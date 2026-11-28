"""Which reader each row of a step goes through (attention-state's device form, which softmax-attention serves): a
row's phase decides it, never its length or the rows beside it (crit_plugins.testing's routing conformance over its
phase x length grid). On the CPU, with recording kernels or readers in place of crit's: each records the path of the
tokens it computes."""
import sys

from crit_plugins import StepBatch
from crit_plugins.device import DECODE, PREFILL
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}
R, P, H = 32, 16, 2


class LatentKernels:
    """crit's latent-attention module's launches, recording: each writes its code into the rows it computes."""

    MAX_COLUMNS = 64
    CODES = {1.0: "decode reader", 2.0: "fused pieces", 3.0: "prompt reader"}

    @staticmethod
    def attend(q, cache, tables, rows, starts, firsts, columns, page, counts=None, out=None, lse=None, fused=False,
               **common):
        counts = counts.tolist() if counts is not None else [columns] * len(starts)
        for start, count in zip(starts.tolist(), counts):
            out[start:start + count] = 2.0 if fused else 1.0

    @staticmethod
    def prefill(q, cache, table, position, page, out=None, lse=None, **common):
        out.fill_(3.0)

    @staticmethod
    def prefill_rows(q, cache, tables, firsts, counts, positions, page, out=None, lse=None, **common):
        for first, count in zip(firsts, counts):
            out[first:first + count] = 3.0


def prefixed(pool, batch, write):
    """Open each row's sequence and write its positions before the row's first, then the step's own: (slots)."""
    for r in range(batch.rows):
        sequence, _, _, position = batch.row(r)
        pool.open(sequence)
        if position:
            write(pool.reserve(StepBatch.build([(sequence, 0, position, PREFILL)], batch.device)), position)
            pool.keep(sequence, position)
    slots = pool.reserve(batch)
    write(slots, batch.tokens)
    return slots


class LatentRoutingTests(PluginTestCase):
    """The latent reader: a prompt chunk of up to 64 tokens went through the decode reader, so a short prompt's rows
    had other bits than the same rows in a longer one (Gemma 4's drafter)."""

    def setUp(self):
        self.torch = __import__("torch")

    def route(self, prompt_reader):
        torch = self.torch
        parts = self.assemble(USE, options={"prompt_reader": prompt_reader})
        attention = parts["attention"]
        attention.device_form._latent_module = lambda pool, q: LatentKernels

        def run(batch):
            pool = parts["kv_cache"].pool([0], 1, R + P, 1024, "cpu", "bf16", page=16, value_dim=0)
            prefixed(pool, batch, lambda slots, n: pool.write(0, slots, torch.zeros(n, 1, R + P), None))
            out = attention.forward(0, torch.zeros(batch.tokens, H, R + P), pool, batch, latent=R)
            return [LatentKernels.CODES[float(v)] for v in out[:, 0, 0]]
        return run

    def test_a_rows_reader_follows_its_phase_at_every_length(self):
        for prompt_reader, prompt in (("prefill", "prompt reader"), ("decode", "fused pieces")):
            found = self.assertRoutesByPhase(self.route(prompt_reader))
            self.assertEqual({phase: list(paths) for phase, paths in found.items()},
                             {PREFILL: [(prompt,)], DECODE: [("decode reader",)]}, prompt_reader)

    def test_a_decode_row_past_the_decode_readers_columns_is_refused(self):
        with self.assertRaisesRegex(ValueError, "decode row of up to 64"):
            self.route("prefill")(StepBatch.build([("a", 0, 65, DECODE)], "cpu"))


class SoftmaxRoutingTests(PluginTestCase):
    """The softmax readers: a row of up to the pool's window went through the window reader by its length unless the
    model marked it (attention-state read prompt chunks of up to 16 tokens as decode rows: 4.1 nats on Gemma 4), and
    over FP8 and INT8 pages a prompt chunk of up to the window read its own positions from the pages, not fresh. On
    the CPU: prompt_reader decode reads window rows and prompt pieces over the page tables (recorded); prompt_reader
    prefill reads every row in torch, a prompt row's own positions fresh (recorded where it takes them)."""

    def setUp(self):
        self.torch = __import__("torch")

    def route(self, prompt_reader):
        torch = self.torch
        parts = self.assemble(USE, options={"prompt_reader": prompt_reader})
        form = parts["attention"].device_form
        code = sys.modules[type(form).__module__]
        pages = sys.modules[type(parts["kv_cache"].pool([0], 2, 16, 64, "cpu", "bf16")).__module__]
        labels = {}

        def mark(first, count, label):
            for t in range(first, first + count):
                labels[t] = label

        def windows(layer, q, pool, meta, *args, **kwargs):
            for start, count in zip(meta.starts.tolist(), meta.counts.tolist()):
                mark(start, count, "window reader")

        def pieces(layer, q, pool, batch, rows, *args, **kwargs):
            for r in rows:
                mark(batch.row(r)[1], batch.row(r)[2], "prompt pieces")

        def with_fresh(gathered, fresh, first, count):
            mark(first, count, "torch, own positions fresh" if fresh is not None else "torch, own positions paged")
            return real(gathered, fresh, first, count)
        real = code._with_fresh
        form._windows, form._decode_rows = windows, pieces

        def run(batch):
            labels.clear()
            fresh = pages.FreshBuffer(torch.empty(1 << 20, dtype=torch.uint8))
            pool = pages.KVPool([0], 2, 16, 1024, "cpu", "fp8:row", page=16, window=16, fresh=fresh)
            prefixed(pool, batch, lambda slots, n: pool.write(0, slots, torch.randn(n, 2, 16), torch.randn(n, 2, 16)))
            code._with_fresh = with_fresh
            try:
                form.forward(0, torch.randn(batch.tokens, 4, 16), pool, batch)
            finally:
                code._with_fresh = real
            return [labels.get(t, "unread") for t in range(batch.tokens)]
        return run

    def test_a_rows_reader_follows_its_phase_at_every_length(self):
        for prompt_reader, prompt, decode in (("decode", "prompt pieces", "window reader"),
                                              ("prefill", "torch, own positions fresh", "torch, own positions paged")):
            found = self.assertRoutesByPhase(self.route(prompt_reader))
            self.assertEqual({phase: list(paths) for phase, paths in found.items()},
                             {PREFILL: [(prompt,)], DECODE: [(decode,)]}, prompt_reader)
