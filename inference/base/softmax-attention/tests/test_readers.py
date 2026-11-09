"""prompt_reader "decode": every row kind over the pages in the decode's reduction, so a position's attention has the
same bits as a decode row alone, as a verify window's column, as a round's column, in a step's prompt row and in a
captured prompt chunk (served == scored), over FP8, fp8:row, INT8 and BF16 pages, at a 27B attention shape, before and
past the length where partitions hold several pieces; through the registry's split-KV reader and through its torch
fallback over the page tables; in two pools of different page kinds in one model; with block lists, a prompt row,
a decode row and a round's columns alike through the registry's kernel. prompt_reader "prefill" (the prompt reader,
another reduction) stays within the bound its facts state. The fresh buffer is off: a prompt chunk's own fresh keys
and values are a stated difference (test_fresh)."""
from crit_plugins import StepBatch
from crit_plugins.device import PromptBatch, RoundBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}
HQ, HKV, D = 24, 4, 256
CHUNK = 100
KINDS = ("int8", "fp8:row", "fp8", "bf16")


class ReaderAgreementTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("rounds and captured chunks run on CUDA")
        self.torch = __import__("torch")

    def engine(self, kernels="auto", prompt_reader="decode"):
        parts = self.assemble(USE, options={"prompt_reader": prompt_reader, "kernels": kernels},
                              use_options={"attention-state": {"page": 64, "fresh_mib": 0}})
        return parts["kv_cache"], parts["attention"]

    def pool(self, kv, dtype, layers=(0,)):
        pool = kv.pool(list(layers), HKV, D, 8192, "cuda", dtype)
        if dtype == "fp8":
            for layer in layers:
                pool.set_scales(layer, 0.05, 0.04)
        return pool

    def rows(self, positions, seed, heads):
        """Rows that are a function of the position alone, so every path reads the same keys, values and queries."""
        torch = self.torch
        return torch.stack([torch.randn(heads, D, generator=torch.Generator().manual_seed(seed * 1_000_003 + p))
                            for p in positions]).cuda()

    def kv(self, positions, layer=0):
        return self.rows(positions, 1 + 7 * layer, HKV), self.rows(positions, 2 + 7 * layer, HKV)

    def q(self, positions):
        return self.rows(positions, 3, HQ).to(self.torch.bfloat16)

    def step(self, attention, pool, rows, layer=0):
        """A StepBatch of (sequence, first position, count) rows: its keys and values written, its attention read."""
        batch = StepBatch.build(rows, "cuda")
        positions = [p + i for _, p, n in rows for i in range(n)]
        pool.write(layer, pool.reserve(batch), *self.kv(positions, layer))
        return attention.forward(layer, self.q(positions), pool, batch)

    def blocks(self, positions, size, k=48):
        """Each token's block list, a function of its position alone: every third complete block below its own (the
        most recent k - 1 of them) and the block holding the position, ascending, -1 past them."""
        rows = []
        for p in positions:
            own = p // size
            listed = list(range(own - 3, -1, -3))[:k - 1][::-1] + [own]
            rows.append(listed + [-1] * (k - len(listed)))
        return self.torch.tensor(rows, dtype=self.torch.int32, device="cuda")

    def sparse(self, attention, pool, batch, positions, size, layer=0):
        """The batch's keys and values written, its attention read over each token's block list."""
        pool.write(layer, pool.reserve(batch), *self.kv(positions, layer))
        return attention.forward(layer, self.q(positions), pool, batch, None, self.blocks(positions, size), size)

    def agree(self, attention, pool, prefix, case, layer=0, sequence="a", other="b"):
        """The chunk prefix .. prefix + CHUNK as a step's prompt row, its positions decoded alone, in a verify window,
        in a round and captured: the same bits."""
        torch = self.torch
        pool.open(sequence)
        if prefix:
            self.step(attention, pool, [(sequence, 0, prefix)], layer)
            pool.keep(sequence, prefix)
        prompt = self.step(attention, pool, [(sequence, prefix, CHUNK)], layer)
        # (The sequence's valid length stays at prefix: each later row reads up to its own position.)
        for at in (0, 1, 37, 63, 64, 99):
            self.assertTrue(torch.equal(self.step(attention, pool, [(sequence, prefix + at, 1)], layer),
                                        prompt[at:at + 1]), case + ("decode", at))
        # A verify window of 8 columns, beside another sequence's decode row.
        pool.open(other)
        self.step(attention, pool, [(other, 0, 30)], layer)
        pool.keep(other, 30)
        window = self.step(attention, pool, [(other, 30, 1), (sequence, prefix + 10, 8)], layer)
        self.assertTrue(torch.equal(window[1:], prompt[10:18]), case + ("window",))
        # A round's 8 columns.
        pool.bind(0, sequence, prefix + CHUNK + 16)
        batch = RoundBatch(1, 7, "cuda").fill(1, 7, torch.tensor([prefix + 40], device="cuda"),
                                              torch.zeros(1, 8, dtype=torch.int64, device="cuda"))
        positions = list(range(prefix + 40, prefix + 48))
        pool.write(layer, pool.reserve(batch), *self.kv(positions, layer))
        self.assertTrue(torch.equal(attention.forward(layer, self.q(positions), pool, batch), prompt[40:48]),
                        case + ("round",))
        pool.unbind(0)
        # The chunk as a prompt batch of its tokens in a 128-token bucket, eagerly and replayed from a CUDA graph.
        chunk = PromptBatch(128, "cuda")
        positions = list(range(prefix, prefix + 128))
        q, kv = self.q(positions), self.kv(positions, layer)

        def run():
            pool.write(layer, pool.reserve(chunk), *kv)
            return attention.forward(layer, q, pool, chunk)
        pool.stage_prompt(chunk.fill(128, sequence, prefix, list(range(CHUNK))))
        eager = run()
        self.assertTrue(torch.equal(eager[:CHUNK], prompt), case + ("prompt batch",))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        pool.stage_prompt(chunk.fill(128, sequence, prefix, list(range(CHUNK))))
        graph.replay()
        pool.commit_prompt(chunk)
        self.assertTrue(torch.equal(captured[:CHUNK], prompt), case + ("captured",))

    def test_every_row_kind_reads_the_same_bits_in_the_decode_reduction(self):
        for kernels in ("auto", "torch"):
            kv, attention = self.engine(kernels)
            if kernels == "auto" and attention.device_form.kernels is None:
                continue
            for dtype in KINDS:
                for prefix in ((0, 3000) if kernels == "auto" else (0, 300)):
                    self.agree(attention, self.pool(kv, dtype), prefix, (kernels, dtype, prefix))

    def test_block_lists_read_the_same_bits_in_a_prompt_row_a_decode_row_and_a_round(self):
        # Block-sparse lists take one partition count in every launch of the registry's split-KV reader (a prompt
        # row's pieces, a step's window rows, a round's columns): a decoded token reads its listed blocks in the prompt
        # row's reduction, past the length where the device's partitions would split a row differently. (The torch
        # fallback reads a prompt row's lists over the gathered rows, another order.)
        torch, size, prefix = self.torch, 16, 3000
        kv, attention = self.engine()
        if attention.device_form.kernels is None:
            self.skipTest("the registry's split-KV reader does not load here")
        for dtype in ("fp8", "bf16"):
            pool = self.pool(kv, dtype)
            pool.open("a")
            self.step(attention, pool, [("a", 0, prefix)])
            pool.keep("a", prefix)
            positions = list(range(prefix, prefix + CHUNK))
            prompt = self.sparse(attention, pool, StepBatch.build([("a", prefix, CHUNK)], "cuda"), positions, size)
            for at in (0, 1, 37, 63, 64, 99):
                decoded = self.sparse(attention, pool, StepBatch.build([("a", prefix + at, 1)], "cuda"),
                                      [prefix + at], size)
                self.assertTrue(torch.equal(decoded, prompt[at:at + 1]), (dtype, "decode", at))
            pool.bind(0, "a", prefix + CHUNK + 16)
            batch = RoundBatch(1, 7, "cuda").fill(1, 7, torch.tensor([prefix + 40], device="cuda"),
                                                  torch.zeros(1, 8, dtype=torch.int64, device="cuda"))
            columns = self.sparse(attention, pool, batch, list(range(prefix + 40, prefix + 48)), size)
            self.assertTrue(torch.equal(columns, prompt[40:48]), (dtype, "round"))
            pool.unbind(0)

    def test_two_pools_of_different_page_kinds_each_read_in_the_decode_reduction(self):
        # One model: layer 0 over fp8:row pages, layer 1 over BF16 pages (a second pool, the same plugins).
        kv, attention = self.engine()
        quantized, wide = self.pool(kv, "fp8:row", (0,)), self.pool(kv, "bf16", (1,))
        self.agree(attention, quantized, 600, ("fp8:row pool",), layer=0)
        self.agree(attention, wide, 600, ("bf16 pool",), layer=1)
        facts = kv.facts()
        per_token = sum(t[:, 0, 0].numel() * t.element_size() for pool in (quantized, wide)
                        for t in (pool.k, pool.v, pool.k_scales, pool.v_scales) if t is not None)
        self.assertEqual(facts["kv_bytes_per_token"], per_token)     # both pools counted
        self.assertIn("decode rows read them", facts["prompt_kv"])

    def test_the_prefill_reader_stays_within_its_stated_bound(self):
        # prompt_reader "prefill": prompt rows through the prompt reader, decode rows through the split-KV reader:
        # another reduction, within the readers' rounding the facts state.
        kv, attention = self.engine(prompt_reader="prefill")
        self.assertEqual(attention.facts()["served_vs_scored"], "within the readers' rounding")
        for dtype in ("int8", "fp8:row"):
            pool = self.pool(kv, dtype)
            pool.open("a")
            self.step(attention, pool, [("a", 0, 600)])
            pool.keep("a", 600)
            prompt = self.step(attention, pool, [("a", 600, CHUNK)])
            for at in (0, 50, 99):
                decoded = self.step(attention, pool, [("a", 600 + at, 1)])
                self.assertLess(float((decoded.float() - prompt[at:at + 1].float()).abs().max()), 1.6e-2, dtype)

    def test_the_facts_state_the_honesty_class(self):
        facts = self.engine()[1].facts()
        self.assertEqual(facts["served_vs_scored"], "exact")
        self.assertIn("decode's reduction", facts["prompt_reads"])
