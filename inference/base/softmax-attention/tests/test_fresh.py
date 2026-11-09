"""A prompt row over FP8 and INT8 pages (a step's row, or a prompt batch's chunk, of more than the pool's window)
attends its own chunk's keys and values as the projections gave them (attention-state's fresh buffer) and the earlier
positions from the pages; a decode row beside it and a decode step after it read the chunk from the pages, as do
pieces of up to the window, in a step and in a prompt batch alike, captured or not, with either prompt reader;
over BF16 pages nothing is kept.
To show which keys and values a row reads, most tests put other values in the fresh buffer than the pages hold (as the
kernel's own test does): its rows then equal float64 attention over [the pages' prefix, those values] or over the
pages, and are far from the other. The pages' values are read dequantized in FP32 (rows_at), as the readers use them."""
from crit_plugins import StepBatch
from crit_plugins.device import PromptBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}
HQ, HKV, D = 24, 4, 256
WINDOW = 16
NEAR = 2e-2     # the prompt readers' rounding against float64 (BF16 queries and probabilities)


class FreshChunkTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("the fresh buffer is a CUDA engine's")
        self.torch = __import__("torch")
        self.g = self.torch.Generator().manual_seed(5)

    def engine(self, dtype, kernels="auto", reader="prefill"):
        parts = self.assemble(USE, options={"kernels": kernels, "prompt_reader": reader},
                              use_options={"attention-state": {"page": 64, "window": WINDOW, "fresh_mib": 16}})
        pool = parts["kv_cache"].pool([0], HKV, D, 8192, "cuda", dtype)
        if dtype == "fp8":
            pool.set_scales(0, 0.05, 0.04)
        for sequence in ("a", "b"):
            pool.open(sequence)
        return pool, parts["attention"]

    def rows(self, count, heads):
        return self.torch.randn(count, heads, D, generator=self.g).cuda()

    def reference(self, q, keys, values, first):
        """float64 causal attention of q's rows at positions first.. over keys / values [n][kv heads][D]."""
        torch = self.torch
        q, keys, values = q.double(), keys.double(), values.double()
        out = torch.empty(q.shape[0], HQ, D, dtype=torch.float64, device="cuda")
        for t in range(q.shape[0]):
            for h in range(HQ):
                s = keys[:first + t + 1, h // (HQ // HKV)] @ q[t, h] * D ** -0.5
                out[t, h] = torch.softmax(s, 0) @ values[:first + t + 1, h // (HQ // HKV)]
        return out

    def stored(self, pool, sequence, count):
        """The sequence's first `count` positions as its pages hold them, dequantized in FP32."""
        torch = self.torch
        table = torch.tensor(pool.table(sequence), dtype=torch.int64, device="cuda")
        n = torch.arange(count, device="cuda")
        return pool.rows_at(0, table[n // pool.page] * pool.page + n % pool.page)

    def prefix(self, pool, sequence, count):
        batch = StepBatch.build([(sequence, 0, count)], "cuda")
        pool.write(0, pool.reserve(batch), self.rows(count, HKV), self.rows(count, HKV))
        pool.keep(sequence, count)

    def substitute(self, pool, tokens):
        """Other keys and values in the fresh buffer than the pages hold for the step's `tokens` tokens."""
        torch = self.torch
        keys, values = self.rows(tokens, HKV).to(torch.bfloat16), self.rows(tokens, HKV).to(torch.bfloat16)
        pool.fresh_k[:tokens].copy_(keys)
        pool.fresh_v[:tokens].copy_(values)
        return keys, values

    def against(self, pool, sequence, q, frontier, own):
        """(over own, over the pages): float64 attention of a chunk at `frontier` over [its prefix as the pages hold
        it, `own` (keys, values) for its own positions], and over everything as the pages hold it."""
        torch = self.torch
        prefix_k, prefix_v = self.stored(pool, sequence, frontier)
        fresh = self.reference(q, torch.cat([prefix_k.double(), own[0].double()]),
                               torch.cat([prefix_v.double(), own[1].double()]), frontier)
        return fresh, self.reference(q, *self.stored(pool, sequence, frontier + q.shape[0]), frontier)

    def errors(self, found, fresh, paged):
        return float((found.double() - fresh).abs().max()), float((found.double() - paged).abs().max())

    def assertReads(self, found, fresh, paged, own, case):
        """found reads its own positions from the fresh rows (own) or from the pages: near that one, far from the
        other."""
        error, other = self.errors(found, fresh, paged)
        if not own:
            error, other = other, error
        self.assertLess(error, NEAR, case)
        self.assertGreater(other, 4 * error, case)

    def test_a_prompt_row_reads_itself_fresh_a_decode_row_beside_it_and_after_it_the_pages(self):
        torch = self.torch
        for kernels, reader in (("auto", "prefill"), ("torch", "prefill"), ("auto", "decode"), ("torch", "decode")):
            for dtype in ("fp8:row", "int8", "fp8"):
                case = (kernels, reader, dtype)
                pool, attention = self.engine(dtype, kernels, reader)
                self.assertGreater(pool.fresh_tokens, 200, case)
                self.prefix(pool, "a", 300)
                self.prefix(pool, "b", 4)
                q = self.rows(121, HQ).to(torch.bfloat16)
                # b decodes one token (a window row) beside a's prompt row of 120 tokens.
                batch = StepBatch.build([("b", 4, 1), ("a", 300, 120)], "cuda")
                pool.write(0, pool.reserve(batch), self.rows(121, HKV), self.rows(121, HKV))
                keys, values = self.substitute(pool, 121)
                out = attention.forward(0, q, pool, batch)
                self.assertReads(out[1:], *self.against(pool, "a", q[1:], 300, (keys[1:], values[1:])), True, case)
                self.assertReads(out[:1], *self.against(pool, "b", q[:1], 4, (keys[:1], values[:1])), False, case)
                pool.keep("a", 420)
                # A decode step after the prompt reads the pool: the chunk at its pages' precision.
                step = StepBatch.build([("a", 420, 1)], "cuda")
                pool.write(0, pool.reserve(step), self.rows(1, HKV), self.rows(1, HKV))
                self.assertIsNone(pool.fresh_rows(0, 1), "a decode step keeps nothing fresh")
                q1 = self.rows(1, HQ).to(torch.bfloat16)
                want = self.reference(q1, *self.stored(pool, "a", 421), 420)
                self.assertLess(float((attention.forward(0, q1, pool, step).double() - want).abs().max()), NEAR, case)

    def test_a_prompt_row_is_float64_attention_over_its_prefix_pages_and_its_own_chunk(self):
        # The fresh buffer as write() left it: the chunk as the projections gave it; over FP8 pages (their quantization
        # shows at these values) nearer than attention over the chunk's pages.
        torch = self.torch
        for dtype, reader in (("fp8:row", "prefill"), ("int8", "prefill"), ("fp8", "prefill"), ("fp8:row", "decode")):
            pool, attention = self.engine(dtype, reader=reader)
            self.prefix(pool, "a", 300)
            keys, values = self.rows(120, HKV), self.rows(120, HKV)
            q = self.rows(120, HQ).to(torch.bfloat16)
            batch = StepBatch.build([("a", 300, 120)], "cuda")
            pool.write(0, pool.reserve(batch), keys, values)
            out = attention.forward(0, q, pool, batch)
            own = (keys.to(torch.bfloat16), values.to(torch.bfloat16))
            error, other = self.errors(out, *self.against(pool, "a", q, 300, own))
            self.assertLess(error, NEAR, (dtype, reader))
            if dtype != "int8":
                self.assertLess(error, other, (dtype, reader))

    def test_a_prompt_batch_reads_its_chunk_fresh_past_the_window_and_its_pages_up_to_it(self):
        torch = self.torch
        for kernels, reader in (("auto", "prefill"), ("torch", "prefill"), ("auto", "decode"), ("torch", "decode")):
            for dtype in ("fp8:row", "int8"):
                pool, attention = self.engine(dtype, kernels, reader)
                self.prefix(pool, "a", 200)
                frontier = 200
                for count, own in ((40, True), (WINDOW, False), (9, False), (WINDOW + 1, True)):
                    case = (kernels, reader, dtype, count)
                    batch = PromptBatch(64, "cuda").fill(64, "a", frontier, list(range(count)))
                    pool.stage_prompt(batch)
                    q = self.rows(64, HQ).to(torch.bfloat16)
                    pool.write(0, pool.reserve(batch), self.rows(64, HKV), self.rows(64, HKV))
                    keys, values = self.substitute(pool, 64)
                    out = attention.forward(0, q, pool, batch)
                    pool.commit_prompt(batch)
                    self.assertReads(out[:count], *self.against(pool, "a", q[:count], frontier,
                                                                (keys[:count], values[:count])), own, case)
                    frontier += count

    def test_one_captured_prompt_chunk_reads_fresh_or_its_pages_by_its_count(self):
        torch = self.torch
        for dtype, reader in (("fp8:row", "prefill"), ("int8", "prefill"), ("fp8:row", "decode"), ("int8", "decode")):
            pool, attention = self.engine(dtype, reader=reader)
            self.prefix(pool, "a", 200)
            batch = PromptBatch(64, "cuda")
            q = torch.empty(64, HQ, D, dtype=torch.bfloat16, device="cuda")

            def stage(count, frontier):
                pool.stage_prompt(batch.fill(64, "a", frontier, list(range(count))))
                pool.write(0, pool.reserve(batch), self.rows(64, HKV), self.rows(64, HKV))
                q.copy_(self.rows(64, HQ))
                return self.substitute(pool, 64)
            stage(40, 200)
            attention.forward(0, q, pool, batch)               # its first run eagerly, then the attention captured
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = attention.forward(0, q, pool, batch)
            frontier = 200
            for count, own in ((40, True), (9, False), (WINDOW, False), (50, True)):
                keys, values = stage(count, frontier)
                graph.replay()
                pool.commit_prompt(batch)
                self.assertReads(out[:count], *self.against(pool, "a", q[:count], frontier,
                                                            (keys[:count], values[:count])), own,
                                 (dtype, reader, count))
                frontier += count

    def test_a_step_piece_up_to_the_window_reads_its_pages(self):
        torch = self.torch
        pool, attention = self.engine("fp8:row")
        self.prefix(pool, "a", 200)
        q = self.rows(WINDOW, HQ).to(torch.bfloat16)
        batch = StepBatch.build([("a", 200, WINDOW)], "cuda")
        pool.write(0, pool.reserve(batch), self.rows(WINDOW, HKV), self.rows(WINDOW, HKV))
        self.assertIsNone(pool.fresh_rows(0, WINDOW), "a step of window rows keeps nothing fresh")
        want = self.reference(q, *self.stored(pool, "a", 200 + WINDOW), 200)
        self.assertLess(float((attention.forward(0, q, pool, batch).double() - want).abs().max()), NEAR)

    def test_16_bit_pages_keep_nothing_fresh(self):
        torch = self.torch
        pool, attention = self.engine("bf16")
        self.assertEqual(pool.fresh_tokens, 0)
        self.prefix(pool, "a", 100)
        batch = StepBatch.build([("a", 100, 50)], "cuda")
        pool.write(0, pool.reserve(batch), self.rows(50, HKV), self.rows(50, HKV))
        self.assertIsNone(pool.fresh_rows(0, 50))
        attention.forward(0, self.rows(50, HQ).to(torch.bfloat16), pool, batch)
