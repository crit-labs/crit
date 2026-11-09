"""softmax-attention over a RoundBatch (rounds:attention), trees (attention:tree) and block lists (attention:blocks):
against a float64 reference per page type, batched rows equal to rows alone, a verify window accepted at every length
and continued equal to serial decoding, one CUDA graph replayed at other lengths, and work that follows the rows'
lengths. Bit for bit where the registry's split-KV reader serves; within the fallback's FP32 rounding otherwise."""
from types import SimpleNamespace

from crit_plugins import StepBatch
from crit_plugins.device import RoundBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}
HQ, HKV, D = 8, 2, 128
TYPES = ("bf16", "fp16", "fp8", "fp8:row", "int8")


class RoundAttentionTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("captured rounds run on CUDA")
        self.torch = __import__("torch")
        parts = self.assemble(USE, options={"decode": "split"})
        self.kv, self.attention = parts["kv_cache"], parts["attention"]
        self.split = self.attention.device_form.kernels is not None
        self.plain = self.assemble(USE, options={"kernels": "torch"})["attention"]

    def pool(self, dtype, tokens=4096, page=16, hkv=HKV, d=D):
        pool = self.kv.pool([0], hkv, d, tokens, "cuda", dtype, page=page)
        if dtype == "fp8":
            pool.set_scales(0, 0.02, 0.03)
        return pool

    def kv_at(self, positions, seed, hkv=HKV, d=D):
        """Keys and values of `positions`, a function of the position alone (so serial and speculative agree)."""
        torch = self.torch
        rows = []
        for p in positions:
            g = torch.Generator().manual_seed(seed * 1_000_003 + int(p))
            rows.append(torch.randn(2, hkv, d, generator=g))
        both = torch.stack(rows).cuda()
        return both[:, 0], both[:, 1]

    def queries(self, positions, seed, hq=HQ, d=D):
        torch = self.torch
        return torch.stack([torch.randn(hq, d, generator=torch.Generator().manual_seed(seed * 7 + int(p)))
                            for p in positions]).cuda().to(torch.bfloat16)

    def prefill(self, pool, sequence, count, seed):
        pool.open(sequence)
        batch = StepBatch.build([(sequence, 0, count)], "cuda")
        keys, values = self.kv_at(range(count), seed, pool.kv_heads, pool.head_dim)
        pool.write(0, pool.reserve(batch), keys, values)
        pool.keep(sequence, count)

    def round(self, pool, batch, frontier, width, values_seed, junk_from=None, q_seed=3):
        """One round's attention: its columns' KV written (columns from junk_from on with other values), then read."""
        torch = self.torch
        rows = len(frontier)
        batch.fill(rows, width, torch.tensor(frontier, device="cuda"),
                   torch.zeros(rows, width + 1, dtype=torch.int64, device="cuda"))
        positions = [f + c for f in frontier for c in range(width + 1)]
        keys, values = self.kv_at(positions, values_seed, pool.kv_heads, pool.head_dim)
        if junk_from is not None:
            for i, p in enumerate(positions):
                if (i % (width + 1)) >= junk_from:
                    keys[i], values[i] = keys[i] * -2 + 1, values[i] * 3
        pool.write(0, pool.reserve(batch), keys, values)
        return self.attention.forward(0, self.queries(positions, q_seed), pool, batch)

    def reference(self, pool, sequence, positions, q):
        torch = self.torch
        keys, values = pool.gather(0, sequence, max(positions) + 1)
        keys, values = keys.double(), values.double()
        out = []
        g = q.shape[1] // keys.shape[1]
        for t, p in enumerate(positions):
            row = []
            for h in range(q.shape[1]):
                s = keys[:p + 1, h // g] @ q[t, h].double() * q.shape[-1] ** -0.5
                row.append(torch.softmax(s, 0) @ values[:p + 1, h // g])
            out.append(torch.stack(row))
        return torch.stack(out)

    def same(self, a, b, what):
        if self.split:
            self.assertTrue(self.torch.equal(a, b), what)
        else:
            self.torch.testing.assert_close(a.float(), b.float(), atol=2e-2, rtol=2e-2, msg=what)

    def test_it_provides_rounds_trees_and_blocks(self):
        self.assertImplements(self.assemble(USE))

    def test_by_default_a_step_decodes_a_row_as_a_round_does(self):
        """With the plugin's default options (FlashInfer installed or not), a row a step decodes alone and the same
        row as a round's column, and a step's verify window and the round's: the same bits, on every page type the
        split-KV reader reads (plain decoding against drafted rounds)."""
        torch = self.torch
        parts = self.assemble(USE)
        kv, attention = parts["kv_cache"], parts["attention"]
        if attention.device_form.kernels is None:
            self.skipTest("the registry's split-KV reader is not here")
        for dtype in TYPES:
            pool = kv.pool([0], HKV, D, 4096, "cuda", dtype)
            if dtype == "fp8":
                pool.set_scales(0, 0.02, 0.03)
            for s, n in (("a", 300), ("b", 37)):
                self.prefill(pool, s, n, seed=5)
            pool.bind(0, "a", 400)
            pool.bind(1, "b", 400)
            batch = RoundBatch(2, 3, "cuda")
            batch.fill(2, 3, torch.tensor([300, 37], device="cuda"),
                       torch.zeros(2, 4, dtype=torch.int64, device="cuda"))
            positions = [f + c for f in (300, 37) for c in range(4)]
            keys, values = self.kv_at(positions, 5)
            pool.write(0, pool.reserve(batch), keys, values)
            q = self.queries(positions, 9)
            rounded = attention.forward(0, q, pool, batch)
            for r, sequence in enumerate("ab"):
                first = (300, 37)[r]
                for count in (1, 4):   # the pending token alone (plain decoding), the whole window
                    step = StepBatch.build([(sequence, first, count)], "cuda")
                    span = slice(4 * r, 4 * r + count)
                    pool.write(0, pool.reserve(step), keys[span], values[span])
                    found = attention.forward(0, q[span], pool, step)
                    self.assertTrue(torch.equal(found, rounded[span]), (dtype, sequence, count))

    def test_round_rows_match_float64_attention_on_every_page_type(self):
        for dtype in TYPES:
            pool = self.pool(dtype)
            for s, n in (("a", 300), ("b", 37)):
                self.prefill(pool, s, n, seed=1)
            pool.bind(0, "a", 400)
            pool.bind(1, "b", 400)
            batch = RoundBatch(2, 4, "cuda")
            out = self.round(pool, batch, [300, 37], 4, values_seed=1)
            q = self.queries([300 + c for c in range(5)] + [37 + c for c in range(5)], 3)
            for r, (sequence, f) in enumerate((("a", 300), ("b", 37))):
                want = self.reference(pool, sequence, [f + c for c in range(5)], q[5 * r:5 * r + 5])
                self.torch.testing.assert_close(out[5 * r:5 * r + 5].double(), want, atol=1e-2, rtol=2e-2,
                                                msg=dtype)

    def test_a_verify_window_accepted_at_every_length_and_continued_equals_serial_decoding(self):
        width = 5
        for dtype in ("bf16", "fp8:row", "int8"):
            serial_pool = self.pool(dtype)
            self.prefill(serial_pool, "s", 100, seed=2)
            serial_pool.bind(0, "s", 200)
            batch = RoundBatch(1, width, "cuda")
            serial = {}
            for p in range(100, 100 + 2 * (width + 1)):
                serial[p] = self.round(serial_pool, batch, [p], 0, values_seed=2)[0]
            for kept in range(1, width + 2):     # the pending token's column and 0..width accepted drafts
                pool = self.pool(dtype)
                self.prefill(pool, "s", 100, seed=2)
                pool.bind(0, "s", 200)
                out = self.round(pool, batch, [100], width, values_seed=2, junk_from=kept)
                for c in range(kept):
                    self.same(out[c], serial[100 + c], (dtype, kept, c))
                pool.commit_round(SimpleNamespace(rows=1, frontier=self.torch.tensor([100 + kept], device="cuda"),
                                                  n_emitted=self.torch.tensor([kept], device="cuda")))
                self.assertEqual(pool.length("s"), 100 + kept)
                again = self.round(pool, batch, [100 + kept], width, values_seed=2)
                for c in range(width + 1):
                    self.same(again[c], serial[100 + kept + c], (dtype, kept, "continued", c))

    def test_rows_of_a_round_equal_the_rows_alone(self):
        pool = self.pool("fp8:row")
        frontiers = {"a": 900, "b": 17, "c": 2000}
        for i, (s, n) in enumerate(frontiers.items()):
            self.prefill(pool, s, n, seed=4 + i)
        for slot, (s, f) in enumerate(frontiers.items()):
            pool.bind(slot, s, f + 50)
        batch = RoundBatch(3, 3, "cuda")
        together = self.round(pool, batch, list(frontiers.values()), 3, values_seed=9)
        for slot in range(3):
            pool.unbind(slot)
        for r, (s, f) in enumerate(frontiers.items()):
            pool.bind(0, s, f + 50)
            alone = self.round(pool, batch, [f], 3, values_seed=9)
            self.same(alone, together[4 * r:4 * r + 4], s)
            pool.unbind(0)

    def test_a_round_replays_from_one_cuda_graph_at_other_lengths(self):
        torch = self.torch
        pool = self.pool("int8", tokens=8192)
        for s in "ab":
            self.prefill(pool, s, 3000, seed=6)
        pool.bind(0, "a", 3100)
        pool.bind(1, "b", 3100)
        batch = RoundBatch(2, 3, "cuda")
        frontier = torch.tensor([10, 20], device="cuda")
        columns = torch.zeros(2, 4, dtype=torch.int64, device="cuda")
        q = self.queries(range(8), 5)

        def run():
            batch.fill(2, 3, frontier, columns)
            return self.attention.forward(0, q, pool, batch)
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        for a, b in ((10, 20), (2990, 5), (64, 1500), (1, 2)):
            frontier.copy_(torch.tensor([a, b]))
            graph.replay()
            self.same(out, run(), (a, b))
            want = self.reference(pool, "a", [a + c for c in range(4)], q[:4])
            torch.testing.assert_close(out[:4].double(), want, atol=1e-2, rtol=2e-2)

    def test_a_step_decodes_as_a_round_does_with_the_split_reader(self):
        if not self.split:
            self.skipTest("the registry's split-KV reader is not loaded")
        pool = self.pool("bf16")
        self.prefill(pool, "a", 500, seed=7)
        q = self.queries([500, 501, 502], 8)
        step = StepBatch.build([("a", 500, 3)], "cuda")
        keys, values = self.kv_at([500, 501, 502], 7)
        pool.write(0, pool.reserve(step), keys, values)
        stepped = self.attention.forward(0, q, pool, step)
        pool.keep("a", 500)
        pool.bind(0, "a", 600)
        batch = RoundBatch(1, 2, "cuda").fill(1, 2, self.torch.tensor([500], device="cuda"),
                                              self.torch.zeros(1, 3, dtype=self.torch.int64, device="cuda"))
        pool.write(0, pool.reserve(batch), keys, values)
        self.assertTrue(self.torch.equal(self.attention.forward(0, q, pool, batch), stepped))

    def test_a_round_column_equals_the_steps_decode_row_in_the_queries_dtype(self):
        torch = self.torch
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            pool = self.pool("bf16")
            self.prefill(pool, "a", 50, seed=11)
            keys, values = self.kv_at([50], 11)
            q = self.queries([50], 12).to(dtype)
            step = StepBatch.build([("a", 50, 1)], "cuda")
            pool.write(0, pool.reserve(step), keys, values)
            stepped = self.attention.forward(0, q, pool, step)
            pool.keep("a", 50)
            pool.bind(0, "a", 100)
            batch = RoundBatch(1, 0, "cuda").fill(1, 0, torch.tensor([50], device="cuda"),
                                                  torch.zeros(1, 1, dtype=torch.int64, device="cuda"))
            pool.write(0, pool.reserve(batch), keys, values)
            rounded = self.attention.forward(0, q, pool, batch)
            self.assertEqual((stepped.dtype, rounded.dtype), (dtype, dtype))
            self.assertTrue(torch.equal(rounded, stepped), dtype)

    def test_a_tree_and_block_lists_match_a_dense_masked_reference(self):
        torch = self.torch
        pool = self.pool("bf16")
        self.prefill(pool, "a", 600, seed=8)
        pool.bind(0, "a", 700)
        batch = RoundBatch(1, 7, "cuda")
        out_plain = self.round(pool, batch, [600], 7, values_seed=8)
        q = self.queries(range(600, 608), 3)
        parents = torch.tensor([-1, 0, 0, 1, 1, 2, 5, 3], device="cuda")
        keys, values = pool.gather(0, "a", 608)
        keys, values = keys.double(), values.double()
        out, lse = self.attention.forward(0, q, pool, batch, parents=parents, lse=True)
        for c in range(8):
            anc, p = {c}, int(parents[c])
            while p >= 0:
                anc.add(p)
                p = int(parents[p])
            seen = torch.tensor([n < 600 or (n - 600) in anc for n in range(608)], device="cuda")
            for h in range(HQ):
                s = keys[:, h // 4] @ q[c, h].double() * D ** -0.5
                s = s.masked_fill(~seen, float("-inf"))
                torch.testing.assert_close(out[c, h].double(), torch.softmax(s, 0) @ values[:, h // 4],
                                           atol=1e-2, rtol=2e-2)
                torch.testing.assert_close(lse[c, h].double(), torch.logsumexp(s, 0), atol=2e-3, rtol=1e-4)
        chain = self.attention.forward(0, q, pool, batch, parents=torch.arange(-1, 7, device="cuda"))
        self.same(chain, out_plain, "a chain tree is the causal window")
        blocks = torch.tensor([[0, 3, 9, 149, -1, -1]] * 8, dtype=torch.int32, device="cuda")
        sparse = self.attention.forward(0, q, pool, batch, blocks=blocks, block_size=4)
        n = torch.arange(608, device="cuda")
        for c in range(8):
            seen = (n // 4)[:, None].eq(blocks[c][blocks[c] >= 0].long()[None, :]).any(1) & (n <= 600 + c)
            for h in range(HQ):
                s = (keys[:, h // 4] @ q[c, h].double() * D ** -0.5).masked_fill(~seen, float("-inf"))
                torch.testing.assert_close(sparse[c, h].double(), torch.softmax(s, 0) @ values[:, h // 4],
                                           atol=1e-2, rtol=2e-2)

    def test_work_follows_the_rows_length_not_the_capacity(self):
        # No timing (a plugin test shares the GPU with whatever else runs): one graph captured at a fixed capacity is
        # replayed at a long and then a short frontier, every position past the frontier poisoned first (NaN key and
        # value scales). A read of any of them would make the output NaN or move it, so each replay reads its row's own
        # positions only, whatever the capacity the tables hold. (Its speed: the kernel's benchmarks, exclusive.)
        if not self.split:
            self.skipTest("the torch fallback reads every position the round's tables hold")
        torch = self.torch
        pool = self.kv.pool([0], 4, 256, 210_000, "cuda", "fp8:row", page=64)
        pool.open("a")
        batch = StepBatch.build([("a", 0, 200_000)], "cuda")
        slots = pool.reserve(batch)
        for at in range(0, 200_000, 20_000):
            part = torch.randn(20_000, 4, 256, device="cuda")
            pool.write(0, slots[at:at + 20_000], part, part)
        pool.keep("a", 200_000)
        pool.bind(0, "a", 201_000)
        round_batch = RoundBatch(1, 0, "cuda")
        q = torch.randn(1, 24, 256, device="cuda").to(torch.bfloat16)
        frontier = torch.zeros(1, dtype=torch.int64, device="cuda")
        columns = torch.zeros(1, 1, dtype=torch.int64, device="cuda")

        def run():
            round_batch.fill(1, 0, frontier, columns)
            return self.attention.forward(0, q, pool, round_batch)
        run()
        graph = torch.cuda.CUDAGraph()     # one graph at a fixed capacity, its work read from the frontier
        with torch.cuda.graph(graph):
            out = run()
        table = torch.tensor(pool.table("a"), device="cuda")
        for at in (199_999, 150_000, 4_000):
            frontier.fill_(at)
            want = run().clone()           # eagerly, before the positions past the frontier are poisoned
            past = torch.arange(at + 1, 201_000, device="cuda")
            slot = table[past // 64].long() * 64 + past % 64
            for scales in (pool.k_scales, pool.v_scales):
                scales[0].view(-1, 4).index_fill_(0, slot, float("nan"))
            graph.replay()
            self.assertTrue(bool(torch.isfinite(out).all()), at)
            self.assertTrue(torch.equal(out, want), at)
