"""kv:sliding: a pool of sliding-window layers keeps each sequence as a ring of pages, and its attention (read with
the pool's window) equals a full pool's read with the same window: prompt chunks, decode steps, verify windows rolled
back, forks, on crit's kernels (a bf16 pool, bit for bit) and on the torch fallback (an fp32 pool); a sequence holds at
most its ring whatever its length, binding a round slot takes no page past it, and what the ring cannot hold is
refused."""
import math

from types import SimpleNamespace

from crit_plugins import StepBatch
from crit_plugins.device import RoundBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}
WINDOW, SPAN, PAGE = 40, 24, 16


class SlidingPoolTests(PluginTestCase):
    def setUp(self):
        if not self.device() or self.device() == "cpu":
            self.skipTest("the device form needs a GPU")
        self.torch = __import__("torch")
        parts = self.assemble(USE)
        self.kv, self.attention = parts["kv_cache"], parts["attention"]

    def pools(self, dtype, dim):
        device = self.device()
        full = self.kv.pool([0], 2, dim, 2048, device, dtype, page=PAGE)
        ring = self.kv.pool([0], 2, dim, 256, device, dtype, page=PAGE, sliding=WINDOW, span=SPAN, positions=2048)
        return full, ring

    def step(self, pools, rows, dim, seed):
        torch = self.torch
        batch = StepBatch.build(rows, self.device())
        g = torch.Generator().manual_seed(seed)
        keys, values = torch.randn(batch.tokens, 2, dim, generator=g), torch.randn(batch.tokens, 2, dim, generator=g)
        q = torch.randn(batch.tokens, 4, dim, generator=g).to(self.device())
        out = []
        for pool, sliding in zip(pools, (WINDOW, 0)):
            slots = pool.reserve(batch)
            pool.write(0, slots, keys.to(self.device()), values.to(self.device()))
            out.append(self.attention.forward(0, q, pool, batch, sliding=sliding))
        return out

    def check(self, dtype, dim):
        torch = self.torch
        full, ring = self.pools(dtype, dim)
        same = (lambda a, b: torch.equal(a, b)) if dtype == "bf16" else (
            lambda a, b: float((a.double() - b.double()).abs().max()) < 1e-5)
        for pool in (full, ring):
            pool.open("a"), pool.open("b")
        a, b = self.step((full, ring), [("a", 0, 24), ("b", 0, 7)], dim, 1)   # prompt chunks
        self.assertTrue(same(a, b))
        length = {"a": 24, "b": 7}
        for i in range(60):   # decode steps, every fifth a verify window of 4 that keeps 2
            width = 4 if i % 5 == 4 else 1
            rows = [(s, length[s], width) for s in "ab"]
            a, b = self.step((full, ring), rows, dim, 10 + i)
            self.assertTrue(same(a, b), (dtype, i))
            for s in "ab":
                kept = length[s] + (2 if width > 1 else 1)
                for pool in (full, ring):
                    pool.keep(s, kept)
                length[s] = kept
        held = ring.held("a") + ring.held("b")
        self.assertLessEqual(held, 2 * ring.ring)
        self.assertEqual(ring.ring, math.ceil((WINDOW + SPAN) / PAGE) + 1)
        # A fork shares the ring until either writes; both branches go on as a full pool's do.
        for pool in (full, ring):
            pool.fork("a", "c")
        length["c"] = length["a"]
        for i in range(12):
            a, b = self.step((full, ring), [(s, length[s], 1) for s in "ac"], dim, 100 + i)
            self.assertTrue(same(a, b), (dtype, "fork", i))
            for s in "ac":
                for pool in (full, ring):
                    pool.keep(s, length[s] + 1)
                length[s] += 1
        # Binding a round slot up to far past the ring takes no page past it.
        free, before = ring.free_pages(), ring.held("b")
        ring.bind(0, "b", length["b"] + 1500)
        self.assertEqual(ring.held("b"), ring.ring)
        self.assertEqual(ring.free_pages(), free - (ring.ring - before))
        ring.unbind(0)
        self.assertEqual(ring.held("b"), ring.ring)   # wrapped: its ring holds the window

    def test_a_ring_reads_as_a_full_pool_under_its_window_on_crits_kernels(self):
        self.check("bf16", 64)

    def test_a_ring_reads_as_a_full_pool_under_its_window_on_the_torch_fallback(self):
        self.check("fp32", 8)

    def test_captured_rounds_over_a_ring_read_as_over_a_full_pool_and_replay_past_its_wrap(self):
        torch = self.torch
        full, ring = self.pools("bf16", 64)
        g = torch.Generator().manual_seed(3)

        def kv(positions):   # keys and values a function of the position alone
            rows = [torch.randn(2, 2, 64, generator=torch.Generator().manual_seed(17 * int(p) + 5)) for p in positions]
            both = torch.stack(rows).to(self.device())
            return both[:, 0], both[:, 1]
        for pool in (full, ring):
            pool.open("a")
            for at in range(0, 48, SPAN):   # the prompt in chunks of the span
                batch = StepBatch.build([("a", at, SPAN)], self.device())
                pool.write(0, pool.reserve(batch), *kv(range(at, at + SPAN)))
            pool.keep("a", 48)
            pool.bind(0, "a", 1500)
        self.assertEqual(ring.held("a"), ring.ring)
        width, frontier = 3, 48
        batch = RoundBatch(1, width, self.device())
        for i in range(120):   # rounds of a verify window of 4, keeping 1 to 4 of its columns
            out = []
            for pool, sliding in ((full, WINDOW), (ring, 0)):
                batch.fill(1, width, torch.tensor([frontier], device=self.device()),
                           torch.zeros(1, width + 1, dtype=torch.int64, device=self.device()))
                positions = range(frontier, frontier + width + 1)
                pool.write(0, pool.reserve(batch), *kv(positions))
                q = torch.randn(width + 1, 4, 64, generator=torch.Generator().manual_seed(i)).to(self.device())
                out.append(self.attention.forward(0, q.to(torch.bfloat16), pool, batch, sliding=sliding))
            self.assertTrue(torch.equal(out[0], out[1]), i)
            kept = 1 + int(torch.randint(0, width + 1, (1,), generator=g))
            for pool in (full, ring):
                pool.commit_round(SimpleNamespace(rows=1, frontier=torch.tensor([frontier + kept], device=self.device()),
                                                  n_emitted=torch.tensor([kept], device=self.device())))
            last, frontier = frontier, frontier + kept
        self.assertGreater(frontier, 4 * ring.ring * PAGE)   # wrapped its ring several times
        # One CUDA graph of a round over the ring replays at other frontiers, as the full pool reads them.
        front = torch.tensor([frontier], device=self.device())
        columns = torch.zeros(1, width + 1, dtype=torch.int64, device=self.device())
        q = torch.randn(width + 1, 4, 64, generator=g).to(self.device(), torch.bfloat16)

        def run(pool, sliding):
            batch.fill(1, width, front, columns)
            return self.attention.forward(0, q, pool, batch, sliding=sliding)
        run(ring, 0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(ring, 0)
        # At frontiers whose columns the last round wrote and whose windows the ring still holds (its last ring - 1
        # pages hold every position written).
        for at in (last, last - 5, last - 15):
            front.fill_(at)
            graph.replay()
            self.assertTrue(torch.equal(captured, run(full, WINDOW)), at)

    def test_what_a_ring_cannot_hold_is_refused(self):
        device = self.device()
        with self.assertRaisesRegex(ValueError, "give the most positions"):
            self.kv.pool([0], 2, 64, 256, device, "bf16", page=PAGE, sliding=WINDOW)
        ring = self.kv.pool([0], 2, 64, 256, device, "bf16", page=PAGE, sliding=WINDOW, span=SPAN, positions=512)
        ring.open("a")
        with self.assertRaisesRegex(ValueError, "past the sliding pool's span"):
            ring.reserve(StepBatch.build([("a", 0, SPAN + 1)], device))
        with self.assertRaisesRegex(ValueError, "past the pool's page tables"):
            ring.reserve(StepBatch.build([("a", 520, 1)], device))
        q = self.torch.zeros(1, 4, 64, device=device)
        ring.reserve(StepBatch.build([("a", 0, 1)], device))
        with self.assertRaisesRegex(ValueError, "over a sliding pool that keeps"):
            self.attention.forward(0, q, ring, StepBatch.build([("a", 0, 1)], device), sliding=WINDOW + 1)
