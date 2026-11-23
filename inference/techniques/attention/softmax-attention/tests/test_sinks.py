"""attention:sinks and attention:sliding: forward(..., sinks=) gives every token one more position with no value a query head, on crit's
kernels (a bf16 pool) and on the torch fallback (an fp32 pool), decode rows, verify windows and prompt rows alike, with
and without log-sum-exps; sinks of -inf change no bit."""

from crit_plugins import StepBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}


class SinkTests(PluginTestCase):
    def setUp(self):
        if not self.device() or self.device() == "cpu":
            self.skipTest("the device form needs a GPU")
        self.torch = __import__("torch")
        self.parts = self.assemble(USE)
        self.kv, self.attention = self.parts["kv_cache"], self.parts["attention"]

    def run_case(self, dtype, d):
        torch = self.torch
        device = self.device()
        pool = self.kv.pool([0], 2, d, 1024, device, dtype, page=16)
        g = torch.Generator().manual_seed(5)
        keys = {s: torch.randn(80, 2, d, generator=g) for s in "ab"}
        values = {s: torch.randn(80, 2, d, generator=g) for s in "ab"}
        for s in "ab":
            pool.open(s)

        def write(rows):
            batch = StepBatch.build(rows, device)
            slots = pool.reserve(batch)
            k = torch.cat([keys[s][p:p + n] for s, p, n in rows]).to(device)
            v = torch.cat([values[s][p:p + n] for s, p, n in rows]).to(device)
            pool.write(0, slots, k, v)
            return batch

        write([("a", 0, 40), ("b", 0, 20)])
        batch = write([("a", 40, 1), ("b", 20, 30)])   # a decode row and a prompt row
        q = torch.randn(batch.tokens, 4, d, generator=torch.Generator().manual_seed(6)).to(device)
        sinks = torch.tensor([0.5, 6.0, float("-inf"), -1.0], device=device)
        out, lse = self.attention.forward(0, q, pool, batch, lse=True, sinks=sinks)
        scale = d ** -0.5
        qd = q.to(torch.bfloat16).double() if dtype == "bf16" else q.double()
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            k, v = pool.gather(0, sequence, position + count)
            k, v = k.double(), v.double()
            for t in range(count):
                for h in range(4):
                    s = (k[: position + t + 1, h // 2] @ qd[first + t, h]) * scale
                    s = torch.cat([s, torch.tensor([float(sinks[h])], dtype=torch.float64, device=device)])
                    p = torch.softmax(s, 0)
                    want = p[:-1] @ v[: position + t + 1, h // 2]
                    self.assertLess(float((out[first + t, h].double() - want).abs().max()), 1.5e-2, (dtype, r, t, h))
                    self.assertLess(abs(float(lse[first + t, h]) - float(torch.logsumexp(s, 0))), 2e-3)
        none = torch.full((4,), float("-inf"), device=device)
        plain = self.attention.forward(0, q, pool, batch)
        with_none = self.attention.forward(0, q, pool, batch, sinks=none)
        if dtype == "bf16":   # crit's kernels: no bit changes (torch's SDPA gives way to the masked softmax)
            self.assertTrue(torch.equal(with_none, plain))
        self.assertLess(float((with_none.double() - plain.double()).abs().max()), 1e-5 if dtype == "fp32" else 1e-9)
        with self.assertRaisesRegex(ValueError, "sinks are contiguous FP32"):
            self.attention.forward(0, q, pool, batch, sinks=sinks[:3])

    def sliding_case(self, dtype, d, window):
        torch = self.torch
        device = self.device()
        pool = self.kv.pool([0], 2, d, 1024, device, dtype, page=16)
        g = torch.Generator().manual_seed(8)
        keys, values = torch.randn(90, 2, d, generator=g), torch.randn(90, 2, d, generator=g)
        pool.open("a")
        for rows in ([("a", 0, 60)], [("a", 60, 30)]):   # a prompt, then a prompt row past the window
            batch = StepBatch.build(rows, device)
            slots = pool.reserve(batch)
            pool.write(0, slots, keys[rows[0][1]:rows[0][1] + rows[0][2]].to(device),
                       values[rows[0][1]:rows[0][1] + rows[0][2]].to(device))
        q = torch.randn(30, 4, d, generator=torch.Generator().manual_seed(9)).to(device)
        sinks = torch.tensor([0.5, -1.0, float("-inf"), 2.0], device=device)
        out = self.attention.forward(0, q, pool, batch, sliding=window, sinks=sinks)
        qd = q.to(torch.bfloat16).double() if dtype == "bf16" else q.double()
        k, v = pool.gather(0, "a", 90)
        k, v = k.double(), v.double()
        for t in range(30):
            at = torch.arange(90, device=device)
            seen = (at <= 60 + t) & (at > 60 + t - window)
            for h in range(4):
                s = (k[:, h // 2] @ qd[t, h]) * d ** -0.5
                s = torch.cat([s.masked_fill(~seen, float("-inf")),
                               torch.tensor([float(sinks[h])], dtype=torch.float64, device=device)])
                want = torch.softmax(s, 0)[:-1] @ v[:, h // 2]
                self.assertLess(float((out[t, h].double() - want).abs().max()), 1.5e-2, (dtype, t, h))

    def test_sliding_windows_on_crits_kernels_and_the_torch_fallback(self):
        for window in (1, 16, 45):
            self.sliding_case("bf16", 64, window)
            self.sliding_case("fp32", 8, window)

    def test_sinks_on_crits_kernels(self):
        self.run_case("bf16", 64)

    def test_sinks_on_the_torch_fallback(self):
        self.run_case("fp32", 8)
