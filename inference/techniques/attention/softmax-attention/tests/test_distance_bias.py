"""attention:distance-bias: forward(..., bias=) adds each token's own scores to the positions just behind it, by how far
behind (nothing further back), with sinks, sliding windows and log-sum-exps, for decode rows and prompt rows alike, on
the device form's torch readers (the CPU's, and the GPU's for any call with a bias)."""

from crit_plugins import StepBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}


class DistanceBiasTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch")
        self.where = self.device() or "cpu"
        self.reader("prefill")

    def reader(self, prompt_reader):
        """prefill: each row's gathered positions (the masked softmax); decode: the passes over the page tables that
        rounds and captured prompt chunks read through."""
        self.parts = self.assemble(USE, options={"prompt_reader": prompt_reader})
        self.kv, self.attention = self.parts["kv_cache"], self.parts["attention"]

    def case(self, window, extent, sinks, dtype="fp32"):
        torch, device = self.torch, self.where
        d = 8 if dtype == "fp32" else 64   # bf16 pages: crit's kernels read them (d a multiple of 16)
        pool = self.kv.pool([0], 2, d, 1024, device, dtype, page=16)
        g = torch.Generator().manual_seed(11)
        keys = {s: torch.randn(80, 2, d, generator=g) for s in "ab"}
        values = {s: torch.randn(80, 2, d, generator=g) for s in "ab"}
        for s in "ab":
            pool.open(s)

        def write(rows):
            batch = StepBatch.build(rows, device)
            slots = pool.reserve(batch)
            pool.write(0, slots, torch.cat([keys[s][p:p + n] for s, p, n in rows]).to(device),
                       torch.cat([values[s][p:p + n] for s, p, n in rows]).to(device))
            return batch

        write([("a", 0, 40), ("b", 0, 20)])
        batch = write([("a", 40, 1), ("b", 20, 30)])   # a decode row and a prompt row
        q = torch.randn(batch.tokens, 4, d, generator=torch.Generator().manual_seed(12)).to(device)
        if dtype == "bf16":
            q = q.to(torch.bfloat16)
        bias = (torch.randn(batch.tokens, 4, extent, generator=torch.Generator().manual_seed(13)) * 2).to(device)
        given = None if sinks is None else torch.tensor(sinks, device=device)
        out, lse = self.attention.forward(0, q, pool, batch, lse=True, bias=bias, sinks=given, sliding=window)
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            k, v = (x.double() for x in pool.gather(0, sequence, position + count))
            tol = 1e-4 if dtype == "fp32" else 1.5e-2
            for t in range(count):
                at = torch.arange(position + t + 1, device=device)
                behind = position + t - at
                for h in range(4):
                    s = (k[: position + t + 1, h // 2] @ q[first + t, h].double()) * d ** -0.5
                    near = behind < extent
                    s = s + torch.where(near, bias[first + t, h].double()[behind.clamp(max=extent - 1)], 0.0)
                    if window:
                        s = s.masked_fill(behind >= window, float("-inf"))
                    if given is not None:
                        s = torch.cat([s, torch.tensor([sinks[h]], dtype=torch.float64, device=device)])
                    p = torch.softmax(s, 0)
                    want = p[:position + t + 1] @ v[: position + t + 1, h // 2]
                    self.assertLess(float((out[first + t, h].double() - want).abs().max()), tol,
                                    (dtype, window, extent, r, t, h))
                    self.assertLess(abs(float(lse[first + t, h]) - float(torch.logsumexp(s, 0))), tol)
        # A bias of zeros is no bias.
        plain = self.attention.forward(0, q, pool, batch, sinks=given, sliding=window)
        zeros = self.attention.forward(0, q, pool, batch, sinks=given, sliding=window, bias=torch.zeros_like(bias))
        self.assertLess(float((zeros.double() - plain.double()).abs().max()), 1e-5 if dtype == "fp32" else 2e-2)

    def test_a_bias_behind_each_token_with_and_without_a_window_and_sinks(self):
        for reader in ("prefill", "decode"):
            self.reader(reader)
            for window, extent, sinks in ((0, 16, None), (0, 64, [0.5, 6.0, float("-inf"), -1.0]),
                                          (16, 16, [1.0] * 4), (24, 7, None)):
                for dtype in ("fp32",) + (("bf16",) if self.where != "cpu" else ()):
                    with self.subTest(reader=reader, window=window, extent=extent, dtype=dtype):
                        self.case(window, extent, sinks, dtype)

    def test_a_bias_of_another_shape_or_with_block_lists_is_refused(self):
        torch, device = self.torch, self.where
        pool = self.kv.pool([0], 2, 8, 256, device, "fp32", page=16)
        pool.open("a")
        batch = StepBatch.build([("a", 0, 4)], device)
        pool.write(0, pool.reserve(batch), torch.zeros(4, 2, 8, device=device), torch.zeros(4, 2, 8, device=device))
        q = torch.zeros(4, 4, 8, device=device)
        with self.assertRaisesRegex(ValueError, "bias is contiguous FP32"):
            self.attention.forward(0, q, pool, batch, bias=torch.zeros(4, 3, 5, device=device))
        with self.assertRaisesRegex(ValueError, "no latent rows or block lists"):
            self.attention.forward(0, q, pool, batch, bias=torch.zeros(4, 4, 5, device=device),
                                   blocks=torch.zeros(4, 1, dtype=torch.int32, device=device), block_size=16)
