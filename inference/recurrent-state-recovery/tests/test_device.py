"""recurrent-state-recovery's device form against its reference's (sequential-recurrence): decode and verify rows give
the same bits, prompt rows (its two-pass chunked prefill where recurrent-scans is installed, the scan otherwise) agree
with the float64 recurrence within 2e-3 + 2e-2 |x|, and its three commit strategies keep a verify window's accepted
prefix with the same bits, which continue as the base plugin's serial decode."""
from crit_plugins import StepBatch
from crit_plugins.device import RoundBatch
from crit_plugins.testing import PluginTestCase

PROMPT = dict(atol=2e-3, rtol=2e-2)


class View:
    def __init__(self, rows, width, n_emitted):
        self.rows, self.width, self.n_emitted = rows, width, n_emitted


class DeviceTests(PluginTestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.dev = self.device()
        g = torch.Generator().manual_seed(31)
        self.kh, self.h, self.dk, self.dv = (2, 6, 64, 64) if self.dev == "cuda" else (1, 2, 8, 4)
        T = 160
        self.inputs = {"q": torch.randn(T, self.kh, self.dk, generator=g), "k": torch.randn(T, self.kh, self.dk, generator=g),
                       "v": torch.randn(T, self.h, self.dv, generator=g), "g": -torch.rand(T, self.h, generator=g) * 0.5,
                       "b": torch.rand(T, self.h, generator=g), "x": torch.randn(T, 24, generator=g)}
        self.inputs = {k: v.to(self.dev) for k, v in self.inputs.items()}
        self.weight = torch.randn(24, 4, generator=g).to(self.dev)
        self.geometry = {"keys": self.dk, "values": self.dv, "epsilon": 1e-6, "query_scale": self.dk ** -0.5,
                         "normalize_keys": True, "normalize_queries": True}

    def pool(self, plugin, widest=0):
        pool = plugin.pool([0], self.h, self.dk, self.dv, self.dev, 4, widest, (24, 4), self.kh, 1)
        pool.open("a")
        return pool

    def rows(self, plugin, pool, batch, first, count, accept=True):
        span = slice(first, first + count)
        x = plugin.convolve(0, self.inputs["x"][span], self.weight, None, pool, batch)
        out = plugin.forward(0, *(self.inputs[n][span] for n in "qkvgb"), pool, batch, self.geometry)
        if accept and not batch.round:
            pool.accept("a", count)
        return out, x

    def step(self, plugin, pool, first, count):
        return self.rows(plugin, pool, StepBatch.build([("a", first, count)], self.dev), first, count)

    def test_it_has_the_device_form_and_the_rounds_methods(self):
        self.assertImplements()

    def test_decode_rows_have_the_bits_of_the_reference_plugins(self):
        torch = self.torch
        mine, base = self.assemble()["recurrence"], self.reference()["recurrence"]
        pools = (self.pool(mine), self.pool(base))
        for t in range(6):
            (a, xa), (b, xb) = (self.step(p, pool, t, 1) for p, pool in zip((mine, base), pools))
            self.assertTrue(torch.equal(a, b) and torch.equal(xa, xb))
        self.assertTrue(torch.equal(pools[0].state(0, "a"), pools[1].state(0, "a")))

    def test_prompt_rows_follow_the_float64_recurrence_and_the_reference_plugin(self):
        torch = self.torch
        pinned = {"exact_rows": 16}   # 150 tokens through the chunked kernels, wherever the crossover is
        mine = self.assemble(options=pinned)["recurrence"]
        base = self.reference(options={"sequential-recurrence": pinned})["recurrence"]
        found = {}
        for name, plugin in (("mine", mine), ("base", base)):
            pool = self.pool(plugin)
            out, _ = self.step(plugin, pool, 0, 150)
            found[name] = (out, pool.state(0, "a").clone())
        d = {n: t[:150].double() for n, t in self.inputs.items()}
        normal = lambda x: x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)  # noqa: E731
        S = torch.zeros(self.h, self.dk, self.dv, dtype=torch.float64, device=self.dev)
        ratio, ref = self.h // self.kh, []
        for t in range(150):
            k = normal(d["k"][t]).repeat_interleave(ratio, 0)
            q = normal(d["q"][t]).repeat_interleave(ratio, 0) * self.dk ** -0.5
            e = torch.exp(d["g"][t])
            delta = d["b"][t][:, None] * (d["v"][t] - e[:, None] * torch.einsum("hk,hkv->hv", k, S))
            S = e[:, None, None] * S + k[:, :, None] * delta[:, None, :]
            ref.append(torch.einsum("hk,hkv->hv", q, S))
        ref = torch.stack(ref)
        for name in found:
            torch.testing.assert_close(found[name][0].double(), ref, **PROMPT)
            torch.testing.assert_close(found[name][1].double(), S, **PROMPT)
        torch.testing.assert_close(found["mine"][0], found["base"][0], **PROMPT)

    def test_every_strategy_keeps_the_accepted_prefix_with_the_same_bits(self):
        torch = self.torch
        width = 3
        serial_plugin = self.reference()["recurrence"]
        for n in range(width + 2):
            serial = self.pool(serial_plugin)
            expected = [self.step(serial_plugin, serial, t, 1) for t in range(2 + n + 2)]
            for commit in ("deferred", "checkpoints", "recompute"):
                with self.subTest(n=n, commit=commit):
                    plugin = self.assemble(options={"commit": commit})["recurrence"]
                    pool = self.pool(plugin, widest=width)
                    for t in range(2):
                        self.step(plugin, pool, t, 1)
                    pool.bind(0, "a")
                    batch = RoundBatch(1, width, self.dev).fill(
                        1, width, torch.tensor([2], device=self.dev), torch.zeros(1, width + 1, dtype=torch.int64, device=self.dev))
                    out, x = self.rows(plugin, pool, batch, 2, width + 1)
                    pool.commit_round(View(1, width, torch.tensor([n], device=self.dev)))
                    pool.unbind(0)
                    for c in range(n):
                        self.assertTrue(torch.equal(out[c:c + 1], expected[2 + c][0]))
                        self.assertTrue(torch.equal(x[c:c + 1], expected[2 + c][1]))
                    for t in range(2):
                        got = self.step(plugin, pool, 2 + n + t, 1)
                        self.assertTrue(torch.equal(got[0], expected[2 + n + t][0]))
                        self.assertTrue(torch.equal(got[1], expected[2 + n + t][1]))
                    self.assertTrue(torch.equal(pool.state(0, "a"), serial.state(0, "a")))
                    self.assertTrue(torch.equal(pool.window(0, "a"), serial.window(0, "a")))
