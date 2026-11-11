"""grouped-experts: AWQ and GPTQ expert matrices packed into int4_zp:<group> bundles (code/bundles.py), on the kernel
and the torch fallback, against the FP64 reference of the checkpoint's own values."""
import sys

import test_device as base

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def pack_words(torch, v, order):
    v = v.reshape(*v.shape[:-1], -1, 8).long()
    word = torch.zeros(v.shape[:-1], dtype=torch.int64)
    for p in range(8):
        word |= (v[..., order[p]] & 15) << (4 * p)
    return torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)


def checkpoint(torch, n, k, group, g, format):
    """(stored tensors as the format stores them, FP64 values [n][k])."""
    q = torch.randint(0, 16, (k, n), generator=g)
    z = torch.randint(1, 16, (k // group, n), generator=g)
    s = ((torch.rand(k // group, n, generator=g) + 0.5) / 64).half()
    values = ((q.double() - z.double().repeat_interleave(group, 0)) * s.double().repeat_interleave(group, 0)).t()
    if format == "awq":
        return {"qweight": pack_words(torch, q, AWQ_ORDER), "qzeros": pack_words(torch, z, AWQ_ORDER),
                "scales": s}, values
    seq = list(range(8))
    return {"qweight": pack_words(torch, q.t(), seq).t().contiguous(), "qzeros": pack_words(torch, z - 1, seq),
            "scales": s, "g_idx": torch.arange(k) // group}, values


class Int4ZpBundleTests(base.DeviceExpertsTests):
    """The device tests' helpers; only the tests written here run (test_device runs its own)."""

    def test_awq_and_gptq_experts_match_the_checkpoints_fp64_values(self):
        torch = self.torch
        H, I, E, group = base.H, base.I, 5, 64
        for format in ("awq", "gptq"):
            for kernel in self.paths():
                with self.subTest(format=format, kernel=kernel):
                    experts = self.assemble(base.USE, {"kernel": kernel})["experts"]
                    g = torch.Generator().manual_seed(11)
                    values = {}
                    for e in range(E):
                        gate, gv = checkpoint(torch, I, H, group, g, format)
                        up, uv = checkpoint(torch, I, H, group, g, format)
                        down, dv = checkpoint(torch, H, I, group, g, format)
                        values[e] = (torch.cat([gv, uv]), dv)
                        experts.load(3, e, (base.on(gate, self.device()), base.on(up, self.device())),
                                     base.on(down, self.device()), format)
                    self.assertEqual(experts.layers[3]["format"], f"int4_zp:{group}")
                    module = sys.modules[type(experts).__module__]
                    store, _ = experts._own(3)
                    gu, dn = module.bundles.matrices(store, module.bundles.layout(f"int4_zp:{group}", H, I))
                    for e in range(E):   # the bundles hold the checkpoint's values exactly
                        self.assertTrue(torch.equal(gu[e].double().cpu(), values[e][0]))
                        self.assertTrue(torch.equal(dn[e].double().cpu(), values[e][1]))
                    x, ids, w = self.routes(40, 2, E, seed=3)
                    out = experts.forward(3, x, ids, w).double().cpu()
                    want = torch.zeros(40, H, dtype=torch.float64)
                    xs, idc, wc = x.double().cpu(), ids.cpu(), w.double().cpu()
                    for t in range(40):
                        for j in range(2):
                            e = int(idc[t, j])
                            if e < 0:
                                continue
                            gu_ = xs[t] @ values[e][0].T
                            h = (torch.nn.functional.silu(gu_[:I]) * gu_[I:]).float().bfloat16().double()
                            want[t] += wc[t, j] * (h @ values[e][1].T)
                    err = float((out - want).abs().max() / want.abs().max())
                    self.assertLess(err, 5e-3, (format, kernel, err))
                    # a route's bits alone and in the call
                    alone = experts.forward(3, x[7:8].clone(), ids[7:8].clone(), w[7:8].clone())
                    self.assertTrue(torch.equal(alone[0].cpu(), out[7].float()), (format, kernel))

    def test_act_order_experts_are_refused_and_groups_checked(self):
        torch = self.torch
        experts = self.assemble(base.USE)["experts"]
        g = torch.Generator().manual_seed(2)
        gate_up, _ = checkpoint(torch, 2 * base.I, base.H, 64, g, "gptq")
        down, _ = checkpoint(torch, base.H, base.I, 64, g, "gptq")
        shuffled = dict(gate_up, g_idx=gate_up["g_idx"][torch.randperm(base.H, generator=g)])
        with self.assertRaises(ValueError):
            experts.load(3, 0, shuffled, down, "gptq")
        with self.assertRaises(ValueError):
            experts.layout("int4_zp", base.H, base.I)
        with self.assertRaises(ValueError):
            experts.layout("int4_zp:48", base.H, base.I)


# the inherited tests are base's own; run only these
for name in [n for n in dir(base.DeviceExpertsTests) if n.startswith("test_")]:
    setattr(Int4ZpBundleTests, name, None)



# It inherits DeviceExpertsTests for its helpers; the inherited tests run in test_device, not again here.
for _name in [n for n in dir(base.DeviceExpertsTests) if n.startswith("test") and n not in vars(Int4ZpBundleTests)]:
    setattr(Int4ZpBundleTests, _name, None)
