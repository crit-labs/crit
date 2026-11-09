"""host-expert-tier's device form on the CPU: nvfp4 (W4A4) and int4 (W4A8) experts computed in host-expert-arithmetic's
declared integer block order, the native tier and the torch path giving the same bits, a token's result the same alone
as beside other rows, a store of bundles read in place the same as loaded experts, and within the quantization's error
of the dense FP64 block of the same weights."""
import random
import unittest

from crit_plugins.testing import PluginTestCase

try:
    import torch
except ImportError:  # the device form's tensors are torch's
    torch = None

USE = {"expert_source": "expert-tensors"}
HIDDEN, INTER, EXPERTS = 256, 128, 6


def stored(generator, hidden, inter, format, per_row):
    """An expert's stored tensors by role: random codes, E4M3 block scales, global and input scales (divisors)."""
    def matrix(rows, cols, input_scale):
        return {"weight": torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator),
                "scale": torch.randint(0x28, 0x48, (rows, cols // 16), dtype=torch.uint8, generator=generator),
                "global_scale": (torch.rand(rows, generator=generator) * 100 + 50) if per_row else torch.tensor(80.0),
                "input_scale": torch.tensor(input_scale)}
    return matrix(2 * inter, hidden, 12.0), matrix(hidden, inter, 3.0)


def dense(matrix, format):
    """The FP64 weights a stored matrix means (W = value * scale / global)."""
    from importlib import import_module
    codes = matrix["weight"]
    low, high = (codes & 15).long(), (codes >> 4).long()
    nib = torch.stack([low, high], -1).reshape(codes.shape[0], -1)
    if format == "int4":
        values = torch.where(nib < 8, nib, nib - 16).double()
    else:
        magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
        values = torch.where(nib < 8, magnitude[nib & 7], -magnitude[nib & 7])
    table = torch.tensor([float(torch.tensor(b, dtype=torch.uint8).view(torch.float8_e4m3fn).float()) for b in range(256)],
                         dtype=torch.float64)
    scales = table[matrix["scale"].long()].repeat_interleave(16, dim=1)
    g = matrix["global_scale"].double().reshape(-1, 1)
    return values * scales / g


@unittest.skipIf(torch is None, "the device form's tensors are torch's")
class HostTierDeviceTests(PluginTestCase):
    def experts(self, format="nvfp4", native="auto", seed=7):
        parts = self.assemble(USE, {"native": native, "workers": 3, "pin": False})
        experts, generator = parts["experts"], torch.Generator().manual_seed(seed)
        matrices = {}
        for e in range(EXPERTS):
            gate_up, down = stored(generator, HIDDEN, INTER, format, per_row=e % 2 == 0)
            matrices[e] = (gate_up, down)
            experts.load(3, e, gate_up, down, format)
        return experts, matrices

    def routes(self, seed=11, tokens=5, top=4):
        rng = random.Random(seed)
        ids = torch.tensor([rng.sample(range(EXPERTS), top) for _ in range(tokens)], dtype=torch.int64)
        ids[1, 2] = -1
        weights = torch.rand(tokens, top, generator=torch.Generator().manual_seed(seed))
        x = torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(seed + 1)).bfloat16()
        return x, ids, weights

    def test_it_has_the_device_form_and_bundles(self):
        self.assertImplements(self.assemble(USE))

    def test_each_route_is_the_declared_order_and_the_sum_is_in_route_order(self):
        for format in ("nvfp4", "int4"):
            with self.subTest(format=format):
                experts, _ = self.experts(format)
                x, ids, weights = self.routes()
                out = experts.forward(3, x, ids, weights)
                tier = experts._tier(format, HIDDEN, INTER)
                y = torch.zeros(ids.numel(), HIDDEN)
                for r, e in enumerate(ids.reshape(-1).tolist()):
                    if e >= 0:
                        y[r] = tier.expert_rows(experts.keys[(3, e)], x[r // ids.shape[1]:r // ids.shape[1] + 1].float())[0]
                expected = tier.declared.combine(y, ids, weights)
                self.assertTrue(torch.equal(out.view(torch.int32), expected.view(torch.int32)))

    def test_the_native_tier_and_the_torch_path_give_the_same_bits(self):
        for format in ("nvfp4", "int4"):
            with self.subTest(format=format):
                native, _ = self.experts(format, "auto")
                if not native._tier(format, HIDDEN, INTER).native:
                    self.skipTest("the native tier does not build here")
                fallback, _ = self.experts(format, "off")
                x, ids, weights = self.routes(seed=5, tokens=7, top=6)
                self.assertTrue(torch.equal(native.forward(3, x, ids, weights), fallback.forward(3, x, ids, weights)))
                self.assertIn("native", native.facts()["cpu_expert_rows"])
                self.assertIn("torch", fallback.facts()["cpu_expert_rows"])

    def test_a_token_alone_gives_its_bits_beside_other_rows(self):
        experts, _ = self.experts()
        x, ids, weights = self.routes(seed=3, tokens=8, top=6)
        batched = experts.forward(3, x, ids, weights)
        for t in range(8):
            alone = experts.forward(3, x[t:t + 1].clone(), ids[t:t + 1].clone(), weights[t:t + 1].clone())
            self.assertTrue(torch.equal(alone[0], batched[t]), f"token {t}")

    def test_fp32_or_bf16_x_give_the_same_bits(self):
        experts, _ = self.experts()
        x, ids, weights = self.routes(seed=13)
        xf = x.float() + 1e-3 * torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(experts.forward(3, xf, ids, weights), experts.forward(3, xf.bfloat16(), ids, weights)))

    def test_a_store_of_bundles_read_in_place_gives_the_loaded_experts_bits(self):
        experts, matrices = self.experts()
        store = torch.stack([experts.pack(*matrices[e], "nvfp4") for e in range(EXPERTS)][::-1])
        slots = torch.tensor([EXPERTS - 1 - e for e in range(EXPERTS)], dtype=torch.int32)
        self.assertEqual(store.shape[1], experts.layout("nvfp4", HIDDEN, INTER)["bundle_bytes"])
        x, ids, weights = self.routes()
        self.assertTrue(torch.equal(experts.forward_bundles(3, x, ids, weights, store, slots, "nvfp4"),
                                    experts.forward(3, x, ids, weights)))

    def test_it_is_within_the_quantization_error_of_the_dense_block(self):
        for format, bound in (("nvfp4", 0.25), ("int4", 0.08)):
            with self.subTest(format=format):
                experts, matrices = self.experts(format)
                x, ids, weights = self.routes(seed=9, tokens=6, top=4)
                out = experts.forward(3, x, ids, weights).double()
                want = torch.zeros_like(out)
                for t in range(ids.shape[0]):
                    for j in range(ids.shape[1]):
                        e = int(ids[t, j])
                        if e < 0:
                            continue
                        gu, dn = (dense(m, format) for m in matrices[e])
                        g, u = (gu @ x[t].double()).chunk(2)
                        want[t] += float(weights[t, j]) * (dn @ (torch.nn.functional.silu(g) * u))
                error = (out - want).norm() / want.norm()
                self.assertLess(float(error), bound)

    def test_it_refuses_a_format_without_a_declared_integer_order(self):
        experts = self.assemble(USE)["experts"]
        with self.assertRaisesRegex(ValueError, "nvfp4, int4"):
            experts.load(0, 0, torch.zeros(2 * INTER, HIDDEN), torch.zeros(HIDDEN, INTER), "bf16")
        gate_up, down = stored(torch.Generator().manual_seed(1), HIDDEN, INTER, "nvfp4", True)
        del gate_up["input_scale"]
        with self.assertRaisesRegex(ValueError, "input scales"):
            experts.load(0, 0, gate_up, down, "nvfp4")
