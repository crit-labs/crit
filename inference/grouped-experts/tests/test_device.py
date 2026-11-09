"""grouped-experts' device form against an FP64 reference of the same decoded weights in every format, through the
registry kernel where it loads (LOCAL_KERNELS) and through the torch fallback (option kernel false): a route's bits the
same alone and batched, in decode rows and prompt-sized calls, on a repeat, over a slot store with a slot table, and in
a captured CUDA graph whose workspace outlives later prompt shapes; a load after a forward; E8M0 bytes 0 and 255
decoded alike on both paths; a model whose layers declare W4A4 and W4A16 run each as declared; what it cannot compute
refused (formats, casts, bundles and stores that do not carry what their layer declares, the declared arithmetic
without the kernel).

Tolerances: max |out - FP64| <= 5e-3 x max |FP64| for 16-bit activations (the BF16 intermediate is what the
reference rounds too; FP32 sums), 3e-2 for W4A4 (an FP32 intermediate a rounding away from an E2M1 boundary takes the
other code)."""
import sys

from crit_plugins.testing import PluginTestCase

KERNEL = "crit-labs/grouped-experts"

USE = {"expert_source": "expert-tensors"}
CASES = [("bf16", "bf16"), ("fp8:channel", "bf16"), ("fp8:block", "bf16"), ("nvfp4", "bf16"), ("nvfp4", "nvfp4"),
         ("int4", "bf16"), ("mxfp4", "bf16")]
H, I = 256, 128


def stored(fmt, rows, cols, g, inputs=None):
    """A random matrix as a checkpoint stores it, by role (what load() takes)."""
    import torch
    if fmt == "bf16":
        return (torch.randn(rows, cols, generator=g) * 0.05).bfloat16()
    if fmt.startswith("fp8"):
        weight = (torch.randn(rows, cols, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
        if fmt == "fp8:channel":
            scale = torch.rand(rows, generator=g) * 0.01 + 0.0005
        else:
            scale = torch.rand(-(-rows // 128), -(-cols // 128), generator=g) * 0.01 + 0.0005
        return {"weight": weight, "scale": scale}
    weight = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g)
    if fmt == "mxfp4":
        return {"weight": weight, "scale": torch.randint(120, 127, (rows, cols // 32), dtype=torch.uint8, generator=g)}
    scale = (torch.randint(5, 9, (rows, cols // 16), generator=g) * 8
             + torch.randint(0, 8, (rows, cols // 16), generator=g)).to(torch.uint8).view(torch.float8_e4m3fn)
    found = {"weight": weight, "scale": scale, "global_scale": torch.rand((), generator=g) * 40 + 20}
    if inputs is not None:
        found["input_scale"] = torch.tensor(inputs)
    return found


def quantize_int8(v, block=32):
    """crit's declared INT8 quantizer (host_int quantize_int8) as values: per `block` values d = amax / 127, each value
    rint(v / d) clamped to 127, times d."""
    import torch
    rows, cols = v.shape
    blocks = v.float().reshape(rows, cols // block, block)
    d = torch.nan_to_num(blocks.abs(), nan=0.0).amax(-1, keepdim=True) / 127.0
    q = torch.where(d > 0, torch.round(blocks / d.clamp(min=1e-38)).clamp(-127, 127), torch.zeros_like(blocks))
    return (q * d).reshape(rows, cols)


def on(found, device):
    """Stored tensors on the device (the bundles land where their tensors are when the engine names no device)."""
    if isinstance(found, dict):
        return {k: v.to(device) for k, v in found.items()}
    return found.to(device)


class DeviceExpertsTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None

    def experts(self, fmt, activations="bf16", kernel=True, E=6, seed=0, arithmetic="tensor", layer=3, inputs=True,
                into=None):
        """A plugin (or `into`) with E experts of `fmt` loaded in `layer`, their matrices carrying input_scale (each
        expert's pair kept in self.inputs) unless inputs is False."""
        torch = self.torch
        experts = into or self.assemble(USE, {"activations": activations, "kernel": kernel,
                                              "arithmetic": arithmetic})["experts"]
        g = torch.Generator().manual_seed(seed)
        self.inputs = {}
        for e in range(E):
            ins = (float(torch.rand(1, generator=g)) * 200 + 300, float(torch.rand(1, generator=g)) * 40 + 20)
            self.inputs[e] = ins
            experts.load(layer, e, on(stored(fmt, 2 * I, H, g, ins[0] if inputs else None), self.device()),
                         on(stored(fmt, H, I, g, ins[1] if inputs else None), self.device()), fmt)
        return experts

    def routes(self, T, top, E, seed, holes=True):
        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        ids = torch.stack([torch.randperm(E, generator=g)[:top] for _ in range(T)])
        if holes:
            ids[torch.rand(T, top, generator=g) < 0.1] = -1
        x = torch.randn(T, H, generator=g).bfloat16()
        device = self.device()
        return x.to(device), ids.to(device), torch.rand(T, top, generator=g).to(device)

    def reference(self, experts, fmt, x, ids, w, a4, int8=False, layer=3):
        """The FP64 block over the bundles' decoded weights, x in BF16 and h rounded as the plugin rounds them."""
        torch = self.torch
        module = sys.modules[type(experts).__module__]
        bundles, fallback = module.bundles, module.fallback
        store, _ = experts._own(layer)
        found = bundles.layout(fmt, H, I)
        store = store.cpu()
        gate_up, down = bundles.matrices(store, found)
        gate_up, down = gate_up.double().cpu(), down.double().cpu()
        xs, out = x.double().cpu(), torch.zeros(x.shape[0], H, dtype=torch.float64)
        for t in range(x.shape[0]):
            for j in range(ids.shape[1]):
                e = int(ids[t, j])
                if e < 0:
                    continue
                g_in, g_dn = self.inputs[e]
                xin = (fallback.quantize(xs[t:t + 1].float(), torch.tensor([g_in])).double()[0] / g_in if a4
                       else quantize_int8(xs[t:t + 1].float()).double()[0] if int8 else xs[t])
                gu = gate_up[e] @ xin
                h = torch.nn.functional.silu(gu[:I]) * gu[I:]
                h = (fallback.quantize(h[None].float(), torch.tensor([g_dn])).double()[0] / g_dn if a4
                     else quantize_int8(h[None].float()).double()[0] if int8 else h.float().bfloat16().double())
                out[t] += float(w[t, j]) * (down[e] @ h)
        return out

    def paths(self):
        return (True, False) if self.device() == "cuda" else (False,)

    def with_kernel(self):
        """A skip where the kernel is not here."""
        if self.device() != "cuda":
            self.skipTest("the kernel runs on CUDA")
        if self.assemble(USE)["experts"].kernel is None:
            self.skipTest(f"no build of {KERNEL} here (LOCAL_KERNELS)")

    def test_every_format_matches_the_fp64_reference(self):
        for kernel in self.paths():
            for fmt, activations in CASES:
                with self.subTest(fmt=fmt, activations=activations, kernel=kernel):
                    experts = self.experts(fmt, activations, kernel)
                    x, ids, w = self.routes(11, 3, 6, 1)
                    out = experts.forward(3, x, ids, w).double().cpu()
                    ref = self.reference(experts, fmt, x, ids, w, activations == "nvfp4")
                    err = float((out - ref).abs().max() / ref.abs().max())
                    self.assertLess(err, 3e-2 if activations == "nvfp4" else 5e-3, (fmt, activations, kernel))

    def test_the_declared_arithmetic_matches_the_fp64_reference_and_keeps_its_bits(self):
        torch = self.torch
        self.with_kernel()
        for fmt in ("nvfp4", "int4"):
            with self.subTest(fmt=fmt):
                experts = self.experts(fmt, "nvfp4" if fmt == "nvfp4" else "bf16", True, E=8, seed=14,
                                       arithmetic="declared")
                x, ids, w = self.routes(9, 3, 8, 15)
                out = experts.forward(3, x, ids, w)
                ref = self.reference(experts, fmt, x, ids, w, fmt == "nvfp4", fmt == "int4")
                self.assertLess(float((out.double().cpu() - ref).abs().max() / ref.abs().max()), 3e-2)
                alone = experts.forward(3, x[4:5].clone(), ids[4:5].clone(), w[4:5].clone())
                self.assertTrue(torch.equal(alone[0], out[4]))
                self.assertEqual(experts.facts()["experts"]["arithmetic"], "declared")

    def test_the_declared_arithmetic_is_refused_without_the_kernel(self):
        with self.assertRaisesRegex(ValueError, f"declared runs only on the registry kernel {KERNEL}.*kernel is false"):
            self.assemble(USE, {"arithmetic": "declared", "activations": "nvfp4", "kernel": False})

    def test_a_route_keeps_its_bits_alone_batched_and_in_a_prompt_call(self):
        torch = self.torch
        for kernel in self.paths():
            for fmt, activations in (("nvfp4", "nvfp4"), ("bf16", "bf16"), ("fp8:block", "bf16")):
                with self.subTest(fmt=fmt, kernel=kernel):
                    experts = self.experts(fmt, activations, kernel, E=16, seed=2)
                    x, ids, w = self.routes(300 if kernel else 40, 4, 16, 3)   # 1200 routes: a prompt-sized call
                    whole = experts.forward(3, x, ids, w)
                    for first, count in ((0, 1), (5, 1), (9, 8), (20, 13)):
                        part = experts.forward(3, x[first:first + count].clone(), ids[first:first + count].clone(),
                                               w[first:first + count].clone())
                        self.assertTrue(torch.equal(part, whole[first:first + count]), (fmt, first, count))
                    self.assertTrue(torch.equal(experts.forward(3, x, ids, w), whole))

    def test_a_slot_store_with_a_slot_table_and_another_tiers_rows(self):
        torch = self.torch
        for kernel in self.paths():
            with self.subTest(kernel=kernel):
                experts = self.experts("nvfp4", "bf16", kernel, E=8, seed=4)
                x, ids, w = self.routes(9, 3, 8, 5, holes=False)
                base = experts.forward(3, x, ids, w)
                own, _ = experts._own(3)
                perm = torch.randperm(8, generator=torch.Generator().manual_seed(6)).to(own.device)
                store = torch.zeros(11, own.shape[1], dtype=torch.uint8, device=own.device)
                store[perm + 3] = own
                slots = (perm + 3).int()
                self.assertTrue(torch.equal(experts.forward_bundles(3, x, ids, w, store, slots, "nvfp4"), base))
                rows = experts.route_outputs(3, x, ids, store, slots, "nvfp4").clone()
                slots[5] = -2   # expert 5 on the CPU tier: skipped here, its rows joined in route order
                skipped = experts.forward_bundles(3, x, ids, w, store, slots, "nvfp4")
                tier = ids.reshape(-1) == 5
                y = experts.route_outputs(3, x, ids, store, slots, "nvfp4").clone()
                y[tier] = rows[tier]
                self.assertTrue(torch.equal(experts.combine(y, ids, w), base))
                if bool(tier.any()):
                    self.assertFalse(torch.equal(skipped, base))

    def test_a_captured_call_replays_new_routes(self):
        torch = self.torch
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        for kernel in self.paths():
            for tokens in (1, 8, 300):
                with self.subTest(kernel=kernel, tokens=tokens):
                    experts = self.experts("nvfp4", "nvfp4", kernel, E=32, seed=7)
                    x, ids, w = self.routes(tokens, 4, 32, 8)
                    out = experts.forward(3, x, ids, w)   # the shape's first call, outside the capture
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out = experts.forward(3, x, ids, w)
                    for seed in (9, 10):
                        nx, nids, nw = self.routes(tokens, 4, 32, seed)
                        x.copy_(nx)
                        ids.copy_(nids)
                        w.copy_(nw)
                        graph.replay()
                        eager = experts.forward(3, x.clone(), ids.clone(), w.clone())
                        self.assertTrue(torch.equal(out, eager))

    def test_the_kernel_and_the_fallback_agree_within_their_sums(self):
        if self.device() != "cuda":
            self.skipTest("the kernel runs on CUDA")
        a, b = self.experts("nvfp4", "nvfp4", True), self.experts("nvfp4", "nvfp4", False)
        if a.kernel is None:
            self.skipTest("no build of crit-labs/grouped-experts here (LOCAL_KERNELS)")
        x, ids, w = self.routes(7, 3, 6, 11)
        fa, fb = a.forward(3, x, ids, w), b.forward(3, x, ids, w)
        self.assertLess(float((fa - fb).abs().max() / fb.abs().max()), 3e-2)
        self.assertEqual(a.facts()["experts"]["path"], "kernel")
        self.assertEqual(b.facts()["experts"]["path"], "torch")

    def test_a_workspace_a_capture_used_outlives_later_prompt_shapes(self):
        torch = self.torch
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        for kernel in self.paths():
            with self.subTest(kernel=kernel):
                experts = self.experts("bf16", "bf16", kernel, E=32, seed=21)
                x, ids, w = self.routes(300, 4, 32, 22)   # 1,200 routes: a prompt-sized workspace
                experts.forward(3, x, ids, w)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = experts.forward(3, x, ids, w)
                graph.replay()
                expected = out.clone()
                key = next(k for k in experts.workspaces if k[:2] == (300, 4))
                for tokens in (400, 500, 600):   # more prompt shapes than the plugin keeps
                    experts.forward(3, *self.routes(tokens, 4, 32, tokens))
                self.assertIn(key, experts.workspaces)
                torch.cuda.synchronize()
                victims = [torch.full((300 * 4, H), 7.0, device="cuda") for _ in range(8)]
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(all(bool((v == 7.0).all()) for v in victims))
                self.assertTrue(torch.equal(out, expected))
                g = torch.Generator().manual_seed(23)
                with self.assertRaisesRegex(ValueError, "captured"):   # a new expert would move the captured store
                    experts.load(3, 32, stored("bf16", 2 * I, H, g).cuda(), stored("bf16", H, I, g).cuda())

    def test_a_load_after_a_forward_keeps_the_layers_other_experts(self):
        torch = self.torch
        device = self.device()
        for kernel in self.paths():
            with self.subTest(kernel=kernel):
                experts = self.experts("nvfp4", "bf16", kernel, E=6, seed=31)
                x, ids, w = self.routes(7, 3, 6, 32)
                experts.forward(3, x, ids, w)
                fresh = self.experts("nvfp4", "bf16", kernel, E=6, seed=31)
                g = torch.Generator().manual_seed(33)
                for e in (2, 6):   # in place, then a new expert (the store rebuilt with every expert)
                    gate_up, down = on(stored("nvfp4", 2 * I, H, g), device), on(stored("nvfp4", H, I, g), device)
                    experts.load(3, e, gate_up, down, "nvfp4")
                    fresh.load(3, e, gate_up, down, "nvfp4")
                    routed = ids.clone()
                    routed[0, 0] = e
                    self.assertTrue(torch.equal(experts.forward(3, x, routed, w), fresh.forward(3, x, routed, w)), e)

    def test_a_mixed_model_runs_each_layer_as_its_stored_matrices_declare(self):
        # One W4A4 layer (its NVFP4 matrices carry input_scale) and one W4A16 layer (they do not), as a checkpoint
        # whose main experts declare W4A4 beside MTP experts that do not: each against FP64, loaded and from a store.
        torch = self.torch
        device = self.device()
        for kernel in self.paths():
            with self.subTest(kernel=kernel):
                experts = self.experts("nvfp4", "nvfp4", kernel, E=6, seed=36, layer=1)
                w4a4_inputs = dict(self.inputs)
                self.experts("nvfp4", "nvfp4", kernel, E=6, seed=37, layer=2, inputs=False, into=experts)
                x, ids, w = self.routes(9, 3, 6, 38)
                outs = {}
                for layer, a4 in ((1, True), (2, False)):
                    self.inputs = w4a4_inputs if a4 else {e: (0.0, 0.0) for e in range(6)}
                    outs[layer] = experts.forward(layer, x, ids, w)
                    ref = self.reference(experts, "nvfp4", x, ids, w, a4, layer=layer)
                    err = float((outs[layer].double().cpu() - ref).abs().max() / ref.abs().max())
                    self.assertLess(err, 3e-2 if a4 else 5e-3, (layer, a4))
                self.assertEqual(experts.facts()["experts"]["layers"],
                                 {"1": {"format": "nvfp4", "activations": "nvfp4"},
                                  "2": {"format": "nvfp4", "activations": "bf16"}})
                # The same two layers from one residency store: bundles packed per layer, a slot table per layer.
                cache = self.assemble(USE, {"activations": "nvfp4", "kernel": kernel})["experts"]
                rows = [experts._own(1)[0], experts._own(2)[0]]
                store = torch.cat(rows)
                for layer, first in ((1, 0), (2, 6)):
                    slots = torch.arange(first, first + 6, dtype=torch.int32, device=device)
                    got = cache.forward_bundles(layer, x, ids, w, store, slots, "nvfp4")
                    self.assertTrue(torch.equal(got, outs[layer]), layer)
                self.assertEqual(cache.facts()["experts"]["layers"]["2"]["activations"], "bf16")

    def test_w4a4_bundles_and_stores_must_carry_what_their_layer_declares(self):
        torch = self.torch
        device = self.device()
        for kernel in self.paths():
            with self.subTest(kernel=kernel):
                experts = self.assemble(USE, {"activations": "nvfp4", "kernel": kernel})["experts"]
                g = torch.Generator().manual_seed(34)
                with self.assertRaisesRegex(ValueError, "or neither does"):   # one matrix of an expert with, one without
                    experts.pack(stored("nvfp4", 2 * I, H, g, 300.0), stored("nvfp4", H, I, g), "nvfp4")
                with_scales = [experts.pack(stored("nvfp4", 2 * I, H, g, 300.0), stored("nvfp4", H, I, g, 30.0),
                                            "nvfp4", layer=0) for _ in range(4)]   # layer 0 declares W4A4
                bare = [experts.pack(stored("nvfp4", 2 * I, H, g), stored("nvfp4", H, I, g), "nvfp4", layer=7)
                        for _ in range(4)]                                        # layer 7 declares W4A16
                with self.assertRaisesRegex(ValueError, "layer 0 mixes"):
                    experts.pack(stored("nvfp4", 2 * I, H, g), stored("nvfp4", H, I, g), "nvfp4", layer=0)
                x, ids, w = self.routes(3, 2, 4, 35, holes=False)
                slots = torch.arange(4, dtype=torch.int32, device=device)
                with self.assertRaisesRegex(ValueError, "lack them"):   # a W4A4 layer's store packed without its scales
                    experts.forward_bundles(0, x, ids, w, torch.stack(bare).to(device), slots, "nvfp4")
                mixed = torch.stack(with_scales[:2] + bare[:2]).to(device)
                with self.assertRaisesRegex(ValueError, "mix input scales"):   # an undeclared layer's bundles disagree
                    experts.forward_bundles(5, x, ids, w, mixed, slots, "nvfp4")
                full = torch.stack(with_scales + [torch.zeros_like(bare[0])]).to(device)
                out = experts.forward_bundles(0, x, ids, w, full, slots, "nvfp4")   # slot 4, empty, is not named
                self.assertTrue(bool(torch.isfinite(out).all()) and bool(out.abs().max() > 0))
                plain = experts.forward_bundles(7, x, ids, w, torch.stack(bare).to(device), slots, "nvfp4")
                self.assertTrue(bool(torch.isfinite(plain).all()) and bool(plain.abs().max() > 0))
                loads = self.assemble(USE, {"activations": "nvfp4", "kernel": kernel})["experts"]
                loads.load(4, 0, stored("nvfp4", 2 * I, H, g, 300.0), stored("nvfp4", H, I, g, 30.0), "nvfp4")
                with self.assertRaisesRegex(ValueError, "layer 4 mixes"):
                    loads.load(4, 1, stored("nvfp4", 2 * I, H, g), stored("nvfp4", H, I, g), "nvfp4")

    def test_mxfp4_scale_bytes_0_and_255_decode_alike_on_both_paths(self):
        torch = self.torch
        device = self.device()
        for kernel in self.paths():
            with self.subTest(kernel=kernel):
                experts = self.assemble(USE, {"kernel": kernel})["experts"]
                g = torch.Generator().manual_seed(41)
                for e in range(2):
                    gate_up, down = stored("mxfp4", 2 * I, H, g), stored("mxfp4", H, I, g)
                    if e == 0:   # gate weights times 2^-127 (BF16 subnormals and their neighbours), up times 1
                        gate_up["scale"][:I], gate_up["scale"][I:] = 0, 127
                    else:        # one gate block NaN
                        gate_up["scale"][1, 1] = 255
                    experts.load(3, e, on(gate_up, device), on(down, device), "mxfp4")
                self.inputs = {0: (0.0, 0.0), 1: (0.0, 0.0)}
                x = (torch.randn(2, H, generator=g) * 2.0 ** 20).bfloat16().to(device)   # products of 2^-127 stay normal
                ids = torch.tensor([[0], [1]], device=device)
                w = torch.ones(2, 1, device=device)
                out = experts.forward(3, x, ids, w).double().cpu()
                ref = self.reference(experts, "mxfp4", x, ids, w, False)
                self.assertTrue(bool(ref[0].abs().max() > 0))
                self.assertLess(float((out[0] - ref[0]).abs().max() / ref[0].abs().max()), 5e-3)
                self.assertTrue(bool(torch.isnan(out[1]).all()) and bool(torch.isnan(ref[1]).all()))

    def test_a_long_chunk_in_parts_and_a_shorter_call_keep_their_bits_in_one_workspace(self):
        """A chunk past a part's tokens runs in parts combined as they go (never a row of every route at once), and a
        prompt-sized call of fewer tokens than a kept prompt workspace runs in its buffers: the same bits as one call,
        and one prompt workspace kept."""
        torch = self.torch
        if self.device() != "cuda":
            self.skipTest("the workspace memory is the device's")
        for kernel in self.paths():
            with self.subTest(kernel=kernel):
                whole = self.experts("bf16", "bf16", kernel, E=32, seed=41)
                x, ids, w = self.routes(3000, 4, 32, 42)
                expected = whole.forward(3, x, ids, w)                    # one call (a part holds all 3,000)
                parts = self.experts("bf16", "bf16", kernel, E=32, seed=41)
                parts.workspace_bytes = 512 * 4 * (H * 4 + I * 2 + 40)    # 512 tokens a part: 5 x 512 + 440
                self.assertEqual(parts._step(4, H, I, 32, False, False, "bf16"), 512)
                found = parts.forward(3, x, ids, w)
                self.assertTrue(torch.equal(found, expected))
                del found
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                found = parts.forward(3, x, ids, w)   # again, its workspaces made: what a call holds besides them
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(found, expected))
                if kernel:   # the kernel's call holds its output and the parts' sums, never a row of every route
                    self.assertLess(torch.cuda.max_memory_allocated() - before, 3000 * 4 * H * 4)
                prompts = [k for k in parts.workspaces if k[0] * k[1] > 1024]
                self.assertEqual([k[0] for k in prompts], [512])          # the 440-token part ran in the 512 one's
                found = parts.forward(3, x[:400].clone(), ids[:400].clone(), w[:400].clone())
                self.assertTrue(torch.equal(found, expected[:400]))
                self.assertEqual([k for k in parts.workspaces if k[0] * k[1] > 1024], prompts)

    def test_long_calls_split_within_the_kernels_items(self):
        experts = self.assemble(USE, {"workspace_mib": 65536})["experts"]
        for top, hidden, inter, count in ((1, 64, 64, 8), (8, 64, 64, 512), (10, 2560, 640, 4096), (2, 64, 64, 60000)):
            step = experts._step(top, hidden, inter, count, False, False, "bf16")
            routes = step * top
            self.assertLessEqual(-(-routes // 8) + min(count, routes), 65535, (top, count, step))

    def test_captured_prompt_buckets_give_the_unpadded_rows_and_their_padding_routes_nowhere(self):
        torch = self.torch
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        from crit_plugins.prompt_graphs import BUCKETS, PromptGraphs
        for kernel in self.paths():
            with self.subTest(kernel=kernel):
                experts = self.experts("nvfp4", "nvfp4", kernel, E=16, seed=51)
                g = torch.Generator().manual_seed(52)
                hidden = torch.randn(97, H, generator=g).bfloat16().cuda()   # each token's hidden row
                routes = torch.stack([torch.randperm(16, generator=g)[:4] for _ in range(97)]).cuda()
                weights = torch.rand(97, 4, generator=g).cuda()

                def forward(batch):   # a model's routed layer over a PromptBatch: its padding NaN, its ids masked
                    t = batch.tokens_t % 97
                    x = torch.where(batch.valid_t[:, None], hidden[t], torch.full_like(hidden[t], float("nan")))
                    ids = routes[t].masked_fill(~batch.valid_t[:, None], -1)
                    return experts.forward(3, x, ids, weights[t])

                prompts = PromptGraphs(forward, device="cuda", buckets=BUCKETS)
                self.assertEqual(prompts.prepare(), list(BUCKETS))
                for count in (5, 16, 40, 128, 300, 512):
                    tokens = [(7 * i + count) % 97 for i in range(count)]
                    out = prompts.run(0, 0, tokens).clone()
                    t = torch.tensor(tokens, device="cuda")
                    alone = experts.forward(3, hidden[t], routes[t], weights[t])
                    self.assertEqual(out.shape[0], min(b for b in BUCKETS if b >= count))
                    self.assertTrue(torch.equal(out[:count], alone), count)
                    self.assertTrue(bool((out[count:] == 0).all()), count)

    def test_formats_it_cannot_compute_are_refused(self):
        torch = self.torch
        experts = self.assemble(USE, {"activations": "nvfp4"})["experts"]
        g = torch.Generator().manual_seed(13)
        with self.assertRaisesRegex(ValueError, "computes"):
            experts.load(0, 0, torch.zeros(4, 64), torch.zeros(64, 2), "fp8:tensor")
        with self.assertRaisesRegex(ValueError, "or neither does"):
            experts.load(0, 0, stored("nvfp4", 2 * I, H, g), stored("nvfp4", H, I, g, 30.0), "nvfp4")
        with self.assertRaisesRegex(ValueError, "multiples of 64"):
            experts.load(0, 0, torch.zeros(2 * 40, 96).bfloat16(), torch.zeros(96, 40).bfloat16(), "bf16")
        plain = self.assemble(USE)["experts"]
        bad = stored("nvfp4", 2 * I, H, g)
        del bad["global_scale"]
        with self.assertRaisesRegex(ValueError, "global_scale"):
            plain.load(0, 0, bad, stored("nvfp4", H, I, g), "nvfp4")
        with self.assertRaisesRegex(KeyError, "no experts of layer 9"):
            plain.forward(9, torch.zeros(1, H), torch.zeros(1, 1, dtype=torch.int64), torch.ones(1, 1))
        with self.assertRaisesRegex(ValueError, "computes"):
            plain.forward_bundles(9, torch.zeros(1, H), torch.zeros(1, 1, dtype=torch.int64), torch.ones(1, 1),
                                  torch.zeros(1, 16, dtype=torch.uint8), None, "fp8:tensor")
        if self.device() == "cuda" and plain.kernel is not None:
            declared = self.assemble(USE, {"arithmetic": "declared"})["experts"]
            with self.assertRaisesRegex(ValueError, "declared computes"):
                declared.load(0, 0, stored("bf16", 2 * I, H, g), stored("bf16", H, I, g), "bf16")
            with self.assertRaisesRegex(ValueError, "activations nvfp4"):
                declared.load(0, 0, stored("nvfp4", 2 * I, H, g, 300.0), stored("nvfp4", H, I, g, 30.0), "nvfp4")

    def test_pack_casts_nothing_and_keeps_fp8_block_grids_apart(self):
        torch = self.torch
        experts = self.assemble(USE)["experts"]
        bundles = sys.modules[type(experts).__module__].bundles
        g = torch.Generator().manual_seed(16)
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            experts.pack(torch.zeros(2 * I, H, dtype=torch.float16), torch.zeros(H, I).bfloat16(), "bf16")
        wide = stored("nvfp4", 2 * I, H, g)
        wide["scale"] = wide["scale"].float()   # E4M3 values as FP32, which a cast to bytes would truncate
        with self.assertRaisesRegex(ValueError, "casts nothing"):
            experts.pack(wide, stored("nvfp4", H, I, g), "nvfp4")
        e5m2 = stored("fp8:channel", 2 * I, H, g)
        e5m2["weight"] = e5m2["weight"].float().to(torch.float8_e5m2)
        with self.assertRaisesRegex(ValueError, "casts nothing"):
            experts.pack(e5m2, stored("fp8:channel", H, I, g), "fp8:channel")
        hidden, inter = 256, 192   # inter 64 mod 128: gate's last block and up's first are halves of their grids
        gate, up = stored("fp8:block", inter, hidden, g), stored("fp8:block", inter, hidden, g)
        weight = torch.cat([gate["weight"].view(torch.uint8), up["weight"].view(torch.uint8)]).view(torch.float8_e4m3fn)
        bundle = experts.pack({"weight": weight, "scale": torch.cat([gate["scale"], up["scale"]])},
                              stored("fp8:block", hidden, inter, g), "fp8:block")
        gate_up, _ = bundles.matrices(bundle[None], bundles.layout("fp8:block", hidden, inter))

        def decoded(part, rows):
            return part["weight"].float() * part["scale"].repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows]

        self.assertTrue(torch.equal(gate_up[0], torch.cat([decoded(gate, inter), decoded(up, inter)])))
        fused = {"weight": weight, "scale": torch.rand(3, 2, generator=g)}   # one grid over the 384 fused rows
        with self.assertRaisesRegex(ValueError, "gate's 128 x 128 grid, then up's"):
            experts.pack(fused, stored("fp8:block", hidden, inter, g), "fp8:block")

