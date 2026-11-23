"""CSF-compressed scales: expert_store.csf_decode gives back what csf_encode (and an independent encoder of ModelOpt's
layout, crit_core.model.schemes csf_scale_shape) compressed, exceptions included, on the host and on the device; a
bundle store that keeps the scales compressed (the expert_source plugin's csf_scales option: put() encodes them from the
slot-layout bundle, or takes the checkpoint's, whole or as row-stacked parts) fills the cache's slots with the slot
layout's bundles byte for byte, by the CUDA program fill and by the torch ops, in a captured plan, a prompt chunk and a
preload; a plain store takes no compressed scales."""
from pathlib import Path
import random
import sys

from crit_plugins.testing import PluginTestCase

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import expert_store  # noqa: E402


def encode(scales):
    """An encoder written from the format's description alone (not csf_encode's code)."""
    torch = __import__("torch")
    rows, groups = scales.shape
    base = scales.min(1).values
    codes = scales.int() - base.int()[:, None]
    words = []
    for r, c in (codes > 15).nonzero().tolist():
        words.append((r * groups + c) | (int(scales[r, c]) << 24))
    codes = torch.where(codes > 15, torch.zeros_like(codes), codes).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(rows // 16, 16, groups // 2)
    fixed = torch.cat([base.view(rows // 16, 16), packed.reshape(rows // 16, -1)], 1)
    exceptions = torch.tensor([w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=torch.int32)
    return fixed.contiguous(), exceptions


def scales_like(rng, rows, groups):
    torch = __import__("torch")
    return torch.tensor([[rng.randrange(100, 112) if rng.random() > 0.03 else rng.randrange(0, 256)
                          for _ in range(groups)] for _ in range(rows)], dtype=torch.uint8)


# A stand-in slot layout of the experts plugin's NVFP4 bundle (hidden 64, inter 32): codes, E4M3 scales, inverses.
LAYOUT = {"bundle_bytes": 0, "components": {}}
_at = 0
for _name, _dtype, _shape in (("gate_up", "uint8", (64, 32)), ("gate_up_scale", "uint8", (64, 4)),
                              ("gate_up_inverse", "float32", (64,)), ("down", "uint8", (64, 16)),
                              ("down_scale", "uint8", (64, 2)), ("down_inverse", "float32", (64,)),
                              ("input", "float32", (4,))):
    LAYOUT["components"][_name] = (_at, _dtype, _shape)
    _at += -(-expert_store._component_bytes(_dtype, _shape) // 16) * 16
LAYOUT["bundle_bytes"] = _at


class CsfTests(PluginTestCase):
    def test_csf_scales_decode_to_the_scales_encoded(self):
        torch = __import__("torch") if self.device() else None
        rng = random.Random(5)
        for rows, groups in ((16, 2), (64, 160), (640, 40)):
            scales = scales_like(rng, rows, groups)
            from crit_core.model.schemes import csf_scale_shape
            for fixed, exceptions in (encode(scales), expert_store.csf_encode(scales)):
                self.assertEqual(csf_scale_shape(tuple(fixed.shape)), (rows, groups))
                for device in {"cpu", self.device()}:
                    out = expert_store.csf_decode(fixed.to(device), exceptions.to(device))
                    self.assertTrue(torch.equal(out.cpu(), scales), (rows, groups, device))
                stacked = expert_store.csf_decode(torch.stack([fixed, fixed]), torch.stack([exceptions, exceptions]))
                self.assertTrue(torch.equal(stacked[1], scales))
        with self.assertRaisesRegex(ValueError, "CSF"):
            expert_store.csf_decode(torch.zeros(1, 20, dtype=torch.uint8))
        self.assertEqual(expert_store.csf_scales(" gate_up_scale:256, down_scale:0 "),
                         {"gate_up_scale": 256, "down_scale": 0})
        self.assertIsNone(expert_store.csf_scales(""))
        with self.assertRaisesRegex(ValueError, "component:capacity"):
            expert_store.csf_scales("gate_up_scale")
        gate, up = scales_like(rng, 32, 40), scales_like(rng, 48, 40)   # a checkpoint's gate and up scales
        fixed, exceptions = expert_store.csf_stack([expert_store.csf_encode(gate), expert_store.csf_encode(up)])
        self.assertTrue(torch.equal(expert_store.csf_decode(fixed, exceptions), torch.cat([gate, up])))

    def test_a_csf_store_fills_slots_with_the_slot_layouts_bundles(self):
        torch = __import__("torch") if self.device() else None
        device = self.device()
        rng = random.Random(9)
        g = torch.Generator().manual_seed(9)
        layers, experts = 2, 12
        csf = {"expert-tensors": {"csf_scales": "gate_up_scale:256,down_scale:128"}}
        slot_bundles = {}
        for kernel in ((True, False) if device == "cuda" else (False,)):
            source = self.assemble(use={"expert_source": "expert-tensors"}, use_options=csf)["expert_source"]
            store = source.open_store(LAYOUT, layers, experts, device)   # the slot layout: the option compresses
            self.assertEqual(store.compressed, ("down_scale", "gate_up_scale"))
            self.assertEqual(store.layout["slot_bytes"], LAYOUT["bundle_bytes"])
            self.assertLess(store.bundle_bytes, LAYOUT["bundle_bytes"] + (256 + 128) * 4 + 32)
            for l in range(layers):
                for e in range(experts):
                    if (l, e) not in slot_bundles:
                        bundle = torch.randint(0, 256, (LAYOUT["bundle_bytes"],), dtype=torch.uint8, generator=g)
                        compressed = {}
                        for name in ("gate_up_scale", "down_scale"):
                            offset, _, shape = LAYOUT["components"][name]
                            scales = scales_like(rng, *shape)
                            bundle[offset:offset + scales.numel()] = scales.view(-1)
                            compressed[name] = expert_store.csf_encode(scales)
                        slot_bundles[(l, e)] = (bundle, compressed)
                    bundle, compressed = slot_bundles[(l, e)]
                    if e % 3 == 0:     # the checkpoint's CSF scales, as they are
                        store.put(l, e, bundle, compressed)
                    elif e % 3 == 1:   # gate and up scales as two CSF tensors, stacked by rows
                        offset, _, shape = LAYOUT["components"]["gate_up_scale"]
                        rows = bundle[offset:offset + shape[0] * shape[1]].view(shape)
                        halves = [expert_store.csf_encode(rows[:32]), expert_store.csf_encode(rows[32:])]
                        store.put(l, e, bundle, {"gate_up_scale": halves})
                    else:              # encoded by the store from the bundle's own scale bytes
                        store.put(l, e, bundle)
            plugin = self.assemble(options={"native": kernel, "ring_slots": 12})["expert_residency"]
            cache = plugin.cache(layers, experts, LAYOUT, 28, device, store)

            def check(layer, table, routed):
                for e in routed:
                    self.assertTrue(torch.equal(cache.store[table[e]].cpu(), slot_bundles[(layer, e)][0]),
                                    (kernel, layer, e))

            for _ in range(10):
                layer = rng.randrange(layers)
                ids = [rng.randrange(experts) for _ in range(5)]
                check(layer, cache.plan(layer, torch.tensor(ids, device=device)).tolist(), ids)
            ids = [rng.randrange(experts) for _ in range(6)]
            _, table = cache.prompt(1, torch.tensor(ids, device=device).view(3, 2))
            check(1, table.tolist(), ids)
            for layer in (0, 1, 0):   # long chunks: every non-resident expert staged (the copy engine on CUDA)
                ids = [rng.randrange(experts) for _ in range(24)]
                _, table = cache.prompt(layer, torch.tensor(ids, device=device).view(12, 2))
                if device == "cuda":
                    torch.cuda.synchronize()
                check(layer, table.tolist(), range(experts) if device == "cuda" else ids)   # staged: every expert
            cache.end_prompt()
            fresh = plugin.cache(layers, experts, LAYOUT, 28, device, store)
            pairs = fresh.preload([3, 14, 20])
            for key, slot in pairs:
                self.assertTrue(torch.equal(fresh.store[slot].cpu(), slot_bundles[divmod(key, experts)][0]))
            if device == "cuda":
                ids_t = torch.tensor([1, 2, 3, 4], device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    cache.plan(0, ids_t)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        out = cache.plan(0, ids_t)
                for _ in range(4):
                    routed = rng.sample(range(experts), 4)
                    ids_t.copy_(torch.tensor(routed))
                    graph.replay()
                    torch.cuda.synchronize()
                    check(0, out.tolist(), routed)

    def test_stores_take_slot_layout_bundles_and_refuse_what_they_do_not_keep(self):
        torch = __import__("torch") if self.device() else None
        device = self.device()
        bundle = torch.zeros(LAYOUT["bundle_bytes"], dtype=torch.uint8)
        offset, _, shape = LAYOUT["components"]["down_scale"]
        bundle[offset:offset + shape[0] * shape[1]] = 100
        plain = self.assemble(use={"expert_source": "expert-tensors"})["expert_source"].open_store(LAYOUT, 1, 2, device)
        self.assertEqual(plain.compressed, ())
        with self.assertRaisesRegex(ValueError, "keeps no compressed scales"):
            plain.put(0, 0, bundle, {"down_scale": expert_store.csf_encode(torch.full(shape, 100, dtype=torch.uint8))})
        source = self.assemble(use={"expert_source": "expert-tensors"},
                               use_options={"expert-tensors": {"csf_scales": "down_scale:0"}})["expert_source"]
        packed = source.open_store(LAYOUT, 1, 2, device)
        with self.assertRaisesRegex(ValueError, "not \\['gate_up_scale'\\]"):
            packed.put(0, 0, bundle, {"gate_up_scale": expert_store.csf_encode(torch.zeros(64, 4, dtype=torch.uint8))})
        with self.assertRaisesRegex(ValueError, "slot-layout bundle"):
            packed.put(0, 0, bundle[:-16])
        packed.put(0, 0, bundle)   # down_scale's bytes all within 15 of their row's base: no exception words needed
        bundle[offset] = 200
        with self.assertRaisesRegex(ValueError, "exceptions, the store holds 0"):
            packed.put(0, 1, bundle)
        files = self.assemble(use={"expert_source": "ssd-expert-materialization"},
                              use_options={"ssd-expert-materialization": {"csf_scales": "down_scale:4"}})
        tier = files["expert_source"].open_store(LAYOUT, 1, 2, device)   # the file tier keeps them compressed too
        self.assertEqual((tier.compressed, tier.bundle_bytes), (("down_scale",), packed.bundle_bytes + 16))
        tier.put(0, 1, bundle)
        into = torch.empty(1, tier.bundle_bytes, dtype=torch.uint8)
        tier.read([1], into)
        self.assertTrue(torch.equal(into[0], expert_store.store_bundle(tier.layout, bundle)))
