"""device-expert-cache over opaque byte rows (experts:rows): a mixture of experts whose experts are stored in a layout
of its own (int8 weights and FP16 scales per output, 27 bytes an expert: no grouped-experts bundle, no multiple of
16) keeps them in row_store(), plans each layer's routes through row_cache() as cache() plans bundles, and reads its
experts back from cache.rows(): every slot a plan hands out holds its key's row byte for byte, and the block's output
over the cached rows equals its output over the host rows. The directory's rule (hits, fills, telemetry) is the
bundles' rule, which test_device.py holds to its host twin."""
import random

from crit_plugins.testing import PluginTestCase

IN, OUT = 7, 3
ROW = IN * OUT + 2 * OUT   # int8 weights, then an FP16 scale per output: 27 bytes


class OwnLayoutExperts:
    """A toy experts block in its own row layout: row = W (int8 [OUT][IN], row-major) then scales (FP16 [OUT])."""

    def __init__(self, torch, layers, experts, seed=0):
        self.torch = torch
        g = torch.Generator().manual_seed(seed)
        self.weights = torch.randint(-127, 128, (layers * experts, OUT, IN), dtype=torch.int8, generator=g)
        self.scales = (torch.rand(layers * experts, OUT, generator=g) / 64).to(torch.float16)

    def rows(self):
        torch = self.torch
        return torch.cat([self.weights.view(len(self.weights), -1).view(torch.uint8),
                          self.scales.view(torch.uint8).view(len(self.scales), -1)], 1)

    def decode(self, rows):
        """[n][ROW] uint8 rows -> (W [n][OUT][IN] FP32, scales [n][OUT] FP32), from the layout's bytes alone."""
        torch = self.torch
        rows = rows.contiguous()
        w = rows[:, :IN * OUT].view(torch.int8).view(-1, OUT, IN).float()
        s = rows[:, IN * OUT:ROW].contiguous().view(torch.float16).float()
        return w, s

    def forward(self, rows, x, ids, weights):
        """sum_k weights[t, k] * (W_ids[t, k] x[t]) * scales: rows indexed by the slots the ids map to."""
        w, s = self.decode(rows)
        out = self.torch.einsum("tkoi,ti->tko", w[ids], x) * s[ids]
        return (out * weights.unsqueeze(-1)).sum(1)


def _device(torch):
    """CUDA where torch has it, else the CPU: the cache's torch ops run there too, so the row interface is checked on
    a host without a GPU (the plugin serves on CUDA alone)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class HostRows:
    """A row source of the caller's own (row_cache takes any store with keys, bundle_bytes, layout, mapped and read()):
    padded rows in a host tensor, read on the host, as a file tier's are."""

    mapped = False

    def __init__(self, torch, rows, slot):
        self.keys, self.bundle_bytes, self.layout = len(rows), slot, {"bundle_bytes": slot}
        self.all = torch.zeros(len(rows), slot, dtype=torch.uint8)
        self.all[:, :rows.shape[1]] = rows

    def read(self, keys, into):
        for i, key in enumerate(keys):
            into[i].copy_(self.all[key])


class RowCacheTests(PluginTestCase):
    def row_store(self, residency, layers, experts, device, rows):
        """residency's row_store(), where host memory regions can be made (libc's mmap: not on Windows)."""
        try:
            store = residency.row_store(layers, experts, ROW, device, rows)
        except Exception as exc:
            if type(exc).__name__ == "HostMemoryUnavailable":
                self.skipTest(str(exc))
            raise
        self.addCleanup(store.close)
        return store

    def test_it_provides_the_rows_extension(self):
        from crit_plugins.device import extension_missing
        self.assertEqual(extension_missing("experts:rows", self.assemble()["expert_residency"]), [])

    def test_an_own_layout_reads_its_experts_back_from_the_cache_of_its_row_store(self):
        import torch
        device = _device(torch)
        block = OwnLayoutExperts(torch, 3, 12)
        residency = self.assemble(options={"native": False, "ring_slots": 0})["expert_residency"]
        store = self.row_store(residency, 3, 12, device, block.rows())
        self.assertEqual((store.row_bytes, store.bundle_bytes), (ROW, 32))
        self.check_layout(residency, block, store, device)

    def test_an_own_layout_reads_its_experts_back_from_the_cache_of_a_source_of_its_own(self):
        import torch
        device = _device(torch)
        block = OwnLayoutExperts(torch, 3, 12, seed=1)
        residency = self.assemble(options={"native": False, "ring_slots": 0})["expert_residency"]
        self.check_layout(residency, block, HostRows(torch, block.rows(), 32), device)

    def check_layout(self, residency, block, store, device):
        import torch
        layers, experts, slots, top = 3, 12, 20, 2
        cache = residency.row_cache(layers, experts, ROW, slots, device, store)
        self.assertEqual(tuple(cache.rows().shape[1:]), (ROW,))
        host = block.rows()
        rng = random.Random(4)
        for _ in range(40):
            layer, tokens = rng.randrange(layers), rng.choice([1, 2, 3])
            ids = torch.tensor([[rng.randrange(experts) for _ in range(top)] for _ in range(tokens)], device=device)
            table = cache.plan(layer, ids)
            routed = set(ids.flatten().tolist())
            for e, slot in enumerate(table.tolist()):
                if e in routed:
                    self.assertGreaterEqual(slot, 0)
                    self.assertTrue(torch.equal(cache.rows()[slot].cpu(), host[layer * experts + e]), (layer, e))
            x = torch.randn(tokens, IN, device=device)
            weights = torch.rand(tokens, top, device=device)
            cached = block.forward(cache.rows(), x, table.long()[ids], weights)
            direct = block.forward(host.to(device), x, layer * experts + ids, weights)
            self.assertTrue(torch.equal(cached, direct))
        seen = cache.telemetry(sync=True)
        self.assertGreater(seen["hits"], 0)
        self.assertEqual(seen["fill_bytes"], seen["fills"] * 32, "a fill copies the padded slot")

    def test_rows_are_checked_against_their_slots_and_store(self):
        import torch
        device = _device(torch)
        residency = self.assemble(options={"native": False, "ring_slots": 0})["expert_residency"]
        with self.assertRaisesRegex(ValueError, "slot of 16"):
            residency.cache(1, 4, {"bundle_bytes": 16, "row_bytes": 17}, 8, device, HostRows(torch, torch.zeros(4, 5,
                            dtype=torch.uint8), 16))
        with self.assertRaisesRegex(ValueError, "fill slots of 32"):
            residency.row_cache(1, 4, 20, 8, device, HostRows(torch, torch.zeros(4, 5, dtype=torch.uint8), 16))
        store = self.row_store(residency, 1, 4, device, torch.zeros(4, ROW, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "fill slots of 16"):
            residency.row_cache(1, 4, 5, 8, device, store)

    def test_a_row_store_takes_its_rows_in_their_own_size(self):
        import torch
        device = _device(torch)
        residency = self.assemble()["expert_residency"]
        try:
            with self.assertRaisesRegex(ValueError, "not 3"):
                residency.row_store(1, 4, 5, device, [bytes(5)] * 3)
            with self.assertRaisesRegex(ValueError, "not 5"):
                residency.row_store(1, 2, 5, device, [bytes(5), bytes(4)])
        except Exception as exc:
            if type(exc).__name__ == "HostMemoryUnavailable":
                self.skipTest(str(exc))
            raise
        store = self.row_store(residency, 1, 2, device, [bytes(range(ROW)), bytes(ROW)])
        self.assertEqual(bytes(store.bundle(0, 0)[:ROW].tolist()), bytes(range(ROW)))
        self.assertEqual(store.bundle(0, 0)[ROW:].tolist(), [0] * (32 - ROW), "padded with zeros")
