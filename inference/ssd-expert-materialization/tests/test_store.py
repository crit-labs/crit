"""ssd-expert-materialization's bundle file (the file tier of experts:store): read() gives the bundles put, the host
tier serves repeats, a reopened file needs no puts when it was written for the same checkpoint; it has no device-readable region, so a device expert cache fills
its slots from read() on the host, exactly, and refuses to capture such a plan."""
import os
import tempfile

from crit_plugins.testing import PluginTestCase


class BundleFileTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None
        g = self.torch.Generator().manual_seed(1)
        self.bundles = self.torch.randint(0, 256, (2, 6, 2048), dtype=self.torch.uint8, generator=g)

    IDENTITY = {"shards": [["model-00001.safetensors", 8192, "51e0"]]}

    def filled(self, source, path=None, identity=None):
        store = source.open_store({"bundle_bytes": 2048}, 2, 6, self.device(), path, identity)
        if not getattr(store, "complete", False):
            for l in range(2):
                for e in range(6):
                    store.put(l, e, self.bundles[l, e])
        return store

    def test_read_gives_the_bundles_and_the_host_tier_serves_repeats(self):
        torch = self.torch
        for native, host_bytes in ((True, 0), (True, 3 * 2048), (False, 1 << 20)):
            with self.subTest(native=native, host_bytes=host_bytes):
                source = self.assemble(options={"host_bytes": host_bytes})["expert_source"]
                store = self.filled(source)
                if not native:
                    store._reader_module = None
                into = torch.empty(4, 2048, dtype=torch.uint8)
                for _ in range(2):
                    store.read([7, 0, 7, 11], into)
                    self.assertTrue(torch.equal(into, self.bundles.view(12, 2048)[[7, 0, 7, 11]]))
                self.assertEqual(store.stats["file_reads"], 3 if host_bytes else 6)   # each distinct bundle once a read
                self.assertEqual(store.stats["tier_hits"], 0 if not host_bytes else 4)
                with self.assertRaisesRegex(ValueError, "immutable"):
                    store.put(0, 0, self.bundles[0, 0])
                with self.assertRaises(RuntimeError):
                    store.region()

    def test_a_bundle_file_at_a_path_reopens_without_puts(self):
        torch = self.torch
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = os.path.join(folder.name, "bundles.bin")
        with self.assertRaisesRegex(ValueError, "identity"):
            self.filled(self.assemble()["expert_source"], path)   # a kept file is bound to its checkpoint
        store = self.filled(self.assemble()["expert_source"], path, self.IDENTITY)
        store.read([1], torch.empty(1, 2048, dtype=torch.uint8))
        store.close()
        self.assertTrue(os.path.exists(path))
        source = self.assemble()["expert_source"]
        again = source.open_store({"bundle_bytes": 2048}, 2, 6, self.device(), path, self.IDENTITY)
        self.assertTrue(again.complete)
        into = torch.empty(1, 2048, dtype=torch.uint8)
        again.read([10], into)
        self.assertTrue(torch.equal(into[0], self.bundles[1, 4]))
        again.close()
        other = source.open_store({"bundle_bytes": 2048}, 2, 6, self.device(), path,
                                  {"shards": [["model-00001.safetensors", 8192, "0bad"]]})
        self.assertFalse(other.complete)   # another checkpoint writes the file anew, its index gone first
        self.assertFalse(os.path.exists(path + ".json"))
        other.close()

    def test_a_device_cache_fills_from_the_file_tier(self):
        torch, device = self.torch, self.device()
        assembly = self.assemble(use={"expert_residency": "device-expert-cache"},
                                 use_options={"device-expert-cache": {"ring_slots": 0}})
        store = self.filled(assembly["expert_source"])
        cache = assembly["expert_residency"].cache(2, 6, {"bundle_bytes": 2048}, 6, device, store)
        self.assertRegex(" ".join(assembly["expert_residency"].round_problems()), "file tier")
        for layer, ids in ((0, [1, 2, 3]), (1, [5, 0]), (0, [1, 4]), (1, [2, 3, 4, 5])):
            table = cache.plan(layer, torch.tensor(ids, device=device)).tolist()
            for e in ids:
                self.assertGreaterEqual(table[e], 0)
                self.assertTrue(torch.equal(cache.store[table[e]].cpu(), self.bundles[layer, e]))
        if device == "cuda":
            ids = torch.tensor([1, 2], device="cuda")
            graph = torch.cuda.CUDAGraph()
            with self.assertRaisesRegex(RuntimeError, "file tier"):
                with torch.cuda.graph(graph):
                    cache.plan(0, ids)
