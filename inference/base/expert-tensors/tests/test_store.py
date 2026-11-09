"""expert-tensors' bundle store (experts:store): bundles put into host memory come back byte for byte from its host
bytes, its device view and read(); a store file is sealed by its first region() and reopens complete for the checkpoint and
layout it was written for alone; a path one open store maps is refused to another."""
import os
import tempfile
import unittest

from crit_plugins.testing import PluginTestCase

# The store maps its file into host memory with libc's mmap (device-expert-cache's host_memory), which Windows lacks.
NO_HOST_MEMORY = "host memory regions need libc's mmap, which Windows does not have"


@unittest.skipIf(os.name == "nt", NO_HOST_MEMORY)
class BundleStoreTests(PluginTestCase):
    def bundles(self, layers, experts, size, seed=0):
        torch = __import__("torch")
        g = torch.Generator().manual_seed(seed)
        return {(l, e): torch.randint(0, 256, (size,), dtype=torch.uint8, generator=g)
                for l in range(layers) for e in range(experts)}

    def test_bundles_come_back_byte_for_byte(self):
        torch = __import__("torch") if self.device() else None
        device = self.device()
        source = self.assemble()["expert_source"]
        layout = {"bundle_bytes": 4096, "components": {"gate_up": (0, "uint8", [32, 64]), "down": (2048, "uint8", [32, 64])}}
        store = source.open_store(layout, 2, 5, device)
        bundles = self.bundles(2, 5, 4096)
        for (l, e), b in bundles.items():
            store.put(l, e, b.to(device))
        host, view = store.region()
        whole = torch.cat([bundles[(l, e)] for l in range(2) for e in range(5)])
        self.assertTrue(torch.equal(host, whole))
        if device == "cuda":
            self.assertTrue(torch.equal(view.cpu(), whole))
            self.assertEqual(view.data_ptr(), store.address(0, 0))
        self.assertEqual(store.address(1, 3) - store.address(0, 0), 8 * 4096)
        into = torch.empty(3, 4096, dtype=torch.uint8, device=device)
        store.read([7, 0, 7], into)
        self.assertTrue(torch.equal(into.cpu(), torch.stack([bundles[(1, 2)], bundles[(0, 0)], bundles[(1, 2)]])))
        with self.assertRaisesRegex(ValueError, "sealed"):
            store.put(0, 0, bundles[(0, 0)])
        with self.assertRaisesRegex(ValueError, "4096 bytes"):
            source.open_store(layout, 1, 1, device).put(0, 0, bundles[(0, 0)][:16])

    def test_a_store_file_reopens_complete_for_its_checkpoint_alone(self):
        torch = __import__("torch") if self.device() else None
        device = self.device()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = os.path.join(folder.name, "experts.bundles")
        identity = {"shards": [["model-00001.safetensors", 4096, "9f2c"]]}
        bundles = self.bundles(1, 4, 1024, seed=2)
        source = self.assemble(options={"store_path": path})["expert_source"]
        with self.assertRaisesRegex(ValueError, "identity"):
            source.open_store({"bundle_bytes": 1024}, 1, 4, device)   # a kept file is bound to its checkpoint
        store = source.open_store({"bundle_bytes": 1024}, 1, 4, device, identity=identity)
        for (l, e), b in bundles.items():
            store.put(l, e, b)
        store.region()
        with self.assertRaisesRegex(ValueError, "open as another bundle store"):
            self.assemble()["expert_source"].open_store({"bundle_bytes": 1024}, 1, 4, device, path, identity)
        store.close()
        again = self.assemble()["expert_source"].open_store({"bundle_bytes": 1024}, 1, 4, device, path, identity)
        self.assertTrue(again.complete)
        self.assertTrue(torch.equal(again.bundle(0, 3), bundles[(0, 3)]))
        with self.assertRaisesRegex(ValueError, "sealed"):
            again.put(0, 0, bundles[(0, 0)])
        again.close()
        other = self.assemble()["expert_source"].open_store({"bundle_bytes": 1024}, 1, 4, device, path,
                                                            {"shards": [["model-00001.safetensors", 4096, "77aa"]]})
        self.assertFalse(other.complete)   # another checkpoint writes the file anew
        self.assertFalse(os.path.exists(path + ".json"))   # its index goes before the bytes change
        other.close()
        layout = self.assemble()["expert_source"].open_store({"bundle_bytes": 2048}, 1, 4, device, path, identity)
        self.assertFalse(layout.complete)   # another layout too
        layout.close()
