"""memory-table-rows' device form indexes the rows its host form returns, from a table on the device or (host) in
registered host memory read over the link, and its lookup captures in a CUDA graph either way; store() returns a
table handle, as lookup-table-rows' does, so a plugin holds several tables and the two plugins are interchangeable."""
import os

from crit_plugins.testing import PluginTestCase

NO_HOST_MEMORY = "host memory regions need libc's mmap, which Windows does not have"


class DeviceTableTests(PluginTestCase):
    def test_lookup_returns_the_rows_of_the_ids(self):
        torch = __import__("torch") if self.device() else None
        for host in (False, True):
            with self.subTest(host=host):
                if host and os.name == "nt":
                    self.skipTest(NO_HOST_MEMORY)
                table = self.assemble(options={"host": host})["embeddings"]
                rows = [[float(i * 3 + j) for j in range(3)] for i in range(5)]
                table.load(rows)
                table.store(torch.tensor(rows, device=self.device()))
                ids = [4, 0, 4, 2]
                found = table.lookup(torch.tensor(ids, device=self.device()))
                self.assertEqual(found.device.type, torch.device(self.device()).type)
                self.assertEqual(found.tolist(), table.rows(ids))
                with self.assertRaises(IndexError):
                    table.lookup(torch.tensor([5], device=self.device()))
                self.assertEqual(table.facts()["embeddings"]["table"], "host" if host else "device")

    def test_a_lookup_captures_in_a_cuda_graph(self):
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        torch = __import__("torch")
        for host in (False, True):
            with self.subTest(host=host):
                if host and os.name == "nt":
                    self.skipTest(NO_HOST_MEMORY)
                table = self.assemble(options={"host": host})["embeddings"]
                table.store(torch.arange(15, dtype=torch.bfloat16, device="cuda").view(5, 3))
                ids = torch.tensor([3, 1], device="cuda")
                table.lookup(ids)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = table.lookup(ids)
                ids.copy_(torch.tensor([0, 4]))
                graph.replay()
                self.assertEqual(out.float().tolist(), [[0.0, 1.0, 2.0], [12.0, 13.0, 14.0]])

    def test_tables_are_handles_as_lookup_table_rows_gives_them(self):
        torch = __import__("torch") if self.device() else None
        from crit_plugins.assemble import assemble
        device = self.device()
        g = torch.Generator().manual_seed(2)
        first, second = torch.randn(7, 4, generator=g), torch.randn(9, 6, generator=g)
        codes = torch.randint(0, 256, (11, 16), dtype=torch.uint8, generator=g)    # NVFP4: 32 values a row
        scales = torch.randint(0, 0x7F, (11, 2), dtype=torch.uint8, generator=g)    # E4M3, no NaN
        other = assemble({"use": {"embeddings": "lookup-table-rows"}, "options": {}}, self.registry())
        self.addCleanup(other.close)
        reference = other["embeddings"].store(codes.to(device), "nvfp4", scales.to(device), 0.5)
        ids = torch.tensor([3, 0, 10, 3], device=device)
        for host in (False, True):
            with self.subTest(host=host):
                if host and os.name == "nt":
                    self.skipTest(NO_HOST_MEMORY)
                plugin = self.assemble(options={"host": host})["embeddings"]
                a, b = plugin.store(first.to(device)), plugin.store(second.to(device))
                self.assertTrue(torch.equal(a.lookup(torch.tensor([6, 1], device=device)).cpu(), first[[6, 1]]))
                self.assertTrue(torch.equal(b.lookup(torch.tensor([8], device=device)).cpu(), second[[8]]))
                self.assertTrue(torch.equal(plugin.lookup(torch.tensor([2], device=device)).cpu(), second[[2]]))
                self.assertEqual(plugin.facts()["embeddings"]["tables"], 2)
                a.close()
                with self.assertRaisesRegex(ValueError, "closed"):
                    a.lookup(torch.tensor([0], device=device))
                packed = plugin.store(codes.to(device), "nvfp4", scales.to(device), 0.5)
                self.assertTrue(torch.equal(packed.lookup(ids).cpu(), reference.lookup(ids).cpu()))
                self.assertEqual(packed.lookup(ids).dtype, torch.bfloat16)

    def test_a_table_off_the_device_is_read_from_its_files_into_host_memory(self):
        # embeddings:host: model-template opens a table its plan put off the device where the checkpoint stores it
        # ({paths, offsets, rows}); this plugin reads those rows once into registered host memory, so a lookup gives
        # the rows lookup-table-rows' file reader gives, and captures in a CUDA graph.
        if os.name == "nt":
            self.skipTest(NO_HOST_MEMORY)
        import tempfile
        torch = __import__("torch")
        from crit_plugins.assemble import assemble
        device = self.device()
        g = torch.Generator().manual_seed(3)
        rows = torch.randn(13, 8, generator=g).to(torch.bfloat16)
        codes = torch.randint(0, 256, (13, 16), dtype=torch.uint8, generator=g)
        scales = torch.randint(0, 0x7F, (13, 2), dtype=torch.uint8, generator=g)
        folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, folder, True)

        def files(tensor, name):   # the rows split over two files, each after a header of 24 bytes
            data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            cut = 5 * (len(data) // tensor.shape[0])
            paths = []
            for n, part in enumerate((data[:cut], data[cut:])):
                paths.append(os.path.join(folder, f"{name}{n}.bin"))
                with open(paths[-1], "wb") as handle:
                    handle.write(b"h" * 24 + part)
            return {"paths": paths, "offsets": [24, 24], "rows": [5, 8]}
        other = assemble({"use": {"embeddings": "lookup-table-rows"}, "options": {}}, self.registry())
        self.addCleanup(other.close)
        plugin = self.assemble()["embeddings"]
        self.assertIn("embeddings:host", self.registry()["memory-table-rows"]["provides"])
        ids = torch.tensor([12, 0, 5, 4, 12], device=device)
        plain = plugin.open(files(rows, "bf16"), 8, torch.bfloat16, device)
        self.assertTrue(torch.equal(plain.lookup(ids).cpu(), rows[ids.cpu()]))
        self.assertEqual(plugin.facts()["embeddings"]["table"], "host")
        packed = plugin.open(files(codes, "codes"), 32, torch.bfloat16, device, format="nvfp4",
                             scales=files(scales, "scales"), scale=0.5)
        reference = other["embeddings"].open(files(codes, "rcodes"), 32, torch.bfloat16, device, format="nvfp4",
                                             scales=files(scales, "rscales"), scale=0.5)
        self.assertTrue(torch.equal(packed.lookup(ids).cpu(), reference.lookup(ids).cpu()))
        plugin.prefetch([1, 2])   # nothing to read ahead, and no error
        if device == "cuda":
            plain.lookup(ids)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = plain.lookup(ids)
            ids.copy_(torch.tensor([1, 2, 3, 6, 7]))
            graph.replay()
            self.assertTrue(torch.equal(out.cpu(), rows[[1, 2, 3, 6, 7]]))
