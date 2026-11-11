"""attention-state's device form: the paged KV pool keeps each sequence's rows where its host form keeps them, in torch
and through crit's paged-kv-attention writes (the same codes and scales)."""
from crit_plugins import StepBatch
from crit_plugins.testing import PluginTestCase


class KVPoolTests(PluginTestCase):
    def pool(self, dtype="fp32", tokens=256, page=16, layers=(0, 3), kernels="auto", dim=4):
        self.torch = __import__("torch")
        device = self.device()
        kv = self.assemble(options={"page": page, "kernels": kernels})["kv_cache"]
        return kv.pool(list(layers), 2, dim, tokens, device, dtype), device

    def rows(self, n, seed, width=4):
        g = self.torch.Generator().manual_seed(seed)
        return self.torch.randn(n, 2, width, generator=g)

    def step(self, pool, device, rows, layer=3, seed=0):
        """Reserve and write `rows` of (sequence, first position, count); returns the keys and values written."""
        batch = StepBatch.build(rows, device)
        slots = pool.reserve(batch)
        width = pool.head_dim
        keys, values = self.rows(batch.tokens, seed, width).to(device), self.rows(batch.tokens, seed + 1, width).to(device)
        pool.write(layer, slots, keys, values)
        return batch, keys, values

    def test_a_step_of_many_sequences_writes_each_where_it_reads_back(self):
        pool, device = self.pool()
        for sequence in "abc":
            pool.open(sequence)
        batch, keys, values = self.step(pool, device, [("a", 0, 20), ("b", 0, 1), ("c", 0, 5)])
        for r in range(batch.rows):
            sequence, first, count, _ = batch.row(r)
            k, v = pool.gather(3, sequence, count)
            self.assertTrue(self.torch.equal(k.cpu(), keys[first:first + count].cpu()))
            self.assertTrue(self.torch.equal(v.cpu(), values[first:first + count].cpu()))
        # Another layer of the pool is apart.
        self.assertEqual(float(pool.gather(0, "a", 20)[0].abs().sum()), 0.0)
        self.assertEqual(pool.free_pages(), 16 - 2 - 1 - 1)

    def test_keep_rolls_a_verify_window_back_and_frees_its_pages(self):
        pool, device = self.pool()
        pool.open("a")
        self.step(pool, device, [("a", 0, 16)])
        pool.keep("a", 16)
        _, keys, _ = self.step(pool, device, [("a", 16, 4)], seed=5)   # a verify window of 4
        pool.keep("a", 17)   # one row accepted
        self.assertEqual((pool.length("a"), len(pool.table("a"))), (17, 2))
        self.assertTrue(self.torch.equal(pool.gather(3, "a", 17)[0][16].cpu(), keys[0].cpu()))
        pool.keep("a", 16)
        self.assertEqual(len(pool.table("a")), 1)
        with self.assertRaisesRegex(ValueError, "holds 16 reserved positions"):
            pool.keep("a", 20)

    def test_a_fork_shares_its_parents_pages_until_one_writes(self):
        pool, device = self.pool()
        pool.open("a")
        _, keys, _ = self.step(pool, device, [("a", 0, 20)])
        pool.keep("a", 20)
        pool.fork("a", "b")
        self.assertEqual(pool.table("a"), pool.table("b"))
        self.step(pool, device, [("b", 20, 3)], seed=9)   # into the page they share: b copies it first
        self.assertEqual(pool.table("a")[0], pool.table("b")[0])
        self.assertNotEqual(pool.table("a")[1], pool.table("b")[1])
        self.assertTrue(self.torch.equal(pool.gather(3, "b", 20)[0].cpu(), keys.cpu()))
        self.assertTrue(self.torch.equal(pool.gather(3, "a", 20)[0].cpu(), keys.cpu()))
        pool.close("a")
        pool.close("b")
        self.assertEqual(pool.free_pages(), 16)

    def test_an_fp8_pool_stores_e4m3_with_its_layers_scales(self):
        for kernels in ("auto", "torch"):
            pool, device = self.pool("fp8", kernels=kernels)
            pool.set_scales(3, 0.5, 2.0)
            pool.open("a")
            _, keys, values = self.step(pool, device, [("a", 0, 8)])
            k, v = pool.gather(3, "a", 8)
            # E4M3 keeps 3 mantissa bits (a relative step of 1/8), then the BF16 the pool computes in.
            for found, wrote in ((k, keys), (v, values)):
                error = (found.float().cpu() - wrote.cpu()).abs()
                self.assertTrue(bool((error <= 0.07 * wrote.cpu().abs() + 0.02).all()), float(error.max()))

    def test_row_scaled_pools_store_codes_with_a_scale_per_position_and_head(self):
        for dtype, top in (("int8", 127), ("fp8:row", 448)):
            pool, device = self.pool(dtype)
            pool.open("a")
            _, keys, values = self.step(pool, device, [("a", 0, 20)])
            k, v = pool.gather(3, "a", 20)
            for found, wrote in ((k, keys), (v, values)):
                # INT8: half a code step of the row's own scale; E4M3: 1/16 of the value; then BF16.
                step = wrote.cpu().abs().amax(dim=-1, keepdim=True) / top
                error = (found.float().cpu() - wrote.cpu()).abs()
                bound = 0.5 * step + 0.008 * wrote.cpu().abs() + 1e-6 if dtype == "int8" else \
                    0.07 * wrote.cpu().abs() + 4 * step
                self.assertTrue(bool((error <= bound).all()), (dtype, float(error.max())))
            # A fork's copied page keeps its scales.
            pool.keep("a", 20)
            pool.fork("a", "b")
            self.step(pool, device, [("b", 20, 3)], seed=9)
            self.assertTrue(self.torch.equal(pool.gather(3, "b", 20)[0].cpu(), k.cpu()))

    def test_row_scaled_codes_are_the_torch_rules_whatever_path_writes_them(self):
        # With crit's paged-kv-attention kernel loaded, a CUDA pool writes keys and values in one launch: the same
        # codes and scales as int8_rows and fp8_rows compute.
        from crit_plugins.assemble import load_module
        pages = load_module(self.registry()["attention-state"]).pages
        for dtype, rule in (("int8", pages.int8_rows), ("fp8:row", pages.fp8_rows)):
            for kernels in ("auto", "torch"):
                pool, device = self.pool(dtype, page=16, layers=(0, 3), kernels=kernels, dim=64)
                pool.open("a")
                _, keys, values = self.step(pool, device, [("a", 0, 37)])
                table = self.torch.tensor(pool.table("a"), device=device)
                for store, scales, wrote in ((pool.k[1], pool.k_scales[1], keys), (pool.v[1], pool.v_scales[1], values)):
                    codes, scale = rule(wrote)
                    found = store.index_select(0, table).reshape(-1, 2, 64)[:37]
                    if dtype == "fp8:row":
                        found, codes = found.view(self.torch.uint8), codes.view(self.torch.uint8)
                    self.assertTrue(self.torch.equal(found.cpu(), codes.cpu()), (dtype, kernels))
                    self.assertTrue(self.torch.equal(scales.index_select(0, table).reshape(-1, 2)[:37].cpu(),
                                                     scale.cpu()), (dtype, kernels))

    def test_a_full_pool_is_refused(self):
        pool, device = self.pool(tokens=32)
        pool.open("a")
        with self.assertRaisesRegex(MemoryError, "2 pages of 16 tokens are all in use"):
            self.step(pool, device, [("a", 0, 40)])
