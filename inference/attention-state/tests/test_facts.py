"""attention-state's engine facts: the device bytes one more token of context takes across the pools it made, at each
pool's precision, which crit-eval sizes the KV cache at a context by."""
from crit_plugins.testing import PluginTestCase

LAYERS, KV_HEADS, HEAD_DIM = [0, 1], 2, 8


class FactsTests(PluginTestCase):
    def assembly(self, page=4):
        self.torch = __import__("torch")
        return self.assemble(options={"page": page})

    def test_no_pool_reports_no_kv_bytes(self):
        self.assertNotIn("kv_bytes_per_token", self.assembly().facts())

    def test_each_precision_counts_its_own_bytes(self):
        # Per layer: keys and values of 2 heads x 8 dims at the element size; an int8 pool adds an FP32 key and
        # value scale per head.
        for dtype, expected in (("bf16", 2 * 64), ("fp32", 2 * 128), ("fp8", 2 * 32), ("int8", 2 * (32 + 16))):
            with self.subTest(dtype=dtype):
                assembly = self.assembly()
                assembly["kv_cache"].pool(LAYERS, KV_HEADS, HEAD_DIM, 64, self.device(), dtype)
                self.assertEqual(assembly["kv_cache"].facts()["kv_bytes_per_token"], expected)
                self.assertEqual(assembly.facts()["kv_bytes_per_token"], expected)

    def test_the_pools_add_up_whatever_their_page_or_value_width(self):
        # A model's FP8 cache of two layers beside a BF16 cache of one more layer, its values 4 wide on another page.
        assembly = self.assembly(page=16)
        kv = assembly["kv_cache"]
        kv.pool(LAYERS, KV_HEADS, HEAD_DIM, 100, self.device(), "fp8")
        kv.pool(["draft"], KV_HEADS, HEAD_DIM, 30, self.device(), "bf16", page=8, value_dim=4)
        self.assertEqual(kv.facts()["kv_bytes_per_token"], 2 * 32 + (32 + 16))

    def test_the_count_is_the_pages_bytes_over_their_tokens(self):
        for dtype in ("bf16", "fp8", "int8"):
            with self.subTest(dtype=dtype):
                kv = self.assembly()["kv_cache"]
                pool = kv.pool(LAYERS, KV_HEADS, HEAD_DIM, 64, self.device(), dtype)
                tensors = [pool.k, pool.v] + ([pool.k_scales, pool.v_scales] if dtype == "int8" else [])
                held = sum(t.numel() * t.element_size() for t in tensors)
                # every page tensor holds the sequences' pages and the dump page after them (index pool.pages)
                self.assertEqual(kv.facts()["kv_bytes_per_token"] * (pool.pages + 1) * pool.page, held)
