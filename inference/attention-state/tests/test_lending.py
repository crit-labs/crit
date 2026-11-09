"""attention-state's pool lends what it does not use: one allocation (pages, then spare bytes), pages taken lowest
first, a floor one past the highest page in use, and a borrower called before a page at or past its bound is handed
out and after the floor falls to its low bound."""
from crit_plugins import StepBatch
from crit_plugins.testing import PluginTestCase


class LendingTests(PluginTestCase):
    def pool(self, spare=0, dtype="bf16"):
        self.torch = __import__("torch")
        kv = self.assemble(options={"page": 4})["kv_cache"]
        return kv.pool([0, 1], 2, 8, 64, self.device(), dtype, spare=spare)

    def test_the_pages_and_the_spare_bytes_are_one_allocation(self):
        for dtype in ("bf16", "fp8", "int8"):
            with self.subTest(dtype=dtype):
                pool = self.pool(spare=1000, dtype=dtype)
                info = pool.lendable()
                memory = info["memory"]
                self.assertEqual(memory.numel(), info["spare"][1])
                self.assertEqual(info["spare"][1] - info["spare"][0], 1000)
                tensors = [pool.k, pool.v] + ([pool.k_scales, pool.v_scales] if dtype == "int8" else [])
                self.assertEqual(len(info["regions"]), 2 * len(tensors))
                for i, (first, per_page, pages) in enumerate(info["regions"]):
                    tensor = tensors[i // 2][i % 2]
                    self.assertTrue(tensor.is_contiguous())
                    self.assertEqual(tensor.data_ptr() - memory.data_ptr(), first)
                    self.assertEqual((pages, per_page), (16, tensor[0].numel() * tensor.element_size()))
                    self.assertLessEqual(first + pages * per_page, info["spare"][0])

    def test_the_lowest_free_page_goes_first_and_the_floor_follows_the_highest(self):
        pool = self.pool()
        for name in "ab":
            pool.open(name)
        pool.reserve(StepBatch.build([("a", 0, 8), ("b", 0, 4)], self.device()))
        self.assertEqual((pool.table("a"), pool.table("b"), pool.floor), ([0, 1], [2], 3))
        pool.close("a")
        self.assertEqual(pool.floor, 3)
        pool.open("c")
        pool.reserve(StepBatch.build([("c", 0, 4)], self.device()))
        self.assertEqual(pool.table("c"), [0])
        pool.close("b")
        self.assertEqual(pool.floor, 1)

    def test_the_borrower_hears_of_the_floor_before_the_pages_are_handed_out(self):
        pool = self.pool()
        calls = []

        def lender(floor):
            calls.append((floor, pool.floor, len(pool.free)))
            return (1, 100 if floor > 3 else 3)   # again when the floor falls to 1, or page 3 (then 100) is taken

        pool.on_reserve(lender)
        self.assertEqual(calls, [(0, 0, 16)])
        pool.open("a")
        pool.reserve(StepBatch.build([("a", 0, 12)], self.device()))    # pages 0, 1, 2: below the bound
        self.assertEqual(len(calls), 1)
        pool.reserve(StepBatch.build([("a", 12, 2)], self.device()))    # page 3: the bound
        self.assertEqual(calls[-1], (4, 3, 13))    # told the new floor before the page left the free heap
        pool.keep("a", 4)                          # pages 1, 2, 3 freed: the floor falls to 1
        self.assertEqual(calls[-1][0], 1)
        pool.on_reserve(None)
        pool.close("a")
        self.assertEqual(len(calls), 3)
