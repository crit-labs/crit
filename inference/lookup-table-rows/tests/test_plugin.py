from crit_plugins.slots import CASES
from crit_plugins.testing import PluginTestCase


class EmbeddingsChecks:
    """What every embeddings plugin must do."""

    def test_it_has_the_embeddings_interface(self):
        self.assertImplements()

    def test_rows_come_back_in_request_order(self):
        embeddings = self.assemble()["embeddings"]
        embeddings.load([[float(i), -float(i), 0.5] for i in range(5)])
        self.assertEqual(embeddings.rows([3, 0, 3]), [[3.0, -3.0, 0.5], [0.0, -0.0, 0.5], [3.0, -3.0, 0.5]])
        self.assertEqual(embeddings.rows([]), [])
        with self.assertRaises(IndexError):
            embeddings.rows([5])
        with self.assertRaises(IndexError):
            embeddings.rows([-1])


class LookupTableRowsTests(EmbeddingsChecks, PluginTestCase):
    def test_it_is_the_in_memory_table_within_fp32_rounding(self):
        for workers, cache_rows in ((1, 0), (8, 0), (4, 3)):
            with self.subTest(workers=workers, cache_rows=cache_rows):
                self.assertMatchesReference(options={"workers": workers, "cache_rows": cache_rows},
                                            atol=1e-12, rtol=1e-7)

    def test_hot_rows_are_served_from_memory(self):
        embeddings = self.assemble(options={"cache_rows": 4})["embeddings"]
        CASES["embeddings"]({"embeddings": embeddings})
        self.assertEqual(embeddings.table.reads, 7)
        self.assertEqual(embeddings.table.hits, 2)
