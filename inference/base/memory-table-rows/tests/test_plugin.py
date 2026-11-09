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


class MemoryTableRowsTests(EmbeddingsChecks, PluginTestCase):
    def test_the_cases_run(self):
        out = CASES["embeddings"](self.assemble())
        self.assertEqual(out["prefill"][2], out["decode"][0])
