"""Eval's full-vocabulary distribution comparison: FP64 host math, and the CUDA reference kernel when nvcc and a GPU exist."""
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from crit_eval import logprob, metrics, probe  # noqa: E402


class HostComparisonTests(unittest.TestCase):
    def test_log_softmax_nll_and_divergence(self):
        m = metrics
        row = m.log_softmax([1000.0, 999.0, -5.0, float("inf")], valid=3)   # stable at large logits, padding unread
        self.assertAlmostEqual(sum(math.exp(x) for x in row), 1.0, places=12)
        self.assertAlmostEqual(m.nll(row, 0), -row[0])
        self.assertEqual(m.kl_divergence(row, row), 0.0)
        shifted = m.log_softmax([999.0, 1000.0, -5.0])
        expect = sum(math.exp(p) * (p - q) for p, q in zip(row, shifted))
        self.assertAlmostEqual(m.kl_divergence(row, shifted), expect)
        self.assertGreater(expect, 0)
        for bad in (lambda: m.log_softmax([1.0, float("nan")]), lambda: m.log_softmax([1.0], valid=2),
                    lambda: m.nll(row, 3), lambda: m.kl_divergence(row, row[:2]),
                    lambda: m.compare([[1.0]], [], [0])):
            with self.assertRaises(ValueError):
                bad()

    def test_summary_perplexity_and_quantiles(self):
        s = metrics.Summary()
        for nll, kld in ((math.log(2), 0.0), (math.log(8), 0.3), (math.log(4), 0.1), (math.log(4), 0.2)):
            s.add(nll, kld)
        self.assertAlmostEqual(s.perplexity(), 4.0)
        self.assertAlmostEqual(s.kld_mean(), 0.15)
        self.assertEqual((s.kld_quantile(0.5), s.kld_quantile(0.99), s.kld_max()), (0.1, 0.3, 0.3))
        for bad in (lambda: s.add(-1.0), lambda: s.add(1.0, float("inf")), lambda: s.kld_quantile(0),
                    lambda: metrics.Summary().perplexity(), lambda: metrics.Summary().kld_mean()):
            with self.assertRaises(ValueError):
                bad()

    def test_compare_and_lumped_bound(self):
        reference = [[2.0, 1.0, 0.0, 99.0], [0.0, 3.0, 1.0, 99.0]]   # padding entries are ignored
        candidate = [[2.1, 0.9, 0.0, -99.0], [0.0, 3.0, 1.0, -99.0]]
        ref, cand = metrics.compare(reference, candidate, targets=[0, 1], valid=3)
        self.assertEqual(cand.kld[1], 0.0)
        self.assertGreater(cand.kld[0], 0)
        self.assertLess(abs(cand.perplexity() / ref.perplexity() - 1), 0.05)
        self.assertEqual(cand.kld_quantile(0.99), cand.kld_max())
        # The suite's compact top-k-plus-tail KL is a lower bound on the full-vocabulary value.
        r, c = metrics.log_softmax(reference[0], 3), metrics.log_softmax(candidate[0], 3)
        self.assertLessEqual(metrics.lumped_kl(r[:2], c[:2]), metrics.kl_divergence(r, c) + 1e-15)


class DeviceReferenceTests(unittest.TestCase):
    def test_cuda_rows_match_fp64(self):
        try:
            nvcc = probe.find_nvcc()
        except SystemExit:
            self.skipTest("no CUDA toolkit")
        if not shutil.which("nvidia-smi"):
            self.skipTest("no CUDA device")
        with tempfile.TemporaryDirectory() as directory:
            binary = logprob.build(nvcc, directory, ROOT / "tests/eval/logprob_compare_tests.cu")
            done = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("distribution comparison ok", done.stdout)

    def test_sources_are_in_eval(self):
        self.assertTrue(logprob.SOURCE.is_file() and logprob.HEADER.is_file())
        self.assertIn("enqueue_row_compare", logprob.HEADER.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
