"""Host fixtures for the recurrent-layer kernels' references: the chunked Mamba-2 evaluation equals the token-by-token
recurrence for any chunk size (partial chunks included), and the causal convolution carries its window."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    name = "crit_test_recurrent_layer_kernels_ssd"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "inference" / "recurrent-layer-kernels" / "code" / "ssd.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RecurrentKernelReferenceTests(unittest.TestCase):
    def test_chunked_ssd_equals_the_recurrence(self):
        m = load()
        rng = random.Random(3)
        P, N, T = 3, 4, 23
        H0 = [[rng.gauss(0, 0.2) for _ in range(N)] for _ in range(P)]
        steps = [([rng.gauss(0, 1) for _ in range(P)], rng.uniform(0.01, 0.3), [rng.gauss(0, 0.5) for _ in range(N)],
                  [rng.gauss(0, 0.5) for _ in range(N)]) for _ in range(T)]
        direct = m.ssd_sequential(H0, steps, A=-1.3, D=0.7)
        for chunk in (1, 4, 8, 64):
            self.assertLess(m.max_difference(direct, m.ssd_chunked(H0, steps, A=-1.3, D=0.7, chunk=chunk)), 1e-12)
        with self.assertRaises(ValueError):
            m.ssd_chunked(H0, steps, A=-1.0, chunk=0)

    def test_convolution_carries_its_window(self):
        m = load()
        rows = [1.0, -2.0, 0.5, 3.0, 1.5]
        whole, state = m.conv(rows, [0.2, -0.1, 0.4, 0.3], [0.0, 0.0, 0.0], silu=False)
        first, s1 = m.conv(rows[:2], [0.2, -0.1, 0.4, 0.3], [0.0, 0.0, 0.0], silu=False)
        rest, s2 = m.conv(rows[2:], [0.2, -0.1, 0.4, 0.3], s1, silu=False)
        self.assertEqual(whole, first + rest)
        self.assertEqual(state, s2)
        self.assertEqual(state, rows[-3:])
        with self.assertRaises(ValueError):
            m.conv(rows, [1.0, 2.0], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
