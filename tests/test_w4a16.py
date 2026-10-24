"""Host fixtures for the W4A16 reference: GPTQ (with and without act-order, both zero conventions), AWQ and
compressed-tensors packings unpack to the same weights, the act-order projection equals the original, MXFP4 values."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    name = "crit_test_w4a16_w4"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "inference" / "w4a16-projections" / "code" / "w4.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class W4A16Tests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(7)
        self.m = load()
        self.N, self.K, self.group = 16, 32, 8
        G = self.K // self.group
        self.q = [[rng.randrange(16) for _ in range(self.K)] for _ in range(self.N)]
        self.z = [[rng.randrange(1, 16) for _ in range(G)] for _ in range(self.N)]
        self.s = [[rng.choice([0.01, 0.03]) for _ in range(G)] for _ in range(self.N)]
        self.want = [[(self.q[n][k] - self.z[n][k // self.group]) * self.s[n][k // self.group] for k in range(self.K)]
                     for n in range(self.N)]
        self.x = [rng.gauss(0, 1) for _ in range(self.K)]

    def gptq(self, order, g_idx, zero_offset):
        m, N, K, G = self.m, self.N, self.K, self.K // self.group
        qweight = [[0] * N for _ in range(K // 8)]
        for n in range(N):
            for j, k in enumerate(order):
                qweight[k // 8][n] |= self.q[n][j] << (4 * (k % 8))
        qzeros = [[sum((self.z[8 * w + i][g] - zero_offset) << (4 * i) for i in range(8)) for w in range(N // 8)] for g in range(G)]
        scales = [[self.s[n][g] for n in range(N)] for g in range(G)]
        return m.from_gptq(qweight, qzeros, scales, K, N, self.group, g_idx=g_idx, zero_offset=zero_offset)

    def test_gptq_plain_and_act_order(self):
        m, K = self.m, self.K
        for zero_offset in (0, 1):
            codes, sc, zr, perm = self.gptq(list(range(K)), None, zero_offset)
            self.assertEqual(perm, list(range(K)))
            self.assertEqual(m.int4_weights(codes, sc, zr), self.want)
        rng = random.Random(2)
        order = list(range(K))
        rng.shuffle(order)
        g_idx = [0] * K
        for j, k in enumerate(order):
            g_idx[k] = j // self.group
        canonical = sorted(range(K), key=lambda k: (g_idx[k], k))
        codes, sc, zr, perm = self.gptq(canonical, g_idx, 1)
        self.assertEqual(perm, canonical)
        w = m.int4_weights(codes, sc, zr)
        self.assertEqual(w, self.want)
        y = m.project(w, self.x, perm)
        ref = [sum(self.want[n][j] * self.x[canonical[j]] for j in range(K)) for n in range(self.N)]
        self.assertTrue(all(abs(a - b) < 1e-12 for a, b in zip(y, ref)))

    def test_awq_and_compressed(self):
        m, N, K, G = self.m, self.N, self.K, self.K // self.group
        awq = [[sum(self.q[8 * w + m.AWQ_ORDER[i]][k] << (4 * i) for i in range(8)) for w in range(N // 8)] for k in range(K)]
        awq_z = [[sum(self.z[8 * w + m.AWQ_ORDER[i]][g] << (4 * i) for i in range(8)) for w in range(N // 8)] for g in range(G)]
        scales = [[self.s[n][g] for n in range(N)] for g in range(G)]
        self.assertEqual(m.int4_weights(*m.from_awq(awq, awq_z, scales, K, N, self.group)[:3]), self.want)
        packed = [[sum(self.q[n][8 * w + i] << (4 * i) for i in range(8)) for w in range(K // 8)] for n in range(N)]
        self.assertEqual(m.int4_weights(*m.from_compressed(packed, self.s, K, N, self.group, zero_point=self.z)[:3]), self.want)
        symmetric = m.int4_weights(*m.from_compressed(packed, self.s, K, N, self.group)[:3])
        self.assertEqual(symmetric[0][0], (self.q[0][0] - 8) * self.s[0][0])
        canon = m.pack_canonical(self.q)
        self.assertEqual(canon[0][0], self.q[0][0] | (self.q[0][1] << 4))

    def test_mxfp4(self):
        m = self.m
        blocks = [[bytes([0x9F, 0x07] + [0] * 14)]]
        values = m.mxfp4_weights(blocks, [[127 - 1]], 32, 1)[0]
        self.assertEqual(values[:4], [-3.0, -0.25, 3.0, 0.0])   # codes 15, 9, 7, 0 times 2^-1


if __name__ == "__main__":
    unittest.main()
