"""Host fixture for latent attention: the absorbed form (one shared latent cache, absorbed queries, W_UV after the
latent output) equals per-head attention over decompressed keys and values, and the cache-size arithmetic."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    name = "crit_test_latent_attention_reference"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "inference" / "latent-attention" / "code" / "reference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LatentAttentionTests(unittest.TestCase):
    def test_absorbed_equals_naive(self):
        m = load()
        rng = random.Random(5)
        heads, rank, rope, nope, vdim, L = 3, 6, 2, 4, 5, 9
        g = lambda: rng.gauss(0, 1)  # noqa: E731
        latents = [[g() for _ in range(rank)] for _ in range(L)]
        ropes = [[g() for _ in range(rope)] for _ in range(L)]
        w_uk = [[[g() for _ in range(rank)] for _ in range(nope)] for _ in range(heads)]
        w_uv = [[[g() for _ in range(rank)] for _ in range(vdim)] for _ in range(heads)]
        q_nope = [[g() for _ in range(nope)] for _ in range(heads)]
        q_rope = [[g() for _ in range(rope)] for _ in range(heads)]
        scale = 0.3
        want = m.naive(q_nope, q_rope, latents, ropes, w_uk, w_uv, scale)
        queries = [m.absorbed_query(q_nope[h], w_uk[h]) + q_rope[h] for h in range(heads)]
        cache = [c + r for c, r in zip(latents, ropes)]
        lat = m.latent_attention(queries, cache, rank, scale)
        got = [m.matvec(w_uv[h], lat[h]) for h in range(heads)]
        self.assertTrue(all(abs(a - b) < 1e-9 for ra, rb in zip(want, got) for a, b in zip(ra, rb)))

    def test_cache_size(self):
        m = load()
        # DeepSeek-V3: 61 layers of 512 + 64 values per token, against 128 heads x (192 + 128) for full per-head KV
        self.assertEqual(m.cache_values_per_token(512, 64, 61), 35136)
        self.assertLess(m.cache_values_per_token(512, 64, 1) * 70, 128 * (192 + 128))


if __name__ == "__main__":
    unittest.main()
