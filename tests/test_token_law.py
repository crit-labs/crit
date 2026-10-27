"""Host fixtures for the device sampling law's Python mirror: Core's processing order, the full
vocabulary without a candidate cap, min_p, and speculative verification that keeps the target law."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    name = "crit_test_token_law_sampling"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "inference" / "token-sampling" / "code" / "sampling.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TokenLawTests(unittest.TestCase):
    def test_processing_follows_cores_order(self):
        m = load()
        row = [2.0, -1.0, 0.5, 3.0, -math.inf]
        # bias first, then repetition (divides positive, multiplies negative), then frequency and presence
        v = m.process(row, bias={0: 1.0, 3: -math.inf}, repetition=2.0, frequency=0.5, presence=0.25,
                      counts={0: 1}, prompt=(1,), staged=(0, 2))
        self.assertEqual(v[0], 3.0 / 2.0 - (0.5 * 2 + 0.25))
        self.assertEqual(v[1], -1.0 * 2.0)                 # prompt token: repetition only
        self.assertEqual(v[2], 0.5 / 2.0 - (0.5 + 0.25))   # staged once
        self.assertEqual(v[3], -math.inf)                  # banned by bias
        self.assertEqual(v[4], -math.inf)                  # masked stays masked
        suppressed = m.process(row, min_tokens=3, suppress=(3, 99), outputs=2)
        self.assertEqual(suppressed[3], -math.inf)
        self.assertEqual(m.process(row, min_tokens=3, suppress=(3,), outputs=3)[3], 3.0)
        # Core's suppression_yields: a constraint allowing only suppressed tokens keeps them selectable
        self.assertEqual(m.process(row, min_tokens=3, suppress=(3,), outputs=2, yields=True)[3], 3.0)

    def test_full_vocabulary_and_min_p(self):
        m = load()
        row = [0.0] * 200
        self.assertEqual(len(m.nucleus(row, 1.0)), 200)    # no 64-candidate cap
        self.assertEqual(len(m.nucleus(row, 1.0, top_k=64)), 64)
        draws = {m.select(row, 1.0, seed=3, position=c) for c in range(400)}
        self.assertTrue(any(t >= 64 for t in draws))
        peaked = [5.0, 4.0, 0.0, -1.0]
        kept = m.nucleus(peaked, 1.0, min_p=0.1)
        self.assertEqual([t for t, _ in kept], [0, 1])     # exp(-1) >= 0.1 > exp(-5)
        self.assertAlmostEqual(math.fsum(p for _, p in kept), 1.0)

    def test_verification_keeps_the_target_law(self):
        m = load()
        target = [1.2, 0.8, 0.5, 0.1, -0.5, -2.0]
        drafter = [-1.0, 1.5, 0.2, 0.9, 0.0, -0.4]
        p = dict(m.nucleus(target, 1.0))
        q = dict(m.nucleus(drafter, 1.0))
        n = 6000
        for sampled in (False, True):
            hist = [0] * len(target)
            for seed in range(1, n + 1):
                if sampled:
                    d = m.walk(sorted(q.items(), key=lambda e: (-drafter[e[0]], e[0])), m.uniform(seed, 0, 0, m.DRAFT))
                    out = m.verify([target, target], [d], 1.0, seed=seed, draft_laws=[q])
                else:
                    out = m.verify([target, target], [1], 1.0, seed=seed)
                hist[out[0]] += 1
            for t, pr in p.items():
                self.assertLess(abs(hist[t] / n - pr), 4.5 * math.sqrt(pr * (1 - pr) / n) + 1e-3)
        # greedy targets reduce to argmax predictions
        self.assertEqual(m.verify([target, target, target], [0, 3], 0.0), [0, 0, 0])
        # greedy drafts: every column is the target's own draw, so the round emits what plain sampling would
        out = m.verify([target, target], [0], 1.0, seed=11)
        self.assertEqual(out, [m.select(target, 1.0, seed=11, position=0), m.select(target, 1.0, seed=11, position=1)])
        # sampled drafts: a draft the rejection test accepts is written as itself; a rejected one never is
        d = 1
        out = m.verify([target, target], [d], 1.0, seed=12, draft_laws=[q])
        self.assertEqual(out[0] == d, m.uniform(12, 0, 0, m.ACCEPT) * q[d] < p[d])
        # a cut column (draft -1: the drafter stopped) is the target's own draw whatever q holds, like the bonus
        out = m.verify([target, target], [-1], 1.0, seed=13, draft_laws=[q])
        self.assertEqual(out, [m.select(target, 1.0, seed=13, position=0), m.select(target, 1.0, seed=13, position=1)])


if __name__ == "__main__":
    unittest.main()
