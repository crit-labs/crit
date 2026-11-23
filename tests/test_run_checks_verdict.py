"""crit run serve --check's verdict: a hint item (crit-eval's gate pre-check) is reported and leaves the status."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("inference/core/python", "models/core/python", "cli"):
    sys.path.insert(0, str(ROOT / _folder))

from crit.run.checks import REQUIRED, verdict  # noqa: E402


class VerdictTests(unittest.TestCase):
    def test_a_failed_hint_leaves_the_status_and_shows_in_the_rationale(self):
        items = [{"name": name, "state": "passed"} for name in REQUIRED]
        status, rationale = verdict(items + [{"name": "precheck_lower_context", "state": "failed", "hint": True}])
        self.assertEqual(status, "passed")
        self.assertIn("precheck_lower_context failed (hint)", rationale)

    def test_a_failed_check_still_fails(self):
        items = [{"name": name, "state": "passed"} for name in REQUIRED]
        self.assertEqual(verdict(items + [{"name": "precheck", "state": "failed"}])[0], "failed")


if __name__ == "__main__":
    unittest.main()
