"""crit_kernels.build.select: a configs= build runs the configuration a sweep selects, counted with the structure's
defaults first; an index past the build's configurations is refused with them named."""
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("inference/core/python", "cli", "kernels"):
    sys.path.insert(0, str(ROOT / _folder))
from crit_kernels.build import select  # noqa: E402


class SelectTests(unittest.TestCase):
    def test_an_index_past_the_builds_configurations_is_refused_naming_them(self):
        # A build given 48 configurations, one equal to the defaults, holds 48 (the defaults first, not twice): the
        # 49th index raised IndexError at the kernel's first launch, which a hand-made knob probe read as 47 of its 48
        # configurations failing.
        name = "crit_test_select_fixture"
        tuning = types.ModuleType(name + "._crit_tuning")
        tuning.TUNING = {"window": {"configs": [{"STAGES": 3}, {"STAGES": 2}], "table": []}}
        tuning.SELECTED = {}
        sys.modules[tuning.__name__] = tuning
        try:
            module = types.SimpleNamespace(__name__=name)
            select(module, "window", 1)
            self.assertEqual(tuning.SELECTED, {"window": 1})
            with self.assertRaisesRegex(IndexError, r"window was built with 2 configurations .*not 3"):
                select(module, "window", 2)
            self.assertEqual(tuning.SELECTED, {"window": 1})
            select(module, "window", -1)
            self.assertEqual(tuning.SELECTED, {})
        finally:
            sys.modules.pop(tuning.__name__, None)


if __name__ == "__main__":
    unittest.main()
