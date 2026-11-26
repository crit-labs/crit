"""Residency profiles on the host: a profile loads where its geometry is the cache's, and a profile of another geometry
is refused with both geometries named (a cache with a draft layer routes one group more than the model's layers)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[1] / "code"


def _module():
    import sys
    sys.path.insert(0, str(CODE))
    try:
        spec = importlib.util.spec_from_file_location("residency_profile_under_test", CODE / "residency_profile.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(CODE))


class ResidencyProfileTests(unittest.TestCase):
    def test_a_profile_of_the_caches_geometry_loads_and_another_is_refused_naming_both(self):
        try:
            residency_profile = _module()
        except ImportError as error:   # device_cache needs torch
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            path.write_text(json.dumps({"kind": residency_profile.KIND, "groups": 2, "experts": 4, "slots": 3,
                                        "keys": [5, 0, 7]}), encoding="utf-8")
            self.assertEqual(residency_profile.load_profile(path, 2, 4, 2), [5, 0])
            with self.assertRaisesRegex(ValueError, r"2 groups x 4 experts, where the cache routes 3 x 4"):
                residency_profile.load_profile(path, 3, 4, 2)


if __name__ == "__main__":
    unittest.main()
