"""sandwich: its code mounts beside model-template's and registers what entry.json says it provides."""
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]


class TechniqueTests(unittest.TestCase):
    def test_it_registers_what_it_provides(self):
        from crit.models import mounted   # crit's CLI: the technique as model-template mounts it
        entry = json.loads((HERE / "entry.json").read_text(encoding="utf-8"))
        module = mounted(entry["source"])
        for kind, names in entry["provides"].items():
            self.assertEqual(sorted(getattr(module, kind.upper())), sorted(names))


if __name__ == "__main__":
    unittest.main()
