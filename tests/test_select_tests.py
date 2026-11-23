"""tests/select_tests.py: the test modules a change touches, by the imports they follow and the library items they
name."""
from pathlib import Path
import tempfile
import unittest

import select_tests

FILES = {
    "tests/test_store.py": "from support import runs\n",
    "tests/test_cli.py": "def test():\n    from pkg.cli import main\n",
    "tests/test_library.py": "PLUGIN = 'my-plugin'\nNOTES = 'NOTES.md'\n",
    "tests/test_alone.py": "import json\n",
    "tests/support/__init__.py": "",
    "tests/support/runs.py": "from pkg.run.store import Store\n",
    "cli/pkg/__init__.py": "VERSION = 1\n",
    "cli/pkg/cli.py": "from . import records\n",
    "cli/pkg/records.py": "",
    "cli/pkg/unused.py": "",
    "cli/pkg/run/__init__.py": "",
    "cli/pkg/run/store.py": "from ..records import read\n",
    "library/slot/my-plugin/plugin.json": "{}",
    "library/slot/my-plugin/code/rows.py": "",
    "library/slot/other-plugin/plugin.json": "{}",
    "NOTES.md": "",
}


class SelectTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        for name, text in FILES.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def select(self, *changed):
        return select_tests.select(changed, self.root, ("tests", "cli"))

    def test_a_changed_module_selects_the_tests_that_import_it_through_any_module(self):
        self.assertEqual(self.select("cli/pkg/run/store.py"), (["test_store"], []))
        # records is imported by the store (relatively, two packages up) and by the CLI, itself imported in a function
        self.assertEqual(self.select("cli/pkg/records.py"), (["test_cli", "test_store"], []))
        # importing a module runs its packages' __init__
        self.assertEqual(self.select("cli/pkg/__init__.py"), (["test_cli", "test_store"], []))
        self.assertEqual(self.select("tests/support/runs.py"), (["test_store"], []))
        self.assertEqual(self.select("tests/test_alone.py"), (["test_alone"], []))

    def test_a_file_no_import_reaches_selects_the_tests_that_name_its_item_or_itself(self):
        self.assertEqual(self.select("library/slot/my-plugin/code/rows.py"), (["test_library"], []))
        self.assertEqual(self.select("library/slot/my-plugin/plugin.json"), (["test_library"], []))
        self.assertEqual(self.select("NOTES.md"), (["test_library"], []))

    def test_a_file_no_test_imports_or_names_is_unmatched(self):
        self.assertEqual(self.select("cli/pkg/unused.py", "library/slot/other-plugin/plugin.json", "cli/pkg/cli.py"),
                         (["test_cli"], ["cli/pkg/unused.py", "library/slot/other-plugin/plugin.json"]))

    def test_every_test_module_of_this_checkout_reaches_itself(self):
        modules = select_tests.index()
        reached = select_tests.reached(modules)
        self.assertIn("test_select_tests", reached)
        self.assertIn("select_tests", reached["test_select_tests"])
        self.assertIn("crit.run.store", reached["test_run_records"])


if __name__ == "__main__":
    unittest.main()
