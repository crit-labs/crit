"""crit from a checkout runs on that checkout's Core: a crit-core installed from elsewhere is refused with the install
that fixes it, and Core's loader finds the crit_core_steps library `crit core build` builds in the checkout."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))
sys.path.insert(0, str(ROOT / "cli"))

from crit_core import native  # noqa: E402


class CoreOfTheCheckoutTests(unittest.TestCase):
    def run_crit(self, folder, *arguments):
        code = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; from crit.cli import main; "
                "raise SystemExit(main(sys.argv[3:]))")
        return subprocess.run([sys.executable, "-B", "-c", code, folder, str(ROOT / "cli"), *arguments],
                              capture_output=True, text=True, timeout=120, cwd=folder)

    def test_a_core_installed_from_elsewhere_is_refused_with_the_fix(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "crit_core").mkdir()
            Path(folder, "crit_core", "__init__.py").write_text('__version__ = "0.0.0"\n', encoding="utf-8")
            result = self.run_crit(folder, "plugins", "list")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("runs on its own Core", result.stderr)
            self.assertIn("pip install -e " + str((ROOT / "core" / "python").resolve()), result.stderr)
            self.assertEqual(self.run_crit(folder, "--version").returncode, 0)

    def test_the_checkouts_own_core_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_crit(folder, "--version")
            self.assertEqual(result.returncode, 0, result.stderr)
            import crit.cli
            self.assertIsNone(crit.cli.core_mismatch())


class StepLibraryTests(unittest.TestCase):
    def test_the_variable_wins(self):
        with patch.dict(os.environ, {native.STEP_LIBRARY: "/somewhere/libcrit_core_steps.so"}):
            self.assertEqual(native.step_library(), "/somewhere/libcrit_core_steps.so")

    def test_without_the_variable_the_checkouts_build_is_found(self):
        environment = {k: v for k, v in os.environ.items() if k != native.STEP_LIBRARY}
        with patch.dict(os.environ, environment, clear=True):
            found = native.step_library()
        built = [ROOT / folder / name for folder in native.STEP_LIBRARY_FOLDERS for name in native.STEP_LIBRARY_FILES
                 if (ROOT / folder / name).is_file()]
        self.assertEqual(found, str(built[0]) if built else None)

    def test_crit_core_where_names_it_or_says_how_to_build_it(self):
        environment = {k: v for k, v in os.environ.items() if k != native.STEP_LIBRARY}
        result = subprocess.run([sys.executable, "-B", "-c", "import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
                                 "from crit.core import main; raise SystemExit(main(['where']))",
                                 str(ROOT / "core" / "python"), str(ROOT / "cli")],
                                capture_output=True, text=True, timeout=60, env=environment)
        if result.returncode:
            self.assertIn("crit core build", result.stderr)
        else:
            self.assertTrue(Path(result.stdout.strip()).is_file())


if __name__ == "__main__":
    unittest.main()
