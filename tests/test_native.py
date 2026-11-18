"""crit_plugins.native: what is built on this machine at first use is built once, cached by content and recorded,
and a build that failed is remembered for the process."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))

from crit_plugins import native  # noqa: E402


def buildable():
    try:
        native.compiler()
        return True
    except native.NativeUnavailable:
        return False


@unittest.skipUnless(buildable(), "no C++ compiler for native code built at load")
class LibraryTests(unittest.TestCase):
    def test_a_library_is_built_once_and_a_source_that_fails_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            good, bad = Path(folder) / "good.cpp", Path(folder) / "bad.cpp"
            good.write_text('extern "C" int answer() { return 42; }\n', encoding="utf-8")
            bad.write_text("this is not C++\n", encoding="utf-8")
            os.environ["CRIT_NATIVE_CACHE"] = str(Path(folder) / "cache")
            try:
                lib = native.library("crit_test_good", [good])
                self.assertEqual(lib.answer(), 42)
                self.assertIs(native.library("crit_test_good", [good]), lib)
                with self.assertRaisesRegex(native.NativeUnavailable, "bad.cpp did not compile"):
                    native.library("crit_test_bad", [bad])
                bad.unlink()   # the failure is remembered: the source is not read or compiled again
                self.assertEqual(native.built()["crit_test_good"]["source"], "built")
                self.assertEqual(native.built()["crit_test_bad"]["source"], "unavailable")
            finally:
                del os.environ["CRIT_NATIVE_CACHE"]


if __name__ == "__main__":
    unittest.main()
