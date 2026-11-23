"""crit_plugins.native: what is built on this machine at first use is built once, cached by content and recorded,
and a build that failed is remembered for the process."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "core" / "python"))

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

    def test_a_copy_of_the_same_sources_reuses_the_build_and_opens_a_library_of_its_own(self):
        from unittest import mock
        source = 'extern "C" int counter() { static int n = 0; return ++n; }\n#include "value.h"\n'
        with tempfile.TemporaryDirectory() as folder:
            copies = []
            for name in ("engine-a", "engine-b"):
                code = Path(folder) / name / "code"
                code.mkdir(parents=True)
                (code / "count.cpp").write_text(source, encoding="utf-8")
                (code / "value.h").write_text("#define VALUE 1\n", encoding="utf-8")
                copies.append(code)
            compiled = []
            real = native._compile

            def counting(*args):
                compiled.append(args[1][0][0])
                return real(*args)
            with mock.patch.dict(os.environ, {"CRIT_NATIVE_CACHE": str(Path(folder) / "cache")}), \
                    mock.patch.object(native, "_compile", counting):
                a = native.library("crit_test_count", [copies[0] / "count.cpp"])
                b = native.library("crit_test_count", [copies[1] / "count.cpp"])
                self.assertEqual(len(compiled), 1)                     # built once for both copies
                self.assertEqual((a.counter(), a.counter(), b.counter()), (1, 2, 1))   # each copy its own state
                # a header beside a source is part of what it builds from: a changed one builds again
                (copies[1] / "value.h").write_text("#define VALUE 2\n", encoding="utf-8")
                native.library("crit_test_count", [copies[1] / "count.cpp"])
                self.assertEqual(len(compiled), 2)


class StagingTests(unittest.TestCase):
    """extension(): torch's builder keys its builds by their paths, so two copies of the same files are hashed by
    content and built from one staged copy of them."""

    def test_two_copies_hash_alike_and_stage_into_one_folder(self):
        import hashlib
        from unittest import mock
        with tempfile.TemporaryDirectory() as folder:
            roots = []
            for name in ("a", "b"):
                code, include = Path(folder) / name / "code", Path(folder) / name / "include"
                (include / "crit").mkdir(parents=True)
                code.mkdir(parents=True)
                (code / "op.cu").write_text('#include "op.h"\n#include "crit/shared.h"\n', encoding="utf-8")
                (code / "op.h").write_text("// op\n", encoding="utf-8")
                (code / "notes.py").write_text("# not a header\n", encoding="utf-8")
                (include / "crit" / "shared.h").write_text("// shared\n", encoding="utf-8")
                roots.append((code, include))

            def digest(code, include):
                h = hashlib.sha256()
                native._content(h, [code / "op.cu"], [include])
                return h.hexdigest()
            self.assertEqual(digest(*roots[0]), digest(*roots[1]))
            with mock.patch.dict(os.environ, {"CRIT_NATIVE_CACHE": str(Path(folder) / "cache")}):
                staged = [native._stage("crit_test_op", digest(*r), [r[0] / "op.cu"], [r[1]]) for r in roots]
                self.assertEqual(staged[0], staged[1])
                (sources, includes) = staged[0]
                self.assertEqual(sources[0].read_text(encoding="utf-8"), (roots[0][0] / "op.cu").read_text(encoding="utf-8"))
                self.assertTrue((sources[0].parent / "op.h").is_file())
                self.assertTrue((includes[0] / "crit" / "shared.h").is_file())
                self.assertFalse((sources[0].parent / "notes.py").exists())
            (roots[1][1] / "crit" / "shared.h").write_text("// changed\n", encoding="utf-8")
            self.assertNotEqual(digest(*roots[0]), digest(*roots[1]))


if __name__ == "__main__":
    unittest.main()
