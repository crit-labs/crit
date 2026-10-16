"""crit_core.files and crit_core.jsonfile: model file sources, names, size caps and the strict JSON reader (offline)."""
import errno
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core/python"))

from crit_core import jsonfile  # noqa: E402
from crit_core.files import (CallableFiles, DirectoryFiles, MappingFiles, ModelFiles, ModelFilesError,  # noqa: E402
                             file_name, model_files)


class _Unreadable:
    """A file whose bytes must not be read (it can still be stat-ed)."""

    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *details):
        self.handle.close()

    def fileno(self):
        return self.handle.fileno()

    def read(self, *details):
        raise AssertionError("read before the size check")


def unreadable_open(file, mode="r", *details, **options):
    return _Unreadable(open(file, mode, *details, **options))


def symlink_or_skip(test, target, link):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as error:
        test.skipTest(f"no symlinks here: {error}")


class FileNameTests(unittest.TestCase):
    def test_portable_relative_names_only(self):
        for name in ("config.json", "additional_chat_templates/tool_use.jinja", "CONSOLE.txt", "con1.txt", "a/b.c",
                     "COMX", "lpt", ".hidden", "a..b"):
            with self.subTest(name=name):
                self.assertEqual(file_name(name), name)
        refused = ("", "/etc/passwd", "a//b", "a/", "./a", "a/./b", "../x", "a/../b", "a\\b", "a\0b",
                   "C:/Windows/win.ini", "C:secret.txt", "tokenizer.json:stream", "a/b:c", ":", "CON", "con", "NUL.txt",
                   "aux.tar.gz", "COM1", "com9.json", "LPT1", "a/PRN", "x/nul/y", "CON .txt", "COM\u00b9",
                   "lpt\u00b3.txt", "a/LPT\u00b2", "com\u00b2 .json", 7, None, b"config.json")
        for name in refused:
            with self.subTest(name=name), self.assertRaisesRegex(ModelFilesError, "invalid model file name"):
                file_name(name)


class ReadLimitTests(unittest.TestCase):
    def test_every_source_refuses_files_above_max_bytes_before_recording_them(self):
        data = b"x" * 10
        for files in (MappingFiles({"a.bin": data}), CallableFiles({"a.bin": data}.get)):
            with self.subTest(files=type(files).__name__):
                with self.assertRaisesRegex(ModelFilesError, "^a.bin: 10 bytes exceeds the 9-byte limit$"):
                    files.read("a.bin", max_bytes=9)
                self.assertEqual(files.records, ())
                self.assertEqual(files.read("a.bin", max_bytes=10), data)
                self.assertEqual(files.read("a.bin"), data)
                self.assertIsNone(files.read("absent.bin", max_bytes=1))
                for value in (0, -1, 1.5, True, "9"):
                    with self.subTest(max_bytes=value), self.assertRaisesRegex(ValueError, "max_bytes"):
                        files.read("a.bin", max_bytes=value)

    def test_a_directory_checks_the_size_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "big.json")
            with open(path, "wb") as handle:
                handle.truncate(400 << 20)   # sparse
            files = DirectoryFiles(directory)
            with mock.patch("crit_core.files.open", unreadable_open, create=True):
                with self.assertRaisesRegex(ModelFilesError, "^big.json: 419430400 bytes exceeds the 16777216-byte "
                                                             "limit$"):
                    files.read("big.json", max_bytes=16 << 20)
            self.assertEqual(files.records, ())
            path.write_bytes(b"12345")
            self.assertEqual(files.read("big.json", max_bytes=5), b"12345")

    def test_a_directory_raises_for_names_that_exist_but_are_not_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "folder.json").mkdir()
            (root / "file.json").write_text("{}")
            files = DirectoryFiles(root)
            with self.assertRaisesRegex(ModelFilesError, "^folder.json: not a regular file$"):
                files.read("folder.json")
            for absent in ("missing.json", "nested/missing.json", "file.json/inner"):
                with self.subTest(name=absent):
                    self.assertIsNone(files.read(absent))
            if hasattr(os, "mkfifo"):
                os.mkfifo(root / "pipe")
                with self.assertRaisesRegex(ModelFilesError, "^pipe: not a regular file$"):
                    files.read("pipe")
            symlink_or_skip(self, root / "blobs" / "missing-blob", root / "dangling.json")
            with self.assertRaisesRegex(ModelFilesError, "^dangling.json: open: "):
                files.read("dangling.json")
            symlink_or_skip(self, root / "file.json", root / "linked.json")
            self.assertEqual(files.read("linked.json"), b"{}")
            # Links that exist but resolve to nothing fail like a dangling one: a loop, and a target through a file.
            os.symlink("loop.json", root / "loop.json")
            os.symlink("file.json/inner", root / "through.json")
            os.symlink("loop.json/inner", root / "past-loop.json")
            for name in ("loop.json", "through.json", "past-loop.json", "loop.json/inner"):
                with self.subTest(name=name), self.assertRaisesRegex(ModelFilesError, f"^{name}: open: "):
                    files.read(name)
            self.assertEqual([record.path for record in files.records], ["linked.json"])

    def test_a_directory_raises_when_a_file_cannot_be_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "file.json").write_text("{}")
            files = DirectoryFiles(directory)
            denied = PermissionError(errno.EACCES, os.strerror(errno.EACCES))
            with mock.patch("crit_core.files.os.open", side_effect=denied):
                with self.assertRaisesRegex(ModelFilesError, f"^file.json: open: {os.strerror(errno.EACCES)}$"):
                    files.read("file.json")
            with mock.patch("crit_core.files.os.lstat", side_effect=denied):
                with self.assertRaisesRegex(ModelFilesError, f"^file.json: open: {os.strerror(errno.EACCES)}$"):
                    files.read("file.json")
            self.assertEqual(files.records, ())
            self.assertEqual(files.read("file.json"), b"{}")


class NamesTests(unittest.TestCase):
    def test_names_filter_by_prefix_first_and_skip_refused_names(self):
        listed = CallableFiles(lambda name: None, names=["bad\\name", "a/../b", "templates/x.jinja", "templates/CON",
                                                        "templates/y:z", "other/C:x", "templates/n/deep.jinja", 7])
        self.assertEqual(listed.names("templates/"), ["templates/n/deep.jinja", "templates/x.jinja"])
        self.assertEqual(listed.names(), ["templates/n/deep.jinja", "templates/x.jinja"])
        with tempfile.TemporaryDirectory() as directory:
            for files in (listed, MappingFiles({}), DirectoryFiles(directory)):
                with self.subTest(files=type(files).__name__), self.assertRaisesRegex(TypeError, "prefix must be"):
                    files.names(None)

    def test_a_directory_lists_only_the_folder_the_prefix_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("templates/a.jinja", "templates/sub/b.jinja", "other/c.txt", "model-1.bin", "model-2.bin",
                         "readme.md"):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text("x")
            (root / "templates" / "folder.jinja").mkdir()
            if os.name == "posix":   # legal POSIX names that crit_core.files refuses
                (root / "notes\\draft.txt").write_text("x")
                (root / "templates" / "odd:name.jinja").write_text("x")
            files = DirectoryFiles(root)
            walked = []
            original = Path.rglob

            def rglob(path, pattern, *details, **options):
                walked.append(path)
                return original(path, pattern, *details, **options)

            with mock.patch.object(Path, "rglob", rglob):
                self.assertEqual(files.names("templates/"), ["templates/a.jinja", "templates/sub/b.jinja"])
                self.assertEqual(walked, [root / "templates"])
                self.assertEqual(files.names("model-"), ["model-1.bin", "model-2.bin"])
                self.assertEqual(files.names("absent/"), [])
                self.assertEqual(files.names("../"), [])
            self.assertEqual(files.names(), ["model-1.bin", "model-2.bin", "other/c.txt", "readme.md",
                                             "templates/a.jinja", "templates/sub/b.jinja"])


class ModelFilesHelperTests(unittest.TestCase):
    def test_paths_become_directories_sources_pass_through_and_other_types_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            for source in (directory, Path(directory)):
                with self.subTest(source=source):
                    files = model_files(source)
                    self.assertIsInstance(files, DirectoryFiles)
                    self.assertEqual(files.root, Path(directory))
            with self.assertRaisesRegex(ModelFilesError, "is not a directory"):
                model_files(Path(directory, "missing"))
        source = MappingFiles({})
        self.assertIs(model_files(source), source)

        class LookAlike:
            def read(self, name, max_bytes=None):
                return None

            def names(self, prefix=""):
                return []

        for value in (42, None, b"/model", LookAlike()):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "directory path or a crit_core.files"):
                model_files(value)

    def test_subclasses_implementing_only_read_and_names_still_work(self):
        class Legacy(ModelFiles):
            def _read(self, name):
                return b"abc" if name == "a" else None

            def _names(self):
                return ["a", "bad\\name"]

        files = Legacy()
        self.assertEqual((files.read("a"), files.read("b"), files.names()), (b"abc", None, ["a"]))
        with self.assertRaisesRegex(ModelFilesError, "^a: 3 bytes exceeds the 2-byte limit$"):
            files.read("a", max_bytes=2)

    def test_directory_subclasses_overriding_read_or_names_are_consulted(self):
        class Hidden(DirectoryFiles):   # the documented hooks, below the class that adds the limited ones
            def _read(self, name):
                return None if name == "secret.json" else super()._read(name)

            def _names(self):
                return [name for name in super()._names() if name != "secret.json"]

        class Limited(DirectoryFiles):
            def _read_limited(self, name, max_bytes):
                self.calls.append((name, max_bytes))
                return super()._read_limited(name, max_bytes)

            def _names_under(self, prefix):
                self.calls.append(prefix)
                return super()._names_under(prefix)

        with tempfile.TemporaryDirectory() as directory:
            for name in ("secret.json", "open.json", "sub/x.json"):
                Path(directory, name).parent.mkdir(exist_ok=True)
                Path(directory, name).write_text("{}")
            hidden = Hidden(directory)
            self.assertEqual((hidden.read("secret.json"), hidden.read("open.json", max_bytes=2)), (None, b"{}"))
            self.assertEqual((hidden.names(), hidden.names("sub/")), (["open.json", "sub/x.json"], ["sub/x.json"]))
            with self.assertRaisesRegex(ModelFilesError, "^open.json: 2 bytes exceeds the 1-byte limit$"):
                hidden.read("open.json", max_bytes=1)
            limited = Limited(directory)
            limited.calls = []
            self.assertEqual((limited.read("open.json", max_bytes=2), limited.names("sub/")), (b"{}", ["sub/x.json"]))
            self.assertEqual(limited.calls, [("open.json", 2), "sub/"])


class JsonFileTests(unittest.TestCase):
    def test_defaults_match_the_native_reader(self):
        self.assertEqual((jsonfile.MAX_BYTES, jsonfile.MAX_DEPTH, jsonfile.MAX_CONFIG_BYTES),
                         (128 << 20, 128, 16 << 20))
        for options in ({"max_bytes": 0}, {"max_depth": 0}, {"max_bytes": True}, {"max_depth": -1}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "json limits must be positive"):
                jsonfile.loads("{}", **options)

    def test_unpaired_surrogate_escapes_are_errors_as_in_the_native_reader(self):
        for text in ('"\\ud800"', '{"\\ud800": 1}', '{"a": "\\udc00"}', '["x\\ud83d\\u0041"]', '["\\ude00\\ud83d"]',
                     '{"a": {"b": ["\\udbff"]}}', '"\\ud83d"'):
            with self.subTest(text=text), self.assertRaisesRegex(jsonfile.JsonError,
                                                                 "^x.json: unpaired surrogate escape$"):
                jsonfile.loads(text.encode(), "x.json")
        self.assertEqual(jsonfile.loads(b'{"a": "\\ud83d\\ude00", "\\uD83D\\uDE00": 1}'),
                         {"a": "\U0001F600", "\U0001F600": 1})
        self.assertEqual(jsonfile.loads(b'["\\\\ud800"]'), ["\\ud800"])   # an escaped backslash, not an escape
        with self.assertRaisesRegex(jsonfile.JsonError, "^x.json: not UTF-8 \\(surrogates not allowed at character 2"):
            jsonfile.loads('["\ud800"]', "x.json")   # text input holding a lone surrogate itself

    def test_integers_beyond_the_digit_limit_are_json_errors_naming_the_source(self):
        for allow_real in (True, False):
            with self.subTest(allow_real=allow_real), self.assertRaisesRegex(jsonfile.JsonError,
                                                                             "^x.json: Exceeds the limit"):
                jsonfile.loads('{"a": ' + "1" * 5000 + "}", "x.json", allow_real=allow_real)
        self.assertEqual(jsonfile.loads('{"a": ' + "9" * 300 + "}")["a"], int("9" * 300))   # exact, any size

    def test_non_finite_literals_load_only_when_asked(self):
        text = '{"a": [NaN, Infinity, -Infinity, 1.5]}'
        for literal in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(literal=literal), self.assertRaisesRegex(jsonfile.JsonError,
                                                                       f"^x.json: non-finite number {literal}$"):
                jsonfile.loads(f'[{literal}]', "x.json")
        values = jsonfile.loads(text, allow_nonfinite=True)["a"]
        self.assertTrue(math.isnan(values[0]))
        self.assertEqual(values[1:], [math.inf, -math.inf, 1.5])
        for text in ("[1e400]", "[-1e999]"):
            with self.subTest(text=text), self.assertRaisesRegex(jsonfile.JsonError, "non-finite number"):
                jsonfile.loads(text, allow_nonfinite=True)
        self.assertEqual(jsonfile.loads("[1e-400]", allow_nonfinite=True), [0.0])
        with self.assertRaisesRegex(jsonfile.JsonError, "only integers"):
            jsonfile.loads("[NaN]", allow_real=False, allow_nonfinite=True)

    def test_nesting_is_bounded_like_the_native_reader(self):
        def nested(depth):
            return "[" * depth + "]" * depth

        self.assertEqual(jsonfile.loads(nested(128)), jsonfile.loads(nested(128)))
        for depth in (129, 500, 1000, 5000, 200000):
            with self.subTest(depth=depth), self.assertRaisesRegex(jsonfile.JsonError,
                                                                   "^x.json: nesting exceeds the depth limit of 128$"):
                jsonfile.loads(nested(depth), "x.json")
        self.assertEqual(jsonfile.loads('{"a": ' + nested(63) + "}", max_depth=64)["a"], jsonfile.loads(nested(63)))
        with self.assertRaisesRegex(jsonfile.JsonError, "depth limit of 64$"):
            jsonfile.loads('{"a": ' + nested(64) + "}", max_depth=64)
        with self.assertRaisesRegex(jsonfile.JsonError, "depth limit of 3$"):
            jsonfile.loads('[{"a": [{}]}]', max_depth=3)
        self.assertEqual(jsonfile.loads('["[[[[", {"a": "]]"}]', max_depth=2), ["[[[[", {"a": "]]"}])

    def test_read_checks_the_file_before_reading_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "a.json")
            with open(path, "wb") as handle:
                handle.truncate(200 << 20)
            with mock.patch("crit_core.jsonfile.open", unreadable_open, create=True), \
                    self.assertRaisesRegex(jsonfile.JsonError, "209715200 bytes exceeds the 134217728-byte limit$"):
                jsonfile.read(path)
            path.write_text('{"a": NaN}')
            self.assertTrue(math.isnan(jsonfile.read(path, allow_nonfinite=True)["a"]))
            with self.assertRaisesRegex(jsonfile.JsonError, "not a regular file$"):
                jsonfile.read(directory)
            with self.assertRaises(FileNotFoundError):
                jsonfile.read(Path(directory, "missing.json"))


if __name__ == "__main__":
    unittest.main()
