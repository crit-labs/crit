"""python-bindings on the host: the header, its C++ definitions and the ctypes loader agree; no device needed."""
import ctypes
import importlib.util
import os
from pathlib import Path
import re
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[1] / "inference/python-bindings/code"
spec = importlib.util.spec_from_file_location("crit_test_python_bindings", CODE / "crit_inference.py")
bindings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bindings)


def definitions(source):
    """{name: (return letter, argument letters)} of the crit_* functions a C++ file defines."""
    text = re.sub(r"//[^\n]*", "", source.read_text(encoding="utf-8"))
    table = {}
    for ret, name, args in re.findall(r"^(crit_status|int|size_t|uint32_t|const char\*) (crit_\w+)\(([^)]*)\)\s*\{",
                                      text, re.M):
        letters = "" if args.strip() == "void" else "".join(
            "p" if "*" in arg else bindings.LETTERS[" ".join(arg.split()).rsplit(" ", 1)[0]] for arg in args.split(","))
        table[name.removeprefix("crit_")] = ("s" if "*" in ret else bindings.LETTERS[ret], letters)
    return table


class PythonBindings(unittest.TestCase):
    def test_every_declaration_is_defined_with_its_signature(self):
        declared = bindings.signatures()
        self.assertEqual(declared, definitions(CODE / "crit_inference.cpp"))
        self.assertGreater(len(declared), 70)
        self.assertEqual(declared["st_gemv_fp8"], ("e", "pppiipiipiip"))
        self.assertEqual(declared["error_string"], ("s", "i"))

    def test_the_header_abi_and_plan_size_match_the_loader(self):
        header = (CODE / "crit_inference.h").read_text(encoding="utf-8")
        self.assertIn(f"#define CRIT_INFERENCE_ABI {bindings.ABI_VERSION}\n", header)
        self.assertIn(f"#define CRIT_PLAN_BYTES {bindings.PLAN_BYTES}\n", header)
        self.assertEqual(ctypes.sizeof(bindings.plan_buffer()), bindings.PLAN_BYTES)

    def test_a_missing_library_names_its_path_and_the_target(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "libcrit_inference_python.so"
            with self.assertRaisesRegex(FileNotFoundError, re.escape(str(missing)) + ".*crit_inference_python target"):
                bindings.Library(missing)
            previous = os.environ.get("CRIT_INFERENCE_LIBRARY")
            os.environ["CRIT_INFERENCE_LIBRARY"] = str(missing)
            try:
                self.assertEqual(bindings.default_library(), missing)
            finally:
                if previous is None:
                    del os.environ["CRIT_INFERENCE_LIBRARY"]
                else:
                    os.environ["CRIT_INFERENCE_LIBRARY"] = previous

    def test_read_into_fills_host_memory_and_refuses_a_short_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.bin"
            path.write_bytes(bytes(range(32)))
            target = ctypes.create_string_buffer(8)
            bindings.read_into(ctypes.addressof(target), path, 10, 8)
            self.assertEqual(target.raw, bytes(range(10, 18)))
            with self.assertRaisesRegex(OSError, "2 bytes short at offset 32"):
                bindings.read_into(ctypes.addressof(target), path, 26, 8)


if __name__ == "__main__":
    unittest.main()
