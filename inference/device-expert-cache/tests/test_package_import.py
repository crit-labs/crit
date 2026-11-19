"""The code folder imports as a package's subpackage, as an engine's plugin loads it (crit_plugins.assemble's
sources), not only from code/: a module importing a sibling by its bare name fails there."""
import importlib
from pathlib import Path
import sys
import types
import unittest

CODE = Path(__file__).resolve().parents[1] / "code"


class PackageImportTests(unittest.TestCase):
    def test_every_module_imports_as_a_subpackage(self):
        name = "crit_test_device_expert_cache_code"
        package = types.ModuleType(name)
        package.__path__ = [str(CODE)]
        saved = list(sys.path)
        sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != CODE]
        sys.modules[name] = package
        try:
            for module in ("residency_profile", "device_cache", "trace_replay", "replay"):
                if (CODE / f"{module}.py").is_file():
                    importlib.import_module(f"{name}.{module}")
        finally:
            sys.path[:] = saved
            for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
                del sys.modules[key]


if __name__ == "__main__":
    unittest.main()
