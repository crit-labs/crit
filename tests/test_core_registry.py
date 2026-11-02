"""Core's registry of model-family knowledge (crit_core.registry): sources registered at run time or named by entry
points, imported only when their kind is first read, the first source that gives a key keeping it."""
from pathlib import Path
import importlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))
from crit_core import registry  # noqa: E402
from crit_core.registry import RegistryError  # noqa: E402


class Fresh:
    """The registry as a new process holds it (no sources, entry points read again), restored afterwards."""

    def __enter__(self):
        self.saved = ({kind: list(sources) for kind, sources in registry._sources.items()}, dict(registry._merged),
                      dict(registry._installed))
        for sources in registry._sources.values():
            sources.clear()
        registry._merged.clear()
        registry._installed.clear()
        for kind in registry.KINDS:
            registry._generation[kind] += 1

    def __exit__(self, *_):
        sources, merged, installed = self.saved
        for kind in registry.KINDS:
            registry._sources[kind][:] = sources[kind]
            registry._generation[kind] += 1
        registry._merged.clear()
        registry._merged.update(merged)
        registry._installed.clear()
        registry._installed.update(installed)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        sys.path.insert(0, str(self.folder))
        self.modules = []
        fresh = Fresh()
        fresh.__enter__()
        self.addCleanup(fresh.__exit__)
        # Only the test's own distributions: crit-eval installed beside crit registers entry points of both kinds.
        import importlib.metadata
        folder = str(self.folder)
        own = patch.object(importlib.metadata, "entry_points", lambda group: [
            point for dist in importlib.metadata.distributions(path=[folder]) for point in dist.entry_points
            if point.group == group])
        own.start()
        self.addCleanup(own.stop)

    def tearDown(self):
        sys.path.remove(str(self.folder))
        for name in self.modules:
            sys.modules.pop(name, None)
        self.temp.cleanup()

    def module(self, name, source):
        (self.folder / f"{name}.py").write_text(source, encoding="utf-8")
        self.modules.append(name)
        # The import system caches each folder's listing by its timestamp, which a file written in the same tick
        # leaves unchanged on Windows.
        importlib.invalidate_caches()

    def test_sources_load_when_their_kind_is_first_read(self):
        self.module("crit_test_rules", "RULES = {'arch': 'from a module'}\n")
        registry.register("gguf_config", "crit_test_rules:RULES")
        registry.register("gguf_config", {"arch": "from a mapping", "other": 2})
        registry.register("gguf_config", "crit_test_rules:RULES")   # once only
        self.assertNotIn("crit_test_rules", sys.modules)
        entries = registry.entries("gguf_config")
        self.assertEqual(dict(entries), {"arch": "from a module", "other": 2})   # the first source keeps "arch"
        self.assertEqual(dict(registry.entries("chat_templates")), {})
        with self.assertRaises(TypeError):
            entries["new"] = 1   # read-only
        registry.register("gguf_config", {"later": 3})
        self.assertEqual(registry.entries("gguf_config")["later"], 3)

    def test_a_source_that_registers_another_while_it_loads(self):
        self.module("crit_test_more", "from crit_core import registry\n"
                                      "registry.register('chat_templates', {'b': 2})\nTEMPLATES = {'a': 1}\n")
        registry.register("chat_templates", "crit_test_more:TEMPLATES")
        self.assertEqual(dict(registry.entries("chat_templates")), {"a": 1, "b": 2})

    def test_installed_entry_points(self):
        self.module("crit_test_provider", "TEMPLATES = {'family': 'installed', 'own': 'installed'}\n")
        info = self.folder / "crit_test_provider-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: crit-test-provider\nVersion: 1.0\n",
                                       encoding="utf-8")
        (info / "entry_points.txt").write_text("[crit_core.chat_templates]\nprovider = crit_test_provider:TEMPLATES\n",
                                               encoding="utf-8")
        registry.register("chat_templates", {"own": "registered"})
        self.assertEqual(dict(registry.entries("chat_templates")), {"own": "registered", "family": "installed"})
        self.assertNotIn("crit_test_provider", dict(registry.entries("gguf_config")))

    def test_refusals(self):
        self.module("crit_test_scalar", "RULES = 7\n")
        with self.assertRaisesRegex(RegistryError, "no registry kind 'templates'"):
            registry.register("templates", {})
        with self.assertRaisesRegex(RegistryError, "no registry kind"):
            registry.entries("templates")
        with self.assertRaisesRegex(RegistryError, "a mapping or 'module:attribute', not list"):
            registry.register("gguf_config", [])
        registry.register("gguf_config", "crit_test_scalar:RULES")
        with self.assertRaisesRegex(RegistryError, r"'crit_test_scalar:RULES' is not a mapping \(int\)"):
            registry.entries("gguf_config")
        with Fresh():
            registry.register("gguf_config", "crit_test_absent:RULES")
            with self.assertRaisesRegex(RegistryError, "source 'crit_test_absent:RULES' does not load"):
                registry.entries("gguf_config")


if __name__ == "__main__":
    unittest.main()
