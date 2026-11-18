"""Core's layering: Core knows no model, no plugin and nothing above it.

Three rules, read from the source tree (nothing is imported or built):

1. Core's Python (crit_core, crit_plugins) imports the standard library, itself and the few libraries it is built
   on. Nothing above it: not the CLI, Eval, the API, the plugin library or a model package.
2. Core's C++ includes its own headers and system headers only.
3. Core names no plugin of the library and no model family. The names it holds today (examples in docstrings, one
   message, the formats' own architecture names) are counted in support/core_layering_allowlist.json. The list may
   only shrink: a new name fails, and so does an entry that no longer applies, so the entry leaves with its fix.
"""
import ast
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
PACKAGES = ("crit_core", "crit_plugins")
ALLOWLIST = ROOT / "tests/support/core_layering_allowlist.json"

# What Core is built on. crit_kernels is the kernel workbench: Core's kernel loader calls it (crit_plugins.kernels),
# and its exactness levels are the ones an engine's configuration asks (crit_core.engine).
LIBRARIES = {
    "crit_core": {"crit_core", "crit_kernels", "crit_plugins", "jinja2", "numpy", "tokenizers", "torch"},
    "crit_plugins": {"crit_core", "crit_plugins", "crit_kernels", "numpy", "torch"},
}
SOURCE = {".py", ".h", ".cpp", ".cu", ".cuh", ".json", ".cmake", ".txt"}
# Folder names of the library that are plain words, not plugin ids.
PLAIN = {"base", "examples", "code", "tests"}
# Headers Core's own build writes (core/CMakeLists.txt).
GENERATED = {"crit_step_settings.inc"}
FAMILIES = ("qwen", "llama", "gemma", "deepseek", "glm", "mistral", "nanbeige", "minimax", "gpt", "granite", "kimi",
            "hermes", "phi")


def relative(path):
    return path.relative_to(ROOT).as_posix()


def core_files():
    return sorted(f for f in CORE.rglob("*") if f.is_file() and f.suffix in SOURCE and "__pycache__" not in f.parts)


def plugin_ids():
    folders = [*(ROOT / "inference").iterdir(), *(ROOT / "inference/base").iterdir(),
               *(ROOT / "kernels/techniques").glob("*/*")]
    return sorted({f.name for f in folders if f.is_dir()} - PLAIN)


def counts(words, pattern, lower=False):
    """{word: {file: occurrences}} of each word in Core's source."""
    found = {}
    for path in core_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if lower:
            text = text.lower()
        for word in words:
            n = len(re.findall(pattern.format(re.escape(word)), text))
            if n:
                found.setdefault(word, {})[relative(path)] = n
    return found


def measured():
    return {"plugins": counts(plugin_ids(), r"(?<![\w-]){}(?![\w-])"),
            "families": counts(FAMILIES, r"(?<![a-z0-9]){}(?![a-z])", lower=True)}


class CoreLayering(unittest.TestCase):
    def test_python_imports_nothing_above_core(self):
        standard = set(sys.stdlib_module_names)
        above = []
        for package in PACKAGES:
            for path in sorted((CORE / "python" / package).rglob("*.py")):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        modules = [node.module]
                    else:
                        continue
                    for module in modules:
                        top = module.split(".")[0]
                        if top not in standard and top not in LIBRARIES[package]:
                            above.append(f"{relative(path)}:{node.lineno} imports {module}")
        self.assertEqual(above, [], "Core imports a module above it")

    def test_cpp_includes_core_and_system_headers_only(self):
        include = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.M)
        own = {f.name for f in CORE.rglob("*") if f.suffix in {".h", ".cuh"}}
        foreign = []
        for path in core_files():
            if path.suffix not in {".h", ".cpp", ".cu", ".cuh"}:
                continue
            for quote, header in include.findall(path.read_text(encoding="utf-8", errors="ignore")):
                if header.startswith("crit/") and not (CORE / "include" / header).is_file():
                    foreign.append(f"{relative(path)} includes {header}, which Core does not hold")
                elif quote == '"' and not header.startswith("crit/") and Path(header).name not in own | GENERATED:
                    foreign.append(f"{relative(path)} includes {header}")
        self.assertEqual(foreign, [], "Core's C++ includes a header outside Core")

    def test_core_names_no_plugin_and_no_model_family(self):
        allowed = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
        problems = []
        for kind, found in measured().items():
            keys = {(w, f) for w, files in found.items() for f in files}
            keys |= {(w, f) for w, files in allowed[kind].items() for f in files}
            for word, path in sorted(keys):
                have = found.get(word, {}).get(path, 0)
                may = allowed[kind].get(word, {}).get(path, 0)
                if have > may:
                    problems.append(f"{path} names {word!r} {have} times (allowed {may}): Core names no "
                                    f"{'plugin' if kind == 'plugins' else 'model family'}")
                elif have < may:
                    problems.append(f"{path} names {word!r} {have} times, the allowlist says {may}: lower the entry")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:      # after removing names from Core: the list as the tree stands
        ALLOWLIST.write_text(json.dumps(measured(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    else:
        unittest.main()
