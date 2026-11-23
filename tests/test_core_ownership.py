"""Core's ownership: what decides how an engine computes and schedules on the device is Core's, defined once.

core/README.md#what-core-decides gives the rule. Three checks hold it, read from the source tree (nothing is imported
or built):

1. Each value Core owns (OWNED: the exactness levels and phases, the quality modes, the cancel grace, the device
   memory an engine leaves free) is assigned in its one Core module, and anywhere else in crit (the CLI, the API, the
   kernels, smoke, the library, Core's other modules) it is imported. Its names assigned elsewhere count, and so
   does a literal copy of its values under any name (a tuple of the quality modes as an option's choices).
2. Core's Python imports nothing from crit_kernels.exactness: the levels are Core's, and crit_kernels reads a
   kernel.json against them.
3. Core's Python keeps no engine's policy in module state: no function rebinds a module global (`global`) but the
   process's own state (PROCESS), so two engines in one process never see each other's policy.

Today's exceptions are counted in support/core_ownership_allowlist.json. The list may only shrink: a new exception
fails, and so does an entry that no longer applies, so the entry leaves with its fix. Tests may spell the values out.
"""
import ast
import builtins
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "inference" / "core"
PYTHON = CORE / "python"
ALLOWLIST = ROOT / "tests/support/core_ownership_allowlist.json"

# What Core owns: its home (the one module that assigns it), the name it has there, the names a copy goes by
# elsewhere, and its values where a literal collection of them is a copy under any name.
OWNED = {
    "exactness levels": {"home": "inference/core/python/crit_plugins/exactness.py", "name": "LEVELS",
                         "names": ("LEVELS", "EXACTNESS_LEVELS"),
                         "values": {"repeatable", "row-exact", "exact", "declared"}},
    # A dict keyed by the phases is a value per phase (a plugin's exactness()), not a copy: only collections count.
    "exactness phases": {"home": "inference/core/python/crit_plugins/exactness.py", "name": "PHASES", "names": (),
                         "values": {"decode", "prompt"}, "keys": False},
    "quality modes": {"home": "inference/core/python/crit_plugins/exactness.py", "name": "QUALITY_MODES",
                      "names": ("QUALITY_MODES", "MODE_LEVELS"), "values": {"speed", "preserve"}},
    "cancel grace": {"home": "inference/core/python/crit_plugins/steps.py", "name": "GRACE",
                     "names": ("GRACE", "CANCEL_GRACE", "CANCEL_GRACE_SECONDS")},
    "VRAM headroom": {"home": "inference/core/python/crit_plugins/memory.py", "name": "HEADROOM",
                      "names": ("HEADROOM", "VRAM_HEADROOM", "VRAM_HEADROOM_GIB", "VRAM_HEADROOM_BYTES",
                                "DEFAULT_HEADROOM")},
}
# Module state of the process, not of an engine: what `global` may rebind in Core's Python.
PROCESS = {
    "inference/core/python/crit_plugins/compiles.py": {"_recording"},   # the process's compile recorder (torch's are process-wide)
    "inference/core/python/crit_plugins/host/__init__.py": {"_current"},   # the process's one host: its cores and worker pool
    "inference/core/python/crit_plugins/memory.py": {"_CU"},   # the CUDA driver library, loaded once a process
    "inference/core/python/crit_plugins/policy.py": {"_process"},   # the policy outside any engine (tests, a quick try)
}
SKIP = {"__pycache__", "build", "manifests", "tests"}   # tests spell values out; checkpoint manifests are data
BUILTINS = set(dir(builtins))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def sources():
    """crit's Python source outside the tests."""
    return sorted(f for f in ROOT.rglob("*.py")
                  if not SKIP & set(f.relative_to(ROOT).parts[:-1]) and not f.name.startswith("test_"))


def literal(node):
    """Whether an assigned value is written out rather than taken from elsewhere: it names nothing but builtins."""
    return all(n.id in BUILTINS for n in ast.walk(node) if isinstance(n, ast.Name)) and \
        not any(isinstance(n, ast.Attribute) for n in ast.walk(node))


def collection(node):
    """(the strings of a literal tuple, list or set of strings, or of a dict's keys, whether they are a dict's keys),
    or None for anything else."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items, keys = node.elts, False
    elif isinstance(node, ast.Dict) and None not in node.keys:
        items, keys = node.keys, True
    else:
        return None
    found = [item.value for item in items if isinstance(item, ast.Constant) and isinstance(item.value, str)]
    return (set(found), keys) if found and len(found) == len(items) else None


def bump(found, kind, path, what):
    entry = found[kind].setdefault(path, {})
    entry[what] = entry.get(what, 0) + 1


def measured():
    found = {"assigned": {}, "copies": {}, "imports": {}, "globals": {}, "unhomed": {}}
    names = {name: concept for concept, owned in OWNED.items() for name in owned["names"]}
    homes = {}
    for path in sources():
        where = relative(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        counted = set()   # the values of assignments counted as assigned, not again as copies
        for node in ast.walk(tree):
            targets = (node.targets if isinstance(node, ast.Assign) else
                       [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
            for target in targets:
                name = getattr(target, "id", None)
                if node.value is None or not literal(node.value):
                    continue
                for concept, owned in OWNED.items():
                    if where == owned["home"] and name == owned["name"]:
                        homes.setdefault(concept, set()).add(name)
                        counted.add(id(node.value))
                concept = names.get(name)
                if concept is not None and where != OWNED[concept]["home"]:
                    bump(found, "assigned", where, name)
                    counted.add(id(node.value))
            written = collection(node)
            if written is not None and id(node) not in counted:
                values, keys = written
                for concept, owned in OWNED.items():
                    if owned.get("values") == values and (owned.get("keys", True) or not keys)                             and where != owned["home"]:
                        bump(found, "copies", where, concept)
        if not path.is_relative_to(PYTHON):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (
                    node.module == "crit_kernels.exactness"
                    or node.module == "crit_kernels" and any(a.name == "exactness" for a in node.names)) or \
                    isinstance(node, ast.Import) and any(a.name.startswith("crit_kernels.exactness") for a in node.names):
                bump(found, "imports", where, "crit_kernels.exactness")
            if isinstance(node, ast.Global):
                for name in node.names:
                    if name not in PROCESS.get(where, ()):
                        bump(found, "globals", where, name)
    for concept, owned in OWNED.items():
        if owned["name"] not in homes.get(concept, ()):
            found["unhomed"].setdefault(owned["home"], {})[concept] = 1
    return found


WHY = {"assigned": "a value Core owns is assigned outside its Core module: import it from there",
       "copies": "a literal copy of values Core owns: import them from Core",
       "imports": "Core imports crit_kernels.exactness: the levels are Core's (crit_plugins.exactness)",
       "globals": "Core keeps an engine's policy in module state: it belongs to the engine (its LoadContext)",
       "unhomed": "a value Core owns has no definition in its Core module"}


class CoreOwnership(unittest.TestCase):
    def test_owned_values_live_in_core_and_policy_in_the_engine(self):
        allowed = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
        problems = []
        for kind, found in measured().items():
            may_kind = allowed.get(kind, {})
            keys = {(p, w) for p, entry in found.items() for w in entry} | \
                   {(p, w) for p, entry in may_kind.items() for w in entry}
            for path, what in sorted(keys):
                have = found.get(path, {}).get(what, 0)
                may = may_kind.get(path, {}).get(what, 0)
                if have > may:
                    problems.append(f"{path}: {what} {have} times (allowed {may}): {WHY[kind]}")
                elif have < may:
                    problems.append(f"{path}: {what} {have} times, the allowlist says {may}: lower the entry")
        self.assertEqual(problems, [])

    def test_the_homes_are_core_modules(self):
        for concept, owned in OWNED.items():
            self.assertTrue(owned["home"].startswith("inference/core/python/"), concept)
        for path in PROCESS:
            self.assertTrue((ROOT / path).is_file(), path)


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:      # after a fix: the list as the tree stands
        ALLOWLIST.write_text(json.dumps(measured(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    else:
        unittest.main()
