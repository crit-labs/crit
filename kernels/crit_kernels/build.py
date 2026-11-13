"""Build one of crit's kernels for this machine's GPU, with a tuning table or with its default knobs.

crit's kernels are source (kernels/techniques/<family>/<id>/, described by kernel.json); nothing is downloaded and no binary is
shipped. A CUDA kernel compiles on first use with torch's extension builder for the current device only (one
architecture, so a minute or two, not a matrix of builds), into a cache keyed by everything that changes the binary:
the kernel's files, the tuning, the device's architecture and the torch and CUDA versions. A Triton kernel needs no
compile step; its package is copied and its tuning written beside it.

Knobs: each structure in kernel.json declares compile-time knobs (CUDA) or launch parameters (Triton) with the values a
sweep may try and a default. A tuning table (crit kernels sweep) names, per shape key and row range, the knob values to
use. The build writes it where the kernel's code reads it:

- CUDA: crit_tuning.h (included by the kernel's sources) defines, per structure S (its id with '-' as '_'),
  CRIT_CONFIGS_S(X) listing X(index, knob values in kernel.json order) for every configuration the table uses (index 0
  is always the default knobs), and crit_lookup_S(key, rows[, length]), the configuration index for a shape key, row
  count and (where the structure declares "lengths") length (0 where the table has no entry; a table entry without a
  "length" range holds for every length, and a call without a length, -1, matches every entry's). A sweep build defines CRIT_SWEEP_BUILD, so a kernel may instantiate only what the
  sweep calls.
- Triton: _crit_tuning.py in the package, with knobs(structure, key, rows, length=None) -> {knob: value}. A Triton kernel whose
  kernel.json has a "cuda" section holds CUDA sources too: they get crit_tuning.h as a CUDA kernel's do, and
  _crit_cuda.py in the package compiles them for the build's GPU, one shared library a variant (the compile-time
  choices its code makes a call by), into the build folder on the variant's first use (_CUDA_MODULE).

A structure may alias another kernel's structure (kernel.json "alias": {"kernel": id, "of": structure}, the same knobs
and defaults): the kernel's Python calls forward to that kernel, built with the alias's tuning table for that structure
(the target's own tuning where the alias's table is empty: the file use_tuning_source() names, which crit_plugins sets
to the engine's tuning folder, else CRIT_KERNEL_TUNING/<id>.json), and a sweep of the alias,
its builds with configs= and select() follow it there. The build writes _crit_aliases.py into the package (module(S):
the target's package for alias S, built on first use, reported to observe_aliases()' observers) and builds the
targets with the kernel (_ALIAS_MODULE).

load(kernel_id) returns the kernel's Python package (its torch-ext/<package>/ with the ops registered), imported under
a name unique to the build, so builds with different knobs load side by side in one process (the sweep does).
"""
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import threading
from pathlib import Path

SCHEMA = "crit-kernel-v1"
TUNING_SCHEMA = "crit-kernel-tuning-v1"
_lock = threading.Lock()
_libraries = {}   # package -> the path its library was loaded from in this process (once: a second load of the
                 # same library from another path would register its ops again)


# The techniques and the workbench's starters in crit's kernels folder; "<id>/kernel.json" in any other root (a
# workbench root such as work/kernels, or one family's folder).
KERNEL_GLOBS = ("techniques/*/*/kernel.json", "starters/*/kernel.json", "*/kernel.json")


def library():
    """The kernels folder (core/, techniques/<family>/<id>/, starters/): a checkout's kernels/, or an installed crit's
    crit/library/kernels."""
    here = Path(__file__).resolve().parent.parent
    if any(here.glob(KERNEL_GLOBS[0])):
        return here
    spec = importlib.util.find_spec("crit.library.kernels")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("crit's kernel library is not installed (crit.library.kernels)")
    return Path(list(spec.submodule_search_locations)[0])


def manifests(root=None):
    """Every kernel's kernel.json, id -> (folder, manifest)."""
    root = Path(root) if root else library()
    found = {}
    for path in sorted(p for pattern in KERNEL_GLOBS for p in root.glob(pattern)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"{path}: schema is {manifest.get('schema')!r}, expected {SCHEMA}")
        if manifest["id"] != path.parent.name:
            raise ValueError(f"{path}: id {manifest['id']!r} differs from its folder's name")
        found[manifest["id"]] = (path.parent, manifest)
    return found


def search_roots(root=None):
    """The folders find() looks in, in order: root (when given), crit's kernels folder, then CRIT_KERNEL_PATH's folders
    (separated as PATH is: workbench roots holding <id>/kernel.json, which a workbench folder's kernel.py and tests
    add themselves)."""
    out = [Path(root)] if root else []
    try:
        out.append(library())
    except FileNotFoundError:
        if not out:
            raise
    out += [Path(p) for p in os.environ.get("CRIT_KERNEL_PATH", "").split(os.pathsep) if p]
    return out


def find(kernel_id, root=None):
    """The kernel's (folder, manifest): the first of search_roots(root) that holds it."""
    roots = search_roots(root)
    for folder in roots:
        found = manifests(folder) if folder.is_dir() else {}
        if kernel_id in found:
            return found[kernel_id]
    raise KeyError(f"no kernel {kernel_id!r} in {', '.join(str(r) for r in roots)} "
                   "(kernels/techniques/<family>/<id>/kernel.json, or <root>/<id>/kernel.json with --root)")


def c_name(structure):
    return structure.replace("-", "_")


def knob_names(manifest, structure):
    return list(manifest["structures"][structure].get("knobs", {}))


def defaults(manifest, structure):
    return {name: spec["default"] for name, spec in manifest["structures"][structure].get("knobs", {}).items()}


def default_tuning(manifest):
    """A tuning with every structure on its default knobs, plus each structure's default_rows (knob overrides by row
    range, shape-independent: the kernel's hand-set rules for an untuned build)."""
    structures = {}
    for sid, spec in manifest.get("structures", {}).items():
        base = defaults(manifest, sid)
        configs, table = [base], []
        for rule in spec.get("default_rows", []):
            config = dict(base, **rule["knobs"])
            if config not in configs:
                configs.append(config)
            table.append({"key": None, "rows": rule["rows"], "config": configs.index(config)})
        structures[sid] = {"configs": configs, "table": table}
    return {"schema": TUNING_SCHEMA, "kernel": manifest["id"], "structures": structures}


def check_tuning(manifest, tuning):
    """The tuning, checked against the manifest: every knob a declared one with a declared value, config 0 the
    structure's defaults (filled in when missing), every table entry naming a configuration."""
    if tuning.get("schema") != TUNING_SCHEMA or tuning.get("kernel") != manifest["id"]:
        raise ValueError(f"tuning is {tuning.get('schema')} for {tuning.get('kernel')!r}, expected {TUNING_SCHEMA} for "
                         f"{manifest['id']!r}")
    out = {}
    for sid, spec in manifest.get("structures", {}).items():
        given = (tuning.get("structures") or {}).get(sid) or {"configs": [], "table": []}
        knobs = spec.get("knobs", {})
        configs = [dict(defaults(manifest, sid), **c) for c in given.get("configs") or []]
        if not configs or configs[0] != defaults(manifest, sid):
            configs.insert(0, defaults(manifest, sid))
            shift = 1
        else:
            shift = 0
        for config in configs:
            for name, value in config.items():
                if name not in knobs:
                    raise ValueError(f"{manifest['id']}/{sid}: tuning sets unknown knob {name!r}")
                if value not in knobs[name]["values"]:
                    raise ValueError(f"{manifest['id']}/{sid}: knob {name}={value!r} is not one of "
                                     f"{knobs[name]['values']}")
        table = []
        for entry in given.get("table") or []:
            index = entry["config"] + shift
            if not 0 <= index < len(configs):
                raise ValueError(f"{manifest['id']}/{sid}: a table entry names configuration {entry['config']}")
            lo, hi = entry["rows"]
            found = {"key": entry.get("key"), "rows": [int(lo), int(hi)], "config": index}
            if entry.get("length") is not None:
                found["length"] = [int(entry["length"][0]), int(entry["length"][1])]
            table.append(found)
        out[sid] = {"configs": configs, "table": table}
    for sid in (tuning.get("structures") or {}):
        if sid not in manifest.get("structures", {}):
            raise ValueError(f"{manifest['id']}: tuning names unknown structure {sid!r}")
    return {"schema": TUNING_SCHEMA, "kernel": manifest["id"], "structures": out}


def _literal(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    raise ValueError(f"a CUDA knob is an integer or a boolean, not {value!r}")


def tuning_header(manifest, tuning, sweep=False, swept=()):
    """crit_tuning.h for a CUDA kernel: the configurations and lookup per structure (module docstring). A sweep build
    defines CRIT_SWEEP_BUILD and CRIT_SWEEP_STRUCTURE_<S> for each structure it tunes, so a kernel may instantiate only
    those structures' calls."""
    lines = ["// Generated by crit_kernels.build from the kernel's tuning: do not edit.", "#pragma once", ""]
    if sweep:
        lines += ["#define CRIT_SWEEP_BUILD 1"] + [f"#define CRIT_SWEEP_STRUCTURE_{c_name(s)} 1" for s in swept] + [""]
    for sid, spec in manifest.get("structures", {}).items():
        name, knobs = c_name(sid), list(spec.get("knobs", {}))
        key = spec.get("key", [])
        t = tuning["structures"][sid]
        entries = " ".join(f"X({i}{''.join(', ' + _literal(c[k]) for k in knobs)})" for i, c in enumerate(t["configs"]))
        lines.append(f"// {sid}: X(index{''.join(', ' + k for k in knobs)})")
        lines.append(f"#define CRIT_CONFIGS_{name}(X) {entries}")
        lines.append(f"#define CRIT_CONFIG_COUNT_{name} {len(t['configs'])}")
        rows = []
        for e in t["table"]:
            k = e["key"]
            wild = 1 if k is None else 0
            values = [0] * len(key) if k is None else [int(k[f]) for f in key]
            span = e.get("length") or [0, 1 << 62]
            rows.append("{" + ", ".join([str(wild)] + [str(v) for v in values] + [str(e["rows"][0]), str(e["rows"][1]),
                                                                                   str(e["config"]), str(int(span[0])),
                                                                                   str(int(span[1]))]) + "}")
        width = len(key) + 6
        lines.append(f"static const long long crit_table_{name}[][{width}] = {{{', '.join(rows) or '{0}'}}};")
        lines.append(f"static const int crit_table_size_{name} = {len(t['table'])};")
        args = "".join(f"long long {f}, " for f in key)
        lines.append("// A sweep build selects one of its configurations at run time (crit_select_S, called through")
        lines.append("// ctypes by crit_kernels.build.select): it wins over the table while it is 0 or more.")
        lines.append("// Hidden: one variable per library, so two kernels with a structure of the same name, loaded in one")
        lines.append("// process, never share it (a default-visibility symbol would bind both to the first library's copy).")
        lines.append(f'__attribute__((visibility("hidden"))) inline int crit_selected_{name} = -1;')
        lines.append(f'extern "C" __attribute__((visibility("default"), used)) inline void crit_select_{name}(int i) {{')
        lines.append(f"  crit_selected_{name} = i;")
        lines.append("}")
        lines.append(f"// The configuration for a shape ({', '.join(key) or 'no key'}), row count and length: the first")
        lines.append("// entry whose key matches (or holds for every key), whose row range holds the rows and whose length")
        lines.append("// range the length (a length below 0: any), else 0 (defaults).")
        lines.append(f"static inline int crit_lookup_{name}({args}int rows, long long length = -1) {{")
        lines.append(f"  if (crit_selected_{name} >= 0) return crit_selected_{name};")
        lines.append(f"  const long long key[{max(len(key), 1)}] = {{{', '.join(key) or '0'}}};")
        lines.append("  (void)key;")
        lines.append(f"  for (int i = 0; i < crit_table_size_{name}; ++i) {{")
        lines.append(f"    const long long* e = crit_table_{name}[i];")
        lines.append("    bool match = e[0] != 0;")
        lines.append("    if (!match) {")
        lines.append("      match = true;")
        lines.append(f"      for (int j = 0; j < {len(key)}; ++j) match = match && e[1 + j] == key[j];")
        lines.append("    }")
        lines.append(f"    if (match && rows >= e[{len(key) + 1}] && rows <= e[{len(key) + 2}] && "
                     f"(length < 0 || (length >= e[{len(key) + 4}] && length <= e[{len(key) + 5}]))) "
                     f"return (int)e[{len(key) + 3}];")
        lines.append("  }")
        lines.append("  return 0;")
        lines.append("}")
        lines.append("")
    return "\n".join(lines) + "\n"


def tuning_module(manifest, tuning):
    """_crit_tuning.py for a Triton kernel: knobs(structure, key, rows, length=None)."""
    keys = {sid: spec.get("key", []) for sid, spec in manifest.get("structures", {}).items()}
    return ('"""Generated by crit_kernels.build from the kernel\'s tuning: do not edit."""\n'
            "import json\n\n"
            f"TUNING = json.loads({json.dumps(json.dumps(tuning['structures'], sort_keys=True))})\n"
            f"KEYS = json.loads({json.dumps(json.dumps(keys, sort_keys=True))})\n"
            "SELECTED = {}   # structure -> configuration index a sweep selected (crit_kernels.build.select)\n\n\n"
            "def knobs(structure, key, rows, length=None):\n"
            "    \"\"\"The knob values for a shape key (a mapping of the structure's key fields, or None), row count and\n"
            "    length (None: any): the first table entry whose key matches (or holds for every key), whose row range\n"
            "    holds rows and whose length range (none: every length) the length, else the structure's defaults.\"\"\"\n"
            "    t = TUNING[structure]\n"
            "    if structure in SELECTED:\n"
            "        return dict(t['configs'][SELECTED[structure]])\n"
            "    fields = KEYS[structure]\n"
            "    for e in t['table']:\n"
            "        k = e['key']\n"
            "        if (k is None or (key is not None and all(int(k[f]) == int(key[f]) for f in fields))) \\\n"
            "                and e['rows'][0] <= rows <= e['rows'][1] \\\n"
            "                and (length is None or 'length' not in e or e['length'][0] <= length <= e['length'][1]):\n"
            "            return dict(t['configs'][e['config']])\n"
            "    return dict(t['configs'][0])\n")


IGNORED = {"tests", "benchmarks", "bench.py"}   # a kernel's files that never reach its build
# A workbench folder's own files at its top (crit kernels new, crit eval workbench): its measurement adapter and
# harness, the copy of the build crit-eval profiles, and their results.
WORKBENCH = {"kernel.py", "harness.py", "build", "check.json", "bench.json", "profile.json", "profile-ncu.txt"}


def _ignored(relative):
    """Whether a kernel's file (its path in the kernel's folder) is outside its build."""
    parts = relative.parts
    return bool(IGNORED & set(parts)) or (bool(parts) and parts[0] in WORKBENCH)


def _files_digest(folder):
    """The digest of what a kernel's build is made of: every file but its docs, tests, benchmarks and sweep bench, so
    editing those neither rebuilds the kernel nor stales a tuning."""
    h = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                       and not p.name.endswith(".pyc") and p.suffix != ".md"
                       and not _ignored(p.relative_to(folder))):
        h.update(path.relative_to(folder).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def core_folder(root=None):
    """The kernel core: building blocks every kernel may use, no algorithm in them (kernels/core/: CUDA headers in
    include/crit_kcore, Triton helpers in python/crit_kcore)."""
    return (Path(root) if root else library()) / "core"


def core_root(folder):
    """The kernels folder whose core a kernel's folder builds with: the nearest one above it that holds a core
    (crit's kernels/ for kernels/techniques/<family>/<id> and kernels/starters/<id>), else crit's own (a workbench
    folder outside crit)."""
    for parent in Path(folder).resolve().parents:
        if (parent / "core" / "include" / "crit_kcore").is_dir():
            return parent
    return library()


_CORE_HEADER = re.compile(r'#\s*include\s*[<"]crit_kcore/(\w+)\.cuh[>"]')
_CORE_MODULE = re.compile(r"(?:from\s+crit_kcore\s+import\s+(\([^)]*\)|[^\n#]+)|crit_kcore\.(\w+))")


def _core_used(text):
    """The core areas a source names: (CUDA headers, Triton modules)."""
    headers = set(_CORE_HEADER.findall(text))
    modules = set()
    for names, dotted in _CORE_MODULE.findall(text):
        if dotted:
            modules.add(dotted)
        for name in names.replace("(", " ").replace(")", " ").split(","):   # "a as b" names a
            if name.split() and not name.split()[0].startswith("#"):
                modules.add(name.split()[0])
    return headers, modules


def _core_digest(root=None, folder=None):
    """The digest of the core blocks a kernel uses: the headers and Triton modules its sources name, with the ones
    those name in turn, so a core edit rebuilds (and stales the tuning of) only the kernels that use what changed.
    folder None: the whole core."""
    core = core_folder(root)
    if not core.is_dir():
        return ""
    if folder is None:
        return _files_digest(core)
    files = _core_files(core, folder)
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(path.relative_to(core).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest() if files else ""


def _core_files(core, folder):
    """The core files a kernel's sources name, with the ones those name in turn."""
    include, python = core / "include" / "crit_kcore", core / "python" / "crit_kcore"
    headers, modules = set(), set()
    for path in folder.rglob("*"):
        if (path.is_file() and path.suffix in (".cu", ".cuh", ".h", ".hpp", ".cpp", ".py")
                and not _ignored(path.relative_to(folder))):
            found = _core_used(path.read_text(errors="replace"))
            headers |= found[0]
            modules |= found[1]
    files, todo = set(), [include / f"{h}.cuh" for h in headers] + [python / f"{m}.py" for m in modules]
    if modules:
        todo.append(python / "__init__.py")
    while todo:
        path = todo.pop()
        if path in files or not path.is_file():
            continue
        files.add(path)
        h2, m2 = _core_used(path.read_text(errors="replace"))
        todo += [include / f"{h}.cuh" for h in h2] + [python / f"{m}.py" for m in m2]
    return files


def source_digest(kernel_id, root=None):
    """The digest of a kernel's files: a tuning names it, so a tuning made for other source is noticed."""
    folder, _ = find(kernel_id, root)
    return hashlib.sha256((_files_digest(folder) + _core_digest(core_root(folder), folder)).encode()).hexdigest()


def device_arch():
    """The current CUDA device's architecture as nvcc names it (sm_120a style for sm_90 and later, whose
    architecture-specific instructions crit's kernels may use), and its capability string ('12.0')."""
    import torch
    major, minor = torch.cuda.get_device_capability()
    num = f"{major}{minor}"
    return (num + "a" if major >= 9 else num), f"{major}.{minor}"


def device_capability():
    """The current CUDA device's capability string ('12.0'), or None where there is no CUDA device to ask."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return device_arch()[1]
    except Exception:  # noqa: BLE001 - no torch, no driver: nothing to compare a tuning with
        return None


def foreign_tuning(tuning, capability=None):
    """Why a tuning was not made on this GPU (a phrase), or None when it was or that cannot be told. The sweep writes the
    GPU's name and compute capability into every tuning; knob choices measured on one GPU generation are no evidence
    for another (docs/CONTRIBUTION_CRITERIA.md), so a tuning swept on another capability is not applied where the
    tuning is found implicitly (an engine's tuning folder, CRIT_KERNEL_TUNING): the kernel builds with its default knobs
    and says so. A tuning without a capability (written by hand, or before the sweep recorded it) is taken as it is.
    capability: this GPU's ('12.0'), else the current CUDA device's."""
    made = (tuning or {}).get("capability")
    if not made:
        return None
    here = capability or device_capability()
    if here is None or str(made) == str(here):
        return None
    return (f"its tuning was swept on {tuning.get('device') or 'another GPU'} (compute capability {made}), not on this "
            f"GPU (compute capability {here})")


_foreign_warned = set()   # kernels whose foreign tuning was reported on stderr


def warn_foreign_tuning(kernel_id, reason, kept=False):
    """Say once per process and kernel that its tuning comes from another GPU: dropped for the default knobs, or kept
    (kept=True) where a command was given the tuning explicitly."""
    if kernel_id in _foreign_warned:
        return
    _foreign_warned.add(kernel_id)
    then = ("it is used as given, but its choices are not evidence for this GPU" if kept else
            "it builds with its default knobs instead")
    print(f"crit: kernel {kernel_id}: {reason}; {then}. `crit kernels sweep {kernel_id} --model CHECKPOINT --out "
          f"TUNING` tunes it on this GPU.", file=sys.stderr)


def cache_root():
    return Path(os.environ.get("CRIT_KERNEL_CACHE") or Path.home() / ".cache" / "crit" / "kernels")


def _generator_digest():
    """The digest of the code that writes a build's generated files (crit_tuning.h, _crit_tuning.py), so a change to
    what they hold rebuilds instead of reusing a build made with the old ones."""
    import inspect
    return hashlib.sha256((inspect.getsource(tuning_header) + inspect.getsource(tuning_module)).encode()).hexdigest()


def _build_key(folder, manifest, tuning, sweep, archs=None, swept=()):
    import torch
    parts = [_files_digest(folder), _core_digest(core_root(folder), folder), _generator_digest(), json.dumps(tuning, sort_keys=True), str(sweep), ",".join(sorted(swept)),
             torch.__version__,
             str(torch.version.cuda), platform.python_version(), platform.system()]
    if manifest["language"] == "cuda" or _mixed(manifest):
        parts.append(",".join(archs) if archs else device_arch()[0])
    if _mixed(manifest):
        parts.append(hashlib.sha256(_CUDA_MODULE.encode()).hexdigest())
    if cross_aliases(manifest):   # only these kernels' keys hold it: every other kernel's key is as before
        import inspect
        parts.append(hashlib.sha256((_ALIAS_MODULE + inspect.getsource(alias_build_args) +
                                     inspect.getsource(_target_tuning_file)).encode()).hexdigest())
        parts.append(json.dumps(cross_aliases(manifest), sort_keys=True))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def cross_aliases(manifest):
    """The structures that alias another kernel's structure: {structure: {"kernel": id, "of": structure}}."""
    return {sid: {"kernel": spec["alias"]["kernel"], "of": spec["alias"]["of"]}
            for sid, spec in manifest.get("structures", {}).items()
            if isinstance(spec.get("alias"), dict) and spec["alias"].get("kernel")}


def _alias_specs(manifest, tuning, sweep, swept, root=None):
    """Each alias structure's target and the tuning it carries across: its configurations and table in this build, and
    whether the build is a sweep's of it (then the target's build is the sweep's of the target structure, the same
    configurations in the same order, so select() indices agree)."""
    out = {}
    for sid, target in cross_aliases(manifest).items():
        _, other = find(target["kernel"], root)
        if target["of"] not in other.get("structures", {}):
            raise ValueError(f"{manifest['id']}/{sid} aliases {target['kernel']}/{target['of']}, which does not exist")
        if other["structures"][target["of"]].get("knobs", {}) != manifest["structures"][sid].get("knobs", {}):
            raise ValueError(f"{manifest['id']}/{sid} and {target['kernel']}/{target['of']} it aliases differ in "
                             "their knobs")
        t = tuning["structures"][sid]
        out[sid] = dict(target, configs=t["configs"], table=t["table"], sweep=bool(sweep and sid in swept))
    return out


_tuning_source = None    # kernel id -> the tuning file to build it with, or None (use_tuning_source)
_alias_observers = []    # callables told of every target an alias loads (observe_aliases)


def use_tuning_source(source):
    """Where an alias's target finds its own tuning: source(kernel_id) gives the tuning file (or None: its defaults).
    crit_plugins sets it to the engine's (use_tuning(), CRIT_KERNEL_TUNING); None goes back to CRIT_KERNEL_TUNING."""
    global _tuning_source
    _tuning_source = source


def observe_aliases(observer):
    """observer(record) for every target an alias loads: {"kernel": the alias's kernel, "structure", "target",
    "of", "source": "tuned" or "built", "tuning": what tuned it (a file, or the alias's table) or None}."""
    if observer not in _alias_observers:
        _alias_observers.append(observer)


def _target_tuning_file(kernel_id):
    if _tuning_source is not None:
        found = _tuning_source(kernel_id)
        return Path(found) if found else None
    folder = os.environ.get("CRIT_KERNEL_TUNING")
    path = Path(folder) / f"{kernel_id}.json" if folder else None
    return path if path and path.is_file() else None


def alias_loaded(kernel_id, structure, spec):
    """Tell the observers that alias `structure` of kernel_id loaded its target (called by _crit_aliases.module)."""
    if spec["sweep"]:
        source, tuning = "built", None
    elif spec["table"]:
        source, tuning = "tuned", f"{kernel_id}'s tuning of {structure}"
    else:
        path = _target_tuning_file(spec["kernel"])
        source, tuning = ("tuned", str(path)) if path else ("built", None)
    record = {"kernel": kernel_id, "structure": structure, "target": spec["kernel"], "of": spec["of"],
              "source": source, "tuning": tuning}
    for observer in list(_alias_observers):
        observer(dict(record))


def alias_build_args(spec, root=None):
    """build()'s arguments for an alias's target (an _alias_specs entry): a sweep's build of the target structure with
    the alias's configurations, else the target's tuning with the alias's configurations and table for that structure
    where it names a table or more configurations than the defaults (a build with configs=: select() picks among them),
    the target's own tuning file in CRIT_KERNEL_TUNING, or its defaults, otherwise."""
    if spec["sweep"]:
        return {"configs": {spec["of"]: spec["configs"][1:]}, "sweep": True, "root": root}
    _, manifest = find(spec["kernel"], root)
    path = _target_tuning_file(spec["kernel"])
    tuning = read_tuning(path) if path else None
    foreign = foreign_tuning(tuning) if tuning else None
    if foreign:   # found implicitly, in the engine's tuning folder: another GPU's choices are not applied here
        warn_foreign_tuning(spec["kernel"], foreign)
        tuning = None
    tuning = tuning or default_tuning(manifest)
    if spec["table"] or len(spec["configs"]) > 1:   # a table, or configurations a select() picks from (configs=)
        tuning = dict(tuning, structures=dict(tuning.get("structures") or {},
                                              **{spec["of"]: {"configs": spec["configs"], "table": spec["table"]}}))
    return {"tuning": tuning, "root": root}


# _crit_aliases.py, written into the package of a kernel with structures that alias another kernel's (cross_aliases):
# module(S) is the target's package for alias S, built (alias_build_args) and imported on first use; select(S, i)
# selects configuration i of the target structure (a sweep's build).
_ALIAS_MODULE = """\"\"\"Generated by crit_kernels.build for a kernel whose structures alias other kernels' structures: do
not edit. module(structure): the target kernel's package for an alias structure (ALIASES), built with the tuning the
alias carries across; select(structure, index): a sweep's configuration of the target structure.\"\"\"
import json
import threading

ALIASES = json.loads(__ALIASES__)
KERNEL = __KERNEL__
ROOT = __ROOT__
SELECTED = {}
_loaded = {}
_lock = threading.Lock()


def module(structure):
    found = _loaded.get(structure)
    if found is None:
        from crit_kernels import build as kbuild
        with _lock:
            found = _loaded.get(structure)
            if found is None:
                spec = ALIASES[structure]
                found = kbuild.load(spec["kernel"], **kbuild.alias_build_args(spec, ROOT))
                kbuild.alias_loaded(KERNEL, structure, spec)
                if structure in SELECTED:
                    kbuild.select(found, spec["of"], SELECTED[structure])
                _loaded[structure] = found
    return found


def select(structure, index):
    if index < 0:
        SELECTED.pop(structure, None)
    else:
        SELECTED[structure] = int(index)
    found = _loaded.get(structure)
    if found is not None:
        from crit_kernels import build as kbuild
        kbuild.select(found, ALIASES[structure]["of"], index)
"""


def _mixed(manifest):
    """A kernel of another language (Triton) that holds CUDA sources too (kernel.json "cuda")."""
    return manifest["language"] != "cuda" and bool(manifest.get("cuda"))


def _templates():
    return Path(__file__).resolve().parent / "templates"


# _crit_cuda.py, written into a mixed kernel's package (kernel.json "cuda"): its CUDA sources compiled for the build's
# GPU, one shared library a variant, on the variant's first use (or ahead of it: build(variants=...)), into the build
# folder, with the build's crit_tuning.h and its copy of the core headers it uses.
_CUDA_MODULE = '''"""Generated by crit_kernels.build for a kernel with CUDA sources beside its Triton modules: do not edit.

library(**variant): the kernel's CUDA sources compiled as one shared library for a variant (kernel.json "cuda"
"variant": the compile-time defines CRIT_VARIANT_<name> the package picks a call by; "device": fields filled from the
GPU's properties), for this build's GPU, with this build's crit_tuning.h, cached in the build folder and loaded with
ctypes (called on the caller's stream, so CUDA graphs capture its launches). select(structure, index): a sweep's
configuration for every library of this build (crit_select_<S>), loaded or not yet loaded."""
import ctypes
import json
import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path

CONFIG = json.loads(__CONFIG__)
BUILD = Path(__file__).resolve().parent.parent
SELECTED = {}   # structure -> configuration index (crit_kernels.build.select)
_lock = threading.Lock()
_loaded = {}


def _nvcc():
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    found = Path(home) / "bin" / ("nvcc.exe" if platform.system() == "Windows" else "nvcc") if home else None
    if found is None or not found.exists():
        which = shutil.which("nvcc")
        found = Path(which) if which else None
    if found is None:
        raise RuntimeError(f"{CONFIG['kernel']}'s CUDA sources compile with nvcc: none on PATH or in CUDA_HOME")
    return found


def device_fields(device=None):
    """The variant's fields read from the GPU (kernel.json "cuda" "device": field -> torch device property)."""
    import torch
    props = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
    return {name: int(getattr(props, prop)) for name, prop in CONFIG["device"].items()}


def name(variant):
    return "-".join(f"{k}{int(v)}" for k, v in sorted(variant.items()))


def check(variant, device_too=True):
    want = set(CONFIG["variant"]) | (set(CONFIG["device"]) if device_too else set())
    if set(variant) != want:
        raise ValueError(f"{CONFIG['kernel']}: a CUDA variant names {sorted(want)}, not {sorted(variant)}")
    for k, values in CONFIG["variant"].items():
        if values and int(variant[k]) not in values:
            raise ValueError(f"{CONFIG['kernel']}: variant field {k}={variant[k]} is not one of {values}")


def compile_variant(variant, archs=None):
    """The variant's library path, compiled now when it is not in the build folder yet."""
    archs = archs or CONFIG["archs"]
    out = BUILD / "cuda" / (name(variant) + ("" if archs == CONFIG["archs"] else "-" + "-".join(archs)) +
                            (".dll" if platform.system() == "Windows" else ".so"))
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}.{threading.get_ident()}")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]
    # -fno-gnu-unique: a process loads several variants' libraries, which hold the same inline and template symbols;
    # GNU-unique binding would make every library use the first one's.
    pic = [] if platform.system() == "Windows" else ["-Xcompiler", "-fPIC", "-Xcompiler", "-fno-gnu-unique"]
    command = [str(_nvcc()), "-O3", "-std=c++17", "-shared", *pic, *gencode, "-I", str(BUILD / "include"),
               "-I", str(BUILD / "cuda" / "src"), *CONFIG["flags"],
               *[f"-DCRIT_VARIANT_{k}={int(v)}" for k, v in sorted(variant.items())],
               *[str(BUILD / "cuda" / "src" / s) for s in CONFIG["sources"]], "-o", str(tmp)]
    run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{CONFIG['kernel']} CUDA variant {name(variant)} did not compile:\\n{run.stderr[-4000:]}")
    os.replace(tmp, out)
    return out


def library(device=None, **variant):
    """The variant's library (variant: the kernel.json "cuda" "variant" fields; the device ones are read here)."""
    full = dict(variant, **device_fields(device))
    key = name(full)
    found = _loaded.get(key)
    if found is not None:
        return found
    with _lock:
        found = _loaded.get(key)
        if found is None:
            check(full)
            found = ctypes.CDLL(str(compile_variant(full)))
            for structure, index in SELECTED.items():
                _apply(found, structure, index)
            _loaded[key] = found
    return found


def _apply(lib, structure, index):
    function = getattr(lib, "crit_select_" + structure.replace("-", "_"), None)
    if function is not None:
        function(int(index))


def select(structure, index):
    with _lock:
        if index < 0:
            SELECTED.pop(structure, None)
        else:
            SELECTED[structure] = int(index)
        for lib in _loaded.values():
            _apply(lib, structure, index)
'''


def _stage_cuda(folder, manifest, tuning, sweep, swept, stage, package, archs):
    """A mixed kernel's CUDA part in the build folder: crit_tuning.h and the core headers its sources use in include/,
    its sources in cuda/src/, and the package's _crit_cuda.py (_CUDA_MODULE) that compiles them a variant at a time."""
    spec = manifest["cuda"]
    include = stage / "include"
    include.mkdir(exist_ok=True)
    (include / "crit_tuning.h").write_text(tuning_header(manifest, tuning, sweep, swept))
    core = core_folder(core_root(folder))
    for path in _core_files(core, folder) if core.is_dir() else ():
        if path.suffix == ".cuh":
            target = include / path.relative_to(core / "include")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target)
    src = stage / "cuda" / "src"
    for s in spec["sources"]:
        target = src / s
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(folder / s, target)
    config = {"kernel": manifest["id"], "sources": list(spec["sources"]), "flags": list(spec.get("flags", [])),
              "variant": spec.get("variant", {}), "device": spec.get("device", {}),
              "archs": list(archs) if archs else [device_arch()[0]]}
    (stage / package / "_crit_cuda.py").write_text(
        _CUDA_MODULE.replace("__CONFIG__", json.dumps(json.dumps(config, sort_keys=True))))


def _cuda_module(out, package):
    """A mixed build's _crit_cuda module, loaded on its own (it imports nothing from its package)."""
    path = Path(out) / package / "_crit_cuda.py"
    name = f"_crit_cuda_{Path(out).name.replace('-', '_')}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _cuda_variants(out, package, manifest, variants, archs):
    """Compile a mixed build's variants ahead of their use: `variants` (kernel.json "cuda" "variant" fields; the
    device's filled in), or with archs (a build check) the manifest's "check" variants for those architectures."""
    module = _cuda_module(out, package)
    if archs:
        for variant in manifest["cuda"].get("check", []):
            module.check(variant, device_too=False)
            module.compile_variant(dict(variant), archs=list(archs))
        return
    fields = module.device_fields()
    for variant in variants or []:
        full = dict(variant, **fields)
        module.check(full)
        module.compile_variant(full)


def build(kernel_id, *, tuning=None, knobs=None, configs=None, sweep=False, root=None, verbose=False, archs=None,
          variants=None):
    """Compile (CUDA) or stage (Triton) kernel_id and return (package folder, package name, build key) without
    importing it. tuning: a tuning dict for this kernel (None: its default tuning); knobs: {structure: {knob: value}},
    a single configuration per structure, which replaces the tuning; configs: {structure: [knob dicts]}, several
    configurations a structure holds at once (the sweep's builds: the defaults are index 0, the given ones follow in
    order, select() picks one at run time), which replaces the tuning; sweep: define
    CRIT_SWEEP_BUILD; archs: compile for these architectures (nvcc names: 80, 86, 89, 90a, 100a, 120a) instead of the
    current device's, to check that the kernel builds for every one it claims (such a build is not for loading; a
    mixed kernel compiles its kernel.json "cuda" "check" variants); variants: a mixed kernel's CUDA variants to compile
    now rather than on first use (a sweep's builds, before it holds the GPU)."""
    folder, manifest = find(kernel_id, root)
    if knobs is not None:
        tuning = {"schema": TUNING_SCHEMA, "kernel": kernel_id,
                  "structures": {sid: {"configs": [dict(defaults(manifest, sid), **(knobs.get(sid) or {}))],
                                       "table": [{"key": None, "rows": [0, 1 << 30], "config": 0}]}
                                 for sid in manifest.get("structures", {})}}
        tuning = check_tuning(manifest, tuning)
        # check_tuning puts the defaults first when the sweep's configuration differs from them: point every row at
        # the sweep's configuration, the last one.
        for t in tuning["structures"].values():
            for e in t["table"]:
                e["config"] = len(t["configs"]) - 1
    elif configs is not None:
        structures = {}
        for sid in manifest.get("structures", {}):
            base = defaults(manifest, sid)
            listed = [base] + [dict(base, **c) for c in configs.get(sid, []) if dict(base, **c) != base]
            structures[sid] = {"configs": listed, "table": []}
        tuning = check_tuning(manifest, {"schema": TUNING_SCHEMA, "kernel": kernel_id, "structures": structures})
    else:
        foreign = foreign_tuning(tuning) if (tuning and not archs) else None
        if foreign:   # given explicitly (crit kernels build --tuning, gates --tuning): kept, with a warning
            warn_foreign_tuning(kernel_id, foreign, kept=True)
        tuning = check_tuning(manifest, tuning) if tuning else default_tuning(manifest)
    swept = sorted(configs) if (sweep and configs) else sorted(knobs) if (sweep and knobs) else []
    key = _build_key(folder, manifest, tuning, sweep, archs, swept)
    out = cache_root() / f"{kernel_id}-{key}"
    package = f"{manifest['package']}_{key}"
    done = out / "complete"
    if cross_aliases(manifest) and not archs:   # the targets the package forwards to, built with it (before a sweep
        for spec in _alias_specs(manifest, tuning, sweep, swept, root).values():   # holds the GPU)
            build(spec["kernel"], verbose=verbose, **alias_build_args(spec, root))
    if done.exists():
        if _mixed(manifest) and (variants or archs):
            _cuda_variants(out, package, manifest, variants, archs)
        return out, package, key
    stage = out.with_name(out.name + f".tmp{os.getpid()}")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    pkg = stage / package
    shutil.copytree(folder / "torch-ext" / manifest["package"], pkg,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_ops.py"))
    (stage / "tuning.json").write_text(json.dumps(tuning, indent=1, sort_keys=True))
    if manifest["language"] == "cuda":
        _compile(folder, manifest, tuning, sweep, stage, package, verbose, archs, swept)
        (pkg / "_ops.py").write_text(
            "import torch\n\n"
            f"ops = getattr(torch.ops, {package!r})\n\n\n"
            "def add_op_namespace_prefix(op_name: str) -> str:\n"
            f"    return f\"{package}::{{op_name}}\"\n")
    else:
        (pkg / "_ops.py").write_text("ops = None\n\n\ndef add_op_namespace_prefix(op_name: str) -> str:\n"
                                     "    return op_name\n")
        if _mixed(manifest):
            _stage_cuda(folder, manifest, tuning, sweep, swept, stage, package, archs)
    (pkg / "_crit_tuning.py").write_text(tuning_module(manifest, tuning))
    if cross_aliases(manifest):
        specs = _alias_specs(manifest, tuning, sweep, swept, root)
        (pkg / "_crit_aliases.py").write_text(
            _ALIAS_MODULE.replace("__ALIASES__", json.dumps(json.dumps(specs, sort_keys=True)))
            .replace("__KERNEL__", repr(kernel_id)).replace("__ROOT__", repr(str(root)) if root else "None"))
    (stage / "complete").write_text(json.dumps({"kernel": kernel_id, "package": package}))
    if out.exists():
        shutil.rmtree(stage, ignore_errors=True)   # another process built it meanwhile
    else:
        try:
            stage.rename(out)
        except OSError:
            shutil.rmtree(stage, ignore_errors=True)
    if _mixed(manifest) and (variants or archs):
        _cuda_variants(out, package, manifest, variants, archs)
    return out, package, key


def _compile(folder, manifest, tuning, sweep, stage, package, verbose, archs=None, swept=()):
    from torch.utils import cpp_extension
    arch, capability = device_arch() if not archs else (None, None)
    allowed = [c.rstrip("a") for c in manifest.get("capabilities", [])]
    if not archs and allowed and capability not in allowed:
        raise RuntimeError(f"{manifest['id']} builds for compute capabilities {', '.join(manifest['capabilities'])}, "
                           f"not this device's {capability}")
    include = stage / "include"
    include.mkdir()
    (include / "crit_tuning.h").write_text(tuning_header(manifest, tuning, sweep, swept))
    shutil.copy(_templates() / "registration.h", include / "registration.h")
    sources = [str(folder / s) for s in manifest["sources"]]
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in (archs or [arch])]
    flags = ["-O3", *gencode, "-DCUDA_KERNEL"] + manifest.get("cuda_flags", [])
    cxx = ["-O3", "-DCUDA_KERNEL"] + manifest.get("cxx_flags", [])
    if platform.system() == "Windows":
        cxx = ["/O2", "/DCUDA_KERNEL"] + manifest.get("cxx_flags", [])
    ldflags = [f"-l{lib}" for lib in manifest.get("libraries", [])]
    build_dir = stage / "build"
    build_dir.mkdir()
    core_include = str(core_folder(core_root(folder)) / "include")
    cpp_extension.load(name=package, sources=sources, extra_include_paths=[str(include), core_include, str(folder),
                                                                           str(folder / "torch-ext")],
                       extra_cflags=cxx, extra_cuda_cflags=flags, extra_ldflags=ldflags, build_directory=str(build_dir),
                       is_python_module=False, with_cuda=True, verbose=verbose)
    built = list(build_dir.glob(f"{package}*.so")) + list(build_dir.glob(f"{package}*.pyd"))
    if not built:
        raise RuntimeError(f"{manifest['id']}: the build left no library in {build_dir}")
    _libraries[package] = str(built[0])   # torch's builder loaded it from there: import_build must not load it again
    shutil.copy(built[0], stage / package / built[0].name)
    shutil.rmtree(build_dir, ignore_errors=True)


def import_build(out, package):
    """Import a built package (the folder build() returned): its library first, so its ops are registered."""
    with _lock:
        if package in sys.modules:
            return sys.modules[package]
        pkg = Path(out) / package
        import torch
        if package not in _libraries:
            for lib in sorted(pkg.glob(f"{package}*.so")) + sorted(pkg.glob(f"{package}*.pyd")):
                torch.ops.load_library(str(lib))
                _libraries[package] = str(lib)
        for candidate in (core_folder() / "python",):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))   # Triton kernels import crit_kcore
        spec = importlib.util.spec_from_file_location(package, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[package]
            raise
        return module


def load(kernel_id, *, tuning=None, knobs=None, configs=None, sweep=False, root=None, verbose=False):
    """The kernel's Python package, built for this machine (build()) and imported."""
    out, package, _ = build(kernel_id, tuning=tuning, knobs=knobs, configs=configs, sweep=sweep, root=root,
                            verbose=verbose)
    return import_build(out, package)


def select(module, structure, index):
    """Make a build made with configs= run configuration `index` of `structure` for every call (-1: back to its
    table). CUDA: crit_select_<S> in its library; Triton: its _crit_tuning.SELECTED."""
    import ctypes
    tuning_module_ = sys.modules.get(module.__name__ + "._crit_tuning")
    if tuning_module_ is not None:
        if index < 0:
            tuning_module_.SELECTED.pop(structure, None)
        else:
            tuning_module_.SELECTED[structure] = index
    aliases = sys.modules.get(module.__name__ + "._crit_aliases")   # an alias: the target structure's build
    if aliases is not None and structure in aliases.ALIASES:
        aliases.select(structure, index)
    cuda_module = sys.modules.get(module.__name__ + "._crit_cuda")   # a mixed kernel's CUDA libraries
    if cuda_module is not None:
        cuda_module.select(structure, index)
    # The library as this process loaded it (the same path: another path would load and register it again).
    path = _libraries.get(module.__name__)
    if path is not None:
        function = getattr(ctypes.CDLL(path), "crit_select_" + c_name(structure), None)
        if function is not None:
            function(int(index))


def read_tuning(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != TUNING_SCHEMA:
        raise ValueError(f"{path}: not a {TUNING_SCHEMA} file")
    return data


__all__ = ["SCHEMA", "TUNING_SCHEMA", "alias_build_args", "alias_loaded", "build", "cross_aliases", "observe_aliases",
           "select", "use_tuning_source", "cache_root", "check_tuning", "core_root", "default_tuning", "defaults",
           "device_capability", "find", "foreign_tuning",
           "import_build", "library", "load", "manifests", "read_tuning", "search_roots", "source_digest", "tuning_header",
           "tuning_module", "warn_foreign_tuning"]
