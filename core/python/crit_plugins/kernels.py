"""The kernels plugins run that no provider ships: crit's own kernels, built on this machine from crit's kernel library.

crit's kernels are source in crit's kernels/ folder (kernels/techniques/<family>/<id>/, a kernel.json each), templates whose knobs
(warps, tiles, splits, loads in flight) a sweep tunes for a model on a GPU (crit kernels sweep). Nothing is downloaded:
a kernel compiles for the current GPU the first time a plugin asks for it (crit_kernels.build, cached after that), with
the tuning table of the engine (CRIT_KERNEL_TUNING, a folder of <kernel id>.json files the sweep wrote, or
use_tuning()) or, without one, the kernel's default knobs. A plugin computes with a provider first (torch, cuBLASLt
through torch, FlashInfer, fla, ...), then with a crit kernel, and last with its plain torch fallback when the kernel
cannot be built here (no CUDA toolkit, an architecture it does not cover, crit's kernels package not installed).

A plugin declares each kernel it loads in its plugin.json:

    "kernels": [{"name": "grouped-experts", "fallback": "the same arithmetic in torch"}]

(name: the kernel's id in crit's library; fallback, required, says what the plugin runs without it; backend, optional,
the build it loads where that is not torch's own device backend), and asks for it once, when it is created, by name
from its own folder:

    from pathlib import Path
    from crit_plugins.kernels import load_kernel
    grouped = load_kernel("grouped-experts", folder=Path(__file__).parent)
    if grouped is None:
        ...  # the torch fallback; the engine's facts say so

A kernel the manifest does not declare is refused (PluginError): the manifest is the list of what the plugin loads.

A kernel whose structure aliases another kernel's (kernel.json "alias" with "kernel") forwards those calls to that
kernel, which crit's build loads with the same tuning folder (use_tuning(), CRIT_KERNEL_TUNING) and which is recorded
here too, under its own id with "via" naming the alias, for the plugins that loaded the aliasing kernel.

A plugin that needs one configuration a shape at every row count (a tuning picks by the row count) asks for the
kernel with its default knobs, whatever the engine's tuning: load_kernel(name, folder=..., tuned=False), recorded
beside the tuned record as name + UNTUNED.

Each request is recorded (loaded()), and the engine's facts carry the record under "kernels": per kernel the source it
came from (tuned: built with the engine's tuning table; built: built with its default knobs; fallback) and, for a
fallback, why.
"""
import os
import threading
from pathlib import Path

SOURCES = ("tuned", "built", "fallback")

_lock = threading.RLock()   # re-entered when a kernel's alias loads its target during a load
_loaded = {}     # kernel id -> the record of its last request
_modules = {}    # (kernel id, tuning file) -> module
_tuning = None   # the folder use_tuning() named


def use_tuning(folder):
    """Build kernels with the tuning tables in folder (<kernel id>.json each, crit kernels sweep's output): the
    engine's tuning. None: CRIT_KERNEL_TUNING, else the kernels' default knobs."""
    global _tuning
    _tuning = None if folder is None else Path(folder)


def tuning_file(name):
    """The tuning file for kernel `name` this process uses, or None."""
    folder = _tuning or (Path(os.environ["CRIT_KERNEL_TUNING"]) if os.environ.get("CRIT_KERNEL_TUNING") else None)
    if folder is None:
        return None
    path = folder / f"{name}.json"
    return path if path.is_file() else None


_warned = set()   # kernels whose fallback was reported on stderr


def _warn_fallback(name, plugin, reason):
    """Say once per process that a crit kernel is not here and its plugin runs the fallback: a plugin's fallback can be
    many times slower (a row a call, the provider's general kernel), and an engine that serves on it looks merely
    slow."""
    if name in _warned:
        return
    _warned.add(name)
    import sys
    print(f"crit: kernel {name} is not available here ({reason}); {plugin or 'its plugin'} runs its fallback, which "
          f"may be much slower. crit builds its kernels on first use: it needs crit installed, torch with CUDA, the "
          f"CUDA toolkit (nvcc) and ninja; `crit kernels build {name}` shows the build's errors.", file=sys.stderr)


def _record(name, source, plugin, reason=None, tuning=None):
    if source == "fallback":
        _warn_fallback(name, plugin, reason or "unavailable")
    record = {"source": source}
    if tuning:
        record["tuning"] = str(tuning)
    if reason:
        record["reason"] = reason
    if plugin:
        record["plugins"] = sorted(set((_loaded.get(name) or {}).get("plugins", [])) | {plugin})
    _loaded[name] = record
    return record


_manifests = {}   # (plugin folder, plugin.json's mtime and size) -> its checked manifest
_declared = {}    # (folder as given, kernel name) -> (plugin.json path, mtime, size, (entry, plugin id))


def declared(folder, name):
    """The entry {name, fallback, backend} the plugin in `folder` declares for kernel `name`, and the plugin's id;
    PluginError when its plugin.json does not declare it. The manifest is read once per folder (again when its
    plugin.json changes)."""
    from .manifest import MANIFEST, PluginError, load_manifest
    # A plugin asks for its kernels from its hot paths (a matmul a call), so the answer is kept by the folder as
    # given and checked against one stat of its plugin.json, without resolving the path again.
    known = _declared.get((str(folder), name))
    if known is not None:
        path, mtime, size, answer = known
        try:
            stat = os.stat(path)
        except OSError:
            stat = None
        if stat is not None and (stat.st_mtime_ns, stat.st_size) == (mtime, size):
            return answer
    given = str(folder)
    folder = Path(folder).resolve()
    stat = (folder / MANIFEST).stat()
    key = (str(folder), stat.st_mtime_ns, stat.st_size)
    manifest = _manifests.get(key)
    if manifest is None:
        manifest = _manifests[key] = load_manifest(folder)
    for kernel in manifest["kernels"]:
        if kernel["name"] == name:
            _declared[(given, name)] = (str(folder / MANIFEST), stat.st_mtime_ns, stat.st_size, (kernel, manifest["id"]))
            return kernel, manifest["id"]
    raise PluginError(f"{manifest['id']} loads the kernel {name}, which its plugin.json does not declare: add "
                      f'{{"name": "{name}", "fallback": "..."}} to its "kernels"')


def _alias_loaded(info):
    """crit_kernels.build's report of a target an alias loaded: recorded under the target's id, for the plugins that
    loaded the aliasing kernel."""
    with _lock:
        plugins = (_loaded.get(info["kernel"]) or {}).get("plugins") or []
        record = {"source": info["source"], "via": f"{info['kernel']}/{info['structure']}"}
        if info.get("tuning"):
            record["tuning"] = str(info["tuning"])
        known = _loaded.get(info["target"]) or {}
        merged = sorted(set(known.get("plugins") or []) | set(plugins))
        if merged:
            record["plugins"] = merged
        if known.get("source") and "via" not in known:   # the target loaded on its own too: keep that record
            known = dict(known, plugins=merged) if merged else dict(known)
            _loaded[info["target"]] = known
        else:
            _loaded[info["target"]] = record


def _connect(build_module):
    """crit's build takes this process's tuning folder for the targets aliases load, and reports them here."""
    build_module.use_tuning_source(lambda kernel_id: tuning_file(kernel_id))
    build_module.observe_aliases(_alias_loaded)


def _report_aliases(build_module, module):
    """Record the targets a kernel's aliases already loaded in this process (a cached package after forget())."""
    import sys
    aliases = sys.modules.get(getattr(module, "__name__", "") + "._crit_aliases")
    for structure in list(getattr(aliases, "_loaded", {})):
        build_module.alias_loaded(aliases.KERNEL, structure, aliases.ALIASES[structure])


UNTUNED = ":untuned"   # loaded()'s suffix of a kernel's record built with its default knobs on request (tuned=False)


def load_kernel(name, *, folder, tuned=True):
    """The Python package of crit's kernel `name`, built for this machine (with the engine's tuning when there is one),
    or None when it cannot be built here: the caller then runs its fallback. folder: the asking plugin's folder, whose
    plugin.json declares the kernel (declared()). tuned=False: built with its default knobs whatever the engine's
    tuning (a plugin that needs one configuration a shape at every row count, where a tuning picks by the row count),
    recorded in loaded() as `name` + UNTUNED (source built, or fallback with its reason), beside the tuned record."""
    _, plugin = declared(folder, name)
    tuning = tuning_file(name) if tuned else None
    label = name if tuned else name + UNTUNED
    key = (name, str(tuning)) if tuned else (name, UNTUNED)
    with _lock:
        if key in _modules:
            record = _loaded[label]
            _record(label, record["source"], plugin, record.get("reason"), record.get("tuning"))
            if _modules[key] is not None:
                from crit_kernels import build as build_module
                _report_aliases(build_module, _modules[key])
            return _modules[key]
        try:
            from crit_kernels import build as build_module
            from crit_kernels.build import load, read_tuning
        except ImportError:
            _record(label, "fallback", plugin, "crit's kernels package (crit_kernels) is not installed")
            _modules[key] = None
            return None
        if hasattr(build_module, "use_tuning_source"):
            _connect(build_module)
        try:
            module = load(name, tuning=read_tuning(tuning) if tuning else None)
        except Exception as error:  # noqa: BLE001 - any failure to build or import is a fallback with its reason
            text = str(error).strip()
            reason = f"{type(error).__name__}: {text.splitlines()[-1] if text else ''}"[:300]
            _record(label, "fallback", plugin, reason)
            _modules[key] = None
            return None
        _record(label, "tuned" if tuning else "built", plugin, tuning=tuning)
        if hasattr(build_module, "alias_loaded"):
            _report_aliases(build_module, module)
        _modules[key] = module
        return module


def record_fallback(name, *, folder, reason):
    """Record that the plugin in `folder` runs its fallback for kernel `name` although the kernel may be built: a shape
    or format it does not cover (reason, a line of text). The engine's facts then say so, as for a kernel that is not
    available."""
    _, plugin = declared(folder, name)
    with _lock:
        _record(name, "fallback", plugin, reason)


def loaded():
    """Every crit kernel requested in this process: kernel id -> {source, tuning, plugins, reason}; a kernel a
    plugin asked for with its default knobs (load_kernel(..., tuned=False)) as id + UNTUNED."""
    with _lock:
        return {name: dict(record) for name, record in _loaded.items()}


def fallbacks():
    """The kernels that fell back, kernel id -> why."""
    return {name: record.get("reason", "") for name, record in loaded().items() if record["source"] == "fallback"}


def forget():
    """Drop every record and cached module (tests, and a process that reloads its plugins)."""
    with _lock:
        _loaded.clear()
        _modules.clear()
        _manifests.clear()
        _declared.clear()


__all__ = ["SOURCES", "UNTUNED", "declared", "fallbacks", "forget", "load_kernel", "loaded", "record_fallback", "tuning_file",
           "use_tuning"]
