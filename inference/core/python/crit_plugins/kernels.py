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
fallback, why. A tuning table swept on a GPU of another compute capability (the sweep records it) is not applied: the
kernel is built with its default knobs, recorded as built with the reason, and stderr says so once.

The engine also says how exact its rows must be (use_exactness(), from the configuration's "exactness": a level of
crit_plugins.exactness for the decode rows and one for the prompt rows). A plugin that can compute a phase's rows in
more than one way reads the level asked (asks("prompt")) where its own option leaves the choice open, takes the fastest
way that keeps it, and says what its rows keep in its exactness(); Core refuses an engine whose plugin keeps less than
the configuration asks. Where the configuration asks nothing, asks() is None and the plugin's options decide alone.

Exactness is the order of a row's sums. The precision of what is summed follows the quality mode the engine states
(use_quality_mode(), from the configuration's "quality_mode": speed or preserve, the mode of the run that built it): a
plugin whose precision option is left to choose (a projections plugin's head) reads quality_mode() and takes the
lower precision the speed mode starts from, and the stored precision in preserve or where no mode is stated.

The prompt activations are asked as the exactness is (use_activations(), from the configuration's
"prompt_activations": the activations the engine's build plan declared, which crit run writes into the candidate). A
projections plugin whose activation option for a stored format is left to choose reads activation(format) first, and
the quality mode only where nothing is asked; Core refuses an engine whose facts' prompt_activations are not what the
configuration asks, so a plan and the engine it built cannot state two activations.

The tuning, the exactness, the quality mode and the activations are the engine's policy (crit_plugins.policy): Core's
loader gives each engine its own and activates it around every call into the engine's plugins, so the functions here
answer for the engine whose plugin calls them, and two engines in one process never see each other's. The use_*
functions set the active engine's, or outside any engine the process's own (tests, a quick try). The record of the
kernels requested (loaded()) is the engine's too; the built modules are the process's, shared by every engine that
asks for a kernel with the same tuning file.
"""
import os
import sys
import threading
from pathlib import Path

from .exactness import PHASES, QUALITY_MODES, check_mode, keeps, setting
from .policy import current

SOURCES = ("tuned", "built", "fallback")

_lock = threading.RLock()   # re-entered when a kernel's alias loads its target during a load
_modules = {}    # (kernel id, tuning file) -> module, built once a process
_built = {}      # the same keys -> (source, reason, tuning) of that build, which every engine asking for it records


def use_tuning(folder):
    """Build kernels with the tuning tables in folder (<kernel id>.json each, crit kernels sweep's output): the
    engine's tuning (the active engine's policy, else the process's). None: CRIT_KERNEL_TUNING, else the kernels'
    default knobs."""
    current().tuning = None if folder is None else Path(folder)


def tuning_file(name):
    """The tuning file for kernel `name` the active engine's kernels build with, or None."""
    folder = current().tuning_folder()
    if folder is None:
        return None
    path = folder / f"{name}.json"
    return path if path.is_file() else None


def use_exactness(value, quality_mode=None):
    """The exactness the engine asks of its rows (crit_plugins.exactness: declared, exact, row-exact, repeatable):
    one level for both phases or {"decode": level, "prompt": level}; a phase left out takes the quality mode's level
    when one is given (exact in preserve, row-exact in speed). None with no quality mode: the engine asks nothing.
    Returns the levels asked, or None; ValueError for a level or phase that does not exist."""
    policy = current()
    policy.exactness = None if value is None and quality_mode is None else setting(value, quality_mode)
    return exactness()


def exactness():
    """The levels the engine asks, {"decode": level, "prompt": level}, or None when it asks none."""
    asked = current().exactness
    return None if asked is None else dict(asked)


def use_quality_mode(mode):
    """The quality mode the engine states (QUALITY_MODES: speed or preserve, the mode of the run that built it), or
    None: it states none. Returns the mode; ValueError for another name."""
    if mode is not None:
        check_mode(mode)
    current().quality_mode = mode
    return mode


def quality_mode():
    """The quality mode the engine states (speed or preserve), or None when it states none: a precision option left
    to choose then keeps the stored precision, as it does in preserve."""
    return current().quality_mode


def asks(phase):
    """The level the engine asks for a phase ("decode": decode and verify rows; "prompt": prompt rows), or None when
    it asks none: the plugin's own options then decide."""
    if phase not in PHASES:
        raise ValueError(f"exactness is asked for the decode or the prompt phase, not {phase!r}")
    asked = current().exactness
    return None if asked is None else asked[phase]


def use_activations(value):
    """The prompt activations the engine is asked to serve (configuration "prompt_activations": what its build plan
    declared, which crit run writes into the candidate): one of crit_core.serving.PROMPT_ACTIVATIONS (bf16, fp16_acc,
    fp8, stored) for every quantized format, or one per stored format ({"nvfp4": "stored", "fp8": "fp8"}; a format's
    {activation: share} mix asks nothing of it). None: the engine is asked nothing and the plugins' options decide.
    Core's loader refuses an engine whose facts serve other activations than asked. Returns the value."""
    current().activations = None if value is None else (value if isinstance(value, str) else dict(value))
    return activations_asked()


def activations_asked():
    """The prompt activations the engine is asked to serve (use_activations), or None."""
    asked = current().activations
    return asked if asked is None or isinstance(asked, str) else dict(asked)


def activation(format):
    """The prompt activation asked for a stored format's matrices ("stored": the format's own, W8A8 or W4A4; "fp8",
    "fp16_acc", "bf16"), or None where nothing is asked of it: a projections plugin whose activation option for the
    format is left to choose (auto) takes this one, so an engine serves what its plan declared."""
    asked = current().activations
    if asked is None:
        return None
    found = asked if isinstance(asked, str) else asked.get(format)
    return found if isinstance(found, str) else None


def allows(phase, level):
    """Whether rows of this level may run in the phase: the engine asks nothing there, or asks that level or a lower
    one. A plugin picks the fastest of its ways for which this holds."""
    asked = asks(phase)
    return asked is None or keeps(phase, level, asked)   # the decode rows' exact is row-exact (crit_plugins.exactness)


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
    """Record a request of kernel `name` in the active engine's record (loaded())."""
    if source == "fallback":
        _warn_fallback(name, plugin, reason or "unavailable")
    record = {"source": source}
    if tuning:
        record["tuning"] = str(tuning)
    if reason:
        record["reason"] = reason
    requests = current().kernels
    if plugin:
        record["plugins"] = sorted(set((requests.get(name) or {}).get("plugins", [])) | {plugin})
    requests[name] = record
    return record


def _outcome(key, label, source, plugin, reason=None, tuning=None):
    """A build's outcome: recorded for the engine asking, and kept with the module for the next engine that asks."""
    _built[key] = (source, reason, tuning)
    return _record(label, source, plugin, reason, tuning)


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
        requests = current().kernels
        plugins = (requests.get(info["kernel"]) or {}).get("plugins") or []
        record = {"source": info["source"], "via": f"{info['kernel']}/{info['structure']}"}
        if info.get("tuning"):
            record["tuning"] = str(info["tuning"])
        known = requests.get(info["target"]) or {}
        merged = sorted(set(known.get("plugins") or []) | set(plugins))
        if merged:
            record["plugins"] = merged
        if known.get("source") and "via" not in known:   # the target loaded on its own too: keep that record
            known = dict(known, plugins=merged) if merged else dict(known)
            requests[info["target"]] = known
        else:
            requests[info["target"]] = record


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


def plugin_folder(module, default):
    """The folder of the plugin under whose package the module named `module` was imported: the nearest package of the
    name that carries a plugin manifest (crit_plugins.assemble.load_module's packages). Code a plugin imports from
    another technique's code folder (a source that is code only, copied to an engine's _sources) belongs to the
    importing plugin, whose plugin.json then declares the kernels that code loads. `default` for a module imported on
    its own (a technique's code run from its folder, its tests)."""
    parts = module.split(".")
    for end in range(len(parts) - 1, 0, -1):
        manifest = getattr(sys.modules.get(".".join(parts[:end])), "__crit_manifest__", None)
        if manifest is not None:
            return manifest["folder"]
    return default


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
            source, reason, built_tuning = _built[key]
            _record(label, source, plugin, reason, built_tuning)
            if _modules[key] is not None:
                from crit_kernels import build as build_module
                _report_aliases(build_module, _modules[key])
            return _modules[key]
        try:
            from crit_kernels import build as build_module
            from crit_kernels.build import load, read_tuning
        except ImportError:
            _outcome(key, label, "fallback", plugin, "crit's kernels package (crit_kernels) is not installed")
            _modules[key] = None
            return None
        if hasattr(build_module, "use_tuning_source"):
            _connect(build_module)
        foreign = None
        try:
            data = read_tuning(tuning) if tuning else None
            check = getattr(build_module, "foreign_tuning", None)
            foreign = check(data) if (data and check) else None
            if foreign:   # swept on another GPU: its choices are no evidence here, so the default knobs
                build_module.warn_foreign_tuning(name, foreign)
                data = None
            module = load(name, tuning=data)
        except Exception as error:  # noqa: BLE001 - any failure to build or import is a fallback with its reason
            text = str(error).strip()
            reason = f"{type(error).__name__}: {text.splitlines()[-1] if text else ''}"[:300]
            _outcome(key, label, "fallback", plugin, reason)
            _modules[key] = None
            return None
        if foreign:
            _outcome(key, label, "built", plugin, f"default knobs: {foreign}")
        else:
            _outcome(key, label, "tuned" if tuning else "built", plugin, tuning=tuning)
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
    """Every crit kernel the active engine's plugins requested (outside any engine: the process's requests): kernel
    id -> {source, tuning, plugins, reason}; a kernel a plugin asked for with its default knobs (load_kernel(...,
    tuned=False)) as id + UNTUNED."""
    with _lock:
        return {name: dict(record) for name, record in current().kernels.items()}


def fallbacks():
    """The kernels that fell back, kernel id -> why."""
    return {name: record.get("reason", "") for name, record in loaded().items() if record["source"] == "fallback"}


def forget():
    """Drop the active engine's (else the process's) record and every cached module (tests, and a process that
    reloads its plugins)."""
    with _lock:
        current().kernels.clear()
        _modules.clear()
        _built.clear()
        _manifests.clear()
        _declared.clear()


__all__ = ["QUALITY_MODES", "SOURCES", "UNTUNED", "activation", "activations_asked", "allows", "asks", "declared",
           "exactness", "fallbacks", "forget", "load_kernel", "loaded", "plugin_folder", "quality_mode",
           "record_fallback", "tuning_file", "use_activations", "use_exactness", "use_quality_mode", "use_tuning"]
