"""The kernels plugins run that no provider ships: crit-labs' kernels in the Hugging Face kernel registry.

crit hosts no kernel. A plugin computes with a provider first (torch, cuBLASLt through torch, FlashInfer, fla, ...),
then with a crit-labs kernel pulled from the registry by name at a pinned revision through the `kernels` package, and
last with its plain torch fallback when neither is there. Every crit-labs kernel is a kernel-builder project in the
crit-kernels repository (crit-labs/<name> on the Hub). Before a kernel is published, or to test a local build of it,
the `kernels` package's LOCAL_KERNELS variable maps its repository id to a local build
("crit-labs/<name>=/path/to/project:...").

A plugin declares each registry kernel it loads in its plugin.json, as a dependency with its pin:

    "kernels": [{"name": "crit-labs/grouped-experts", "version": 1, "revision": null,
                 "fallback": "the same arithmetic in torch"}]

(revision null until the kernel is published, then the registry commit it is pinned to; fallback, required, says what
the plugin runs without it; backend, optional, the build it loads where that is not torch's own device backend), and
asks for it once, when it is created, by name from its own folder, which supplies the pin:

    from pathlib import Path
    from crit_plugins.kernels import registry_kernel
    grouped = registry_kernel("crit-labs/grouped-experts", folder=Path(__file__).parent)
    if grouped is None:
        ...  # the torch fallback; the engine's facts say so

A kernel the manifest does not declare is refused (PluginError): the manifest is the list of what the plugin loads.

Each request is recorded (loaded()), and the engine's facts carry the record under "kernels": per kernel the source it
came from (local, hub or fallback), its revision and, for a fallback, why. A fresh clone with no published kernel
serves correctly on its fallbacks, and says so.
"""
import threading

SOURCES = ("local", "hub", "fallback")

_lock = threading.Lock()
_loaded = {}     # repository id -> the record of its last request
_modules = {}    # (repository id, version, revision) -> module


def _local_overrides():
    import os
    found = {}
    for entry in (os.environ.get("LOCAL_KERNELS") or "").split(":"):
        if "=" in entry:
            repo, path = entry.split("=", 1)
            found[repo] = path
    return found


def _record(repo, source, plugin, version, revision, reason=None):
    record = {"source": source, "version": version, "revision": revision}
    if reason:
        record["reason"] = reason
    if plugin:
        record["plugins"] = sorted(set((_loaded.get(repo) or {}).get("plugins", [])) | {plugin})
    _loaded[repo] = record
    return record


_manifests = {}   # (plugin folder, plugin.json's mtime and size) -> its checked manifest
_declared = {}    # (folder as given, kernel name) -> (plugin.json path, mtime, size, (pin, plugin id))


def declared(folder, name):
    """The pin {name, version, revision} the plugin in `folder` declares for registry kernel `name` (ORG/NAME), and
    the plugin's id; PluginError when its plugin.json does not declare it. The manifest is read once per folder (again
    when its plugin.json changes)."""
    import os
    from pathlib import Path
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
    raise PluginError(f"{manifest['id']} loads the registry kernel {name}, which its plugin.json does not declare: add "
                      f'{{"name": "{name}", "version": 1, "revision": null, "fallback": "..."}} to its "kernels"')


def registry_kernel(name, *, folder):
    """The module of the registry kernel `name` (ORG/NAME), or None when it is not available here (the `kernels`
    package is missing, the kernel is not published or has no build for this machine, the registry cannot be reached
    offline): the caller then runs its fallback. folder: the asking plugin's folder, whose plugin.json declares the
    kernel with its version, pinned revision and backend (declared()). A local build named in LOCAL_KERNELS wins."""
    pin, plugin = declared(folder, name)
    repo, version, revision, backend = name, pin["version"], pin["revision"], pin.get("backend")
    key = (repo, version, revision, backend)
    with _lock:
        if key in _modules:
            _record(repo, _loaded[repo]["source"], plugin, version, revision, _loaded[repo].get("reason"))
            return _modules[key]
        local = repo in _local_overrides()
        try:
            import kernels
        except ImportError:
            _record(repo, "fallback", plugin, version, revision, "the kernels package is not installed")
            _modules[key] = None
            return None
        options = {"trust_remote_code": [repo]}
        if backend:
            options["backend"] = backend
        if revision:
            options["revision"] = revision
        elif version is not None:
            options["version"] = version
        try:
            module = kernels.get_kernel(repo, **options)
        except Exception as error:  # noqa: BLE001 - any failure to resolve or import is a fallback with its reason
            reason = f"{type(error).__name__}: {error}".splitlines()[0][:300]
            _record(repo, "fallback", plugin, version, revision, reason)
            _modules[key] = None
            return None
        _record(repo, "local" if local else "hub", plugin, version, revision)
        _modules[key] = module
        return module


def record_fallback(name, *, folder, reason):
    """Record that the plugin in `folder` runs its fallback for the registry kernel `name` (ORG/NAME) although the kernel
    may be installed: a shape or format it does not cover (reason, a line of text). The engine's facts then say so, as
    for a kernel that is not available."""
    pin, plugin = declared(folder, name)
    with _lock:
        _record(name, "fallback", plugin, pin["version"], pin["revision"], reason)


def loaded():
    """Every registry kernel requested in this process: repository id -> {source, version, revision, plugins, reason}."""
    with _lock:
        return {repo: dict(record) for repo, record in _loaded.items()}


def fallbacks():
    """The registry kernels that fell back, repository id -> why."""
    return {repo: record.get("reason", "") for repo, record in loaded().items() if record["source"] == "fallback"}


def forget():
    """Drop every record and cached module (tests, and a process that reloads its plugins)."""
    with _lock:
        _loaded.clear()
        _modules.clear()
        _manifests.clear()
        _declared.clear()


__all__ = ["SOURCES", "fallbacks", "forget", "loaded", "record_fallback", "registry_kernel"]
