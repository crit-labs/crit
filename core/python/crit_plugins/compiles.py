"""Kernel compiles and extension builds, watched: what a plugin compiles once its engine serves.

A plugin compiles every kernel it calls at load (or in the warm-up its model runs at load), never on a request: a
request that compiles waits seconds for it, and its first token with it. Core's loader starts a watch once the warm
requests ran and keeps it for the engine's life, so every engine says what it compiled while it served (its
telemetry's core.compiles_after_load), the agent's own plugins too, and crit run serve --check fails one that
compiled while its checks ran. A plugin's tests use the same watch (crit_plugins.testing.compiles).

    watch = compiles.Watch()     # watching from here on
    ...
    watch.report()               # {"count": n, "first": [up to FIRST descriptions]}, cheap to read often
    watch.stop()                 # the descriptions, with the cache entries made outside triton.jit since it started

What it sees, using only what is imported (nothing is loaded for it; while Core records, a hook goes in the moment
its module is imported):
- a Triton JIT compile: Triton's jit_cache_hook, which runs whenever a kernel's specialization is not yet in memory
  (compiled now or read from Triton's cache);
- a torch extension build or load (torch.utils.cpp_extension's _jit_compile);
- at stop(), new entries in Triton's cache beyond the JIT compiles (kernels compiled outside triton.jit, such as
  torch.compile's) and new torch extension build folders.

One hook of each kind is installed for the process, chained to the hook that was there, and every watch that is
running gets each event, so watches nest and end in any order.

Why a kernel compiled again. Triton specializes a runtime argument it is not told to leave alone: an integer equal to
1 becomes a constant, one divisible by 16 says so, a pointer says it is 16-byte aligned; each mix is a kernel of its
own. An argument that follows the request (a token count, a key count, a position, the stride of a buffer made per
call) then compiles a kernel the first time a request brings a new mix, whatever ran at load. Core's loader records
every compile from the start of the load (record()), and a compile with the constexprs of one the process made
before names the arguments whose specialization changed ("T is now 1, was divisible by 16"): such an argument
belongs in its kernel's do_not_specialize (do_not_specialize_on_alignment for a pointer), or its launcher keeps it
in one class (a buffer padded to 16). A compile with constexprs the process never compiled is a class of its own,
which a load's warm-up reaches or a request does first.
"""
import importlib.abc
import os
import sys
import threading

FIRST = 8   # the compiles a report names one by one

_lock = threading.Lock()
_watches = []      # the running watches
_installed = {}    # what the hooks replaced: "triton" -> (runtime, previous hook), "extension" -> (module, function)
_recording = False  # record() ran: the hooks stay for the process and every compile is kept in _compiled
_compiled = {}     # "module.kernel" -> [(constexprs, {argument: its specialization})], each compile the process made


def _entries(folder):
    try:
        return set(os.listdir(folder)) if folder and os.path.isdir(folder) else set()
    except OSError:
        return set()


def _triton_runtime():
    triton = sys.modules.get("triton")
    runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
    return runtime if runtime is not None and hasattr(runtime, "jit_cache_hook") else None


def _triton_cache():
    triton = sys.modules.get("triton")
    return getattr(getattr(getattr(triton, "knobs", None), "cache", None), "dir", None) if triton is not None else None


def _extension():
    extension = sys.modules.get("torch.utils.cpp_extension")
    return extension if extension is not None and hasattr(extension, "_jit_compile") else None


def _extension_root(extension):
    if extension is None:
        return None
    try:
        return os.environ.get("TORCH_EXTENSIONS_DIR") or extension.get_default_build_root()
    except Exception:   # noqa: BLE001 - no build root: nothing to compare
        return None


def _tell(what, jit=False):
    with _lock:
        watches = list(_watches)
    for watch in watches:
        watch._add(what, jit)


def _specialization(kwargs):
    """(constexprs, {runtime argument: its specialization}) of a Triton compile, from its hook's arguments: a
    declared constexpr's value, and per runtime argument "=value" (an integer Triton made a constant: 1, or None),
    "divisible by 16" (an integer) or "16-byte aligned" (a pointer), else "any"."""
    fn, options = kwargs.get("fn"), kwargs.get("compile") or {}
    params = list(getattr(getattr(fn, "jit_function", None), "params", None) or [])
    signature = options.get("signature") or {}
    constants = {key[0] if isinstance(key, tuple) and len(key) == 1 else key: value
                 for key, value in (options.get("constants") or {}).items()}
    attrs = (options.get("configs") or [{}])[0] or {}
    divisible = {key[0] for key, values in attrs.items() if isinstance(key, tuple) and len(key) == 1
                 and any(isinstance(v, (list, tuple)) and v and v[0] == "tt.divisibility" for v in values)}
    declared, runtime = {}, {}
    for index, (name, kind) in enumerate(signature.items()):
        param = params[index] if index < len(params) else None
        if param is not None and getattr(param, "is_constexpr", False):
            declared[name] = constants.get(index)
        elif kind == "constexpr":
            runtime[name] = f"={constants.get(index)}"
        elif index in divisible:
            runtime[name] = "16-byte aligned" if str(kind).startswith("*") else "divisible by 16"
        else:
            runtime[name] = "any"
    return declared, runtime


def _why(name, declared, runtime):
    """Why the process compiles kernel `name` again (a compile it made before with the same constexprs, whose
    runtime arguments specialized otherwise), or why it is a class of its own; records this compile."""
    earlier = _compiled.setdefault(name, [])
    same = [found for constexprs, found in earlier if constexprs == declared]
    earlier.append((declared, runtime))
    if same:
        last = same[-1]
        changed = [f"{arg} is now {_said(now)}, was {_said(last.get(arg))}" for arg, now in runtime.items()
                   if last.get(arg) != now]
        if changed:
            return ("; compiled again for the same constexprs: " + ", ".join(changed) + " (an argument that follows "
                    "the request belongs in the kernel's do_not_specialize, a pointer's in "
                    "do_not_specialize_on_alignment, or its launcher keeps it in one class)")
        return "; compiled again for the same constexprs and specializations (other options: warps, stages)"
    if earlier[:-1]:
        return "; constexprs this process never compiled it with (a class no warm-up reached)"
    return ""


def _said(specialization):
    return "unknown" if specialization is None else specialization[1:] if specialization.startswith("=") \
        else specialization


def _describe(kwargs):
    """A Triton JIT compile, as its hook's arguments name it: the kernel, its signature, its constexprs, the runtime
    arguments Triton specialized, and why it compiled again (the module's description)."""
    fn, options = kwargs.get("fn"), kwargs.get("compile") or {}
    declared, runtime = _specialization(kwargs)
    signature = ", ".join(f"{name}: {kind}" for name, kind in (options.get("signature") or {}).items()
                          if kind != "constexpr")
    constants = ", ".join(f"{name}={value}" for name, value in declared.items())
    special = ", ".join(f"{name}{value if value.startswith('=') else ' ' + value}" for name, value in runtime.items()
                        if value != "any")
    name = f"{getattr(fn, 'module', '?')}.{getattr(fn, 'name', '?')}"
    with _lock:
        why = _why(name, declared, runtime)
    return (f"Triton compile of {name}({signature}) constexprs {constants or 'none'}"
            f"{'; specialized ' + special if special else ''}{why}")


def _install():
    """The process's hooks, once each, where their module is imported; called under _lock. A hook something else
    replaced since (a test standing in for a build) is installed again over the replacement."""
    for kind, (owner, _, ours) in list(_installed.items()):
        if getattr(owner, "jit_cache_hook" if kind == "triton" else "_jit_compile", None) is not ours:
            del _installed[kind]
    runtime = _triton_runtime()
    if runtime is not None and "triton" not in _installed:
        previous = runtime.jit_cache_hook

        def hook(*args, **kwargs):
            try:
                described = _describe(kwargs)
                if _watches:
                    _tell(described, jit=True)
            except Exception:   # noqa: BLE001 - a description never breaks a compile
                pass
            return previous(*args, **kwargs) if previous else False
        runtime.jit_cache_hook = hook
        _installed["triton"] = (runtime, previous, hook)
    extension = _extension()
    if extension is not None and "extension" not in _installed:
        jit = extension._jit_compile

        def building(name, *args, **kwargs):
            _tell(f"torch extension build or load of {name}")
            return jit(name, *args, **kwargs)
        extension._jit_compile = building
        _installed["extension"] = (extension, jit, building)


class _OnImport(importlib.abc.MetaPathFinder):
    """While Core records, the hooks go in the moment their module is imported (Triton by the first kernel package or
    plugin that uses it, torch's extension builder by the first extension): Core imports neither, so it waits for
    whoever brings them, and the load's first compile is recorded like any other."""

    def find_spec(self, name, path=None, target=None):
        if name not in WATCHED:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        run = loader.exec_module

        def exec_module(module):
            run(module)
            with _lock:
                if _recording or _watches:
                    _install()
        loader.exec_module = exec_module
        return spec


WATCHED = ("triton", "torch.utils.cpp_extension")   # the modules whose import installs the hooks while recording
_on_import = _OnImport()


def record(on=True):
    """Keep every compile from here on (Core's loader, as an engine's load starts): the hooks stay for the process,
    so a later compile of a kernel with constexprs it compiled before names what changed (the module's
    description); a hook whose module is not imported yet goes in when it is. record(False) stops it (a test), the
    hooks going once no watch runs."""
    global _recording
    with _lock:
        _recording = bool(on)
        if _recording:
            if _on_import not in sys.meta_path:
                sys.meta_path.insert(0, _on_import)
            _install()
        else:
            if _on_import in sys.meta_path:
                sys.meta_path.remove(_on_import)
            if not _watches:
                _uninstall()


def _uninstall():
    """The hooks put back once no watch runs (where nothing replaced them since, and nothing records); called under
    _lock."""
    if _recording:
        return
    for kind, (owner, previous, ours) in list(_installed.items()):
        attribute = "jit_cache_hook" if kind == "triton" else "_jit_compile"
        if getattr(owner, attribute, None) is ours:
            setattr(owner, attribute, previous)
        del _installed[kind]


class Watch:
    """The compiles and builds from its creation to stop() (the module's description)."""

    def __init__(self):
        self.found, self.count, self.jit = [], 0, 0
        self._cache, self._built = _triton_cache(), _extension_root(_extension())
        self._cached, self._extensions = _entries(self._cache), _entries(self._built)
        self._stopped = False
        with _lock:
            _install()
            _watches.append(self)

    def _add(self, what, jit=False):
        self.count += 1
        self.jit += bool(jit)
        self.found.append(what)

    def report(self):
        """{"count": the compiles and builds seen so far, "first": the first FIRST of them}: no file is read. A hook
        whose module was imported since the watch began (Triton, torch's extensions) is installed now."""
        with _lock:
            if not self._stopped:
                _install()
        return {"count": self.count, "first": list(self.found[:FIRST])}

    def stop(self):
        """End the watch: the descriptions of everything it saw, with the Triton cache entries made outside
        triton.jit and the new extension build folders since it started. A second stop() returns the same."""
        if not self._stopped:
            self._stopped = True
            with _lock:
                if self in _watches:
                    _watches.remove(self)
                if not _watches:
                    _uninstall()
            new = sorted(_entries(self._cache) - self._cached)
            if len(new) > self.jit:
                self._add(f"{len(new) - self.jit} new Triton cache entries beyond the JIT compiles (compiled outside "
                          f"triton.jit): {', '.join(new[:FIRST])}")
            for name in sorted(_entries(self._built) - self._extensions):
                self._add(f"a new torch extension build directory {name}")
        return list(self.found)
