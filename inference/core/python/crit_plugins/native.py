"""What is built on this machine the first time it is needed: C++ for the host and device code for its architecture.

A plugin's native code, and Core's own (crit_plugins.host), is source. It is compiled here, for this CPU and this
device, cached by content, and recorded, so an engine's facts say what was built, what was not, and why:

    lib = library("crit_host_rows", [(source, flags), ...], includes)   a shared library opened with ctypes
    module = extension("crit_host_handoff", [sources], includes)        a torch extension for the device's architecture
    built()                                                             {name: {kind, source, reason, for}}

library(): the system's C++ compiler (CXX, c++, g++ or clang++), no torch headers, so a build takes seconds. Every
library is cached under a name hashed from the compiler, the flags and the content of every source and of every header
its folders hold (the sources' own folders and the include folders, by their names within them), not where they lie:
a plugin copied unchanged into another engine (every candidate's package holds its own copy) reuses the build, and a
changed source or header builds again. Each copy is still opened as a library of its own (a copy of the build per
source folder), so two engines in one process never share a library's state.

extension(): torch's extension builder (the CUDA toolkit's compiler for .cu sources) for the current device's
architecture, under a name hashed as a library's is, loaded once a process. torch's builder keys a build by its paths
too, so the sources and headers are built from a copy of them under the cache named by that hash (_stage): an
unchanged plugin in another engine's package compiles nothing.

Where the compiler is missing or fails, both raise NativeUnavailable with the reason (recorded too): the caller runs
its fallback, or refuses to load when it has none.
"""
import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

X86 = platform.machine().lower() in ("x86_64", "amd64")
# A library's flags: contraction off, so a product and a sum stay two roundings as their source says.
FLAGS = ("-std=c++20", "-O3", "-fPIC", "-pthread", "-ffp-contract=off")
HEADERS = (".h", ".hpp", ".inc", ".cuh")

_lock = threading.RLock()
_libraries = {}    # (name, digest) -> ctypes library
_extensions = {}   # (name, digest, arch) -> module
_built = {}        # name -> the record of its last request
_failed = {}       # a build's key -> why it failed: a build that failed is not tried again in this process


class NativeUnavailable(RuntimeError):
    """No compiler here, or the build failed: the caller runs its fallback."""


def _record(name, kind, source, reason=None, target=None):
    record = {"kind": kind, "source": source}
    if target:
        record["for"] = target
    if reason:
        record["reason"] = reason
    _built[name] = record


def built():
    """Every library and extension asked for in this process: name -> {kind (library, extension), source (built,
    unavailable), for (the compiler, or the device architecture), reason (why it is unavailable)}."""
    with _lock:
        return {name: dict(record) for name, record in _built.items()}


def compiler():
    """The C++ compiler libraries are built with; NativeUnavailable where there is none (and on Windows, whose
    libraries this does not build)."""
    found = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not found or platform.system() == "Windows":
        raise NativeUnavailable("no C++ compiler (c++, g++ or clang++, or CXX) for native code built at load")
    return found


def cache_root():
    """Where libraries are kept: CRIT_NATIVE_CACHE, else beside torch's extensions, else crit_core.cache_home()'s
    native."""
    from crit_core import cache_home
    root = os.environ.get("CRIT_NATIVE_CACHE")
    if not root:
        torch_ext = os.environ.get("TORCH_EXTENSIONS_DIR")
        root = os.path.join(torch_ext, "crit-native") if torch_ext else str(cache_home("native"))
    return Path(root)


def _folders(sources, includes):
    """The folders a build reads: each source's own folder (its quoted includes) and each include folder, once each,
    in order, resolved."""
    found = []
    for folder in [Path(s).resolve().parent for s in sources] + [Path(i).resolve() for i in includes]:
        if folder not in found:
            found.append(folder)
    return found


def _folder_headers(folder):
    """A folder's headers, at any depth, by their path within it, in a fixed order."""
    return sorted((p for p in Path(folder).rglob("*") if p.suffix in HEADERS and p.is_file()
                   and "__pycache__" not in p.parts), key=lambda p: p.relative_to(folder).as_posix())


def _content(h, sources, includes, extra=()):
    """Hash what a build reads by content and by name within its folders, not by where the folders lie: the sources
    (each by its folder's number), every header of those folders and of the include folders, and `extra` files."""
    folders = _folders(sources, includes)
    for source in sources:
        source = Path(source).resolve()
        h.update(f"source {folders.index(source.parent)}/{source.name}\n".encode() + source.read_bytes())
    h.update(repr([folders.index(Path(i).resolve()) for i in includes]).encode())
    for j, folder in enumerate(folders):
        for header in _folder_headers(folder):
            h.update(f"header {j}/{header.relative_to(folder).as_posix()}\n".encode() + header.read_bytes())
    for path in extra:
        h.update(f"extra {Path(path).name}\n".encode() + Path(path).read_bytes())


def _stage(name, digest, sources, includes):
    """The sources and the headers of their folders and of the include folders, copied once under the cache into a
    folder named by their hash (folder j as d<j>, its files at their paths within it): (staged sources, staged
    includes). torch's builder keys a build by its paths too, so every engine whose files are the same builds from one
    copy."""
    folders = _folders(sources, includes)
    stage = cache_root() / "sources" / f"{name}-{digest}"
    resolved = [Path(s).resolve() for s in sources]
    staged_sources = [stage / f"d{folders.index(s.parent)}" / s.name for s in resolved]
    staged_includes = [stage / f"d{folders.index(Path(i).resolve())}" for i in includes]
    if not (stage / ".complete").exists():
        partial = stage.with_name(stage.name + f".tmp{os.getpid()}.{threading.get_ident()}")
        shutil.rmtree(partial, ignore_errors=True)
        for j, folder in enumerate(folders):
            for path in _folder_headers(folder) + [s for s in resolved if s.parent == folder]:
                target = partial / f"d{j}" / path.relative_to(folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        (partial / ".complete").write_text("")
        try:
            partial.rename(stage)
        except OSError:   # another process staged it meanwhile
            shutil.rmtree(partial, ignore_errors=True)
    return staged_sources, staged_includes


def _units(units):
    return [(Path(unit[0]), tuple(unit[1])) if isinstance(unit, (tuple, list)) else (Path(unit), ()) for unit in units]


def library(name, units, includes=(), defines=(), flags=FLAGS):
    """The ctypes library `name` built from units (sources, or (source, its own flags) pairs) with the include
    folders and defines, cached by content. Raises NativeUnavailable when it cannot be built or loaded here."""
    units = _units(units)
    includes = [Path(i).resolve() for i in includes]
    try:
        cxx = compiler()
        version = subprocess.run([cxx, "--version"], capture_output=True, text=True, timeout=60).stdout
        h = hashlib.sha256(cxx.encode() + version.encode() + repr((tuple(flags), tuple(defines))).encode())
        h.update(repr([own for _, own in units]).encode())
        _content(h, [path for path, _ in units], includes)
        key = h.hexdigest()[:16]
        # Where this copy of the sources lies: each copy opens a library of its own (its state is its engine's).
        copy = hashlib.sha256(repr([str(path.resolve()) for path, _ in units]).encode()).hexdigest()[:12]
        with _lock:
            if (name, key, copy) in _libraries:
                return _libraries[(name, key, copy)]
            if (name, key) in _failed:
                raise NativeUnavailable(_failed[(name, key)])
            folder = cache_root() / f"{name}-{key}"
            path = folder / f"lib{name}.so"
            if not path.exists():
                folder.mkdir(parents=True, exist_ok=True)
                import fcntl   # POSIX only, as the build is: compiler() refuses Windows before a build gets here
                with open(folder / ".lock", "w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if not path.exists():
                        try:
                            _compile(cxx, units, includes, tuple(flags), tuple(defines), folder, path)
                        except NativeUnavailable as error:
                            _failed[(name, key)] = str(error)
                            raise
            # The build as this copy's own file: a library is opened once a file (a link to it would be the same one).
            opened = folder / "copies" / f"lib{name}-{copy}.so"
            if not opened.exists():
                opened.parent.mkdir(parents=True, exist_ok=True)
                partial = opened.with_name(opened.name + f".tmp{os.getpid()}.{threading.get_ident()}")
                shutil.copyfile(path, partial)
                os.replace(partial, opened)
            try:
                loaded = ctypes.CDLL(str(opened), mode=ctypes.RTLD_LOCAL)
            except OSError as error:
                _failed[(name, key)] = f"{opened} does not load: {error}"
                raise NativeUnavailable(_failed[(name, key)]) from error
            _libraries[(name, key, copy)] = loaded
            _record(name, "library", "built", target=Path(cxx).name)
            return loaded
    except NativeUnavailable as error:
        with _lock:
            _record(name, "library", "unavailable", str(error).splitlines()[0][:300])
        raise


def _compile(cxx, units, includes, flags, defines, folder, library_path):
    common = [*flags, *defines, *(f"-I{i}" for i in includes)]
    with tempfile.TemporaryDirectory(dir=folder) as work:
        objects = []
        for i, (source, own) in enumerate(units):
            obj = Path(work) / f"{i}-{source.stem}.o"
            run = subprocess.run([cxx, *common, *own, "-c", str(source), "-o", str(obj)], capture_output=True, text=True)
            if run.returncode:
                raise NativeUnavailable(f"{source.name} did not compile:\n{run.stderr[-3000:]}")
            objects.append(str(obj))
        partial = Path(work) / library_path.name
        run = subprocess.run([cxx, "-shared", "-pthread", *objects, "-o", str(partial)], capture_output=True, text=True)
        if run.returncode:
            raise NativeUnavailable(f"{library_path.name} did not link:\n{run.stderr[-3000:]}")
        os.replace(partial, library_path)


def device_arch(device=None):
    """The architecture torch's builder compiles device code for on `device` (the current CUDA device): "12.0a"."""
    import torch
    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}.{minor}" + ("a" if major >= 9 else "")


def extension(name, sources, includes=(), headers=(), arch=None, flags=("-O3", "-std=c++20"), cuda_flags=None,
              link=(), cuda=True):
    """The torch extension `name` built from sources (CUDA C++ and C++) for the current device's architecture (or
    `arch`, "12.0a"), with the include folders; headers: more files its build depends on; flags: the C++ compiler's,
    cuda_flags: the CUDA compiler's (default: flags); link: linker flags; cuda False: host C++ alone, for a
    machine without a CUDA device too. Built once a process and cached across processes under a name hashed from
    the flags and the paths and contents of the sources, the include folders' headers and `headers`. Raises
    NativeUnavailable when it does not build here (no CUDA toolkit, no compiler, a header that does not match)."""
    sources = [Path(s) for s in sources]
    includes = [Path(i).resolve() for i in includes]
    cuda_flags = tuple(flags if cuda_flags is None else cuda_flags)
    try:
        if not cuda:
            arch = "cpu"
        else:
            try:
                arch = arch or device_arch()
            except Exception as error:   # noqa: BLE001 - no torch, no CUDA device
                raise NativeUnavailable(f"no CUDA device to build {name} for: "
                                        f"{str(error).splitlines()[0][:200]}") from error
        h = hashlib.sha256(repr((tuple(flags), cuda_flags, tuple(link), bool(cuda))).encode())
        _content(h, sources, includes, headers)
        digest = h.hexdigest()[:12]
        key = (name, digest, arch)
        with _lock:
            if key in _extensions:
                return _extensions[key]
            if key in _failed:
                raise NativeUnavailable(_failed[key])
            try:
                from torch.utils.cpp_extension import load
                sources, includes = _stage(f"{name}_{arch.replace('.', '_')}", digest, sources, includes)
                # The target as the CUDA compiler's own flag: torch's builder then adds none (no environment read).
                target = [f"-gencode=arch=compute_{arch.replace('.', '')},code=sm_{arch.replace('.', '')}"] if cuda else []
                module = load(name=f"{name}_{arch.replace('.', '_')}_{digest}", sources=[str(s) for s in sources],
                              extra_include_paths=[str(i) for i in includes], extra_cflags=list(flags),
                              extra_cuda_cflags=[*cuda_flags, *target], extra_ldflags=list(link), with_cuda=bool(cuda),
                              verbose=False)
            except Exception as error:   # noqa: BLE001 - no compiler, no toolkit, a source that does not compile
                text = str(error).strip()
                _failed[key] = (f"{name} did not build for {arch}: "
                                f"{text.splitlines()[0][:300] if text else type(error).__name__}")
                raise NativeUnavailable(_failed[key]) from error
            _extensions[key] = module
            _record(name, "extension", "built", target=arch)
            return module
    except NativeUnavailable as error:
        with _lock:
            _record(name, "extension", "unavailable", str(error).splitlines()[0][:300])
        raise


__all__ = ["FLAGS", "NativeUnavailable", "X86", "built", "cache_root", "compiler", "device_arch", "extension",
           "library"]
