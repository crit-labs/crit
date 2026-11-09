"""The host rows' C++ (host_int, host_nvfp4 and their AVX2 / AVX-512 files) built into a shared library when a plugin
loads, and opened with ctypes: no torch headers, so a build takes seconds, and the library works on any CPU the OS runs
(the ISA files are compiled with their own flags and chosen at run time through host_isa).

A technique that builds on these rows (host-expert-tier) adds its own sources to the same library through `build`.
Each library is cached under a name hashed from the compiler, the flags and every source and header's path and content,
so two copies of a plugin (two engines) never share a build, and a changed source builds again. Without a C++ compiler,
or on a compiler that fails, `build` raises NativeUnavailable and the plugin runs its torch fallback (declared.py), which
gives the same bits, slower.
"""
import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
X86 = platform.machine().lower() in ("x86_64", "amd64")
BASE = ("-std=c++20", "-O3", "-fPIC", "-pthread", "-ffp-contract=off")
AVX2 = ("-mavx2", "-mfma")
AVX512 = ("-mavx512f", "-mfma")
AVX512_VNNI = ("-mavx512f", "-mavx512bw", "-mavx512vnni", "-mfma")


class NativeUnavailable(RuntimeError):
    """No C++ compiler here, or the build failed: the caller runs its torch fallback."""


def arithmetic_units():
    """The host rows' translation units with their own flags: (path, flags)."""
    units = [(HERE / name, ()) for name in ("host_isa.cpp", "host_nvfp4.cpp", "host_int.cpp", "host_capi.cpp")]
    if X86:
        units += [(HERE / "host_nvfp4_avx2.cpp", AVX2), (HERE / "host_nvfp4_avx512.cpp", AVX512),
                  (HERE / "host_int_avx2.cpp", AVX2), (HERE / "host_int_avx512.cpp", AVX512_VNNI)]
    return units


def compiler():
    found = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not found or platform.system() == "Windows":
        raise NativeUnavailable("no C++ compiler (c++, g++ or clang++, or CXX) for the host rows")
    return found


def cache_root():
    root = os.environ.get("CRIT_NATIVE_CACHE")
    if not root:
        torch_ext = os.environ.get("TORCH_EXTENSIONS_DIR")
        root = os.path.join(torch_ext, "crit-native") if torch_ext else os.path.join(Path.home(), ".cache", "crit", "native")
    return Path(root)


def _digest(cxx, units, includes, defines):
    h = hashlib.sha256()
    version = subprocess.run([cxx, "--version"], capture_output=True, text=True, timeout=60).stdout
    h.update(cxx.encode() + version.encode() + repr((BASE, tuple(defines))).encode())
    for path, flags in units:
        h.update(str(path).encode() + repr(tuple(flags)).encode() + Path(path).read_bytes())
    for folder in includes:
        for header in sorted(Path(folder).glob("*")):
            if header.suffix in (".h", ".inc", ".hpp"):
                h.update(str(header).encode() + header.read_bytes())
    return h.hexdigest()[:16]


_LOADED = {}


def build(name, units, includes, defines=()):
    """The ctypes library `name` built from units [(source, flags)] with `includes`, cached by content. Raises
    NativeUnavailable when it cannot be built here."""
    cxx = compiler()
    includes = [Path(i).resolve() for i in includes]
    defines = tuple(defines) + (("-DCRIT_HOST_ISA_FILES=1",) if X86 else ())
    key = _digest(cxx, units, includes, defines)
    if (name, key) in _LOADED:
        return _LOADED[(name, key)]
    folder = cache_root() / f"{name}-{key}"
    library = folder / f"lib{name}.so"
    if not library.exists():
        folder.mkdir(parents=True, exist_ok=True)
        import fcntl   # POSIX only, as the build is: compiler() refuses Windows before a build gets here
        with open(folder / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not library.exists():
                _compile(cxx, units, includes, defines, folder, library)
    try:
        loaded = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
    except OSError as error:
        raise NativeUnavailable(f"{library} does not load: {error}") from error
    _LOADED[(name, key)] = loaded
    return loaded


def _compile(cxx, units, includes, defines, folder, library):
    flags = [*BASE, *defines, *(f"-I{i}" for i in includes)]
    with tempfile.TemporaryDirectory(dir=folder) as work:
        objects = []
        for i, (source, own) in enumerate(units):
            obj = Path(work) / f"{i}-{Path(source).stem}.o"
            run = subprocess.run([cxx, *flags, *own, "-c", str(source), "-o", str(obj)], capture_output=True, text=True)
            if run.returncode:
                raise NativeUnavailable(f"{Path(source).name} did not compile:\n{run.stderr[-3000:]}")
            objects.append(str(obj))
        partial = Path(work) / library.name
        run = subprocess.run([cxx, "-shared", "-pthread", *objects, "-o", str(partial)], capture_output=True, text=True)
        if run.returncode:
            raise NativeUnavailable(f"lib{library.stem} did not link:\n{run.stderr[-3000:]}")
        os.replace(partial, library)


def includes(quantized):
    """The include folders of the host rows: this folder and quantized-projections' code (nvfp4.h)."""
    return [HERE, Path(quantized)]
