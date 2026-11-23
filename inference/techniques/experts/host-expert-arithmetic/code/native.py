"""The host rows' C++ (host_int, host_nvfp4 and their AVX2 / AVX-512 files) built into a shared library when a plugin
loads, and opened with ctypes: Core's native build (crit_plugins.native.library: no torch headers, so a build takes
seconds, cached by content), and a library that works on any CPU the OS runs (the ISA files are compiled with their
own flags and chosen at run time through host_isa).

A technique that builds on these rows (host-expert-tier) adds its own sources to the same library through `build`.
Without a C++ compiler, or on a compiler that fails, `build` raises NativeUnavailable and the plugin runs its torch
fallback (declared.py), which gives the same bits, slower.
"""
from pathlib import Path

from crit_plugins.native import X86, NativeUnavailable, library

HERE = Path(__file__).resolve().parent
AVX2 = ("-mavx2", "-mfma")
AVX512 = ("-mavx512f", "-mfma")
AVX512_VNNI = ("-mavx512f", "-mavx512bw", "-mavx512vnni", "-mfma")


def arithmetic_units():
    """The host rows' translation units with their own flags: (path, flags)."""
    units = [(HERE / name, ()) for name in ("host_isa.cpp", "host_nvfp4.cpp", "host_int.cpp", "host_capi.cpp")]
    if X86:
        units += [(HERE / "host_nvfp4_avx2.cpp", AVX2), (HERE / "host_nvfp4_avx512.cpp", AVX512),
                  (HERE / "host_int_avx2.cpp", AVX2), (HERE / "host_int_avx512.cpp", AVX512_VNNI)]
    return units


def build(name, units, includes, defines=()):
    """The ctypes library `name` built from units [(source, flags)] with `includes`, cached by content. Raises
    NativeUnavailable when it cannot be built here."""
    return library(name, units, includes, tuple(defines) + (("-DCRIT_HOST_ISA_FILES=1",) if X86 else ()))


def includes(quantized):
    """The include folders of the host rows: this folder and quantized-projections' code (nvfp4.h)."""
    return [HERE, Path(quantized)]


__all__ = ["NativeUnavailable", "arithmetic_units", "build", "includes"]
