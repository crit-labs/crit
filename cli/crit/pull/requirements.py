"""Whether this machine meets a crit-hub card's requirements: its platform, the GPU's architecture and memory (from a
`crit eval probe` result) and the CUDA toolkit (nvcc, else the probe's CUDA runtime), or for a card whose backend is
hip, the ROCm version (the probe's HIP runtime); and whether a prebuilt artifact's toolchain matches it (toolchain)."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ARCH = re.compile(r"sm_(\d{1,2})(\d)([a-z]?)")


def platform():
    """This machine as crit-hub names platforms: windows, macos, wsl2 or linux."""
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    try:
        wsl = "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        wsl = False
    return "wsl2" if wsl or os.environ.get("WSL_DISTRO_NAME") else "linux"


def arch_runs(required, capability):
    """Whether code built for `required` (sm_120a) runs on a GPU of compute capability `capability` (12.0): an
    architecture-specific build (a) only on that exact capability, another on the same major at that minor or later."""
    found, wanted = ARCH.fullmatch(required), re.fullmatch(r"(\d+)\.(\d+)", str(capability or ""))
    if not found or not wanted:
        return False
    major, minor, suffix = int(found[1]), int(found[2]), found[3]
    if (int(wanted[1]), int(wanted[2])) == (major, minor):
        return True
    return suffix != "a" and int(wanted[1]) == major and int(wanted[2]) >= minor


def nvcc():
    """The CUDA compiler and the toolkit version it reports ("13.1"): the one on PATH, else CUDA_HOME's, else
    /usr/local/cuda's; (None, None) without one."""
    candidates = [shutil.which("nvcc")]
    if os.environ.get("CUDA_HOME"):
        candidates.append(str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"))
    candidates.append("/usr/local/cuda/bin/nvcc")
    for path in filter(None, candidates):
        try:
            output = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        found = re.search(r"release (\d+)\.(\d+)", output)
        if found:
            return path, f"{found[1]}.{found[2]}"
    return None, None


def toolkit():
    """The CUDA toolkit nvcc reports ("13.1"), or None without nvcc."""
    return nvcc()[1]


def at_least(found, required):
    """Whether CUDA `found` ("13.2") serves code that needs `required` ("13.1"): the same major, that minor or later."""
    a, b = (re.fullmatch(r"(\d+)\.(\d+)", str(v or "")) for v in (found, required))
    return bool(a and b) and int(a[1]) == int(b[1]) and int(a[2]) >= int(b[2])


def hip_version(runtime):
    """ROCm's version ("6.4") from a HIP runtime version (HIP_VERSION: major * 10^7 + minor * 10^5 + patch), or None."""
    if not isinstance(runtime, int) or runtime <= 0:
        return None
    return f"{runtime // 10_000_000}.{runtime // 100_000 % 100}"


def platform_row(platforms, here, verb="measured"):
    """Whether this machine is one of `platforms`. Linux and WSL2 stand in for each other, with a note: both are Linux
    and the engine's CUDA stack is the same."""
    platforms = platforms or []
    linux = {"linux", "wsl2"}
    met = here in platforms or (here in linux and bool(linux & set(platforms)))
    return {"requirement": "platform", "required": platforms, "found": here, "met": met,
            **({"note": f"{verb} on {', '.join(platforms)}"} if met and here not in platforms else {})}


def check(requirements, device, cuda=None, here=None):
    """One row per requirement: {requirement, required, found, met}. `device` is a crit eval probe result; `cuda`
    the toolkit's version (default: nvcc's); `here` the platform (default: this machine's). Linux and WSL2 stand in
    for each other, with a note: both are Linux and the engine's CUDA stack is the same."""
    rows = [platform_row(requirements.get("platforms"), here or platform())]
    backend = requirements.get("backend", "cuda")
    if backend != "cuda":
        return rows + other_backend(requirements, device, backend)
    capability = device.get("compute_capability")
    arch = requirements.get("gpu_arch") or []
    rows.append({"requirement": "gpu_arch", "required": arch, "found": f"{device.get('device')} (compute capability "
                 f"{capability})", "met": any(arch_runs(a, capability) for a in arch)})
    total = device.get("memory_total_bytes")
    memory = requirements.get("gpu_memory_gb")
    rows.append({"requirement": "gpu_memory_gb", "required": memory,
                 "found": round(total / 1e9, 1) if isinstance(total, (int, float)) else None,
                 "met": isinstance(total, (int, float)) and isinstance(memory, (int, float)) and total / 1e9 >= memory})
    found, source = (cuda, "nvcc") if cuda else (None, None)
    if found is None:
        found, source = toolkit(), "nvcc"
    if found is None and isinstance(device.get("cuda_runtime"), int):
        runtime = device["cuda_runtime"]
        found, source = f"{runtime // 1000}.{runtime % 1000 // 10}", "the probe's CUDA runtime (no nvcc found)"
    rows.append({"requirement": "cuda", "required": requirements.get("cuda"), "found": found,
                 "met": at_least(found, requirements.get("cuda")), **({"note": source} if found else {})})
    return rows


def memory_row(requirements, device):
    total = device.get("memory_total_bytes")
    memory = requirements.get("gpu_memory_gb")
    return {"requirement": "gpu_memory_gb", "required": memory,
            "found": round(total / 1e9, 1) if isinstance(total, (int, float)) else None,
            "met": isinstance(total, (int, float)) and isinstance(memory, (int, float)) and total / 1e9 >= memory}


def other_backend(requirements, device, backend):
    """The architecture, memory and toolkit rows of a card for a backend other than CUDA. A hip card needs a HIP probe
    whose architecture it was built for (code objects are per architecture, gfx1201) and a ROCm of its major version
    at its minor or later; crit pull checks no metal or vulkan device yet, so those rows are unmet."""
    arch = requirements.get("gpu_arch") or []
    probed = device.get("backend") or "cuda"
    if backend != "hip":
        return [{"requirement": "backend", "required": backend, "found": probed, "met": False,
                 "note": "crit pull checks cuda and hip devices only"}]
    rows = [{"requirement": "gpu_arch", "required": arch, "found": f"{device.get('device')} ({device.get('arch')})",
             "met": probed == "hip" and device.get("arch") in arch}]
    rows.append(memory_row(requirements, device))
    found = hip_version(device.get("runtime_version")) if probed == "hip" else None
    rows.append({"requirement": "rocm", "required": requirements.get("rocm"), "found": found,
                 "met": at_least(found, requirements.get("rocm")),
                 **({"note": "the probe's HIP runtime"} if found else {})})
    return rows


def version(value):
    """(13, 1) from "13.1", or None."""
    found = re.fullmatch(r"(\d+)\.(\d+)", str(value or ""))
    return (int(found[1]), int(found[2])) if found else None


def driver_cuda():
    """The newest CUDA version the NVIDIA driver runs ("13.1"): cuDriverGetVersion from the driver library, else
    nvidia-smi's header; None without a driver."""
    import ctypes
    for name in (["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"]):
        try:
            library = ctypes.WinDLL(name) if sys.platform == "win32" else ctypes.CDLL(name)
            value = ctypes.c_int(0)
            if library.cuDriverGetVersion(ctypes.byref(value)) == 0 and value.value > 0:
                return f"{value.value // 1000}.{value.value % 1000 // 10}"
        except (OSError, AttributeError):
            continue
    try:
        output = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    return f"{found[1]}.{found[2]}" if found else None


def toolchain(built, device, driver=None, here=None, python=None):
    """Whether a prebuilt artifact runs on this machine, one row per field of the toolchain it was built with
    (`built`, a card download's toolchain): the platform, the GPU architecture (as for requirements), the CUDA
    version it was compiled with against the newest one the driver runs (`driver`, default: this driver's; a newer
    driver runs older code), or for hip the ROCm version, and the Python version its extension modules were built
    for (`python`, default: this interpreter's "3.12"). No nvcc is needed: nothing is compiled."""
    rows = [platform_row(built.get("platforms"), here or platform(), "built")]
    backend = built.get("backend", "cuda")
    if backend == "hip":
        rows += [row for row in other_backend(built, device, backend) if row["requirement"] != "gpu_memory_gb"]
    elif backend != "cuda":
        rows += other_backend(built, device, backend)
    else:
        capability = device.get("compute_capability")
        arch = built.get("gpu_arch") or []
        rows.append({"requirement": "gpu_arch", "required": arch, "found": f"{device.get('device')} (compute "
                     f"capability {capability})", "met": any(arch_runs(a, capability) for a in arch)})
        if "cuda" in built:
            found = driver if driver is not None else driver_cuda()
            rows.append({"requirement": "cuda", "required": built["cuda"], "found": found, "note": "the driver's CUDA",
                         "met": bool(version(found) and version(built["cuda"]))
                                and version(found) >= version(built["cuda"])})
    if "python" in built:
        found = python or f"{sys.version_info[0]}.{sys.version_info[1]}"
        rows.append({"requirement": "python", "required": built["python"], "found": found,
                     "met": version(found) is not None and version(found) == version(built["python"])})
    return rows


def unmet(rows):
    return [f"{row['requirement']}: needs {row['required']}, this machine has {row['found']}" for row in rows if not row["met"]]
