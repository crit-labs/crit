"""Whether this machine meets a crit-hub card's requirements: its platform, the GPU's architecture and memory (from a
`crit eval probe` result) and the CUDA toolkit (nvcc, else the probe's CUDA runtime)."""
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


def check(requirements, device, cuda=None, here=None):
    """One row per requirement: {requirement, required, found, met}. `device` is a crit eval probe result; `cuda`
    the toolkit's version (default: nvcc's); `here` the platform (default: this machine's). Linux and WSL2 stand in
    for each other, with a note: both are Linux and the engine's CUDA stack is the same."""
    rows = []
    here = here or platform()
    platforms = requirements.get("platforms") or []
    linux = {"linux", "wsl2"}
    met = here in platforms or (here in linux and bool(linux & set(platforms)))
    rows.append({"requirement": "platform", "required": platforms, "found": here, "met": met,
                 **({"note": f"measured on {', '.join(platforms)}"} if met and here not in platforms else {})})
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


def unmet(rows):
    return [f"{row['requirement']}: needs {row['required']}, this machine has {row['found']}" for row in rows if not row["met"]]
