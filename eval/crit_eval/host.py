"""The machine a result was measured on, stored as the result's `host`.

Host hardware moves speed a lot: an engine that streams weights or experts from host memory is bound by the GPU's
PCIe link (generation and width) and the host's RAM, and a CPU expert path by the CPU's cores. describe() records
the OS (and whether it is WSL), the CPU model and core counts, the physical RAM (under WSL the VM's) and per GPU
its VRAM, driver and PCIe link. Every probe is best-effort: a value that cannot be read is None and never fails a run.

A GPU lowers its current PCIe link generation while idle, so `gen_current` is sampled at the start of the run
(`current_sampled`: "idle"); `gen_max` is the most the GPU and this host's slot allow together, `gen_gpu_max` and
`gen_host_max` each side's own maximum (drivers that report them).
"""
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .probe import host_memory

GPU_FIELDS = ("index", "name", "memory.total", "driver_version", "pcie.link.gen.current", "pcie.link.gen.max",
              "pcie.link.width.current", "pcie.link.width.max")
# Newer drivers only; queried apart so an older nvidia-smi still answers the fields above.
GPU_SIDE_FIELDS = ("pcie.link.gen.gpumax", "pcie.link.gen.hostmax")
MIB = 1024 ** 2


def _wsl():
    return "microsoft" in platform.release().lower() or "WSL_DISTRO_NAME" in os.environ


def _os_release():
    """PRETTY_NAME from /etc/os-release, or None."""
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            for line in handle:
                name, _, value = line.strip().partition("=")
                if name == "PRETTY_NAME":
                    return value.strip('"') or None
    except OSError:
        pass
    return None


def operating_system():
    return {"system": platform.system() or None, "release": platform.release() or None,
            "version": platform.version() or None,
            "distribution": _os_release() if sys.platform.startswith("linux") else None, "wsl": _wsl()}


def _cpuinfo():
    """(model name, physical cores) from /proc/cpuinfo; physical cores None without topology fields."""
    model, cores, block = None, set(), {}
    with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
        for line in [*handle, ""]:
            name, _, value = line.partition(":")
            name = name.strip()
            if not name:   # a blank line ends one logical CPU's block
                if "core id" in block:
                    cores.add((block.get("physical id"), block["core id"]))
                block = {}
                continue
            block[name] = value.strip()
            if name == "model name" and model is None:
                model = value.strip()
    return model, len(cores) or None


def _windows_cpu_model():
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip() or None


def _windows_physical_cores():
    """Processor cores (RelationProcessorCore entries) of the calling thread's processor group, or None."""
    import ctypes

    class Info(ctypes.Structure):   # SYSTEM_LOGICAL_PROCESSOR_INFORMATION
        _fields_ = [("mask", ctypes.c_size_t), ("relationship", ctypes.c_int), ("union", ctypes.c_ulonglong * 2)]
    kernel32 = ctypes.windll.kernel32
    length = ctypes.c_ulong(0)
    kernel32.GetLogicalProcessorInformation(None, ctypes.byref(length))
    entries = (Info * (length.value // ctypes.sizeof(Info)))()
    if not length.value or not kernel32.GetLogicalProcessorInformation(entries, ctypes.byref(length)):
        return None
    return sum(1 for entry in entries if entry.relationship == 0) or None


def cpu():
    model = physical = None
    try:
        if sys.platform == "win32":
            model = _windows_cpu_model()
            physical = _windows_physical_cores()
        elif sys.platform.startswith("linux"):
            model, physical = _cpuinfo()
    except (OSError, ValueError, AttributeError, ImportError):
        pass
    return {"model": model or platform.processor() or None, "physical_cores": physical,
            "logical_cores": os.cpu_count()}


def _nvidia_smi():
    found = shutil.which("nvidia-smi")
    if found:
        return found
    wsl = Path("/usr/lib/wsl/lib/nvidia-smi")
    return str(wsl) if wsl.is_file() else None


def _query(binary, fields):
    """One row of values per GPU for fields, '[N/A]'-like values as None; None when nvidia-smi fails."""
    try:
        done = subprocess.run([binary, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    rows = [[None if value.startswith("[") or not value else value for value in (v.strip() for v in line.split(","))]
            for line in done.stdout.splitlines() if line.strip()]
    return rows if rows and all(len(row) == len(fields) for row in rows) else None


def _int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def gpus():
    """Per GPU: index, name, VRAM, driver and PCIe link; None where nvidia-smi is missing or fails."""
    binary = _nvidia_smi()
    rows = _query(binary, GPU_FIELDS) if binary else None
    if rows is None:
        return None
    sides = _query(binary, GPU_SIDE_FIELDS)
    if sides is not None and len(sides) != len(rows):
        sides = None
    result = []
    for i, row in enumerate(rows):
        values = dict(zip(GPU_FIELDS, row))
        side = dict(zip(GPU_SIDE_FIELDS, sides[i])) if sides else {}
        memory = _int(values["memory.total"])
        result.append({
            "index": _int(values["index"]), "name": values["name"],
            "memory_total_bytes": None if memory is None else memory * MIB, "driver": values["driver_version"],
            "pcie": {"gen_current": _int(values["pcie.link.gen.current"]),
                     "width_current": _int(values["pcie.link.width.current"]), "current_sampled": "idle",
                     "gen_max": _int(values["pcie.link.gen.max"]), "width_max": _int(values["pcie.link.width.max"]),
                     "gen_gpu_max": _int(side.get("pcie.link.gen.gpumax")),
                     "gen_host_max": _int(side.get("pcie.link.gen.hostmax"))},
        })
    return result


def describe():
    """The result's `host`: platform and python as before, plus OS, CPU, RAM and GPUs."""
    def safely(probe, fallback=None):
        try:
            return probe()
        except Exception:   # a probe never fails the run
            return fallback
    return {"platform": safely(platform.platform), "python": safely(platform.python_version),
            "os": safely(operating_system), "cpu": safely(cpu),
            "memory_total_bytes": safely(lambda: host_memory()[0]), "gpus": safely(gpus)}


def summary(host):
    """One line naming the host's hardware, from a result's `host` (older results hold only platform)."""
    gib = lambda n: f"{n / 1024 ** 3:.0f} GiB"
    system = host.get("os") or {}
    parts = [(system.get("distribution") or " ".join(filter(None, (system.get("system"), system.get("release")))))
             + (" (WSL)" if system.get("wsl") else "") or host.get("platform") or "unknown host"]
    processor = host.get("cpu") or {}
    if processor.get("model"):
        cores = "/".join(str(n) for n in (processor.get("physical_cores"), processor.get("logical_cores")) if n)
        parts.append(processor["model"] + (f" ({cores} cores)" if cores else ""))
    if host.get("memory_total_bytes"):
        parts.append(gib(host["memory_total_bytes"]) + " RAM")
    for gpu in host.get("gpus") or []:
        link = gpu.get("pcie") or {}
        text = gpu.get("name") or "GPU"
        if gpu.get("memory_total_bytes"):
            text += " " + gib(gpu["memory_total_bytes"])
        if link.get("gen_max"):
            text += f", PCIe {link['gen_max']}.0 x{link.get('width_max') or '?'}"
            if link.get("gen_gpu_max") and link["gen_gpu_max"] > link["gen_max"]:
                text += f" (GPU supports {link['gen_gpu_max']}.0)"
        parts.append(text)
    return "; ".join(parts)
