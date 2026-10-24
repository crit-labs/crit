"""Measure the device ceilings Eval compares engines with: `crit eval probe --out device.json`.

Builds probe/device_probe.cu with nvcc (CUDA toolkit with cuBLASLt), runs it several times and
keeps the median of each value: achievable copy bandwidth (GB/s, bytes read plus written) for
the decode pass floor, and dense GEMM TFLOPS per weight format (bf16, fp8, nvfp4) for the
prefill floor. It also keeps the least free device memory any run saw, for the max context,
and the host's physical and available memory. Pass the file to `crit eval run --device
device.json`. Probe on the machine and in the environment the engine runs in, with the same
other GPU load as the evaluation.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from . import EVAL_DIR, metrics, portable_path, write_json

SOURCE = EVAL_DIR / "probe" / "device_probe.cu"


def find_nvcc(explicit=None):
    candidates = [explicit] if explicit else []
    candidates.append(shutil.which("nvcc"))
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if root:
            candidates.append(str(Path(root) / "bin" / "nvcc"))
    candidates += sorted((str(p) for p in Path("/usr/local").glob("cuda-*/bin/nvcc")), reverse=True)
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit("nvcc not found: install the CUDA toolkit or pass --nvcc PATH")


def host_compiler():
    """nvcc arguments naming the newest MSVC x64 compiler on Windows when cl.exe is not on PATH (outside a
    developer prompt); nvcc sets up the compiler's environment from -ccbin itself. Elsewhere none."""
    if os.name != "nt" or shutil.which("cl"):
        return []
    vswhere = (Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) /
               "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if not vswhere.is_file():
        return []
    done = subprocess.run([str(vswhere), "-products", "*", "-requires",
                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                          capture_output=True, text=True)
    compilers = [cl for line in done.stdout.splitlines() if line.strip()
                 for cl in (Path(line.strip()) / "VC" / "Tools" / "MSVC").glob("*/bin/Hostx64/x64/cl.exe")]
    version = lambda cl: tuple(int(part) if part.isdigit() else 0 for part in cl.parents[3].name.split("."))
    return ["-ccbin", str(max(compilers, key=version).parent)] if compilers else []


def build(nvcc, directory):
    binary = Path(directory) / ("device_probe.exe" if os.name == "nt" else "device_probe")
    command = [nvcc, *host_compiler(), "-O2", "-std=c++17", "-o", str(binary), str(SOURCE), "-lcublasLt"]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:   # nvcc reports a missing host compiler on stdout
        raise SystemExit("building the device probe failed:\n" + (done.stdout + done.stderr)[-4000:])
    return binary


def combine(samples):
    """Median of each measured value over repeated probe runs."""
    first = samples[0]
    formats = sorted({f for s in samples for f in s.get("tflops", {})})
    return {
        "device": first.get("device"), "compute_capability": first.get("compute_capability"),
        "cuda_runtime": first.get("cuda_runtime"),
        "bandwidth_gbs": metrics.median([s["bandwidth_gbs"] for s in samples if s.get("bandwidth_gbs")]),
        "tflops": {f: metrics.median([s["tflops"][f] for s in samples if f in s.get("tflops", {})]) for f in formats},
        "memory_total_bytes": max((s["memory_total_bytes"] for s in samples if s.get("memory_total_bytes")), default=None),
        # The least free memory any run saw: the max context must fit the device as it is shared.
        "memory_free_bytes": min((s["memory_free_bytes"] for s in samples if s.get("memory_free_bytes")), default=None),
    }


def host_memory():
    """(physical, available) host memory in bytes, or (None, None) where it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total, status.available
            return None, None
        fields = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                name, _, value = line.partition(":")
                fields[name] = int(value.split()[0]) * 1024
        return fields.get("MemTotal"), fields.get("MemAvailable")
    except (OSError, ValueError, AttributeError):
        return None, None


def probe(out, nvcc=None, repeats=3, log=print):
    nvcc = find_nvcc(nvcc)
    with tempfile.TemporaryDirectory() as directory:
        binary = build(nvcc, directory)
        samples = []
        for i in range(repeats):
            done = subprocess.run([str(binary)], capture_output=True, text=True)
            if done.returncode != 0:
                raise SystemExit("device probe failed:\n" + (done.stdout + done.stderr)[-4000:])
            sample = json.loads(done.stdout)
            samples.append(sample)
            log(f"probe {i + 1}/{repeats}: {sample['bandwidth_gbs']:.0f} GB/s, "
                + ", ".join(f"{k} {v:.0f} TFLOPS" for k, v in sample.get("tflops", {}).items()))
    result = combine(samples)
    result["host_memory_total_bytes"], result["host_memory_available_bytes"] = host_memory()
    result.update(samples=samples, nvcc=portable_path(nvcc),
                  probe_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  measured_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(out, result)
    return result
