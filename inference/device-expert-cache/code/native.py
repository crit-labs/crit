"""directory_ops.cu built at load through torch's extension builder, once a process for the device's architecture and
cached across processes under a name hashed from the source's path and bytes (each engine has its own copy of the
plugin); None where it does not build, and device_directory.py runs its torch ops instead."""
import hashlib
import os
from pathlib import Path
import threading
import warnings

HERE = Path(__file__).resolve().parent
_BUILT = {}
_LOCK = threading.Lock()
_WHY = {}


def directory_ops(device, arch=None):
    """The directory's plan and fill on `device` (a CUDA device), or None (the reason in why(device))."""
    import torch
    device = torch.device(device)
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    arch = arch or f"{major}.{minor}"
    with _LOCK:
        if arch not in _BUILT:
            _BUILT[arch] = _build(arch)
    return _BUILT[arch]


def why(arch):
    return _WHY.get(arch)


def _build(arch):
    from torch.utils.cpp_extension import load
    source = HERE / "directory_ops.cu"
    copy = hashlib.sha256(str(source).encode() + source.read_bytes()).hexdigest()[:12]
    before = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load(name=f"crit_expert_directory_{arch.replace('.', '_')}_{copy}", sources=[str(source)],
                    extra_cuda_cflags=["-O3", "-std=c++20"], extra_cflags=["-O3", "-std=c++20"], verbose=False)
    except Exception as error:   # no compiler, or headers that do not match: the torch ops serve
        _WHY[arch] = f"{type(error).__name__}: {error}".splitlines()[0][:300]
        warnings.warn(f"device-expert-cache: the directory's CUDA ops did not build for {arch} ({_WHY[arch]}); its "
                      "torch ops plan and fill instead (slower, same rule)")
        return None
    finally:
        if before is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = before
