"""directory_ops.cu built at load through Core's native build (crit_plugins.native.extension), once a process for the
device's architecture and cached across processes under a name hashed from the source's path and bytes (each engine
has its own copy of the plugin); None where it does not build, and device_directory.py runs its torch ops instead."""
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
    from crit_plugins.native import NativeUnavailable, extension
    try:
        return extension("crit_expert_directory", [HERE / "directory_ops.cu"], arch=arch)
    except NativeUnavailable as error:   # no compiler, or headers that do not match: the torch ops serve
        _WHY[arch] = str(error).splitlines()[0][:300]
        warnings.warn(f"device-expert-cache: the directory's CUDA ops did not build for {arch} ({_WHY[arch]}); its "
                      "torch ops plan and fill instead (slower, same rule)")
        return None
