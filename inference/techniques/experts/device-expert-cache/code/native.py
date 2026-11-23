"""The directory's plan and fill on the device: crit's kernel expert-directory (kernels/techniques/experts/
expert-directory), built for the device by crit's kernel build through crit_plugins.kernels.load_kernel, once a process,
and recorded in the engine's facts; None where it does not build (no CUDA compiler, no crit_kernels: the kernel's
fallback with its reason), and device_directory.py runs its torch ops instead. The plugin that loads it is the one
whose package imports this code (crit_plugins.kernels.plugin_folder): device-expert-cache itself, or a plugin that
imports its code as a source (expert-offload), whose plugin.json declares the kernel too."""
from pathlib import Path
import threading
import warnings

HERE = Path(__file__).resolve().parent
KERNEL = "expert-directory"   # plugin.json "kernels"
_BUILT = {}
_LOCK = threading.Lock()


def directory_ops(device):
    """The directory's plan and fill on `device` (a CUDA device): crit's kernel expert-directory, or None (the reason
    in why())."""
    import torch
    device = torch.device(device)
    if device.type != "cuda":
        return None
    from crit_plugins.kernels import plugin_folder
    folder = plugin_folder(__name__, HERE.parent)
    with _LOCK:   # once a process for each plugin that asks (load_kernel builds it once)
        if folder not in _BUILT:
            _BUILT[folder] = _build(folder)
    return _BUILT[folder]


def why():
    """Why the kernel did not build here (crit_plugins.kernels.fallbacks), or None."""
    from crit_plugins.kernels import fallbacks
    return fallbacks().get(KERNEL)


def _build(folder):
    from crit_plugins.kernels import load_kernel
    module = load_kernel(KERNEL, folder=folder)
    if module is None:   # no compiler, or headers that do not match: the torch ops serve
        warnings.warn(f"device-expert-cache: crit's kernel {KERNEL} did not build here ({why()}); the directory's "
                      "torch ops plan and fill instead (slower, same rule)")
    return module
