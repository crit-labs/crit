"""expert-directory's sweep and gates bench: the fill, a plan's copies of whole expert bundles into cache slots, at the
model's NVFP4 bundle size (gate, up and down: 4-bit values and an FP8 scale per 16), from a store in mapped host memory
read over the link, as an engine's store is (pinned memory at its cudaHostGetDevicePointer address; where the platform
has no such mapping, from device memory). The link bounds the fill: on an RTX 5090 over its PCIe link every grid of 4
blocks or more and every load width copies at 26 GB/s, the copy engine's rate from the same memory (README). Copy i
goes to slot i, listed last first, from host bundle i % HOST_KEYS (a function of i alone), so a row's bytes do not
depend on the rows beside it and the host store stays small. The plan's decisions are device_cache.LayerDirectory's,
which device-expert-cache's tests check plan for plan against its host twin."""
import ctypes
import os

import torch

HOST_KEYS = 64   # distinct host bundles (177 MB of 2.76 MB bundles): rows past it reread them, over the link still


def bundle_bytes(hidden, intermediate):
    """An expert's NVFP4 bundle (gate, up, down: hidden x intermediate each, 4 bits a value and an FP8 scale per 16),
    padded to 16 bytes."""
    size = 3 * hidden * intermediate // 2 + 3 * hidden * intermediate // 16
    return (size + 15) // 16 * 16


_cudart = None


def _runtime():
    """The CUDA runtime torch loaded, through ctypes (cudaHostGetDevicePointer), or None."""
    global _cudart
    if _cudart is None:
        path = None
        try:
            with open("/proc/self/maps") as maps:
                path = next((line.split()[-1] for line in maps if "libcudart" in line), None)
            lib = ctypes.CDLL(path or "libcudart.so")
        except OSError:
            return None
        lib.cudaHostGetDevicePointer.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint)
        _cudart = lib
    return _cudart


class _View:
    def __init__(self, address, size):
        self.__cuda_array_interface__ = {"shape": (int(size),), "typestr": "|u1", "data": (int(address), False),
                                         "version": 3, "strides": None}


def mapped_host(tensor, device):
    """(pinned CPU tensor, a CUDA uint8 tensor over its device address) for a uint8 tensor's bytes, or None where the
    platform maps no pinned memory for the device (the caller keeps the pinned tensor alive while either is used)."""
    if os.name == "nt":
        return None
    torch.cuda.init()
    lib = _runtime()
    if lib is None:
        return None
    pinned = tensor.pin_memory()
    pointer = ctypes.c_void_p()
    if lib.cudaHostGetDevicePointer(ctypes.byref(pointer), ctypes.c_void_p(pinned.data_ptr()), 0) or not pointer.value:
        return None
    with torch.cuda.device(torch.device(device)):
        view = torch.as_tensor(_View(pointer.value, pinned.numel()), device=device)
    return pinned, view.view(pinned.shape)


class Fill:
    structure = "fill"

    def __init__(self):
        self._host = {}   # bundle -> (pinned, mapped): one host store a bundle size

    def shapes(self, model):
        hidden = model.get("hidden_size")
        intermediate = model.get("moe_intermediate_size", "expert_intermediate_size")
        if not hidden or not intermediate:
            return []
        return [{"bundle": bundle_bytes(int(hidden), int(intermediate))}]

    def inputs(self, shape, rows, device):
        bundle = int(shape["bundle"])
        if bundle not in self._host:
            k = torch.arange(HOST_KEYS, dtype=torch.int64)
            bundles = ((k[:, None] * 131 + torch.arange(bundle)[None, :] * 7) % 251).to(torch.uint8)
            self._host[bundle] = mapped_host(bundles, device) if torch.device(device).type == "cuda" else None
            if self._host[bundle] is None:   # no mapping here: the same bundles from device memory
                self._host[bundle] = (bundles, bundles.to(device))
        pinned, source = self._host[bundle]
        order = torch.arange(rows - 1, -1, -1, device=device, dtype=torch.int32)
        jobs = torch.zeros(rows + 1, 2, dtype=torch.int32, device=device)
        jobs[:rows, 0] = order % HOST_KEYS
        jobs[:rows, 1] = order
        keys = torch.arange(rows, device=device, dtype=torch.int64) % HOST_KEYS
        return {"source": source, "pinned": pinned, "store": torch.zeros(rows, bundle, dtype=torch.uint8, device=device),
                "jobs": jobs, "count": torch.tensor([rows], dtype=torch.int32, device=device), "keys": keys,
                "slots": torch.arange(rows, device=device, dtype=torch.int64),
                "first": torch.empty(0, dtype=torch.int32, device=device)}

    def run(self, module, s):
        module.fill(s["source"], s["store"], s["jobs"], s["count"], 0, 16, s["first"])
        return s["store"]

    def reference(self, s):
        return s["source"].index_select(0, s["keys"])   # slot i holds host bundle i % HOST_KEYS

    def check(self, got, want, s):
        if torch.equal(got, want):
            return None
        bad = int((got != want).any(dim=1).sum())
        return f"{bad} of {got.shape[0]} slots differ from their bundles"

    def first_row(self, out):
        return out[:1]

    def bytes(self, shape, rows):
        return 2 * rows * int(shape["bundle"])

    def yardstick(self, s):
        source, store, keys, slots = s["source"], s["store"], s["keys"], s["slots"]
        return lambda: store.index_copy_(0, slots, source.index_select(0, keys))
