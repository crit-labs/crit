"""The host rows from Python: host_capi.cpp built by native.py and called through ctypes on CPU torch tensors (contiguous,
the dtypes host_capi names). ctypes releases the GIL during each call, so `threads` splits a matrix's rows over a
thread pool; rows are independent, so any split gives the same bits."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import os

from . import native

c_size, c_ptr, c_int, c_float, c_char_p = ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_char_p
ERROR_BYTES = 2048
ISA = {"best": -1, "scalar": 0, "portable": 1, "avx2": 2, "avx512": 3}


def ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def check(tensor, name, dtype):
    if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.dtype != dtype:
        raise ValueError(f"{name} must be a contiguous CPU {dtype} tensor, not {tensor.dtype} on {tensor.device}")


def declare(library, name, restype, *argtypes):
    function = getattr(library, name)
    function.restype, function.argtypes = restype, list(argtypes)
    return function


class HostRows:
    """host_capi's functions over one loaded library (build(), or a technique's library that holds them too)."""

    def __init__(self, library):
        self.library = library
        d = lambda name, restype, *args: declare(library, name, restype, *args)  # noqa: E731
        self._int_isa = d("crit_hx_int_isa", c_char_p)
        self._isa = d("crit_hx_isa", c_char_p)
        self._int_rows = d("crit_hx_int_rows", c_int, c_ptr, c_ptr, c_ptr, c_float, c_size, c_size, c_ptr, c_ptr, c_int,
                           c_int, c_int, c_size, c_size, c_int, c_ptr, c_size, c_ptr, c_size)
        self._nvfp4_rows = d("crit_hx_nvfp4_rows", c_int, c_ptr, c_ptr, c_ptr, c_size, c_size, c_ptr, c_int, c_size,
                             c_size, c_int, c_ptr, c_size, c_ptr, c_size)
        self._q4 = d("crit_hx_quantize_e2m1x2", c_int, c_ptr, c_int, c_size, c_float, c_ptr, c_ptr, c_ptr, c_size)
        self._q8 = d("crit_hx_quantize_int8", c_int, c_ptr, c_int, c_size, c_int, c_ptr, c_ptr, c_ptr, c_size)
        self._swiglu = d("crit_hx_swiglu", None, c_ptr, c_size, c_size, c_ptr)
        self._combine = d("crit_hx_combine", None, c_ptr, c_ptr, c_int, c_ptr, c_size, c_size, c_size, c_ptr)
        self._exp = d("crit_hx_exp", c_float, c_float)
        self._pool = None

    @classmethod
    def build(cls, quantized):
        """The host rows built from this technique's code (native.py); quantized: quantized-projections' code folder."""
        return cls(native.build("crit_host_rows", native.arithmetic_units(), native.includes(quantized)))

    def int_isa(self):
        return self._int_isa().decode()

    def isa(self):
        return self._isa().decode()

    def _call(self, function, *args):
        error = ctypes.create_string_buffer(ERROR_BYTES)
        if function(*args, error, ERROR_BYTES):
            raise ValueError(error.value.decode() or "host rows failed")

    def _split(self, rows, threads, run):
        threads = max(1, min(int(threads or 1), (rows + 63) // 64))
        if threads == 1:
            run(0, rows)
            return
        if self._pool is None or self._pool._max_workers < threads:
            self._pool = ThreadPoolExecutor(max_workers=max(threads, os.cpu_count() or 1))
        step = (rows + threads - 1) // threads
        for future in [self._pool.submit(run, r0, min(rows, r0 + step)) for r0 in range(0, rows, step)]:
            future.result()

    def int_rows(self, codes, scales, post, values, act_scales, block, weights="e2m1x2", factor=1.0, out=None,
                 threads=1, isa="best"):
        """codes [N][K / 2] uint8, scales [N][K / 16] uint8 (E4M3), post [N] FP32 or None, values [T][K] int8,
        act_scales [T][K / block] FP32 -> out [T][N] FP32 (each row times factor * post, or factor)."""
        import torch
        rows, cols, tokens = codes.shape[0], codes.shape[1] * 2, values.shape[0]
        check(codes, "codes", torch.uint8)
        check(scales, "scales", torch.uint8)
        check(values, "values", torch.int8)
        check(act_scales, "act_scales", torch.float32)
        if post is not None:
            check(post, "post", torch.float32)
        if tuple(scales.shape) != (rows, cols // 16) or values.shape[1] != cols or tuple(act_scales.shape) != (
                tokens, cols // block) or (post is not None and post.numel() != rows):
            raise ValueError("int_rows: codes [N][K / 2], scales [N][K / 16], post [N], values [T][K], act_scales "
                             "[T][K / block]")
        out = torch.empty(tokens, rows, dtype=torch.float32) if out is None else out
        check(out, "out", torch.float32)
        args = (ptr(codes), ptr(scales), ptr(post) if post is not None else None, float(factor), rows, cols,
                ptr(values), ptr(act_scales), tokens, int(block), 1 if weights == "int4" else 0)
        self._split(rows, threads, lambda r0, r1: self._call(self._int_rows, *args, r0, r1, ISA[isa], ptr(out), rows))
        return out

    def nvfp4_rows(self, codes, scales, post, x, out=None, threads=1, isa="best"):
        """NVFP4 rows in the declared FP32 order against FP32 x [T][K], times post [N] (or 1)."""
        import torch
        rows, cols, tokens = codes.shape[0], codes.shape[1] * 2, x.shape[0]
        check(codes, "codes", torch.uint8)
        check(scales, "scales", torch.uint8)
        check(x, "x", torch.float32)
        if post is not None:
            check(post, "post", torch.float32)
        out = torch.empty(tokens, rows, dtype=torch.float32) if out is None else out
        args = (ptr(codes), ptr(scales), ptr(post) if post is not None else None, rows, cols, ptr(x), tokens)
        self._split(rows, threads, lambda r0, r1: self._call(self._nvfp4_rows, *args, r0, r1, ISA[isa], ptr(out), rows))
        return out

    def quantize_e2m1x2(self, x, global_scale):
        import torch
        x = x.float().contiguous()
        values = torch.empty(x.shape, dtype=torch.int8)
        scales = torch.empty(x.shape[0], x.shape[1] // 16, dtype=torch.float32)
        self._call(self._q4, ptr(x), x.shape[0], x.shape[1], float(global_scale), ptr(values), ptr(scales))
        return values, scales

    def quantize_int8(self, x, block=32):
        import torch
        x = x.float().contiguous()
        values = torch.empty(x.shape, dtype=torch.int8)
        scales = torch.empty(x.shape[0], x.shape[1] // block, dtype=torch.float32)
        self._call(self._q8, ptr(x), x.shape[0], x.shape[1], int(block), ptr(values), ptr(scales))
        return values, scales

    def swiglu(self, gate_up):
        import torch
        gate_up = gate_up.float().contiguous()
        out = torch.empty(gate_up.shape[0], gate_up.shape[1] // 2, dtype=torch.float32)
        self._swiglu(ptr(gate_up), gate_up.shape[0], gate_up.shape[1] // 2, ptr(out))
        return out

    def combine(self, y, ids, weights, out=None):
        """out[t] = acc, acc = fma(weights[t][j], y[t * top + j], acc) for j ascending over ids[t][j] >= 0."""
        import torch
        y, weights = y.float().contiguous(), weights.float().contiguous()
        ids = ids.contiguous()
        if ids.dtype not in (torch.int64, torch.int32):
            raise ValueError("ids are int64 or int32")
        tokens, top = ids.shape
        out = torch.empty(tokens, y.shape[-1], dtype=torch.float32) if out is None else out
        self._combine(ptr(y), ptr(ids), 1 if ids.dtype == torch.int64 else 0, ptr(weights), tokens, top, y.shape[-1],
                      ptr(out))
        return out

    def exp(self, x):
        return self._exp(float(x))
