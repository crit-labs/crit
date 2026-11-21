"""Radix select's keys for Triton kernels (radix.cuh's radix_key for CUDA ones): an FP32 value as a uint32 key of the
same order (-0 just below +0), a NaN as 0, below every number, -inf included."""
import triton
import triton.language as tl


@triton.jit
def radix_key(x):
    u = x.to(tl.uint32, bitcast=True)
    key = tl.where((u & 0x80000000) != 0, ~u, u | 0x80000000)
    return tl.where(x != x, 0, key).to(tl.uint32)
