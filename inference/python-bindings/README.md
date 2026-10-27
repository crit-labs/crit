# Call the CUDA entries from a Python engine

One shared library with a plain C ABI over the CUDA entries an engine's forward
pass uses most, and a standard-library `ctypes` loader. A Python engine passes
its own device addresses and streams (`tensor.data_ptr()`,
`torch.cuda.current_stream().cuda_stream`) and gets each entry's result in its
buffers, without writing or compiling a Python extension around them.

## When to use it

When the forward pass is driven from Python and calls several of
[stored-tile-gemm](../stored-tile-gemm/README.md),
[recurrent-layer-kernels](../recurrent-layer-kernels/README.md),
[recurrent-state-recovery](../recurrent-state-recovery/README.md),
[fp8-kv-attention](../fp8-kv-attention/README.md),
[token-sampling](../token-sampling/README.md) or the greedy accept of
[device-speculative-round](../device-speculative-round/README.md). Not for a
native C++ engine, which links the entries directly, nor for calls that need an
entry's C++-only extension points.

## How it works

**The ABI.** [crit_inference.h](code/crit_inference.h) declares every function
with C types only: pointers, `int`, fixed-width integers, `size_t`, `float` and
`double`. Device buffers are `void*` or typed pointers, streams are
`cudaStream_t` passed as `void*` (0 for the legacy default stream), flags are
ints. Launch functions return the entry's `cudaError_t` and only enqueue. Names
carry their entry: `crit_st_*` stored-tile-gemm, `crit_rk_*`
recurrent-layer-kernels, `crit_rr_*` recurrent-state-recovery, `crit_kv_*`
fp8-kv-attention, `crit_ts_*` token-sampling, `crit_dr_*` device round.

**Forwarding.** [crit_inference.cpp](code/crit_inference.cpp) rebuilds each
entry's argument structs (`Fp8Weights`, `Conv`, `DeltaInputs`, `Cache`,
`Decode`, `Rows`, `Verify`, `AcceptRows`, ...) from the flat arguments and calls
the entry. It adds no checks beyond null plan or row buffers and an activation
code outside 0-2: every other refusal is the entry's own, with its code.

**Plans and rows.** Stored-tile plans are opaque 64-byte host buffers that
`crit_st_plan*` fill (with the scratch sizes the engine allocates) and every
launch reads by value. Token sampling's `Request` and `Params` rows are packed
on the host by `crit_ts_request_pack` / `crit_ts_params_pack`; the engine copies
them to device memory, where launches read and advance them.

**The loader.** [crit_inference.py](code/crit_inference.py) has no signature
table of its own. `signatures()` reads every `CRIT_API` declaration from the
header next to it, and `Library` sets each function's `restype` and `argtypes`
from there, so a Python call cannot pass arguments in an order the C side does
not declare. Status-returning functions raise `CudaError(function, code)`.
`lib.st_gemv_fp8(...)` calls `crit_st_gemv_fp8`; helpers return plans
(`gemm_plan`, `gemv_plan`, `a16_plan`), packed rows (`request`, `params`), the
packed result layout (`packed_layout`), mapped host memory (`host_alloc`) and
read files into it (`read_into`).

## Building

`crit_inference_python` is a target of Inference's CMake project when
`CRIT_INFERENCE_CUDA` is on. Build for the device's exact architecture so the
NVFP4 MMA is compiled (`st_fp4_mma_compiled()` reports it):

```text
cmake -S inference -B build -DCRIT_INFERENCE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build --target crit_inference_python
```

Copy the library next to `crit_inference.py` and `crit_inference.h` in the
engine's package, or point `CRIT_INFERENCE_LIBRARY` at it.

## Worked example

The header is the loader's signature table; reading it needs no device or
built library, and loading without a library names what to build:

```python
import ctypes
import crit_inference

table = crit_inference.signatures()
ret, args = table["st_gemv_fp8"]
print(len(table), "functions; crit_st_gemv_fp8:", ret, args)
assert (ret, args) == ("e", "pppiipiipiip")  # status(plan, tiles, row_scale, n, k, x, ldx, m, y, ldy, add, stream)
assert table["ts_request_pack"][1].startswith("pddd")
assert ctypes.sizeof(crit_inference.plan_buffer()) == crit_inference.PLAN_BYTES

try:
    crit_inference.Library("missing/libcrit_inference_python.so")
except FileNotFoundError as error:
    assert "crit_inference_python target" in str(error)

# With the library built, an engine's decode projection reads:
#   lib = crit_inference.Library()
#   plan = lib.gemv_plan(fp4=False, n=N, k=K)
#   lib.st_gemv_fp8(plan, tiles.data_ptr(), row_scale.data_ptr(), N, K, x.data_ptr(), K, 1,
#                   y.data_ptr(), N, 0, torch.cuda.current_stream().cuda_stream)
```

## Adapting it

Bind another entry by declaring its functions in the header, forwarding them in
the C++ file and adding its sources to the target; the host test then checks
the two agree. Raise `CRIT_INFERENCE_ABI` and `ABI_VERSION` together when an
existing function's arguments change. Model-specific kernels (norms, rotary
embedding, gates, embeddings) belong to the engine; it can build them into its
own library the same way.

## Validation

[test_inference_bindings.py](../../tests/test_inference_bindings.py) checks on
the host that every declaration has a definition with the same types, the ABI
version and plan size match the loader, a missing library is refused with its
path, and file reads into host memory are exact.
[python_bindings_check.py](../../tests/inference/python_bindings_check.py)
(ctest `crit_inference_python`) loads the built library on a device: every
function exported, greedy select and sample through packed rows, grouped
accepted counts, and a refused launch raised with its code (passed on an RTX 5090, sm_120a). The kernels
themselves are validated by their entries' own tests; the bindings forward
unchanged. No performance claim: a call adds host-side `ctypes` conversion
(microseconds) and nothing inside a captured graph.
