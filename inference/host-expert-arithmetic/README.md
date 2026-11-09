# Host arithmetic for block-scaled experts

Compute block-scaled expert rows (NVFP4, MXFP4, MXFP8, MXINT8, INT4 or an engine's own block format) on the CPU
with exactly the arithmetic of the device kernel that would otherwise compute them. A row's dot product is fixed by
a declared operation order: lanes, product rounding, block order, reduction tree and final multiply. A device kernel
that declares the same order gives the same bits, so where an expert ran never changes the model's output. The
entry supplies a portable scalar oracle, a portable vector path that any compiler vectorizes for its CPU (Arm NEON
and SVE, x86 AVX, RISC-V V), hand-written AVX2 and AVX-512 paths for NVFP4, one device header for CUDA and HIP, the
same functions for Metal, and a scale-decode hook for checkpoint-specific scale encodings.

## When to use it

Use it when [measured dispatch](../heterogeneous-expert-dispatch/README.md) puts part of an expert-routed
step's misses on the CPU, and the output must not depend on that split. Measure that share from the first
candidate of a model whose experts do not fit the device (recon's `sparse_device_host` scenario) whenever host
memory reads faster than the host link copies (`crit eval probe`: `host_memory_gbs.read` above the pinned
`host_to_device_gbs`), as it does on many desktop machines: recon's placement then prices misses computed on the
CPU below misses copied over the link, so the CPU is a primary source of misses there, not a later refinement. The split varies with cache state,
host load and request history, so greedy output stops repeating if host and device round differently. One
engine saw decision forks from mixing CPU W4A8 with device W4A4. In quality mode `speed` Eval passes such
forks when they part at near-ties ([forks](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#forks-in-quality-mode-speed)), so there the exact rows are one option to measure against
the host's own faster arithmetic, not a requirement.

Don't use it when every expert fits on the device, or when measured dispatch gives the CPU no share on the
target machine. That happens when host compute time exceeds the transfer it replaces. Don't use it either when
the device kernel's order can't be declared (a library GEMM with an unspecified reduction): either the
device side is changed to a declared order or the split is not exact.

## How it works

**Format.** A `Format` names the block layout: 4-bit codes (two a byte, low nibble the even element) or 8-bit
codes, 16 or 32 elements per block scale, and the element values by code. The default is crit's NVFP4 layout from
[quantized projections](../quantized-projections/README.md) (`crit::nvfp4`, `nvfp4.py`): E2M1 codes
`[rows][cols / 2]`, one E4M3 scale per 16 values. `mxfp4_format`, `mxfp8_e4m3_format`, `mxfp8_e5m2_format` and
`mxint8_format` give the OCP microscaling layouts (32 per E8M0 scale, `e8m0_scales`), and `int4_values` an INT4
table for a layout that names its own block. A table of its own covers any other element encoding.

**Declared order.** For a row and one token's activations `x`:

1. Element `k` goes to accumulator lane `k % lanes`, with `lanes` 16 or 32.
2. Each term is `lane = fma(round(e2m1(code_k) * scale[k / 16]), x[k], lane)`. The product is rounded to
   FP32 before the FMA, and elements are visited in increasing `k`.
3. The lanes are reduced by a halving tree, `(i, i + lanes/2)`, then `/4`, down to lane 0, or sequentially,
   `lane 0 + lane 1 + ...`.
4. The sum is multiplied by `post`, which is the global-scale factor exactly as the device applies it, for
   example a precomputed `1 / global_scale`.

Steps 1 and 2 describe the `scaled_elements` product. The other common GEMV order is `block_dot`: block `b`
goes to lane `b % lanes`, its 16 unscaled codes are dotted with `x` sequentially from 0, and then
`lane = fma(dot, scale[b], lane)`. Steps 3 and 4 are the same for both. A kernel that has each lane load
whole 16-value blocks and reduce with a warp sum usually has this order, which also allows rows whose block
count isn't a multiple of the lanes.

No other operation is fused or reordered. On the host, the files compile without FP contraction. On the
device, [`dot_row_halving<L>`](code/host_nvfp4_device.cuh) and `dot_row_block<L>` use `__fmul_rn`,
`__fmaf_rn` and a shuffle tree that pairs lanes the way the host's halving reduction does. A xor butterfly
leaves lane 0 the same value.

**Paths.** `rows_scalar` is the oracle. The portable path (`Isa::portable`, what `best_isa()` returns on a CPU without
the hand-written paths) runs the same operations with every inner loop over independent lanes and a fixed lane
count, so the compiler vectorizes it for its target without changing any lane's order: it is the vector path on Arm,
RISC-V and any other CPU, and for every format other than NVFP4 on x86, where it is compiled once more with AVX2 and
with AVX-512 enabled. For NVFP4 on x86 the hand-written paths are faster. The AVX2 path (two YMM per 16 lanes) and the AVX-512F path (one ZMM
per 16 lanes) perform each lane's operations in the same order. They interleave four rows so the FMA chains
overlap, which leaves each row's arithmetic unchanged. `best_isa()` reads cpuid and xgetbv at run time, so one
binary runs on any x86-64 CPU and uses the widest path the CPU and OS support. Rows are independent, so any
split across threads gives the same bits. Each ISA file is compiled with its own flags and takes the E2M1
table as an argument, so no inline function is shared between files built for different instruction sets.

**Scale hook.** `ScaleDecoder` writes a row's block scales as the exact floats the device multiplies by. The
standard decoder reads E4M3 bytes; a decoder that is a pure byte table also sets `byte_table`, and the kernels
then look scales up inline instead of decoding each row first. A checkpoint that stores scales another way, such as deltas from a
per-slab base with exception lists, supplies its own decoder in the engine. A scale stored at a power-of-two
offset with a compensating `post` (for example halved scales against doubled E2M1 codes) gives the same bits.

## Integer block rows

The FP32 orders above are exact but slow on a CPU: one FMA per token for every 16 weights, about 8 GB/s of weights per
thread for one token and far less for a verify window's several (14-16 GB/s on 6 threads of a Ryzen 5 9600 at 3 tokens
an expert, below a PCIe 4.0 link's 25-28). A CPU tier built on them can lose to copying its misses.

`host_int.h` adds an order that is exact *and* fast. A 4-bit weight's value times 2 is a small integer (E2M1: 0, 1, 2,
3, 4, 6, 8, 12 and negatives; INT4: -8..7), and so is an activation quantized per block (E2M1 x 2 for 4-bit
activations, -127..127 for 8-bit ones). A 16-block's dot of those integers is an exact int32 in any order, so
vpdpbusd, vpmaddubsw, Arm's SDOT, dp4a and int8 tensor cores all agree on it. Only the FP32 step is declared: block b
to lane b % 16 (or 32), `lane = fma(float(sum), wscale_b * ascale_b, lane)` with the product rounded, blocks in order,
the halving (or sequential) reduction, times post.

- **Where it is exact.** Exact here means the same bits on every processor, not drift-free. For a checkpoint whose
  experts declare 4-bit input activations (NVFP4 W4A4) it is the arithmetic the checkpoint declares and the device's
  W4A4 GEMMs run; for weight-only checkpoints, 8-bit activations are a quantization of the engine's own. Against
  Eval's reference (the checkpoint's stored weights with exact activations) both are drift, whatever the checkpoint
  declares: `speed`'s budget admits it, and `preserve` measures it against its budget like any precision
  change, with the FP32 rows as the drift-free order.
- **The rest of the expert, declared too.** `quantize_e2m1x2`, `quantize_int8`, `exp_declared`, `silu_declared` and
  `swiglu_declared` use IEEE operations only (the exp is FMAs and an exact power-of-two scaling, no library `expf`), so
  an expert's intermediate is quantized to the same codes on either processor. `host_int_device.cuh` has them for CUDA
  and HIP, named `_device`, with `dot_row_int<L>`.
- **Paths.** The scalar oracle; AVX2 (vpmaddubsw) and AVX-512 with BW and VNNI (vpdpbusd), chosen at run time by
  `best_int_isa()`. A host store may keep experts in `Layout::interleaved` (`interleave_row`): a vector lane then
  holds one block, so the block sums need no reduction inside a vector. Every path and layout gives the oracle's bits.
- **Kernels.** crit-labs' `int-block-experts` registry kernel runs the same order on the device
  ([operations.json](operations.json) says how to check one):
  decode rows for 1 to 8 tokens a matrix and a prompt GEMM on int8 tensor cores, the same bits as the CPU rows.
- **Cost.** On one Ryzen 5 9600 (6 threads, WSL2) the interleaved AVX-512 rows read 40-42 GB/s of weights for one token,
  26 GB/s for three and 16-22 GB/s for eight (`host_int_tests` prints its own); the rows layout runs at 30-40, 20 and
  11-16. Beside link copies, the two share host memory's read rate: time both together before setting the split.

## The plugin

As a projections plugin on the CPU (`devices: ["cpu"]`), it multiplies a checkpoint's stored matrices with CPU torch
tensors: `load(tensors, format, role=None)` for `nvfp4` and `int4` (`{weight, scale, global_scale, input_scale}`, E4M3
scale per 16) and `bf16`; `matmul(x, matrix, out_dtype=None)`. The option `activations` picks what an NVFP4 matrix
multiplies: `fp32` (weight only, the 16-lane FP32 order above), `e2m1x2` (W4A4 at the matrix's input_scale) or `int8`
(W4A8 per 32), the last two in the integer block order; INT4 matrices run W4A8 or W4A4. Activations are quantized per
row and every row is computed alone, so a row's bits never depend on the rows beside it at any row count (the facts say
`row_invariance` `prompt_rows`); 16-bit matrices go through torch a row at a time. The rows run natively: `native.py`
builds `host_capi.cpp` with the host rows into a shared library when the plugin loads (each ISA file with its own
flags, cached by a hash of the compiler, the flags and every source), and `rows.py` calls it through ctypes, splitting
a matrix's rows over threads. Without a C++ compiler the same orders run in torch (`declared.py`, option `native: off`
forces it), with the same bits: torch has no FMA, so `declared.fma32` rounds a * b + c once (the product exact in FP64,
TwoSum's exact error, the FP64 sum rounded to FP32 and moved one step where it lies exactly between two FP32 values).
`declared.py` runs on any torch device, so it is also an exact device-side twin of the CPU rows, slow.

## What the code supplies

| File | Supplies |
| --- | --- |
| [`host_nvfp4.h`](code/host_nvfp4.h), [`host_nvfp4.cpp`](code/host_nvfp4.cpp) | `Order`, `Format` and its tables, `MatrixView`, `ScaleDecoder` with `e4m3_scales` and `e8m0_scales`, `dot_rows` for 1 to 8 tokens with a mask; the scalar oracle |
| [`host_isa.h`](code/host_isa.h), [`host_isa.cpp`](code/host_isa.cpp) | `Isa`, `supported`, `best_isa`: cpuid/xgetbv detection shared with other host paths that compile AVX2 and AVX-512 files (`crit_inference_isa_sources`) |
| [`host_nvfp4_avx2.cpp`](code/host_nvfp4_avx2.cpp), [`host_nvfp4_avx512.cpp`](code/host_nvfp4_avx512.cpp) | The two SIMD paths, bitwise equal to the oracle |
| [`host_nvfp4_portable.inc`](code/host_nvfp4_portable.inc) | The portable rows, compiled for the baseline target and again with AVX2 and AVX-512 |
| [`host_nvfp4_device.cuh`](code/host_nvfp4_device.cuh) | `dot_row_halving_format` and `dot_row_block_format<lanes, bits, block>` for any format, and the NVFP4 `dot_row_halving` and `dot_row_block`: the device side of both products with the halving reduction, one source for CUDA and HIP |
| [`host_nvfp4_device.metal`](code/host_nvfp4_device.metal) | The same functions in the Metal Shading Language, for a library built with fast math off |
| [`host_int.h`](code/host_int.h), [`host_int.cpp`](code/host_int.cpp), [`host_int_avx2.cpp`](code/host_int_avx2.cpp), [`host_int_avx512.cpp`](code/host_int_avx512.cpp) | The integer block order: `IntMatrix` (rows or interleaved), `IntActs`, `dot_rows_int`, `interleave_row`, the declared quantizers, exp, SiLU and SwiGLU; scalar oracle, AVX2 and AVX-512 VNNI paths, bitwise equal |
| [`host_int_device.cuh`](code/host_int_device.cuh) | `dot_row_int<L>` and the declared functions on CUDA and HIP |
| [`implementation.py`](code/implementation.py) | Exact FP32 reference of the declared order (one rounding per operation), standard library only |
| [`host_capi.cpp`](code/host_capi.cpp) | A C interface over the rows, quantizers, SwiGLU and the route-order combine, for ctypes |
| [`native.py`](code/native.py) | Builds the host rows (and a technique's own C++ beside them) into a shared library at load; `NativeUnavailable` without a compiler |
| [`rows.py`](code/rows.py) | `HostRows`: the C interface over CPU torch tensors, rows split over threads |
| [`declared.py`](code/declared.py) | The integer block order, the FP32 order, the quantizers, `exp_declared`/SwiGLU and the combine in torch on any device, bit for bit the C++ (`fma32`) |

## What the engine supplies

- **The expert's composition**: which projections (gate, up, down), the activation between them and the
  routing weight. The activation must also match bitwise. Exact ones (ReLU, ReLU²) do so trivially.
  Transcendental ones (SiLU, GELU) need the same `exp` and division sequence on both processors, because
  `expf` differs between CUDA and C libraries in the last bits. CUDA's `expf` finishes with the hardware's
  approximate `ex2`, which no host code reproduces, so an exact split needs a device kernel whose `exp` is
  built only from IEEE operations (FMA, multiply, add, exponent scaling) that the host can repeat. The way
  contributions are summed also has to match: if the host returns a sum over its experts that the device adds
  to its own sum, the order depends on the split. Returning per-expert rows that the device sums in its plan
  order keeps it fixed.
- **A device kernel that declares the same order**: `dot_row_halving_format` or `dot_row_block_format` (CUDA, HIP), their
  Metal versions, or its own kernel written to the same rules and checked against `dot_rows` with the CUDA fixture's
  method. The HIP and Metal versions are written to the order but not run in crit's tests (no AMD or Apple GPU in
  them): check them on the target before relying on the split.
- **Its scale decoder** when the checkpoint's scales are not plain E4M3 bytes, and `post` as the device
  computes it.
- **Threads and the CPU share.** Split rows or experts over a worker pool, for example behind the
  [host-work mailbox](../host-work-mailbox/README.md). Leave the share to measured dispatch with the host
  lane timed on the target machine. Host time per expert grows with its weight bytes until the workers
  saturate host DRAM, and the device pays each miss's transfer at the measured host-to-device rate. Link copies
  read the same host DRAM, so with both paths busy each runs below its rate alone: time the lanes concurrently
  before setting the share. Past the
  share where host time exceeds the transfer it replaces, decode slows down. That point depends on cores,
  memory bandwidth, the link and the expert size, so no share carries over between machines.

## Worked example

Run from `code/`. The same bytes under another reduction, another lane count, or a double-precision dot
round differently. A scale hook that stores halved scales against a doubled `post` keeps every bit.

```python
import random
from implementation import SEQUENTIAL, bits, codes_of, dot_row, f32, standard_scales
from nvfp4 import e2m1_value

rng = random.Random(7)
cols = 256
rows = [(bytes(rng.randrange(256) for _ in range(cols // 2)),
         bytes(rng.randrange(0x20, 0x60) for _ in range(cols // 16))) for _ in range(24)]
x = [f32(rng.gauss(0.0, 1.0)) for _ in range(cols)]

# The order is part of the contract: other reductions or lane counts round differently.
oracle = [dot_row(p, standard_scales(s), x) for p, s in rows]
sequential = [dot_row(p, standard_scales(s), x, reduce=SEQUENTIAL) for p, s in rows]
wide = [dot_row(p, standard_scales(s), x, lanes=32) for p, s in rows]
assert any(bits(a) != bits(b) for a, b in zip(oracle, sequential))
assert any(bits(a) != bits(b) for a, b in zip(oracle, wide))

# A scale hook may store scales differently, not change the arithmetic: halved scales with a doubled
# post-multiplier are exact powers of two away, so every bit survives.
halved = [dot_row(p, [v / 2 for v in standard_scales(s)], x, post=2.0) for p, s in rows]
assert [bits(v) for v in halved] == [bits(v) for v in oracle]

# A double-precision dot is close in value but is not the declared order, so it is not the device's bits.
exact = [f32(sum(e2m1_value(c) * v * xv for c, v, xv in
                 zip(codes_of(p, cols), [sv for sv in standard_scales(s) for _ in range(16)], x)))
         for p, s in rows]
assert all(abs(a - b) <= 1e-4 * max(1.0, abs(b)) for a, b in zip(oracle, exact))
assert any(bits(a) != bits(b) for a, b in zip(oracle, exact))
```

## Adapting it

Preserve:

- one rounding per declared operation, with no contraction the order does not name;
- the lane assignment, the block order and the reduction pairing;
- `post` as the device applies it;
- scales decoded to the device's exact floats;
- the same activation arithmetic on both processors;
- a measured CPU share.

You can change row interleaving, threading, the token batch, how scales are cached, the instruction set (a
NEON path must match the oracle the same way), and the order itself, as long as the device declares the same
one.

Adding an order (another lane count, a pairwise first step) means adding it to the oracle, every SIMD path,
the device function and the fixtures, with bitwise checks against the oracle.

## Costs and validation

Per token, a row reads its codes once, so throughput is bound by decoding and FMAs until the workers saturate
DRAM. For scale only: on one Ryzen 5 9600 (Windows, single thread, 4096-column rows) the fixture measured
about 8 GB/s of weights for one token on AVX-512 (GCC under WSL2, 1024-column rows), 3.5 GB/s on AVX2 and
under 0.1 GB/s on the scalar oracle,
which is a reference, not a fallback for speed.

The [host fixture](../../tests/inference/host_nvfp4_tests.cpp) checks the AVX2 and AVX-512 paths bitwise
against the oracle, for both lane counts, both reductions, the standard decoder and an engine-style hook, and
token masks up to 8 tokens. Shapes have 64 to 4096 columns. It also checks that splits over 2, 3 and 7
threads change nothing, that values agree with `crit::nvfp4::project` within FP32 rounding, that the two
reductions and the two lane counts give different bits on most rows, and that every path gives the Python
reference's known answers for inputs from a portable generator. It passes with MSVC on Windows and GCC on
Linux.

The [CUDA fixture](../../tests/inference/host_nvfp4_cuda_tests.cu) checks 1408 `scaled_elements` rows and 1152
`block_dot` rows, 512 to 4096 columns, on an RTX 5090 against the oracle and every host path: the device
functions matched all of them bitwise, while a plainly written device loop, which nvcc may contract, differed
on most. In one MoE engine on that machine, using these rows for its CPU-computed misses in place of its own
AVX-512 code (the same declared order) gave token-identical output in every run; decode was within 3% and
short-prompt first tokens about 16% slower than the engine's hand-tuned rows. In a second MoE engine, whose kernels use the
`block_dot` order and whose checkpoint stores CSF-encoded scales, `block_dot` rows through a scale hook for
those scales matched the engine's own device row arithmetic on 20,480 rows of real expert weights. That
engine's expert outputs still differ across processors: its SiLU uses CUDA's `expf`, and its host returns a sum
over experts rather than per-expert rows.
