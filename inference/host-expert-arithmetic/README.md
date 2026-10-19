# Host arithmetic for NVFP4 experts

Compute NVFP4 block-scaled expert rows on the CPU with exactly the arithmetic of the device kernel that would
otherwise compute them. A row's dot product is fixed by a declared operation order: lanes, product rounding,
block order, reduction tree and final multiply. A device kernel that declares the same order gives the same
bits, so where an expert ran never changes the model's output. The entry supplies a portable scalar oracle,
AVX2 and AVX-512 paths chosen by CPU feature detection, the matching CUDA device function and a scale-decode
hook for checkpoint-specific scale encodings.

## When to use it

Use it when [measured dispatch](../heterogeneous-expert-dispatch/README.md) puts part of an expert-routed
step's misses on the CPU, and the output must not depend on that split. The split varies with cache state,
host load and request history, so greedy output stops repeating if host and device round differently. One
engine saw decision forks from mixing CPU W4A8 with device W4A4.

Don't use it when every expert fits on the device, or when measured dispatch gives the CPU no share on the
target machine. That happens when host compute time exceeds the transfer it replaces. Don't use it either when
the device kernel's order can't be declared (a library GEMM with an unspecified reduction): either the
device side is changed to a declared order or the split is not exact.

## How it works

**Format.** Weights use crit's NVFP4 layout from
[quantized projections](../quantized-projections/README.md): codes `[rows][cols / 2]`, low nibble the even
element, one 8-bit block scale per 16 values. The E2M1 and E4M3 values come from that entry's codec
(`crit::nvfp4`, `nvfp4.py`), so the format has one definition in crit.

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

**Paths.** `rows_scalar` is the oracle. The AVX2 path (two YMM per 16 lanes) and the AVX-512F path (one ZMM
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

## What the code supplies

| File | Supplies |
| --- | --- |
| [`host_nvfp4.h`](code/host_nvfp4.h), [`host_nvfp4.cpp`](code/host_nvfp4.cpp) | `Order`, `MatrixView`, `ScaleDecoder` and `e4m3_scales`, `Isa`, `supported`, `best_isa`, `dot_rows` for 1 to 8 tokens with a mask; the scalar oracle and CPU feature detection |
| [`host_nvfp4_avx2.cpp`](code/host_nvfp4_avx2.cpp), [`host_nvfp4_avx512.cpp`](code/host_nvfp4_avx512.cpp) | The two SIMD paths, bitwise equal to the oracle |
| [`host_nvfp4_device.cuh`](code/host_nvfp4_device.cuh) | `dot_row_halving<16 or 32>` and `dot_row_block<16 or 32>`, the device side of both products with the halving reduction |
| [`implementation.py`](code/implementation.py) | Exact FP32 reference of the declared order (one rounding per operation), standard library only |

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
- **A device kernel that declares the same order**: `dot_row_halving` or `dot_row_block`, or its own kernel written to the same
  rules and checked against `dot_rows` with the CUDA fixture's method.
- **Its scale decoder** when the checkpoint's scales are not plain E4M3 bytes, and `post` as the device
  computes it.
- **Threads and the CPU share.** Split rows or experts over a worker pool, for example behind the
  [host-work mailbox](../host-work-mailbox/README.md). Leave the share to measured dispatch with the host
  lane timed on the target machine. Host time per expert grows with its weight bytes until the workers
  saturate host DRAM, and the device pays each miss's transfer at the measured host-to-device rate. Past the
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
