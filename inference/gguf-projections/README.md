# GGUF block projections

Projections that read GGUF's quantized weight blocks as they are stored in the file:
Q4_0, Q8_0, Q4_K, Q5_K and Q6_K, the block formats most GGUF checkpoints use for their
large matrices. A host reference decodes every block bit for bit as gguf-py's
`quants.dequantize` does (the format's own reference decoder, which Eval's GGUF path
already trusts), and CUDA kernels decode the same bits in registers for a GEMV over up
to 8 activation rows, or write FP32 copies for prompt GEMMs.

## When to use it

When an engine serves a GGUF checkpoint without re-encoding it: the stored blocks are
read once per decode step, so a projection costs the file's bits per weight (4.5 for
Q4_0 and Q4_K, 5.5 for Q5_K, 6.6 for Q6_K, 8.5 for Q8_0). Not for the IQ and other
codebook formats (a different decoder per format), or when the engine re-encodes the
weights into its own layout (then [quantized-projections](../quantized-projections/README.md)
or [stored-tile-gemm](../stored-tile-gemm/README.md) fit).

## How it works

A `rows x cols` matrix stores row after row, each as `cols / values` consecutive blocks
(GGUF's `ne0` is the row length). Per format, with `f16` scales converted exactly:

| Format | Values | Bytes | Value |
| --- | --- | --- | --- |
| Q8_0 | 32 | 34 | `d * q`, 32 int8 codes after `d` |
| Q4_0 | 32 | 18 | `d * (q - 8)`, element `e < 16` in the low nibble of byte `e`, `e >= 16` in the high nibble of byte `e - 16` |
| Q4_K | 256 | 144 | `(d * sc_j) * q - dmin * m_j` for sub-block `j = e / 32`; 6-bit `sc_j`, `m_j` packed in 12 bytes; nibble `h = j % 2` of byte `32 (j / 2) + e % 32` |
| Q5_K | 256 | 176 | as Q4_K, with bit `j` of byte `e % 32` of the 32 high-bit bytes as the code's fifth bit |
| Q6_K | 256 | 210 | `(d * s_(e / 16)) * (q - 32)`: low nibble from 128 bytes, two high bits from 64 bytes, 16 int8 scales, `d` last |

Every value is computed with gguf-py's operations in its order, each FP32 multiply and
subtract rounded once: on the device with `__fmul_rn` and `__fsub_rn`, so the compiler
cannot fuse `(d * sc) * q - m` into one FMA, which would round differently. The host
reference in Python rounds each double result to FP32 (exact for products of FP32
values, and correctly rounded for their differences).

**GEMV.** A warp owns one weight row. For 256-value formats lane `l` decodes values
`[8l, 8l + 8)` of each block in turn; for 32-value formats four lanes share a block and
a warp step covers eight blocks. Each lane multiplies its values with the activations of
every batch row (float4 loads) and the warp reduces in a fixed order, so results are
deterministic. `dequantize` writes FP32 copies for prompt GEMMs.

## Worked example

From `code/`:

```python
import struct
from gguf_blocks import FORMATS, dequantize, project, test_blocks

block = struct.pack("<e", 0.5) + bytes([0x10] * 16)        # Q4_0: d = 0.5, low nibbles 0, high nibbles 1
assert dequantize("Q4_0", block) == [-4.0] * 16 + [-3.5] * 16
q8 = struct.pack("<e", 0.25) + struct.pack("<32b", *range(-16, 16))
assert dequantize("Q8_0", q8)[:3] == [-4.0, -3.75, -3.5]
data = test_blocks("Q4_K", 2, 7)                            # deterministic blocks the native tests generate too
assert len(dequantize("Q4_K", data)) == 512 and FORMATS["Q4_K"] == (256, 144)
y = project("Q4_K", data, 2, [1.0] * 256)                   # two rows of one block each
assert abs(y[0] - sum(dequantize("Q4_K", data)[:256])) < 1e-9
```

The CUDA interface ([gguf_cuda.h](code/gguf_cuda.h), [gguf_cuda.cu](code/gguf_cuda.cu),
[gguf_blocks.h](code/gguf_blocks.h)):

```cpp
namespace gb = crit::gguf_blocks;
gb::gemv(gb::Kind::q4_k, blocks, rows, cols, x, batch, y, stream);   // x [batch][cols], y [batch][rows], FP32
gb::dequantize(gb::Kind::q6_k, blocks, block_count, out, stream);   // FP32 values for a prompt GEMM
```

## Invariants

- One FP32 rounding per multiply and subtract, in gguf-py's order; never let the compiler
  contract them into FMAs.
- Rows of blocks exactly as the file stores them; a GEMV reads every byte once.
- The GEMV's per-row reduction order is fixed, so equal inputs give equal bits.

## Costs and validation

`test_reusable_techniques.py` checks the Python reference against SHA-256 digests of
gguf 0.19.0's `quants.dequantize` output on deterministic blocks of each format (and
against gguf-py itself when it is installed). `tests/inference/gguf_projections_tests.cpp`
checks the C++ host decode against the same digests, the device decode against the host
bit for bit, and the GEMV on a 37 x 1024 matrix for 1, 3 and 8 activation rows against
an FP64 product of the decoded weights (within 1e-5 of the summed magnitudes).

Measured on an RTX 5090 (1,438 GB/s from `crit eval probe`), weights cycling through
copies that together exceed its 96 MB L2 cache several times, GEMV time and bytes of
blocks read per second for a 14336 x 4096 matrix: one activation row took 39 us
(Q4_0, 841 GB/s), 45 us (Q8_0, 1,394 GB/s), 37 us (Q4_K, 885 GB/s), 47 us (Q5_K,
855 GB/s) and 47 us (Q6_K, 1,023 GB/s); eight rows took 97 to 126 us (265 to 645 GB/s),
where the per-lane activation loop, not the weight bytes, limits the kernel. A 4096 x
4096 matrix took 15 to 19 us for one row. These illustrate one device; wider batches
want an activation tile in shared memory or a tensor-core kernel.
