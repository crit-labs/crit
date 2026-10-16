# Low-bit and row-scaled FP8 projection

Represent weights with signed integer codes and one FP32 scale per group, then project
using exactly that representation. The entry defines one complete small codec, plus a
native E4M3FN/BF16 row codec with a CUDA small-batch projection, a fixed-G32 C++
integer codec and an NVFP4 reference codec. Encoded model formats are interchangeable
only when their exact word, scale and layout semantics match.

## When to use it

Use it to reason about weight bandwidth, packing and numerical identity, or as a host
oracle for an implementation of one of these exact codecs. An existing encoded artifact
needs its own decoder unless its layout and arithmetic match. Quantizing weights reduces
traffic, but decode cost and approximation may outweigh the saving for a workload.

## How it works

For an `R x W` matrix and group width `G` (`W % G == 0`), per row group:

```text
L = 2^(bits-1) - 1                       bits is 4 or 8
s = round_to_FP32(max(abs(group)) / L)    an all-zero group uses s = 1
q_i = clamp(round_ties_to_even(w_i / s), -L, L)
reconstructed_w_i = signed(q_i) * s
```

Codes derive from the represented FP32 scale. Scales that cannot be positive finite
FP32 are rejected, and externally built `QuantizedRows` must already hold represented
FP32 values (use `represented_scale(value)` to round one). The encoder uses `[-7, 7]`
or `[-127, 127]`; the decoder accepts the full signed code space. Rows are flattened
row-major; 8-bit codes take one byte, 4-bit codes put even elements in the low nibble
and odd in the high. Only the last byte of the matrix may be padded. The scale index
of flattened element `i` is `i // G`. The host projection computes
`y_r = sum_c reconstructed_w[r,c] * activation[c]` without activation quantization.

**FP8 rows.** `RowWeights` hold `K` E4M3FN code bytes and one BF16 multiplier per row;
the weight is `decode(code) * decode_bf16(scale)`. Codes `0x7f` and `0xff` (NaN) are
invalid; scales are nonnegative finite BF16, not negative zero, and a zero scale
requires only signed-zero codes. `fp8::quantize` rounds `maxabs/448` to BF16, then
rounds codes to nearest-even E4M3FN with finite saturation (all-zero rows get zero
scales; an underflowed nonzero scale becomes BF16 word `0x0001`).

**G32 integers.** [grouped_int.h](code/grouped_int.h) uses the same signed layout with
`G = 32`. `quant::Rows` validates extents and scales once and borrows owner-held
storage; `project` takes token-major FP32 activations and accumulates in double
without expanding the matrix. `CRIT_GROUPED_INT_AVX2` enables an AVX2 path (select it
only on an AVX2 CPU), with optional 8-bit G32 activation quantization; that changes
model arithmetic and needs its own quality check. This layout is not GGML Q4_0/Q8_0.

**NVFP4.** A value decodes as `e2m1(code) * e4m3(block_scale) / global_scale`: E2M1
codes, one E4M3FN scale per 16 row values and one FP32 divisor per tensor. Byte `i`
holds code `2i` in the low nibble and `2i+1` in the high ("adjacent"). Other layouts
(interleaved half-blocks, for example) have identical byte counts and must be converted
explicitly. Quantization uses `global = 6 * 448 / amax(tensor)`, block scale
`e4m3(amax(block) * global / 6)` and codes `e2m1(x * global / scale)`, ties to even,
saturating at 6 and 448; this is the codec's rule, not a claim of bit equality with any
producer. The projection decodes each 16-value block once per row and reuses it for
every token, so decode cost is per weight, not per weight and token.

## What the code supplies

| File | Supplies |
| --- | --- |
| [implementation.py](code/implementation.py) | `QuantizedRows`, `quantize`, `dequantize`, `project` (optional `backend=` hook) |
| [row_fp8.h](code/row_fp8.h) | BF16/E4M3FN conversion, row validation, double-accumulating oracle |
| [fp8_cuda.h](code/fp8_cuda.h) | `project_cuda`: SIMT projection for 1-8 tokens on SM80+, one warp per output row, FP32 accumulate, no tensor cores |
| [grouped_int.h](code/grouped_int.h) | G32 integer codec and CPU projection |
| [nvfp4.py](code/nvfp4.py), [nvfp4.h](code/nvfp4.h) | NVFP4 codec; FP64 oracle in Python, fixed-order FP32 in C++ |

## What the engine supplies

Treat code bytes, represented scales, geometry and codec version as one immutable
representation, and allocate weights and workspace through crit resource services,
retained through completion. A `backend(weights, activation)` must consume that exact
layout and return one value per row; the reference returns its result unchecked. A
native kernel can decode groups while accumulating; if it also quantizes activations,
it must define their blocks, scales and rounding.

For `project_cuda`, upload validated `RowWeights`, pass BF16 input `[T,K]`, an
exclusive FP32 output `[T,N]` and the stream. `DeviceRows` carries code and scale spans
plus `N,K`. The launch checks extents, alignment and aliases without reading device
data or synchronizing; retain every allocation through stream completion, even after a
launch error. Build with the actual GPU's architecture:

```text
cmake -S inference/quantized-projections/code -B build/fp8 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/fp8 --config Release
```

Native FP4 tensor-core providers belong in [Kernels](../../kernels/README.md), qualified
against the NVFP4 reference.

## Worked example

Run from `code/`. An odd number of 4-bit elements leaves one padded nibble.

```python
from implementation import quantize, dequantize, project

weights = quantize([[-7.0, 7.0, 0.0]], group=3, bits=4)
assert weights.codes == bytes([0x79, 0x00])
assert weights.scales == (1.0,)
assert dequantize(weights) == ((-7.0, 7.0, 0.0),)
assert project(weights, (1.0, 2.0, 3.0)) == (7.0,)
try:
    project(weights, (1.0,))
except ValueError:
    pass
else:
    raise AssertionError("activation geometry mismatch was admitted")
```

## Adapting it

Changing a codebook, group structure, scale precision, zero point or packing order
creates a different codec; do not relabel those bytes as this one. Changing reduction
order or activation encoding changes the arithmetic and must be verified on its own.
Group width, tiling, placement and fusion are engine choices.
[Artifact loading](../artifact-loading/README.md) provides the exact encoded bytes and
[expert dispatch](../heterogeneous-expert-dispatch/README.md) chooses their source;
neither may silently change the representation or apply a routing weight twice. A fused
[normalization](../rmsnorm-bf16/README.md) and projection must make intermediate
rounding explicit.

## Costs and validation

Storage is `ceil(R * W * bits / 8) + 4 * R * W / G` bytes with FP32 scales; smaller
groups trade scale traffic for accuracy. The Python fallback dequantizes the whole
matrix, so it is a formula oracle, not a fast path. Malformed extents, incomplete
groups, nonfinite values and unrepresentable scales are rejected; finite inputs can
still overflow an accumulation.

The [NVFP4 fixture](../../tests/inference/nvfp4_tests.cpp) checks every E2M1 and finite
E4M3 code, ties, saturation, zero blocks, an error bound and batched projection against
an FP64 dot. [Host fixtures](../../tests/test_techniques.py) cover nibble decoding, odd
extents, zero groups, invalid representations and hook dispatch.
[Native host fixtures](../../tests/inference/inference_io_codec_tests.cpp) cover every
finite FP8 code, signed zeros, scales and invalid encodings. The optional
[GPU fixtures](../../tests/inference/cuda/kernels.cpp) compare 320 projection shapes,
including tails, odd widths, every batch size and graph replay, against the double
oracle. None of these qualify a converted model or establish a speedup.
