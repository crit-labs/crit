# Stored-tile low-precision GEMM

Lay block-scaled low-precision weights (FP8 with row scales, NVFP4 with 16-value E4M3
group scales) out once, value-preserving, in tiles whose byte order is a tensor-core
fragment, and have both the decode GEMV and the prompt GEMM read those bytes directly.
In one measured engine (NVFP4 MLP matrices of 17408 x 5120 at 1,536 tokens) the stored-tile
GEMM ran at 670 TFLOPS against 550 for the library GEMM with re-layout, and a short
prompt's time to first token fell from 58 to 38 ms because prompts under 128 tokens no
longer ran in FP16.

## When to use it

Use it when a library GEMM would need its own weight layout and scale swizzle: a second
copy doubles the largest allocation, and per-call re-layout costs more than the
low-precision math saves on short prompts. Skip it when a second copy is affordable or
the device lacks a matching MMA. The kernels need sm_89+ for FP8 and sm_120a / sm_121a
for NVFP4; the layout method applies to any MMA.

## How it works

**One tile layout.** Lane `l = 4g + t` of a warp owns 16 bytes of each tile. For FP8
tiles of 16 rows x 32 columns, for each 16-column half `s` the lane holds
`(row g, cols 16s+2t, +1)`, `(row g+8, same)`, `(row g, cols +8, +9)`,
`(row g+8, same)`. NVFP4 tiles of 16 x 64 hold the same pattern per 16-column scale
group as packed E2M1 bytes (low nibble first), plus a 64-byte scale tile
`[g][row g: groups 0..3][row g+8: groups 0..3]`. Each lane's bytes are its A fragment
of `mma.m16n8k16` with FP16 operands, which a decode GEMV converts and multiplies.
Repacking only moves bytes.

**FP8: permute k inside the tile.** `mma.m16n8k32 e4m3` wants lane `t` to hold
`k = 4t .. 4t+3`. A dot product does not care about `k` order if both operands agree,
so within each half MMA slot `4t + i` holds column `2t + (i < 2 ? i : 6 + i)`, which is
what the lane already holds: `a0 = byte_perm(w0, w1, 0x5410)` (row g),
`a1 = byte_perm(w0, w1, 0x7632)` (row g+8), and `a2`/`a3` likewise from the second
half. Activations are quantized per token (E4M3, scale `amax / 448`) and packed in the
same slot order.

**NVFP4: lanes exchange values.** `mma.m16n8k64 kind::mxf4nvf4 scale_vec::4X` applies
one scale per 16 values of `k` and puts each scale group of a row inside one lane pair
(same `t >> 1`), while the tile spreads every group over all four lanes of a quad. No
permutation keeps groups intact, so lane `t` keeps its words for groups `t >> 1` and
`(t >> 1) + 2` and swaps the other two with lane `t ^ 2` (one `__shfl_xor` per register
pair). Slot `(lane t, register pair r, nibble i)` then holds column
`16s + 2t' + (i&3 < 2 ? i&3 : 6 + (i&3))` with `s = (t >> 1) + 2r` and
`t' = (i < 4 ? t : t ^ 2)`: `a0 = byte_perm(own, got, 0x6420)`,
`a1 = byte_perm(own, got, 0x7531)`. Activations use the checkpoint's NVFP4 input
quantization (static global scale `gx`, scale `e4m3(amax gx / 6)` per 16 values) in
that slot order. The result is scaled by `1 / (weight_global_scale * gx)`.

**Measure the fragment layout.** `fragment_probe.cu` finds the block-scaled MMA's scale
mapping by setting one scale byte to 2.0 at a time. On the tested device, row `g`'s
scales come from lane `4g`, row `g+8`'s from lane `4g+1`, column `g`'s from lane `4g`,
and byte `j` covers `k = 16j .. 16j+15`.

**Pipeline.** A block of 8 warps covers 8 x 16 x `RT` rows and `8 NT` tokens. In the FP8
kernel one activation fragment from shared memory feeds `RT` MMAs; the NVFP4 kernel loops
over row tiles outside token groups instead, reading each activation fragment once per row
tile so that only one row tile's weight fragment is live (same bits either way). Weight tiles stream through a
`PF`-deep register ring (L1 no-allocate), activations through a 3-stage `cp.async`
ring. When the grid is too small, K is split into up to 8 parts whose FP32 partials a
finish kernel sums in a fixed order.

## What the code supplies

- [stored_tile_gemm.h](code/stored_tile_gemm.h) / [stored_tile_gemm.cu](code/stored_tile_gemm.cu):
  `enqueue_repack_fp8`, `enqueue_repack_fp4` (row-major to tiles), `plan(fp4, N, K, T, ...)`
  returning a `Plan` (tokens per block, row tiles per warp, split, padding and scratch
  sizes), `enqueue_fp8(Fp8Weights, x, T, y, ldo, add, plan, scratch, stream)`,
  `enqueue_fp4(Fp4Weights, ...)` and `fp4_mma_compiled()`. BF16 activations `[T][K]`
  in, FP32 `[T][ldo]` out, optionally added to `y`.
- [fragment_probe.cu](code/fragment_probe.cu): standalone layout probe, exit 0 on match.
- [layout.py](code/layout.py): host model of tiles, register assembly and both MMAs
  (`check_fp8`, `check_fp4`).

## What the engine supplies

The stored tensors, the decode GEMV over the tiles, fusion, scratch buffers and `plan`
parameters. Repack each matrix once at load; the tiles replace the row-major copy. FP8 row scales stay
as they are; NVFP4 scales move into scale tiles. Size scratch from `Plan` for the largest
prompt chunk: `activation_bytes`, `token_scale_count` floats (FP8) and `partial_floats`.
A plan serves launches of its own format, `N` and `K` with 1 to its `T` tokens; a launch
with more tokens, another `N` or `K`, or the other format's plan returns
`cudaErrorInvalidValue` instead of writing past the scratch. Left at its default
(`min_blocks = 0`), `plan` splits K until the grid holds two blocks per SM of the current
device; `wide_tokens = 256` was measured on one sm_120 device and is the engine's to
measure. The split sets the accumulation order, so results can differ between devices.
Calls are asynchronous and borrow every buffer until the stream reaches them; `add`
fuses a residual add. `N` must be a multiple of 128 (256
with two row tiles per warp) and `K` of 32 (FP8) or 64 (NVFP4); other shapes return
`cudaErrorInvalidValue`.

Activation quantization is part of the arithmetic (W8A8 per-token for FP8, the
checkpoint's static input scales for NVFP4), so measure quality for the whole model with
Eval's [full-vocabulary comparison](../../docs/eval/MEASUREMENTS.md#full-vocabulary-comparison).

## Worked example

The host model assembles registers as the kernels do; a wrong selector or exchange
partner breaks the equality.

```python
import layout

print("fp8 tile product error:", layout.check_fp8(seed=0))
print("nvfp4 tile product error:", layout.check_fp4(seed=0))
slots = layout.fp4_activation_columns()
assert sorted(slots.values()) == list(range(64))  # every column held exactly once per token
assert layout.check_fp8(seed=1) == 0.0 and layout.check_fp4(seed=1) == 0.0
assert layout.check_fp8(seed=0, selectors=(0x3210, 0x7654)) > 0  # plain word copies are the wrong k order
assert layout.check_fp4(seed=0, partner_xor=1) > 0  # exchanging with lane t ^ 1 splits scale groups
print("slot (t=2, pair 1, nibble 5) holds column", layout.fp4_mma_column(2, 1, 5))
```

## Adapting it

Preserve every stored value (repacking never re-encodes), make the activation slot order
equal the weight slot order, keep every scale group within one MMA scale group, and keep
the accumulation order fixed per launch shape so repeated prompts give identical
results. For another MMA, define the k order it needs, permute within scale groups and
exchange between lanes where a group spans too many lanes. The block shape (`NT`, `RT`,
`PF`), split threshold and activation quantization (fusable into its producer) are free.
The [quantized-projections](../quantized-projections/README.md) entry covers the codecs.

## Costs and validation

Each GEMM adds an activation packing pass (plus per-token amax for FP8) and, when split,
`SK x T x N` partial floats, which matters for short prompts. Without the instruction the
NVFP4 kernel traps; check `fp4_mma_compiled()`. Wide activation ranges lose precision as
in any W8A8 / W4A4 scheme.

[layout.py](code/layout.py) checks permutation, exchange and activation order with exact
arithmetic. The [CUDA test](../../tests/inference/cuda/stored_tile_tests.cu) compares both
GEMMs with an FP64 reference using the same activation quantization for 1 to 300 tokens,
with and without the split, one or two row tiles per warp, `add` and fewer tokens than
the plan's, and checks that a plan mismatch is refused; largest relative
error was 1.5e-7 (FP8) and 6e-8 (NVFP4) on an sm_120 device.
[fragment_probe.cu](code/fragment_probe.cu) checks the layout on the device. No speed
claim is made for other devices or shapes.
