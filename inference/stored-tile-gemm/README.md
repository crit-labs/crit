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
the device lacks a matching MMA. The W8A8 / W4A4 prompt GEMMs need sm_89+ for FP8 and
sm_120a / sm_121a for NVFP4; the decode GEMV and the BF16-activation prompt GEMMs run both
formats on sm_80+; the layout method applies to any MMA.

## How it works

**One tile layout.** Lane `l = 4g + t` of a warp owns 16 bytes of each tile. For FP8
tiles of 16 rows x 32 columns, for each 16-column half `s` the lane holds
`(row g, cols 16s+2t, +1)`, `(row g+8, same)`, `(row g, cols +8, +9)`,
`(row g+8, same)`. NVFP4 tiles of 16 x 64 hold the same pattern per 16-column scale
group as packed E2M1 bytes (low nibble first), plus a 64-byte scale tile
`[g][row g: groups 0..3][row g+8: groups 0..3]`. Each lane's bytes are its A fragment
of `mma.m16n8k16` with 16-bit operands, which the decode GEMV converts and multiplies.
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

**Decode / verify GEMV.** For 1 to 16 activation rows the GEMV reads the same tiles:
each lane decodes its bytes exactly to BF16 (E4M3; E2M1 times its E4M3 group scale, at
most 6 significant bits) and runs `mma.m16n8k16` BF16 with the activations as they are, FP32
accumulation, and the FP8 row scale (a multiply) or the NVFP4 global scale (a division)
after the reduction. A block owns 32-row pairs; `groups` warps of a pair split K (warp `q`
takes tiles `q, q + groups, ...` in order) and their partials are summed in warp order.
Rows 8 to 15 use a second MMA that does not touch rows 0 to 7, so each row's bits are
independent of `M` and of the other rows: a verify row equals plain decode, which
speculative exactness needs. `plan_gemv` takes the largest `groups` of 4, 8 and 16 whose
grid still fits in one wave of the device's resident warps (SM count times the kernel's
occupancy): narrow matrices split K to fill SMs, wide ones keep long chains. The reduction
order then depends on the device, as a GEMM plan's split does; pass `groups` to fix it.
Below sm_89 E4M3 is decoded in software (the magnitude bits placed in an FP16 and one exact
multiply by 256), and without the sm_100 / sm_120 E2M1 conversion a byte permute indexes an
8-entry table; both give the hardware's values. A gated variant writes `bf16(act(gate) * up)`
from gate and up rows bitwise equal to the plain GEMV's.

**BF16-activation prompt GEMM (W8A16 / W4A16).** A third prompt path for checkpoints whose
quality budget cannot afford activation quantization: the GEMV's exact decode and BF16 MMA
for any number of tokens, with no activation packing. A block of 8 warps covers 256 rows (two
row tiles per warp, so gate and up pair as in the gated launches) by 64 tokens; weights
stream through a two-chunk register ring, activations through a four-stage `cp.async` ring
in shared memory (64 columns per stage, 16-byte chunks XOR-swizzled by token) read with
`ldmatrix`. Each MMA chain covers 512 columns and is folded into an FP32 running sum in K
order, so a long K keeps FP32 accuracy instead of one tensor-core chain over all of it. K is
split like the W8A8 / W4A4 plans (one wave of resident blocks) and the parts are summed in
order. Rows past a 256-row block and an FP8 `K % 64 == 32` tail are handled (`N % 16`,
`K % 32` FP8 / `K % 64` NVFP4); with the software decodes it runs below sm_89.

**Placement.** Rows come in 16-row tiles, so matrices with the same K combine by tile row:
`enqueue_repack_fp8_rows` / `enqueue_repack_fp4_rows` write source tile row `r` to destination
tile row `first + step r`. `first` = rows before / 16 with `step` 1 concatenates (q | k | v);
`first` 0 and 1 with `step` 2 interleave gate and up, and `enqueue_place_row_scales` places FP8
row scales the same way. Each checkpoint tensor is repacked straight into its place.

**Gated MLP epilogue.** With gate and up interleaved, the two row tiles of a warp (`RT` = 2)
are the gate and up rows of the same 16 intermediate rows, so `enqueue_fp8_gated` /
`enqueue_fp4_gated` write `bf16(act(gate) * up)` from the accumulators, with gate and up
scaled exactly as the plain launch scales them and `act` one of SiLU, tanh-GELU or erf-GELU in
FP32. No FP32 `[T][2I]` output exists; a split plan's partials are summed and activated by
the finish kernel. GELU is computed without the cancellation of `1 + tanh` and `1 + erf` for
negative inputs, and divisions use `__fdividef` (2 FP32 ulp, well below BF16).

## What the code supplies

- [stored_tile_gemm.h](code/stored_tile_gemm.h) / [stored_tile_gemm.cu](code/stored_tile_gemm.cu):
  `enqueue_repack_fp8`, `enqueue_repack_fp4` (row-major to tiles), `plan(fp4, N, K, T, ...)`
  returning a `Plan` (tokens per block, row tiles per warp, split, padding and scratch
  sizes), `enqueue_fp8(Fp8Weights, x, T, y, ldo, add, plan, scratch, stream)`,
  `enqueue_fp4(Fp4Weights, ...)` and `fp4_mma_compiled()`. BF16 activations `[T][K]`
  in, FP32 `[T][ldo]` out, optionally added to `y`. Placement: `enqueue_repack_fp8_rows`,
  `enqueue_repack_fp4_rows`, `enqueue_place_row_scales`. Gated MLP:
  `enqueue_fp8_gated(Fp8Weights, x, T, Activation, out, ldo, plan, scratch, stream)` and
  `enqueue_fp4_gated(Fp4Weights, up_weight_global_scale, ...)`, BF16 `[T][ldo]` out.
- [stored_tile_bf16.cu](code/stored_tile_bf16.cu) (declared in the same header): the GEMV,
  `plan_gemv(fp4, N, K, groups)` returning a `GemvPlan`,
  `enqueue_gemv_fp8(Fp8Weights, x, ldx, M, y, ldy, add, plan, stream)`, `enqueue_gemv_fp4` and
  the gated `enqueue_gemv_fp8_gated` / `enqueue_gemv_fp4_gated` for `M` = 1 to 16 BF16 rows;
  and the BF16-activation prompt GEMMs, `plan_a16(fp4, N, K, T, min_blocks)` returning an
  `A16Plan`, `enqueue_fp8_a16(Fp8Weights, x, T, y, ldo, add, plan, partials, stream)`,
  `enqueue_fp4_a16` and the gated `enqueue_fp8_a16_gated` / `enqueue_fp4_a16_gated`.
- [stored_tile_device.cuh](code/stored_tile_device.cuh): device helpers the kernels share
  (exact decodes, the BF16 MMA, tile loads, the gated activation).
- [fragment_probe.cu](code/fragment_probe.cu): standalone layout probe, exit 0 on match.
- [layout.py](code/layout.py): host model of tiles, register assembly and both MMAs
  (`check_fp8`, `check_fp4`).

## What the engine supplies

The stored tensors, other fusions, scratch buffers and plan parameters. Repack each matrix
once at load, straight into its place in a combined matrix; the tiles replace the row-major
copy. FP8 row scales stay as they are (placed like their rows); NVFP4 scales move into scale
tiles. Calls are asynchronous and borrow every buffer until the stream reaches them; `add`
fuses a residual add.

Prompt GEMMs: size scratch from `Plan` for the largest prompt chunk: `activation_bytes`,
`token_scale_count` floats (FP8) and `partial_floats`. A plan serves launches of its own
format, `N` and `K` with 1 to its `T` tokens; a launch with more tokens, another `N` or `K`,
or the other format's plan returns `cudaErrorInvalidValue` instead of writing past the
scratch. A launch packs and computes only the `ceil(T / (8 NT))` token blocks its `T` needs,
and a token's result depends only on the plan and its own row, so a short last chunk costs
what it uses and has the bits it would have had in a full launch (on one sm_120 device, 64
tokens under a 2,048-token plan took 245 instead of 918 us for an 8192 x 8192 FP8 matrix).
Left at its default (`min_blocks = 0`), `plan` splits K only while the split grid still fits
in one wave of resident blocks (the device's SM count times the occupancy of the kernel the
plan launches): the parts fill SMs the unsplit grid leaves idle, and no part starts a second
wave or adds partials beyond it. A split past one wave can still pay when the MMA rate is high
against memory bandwidth, which no device query states; pass `min_blocks` to choose it.
`wide_tokens = 256` was measured on one sm_120 device and is the engine's to measure. The
split sets the accumulation order, so results can differ between devices. `N` must be a
multiple of 128 (256 with two row tiles per warp) and `K` of 32 (FP8) or 64 (NVFP4); other
shapes return `cudaErrorInvalidValue`. A gated launch needs a plan with two row tiles per
warp (`N` = 2 I a multiple of 256); NVFP4 gate and up must share `input_global_scale`, since
the activations are quantized once, and keep their own weight global scales.

BF16-activation GEMMs: make an `A16Plan` per (format, `N`, `K`, largest chunk) at load and
give split plans `partial_floats` floats of scratch; `x` is dense `[T][K]` and 16-byte aligned.
They are not bitwise equal to the GEMV (another K order) and need no activation scales.

Decode GEMV: make one `GemvPlan` per matrix at load and use it for every decode and verify
call, so all rows share one reduction order. `N` must be a multiple of 16 (32 for the gated
variant), `K` of 32 (FP8) or 64 (NVFP4), `M` 1 to 16, and `x` rows 4-byte aligned with an
even stride; the GEMV reads the tiles only and needs no scratch.

Activation quantization is part of the W8A8 / W4A4 arithmetic (per-token for FP8, the
checkpoint's static input scales for NVFP4), so measure quality for the whole model with
Eval's [full-vocabulary comparison](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#full-vocabulary-comparison);
where it exceeds the budget, the BF16-activation GEMMs read the same tiles without it.

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
A GEMV adaptation keeps each row's MMA chain independent of the other rows and of `M` (rows
8 to 15 in their own MMA, no reduction across rows) and uses one plan for decode and verify
of a matrix; its group counts and ring depth are free. A BF16-activation GEMM keeps the
exact decode and a bounded chain length before each FP32 fold; its stages, ring and fold
length are free.
The [quantized-projections](../quantized-projections/README.md) entry covers the codecs.

## Costs and validation

Each GEMM adds an activation packing pass (plus per-token amax for FP8) and, when split,
`SK x T x N` partial floats, which matters for short prompts. Without the instruction the
NVFP4 kernel traps; check `fp4_mma_compiled()`. Wide activation ranges lose precision as
in any W8A8 / W4A4 scheme. The gated launches save the FP32 `[T][2I]` buffer and its
activation pass but compute the activation in the epilogue; on one sm_120 device a gated
28672 x 4096 GEMM at 2,048 tokens took 1,439 us (FP8) and 524 us (NVFP4) against 1,376 and
613 us for the plain GEMM alone and 1,541 and 777 us for the plain GEMM plus a separate
activation kernel.

The GEMV reads each weight byte once per call. On one RTX 5090 (170 SMs) the derived plans
read FP8 at 1,420 GB/s (4096 x 4096, 16 groups) to 1,700 GB/s (128256 x 4096, 4 groups) for
one row and 1,270 to 1,660 GB/s for 16 rows, NVFP4 at 1,170 to 1,690 GB/s for one row and 930
to 1,520 GB/s for 16; crit's row-major FP8 GEMV
([quantized-projections](../quantized-projections/README.md)) read 860 to 1,190 GB/s for
one row and 390 to 510 GB/s for 8 on the same shapes. Against fixed shapes (16 warps up to
N = 8192, else 8), the derived group count was up to 36% faster for narrow NVFP4 matrices
and at most 3% slower (FP8, 16 rows).

On the same device, for 40 prompt GEMMs (FP8 and NVFP4; 4096 x 4096, 5120 x 17408,
14336 x 5120, 28672 x 4096 and 4096 x 14336; 64 to 2,048 tokens, each with its own plan) the
one-wave split took 10,324 us in all against 11,213 us for a split to two blocks per SM and
10,258 us for the best split of each case; it was within 5% of that best in 35 cases (the
two-blocks rule in 15), at most 16% slower than the two-blocks rule (FP8 14336 x 5120, 256
tokens, which gains from a split past one wave) and up to 64% faster (NVFP4 4096 x 4096,
512 tokens).

The BF16-activation GEMMs run at the BF16 tensor-core rate, not the FP8 / FP4 one: on the
same device and shapes, 155-205 TFLOPS from 512 tokens (FP8 and NVFP4 alike), 86-102% of
cuBLAS BF16 x BF16 on a dequantized BF16 copy of the weights (which the tiles avoid) and 1.3
to 1.7x faster than it at 64 tokens, against 233-350 TFLOPS for W8A8 and 540-790 for W4A4.

[layout.py](code/layout.py) checks permutation, exchange and activation order with exact
arithmetic. The [CUDA test](../../tests/inference/cuda/stored_tile_tests.cu) compares both
GEMMs with an FP64 reference using the same activation quantization for 1 to 300 tokens,
with and without the split, one or two row tiles per warp, `add` and fewer tokens than
the plan's; largest relative error was 1.5e-7 (FP8) and 6e-8 (NVFP4) on an sm_120 device.
It also checks that every token of a shorter launch has the bits of the plan's full launch;
that parts repacked into place (three concatenated, gate | up interleaved) give the bytes of
the fused matrix's repack; that the gated launches equal `bf16(act(gate) * up)` of the plain
launch's FP32 rows within one BF16 ulp for all three activations, split and unsplit, and
match FP64 within BF16 rounding; and that plan mismatches are refused. For the GEMV (4, 8
and 16 groups; 37 K tiles, so warps take uneven shares; 17 row tiles, so one pair is half
empty) it checks that row `m` of every launch with `M` = 1 to 16, and of a 16-row launch with
the rows reversed, equals the one-row launch bitwise; repeats and `add` exactly; error
against FP64 at most 1.3e-7 of `sum |w x|`; that every E4M3 code and every E2M1 code times
every E4M3 scale decodes exactly; that the gated GEMV equals `act(gate) * up` of the plain
GEMV's rows within one BF16 ulp and is row invariant; and that invalid launches are refused.
For the BF16-activation GEMMs it checks 1 to 200 tokens against FP64 (at most 5e-7 of
`sum |w x|`, also over K = 16384 unsplit and split), unsplit and split into 8, with a 16-row
block tail, an FP8 32-column K tail and a partial 512-column chain; that shorter launches
give the plan's bits; `add`; the gated launches within one BF16 ulp of the plain rows; and
rejections. A PTX build for sm_80 run on an sm_120 device passes the same GEMV and
BF16-activation GEMM checks through the software decodes.
[fragment_probe.cu](code/fragment_probe.cu) checks the layout on the device. No speed
claim is made for other devices or shapes.
