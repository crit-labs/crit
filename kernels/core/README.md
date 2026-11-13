# Kernel core

The building blocks crit's kernel templates are made of, with no algorithm in them: format decoders, tensor-core
multiply wrappers per GPU generation, loaders and pipelines, reductions (crit's fixed-order rule among them) and
epilogues. A template (a technique, `kernels/techniques/<family>/<id>/`) composes them; a mechanism learned from an engineered
library enters here once and every template that uses it gains; a new GPU generation or backend is a core addition.

- CUDA: header-only, `include/crit_kcore/<area>.cuh`, included as `#include "crit_kcore/<area>.cuh"` (the build adds
  `include/` to every kernel's include path and the core to every kernel's build key).
- Triton: `python/crit_kcore/<area>.py`, imported as `from crit_kcore import <area>` (the build puts `python/` on the
  path when it imports a kernel).
- `tests/`: each block checked against a reference on the device (`<area>_tests.cu` with `test_core.py`, the
  Triton helpers in `test_core_triton.py`).

Every block is one definition: a kernel includes it instead of carrying its own copy, so a fix or a faster
instruction sequence reaches every kernel, and blocks keep their bits (a block that changes what it computes is a new
block, not an edit). Run the tests with `python -m pytest kernels/core/tests` on a GPU: each `<area>_tests.cu` is
built for the device's specific target (sm_120a), its plain target (sm_120) and as compute_80 PTX (run through the
JIT with `__CUDA_ARCH__` 800), so one GPU checks every path; `test_every_arch_builds` compiles each for 80, 86, 89,
90a, 100a and 120a.

## arch (`arch.cuh`)

`CRIT_KCORE_HAS_*` macros: 1 where the architecture being compiled has the instruction, 0 elsewhere and on the host
pass: `CP_ASYNC` (8.0), `FP8_MMA` (8.9: e4m3 mma.sync and cvt), `TMA` and `PDL` (9.0: cp.async.bulk, mbarrier
transaction counts, griddepcontrol), `WGMMA` (sm_90a), `TCGEN05` (sm_100 family specific targets), `E2M1_CVT`
(cvt.rn.f16x2.e2m1x2: family and specific targets from 10.0), `BLOCK_SCALED_MMA` (sm_120a, sm_121a). Guard an
instruction with `#if`, pick a path with `if constexpr`.

## formats (`formats.cuh`, `python/crit_kcore/formats.py`)

Exact decoders of what crit's kernels read, in `crit::kcore`: 16-bit pairs (`pack_bf16x2`, `bf16_lo/hi`,
`to_float`/`from_float`); E4M3 (`e4m3_value`: integer path, NaN codes read +-480, for scales; `e4m3_cvt`: the CUDA
conversion; `e4m3x2_to_half2`; `e4m3x4_to_bf16x2`); E8M0 (`e8m0_value`, `e8m0_bf16x2`); E2M1 (`e2m1_value`,
`e2m1_magnitude`, `e2m1x8_to_half2` with `e2m1_half2_unit()` (1 with the conversion instruction, 2^-14 by bit pattern),
`e2m1x8_to_float`, `e2m1x8_to_bf16x2` by byte permutation on every architecture, `e2m1x2/x8_to_bf16x2_scaled`);
INT4 (`int4_value`, `int4x8_to_bf16x2_scaled`); 4-bit codes as integers for dp4a and the s8 MMA (`four_bit_int`,
`four_bit_int8x4`: INT4 or 2 x E2M1); a byte-to-pair table (`four_bit_pair_table`). Encoders: `e4m3_encode`
(integer, non-negative, saturating; equal to the conversion on every non-negative float, checked exhaustively),
`e4m3_encode_satfinite`, `e4m3x2_encode_satfinite`, `e2m1_code` / `e2m1_code_signed` (nearest, ties to the even
code), `e2m1x2_encode` (one cvt.rn.satfinite.e2m1x2 where it exists, the same codes), crit's NVFP4 quantizer
steps (`nvfp4_block_scale`, `nvfp4_operand`). INT4 with a zero point a group (AWQ, GPTQ;
`tests/formats_int4_tests.cu`): `nibble` (sequential packing: crit's rows, GPTQ's qweight and qzeros), `awq_nibble` /
`awq_code` / `awq_to_sequential` (AWQ's 0, 2, 4, 6, 1, 3, 5, 7 order), `gptq_zero` (the stored nibble plus GPTQ's +1
offset where the checkpoint uses it), `gptq_group` (act order's g_idx), `f16_value`, `int4_zp_value` and
`int4x8_zp_to_float` ((code - zero) * scale, exact in FP32), `int4x8_zp_to_bf16x2_split` (the value as two BF16
parts, hi + lo exactly, for two BF16 MMAs).
Layouts: `blocked_scale_offset` (cuBLASLt's 128 x 4 tiles), `scale_offset` (row major or blocked),
`blocked_scale_bytes` (host and device).

- Triton (`python/crit_kcore/formats.py`): the weight-format codes CUDA and Triton kernels share (NVFP4 0, FP8 1,
  MXFP8 2, BF16 3, INT4 with a zero point 4) and exact decoders to FP32 (`bf16_values`, `f16_values`,
  `int4_zp_offsets`, `int4_zp_values`), plus `split_bf16` (a value as a BF16 part and the exact rest, for dots on
  values BF16 cannot hold).

## mma (`mma.cuh`)

One wrapper per tensor-core instruction, its fragment layout in the header's comment (`mma_gid`, `mma_tig`,
`mma_c_row`, `mma_c_col`, `quad_transpose` for 16-byte row reads): mma.sync m16n8k16 BF16/FP16 and m16n8k8 BF16 (8.0),
s8 m16n8k16 / k32 (8.0, an exact-sum `_zero` form), E4M3 m16n8k32 (8.9, accumulating and `_zero`), and the sm_120a
block-scaled kinds: `mma_nvf4_m16n8k64` (E2M1, UE4M3 per 16: NVFP4), `mma_mxf4_m16n8k64` (E2M1, UE8M0 per 32) and
`mma_mxf8_e4m3_m16n8k32` (E4M3, UE8M0 per 32), the lane supplying the scales of A row gid + 8 (tig & 1) and of B
column gid. A wrapper the target lacks traps. `tests/mma_tests.cu` packs fragments by the documented layouts and
checks each instruction's exact result.

Warpgroup and 5th-generation MMA, built and assembled for their targets but not run here (no sm_90 or sm_100 device):
`wgmma_desc`, `wgmma_fence` / `wgmma_commit` / `wgmma_wait<N>` and `Wgmma<N>::bf16<TA, TB>` (m64nNk16, N 8 to 256,
sm_90a); `umma_desc`, `umma_idesc_f16`, `tcgen05_alloc` / `dealloc` / `relinquish`, the fences, `tcgen05_mma_f16`
(kind::f16, cta_group 1), `tcgen05_commit` and `tcgen05_ld_32x32b_x8` / `tcgen05_wait_ld` (sm_100a). Descriptor
layouts after CUTLASS (BSD-3-Clause). A kernel that adopts them should add a device test on that hardware.

## load (`load.cuh`, `python/crit_kcore/load.py`)

Moves bytes, changes no value: `smem_u32`; `ldg16` (zeros when invalid), `ld_stream16` / `ld_stream8`
(L1::no_allocate) and `ld16<STREAM>` / `ld8<STREAM>` for a STREAM knob; cp.async (`cp_async16`: .cg with zero fill,
`cp_async_ca<4|8|16>`, `cp_async_commit`, `cp_async_wait<N>`) and `CpAsyncRing<STAGES>` (the multistage ring:
`slot`, `commit`, `wait`); `ldmatrix_x1/x2/x4` and `_trans`; the shared-memory swizzles (`swizzle<B, M, S>`,
`swizzle_128b/64b/32b`, the TMA's modes); mbarriers (`mbarrier_init`, `fence_barrier_init`, `mbarrier_arrive`,
`mbarrier_arrive_expect_tx`, `mbarrier_wait`), the TMA (`tma_load_1d`, `tma_load_2d`, `tma_store_2d` with
`fence_proxy_async`, `tma_store_commit` / `tma_store_wait`, `tma_prefetch_map`) and the host's `make_tensor_map_2d`
(cuTensorMapEncodeTiled through the runtime's entry point); programmatic dependent launch (`griddep_wait`,
`griddep_launch_dependents`, host `launch_programmatic`). Triton: `grouped_tile` (the grouped program order of an
output's tiles, for L2 reuse). Forms after CUTLASS (BSD-3-Clause) and FlashInfer (Apache-2.0), written from the PTX
ISA.

## reduce (`reduce.cuh`, `python/crit_kcore/reduce.py`)

One order per reduction, fixed by its shape: `warp_sum` / `warp_max` / `warp_min` (xor butterflies over WIDTH lanes),
`warp_max_from<FROM>` (across the higher lane bits only, such as an MMA column's 8 lanes), `tree_sum_down<WIDTH>`
(the halving tree by shuffle-down, each add rounded), `kLaneOrder16` + `fold16` (the same 16-lane tree folded by one
thread as lanes finish, the same bits), `block_sum` (warps in order), `block_scan<MAX>` (inclusive sum or max),
`nanmax`, the selection keys (`order_key`, `kNanKey`, `kPastKey`) and `warp_argmax` (larger key, then lower index),
and `softmax_merge` (a partial (max, sum) into a running one, -inf safe). Triton: `tree_sum` / `tree_sum_rows` (the
pairwise tree of adjacent elements: the same bits for every layout and warp count, unlike tl.sum) and
`sum_partials` (partials added in part order).

## epilogue (`epilogue.cuh`, `python/crit_kcore/epilogue.py`)

`exp_declared` and `swiglu_declared` (crit's declared functions: every step rounded, the same bits on every device
and on the CPU tiers), `sigmoid` / `silu`, `store` (by type or by a runtime `Out` kind) and `store_c_fragment` (an
m16n8 accumulator into a row-major matrix, 16-bit pairs as one store, cut at the edges). Triton: `sigmoid` and
`act(g, ACT)` (silu, gelu, gelu_tanh, sigmoid, identity).

## scan (`scan_columns.cuh`, `scan_stage.cuh`, `scan_chunk.cuh`)

Column scans (a linear recurrence token after token with the state in registers, the same expression for every
column, so a row's bits never depend on the rows beside it): `decay_exact` (a token's decay as FP64 exp rounded once),
`delta_update` (the gated-delta update expression), `column_sum4` / `column_sum8` / `column_sum<N>` (a lane's partial
dot products reduced with the 32-lane butterfly's bits). Chunked scans (the WY / UT-transform form: a parallel pass
builds each chunk's state-independent matrices, a sequential pass carries the state across chunks on tensor cores):
`unit_lower_inverse_column` (a column of (I + A)^-1 a thread, FP32 forward substitution), `f16_a_of_accumulators` (MMA
accumulators reused as the next product's FP16 A operand, no data moved between lanes) and `TransposedState` (a warp's
slice of the state, transposed, kept in accumulators).

- Staged state tiles (`scan_stage.cuh`): `stage_columns` copies a slot of 32 value columns (32 R key rows) of a
  [dk][dv] state into shared memory by cp.async, whole 128-byte rows, chunks swizzled by the reading lane
  (`staged_offset`), and `read_columns` gives each lane the values a direct load would (zeros past dk and dv), so a
  column scan can keep several slots in flight without registers (after FlashInfer's gated-delta decode, Apache-2.0).

## attention_softmax (`python/crit_kcore/attention_softmax.py`)

The online softmax a flash-style attention loop is made of, with no attention algorithm in it (which positions a
program reads and in what order are the kernel's): `step` (a block of scores into the running maximum, -inf safe,
natural or base-2 exponents with the scale folded in), `row_sum` (a block's row sums on the tensor cores, a dot with a
column of ones, so the order is the MMA's k-order whatever the tile, or `tl.sum` in FP32 where the order may follow
the layout), `fold` (partial states from different key ranges combined in a fixed order), `sink` (a per-head sink
logit folded into a row's final state as one more position with no value), `finish` and `lse` (the
base-2 log-sum-exp turned back to natural log by one FMA, the same bits for every caller). After FlashAttention-2 and
FlashInfer (Apache-2.0).

- CUDA (`include/crit_kcore/attention_softmax.cuh`, tests `tests/attention_tests.cu`): the same online softmax for a
  warp that owns 16 query rows of mma.sync m16n8 accumulators, every step written out so a row's bits never depend on
  the rows beside it: `ex2`, `softmax_step_m16` (a block of base-2 scores into the running maximum and the lane's
  partial sums, -inf safe, alpha exactly 1 where the maximum stays), `softmax_row_total` (the quad's parts once at the
  end), `probabilities_to_a` (two accumulators as the next m16n8k16 A operand), `softmax_fold2` / `fold_output` (a
  partial state folded in order; an empty one changes no bit), `softmax_sink2` (a sink logit folded last),
  `softmax_finish` and `softmax_lse2`.

## exl3 (`exl3.cuh`)

The EXL3 trellis format's decoder: `exl3_value<CB>` (a 16-bit state's FP16 value in the 3INST, MCG or MUL1
codebook: integer arithmetic and one FP16 rounding), `exl3_state` (position p's 16-bit window of a tile's circular
bitstream, most significant bit first, at a whole or half-integer bitrate), `exl3_words` and `exl3_row` /
`exl3_col` (the fragment order that places a lane's eight positions in the 16 x 16 tile). `tests/exl3_tests.cu`
checks every state of each codebook, every position's window at every bitrate and the order against host
references; exl3-rows' tests check the decoded tiles against the format's own decoder bit for bit.

## gguf (`gguf.cuh`)

GGUF block formats' decoders, in `crit::kcore::gguf`: `subblock<F>(block, ib, q, s0, s1, m)` gives a block's 32-value
sub-block `ib` as int8 codes four a word (what `dp4a` and the s8 MMA multiply) and its FP32 scales (the stored FP16
scale times the sub-block's integer scale, rounded once; `s1` for values 16 to 31 with 16-value scales, `m` for a
minimum, the negated offset of Q4_1 and Q5_1, and with Q2_K's minimum a 16 values `m0` and `m1`), so `value()` =
`fsub(fmul(s, q), m)` is the stored value bit for bit; IQ1's offset of an eighth folds into the codes (8 g +- 1 over
d l / 8, exact); `block_bytes`, `block_values`, `two_scales`, `has_min`, `two_mins` and `supported` by GGUF type id.
Formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4 (their own IQ grids, in `gguf_grids.cuh` beside IQ3_S's, and IQ4 values). gguf-rows' tests check every format's decoded values against the format's own reference
decoder bit for bit. With it, `reduce.cuh`'s `lane_at32` and `fold32`: the 32-lane halving tree (the warp's xor
butterfly) folded by one thread from finished lanes as they come, a binary counter of six partial sums, the bits of the
butterfly, for a GEMM that keeps a warp-wide declared order.

## blockscaled (`blockscaled_order.cuh`, `blockscaled_mma.cuh`, `blockscaled_stage.cuh`)

crit's declared order for a block-scaled dot (each block summed exactly or on the tensor cores, one FP32 step a block
on lane b % 16, the 16 lanes reduced by the halving tree), in a form a GEMM can hold: `lane_at` and `GroupVisits<G,
RUN>` visit the blocks lane by lane in the order the tree consumes the lanes, so an output keeps one running lane and
four partial sums (folded with reduce's `fold16`, the bits of 16 lanes kept to the end). `block_sum_e4m3` (a
zero-accumulator E4M3 MMA in crit's fragment order), `block_sum_s8` + `biased_sum` (the s8 MMA on a biased
accumulator: the exact block sum is one FP32 subtraction), `step` (the declared fma), `nibbles_to_s8x4` (four E2M1 x 2
or INT4 codes to int8 in three byte permutes) and `pair_slot` (rows g and g + 8 side by side for 8-byte scale loads).

## tile (`tile.cuh`)

K-major byte tiles for GEMMs that accumulate in the MMA, from CUTLASS's multistage mainloop (BSD-3-Clause, written from
the PTX ISA): `offset<BK>` (a 32-256-byte row's 16-byte chunks swizzled by the row, conflict-free for ldmatrix),
`KTileCopy` (a CTA's cp.async copies of a tile, one source pointer a thread, zero fill past the rows and K), `ldsm_a` /
`ldsm_b2` / `ldsm_b1` (a k-step of 32 bytes as the 8-bit m16n8k32 or 4-bit m16n8k64 A fragment, or two or one B
fragments), `copy_rows` + `row_copy_width` (a stage's scale bytes by the widest aligned cp.async), `tile_of` (a
grouped CTA order) and `store_tile` + `store_chunk` (an accumulator tile through shared memory as whole rows); tests
`tests/tile_tests.cu`.

## gemm_epilogue, mma_fp8 (`gemm_epilogue.cuh`, `mma_fp8.cuh`)

For GEMMs whose scales do not change along K. `gemm_epilogue.cuh` (`crit::kcore::gemm_out`): an accumulator tile
(tile.cuh's warp layout) times a rank-1 factor (row scale x column scale, read into registers by the caller), rounded
once to FP32, BF16 or FP16: `store_scaled` from registers (pairs), `store_scaled_staged` + `staged_chunk` through
shared memory as whole rows in 16-byte stores. `mma_fp8.cuh`: `mma_e4m3_m16n8k32_fast`, the accumulating E4M3
m16n8k32 at the target's best rate: on sm_120a the block-scaled kind::mxf8f6f4 with unit E8M0 scales (`kE8M0Ones`),
twice the plain instruction's rate on an RTX 5090 (120 G against 61 G MMAs a second from registers), elsewhere the
plain e4m3 MMA; `mma_e4m3_m16n8k32_fast_zero`, its form from a zero accumulator. Used by fp8-block-rows'
gemm-native-channel and gemm-native-blocks and fixed-tile-gemm's cuda-mm (their tests check them against the FP64
product).

## multiply (`multiply.cuh`, `declared_rows.cuh`)

The ways of multiplying a weight row by an activation, apart from how the weights are decoded: the exact FP32 path,
crit's declared integer block order (dp4a or s8 block sums) and its declared FP8 block order (E4M3 block sums),
chosen by `path_of(weight format, activation format, arch)`. `ScaleGrid` carries FP8's 128 x 128 scale grid and
per-row factors as decoder side data. `declared_rows.cuh` holds the declared decode rows on those paths as templates
over the caller's matrix, activation and job records, so a kernel instantiates them instead of keeping its own copy:
decode-gemv's `rows-int-block` and `rows-fp8-block`, which int-block-experts' and fp8-block-rows' `rows` alias.
