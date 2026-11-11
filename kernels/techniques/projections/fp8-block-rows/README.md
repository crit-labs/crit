# fp8-block-rows

Matched FP8 projections from [crit](https://github.com/crit-labs/crit)'s quantized-projections technique: E4M3 weights
(MXFP8 with its E8M0 block scales, an FP8 checkpoint's 128 x 128 FP32 block scales, or one scale per row) against E4M3
activations quantized per group, in crit's declared per-output order. Per 32-element block: one e4m3 `mma.sync`
m16n8k32 with a zero accumulator (weights the A operand, tokens B, crit's fragment order), then `lane b % 16 =
fma(block sum, weight scale * activation scale, lane)` with the product rounded, the halving tree over 16 lanes, times
post.

**Why it exists.** A decode kernel shaped for 1-8 tokens and a prompt GEMM usually sum in different orders, so a token's
logits depend on whether it was decoded, verified in a speculative round or prefilled. crit's honesty, request-history
and prompt-cache gates compare exactly that. Here `rows` (decode) and `gemm` (prompt) give the same bits for every
output, and an output depends only on its own row and token, never on the job's other tokens. Decode no longer has to
run the prompt kernel on a handful of rows to stay exact.

## Applicability

| | |
| --- | --- |
| Weight formats | E4M3 codes [rows][cols]; scales FP32 or E8M0 bytes [ceil(rows / scale_rows)][cols / scale_k]: MXFP8 (1, 32), 128 x 128 blocks (128, 128), per row (1, cols); a post factor per matrix |
| Activations | E4M3 codes per element and an FP32 scale per group (a multiple of 32); `quantize` makes them from FP32 or BF16 with crit's declared quantizer: MXFP8's 2^(floor(log2 amax) - 8) or amax / 448, round to nearest even, saturating |
| GPUs | NVIDIA sm_89 and newer (e4m3 `mma.sync`); not HIP |
| Shapes | cols a multiple of 32; decode jobs of 1 to 8 tokens on one matrix; prompt jobs of any length; any rows |
| Numerics | bitwise equal to the declared order's reference kernel on the target GPU (`tests/fp8_block_tests.cu`); within 2e-7 of FP64 relative to the products' magnitude on an RTX 5090. The tensor core's internal block sum is not specified by PTX, so the bits hold on one GPU architecture: check on yours |
| Determinism | same bits on a repeat; an output independent of its job's other tokens and of the kernel that computed it (`rows`, `gemm`, `gemm-staged`; `gemm-native` and `gemm-native-channel` give that up, see below) |
| Outputs | FP32; the prompt GEMMs `gemm_native` and `gemm_native_channel` also BF16 (an FP32 out's value rounded once to nearest even: `out` of dtype bfloat16 selects it) |
| Measured speed | RTX 5090: decode rows read 13.1 MB MXFP8 matrices at 0.85-1.03 TB/s for 1-8 tokens, back to back, about a plain streaming read's rate at that size (1.07-1.13 TB/s); prompt GEMM 79-97 TFLOPS on 512-4096 tokens x 5120 x 2560, without shared-memory staging yet |
| Yardstick | cuBLASLt's MXFP8 GEMM (torch._scaled_mm, E4M3 with blocked E8M0 scales on both operands, FP32 output) on the same bytes, the activations' power-of-two group scales converted to E8M0 once outside the timed call (kernel.json `yardstick`, the sweep's library_us). RTX 5090, `crit kernels sweep`, tuned vs cuBLASLt in us. Decode rows at 1 / 2 / 4 / 8 tokens: 5120 x 2560 14.3 / 14.3 / 14.3 / 14.3 against 22.5 (1.58x cuBLASLt's speed), 12288 x 5120 47.1 / 47.4 / 49.1 / 49.1 against 65.5 / 67.6 / 69.6 / 65.5 (1.33-1.43x). Prompt GEMM at 32 / 128 / 512 / 2048 tokens: 5120 x 2560 51 / 57 / 156 / 537 against 22 / 22 / 27 / 132 (0.17-0.44x), 12288 x 5120 137 / 268 / 743 / 2457 against 51 / 53 / 182 / 557 (0.20-0.37x): the prompt GEMM is 2.3-6x slower than cuBLASLt (whose rows change with the token count), the gap to close |

## Structures and knobs

kernel.json lists them; the default knobs run the kernels as they were before knobs (16 warps and 2 rounds of loads a
decode CTA, 8 warps of 4 n-tiles a GEMM CTA).

| Structure | What | Knobs (none changes an output's order) |
| --- | --- | --- |
| rows | decode rows, 1 to 8 tokens a job | WARPS (4, 8, 16: the 16 lanes over them), AHEAD (1-4 rounds of loads in flight), STREAM (weight codes with L1::no_allocate) |
| gemm | the prompt GEMM, any token count | WARPS (2-16, 16 weight rows each), TILES (1-8 n-tiles of 8 tokens a warp pass), UNROLL (0: the compiler's, 1, 2, 4) |
| gemm-staged | the prompt GEMM with shared-memory staging (`gemm_staged`, the same bits) | BM, BN (32-128 weight rows, 16-64 tokens a CTA), WM, WN (warps along each), GROUP (1, 2 sibling lanes a stage), RUN (1, 2 blocks of each lane a stage), STAGES (the cp.async ring) |
| gemm-native | the prompt GEMM accumulated in the tensor cores (`gemm_native`): NOT the declared order, not row invariant | TM, TN (64-256 tokens, weight rows a CTA), WM, WN (warps along each), BK (64, 128 bytes of K a stage), STAGES (3, 4), GROUP (weight-row tiles a group of the CTA order), EPI (0: stores from registers while the next tile loads, 1: through shared memory) |
| gemm-native-channel | the prompt GEMM of per-channel matrices (`gemm_native_channel`; `gemm_native(mx=False)` takes it for them): NOT the declared order, not row invariant | TM, TN (64-256), WM, WN, BK (64, 128), STAGES (2-4), GROUP, EPI, as gemm-native's |
| gemm-native-blocks | the prompt GEMM of scales that change along K (`gemm_native(mx=False)` for every other table: 128 x 128 FP32 blocks, MXFP8 with activation scales that are not powers of two): NOT the declared order, not row invariant | as gemm-native-channel's; default_rows: 1-512 tokens 64 x 64 tiles, 4 stages |

The kernels dispatch through the build's `crit_tuning.h`, keyed by the call's max_rows and activation width (N, K) and
its token count (rows: the activation rows given, at most 8; gemm and gemm-staged: max_count). Every lane sums its
blocks in increasing order and the tree is fixed whichever warp, pass, stage or unroll computes them, so every
configuration gives the same bits (the tests check settings of every structure against the defaults; gemm-staged's
defaults are its best configuration on the largest prompts swept). `gemm-native` is the exception: its knobs change
which blocks the tensor core sums together, so its bits may change with them and with the token count (kernel.json
`"row_invariant": false` and its `order`).

**`rows`** is an alias of decode-gemv's `rows-fp8-block` (kernel.json `"alias": {"kernel": "decode-gemv", "of": "rows-fp8-block"}`):
this package's `rows()` forwards to decode-gemv's `rows_fp8_block`, built with this kernel's tuning table for `rows` (an old
tuning still applies; without one, decode-gemv's own tuning), and a sweep of `rows` and `select()` follow it there
(kernels/TEMPLATES.md, `alias`). The kernel is the kernel core's declared FP8 rows (`declared_rows.cuh` on
`multiply.cuh`'s declared paths), the operations this kernel's rows ran before: an A/B of every `rows` configuration
through the forwarding against the build before the move gave the same bits (RTX 5090). The C++ `enqueue_rows` stays
for C++ callers and the C++ tests, on the same core template.

## Build, test, tune

```sh
crit kernels build fp8-block-rows                                 # for this GPU, default knobs
crit kernels build fp8-block-rows --arch 89 --arch 90a --arch 100a --arch 120a   # build check
python -m pytest kernels/techniques/projections/fp8-block-rows/tests -q
crit kernels sweep fp8-block-rows --model CHECKPOINT --out TUNING_DIR   # or --shapes FILE, --structure gemm
```

The pytest tests load it with `crit_kernels.build.load("fp8-block-rows")`: the decode rows and the GEMM bit for bit,
the GEMM's repeat, and other knob settings of both with the defaults' bits. The sweep's benches (bench.py) take the
checkpoint's FP8 matrices as MXFP8-scaled E4M3 codes against E4M3 activations with power-of-two group scales, check
each configuration against an FP64 reference and for row invariance, and time decode at 1 to 8 tokens and the GEMM at
32 to 2048 tokens.

## Check it first

Run `tests/fp8_block_tests.cu` on the target (`nvcc -std=c++20 -O2 -arch=sm_<cc> -I fp8_block_cuda -I
../../../core/include tests/fp8_block_tests.cu fp8_block_cuda/fp8_block.cu` in the kernel's folder; outside crit's build
the source takes its default knobs): the decode rows and both GEMMs against the reference kernel bit for bit, over
three scale layouts and shuffled jobs, the FP64 bound, and the quantizer against its host twin. Then
`tests/fp8_block_order_tests.cu` (`nvcc -std=c++20 -O2 -arch=sm_<cc> -I tests/order tests/fp8_block_order_tests.cu`)
checks the order's assumption itself on the same GPU: three implementations of the declared order
(`tests/order/fp8_block_device.cuh`, the reference this project's kernels follow) with rows and tokens moved across
tile positions and companions, bit for bit, and the device quantizer against its host twin (`tests/order/fp8_block.h`,
a copy of crit's quantized-projections host code).

## The staged GEMM

`gemm-staged` is the prompt GEMM rebuilt from crit's kernel core (`kernels/core/include/crit_kcore/blockscaled_*.cuh`
on `mma.cuh` and `load.cuh`), the mechanisms of a multistage tensor-core GEMM (CUTLASS's mainloop, BSD-3-Clause;
written anew, no code copied) in the declared order. The blocks are visited lane by lane in the order the halving tree
takes the lanes (`GroupVisits`, GROUP sibling lanes and RUN blocks of each a visit), so an output holds its running
lanes and the tree's partial sums (5 registers) instead of 16 lanes, and every lane still takes its blocks in
increasing order. Each visit is one stage of a cp.async ring: the blocks' 32 bytes of every weight row and token, the
weight scales' words (E8M0 bytes at any alignment, or FP32) and the token scales; the CTA decodes a stage's weight
scales to FP32 one stage ahead of its MMAs. A warp runs one e4m3 m16n8k32 MMA with a zero accumulator a 16 x 8 tile
and block (`block_sum_e4m3`, crit's fragment order: the same k positions as the reference kernel), then the declared
step. Its outputs are `gemm`'s and `rows`' bits (the bench checks every configuration against the untuned `gemm`, and
tests/fp8_block_tests.cu against the reference kernel); row invariant. `gemm_staged` takes `gemm`'s arguments;
activation rows 16-byte aligned.

RTX 5090, `crit kernels sweep fp8-block-rows --structure gemm-staged` (137 configurations) on the shapes above, cold
L2, tuned kernel against cuBLASLt's MXFP8 GEMM (us), and the register `gemm`'s tuned time for comparison:

| Shape (N x K) | 32 | 128 | 512 | 2,048 tokens |
| --- | --- | --- | --- | --- |
| 5120 x 2560 | 22.5 / 20.5 (0.91), `gemm` 51.2 | 30.7 / 22.5 (0.73), 57.3 | 81.9 / 26.6 (0.33), 155.6 | 303.1 / 131.1 (0.43), 536.6 |
| 12288 x 5120 | 71.6 / 51.2 (0.72), `gemm` 137.2 | 161.1 / 53.2 (0.33), 268.3 | 389.1 / 182.3 (0.47), 743.4 | 1,355.1 / 555.1 (0.41), 2,456.6 |

The staged GEMM is 1.7 to 2.3 times the register GEMM's speed; it reaches 0.8 of cuBLASLt only at 32 tokens. What
holds it back is measured, not the arithmetic: from registers alone, the declared order's block MMA and FP32 step run
58 G MMAs a second (a microbenchmark of `block_sum_e4m3` and `step`, 95% of the e4m3 MMA rate, 61 G); the kernel
reaches 22 G at 2,048 tokens. An output needs 5 registers (its running lane and four partial sums of the tree), so a CTA
holds at most 128 x 64 outputs at two warps a scheduler: 43 multiply-adds a byte of L2 traffic, read as 32-byte pieces
of rows (the lane-major order visits blocks 512 bytes apart), with each block's fragments loaded from shared memory
just before its MMAs. On 2,048 x 5120 x 2560 the MMAs on fragments from shared memory alone take 162 us, the barriers
add 18 and the copies another 95 (303 in the sweep with a cold L2), against cuBLASLt's 131 us with larger tiles. The
remaining levers: fragment double-buffering in registers freed from the producer's state, the weights repacked once
at load in the lane-major visiting order (contiguous reads), and warp-specialized producers (bulk 32-byte TMA copies
measured 10 times slower than cp.async for this gather on an RTX 5090, so not those).

## The native GEMM

`gemm-native` gives up the declared order for the prompt GEMM's speed, with the owner's approval: it accumulates in the
tensor cores the way cuBLASLt and CUTLASS do, so an output is within FP32 accumulation of the exact product but its bits
are not `rows`' or `gemm`'s and may change with the tuning and the job's token count (not row invariant; the decode rows
and the declared GEMMs stay the default, and a plugin that selects this structure says so in its facts). Built from
crit's kernel core (`tile.cuh`: `KTileCopy`, `ldsm_a`, `ldsm_b2`, `copy_rows`, `store_tile`, `tile_of`, on `load.cuh`'s
cp.async and ldmatrix and `mma.cuh`'s block-scaled MMA), with the mechanisms of CUTLASS's multistage GEMMs
(BSD-3-Clause; written anew, no code copied): CTA tiles of TM tokens x TN weight rows (the tokens the MMA's A operand),
a STAGES-deep cp.async ring of BK-byte K-major tiles with 128-byte rows' 16-byte chunks swizzled by the row
(conflict-free ldmatrix), fragments of the next k-step read while the current one's MMAs run and the next stage's
barrier before a stage's last k-step, persistent CTAs (as many as the GPU holds, each taking tiles in a grouped order)
that issue the next tile's first stages before the epilogue. `gemm_native(..., mx=True)` states the call is MXFP8
throughout (activation scales MXFP8's powers of two at group 32, from `quantize(kind=MX)`; every matrix E8M0 per 32 of
one row with cols a multiple of 128): an sm_120a GPU then runs kind::mxf8f6f4 block-scaled m16n8k32 MMAs, the tensor
core applying both E8M0 scales and summing into its FP32 accumulator (a matrix that is not MXFP8 gets NaN outputs).
`mx=False` runs the channel GEMM (scales constant along K) or the blocks GEMM (every other layout), below; a GPU
without block-scaled MMAs runs the generic path for `mx=True`: an e4m3 MMA from a zero accumulator a 32-element block,
the product of the two FP32 scales applied on the CUDA cores.

Numerics: within 1e-5 of the FP64 product relative to the products' magnitude on the block-scaled path (the tensor
core's own sum of a block is not FP32 round to nearest: 3.3e-6 measured at K = 2,560 on an RTX 5090), 2e-6 on the
generic path (6e-7 measured; the blocks GEMM, which runs the block-scaled MMA too, 1e-5); the bench checks every configuration against the FP64 reference,
`tests/test_fp8_block_rows.py` both paths with two jobs, tokens and rows off the tiles.

RTX 5090 (sm_120a), torch 2.14.1, `crit kernels sweep fp8-block-rows --structure gemm-native` (136 configurations, none
dropped) on the MXFP8 matrices of the shapes above, random E4M3 codes with E8M0 scales and power-of-two activation
scales, cold L2; tuned kernel against cuBLASLt's MXFP8 GEMM (FP32 output, timed as a graph replay), us:

| Shape (N x K) | 128 | 512 | 2,048 | 4,096 tokens |
| --- | --- | --- | --- | --- |
| 5120 x 2560 | 14.3 / 22.5 (1.57) | 28.6 / 26.6 (0.93) | 145.4 / 131.1 (0.90) | 276.5 / 249.9 (0.90) |
| 12288 x 5120 | 59.0 / 53.2 (0.90) | 204.8 / 180.3 (0.88) | 620.6 / 553.0 (0.89) | 1,167.4 / 1,054.8 (0.90) |

Picks: 64 x 64 tiles of 128-byte stages at 128 tokens on the small matrix, 128 x 128 (8 warps of 64 x 32) with 3 stages
of 64 bytes elsewhere, 128 x 256 with the shared-memory epilogue at 2,048 tokens and more on the small one. The
block-scaled MMA itself runs at 1,000 TFLOPS on this GPU from registers (122 G m16n8k32 a second); the kernel reaches
390-440 TFLOPS on 2,048 tokens and more, cuBLASLt 430-490: what is left is the mainloop's share of waits on shared
memory with one CTA an SM (the ring needs 52-99 KB). Remaining levers: warp-specialized producers (TMA loads into an
mbarrier ring, as CUTLASS's sm120 kernels do) and a stream-K split for the token counts that leave SMs idle.

## The channel GEMM

`gemm-native-channel` is the prompt GEMM of an FP8 checkpoint's per-channel W8A8 matrices: FP32 weight scales that do
not change along K (scale_k = cols: one a row, a block of scale_rows rows, or the tensor) against activations with
one FP32 scale a token (scales [tokens, 1], quantized-projections' fp8_rows). No scale changes inside a sum, so every
block of K sums in the tensor cores' FP32 accumulator and the scales leave the mainloop: a stage of the cp.async ring
holds codes only, and the epilogue multiplies each sum by token scale x (weight scale x post) and rounds it once to
FP32 or BF16 (the engine's BF16 activations come straight out; no elementwise pass follows, kernel core
`gemm_epilogue.cuh`: `store_scaled` from registers, `store_scaled_staged` through shared memory as whole 16-byte rows).
The MMA is kernel core `mma_fp8.cuh`'s `mma_e4m3_m16n8k32_fast`: on sm_120a the block-scaled kind::mxf8f6f4 m16n8k32
with unit E8M0 scales, which runs at twice the plain e4m3 mma.sync's rate (RTX 5090 from registers: 120 G against 61
G MMAs a second; the generic path's per-block MMA, its zero-accumulator e4m3 one plus four FMAs, is what made it 2.6x
slower than cuBLASLt); elsewhere the plain e4m3 MMA. The ring, the fragments read a k-step ahead, the swizzled tiles
and the persistent grouped CTA order are gemm-native's (`tile.cuh`). `gemm_native(..., mx=False)` takes this kernel
when the table's every matrix has FP32 scales constant along K (`matrix_table` marks it `crit_channel`) and the
activations have one scale a token; `gemm_native_channel` calls it directly. A job whose matrix is not channel-scaled
gets NaN outputs (tests/test_fp8_block_rows.py checks both, and FP32 and BF16 against the FP64 product).

Numerics: each MMA adds its 32-element block into the FP32 accumulator in increasing order of K, the scales applied
once to the sum; within 1e-5 of the FP64 product relative to the products' magnitude (the block-scaled MMA's own block
sum is not FP32 round to nearest), plus BF16's rounding with BF16 out. Not the declared order and not row invariant
(kernel.json `"row_invariant": false`). There is no split of K and the configuration is the only thing the call's
token count chooses: at one configuration (a fixed max_count), an output's bits do not depend on the job's device-side
token count or on which persistent CTA takes its tile.

RTX 5090 (sm_120a), torch 2.14.1, `crit kernels sweep fp8-block-rows --structure gemm-native-channel` (280
configurations, none dropped) on Qwen3.8-27B's FP8 matrices as the engine stacks them (`--model`'s shapes, swept in two
pieces with `--shapes` to keep each hold of the shared GPU short: q|k|v 14336 x 5120, the gated-delta q|k|v|z 16384 x
5120, layers 56-63's MLP gate|up 34816 x 5120 and down 5120 x 17408, attention o and gated-delta out 5120 x 6144,
lm_head 248320 x 5120), random E4M3 codes with FP32 row scales and amax activation scales, BF16 out, cold L2; tuned
kernel / cuBLASLt's row-wise scaled FP8 GEMM (torch._scaled_mm, BF16 out, timed as a graph replay) in us, vs_library
(> 1: crit faster), and the untuned defaults' vs_library:

| Shape (N x K) | 128 | 512 | 2,048 | 8,192 tokens |
| --- | --- | --- | --- | --- |
| 14336 x 5120 | 61.4 / 79.9 (1.30; 1.27) | 192.5 / 223.3 (1.16; 1.16) | 626.8 / 745.5 (1.19; 1.19) | 2,245.7 / 2,851.9 (1.27; 1.27) |
| 16384 x 5120 | 69.6 / 81.9 (1.18; 1.18) | 215.1 / 288.8 (1.34; 1.19) | 720.9 / 874.5 (1.21; 1.21) | 2,563.1 / 3,251.2 (1.27; 1.27) |
| 34816 x 5120 | 139.3 / 172.1 (1.24; 1.22) | 419.9 / 512.0 (1.22; 1.22) | 1,425.5 / 1,757.2 (1.23; 1.22) | 5,572.7 / 7,158.5 (1.28; 1.24) |
| 5120 x 6144 | 34.8 / 49.2 (1.41; 0.96) | 75.8 / 90.1 (1.19; 1.19) | 270.4 / 333.9 (1.24; 1.20) | 997.1 / 1,273.9 (1.28; 1.27) |
| 5120 x 17408 | 81.9 / 122.9 (1.50; 1.00) | 217.1 / 251.3 (1.16; 1.14) | 688.2 / 903.7 (1.31; 1.31) | 2,742.9 / 3,675.1 (1.34; 1.34) |
| 248320 x 5120 (lm_head) | 950.3 / 986.8 (1.04; 1.03) | 2,746.8 / 3,306.5 (1.20; 1.20) | 10,869.7 / 13,449.2 (1.24; 1.24) | 43,726.4 / 52,857.2 (1.21; 1.20) |

(lm_head at 128 tokens reads its 1.27 GB of codes at 1.34 TB/s: memory bound.) Picks: the defaults (128 x 128, 8 warps of 64 x 32, 3 stages of 128 bytes, the staged epilogue) on most shapes;
128 x 256 with 2 stages on the wide matrices at 512 tokens and more, 256 x 128 at 8,192 on some; at 128 tokens the
narrow matrices (5,120 rows) need smaller tiles to fill the GPU's 170 SMs (64 x 128 on 5120 x 6144, 64 x 64 with 4
warps and the register epilogue on 5120 x 17408): the defaults' 40 CTAs leave most SMs idle there (0.96 and 1.00).
The kernel reaches 480-530 TFLOPS on 2,048 tokens and more (cuBLASLt 400-420). What is left: a split of K (or stream-K) for short prompts
on narrow matrices, which the tuned small tiles cover today; TMA loads into an mbarrier ring with warp-specialized
producers (the kernel core has the TMA and mbarrier blocks, `load.cuh`), not needed to pass cuBLASLt here.

## The blocks GEMM

`gemm-native-blocks` is `gemm_native(..., mx=False)` for every table the channel GEMM does not take: an FP8 block
checkpoint's 128 x 128 FP32 weight scales against activations quantized per group of 128 (its recipe's dynamic
activations), the same weights against one scale a token, row scales against grouped activations, MXFP8 matrices
against activations whose scales are not powers of two. A scale span is the run of 32-element blocks over which neither
operand's scale changes (the gcd of scale_k and the activation group; all of K for an operand whose scale is constant
along K). Each span's MMAs sum in a span accumulator in the tensor cores (`mma_fp8.cuh`'s fast MMA, its
zero-accumulator form `mma_e4m3_m16n8k32_fast_zero` on a span's first block), and one FMA a span folds it into the
output's FP32 sum times the span's activation scale x weight scale; a scale constant along K, and post, is applied once
in the epilogue (gemm_epilogue.cuh, FP32 or BF16 out). Where a weight block of scale_rows rows holds the CTA's TN rows
whole (128 x 128 blocks with TN 64 or 128), the CTA has one weight scale a span: one product a token, one FMA an
output. The stage's FP32 scales (E8M0 bytes decoded in place by the threads that copied them) go to a ring of
STAGES + 1 slots, one more than the code ring, so the slot a warp still folds from is never the one being refilled;
the code ring, fragments and persistent order are the channel GEMM's. 1e-5 of the FP64 product's magnitude (the
bench's check); tests/test_fp8_block_rows.py runs five layouts in FP32 and BF16.

Yardstick: cuBLASLt has no FP8 GEMM with FP32 block scales on sm_120 (torch._scaled_mm's 1 x 128 / 128 x 128 scaling
returns CUBLAS_STATUS_NOT_SUPPORTED with torch 2.14.1 on an RTX 5090), so the yardstick is its MXFP8 GEMM on the same
codes (E8M0 scales per 32 on both operands, applied by the tensor cores), BF16 out: block-scaled FP8 at the library's
best rate, a GEMM with less work than FP32 block scales (no span sums and folds).

RTX 5090, `crit kernels sweep fp8-block-rows --structure gemm-native-blocks --model` an FP8 block checkpoint (DeepSeek
V4 Flash W4A16-FP8: its attention projections, 128 x 128 FP32 weight blocks, activations per 128; 176 configurations,
none dropped), random codes and scales, cold L2; tuned kernel / cuBLASLt MXFP8 (us), vs_library (untuned in
parentheses):

| Shape (N x K) | 128 | 512 | 2,048 | 8,192 tokens |
| --- | --- | --- | --- | --- |
| 64 x 4096 | 24.5 / 28.7 (1.17; 0.54) | 24.5 / 30.7 (1.25; 0.58) | 26.6 / 30.7 (1.15; 0.58) | 30.7 / 40.9 (1.33; 0.71) |
| 256 x 4096 | 24.5 / 30.7 (1.25; 0.58) | 24.5 / 30.7 (1.25; 0.58) | 26.6 / 30.7 (1.15; 0.56) | 67.6 / 49.2 (0.73; 0.73) |
| 512 x 4096 | 24.5 / 30.7 (1.25; 0.58) | 24.5 / 30.7 (1.25; 0.58) | 38.9 / 33.8 (0.87; 0.61) | 129.1 / 91.2 (0.71; 0.71) |
| 1024 x 4096 | 24.5 / 30.7 (1.25; 0.58) | 26.6 / 30.7 (1.15; 0.58) | 63.5 / 41.0 (0.65; 0.65) | 260.1 / 192.6 (0.74; 0.74) |
| 8192 x 1024 | 14.3 / 14.3 (1.00; 0.78) | 34.8 / 22.5 (0.65; 0.65) | 125.0 / 86.0 (0.69; 0.69) | 424.0 / 322.0 (0.76; 0.76) |
| 32768 x 1024 | 38.9 / 34.8 (0.89; 0.89) | 129.1 / 88.0 (0.68; 0.68) | 432.2 / 327.7 (0.76; 0.76) | 1,611.0 / 1,251.4 (0.78; 0.78) |
| 8192 x 4096 | 40.9 / 43.0 (1.05; 0.75) | 129.0 / 91.8 (0.71; 0.71) | 436.3 / 317.5 (0.73; 0.73) | 1,509.4 / 1,114.2 (0.74; 0.74) |
| 4096 x 8192 | 55.3 / 59.4 (1.07; 0.58) | 118.3 / 81.9 (0.69; 0.69) | 480.8 / 338.0 (0.70; 0.70) | 1,576.6 / 1,130.6 (0.72; 0.72) |

The untuned column was measured before default_rows (1-512 tokens on 64 x 64 tiles, the sweep's pick there) was
added. Where the arithmetic dominates (512 tokens and more on 1,024 rows and up) the blocks GEMM runs at 0.65-0.78 of
the MXFP8 library GEMM, 315-350 TFLOPS against the channel GEMM's 470-530: measured on 8192 x 4096 x 2,048 (iso, one
configuration), the blocks GEMM on channel scales (no span, no fold) loses 7% to the channel GEMM (its span
accumulator: 252 registers a thread), weight blocks with one activation scale a token another 14%, activations per
128 another 12%. Tried and measured slower: folding a span tile by tile ahead of the next span's MMAs (spills), and
moving each scale's power of two into the block-scaled MMA with the accumulators rescaled by the mantissas' ratio at
each span (no span accumulator, but 0.55 of the yardstick). What is left: the fold's cost against a library GEMM that
applies power-of-two scales in the tensor cores; an activation quantizer with power-of-two group scales (MXFP8's)
would let these matrices take the block-scaled path for their activations and fold only the weight blocks.
