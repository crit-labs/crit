Integer block expert projections, crit's kernel for its host-expert-arithmetic technique (operation
`integer-block-rows`), built for the engine's device with the engine's tuning or the default knobs:
4-bit weights (NVFP4's E2M1 codes or INT4, one E4M3 scale per 16) against activations quantized per block (E2M1 x 2
for 4-bit activations, INT8 for 8-bit ones), in crit's declared integer order. Each 16-block's dot of the integer values
is exact on any processor; the FP32 step is declared (block b to lane b % 16, `lane = fma(sum, weight scale *
activation scale, lane)` with the product rounded, the halving tree over 16 lanes, times post).

**Why it exists.** The decode rows, the prompt GEMM and crit's CPU rows (`host_int.h`: AVX-512 VNNI, AVX2, scalar)
give the same bits for every output. A CPU expert tier, a device expert cache and the prompt path can split an expert's
work any way, and move experts between tiers while serving, without changing a token: what crit's repeatability,
request-history and honesty gates ask, at integer speed on every processor.

## Applicability

| | |
| --- | --- |
| Weight formats | E2M1 codes (NVFP4) or INT4, two a byte (low nibble the even element), E4M3 scale per 16 weights, a post factor per matrix; rows layout [rows][cols / 2] |
| Activations | int8 values per element (E2M1 x 2 or INT8) and an FP32 scale per 16 or 32 values; the declared quantizers make them from FP32 |
| GPUs | NVIDIA sm_80 and newer (int8 `mma.sync` m16n8k16, `dp4a`); not HIP |
| Shapes | cols a multiple of 64; decode jobs of 1 to 8 tokens on one matrix; prompt jobs of any length; any rows |
| Numerics | bitwise equal to the declared order's scalar oracle and to crit's CPU rows (tests/int_block_tests.cu, an RTX 5090) |
| Determinism | same bits on a repeat; an output independent of its job's other tokens and of the kernel that computed it |
| Measured speed | RTX 5090: decode rows 1.9-3.4 µs per 2.76 MB matrix for 1-8 tokens (from L2); prompt GEMM 24 TOPS on 4096 x 1920 x 2560 in registers (`gemm`); 1.5 to 3.6 times `gemm`'s speed with shared-memory staging (`gemm-staged`, Against the library); `gemm-native` (NVFP4 W4A4, not the declared order) 0.93 to 0.96 of cuBLASLt's NVFP4 GEMM on Qwen3.8-27B's gate|up and 0.89 to 0.93 on its down at 512-8,192 tokens, 0.93 and 1.14 at 128 (The native GEMM) |

## Interface

```python
from crit_kernels.build import load      # a plugin: crit_plugins.kernels.load_kernel("int-block-experts", folder=...)
ib = load("int-block-experts")
table = ib.matrix_table([(codes, scales, post, int4)])            # one Matrix per matrix
values, scales = ib.quantize_int8(x, 32)                           # or ib.quantize_e2m1x2(x, global_scale)
ib.rows(table, ib.row_jobs([(0, act_rows, out_rows)], dev), max_rows, values, scales, 32, out)   # decode, 1-8 tokens a job
ib.gemm(table, ib.gemm_jobs([(0, first, count)], dev), max_rows, max_count, values, scales, 32, out)   # prompt
ib.gemm_staged(table, ib.gemm_jobs([(0, first, count)], dev), max_rows, max_count, values, scales, 32, out)   # the same bits
codes, bytes_ = ib.quantize_nvfp4(x, global_scale, tiled=False)    # x FP32 or BF16: quantize_e2m1x2's values as NVFP4
ib.gemm_native(table, ib.gemm_jobs([(0, first, count)], dev), max_rows, max_count, codes, bytes_, global_scale, out,
               col_scale=None, tiled=0)   # not the declared order; out FP32 or BF16; tiled: ib.TILED_ACTS | ib.TILED_WEIGHTS
h = ib.swiglu(gate_up)
```

## Structures and knobs

| Structure | Knob | Values | Default | What |
| --- | --- | --- | --- | --- |
| `rows` (decode) | THREADS | 64, 128, 256, 512 | 256 | threads a block, a half-warp an output row |
| | ROWS | 1, 2, 4 | 1 | output rows a half-warp owns (a token's activation block read once for them) |
| | STREAM | 0, 1 | 0 | read codes with L1::no_allocate |
| | UNROLL | 1, 2 | 1 | blocks a lane loads before computing them |
| `gemm` (prompt) | WARPS | 2, 4, 8, 16 | 8 | warps a block |
| | TILES | 1, 2, 4 | 4 | m16n8k16 tiles (8 weight rows each) a warp runs on one A fragment |
| | TOKZ | 1, 2, 4 | 1 | 16-token tiles a block takes along grid.z |
| `gemm-staged` (prompt) | BM, BN | 32-128, 16-64 | 128, 64 | weight rows and tokens a CTA |
| | WM, WN | 2-4, 1-4 | 4, 2 | warps along the rows and the tokens (a warp BM / WM x BN / WN) |
| | GROUP | 1, 2 | 2 | sibling lanes of the tree a stage carries a block of |
| | RUN | 1, 2 | 1 | consecutive blocks of each lane a stage carries |
| | STAGES | 4 | 4 | stages of the cp.async ring |
| `gemm-native` (prompt, NOT the declared order) | TM, TN | 64-256, 64-256 | 128, 128 | tokens and weight rows a CTA (TM 192 too) |
| | WM, WN | 1-6, 2-4 | 2, 4 | consumer warps along the tokens and the rows (beside the producer warp on sm_120a; 12 stream B) |
| | BK | 64, 128 | 64 | bytes of every row a stage (128 or 256 values) |
| | STAGES | 2-6 | 4 | stages of the ring (sm_120a: as many as fit up to this; elsewhere 3 or 4) |
| | GROUP | 8 | 8 | weight-row tiles a group of the persistent CTAs' tile order |
| | EPI | 0, 1, 2 | 1 | sm_120a, BF16 out: 0 register stores, 1 / 2 each warp's 16-row slices by TMA stores from one / two boxes; other GPUs, FP32 out: 1 through shared memory |
| | SPLIT | 0, 1, 2 | 0 | sm_120a: 0 stream-K for launches of few tiles and a long K, 1 every tile whole, 2 stream-K for every last partial wave |
| `declared-quantizers` | none | | | the quantizers and SwiGLU |

The tuned structures are keyed by the matrix shape (N: the largest matrix's rows, K: the activations' width) and the row
count (activation rows for `rows`, tuned at 1, 2, 4 and 8; the job's tokens for `gemm` and `gemm-staged`, tuned at 16,
128, 1,024 and 4,096; `gemm-native` at 128, 512, 1,024, 2,048, 4,096 and 8,192). No knob of the declared structures changes an
output's operations or their order (no order knob): every configuration gives the declared order's bits; `gemm-native`
is outside that promise (kernel.json `"row_invariant": false` and its `order`). The defaults of `rows` and `gemm` are
the launches the kernels had before they had knobs; those of `gemm-staged` its best configuration on the largest prompts
swept.

**`gemm-staged`** is the prompt GEMM rebuilt from crit's kernel core (`kernels/core/include/crit_kcore/blockscaled_*.cuh`
on `mma.cuh`, `load.cuh` and `formats.cuh`), the mechanisms of a multistage tensor-core GEMM (CUTLASS's mainloop,
BSD-3-Clause; written anew, no code copied) in the declared order: the blocks are visited lane by lane in the order
the halving tree takes the lanes (`GroupVisits`), so an output holds its running lane and the tree's partial sums (5
registers) instead of 16 lanes, and every lane still takes its blocks in increasing order. Each visit is one stage of a
4-deep cp.async ring: the token values, weight codes and scale words, decoded once a CTA one stage ahead of the MMAs
(codes to int8 by three byte permutes, `nibbles_to_s8x4`; E4M3 scales to FP32); the warps read their fragments by
ldmatrix and run one s8 m16n8k16 MMA a 16 x 8 tile and block onto an FP32 bias, so the exact block sum is one
subtraction (`block_sum_s8`, `biased_sum`: no integer-to-float conversion), then the declared step. Its outputs are the
`gemm`'s and the CPU rows' bits (row invariant; the bench checks every configuration against the untuned `gemm`).

**`rows`** is an alias of decode-gemv's `rows-int-block` (kernel.json `"alias": {"kernel": "decode-gemv", "of": "rows-int-block"}`):
this package's `rows()` forwards to decode-gemv's `rows_int_block`, built with this kernel's tuning table for `rows` (an old
tuning still applies; without one, decode-gemv's own tuning), and a sweep of `rows` and `select()` follow it there
(kernels/TEMPLATES.md, `alias`). The kernel is the kernel core's declared integer rows (`declared_rows.cuh` on
`multiply.cuh`'s declared paths), the operations this kernel's rows ran before: an A/B of every `rows` configuration
through the forwarding against the build before the move gave the same bits (RTX 5090). The C++ `enqueue_rows` stays
for C++ callers and the C++ tests, on the same core template.

## Build, test, tune

```sh
crit kernels build int-block-experts                              # this GPU, default knobs (or --tuning FILE)
python -m pytest kernels/techniques/experts/int-block-experts/tests -q       # needs nvcc for the C++ oracle test
crit kernels sweep int-block-experts --model CHECKPOINT --out TUNING_DIR
```

`tests/int_block_tests.cu` checks the rows and the GEMM against the scalar oracle, bit for bit, and the quantizers and
SwiGLU against their declared host functions; the pytest compiles it against a `crit_tuning.h` of the default knobs and
of another configuration (`crit_kernels.build.tuning_header`). Run it on the target first. An engine that splits
experts between the CPU and the GPU checks crit's `host_int_tests` too, so its CPU rows match on the target CPU. The
sweep's benches (`bench.py`) take the expert shapes (gate/up [2 inter][hidden], down [hidden][inter]) from the
checkpoint's config and keep a configuration only when every output equals the untuned build's bits; `gemm-native`'s
bench takes a dense NVFP4 model's MLP matrices (gate|up fused [2 intermediate][hidden], down [hidden][intermediate])
where the model has no experts, and checks every output against the FP64 product.

## Against the library

Each tuned structure names its yardstick in `kernel.json`, which the sweep times on the same shapes (`library_us`,
`vs_library` = library time / tuned time, above 1 the kernel is faster):

- `rows`: cuBLAS BF16 (`torch.matmul`) on the weights widened to BF16 and the INT8 activations dequantized, both once
  (no library multiplies E2M1 weights by INT8 activations in blocks of 32; it reads four times the weight bytes).
- `gemm-native`: cuBLASLt's NVFP4 block-scaled GEMM (`torch._scaled_mm`) on the same codes and tiled scale bytes as
  the kernel, BF16 output like the kernel's (The native GEMM).
- `gemm` and `gemm-staged`: cuBLASLt's NVFP4 block-scaled GEMM (`torch._scaled_mm` on FP4 tensor cores) on the stored
  codes and E4M3 scales (tiled to its 128 x 4 layout), the activations re-quantized to NVFP4 once (no library takes
  INT8 activations in blocks of 32 against 4-bit weights). It computes in a different order and with 4-bit
  activations, so it is a speed bound, not the same bits.

RTX 5090, torch 2.14.1, a sweep on the expert shapes of a 512-expert model (hidden 2,560, inter 640: gate/up 1280 x
2560, down 2560 x 640), random weights, tuned kernel against library (us):

| Structure | Shape (N x K) | 1 | 2 | 4 | 8 tokens |
| --- | --- | --- | --- | --- | --- |
| `rows` | 1280 x 2560 | 12.3 / 10.2 (0.83) | 16.1 / 16.4 (1.02) | 22.5 / 16.4 (0.73) | 36.9 / 16.4 (0.44) |
| `rows` | 2560 x 640 | 6.1 / 6.1 (1.00) | 8.2 / 8.2 (1.00) | 10.2 / 8.2 (0.80) | 14.3 / 8.2 (0.57) |

| Structure | Shape (N x K) | 16 | 128 | 1,024 | 4,096 tokens |
| --- | --- | --- | --- | --- | --- |
| `gemm` | 1280 x 2560 | 71.7 / 13.6 (0.19) | 79.9 / 13.7 (0.17) | 247.5 / 13.7 (0.055) | 1,062.9 / 26.6 (0.025) |
| `gemm` | 2560 x 640 | 18.4 / 8.2 (0.44) | 22.5 / 8.2 (0.36) | 106.5 / 10.2 (0.096) | 399.4 / 20.5 (0.051) |

The decode rows match cuBLAS at 1-2 tokens and fall behind it from 4 (each token's rows run their own dp4a chain while
cuBLAS reads the widened matrix once for all of them). The prompt GEMM is 5 to 40 times slower than cuBLASLt's FP4
GEMM: the price of the declared order on int8 `mma.sync` without shared-memory staging, and the gap a faster GEMM in
the same order has to close.

`gemm-staged`, the same sweep (`--structure gemm-staged`, 140 configurations, every one with the untuned `gemm`'s
bits), tuned kernel against the library (us), and the register `gemm`'s tuned time for comparison:

| Shape (N x K) | 16 | 128 | 1,024 | 4,096 tokens |
| --- | --- | --- | --- | --- |
| 1280 x 2560 | 36.8 / 12.3 (0.33), `gemm` 71.7 | 36.8 / 12.3 (0.33), 79.9 | 75.8 / 12.3 (0.16), 247.5 | 294.9 / 26.6 (0.09), 1,062.9 |
| 2560 x 640 | 12.3 / 8.2 (0.67), `gemm` 18.4 | 14.3 / 8.2 (0.57), 22.5 | 44.7 / 8.2 (0.18), 106.5 | 153.6 / 20.4 (0.13), 399.4 |

The staged GEMM is 1.5 to 3.6 times the register GEMM's speed, still 1.5 to 11 times slower than cuBLASLt's FP4 GEMM.
Most of that gap is the arithmetic the yardstick does not do: cuBLASLt multiplies 4-bit activations on FP4 tensor
cores that apply the block scales themselves (about 4 times the INT8 MMA rate on an RTX 5090), while the declared
order takes INT8 activations, one INT8 MMA a 16-element block (half of what an m16n8k32 does) and three FP32
operations a block and output on the CUDA cores (the bias subtraction, the scale product, the FMA). From registers
alone that is 94 G block MMAs a second on an RTX 5090 (a microbenchmark of `block_sum_s8` and `step`, 77% of the INT8
MMA rate); the kernel reaches 22 G at 4,096 tokens, held back like fp8-block-rows' staged GEMM (its README) by an
output's 5 registers (a CTA holds at most 128 x 64 outputs: two warps a scheduler, fragments loaded just in time, 32
bytes a row gathered from L2 each visit). Below 0.8 of the yardstick on every shape; the remaining levers are
fragment double-buffering in registers freed from the producer's state, a weight layout repacked once at load in the
lane-major visiting order (contiguous reads), and warp-specialized producers.

## The native GEMM

`gemm-native` gives up the declared order for the prompt GEMM's speed, with the owner's approval: it accumulates in the
tensor cores the way cuBLASLt and CUTLASS do, so an output is within FP32 accumulation of the exact product but its bits
are not the declared order's (not `rows`', `gemm`'s or the CPU rows') and change with the tuning (not row invariant; the
declared structures stay the default, and a plugin that selects this one says so in its facts and keeps the others
selectable). For a fixed configuration, `max_count` and `max_rows` an output's bits do not depend on its job's own
token count (the device-side count), on which CTA takes its tile or on the order a shared tile's segments finish, so a
caller that passes one `max_count` per matrix gets one set of bits per row. It takes 4-bit activations:
`quantize_nvfp4` packs `quantize_e2m1x2`'s values as NVFP4 (the same E2M1 codes, two a byte, and each 16-block's E4M3
scale byte before the division by the global divisor; tests check the two quantizers agree, also on BF16 input, which
it widens exactly), the arithmetic cuBLASLt's NVFP4 GEMM does; INT8 activations stay on `gemm` / `gemm-staged`.
`quantize_nvfp4` is one thread a 16-block (vector loads, the E4M3 conversion instruction and a comparison E2M1 encoder,
the same codes as the declared search) and writes the scale bytes row major or in cuBLASLt's 128 x 4 tiled layout.

**sm_120a, E2M1 weights** (`int4=False`): a warp-specialized persistent GEMM after CUTLASS's sm_90 / sm_120 TMA GEMMs
(BSD-3-Clause; the mechanisms, written anew from the PTX ISA). A producer warp fills a ring of STAGES shared-memory
stages: each stage two TMA tile copies (tensor maps over the activation codes and the matrix's codes, TM or TN rows x
BK bytes a box, swizzled 64B / 128B as the kernel core's `tile.cuh` offset so ldmatrix reads are conflict free, rows
and K past the tensors zero-filled) and one bulk copy a 128-row tile of tiled scale bytes, all completing on the
stage's full mbarrier; WM x WN consumer warps wait on it, read their fragments by ldmatrix at fixed per-lane offsets
(the next k-step's while the current one's MMAs run, the next stage's first while a stage's last MMAs run), run
kind::mxf4nvf4 block-scaled m16n8k64 MMAs (UE4M3 scales per 16 on both operands, applied by the tensor core, summed in
its FP32 accumulator) and release the stage on its empty mbarrier. Persistent CTAs take tiles in `tile_of`'s grouped
order, and the producer loads the next tile's first stages while the consumers store the last one. The epilogue
multiplies by post / global and, given `col_scale`, each column's factor (a stacked matrix's own inverse global scale).
BF16 outputs of whole tiles go through shared memory (`EPI` 1, the default; 2 alternates two boxes): each consumer
warp writes its 16-row slices into its own staging box (rows of its tile's width, in TMA's 32B / 64B / 128B swizzle,
so the writes are conflict free) and its lane 0 issues one TMA store a slice, which runs while the warp goes on to the
next tile's MMAs; FP32 outputs, partial tiles and `EPI` 0 store from registers. 192 x 128 tiles run 12 consumer warps of
32 x 64 (13 warps: 128 registers a thread), which hold one k-step's A fragments a step ahead and stream B two n8 tiles
at a time; a 192-row tile starts at row 0 or 64 of its 128-row scale tile, so a stage holds two scale tiles. Launches
with few tiles and a long K (tiles filling at most half the CTAs, 64 stages of K or more; `SPLIT` 2: the tiles past
every launch's last whole wave) share the tiles out over the CTAs k-iteration by k-iteration (stream-K): each
segment's FP32 partial goes to a workspace the op allocates, and a tile's last segment adds them in segment order.
The tensor maps need the host's view of the matrices: `matrix_table` records the codes' pointers on the table tensor
(`table.crit_host`), and the weights' maps are kept (the activations' made a call); a table without the record (a
copy, or packed by hand) runs the same kernel with the producer's lanes copying by cp.async, about 0.6 times as fast.
Scale bytes may be row major (copied in 4-byte pieces) or tiled (`tiled`: `TILED_ACTS` with every job starting at a
multiple of 128 rows, `TILED_WEIGHTS`), so a caller with cuBLASLt-layout scales passes them as they are.

**INT4 weights and other GPUs**: the cp.async kernel of the previous round (`native_tiles`): a STAGES-deep cp.async ring
of swizzled tiles (`tile.cuh`), each 16-block's exact sum on the s8 MMA (`block_sum_s8` on codes widened by
`nibbles_to_s8x4`) times the two scales on the CUDA cores; row-major scale bytes only.

Numerics: within 2e-6 of the FP64 product relative to the products' magnitude (FP4 products and their block sums are
exact in FP32, only the accumulation rounds), plus one BF16 rounding for BF16 outputs; the bench checks every
configuration against the FP64 reference of the stored values, `tests/test_int_block_experts.py` both paths with two
jobs, tokens and rows off the tiles, a K tail, BF16 outputs with column scales, every scale layout, the tensor-map and
cp.async copies, a stream-K launch and repeats (the same bits), with the untuned knobs (TMA-store epilogue), register
stores with stream-K past the last wave, and 192-row tiles.

RTX 5090 (sm_120a, 170 SMs), torch 2.14.1, CUDA 13.1, `crit kernels sweep int-block-experts --structure gemm-native
--model unsloth-Qwen3.8-27B-NVFP4` (layers 0-55's MLP matrices: gate|up 34816 x 5120, down 5120 x 17408; random E2M1
weights and E4M3 scales, BF16 activations quantized once, every scale byte tiled, BF16 output, cold L2), shared GPU (other
builders' work beside it: a few percent of noise); kernel against cuBLASLt's NVFP4 GEMM (`torch._scaled_mm`, the same
bytes, BF16 output, timed as a graph replay), tuned / untuned defaults / library, us (vs_library of the tuned and the
untuned kernel):

| Tokens | gate\|up 34816 x 5120 | down 5120 x 17408 |
| --- | --- | --- |
| 128 | 84.0 / 84.0 / 77.8 (0.93, 0.93) | 59.4 / 65.1 / 67.6 (1.14, 1.04) |
| 512 | 236.9 / 243.8 / 219.2 (0.93, 0.90) | 110.6 / 110.6 / 98.6 (0.89, 0.89) |
| 1,024 | 415.8 / 424.0 / 387.1 (0.93, 0.91) | 219.2 / 222.9 / 202.8 (0.93, 0.91) |
| 2,048 | 781.7 / 786.5 / 737.3 (0.94, 0.94) | 399.4 / 404.7 / 370.8 (0.93, 0.92) |
| 4,096 | 1,498.2 / 1,513.5 / 1,439.8 (0.96, 0.95) | 761.1 / 772.1 / 708.7 (0.93, 0.92) |
| 8,192 | 2,944.1 / 2,962.5 / 2,838.2 (0.96, 0.96) | 1,537.7 / 1,568.8 / 1,368.1 (0.89, 0.87) |

54 configurations, none dropped (the bench's `valid`: 128 x 128 tiles with 4 or 8 consumer warps and 192 x 128 with
12, as many stages as fit with or without the staging boxes, EPI 0 or 1, SPLIT 0 or 2; smaller tiles only for
matrices under 4,096 rows or inputs). Picks: 4 consumer warps of 64 x 64 with two 128-byte stages and the TMA-store
epilogue nearly everywhere, 192 x 128 for gate|up at 8,192, register stores and 5 stages for gate|up at 128; stream-K
past the last wave (SPLIT 2) for gate|up at 512 and down at 2,048-4,096. The untuned defaults (128 x 128, 8 consumer
warps of 64 x 32, BK 64, 4 stages, EPI 1, SPLIT 0) are within 3% of the picks from 512 tokens on, 10% at 128.
Against the previous round's cp.async kernel (FP32 output, row-major scales, default knobs, same GPU) at 4,096 tokens:
gate|up 1,682 us before, 1,514 untuned now; down 818 before, 772 now; at 128 tokens 94.6 and 84.4 before, 84.0 and 65.1
now. Against the first sweep of this kernel (register stores, no 192-row tiles, the same GPU): 0 to +4% tuned at
512-8,192 tokens.

The activations' quantization is the same for both and not in the times: `quantize_nvfp4` of BF16 x with tiled scales
takes 6.5-74 us on gate|up's input (128-8,192 tokens x 5,120) and 9-238 us on down's (x 17,408), the memory bound of
reading BF16 (about 1.4 TB/s); counted in on both sides the ratios move toward 1 by at most 0.03.

Where it stands: at 1,024 tokens and more the kernel runs about 950-1,020 TFLOPS against cuBLASLt's 1,000-1,100.
What was measured on the way (RTX 5090, 4,096 tokens unless said): one bulk copy a row of codes into padded shared
memory instead of the TMA tiles is 8 times slower (the copy engine's per-request cost); cp.async by four producer
warps 0.75 of the library; no producer warp (the warps issuing the copies themselves) 0.65-0.70; 4 consumer warps
with 64 x 64 warp tiles 0.80; 128 x 64 tiles 0.60 and 256 x 128 or 128 x 256 tiles (64 x 64 warp tiles held in 168
registers by streaming B, or spilling) 0.50-0.55; a fifth stage, 16-byte BF16 stores after a quad transpose: no change;
leaving the stores out entirely: +9% on gate|up, +2% on down, of which the TMA-store epilogue recovers about half
(+4-5% on gate|up, +2-4% on down; one CTA-wide 32 KB staging tile measures the same as the per-warp boxes, which leave
room for a fourth stage); 192 x 128 tiles: +2-5% at 8,192 tokens, lower at 512 (2.7 token tiles of 512 round up to 3)
and with fewer than 3 stages (with a whole-tile staging buffer, 2 stages: 0.72-0.82); stream-K past the last wave
(SPLIT 2): +1-2% on down at 8,192 (2,560 tiles: 15.06 waves), worse where a launch is about one wave. Nine warps (8
consumers and the producer) get 168 registers a thread on this GPU (a sub-partition's 16K registers hold three warps
of 170; 13 warps four of 128), and sm_120 has no setmaxnreg to move registers from the producer to the consumers, so a
64 x 64 warp tile does not fit beside the producer. Not built: two consumer groups taking alternate tiles (with
mma.sync one group of 4 warps reaches 0.80 alone, so the alternation would trade MMA issue for the epilogue the TMA
stores already hide); left: the rest of the epilogue's share and the last wave's idle SMs on down at 8,192 tokens,
MXFP4 weights (kind::mxf4, UE8M0 scales per 32) on the same kernel.
MXFP4 weights (kind::mxf4, UE8M0 scales per 32) on the same kernel.
