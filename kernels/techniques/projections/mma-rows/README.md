# mma-rows

Projections of 1 to 64 activation rows on tensor cores in one fixed tile, from
[crit](https://github.com/crit-labs/crit)'s quantized-projections technique: a matrix stored as NVFP4, FP8 (E4M3 with
a row scale), MXFP8, BF16 or INT4 with a zero point a group (AWQ, GPTQ) times BF16 rows, the activations used as they are (weight only: W4A16, W8A16) or quantized
as a prompt's are (FP8 per row: W8A8; NVFP4 per 16-value block: W4A4), FP32 sums; on the sm_120 family, W4A4 rows on
the block-scaled FP4 MMA (E2M1 codes times E2M1 codes, both block scales inside the MMA); and the two activation
quantizers alone, for a provider's GEMM on the same activations.

**Why it exists.** Decode steps, speculative verify windows and short prompts multiply a few rows by every weight: the
weights' read is the cost. A CUDA-core GEMV grows compute bound past a few rows, and a provider's GEMM picks its
kernel, and so each output's sum, by the row count, so a row changes with the rows beside it. Here every call runs one
fixed MMA tile (8 rows a column of the MMA, the weights its A operand), each weight is read once a call whatever the
rows, and a sum's order depends on the matrix and the plan only: a row gets the same bits alone, in a decode call of
16 rows or in a prompt's chunk of 64, which keeps batched decoding, verify windows and padded prompt buckets equal to
serial decoding.

## Applicability

| | |
| --- | --- |
| Weight formats | NVFP4: codes [N][K / 2] (element 2i in the low nibble of byte i), E4M3 block scales [N][K / 16] row major or in cuBLASLt's 128 x 4 tiled layout, inverse global scale [N] FP32. FP8: E4M3 [N][K] with FP32 row scales [N]. MXFP8: E4M3 [N][K] with E8M0 exponents [N][K / 32] (modelopt's layout). BF16: [N][K]. INT4 with a zero point (weights `int4_zp`, code 4): crit's rows as decode-gemv's `repack_awq` / `repack_gptq` write them, codes [N][K / 2] unsigned, FP16 scales and uint8 zero points [N][K / group] (`zeros`, `group`: a multiple of 32 dividing K), BF16 activations; a value (code - zero) * scale has up to 15 significant bits, so the kernel core's split decoder gives it as hi + lo, two BF16 values, and each MMA of the weights runs twice (hi, then lo) on the same activations: every product exact, twice the MMAs |
| Activations | BF16 [M][K], M 1 to 64, rows on 16-byte boundaries; used as they are, or quantized by the kernel's pre-pass: FP8 rows (scale = amax * fp32(1 / 448), at least 1e-12; values e4m3(clamp(x / scale)), the scale after the sum) or NVFP4 blocks under the input scale g (block scale e4m3(min(amax * fp32(1 / 6) * g, 448)), values the nearest E2M1 magnitude, ties to the even code; fp32(1 / g) after the sum) |
| FP4 MMA | acts `nvfp4_mma` (sm_120a, 121a; `fp4_mma()` says whether the device runs it): NVFP4 weights with their scales in the tiled layout, K a multiple of 128, 1 to 64 rows. The pre-pass quantizes the rows to NVFP4 codes and tiled scales (the `quantize_nvfp4` op's bits); then `mma.sync` kind::mxf4nvf4 m16n8k64 multiplies the weights' E2M1 codes (the A operand: a quad reads 64 bytes of a row a step, two k64 MMAs, and three shuffles transpose them across the quad) by the activations' (the B operand, 8 tokens a column) with both operands' E4M3 block scales applied inside the MMA, FP32 sums, the inverse global and input scales after. The warp tile is the same at every row count, so a row's bits are the same alone and among up to 63 others; within 1e-5 of the BF16-MMA W4A4 path's values and 2e-5 of the FP64 product |
| Output | FP32, BF16 or FP16, rounded once |
| Shapes | K a multiple of 32, any N |
| GPUs | NVIDIA sm_80 and newer (mma.m16n8k16 with BF16 operands); the hardware E2M1 conversion on sm_100a, 103a, 110a, 120a, 121a, the bit-pattern decode elsewhere (the same values); the pre-pass overlaps the weight stream through programmatic dependent launch from sm_90 |
| Numerics | every value enters the MMA exactly in BF16 (an E2M1 code times its E4M3 scale, an E4M3 code, times its power of two for MXFP8, a BF16 weight; a BF16 activation or its quantized value), FP32 accumulation, the row, activation and global scales after the sum. Within 2e-5 of the FP64 product relative to the products' magnitude (`tests/test_mma_rows.py`) |
| Determinism | a sum's order is fixed by N, K and the plan's splits of K (plan(): the build's tuning for the matrix, or untuned from the shape, the device's multiprocessor count and the decode tile's occupancy there, decode_ctas); the rows a call takes, `groups`, `warps`, the FP4 MMA's HOIST and the prompt tile's PW and PT change no bits. The split CTAs' partial sums add in split order (the last CTA of a row tile adds them). A row's bits are the same alone and among up to 63 others, and on a repeat; `mm_prompt` gives every row of a call of any length the same bits |
| Capture | no allocation and no host synchronization (the caller gives scratch, partial-sum and counter buffers; the counters are left zero): a CUDA graph captures a call |
| Measured speed | RTX 5090, a call's GPU time in a captured graph over matrices that leave the 96 MB L2, at 1 / 4 / 16 rows: NVFP4 W4A16 34816 x 5120 68 / 70 / 99 us (1.47 / 1.43 / 1.02 TB/s), 5120 x 17408 36 / 39 / 56 us; W4A4 (the activations quantized by the pre-pass) the same within 2%; FP8 W8A8 16384 x 5120 55 / 56 / 66 us (1.54 TB/s at 1 row), 5120 x 6144 24 / 25 / 32 us, a 248320 x 5120 head 774 / 778 / 797 us (1.64 / 1.59 TB/s); with plan()'s splits, a 33B llama-layout model's four NVFP4 roles (qkv 9216, o 7168, gate/up 44032 x 7168, down 7168 x 22016) take 220.7 / 223.5 / 230.6 us at 1 / 2 / 4 rows against a CUDA-core GEMV's (decode-gemv) 210.3 / 224.4 / 241.5, and its 44032 x 7168 gate/up 114.8 us at 1 row (125.4 with the earlier plan's 688 CTAs for 680 resident); about twice the GEMV's speed at 16 rows; FP8 W8A16 within 1-5% of the GEMV at 1-4 rows. Past 16 rows the MMAs and decodes grow with the rows (64 rows: 3-5x the 1-row time), so a provider's GEMM wins sooner for activations it takes as they are |
| INT4 speed | RTX 5090, INT4 with a zero point (group 128, FP16 scales), one call in a captured graph over 4 copies of the matrix (weights from DRAM), plan()'s splits, at 1 / 2 / 4 / 8 / 16 rows: 34816 x 5120 73.8 / 85.4 / 78.1 / 88.6 / 134.3 us against torch's W4A16 INT4 GEMM (torch._weight_int4pack_mm, tinygemm) 60.9 / 72.7 / 68.4 / 125.0 / 253.5 (0.83 / 0.85 / 0.88 / 1.41 / 1.89x its speed); 5120 x 17408 40.4 / 40.5 / 53.0 / 47.7 / 69.8 against 32.1 / 33.6 / 39.8 / 64.0 / 149.8 (0.79 / 0.83 / 0.75 / 1.34 / 2.15x); 17408 x 5120 39.5 / 40.2 / 41.8 / 45.1 / 69.6 against 32.4 / 33.1 / 35.0 / 66.7 / 170.0 (0.82-0.84x to 4 rows, 1.48 / 2.44x at 8 / 16). The two MMAs a step (hi and lo) and the scale and zero decode make the 1-4 row calls 15-25% slower than the library; decode-gemv's CUDA-core GEMV takes those rows level with it (its README) |
| FP4 MMA speed | RTX 5090, through quantized-projections (pre-pass included), at 1 / 4 / 8 / 9 / 16 rows: 34816 x 5120 68.8 / 69.1 / 69.3 / 70.2 / 70.6 us (1.46 / 1.42 TB/s at 1 / 16 rows), 5120 x 17408 38.3 / 39.2 / 40.6 / 41.6 / 43.9 us (1.31 / 1.14 TB/s); the W4A16 tile on the same matrices 67.6 / 70.4 / 74.1 / 84.8 / 93.2 and 34.5 / 36.3 / 39.9 / 43.6 / 53.4: 2% and 11% slower at 1 row, faster from 8 rows on, 24% and 18% faster at 16. The pre-pass costs about 1 us a call |
| Yardsticks | kernel.json `yardstick`, the sweep's library_us. INT4 weights: torch's W4A16 INT4 group GEMM (torch._weight_int4pack_mm) on the same codes repacked once (INT4 speed above). decode-tile and prompt-tile: NVFP4 weights against flashinfer's mm_bf16_fp4 (W4A16, backend cute-dsl-native) on the stored codes, the block scales in its 128 x 4 layout once, the per-row inverse scale left out; BF16 against cuBLAS (torch.matmul) on the same matrix; FP8 against cuBLAS on the values times their row scales widened to BF16 once (no library takes FP8 weights against BF16 activations; twice the weight bytes). fp4-tile: cuBLASLt's block-scaled NVFP4 GEMM (torch._scaled_mm) on the same weight bytes, the activations quantized once outside its timed call (the tile's time includes its pre-pass). RTX 5090, `crit kernels sweep`, tuned vs library in us. decode-tile (before the staged tile) at 1 / 2 / 4 / 8 / 16 rows: NVFP4 34816 x 5120 67.6 / 69.6 / 69.6 / 69.7 / 84.0 against 69.0 / 69.0 / 69.0 / 69.0 / 71.4 (1.02 / 0.99 / 0.99 / 0.99 / 0.85x the library's speed); with the staged tile (SW, SPF) swept on four of a 27B's matrices (2026-11-10), each call in a captured graph over copies of the matrix that leave the L2, both kernels timed the same way, at 1 / 2 / 4 / 8 / 12 / 16 rows: NVFP4 34816 x 5120 65.9 / 67.0 / 67.8 / 69.6 / 75.7 / 76.7 against 66.8 / 66.8 / 67.5 / 68.6 / 69.2 / 70.1 (1.01 / 1.00 / 1.00 / 0.99 / 0.91 / 0.91x), 17408 x 5120 34.2 / 37.5 / 37.5 / 37.9 / 38.9 / 38.9 against 33.7 / 34.2 / 34.1 / 35.0 / 35.7 / 36.1 (0.99 / 0.91 / 0.91 / 0.92 / 0.92 / 0.93x), 5120 x 17408 35.7 / 35.8 / 35.9 / 36.6 / 39.6 / 40.4 against 33.9 / 34.0 / 34.4 / 34.0 / 35.4 / 35.9 (0.95 / 0.95 / 0.96 / 0.93 / 0.89 / 0.89x), where the register tile alone (the same plans) took 0.67-0.85x at 16 rows; that sweep's own cold-L2 single calls gave the tile 67.6-79.8, 38.9-40.9 and 36.9-43.0 us but flashinfer 125-127, 65-92 and 67-69 us (its untuned default tactic in that harness, about twice its graph time), so the graph times are the comparison. Of the remaining gap at 16 rows, the decode is 0-5% (codes passed to the MMAs undecoded) and the MMAs 5-10% (replaced by adds): the rest is the weight stream's pipelining. FP8 16384 x 5120 55.3 / 55.3 / 55.3 / 57.3 / 59.4 against 102.4 / 114.7 / 114.7 / 115.9 / 116.7 (1.85-2.08x). fp4-tile at 1 / 2 / 4 / 8 / 16 rows: 34816 x 5120 71.7 / 73.2 / 73.1 / 71.7 / 73.7 against 98.3 / 88.1 / 90.2 / 95.6 / 92.2 (1.20-1.37x), 5120 x 17408 40.9 / 40.9 / 40.9 / 43.0 / 47.1 against 57.3 / 57.3 / 57.3 / 57.3 / 58.7 (1.25-1.40x). prompt-tile at 64 / 512 / 2048 rows: NVFP4 34816 x 5120 182 / 1053 / 4028 against 163 / 941 / 3567 (0.89 / 0.89 / 0.89x); BF16 5120 x 6144 49 / 219 / 803 against 86 / 160 / 635 (1.75 / 0.73 / 0.79x). flashinfer's W4A16 GEMM matches the NVFP4 decode tile to 8 rows and is 7-12% ahead of the staged tile at 12-16 rows (11-15% of the prompt tile on prompts) |

## Prompt rows with the decode tile's bits

`mm_prompt` multiplies any number of rows, each output with the bits `mm_rows` gives the same row (the same matrix,
activations and splits): the same MMA with every operand where the decode tile puts it, the same steps and sub-steps
of K in the same order, each split of K a chain from zero, the splits added in split order as the decode tile's last
CTA adds them, the same scales after the sum. A prompt's rows then score exactly as the same tokens decode (an engine
whose served and scored log-probabilities must agree), while a CTA reuses each decoded weight for 64 rows as a GEMM
does: 8 warps of 16 matrix rows each, the 64 rows' sub-steps staged in shared memory (cp.async, two buffers, a swizzle
that keeps the B fragments' reads free of bank conflicts), every split of K in the CTA with a running total, so no
partial-sum or counter buffers. Quantized activations take the decode tile's pre-pass for all the call's rows.
Measured on an RTX 5090 (one call in a captured graph, BF16 out) at 4,096 rows: NVFP4 W4A16 44032 x 7168 15.1 ms (171
TFLOPS), 7168 x 22016 8.1 ms (160), 34816 x 5120 8.7 ms (169), FP8 W8A16 16384 x 5120 4.3 ms (160); 150-170 TFLOPS from
1,024 rows, 114-160 at 256; against cuBLAS's BF16 GEMM over the same matrix decoded beforehand (its decode not
counted) 75-80% of its rate, and twice the rate of the decode tile in calls of 64 rows (70-89 TFLOPS).

A CTA whose 64 activation rows run past M (a prompt's last tile) stages and multiplies only its columns of eight
rows that hold a row below M: a 65-row prompt's second CTA does one column's work instead of eight. Outputs are
unchanged. On an RTX 5090, a 5120 x 5120 matrix at 65 to 200 rows took 2 to 6% less (BF16 at 65 rows 88 to 83 us), and
full tiles stayed within the runs' spread.

## Gate and up in one matrix (glu)

An MLP's gate and up projections read the same activations, and the activation product after them reads both outputs
back. `mm_rows(..., glu=True)` and `mm_prompt(..., glu=True)` (the FP4 MMA too) take the two as one matrix whose rows
are interleaved by eights (`interleave_glu(gate, up)` for codes, row-major block scales, row scales or BF16 weights;
NVFP4 scales go into the tiled layout after they are interleaved). A thread's MMA fragment then holds a gate row and
its up row for the same tokens, so the epilogue writes `silu(gate) * up`, out [M][N / 2]: each sum finished as
without glu and rounded to out's dtype, then the product in FP32, rounded once. With splits of K, the tile's last
CTA adds both sums in split order, one thread a pair. Its outputs are within one BF16 step (the activation's
exponential) of the two projections followed by `silu_mul`, and a row keeps its bits alone, beside others and in the
prompt tile ([tests/test_glu.py](tests/test_glu.py)). On an RTX 5090, a 34816 x 5120 gate/up matrix in CUDA graphs:
level with the projection and `silu_mul` from 1 to 16 rows (NVFP4 64 / 94 / 152 us at 1 / 4 / 16 rows), 2 to 6% faster
at 64 rows (NVFP4 546 to 513 us, FP8 574 to 564, BF16 603 to 594).

## Structures and knobs

kernel.json lists them; the default knobs run what the kernel ran before it had knobs. Order knobs in bold; plan()
reads decode-tile's and fp4-tile's per matrix (one plan for every row count), so all of them keep one value a matrix.

| Structure | What | Knobs |
| --- | --- | --- |
| decode-tile | `mm_rows`, 1 to 64 rows on mma.m16n8k16, every weight format (the key's weights code; a format is a `Row` decoder over the kernel core's, `tile_parts.cuh`) and activation quantization; up to 16 rows also the staged tile | **GROUPS**, **WARPS**, **SPLITS** (0: plan()'s occupancy rule), SW (0, or the staged tile's warps a CTA: 4, 8), SPF (the staged tile's weights a step ahead) |
| fp4-tile | `mm_rows` with acts nvfp4_mma, the block-scaled FP4 MMA (sm_120 family) | **WARPS**, **SPLITS** (0: the rule), HOIST (steps of weights loaded together) |
| prompt-tile | `mm_prompt`, any number of rows with the decode tile's bits | PW (warps a CTA), PT (rows a CTA) |
| quantizers | `quantize_fp8_rows`, `quantize_nvfp4` | none |

GROUPS, WARPS and SPLITS are arguments of a call, which plan() takes from the build's `_crit_tuning.knobs()`; SW,
SPF, HOIST, PW and PT are template parameters the build's `crit_tuning.h` picks per matrix and row count (a build holds
only the staged tiles its tuning names).

## The staged decode tile

At 9 to 16 rows the register tile's cost is its activations: each warp reads its B fragments (64 bytes a token a
sub-step) from L1 and L2 itself, so a call of 16 rows reads the activations once a warp, four times the weights' bytes
on a 5,120-wide matrix (and the same activations broadcast to every token ran 15-20% faster: the reads, not the MMAs,
were the cost). The staged tile (SW 4 or 8, up to 16 rows) runs the same chains, every MMA with the operands the
register tile gives it in the same order, the same splits and the same reduce, so a row's bits are the register
tile's whatever SW and SPF a row count takes; what changes is where the B fragments come from: a CTA of SW warps
stages each step's activations of every token once in shared memory (cp.async, three steps in flight, one barrier a
step, the 16-byte chunks XOR-swizzled by quad position and token parity so a read phase covers all 32 banks), and its
warps read them from there. SPF loads each step's weights a step ahead (two steps of weights in registers), which the
staged tile's fewer resident warps need at 16 rows. The mechanism (activations staged once a CTA by cp.async in a
multi-stage ring, read back for the MMAs) follows flashinfer's W4A16 GEMM (mm_bf16_fp4, Apache-2.0); no code of it is
used.

## Build, test, tune

```sh
crit kernels build mma-rows                                      # for this GPU, default knobs
crit kernels build mma-rows --arch 80 --arch 86 --arch 89 --arch 90a --arch 100a --arch 120a   # build check
python -m pytest kernels/techniques/projections/mma-rows/tests -q
crit kernels sweep mma-rows --model CHECKPOINT --out TUNING_DIR      # or --shapes FILE, --structure decode-tile
```

The sweep's benches (bench.py) take the checkpoint's NVFP4, FP8 and BF16 matrices (key: format code 0 NVFP4, 1 FP8,
3 BF16, N, K), call the kernel with the plan plan() gives, and check every configuration against an FP64 reference of
the stored values (the FP4 MMA's against the quantized activations' values) and for row invariance before timing it.
Plugins load the kernel with `crit_plugins.kernels.load_kernel("mma-rows", ...)`, built with the engine's tuning
(`CRIT_KERNEL_TUNING`) or the default knobs.

The tests load it with `crit_kernels.build.load("mma-rows")`. `tests/test_mma_rows.py`: the FP64 bounds for every
format with and without splits of K, every row count 1 to 64 alone and batched bit for bit whatever the groups and
warps, the quantizers' bits against the torch quantizers they follow, and graph capture with the counters left zero;
where the FP4 MMA runs, its values against FP64 and the BF16-MMA W4A4 path, its rows' bits at every row count 1 to 64
across warps and offsets, its capture with the pre-pass, and its refusals; a build with other knobs than the defaults
takes its plan from the tuning and gives the default build's bits for the same plan (decode tile, FP4 MMA, prompt
tile). The staged tile (a build holding every SW and SPF, each selected): every format and quantization, row counts 1
to 16, groups, warps and splits of 1 to all of K's steps, a partial last step and N past the last tile, bit for bit
the register tile's, the split counters left zero. `tests/test_prompt_tile.py`: `mm_prompt` against `mm_rows` bit for bit for every format and activation
quantization, splits 1 to the steps of K, rows alone and in decode calls of 2 to 16 and 64, chunks of 1 to 129 rows,
every output type, graph capture; at the role shapes of two checkpoints with plan()'s splits, chunks of 1 to 4,096
rows.

The NVFP4 decode follows a crit-labs engine's tensor-core NVFP4 GEMV (Apache-2.0), generalized to every format, split
K and the quantized activations.
