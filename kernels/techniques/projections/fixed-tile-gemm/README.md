# fixed-tile-gemm

A row-invariant prompt GEMM from [crit](https://github.com/crit-labs/crit)'s quantized-projections technique: `x [M][K]`
times `dense [N][K]` (both BF16 or both FP16) on tensor cores with FP32 accumulation, in Triton, in fixed tiles with one
K loop whatever `M` is: no autotuning and no split of K.

**Why it exists.** A provider's GEMM picks its kernel, and so the order of each output's sum, by the row count. A prompt
row's logits then depend on which other rows shared its GEMM, and an engine that batches the prompts of several
sequences into one pass gives each a result different from a pass of its own. Here a row's result is the same bits
whatever rows share the call, so batched prefill matches serial prefill bit for bit. crit's quantized-projections plugin
runs it for BF16 matrices and for NVFP4 ones decoded to BF16 (`prompt_gemm: fixed_tiles`) and reports the rows'
invariance in the engine's facts.

**Built from the kernel core** (`kernels/core`): the grouped tile order is `crit_kcore.load.grouped_tile`.

## Applicability

| | |
| --- | --- |
| Operands | `x [M][K]`, `dense [N][K]`, both BF16 or both FP16, unit column stride; any `M`, `N`, `K` |
| Output | `[M][N]` in x's dtype or the one asked for (FP32 for logits) |
| GPUs | NVIDIA with Triton's tensor-core `tl.dot` (sm_80 and newer) |
| Tiles | untuned: 16 x 64 x 64 (4 warps, 6 stages) at 1 to 32 rows; else 128 x 128 x 64 (8 warps, 3 stages), or 64 x 64 x 64 (4 warps, 4 stages) where the large tile would not fill the device twice over (a tuning sets the tile, warps and stages per shape and row count, see below): the tile depends on the row count, and the two tiles give a row the same bits (one K loop of 64, the same dot), verified on sm_120 by the tests; run them on another GPU before relying on it there |
| Numerics | FP32 accumulation of the 16-bit products, within 1e-5 of the FP64 product relative to its largest output (`tests/test_fixed_tile_gemm.py`) |
| Determinism | a row's bits independent of the call's other rows and of the tile switch; the same bits on a repeat |
| Measured speed | RTX 5090, 4,096 rows: 6-12% slower than torch's GEMM on the same operands (measured inside crit before the move) |
| Yardstick | cuBLAS (torch.matmul) on the same BF16 operands, its heuristic's kernel at each row count (kernel.json `yardstick`, the sweep's library_us). RTX 5090, `crit kernels sweep` on two matrices, tuned vs cuBLAS in us at 16 / 128 / 512 / 2048 / 8192 rows: 34816 x 5120 229 / 289 / 915 / 3375 / 14037 against 225 / 298 / 952 / 3482 / 16034 (0.98 / 1.03 / 1.04 / 1.03 / 1.14x cuBLAS's speed); 5120 x 17408 137 / 157 / 477 / 1772 / 7217 against 127 / 153 / 492 / 1889 / 7301 (0.92 / 0.97 / 1.03 / 1.07 / 1.01x). Tuned, the fixed tile matches or beats cuBLAS from 512 rows and is 2-8% behind it at 16-128 rows |

## Structures and knobs

kernel.json lists them; the default knobs run MM_TILE with the switch to MM_SMALL, as before knobs.

| Structure | What | Knobs (order knobs in bold) |
| --- | --- | --- |
| mm | the fixed-tile GEMM, any row count, row invariant | BLOCK_M (16-256), BLOCK_N (32-256), **BLOCK_K** (32, 64, 128), WARPS (2, 4, 8), STAGES (2-6), SMALL (1: MM_SMALL's tile where the knobs' tile would not fill the device twice over); default_rows: 1-32 rows 16 x 64, 4 warps, 6 stages, no switch |
| mm-split | `mm_split`: the same GEMM with K split over SPLIT programs a tile, FP32 partials added in split order by a second pass, one rounding; NOT row invariant (the split may follow the row count), deterministic | the tile knobs as mm's, SPLIT (0: up to 4, at least 1,024 values of K each, as many as bring the tiles to the device's multiprocessors); default_rows: 1-32 rows 16 x 64 (4 warps, 6 stages), 33-96 rows 64 x 64 (4 warps, 4 stages) |
| cuda-mm | `mm_cuda`: the CUDA GEMM (below), row invariant whatever the knobs | TM (16-256), TN (64-256), WM, WN (warps along each), BK (64, 128 bytes of K a stage), STAGES (2-4), GROUP, EPI (0: stores from registers, 1: through shared memory); default_rows: 1-32 rows 16 x 64 (4 warps), 33-256 rows 64 x 128 (8 warps), 4 stages |
| cuda-mm-split | `mm_cuda_split`: the CUDA GEMM with K split over SPLIT CTAs a tile (FP32 partials added in split order, one rounding); NOT row invariant, deterministic | cuda-mm's and SPLIT (0: mm-split's rule); default_rows: 1-32 rows as cuda-mm's, 33-96 rows 64 x 128 (8 warps, 4 stages), the defaults from 97 rows (the split fills narrow matrices) |

`tile(M, N, K, device)` gives a call's launch parameters from the build's `_crit_tuning.knobs("mm", {N, K}, M)`. The
tile, warps, stages and the switch choose which outputs a program computes and how its loads are staged, never an
output's order (one K loop, the same dot); BLOCK_K is the K loop's step and an order knob, the same for every row count
of a shape. The bench's `valid()` leaves out tiles whose FP32 accumulators exceed 128 a thread or whose stages exceed
the device's shared memory.

Round 4, untuned, one timing script (a CUDA graph of 10 calls cycling over 2-4 copies of the matrix so the weights
come from DRAM, the lower of two rounds), crit / cuBLAS (torch's F.linear) speed at Qwen3.8-27B's widths:

| Matrix (N x K) | 16 | 32 | 64 | 128 | 512 | 2,048 | 8,192 rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 34816 x 5120, mm / mm_split | 1.01 / 1.01 | 1.04 / 1.04 | 1.01 / 1.02 | 0.97 / 0.99 | 0.94 | 0.90 | 1.00 |
| 5120 x 17408 | 1.11 / 1.14 | 1.02 / 1.01 | 0.85 / 0.98 | 0.91 / 1.07 | 0.89 / 1.00 | 0.98 | 0.90 |
| 10240 x 5120 | 1.04 / 1.05 | 1.02 / 1.01 | 1.04 / 1.05 | 0.91 / 1.00 | 0.86 / 0.98 | 0.95 | 0.94 |
| 5120 x 6144 | 1.09 / 1.13 | 1.02 / 1.01 | 0.91 / 1.00 | 0.93 / 1.01 | 0.84 / 0.96 | 0.98 | 0.97 |
| 14336 x 5120 | 1.19 | 1.08 | 1.02 / 1.04 | 0.82 / 0.80 | 0.94 | 0.95 | 0.96 |

The 16-row tile (round 4) puts mm at or above cuBLAS at 16-32 rows on every shape; mm_split's splits put the narrow
matrices (5,120 outputs: 40-80 tiles on 170 multiprocessors, where cuBLAS splits K too) at 0.98-1.14 through 128 rows.
At 128 rows a tile per shape reaches cuBLAS (a short search of mm-split configurations, which a sweep of the
structure finds: 34816 x 5120 128 x 256, one split, 1.05; 5120 x 17408 128 x 128, four splits, 1.06; 10240 x 5120
64 x 128 x 128, 1.06; 5120 x 6144 128 x 64, two splits, 1.07; 14336 x 5120 128 x 256, three splits, 1.02). From 256
rows the tl.dot GEMM stays 0.93-1.01 of cuBLAS even per shape: the row-invariant mm keeps narrow matrices at 0.85-0.93
from 64 rows (no split of K in a row-invariant structure).

## Build, test, tune

```sh
crit kernels build fixed-tile-gemm                                # stages the Triton package with default knobs
crit kernels build fixed-tile-gemm --arch 80 --arch 86 --arch 89 --arch 90a --arch 100a --arch 120a   # the CUDA GEMM build check
python -m pytest kernels/techniques/projections/fixed-tile-gemm/tests -q
crit kernels sweep fixed-tile-gemm --model CHECKPOINT --out TUNING_DIR   # or --shapes FILE
```

Triton compiles the kernel on first use for the current GPU. The tests load it with
`crit_kernels.build.load("fixed-tile-gemm")`: the FP64 bound, slices of a 45,056-row call that cross the tile switch bit
for bit, rows of a 5120-wide matrix, the 16-row tile's rows equal to the larger tiles' at real widths, mm_split within
FP32 accumulation, deterministic, and its splits, and three other knob settings (tiles, warps, stages, the switch) with the default
build's bits. Run them on the target before relying on the tile switch there. The sweep's bench (bench.py) takes every
linear weight of the checkpoint as a BF16 matrix, checks each configuration against an FP64 reference and for row
invariance, and times it at 16 to 8192 rows. Plugins load it with `crit_plugins.kernels.load_kernel("fixed-tile-gemm",
...)`, built with the engine's tuning (`CRIT_KERNEL_TUNING`) or the default knobs.

## The CUDA GEMM

`cuda-mm` and `cuda-mm-split` are the same GEMM in CUDA (`torch-ext/fixed_tile_gemm/cuda_gemm.cu`, kernel.json
`"cuda"`: crit's mixed CUDA + Triton build compiles it for the build's GPU as one shared library a variant, T 0 BF16
or 1 FP16, with the build's crit_tuning.h, and the package calls it through ctypes on the caller's stream), built from
the kernel core and fp8-block-rows' native GEMMs: tile.cuh's K-major byte tiles (a 16-bit row of BK bytes holds BK / 2
values; a k-step of 32 bytes is one m16n8k16 MMA, whose A and B fragments are ldsm_a's and ldsm_b2's) in a
STAGES-deep cp.async ring swizzled for conflict-free ldmatrix, fragments read a k-step ahead of the MMAs, persistent
CTAs in tile.cuh's grouped order that issue the next tile's first stages before the epilogue, mma.cuh's m16n8k16 BF16 /
FP16 MMA with FP32 accumulation, and gemm_epilogue.cuh's stores (each FP32 sum rounded once to the 16-bit type, or
FP32). After CUTLASS's multistage GEMM (BSD-3-Clause; written anew).

`mm_cuda(x, dense, dtype=None)`: every output is the MMAs' sum over K in k16 steps in increasing order into one FP32
accumulator, whatever the configuration and the row count: the tile, warps, stages and epilogue choose which outputs
a CTA computes and how its loads are staged, never an output's order, so a row's bits are the same in any call (row
invariant, one configuration a shape usable by pinned prompts; the tests check slices and four configurations bit for
bit). `mm_cuda_split(x, dense, dtype=None)`: K split over SPLIT CTAs a tile, each writing FP32 partials, added in
split order by a second pass and rounded once (not row invariant: the split may follow the row count; deterministic).
Both take BF16 or FP16 operands whose rows are 16-byte aligned with K a multiple of 8 (`cuda_ok`), out in x's dtype or
FP32; quantized-projections' native and pinned prompts call them (`prompt_gemm` fixed_tiles keeps the Triton `mm`).

RTX 5090 (sm_120a), torch 2.14.1, `crit kernels sweep fixed-tile-gemm --structure cuda-mm` (470 configurations) and
`--structure cuda-mm-split` (255: the split's space narrowed to 128-byte stages), none dropped, on Qwen3.8-27B's
16-bit prompt shapes (the bench's BF16 operands, cold L2), tuned vs cuBLAS (torch.matmul, timed as a graph replay),
cuda-mm / cuda-mm-split:

| Matrix (N x K) | 16 | 32 | 64 | 128 | 512 | 2,048 | 8,192 rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 34816 x 5120 | 1.03 / 1.03 | 1.03 / 1.03 | 1.01 / 1.01 | 1.09 / 1.01 | 1.06 / 1.03 | 1.07 / 1.04 | 1.19 / 1.13 |
| 5120 x 17408 | 1.26 / 1.44 | 0.90 / 0.99 | 0.87 / 0.97 | 0.92 / 1.17 | 1.13 / 1.10 | 1.12 / 1.08 | 1.05 / 1.02 |
| 10240 x 5120 | 1.08 / 1.08 | 1.03 / 1.02 | 1.08 / 1.11 | 1.12 / 1.12 | 1.11 / 1.07 | 1.06 / 1.06 | 1.05 / 1.06 |
| 5120 x 6144 | 1.28 / 1.46 | 0.96 / 1.09 | 0.96 / 1.01 | 0.91 / 1.00 | 1.10 / 1.10 | 1.10 / 1.10 | 1.08 / 1.08 |
| 14336 x 5120 | 1.25 / 1.25 | 1.10 / 1.10 | 1.06 / 1.04 | 0.99 / 1.00 | 1.02 / 1.03 | 1.04 / 1.06 | 1.05 / 1.10 |

The untuned build (default knobs and default_rows), one script on the bench's operands, cuda-mm / cuda-mm-split vs
cuBLAS (eager timing, the GPU shared with other work, so a few cells move by 0.1 between runs):

| Matrix (N x K) | 16 | 32 | 64 | 128 | 256 | 512 | 2,048 | 8,192 rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 34816 x 5120 | 1.02 / 1.02 | 0.92 / 0.93 | 1.00 / 1.00 | 0.90 / 0.99 | 0.97 / 0.95 | 0.99 / 0.99 | 1.01 / 0.99 | 1.07 / 1.07 |
| 5120 x 17408 | 1.25 / 1.39 | 0.90 / 0.89 | 0.60 / 0.97 | 0.74 / 1.15 | 0.98 / 1.03 | 1.07 / 1.07 | 1.05 / 1.04 | 0.97 / 0.97 |
| 10240 x 5120 | 1.06 / 1.06 | 1.01 / 1.00 | 0.99 / 1.00 | 1.14 / 1.10 | 0.95 / 1.07 | 1.03 / 1.03 | 1.02 / 1.01 | 1.01 / 0.99 |
| 5120 x 6144 | 1.28 / 1.33 | 0.96 / 0.96 | 0.67 / 1.00 | 0.80 / 1.07 | 1.00 / 0.99 | 1.06 / 1.05 | 1.04 / 1.04 | 1.01 / 1.01 |
| 14336 x 5120 | 1.23 / 1.23 | 0.91 / 0.92 | 1.00 / 0.92 | 0.83 / 0.81 | 0.92 / 0.80 | 0.99 / 0.99 | 1.01 / 1.01 | 0.99 / 0.98 |

Untuned, the split GEMM is at 0.9 of cuBLAS or more but on 14336 x 5120 at 128-256 rows (0.80-0.81: two splits of
the 128 x 128 tile there); the row-invariant one falls to 0.60-0.80 on the narrow matrices at 64-128 rows (its
default 64 x 128 tile: 40-80 CTAs) and 0.83 on 14336 x 5120 at 128: shape-independent default_rows cannot pick the
narrow and the wide matrices' tiles at once; a tuning does (the table above).

The split GEMM is at or above cuBLAS everywhere but 5120 x 17408 at 32 and 64 rows (0.99, 0.97: 178 MB of weights
read at 1.4-1.5 TB/s by both). The row-invariant GEMM cannot split K, so on the narrow matrices (5,120 outputs, 40-80
tiles for 170 multiprocessors) at 32-128 rows it stays at 0.87-0.96 where cuBLAS splits K; from 256 rows, and on the
wide matrices at every row count, it is at or above cuBLAS. Picks: 16 x 64 or 16 x 128 tiles (4 warps, 4 stages) at
16-32 rows, 32-64 x 64-128 to 128 rows, 128 x 128 (8 warps of 64 x 32, 3 stages of 128 bytes, staged epilogue) or
128 x 256 (2 stages) from 512 rows. 128 x 128 at 8,192 rows runs about 210 TFLOPS (34816 x 5120: 13.8 ms).
