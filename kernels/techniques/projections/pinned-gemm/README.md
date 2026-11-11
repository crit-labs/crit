# pinned-gemm

Prompt GEMMs whose rows do not depend on the row count, from [crit](https://github.com/crit-labs/crit)'s
quantized-projections technique: cuBLASLt with one split-K-free algorithm per GEMM shape and format, chosen once and
run for every row count, for BF16, FP16, FP32, FP8 (tensor or row scales), MXFP8, MXFP4 and NVFP4 operands.

**Why it exists.** cuBLAS's heuristics pick a GEMM's algorithm by its row count, and algorithms split a row's sum
differently (split-K, stream-K, other K tiles). A prompt row then computes other bits when the prompt is longer, is
cut into other chunks, is padded to a captured bucket or shares a pass with another sequence: a prefix scores its
tokens differently from the whole prompt, and where activations are quantized (W4A4, W8A8) one moved ulp flips codes
downstream. Here a plan holds one algorithm and multiplies any row count with it, so a row's bits depend on its own
values, the matrix and the plan only, at cuBLASLt's own speed.

**Built from the kernel core** (`kernels/core`): the blocked block-scale layout's size is `crit_kcore/formats.cuh`'s.

## Applicability

| | |
| --- | --- |
| Formats | `INPUTS[-SCALE]:OUTPUT`: inputs bf16, fp16, fp32 (no TF32), fp8e4m3, fp8e5m2 (or a pair such as fp8e4m3xfp8e5m2), mxfp8 (E4M3, one UE8M0 scale per 32 along K), mxfp4 (E2M1, UE8M0 per 32), nvfp4 (E2M1, UE4M3 per 16, an FP32 global scale alpha read from device memory); FP8 scales `tensor` (one FP32 element an operand) or `row` (FP32 vectors of a's rows and b's rows); outputs bf16, fp16, fp32. FP4 two a byte (the first element in the low nibble); block scales in cuBLASLt's 128 x 4 blocked layout (`blocked_scales`) |
| Operands | a [M, K] row major (activation rows, any M), b [N, K] row major (a weight); out [M, N] row major, or N columns of a wider row-major matrix (one run of a stacked projection written in place) |
| Shapes | what cuBLASLt takes for the format (FP8 and FP4 need 16-byte aligned rows: K a multiple of 16 for FP8, 32 for FP4); `pin` raises `UnsupportedFormat` where the heuristic lists no split-K-free algorithm |
| GPUs | where the process's cuBLASLt offers the format split-K free: FP8 from sm_89, MXFP8, MXFP4 and NVFP4 from sm_100 (RTX 50 series: sm_120). On an RTX 5090 with cuBLASLt 13.1 FP8 with row scales lists none (FP8 unscaled or with tensor scales into FP32 does: row scales then go after the GEMM). Host code only: cuBLASLt is the library torch loaded, opened by its soname |
| Choice | `pin(fmt, N, K)`: of the split-K-free algorithms cuBLASLt's heuristic lists at REP_ROWS rows (2048 untuned), in its order, the CHOICE-th (the first untuned) that is row exact: a call of each of 22 row counts from 1 to 4096 gives the first rows of the 4096-row call, and five slices at offsets give its rows, bit for bit, on random operands (or the caller's weight; the longest check cut to the rows whose output fits 512 MiB). No timing at pin time: the choice depends on the shape, the format, the device, the cuBLASLt version and the build's tuning only, so every process with that tuning makes the same one. Short calls run the same algorithm: another that agrees with it on test operands may still sum in another order (16-bit outputs hide most of a difference), so none is swapped in |
| Numerics | cuBLASLt's: FP32 accumulation (CUBLAS_COMPUTE_32F), scales and alpha in the epilogue, one rounding to the output type. Within FP32 accumulation of the FP64 product of the dequantized operands (`tests/test_pinned_gemm.py`) |
| Determinism | a row's bits are the same at every row count from 1 to 4096, at any offset, and on a repeat (the algorithm has no split of K and no reduction scheme) |
| Capture | a call with `out` given allocates nothing and runs on the current stream without host synchronization: a CUDA graph captures it. A plan checks a row count with cuBLASLt (host only, cached) the first time it meets it |
| Measured speed | RTX 5090, cuBLASLt 13.1, NVFP4 x NVFP4 into BF16 against torch's scaled matmul at the same row count (its heuristic per M), us at 128 / 512 / 2048 / 8192 rows: 34816 x 5120 66 / 196 / 686 / 2815 against 68 / 202 / 699 / 2996; 5120 x 17408 58 / 88 / 318 / 1292 against 60 / 90 / 323 / 1312; 44032 x 7168 141 / 331 / 1211 / 5048 against 147 / 338 / 1286 / 5478; 7168 x 22016 71 / 186 / 607 / 2332 against 76 / 189 / 612 / 2583. FP8 into FP32 (row and column scales applied after by the caller) 16384 x 5120 at 8192 rows 3052 us for the GEMM against 3349 for torch's rowwise scaled matmul. `pin` takes 0.03-0.1 s a shape (the first in a process about 0.4 s) |
| Yardstick | cuBLASLt unpinned through torch on the same operands (torch.matmul / torch.mm with out_dtype for 16-bit inputs, torch._scaled_mm for FP8 and NVFP4; NVFP4's global alpha left out): the heuristic's algorithm at each row count, split-K included (kernel.json `yardstick`, the sweep's library_us). RTX 5090, `crit kernels sweep`, tuned pin vs unpinned in us at 64 / 512 / 4096 rows: BF16 into BF16 34816 x 5120 235 / 891 / 6841 against 236 / 905 / 6901 (1.00-1.02x the library's speed); BF16 into FP32 5120 x 17408 124 / 520 / 3858 against 120 / 474 / 3466 (0.97 / 0.91 / 0.90x: one algorithm a shape, picked for the shape's rows as a whole); FP8 into FP32 16384 x 5120 71 / 211 / 1324 against 61 / 232 / 1560 (0.86 / 1.10 / 1.18x); NVFP4 into BF16 34816 x 5120 70 / 213 / 1402 against 71 / 216 / 1407, 5120 x 17408 58 / 102 / 687 against 59 / 102 / 689 (1.00-1.02x) |

## Structures and knobs

kernel.json lists them; the default knobs pin what the kernel pinned before it had knobs. The structure is the
cuBLASLt kernel family the format runs on; the tuning is keyed by N, K, IN (a's element bits) and OUT (the output's
bits), which `structure(fmt, N, K)` gives.

| Structure | Formats | Knobs (order knobs in bold) |
| --- | --- | --- |
| lt-dense | bf16, fp16, fp32 inputs | **REP_ROWS** (512, 2048, 8192), **CHOICE** (0-5) |
| lt-fp8 | FP8 E4M3/E5M2, unscaled or tensor or row scales | **REP_ROWS**, **CHOICE** |
| lt-block | MXFP8, MXFP4, NVFP4 | **REP_ROWS**, **CHOICE** |

The knobs are host-side choices pin() reads from the build's `_crit_tuning.knobs()` (`tuned(fmt, N, K)`): REP_ROWS
the row count whose heuristic list is searched, CHOICE which of its distinct row-exact split-K-free algorithms is pinned
(the last one where fewer are listed). Both pick the algorithm, and so a row's summation order: a tuning keeps one value
of each a shape, and every row count of the shape runs the one pinned algorithm, so the row-count independence holds
for a tuned build as for an untuned one. Split-K and reduction schemes stay excluded whatever the knobs say.

## Build, test, tune

```sh
crit kernels build pinned-gemm                                   # for this GPU, default knobs
crit kernels build pinned-gemm --arch 80 --arch 86 --arch 89 --arch 90a --arch 100a --arch 120a   # build check
python -m pytest kernels/techniques/projections/pinned-gemm/tests -q
crit kernels sweep pinned-gemm --model CHECKPOINT --out TUNING_DIR   # or --shapes FILE, --structure lt-block
```

It is host code against cuBLASLt (CUDA 13's toolkit and cuBLAS for the row and block scale modes), compiled for the
current GPU. The tests load it with `crit_kernels.build.load("pinned-gemm")`: every format's rows bit for bit at every
row count from 1 to 4096 and at offsets, the values against the FP64 product, a column block of a wider output, a
captured call, the same choice in every pin (with random weights or the caller's), and a non-default knob setting's
pinned algorithm row exact too. The sweep's benches (bench.py) take the checkpoint's matrices in each structure's
formats (NVFP4 into BF16 and FP32, FP8 into FP32, BF16 into BF16 and FP32), pin each knob setting with the bench's
weight, check it against the FP64 product of the dequantized operands and for row invariance, and time it at 64, 512
and 4096 rows. Plugins load it with `crit_plugins.kernels.load_kernel("pinned-gemm", ...)`, built with the engine's
tuning (`CRIT_KERNEL_TUNING`) or the default knobs.

The cuBLASLt calls follow crit's recorded-selection technique (pinned GEMMs, Apache-2.0), with the choice made
without timing and the library opened from the process.

