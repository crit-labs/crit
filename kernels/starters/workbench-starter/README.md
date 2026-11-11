# workbench-starter

The kernel workbench's starter matmul: `y[columns][N] = x[columns][K] . w[N][K]^T` in BF16, or in E4M3 with one FP32
scale per row of `w` and per column of `x`, FP32 out. One warp per output row and every column in that warp, FP32
accumulation in a fixed order: a column computed alone equals the same column in a call of several.

**Why it exists.** `crit kernels new NAME --op gemv|gemm --format bf16|fp8 --starter --from <this folder>` copies it
into an engine's workbench folder as a correct starting point (allowed from a run's fourth candidate on); the engine
replaces it with the kernel it writes. It is not fast and not meant to be. The workbench compiles every `.cu` and
`.cpp` it copied with torch's extension builder, so the sources stand alone (no generated header).

## Applicability

| | |
| --- | --- |
| Formats | BF16 x BF16 (`run_bf16`); E4M3 x E4M3 with FP32 scales, x per column and w per row (`run_fp8`) |
| Shapes | x [columns <= 16][K], w [N][K], contiguous |
| GPUs | NVIDIA sm_80 and newer (no architecture-specific instructions) |
| Numerics | FP32 sums of exact products, within 1e-5 of FP64 relative to the largest output (`tests/test_workbench_starter.py`) |
| Determinism | a column's bits independent of the other columns and of the column count, the same on a repeat |
| Structure | `matmul`: no knobs (a starting point, eight warps a block) |
| Measured speed | not measured: a starting point |

## Build and test

```sh
crit kernels build workbench-starter                     # compiles for this GPU into crit's kernel cache
crit kernels build workbench-starter --arch 80 --arch 120 # checks the build for other architectures
python -m pytest kernels/starters/workbench-starter/tests # on the target GPU
```

The tests load the build with `crit_kernels.build.load("workbench-starter")` and check both formats against FP64 and
each column alone against the same column in a call of eight.
