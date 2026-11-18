# exl3-rows

Projections of EXL3 checkpoints (Core's `exl3` scheme: trellis codes with sign vectors and a codebook marker), read
at their stored 1 to 8 bits a weight, or 1.5, 2.5 and 3.5 with the MUL1 codebook. The matrix is W [in][out] =
diag(su) H Q H diag(sv): Q the trellis tiles, H the block-diagonal normalized 128-point Hadamard, su and sv the sign
vectors. `gemv` rotates 1 to 16 activation rows (x su H in FP32, stored in FP16), multiplies them by the tiles
decoded on the fly on FP16 tensor cores (a block's tile slices and activations staged in shared memory by cp.async,
each tile decoded once into an MMA's A operand, the rows its B operand, FP32 sums), and rotates the result (y' H sv).
The three launches chain as programmatic dependents (9.0 on), so each starts while the one before finishes. `gemm` takes a prompt's rows through the same tiles kernel in slabs of up to 64, the tiles decoded once a slab
and multiplied by every group of eight rows, so a prompt row has the bits of the same token decoded and the codes
are read at their stored bits once a slab. `dequantize` gives W^T [out][in] in BF16 or FP16 for a GEMM on another
kernel or a library, whose rows are other bits.

**Built from the kernel core** (`kernels/core`): the codebooks, the tile's bitstream and the fragment order of its
positions are `crit_kcore/exl3.cuh`'s, checked against host references in `core/tests/exl3_tests.cu`.

## Applicability

| | |
| --- | --- |
| Inputs | trellis [in / 16][out / 16][16 K] int16 (contiguous); suh [in] and svh [out] FP16 signs (`unpack_signs` for the bitfields su / sv); the codebook: 0 3INST, 1 MCG (a `.mcg` marker), 2 MUL1 (a `.mul1` marker); x [M][in] BF16, FP16 or FP32, M 1 to 16 |
| Outputs | y [M][out] in x's dtype; `decode`: Q [in][out] FP16; `dequantize`: W^T [out][in] |
| GPUs | NVIDIA sm_80 and newer (warp shuffles, `__dp4a`) |
| Shapes | in and out multiples of 128 (both are rotated in 128-value blocks) |
| Numerics | the tiles' values are the format's FP16 values exactly; the rotated rows are rounded to FP16 (the tensor cores' operand), the products exact, the sums FP32, one rounding into the output |
| Determinism | a row's sums run in tile order per split, the four lanes of a column add xor 1 then xor 2, and the splits add in order, all by the shape and device, never the row count: a row has the same bits alone and among up to 16 |
| Accuracy | `decode` equal to the format's decoder bit for bit for every codebook and bitrate (tests: the SHA-256 of decoded seeded tiles); `gemv` within 1e-4 of the output's largest value plus one BF16 rounding of its FP64 reference (FP16 rotated rows times the tiles), and within FP16 activations' rounding of the exact product |

## Interface

```python
from crit_kernels.build import load
er = load("exl3-rows")
w = er.Exl3Matrix(trellis, suh, svh, codebook)   # suh / svh FP16, or er.unpack_signs(su, in) / (sv, out)
y = er.gemv(x, w)                                # 1 to 16 rows
y = er.gemm(x, w)                                # any rows (a prompt), each with gemv's bits
dense = er.dequantize(w)                         # W^T [out][in] BF16
q = er.decode(w)                                 # the tiles' stored FP16 values [in][out]
```

## Structures and knobs

| Structure | Call | Knobs |
| --- | --- | --- |
| gemv | `gemv` | `WARPS` (2, 4, 8: warps a block), `STRIPS` (1, 2, 4: neighbouring tile strips a warp decodes), `ROWS` (1, 2, 4, 8: input rows a stage copies), `STAGES` (2, 3, 4: stages in flight), all speed only (a configuration whose block would take more than 99 KiB of shared memory is not swept); `SPLITS` (0, 1 .. 32: contiguous parts of the input's tiles, one a block row, added in part order; 0 gives 48 warps a multiprocessor with at least two stages a part; an order knob) |

| prompt | `gemm` | none of its own: gemv's configuration for the matrix at 16 rows, eight row groups a tile (`slab_rows(w)`: 64 rows a pass, or 16 where that block would take more than 99 KiB of shared memory); exact, the bits of gemv |

`decode`, `dequantize` and `rotate` have no knobs.

## Measured

RTX 5090 (another engine build shared it, so each figure is the best of 11 replays), CUDA graphs over enough distinct
matrices that every call reads its weights from memory, MCG codebook. The yardsticks: cuBLAS on the dense BF16 matrix
(`torch.matmul`), and the format's own CUDA GEMV on the same codes and signs (its outputs within 1e-3 of this
kernel's, relative to the largest):

| Matrix | Rows | BF16 | 2 bits | format's | 4 bits | format's | 5 bits | format's |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5120 x 17408 | 1 | 111 us | 30 us | 33 us | 37 us | 37 us | 46 us | 46 us |
| 5120 x 17408 | 16 | 123 us | 37 us | 37 us | 47 us | 40 us | 68 us | 49 us |
| 17408 x 5120 | 1 | 114 us | 31 us | 36 us | 39 us | 41 us | 45 us | 50 us |
| 17408 x 5120 | 16 | 129 us | 37 us | 44 us | 47 us | 47 us | 55 us | 56 us |
| 5120 x 5120 | 1 | 35 us | 12 us | 15 us | 15 us | 16 us | 28 us | 32 us |
| 5120 x 5120 | 16 | 40 us | 15 us | 19 us | 19 us | 19 us | 30 us | 30 us |

A tile's words reach shared memory by 16-byte copies with three stages in flight, while the warps decode the stage
before: a plain read of the same bytes on the shared GPU took 19 us at 2 bits and 32 us at 4 bits, so 4 bits is
near the read and 2 bits is bound by the decode arithmetic (with the decode removed, 18.5 us). At 16 rows on wide
matrices the default knobs leave it behind the format's GEMV: the sweep's table per shape and row count narrows it
(5120 x 17408 at 4 bits, 16 rows: 45 us with SPLITS 4).

A prompt's rows (`gemm`, default knobs, one matrix, the best of 10 calls, the GPU shared with other work). The
yardsticks: cuBLAS on a dense BF16 copy that is already there (178 MB a 5120 x 17408 matrix, which a model stored at
4 bits cannot keep for every matrix), and the same after a dequantize a call makes:

| Matrix | Bits | Rows | gemm | BF16 GEMM on a dense copy | with its dequantize |
| --- | --- | --- | --- | --- | --- |
| 5120 x 17408 | 4 | 64 | 142 us | 178 us | 2,635 us |
| 5120 x 17408 | 4 | 512 | 925 us | 550 us | 3,043 us |
| 5120 x 17408 | 4 | 2,048 | 3,479 us | 1,788 us | 4,273 us |
| 5120 x 17408 | 2 | 2,048 | 3,226 us | 1,752 us | 4,222 us |
| 17408 x 5120 | 4 | 2,048 | 3,556 us | 1,896 us | 4,601 us |
| 5120 x 5120 | 4 | 512 | 431 us | 181 us | 963 us |
| 5120 x 5120 | 4 | 2,048 | 1,613 us | 575 us | 1,356 us |

A slab of 64 rows costs about 110 us on the wide matrices where 16 decode rows cost 40 to 47: the tiles' read and
decode are shared by the slab and the rest is its eight MMAs a tile, about 105 TFLOPS. Against a dense copy that is
kept, the exact rows are 1.7 to 2.8 times slower at 512 rows and more; against a copy made for the call they are
faster except on the square matrix at 2,048 rows, and they need no memory beside the codes.

## Build, test, tune

```sh
crit kernels build exl3-rows
python -m pytest kernels/techniques/projections/exl3-rows/tests
crit kernels sweep exl3-rows --model CHECKPOINT --out TUNING
crit kernels gates exl3-rows --model CHECKPOINT
```
