# gguf-rows

GGUF projections at their stored bits: 1 to 8 activation rows times a matrix in a GGUF block format, read straight
from the checkpoint's blocks, with the activations quantized to int8 per 32 values and each sub-block's integer dot on
`dp4a`; the same over a stack of experts, each row its own expert (the gathered form); any number of rows (a prompt)
on the s8 tensor cores in the same order, so a prompt's rows have the decode rows' bits; and the dequantization of a
matrix.

**Why it exists.** An engine on a GGUF checkpoint otherwise decodes every quantized matrix to 16 bits (reading and
writing it once a token) or writes these kernels itself. Here the weights stay in their blocks, every format shares one
kernel through the kernel core's decoders (`crit_kcore/gguf.cuh`: a new format is a decoder), and every row reduces in
one declared order, so a row's bits are the same alone and beside up to seven others, and a gathered row equals the
plain call on its expert.

## Applicability

| | |
| --- | --- |
| Weight formats | GGUF Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4: rows of blocks as the file stores them, `uint8 [N][row bytes]` (or `[experts][N][row bytes]`), K a multiple of the format's block (32 or 256 values) |
| Activations | FP32, BF16 or FP16, quantized by the kernel to int8 per 32 values: 1 to 8 rows (`gemv`), any rows (`gemm`) |
| GPUs | NVIDIA sm_80 and newer (`dp4a`); not HIP |
| Numerics | `dequantize` in FP32 equals the format's own reference decoder bit for bit (and BF16 / FP16 one rounding of it); `gemv` and `gemm` are the declared order of `gguf_rows.h` bit for bit (bench.py's `product`, in torch on the host): a prompt row equals the same row decoded |
| Determinism | same bits on a repeat; a row's output independent of the rows beside it and, gathered, of the other rows' experts |
| Measured speed (gemm) | RTX 5090, 14336 x 4096, cold L2, default knobs (WN 4, WT 2, TB 4, RUN 2, STAGES 2), the call's staging and quantizing included; 512 tokens: Q8_0 107 TOPS, Q4_K 95, Q6_K 92, IQ4_NL 111, IQ3_S 112, IQ4_XS 111; 2,048 tokens: Q8_0 128, Q4_K 110, Q6_K 107, IQ4_NL 129, IQ3_S 129, IQ4_XS 129. Dequantizing to BF16 and cuBLAS's BF16 GEMM on the same matrices: 50-52 TFLOPS at 512 tokens (gemm 1.8 to 2.2 times faster), 119-121 at 2,048 (gemm 1.06 to 1.08 times faster for the formats with one scale, 0.90 to 0.92 for Q4_K and Q6_K, whose terms take a minimum or two scales) |
| Measured speed, the other formats | the same RTX 5090 and setup, warm weights for gemm. gemm at 512 and 2,048 tokens: Q4_0 118 and 132 TOPS, Q5_0 117 and 131, MXFP4 118 and 132, IQ2_XXS 119 and 132, IQ3_XXS 119 and 131, IQ1_S 117 and 126-131 (one scale a sub-block: 1.07 times cuBLAS at 2,048); Q4_1 99 and 112, Q5_1 87-98 and 111, Q5_K 98 and 111, Q3_K 97 and 109, Q2_K 91 and 105, IQ2_XS 97 and 109, IQ2_S 96 and 109, IQ1_M 94-96 and 106 (two scales or a minimum: 0.86 to 0.92 times cuBLAS at 2,048, 1.7 to 2.3 times it at 512). gemv, 1 row on 14336 x 4096 and 4096 x 14336: Q4_0 1,355-1,365 GB/s, Q4_1 1,325-1,414, Q5_0 1,364-1,400, Q5_1 1,383-1,442, MXFP4 1,162-1,208, Q5_K 1,143-1,209, Q3_K 882-914, Q2_K 1,013-1,089, IQ3_XXS 1,096-1,130, IQ2_S 953-1,018, IQ2_XS 942-998, IQ2_XXS 930-948, IQ1_M 754-796, IQ1_S 732-771 (at 1.6 to 2.6 bits a weight the decode, not the read, sets the time: IQ1_S's 11.5 MB take 15 us against 7.7 us of reading) |
| Memory (gemm) | a staging buffer a call: K + K / 4 bytes a weight row (codes and two planes of scales; Q2_K three), the matrix's rows taken in chunks that fit 128 MiB (all of a 14336 x 4096 matrix: 75 MiB); half of what dequantizing to BF16 holds |
| Measured speed (gemv) | RTX 5090 (crit eval probe: 1,487 GB/s copy bandwidth), weights read cold (matrices rotated past L2), the call's three launches timed in a CUDA graph: 1 row on 14336 x 4096 and 4096 x 14336 reads Q8_0 at 1,357-1,429 GB/s, Q4_K 1,236-1,257, Q6_K 1,247-1,250, IQ4_NL 1,124-1,177, IQ3_S 1,127-1,130, IQ4_XS 1,056-1,128 (71-96% of the probe); 4 rows 742-1,348 GB/s, 8 rows 545-1,032 GB/s; a 2816 x 2816 matrix (3.4-8.4 MB) takes 7.7-13.4 us for 1 row |

## The declared order

1. Each activation row's 32-value blocks become `q = rint(x / d)`, `d = max |x| / 127` in FP32 (`q = 0` where `d` is
   0; a NaN in a block makes `d` NaN), with the block's sum of `q`.
2. A matrix row's 32-value sub-blocks `j`, in order, each give an FP32 term from the exact integer dot of the stored
   codes and `q`: `fmul(fmul(s, d), dot)`; with two scales a sub-block (16-value scales: Q2_K, Q3_K, Q6_K, IQ2_XS,
   IQ2_S, IQ1_M) the sum of one such
   term a half; less `fmul(fmul(m, d), sum)` where the format has a minimum (Q4_K, Q5_K; Q4_1 and Q5_1, whose stored
   offset is added, with `m` its negation); with a minimum a 16 values (Q2_K) less `fmul(fmul(m0, d), first)`, then
   less `fmul(fmul(m1, d), sum - first)`, `first` the exact sum of the block's first 16 `q`. `s` and `m` are the stored
   FP16 scale times the sub-block's integer scale, rounded once (the IQ2 and IQ3_XXS scales as their formats compute
   them, `fmul(fmul(d, 0.5 + l), 0.25)` or `0.5`; IQ1's codes are `8 g +- 1` over `d l / 8`, exactly its values).
3. The sub-blocks split into `splits` contiguous parts; in a part, sub-block `j` goes to lane `(j - start) % 32`, each
   lane adds its terms in order, the lanes reduce by the xor butterfly (16, 8, 4, 2, 1), the parts add in part order,
   and the sum rounds once to the output's dtype. `splits` depends on the matrix's shape and the device (or the
   tuning's `SPLITS`), never on the rows.

## Interface

```python
from crit_kernels.build import load      # a plugin: crit_plugins.kernels.load_kernel("gguf-rows", folder=...)
gr = load("gguf-rows")
w = gr.GgufMatrix(blocks, "Q4_K", k)     # blocks: uint8 [N][row bytes] on the GPU, as the GGUF file stores the rows
y = gr.gemv(x, w)                        # x [1..8][K] -> y [M][N] in x's dtype
y = gr.gemv(x, experts, ids)             # experts over [E][N][row bytes], ids int32 [M]: row m times expert ids[m]
y = gr.gemm(x, w)                        # any rows on the s8 tensor cores, gemv's bits a row (matmul picks either)
dense = gr.dequantize(w)                 # W [N][K] BF16 (torch.float32: the stored values exactly)
```

## Knobs (structure `gemv`)

| Knob | Values | Effect |
| --- | --- | --- |
| `WARPS` | 2, 4, 8 | warps a block, one output column a warp |
| `SPLITS` | 0, 1, 2, 4, 8, 16 | contiguous parts of a row's sub-blocks (0: enough warps for 32 a multiprocessor, at least 32 sub-blocks a part); an order knob |

Structure `gemm` (the prompt rows, 9 rows and more), in two steps a call. First every sub-block of the matrix is
decoded once, as gemv decodes it (the kernel core's decoders: int8 codes and FP32 scales, bit for bit), into a
staging buffer laid out sub-block by sub-block (codes `[J][rows][32]`, scales `[2][J][rows]`), a chunk of rows at a
time; the activations are quantized in the same sub-block-major layout. Then a CTA takes 16 WN weight rows x 8 TB WT
tokens, WN x WT warps of 16 rows x 8 TB tokens each, the token tiles along the grid's x so that CTAs beside each other
share a weight tile. Each part's sub-blocks are visited lane by lane in the order the 32-lane tree consumes the lanes
(`crit_kcore/reduce.cuh`'s `lane_at32`), every lane's sub-blocks in increasing order, RUN of them a stage; a stage's
codes, scales and tokens are copied into shared memory by cp.async, STAGES - 1 stages ahead of its MMAs; one s8
m16n8k32 MMA a 16 x 8 tile gives the exact dots (two m16n8k16 for Q6_K's 16-value scales), the term goes onto the
output's running lane as gemv adds it, and a finished lane folds into the tree's partial sums (`fold32`'s binary
counter): an output holds its running lane and five partial sums instead of 32 lanes. Neither the chunks nor the tile
change an output's bits.

Why it stages: decoding inside the GEMM's CTAs repeats each sub-block's decode in every token tile (16 times for
2,048 tokens at 128 a tile), and copying the scattered bytes a K-quant sub-block's decode reads (a superblock's
scales, its run of codes) as 4-byte copies was the slowest part of the kernel: an RTX 5090 took 70 ms on Q6_K's
copies alone at 14336 x 4096 and 2,048 tokens, against 2.3 ms for the whole call now.

| Knob | Values | Effect |
| --- | --- | --- |
| `WN` | 1, 2, 4 | warps across the weight rows, 16 rows each |
| `WT` | 1, 2, 4 | warps across the tokens |
| `TB` | 1, 2, 4 | 8-token MMA tiles a warp |
| `RUN` | 1, 2, 4 | sub-blocks of a lane a stage (one barrier a stage) |
| `STAGES` | 2, 3 | stages in flight, STAGES - 1 copied ahead by cp.async |

`crit kernels sweep` on an RTX 5090 (14336 x 4096 Q4_K, 4096 x 14336 Q6_K, 4096 x 4096 Q8_0) kept the defaults from 256
rows up; at 16 and 64 rows smaller tiles (TB 1 or 2, WN x WT 2 x 2 or 4 x 1) are 1.2 to 1.5 times faster, which an
engine's tuning picks.

## How it was checked

`tests/test_gguf_rows.py` on an RTX 5090, for every format (20 of them) on three shapes: the FP32 dequantization and bench.py's host
decoders against gguf-py's `dequantize` bit for bit; `gemv` against the declared order bit for bit at 1, 2, 3, 5 and 8
rows, in BF16, FP16 and FP32, and at 1, 2, 3 and 8 parts; `gemm` against `gemv` and the declared order bit for bit
at 1, 9, 33 and 100 rows, and staged in chunks against one staging; a row alone against the same row among others; a gathered row
against the plain call on its expert; a NaN reaching its row only; and the shapes and formats it refuses.
`crit kernels gates gguf-rows --shapes FILE` (nonfinite, repeat, edges, sums, limits) and the build for every
architecture in kernel.json.

No library computes this arithmetic (int8 activations per 32 values against the stored GGUF blocks in a fixed order),
so the structure has no yardstick: its report gives the achieved GB/s against the device's.
