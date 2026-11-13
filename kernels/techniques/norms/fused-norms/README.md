# fused-norms

The elementwise work around a transformer's matmuls, one Triton kernel per call, for crit's fused-norms technique (its
`norms` slot): RMSNorm with a residual added in place (an FP32 residual stream beside 16-bit rows), an offset weight (a zero-centred weight stored as weight - 1) and a
chosen output dtype; query/key head norms, alone or with rotary positions; the gated RMSNorm of linear-attention
outputs; rotary positions (half or interleaved pairs, partial rotation, inverse frequencies or a cos|sin table); SiLU,
GELU and sigmoid products; L2 norms; the mixes of several residual streams: a gated, weighted or plain sum over the
streams and one row added into every stream with a weight per stream; the softplus log-decay of gated delta layers;
and fused forms where a layer runs two of these back to back (its queries and keys; a stream add and the norms after).

**Why it exists.** Written in torch, each of these is a chain of small kernels (a cast, a square, a mean, an rsqrt,
multiplies, a cast back): a decode round launches hundreds of kernels that each move a few kilobytes, and a prompt pass
reads the same activations several times. Here each call is one launch that reads its inputs once. And torch's own
mean over a row does not keep a row's bits across row counts (a row of a few-row call sums in another order than in a
long call), which an engine whose decode, verify and prompt rows must agree cannot use; every kernel here reduces a
row in an order fixed by its width alone.

**Built from the kernel core** (`kernels/core`): the sigmoid and the activations are `crit_kcore.epilogue`'s.

## Applicability

| | |
| --- | --- |
| Functions | `rms_norm`, `rms_norm_heads`, `rms_norm_rope`, `gated_rms_norm`, `rope`, `silu_mul`, `gelu_mul`, `sigmoid_mul`, `l2_norm`, `stream_sum`, `stream_add`, `softplus_decay` (a gated delta layer's log-decay and beta gate), and fused forms of calls a layer runs back to back, each output with the bits of the separate calls: `rms_norm_rope_pair` and `rope_pair` (a layer's queries and keys in one launch), `stream_add_norm` (a row added into the residual streams, then their norms), `rms_norm_fp8_rows` and `rms_norm_nvfp4` (a norm quantized for a W8A8 or W4A4 projection: the codes and scales of `rms_norm`'s BF16 rows followed by mma-rows' `quantize_fp8_rows` or `quantize_nvfp4`, bit for bit, without writing the rows); signatures and semantics in the package docstrings. Every norm weight may be one row per group (`[groups][dim]` for x `[..., groups, dim]`: a weight per head or per residual stream) |
| Dtypes | BF16, FP16, FP32 inputs, each read in its own dtype (a residual or gate may differ from the rows); FP32 inside; one rounding to `out_dtype` (BF16, FP16, FP32) |
| Layouts | the last dimension contiguous (`out` too), the other dimensions any strides (a query beside its gate in one projection output, a slice of a fused projection); layouts whose leading dimensions do not merge into two levels, or whose base address or row stride is not a multiple of 16 (a width of 4,100, an odd token stride), are copied into aligned buffers first, so a launch never specializes on the rows given |
| GPUs | NVIDIA GPUs Triton compiles for (sm_80 and newer; checked on sm_120); on sm_90 and newer the activation products and one-pass long rows launch with programmatic dependent launch (the kernel waits for the one before with griddepcontrol.wait before it reads anything) |
| Shapes | any row count; rows up to 16,384 values in the rows kernel, wider rows (norms and gates) in the long-rows kernel: held in registers and read once up to 32,768 values, wider in two passes of 4,096; rope heads up to 16,384 values |
| Numerics | within FP32 rounding of a float64 reference with FP32 out, within one output rounding with 16-bit out (`tests/test_fused_norms.py`); the rotary angle is `fp32(position) * fp32(inverse frequency)`, as checkpoints' reference code computes it; with a residual, the sum as stored in the residual's dtype is what is normalized |
| Determinism | same bits on a repeat; a row's bits independent of the rows beside it and of the row count (1 to 8 rows and 320 checked at four widths); CUDA graph replay equal to eager calls |
| Capture | no host synchronization, nothing allocated but the result (or `out=`); a shape's first call compiles, so call each shape once before capturing a graph |
| Measured speed | RTX 5090, default knobs, `benchmarks/shapes.py`, GPU time per call in a CUDA graph against the torch chain a model writes by hand (BF16 rows, FP32 residual): Qwen3.8-27B's shape (hidden 5,120, 24 + 4 heads of 256, 64 rotary values) at 1 / 8,192 rows: residual + RMSNorm 2.4 / 331 us (chain of 9 kernels: 11.2 / 1,159), q and k head norm + rope 5.0 / 156 us (46 kernels: 58.7 / 1,722), gated RMSNorm 1.7 / 193 us (13: 17.9 / 2,543), SiLU product 1.7 / 543 us (2: 3.2 / 914); Nanbeige4.3-33B's (7,168, 56 + 8 heads of 128): q and k rope 3.5 us a decode row (26 kernels: 34.2); four residual streams of 2,560 (norm, gated mean, a row added to each) 5.9 us a decode row (11 kernels: 14.1). The fused forms in a 27B target pass at 4 to 16 rows: q and k of 16 layers 25 us in pairs against 49-50 us apart; the decay and beta of 48 gated delta layers 74 us against 350-365 us as six torch kernels a layer. At 8,192 rows the norms and products run at about the memory rate. Host: Triton's launch path, 25-85 us of CPU time a call on a loaded machine |

## Knobs

The launch constants are knobs (`kernel.json`), read per call shape from the build's tuning (`_crit_tuning.py`, written
by the build from a sweep's table, or the defaults, which are the constants the kernel had before it had knobs: an
untuned build launches exactly what it launched then).

| Structure | Kernel, functions | Key | Knobs (default) |
| --- | --- | --- | --- |
| `rows` | `rows_kernel` without rope: `rms_norm`, `rms_norm_heads`, `gated_rms_norm`, `l2_norm`, `stream_add_norm`; `rows_quant_kernel` (`rms_norm_fp8_rows`, `rms_norm_nvfp4`) on the same knobs, so its rows have `rms_norm`'s bits | `D` | `ELEMENTS` values a program holds, HB = ELEMENTS / BD rows (4,096); `SPAN` values a warp covers, warps = BD x HB / SPAN within 1-16 (512). Both order knobs |
| `rope` | `rows_kernel` with rope and `pair_kernel`: `rope`, `rms_norm_rope`, `rope_pair`, `rms_norm_rope_pair` | `D` | `ELEMENTS` (4,096; HB = ELEMENTS / the row's block whatever the head count, so a pair's tensors keep their separate calls' bits), `SPAN` (512). Both order knobs |
| `long-rows` | `long_rows_kernel`: the norms of rows wider than 16,384 values | `D` | `CHUNK` (0: the row's block in one pass up to 32,768 values, else 4,096 values a step in two passes; a CHUNK at least the row's block: one pass), `WARPS` (0: 16 in one pass, 8 in two). Both order knobs. `PDL` (3, one-pass rows): programmatic dependent launch, speed only |
| `act-mul` | `act_mul_kernel`: `silu_mul`, `gelu_mul`, `sigmoid_mul` | `N` | `BLOCK` values a program (1,024; by row count, `default_rows`: 512 at 1-8 rows, 2,048 past 96), `WARPS` (4; 8 past 96 rows), `PDL` programmatic dependent launch (0 none; 1, 2, 3: wait for the kernel before, let the next start after the program's loads, its stores or its exit; untuned: 1 up to 32 rows, 3 to 96, 2 past) |
| `stream-sum`, `stream-add` | `stream_sum_kernel`, `stream_add_kernel` | `S`, `D` | `BLOCK` (1,024), `WARPS` (4) |
| `softplus-decay` | `softplus_decay_kernel` | `N` | `BLOCK` (1,024), `WARPS` (4) |

Order knobs set how a row is laid out over a program's threads and so the order its sum of squares is added in: they
are looked up by the width alone (never by the row count), so a tuning cannot make a row's bits depend on the rows
beside it, and the sweep keeps one value of them per width. The elementwise structures' knobs change speed only (the
tests check those bits equal the defaults'), so a tuning may pick them per row count. The queries and keys in one
launch must keep the bits of the separate calls: the `rope` bench drops a configuration where they differ.

The long-rows defaults changed in round 4 (one pass in registers where the two passes re-read the row): a long row's
bits are another order than before (within the same bounds), still a function of its width alone. The activation
products' knobs and PDL change speed only: their bits are the same in every configuration.

## Yardstick

Each structure's `yardstick` (kernel.json), which `crit kernels sweep` times on the same inputs beside the tuned kernel
(`library_us`, `vs_library` = library time / tuned time): FlashInfer 0.7.3 where it computes the operation, torch ops
where no library does. `rows`, `long-rows`: `gemma_fused_add_rmsnorm` with a residual (in place, on a BF16 copy of the
FP32 residual: FlashInfer keeps the residual in the rows' dtype, so it moves 8 bytes a value where this kernel moves 12)
or `gemma_rmsnorm` without, the weight in BF16. `rope`: `gemma_rmsnorm` of the queries and of the keys, then
`apply_rope_pos_ids_inplace` (three launches; no FlashInfer call fuses head norms with 1-D rotary positions).
`act-mul`: `silu_and_mul` (programmatic dependent launch on, as FlashInfer runs it by default on sm_90 and newer). `stream-sum`, `stream-add`, `softplus-decay`: the torch ops a model writes (two to five
kernels); no library computes a weighted stream sum, a weighted stream add or a gated delta layer's decay and beta on
their own.

Measured on an RTX 5090 (one short sweep, `--shapes`, CUDA graph replays for the kernel), Qwen3.8-27B's widths unless
named, tuned / library in us:

| Structure | Shape | Rows: tuned / library (vs_library) |
| --- | --- | --- |
| `rows` | 5,120 with an FP32 residual | 1: 4.1 / 4.1 (1.0); 512: 14.3 / 8.2 (0.57); 4,096: 154 / 87.4 (0.57) |
| `rows` | 40 heads of 128 a token | 1: 3.4 / 4.1 (1.20); 512: 6.1 / 10.2 (1.67); 4,096: 32.7 / 63.5 (1.94) |
| `long-rows` | 20,480 | 1: 6.1 / 6.1 (1.0); 64: 6.1 / 8.2 (1.33); 512: 24.5 / 36.8 (1.50) |
| `rope` | q and k norms + rope, 24 + 4 heads of 256, 64 rotary values | 1: 4.1 / 10.2 (2.5); 512: 10.2 / 17.7 (1.74); 4,096: 62.0 / 78.8 (1.27) |
| `act-mul` | SiLU product of 17,408 | 1: 3.8 / 4.1 (1.07); 512: 24.5 / 49.1 (2.0); 4,096: 260 / 516 (1.98) |
| `stream-sum` | 4 streams of 2,560 | 1: 4.1 / 12.3 (3.0); 4,096: 55.3 / 288 (5.2) |
| `stream-add` | 4 streams of 2,560 | 1: 4.1 / 10.2 (2.5); 4,096: 80.5 / 357 (4.4) |
| `softplus-decay` | 48 decay heads | 1: 4.1 / 15.4 (3.75); 4,096: 5.5 / 26.0 (4.7) |

The one structure behind its library is the hidden-width norm with an FP32 residual past 64 rows (FlashInfer on a BF16
residual moves two thirds of the bytes and runs at about 1.9 TB/s against this kernel's 1.6). The sweep's event timer
has about 2 us steps at these times.

Round 4, one timing script (a CUDA graph of 100 calls on the same inputs, the lower of two rounds of five replays; the
calls of a decode round are about 1 us each, below the sweep timer's steps), untuned builds, crit / FlashInfer in us:

| Call | Rows: crit / library (vs_library) |
| --- | --- |
| `silu_mul` of 17,408 (packed BF16 halves) | 1: 0.80 / 0.91 (1.13); 4: 0.82 / 0.93 (1.12); 8: 0.89 / 0.95 (1.07); 16: 0.96 / 1.01 (1.04); 64: 1.68 / 1.70 (1.01); 128: 2.67 / 2.79 (1.04); 512: 8.09 / 8.12 (1.00); 2,048: 127.7 / 134.2 (1.05); 8,192: 541 / 548 (1.01) |
| `rms_norm` of 5,120, BF16 weight, no residual | 1: 1.29 / 1.32 (1.02); 4: 1.42 / 1.60 (1.13); 16: 1.44 / 1.66 (1.15); 128: 1.83 / 1.80 (0.99); 2,048: 8.92 / 9.63 (1.08); 8,192: 109.0 / 109.4 (1.00) |
| `rms_norm` of 20,480 (long rows), no residual | 1: 2.5 / 2.0 (0.81); 64: 3.4 / 2.9 (0.85); 512: 11.4 / 11.6 (1.02); 4,096: 217 / 220 (1.01) |

Without programmatic dependent launch the products took 1.05-1.17 us at 1-16 rows (0.85-0.97 of FlashInfer, which
overlaps each call's start with the end of the one before); with it they lead from 1 to 16 rows. At 512-768 rows with
every input in L2 (the same 50-80 MB read 100 times) they are 0.96-1.0 of FlashInfer. Long rows: one row a program in
one pass took 0.59-0.69 -> 0.81-0.85 at 1-64 rows and 0.90-0.97 -> 1.01-1.02 at 512-4,096 rows; at a few rows the
single program a row is bound by what one SM moves (FlashInfer's one program a row moves the same bytes about 0.5 us
sooner); with an FP32 residual (12 bytes a value against FlashInfer's 8 on a BF16 copy) 0.45-0.65.

## Quantizing norms

A W8A8 or W4A4 projection quantizes its activations first. After a norm, that is a second pass over the hidden rows:
the norm writes them, the quantizer reads them back. `rms_norm_fp8_rows` and `rms_norm_nvfp4` write the codes and
scales in the norm's own launch. Their norm is `rows_kernel`'s on the `rows` knobs, rounded to BF16 as `rms_norm`
writes it, and their quantization is that of mma-rows' quantizers (the FP8 row scale is the row's absolute maximum
/ 448, at least 1e-12; NVFP4 takes one E4M3 scale a 16-value block under the input scale, in cuBLASLt's 128 x 4
tiled layout). [tests/test_norm_quant.py](tests/test_norm_quant.py) checks codes, scales and residual against
`rms_norm` followed by the quantizer, bit for bit, with NaN rows and rows of zeros. On an RTX 5090, in CUDA graphs
with a residual, both formats are level with the two launches at rows of 5,120 values (about 4.5 us from 1 to 64
rows) and faster where the rows are longer or more: 8,192 values, 64 rows 5.7 to 4.4 us (FP8); 2,048 rows
145 to 114 us (FP8) and 140 to 83 us (NVFP4).

## Build, test, tune

```sh
crit kernels build fused-norms                    # stage the package with the default knobs (Triton: no compile step)
python -m pytest kernels/techniques/norms/fused-norms/tests  # on the target GPU
crit kernels sweep fused-norms --model CHECKPOINT --out TUNING_DIR   # or --shapes FILE {structure: [shape, ...]}
```

`tests/test_fused_norms.py` checks every function against float64, rows alone and batched, strided views against
their copies, `out=` and in-place rope, a captured graph against eager calls, the defaults against the constants from
before the knobs, and a build with other knobs (`crit_kernels.build.load("fused-norms", configs=...)` and `select`)
within the same bounds, its rows' bits independent of the row count and its elementwise kernels' bits the defaults'.
`bench.py` holds the sweep's benches, one a structure, with the
model's widths: its norm weights (the hidden width with an FP32 residual, narrower ones as several heads a token), its
head size and head counts, MLP widths, residual streams and decay heads. A plugin loads the package with
`crit_plugins.kernels.load_kernel("fused-norms", folder=...)`, built with the engine's tuning (`CRIT_KERNEL_TUNING`)
or the defaults. `benchmarks/shapes.py` times every function against the torch chains a model writes by hand at three
models' shapes.
