# Low-bit and row-scaled FP8 projection

Represent weights with signed integer codes and one FP32 scale per group, then project
using exactly that representation. The entry defines one complete small codec, plus a
native E4M3FN/BF16 row codec with a double-accumulating oracle, a fixed-G32 C++
integer codec and an NVFP4 reference codec. Encoded model formats are interchangeable
only when their exact word, scale and layout semantics match.

## When to use it

Use it to reason about weight bandwidth, packing and numerical identity, or as a host
oracle for an implementation of one of these exact codecs. An existing encoded artifact
needs its own decoder unless its layout and arithmetic match. Quantizing weights reduces
traffic, but decode cost and approximation may outweigh the saving for a workload.

## How it works

For an `R x W` matrix and group width `G` (`W % G == 0`), per row group:

```text
L = 2^(bits-1) - 1                       bits is 4 or 8
s = round_to_FP32(max(abs(group)) / L)    an all-zero group uses s = 1
q_i = clamp(round_ties_to_even(w_i / s), -L, L)
reconstructed_w_i = signed(q_i) * s
```

Codes derive from the represented FP32 scale. Scales that cannot be positive finite
FP32 are rejected, and externally built `QuantizedRows` must already hold represented
FP32 values (use `represented_scale(value)` to round one). The encoder uses `[-7, 7]`
or `[-127, 127]`; the decoder accepts the full signed code space. Rows are flattened
row-major; 8-bit codes take one byte, 4-bit codes put even elements in the low nibble
and odd in the high. Only the last byte of the matrix may be padded. The scale index
of flattened element `i` is `i // G`. The host projection computes
`y_r = sum_c reconstructed_w[r,c] * activation[c]` without activation quantization.

**FP8 rows.** `RowWeights` hold `K` E4M3FN code bytes and one BF16 multiplier per row;
the weight is `decode(code) * decode_bf16(scale)`. Codes `0x7f` and `0xff` (NaN) are
invalid; scales are nonnegative finite BF16, not negative zero, and a zero scale
requires only signed-zero codes. `fp8::quantize` rounds `maxabs/448` to BF16, then
rounds codes to nearest-even E4M3FN with finite saturation (all-zero rows get zero
scales; an underflowed nonzero scale becomes BF16 word `0x0001`).

**G32 integers.** [grouped_int.h](code/grouped_int.h) uses the same signed layout with
`G = 32`. `quant::Rows` validates extents and scales once and borrows owner-held
storage; `project` takes token-major FP32 activations and accumulates in double
without expanding the matrix. `CRIT_GROUPED_INT_AVX2` enables an AVX2 path (select it
only on an AVX2 CPU), with optional 8-bit G32 activation quantization; that changes
model arithmetic and needs its own quality check. This layout is not GGML Q4_0/Q8_0.

**NVFP4.** A value decodes as `e2m1(code) * e4m3(block_scale) / global_scale`: E2M1
codes, one E4M3FN scale per 16 row values and one FP32 divisor per tensor. Byte `i`
holds code `2i` in the low nibble and `2i+1` in the high ("adjacent"). Other layouts
(interleaved half-blocks, for example) have identical byte counts and must be converted
explicitly. Quantization uses `global = 6 * 448 / amax(tensor)`, block scale
`e4m3(amax(block) * global / 6)` and codes `e2m1(x * global / scale)`, ties to even,
saturating at 6 and 448; this is the codec's rule, not a claim of bit equality with any
producer. `four_over_six` (an option of `quantize`) also tries the block scale
`e4m3(amax(block) * global / 4)`, which maps the block maximum to the E2M1 value 4 and
so avoids the wide 4-to-6 step for values near it, and keeps whichever scale gives the
smaller squared block error (6 on a tie). The layout is unchanged, so every decoder
reads it. On Gaussian 1024 x 4096 weights it lowers the relative weight error from
0.0951 to 0.0870 (Student-t(3): 0.0906 to 0.0869). On Qwen3-1.7B with every decoder
projection RTN-quantized (weights only, BF16 activations, RTX 5090), the mean KL divergence
from BF16 over crit-eval's six quality conversations fell on each, by 14% to 40% (p99 by 5% to
52%), and stayed at 0.102 on the first 8,192 tokens of its long document; token-weighted
0.115 to 0.107. One small model: measure the model at hand before adopting it. The projection decodes each 16-value block once per row and reuses it for
every token, so decode cost is per weight, not per weight and token.

## Matched decode and prompt kernels

Decode runs a projection on 1 to 8 rows, prefill on thousands, and the fast kernel for each sums in its own order, so
a token's logits change with how it was computed: decoded, verified in a speculative round, prefilled or replayed from
a prompt cache. crit's honesty, request-history and prompt-cache gates compare exactly those, and an engine that runs
its prompt GEMM on a handful of decode rows to pass them reads its weights at a fraction of the memory rate. A matched
pair declares one per-output order that a decode kernel and a prompt GEMM both keep.

**FP8 (MXFP8, 128 x 128 blocks, row scales).** One order for E4M3 weights against E4M3 activations quantized per
group. Per 32-element block, the block's products are summed by one e4m3 `mma.sync` m16n8k32 with a zero accumulator,
weights the A operand and tokens the B operand, in a declared fragment order (thread q of a quad holds bytes 8q .. 8q
+ 7 of its row or token: 8q .. 8q + 3 in the first register of the pair, 8q + 4 .. 8q + 7 in the second). Then `lane
b % 16 = fma(block sum, weight scale * activation scale, lane)`, with the product rounded and blocks in order, the 16
lanes reduced by the halving tree ((i, i + 8), then (i, i + 4), (i, i + 2), (i, i + 1)), and times post. A kernel may
compute the lanes in any order, split them over warps and tile any number of rows and tokens. The activation quantizer
is declared as well: MXFP8's shared exponent 2^(floor(log2 amax) - 8), or amax / 448, then E4M3 by round to nearest
even with saturation. Its host twin is in [fp8_block.h](code/fp8_block.h), with E8M0 decoding and an FP64 projection
for error bounds; the order written out as device functions, and the kernels that keep it, are crit-labs'
`fp8-block-rows` kernel.

The order rests on one property of the tensor core: an output's block sum depends only on its own row and token, the
same way at every position of the tile. PTX does not specify the internal accumulation, so the kernel's own fixture
checks the property: three kernels built differently move rows and tokens across tile positions and companions and
must give the same bits. The internal sum may differ between GPU architectures and has no CPU twin, so check on the
target and keep FP8 projections on one GPU.

**4-bit weights.** [host-expert-arithmetic](../host-expert-arithmetic/README.md#integer-block-rows)'s integer block
rows are exact on any processor, so CPU rows, GPU decode rows and a GPU prompt GEMM share one order there too.

The operations a kernel for these formats must compute, and how to verify one against this entry's references, are in
[operations.json](operations.json); the kernels themselves come from libraries or the kernel registry
([Kernels](../../kernels/README.md)).

## What the code supplies

| File | Supplies |
| --- | --- |
| [implementation.py](code/implementation.py) | `QuantizedRows`, `quantize`, `dequantize`, `project` (optional `backend=` hook) |
| [row_fp8.h](code/row_fp8.h) | BF16/E4M3FN conversion, row validation, double-accumulating oracle |
| [grouped_int.h](code/grouped_int.h) | G32 integer codec and CPU projection |
| [nvfp4.py](code/nvfp4.py), [nvfp4.h](code/nvfp4.h) | NVFP4 codec; FP64 oracle in Python, fixed-order FP32 in C++ |
| [fp8_block.h](code/fp8_block.h) | The matched FP8 order's host twin: `quantize_group`, `e8m0`, `project_fp64` |
| [device_matrices.py](code/device_matrices.py) | The device form: stored BF16, FP8 and NVFP4 matrices, torch's scaled matmul, draft-only `reencode` |
| [device_nvfp4.py](code/device_nvfp4.py) | W4A16 decode rows, decoders and the fixed-tile GEMM through crit-labs' registry kernels, torch fallbacks |

## What the engine supplies

Treat code bytes, represented scales, geometry and codec version as one immutable
representation, and allocate weights and workspace through crit resource services,
retained through completion. A `backend(weights, activation)` must consume that exact
layout and return one value per row; the reference returns its result unchecked. A
native kernel can decode groups while accumulating; if it also quantizes activations,
it must define their blocks, scales and rounding.

Device kernels for these formats (FP8 and NVFP4 GEMMs and GEMVs) come from the providers in [Kernels](../../kernels/README.md), qualified
against the NVFP4 reference.

The plugin's device form multiplies a stored NVFP4 matrix in one of two ways (option `activations`): with NVFP4
activations (W4A4) through cuBLASLt's block-scaled GEMM, or with 16-bit activations (W4A16, weight only), which a
preserve-mode engine needs, since quantizing the activations to 4 bits moves every product by several percent. No
provider reachable from torch covers W4A16 NVFP4 (cuBLASLt's block-scaled GEMM takes NVFP4 on both operands; FlashInfer's
FP4 weight-only kernels sit inside its fused MoE), so [device_nvfp4.py](code/device_nvfp4.py) takes crit-labs' registry
kernels, declared with their pins in [plugin.json](plugin.json) (`kernels`): `decode-gemv`, a GEMV for up to 16 rows (the
same order whatever the row count, so a decode row is the same alone and batched; the E2M1 codes decoded by the
hardware conversion where the target has it, sm_100a and later, with the same bits as the portable decode) and a
decoder that writes the matrix in BF16 for torch's GEMM on longer prompts. torch's GEMM picks its kernel by the row
count, so a prompt row's sums depend on how many rows share the GEMM; option `prompt_gemm` fixed_tiles runs them
through `fixed-tile-gemm`, a GEMM whose K loop never changes, so a model can put the prompts of several sequences into
one pass and still give each the result of a pass of its own (`prompt_gemm: pinned` does it for every format,
below). Where a registry kernel is not there (not published
yet, no build for the machine, offline), a torch fallback serves with the same contract and the engine's facts say so
under `kernels`: the GEMV decodes the matrix a chunk of rows at a time to its exact values (e2m1 times the block
scale, at most 6 significant bits) and multiplies one activation row per GEMM with FP32 sums, the decoders are torch
operations with the kernel's bits, and the fixed-tile GEMM becomes one GEMM per prompt row (correct, much slower).
Prompts of up to 16 rows still take the GEMV: a model that batches prompts keeps a lone short one in a pass of its
own. Matrices stacked by rows (q/k/v, gate/up) keep one global scale per row.

**Decode rows and prompt rows.** A call of up to 16 rows (decode steps, verify windows: `row_invariance.rows` in the
facts, which `crit_plugins.device.invariant_rows` reads for a model plugin; 8 where a matrix runs the Triton GEMV, and
option `decode_rows` takes 8 to 16) is decode rows, a longer one prompt rows. `activations` is the prompt rows'
arithmetic and `decode_activations` the decode rows'. With `decode_activations: prompt` (the default) decode rows take
the prompt rows' activations through `decode-gemv`'s GEMVs: an FP8 matrix's rows are quantized to E4M3 with a scale per
row as its prompt rows are, and the E4M3 values (exact in BF16) go through the W8A16 GEMV, times the row's scale; a W4A4
NVFP4 matrix's rows are quantized as its prompt rows are (e2m1 times an E4M3 scale per 16, exact in BF16) and go through
the W4A16 GEMV, which reads the block scales in the layout cuBLASLt takes (one copy), divided by the activation scale.
Decode rows then differ from prompt rows in summation order only: at 27B widths (10,240 x 5,120 and 5,120 x 17,408) a
row alone and the same row in a 64-row call differ by at most 1e-5 of the output's largest value for FP8 and 7e-7 for
W4A4. With `decode_activations: bf16` decode rows multiply 16-bit activations by the stored values (weight only), whatever
the prompt rows do: 2.5e-2 (FP8) and 8.8e-2 (W4A4) of the largest value off the prompt rows there, which an engine whose
prompts and decode must agree (cached prompts, histories) does not take. `decode_formats` sets it per format where
the two formats should differ (`nvfp4=prompt,fp8=bf16`: W4A4 decode rows on the prompt rows' NVFP4 activations, FP8
decode rows on 16-bit ones), each matrix taking its format's at load; the facts' `decode_activations` say each
format's. Weight-only NVFP4, BF16 and MXFP8 matrices
multiply 16-bit decode rows either way (W4A16, W8A16: an E4M3 value times an activation is exact in FP32, the row scale
applies once); weight-only NVFP4 prompt rows multiply the same exact values (e2m1 times the block scale) and apply the
row's global scale to the FP32 sums as the GEMV does, 2e-6 of the largest value off the decode rows. Without
`decode-gemv`, FP8 and W4A4 decode rows under `decode_activations: prompt` go to cuBLASLt one row per call (the facts'
`decode_paths` say `cublaslt`), since its scaled GEMM picks its kernel by the row count: a row of a 2- or 8-row call
differs from the row alone at real widths (a 12,288 x 5,120 FP8 matrix).

**MXFP8.** Format `mxfp8` loads E4M3 weights with one E8M0 exponent per 32 inputs as modelopt stores them (`weight`
[out][in], `scale` [out][in / 32] bytes): decode rows through the W8A16 GEMV (each value times its power of two, exact),
prompt rows on the exact values decoded to BF16 (torch operations) and torch's GEMM or the fixed-tile GEMM. The
checkpoint's own MXFP8 activations are not reproduced (facts `prompt_activations` mxfp8: bf16).

**The tile.** Decode rows go through `mma-rows` (with `gemv: mma` every matrix's, with `gemv: auto`, the default,
every matrix's but those the next paragraph names), a tensor-core tile for 1 to 64 rows of any
format: the weights are the A operand of a BF16 MMA, every value exact in BF16, FP32 sums, each weight read once a call
whatever the rows, the prompt rows' quantized activations computed by its pre-pass (the bits of this plugin's torch
quantizers), and a sum's order fixed by the matrix and the device (the plan's splits of K), never by the rows. Prompt
calls of up to `tile_rows` rows (64 rows a kernel call) go through it too where the prompt arithmetic is the tile's:
`auto` (the default) sends weight-only prompts (NVFP4 W4A16, MXFP8, FP8 with `fp8_activations: bf16`) up to 64 rows,
whose prompt path would decode the whole matrix for a GEMM (a 23-token prompt's NVFP4 projections of a 27B model take
a fifth of the time), and keeps W4A4, W8A8 and BF16 prompts on cuBLASLt, which is faster there past the decode rows
(the crossovers measured on an RTX 5090 over three models' matrices: device_matrices.AUTO_TILES); a number sends every
matrix's prompts that far. A row gets the same bits in any call of up to `row_invariance.rows` rows (the fewest over
the loaded matrices of their decode rows, or their tile rows where decode and prompt arithmetic agree), so a captured
prompt bucket of up to that many rows (`prompts:projections`) gives the unpadded chunk's bits; past it the facts make
no such promise. Weight-only NVFP4 prompts in `fp16_acc` or
`fp8`, and `prompt_gemm: fixed_tiles` (which keeps every prompt row's bits its own way), keep their prompt path; FP16
activations, and quantized activations that are not BF16, take the other paths. Where `mma-rows` is not here, or K is
no multiple of 32, decode rows go through `decode-gemv`'s CUDA-core GEMVs (`gemv: native`), whose cost grows with the
rows past a few, or its Triton GEMV (`gemv: triton`); BF16 rows without the CUDA GEMV run a row at a time through torch.
Each keeps a row's bits alone and batched, in different orders, so a process uses one.

**Decode kernels by matrix.** With `gemv: auto` (the default) each matrix takes, at load, the decode kernel that
measured fastest for its arithmetic, one kernel for every decode row count (a row's bits never depend on how many rows
share a call), and the facts' `decode_kernels` name each role's kernel and decode activations (`decode_paths` the same
by format):

- FP8 matrices' 16-bit decode rows (W8A16) run `decode-gemv`'s dot GEMV: a Triton program of 16 rows by a block of
  the matrix on tensor cores (the E4M3 values exact in BF16, FP32 sums, a fixed split of K added in order, the row
  scale after), its program shape from the matrix's shape and the device. Its cost hardly grows with the rows, where
  the tile's does past 8 (an RTX 5090's 27B `out` 5,120 x 6,144: 22 us at 1 to 16 rows against the tile's 22 to
  30). Where it is not here, or K is no multiple of 128, the tile.
- W4A4 decode rows (`decode_activations: prompt` on NVFP4 matrices that take NVFP4 activations) run `mma-rows`'s
  block-scaled FP4 MMA on the sm_120 family: the activations quantized to NVFP4 as the prompt rows' are, then E2M1
  codes times E2M1 codes with both block scales inside the MMA, FP32 sums, so a decoded token and the same token in a
  prompt (the pinned block-scaled GEMMs) take the same arithmetic in another order. Flat across the rows (27B
  gate_up 68-72 us at 1 to 16 rows against the BF16 tile's 68-94); the tile takes no prompt rows then.
- Every other matrix keeps the tile (`gemv: mma` keeps every matrix on it).

`gemv: auto` became the default with these kernels: their sums run in another order than the tile's, so FP8 W8A16 and
W4A4 decode rows get other bits than under `gemv: mma` (each still the same alone and batched, at every row count);
`gemv: mma` keeps the tile's.

**Prompt GEMMs.** Prompt rows past the tile multiply in the arithmetic `activations` and `prompt` give. FP8 and W4A4
rows are quantized by `mma-rows`' quantizers in one kernel (the same bits as the torch quantizers, which serve without
it) and multiplied by cuBLASLt with the scales in its epilogue and a BF16 output: FP8 with the row and column scales,
W4A4 by its block-scaled GEMM with the activations' and the matrix's global scales as tensor-wide factors, one GEMM a
run of rows sharing a global scale (matrices stacked by rows); FP32 outputs (logits) keep the scales after an FP32 GEMM.
FP8 matrices with `fp8_activations: bf16` (W8A16, an exact variant for preserve-mode engines) widen the E4M3 weights
exactly to BF16 and multiply 16-bit activations with FP32 sums, the row scale after; the facts give every format's
arithmetic (`prompt_arithmetic`: weights, activations, sums). Weight-only, W8A16 and MXFP8 prompt rows decode the
matrix a chunk of rows at a time (row_chunks: multiples of 128 rows of at most 64 MiB of values) into one workspace
every matrix on the device shares, and write each chunk's columns of the output; the workspace,
`mma-rows`' activation scratch, partial sums and counters are made by the factory and declared by `memory()`, so a
matrix's load takes only its own tensors.

**Pinned prompt GEMMs.** cuBLASLt picks a GEMM's algorithm by the row count, and its algorithms split a row's sum
differently, so with `prompt_gemm: provider` a prompt row computes other bits when the prompt is longer, is cut into
other chunks or is padded to a captured bucket: a prefix scores its tokens differently from the whole prompt, by tenths
of a nat on a W4A4 engine, whose activation codes flip downstream of a moved ulp. `prompt_gemm: pinned` runs every
format's prompt GEMMs through `pinned-gemm`: at load, each GEMM shape and format (each run of a stacked W4A4 matrix,
each decoded chunk's width) is pinned to one split-K-free cuBLASLt algorithm, the first of the heuristic's list at
2,048 rows that gives a row the same bits at every count from 1 to 4,096 rows and at other offsets, chosen without
timing so every process makes the same choice; every prompt call of any row count runs it. W4A4 runs take one over
both global scales as the epilogue factor and write their columns of one output; FP8 rows (W8A8) multiply into FP32,
their row scales and the column scales after (cuBLASLt lists no split-K-free algorithm with row scales here), rounded
once; BF16 matrices multiply as stored;
decoded chunks (W4A16 with `prompt: bf16`, W8A16, MXFP8) multiply BF16 into FP32, their scales after. Prompt rows then
skip the tile (tile_rows applies to decode rows only), decode rows keep their own contract, and the facts say
`row_invariance` `prompt_rows` true: a row's bits are the same at every prompt length past the decode rows, in any
chunk, in a padded bucket and in a pass shared with other sequences. Weight-only NVFP4 prompts in `fp16_acc` or `fp8`
are refused. Without `pinned-gemm` the prompt rows go one a call through the provider's path (one algorithm for one
row; correct, much slower). On an RTX 5090 a pinned NVFP4 GEMM is as fast as torch's past 64 rows (8,192-row W4A4
gate/up and down 2.8 and 1.3 ms); FP8 rows pay their scales after the GEMM (about a third more at 8,192 rows), and
small-N BF16 GEMMs and weight-only decoded GEMMs at 17-64 rows, which torch splits along K, are slower (a 96 x 5,120
BF16 matrix 42 against 18 us).

**Exact prompt rows.** Pinned GEMMs keep a prompt row's bits the same at every length, but not the decode rows' bits:
a token scored in a prompt (Eval's honesty check, a prompt cache) and the same token decoded sum in other orders, so
their log-probabilities differ by more than a preserve-mode engine allows. `prompt_gemm: tile` (with `gemv: mma`,
every decode row on the tile: `gemv: auto`'s dot GEMV and FP4 MMA sum in orders of their own; weight-only NVFP4
prompts in bf16) sends every prompt row through `mma-rows`' prompt tile: the decode tile's MMA, the same k-steps and
splits of K (the matrix's plan), the split sums added in the decode tile's order, the activations as the decode rows
take them (`decode_activations`, `decode_formats`), while each decoded weight serves 64 rows a CTA as in a GEMM. Every
prompt row's output then has the bits the same row gets as a decode row, alone or batched, at any prompt length, in
any chunk, padded bucket or shared pass, and the facts say so: `row_invariance` `prompt_exact` true (and
`prompt_rows`), with `prompt_arithmetic` and `prompt_activations` the decode rows'. A matrix the decode tile does not
take, or a build of `mma-rows` without the prompt tile, multiplies its prompt rows through the provider and the facts
say `prompt_exact` false. Quantized activations (W8A8, W4A4) go through the shared activation scratch, as many rows a
call as it holds.

On the GEMVs, every format keeps each row's bits in a call of up to 16 rows, alone or beside the others (the projections slot's
device form promises 8 and a plugin's facts may promise more; tests/test_device.py checks 16 bit for bit for BF16, FP8
per channel and per tensor, MXFP8, W4A4 and W4A16, with either `decode_activations`): torch's BF16 GEMM picks another kernel at 2 rows than at 1 and
differed there, which the BF16 GEMV removes. Calls of more rows (prompt rows) keep it through `prompt_gemm: pinned`
(every format) or `prompt_gemm: fixed_tiles` (BF16, MXFP8 and weight-only matrices with BF16 prompts), and the
plugin's facts say so (`row_invariance`: `rows` 16, `prompt_rows` true once every matrix it loaded multiplies its
prompt rows that way).

**Captured rounds.** The plugin provides `rounds:projections`: every decode and prompt path is device work on buffers
sized at load, without a host synchronization, and the registry kernels, the fallbacks' tables and the decode
workspaces are made at load. Each matrix holds the registry modules its calls use from load (a call looks none up),
and runs every path once at load on zero activations: decode rows at every count it takes, the tile's prompt counts,
and its prompt route over a ladder of row counts up to 2,048, into its default output and FP32. So no request meets a
first use (a Triton kernel's compile, cuBLAS or cuBLASLt picking and loading a kernel for a shape, a CUDA module loaded
on its first launch); matrices of the same shape and arithmetic warm once. A Triton kernel's first call inside a
capture still raises and says so (a shape the warm-up did not meet). tests/test_rounds.py captures every decode
arithmetic at 27B widths, replays it on new inputs and checks the eager bits and each row's bits at 1 to 8 rows;
tests/test_tiles.py checks that no call after load compiles a Triton kernel or asks for a registry kernel.

A model plugin may pass each matrix's role (`load(tensors, format, role)`, in the model's own words), and option
`prompt_roles` gives the roles it names their own prompt arithmetic over `prompt`: `prompt: fp16_acc` with
`prompt_roles: gate_up=fp8` runs the widest matrix on FP8 rows and the rest with FP16 accumulation.

**Draft-only re-encoding.** A matrix only a drafter reads (an MTP layer, a draft head) may be re-encoded at load:
`load(tensors, format, role, encode="nvfp4")` takes a stored BF16 or FP8 matrix to NVFP4 (one FP32 global scale `g =
6 * 448 / amax`, E4M3 scales per 16 values and E2M1 codes by this entry's rule, code for code with nvfp4.py's
`quantize_row` at that `g`) and multiplies it with 16-bit activations (no calibrated activation scale exists for it);
`encode="fp8"` takes a BF16 one to E4M3 with one FP32 scale per row. The encoders are torch operations in FP64 at load.
The model plugin passes `encode` only for draft-only roles, the stored target still verifies every draft (the run's
`as_stored` weights rule covers target weights, not draft-only ones), and the facts record each re-encoded role under
`draft_encoding` (stored and encoded format, matrices, bytes before and after). A draft step reading an MTP layer and
head of a 27B model (hidden 5,120, vocabulary 248,320) moves about 0.95 GB as NVFP4 instead of 1.7 GB as FP8.

## Worked example

Run from `code/`. An odd number of 4-bit elements leaves one padded nibble.

```python
from implementation import quantize, dequantize, project

weights = quantize([[-7.0, 7.0, 0.0]], group=3, bits=4)
assert weights.codes == bytes([0x79, 0x00])
assert weights.scales == (1.0,)
assert dequantize(weights) == ((-7.0, 7.0, 0.0),)
assert project(weights, (1.0, 2.0, 3.0)) == (7.0,)
try:
    project(weights, (1.0,))
except ValueError:
    pass
else:
    raise AssertionError("activation geometry mismatch was admitted")
```

## Adapting it

Changing a codebook, group structure, scale precision, zero point or packing order
creates a different codec; do not relabel those bytes as this one. Changing reduction
order or activation encoding changes the arithmetic and must be verified on its own.
Group width, tiling, placement and fusion are engine choices.
[Artifact loading](../artifact-loading/README.md) provides the exact encoded bytes and
[expert dispatch](../heterogeneous-expert-dispatch/README.md) chooses their source;
neither may silently change the representation or apply a routing weight twice. A fused
normalization and projection must make intermediate rounding explicit.

**Calibrated activations (W4A4 prompts).** ModelOpt NVFP4 checkpoints carry a static
`input_scale` per linear layer. For prompt GEMMs, quantizing activations to NVFP4 with that
global scale (`gs = 1 / input_scale`; per 16 values a UE4M3 scale `e4m3(amax / 6 * gs)` and
E2M1 codes, round to nearest even, saturating) runs both operands on block-scaled FP4 tensor
cores, as NVIDIA's serving stacks run such checkpoints. Quantize each row on its own so a
token's result never depends on the other tokens of a chunk, and check that the scale really
is shared before folding it into one quantization per layer (MoE checkpoints often give all
experts of a layer one `input_scale`). Grouped block-scaled GEMMs reach their rate only with
enough rows per expert, so measure the rate by rows per expert on the target device before
choosing the prompt chunk size, and keep weight-only GEMVs for decode. Check the prompt's
drift against the reference, since activations are quantized too. Both halves come from
providers ([Kernels' sources](../../kernels/SOURCES.md)): FlashInfer's FP4 quantization
writes the activations' E2M1 codes and E4M3 block scales from a global scale, in the
swizzled scale layout cuBLASLt's block-scaled NVFP4 GEMM reads (16-element vector scales;
the scale tensors are tiled, not row-major, so a plain `[rows][cols / 16]` array is the
wrong input), and cuBLASLt or CUTLASS run the GEMM. Check the quantizer's codes and scales
against this entry's NVFP4 codec on a few rows before trusting the layout. (For scale only: CUTLASS
v4.7.1 SM120 ptr-array grouped GEMMs on an RTX 5090 ran at 260 TFLOPS at about 90 rows per
expert and 720-940 at 352-1,408; on one 120B MoE the routed experts took a 64K-token prompt
from 15.3 s to 9.4 s at a prompt KLD mean of 4e-3.)

## Costs and validation

Storage is `ceil(R * W * bits / 8) + 4 * R * W / G` bytes with FP32 scales; smaller
groups trade scale traffic for accuracy. The Python fallback dequantizes the whole
matrix, so it is a formula oracle, not a fast path. Malformed extents, incomplete
groups, nonfinite values and unrepresentable scales are rejected; finite inputs can
still overflow an accumulation.

The [NVFP4 fixture](../../tests/inference/nvfp4_tests.cpp) checks every E2M1 and finite
E4M3 code, ties, saturation, zero blocks, an error bound and batched projection against
an FP64 dot. [Host fixtures](../../tests/test_techniques.py) cover nibble decoding, odd
extents, zero groups, invalid representations and hook dispatch.
[Native host fixtures](../../tests/inference/inference_io_codec_tests.cpp) cover every
finite FP8 code, signed zeros, scales and invalid encodings. None of these qualify a converted model or establish a speedup.

The device form's tests ([tests/test_device.py](tests/test_device.py)) run each path bit for bit for row invariance
(BF16, FP8 per channel and per tensor, W4A4 and W4A16; the registry kernels with LOCAL_KERNELS or from the Hub, and
the torch fallbacks without), the fallback decoders against the kernel's bits, CUDA graph capture, and draft-only
re-encoding against nvfp4.py code for code (ties to even, saturation, zero blocks). The matched FP8 order's device
fixture lives with its kernels (`fp8-block-rows`).
