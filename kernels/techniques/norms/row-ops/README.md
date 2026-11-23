# row-ops

The row operations between a model technique's matrix products, one Triton kernel a call (two where a reduction spans
programs), for any model plugin whose techniques run them: each head vector shortened to an L2 length (times a
learned scale and a per-token factor), a row's few values times a small matrix (a bias a token computes for the
positions behind it), an attention's rows times a sink factor and a sigmoid output gate, scaled adds (a convolution
joined to its input), softmax attention of learned queries over a token's normed summary rows (a residual that
attends over depth), gated joins of a row and a message row with their low-rank gates inside, and attention of a few
queries over a span of a row's cached positions without a causal mask.

**Why it exists.** In torch each of these is a chain of small kernels (casts, squares, sums, sigmoids, products), and a
decode round runs them in every layer; torch's own row reductions also do not keep a row's bits across row counts.
Here each call is one launch that reads its inputs once, and every reduction runs in an order fixed by the row's width
and the structure's order knobs alone.

**Who calls it.** A model technique asks the model for it (`model.kernel("row-ops")`, declared by the models core's
plugin.json) and falls back to its own torch form where it does not load (no CUDA or Triton, crit's kernels package
not installed). The operations, with their FP64 references, are the techniques' (`operations.json` of
[relative-attention](../../../../models/techniques/mixers/relative-attention/README.md),
[block-attention](../../../../models/techniques/residuals/block-attention/README.md) and
[block-drafter](../../../../models/techniques/drafters/block-drafter/README.md)); `crit_kernels.contracts` holds the
tolerances. No function names a model: a call gives the shapes.

## Applicability

| | |
| --- | --- |
| Functions | `head_clip`, `small_product`, `sink_gate`, `scaled_add`, `summary_mix`, `pair_norm`, `gated_join`, `low_rank_gates`, `span_attention`; signatures and semantics in the package docstring |
| Dtypes | FP32, BF16 or FP16 inputs as each function states; FP32 inside; the roundings each operation names |
| GPUs | NVIDIA GPUs Triton compiles for (sm_80 and newer) |
| Determinism | same bits on a repeat; a row's bits independent of the rows beside it and of the row count; CUDA graph replay equal to eager calls (`tests/test_row_ops.py`) |
| Capture | no host synchronization, every allocation from torch's caching allocator; a shape's first call compiles |

## Knobs

The launch constants are knobs (`kernel.json`), read per call shape from the build's tuning (`_crit_tuning.py`: a
sweep's table, or the defaults). The defaults are the constants these kernels had when the techniques kept them beside
their code, so an untuned build launches what those launched. A structure whose programs reduce looks its knobs up by
its shape key alone and marks the ones that set a summation order (`"order": true`), so a row keeps its bits at every
row count; the elementwise structures may pick their speed knobs by row count.

| Structure | Functions | Key | Knobs (default) |
| --- | --- | --- | --- |
| head-clip | `head_clip` | D | WARPS (4, order) |
| small-product | `small_product` | R, E | BLOCK (256), WARPS (4) |
| sink-gate | `sink_gate` | D | WARPS (4) |
| scaled-add | `scaled_add` | | BLOCK (1024), WARPS (4) |
| summary-mix | `summary_mix` | H | CHUNK (256, order), WARPS (4, order) |
| pair-norm | `pair_norm` | H | BLOCK (1024, order), WARPS (4, order) |
| gated-join | `gated_join` | H | BLOCK (1024), WARPS (4) |
| low-rank-gates | `low_rank_gates` | H, R | BR (8), BK (256, order), BH (64), WARPS (4, order) |
| span-attention | `span_attention` | G, D | CHUNK (512, order), BN (64, order), WARPS (4, order) |

`crit kernels sweep row-ops --model MODEL` tunes them for a model on this GPU: the benches (`bench.py`) find the
model's widths in its config's standard keys (hidden width, heads, head width) and in the shapes of its tensors (the
small matrices a row multiplies, the low-rank gates' [rank][2 hidden] matrices), never in a tensor's or a key's name,
and skip a structure whose shapes the model does not have. The yardstick of each structure is the torch chain its
technique falls back to.

## Numerics

Against the FP64 references (`crit_kernels.contracts`): FP32 arithmetic plus one rounding of a 16-bit output for the
clips, small products, sink gates and scaled adds; 2e-5 for the summary mix (FP32 out); 1e-4 for a gated join with its
gates given, 2e-2 of the largest value through low-rank gates (a BF16 rounding of a gate row may fall the other way);
the attention tolerance for span attention (BF16 probabilities times BF16 values, FP32 sums).
