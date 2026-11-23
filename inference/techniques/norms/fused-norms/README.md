# The elementwise work around the matmuls, one launch per call

Every transformer layer normalizes, rotates, gates and multiplies its activations between its matmuls: an RMSNorm
before each block (after adding the block's output to the residual stream), query and key norms, rotary positions, the
gated norm of a linear-attention output, an attention output gate, the SiLU product of the MLP's gate and up rows.
Written by hand in torch, each of these is a chain of small kernels (a cast, a square, a mean, an rsqrt, two
multiplies, a cast back), so a decode round launches hundreds of kernels that each move a few kilobytes, and a prompt
pass reads the same activations several times. This entry names those operations once, by what they compute, as the
`norms` slot, and runs each as one kernel launch.

## When to use it

Use it in a model plugin for every norm, rotary position and gated product of its layers: decode rounds launch one
kernel per operation (a chain of 2 to 46 torch kernels before, as measured below), and a CUDA graph of a round holds that many fewer nodes. It is not
needed where a matmul's epilogue already adds the residual or forms the gated product, or for a normalization the
slot does not name (LayerNorm with a bias, group norms).

## How it works

Every operation reads its inputs in their own dtype (BF16, FP16 or FP32), computes in FP32 and rounds once, to the
dtype asked for (by default the dtype of the tensor it normalizes or gates). The last dimension is contiguous; the
others may have any strides, so a query beside its gate in one projection's output, or one slice of a fused
`[q | k | v | z]` projection, is read where it lies.

| Operation | Computes |
| --- | --- |
| `rms_norm(x, weight, eps, offset=0.0, residual=None, out_dtype=None)` | `x / sqrt(mean(x^2) + eps) * (offset + weight)` per row; with a residual, `residual += x` first, in place and in the residual's own dtype (an FP32 residual stream beside BF16 rows), and the sum as stored is normalized: `(y, residual)`. `offset` 1 for a weight stored as `weight - 1` |
| `rms_norm_heads(x, weight, eps, offset=0.0)` | the same per head of `[tokens][heads][dim]` (query and key norms) |
| `rope(x, positions, freqs, rotary_dim=None, style="half", scale=1.0)` | each head rotated at its token's position on its first `rotary_dim` values (partial rotation leaves the rest). `freqs` are `rotary_dim / 2` FP32 inverse frequencies, the angle `fp32(position) * freq` rounded to FP32 as checkpoints' reference code computes it, or a `[positions][rotary_dim]` table of cos then sin. `half` pairs `(j, j + rotary_dim / 2)` (rotate_half), `interleaved` pairs `(2j, 2j + 1)`; `scale` multiplies cos and sin (an attention factor). In place with `out=x` |
| `rms_norm_rope(x, weight, eps, positions, freqs, ...)` | the head norm, then rope, FP32 between them |
| `gated_rms_norm(x, z, weight, eps, offset=0.0, activation="silu")` | `rms_norm(x) * activation(z)`, silu or sigmoid (a gated linear-attention output) |
| `silu_mul(x, up=None, layout="halves")`, `gelu_mul(..., approximate="none")` | `act(gate) * up`: two tensors, or one row holding both, `[gate | up]` halves or interleaved |
| `sigmoid_mul(x, gate)` | `x * sigmoid(gate)` (an attention output gate) |
| `l2_norm(x, eps=1e-6)` | `x / sqrt(sum(x^2) + eps)` per row |
| `stream_sum(x, gate=None, activation="none", scale=1.0)` | for x `[tokens][streams][dim]`, `scale * sum_s activation(gate[t][s]) * x[t][s]` in stream order, the gate one per value or one per stream (`[tokens][streams]`), or none (`scale` 1 / streams: their mean): several residual streams mixed into one row |
| `stream_add(streams, x, weights=None)` | `streams[t][s] += weights[t][s] * x[t]` for every stream, in place in the streams' dtype: one row added into every residual stream |
| `softplus_decay(x, bias, rate, gate=None)` | `-rate * softplus(x + bias)` per value of x `[tokens][n]` (a slice of a projection's output, read where it lies), with a gate also `sigmoid(gate)`: a gated delta or state-space layer's log-decay (`rate` its `exp(A_log)`) and beta, FP32 out by default; one launch where torch takes six (a cast, a sigmoid, an add, a softplus, a negation, a multiply) |

Where a layer runs two of these back to back, a fused form (`norms:fused`) does both in one launch, each output with
the bits the separate calls give it:

| Fused form | Is |
| --- | --- |
| `rms_norm_rope_pair(q, k, q_weight, k_weight, eps, positions, freqs, ...)` | `rms_norm_rope` of a layer's queries and of its keys at the same positions, each its own weight: `(q, k)` |
| `rope_pair(q, k, positions, freqs, ...)` | `rope` of a layer's queries and keys (families without query/key norms) |
| `stream_add_norm(streams, x, weights, weight, eps, offset=0.0)` | `stream_add`, then `rms_norm` of every stream as stored (a weight per stream or one for all): `(y, streams)` |

Every weight may also be one row per group, `[groups][dim]` for x `[..., groups, dim]`: each head, or each of several
residual streams normalized in one call, its own weight. Models that keep several residual streams (mixing them with
learned, per-token weights around each block) take a norm, a mix and an add per block: three launches where a chain of
torch kernels takes one per stream and step.

**Each row on its own.** A row's bits are independent of the rows beside it and of how many share the call: the
reduction over a row runs in an order fixed by the row's width alone. So a token decoded alone, the same token in a
verify window and the same token in a prompt chunk give the same bits, as crit's concurrency and honesty gates
compare. torch's own `mean` and `sum` over the last dimension do not keep this (on a GPU, rows of a call of a few rows
sum in another order than in a long call at hidden widths); torch's fused `rms_norm` reduces one row at a time and
does, so the torch form reduces through it.

A launch's specialization never depends on the row count: dimensions of size 1 keep their strides, and a layout the
kernel cannot read aligned (a base address or a row stride that is not a multiple of 16, as at a width of 4,100 or a
head view with an odd token stride) is copied into aligned buffers first, so every row of a call site is read by one
layout.

**Capture-safe.** Nothing synchronizes with the host or depends on the values, and every allocation (the result unless
`out` is given, whose last dimension is contiguous too, and those aligned copies) comes from torch's caching allocator:
a CUDA graph captures every call on fixed shapes, positions read from a round's fixed-address `positions_t`. A shape's
first call may compile a kernel, so call each shape once before capturing. Every tensor of a call is on one device.

**Several streams.** A per-token mix of S streams into S streams (each output stream its own weighted sum of the
inputs) is S `stream_sum` calls, each written into its stream's `out=` view.

## Which code serves each call

Every call goes to crit's kernel `fused-norms` (Triton, `kernels/techniques/norms/fused-norms`), loaded by the name plugin.json
declares (`crit_plugins.kernels`) and built for the engine's device with the engine's tuning table (`crit kernels sweep`
tunes its block sizes and warp counts for the model's widths) or its default knobs: one launch that reads its inputs
once. Where the kernel cannot be built (no Triton or CUDA, crit's kernels package not installed) the plugin runs
`torch_norms.py`, the same operations as chains of torch kernels, and the engine's facts say so (`kernels`: tuned,
built or fallback). Kernel or torch is chosen once, when the plugin is created. No library kernel serves a call:
FlashInfer's norms and activation products are the kernel's yardstick (its sweep bench and README times them beside
the kernel), not a path of this plugin. Until round 4 FlashInfer served `rms_norm` / `rms_norm_heads` without a
residual and the packed `silu_mul` / `gelu_mul` (option `flashinfer`, now removed); the kernel matches or beats it on
those calls at Qwen3.8-27B's shapes (the kernel's README: 1.04-1.13 for the SiLU product at 1-16 rows, 1.0-1.05 at
128-8,192, 0.99-1.15 for the norm).

## What the code supplies

| File | Supplies |
| --- | --- |
| [`rows.py`](code/rows.py) | The host form: `rms_norm_rows`, `gated_rms_norm_rows`, `rope_rows`, `act_mul_rows`, `l2_norm_rows`, `stream_sum_rows`, `stream_add_rows`, `softplus_decay_rows` over rows of numbers in FP64, the oracle of every device form (`crit_plugins.slots.INTERFACES["norms"]`) |
| [`torch_norms.py`](code/torch_norms.py) | The device form in plain torch ops (`crit_plugins.device.DEVICE_INTERFACES["norms"]`): the base plugin `torch-norms`, and this plugin's fallback |
| [`conformance.py`](code/conformance.py) | The checks any norms device form passes: every method against the host form (FP32 out within FP32 rounding, 16-bit out within one rounding), each row's bits in calls of 1 to 8 and of hundreds of rows at three models' widths and at unaligned layouts (a width of 4,100, an odd token stride), strided views and outs as their copies, `out=`, refused layouts and in-place rope on large shapes, CUDA graph replay against eager calls, two plugins against each other, and the fused forms against the separate calls bit for bit (`fused_forms`) |

## What the engine supplies

- Its layer arithmetic: which norm, offset and eps each layer uses, the rotary style, dimension and table, read from
  the checkpoint's config (attention-state's `rope_table` derives scaled inverse frequencies).
- The weights on the device, the residual stream's dtype, and warm calls of each shape before it captures a graph.

## Worked example

Run from `code/`: the host form, the arithmetic every device form is held to.

```python
import math
from rows import rms_norm_rows, rope_rows, stream_add_rows, stream_sum_rows

# A block's output added to the residual stream, then RMSNorm with a zero-centred weight (stored as weight - 1).
(y,), (sums,) = rms_norm_rows([[1.0, 2.0, 3.0, 4.0]], [0.0, 0.0, 0.5, 0.5], 0.0, offset=1.0,
                              residual=[[1.0, 0.0, -1.0, 0.0]])
assert sums == [2.0, 2.0, 2.0, 4.0]
rms = math.sqrt(sum(v * v for v in sums) / 4)
assert all(abs(a - b) < 1e-12 for a, b in zip(y, [2 / rms, 2 / rms, 3 / rms, 6 / rms]))

# Partial rotary positions: the first 2 of 4 values turn by the position's angle, the rest pass through.
(turned,) = rope_rows([[1.0, 0.0, 5.0, 6.0]], [3], [0.5])          # one pair, inverse frequency 0.5
assert abs(turned[0] - math.cos(1.5)) < 1e-6 and abs(turned[1] - math.sin(1.5)) < 1e-6 and turned[2:] == [5.0, 6.0]

# Two residual streams: their mean, and one row added into both with a weight each.
streams = [[[1.0, 2.0], [3.0, 6.0]]]
assert stream_sum_rows(streams, scale=0.5) == [[2.0, 4.0]]
assert stream_add_rows(streams, [[1.0, 1.0]], [[2.0, -1.0]]) == [[[3.0, 4.0], [2.0, 5.0]]]
```

## Measured

One RTX 5090 (torch 2.14.1, Triton 3.8, FlashInfer 0.7.3), the kernel's `benchmarks/shapes.py`: each
operation as a model writes it by hand in torch (the chain), as one call of the kernel (its default knobs), and as FlashInfer's
call where it has the operation (a yardstick), with BF16 rows and an FP32 residual stream. GPU time per call inside a CUDA graph
(what a captured decode round pays; median of 7) and kernels launched per call, at a 5,120-wide hidden row with 24
query and 4 KV heads of 256 (64 rotary values, zero-centred weights), 48 gated-norm heads of 128 and a 17,408-wide
MLP. **Bold**: the path this plugin serves the call on.

| Operation | rows | chain: launches, GPU us | kernel: launches, GPU us | FlashInfer: GPU us |
| --- | --- | --- | --- | --- |
| residual add + RMSNorm (FP32 residual) | 1 / 8 / 1,024 / 8,192 | 9, 11.0 / 14.4 / 62.5 / 1,186 | **1, 2.4 / 5.5 / 14.3 / 340** | (a BF16 residual, the unrounded sum: 1.8 / 2.4 / 9.3 / 226) |
| RMSNorm, BF16 weight, no residual | 1 / 8 / 1,024 / 8,192 | 10, 13.0 / 16.4 / 58.0 / 1,091 | **1, 1.9 / 1.8 / 5.4 / 111** | 1.6 / 2.1 / 6.2 / 109 |
| q and k: head norm + rotary positions | 1 / 8 / 1,024 / 8,192 | 46, 58.0 / 66.5 / 157 / 1,767 | **2, 4.7 / 5.5 / 11.9 / 155** | |
| gated RMSNorm | 1 / 8 / 1,024 / 8,192 | 13, 17.3 / 21.4 / 136 / 2,559 | **1, 1.8 / 1.9 / 7.5 / 191** | |
| attention output gate | 1 / 8 / 1,024 / 8,192 | 3, 4.1 / 5.5 / 19.8 / 457 | **1, 1.7 / 1.4 / 12.6 / 191** | |
| SiLU product (packed halves) | 1 / 8 / 1,024 / 8,192 | 2, 2.4 / 4.0 / 98.2 / 922 | **1, 1.2 / 1.7 / 37.6 / 553** (round 4, a graph of 100 calls: 0.80 / 0.89 / 36-44 / 541, FlashInfer 0.91 / 0.95 / 36-38 / 548) | 1.3 / 1.0 / 35.1 / 562 |

A decode row through 16 full-attention and 48 gated-delta layers of that shape launches about 2,700 kernels for these
operations as torch chains (3.4 ms of GPU time in a graph) and 288 as this plugin's calls (0.6 ms). At 8,192 prompt
rows the residual norm (reading the BF16 rows and the FP32 residual, writing both) and the products run at about the
GPU's memory rate. At a 7,168-wide row with 56 + 8 heads of 128 (full rotation): residual norm 3.0 against 11.9 us a
decode row, q and k rope 3.2 against 33.6 us (FlashInfer's rope of q and k together in place: 1.6 us, for frequencies
from a base). Four residual streams of 2,560 mixed around a block (a norm with a weight per stream, a gated mean, a row
added into every stream): 3 launches and 6.0 us against 11 launches and 14.5 us a decode row, 1.38 against 3.70 ms at
8,192 rows. Decode-row numbers vary between runs on the shared machine (the kernel's SiLU product at 8 rows read 9.9 us
in one run and 1.7 us in the next, the residual norm at 8 rows 3.0 and 5.5 us).

**A 27B target pass.** Every norms-slot call one Qwen3.8-27B target pass makes (64 layers: 289 calls) and its gated
delta layers' decay and beta, captured in one CUDA graph, timed after a spin kernel between two events around the
replay (median of 40), every layer reading the same activation buffers (as a round's norms read what its projections
just wrote): at 1 / 4 / 9 / 16 / 36 token rows 652 / 768 / 793 / 819 / 927 us with q and k apart and the decay and
beta as six torch kernels a layer (577 launches), 439 / 496 / 504 / 515 / 617 us with `rms_norm_rope_pair` and
`softplus_decay` (321 launches), the same bits. The decay and beta take 279-370 us as torch kernels and 68-74 us as one
launch a layer; q and k 46-50 us apart and 25 us in pairs. Of the rest at 4 to 16 rows: the 129 residual norms
281-283 us (2.2 us a call), the 48 gated norms 80 us, the 64 SiLU products 49-54 us, the 16 output gates 15-17 us;
none grows with the row count. Each call timed alone after a spin (an eager pass) costs 4 to 6 us: 1.61 to 1.66 ms
for the 289 calls at 1 to 16 rows. The fused forms change no bits: each output equals the separate calls'.

Uncaptured, a call costs the host its launch path: 25 to 90 us of CPU time a call through Triton on that (loaded)
machine, 11 to 26 us through FlashInfer, about 10 us a torch kernel. Through the kernel it is a clear gain for the
chains of 9 to 46 kernels and about even for the attention gate; the SiLU product loses there against FlashInfer's
launch path (an engine's eager steps only). Captured rounds pay none of it.
