# Stream routed experts through a prompt

Read a prompt through routed experts that mostly live in host memory, at a rate set by
the link and the device's arithmetic rather than by the number of chunks. Each prompt
chunk sends every touched expert that isn't resident over the link once per routed layer.
Small chunks therefore re-send the same experts again and again, and run the expert
projections at a handful of rows per expert. The entry supplies a schedule that picks the
chunk from the grouped projection's measured rate curve and overlaps each layer's expert
copies with the layer before it, and a CPU reference that any grouped projection is
checked against before an engine trusts it. The projection itself is a provider's:
[kernels.json](kernels.json) names the candidates, each with its formats, GPUs and shapes.

## When to use it

Use it for prefill of an expert-routed model whose experts do not all fit on the
device: the routed experts are in host memory (or a device cache holds part of them),
and a prompt chunk's time is dominated by copying experts. The sign is a prefill rate
that grows with the chunk size, and a link that is busy while the device waits.

Don't use it when every expert fits on the device: then only the grouped projections
matter, and a provider's grouped GEMM for the weight format is the first thing to
measure. For decode, where a step routes a few tokens, use the
[device expert cache](../device-expert-cache/README.md) and GEMV-shaped expert rows
instead.

## How it works

**Chunk size from the rate curve.** Measure the grouped projection on the target at the
engine's shapes, by rows per expert (`RateCurve`). Its knee, the fewest rows per expert
that reach 90% of the best measured rate, is where more rows stop paying. Under uniform
routing a chunk of `t` tokens touches `E (1 - (1 - k / E)^t)` of a layer's `E` experts
with top-`k` routing, so its rows per active expert are `t k / touched`. `choose_chunk`
returns the smallest chunk that reaches the knee and fits the activation workspace
(`max_tokens`). Larger chunks also cut the copies: once a chunk touches nearly every
expert of a layer, each non-resident expert crosses the link once per chunk, so the
prompt's copies fall in proportion to the number of chunks.

**Layer double-buffering.** A step is one routed layer of one chunk, in execution
order. Two staging buffers alternate: step i + 1's copies run on a copy engine while step
i computes, into the buffer step i - 1 released. Once a chunk touches nearly every expert
(`fetch_whole_layer`), the copies don't wait for the layer's router: the whole
non-resident set of the layer is copied ahead, which needs no prediction. Below that
threshold, copy the routed experts after the router, or lower the chunk's priority on the
link. `timeline` and `prefill_estimate` price a schedule from measured step times and
bytes, and report the copies' own time, the floor no schedule beats. Take the staging
buffers from memory the decode path does not use while a prompt runs, as the
[device expert cache](../device-expert-cache/README.md#how-it-works) lends its
prompt workspace from slots (`staging_bytes` sizes one buffer).

**The projection.** Each routed layer sorts the chunk's routes by expert, gathers the
routed rows, runs every expert's gate/up and down projections with SwiGLU between them, and
sums each token's expert outputs with its routing weights. A grouped GEMM that takes one
pointer per expert reads each expert wherever it sits: a cache slot, a staging buffer or
device-visible host memory. [kernels.json](kernels.json) lists the providers for it:
CUTLASS's ptr-array grouped GEMM (block-scaled for 4-bit formats on GPUs that have it),
cuBLAS's grouped batched GEMM for 16-bit weights, and crit-labs' `grouped-experts` Hub
kernel, a baseline for every format on any NVIDIA GPU from sm_70 (not published yet).
`reference.py` gives the order a stable route sort produces (`route_order`) and the block's
FP64 result (`moe_block`, from weights decoded with `decode_matrix`).

**Chunking is free to change** when the projection splits no reduction by the row count,
fills short tiles with zero rows and combines a token's routes in a fixed order. A token's
result then does not depend on the other tokens of its chunk, and the
[batch invariance rules](../../core/src/batch/README.md#batch-invariance) hold for any
chunk size the schedule picks. Check that on the target too: the same token's output with
two chunk sizes.

## What the code supplies

- [schedule.py](code/schedule.py): `RateCurve`, `expected_touched`,
  `rows_per_active_expert`, `choose_chunk`, `fetch_whole_layer`, `Step`, `timeline`,
  `prefill_estimate` and `staging_bytes`.
- [reference.py](code/reference.py): `decode_matrix` (BF16, FP8 E4M3 with row scales, NVFP4,
  MXFP4), `route_order` and `moe_block`, the FP64 reference a projection is checked against.
- [kernels.json](kernels.json): the provider kernels for the grouped projection, with their
  calls, formats, GPUs, shapes and how to check each against `moe_block`.

## What the engine supplies

The router and its top-k ids and weights per chunk, the grouped projection (from
kernels.json, checked against `moe_block` on the target), the expert pointer table (which
slot, staging buffer or host address holds each expert this step), the staging buffers and
the copy-engine copies into them with their events, the attention and dense layers, and the
measurements: the projection's rate by rows per expert, each step's time at the chosen
chunk, and the non-resident bytes per layer. Per routed layer of a chunk:

```text
[copy stream: wait for step i - 1's compute; copy step i + 1's non-resident experts -> staging[(i + 1) % 2]]
router -> ids, weights (device)
sort the routes by expert; gather the routed rows
wait for step i's copies; expert pointers -> cache slot, staging[i % 2] or host address
grouped gate/up projection; SwiGLU; grouped down projection
combine with the routing weights -> residual add
```

## Worked example

```python
from schedule import RateCurve, choose_chunk, expected_touched, prefill_estimate

curve = RateCurve({16: 100.0, 64: 400.0, 256: 800.0, 512: 850.0})  # TFLOPS by rows per expert
print(curve.knee(0.9), choose_chunk(curve, top_k=8, experts=128, max_tokens=16384))  # 256 4096

# 40 routed layers, 6 MB experts, 70% of them off the device, a 27 GB/s link, 4 us of arithmetic per token and layer
copied = lambda n: int(expected_touched(n, 8, 128) * 0.7 * 6_000_000)
small = prefill_estimate(16384, 2048, 40, lambda n: n * 0.004, copied, 27.0)
large = prefill_estimate(16384, 8192, 40, lambda n: n * 0.004, copied, 27.0)
print(round(small[0]), round(large[0]))  # about 2,570 and 6,200 tok/s: the larger chunk sends the experts a quarter as often
assert large[0] > 2 * small[0]
```

## Adapting it

Preserve: no reduction split by the row count and no atomics in a token's sum, so chunk
sizes can follow the schedule; a staging buffer refilled only after its last step has
finished computing; every routed expert of a step resident, staged or readable before
the projection reads it. Free to change: the projection (any kernel from kernels.json, or
the engine's own, measured against them), fusing the SwiGLU and the combine into it, the
chunk policy and threshold, and computing a few rarely routed experts on the CPU instead of
copying them, by a measured split. FP8 or FP4 activations follow the run's quality mode
and are measured as such.

## Costs and validation

The host suite checks the schedule's knee, chunk choice, touched experts, the
double-buffered timeline and the estimate, and the reference's decoders, route order and
block on hand-computed cases. None of this measures performance. A provider's projection
earns its place only on the target: within its arithmetic's tolerance of `moe_block` on a
few tokens of real routes and weights, and at the rate the schedule's curve was measured
with.
