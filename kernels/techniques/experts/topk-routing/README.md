Top-k expert routing of mixture-of-experts router logits in one kernel, a warp per token row: crit's topk-router plugin
(`inference/base/topk-router`) loads it, built for the engine's device with the engine's tuning or the default knobs, for its
Routing description (heterogeneous-expert-dispatch's `routing.py`). Softmax or sigmoid scores before or after the selection, a per-expert correction bias that
only changes the selection, renormalization (divide by max(sum + epsilon, floor)) and scaling.

- **Canonical ties**: the larger value, then the lower expert id.
- **NaN-safe**: a NaN ranks below every number, -inf included, so a row of NaN logits (a captured decode round's junk
  positions) still picks valid distinct experts (the lowest ids) and gets weights 0; a NaN poisons a softmax over its
  row as it does in torch.
- **One launch, no host synchronization**: a call at a fixed token count is captured in a CUDA graph.

Group limits (groups, selected_groups, group_score_top) are the plugin's torch path.

**Built from the kernel core** (`kernels/core`): the selection keys, the warp argmax and the butterflies are
`crit_kcore/reduce.cuh`'s.

## Applicability

| | |
| --- | --- |
| Inputs | logits [tokens][experts] FP32 rows of unit stride; bias [experts] FP32 or none |
| Outputs | ids [tokens][top] int64 in selection order; weights [tokens][top] FP32 |
| GPUs | NVIDIA sm_80 and newer (warp shuffles only) |
| Shapes | experts up to 1,024, top up to 32 |
| Numerics | FP32: expf, a warp max and a warp sum for the softmax over a row; the picked weights' softmax, sum and renormalization summed in selection order |
| Determinism | the same ids and weights on a repeat and in a replayed graph; a row's picks do not depend on other rows |
| Accuracy | ids equal to an FP64 reference of the rule, weights within 2e-6 (tests, RTX 5090) |

## Interface

```python
from crit_kernels.build import load      # a plugin: crit_plugins.kernels.load_kernel("topk-routing", folder=...)
tr = load("topk-routing")
ids, weights = tr.route(logits, top, bias=None, softmax=True, before=False, renormalize=False,
                        epsilon=0.0, floor=0.0, scaling=1.0, ids=None, weights=None)
```

## Structure and knobs

One structure, `warp-rows` (a warp per token row), keyed by experts and top, tuned at 1, 8, 64, 512 and 4,096 tokens:

| Knob | Values | Default | What |
| --- | --- | --- | --- |
| WARPS | 1, 2, 4, 8, 16 | 4 | warps a block |
| ROWS | 1, 2, 4, 8 | 1 | token rows a warp takes, one after another |

Neither changes what a row computes (no order knob): every configuration gives the same ids and weights bit for bit.
The defaults are the launch the kernel had before it had knobs.

## Build, test, tune

```sh
crit kernels build topk-routing                                  # this GPU, default knobs (or --tuning FILE)
python -m pytest kernels/techniques/experts/topk-routing/tests -q           # against an FP64 reference of the rule
crit kernels sweep topk-routing --model CHECKPOINT --out TUNING_DIR
```

The sweep's bench (`bench.py`) takes the router's experts, top and rule from the checkpoint's config and keeps a
configuration only when its ids and weights equal the untuned build's bit for bit. Run the tests on the target first.

## Measured speed

RTX 5090 (sm_120), default knobs; a CUDA graph of 48 selections, time per selection, against crit's topk-router
in torch (softmax, a stable sort) on the same logits:

| Experts, top | 1 / 4 / 8 / 32 tokens (us) | torch (us) |
| --- | --- | --- |
| 512, 10 | 5.6-6.3 | 27.4-30.4 |
| 256, 8 | 3.5-4.5 | 26.9-29.8 |
| 128, 8 | 2.8-3.5 | 19.4-22.9 |

## Against the library

`warp-rows` names its yardstick in `kernel.json`, which the sweep times on the same shapes (`library_us`, `vs_library`
= library time / tuned time, above 1 the kernel is faster): the torch ops an engine without this kernel runs (softmax
or sigmoid, the bias, gather, the picked weights' softmax or renormalization) around the faster row top-k at each row
count, `torch.topk` below 1,024 rows and FlashInfer's `flashinfer.top_k` (sorted) from 1,024. No library fuses the
rule: FlashInfer's `fused_topk_deepseek` always limits groups (one group of all experts is rejected for top above 1),
and its trtllm-gen routing also builds the permutation for its own GEMMs.

RTX 5090, torch 2.14.1, FlashInfer 0.7.3, a sweep on two router shapes, random logits, tuned kernel against library
(us, vs_library):

| Router | 1 | 8 | 64 | 512 | 4,096 tokens |
| --- | --- | --- | --- | --- | --- |
| 512 experts, top 10, softmax, renormalized | 8.2 / 23.9 (2.9) | 8.2 / 24.3 (3.0) | 8.2 / 23.9 (2.9) | 8.2 / 32.5 (4.0) | 18.4 / 154.9 (8.4) |
| 256 experts, top 8, sigmoid, bias, renormalized | 8.2 / 28.0 (3.4) | 8.2 / 28.0 (3.4) | 8.2 / 28.0 (3.4) | 8.2 / 32.5 (4.0) | 12.3 / 208.3 (17.0) |

The kernel's time up to 512 tokens is the timer's floor of one launch; the library runs four to six launches.
