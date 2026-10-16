# Select tokens on the device: greedy, temperature, top-k and top-p

Device-side token selection for logits produced on the GPU, without copying a
vocabulary-sized row to the host per token. Draws are reproducible from a
per-request seed and greedy requests are untouched. This is the device path of
Core's sampler ([Core sampling](../../core/src/sampling/README.md)).

## When to use it

When generation needs temperature, top-k or top-p inside the decode step (after
or inside a captured graph). Apply repetition penalties, logit biases and grammar
masks ([Core's structured output](../../core/src/constraints/README.md)) to the
logits first. Speculative verification with sampling needs the rejection rules in
[mtp-decode](../mtp-decode/README.md), not independent draws per column.

## How it works

Candidates rank by (logit descending, index ascending). Stage 1: 256 blocks each
write the `k` best of their slice (`k = top_k`, 64 when `top_k` is 0, 1 when
greedy). Stage 2: one block selects the global `k` best. Greedy
(`temperature <= 0`) writes the first; otherwise weights are
`exp((l_r - l_0) / temperature)`, top-p keeps the smallest rank prefix reaching
`top_p`, and a uniform draw picks the rank. These are Core `Sampler` steps 7 to
12 with `top_k` capped at 64 and no min-p. The draw is Core's
`crit::core::uniform(seed, position, stream, DrawPurpose::sample)` (Philox4x32-10),
so draws depend only on seed, stream and position.

## What the code supplies

[sampling.h](code/sampling.h) / [sampling.cu](code/sampling.cu): device-resident
`Params` and `enqueue_select(logits, vocab, params, cand_values, cand_ids, out,
stream)`, linking `crit::core`. [sampling.py](code/sampling.py) mirrors the rules
in FP64 with a bit-identical draw. The host oracle is `crit::core::Sampler`.

## What the engine supplies

One `Params` per sequence, set from `crit::core::resolve_sampling` with
`position = 0`, and rewritten from `TokenHistory::outputs()` after trims. Keep the
plain argmax path for greedy requests. Requests with penalties, logit bias, min-p,
min_tokens or top_k above 64 need Core's host `Sampler`. Report supported options
through `ModelInfo::sampling`.

## Worked example

From `code/`:

```python
import sampling as s

row = [2.0, 1.0, 2.0, -1.0, 0.5]
assert s.select(row) == 0                  # tie between 0 and 2, lowest index wins
assert [t for t, _ in s.nucleus(row, 0.7, top_p=0.9)] == [0, 2, 1]   # 0.446, 0.446, 0.107
draws = [s.select(row, 0.7, 0.9, 0, seed=1, position=c) for c in range(8)]
assert draws == [2, 2, 0, 2, 0, 0, 0, 2]   # crit::core::Sampler draws the same tokens
```

## Adapting it

Preserve the rank order and tie-break, one draw per sampled token at its own
position, and Core's draw. Block count, candidate cap and fusing into the head
kernel are adaptable.

## Costs and validation

Selection is small next to the head projection. FP32 weights can make a draw
within rounding of a cumulative boundary differ from Core's double-precision
`Sampler`; NaN logits are not filtered.
[token_sampling_tests.cpp](../../tests/inference/token_sampling_tests.cpp) (under
`CRIT_INFERENCE_CUDA`) checks tie-break, seed repeatability, agreement with
`Sampler::sample` away from boundaries and draw frequencies. No performance or
quality measurement.
