# Draft against a vocabulary subset of the existing output head

Drafting computes logits only for an indexed subset of `S` rows of the model's
existing output head; verification still uses the full head of `V` rows. Greedy
output is unchanged, because only the target accepts tokens. The subset changes
acceptance and draft cost only.

## When to use it

Use it when the drafter's output head is a large share of its cost: a small
drafter, or one sharing the target's vocabulary projection (an MTP head or a
self-draft), where a modest subset covers most target tokens. It is pointless when
drafting is dominated by something else, when tokens spread evenly across the
vocabulary, or when the draft head is already cheap next to verification.

> **Generalization comes first.** In one measured engine, a 98,304-row subset of a
> 248,320-row head built from mostly English output and sized by crit Eval's
> decode score (+12% there) cut tokens per round on fresh prompts from 2.66 to
> 1.85 in Chinese, 2.60 to 2.02 in Japanese, 2.66 to 1.90 in Russian and 2.68 to
> 2.45 in German. Build the subset from a broad multilingual, multi-domain corpus
> unrelated to any benchmark, never size it by a benchmark score, and check it on
> held-out prompts in several languages (crit Eval's held-out set and generality
> gate) against the full head. Without that evidence, draft with the full head.

## How it works

**Choose the subset.** Count target tokens by decoding the target greedily on a
mixed prompt set (prose, code, reasoning, structured output, the users'
languages); `choose_subset` takes those counts. Take the `S` most frequent IDs
plus required tokens, sorted ascending; ties prefer the lower ID. Without
statistics, `prior_subset` takes the lowest IDs, which approximate frequency in a
merge-ordered (BPE) vocabulary but are weakest for late-merged languages: a
quarter-vocabulary prefix cut German tokens per round from 2.33 to 1.87 in one
engine. Special tokens usually sit at the vocabulary's end; add every one the
target emits (stops, turn delimiters, tool markers) as required, or the subset
mismatches at each one when generation runs past a stop.

**Read and map.** A prefix plus a few required tokens is a few contiguous row
ranges (`SubsetHead.ranges()`), readable as ordinary row blocks without an index
array. The draft head returns subset index `i`; the token is `rows[i]`.

**Gather, do not requantize.** Row `rows[i]` of the subset is the same encoded
bytes, scales and decoder as the model's row. If scales are grouped across rows,
carry each row's group scale unchanged. Re-encoding changes logits and costs
acceptance near ties, never correctness.

**Correctness.** With the same row arithmetic a gathered logit equals the
full-head logit exactly. Under the first-maximum tie rule, if the target argmax
`t*` is in the subset the subset argmax is `t*`; if not, that draft mismatches.
Per-position acceptance is bounded by coverage.

**Cost.** For `K` draft positions and `b` bytes per weight:

```text
full draft head   = K * V * H * b
subset draft head = K * S * (H * b + index_bytes)
saved per round   = K * (V * H * b - S * (H * b + index_bytes))
```

With per-position acceptance `a`, expected tokens per round are
`(1 - a^(K+1)) / (1 - a)`. Coverage overstates what a smaller subset keeps, since
a miss ends the round: a 98,304-row subset covered over 99% of target tokens, yet
cutting it to 49,152 and 32,768 rows dropped tokens per round on a long summary
from 3.54 to 2.83 and 2.38, more than the cheaper head saved. Choose `S` from
measured rounds with `choose_size`.

## What the code supplies

- [subset.py](code/subset.py): `choose_subset`, `prior_subset`, `SubsetHead`
  (mapping, validation, `ranges()`), first-maximum `argmax`, `target_argmax`,
  `draft_argmax`, `coverage`, `expected_tokens`, `draft_head_cost`
  (`DraftHeadCost`) and `choose_size` (most tokens per millisecond).
- [subset_argmax.h](code/subset_argmax.h) / [subset_argmax.cu](code/subset_argmax.cu):
  `enqueue_gather_argmax(GatherArgmax, stream)`, a reference CUDA kernel over FP32
  rows with a row stride, fixed sequential dot products and first-maximum
  reduction. Null `rows` selects the full head, so it also serves as the
  verifier's argmax reference.

## What the engine supplies

The head encoding and decoder, hidden states, frequency statistics, subset size,
and the draft step itself (inside the captured draft step for a GPU
[device round](../device-speculative-round/README.md)). Keep the subset as an
immutable ascending `int32` device array beside the head; recompute it offline or
between requests, never inside a captured round. Use the same row decoder for
draft and verify. Account for the index and any contiguous row copy in Core
resources.

## Worked example

Run from `code/`. Whenever the target's argmax is inside the subset, the draft
equals it.

```python
from subset import SubsetHead, choose_subset, draft_argmax, draft_head_cost, target_argmax

counts = {0: 1, 1: 40, 2: 3, 3: 25, 4: 30, 5: 2}
subset = SubsetHead(choose_subset(counts, 3), vocabulary=6)
assert subset.rows == (1, 3, 4) and subset.index(3) == 1 and subset.token(2) == 4

head = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.5], [0.9, 0.1], [0.5, 0.5], [0.0, 2.0]]
inside = [1.0, 0.2]                    # target argmax is token 1, in the subset
assert target_argmax(head, inside) == draft_argmax(head, inside, subset) == 1
outside = [0.0, 1.0]                   # target argmax is token 5, outside the subset
assert target_argmax(head, outside) == 5 and draft_argmax(head, outside, subset) == 4

cost = draft_head_cost(vocabulary=6, hidden=2, size=3, draft_tokens=4, bytes_per_weight=2)
assert (cost.full_bytes, cost.subset_bytes) == (96, 96)  # tiny heads do not pay for the index

# Choose the size from measured rounds, not coverage: the largest head wins here despite costing more.
from subset import choose_size
best, rates = choose_size({32768: (2.38, 17.0), 49152: (2.83, 17.4), 98304: (3.54, 18.6)})
assert best == 98304 and rates[98304] > rates[49152]

# No statistics: the lowest IDs plus the control token at the end, read as two runs.
from subset import prior_subset
prior = SubsetHead(prior_subset(6, 3, required=(5,)), vocabulary=6)
assert prior.rows == (0, 1, 5) and prior.ranges() == ((0, 2), (5, 6))
assert draft_argmax(head, outside, prior) == 5  # the control token is draftable
```

## Adapting it

Preserve: full-head verification; subset rows gathered without re-encoding;
unique ascending rows with index `i` mapping to `rows[i]`; shared token IDs and
first-maximum tie rule. Adaptable: frequency source or prefix prior, required
tokens, size, contiguous runs or indexed reads, fusing the gather into the draft
GEMV, and per-domain subsets chosen between requests.

## Costs and validation

The index costs `4 * S` bytes (none as contiguous runs); a contiguous copy costs
`S * H * b`. Too small a subset lowers acceptance and can make rounds slower than
ordinary decoding. Duplicate or unsorted rows are rejected.

[Python fixtures](../../tests/test_device_rounds.py) cover subset choice, the
prefix prior, mapping, the inside-subset equality including ties, and a
speculative loop that reproduces ordinary greedy output token for token. The
[CUDA fixture](../../tests/inference/draft_subset_tests.cpp) under
`CRIT_INFERENCE_CUDA` checks on a 1,000-row padded head that gathered logits
equal full-head logits bit for bit and the tie rule holds. Acceptance for a real
model must be measured.
