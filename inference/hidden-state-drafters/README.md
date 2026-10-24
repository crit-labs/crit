# Draft with published drafters that read the target's hidden states

Use a published companion drafter that reads the target's own hidden states, EAGLE-3
(one autoregressive layer over a few of the target's layers) or a block drafter (DFlash,
DSpark: a whole block of drafts in one non-causal pass), with the device speculative round.
Most model families ship no multi-token-prediction head, and for them these drafters are the
difference between copy proposals and several accepted tokens per round. Core reads every
published format into one description; this entry supplies the device glue between the
target's verify, the drafter and the round.

## When to use it

When a drafter trained for the target is published (speculators-format checkpoints for
Llama, Qwen3, gpt-oss, Gemma 4 and others; native EAGLE-3, DFlash and DSpark checkpoints)
and the target has no built-in draft head, or its head accepts less. Use the drafter as
released: crit never trains or fine-tunes one. Published numbers put EAGLE-3 near 3.75 to
4.8 tokens per round on an 8B model at temperature 0, and DFlash's authors report about
2.2x EAGLE-3's speedup at temperature 1. Measure on the engine's own mix before choosing.

Not for drafters that need the target's logits or KV of another model family (those are
ordinary draft models), and not when the taps' extra writes and the drafter's pass cost more
than the tokens they add.

## How it works

**One description.** `crit_core.model.drafters.read_drafter(config, target_layers=...)`
reads speculators configs (`speculators_model_type` eagle3, peagle, dflash, dflash2,
dspark), native EAGLE-3 (an `Eagle3` architecture, default taps `(2, L // 2, L - 3)`),
native DFlash (`dflash_config`), native DSpark and crit's own `crit models drafter` heads
(kind `mtp-head`, one tap: the target's final hidden state) into a `Drafter`: kind, taps, widths,
draft vocabulary, block size, mask token, anchor sampling, causality, a confidence head and
the publisher's draft width. **Taps** are residual-stream indices: tap `k` is the hidden
state after `k` target decoder layers. EAGLE-3 and speculators spell taps directly; native
DFlash and DSpark name the layer whose output is read (`target_layer_ids`, tap = id + 1),
an off-by-one the table keeps in one place. `crit run ... recon --drafter PATH` records
each companion drafter's description in `recon.json` and the drafting plan, checked against
the target's depth and width.

**Taps in verify.** The target's verify writes, for every column, the residual stream at
each tap: `[K + 1][taps][target_hidden]`. Column `c` sits at position `base + c` and reads
the round's pending token (`c = 0`) or draft `c`.

**EAGLE-3 catch-up.** The drafter's input at position `p` is the target's fused taps at `p`
with the embedding of the token at `p + 1`; it predicts the token at `p + 2`. After a round
emits `e_0 .. e_{n-1}`, positions `base .. base + n - 1` are materialized and the token after
`base + c` is `e_c`, so the drafter ingests (taps of column `c`, `e_c`) for `c < n`: a fixed
`K + 1` wide catch-up masked by the device's count, run first in the next draft step, after
Core's truncation has settled `n`. Draft steps after the first in a round feed the drafter's
own output hidden state.

**Block drafting.** The drafter fills `[anchor, mask x (block_size - 1)]` at positions
`frontier .. frontier + block_size - 1` in one pass whose attention is non-causal inside the
block ([flash-prefill-attention](../flash-prefill-attention/README.md) with `causal = false`),
with the target's taps injected as context. Positions `1 .. block_size - 1` give the drafts
(all `block_size` with `sample_from_anchor`). One pass replaces `K` sequential draft steps,
which at batch 1 are bound by launch latency more than by bytes.

**Draft vocabulary.** A reduced draft vocabulary maps back through EAGLE-3's offset table:
target id = draft id + `d2t[draft id]`. Greedy drafts map their argmax; a sampled drafter's
law rows scatter to the target vocabulary for
[token-sampling](../token-sampling/README.md)'s rejection sampling.

**Confidence.** With a confidence head (DSpark) or the drafter's own top probability, drafts
from the first one below a threshold become `-1`, which no prediction matches, so the round
stops there without a narrower graph.

## What the code supplies

| File | Supplies |
| --- | --- |
| [crit_core/model/drafters.py](../../core/python/crit_core/model/drafters.py) | `read_drafter`, `read_drafter_file`, `Drafter` (Core): every published format, one tap convention |
| [drafting.py](code/drafting.py) | Host reference: `eagle_inputs`, `block_input`, `block_drafts`, `map_tokens`, `scatter_law`, `truncate` |
| [hidden_drafters.h](code/hidden_drafters.h) / [.cu](code/hidden_drafters.cu) | CUDA `enqueue_eagle_ingest`, `enqueue_block_input`, `enqueue_map_tokens`, `enqueue_scatter_law`, `enqueue_truncate_drafts` over device-speculative-round's control block and packed result |

## What the engine supplies

The drafter's arithmetic as published (EAGLE-3's fc over the fused taps, its decoder layer
and head; DFlash's layers with target-context injection), the target's tap writes in verify
and prefill, the drafter's KV, and the argmax over the drafter's head (for example
[draft-vocabulary-subset](../draft-vocabulary-subset/README.md)'s gather-argmax). A round
with an EAGLE-3 drafter:

```text
draft:  enqueue_eagle_ingest(control, result) -> catch-up tokens/positions; drafter over them (taps of those columns)
        K drafter steps (own hidden after the first), argmax over its head, enqueue_map_tokens -> drafts
        optional: enqueue_truncate_drafts(drafts, confidence, threshold)
verify: target over K + 1 columns writing logits and taps; predictions (argmax or token-sampling)
```

With a block drafter, `draft` is `enqueue_block_input`, one drafter pass, argmax per block
position and `enqueue_map_tokens`.

## Worked example

From `code/`: the catch-up after a round that emitted two of four possible tokens, a
four-token block, and EAGLE-3's offset table.

```python
from drafting import PAD, block_drafts, block_input, eagle_inputs, map_tokens, scatter_law, truncate

# Verify columns 0..3 read the pending token and three drafts; the round emitted e0 = 11, e1 = 12. The drafter
# ingests (taps of column 0, 11) and (taps of column 1, 12): column c's hidden state with the token after it.
assert eagle_inputs([11, 12], width=4) == [(0, 11), (1, 12), (2, PAD), (3, PAD)]

block = block_input(anchor=7, mask_token=99, block_size=4)       # [pending, mask, mask, mask]
assert block == [7, 99, 99, 99]
assert block_drafts([3, 8, 1, 4]) == [8, 1, 4]                   # mask positions draft; the anchor is known
assert block_drafts([3, 8, 1, 4], sample_from_anchor=True) == [3, 8, 1, 4]

d2t = [0, 4, 7]                                                  # draft ids 0, 1, 2 are target ids 0, 5, 9
assert map_tokens([2, 0, PAD], d2t) == [9, 0, PAD]
assert scatter_law([0.5, 0.3, 0.2], d2t, 10)[5] == 0.3
assert truncate([5, 6, 7], confidence=[0.9, 0.3, 0.8], threshold=0.5) == [5, PAD, PAD]
```

## Adapting it

Preserve: taps as residual-stream indices from Core's reader (never re-derive a format's
convention), the EAGLE-3 alignment (taps at `p` with the token at `p + 1`), catch-up from the
device's settled count, block positions from the frontier, `d2t` mapping before
verification, and drafts beyond a confidence cut that no prediction can match. Adaptable:
fusing the tap writes into the target's residual adds, the drafter's precision, the draft
width up to the publisher's, and the confidence threshold (tuned on general text, never on
Eval's prompts).

## Costs and validation

Taps add `(K + 1) x taps x target_hidden` writes per verify; EAGLE-3 adds `K` small drafter
passes per round, a block drafter one pass over `block_size` positions.

[test_hidden_drafters.py](../../tests/test_hidden_drafters.py) checks Core's reader on
speculators EAGLE-3 / DFlash / DSpark, native EAGLE-3 (default taps), native DFlash and DSpark
(one tap convention across formats), rejected and foreign configs and a drafter directory,
plus the drafting rules. [hidden_drafter_tests.cpp](../../tests/inference/hidden_drafter_tests.cpp)
checks the kernels against those rules, including catch-up after a Core truncation. No
drafter arithmetic or speedup is qualified here.
