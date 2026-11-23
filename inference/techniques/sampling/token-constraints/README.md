# Constrained speculative decoding

Keep structured output (finite choices, JSON objects, JSON-schema/regex/EBNF grammars) exact under
probability-correct speculative decoding. Matchers, masked sampling and the compiler are Core's
[structured-output service](../../../core/src/constraints/README.md); this entry joins a Core `TokenMatcher` with a
raw-logit model state into the `ProbabilityState` that [sampled speculation](../../speculation/mtp-decode/README.md) drafts,
verifies and repairs.

## When to use it

When a request carries a constraint and the engine verifies sampled drafts. Ordinary one-token decoding
needs only Core's matcher and the sampler plugin: mask the row to `matcher.allowed()`, sample, then append to the matcher.

## How it works

`constrained_logits` applies mtp-decode's temperature/top-k/top-p/min-p `Distribution` to the allowed ids only
and scatters it back to the full vocabulary. Masking precedes the filters; filtering first could discard every
valid token. Greedy ties choose the smallest original id.

`ConstrainedProbabilityState` owns one `LogitState` and one `TokenMatcher`. `append` checks the token on a private
matcher copy before touching the model; `rewind` repairs both to the same frontier; `fork` copies both. A failed
model append or rewind poisons the state and the branch must be retired. After EOS the matcher allows only EOS,
keeping terminal verification defined.

## What the code supplies

The host reference [`constrained.py`](code/constrained.py): `constrained_law`, `accept_probability` and `residual`.
The engine supplies the raw target and draft laws and the vocabulary.

## Worked example

Run from `code/` with Python 3.11+:

```python
from constrained import accept_probability, constrained_law, residual

# Vocabulary a, b, c, EOS, ab, x: after "a" the language allows only b and c.
target = constrained_law([1, 0, 2, 3, 2, 99], [1, 2])
draft = constrained_law([2, 3, 0, 1, 0, 99], [1, 2], top_k=5)  # top_k above the allowed count keeps both
assert target[5] == draft[5] == 0 and abs(sum(target) - 1) < 1e-12
assert constrained_law([1, 0, 2, 2, 2, 99], [2, 3, 4], temperature=0)[2] == 1  # greedy: lowest allowed id
p_accept = accept_probability(target, draft, 1)
repair = residual(target, draft)
assert 0 < p_accept < 1 and repair[2] == 1 and repair[5] == 0  # the repair stays inside the language
```

## On the device

The plugin's device form masks each constrained row before Core's law and then
draws by token-sampling's device law: `sample(logits, params)` takes a row's
`crit_plugins.Allowed` (Core's matcher bitmask) as its `allowed`, and
`staged(rows, device, vocabulary)` keeps the masks in a device buffer at a fixed
address, which `set(params)` fills from the host's bitmask words (one
asynchronous copy) before a launch and `sample(logits)` unpacks inside it, so a
step with constrained rows captures in a CUDA graph. Columns past the vocabulary
are never allowed, for an unconstrained row either, and a row a step does not
stage keeps no mask from an earlier one. Both give the host form's tokens
(`select`).

Inside captured decode rounds (`rounds:sampler`) it is token-sampling's round
form (option `rounds`, as token-sampling's) with each round slot's mask staged
beside its rules: `stage(slot, params)` keeps the slot's `allowed` words in a
device buffer at a fixed address, and a round masks every logits row of a
constrained slot before the law (predictions and log-probabilities) and cuts
its drafts, so the slot emits one token a round, its column 0's, under the
allowed set Core staged. `stage_mask(slot, allowed)` replaces only the slot's
set (one copy of its bitmask words), which Core's step engine calls between two
rounds once the request's matcher took the last round's token
([native decode rounds](../../../core/src/runtime/README.md#native-decode-rounds)).
So selecting token-constraints keeps captured rounds for every request: the
round driver's `Rounds.masks` is true with it, and a model plugin binds a
constrained request at width 0.

Sampled verification under a constraint stays the host form (`predictions`):
Core decodes a constrained request plainly, one token a round or step
(`StepRequest.plain`), because its matcher computes the allowed set of the next
position only once the tokens before it are known. A verify window would need
the mask at every draft position, from the matcher advanced along each draft
(and every branch of a tree), before the window is computed.

## Adapting it

Target and draft use their own constrained probabilities in the same acceptance and residual formulas; share
vocabulary, language and EOS (EOS among the stop ids). Device adapters build one mask per verification position and
restore the matcher with accepted KV/recurrent state.

## Costs and validation

Every `next()` reads the matcher's allowed set (Core walks the vocabulary's piece trie, or hits its allowed-set cache for a repeated state) and every append copies matcher history. Empty
support throws; it never falls back to unconstrained generation. No model
or device speed qualification.
