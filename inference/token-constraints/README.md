# Constrained speculative decoding

Keep structured output (finite choices, JSON objects, JSON-schema/regex/EBNF grammars) exact under
probability-correct speculative decoding. Matchers, masked sampling and the compiler are Core's
[structured-output service](../../core/src/constraints/README.md); this entry joins a Core `TokenMatcher` with a
raw-logit model state into the `ProbabilityState` that [sampled speculation](../mtp-decode/README.md) drafts,
verifies and repairs.

## When to use it

When a request carries a constraint and the engine decodes with `SampledSpeculator`. Ordinary one-token decoding
needs only Core: `Sampler::sample(logits, AllowedTokens{matcher.allowed()}, ...)`, then append to the matcher.

## How it works

`constrained_logits` applies mtp-decode's temperature/top-k/top-p/min-p `Distribution` to the allowed ids only
and scatters it back to the full vocabulary. Masking precedes the filters; filtering first could discard every
valid token. Greedy ties choose the smallest original id.

`ConstrainedProbabilityState` owns one `LogitState` and one `TokenMatcher`. `append` checks the token on a private
matcher copy before touching the model; `rewind` repairs both to the same frontier; `fork` copies both. A failed
model append or rewind poisons the state and the branch must be retired. After EOS the matcher allows only EOS,
keeping terminal verification defined.

## What the code supplies

[`constrained_state.h`](code/constrained_state.h) (header-only: `constrained_logits`, `LogitState`,
`ConstraintSampling`, `ConstrainedProbabilityState`), [`example.cpp`](code/example.cpp) (run
`crit_constraint_example` after `cmake -S inference -B build/inference`) and the host reference
[`constrained.py`](code/constrained.py). The engine supplies the raw target and draft states and the vocabulary:

```cpp
auto vocabulary = std::make_shared<const crit::core::ByteVocabulary>(decoded_pieces, eos);
auto target = std::make_unique<ConstrainedProbabilityState>(std::move(raw_target),
    std::make_unique<crit::core::JsonObjectMatcher>(vocabulary));
auto draft = std::make_unique<ConstrainedProbabilityState>(std::move(raw_draft),
    std::make_unique<crit::core::JsonObjectMatcher>(vocabulary));
SampledSpeculator loop(std::move(target), std::move(draft), limits, seed, stops /* include eos */);
```

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

## Adapting it

Target and draft use their own constrained probabilities in the same acceptance and residual formulas; share
vocabulary, language and EOS (EOS among the stop ids). Device adapters build one mask per verification position and
restore the matcher with accepted KV/recurrent state.

## Costs and validation

Every `next()` scans the vocabulary and every append copies matcher history: conservative CPU choices. Empty
support throws; it never falls back to unconstrained generation. [Native fixtures](../../tests/inference/constraint_tests.cpp)
check masking, branches, poisoned states and speculative answer frequencies against exact probabilities. No model
or device speed qualification.
