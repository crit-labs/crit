# Structured output (constraints)

An optional Core service that restricts generated output to a language over its
exact bytes, as OpenAI's `response_format` (JSON mode, JSON schema) and
regex/grammar constraints require. The runtime never calls it. Native code:
[constraints.h](../../include/crit/constraints.h), [constraints.cpp](constraints.cpp)
and [grammar.cpp](grammar.cpp); masks reach sampling through
`Sampler::sample(logits, AllowedTokens{...}, ...)` in [sampling.h](../../include/crit/sampling.h).
Python: [`crit_core.constraints`](../../python/crit_core/constraints). Tests:
[constraints_tests.cpp](../../../tests/core/constraints_tests.cpp) and
[test_core_constraints.py](../../../tests/test_core_constraints.py), among others.

| API | Purpose |
| --- | --- |
| `ByteVocabulary` | Exact decoded bytes per token id; one EOS with an empty piece; other empty pieces are never allowed |
| `TokenMatcher` | Request-private language state: `allowed()`, `append`, `rewind`, `fork`, `ended` |
| `ChoiceGrammar`, `ChoiceMatcher` | A finite set of byte strings, whatever the token segmentation |
| `JsonObjectMatcher`, `JsonLimits` | One JSON object of any shape (JSON mode) |
| `ByteGrammar`, `GrammarMatcher`, `GrammarLimits` | A compiled byte grammar (JSON schema, regex, EBNF) through a bounded Earley recognizer |
| `AllowedTokens`, `Sampler::sample`/`distribution` overloads | Mask everything outside `allowed()` before the sampling rules |
| `constraint_readout` | Restricted probabilities and the allowed set's mass under the raw softmax |
| `crit_core.constraints` (Python) | `compile_language`, `CompilerCache`, `PreparationService`, `byte_vocabulary(Tokenizer)`, `matches`, `json_object_matches` |

## Using it in a session

```cpp
#include <crit/constraints.h>
#include <crit/sampling.h>
using namespace crit::core;

// Once per model: the decoded bytes of every token (crit_core.constraints.byte_vocabulary gives them in Python).
auto vocabulary = std::make_shared<const ByteVocabulary>(decoded_pieces, eos);
// Per request: JSON mode, or a grammar artifact compiled by crit_core.constraints (shared, immutable).
std::unique_ptr<TokenMatcher> matcher = std::make_unique<JsonObjectMatcher>(vocabulary);
// matcher = std::make_unique<GrammarMatcher>(ByteGrammar::load(artifact), vocabulary);
// Per position, after reading logits:
const std::vector<Token> allowed = matcher->allowed();
const Choice choice = sampler.sample(logits, AllowedTokens{allowed}, params, history);
matcher->append(choice.token);   // on accept; rewind(n) together with KV on rollback
```

The matcher and the model describe the same accepted output: append, rewind and
fork both together. Put the vocabulary's EOS among the request's stop tokens;
after EOS, `allowed()` returns only EOS. `allowed()` never returns an empty set:
a dead end or a reached limit throws `domain_error`, and the request fails
instead of falling back to unconstrained output. Reserve completion room in the
output budget, since a valid prefix does not guarantee the rest can finish it.

## Vocabulary

Pieces are the bytes the detokenizer appends for each token, not the stored
spellings, and may end inside a UTF-8 character. Special and control tokens get
empty pieces and are never allowed. `byte_vocabulary(tokenizer, eos=None)` builds
the table from a `crit_core.text.Tokenizer` for byte-level, Metaspace and
SentencePiece byte-fallback decoders; others raise `TextError`. Limits: 1,048,576
entries and 64 MiB of piece bytes.

## Languages

| Language | Accepts | Bounds |
| --- | --- | --- |
| `ChoiceGrammar` | Choices as a byte trie; EOS at a complete choice | 1 to 65,536 choices, 1 MiB |
| `JsonObjectMatcher` | One JSON object; duplicate names and numeric range are not checked | `JsonLimits` defaults: depth 64, output 1 MiB, tokens 32,768 |
| `ByteGrammar` | A `crit-byte-grammar-v1` artifact; `GrammarMatcher` runs a bounded Earley recognizer per byte, so recursive, ambiguous and nullable rules work | 4,096 rules, 4 MiB; `GrammarLimits` defaults: 4,096 output bytes and tokens |

Grammar exhaustion throws `length_error` without changing the matcher.

## Sampling with a mask

`Sampler::sample(logits, AllowedTokens{ids}, params, history, column, top)` and
the matching `distribution` overload mask every id outside `ids` to -inf before
step 2 (bias) of the [sampling order](../sampling/README.md), so temperature,
top_k, top_p and min_p see only allowed tokens. Greedy ties go to the lowest
allowed id. The reported log-probability and `top` alternatives stay raw over
the whole row. Ids must be nonempty, sorted, unique and in range. The overloads
allocate nothing per call.

## Compiling languages (Python)

`compile_language(kind, source)` turns a JSON schema (`"schema"`), a byte regex
(`"regex"`) or line EBNF (`"ebnf"`) into a tokenizer-independent artifact.
Unsupported constructs raise `UnsupportedConstraint` (a `ValueError`); nothing is
silently relaxed. Offline: `python -m crit_core.constraints.compiler schema reply.schema.json reply.grammar`.

The [JSON Schema 2020-12](https://json-schema.org/draft/2020-12/json-schema-validation)
subset: closed objects (`additionalProperties: false`, at most 128 properties,
output as compact JSON in declaration order), arrays with `items`/`minItems`/`maxItems`,
strings with `minLength`/`maxLength`, `boolean`, `null`, `number`, `integer`
(bounded integers need both bounds spanning at most 4,096 values), `enum` and
`const`. `$ref`, unions (`anyOf`, `oneOf`, type lists), `pattern`, `format` and
open objects are rejected.

`regex` matches the whole byte string (no lookaround, backreferences or flags).
`ebnf` takes one `name ::= expression` per line with a required `root`. Neither
is full PCRE or GBNF. Sources are at most 1 MiB. `CompilerCache` is a
caller-owned LRU (16 entries, 8 MiB by default).

## Preparation (Python)

`PreparationService` compiles on bounded CPU workers while requests wait;
identical concurrent requests share one job. Call `request` before model
admission, poll `ready()`/`state` and map pending preparation to
`Admission::deferred`; failures and timeouts reject the request, never fall back
to unconstrained output. `new_matcher(factory)` returns a `PreparedMatcher`
lease that keeps the artifact charged until closed. Workers, queues and bytes are
bounded; byte bounds exclude Python overhead and native charts, which the engine
budgets.

## Checking outputs (Python)

`matches(artifact, data)` and `json_object_matches(data)` check whole outputs;
`crit serve` uses them to verify engine output.

## Costs

The reference matchers scan the whole vocabulary at every position and copy
history per fork; grammar charts grow with ambiguity. Compile ahead of admission
and profile mask cost within the decode step before using it in a
latency-sensitive path; device masks or transition caches are engine work.
