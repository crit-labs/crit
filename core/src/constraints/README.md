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
| `ByteVocabulary` | Exact decoded bytes per token id; one EOS with an empty piece; other empty pieces are never allowed; a byte trie of the pieces |
| `TokenMatcher` | Request-private language state: `allowed()`, `allowed_bits` (a bitmask for device masks), `append`, `rewind`, `fork`, `ended` |
| `AllowedCache` | Bounded, thread-safe LRU of allowed sets keyed by exact matcher state, shared by the matchers of one language |
| `allowed_by_scan` | The full-vocabulary definition of `allowed()` (every token a fork accepts), for tests and measurements |
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
empty pieces and are never allowed. The constructor also builds a byte trie of the
nonempty pieces (tens of milliseconds for a vocabulary of a few hundred thousand
tokens), which every matcher walks. `byte_vocabulary(tokenizer, eos=None)` builds
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

## Computing allowed sets

`allowed()` walks the vocabulary's piece trie depth first and advances the
matcher's own state one byte per edge (a choice-trie node, a JSON cursor, an Earley
chart), pruning a subtree as soon as its prefix is rejected and pieces longer than
the remaining byte room. A prefix that many tokens share is processed once, and a
rejected prefix rejects all of them; the set is collected as a bitmask and returned
sorted. The result equals the full-vocabulary scan (`allowed_by_scan`), including
grammar chart exhaustion: a token's path visits exactly the charts its scan would.

Allowed sets are cached by exact state: the choice-trie node; the whole JSON cursor
(lexical state, container stack) with the byte room up to the longest piece; for
grammars, whether a sentence ended and the items still waiting for a symbol in the
last chart and in every chart their origins reach (completed items have acted
already), renumbered so equal structures at other offsets share a key. A grammar
entry records the chart states its walk used and serves only states whose remaining
budget covers them, so a hit never hides a `length_error`; states with more than
4,096 waiting items are not cached. A matcher gets a private 4 MiB cache
(`allowed_cache_private_bytes`) unless one is passed in; pass one `AllowedCache`
to every matcher of a language (same vocabulary, grammar or choices, and limits)
to share it across requests, since states such as "inside a string" recur. The
cache throws `invalid_argument` when used with another language.

A grammar rule whose productions are all single bytes is a byte class: recognizers
scan it as one terminal instead of predicting its productions (character classes
otherwise add one item per byte to every chart), and the Python recognizer does the
same so both count the same chart states. The compiler builds unbounded repetition
left recursive, which an Earley recognizer extends in constant work per item, where
right recursion leaves a completion chain as long as the repetition.

## Sampling with a mask

On the device, `allowed_bits` fills a bitmask (bit `t % 32` of word `t / 32`) that
[token-sampling](../../../inference/token-sampling/README.md)'s `enqueue_mask` applies
to logits before selection.


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
required and optional ones, output as compact JSON in declaration order), arrays with
`items`/`minItems`/`maxItems`, strings with `minLength`/`maxLength` or `pattern`,
`boolean`, `null`, `number`, `integer` (bounded integers need both bounds spanning at
most 4,096 values), `enum`, `const`, and:

- `anyOf` (1 to 64 schemas, no sibling keywords) and type lists (each listed type
  takes the keywords that apply to it; a keyword no listed type takes is rejected);
- `oneOf` only when its branches are provably disjoint: different JSON kinds,
  disjoint `enum`/`const` values, or a property every branch requires with disjoint
  `enum`/`const` values (a tagged union). Anything else is rejected, never compiled
  as `anyOf`;
- `$defs` at the root and local `$ref` (`#` or `#/$defs/<name>`, no sibling
  keywords), one grammar rule per target, so recursive schemas work; a reference
  cycle that produces no value is rejected;
- `pattern` with ECMA-262 search semantics on the decoded string: unanchored ends
  take any characters, `.` excludes line terminators, `\d` and `\w` are ASCII and
  `\s` includes the Unicode spaces. Classes take ASCII characters only (a negated
  class admits every other character). The matched part is written in one canonical
  encoding (`"` and `\` escaped, control characters as `\n`, `\r`, `\t`, `\b`,
  `\f` or `\u00XX`, everything else raw UTF-8), the encoding
  `json.dumps(..., ensure_ascii=False)` writes; `pattern` with length bounds is rejected.

`format`, `allOf`, `not`, conditionals, nested `$defs`, remote references and open
objects are rejected.

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

A state's first `allowed()` walks the trie; later visits of an equal state are
cache hits that copy the stored set. Forks copy the matcher's history; grammar
charts grow with ambiguity. Compile ahead of admission, share one `AllowedCache`
per language and profile mask cost within the decode step.

Measured on one thread of an AMD Ryzen 5 9600, before (full-vocabulary scan) and
after, per position, on the byte vocabularies of Nemotron 3.5 Super (131,072
tokens) and Qwen3.8 (248,077 tokens):

| State | Before | After, first visit | After, repeat |
| --- | --- | --- | --- |
| JSON mode, object start / value / number | 2.2-5.0 ms | 0.01-0.06 ms | < 0.01 ms |
| JSON mode, inside a string (127K / 245K tokens allowed) | 5.9 / 10.7 ms | 7.0 / 13.8 ms | 0.13 / 0.19 ms |
| JSON schema grammar, start / integer | 1.3-6.3 ms | 0.004-0.03 ms | < 0.01 ms |
| JSON schema grammar, inside a string | 9.4 / 17.9 s | 0.20 / 0.40 s | 0.13 / 0.21 ms |
| 64 choices | 0.8-2.5 ms | 0.004-0.006 ms | < 0.01 ms |

Writing a 24-character string, the JSON matcher's later positions averaged 0.011
and 0.026 ms and the schema grammar's 8.7 and 17.6 ms (two first visits, then cache
hits). Large allowed sets cost a little more on a first visit than the scan did
(the walk carries a cursor per trie node), so the cache, not the walk, is what
makes them cheap. These numbers illustrate one CPU and two vocabularies.
