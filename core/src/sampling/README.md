# Token sampling

[sampling.cpp](sampling.cpp) implements [sampling.h](../../include/crit/sampling.h),
a library sessions call after reading logits (the [runtime](../runtime/README.md)
never calls it). It depends only on the standard library. Request fields and
`TokenLogprob`, `QueryScores` and `LogitBias` come from [core.h](../../include/crit/core.h).
Tests: [sampling_tests.cpp](../../../tests/core/sampling_tests.cpp).

| API | Purpose |
| --- | --- |
| `philox4x32`, `uniform`, `DrawPurpose` | Header-only, `constexpr`, `CRIT_HOST_DEVICE` counter-based draws |
| `SamplingParams`, `resolve_sampling`, `validate` | Resolved request rules and their range checks |
| `TokenHistory` | Prompt, committed and staged outputs as sparse counts |
| `Sampler::sample`, `Sampler::distribution` | One draw with its raw log-probability; the exact law of that draw |
| `AllowedTokens` overloads | The same with a [structured-output constraint](../constraints/README.md)'s allowed set masked first |
| `log_softmax`, `top_logprobs`, `query_logprobs` | Raw log-probabilities of one row |
| `score_logit_rows` | A `QueryScores` for `Model::score_queries` from strided logit rows |
| `accept_greedy` | How many tokens a greedy verify round emits |

## Random draws

Every draw is a pure function of `(seed, position, stream, purpose)`, so rollback,
trimming, cancellation and speculative redo need no generator state. Sampling
uses position = committed outputs plus the verify column, stream =
`SamplingParams::stream` and purpose `DrawPurpose::sample` (0). `draft` (1),
`accept` (2), `residual` (3) and `block` (4, block verification's acceptance draws)
are reserved for speculative sampling; adapters may use 16 (`user`) and above.

`philox4x32` is Philox4x32-10 exactly as in Random123, on unsigned 32-bit words:

```
key     = {k0, k1} = {seed & 0xffffffff, seed >> 32}
counter = {c0, c1, c2, c3} = {position & 0xffffffff, position >> 32, stream, purpose}
repeat 10 times (round 0 to 9):
    if round > 0: k0 += 0x9E3779B9; k1 += 0xBB67AE85
    p0 = 0xD2511F53 * c0   (64-bit product; hi0 = p0 >> 32, lo0 = p0 & 0xffffffff)
    p1 = 0xCD9E8D57 * c2   (64-bit product; hi1, lo1 likewise)
    {c0, c1, c2, c3} = {hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0}
result word[0..3] = {c0, c1, c2, c3}

uniform = double(((uint64(word[0]) << 32) | word[1]) >> 11) * 2^-53     in [0, 1), 53 random bits
```

The conversion is exact, so a device kernel with the same integer steps
reproduces `u` bit for bit. Known answers (Random123 vectors, checked at compile
time by the tests):

| seed | position | stream | purpose | word[0..3] | uniform |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 6627e8d5 e169c58d bc57ac4c 9b00dbd8 | 0xcc4fd1abc2d38 * 2^-53 |
| 2^64 - 1 | 2^64 - 1 | ffffffff | ffffffff | 408f276d 41c83b0e a20bc7c6 6d5451fd | 0x811e4eda83907 * 2^-53 |
| 0x299f31d0a4093822 | 0x85a308d3243f6a88 | 13198a2e | 03707344 | d16cfe09 94fdcceb 5001e420 24126ea1 | |

## Rules

`resolve_sampling` takes each set request field, else the adapter default. A
non-empty request `logit_bias` replaces the default list; `suppress_until_min`
(usually the stop tokens) and `stream` always come from the defaults; its
`vocabulary` is the valid vocabulary. `validate` (also run by `Sampler` on every
call) mirrors the runtime's request checks. Ids in `suppress_until_min` at or
above the vocabulary are ignored, so adapters can pass the runtime's accepted stop
tokens unchanged. `top_k` has no upper bound here; one at or above the selectable
count keeps every token.

An unset seed takes `defaults.seed` (0 unless set), so unseeded requests sharing
one defaults object draw identical `u` and identical prompts give identical
completions. Unless that is intended, put a fresh random seed in `defaults` per
request and record it for replay.

`sample` and `distribution` process one row of exactly `vocabulary()` entries in
IEEE double, in this order, where n(t) counts token t among committed outputs plus
`staged[0, c)` for verify column c:

1. Load: `v[i] = double(logits[i])`. NaN or +inf is rejected; -inf masks. With
   `AllowedTokens`, ids outside the allowed set become -inf.
2. Bias: `v[t] += double(bias)`; a -inf bias bans the token.
3. Suppression: while `outputs() + c < min_tokens`, `v[t] = -inf` for each
   suppress_until_min token. With `AllowedTokens` whose ids are all
   suppress_until_min tokens (`suppression_yields`), suppression is skipped: a
   constraint that only allows a stop (a finished grammar) ends the output
   instead of failing the request.
4. Repetition, once per distinct unmasked token in the prompt (when counted) or
   with n(t) > 0: `v = v > 0 ? min(v / r, DBL_MAX) : max(v * r, -DBL_MAX)`.
5. Frequency and presence, for n(t) > 0: `v -= f * n(t) + p`.
6. Selectable tokens have `v != -inf`; none fails with "no selectable token".
7. Greedy: at temperature 0, the lowest id with the largest v.
8. Rank by v descending, then id ascending. With m the largest v, `w = exp((v - m) / T)`.
9. top_k: when 0 < top_k < selectable count, keep the first top_k in rank order;
   S is their weight sum.
10. top_p: when top_p < 1, keep the shortest rank prefix whose cumulative weight
    reaches `top_p * S` (exact reach stops there; at least one token remains).
11. min_p: keep `w >= min_p` (equality kept; zero weights dropped; the top token
    always stays).
12. Draw: `u = uniform(seed, outputs() + c, stream)`; the token is the first kept
    token in rank order whose cumulative weight exceeds `u * S_kept`, else the
    last kept token.

The kept set is always a rank prefix. Without top_k, S and S_kept are summed in id
order; with top_k, in rank order. `distribution` returns the kept tokens in rank
order with `log p = (v - m) / T - log(S_kept)`; a greedy law is `{token, 0}`.

`u` is identical on every platform; the chosen token is deterministic per build
and platform. Another implementation of these steps agrees except where `exp`
and `log` differ in the last bit near a CDF boundary.

## Log-probabilities

Reported log-probabilities are raw: the unprocessed row, normalized over exactly
the given entries, before temperature, penalties, bias and filters:

```
m       = max over finite x
log_sum = log(sum over x[i] != -inf of exp(double(x[i]) - m))      double, accumulated in id order
logprob[i] = float((double(x[i]) - m) - log_sum)                   -inf for masked entries
```

m and log_sum stay separate so rows of very large logits keep their
normalization. `Choice::logprob` is the chosen token's raw value even when bias or
penalties chose it. Alternatives are ranked by float log-probability descending,
then id ascending. `log_softmax` may write in place only on an exact alias.

## Token history

`TokenHistory` holds the prompt (unless `count_prompt_for_repetition` is false),
committed outputs and one staged round as sparse counts, never O(vocabulary).
Repetition counts distinct prompt and output tokens; frequency and presence count
outputs only. Keep it on the same frontier as KV state:

- Decode: `sample(logits, params, history)`, then `stage({token})` when handing
  the token to Core, then `commit(accepted)` when the runtime commits (0 drops).
- Speculative verify: `stage(drafts)`, `sample` each column c, `accept_greedy`,
  then `rollback()`, `stage(emitted)` and `commit(accepted)`.

Redoing a column after rollback yields the same token for the same logits.
`stage` allocates everything, so `commit` never allocates and a throwing `commit`
changes nothing; a second `stage` before `commit` or `rollback` is a
`logic_error`. `host_bytes(prompt, outputs)` bounds the heap for reservation
through Resources. A history has one owner and is not thread-safe; copies (for
example one per choice when n > 1) are independent.

## Scoring and verification

`score_logit_rows` reads `rows` rows of `stride` floats whose first `valid`
entries are the valid vocabulary, and returns target log-probabilities,
`rows x per_target` query values, optional argmax (lowest id on ties) and
`rows x top_k` alternatives. A masked target yields -inf, which the runtime
rejects.

`accept_greedy(drafts, predictions, remaining, stops)` emits
`predictions[0, emitted)`: the matched drafts, then a correction or bonus token,
stopping early at a stop token or when `remaining` runs out. `stopped` reports a
final stop.

## Cost and limits

Processing is O(V + B + S + H + c) for row size V, bias entries B, suppressed
tokens S and tracked tokens H; top_k uses a heap or selection, and top_p a
2048-bin log-weight histogram, so typical rows avoid a full sort. `Sampler`
allocates scratch once (about 28 bytes per vocabulary entry plus 36 KiB;
`scratch_bytes(V)` gives the exact figure), `sample` never allocates, and one
sampler per worker thread is the intended use.

Vocabulary is 1 to 2^31 entries, `logit_bias` at most 1024 entries
(`logit_bias_limit`), and rows must hold exactly the valid vocabulary (padding
belongs outside the span). Errors are `std::invalid_argument` for bad parameters
or rows (messages match `validate`), `std::out_of_range` for columns or commits
past staged tokens, and `std::logic_error` for a second `stage`.
