# Token sampling

Core's sampling rules: the draw every sampler makes and the order in which a request's sampling fields apply to a
row of logits. The [runtime](../runtime/README.md) never samples; it passes a request's fields
(`SamplingOptions` in [core/request.h](../../include/crit/core/request.h)) to the engine, whose sampler plugin
applies them ([token-sampling](../../../inference/token-sampling/README.md) on the device).
[sampling.h](../../include/crit/sampling.h) is the draw, header-only: `philox4x32`, `uniform` and `DrawPurpose`,
`constexpr` and `CRIT_HOST_DEVICE`, so a device kernel includes it and reproduces every draw bit for bit. Tests:
[sampling_tests.cpp](../../../tests/core/sampling_tests.cpp).

## Random draws

Every draw is a pure function of `(seed, position, stream, purpose)`, so rollback,
trimming, cancellation and speculative redo need no generator state. Sampling
uses position = committed outputs plus the verify column, the request's stream
(for example the choice index when n > 1) and purpose `DrawPurpose::sample` (0). `draft` (1),
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

A request's rules are each set request field, else the engine's default. A non-empty
request `logit_bias` replaces the default list; the tokens `min_tokens` suppresses
(usually the stop tokens) and the stream come from the engine. Suppressed ids at or
above the valid vocabulary are ignored. `top_k` has no upper bound; one at or above
the selectable count keeps every token. An engine gives a request without a seed a
fresh random one and records it for replay.

A row of exactly the valid vocabulary is processed in IEEE double, in this order,
where n(t) counts token t among committed outputs plus the staged tokens before
verify column c:

1. Load: `v[i] = double(logits[i])`. NaN or +inf is rejected; -inf masks. With a
   [constraint](../constraints/README.md), ids outside the allowed set become -inf.
2. Bias: `v[t] += double(bias)`; a -inf bias bans the token.
3. Suppression: while fewer than `min_tokens` tokens precede the draw, `v[t] = -inf`
   for each suppressed token. When a constraint allows only suppressed tokens,
   suppression is skipped: a
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
12. Draw: `u = uniform(seed, outputs + c, stream)`; the token is the first kept
    token in rank order whose cumulative weight exceeds `u * S_kept`, else the
    last kept token.

The kept set is always a rank prefix. Without top_k, S and S_kept are summed in id
order; with top_k, in rank order. The law of the draw is the kept tokens in rank
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
normalization. A chosen token's log-probability is its raw value even when bias or
penalties chose it. Alternatives are ranked by float log-probability descending,
then id ascending.

## Checks

[sampling_tests.cpp](../../../tests/core/sampling_tests.cpp) checks the draw against the Random123 vectors.
token-sampling's device law is checked against a host implementation of the rules above, the fixtures' own
([host_sampler.h](../../../tests/inference/host_sampler.h)), and its Python mirror by
[test_token_law.py](../../../tests/test_token_law.py).
