# crit smoke

`crit smoke` checks a served engine's quality components in seconds, so a build finds its own bugs before an Eval
round is spent on them. Each test checks one component and names it when it fails. None needs a reference model or
Eval's data: a test either compares two ways the engine computes the same thing, or checks a property any working
model has. So it runs on any engine `crit serve` serves, at any stage of a build.

```sh
crit serve --workspace runs --candidate c3 &      # or crit run ... serve --candidate c3
crit smoke --url http://127.0.0.1:8000
```

```
 passed  rows_valid                 0.09s  517 rows finite, ranked and normalised
 failed  causal                     0.00s  prefix of 17 tokens against the whole prompt of 518: 16 rows, ...
 ...
9 passed, 1 failed, 0 skipped in 1.7s
```

It exits 1 when a test fails. A working engine runs the suite in a minute or two (72 to 90 s on a Flash-Next engine
on an RTX 5090). One still running after `--budget` seconds (default 300) stops with a failed `budget` item naming the
tests it did not run: it answers too slowly, or slows down as it serves, and the rest would only hold the machine.
Each request may take `--timeout` seconds (default 120). `--out FILE` writes the items as JSON (`name`, `state`,
`detail`, `seconds`, the shape of `crit run serve --check`'s report), `--only TEST` runs one test, `--model` picks the
candidate.

## The tests

| Test | What it compares | What it catches |
| --- | --- | --- |
| `rows_valid` | Prefill rows of a prose text: finite, in range, ranked, alternatives summing to at most 1 | NaN or inf, a broken softmax or top-k |
| `fluent` | The prose's perplexity, at most 40 (a working model reads it far lower) | Wrong weights, dequantisation, norms, a wrong tokenizer |
| `causal` | Rows of prefixes of 17 to 513 tokens against the whole prompt's first rows | A mask that lets a position see later tokens; prompt lengths at tile, chunk and bucket edges computing other rows |
| `decode_matches_prefill` | 48 tokens decoded after a prefill (forced to the text) against prefilling all of it | KV cache writes and reads, position offsets, decode kernels that disagree with prefill |
| `causal_long` | The same past 1k tokens, up to `--long`: prefixes just past each power of two and between them (1025, 2049, 3001, then 4097 to 8193 with a larger `--long`) against the whole long prompt, on each prefix's last 64 rows | Sparse or windowed attention, chunked prefill and buckets that only start past a length |
| `decode_long` | 48 tokens decoded after prefills of 1100 and 2100 tokens and the whole long prompt, against prefilling all of it | Decode against long KV caches and the attention paths only long prompts take |
| `greedy_is_top` | A greedy reply's tokens against its rows' top-1, and the reply's rows against scoring it as fixed text | Sampling, draft acceptance, the reply's log-probabilities |
| `repeatable` | The same request twice | Uninitialised or racing buffers |
| `isolated` | A request before and after a long, different request | State leaking between requests: KV, recurrent state, prompt cache |
| `plain_equal` | A speculative engine's reply against its plain decode | Drafts accepted that the target would not have chosen |
| `concurrent` | Requests at the same time against each alone | A request reading another's state |
| `recall` | A random word sequence written twice, at 1024 tokens and at `--long` (default 4096): the repeat must read at under 1 nat a token, in prefill and in decode | Attention, rope and the KV cache at long positions |

A test that needs a capability the package does not declare (`chat-score-v1`, `chat-logprobs-v1`,
`chat-tokenize-v1`, `chat-plain-v1`, `chat-concurrent-v1`) is skipped and says which.

## Tolerances

A row's delta is the largest change of the text token's log-probability or of a likely alternative's. A
comparison fails on a mean delta above its limit, on too many rows moving by over its large delta, or on too many
rows choosing another top-1 outside a near-tie:

| Comparison | Profile | Mean | Large rows | Other top-1 | Near-tie |
| --- | --- | --- | --- | --- | --- |
| Prefill against prefill, a reply against its score (`causal`, `causal_long`, `greedy_is_top`) | loose | 0.15 nats | 2% (plus one) over 1 nat | 5% (plus one) | 0.25 |
| Decode against prefill (`decode_matches_prefill`, `decode_long`) | loose | 0.5 nats | 6% (plus one) over 1 nat | 6% (plus one) | 0.25 |
| Every comparison | strict | 0.02 nats | none over 0.25 | none | 0.02 |

A prompt's rows never depend on how long the prompt is, so the prefill comparisons stay tight on every engine. An
engine whose prompt GEMMs pick their kernel by row count (cuBLAS's heuristics, a padded bucket) fails them even when
every kernel is right: NVFP4 activations (W4A4) carry one bf16 ulp through every layer, and on a Qwen3.8-27B NVFP4
engine (RTX 5090) one ulp in one small GEMM moved a 17-token prompt's rows by 0.25 nats on average. Arithmetic that
changes with the row count can fail crit Eval's honesty gate, and a limit above that floor would also pass a bug that
moves a few rows, so pin the prompt kernels (one algorithm for every row count, or fixed row slabs) and the rows
match exactly. Decoding may run other kernels than prefilling (GEMV against GEMM), so it can never match exactly: the
loose decode limits sit above the 0.12 to 0.43 nats that the same engine's decode measured. `--strict` fits an engine
that keeps one arithmetic on every path, `preserve`. Greedy replies may part only at a near-tie (strict: not at all).
A stale cache, a position offset or a mask a decode step gets wrong moves rows by nats. A test that fails on an
engine known to be right means a limit needs to move, so report it.

## What it does not check

Quality drift that stays inside the run's quality budget, speed, device memory, behaviour past thresholds beyond
`--long` (a context bucket at 100k tokens), and an engine opened at a lower context than it was built for (crit
serve opens it at its built context). Eval measures those; `crit run ... serve --check --data` runs
Eval's gate pre-check in minutes.

## Tests of the suite

`tests/test_smoke.py` runs the suite against a toy engine behind a fake crit serve: the working toy passes every
test, and each planted bug (a leaking mask, a decode position offset, state kept between requests, positions past
1500 forgetting the context, a wrong draft acceptance, flattened logits, a mask fault only past 2000 tokens) fails the
test written for it, and the fault past 2000 tokens passes the short tests.
