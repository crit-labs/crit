# crit smoke

`crit smoke` checks a served engine's quality components in seconds, so a build finds its own bugs before an Eval
round is spent on them. Each test checks one component and names it when it fails. None needs a reference model or
Eval's data: a test either compares two ways the engine computes the same thing, or checks a property any working
model has. So it runs on any engine `crit serve` serves, at any stage of a build. Given the checkpoint's own
reference, which recon records once per model from crit Eval's, `fluent` is judged against it
([below](#against-the-checkpoints-reference)).

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
candidate, `--reference RECON_JSON` judges `fluent` against a run's recorded reference (`crit run serve --check` and
`crit run ablate` pass the run's own).

## The tests

| Test | What it compares | What it catches |
| --- | --- | --- |
| `rows_valid` | Prefill rows of a prose text: finite, in range, ranked, alternatives summing to at most 1 | NaN or inf, a broken softmax or top-k |
| `fluent` | With the checkpoint's reference: its texts' answer tokens against the reference's, by the perplexity ratio and the drift per target (median and p90); without one, the prose's perplexity, at most 40 (a working model reads it far lower) | Wrong weights, dequantisation, norms, a wrong tokenizer |
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

An engine that states the [exactness](../docs/PLUGINS.md#exactness) it asks of its rows (its card's
`engine_facts.exactness.asked`, from `configuration.exactness`) is judged by it where the level promises bits,
whatever the profile. Prompt rows asked `row-exact` or higher never change with the prompt's length, so `causal` and
`causal_long` are judged strict; prompt rows asked `exact` (or `declared`), beside decode rows asked `row-exact` or
higher, have the bits of the same token decoded, so `decode_matches_prefill` and `decode_long` are judged strict too.
A lower level, or an engine that states none, is judged by the profile: the drift a path may round by. A failed item
names the level that made it strict.

## Against the checkpoint's reference

An absolute perplexity judges the checkpoint as much as the engine: a quantized checkpoint whose own reference reads
the prose at 74 fails every engine at 40, and leaving smoke out then hides real regressions too. So `fluent` is
relative to the checkpoint where the run has its reference. crit Eval's reference (`crit eval reference`, the
checkpoint teacher-forced in FP32, once per model) scores the quality texts of the run's prepared data, and
`crit run --id RUN recon --data DATA` copies the first four conversations of at most 1,024 tokens (their ids) and the
reference's log-probability of each written-answer token into `work/recon.json` (`gathered.smoke_reference`, with the
reference's perplexity of them and the data it came from; on an existing recon.json it records only that, and
`--force` keeps it for an unchanged checkpoint). `fluent` then prefills each text once and judges the answer tokens:

| Profile | Perplexity over the reference's | Drift per target, median | Drift per target, p90 |
| --- | --- | --- | --- |
| loose | 1.5 | 0.5 nats | 2 nats |
| strict | 1.1 | 0.1 nats | 0.5 nats |

A target's drift is how far the engine's log-probability of the answer token lies from the reference's; the item
reports its median, p90 and maximum beside both perplexities. The limits sit above Eval's own (a perplexity ratio of
1.4 in speed, 1.05 in preserve): smoke catches a broken component, a wrong weight or scale moving the median by nats,
and leaves the drift budget to Eval. Without a reference (a run before its data holds the reference's scores, `crit
smoke --url` without `--reference`) `fluent` falls back to the absolute limit, 40 on smoke's prose, and says so.
`recall`'s limits stay absolute: copying a repeated random text is a property of any working model, and no
reference scores its text.

## Rounding or fault

A failed `decode_matches_prefill` or `decode_long` says how far decoding moved from prefilling, not why: decode runs
other arithmetic than prefill (GEMV against GEMM), so a gap can be rounding as well as a fault in what the two paths
share. On an engine whose facts say `decode_via_prompt` (its model plugin honours the hook, as
[model-template](../models/core/model-template/README.md) does), the test decodes the failed comparison's tokens once more
with the score request's debug option `decode_via_prompt` (`POST /v1/crit/score`, [serving](../docs/SERVING.md)): the
same prefill, KV cache and positions, each decode row computed as a one-token prompt chunk, the prompt rows'
arithmetic. The failed item then names a verdict:

- `rounding`: the rows match prefill within the prompt comparisons' limits (the drift another chunking may round by),
  so the gap is the decode arithmetic's. Pin the decode and prompt kernels to one arithmetic, or ask a higher
  [exactness](../docs/PLUGINS.md#exactness), if the gap matters; a limit that fails an engine known to be right should
  move.
- `fault`: the rows still move, so the fault is in what decode shares with the prompt path: KV cache writes or reads,
  positions, masking.

An engine without the hook says so in the item, which stays a failure without a verdict. The attention plugin picks
its reader from the rows it is given, so a one-token prompt chunk may still read the KV as a decode row does; at
`exact` the prompt rows already read through the decode reader. `tests/test_smoke.py` plants both: a decode rounding
of a W4A4 engine's size, failed strictly, comes back `rounding`, and a decode reading the position before comes back
`fault`.

## What it does not check

Quality drift that stays inside the run's quality budget, speed, device memory, behaviour past thresholds beyond
`--long` (a context bucket at 100k tokens), and an engine opened at a lower context than it was built for (crit
serve opens it at its built context). Eval measures those; `crit run ... serve --check --data` runs
Eval's gate pre-check in minutes.

## Tests of the suite

`tests/test_smoke.py` runs the suite against a toy engine behind a fake crit serve: the working toy passes every
test, and each planted bug (a leaking mask, a decode position offset, state kept between requests, positions past
1500 forgetting the context, a wrong draft acceptance, flattened logits, a mask fault only past 2000 tokens) fails the
test written for it, and the fault past 2000 tokens passes the short tests. A noisy checkpoint (its reference reads
the prose at a perplexity of 80) fails `fluent` without its reference and passes against it with rounding, while
flattened logits, a weaker model and rounding of a nat and a half still fail.
