# Field report: crit-eval's honesty gate fails every engine on a U+FFFD alternative

**Run:** `run-20261111T081913Z-40ce07fd` (crit 0.17.1, crit-eval 0.6.1, quality mode `speed`)

**Model:** Qwen3.6-35B-A3B, UD-IQ4_XS GGUF

**Machine:** RTX 5060 Ti 16 GB on PCIe 3.0, Ryzen 5 3600, 16 GB RAM, Windows 11

**Required context:** 262,144 tokens

## Summary

None of this run's candidates can qualify. One comparison in crit-eval's honesty check, `quality_checks.run_honesty`,
fails because the checker treats the two lists it compares differently. The engine is not at fault.

The engine's two paths agree:
- the served reply (chat API, greedy, `top_logprobs`)
- the scored reply (`POST /v1/crit/score`)

Every shared value is equal: crit-eval itself reports "served replies scored alike within 0.0e+00". Still, the gate
reports:

> json-extraction: the scored and served top-16 lists differ at 1 tokens

No honest change to the engine removes this failure. 0.6.1 is the newest crit-eval release.

## Cause

1. **`crit_eval.run.chat.ids_by_bytes` maps no id to a token whose bytes are `"�".encode()`.** This is on
   purpose: servers show a token that ends inside a UTF-8 character as U+FFFD, so its bytes are lost.
   - Its docstring: "A token that reads as U+FFFD maps to nothing".
2. **`Chat._reply_rows` therefore leaves such an alternative out of the served top-k list.** Here that list has 15
   entries instead of 16.
3. **`engine.score(...)` returns token ids, not bytes.** The same token therefore stays in the scored top-k list,
   which keeps all 16 entries.
4. **`run_honesty` checks only the full list.** For each position it takes each list in turn:
   - `if len(found) < top_k: continue` skips the short served list.
   - The full scored list is tested.
   - A token found only in the scored list fails the position if `value > edge + tolerance`.

   The dropped token appears only in the scored list. Whenever it ranks more than `tolerance` above that list's edge,
   the position fails.

## Evidence

At reply position 159 of the honesty set `json-extraction`:

| | Value |
| --- | --- |
| Token | 5513, the literal U+FFFD piece, bytes `b'\xef\xbf\xbd'` |
| Its log-probability (both paths) | -23.2626 |
| Edge of the scored top-16 list | -27.1424 |
| Distance above the edge | 3.9 nats |
| Tolerance (`speed`) | 2.0 |
| `ids_by_bytes` entry for its bytes | none |

Reproductions, all under `runs/run-20261111T081913Z-40ce07fd/work/dev/`:

- **Against a running `crit serve`:** `honesty_repro.py`, with its output in `honesty_served.log`:

  ```
  json-extraction: 160 tokens
    position 159: scored list holds 16, the other 15; edge -27.1424
      only in scored: id 5513 value -23.2626 bytes b'\xef\xbf\xbd' named by bytes as None
      only in the other: {}
  ```

- **In process:** `honesty_inproc.py` compares the served and replayed decode on the same engine, with the output in
  `honesty_inproc.log` and `honesty_inproc_adapt.log`.
  - The two paths agree bit for bit (`max |diff| 0`).
  - That holds through 4,561 adaptive expert swaps, with nothing flagged.
- **Tokens crit-eval cannot name by bytes:** `honesty_names.py` lists them.

Every candidate whose greedy json-extraction reply ranks token 5513 at that spot fails the same way. Candidates 4 and
6 both did.

## Impact

- **The gate fails every honest engine.** The quality gate fails in the certified (full) Eval. `check --from-eval`
  then records a failed arithmetic check, and `crit run measure` refuses the candidate. The run cannot propose a
  qualified candidate, whatever the engine does.
- **The engine's own checks pass.** Candidate 6's drift checks pass at 74% of the quality budget. All of its other
  gates pass except generality, a single held-out prompt measured once at a jittered length.
- **Candidate 6's certified numbers:**

  | KPI | Value |
  | --- | --- |
  | Decode (sampled) | 61.2 tok/s |
  | Prefill | 1,070 tok/s |
  | Decode at 32k context | 58.8 tok/s |
  | Decode at 262k context | 32.4 tok/s |

## Suggested fix (crit-eval)

In `run_honesty`, treat the ids that `ids_by_bytes` cannot name the same way on both sides. Either:

- remove them from the scored `top` list before comparing, as `_reply_rows` already does for the served list, or
- leave them out of the "only in one list" edge test.

A regression test can use a scored list holding one unnameable id above the edge, and a served list without it. That
pair must pass.

## Toolkit feedback for crit

- `crit serve --check --data` (crit smoke) flagged this failure before the Eval round. Neither crit's nor crit-eval's
  output said it was a checker artifact.
- Once crit-eval names such a token as unjudgeable, a hint in the precheck's output would save a candidate round.
