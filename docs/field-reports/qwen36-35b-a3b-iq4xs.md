# Field report: Qwen3.6-35B-A3B UD-IQ4_XS at 262k context on a 16 GB GPU

This report covers run `run-20261111T081913Z-40ce07fd`. It started under crit 0.17.1 with crit-eval 0.6.1 and was
carried over to crit 0.19 with crit-eval 0.7.0 midway. It records what the run reached, what blocked it, and what
crit should change. Each gap is ranked and comes with a proposed fix.

## Setup

- **Model:** Qwen3.6-35B-A3B (hybrid Gated DeltaNet and full attention, 256 experts, 8 active), Unsloth UD-IQ4_XS GGUF,
  weights as stored.
- **Machine:**
  - GPU: RTX 5060 Ti 16 GB on PCIe 3.0 x16
  - CPU and RAM: Ryzen 5 3600, 16 GB RAM
  - OS: Windows 11
- **Requirements:** 262,144 tokens of context, a decode target of 74 tok/s, quality mode `speed`, 1 GiB device and 2 GiB
  host headroom.
- **Residency:** the weights do not fit in 16 GB. Hot experts live on the device and cold experts on the host. The host
  experts run on the CPU tier (AVX2) or are staged over the link.

## Outcome

| KPI (crit-eval 0.6.1, through `crit serve`) | candidate-0003 | candidate-0005 | **candidate-0006** |
| --- | --- | --- | --- |
| Decode, sampled (tok/s) | 63.1 | 57.2 | **61.2** |
| Prefill (tok/s) | 818 | 922 | **1,070** |
| TTFT, geometric mean (ms) | 1,415 | 1,390 | **1,317** |
| Decode at 32k (tok/s) | 54.4 | 32.7 | **58.8** |
| Decode at 262k (tok/s) | – | – | **32.4** |
| First token at 262k (s) | – | – | **~1,012** |
| Quality drift (share of budget) | 108.8% (fail) | 74.3% | **74.3%** |

- candidate-0006 is the best engine of the run, 83% of the decode target. It did not qualify under crit-eval 0.6.1
  because of the honesty asymmetry (gap 8), which crit-eval 0.7.0 has since fixed.
- candidate-0007 is candidate-0006 ported to crit 0.19's plugin manifests, with no change to the engine's values.
  It builds, and every plugin test and standard serving check passes. Its `serve --check` fails on the new
  full-context check (gap 1). It is not sealed and has not been evaluated under crit-eval 0.7.0.

## What moved the numbers

Every change that touches values was checked bit for bit against the build before it, with equal log-probabilities.

1. **FP8 KV.** This repaired the long-32768 drift (KLD p99 from 4.35 to 3.0) and halved the KV footprint at 262k.
2. **Prompt steps of up to 768 tokens, capped by a step cost model.** Each step targets 0.9 s, because a cancel settles
   only between steps.
3. **Wide-tile prompt attention.** 64 query rows a block gave 1.5x at every length and is row-invariant.
4. **Re-lending.** KV pages go back on loan to host experts after a long sequence ends.
5. **Reclaim fix.** Only the pages a sequence needs are reclaimed. Before the fix, the first page past the lending
   floor sent all 1,571 lent experts home, which cost 40% of decode at 32k.
6. **FP8 decode staging** (8 codes a load), **cross-layer prefetch** (the next layer's 24 hottest host experts copied
   during each prompt layer) and **Q6_K down staging**.

## Where the time goes

**Decode, 14.6 ms a chat step:**

| Term | Time | Floor |
| --- | --- | --- |
| Dense Q8_0 | 4.4 ms | 3.8 ms |
| Idle (host step between graph replays) | 2.6 ms | — |
| CPU-tier wait | 2.0 ms | — |
| Device experts | 1.6 ms | 1.1 ms |
| Q6_K head | 1.2 ms | 1.1 ms |

**Prefill, 687–716 ms a 768-token step:**

| Term | Time |
| --- | --- |
| Staging copies at the link's 6.8 GB/s | ~270–300 ms |
| int8 tile products | 129 ms |
| Q6_K and FP32 GEMVs | 98 ms |
| Idle (per-layer route-count read-back) | 84 ms |
| Gated DeltaNet scan | 60 ms |

The next levers are:
- multi-step native rounds, to remove the decode idle time
- splitting host routes between the CPU and the link
- a chunked delta-rule scan
- removing the per-layer host sync in prefill

## Gaps in crit, ranked

### 1. The full-context serve check has a fixed 600 s client timeout (blocker)

`full_context_checks` (`cli/crit/run/checks.py`) sends one request of about 262,092 tokens. Its client uses
`served.timeout`, which is fixed at 600 s (`cli/crit/run/served.py`, `Served.__init__`). On this machine the
engine needs about 1,012 s for that prompt, a time that Eval's long-context curve itself measured and accepted. So the
check cannot pass however correct the engine is. It then cascades:

- The timeout closes the connection, and `crit serve` treats that as a cancel.
- The cancel lands mid-step (gap 2). The host is retired: "the engine settled no cancel within 1 s".
- `full_context_device_memory` fails with the same `TimeoutError`, although it has nothing to do with time.
- Every 262k smoke test that follows fails against the retired host (gap 3).

The requirement (262k context) and the hardware (a 16 GB GPU with offloaded experts) are both legitimate, and Eval
serves the same request without complaint. The check enforces a speed floor that no other part of crit states.

**Proposed fix:**
- Derive the timeout from the prompt rather than a constant. For example, use the engine's prefill rate measured on
  the one-copy and two-copy requests the check already sends, times a generous factor, bounded by the attempt deadline.
- Alternatively, take a per-run limit from the requirements (for example `max_first_token_seconds`), so the user states
  the floor.
- Report a timeout as `unverified` with the elapsed time, not as `failed` for both items.
- Never let the check's own timeout trigger a cancel that retires the host.

### 2. The 1 s cancel grace is tight for long-context prefill on small GPUs

`CANCEL_GRACE_SECONDS = 1` (`api/crit_api/residency.py`). candidate-0006 sizes its prompt steps with a cost model to
0.9 s, but at long context they still take 0.85–1.3 s: attention grows with the prefix, and host-expert staging varies
with routing. Shrinking the steps further costs prefill throughput on every request.

**Proposed fix:**
- Make the grace a function of the engine's declared or measured step time (for example 2x the p99 prompt step, floored
  at 1 s), or a field in the engine spec that `crit serve` honours up to a cap.
- At the least, document the 1 s budget in SERVING.md and BUILDING_ENGINES.md. The Gemma 4 report raised this too.

### 3. Smoke keeps running against a host that is already retired

After the host was retired, smoke ran `fluent`, `causal` and `decode_matches_prefill` at 120 s each, then failed its
300 s budget after 363 s. Every item failed for the same reason.

**Proposed fix:** when the first item fails with "Resident host exited", stop smoke for that context and report the
remaining items as `not run: host retired`, with the retire reason.

### 4. Moving to 0.19 breaks existing packages without a migration path

- 0.19 refuses the old kernel declarations (`"name": "crit-labs/..."` with version and revision) and accepts only
  `{name, fallback}`. Every package built with 0.17/0.18 base plugins stops loading.
- `crit plugins add --update` fails on the old manifest, because it validates the whole folder first.
- Removing the plugins and adding them again fails for the same reason.
- The only way through was to copy `inference/base/*` and `inference/*/code` by hand and fix the relative
  `../../_sources/...` references.

**Proposed fix:**
- Give `crit plugins add --update` (or a new `crit plugins migrate`) a mode that rewrites old kernel declarations.
- Let it replace vendored base plugins without validating the old ones first.
- List the manifest change under "Breaking" in the CHANGELOG, with the migration command.

### 5. The Core step library is a new hard dependency that only shows up at serve time

0.19 needs `crit_core_steps`, built with `crit core build`, which needs CMake 3.28+, a C++20 compiler and git. The
first `serve --check` failed only after the build, plugin tests and package step had all run. On Windows the build
needs a vcvars64 and CUDA environment, which `crit core build` does not set up.

**Proposed fix:**
- Check for the library in `crit run next` and before `crit run candidate`, and say what to run.
- On Windows, find the VS Build Tools through vswhere and load vcvars64 for the build.

### 6. Resuming after a crit upgrade forces sealing every candidate

- `crit run resume` first required `--external-agent-stopped`, then refused until every candidate was sealed.
- So candidate-0002 to candidate-0005 had to be sealed `check_failed` before work could continue. That is correct for
  their state, but it is irreversible and was forced by the upgrade, not by the run.
- candidate-0006 could not be fixed from its own plan, plan-0006, once its package stopped loading. `crit run next`
  said to seal it and fix it from the same plan, but `crit run candidate` then refused plan-0006 because candidate-0006
  is on the ladder. A new plan (plan-0007) with an empty "port" lever was the only way through.
- `crit run next` also asked to seal candidate-0005, which was already superseded and whose records date from 0.17.1.
  This looks like a misreading of older records.

**Proposed fix:**
- Make `next` and `candidate` agree on whether a candidate can be fixed from its plan.
- Add a lever kind for upgrades ("port to crit X.Y") that needs no invented requirement.
- Let `resume` carry unsealed candidates over and mark them stale, instead of forcing a seal.

### 7. Eval data prepared for one crit-eval version cannot be used with the next

The pre-check refused `work/eval-data` with "prepared for crit-eval-v0.6.0, not crit-eval-v0.7.0: prepare it again
(crit eval prepare) into a new data directory". The message is clear, but the refusal only appears at the end of a
`serve --check` of more than 30 minutes.

**Proposed fix:**
- Check the data version in `crit run next` and when `crit run serve --check` starts.
- Alternatively, have `crit eval prepare` upgrade a directory in place when the source corpus is unchanged.

### 8. Resolved in crit-eval 0.7.0: honesty against unnameable tokens

Under 0.6.1, candidate-0006 failed honesty only because token 5513 (the literal U+FFFD piece) was among the top-16
alternatives at one position. `ids_by_bytes` could not name it, so it was missing from the served list but kept in the
scored list. 0.7.0's `as_served()` drops such tokens from both lists. This is listed here so the run's record is
complete. The fix should be confirmed with a re-evaluation once gap 1 is resolved.

### 9. `plan --check` refuses a lever that only ports

plan-0007's only lever is a port with no change to values. `plan --check` refused `serves: check:functional`, because
candidate-0006's newest functional check had passed, although the package no longer loads under the new crit. The
lever had to claim `check:arithmetic` instead. This is the same root cause as gap 6: there is no honest way to say "this
candidate must be rebuilt for the new toolkit".

## Not resolved in this run

- The decode target of 74 tok/s: candidate-0006 is at 61.2.
- candidate-0007's qualification under crit-eval 0.7.0, which needs gap 1 resolved and the eval data prepared again.
- The lower-context pre-check showed slower decode at 65k after four reloads in one host. Eval's lower_context gate
  passes, and this was not reproduced in process.

## Paths

- Run: `runs/run-20261111T081913Z-40ce07fd`, with its summary in `summary.md`.
- Best engine: candidate-0006, source in `work/candidate-0006-source`.
- Port to 0.19: candidate-0007, source in `work/candidate-0007-source`. The failing check is in
  `work/dev/serve-check-c7b.log`.
