# Field report: an adapting engine stopped as a disturbed device (Qwen3.6-35B-A3B UD-IQ4_XS)

This report continues the adaptation report on the same run (`qwen36-35b-a3b-iq4xs-adaptation.md`). It covers
candidates 12 to 17 on crit 0.19 and crit-eval 0.7.0, and a crit-eval steady-state gap that stopped the last three
full Evals.

- **Model:** Qwen3.6-35B-A3B, Unsloth UD-IQ4_XS GGUF, weights as stored.
- **Machine:** RTX 5060 Ti 16 GB on PCIe 3.0 x16, Ryzen 5 3600, 16 GB RAM, Windows 11.
- **Run terms:** `speed` quality mode, 262,144 tokens of context, decode target 74 tok/s.
- **Engine design:** as in the earlier reports. Routed experts sit in device slots or pinned host memory. The
  residency swaps hot host experts onto the device between decode steps, placement only.

## Outcome

| Candidate | Change | Result |
|---|---|---|
| 12 | candidate 11 with `adapt_swaps` 96 | serve `--check`: a 1.374 s prompt step and 0.96 GiB free at 262k |
| 13 | device reserve 1,925 → 2,100 MiB | full context passes; lower-context pre-check fails on agent-long (58.0 → 52.0 tok/s) |
| 14 | `adapt_swaps` 192 | serve `--check` and every Eval gate pass; the benchmark leaves 0.19 GiB of host RAM free against the 2 GiB headroom, so it is never eligible |
| 15 | candidate 14's package measured again, desktop applications closed | Eval fails `lower_context` on tool-call (−7.9 %), which candidate 14 passed |
| 16 | `adapt()` over every layer at once, every 8 decode steps, thresholds 1.5/1 | serve `--check` passes; Eval stops twice as a disturbed device (code 42 %, math 43 %) |
| 17 | the same `adapt()` every 32 steps, thresholds 3/2 | serve `--check` passes (slowest step 0.976 s, 1.28 GiB free); Eval stops as a disturbed device (math 37 %) |

Candidate 15's Eval, the last one that completed: decode 63.0 tok/s, prefill 984 tok/s, TTFT 1,322 ms.

## Findings

### 1. The host RAM headroom decides this machine, not VRAM

Candidate 14 passed every Eval gate, but its benchmark left 0.19 GiB of the 16 GB host free. The engine pins
5.97 GiB of host experts and holds about 1 GB more. The desktop's own applications take about 6 GB. Measured
with them closed, the RAM gate is within reach. Pinning costs no extra RAM on Windows (`cudaHostRegister` over
the experts' own pages). A mapped GGUF released after each read (`VirtualUnlock`) frees RAM but re-reads at
2.9 GiB/s from standby against 11.8 GiB/s from the working set, too slow for the prompt steps.

### 2. Short replies need a residency that settles within Eval's warmups

tool-call's greedy reply is 31 tokens. With `adapt()` every 32 decode steps it adapts once a request, so Eval
measured it mid-climb: the same package passed (candidate 14) and failed (candidate 15) `lower_context`. Running
`adapt()` over every layer at once (one device-to-host copy of the route counts, the per-layer rule by `topk` and
masks) made it cheap enough to run every 8 steps. In process at 65k, that lifts every case about 14 % even at the
old cadence, and with thresholds 1.5/1 tool-call settles at 91 tok/s.

### 3. crit-eval's steady state measures a staged case's cold first request (crit-eval PR crit-labs/crit-eval#16)

In a staged Eval, stage 1 decodes each speed case once. The other cases run, and stage 2 measures the case again,
with stage 1's request as its warmup (protocol `speed.steady_state`). An engine that adapts to what it serves
meets stage 2 as cold as stage 1, because the residency followed the cases in between. The steady-state rule
compares stage 2's first request only with stage 1's. Both are cold, so they agree within 10 %, and the first
request is measured. The next requests run on the adapted placement, and the spread stops the round:

| Eval | case | stage 1 | stage 2 requests (decode tok/s) | outcome |
|---|---|---|---|---|
| candidate 16, attempt 3 | code | 56.0 | 51.3, 84.6, 87.3 | disturbed, 42 % |
| candidate 16, attempt 4 (resumed) | code | 56.0 | 49.5, 82.4, 87.0, 88.1 | settled after 2 more warmups (−12 % is past the tolerance) |
| candidate 16, attempt 4 (resumed) | math | 55.0 | 51.8, 78.3, 85.5 | disturbed, 43 % |
| candidate 17, attempt 3 | code | 57.9 | 50.8, 84.2, 86.8, 87.9 | settled after 2 more warmups |
| candidate 17, attempt 3 | math | 57.1 | 55.4, 81.6, 85.7 | disturbed, 37 % |

Whether a case settles or stops depends on whether its cold stage-2 request lands just inside or outside 10 % of
its cold stage-1 request. Candidates 14 and 15 passed only because math never adapted there (50 tok/s flat). A
resumed Eval does not help: the resumed attempt loads a fresh engine, and the replayed requests do not warm it.

The proposed crit-eval fix keeps every tolerance. A first measured repetition that agreed only with an earlier
stage's warmup is provisional. When the request after it spreads from it by more than the steady-state tolerance,
it stands as one more warmup, and that request is judged against it under the usual bounds. An engine whose stage 2
agrees with its stage 1 measures as before, and unstaged runs are unchanged. The fix changes the protocol, so it
runs under a new protocol id (`crit-eval-v0.7.1`).

## State of the run

Candidate 17 stands at `eval_stopped`. A full Eval under the fix would certify it or show what else limits it. A
run's Eval pins its crit-eval, so measuring candidate 17 under the fix needs a crit-eval release and the run's
consent to it.
