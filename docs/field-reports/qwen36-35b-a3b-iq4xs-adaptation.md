# Field report: full context, residency adaptation and a plan-check gap (Qwen3.6-35B-A3B UD-IQ4_XS)

This report follows the first field report on the same run (`qwen36-35b-a3b-iq4xs.md`, PR #73). It covers
candidates 7 to 11 on crit 0.19.0 and crit-eval 0.7.0, and the plan for candidate 12.

- **Model:** Qwen3.6-35B-A3B, Unsloth UD-IQ4_XS GGUF, weights as stored.
- **Machine:** RTX 5060 Ti 16 GB on PCIe 3.0 x16, Ryzen 5 3600, Windows 11.
- **Run terms:** `speed` quality mode, 262,144 tokens of context, decode target 74 tok/s.
- **Engine design:** GGUF blocks as stored and routed experts split between device slots and host memory. Host experts are computed by an exact AVX2 CPU tier or staged over the link. FP8 KV pages are lent to host experts. Residency adapts to the routes, placement only.

## Outcome

| Candidate | Change | Result |
|---|---|---|
| 7 | candidate 6 ported to crit 0.19.0 | serve `--check`: full-context checks hit the fixed 600 s timeout; steps reach 1.19 s above 160k context |
| 8 | prompt step cost model with a context term (`step_budget_ms` 850, `step_key_ps` 17000) | every prompt step ≤ 0.94 s at 262k; lower-context pre-check fails (chat 68.7 → 61.0 tok/s) |
| 9 | plus a warm phase for residency (96 swaps a call for the first 16 calls) | its own plugin test was wrong (assumed 6 host experts, fixture has 4) |
| 10 | test fixed | lower-context pre-check fails on agent-long (57.0 → 45.9) |
| 11 | warm phase 48 calls | serve `--check` passes in full; Eval passes every gate but `lower_context` (math −8.7 %, tolerance 5 %) |

Candidate 11's Eval (crit-eval 0.7.0, against candidate 6's KPIs):

| KPI | candidate 11 | candidate 6 |
|---|---|---|
| decode | 57.3 tok/s | 61.2 |
| prefill | 1,013 tok/s | 1,070 |
| TTFT | 1,315 ms | 1,317 |
| decode at 32k / 262k | 55.1 / 30.5 tok/s | 58.8 / 32.4 |
| first token at 262k | 1,314 s | 1,012 s |
| concurrency 1/2/4/8 clients | 43 / 63 / 68 / 79 tok/s, every reply equal to its serial one | |

Prefill and the 262k first token are slower because the prompt steps are smaller. That is the price of keeping each step under the 1 s cancel grace at full context.

## Findings

### 1. The full-context check's fixed 600 s (fixed in PR #77)

The 262k prompt takes about 1,300 s on this machine, and Eval measures and accepts it. `serve --check` gave the request 600 s twice: in its client and in crit serve's own request timeout. PR #77 makes the request wait as long as Core's step count moves, and fail only when no step finishes for 600 s. With `--check`, crit serve bounds a request by the attempt's deadline, as Eval does.

### 2. Keeping every prompt step under the cancel grace needs a context term

The step budget priced a prompt step as `base + n · token`. At long context the attention part grows with the keys already cached, so steps reached 1.19 s above 160k context. A per-key term fixes it:

```
cost = base + n · (token + key · (ctx + n / 2))
```

The cap on prompt rows a step is the root of that quadratic. A least-squares fit of `key` was degenerate (n and ctx are collinear along one prompt), so it was tuned on measured steps instead: `step_key_ps` 17000 with an 850 ms budget keeps every step at or under 0.94 s.

**Lesson:** a step-budget model that ignores context cannot hold the cancel grace at full context.

### 3. Residency adaptation must keep up with workload switches

Adaptation runs every 32 decode steps. It swaps the hottest host experts with the coldest device experts of each layer, at most `adapt_swaps` (8) a call. Eval's lower-context run reopens the engine at 65k and runs the six speed cases in order. Each case gets warmups until decode is steady (with a cap on warmups) and then its repetitions. So the residency adapted to chat and code has to move to math within a few requests.

In process at 65k, the same 24 requests in Eval's order (each case's last request, tok/s):

| setting | chat | code | math | tool-call | long-summary | agent-long |
|---|---|---|---|---|---|---|
| candidate 11 (8 swaps, warm phase 48 calls) | 65.8 | 68.1 | 55.7 | 56.3 | 66.2 | 52.0 |
| `adapt_swaps` 32 | 70.4 | 68.6 | 71.2 | 64.9 | 70.0 | 59.9 |
| `adapt_swaps` 96 | 70.8 | 66.5 | 70.5 | 65.2 | 68.3 | 59.0 |
| `adapt_swaps` 32, warm phase 128 calls | 71.0 | 66.7 | 71.0 | 65.3 | 69.1 | 59.4 |

Math's five requests on candidate 11 were 40.9, 45.8, 51.2, 54.5 and 55.7 tok/s. With 96 swaps they were 48.7, 65.7, 69.2, 69.1 and 70.5. In Eval, math was still moving by 43 % between warmups when the warmup cap was reached. That reproduces the gate's −8.7 %.

Adapting more often costs every step, because each call synchronizes and sorts the route scores. With `adapt_every` 16 decode fell by 5 to 10 tok/s in every case, and with 8 it fell further.

**Lessons:**
- Measure an adaptive engine in Eval's own request order. Two requests per case kept math inside the warm phase and hid the problem.
- A warm phase only helps a fresh engine. A steady-state swap limit that matches it covers every later workload switch.

### 4. Gap in crit: the plan check has no way out of a parent plan's misdeclared activations

Candidate 12's plan (`adapt_swaps` 96) fails `plan --check`:

```
candidate-0011 served prompt activations {'Q8_0': 'stored', 'IQ3_S': 'stored', 'IQ4_XS': 'stored', 'Q6_K': 'fp8'},
its plan plan-0007 declared {... 'Q6_K': 'stored'}: build what the plan declares, or record the measurement that
rejected it (gaps entry activation:stored)
```

`gguf-projections` has no int8 (mmq) kernel for Q6_K, so Q6_K's prompt rows always run as FP8. Plan 6 declared this correctly. Plan 7 used the shorthand `"stored"` for every format, which misstated it. The engine did not change.

The only remedy the check accepts is a gaps entry `activation:stored` with answer `rejected` and a crit-eval quality result that measured the stored format outside the noise. No such measurement exists, because the stored format was never served for Q6_K. The rule exists to stop a plan that declares stored activations while its engine serves BF16. Here the engine served FP8, which `speed` mode admits from candidate 1 on.

**Proposed fix:** when the parent's served activations differ from its plan's declaration but put no format at BF16, accept the plan with a gaps entry `activation:stored` whose answer is `cause`, stating what was served and why. Keep the strict rule (a rejection with a quality measurement) when a format runs at BF16.

### 5. Smaller notes

- **Concurrent GPU jobs:** running the crit unittest suite on the GPU during `serve --check` produced a 5.3 s step and 0 GiB free in the full-context probe. Neither reproduced alone. `serve --check` could name other processes holding device memory when a probe fails.
- **Plugin tests need the build:** a plugin's tests run against the source folder fail because its kernels are only built in the package. Running them against the built package works.

## Next

Candidate 12 is candidate 11 with `gguf-experts` `adapt_swaps` 96. Placement only, so every value stays bit-identical. It goes through the same build, serve `--check` and Eval.
