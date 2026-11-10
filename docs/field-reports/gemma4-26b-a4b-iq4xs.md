# Field report: Gemma 4 26B-A4B (UD-IQ4_XS GGUF) on an RTX 5060 Ti 16 GB

This report covers one crit run end to end: what was built, what was learned, where crit fell short, and what crit
should have provided so the next engine of this kind costs less. Everything below comes from run
`run-20261109T095346Z-1b210ccf` (2026-11-09 to 2026-11-10): its plans, proposals, stops, `work/diagnosis.md`,
`work/profile-candidate-0008.md` and `work/yardstick-decode.md`. The run folder itself is not published.

## The workload

| | |
| --- | --- |
| Model | `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`. 30 layers: 25 sliding-window and 5 full-attention. 128 routed experts, 8 active per token. The gate/up experts are IQ3_S, the down experts IQ4_NL, attention and dense MLP Q8_0, and the tied head is 262,144 × 2,816. |
| Hardware | RTX 5060 Ti 16 GB (about 387 GB/s probed), Windows 11, torch 2.14 cu130, Python 3.14 |
| Quality mode | `speed` |
| Context | 200,000 tokens, with 1 GiB of device headroom required |
| Target | The user's estimate was 82 tok/s decode. Recon's plain-pass ceiling is 118.5 tok/s (3.05 GiB read per token). |

## Outcome

`candidate-0014` is sealed `built`: `crit plugins test` passes 11/11 and `serve --check --no-smoke` passes. It is
**not certified**: no candidate of this model can be measured by Eval (see gap 1).

Decode speed by candidate, measured through `crit serve`:

| Candidate | Decode tok/s | What it added |
| --- | --- | --- |
| 0003 | 4.3–4.8 | Proof of concept: own GGUF GEMVs, torch glue, about 9,500 kernel launches a token |
| 0008 | 22–25 | Decode in CUDA graphs, own split-KV decode attention, a device staging buffer for experts |
| 0009 | 64.5 | Every expert resident on the device (4-bit KV and a weight arena) |
| 0010 | 85–88 | int8 dp4a GEMVs over the stored blocks, fused norms, a two-kernel router, adaptive attention splits |
| 0011 | 89–91 | int8 tensor-core prompt kernel (MMQ) and fused prompt attention. Prefill went from about 225 to about 2,270 tok/s. |
| 0012 | 90–91.5 | Tensor-core split-KV decode attention: 86 tok/s after 2.4k tokens instead of 81 |
| 0013 | 97–100 | Routed experts on a side stream beside the dense MLP |
| **0014** | **99–102** | Wider MMQ tiles and BF16 expert outputs on prompt steps. Prefill reaches about 2,700 tok/s and TTFT is 0.19 s on 512 tokens. |

At long context, decode is 94 tok/s after a 6,820-token prompt (served) and about 87 tok/s at 32k positions (in
process).

## Learnings

### Where the time went, in the order it mattered

1. **Launch overhead first, then residency, then bandwidth.**
   - In candidate-0003 the GPU was busy only 0.16 s of a 0.23 s token. About 9,500 launches at about 15 µs each
     dominated. Capturing the decode step in CUDA graphs per row count gave about 5×.
   - Moving the last five host-resident expert layers onto the device gave another 2.6×. Those layers had cost about
     16–18 ms a token over a 6.5–7 GB/s link. Making room took 4-bit KV and a single weight arena.
   - Only after that did kernel bandwidth matter.
2. **Decode weights should be read in their stored format.**
   - Use int8 activations in the formats' own 32-element units with an FP32 scale, multiplied with `dp4a` straight
     from the GGUF blocks.
   - IQ3_S, IQ4_NL, IQ4_XS and Q8_0 all reached 330–418 GB/s cold, against a 387 GB/s probe (an L2 effect pushes
     some above it).
   - Earlier dequantize-then-multiply paths reached 40% of bandwidth on the IQ formats.
3. **Prompts need tensor cores over the blocks too.**
   - Dequantizing the whole weight to BF16 or FP8 for every prompt chunk capped prefill at about 225 tok/s.
   - An `mma.sync m16n8k32` s8 kernel that decodes blocks into shared memory and sorts expert routes on the device
     (no host sync) reached about 2,700 tok/s.
   - The kernel still runs at only about 10–15 T int8 MAC/s, a small share of peak.
4. **Overlap is cheap once everything sits in graphs.**
   - Running the routed experts on a side stream beside the dense MLP, forked and joined inside the captured graph,
     gave about 8%.
   - It has to stay off for prompt steps: two prompt-sized temporaries alive at once broke the 1 GiB headroom.
5. **Long-context decode needs its own attention kernel.**
   - The derived-key full layers (keys are `rope(k_norm.weight * value)`, see the Gemma 4 notes below) got a BF16
     `mma` split-KV kernel. It stages raw INT4 rows, prefetches the next step in registers, uses a rope recurrence
     instead of a sincos per large angle, and runs the 8 query heads of a group as one 16-row tile.
   - A full layer at 32k positions went from 650 to 193 µs.
6. **Measure in the step, not in isolation.**
   - An aligned Q8_0 repack was 3–11% faster in microbenchmarks but slower in the real step: 100.5 against
     103.2 tok/s, plausibly from cache and arena effects. It was pruned.
   - A 256-column MMQ tile and larger Q8_0 stages were also slower in the step.

### Correctness learnings

- **Check a candidate against an exact FP32 reference, not against smoke alone.**
  - The useful yardstick was an independent FP32 forward pass: transformers' Gemma 4 math in plain torch, with the
    GGUF weights decoded exactly, scored per target over 200 tokens.
  - candidate-0014 reaches mean 2.28, median 0.60 and p90 6.3 nats against it.
  - Single-text perplexity on this checkpoint swings too much to rank candidates.
- **This checkpoint amplifies rounding.**
  - The FP32 reference itself gives some tokens −42 nats, so rounding that flips a router's top 8 or a near-tied top
    token moves single targets by tens of nats.
  - llama.cpp scores this GGUF at perplexity 74 on comparable prose, above smoke's limit of 40.
- **Decode-vs-prefill drift was proven to be rounding, not a bug.**
  - The two kernel families round differently: dp4a GEMV with FP32 accumulation on one side, tiled MMA on the other.
  - Forcing decode rows through the prompt arithmetic made decode equal prefill bit for bit (delta 0.0000 over 48
    rows).
  - That experiment is the cheapest way to separate a KV or position fault from rounding, and crit should offer it
    (gap 11).
- **A capacity bug hid in plain sight.**
  - The torch prompt attention decoded a sequence's whole KV span to FP32: about 1.6 GB for one full layer at
    200,000 tokens.
  - Short tests never hit it, and the capacity check did not catch it before a long prompt would have. A fused
    kernel that expands 4,096 positions at a time keeps prompt memory flat.
- **Making prefixes self-consistent fixed smoke's causal checks.**
  - With the fused prompt attention, a 17-token prefix scores the same rows as the whole prompt, bit for bit.
  - causal, causal_long and recall then passed, where they had failed through candidate-0010.

### Gemma 4 specifics an engine author needs

- Full-attention layers have their own geometry: `num_global_key_value_heads` (2) and `global_head_dim` (512),
  against the sliding layers' `num_key_value_heads` and `head_dim`.
- `attention_k_eq_v`: keys and values are one projection.
  - Store one row (the values, after the weightless norm) and derive each key on read.
  - This gives 5,120 B per token for the 5 full layers: 0.95 GiB at 200,000 tokens in FP8, and half that in INT4.
- The experts-per-token count is spelled `top_k_experts`, not `num_experts_per_tok`.
- In the GGUF, `attention.head_count_kv` is a per-layer list, and `key_length_swa` gives the sliding head size.
- Unsloth's UD-IQ4_XS mixes four block formats, so an engine needs kernels for Q8_0, IQ4_NL, IQ4_XS and IQ3_S.
- The IQ4 lookup constant packs to `0xBFAD9881u`. A one-nibble typo here produced plausible but wrong output.
- The MTP companion drafter (Q8_0, 0.46 GB) does not fit beside every expert at 200,000 tokens with 1 GiB of
  headroom.

### CUDA and Windows notes

- Static shared memory is capped at 48 KB. Kernels needing more must opt in dynamically, or use smaller K tiles
  (the decode attention went to DTK=16).
- For `__constant__` tables read with divergent indices (the IQ3_S grid), `__device__ const` was faster.
- FlashInfer has no wheel for torch 2.14 / cu130 / Python 3.14 on Windows, so every attention kernel had to be
  written here.

## Gaps: what crit should have provided

Each gap lists what happened, its cost to this run, and a proposed fix. They are ordered by impact.

### 1. crit-eval cannot read a Gemma 4 GGUF (blocker)

- **What happened:** crit-eval 0.6.1 builds no reference for a gemma4 GGUF, because
  `crit_eval.architectures.gguf_layout` has no gemma4 layout rules. So no Eval could measure, qualify or certify any
  candidate, and every segment stopped `blocked`.
- **Cost:** no certified result for 14 candidates. Speed numbers come from serve telemetry, not the common suite.
  Quality was judged against a hand-written FP32 reference.
- **Proposed fix:** a gemma4 layout in `gguf_layout` / `gguf_config`, covering per-layer KV heads, the global head
  dim, `attention_k_eq_v`, `top_k_experts` and the IQ3_S / IQ4_NL / IQ4_XS / Q8_0 mix. Preflight should also report
  "Eval cannot score this model" at run create, before any candidate is built.

### 2. The ladder has no path when Eval is unreachable

- **What happened:** the ladder rule puts candidate 11 onward on the full-Eval rung. This run could not reach that
  rung, so plans 0006–0008 had to name candidate-0010 as their parent and re-list every lever built since.
- **Cost:** lineage in the records no longer matches the real history (0014 actually descends from 0013). Each plan
  repeats the previous plans' levers.
- **Proposed fix:** a recorded "rung unreachable: <reason>" state. A blocked run could then continue building on its
  latest built candidate, with lineage intact, without pretending to climb.

### 3. Core's config and GGUF readers lacked per-layer-kind attention geometry

- **What happened:** Core assumed one KV-head count and one head dim for every layer, and two cached tensors (K and
  V). Gemma 4 breaks all three. The KV size, the capacity check and `check_gguf_folder` were all wrong until patched.
- **Cost:** the patches below live, uncommitted, in this checkout:
  - `config_keys.json`: `experts_per_token`, `full_attention_kv_heads`, `full_attention_head_dim`,
    `full_attention_keys_only`.
  - `config.py`: `attention_layers()`; `attention_shape` and `kv_elements_per_token` honour them.
  - `gguf.py`: compares per-layer `head_count_kv` lists and `key_length_swa`.
- **Proposed fix:** land these patches, with tests, and mirror the new keys in the C++ reader (it currently reads
  them in Python only).

### 4. crit serve could lose the engine on a client disconnect

- **What happened:** if a stream write raised `OSError` because the client had gone away, the exception unwound
  through the engine call and retired the host. A fast engine whose next piece beat the disconnect watcher lost its
  resident weights, and reloading them takes minutes.
  - Retirement and the 1-second cancel timeout also printed nothing, so they looked like silent crashes.
- **Cost:** hours of diagnosis in segment 2.
- **Proposed fix:** land the uncommitted patches:
  - `server.py`: a failed write cancels the turn instead of retiring the host.
  - `engine.py`, `residency.py`: one stderr line each when a host is retired or closed.

### 5. The 1-second cancel budget is a hidden constraint

- **What happened:** `crit serve` closes a host that settles no cancel within 1 s. So every prompt chunk must finish
  in under 1 s, at 200,000 tokens of context.
- **Cost:** found by failure, not in documentation. It also shaped the prefill chunk size (512).
- **Proposed fix:** document it in SERVING.md and AGENTS.md, and have `serve --check` time the slowest prompt chunk
  at full context against the budget.

### 6. No GGUF block-format kernels in the library

- **What happened:** no library plugin multiplies by GGUF block formats. This engine wrote its own:
  - dp4a GEMVs and gathered GEMVs
  - an int8 tensor-core MMQ with device-sorted expert segments
  - dequantize kernels
  - activation quantization in the formats' units
  - all checked bit for bit against gguf-py.
- **Cost:** most of the run. Every future GGUF engine will rewrite the same code.
- **Proposed fix:** a `gguf-projections`-style kernels plugin in the library, with Q8_0, Q4_K, Q6_K, IQ4_NL, IQ4_XS
  and IQ3_S at least. The plugin from this run is a ready starting point, with tests.

### 7. Library attention, graphs and expert caches only fit the library's own layouts

- **Round driver:** `crit_plugins.rounds.Rounds` captures graphs only if every slot verify calls has `rounds:<slot>`
  and the KV pool has `bind` / `unbind` / `commit_round`. The engine's ring and derived-key pools could not join, so it
  captured graphs itself with `torch.cuda.CUDAGraph`.
- **Decode attention:** `paged-kv-attention` reads only attention-state's page layout. It has no sliding rings and no
  derived keys.
- **Expert cache:** `device-expert-cache` holds grouped-experts bundles, not GGUF IQ rows.
- **Router:** the router slot must be a router plugin. A fused routing kernel meant copying `topk-router`'s `topk.py`
  into a new plugin, because a library plugin cannot be extended.
- **Proposed fix:** layout-agnostic interfaces. A graph-capture helper should not need draft/verify; attention should
  take a key-derivation hook and ring geometry; the expert cache should hold opaque byte rows; library plugins should
  be subclassable or composable.

### 8. Smoke's limits are absolute, not relative to the checkpoint

- **What happened:**
  - `fluent` fails any engine for this checkpoint: smoke's limit is perplexity 40, and llama.cpp itself scores 74.
  - `decode_matches_prefill` and `decode_long` fail on rounding that this checkpoint amplifies.
  - So `serve --check` had to run with `--no-smoke`, which also hides real regressions.
- **Proposed fix:** set smoke limits relative to a reference measured once per model (llama.cpp perplexity, or the
  FP32 reference of gap 9) and record that reference in recon. Report drift per target (median and p90), not only a
  maximum.

### 9. No exact reference forward pass for quantized checkpoints

- **What happened:** to tell engine faults from checkpoint behaviour, the run wrote its own FP32 reference: GGUF
  weights decoded exactly, then transformers' Gemma 4 math in torch.
- **Proposed fix:** a `crit ref` (or crit-eval) command that writes per-target log-probs for a text from the exact
  dequantized checkpoint, plus a `compare` that reports mean, median, p90 and max nats for a candidate. This run's
  most reliable quality signal was exactly this comparison.

### 10. Model tests skip silently without `CRIT_CORE_STEP_LIBRARY`

- **What happened:** the gemma4-model tests that compare against transformers were skipped, not failed, when
  `CRIT_CORE_STEP_LIBRARY` was unset. For several candidates, "all tests pass" did not include them. `crit serve`
  also needs the variable, and nothing discovers the built DLL.
- **Proposed fix:**
  - `crit plugins test` should print skipped tests with their reasons, and fail when a model test is skipped.
  - Core should locate its built step library from the workspace build folder, with the environment variable as an
    override.

### 11. No built-in way to tell rounding drift from a real fault

- **What happened:** the decisive experiment ran decode rows through the prompt arithmetic and showed bit-for-bit
  equality. It had to be wired by hand.
- **Proposed fix:** a documented engine hook (for example `debug.decode_via_prompt`) that smoke can switch on when
  `decode_matches_prefill` fails. If the delta goes to zero, smoke reports "rounding" instead of "fault".

### 12. Run-record friction

These are individually small, but each cost a retry:

- **Linked plans:** a plan already linked to a candidate cannot be reused. The error does not say to record a new
  plan.
- **Lever `source`:** it must be `"estimate"` for unmeasured changes. The error does not list the allowed values.
- **`lever_fates` fields:** entries take `{fate, detail}`. Using `reason` fails with "unknown field" and no hint.
- **Plans recorded after measurement:** plan-0004 and plan-0006 were recorded after their fixes were measured in the
  work tree. There is no way to mark a plan as retroactive, so the summary had to say it in prose.
- **Proposed fix:** errors that name the allowed values or the next action, and a `retroactive: true` flag on plans.

### 13. No profiling or headroom reporting

- **What happened:**
  - Every speed decision came from hand-run profiles: the per-kernel-family GPU time in
    `work/profile-candidate-0008.md`, and launch counts.
  - The 1 GiB headroom at the prompt step's peak was measured by hand. It came out at about 1.04 GiB, and an earlier
    BF16-outputs attempt silently broke it.
- **Proposed fix:**
  - `crit run ... profile` should report GPU time by kernel family per decode token and per prompt token, launches
    per token, and the share of the recon ceiling.
  - `serve --check` should report the minimum free device memory at full context on prompt and decode steps, and
    fail below the run's headroom.

### 14. Serving lifecycle rough edges

- **What happened:**
  - After a serve process was stopped, its runtime had to be cleaned up with `crit serve --recover-runtime`.
  - Aborted readiness probes printed tracebacks that look like failures but are benign.
- **Proposed fix:** recover a stale runtime automatically when its owning process is gone, and print one line for an
  aborted probe.

## What would push this engine further

- **The int8 prompt kernel (prefill and TTFT):** double-buffered `cp.async` loads, larger tiles, and the block decode
  overlapped with the MMAs. It runs at about 10–15 T MAC/s today.
- **Speculative decoding without a drafter model:** prompt lookup with rejection-sampled rounds. The MTP drafter
  does not fit.
- **Decode glue (103 of 118.5 tok/s in process):**
  - activation quantization fused into the norm kernels (174 quantize launches a token)
  - one-kernel routing selection
  - the head's 0.78 GB a token
