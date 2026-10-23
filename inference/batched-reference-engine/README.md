# Batched reference engine

A complete batched serving engine for a synthetic decoder-only transformer, showing
how the pieces fit: a Core `ConcurrentModel` with mixed batches runs the decode rows
and prompt chunks of many requests as one step, every projection reads its weights
once for all of the step's rows, KV lives in Core's StateCache pages with prefix
reuse, attention runs on [paged-attention](../paged-attention/README.md)'s kernels,
and the engine is exported through the native engine ABI with `concurrent_requests`,
so `crit serve` serves it to concurrent clients (`chat-concurrent-v1`). A request's
output is bit for bit the same whether it runs alone or batched.

## When to use it

As the skeleton of a real engine that serves several requests at once: replace the
synthetic weights and layer with the model's, keep the step layout, the batch
invariance rules, the page ownership and the admission. Also as a fixture for
measuring a scheduler, attention kernels or serving changes under concurrency
without a checkpoint. Not a model: its weights are seeded random values, and its
kernels are plain baselines (one tensor-core GEMM, no graphs, no fused epilogues).

## How it works

**Model.** Pre-norm layers: `x += O(attention(rope(Q), rope(K), V))`,
`x += Down(silu(Gate) * Up)`, RMSNorm without weights, split-half rotary embedding,
grouped-query attention, an FP32 residual stream and BF16 projections. Weights are
uniform with variance 1 / fan-in from `(seed, tensor, index)`, so every device builds
the same model.

**One step.** Core hands `execute` the step's work items. Decode items stage one row
each (the request's last token), prompt items stage their granted chunk; all rows are
concatenated (decode rows first) and each projection is one GEMM over all of them, so
a step's weight bytes are read once however many requests it holds. Each row's key
and value go to the pages its `Round::kv` names; decode rows attend through the
decode kernel and prompt rows through the varlen prefill kernel. Only rows that end a
prompt or decode produce logits (one gathered head GEMM). Plain greedy rows take a
device argmax; rows with temperature, penalties or bias copy their logits and use
Core's `Sampler`.

**Batch invariance.** A request's bits do not depend on its step-mates or on prompt
chunking:

- every row kernel (norm, rotary, SiLU, argmax) reads only its row, and the GEMM's K
  split depends on the weight shape and the multiprocessor count, never on the row
  count (row tiles of 64 rows share each weight tile; zero rows fill short tiles);
- the attention kernel follows the row's phase, not the step's row count, and both
  kernels give a query results that depend only on its position (the prefill kernel
  walks absolute 32-key tiles, so chunk boundaries change nothing; decode uses a
  fixed `blocks`);
- the device argmax picks the lowest id among equal maxima, as Core's greedy sampler.

**State.** Each request is a StateCache sequence. `open` attaches the prompt's
verified full pages (at most `prompt - 1` tokens, since reused rows carry no logits)
and the first prompt step reports them as `reused_prompt_tokens`. Commit keeps the
staged rows and indexes full pages; every step ends with the stream synchronized, so
`retire(epoch())` after commit and retirement makes released pages reusable at once.
Admission reserves a request's whole page need (prompt plus output, with no credit
for shared pages), so a running request never runs out; `scheduling_hint` reports
reusable tokens for reuse-aware admission.

**Serving.** [engine_library.cpp](code/engine_library.cpp) builds the engine from the
adapter's JSON configuration and exports it with `CRIT_EXPORT_ENGINE`;
`crit_core.native.resident_engine` turns it into a resident adapter that serves
`max_concurrency` requests at once.

## Worked example

From `code/`:

```python
from reference import arrange, pages_needed, plan_gemm, split_dot

# A projection's K split depends on its shape and the device, never on the step's rows.
splits, per = plan_gemm(n=1536, k=2048, multiprocessors=170)
assert (splits, per) == (7, 5)   # 24 column tiles, 32 K tiles: 7 splits of at most 5 tiles
x, w = [0.5, -1.0, 0.25] * 683 + [1.0], [0.125] * 2048
assert abs(split_dot(x, w, per) - sum(a * b for a, b in zip(x, w))) < 1e-9
# A mixed step: two decode rows, a prompt chunk that ends its prompt and one that does not.
layout, decode_rows, logits = arrange([("a", "prefill", 3, True), ("b", "decode", 1, False),
                                       ("c", "prefill", 4, False), ("d", "decode", 1, False)])
assert decode_rows == 2 and [r for r, _ in layout[:2]] == ["b", "d"]
assert logits == [0, 1, 4]   # b, d and the last row of a's prompt
assert pages_needed(prompt=53, output=24, page_tokens=16) == 5
```

The C++ engine ([engine.h](code/engine.h), kernels in
[synthetic_model.cu](code/synthetic_model.cu)):

```cpp
namespace be = crit::batched_engine;
be::Config config;
config.shape = {32000, 2048, 8, 16, 4, 128, 5632, 10000.f, /*seed*/ 5};
config.max_concurrency = 64; config.max_batch_tokens = 512; config.context = 4096;
auto engine = be::make_engine(config, std::make_shared<be::Stats>());
auto a = engine->submit(prompt_a, 128), b = engine->submit(prompt_b, 128);  // share steps
const auto result = a.wait();   // result.timing.reused_prompt_tokens reports prefix reuse
```

## Invariants

- Rows of one step never influence each other's bits: row-local kernels, a GEMM split
  fixed per weight, phase-chosen attention kernels with position-only results.
- Write a row's key and value only where its round points, before the attention that
  reads them; commit or discard every round in `commit`; close the sequence at
  retirement.
- Retire released pages only after the work that read them has settled (here, after
  each step's synchronization).
- Admission must keep every admitted request able to finish; reserve before opening.
- The prefix identity names the weights and the KV representation.

## Costs and validation

The kernels need sm_80 or later (BF16 `mma.sync`, `ldmatrix`, `cp.async`, and
paged-attention's requirements). They build for sm_80, sm_86, sm_89 and sm_120; the
tests below ran on sm_120 only.

`tests/inference/batched_engine_tests.cpp` checks the GEMM (with and without split K,
stored and accumulated) against an FP64 host product and bit for bit across row
counts, RMSNorm, rotary embedding, the SiLU product and the argmax against host
references (the argmax against Core's greedy `Sampler`, ties included). Nine
requests (prompts of 1 to 120 tokens, two sharing 53 tokens, greedy and seeded top-k
sampling) run one at a time without prefix reuse and then all at once with other
prompt chunking: every request's 24 tokens are equal, with BF16 and with E4M3 pages,
steps hold up to 9 requests and mix prompt chunks with decode rows, and a repeated
prompt reports 48 reused tokens with an unchanged reply.
`tests/inference/batched_engine_serving_check.py` serves the library through crit
serve's resident host: four concurrent HTTP requests run at once and each reply
equals the same request served alone.

Measured with `crit_batched_engine_throughput` on an RTX 5090 (one model of hidden
size 2,048, 8 layers, 16/4 heads of 128, FFN 5,632, vocabulary 32,000: 0.98 GB of
BF16 weights), 128-token prompts and 128 output tokens per request, completion tokens
over the wall time of the whole wave (Eval's concurrent rate): 535 tokens/s for one
request, 2,347 for 4, 8,351 for 16, 16,627 for 32 and 28,958 for 64, while each
request kept decoding at 540 to 590 tokens/s. One decode step reads every weight
once, so throughput grows with concurrency until compute or KV reads bind. These
numbers illustrate one device and model; they are not settings to copy.
