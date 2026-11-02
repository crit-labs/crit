# Batched reference engine

A complete batched serving engine for a synthetic decoder-only transformer, showing
how the pieces fit: a Core `ConcurrentModel` with mixed batches runs the decode rows
and prompt chunks of many requests as one step (laid out by Core's
[batched steps](../../core/src/batch/README.md)), every projection reads its weights
once for all of the step's rows, KV lives in Core's StateCache pages with prefix
reuse, attention runs on the engine's own [paged attention](#paged-attention) kernels (`code/paged_attention.cu`),
and the engine is exported through the native engine ABI with `concurrent_requests`,
so `crit serve` serves it to concurrent clients (`chat-concurrent-v1`). A request's
output is bit for bit the same whether it runs alone or batched.

## When to use it

As the skeleton of a real engine that serves several requests at once: replace the
synthetic weights and layer with the model's, keep Core's step layout, the batch
invariance rules, the page ownership and Core's page-reserving admission. Also as a fixture for
measuring a scheduler, attention kernels or serving changes under concurrency
without a checkpoint. Not a model: its weights are seeded random values, and its
kernels are plain baselines (one tensor-core GEMM, no graphs, no fused epilogues).

## How it works

**Model.** Pre-norm layers: `x += O(attention(rope(Q), rope(K), V))`,
`x += Down(silu(Gate) * Up)`, RMSNorm without weights, split-half rotary embedding,
grouped-query attention, an FP32 residual stream and BF16 projections. Weights are
uniform with variance 1 / fan-in from `(seed, tensor, index)`, so every device builds
the same model.

**One step.** Core hands `execute` the step's work items and Core's `StepLayout`
places them (`step_item`): decode items stage one row each (the request's last
token), prompt items stage their granted chunk; all rows are concatenated (decode
rows first) and each projection is one GEMM over all of them, so a step's weight
bytes are read once however many requests it holds. Each row's key
and value go to the pages its `Round::kv` names; decode rows attend through the
decode kernel and prompt rows through the varlen prefill kernel. Only rows that end a
prompt or decode produce logits (`logit_rows()`, one gathered head GEMM). Plain greedy rows take a
device argmax; rows with temperature, penalties or bias copy their logits and use
Core's `Sampler`.

**Batch invariance.** A request's bits do not depend on its step-mates or on prompt
chunking:

- every row kernel (norm, rotary, SiLU, argmax) reads only its row, and the GEMM's K
  split depends on the weight shape and the multiprocessor count, never on the row
  count (`plan_gemm`; row tiles of 64 rows share each weight tile; zero rows fill short
  tiles);
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
Core's `PageReservations` reserves a request's whole page need (prompt plus output,
with no credit for shared pages), so a running request never runs out; `scheduling_hint` reports
reusable tokens for reuse-aware admission.

**Serving.** [engine_library.cpp](code/engine_library.cpp) builds the engine from the
adapter's JSON configuration and exports it with `CRIT_EXPORT_ENGINE`;
`crit_core.native.resident_engine` turns it into a resident adapter that serves
`max_concurrency` requests at once.

## Paged attention

The engine's own attention over its StateCache pages, part of the example engine and
not a curated kernel: a real engine takes its attention kernels from a provider
[Kernels](../../kernels/README.md) lists, or writes them.
[paged_attention.cu](code/paged_attention.cu) and [device_pages.cpp](code/device_pages.cpp)
implement [paged_attention.h](code/paged_attention.h) (target `crit_paged_attention`):
the KV write kernel, varlen causal prefill, decode and multi-column verify,
`DevicePages` (a `StateStorage` over device memory, one 2-D copy per array for a
copy-on-write tail) and `BatchTables` (a step's packed tables in one device buffer).
[paged_reference.cpp](code/paged_reference.cpp) implements
[paged_reference.h](code/paged_reference.h), the host half (target
`crit_paged_reference`, no CUDA): the page format, the packed batch tables, BF16 and
E4M3 codecs matching the device conversions, `HostPages` (a host `StateStorage` in the
kernels' layout) and `attend`, an FP64 reference of the kernels.

Row `r` of page `p` is physical row `p * page_tokens + r`, laid out
`[row][kv_heads][head_dim]`, one array per layer for keys and one for values. `bf16`
pages store BF16 values; `e4m3` pages store one OCP E4M3 code per element with one
FP32 scale per row and key/value head (max \|x\| over the head vector / 448, 1 for an
all-zero vector), read as `bf16(code * scale)`. Head dimensions are 64, 128 or 256.
`Tables(max_seqs, max_pages, max_rows)` packs one step into the int32 words the
kernels read: block tables, lengths, query starts and write slots (`add(cache,
sequence)` queries a sequence's staged rows, `add_idle()` pads a captured batch size,
`add_slots(round.kv, page_tokens)` lists where the new rows go); filling never
allocates.

Sequence `s` holds `kv_len` keys and its newest `n` rows are its queries: query `i`
sits at position `p = kv_len - n + i` and sees keys `0 .. p`, its own included, so a
step writes its keys first; query head `h` reads key/value head `h / (heads / kv_heads)`.
A sliding `window`, per-head `sinks` in the softmax denominator, a `softcap` and, for
decode and verify, `tree` masks (bit `j` of column `i`'s mask: column `i` sees column
`j` of its round; with a window, the window exceeds 64 keys) shape what a query sees.

The launch shape depends only on host counts (sequences, blocks, the largest query
count), never on lengths or page ids, so one captured graph serves every length and
table; upload the tables before each replay. Decode splits each sequence's keys into
`blocks` interleaved 128-key pieces combined in block order and prefill walks absolute
32-key tiles, so a query's result depends only on its position: a verify column equals
the same column decoded alone and a sequence's result does not depend on its batch. A
sequence whose device counts break the launch (more queries than
`max_columns`/`max_queries` or than its keys, or keys beyond its table) is not
attended: its rows are NaN and the optional `Batch::fault` flag is set. `DevicePages`
allocates with `cudaMalloc`; reserve its `bytes()` in Core `Resources`.

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

`reference.py` is the host model of those rules; Core implements the step layout and
admission in [batch.h](../../core/include/crit/batch.h), and the split is the engine's
(`plan_gemm` in [synthetic_model.cu](code/synthetic_model.cu)). The C++ engine ([engine.h](code/engine.h),
kernels in [synthetic_model.cu](code/synthetic_model.cu)):

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

The kernels need sm_80 or later (BF16 `mma.sync`, `ldmatrix`, `cp.async`). They build for sm_80, sm_86, sm_89 and sm_120; the
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

`tests/inference/paged_reference_tests.cpp` (no GPU) checks the host half: the codecs
on every BF16 value and every E4M3 code (round trips, ties to even, saturation,
subnormals), the packed tables and their bounds, and `attend` against an independent
dense oracle over StateCache pages with reused prefix pages, a fork's copy-on-write
tail and rows crossing pages, grouped heads, head dimensions 64 and 128, BF16 and E4M3
pages and every variant above; a verify's columns equal the same columns attended
alone and a sequence's result does not depend on its batch.
`tests/inference/cuda_paged_attention_tests.cpp` compares the kernels with it (write
bytes and E4M3 scales equal to `HostPages`, prefill, decode, verify and tree masks
within BF16 probability rounding, bit-equal columns and batches, fault rows), and
`tests/inference/paged_attention_tests.cpp` checks them against attention-state's CPU
reference at every verify width up to 16,384 keys and under graph replay.

Measured with `crit_batched_engine_throughput` on an RTX 5090 (one model of hidden
size 2,048, 8 layers, 16/4 heads of 128, FFN 5,632, vocabulary 32,000: 0.98 GB of
BF16 weights), 128-token prompts and 128 output tokens per request, completion tokens
over the wall time of the whole wave (Eval's concurrent rate): 535 tokens/s for one
request, 2,347 for 4, 8,351 for 16, 16,627 for 32 and 28,958 for 64, while each
request kept decoding at 540 to 590 tokens/s. One decode step reads every weight
once, so throughput grows with concurrency until compute or KV reads bind. These
numbers illustrate one device and model; they are not settings to copy.
