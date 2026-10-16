# Transactional KV, shared pages and explicit-prefix attention

Stage attention state provisionally, expose the intended causal prefix to a query,
and commit only the accepted part. A Python host formula makes that behavior
executable without a page layout or kernel. A native C++20 path stores FP32 or
G32 INT8 KV pages for Core's `StateCache`, which keeps the page, round, sharing,
copy-on-write, prefix and retirement bookkeeping, and supplies tiled CPU
prefill/decode attention with grouped query heads, explicit masks and RoPE.

## When to use it

Use this when request state may be partially accepted, rolled back or
checkpointed, especially during speculative verification, or as a small attention
reference. The native page path suits CPU engines, prefix caching and speculative
branches; it supplies real host backing and copies. It does not describe recurrent
state or every attention variant. Measure paging against contiguous state for
single-request workloads; it pays off with variable-length concurrent state,
prefix sharing or branches.

## How it works

`KVState` holds ordered `committed` and `pending` rows (key width `K`, value width
`V`). `stage` validates and appends to pending within capacity; `commit(n)` keeps
the first `n` pending rows and drops the rest (no argument commits all; a bad count
changes nothing); `rollback()` drops pending; `attend(query)` reads committed rows,
or all rows with `include_pending=True`:

```text
score_i = dot(query, key_i) * scale       default scale = 1 / sqrt(K)
weight_i = exp(score_i - max(score)) / sum_j exp(score_j - max(score))
output[c] = sum_i weight_i * value_i[c]   for c in 0..V-1
```

`include_pending=True` is not a per-query causal mask: an earlier query in a
verification block must see only its own prefix, or use a backend with an
explicit mask.

Natively, bookkeeping and storage are split. Core's `StateCache`
([Core state cache](../../core/src/state/README.md)) maps each sequence's rows to
`PageId`s, stages one round at a time, commits an accepted prefix or discards it,
rewinds, forks with shared pages, reuses verified full prompt pages and stamps
released pages with an epoch. `StatePages` reserves one contiguous FP32 arena;
each page holds `page_rows` token-major `[key | value]` rows. `StatePages::stage`
validates, calls `StateCache::begin` and writes each row where `Round::kv` points,
so a rejected row changes nothing. When a round extends a shared or indexed
partial tail, the cache takes a new page and calls `StatePages::copy_rows` first;
full shared pages are untouched. `StateCache::pages_needed` gives the (advisory)
admission cost.

`view(cache, sequence)` copies the committed block table; `view(cache, sequence,
true)` adds staged rows and `prefix(n)` narrows either. Released pages stay
readable until `retire()` covers the releasing epoch, so in-flight readers are
protected by retirement, not reference counts. A later round can overwrite an
unshared tail page in place, so finish reading those rows before staging over them.

With `prefix_cache` and a nonzero `identity` in `StateCacheConfig`, Core indexes
committed full pages by a token hash chain verified by exact tokens, and
`attach()` references them in an empty sequence. Only full pages are reused: with
two-row pages, a committed `[1,2,3,4]` lets `[1,2,9]` reuse `[1,2]`.

`paged_attention` computes the same formula with online softmax and double
accumulation directly over pages, with `O(V)` accumulator storage. The complete
operation in [attention.h](code/attention.h):

- `HeadTensorView` accepts token-major or head-major FP32 queries with padding;
  KV rows hold all heads' keys followed by all heads' values.
- `rotary` applies caller-supplied positions and frequencies, interleaved or
  split-half, with optional partial rotary dimensions.
- `PagedAttentionPlan` retains one KV image and explicit logical query rows: full
  or chunked prefill, decode, unordered queries. GQA/MQA map
  `kv_head = query_head / (query_heads / kv_heads)`.
- `AttentionMask` combines causal, a backward window (zero means unlimited), an
  optional byte keep-mask `[Q,K]` and additive bias `[Q,Hq,K]`.
- `AttentionResult` holds `[Q,Hq,Dv]` output and double `[Q,Hq]` log-sum-exp;
  fully masked rows return zero and negative infinity.

The plan packs a query tile and streams key tiles from pages; queries sharing a KV
head are rows of one QK GEMM, and online softmax rescales across tiles. There is
no dense KV or full score matrix. `ScalarAttentionGemm` is the default; Kernels'
[OpenBLAS adapter](../../kernels/examples/openblas/attention_gemm.h) uses SGEMM,
qualified by its [retained-provider workflow](../../kernels/examples/openblas/README.md#paged-attention-integration).

## What the code supplies

| File | Supplies |
| --- | --- |
| [implementation.py](code/implementation.py) | `KVState`, its transaction and a stable-softmax formula; optional `backend(query, rows, scale)` |
| [state_pages.h](code/state_pages.h) | `StatePages` (a `crit::core::StateStorage` of FP32 rows) and `StatePagesView` |
| [encoded_kv.h](code/encoded_kv.h) | `EncodedPages`: signed int8 plus one FP32 scale per 32 values (1.125 bytes/value vs 4) |
| [attention.h](code/attention.h) | `paged_attention`, `PagedAttentionPlan`/`BasicAttentionPlan`, masks, RoPE |
| [batch_attention.h](code/batch_attention.h) | `BasicAttentionBatch<KVView>`, `PagedAttentionBatch` |

`EncodedPages` needs K and V widths divisible by 32 and includes `grouped_int.h`
from quantized-projections, so add both code folders to the include path and link
`crit::core`. Inference keeps no page ledger; GPU pages, device attention and
model projections are engine work.

`PagedAttentionBatch` holds one plan per request (each with its own geometry,
context and query rows) and returns results in key order; stale, duplicate or
missing identities fail before arithmetic. Limits cover request count, query rows,
workspace, output and metadata. It executes plans serially on the CPU.

## What the engine supplies

Keep one state image per request or owned branch. Apply position transforms and
key/value construction before staging, and pass the model's own attention scale.
Logical query indices address the retained KV image and set causal visibility
independently of RoPE position IDs. Queries, masks and bias must stay immutable
through `run`. A plan sees one KV image: build a new plan after appends. One plan's
workspace is not reentrant; separate plans can read shared pages concurrently.

A GEMM callable must be synchronous, write every output cell and keep no borrowed
spans. Failure or cancellation returns no result and leaves KV untouched; the
engine still rolls back provisional KV and commits only after all layers succeed.
Map acceptance to the session's committed cursor; a produced token need not have
its own KV row yet, so keep the engine's decode convention.

Call every `StateCache` method from one engine thread; views may be read on
workers. Retire an epoch only after every reader submitted up to it has finished
(`advance_epoch()` after submitting, `retire(e)` once settled); a synchronous CPU
engine calls `retire(epoch())` after each step. A rollback or a submitted device
event does not prove a reader finished. Reserve `pages.reserved_bytes()` once for
the arena (not per branch); `StateCache` reserves its own `metadata_bytes()`.

Prefix tokens describe state **already materialized** in KV. A full prompt hit
supplies no next-token logits or other model state: recompute the last position,
look up an earlier prefix, or retain those results under a compatible identity.
The identity must name weights, layer, adapter, positional conventions and KV
representation (including the codec for encoded pages). Core verifies hits by
tokens, never KV values, so an incomplete identity silently reuses wrong state.
Core rejects `prefix_cache` with recurrent components.

On a GPU, a speculative round can keep the KV frontier on the device: verify
writes K+1 rows at `frontier..frontier+K`, acceptance advances the frontier by
`n_accepted` without a host read, and the next round overwrites rejected rows.
See [device-speculative-round](../device-speculative-round/README.md).

## Worked example

Run from `code/`. The provisional query is allowed to see both staged rows.

```python
from implementation import KVState

state = KVState(capacity=3, key_width=2, value_width=1)
state.stage((1.0, 0.0), (2.0,))
state.stage((0.0, 1.0), (10.0,))
assert state.attend((0.0, 0.0), include_pending=True) == (6.0,)
state.commit(1)
assert state.attend((0.0, 0.0)) == (2.0,)
assert state.pending == []
state.stage((1.0, 1.0), (99.0,))
state.rollback()
assert len(state.committed) == 1
assert state.attend((0.0, 0.0)) == (2.0,)
```

Native examples: [paged_example.cpp](code/paged_example.cpp) commits and attaches
a prompt page, attends over a staged row, discards it and checks page return after
`retire`; [attention_example.cpp](code/attention_example.cpp) runs RoPE, GQA
prefill, prefix reuse and decode; [batch_example.cpp](code/batch_example.cpp) runs
24 requests through Core with 16 lanes and checks every token against a dense
oracle. All use synthetic inputs.

```sh
cmake -S inference -B build/inference -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/inference --config Release
ctest --test-dir build/inference -C Release -R "crit_paged_kv|crit_state_pages|crit_attention" --output-on-failure
```

## Adapting it

Preserve causal indexing, the accepted prefix and the separation of logical
visibility from physical lifetime: a rewind and a completion fence are different
facts. Paged or compressed storage can replace lists and fused attention can
replace the formula, with representation, masks, scaling and arithmetic explicit
and freshly checked. Tie graph bindings to the storage owner, physical address and
the sequence's `layout_generation()`; a page id alone is insufficient.
[Draft verification](../mtp-decode/README.md) supplies an accepted count and
[checkpoint storage](../checkpoint-storage/README.md) can retain a committed image
once its reads settle.

## Costs and validation

Host work is `O(N * (K + V))` for `N` visible rows. `StatePages` payload is
`pages * page_rows * (K + V) * 4` bytes. A fork extending a shared tail needs an
extra page even for one row; if none is free, `begin` throws `CapacityExhausted`
before exposing new KV. For tiled attention with `M = query_tile * (Hq/Hkv)` and
`T = key_tile`, scratch is `4*(M*Dk + T*Dk + T*Dv + M*T + M*Dv) + 8*(M*Dv + 2*M)`
bytes (`workspace_bytes()`); `output_bytes()` is `Q*Hq*(4*Dv+8)`. These exclude
allocator overhead, cache metadata, provider workspace and caller-owned masks, so
the scratch limit is not a total-memory cap.

Invalid geometry, mask errors, nonfinite inputs or GEMM outputs and overflow
fail before a result; all QK cells must be finite even if masked.

Host fixtures (`tests/test_techniques.py`) cover the Python transaction.
[Native fixtures](../../tests/inference/state_pages_tests.cpp) cover staged,
committed, discarded and rewound rows, COW tails and 5,000 randomized operations
against a value oracle; Core's [state fixtures](../../tests/core/state_tests.cpp)
cover the bookkeeping. The [attention fixture](../../tests/inference/attention_tests.cpp)
adds 2,304 comparisons with dense double attention across MHA/GQA/MQA, page
boundaries, chunks and masks, and the batch fixture 74 more. None qualify GPU
fences, complete-model equivalence or throughput. The design follows
[PagedAttention](https://arxiv.org/abs/2309.06180).
