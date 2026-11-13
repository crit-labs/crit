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

For debugging, `StatePages::poison(value)` (NaN by default) fills every page the cache
hands out for new rows, so attention that reads a row the request never wrote, through
a wrong length or mask, sees NaN instead of a previous owner's rows; `fill(page,
value)` writes the pattern into Core's guard page (`StateCacheConfig::guard_page`) for
masked block-table entries.

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
- `rope_table(rope, head_dim, max_position_embeddings, sequence_length)`
  ([rope_table.h](code/rope_table.h)) gives those frequencies from the model's
  `config.json`: the inverse frequency of each rotated pair (`rotary_dim(rope,
  head_dim)` dimensions rotate) and the attention factor that scales cos and sin,
  as Hugging Face's `ROPE_INIT_FUNCTIONS` derive them, for `default`, `mrope`,
  `linear`, dynamic NTK, YaRN (with `mscale`, `attention_factor` and `truncate`),
  `llama3` and `longrope`. `rope` is Core's `RopeConfig` (`load_model_config(...)
  .rope`), `max_position_embeddings` the declared length dynamic NTK needs and
  YaRN, Llama 3 and LongRoPE read when the rope object leaves factor or original
  length out; `sequence_length` picks dynamic NTK's base and LongRoPE's long
  factors past the original length. Other types and missing parameters are
  refused. [rope_table.py](code/rope_table.py) is the Python twin (on
  `crit_core.model`'s `RopeConfig`); `tests/test_rope_tables.py` checks it
  against Transformers and `tests/inference/rope_oracle.py` checks the two agree.
- `PagedAttentionPlan` retains one KV image and explicit logical query rows: full
  or chunked prefill, decode, unordered queries. GQA/MQA map
  `kv_head = query_head / (query_heads / kv_heads)`.
- `AttentionMask` combines causal, a backward window (zero means unlimited), an
  optional byte keep-mask `[Q,K]` and additive bias `[Q,Hq,K]`.
- `AttentionResult` holds `[Q,Hq,Dv]` output and double `[Q,Hq]` log-sum-exp;
  fully masked rows return zero and negative infinity.

The plan packs a query tile and streams key tiles from pages; queries sharing a KV
head are rows of one QK GEMM, and online softmax rescales across tiles. There is
no dense KV or full score matrix. `ScalarAttentionGemm` is the default; an
engine passes another GEMM (a vendor BLAS's SGEMM) through the same call.

## What the code supplies

| File | Supplies |
| --- | --- |
| [implementation.py](code/implementation.py) | `KVState`, its transaction and a stable-softmax formula; optional `backend(query, rows, scale)` |
| [state_pages.h](code/state_pages.h) | `StatePages` (a `crit::core::StateStorage` of FP32 rows) and `StatePagesView` |
| [encoded_kv.h](code/encoded_kv.h) | `EncodedPages`: signed int8 plus one FP32 scale per 32 values (1.125 bytes/value vs 4) |
| [attention.h](code/attention.h) | `paged_attention`, `PagedAttentionPlan`/`BasicAttentionPlan`, masks, RoPE |
| [rope_table.h](code/rope_table.h), [rope_table.py](code/rope_table.py) | `rope_table`, `rotary_dim`, `RopeTable`: rotary frequencies and attention factor from `config.json`'s rope parameters |
| [batch_attention.h](code/batch_attention.h) | `BasicAttentionBatch<KVView>`, `PagedAttentionBatch` |
| [host_page_tier.h](code/host_page_tier.h) | `HostPageTier`: a bounded host tier for the prefix pages Core's `StateCache` evicts ([below](#host-page-tier)) |

`EncodedPages` needs K and V widths divisible by 32 and includes `grouped_int.h`
from quantized-projections, so add both code folders to the include path and link
`crit::core`. Inference keeps no page ledger; GPU pages, device attention and
model projections are engine work. Device attention over paged KV comes from a provider
([Kernels' libraries](../../kernels/README.md#libraries)): FlashInfer's paged and ragged attention
(BF16 and FP8 KV, prefill, decode and a verify's query block) or FlashAttention's paged KV,
which read a block table like Core's `StateCache` gives (`view(...)` above). Store device
pages in a format the provider reads (BF16 or FP8 rows per head); the INT8 G32 rows of
`EncodedPages` are this entry's host codec, which no provider reads, so a device engine
that wants them writes its own kernel and measures the provider on BF16 or FP8 pages as
the yardstick.

`PagedAttentionBatch` holds one plan per request (each with its own geometry,
context and query rows) and returns results in key order; stale, duplicate or
missing identities fail before arithmetic. Limits cover request count, query rows,
workspace, output and metadata. It executes plans serially on the CPU.

**Sliding-window layers as rings (`kv:sliding`).** A pool made with `pool(..., sliding=W, span=S, positions=N)` holds
the layers whose tokens see only their last W positions (local layers): each sequence keeps a ring of
ceil((W + S) / page) + 1 pages (S: the most positions a row writes in one step, a prompt chunk or a verify window; N:
the most positions a sequence reaches, the page tables' width), logical page p in ring page p mod ring. The tables the
attention reads are the ring expanded to the logical pages, so a position older than the window aliases a newer
one's page, which the window never shows; softmax-attention reads such a pool with its window (`attention:sliding`)
and every kernel path of paged-kv-attention reads it unchanged. A sequence holds at most its ring whatever its length,
`bind` takes no page past it for a captured round, a fork shares the ring until either writes, `keep` frees no page
of a sequence that has wrapped, and the plugin's facts count these pools as `kv_sliding_bytes_per_sequence`. Tested
in softmax-attention's `test_sliding_pool.py`: prompt chunks, decode steps, verify windows rolled back, forks and 120
captured rounds over a ring read bit for bit as a full pool under the same window (BF16 on crit's kernels; FP32 on
the torch fallback within 1e-5), and a captured round replays over it.

**Lending the pool's unused pages.** The device pool ([pages.py](code/pages.py) `KVPool`) keeps its page tensors in
one allocation, with `spare` bytes after them for a borrower, and takes the lowest free page first, so every page at
or past its `floor` holds no KV. A device expert cache borrows that memory: `lendable()` gives the allocation, each
page tensor's place in it, the spare bytes and the floor, and `on_reserve(callback)` calls the borrower before a page
at or past the bound it returned is handed out (it reclaims the slots under the new floor first) and after the floor
falls (pages freed). Readers see the same tensors as without a borrower.

### Host page tier

When the device page pool holds fewer pages than the prompts that keep coming back
(several long conversations, an agent's tool loop), Core's prefix cache evicts the
least recently released pages, and the next turn recomputes them. `HostPageTier(slots,
config)` keeps evicted pages in `slots` host page slots instead, for the cache that
`config` describes. It is a ledger like Core's: the engine owns the slots (pinned
memory for a device engine) and every copy.

1. In `StateStorage::evicted(page)`, call `admit(page)`. It returns the slot to copy
   into (the least recently used page leaves when the tier is full), or nothing when
   the tier holds that page already. Copy the page's rows, each component's state
   where `page.snapshots` names a slot and each window group's rows where
   `page.windows` names a page, enqueued on the engine's ordered queue so the copies
   read before any later write to those ids.
2. After `attach(seq, prompt, max_reuse)`, call `continuation(prompt, committed,
   max_reuse, max_round_rows, scope)` (the sequence's reuse scope from `StateCache::open`;
   pages written under another scope never match). It returns the kept pages that continue the prompt,
   each verified by its chain hash and exact tokens, holding every window group's rows
   (a window page a restore round left unwritten would be linked for reuse), ending at
   the last page that holds every component's state, and fitting one round.
3. Restore them in one round of `pages x page_tokens` rows with
   `RoundStates::final_row`: copy each page's rows where `Round::kv` and
   `Round::window_kv` point, the last page's state into `scratch[0]` (an `in_place`
   component's into its live slot), and commit the pages' tokens. Core indexes the
   pages again and snapshots the state at the round's end, as for computed rows.
   Repeat from step 2 until it returns nothing, then compute the rest.

A false hit needs two prefixes with the same chain hash whose page tokens also
match; Core's own index rules that out with exact parent links, which a page outside
the device cannot keep. The ledger allocates nothing after construction, so `admit`
is safe in the noexcept hook. A device engine spills with an asynchronous device-to-host
copy that the next write to those ids waits for (the same stream, or an event), since
the hook runs inside `begin` and must not block, and restores with a host-to-device
copy before the round's first reader. The tier costs `slots` pages of host memory,
which belongs inside [host-memory admission](../host-memory-admission/README.md).
The [tier fixtures](../../tests/inference/host_page_tier_tests.cpp) run a hybrid cache
(one recurrent component, one sliding-window group) with ten device pages: a
24-row conversation evicted by two other prompts is restored for its next turn
(24 rows restored, 4 computed), and every KV row, window row and the live state equal
what recomputing gives; the run rules, token verification under a constant hash and
capacity are checked separately. Device copies and timing are not covered.

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

## On the device

The plugin's device form is a paged pool in torch ([pages.py](code/pages.py)):
every layer's keys and values in pages of `page` positions, each sequence's
page table, copy-on-write forks, and page types BF16, FP16, FP32, FP8 E4M3 with
the layer's key and value scales (`fp8`), FP8 E4M3 with an FP32 scale per
position and KV head (`fp8:row`: the row's absolute maximum over 448, the
quotient rounded to the nearest E4M3 value) and INT8 with an FP32 scale per
position and KV head (`int8`: over 127, rounded to the nearest code, ties to
even). `fp8_rows` and `int8_rows` are the encoding rules; crit's kernel
`paged-kv-attention` (built for the engine's device, with the engine's tuned knobs
or its defaults) writes a step's keys and values in one launch with the same
codes and scales, and torch writes them where it does not build.

Captured decode rounds bind a sequence to a round slot (`rounds:kv_cache`):
`bind(slot, sequence, capacity)` takes its pages up to `capacity` positions
(prompt, output limit and the widest verify window) and puts its page table at
a fixed device address (`round_tables`), copying a page a fork still shares;
`reserve(RoundBatch)` then computes every token's slot on the device from those
tables and the batch's positions, so a captured round never allocates or waits
for the host; a round wider than the pool's `round_rows` is refused. A token
outside every bound capacity (an unbound slot, a position past the capacity)
and a write's slot below 0 go to the dump page, one page past the sequences'
that no table holds for a sequence and no borrower is lent; every unused entry
of a round's or a step's page table points there. `start_round(view)` notes
each slot's length as the round finds it, and `commit_round(view)` sets it to
the round's frontier where the row emitted and keeps the noted one where it
emitted nothing (a held row, or a `keep()` between rounds), on the device; with
a tree verifier's accepted nodes (`view.tree_path`) it first moves their keys
and values to consecutive positions, once a round. `unbind(slot)` reads the
length back (one wait) and frees the pages past it; `move(source, target)`
keeps the decoding rows packed at the front.

Captured prompt chunks (`prompts:kv_cache`, crit_plugins.prompt_graphs):
`stage_prompt(batch)`, outside the graph before each replay, takes the
chunk's pages up to its end and puts its real tokens' KV slots and its page
table in fixed buffers (`prompt_slots`, `prompt_table`), the padding's slots at
the dump page; `reserve(batch)` inside the graph returns those slots, and
`commit_prompt(batch)` sets the length after. A chunk of padding alone (a
capture's warm-up) takes no page. The attention plugins read the prompt table
on the device: crit's prompt reader at a position read on the device
(one graph for every frontier), else the bucket as window rows through the
split-KV reader or its torch fallback; no FlashInfer plan runs in a capture.

[device_attention.py](code/device_attention.py) is the attention slot's device
form over this pool, which softmax-attention and partitioned-attention share:
crit's paged-kv-attention kernels, built for the engine's device with tuned or default knobs
(a split-KV reader of decode and verify windows over every page type, whose partitions walk each row's own length read on the
device, with tree masks and block-sparse lists, tiles every GPU multiplies
with mma.sync and one partition count per pool, so a column's bits are the same
alone, in a window, in a step and in a round; and a prompt reader over every
page type, eager prompt rows one row a launch), then
torch: rows gathered and multiplied by SDPA, or a masked FP32 softmax over the
page tables that captures in a CUDA graph (in a capture it reads every position
the tables hold, so its cost follows the pool's capacity; in an eager step only
the positions its rows reach). The pool's `window` option (16 tokens) is the one
knob of what is a window row: a longer row is a prompt row. No library kernel
reads the pages by default: the plugins' option `flashinfer` (default `off`)
set to `auto` puts FlashInfer's paged prefill on eager prompt rows of 16-bit and
FP8 pages (and with softmax-attention's `decode: flashinfer` its decode on a
step's window rows), a yardstick to compare crit's kernels with in an engine.

The attention plugins' option `reader` picks which of the kernel's structures read
the pages: `split` (the default, the split-KV reader above and crit's earlier prompt
reader) or `flash`, the kernel's flash structures: a row's positions in contiguous
splits of one chunk per pool (`flash_chunk()`, the engine's tuning or 1,024
positions), a window's columns times a KV head's query heads packed into
tensor-core tiles of up to 128 rows, the splits merged (or, for `prompt_reader:
decode`'s prompt pieces, folded in the program) in split order. A column's bits are
then the same alone, in a verify window, in a step and in a round (one reduction,
another than `split`'s), and the facts say so (`reader`). Crit's prompt reader
becomes the flash prompt chunk, which splits a chunk's positions by its length: a
prompt token's bits follow the chunk it is read in, so the facts carry
`prompt_rows: not row-invariant` (as FlashInfer's prefill, within the readers'
rounding of decode rows); `reader: split` keeps the earlier reader. Block lists stay
on the split launch, and the options `partitions` and `piece` do not apply.
`reader: cuda` reads the same launches through the kernel's CUDA flash tile
(`cuda-window`, `cuda-fused`, `cuda-prefill`: the queries in shared memory, 16 rows
a warp, each warp's online softmax in registers, one chunk and block per pool from
`cuda_chunk()`): a column's bits are again the same alone, in a window, in a step and
in a round (its own reduction), its prompt chunk not row-invariant as the flash
one's; int8_qk prompt chunks stay on the Triton flash prompt chunk.

The plugin's `facts()` report `kv_bytes_per_token`: the device bytes one token's
slot takes in every page tensor of the pools the plugin made (keys, values and an
INT8 or row-scaled pool's scales, every layer), summed over the pools. The engine's
facts carry it to crit serve's `engine_facts` and to Eval, which sizes the KV cache
and prices the KV term of its floors by it.

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

The shared [batch engine](code/batch_engine.h) also has a
[memory recovery adapter](code/recoverable_engine.h) for Core's optional preemption
contract. It reserves page rounds transactionally, reports typed pressure before
submission, releases a settled victim's physical pages, and restores its continuation
by replaying accepted history without resampling or publishing it again. This is a
complete recovery recipe for the supplied synthetic attention model; an engine with
recurrent, draft, grammar or other state must include that state in its own image.
The [recovery fixtures](../../tests/inference/recovery_scheduler_tests.cpp) check
physical page reuse and dense-reference output, suspended/restoring cancellation,
failure ownership and quarantine. Native HTTP fixtures exercise this adapter through
the stock resident serving path. These CPU operations settle synchronously.

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
