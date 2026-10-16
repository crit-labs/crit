# State cache

[state_cache.cpp](state_cache.cpp) implements `StateCache` from
[state.h](../../include/crit/state.h): a storage-agnostic ledger of KV pages and
recurrent state slots for one engine. Core never touches state bytes. The engine
owns storage, writes where a `Round` points and copies through two `StateStorage`
hooks (`copy_rows`, `copy_slot`); the cache decides which ids hold what, when ids
may be reused and which committed pages a new prompt may share. One `PageId`
indexes every attention layer; slot ids are per component.

The runtime does not call it. An engine calls `begin` in `advance`, `commit` or
`discard` in its commit hook, `close` when the session closes and `retire` after
readiness. [The native fixture](../../../tests/core/state_tests.cpp) covers the
rules below.

## Configuration

| `StateCacheConfig` field | Meaning |
| --- | --- |
| `pages`, `page_tokens` | Physical KV pages and rows per page; `pages == 0` for attention-free models |
| `sequences` | Open sequences, forks included |
| `max_tokens` | Logical rows per sequence; at most `pages x page_tokens` |
| `max_round_rows` | Most rows one round stages (drafts + 1 for speculation) |
| `prefix_cache`, `identity` | Cross-sequence reuse of full committed pages; nonzero identity names weights, adapter, KV representation and position policy |
| `components` | Recurrent components: `slots` and a `RecurrentCommit` mode each |
| `hash` | Page hash; null selects `chain_hash` |

Each component needs `slots >= sequences + scratch`, where a round takes
`max_round_rows` scratch slots in `column_checkpoints` mode, one in `recompute`
and none in `in_place`. The prefix cache is attention-only: `prefix_cache` with
components is rejected.

## Sequences and rounds

`open()` returns a `SequenceId{index, generation}` with one fresh live slot per
component for the engine to initialize. `close()` discards any pending round and
releases everything. `fork(parent)` shares committed pages and copies live slots;
the parent must have no pending round.

`begin(seq, rows)` stages rows after the committed frontier: it acquires pages and
scratch slots, copies a shared partial tail, then publishes. `try_begin` returns
`nullopt` instead of throwing `CapacityExhausted`; either way a failure changes
nothing. The `Round` holds `id`, `first_row`, `rows`; `kv` (page and offset per
staged row, never shared); `start` (live slot per component, read-only during the
round); and `scratch` (components x rows, row-major; `no_slot` where unused).
Spans stay valid until the sequence's next mutating call.

`commit(seq, round, inputs)` keeps the first `inputs.size()` staged rows (zero
equals `discard`) and records their tokens. `column_checkpoints` swaps live to
`scratch[n-1]`; in `recompute` the engine writes accepted state into `scratch[0]`
from `start`, which becomes live; `in_place` updates live directly. One round may
be pending per sequence; commit consumes it even when it throws.
`rewind(seq, tokens)` shortens an attention-only sequence. `pages_needed` reports
what a round would acquire now. `checkpoint_node(prefix_after, accepted,
boundary_after)` maps an after-token boundary to the verify node holding its state.

## Ownership and reuse of ids

Pages are reference-counted per holding sequence. A released page or slot enters
a retiring list at `epoch()` and becomes allocatable only after `retire(e)` with
`e` at or past that epoch, called once device work up to `e` has settled.
`advance_epoch()` closes an epoch for engines with several batches in flight; a
single-stream engine calls `retire(epoch())` after each step's readiness.
`layout_generation(seq)` changes whenever block table, live slots or scratch take
new ids, so captured graphs know when to rebind.

## Prefix reuse

A page full of committed rows is hashed as `h_i = hash(h_{i-1}, identity, tokens)`
and linked to its parent page (id and generation, or the parent's indexed
duplicate). It is indexed only if its link checks out; the first valid writer per
hash wins. `reusable(prompt, max_reuse)` walks the prompt's full pages through the
index, requiring exact tokens and exact parent links, so a hash collision costs
reuse, never correctness. Cap `max_reuse` at `prompt.size() - 1`, since reused rows
carry no logits. `attach` does the same on an empty sequence and returns the count
for `Progress::reused_prompt_tokens`.

Indexed pages whose last owner leaves stay cached. Allocation takes free pages
first, then evicts the least recently released cached page; `evict(n)` does it
explicitly. Referenced pages are never evicted, and recomputing an evicted prefix
repairs the index level by level.

## Failure atomicity, memory and threading

`begin` and `fork` acquire ids and call hooks before publishing with non-throwing
steps; a throwing hook returns every id and publishes nothing (a cached page
evicted for the copy goes to the free list). The constructor reserves all host
metadata once through `Resources::reserve(Memory::host, ...)` and later calls
never allocate; `metadata_bytes()` reports it. KV and slot storage are the
engine's to reserve. There is no internal locking: call every method from one
thread, normally the engine worker. Mutating the cache from inside a hook is
rejected.

| Exception | When |
| --- | --- |
| `std::invalid_argument` | Bad configuration, rows over `max_round_rows`, negative tokens |
| `std::logic_error` | Stale sequence or round, pending round, attach on a non-empty sequence, rewind with components, reentrant mutation |
| `std::length_error` | A round past `max_tokens` |
| `std::out_of_range` | Commit over staged rows, rewind past committed rows, bad component or checkpoint |
| `CapacityExhausted` (a `std::runtime_error`) | No free sequence, page or slot; retiring ids wait for `retire` |
| `std::runtime_error` | Metadata reservation over the host ceiling |
| `std::overflow_error` | Accounting overflow; round ids or epochs exhausted |

## Limits

Ids are 32-bit (at most `0xfffffffe` pages and slots per component). Reuse is
page-granular and attention-only, with one indexed page per hash. Snapshots for
hybrid reuse, sliding-window or cross-attention groups, and admission watermarks
are not provided.
