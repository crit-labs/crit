# State cache

[state_cache.cpp](state_cache.cpp) implements `StateCache` from
[state.h](../../include/crit/state.h): a storage-agnostic ledger of KV pages and
recurrent state slots for one engine. Core never touches state bytes. The engine
owns storage, writes where a `Round` points and copies through `StateStorage`
hooks (`copy_rows`, `copy_slot`, and `copy_window_rows` with window groups); the
cache decides which ids hold what, when ids may be reused and which committed pages
a new prompt may share. One `PageId` indexes every full-attention layer; each
sliding-window group has a PageId space of its own; slot ids are per component.

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
| `prefix_cache`, `identity` | Cross-sequence reuse of full committed pages; nonzero identity names weights, adapter, KV and recurrent representation and position policy |
| `components` | Recurrent components: `slots`, a `RecurrentCommit` mode and, with the prefix cache, a `snapshots` quota each |
| `windows` | Sliding-window groups: `pages` of their own and a `window` of rows each |
| `hash` | Page hash; null selects `chain_hash` |

Each component needs `slots >= sequences + scratch + snapshots`, where a round takes
`max_round_rows` scratch slots in `column_checkpoints` mode, one in `recompute`
and none in `in_place`, and `snapshots` counts only with the prefix cache, which
requires a nonzero quota for every component. A window group needs pages for one
sequence's window plus a round and a copied tail (at most a whole table), and the
cache needs `pages`: full-attention page ids also name the token rows prefix reuse
matches, even for a model whose attention layers are all windowed (its engine
then stores no bytes for them).

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

`begin(seq, rows, RoundStates::final_row)` is for kernels that compute no state
inside the round, such as a chunked recurrent prefill: its `column_checkpoints`
components behave as `recompute` for that round (one scratch slot, the accepted
state written into `scratch[0]`, snapshots only at the last accepted row), so
Core never takes a page-boundary snapshot from a slot the engine did not write.
The default `every_row` keeps the per-row scratch of a column scan, which a
speculative verify needs; `Round::states` reports the choice.
`rewind(seq, tokens)` shortens an attention-only sequence, but never to a point
whose window would read a window group's dropped pages. `pages_needed` reports the
full-attention pages a round would acquire now. `checkpoint_node(prefix_after, accepted,
boundary_after)` maps an after-token boundary to the verify node holding its state.

## Ownership and reuse of ids

Pages are reference-counted per holding sequence. A released page or slot enters
a retiring list at `epoch()` and becomes allocatable only after `retire(e)` with
`e` at or past that epoch, called once device work up to `e` has settled.
`advance_epoch()` closes an epoch for engines with several batches in flight; a
single-stream engine calls `retire(epoch())` after each step's readiness.
`layout_generation(seq)` changes whenever block table, window tables, live slots or
scratch take new ids, so captured graphs know when to rebind.

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

## Hybrid prefix reuse

With recurrent components, reusing pages is not enough: the sequence also needs each
component's state after the last reused row. A commit that fills a page (one that
gets indexed) copies every component's state after that page's last row into a
snapshot slot through `copy_slot`. The state exists for any boundary inside a
`column_checkpoints` round (that column's scratch slot) and, in `recompute` and
`in_place` modes and in a `final_row` round, only for the round's last accepted row. A boundary that lacks the
state for any component gets no snapshot, and the newest boundaries are snapshotted
first. Snapshots belong to the indexed page and are kept per component in page order.

`reusable` and `attach` end reuse at the last matched page that has a snapshot of
every component. `attach` copies those snapshots into the sequence's live slots
before publishing anything (a throwing copy publishes nothing; the engine
initializes the live slots again), so the engine must not reinitialize them after a
successful attach. A page that leaves the index drops its snapshots.

Snapshot slots are a private pool of at most `snapshots` slots per component, taken
from the free slots once and then recycled: when the pool is full, the oldest
snapshot's slot holds the new one. Only `copy_slot` reads or writes snapshot slots,
so the engine's ordered queue settles every read of an old snapshot before the copy
that reuses its slot, and snapshots never take the slots that sequences and rounds
are guaranteed. `stats().snapshots` counts the pool (its slots count as live). A
throwing snapshot copy in `commit` leaves the round pending and publishes nothing.

## Sliding-window groups

Layers that attend to a window of `window` rows (a row's own included) keep their
keys and values in a window group: its own pages, a window table per sequence
(`window_table`, the same length as the block table) and one `Round::window_kv` row
slot per staged row. Row `r` reads rows `r - window + 1 .. r`, so after each commit
the pages wholly before the next row's window leave the table (`no_page`); they
retire like any page. A group's shared or reusable partial tail is copied through
`copy_window_rows` before a round writes into it, as `copy_rows` does for full
attention.

With the prefix cache, a full window page is linked to the indexed full-attention page
with the same rows when that page is sealed, and stays cached for reuse after its
sequence drops it; allocation evicts cached window pages least recently released
first, and a full-attention page that leaves the index drops its links. Reuse ending
after `n` rows needs, for every group, the window pages holding rows
`n - window + 1 .. n - 1`, so `reusable` and `attach` end at the last matched page
whose window pages are all present; `attach` references them and leaves the pages
before the window as `no_page`. `window_stats(w)` reports a group's pages, with
`indexed_pages` counting the linked ones.

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
| `std::logic_error` | Stale sequence or round, pending round, attach on a non-empty sequence, rewind with components or into dropped window pages, window groups without `copy_window_rows`, reentrant mutation |
| `std::length_error` | A round past `max_tokens` |
| `std::out_of_range` | Commit over staged rows, rewind past committed rows, bad component, window group or checkpoint |
| `CapacityExhausted` (a `std::runtime_error`) | No free sequence, page or slot; retiring ids wait for `retire` |
| `std::runtime_error` | Metadata reservation over the host ceiling |
| `std::overflow_error` | Accounting overflow; round ids or epochs exhausted |

## Limits

Ids are 32-bit (at most `0xfffffffe` pages and slots per component). Reuse is
page-granular, with one indexed page per hash; recurrent reuse is bounded by the
snapshots taken at page boundaries, so a prefix filled inside a `recompute` or
`in_place` round reuses only to its last snapshot. Cross-attention groups and
admission watermarks are not provided.
