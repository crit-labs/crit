# Concurrent request rows and publication

Make compact row identity and staged publication explicit when one engine operation
groups several live requests with different candidate counts, readiness and lifetimes,
while unrelated requests retire independently. Core already owns admission and
scheduling; this entry covers engine-produced rows and their handoff, not a scheduler or
a dynamic-batching backend.

## When to use it

Use it when a grouped operation must publish per-request results safely. Core's
[mixed scheduler](../../core/src/runtime/README.md#mixed-token-work-batches) supplies
decode priority with prefill aging and bounded backfill, and the
[paged batch reference](../attention-state/code/batch_attention.h) with its
[Core example](../attention-state/code/batch_example.cpp) covers the attention part of
such groups. Model arithmetic and kernels belong to the engine.

## How it works

A row carries a request ID, a generation, borrowed candidates and a readiness receipt.
Ready-row compaction preserves input order and applies a row limit. Each selected row is
copied and charged before a group commit can invalidate borrowed model storage. The
commit callback receives all staged rows and only a successful return makes them
publishable; the group is marked consumed before the call, so an ambiguous failure is
never retried. Retirement is independent of publication: owners stay retained until
every receipt tied to them reports physical completion, so an early finisher never waits
on an unrelated request.

**Metadata replicas.** `MetadataTable` keeps bounded opaque byte records in stable slots
keyed by request/generation. Snapshots gather live rows independent of slot order;
diffing two snapshots emits changed rows and removed slots, and applying a delta checks
its exact base, versions, extents and membership before atomically publishing a new
replica snapshot. Slot reuse gets a new version and old snapshots keep their bytes, so
release them after their readers settle.

## What the code supplies

- [implementation.py](code/implementation.py): `Row`, `compact_ready`, `StagedGroup`
  and `RetirementBarrier`. `reserve(nbytes)` returns an accounting owner;
  `model_commit(rows)` performs the engine's state transition. Request IDs and
  generations are positive, tokens nonnegative 32-bit, and every staged row has at least
  one candidate.
- [metadata.py](code/metadata.py): `MetadataTable`, `MetadataSnapshot`,
  `MetadataReplica` (host only; no device upload or pinned pool).
- [host_batch.h](code/host_batch.h): `HostBatch`, described below.

## What the engine supplies

Model execution, fairness, lane allocation and state rollback. A receipt's `done()` must
mean physical completion; the helpers never call `result()`, so establish success before
staging or make `model_commit` reject failed work. Bind each compact index to its request
and generation in every native input, output and state update, and validate the active
identity so a reused lane or ID cannot make an old completion current. Reservation
owners need release semantics for partial construction failure. A commit callback that
mutates some rows and then throws needs an engine recovery policy: the reference
withholds publication and retains owners but cannot undo mutations. The caller
serializes control operations.

## Worked example

Run from `code/`. A pending request is omitted without shifting a later row's identity,
and borrowed candidate arrays can disappear during commit.

```python
from concurrent.futures import Future
from implementation import Row, compact_ready, StagedGroup, RetirementBarrier

ready, pending = Future(), Future()
ready.set_result(None)
borrowed = [4, 5]
rows = [Row(11, 1, borrowed, ready), Row(12, 2, [6], pending),
        Row(13, 4, [7], ready)]
selected = compact_ready(rows, max_rows=2)
assert [r.request_id for r in selected] == [11, 13]
charges = []
def reserve(nbytes):
    charges.append(nbytes)
    return object()

group = StagedGroup(selected, {11: 2, 13: 1}, reserve)
def model_commit(staged):
    assert staged == ((11, 1, (4, 5)), (13, 4, (7,)))
    borrowed.clear()
assert group.commit(model_commit)[0][2] == (4, 5)
assert charges == [8, 4]
try:
    group.commit(model_commit)
except ValueError:
    pass
else:
    raise AssertionError("group committed twice")

barrier = RetirementBarrier([object()], [pending])
assert not barrier.poll()
pending.set_result(None)
assert barrier.poll() and not barrier.owners

from metadata import MetadataTable, MetadataReplica
table = MetadataTable(capacity=4, row_bytes=16)
replica = MetadataReplica(table.empty)
table.set((11, 1), b"first")
first = table.snapshot()
replica.apply(first.diff(replica.snapshot))
held = replica.snapshot
table.set((11, 1), b"second")
delta = table.snapshot().diff(replica.snapshot)
assert delta.payload_bytes == 6
replica.apply(delta)
assert held.rows[0].payload == b"first"
```

## Adapting it

Preserve the request/generation mapping, row and output bounds, borrowed versus owned
lifetimes, and success before publication. Ordering, data structures, budgets and
batching policy are free. [Graph reuse](../../core/src/backends/cuda/README.md#graph-capture)
must bind the same row mapping or a correctly updated indirection table; a stable
physical row-selector table can allow reuse across logical requests at one captured
width, but the selector only chooses storage, while row identity decides who commits.
[Speculative verification](../mtp-decode/README.md) can accept different counts per
row; commit accepted states and tokens for the same requests. A native metadata replica
needs bounded patch upload and must copy-on-write rows rather than overwrite active
readers.

## Native host pipeline

[`HostBatch`](code/host_batch.h) implements Core's optional `BatchExecution` with one
`std::async` CPU task per batch. Construct it in `prepare_batch` with copied work items,
a worker returning `OwnedBatchRow` values, a commit callback, an optional stop callback
and retained owners. `submit(notify)` starts the task and notifies Core; `ready` checks
without blocking; `progress` materializes owned rows; commit is single use even if its
callback throws; `quiesce` requests stop and joins, as does the destructor. Enable Core's
mixed-batch and async capabilities with a bounded positive `max_inflight_batches`.
Core's [pipeline contract](../../core/src/runtime/README.md#asynchronous-batch-pipeline)
owns admission, completion, publication and quarantine.

Workers need independent session and result storage, must not call Core callbacks
recursively, and reserve task memory through the engine's budget. Stop callbacks must be
idempotent; a timeout alone cannot release storage. Device execution implements the same
contract with native events. Same-request speculative pipelining is not implemented.

## Costs and validation

Compaction scans the rows and staging copies selected tokens. Small groups can waste
occupancy; waiting for fuller ones adds latency. Task launch, notification and joining
cost time that must be weighed against the work available to overlap, and a batch using
every lane leaves nothing to overlap. Watch for stale lane identities, copying only a
pointer, publishing after a partial commit failure, and retiring on timeout rather than
completion.

Host fixtures cover variable rows, duplicate identities, output limits, commit failures
and independent retirement. The [portable mechanism fixtures](../../tests/test_portable_mechanisms.py)
compare 500 metadata mutations with full reconstruction and check slot reuse, retained
snapshots, malformed patches and stale deltas. The
[native integration fixture](../../tests/inference/pipeline_tests.cpp) runs a full model
adapter: a blocked producer stays alive while an independent request finishes, plus
cancellation, refill, stale completions, failures, timeout and notification races. None
of these establish throughput or qualify a device implementation.
