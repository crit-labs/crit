# Expert residency

Bounded expert residency keeps a subset of routed expert weights on the preferred
device and manages their replacement safely. This entry supplies three storage
and ownership implementations (choose the one closest to the engine) and optional
host planners for admission, bank sizing and copy grouping.

## When to use it

Use it when the full routed weight set is expensive to keep on the preferred
device and reuse can avoid enough transfers or slower execution to repay cache
management. It is a candidate strategy, not a universal improvement: low reuse,
cheap direct access or expensive replacement can favor a different placement.

## How it works

Residency separates what bytes mean, where they live, and whether they may be read
or replaced. `ExpertKey` identifies revision, layer, expert and encoding; the
engine must make those fields capture every semantic or layout distinction. A slot
generation changes whenever content is replaced, even at the same address.

An expert may span several tensors, scales or metadata arrays; treat its declared
components as one bundle. The fixed-slot protocol is:

`EMPTY or unprotected READY -> LOADING -> READY on complete successful upload`

Starting replacement invalidates the previous content. Failure or abort leaves the
slot empty; relabeling cannot restore partially overwritten bytes. READY is
published only after every component's receipt reports completion and success.
Acquire all current route hits before selecting victims. Each acquisition pins
content until its last physical reader finishes. A READY map is a snapshot, not a
lease, and a completion event from a previous upload is not evidence for the
current generation.

## What the code supplies

| Implementation | Supplied mechanism | Engine still supplies |
| --- | --- | --- |
| [`ExpertCache`](code/implementation.py) | Byte-bounded ready-object cache, acquisition-order eviction, leases and deferred-use receipts | `load(key, nbytes)` returning a ready owner; safe physical release; teardown |
| [`BundleDirectory`](code/implementation.py) | Fixed-slot complete-bundle metadata, pinned uses, upload tickets, component validation, source retention and ordered publication | Reserved backing buffers, copies, completion receipts, native visibility and recovery |
| [`crit::techniques::SettledSlots`](code/settled_slots.h) | C++ fixed-slot directory, content generations, protected uses, temporary region loans with transactional relocation, and slot loans by which owners reclaim memory lent to the cache | Slot geometry, reserved backing, physical completion, transfers and loan policy |
| [`planning.py`](code/planning.py) | Phase-aware admission, compatible-layout bank sizing, dual-contiguous copy runs and demand/cost observations | Allocation, directory integration, exact fallback, native copies/projections and measured costs |

`ExpertCache` capacity is in bytes; the two directories use slot counts. Neither
directory allocates a device cache or supplies a projection kernel; their value is
the eviction and publication state machine, which can be adapted into a native
manager. All directory operations require serialized control.

## What the engine supplies

Declare physical component extents including padding and metadata. A fixed pool
needs about `slot_count * sum(component_bytes)` plus maps, staging and
bookkeeping; reserve it through Core resources alongside context state and scratch.

For `BundleDirectory`: `begin_upload(key)` selects a slot; for each component call
`submit(ticket, name, source_owner, enqueue)`, where `enqueue` returns a
`BundleReceipt` bound to that ticket, component, encoding, extent and completion;
`publish(ticket)` validates all components together. An ambiguous failure requires
backend quiescence before `abort` releases sources. Before reading, `acquire(key)`
and keep the use; call `settle(use, completion)` only after reads physically
finish. Drain uses and uploads before `close()`.

For `ExpertCache`, `ExpertLease.close(completion)` transfers protection to an
in-flight receipt; omitting completion asserts no future reader needs it. A failed
load does not restore entries evicted to make space.

**Admission.** `cache_admission(phase, tokens, experts_per_token, expert_count,
slots)` returns `populate`, `reuse` or `bypass`. Prefill chooses read-only reuse
when slots exist; decode may populate within the token limit when
`min(tokens * experts_per_token, expert_count)` fits. Every rejected or missed
expert still runs through the exact source path; admission is not routing. Pass
`touch=False` to `BundleDirectory.acquire` for read-only reuse, and never call
`begin_upload` from it.

**Banks and copies.** `BankComponent` describes one component (encoding/layout
identity, shape, stride, alignment, guard bytes); `size_banks` shares storage only
across matching revisions, expert counts and component descriptors, splitting a
bank budget in proportion to each group's host payload. Scales indexed by original
expert ID must keep that ID, not the slot. `copy_runs` groups unique
`(expert, slot)` pairs only when both are consecutive and performs no copies.
`observe_cache` separates route occurrences from unique materialization demands
and records caller-measured upload bytes, calls and synchronization time.

## Worked example

### Temporary cache memory loans

The C++ directory can lend a contiguous region of its existing allocation to
scratch work. Construct it with `(capacity, slot_bytes, backing_owner)`; the owner
retains the allocation and its Core reservation, and slots must describe complete
bundles in one contiguous bank.

1. `begin_loan({first, count}, relocations)` claims adjacent slots and optional
   survivor destinations. It rejects protected readers, uploading slots, overlap
   or aliased destinations before mutating anything.
2. Register each copy with `begin_relocation(loan, index, work_owner)` before
   enqueueing it, and `settle_relocation` once dependencies physically settle.
3. `activate_loan` publishes survivors at their new slots. Each scratch user
   registers with `acquire_loan(loan, work_owner)`; `extent` gives its offset and
   length. Scratch contents are uninitialized.
4. `begin_return` stops new borrowers; settle existing users, then `reclaim_loan`
   returns empty slots to the cache. Cancellation uses the same path.

The backing reservation stays charged once; reclaim returns logical cache
capacity, not memory to Core. Capture `binding()` with graph addresses and
validate it before each replay; uploads and loan transitions invalidate it. A
binding check grants no lease. `available_capacity()` reports slots not held by
loans, not immediately evictable slots. The [native example](code/loan_example.cpp)
runs real host copies and checks exact cache recovery:

```sh
cmake -S inference -B build/inference
cmake --build build/inference --config Release --target crit_cache_loan_example
ctest --test-dir build/inference -C Release -R crit_cache_loan_example --output-on-failure
```

### Lending owner memory to the cache

The reverse direction uses the same lifecycle. Memory another owner reserved but cannot use right now can
hold cache slots, and the owner reclaims it before writing. Examples are KV positions beyond the longest
request in flight and a prompt workspace between prompts. Build the directory over all slots, its own and
the lent ones. The directory is address-agnostic, and the engine keeps each slot's address. The owner holds a
loan while it uses its memory:

1. Before the owner writes, for example when a request begins and reaches those positions or a prompt pass
   starts, it calls `begin_slot_loan(slots, relocations)`. The slots can be any set and need not be adjacent.
   Their contents are evicted, and protected readers or uploads reject the whole loan, so settle them and
   retry. Optional survivors move to free slots with the same relocation protocol.
2. `activate_loan`. The owner's writers register with `acquire_loan`. A slot loan has no `extent`, because
   the owner addresses its own memory.
3. When the owner is done (the request ends, the prompt pass finishes), call `begin_return`, settle the
   writers, then `reclaim_loan`. The slots come back EMPTY, never with old contents.

Every transition invalidates `binding()`. A device directory that reads a per-slot valid flag on each replay,
as [device-expert-cache](../device-expert-cache/README.md) does, mirrors `on_loan(slot)` into it: it clears the flag and settles
its readers before the owner's first write is enqueued. It sets the flag again only after `reclaim_loan`. The
owner keeps the memory charged once; the cache gains logical capacity, not memory.

### Bundle publication

Run from `code/`. A two-component bundle cannot become READY early, and a reader
prevents replacement even after uploading has finished.

```python
from concurrent.futures import Future
from implementation import (BundleComponent, BundleDirectory, BundleReceipt,
                            ExpertKey)

components = (BundleComponent("weights", "demo-i8", 4),
              BundleComponent("scales", "demo-f32", 4))
directory = BundleDirectory(1, components, backing_owners=(bytearray(8),))
key = ExpertKey("revision-a", 0, 2, "demo-bundle-v1")
ticket = directory.begin_upload(key)
copies = {component.name: Future() for component in components}
def enqueue(active_ticket, component, source):
    assert len(source) == component.nbytes
    return BundleReceipt(active_ticket, component.name, component.encoding,
                         component.nbytes, copies[component.name])
for component in components:
    directory.submit(ticket, component.name, bytes(4), enqueue)

copies["weights"].set_result(None)
try:
    directory.publish(ticket)
except ValueError:
    pass
else:
    raise AssertionError("incomplete bundle became READY")
assert not directory.ready_slots()
copies["scales"].set_result(None)
assert directory.publish(ticket) == (0, 1)
use = directory.acquire(key)
replacement = ExpertKey("revision-a", 0, 3, "demo-bundle-v1")
try:
    directory.begin_upload(replacement)
except ValueError:
    pass
else:
    raise AssertionError("live reader lost its content")
read = Future()
read.set_result(None)
directory.settle(use, read)
next_ticket = directory.begin_upload(replacement)
assert not directory.ready_slots()
directory.abort(next_ticket)  # no copy was submitted; old bytes stay invalid
assert not directory.ready_slots()
directory.close()

# The second host POC sizes banks, stages coalesced component bytes and checks
# exact original-source fallback without allowing prefill to disturb decode ages.
from example import main
main()
```

## Adapting it

Preserve representation identity, complete-bundle visibility, generation-bound
receipts, protected reads and honest physical accounting. LRU-like eviction, slot
counts, component layout and key representation are reference choices. Cache
membership is performance state and must not alter semantic state or expert
selection. A demand-counting policy should count original routed demands, not
substitutes chosen by an approximate consumer, which would reinforce its own
decisions. Rejecting admission in an exact consumer still runs the original expert.

## With other Inference entries

| Implementation | Supplies | Engine supplies |
| --- | --- | --- |
| [mixed_source_plan / execute_mixed](../heterogeneous-expert-dispatch/code/implementation.py) | Occurrence-preserving source selection, exact fallback, one projection hook | Native staging, projection arithmetic, layouts and ordered readiness |
| Core's [GraphCache](../../core/include/crit/device_graph.h), [CudaStagedGraph](../../core/include/crit/cuda_graph.h) | Binding identity, capture/replay, split phases and completion | Capture-safe arithmetic, stable pools and device ordering |
| Core's [MappedSource](../../core/include/crit/device_memory.h), [cuda_map_host](../../core/include/crit/cuda_memory.h) | Full-range coverage, device alias continuity checks and CUDA host registration | Physical accounting and reader drain before unregister |
| [FileExpertSource](../ssd-expert-materialization/code/implementation.py) | Read ownership and failed-upload staging retention | I/O/upload services and settlement |

With [mixed-source dispatch](../heterogeneous-expert-dispatch/README.md), protect
hits before resolving misses and keep an exact fallback. With
[graph replay](../../core/src/backends/cuda/README.md#graph-capture), keep backing
addresses stable while publishing ordered content maps. With
[file materialization](../ssd-expert-materialization/README.md), read-ahead is not
residency until device content is complete. Budget every owner (weights, state,
bundles, aliases, maps, scratch, staging, pinned host memory, headroom), charge
aliases once, and reclaim bytes only after every reader settles.

## Costs and failure modes

A useful cache saves more slow-source work than it adds in lookup, staging,
synchronization and eviction. Hit rate alone is insufficient: weigh miss cost,
bytes, reuse distance, churn and prefill versus decode, and count cold traffic
separately from warm repeats. A larger pool can displace state or scratch. Common
failures: wrong representation keys, pinning hits after choosing victims,
publishing one tensor of a bundle, reusing a stale copy event, wrapping expert IDs
into a smaller slot range, and treating a failed overwrite as valid old content.

## Validation

Host fixtures cover complete publication, duplicate/foreign tickets and uses,
protected hits, pending reads, failed enqueue/copy and abandoned replacements.
The [settled-slot fixture](../../tests/inference/settled_slots_tests.cpp) checks
slot metadata, receipt identity, fragmented and protected regions, overlapping
loans, stale identities, partial relocation failure, cancellation and binding
invalidation over 1,250 loan/return cycles, and reverse loans: scattered slot loans, a live reader blocking a
reclaim until it settles, a survivor relocated out of reclaimed owner memory, empty return and 500 repeated
requests. In one MoE engine that lends unreachable KV positions and its prompt workspace, holding those as slot
loans (one per request, one per prompt pass) and mirroring `on_loan` into its device valid flags gave
token-identical output at unchanged speed. Planner fixtures check admission
boundaries, bank grouping, guards, alignment and copy-range parity; run the bank
example with `python -B code/example.py` from this entry's directory. Validate a
combined engine on all-hit, all-miss, mixed and repeated routes, slot pressure with
failed uploads, captured replay with mapping replacement, and cold and warm caches
against ordinary placement. These checks do not qualify native copies, device
fences, model results or a speedup.
