# Representation-bound prefix checkpoints

Store opaque serialized state under a compatible identity and exact input prefix,
then find the longest reusable prefix without rerunning its computation. Around
that store the entry adds complete component bundles with private restoration,
paced transfers, residency tiers, prefix/layout coordination, cost-based source
selection and live prefetch.

## When to use it

Use this when repeated requests share meaningful prefixes and restoring state costs
less than recomputing it. The engine needs a deterministic prefix encoding and a
state codec. Unique workloads or expensive-to-restore state may do better with
ordinary recomputation.

## How it works

A record binds a state fingerprint, a codec identifier and exact prefix bytes; its
payload is opaque. The fingerprint must cover everything that changes restored-state
meaning: weights, state representation, position rules and other engine-specific
interpretation. This module cannot infer it.

`CheckpointStore` uses one fingerprint/codec per directory. `put` writes a
versioned envelope to a temporary file, fsyncs it and atomically replaces the
target, within a storage budget:

```text
header: 8-byte magic, three little-endian uint32 lengths, SHA-256 of body
body:   JSON metadata | prefix bytes | opaque encoded state
```

A repeated put for the same identity/prefix must carry the same payload; records
are immutable. `exact(prefix)` verifies one record. `longest_prefix(query)`
validates every record, picks the longest byte prefix of `query`, and raises on a
corrupt or incompatible record rather than skipping it.

**Bundles.** `BundleSpec(fingerprint, codec, components, max_payload_bytes)`
freezes the exact component set (`ComponentSpec`: name, codec, byte bound).
`encode_bundle` writes a manifest with each component's extent and SHA-256;
`decode_bundle` checks the exact set, bounds, gaps/overlap and digests before any
allocation. `RestoreTransaction(plan).submit(allocate, enqueue)` allocates every
destination privately, then copies. Callbacks must register owners with
`operation.retain`, readers with `operation.track_work`, and the copy receipt with
`operation.record_copy` **before** a possibly throwing submission. A receipt's
`done()` must prove physical settlement (a host future that merely dispatched a
CUDA launch is not one). `publish()` returns one `PublishedBundle` only after all
copies succeed; no component is published alone. Ambiguous failures keep the
**whole** owner set retained (at most 16 transactions) until `discard(settle)`
receives an explicit settlement proof. `ActiveWriteLease` keeps a split storage
source pinned until its own write settles.

**Transfers.** `BoundedTransfer(parts, limits, read, write, owners=...)` moves
`TransferPart(name, nbytes, sha256)` components through staging windows bounded by
`TransferLimits(window_bytes, staging_bytes, windows, total_bytes)`. Call
`pump(max_bytes=..., max_submissions=...)` between compute steps; **both read and
write bytes count** against the quota, and it returns bytes submitted, not
completed. Callbacks record receipts with `window.record` before submission;
successful `result()` must return exactly `window.nbytes`. Digest mismatch or a
short transfer fails the batch. `read_file_window`/`write_file_window` are file
adapters; durable commit (flush, fsync, atomic rename) stays with the caller.

**Layout and prefix coordination.** `plan_state_layout` gives each `StateLayout`
(page bytes, page count, power-of-two alignment) its own aligned region and checks
the total against a budget. `common_prefix` finds the greatest boundary every
`PrefixFrontier` accepts, combining alignment quanta by least common multiple and
intersecting exact checkpoint lists until stable. Taking the minimum of each
component's longest hit once is wrong for recurrent state that cannot restore
every shorter prefix. Zero means recompute.

**Tiers.** `MemoryTier` holds complete immutable blobs under a byte capacity.
`TierLease` pins one generation; eviction hides it but keeps it charged until the
last lease closes. `TierMove` copies through `BoundedTransfer` and
`publish(evict_source=True)` installs the destination only if both generations
are unchanged; a stale generation means discard and replan.

**Source selection.** `choose_prefix` gates each `PrefixSource` on exact prefix,
fingerprint, codec, complete component set and readiness, then compares restore
cost (`bytes/bandwidth + batches*latency` per declared edge, no assumed overlap)
plus remaining prefill against recomputation. For `s` tokens after frontier `b`,
prefill is `ceil(s/chunk) * call + s * token + (b*s + s*(s+1)/2) * attention_pair`.
Ties keep recomputation. Coefficients must come from calibration.
`plan_captures` suggests a bounded set of capture points (semantic hints, prompt
tail, geometric history ladder) that avoid indivisible regions such as image spans.

## What the code supplies

| File | Supplies |
| --- | --- |
| [implementation.py](code/implementation.py) | Envelope, checksum, single-writer publication, budget, exact/longest-prefix lookup |
| [bundles.py](code/bundles.py), [bundle_example.py](code/bundle_example.py) | Component inventory, private restore transaction, write leases |
| [transfers.py](code/transfers.py) | Bounded staged transport and file adapters |
| [coordination.py](code/coordination.py) | `plan_state_layout`, `common_prefix` |
| [tiers.py](code/tiers.py) | `MemoryTier`, `TierLease`, `TierMove` with real host copies |
| [planning.py](code/planning.py) | `choose_prefix`, `plan_captures` |
| [prefetch.py](code/prefetch.py) | `PrefetchController`, `tier_prefetch`, `tier_prefix` |

No code interprets KV or recurrent state, copies on a device, evicts
automatically or coordinates writers. Planning results are not leases.

## What the engine supplies

Serialize only settled, committed state, in a dedicated directory per
representation identity, with a codec that keeps everything needed to continue.
Enumerate **all** continuation components (attention KV, recurrent matrix,
convolution windows, target/draft state, and any other carry) and bind codec
versions, weights, conditioning, position interpretation, topology and dtypes in
the fingerprint. Bind weights to verified content or a package seal; matching
names and shapes cannot detect rewritten values.

Choose a prefix encoding whose byte-prefix relation is a valid semantic prefix,
such as fixed-width token IDs written at token boundaries. Arbitrary text prefixes
do not establish identical tokenization. For rendered conversations compare the
next request's actual canonical input with the captured input: a generated reply
can render differently next turn. Include non-token conditioning (media,
preprocessing, positions) in the input or fingerprint. After lookup, validate the
restored position and consume only the unmatched suffix.

Capture only where all continuation state is recoverable; recurrent state cannot
be sliced out of a later checkpoint. Restore into separately reserved storage and
establish readiness before publishing to a live session. After choosing a source,
revalidate and acquire a complete lease, and keep it through all physical copies.

An engine serving one session at a time can apply the same rule without
serialization: keep live state when the next request extends the processed tokens,
keep one in-memory recurrent capture at the end of each prompt for when the
rendered reply differs, and reset otherwise. A failed or interrupted forward must
invalidate the live state. With Core, report the kept prefix through
`Progress::reused_prompt_tokens`.

## Worked example

Run from `code/`. Tokens use a fixed-width encoding; state is deliberately opaque.

```python
import struct
from tempfile import TemporaryDirectory
from implementation import CheckpointStore

def prefix(*tokens):
    return b"".join(struct.pack("<I", token) for token in tokens)

with TemporaryDirectory() as directory:
    store = CheckpointStore(directory, "weights-r1/state-r2", "demo-v1", 4096)
    store.put(prefix(7), b"state-after-one")
    store.put(prefix(7, 9), b"state-after-two")
    query = prefix(7, 9, 11)
    found = store.longest_prefix(query)
    assert found.prefix == prefix(7, 9)
    assert found.payload == b"state-after-two"
    assert query[len(found.prefix):] == prefix(11)
    assert store.longest_prefix(prefix(8)) is None
    try:
        store.put(prefix(7, 9), b"conflicting-state")
    except ValueError:
        pass
    else:
        raise AssertionError("identity collision was admitted")

from bundle_example import main
main()  # complete opaque component inventory and private host restoration

from planning import PrefillCost, PrefixSource, TransferCost, choose_prefix
query = tuple(range(100))
resident = PrefixSource("device", query[:60], "weights+conditioning", "codec", {"kv", "recurrent"})
host = PrefixSource("host", query[:90], "weights+conditioning", "codec", {"kv", "recurrent"},
                    (TransferCost(1000000, 1000000),))
plan = choose_prefix(query, (host, resident), PrefillCost(16, .001, .0001),
                     fingerprint="weights+conditioning", codec="codec", components={"kv", "recurrent"})
assert plan.source is resident  # longer prefix is too costly to transfer
assert plan.predicted_seconds < plan.scratch_seconds

from coordination import StateLayout, PrefixFrontier, plan_state_layout, common_prefix
regions, total = plan_state_layout((StateLayout("kv", 33, 3, 16),
                                    StateLayout("recurrent", 7, 2, 8)), 160)
assert total == 160 and regions[0].stride == 48
frontiers = (PrefixFrontier("kv", "state-v1", 100, 4),
             PrefixFrontier("recurrent", "state-v1", 100, checkpoints=(16, 30, 70)))
assert common_prefix(frontiers, 100, identity="state-v1") == 16

import transfers
from tiers import MemoryTier, TierMove
hot, cold = MemoryTier(32), MemoryTier(32)
hot.put("prefix", "state-v1", b"complete state")
reader = hot.acquire("prefix", "state-v1")
move = TierMove(hot, cold, "prefix", "state-v1",
                transfers.TransferLimits(4, 8, 2, 32), transport=transfers)
while not move.transfer.done():
    move.transfer.pump(max_bytes=8, max_submissions=2)
move.publish(evict_source=True)
assert hot.used == len(reader.data)  # Eviction cannot invalidate a live reader.
reader.close()
assert hot.used == 0
copy = cold.acquire("prefix", "state-v1")
assert copy.data == b"complete state"
copy.close()
```

## Live queued prefetch

`PrefetchController` drives bounded transfers while a request waits: at most
sixteen prepared jobs, paced round-robin under aggregate byte and operation
quotas. Each decision compares estimated remaining transfer plus prefill with
`choose_prefix`'s ready alternatives and recomputation. It waits only for a
compatible complete-state prefix with a known beneficial estimate inside its
monotonic deadline; otherwise it takes a ready shorter prefix or recomputes.
Returned reuse decisions own leases the request must close after restoration.

Use `start(key, prepare, deadline=...)`, `pump`, `decide` and `cancel`; call
`collect` until abandoned jobs with pending readers are gone.

```sh
python -B inference/checkpoint-storage/code/prefetch_example.py
```

## Adapting it

Compatibility is prefix bytes plus codec plus fingerprint. Retain only committed
state; roll back speculative suffixes first. Verify both the envelope and the
restored semantic position. An index, trie, chunked payload, compression or
eviction policy can replace the linear scan; a multi-writer store needs locking or
conditional publication, since `os.replace` alone does not protect the budget.

A device retained-state tier is additional engine work: keep a chosen source
leased through restore, charge shared storage once per physical owner, and compare
restore and spill costs with avoided computation and with the best cheaper-tier
prefix rather than taking the longest prefix. [Attention state](../attention-state/README.md)
supplies a logical commit boundary and [session scheduling](../session-scheduling/README.md)
the matching input cursor; a checkpoint must describe the same committed prefix in both.

## Costs and validation

Longest-prefix lookup reads and hashes all `B` stored bytes; bundle and tier
digests are linear in payload. Restore adds decompression, allocation and
transfers. A `RestorePlan` keeps the whole encoded payload; staging bounds do not
change that. Tier publication temporarily needs another destination-sized
allocation. Quarantined transactions and transfers keep complete payloads and
destinations: include them in failure-path memory planning. The code does not
fsync the parent directory, so crash durability is not established on every
filesystem.

Host fixtures in `tests/` (`test_techniques.py`, `test_checkpoint_bundles.py`,
`test_checkpoint_transfers.py`, `test_portable_mechanisms.py`,
`test_prefetch_controller.py`) cover corruption, incompatible identities,
capacity, all-or-nothing publication, retained owners, quotas, out-of-order
completion, short I/O, digest mismatch, 400 prefix intersections, tier generation
races and prefetch fallback. They do not qualify native DMA, crash recovery or
model continuation. An engine codec should compare continuation from restored
state with recomputation, including boundary and truncated records, rendered
history changes, conditioning changes, failed restores and tier eviction.
