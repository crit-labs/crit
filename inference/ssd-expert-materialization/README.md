# File-backed expert materialization and read-ahead

Encoded expert weights stay in an immutable file; demand reads selected regions with
length and digest checks and optionally uploads them, and read-ahead can overlap file
latency with other work. It works with ordinary files; the name does not assume a
storage device or promise SSD bandwidth.

## When to use it

Use it when experts are served from a file rather than held in memory. Choose read-ahead
only when predictions arrive early enough and do not overload I/O, staging memory or the
transfer path. Deciding which experts stay resident is a separate concern.

## How it works

An expert key selects an explicit byte offset, extent, encoding and digest. Every demand
read checks length and digest. Read-ahead submits that same checked read to a
caller-owned executor and deduplicates outstanding work per key. Materialization
consumes the pending result or reads synchronously, then passes the bytes and encoding to
an upload hook. The result retains source bytes, destination owner and readiness; upload
completion is distinct from read completion. If enqueue throws after possibly starting a
transfer, the source bytes and upload callable are retained for explicit recovery.

**Host tier.** When host memory holds only part of the experts, give the source a
`HostExpertTier(capacity)`: a bounded host copy between the file and the device cache, least
recently used first out. A demand read is served from it when it holds the expert, and an expert
read from the file is held there. When the device cache evicts an expert, copy its slot to host
memory and `retain(key, bytes, protected)` it instead of dropping it: the next miss then crosses
the host link instead of rereading the file. Retained bytes are checked like a file read. An
expert the current route selects (`protected`) is never evicted to make room, and an expert that
cannot fit is refused (`False`), never half held. The capacity is part of the engine's
[host-memory admission](../host-memory-admission/README.md), not memory beside it.

**Native gather.** `MappedFile(path, maximum_bytes)` maps the file through
`crit::core::ReadOnlyFile` ([Core files](../../core/src/io/README.md#read-only-files));
`file()` exposes it for `identity()` and `verify_unchanged()`.
`gather(regions, limits, cancellation)` returns private concatenated bytes in request
order, keeping duplicates. It merges overlapping or adjacent intervals, advises them with
Core's `advise_willneed` (a hint, never counted as completed I/O), then copies with one
worker or a bounded set for larger outputs, joining all workers before any failure
escapes. Cancellation is checked between regions and at 1 MiB boundaries.
`GatherLimits{output_bytes, regions, workers, parallel_threshold}` bounds each call
(1-64 workers, threshold in output bytes).

## What the code supplies

- [implementation.py](code/implementation.py): `ExpertRegion`, `FileExpertSource` and
  `Materialization` with checked demand reads, executor read-ahead, pending-read reuse,
  upload handoff, `quiesce()` and failed-upload retention; `HostExpertTier`, the bounded host
  tier with `retain()` for device evictions and hit, miss, eviction and refusal counts.
- [mapped_gather.h](code/mapped_gather.h): the C++20 sparse gather above.
- [host_tier.h](code/host_tier.h): the native host tier and its read queue. `HostTier` is a
  ledger over pinned host slots the engine owns, one bundle each: `find` serves a committed hit,
  `admit(key, protect)` hands out a free slot or the least recently used unprotected one (or
  refuses), `commit` publishes a slot once its fill landed and `abandon` frees a failed one; it
  counts hits, misses, admissions, evictions and refusals. `ReadQueue` fills such slots from the
  file with a pool of positioned readers (Core's `ReadOnlyFile` read slots): demand reads before
  read-ahead, one read per pending key (a demand for a queued read-ahead moves it first), and a
  `Ticket` per read to wait on before `commit`.
- [bundle_file.py](code/bundle_file.py): `FileBundleStore`, the plugin's bundle store (`open_store`, the file tier
  of "experts:store"): every (layer, expert) bundle in one file in key order, written by `put()` (or a complete file
  with the same layout reopened, its index beside it), read by `read(keys, into)` through lookup-table-rows' native
  row reader (the bundles are its rows; positioned reads in Python without it) behind a `HostExpertTier` of
  `host_bytes`. It has no device-readable region: device-expert-cache's `ExpertCache` fills its slots from `read()`
  into pinned memory on the host, so its plans over this store synchronize and are not captured; a mapped store
  (expert-tensors' `open_store`) serves captured rounds.

No route predictor and no device copy: the engine copies a committed slot to the device cache
(the slot is pinned, so a hit uploads directly) and copies an evicted device slot into an
admitted one before committing it. No model parsing; overlapping named regions are not
rejected.

## What the engine supplies

Descriptors from trusted artifact metadata with the exact encoded extent, an executor for
`prefetch(key, executor)`, and an `upload(encoding, bytes)` hook returning
`(owner, readiness)`. Hold the materialization until asynchronous reads finish and check
destination success before use. Bound queued reads, staging bytes and outstanding
transfers; without a host tier the reference deduplicates only pending keys, so later demand
rereads the file. Reserve staging for urgent demand and prioritize it over speculative queues.
For device evictions into the host tier, enqueue the device-to-host copy of the evicted slot
before anything overwrites it, retain only after that copy completed, and pass the experts the
current token routes as `protected`. Hold the native tier in pinned host memory so a hit uploads
directly.

Native mapping needs a verified immutable artifact. Windows denies write/delete sharing
while open; POSIX callers must prevent mutation and can detect it with
`file().verify_unchanged()`. Mapping computes no checksums and OS mapping faults are not
recoverable C++ errors, so use the checked Python source or an integrity service when
per-read digests are required. A native backend may also use resident host ranges or
aligned direct reads; direct I/O must reserve the rounded extent, validate the complete
read and expose only the requested bytes, with checked ordinary reads as fallback.

On shutdown, stop new callers before `quiesce()`, which waits for pending reads, closes
the source and reports the first read error. It does not drain uploads or stop the
executor. `recover_uploads(quiesce_uploads)` releases failed-upload staging only after
the callback returns `True` as physical quiescence attestation.

## Worked example

Run from `code/`. A prefetched read is reused, while upload readiness stays a separate
event.

```python
from concurrent.futures import Future, ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from implementation import ExpertRegion, FileExpertSource

with TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=1) as executor:
    path = Path(directory) / "weights.bin"
    path.write_bytes(b"ABCDEFGH")
    region = ExpertRegion(0, 4, "demo-i8", sha256(b"ABCD").hexdigest())
    source = FileExpertSource(path, {"expert-a": region})
    first = source.prefetch("expert-a", executor)
    assert source.prefetch("expert-a", executor) is first
    copied = Future()
    def upload(encoding, data):
        assert encoding == "demo-i8" and data == b"ABCD"
        return bytearray(data), copied
    materialized = source.materialize("expert-a", upload)
    assert materialized.encoded_source == b"ABCD"
    assert not materialized.readiness.done()
    assert not source.pending
    source.quiesce()  # file reads are done; upload readiness is still separate
    assert not copied.done()
    copied.set_result(None)
    assert bytes(materialized.owner) == b"ABCD"
```

## Adapting it

Preserve representation identity, checked complete reads, source retention through
asynchronous transfer, and separate read and upload settlement. File API, checksum
strategy, staging allocator, prediction policy and queues may change. With
[expert residency](../expert-residency/README.md), a successful read is not a READY
device bundle until all components upload. With
[mixed-source dispatch](../heterogeneous-expert-dispatch/README.md), read-ahead may
prepare likely misses, but speculation must never change which expert executes.

## Costs and validation

Read-ahead pays off only if hidden latency repays wasted reads and staging; track useful
and unused prefetched bytes, demand stall time and peak staging, not just prediction
accuracy. A host tier costs its capacity in host memory and one device-to-host copy per
eviction, which overlaps the main stream; it pays when misses would otherwise come from the
file, since the host link is usually several times faster than storage. Track tier hits, bytes
read from the file per token and refusals. For scale only: one engine serving a mixture-of-experts
model larger than its memory from NVMe on a 12 GB GPU with 16 GB of RAM read 44.7 and 39.7 MiB
per generated token in two repeated requests without its 8 GiB tier and 35.4 and 12.4 MiB with
it (13.9 and 14.3 against 16.3 and 22.0 tok/s); another engine measured fewer misses with
identical tokens. Neither is a crit measurement. Checksums are O(B) per region and the reference builds a complete `bytes`
object. Mapping size is not resident RAM, and warm results do not establish cold-source
behavior. Common failures: unbounded futures holding large arrays, treating prefetch
completion as device readiness, dropping staging after enqueue failure, and closing
files while producers can still submit.

[Native fixtures](../../tests/inference/inference_io_codec_tests.cpp) check page
boundaries, duplicate and overlapping regions, order, sequential/parallel equivalence,
empty files, bounds and cancellation. Host fixtures cover exact bytes, pending reads,
failures, ambiguous upload recovery, host-tier hits after the file changed, checked retention
of evicted experts, and that the current route is never evicted. Storage throughput, pinned memory, transfer
overlap and cache profitability remain unmeasured.
