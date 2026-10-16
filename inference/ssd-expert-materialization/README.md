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
  upload handoff, `quiesce()` and failed-upload retention.
- [mapped_gather.h](code/mapped_gather.h): the C++20 sparse gather above.

No route predictor, native storage queue, pinned staging pool, device copy or cache
manager; no model parsing; overlapping named regions are not rejected.

## What the engine supplies

Descriptors from trusted artifact metadata with the exact encoded extent, an executor for
`prefetch(key, executor)`, and an `upload(encoding, bytes)` hook returning
`(owner, readiness)`. Hold the materialization until asynchronous reads finish and check
destination success before use. Bound queued reads, staging bytes and outstanding
transfers; the reference deduplicates only pending keys, so later demand can reread.
Reserve staging for urgent demand and prioritize it over speculative queues.

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
accuracy. Checksums are O(B) per region and the reference builds a complete `bytes`
object. Mapping size is not resident RAM, and warm results do not establish cold-source
behavior. Common failures: unbounded futures holding large arrays, treating prefetch
completion as device readiness, dropping staging after enqueue failure, and closing
files while producers can still submit.

[Native fixtures](../../tests/inference/inference_io_codec_tests.cpp) check page
boundaries, duplicate and overlapping regions, order, sequential/parallel equivalence,
empty files, bounds and cancellation. Host fixtures cover exact bytes, pending reads,
failures and ambiguous upload recovery. Storage throughput, pinned memory, transfer
overlap and cache profitability remain unmeasured.
