# Validated tensor-region loading

Load exact encoded byte regions from an immutable artifact, check them against an
admitted manifest, and retain both ends of an asynchronous materialization. The format
parser and tensor arithmetic belong to the engine.

## When to use it

Use it when an artifact holds separately addressable tensors or components and you want
to load only admitted regions, before placement, lazy loading or upload. An existing
verified immutable reader can supply the same guarantees without a second read-and-hash
pass per operation.

## How it works

`TensorRegion` binds a name to `(offset, nbytes, shape, encoding, sha256)`. Offset and
size describe storage, shape and encoding describe interpretation, and neither is
derived from the other: padding, packed blocks and scales make
`product(shape) * element_size` an invalid extent calculation.

1. Check file size, unique names, nonnegative extents, positive dimensions, an encoding
   and canonical digests.
2. Sort the half-open ranges and reject overlap (aliases need their own policy).
3. `read(name)` reads exactly `nbytes` and rejects truncation or changed content by
   digest.
4. `materialize` validates bytes, allocates the destination, invokes the copy service
   and returns destination, source bytes and readiness together.

The digest at each read binds the returned bytes; the size check does not make a mutable
file immutable, and the manifest's authenticity is the caller's concern.

## What the code supplies

[implementation.py](code/implementation.py): `TensorRegion`, `ArtifactReader` and
`Materialization`, with bounded reads and SHA-256 checks into Python bytes. It does not
parse formats, infer strides, map files, allocate device memory or upload. Empty regions
are admitted.

## Native region reads: Core's chunked reader

Native engines use Core's files library ([io](../../core/src/io/README.md#read-only-files)).
`crit::core::ReadOnlyFile::open(path).read_chunks(offset, length, consume, options)`
calls `consume(bytes, file_offset)` per fixed chunk in file order on the calling thread
while the next chunk is read on another thread. The consumer can hash
(`crit::core::Sha256`), copy to pinned memory or submit an upload; a chunk span is valid
only during the call. `ReadOptions::direct = true` reads unbuffered (`O_DIRECT`,
`FILE_FLAG_NO_BUFFERING`) through aligned buffers so large weights do not evict the page
cache; file systems that refuse fall back to buffered reads with
`ReadStats::fallback_reason`. A region past the end throws `std::out_of_range`, and
`verify_unchanged()` rejects a file replaced since open. Choose a chunk of a few MiB that
is a multiple of the alignment, and pair large staging with
[host memory admission](../host-memory-admission/README.md).

For safetensors, `crit::core::Checkpoint` ([model](../../core/src/model/README.md#tensor-bytes))
supplies descriptors, per-tensor `read_chunks`, per-shard `stream_shard` and file
digests. Manifest digest checks and representation metadata stay with `ArtifactReader`
and the engine's descriptors.

## What the engine supplies

A format parser, representation-complete descriptors and these hooks:

| Hook | Obligation |
| --- | --- |
| `allocate(nbytes)` | Return a destination owner after acquiring the byte reservation; clean up partial allocation failures |
| `copy(owner, payload)` | Submit the exact copy and return readiness for that physical operation |
| `retain_failed((owner, payload))` | Retain both endpoints if copy submission throws after possibly starting work |

Keep `Materialization` alive through completion and check success before use. For
repeated engine switches, keep model content independent of a session;
[crit serve's resident host](../../docs/SERVING.md#model-access-and-allocation) verifies
a private read-only snapshot once and rejects source changes before reuse. A display
name, read-only descriptor or unchanged path alone does not establish immutable bytes.

## Worked example

Run from `code/`. A host future stands in for an outstanding copy.

```python
from concurrent.futures import Future
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from implementation import ArtifactReader, TensorRegion

with TemporaryDirectory() as directory:
    path = Path(directory) / "weights.bin"
    path.write_bytes(b"headABCDtail")
    region = TensorRegion("projection", 4, 4, (2, 2), "u8-demo",
                          sha256(b"ABCD").hexdigest())
    reader = ArtifactReader(path, [region])
    copied = Future()
    failed = []
    def copy(destination, payload):
        destination[:] = payload
        return copied
    material = reader.materialize("projection", bytearray, copy, failed.append)
    assert material.source_bytes == b"ABCD"
    assert not material.readiness.done()
    copied.set_result(None)
    material.readiness.result()
    assert bytes(material.owner) == b"ABCD"
    path.write_bytes(b"headWXYZtail")
    try:
        reader.read("projection")
    except ValueError:
        pass
    else:
        raise AssertionError("changed bytes were admitted")
```

## Adapting it

Keep interpretation attached to exact bytes, keep staging and destination owned until
all physical readers settle, and publish only successfully copied content. Streaming,
mapped, aligned or batched reads may replace the whole-region read if they preserve the
admitted subrange, digest and lifetimes; a content cache can amortize verification with
an explicit identity and invalidation rule.
[File-backed expert materialization](../ssd-expert-materialization/README.md) adds
demand and read-ahead, and [quantized projections](../quantized-projections/README.md)
interpret a representation; neither may guess format from size. Package provenance
belongs to the [construction coordinator](../../docs/RUN_RECORDS.md).

## Costs and validation

A `B`-byte region costs `O(B)` read and hash work and `B` retained host bytes plus the
destination; repeated verified reads can dominate small requests. Reject bad extents and
checksums before allocation and budget staging as well as final storage.

Core's [files fixture](../../tests/core/io_tests.cpp) covers direct and buffered chunked
reads, unaligned regions, ordering, past-end reads and consumer exceptions.
[Host fixtures](../../tests/test_techniques.py) cover extents, overlap, changed bytes and
retained endpoints after successful or failed copies. Native transfer, format
interpretation and model fidelity are not covered.
