# Audit model imports and requested capabilities

Check a Safetensors checkpoint's metadata against an engine's exact tensor roles before
loading any payload. A converted, renamed, fused or sharded checkpoint can contain many
plausible names and still omit a requested MTP projection or pair packed weights with
the wrong scale geometry; counts and a descriptive BF16 label cannot prove the stored
dtype. The audit reads headers through Core's strict reader
(`crit_core.model.inspect_safetensors` and `inspect_model`,
[Core model files](../../core/python/crit_core/model/README.md)) and never loads tensor
values.

## When to use it

Use it before an engine loads tensor payloads, for single files and strict flat shard
indexes. Use an already qualified importer instead when it supplies equivalent checks.
GGUF, pickle/PyTorch files, remote URLs and sub-byte dtypes are unsupported; FP8 support
covers only storage size and dtype/shape checks. Architecture configuration, tokenizer
semantics, codecs and inference remain engine work.

## How it works

**Container.** The audit calls Core's reader with `index_policy="strict"`,
`follow_symlinks=False` and its own `Limits` (native engines get the same reader as
`crit::core::Checkpoint` with `IndexPolicy::strict` and `follow_symlinks = false`,
[crit/safetensors.h](../../core/include/crit/safetensors.h)). The reader parses the
bounded header JSON (duplicate fields and non-integer numbers are errors), computes
each tensor's storage bytes from its actual `dtype`, `shape` and `data_offsets`, and
requires ordered, nonoverlapping, exact coverage of the payload with no truncation.
`__metadata__` must map strings to strings and never overrides a dtype. This follows
the public [Safetensors container layout](https://github.com/safetensors/safetensors#format).

The audit then accepts only `BOOL`, `U8`-`U64`, `I8`-`I64`, `F8_E4M3`, `F8_E5M2`,
`F16`, `BF16`, `F32` and `F64` (names and widths from the public
[storage dtype definitions](https://github.com/safetensors/safetensors/blob/main/safetensors/src/tensor.rs)),
rejecting `F8_E8M0` and the F4/F6 formats Core itself accepts. It cannot detect NaNs,
bad scales, a wrong codebook or a dishonest storage label in the payload.

**Indexes.** A `.safetensors.index.json` may hold only `weight_map` and optional
`metadata.total_size`. Every referenced header is read; the union of header tensors
must equal the index keys, each in its declared shard with no duplicates, and
`total_size` must equal the summed extents. Shard names must be portable flat ASCII
`.safetensors` filenames; paths, subdirectories, drive/stream names, reserved Windows
device names and static symlinks are rejected.

**Roles and capabilities.** The engine supplies `TensorRole(role, key, shape, dtype)`
requirements and `audit_model` matches each against the actual headers; missing,
misnamed, wrong-shape or wrong-dtype tensors fail. Several roles may bind the same
tensor (tied weights), but duplicate role names fail. `reject_unexpected` requires the
artifact to contain only these keys. A `CapabilityRequirement` lists the roles a
feature needs and reports `metadata_ready`, or `not_requested` when disabled; a
disabled feature's roles are still required, so omit optional roles from the schema.

**Packed matrices.** `PackedMatrix` declares a codec identity, a matrix prefix (data
and scale keys must start with it plus a dot), data/scale roles, rows, full logical K,
group width, bits and the local slice `[k_begin,k_end)`. It covers 4/8-bit codes packed
into `U8` or `I32` words with `F16`/`BF16`/`F32`/`F64` group scales. With
`P = storage_word_bits / bits` and `K = k_end - k_begin`:

```text
packed_data.shape = (rows, ceil(K / P))
group_scales.shape = (rows, ceil(K / group_size))
k_begin aligns to both group_size and P
k_end aligns to both, unless it is the final full_k boundary
```

This rejects full-size scales paired with local packed weights. It applies to repacked
contiguous local tensors, not strided views. Code order, signedness, zero points and
kernel layouts need the codec's own validation.

## What the code supplies

[implementation.py](code/implementation.py): `inspect_safetensors`, `inspect_model`
(Core errors become `AuditError`), `ModelSchema`, `audit_model` and the geometry check.
`TensorInfo`, `ShardInfo` and `ModelInventory` are Core's records. `Limits` sets audit
defaults: 1 MiB headers/indexes, 8192 tensors, 32 shards, rank 8, dimensions below 2^31,
64 GiB per file, 512 GiB total and 1024 UTF-8 bytes per name; raise them consciously.

`probe_proposer` calls a supplied synchronous proposer once after the requested
capability passes, with up to 4096 prompt tokens, a declared vocabulary and 1-64
maximum output tokens, and checks the returned length and token bounds. It
distinguishes proposals from an explicit empty result; exceptions propagate. It is not
a trained head, verification, state repair or timeout.

[example.py](code/example.py) builds a tiny zero-valued container, checks MTP roles and
a local packed-scale slice, and observes the toy proposer.

## What the engine supplies

A role schema from a trusted architecture/config adapter covering every projection,
norm, embedding, routing and state role of the chosen path, with explicit
source-to-runtime names when fusing or renaming. Core's reader checks file identity
(device, inode, size, mtime, ctime) around the header read and hashes header/index bytes
only, which is not payload identity and not an adversarial sandbox; freeze and
revalidate the source before loading, and use
[artifact loading](../artifact-loading/README.md) for checked region reads. A trained
proposer must share the target tokenizer; verification and rejection correction belong
to [speculative execution](../mtp-decode/README.md), and every private mutable state
component must be repaired, not just a KV count. Metadata success is not a reservation,
a readiness receipt or a quality qualification.

## Worked example

Run from `code/`; temporary tensor files are removed on exit. The example expects an
actual BF16 MTP projection, checks that labeling it F16 fails, and observes one bounded
toy proposal.

```python
from example import run

run()
```

## Adapting it

Preserve exact name/shape/storage-dtype matching and the relationship between local K
words, scale groups and matrix namespace. New dtypes, nested indexes, codebooks or
layouts need explicit validation. A file that parses is not runtime-compatible:
container, architecture and encoding support are separate, and a schema can be wrong
even when every check passes. Record the codec, source identity and adaptation with the
engine candidate, and keep model-specific names in the engine.

## Costs and validation

Only bounded headers and indexes are read: O(B + T) metadata memory and O(T log T)
sorting for T tensors and B metadata bytes. Malformed or duplicate JSON, unsupported
dtypes, bad offsets, holes, overlap, truncation, contradictory indexes, missing roles
and packed-slice errors raise `AuditError`; filesystem and hook errors propagate.

[Host fixtures](../../tests/test_model_import_audit.py) cover actual versus descriptive
dtypes, FP8 extents, missing or misnamed MTP roles, capability status, duplicate JSON
fields, extent and shape failures, scalars and empty tensors, malformed or escaping
indexes and local slice boundaries. Core's [reader tests](../../tests/test_core_model.py)
show the reader never touches payload and detects identity changes during the read. Run
from the repository root:

```text
python -B -m unittest discover -s tests -p test_model_import_audit.py -v
```

FP8 decoding, converted-model arithmetic, trained MTP quality and engine speed are not
covered.
