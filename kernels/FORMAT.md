# Kernels records

Kernels reads strict UTF-8 JSON records under 1 MiB; unknown schema fields, duplicate keys and
nonfinite numbers are rejected. Version 1 has no executable expressions or provider imports. A
catalog holds 1–256 top-level implementation JSON files with unique IDs.

## Operation and target

```json
{
  "schema_version": 1,
  "record_kind": "kernels_request",
  "operation": {
    "name": "gemm", "dtype": "float32", "layout": "row-major",
    "m": 7, "n": 11, "k": 16, "transpose_a": false,
    "transpose_b": false, "aliasing": "disjoint"
  },
  "target": {
    "os": "windows", "arch": "x86_64", "backend": "cpu",
    "abi": "cblas-ilp64", "features": ["sse2", "avx2"]
  }
}
```

Each namespace has at most 128 flat facts. Names start with a lowercase letter and have up to 64
lowercase letters, digits or underscores. Values are nonempty strings (≤ 4,096 characters),
finite numbers (magnitude ≤ 2⁶³), booleans, lists of distinct strings (≤ 128 items) or `null`
for unknown. `operation.name` is required. Describe every relevant fact (encoding and scales,
strides, masks, positional convention, accumulation, aliasing, shapes, workspace, architecture,
ABI, runtime versions); nothing is inferred from a model name.

Inspection always checks `operation.name`, `operation.dtype`, `operation.layout`, `target.os`,
`target.arch`, `target.backend` and `target.abi`, plus every requested or constrained fact. Each
needs both a request value and a descriptor rule. A known mismatch outranks unknowns.

## Implementation declaration

| Field | Meaning |
| --- | --- |
| `schema_version`, `record_kind` | `1`, `kernels_implementation` |
| `id` | 1–128 lowercase letters, digits, dots, underscores or hyphens; first character alphanumeric |
| `summary` | What this implementation/build supplies |
| `provenance` | `{repository, revision, variant}` |
| `license` | Reviewed license expression; does not license crit itself |
| `requirements` | Map from namespaced fact to a rule; at most 256 facts |
| `contract` | Nonempty `semantics`, `invocation`, `workspace`, `completion`, `ownership` |
| `artifacts` | Up to 4,096 exact `{path, bytes, sha256, role}` entries |
| `runtime_dependencies` | Explicit runtime prerequisites; empty is allowed but does not prove closure |
| `unverified` | Open applicability questions; any item keeps applicability unknown |

Only exact revisions can be retained as compatible: `git:` plus a full 40- or 64-character
lowercase commit hash, or `sha256:` plus the 64-character digest of a source or binary
distribution. The variant identifies packaging, compiler/ABI and device specialization. Keep
build logs and provider metadata as artifacts where they back these claims; a binary hash does
not recover its source revision.

Rules (one form per fact, all ANDed):

```json
{
  "operation.dtype": {"one_of": ["float32"]},
  "operation.transpose_a": {"one_of": [false, true]},
  "operation.k": {"minimum": 8, "maximum": 65536, "multiple_of": 8},
  "target.features": {"contains": ["avx2"]},
  "operation.alpha": {"any_value": true}
}
```

`one_of` compares types as well as values (`true` ≠ `1`, `16` ≠ `16.0`). Bounds are inclusive,
any subset of `minimum`, `maximum`, `multiple_of`; `multiple_of` is a positive integer and needs
an integer value. `contains` needs a string list with every listed item. `any_value: true`
accepts every known value; justify it in the contract. Missing or null values stay unknown.

Artifact paths are canonical forward-slash paths relative to the supplied artifact root.
Absolute paths, traversal, duplicate or case-aliased names, symlinks, reparse traversal,
file/directory collisions, disagreeing parent spellings and nonportable Windows names (reserved
device names, trailing spaces or dots) are rejected. Byte counts and lowercase SHA-256 digests
come from the files, for example via `crit kernels inventory`. Each row has one role: `source`,
`header`, `binary`, `runtime`, `notice`, `build`, `metadata` or `test`. Retention needs at least
one source or binary and one notice; all selected entries are copied.

The invocation contract names exported symbols, integer widths, calling convention and launch
arguments, who allocates scratch and outputs, which stream orders work, what proves completion
and how long inputs and workspace must live.

## Assessments and retained dependencies

`inspect` returns a `kernels_assessment` with the request digest and per implementation its
descriptor digest, provenance, license, actual/required values per fact, reasons, open questions
and one of `declared_compatible`, `rejected`, `unknown`. Nothing is loaded, and artifact hashes
are only declarations until retention reads the bytes.

```text
request.json       exact operation and target
catalog.json       all compared declarations (rejected artifacts are not copied)
assessment.json    reproducible decisions
artifacts/         selected source/binaries/notices and other declared files
dependency.json    selected ID, rationale, input digests and full file inventory
```

`dependency.json` is written last; a directory's existence alone does not establish validity.
`verify` rejects missing, added or changed files, recomputes all decisions, checks the selected
bytes against the declaration and returns the record digest for the candidate's sealed
inventory. An unsealed directory can be rewritten, so self-consistency is not proof of origin.
Keep these records and builds in the run's `work/` area until copied into candidate inventories
([Building engines](../docs/BUILDING_ENGINES.md)).
