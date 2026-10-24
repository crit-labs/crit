# Files, JSON and digests

Dependency-free building blocks for reading model artifacts: a strict JSON
reader, read-only files with change detection, chunked and mapped reads, and
SHA-256. They know nothing about model formats; the safetensors and config
readers build on them.

| Header | Source | Ships |
| --- | --- | --- |
| [json.h](../../include/crit/json.h) | [json.cpp](json.cpp) | `FormatError`, `json::Limits`, `json::Kind`, `json::Value`, `json::Member`, `json::Document` |
| [file.h](../../include/crit/file.h) | [file.cpp](file.cpp) | `FileIdentity`, `ReadOptions`, `ReadStats`, `ChunkConsumer`, `ReadOnlyFile`, `display_path` |
| [sha256.h](../../include/crit/sha256.h) | [sha256.cpp](sha256.cpp) | `Sha256`, `to_hex`, `sha256_hex`, `sha256_file` |

Host memory admission lives in [memory](../memory/README.md#host-memory-admission).

## Strict JSON

`json::Document::parse(text, source_name, limits)` and
`json::Document::read_file(path, limits)` accept RFC 8259 JSON and nothing else:

- Strict UTF-8 in keys and values; no byte order mark, comments, trailing commas,
  single quotes or raw control characters. `\u` escapes must pair surrogates.
- `NaN`, `Infinity` and `-Infinity` load only with `Limits::allow_nonfinite`
  (off by default; the model loaders set it for `config.json` and
  `generation_config.json`). Overflowing reals such as `1e400` stay errors.
- Duplicate keys are rejected at every depth, comparing unescaped keys.
- Integers in `[-2^63, 2^64)` stay exact. Larger integers still parse, report
  `exceeds_64_bits()` and throw "integer exceeds 64 bits" from numeric
  accessors. Reals are parsed locale-independently and must be finite.
- Only trailing whitespace may follow the root value.

`Limits` bounds bytes (128 MiB), depth (128) and nodes (2^24 values). The parser
is iterative, so deep nesting fails at the limit instead of exhausting the stack.
`read_file` opens with `ReadOnlyFile`, checks the size before reading and
verifies afterwards that the file did not change.

A `Document` owns an immutable arena; copies share it, and every `Value`, span and
`string_view` stays valid while any copy lives.

| `Value` member | Meaning |
| --- | --- |
| `kind()`, `offset()`, `is_null()` | Kind and source byte offset |
| `boolean()`, `u64()`, `i64()`, `real()`, `string()` | Typed values; a wrong kind or an integer that does not fit throws `FormatError` |
| `array()`, `object()` | Elements, or members in document order |
| `find(key)` | Member value or `nullptr`; logarithmic |

## Read-only files

`ReadOnlyFile::open(path, follow_symlinks = true)` opens a regular file read-only
and captures its `FileIdentity`. With `follow_symlinks = false` a final-component
symbolic link (or reparse point) is refused; a path containing NUL is refused
before any OS call. A relative path is made absolute once at open, so later
working-directory changes do not look like file changes.

On Windows the handle denies writers, renames and deletes while it lives; POSIX
cannot prevent mutation, so `verify_unchanged()` detects it by comparing the
held handle and the path against the identity captured at open (device, inode or
file id, size and change times).

`read_exact(offset, dst)` reads exactly `dst.size()` bytes with a positioned read,
safe from several threads. On Windows, synchronous reads through one handle are
serialised by its file object, so threads sharing it do not scale. `set_readers(n)`
opens `n - 1` private handles reopened from the held one (`ReOpenFile`, the same
file, random-access hint) before concurrent reads begin, and
`read_exact(offset, dst, worker)` reads through slot `worker` (slot 0, and ids past
the slots, use the held handle). POSIX `pread` is concurrent on one descriptor, so
there every slot uses it. The caller chooses the slot count, for example from a
measurement of the volume; Core sets none.

`read_chunks(offset, length, consume, options)` streams a region to `consume` in
file order on the calling thread; each span is valid only during the call, and
`ReadStats` reports bytes, calls and whether direct I/O was used and why not.

- `ReadOptions` defaults: 8 MiB chunks, 4096 alignment, buffered, overlap on.
  `alignment` must be a power of two and `chunk_bytes` a multiple of it.
- Direct I/O uses a second handle (`O_DIRECT` on Linux, `FILE_FLAG_NO_BUFFERING`
  on Windows, `F_NOCACHE` on macOS); a refused open or first read falls back to
  buffered reads with `fallback_reason` set.
- With overlap, one reader thread fills the next chunk while the consumer runs,
  using at most two buffers of `min(chunk_bytes, region)`.

`map()` returns a read-only view of the whole file, created once on first use
after verifying the handle identity. `mapping_owner()` keeps the mapping alive
after the `ReadOnlyFile` is gone. On POSIX, truncating a mapped file raises
`SIGBUS` on access, so pair mappings with `verify_unchanged()` and controlled
storage. `advise_willneed` hints a range and never fails.

## SHA-256

`Sha256` is a portable FIPS 180-4 implementation; `finish()` returns the digest
and resets. `to_hex` is lowercase, equal to Python's `hexdigest()`.
`sha256_file(file, options)` hashes through `read_chunks` (direct I/O by default)
and then calls `verify_unchanged()`, giving the digest of crit run-record
manifest rows (`path`, `bytes`, `sha256`).

## Errors

| Failure | Exception |
| --- | --- |
| Malformed data | `FormatError`: `"<source>: <where>: <reason>"`, with `source()` and `where()` |
| OS failure | `std::system_error` with `errno` or `GetLastError`, message `"<path>: open"`, `stat`, `read` or `map` |
| Refused symbolic link | `std::system_error`, `"<path>: open: symbolic link refused"` |
| Not a regular file, path with NUL, bad options or limits | `std::invalid_argument` |
| Region past the size at open | `std::out_of_range` |
| File shrank or changed | `std::runtime_error` |
| Moved-from `ReadOnlyFile` | `std::logic_error` |

The [native fixture](../../../tests/core/io_tests.cpp) generates every input at run
time and covers JSON rejection, UTF-8 sweeps, fuzzing, NIST SHA-256 vectors,
change detection, the direct/buffered/overlap read matrix and mapping lifetime.
