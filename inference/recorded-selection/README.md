# Record implementation choices instead of timing them at load

When an engine picks among interchangeable implementations of one operation (kernel
variants, tile shapes, a math library's algorithms for a shape), tune once, record the
choices in the frozen configuration and replay them. Picking by timing at load resolves
near-ties differently from process to process; different implementations round
differently, so with greedy decoding two processes of the same package generate
different text, and a single-process repeat check cannot see it.

## When to use it

Use it whenever several candidates exist for an operation. It is unnecessary when every
candidate is bitwise identical for the engine's inputs or only one is ever used.
Physical batch limits are candidates too: the
[speculative batch example](../mtp-decode/README.md#physical-batch-size-is-independent-of-proposal-width)
records a draft cap from timed synthetic runs and replays it. Bind such caps and all
layout, model, backend and shape constraints into the selection identity, and validate
legality and workspace admission at load; a recorded index proves neither.

## How it works

**Key.** A choice belongs to an operation shape (`operation`, fixed dimensions) and a
width bucket: the smallest power of two holding the variable width, clamped to a minimum
and cap. Buckets keep the table small and cover untuned widths.

**Candidates.** The table stores an index into a stable candidate list (a library's
heuristic list for the bucket width, or the engine's own variants). Record the library
version and device beside the table and retune when either changes.

**Replay.** A recorded index is used as is; an index beyond the list is an error. With
no entry and no tuning, candidate 0 is taken (deterministic, maybe slower). Only explicit
tuning times candidates: fastest wins, ties keep the lower index, candidates that cannot
run are skipped, and the first choice for a key is kept; a later call for that key
whose list no longer holds the kept index is an error too. Keys are nonempty and contain
no quotes or backslashes, so the exported table reads back. Before using a bucket's choice
at another width, ask the backend whether it accepts that implementation for the exact
shape; if not, use the backend's first candidate for that shape.

**Check across processes.** `repeatability.py` runs a command in fresh processes,
substitutes `{output}` with a per-run file, compares the JSON results and names the
first differing path and sequence index.

**Replay modes.** [`replay.py`](code/replay.py) supplies `ExecutionShape`, `ReplayOffer`
and `select_replay` to choose between eager execution, full replay and partial replay.
Offers are filtered by declared modes, phase, complete compatibility identity, shape and
available bytes, then ranked by `run_ns * expected_uses + setup_ns` (zero setup for a
ready offer); ties keep offer order and rejected offers carry reasons. Padding needs
explicit model permission and a measured cost that includes it. Cold offers must fit
capture peak and retained bytes plus workspace; ready offers only their workspace. If
nothing fits, selection raises. The selector does not capture, launch or allocate; use
Core's [graph capture harness](../../core/src/backends/cuda/README.md#graph-capture).

## What the code supplies

- [selection.py](code/selection.py): `bucket(width, minimum, cap)`,
  `key(operation, dims, bucket)`, `validate(table)` and `Table(recorded, tuning)` with
  `choose(key, count, time)` returning `(index, source)` and `export()`.
- [selection.h](code/selection.h): header-only C++20 `bucket`, `Table::choose` with a
  timing callback, `to_json()` and a strict `Table::parse`.
- [repeatability.py](code/repeatability.py): `first_difference`, `compare` and a runner
  (`--runs`, `--timeout`, `--report`) exiting 0 identical, 1 different, 2 a run failed.
- [replay.py](code/replay.py): the replay-mode selector above.

## What the engine supplies

The candidate lists, a synchronized timing harness on representative inputs, the
exact-shape acceptance check, the command that writes comparable output, and a replay
identity binding representation, backend, allocation/binding generations and operation
semantics (revalidated with memory before executing). At load, create the table from the
configuration; per call, compute the key, look up the cached plan for the exact width,
and on a miss choose, check against the exact shape and cache. A warm-up pass (one
execution per bucket) moves plan creation off the first requests. A development command
sets `tuning`, runs the warm-up and writes `export()` / `to_json()` into the candidate
configuration.

## Worked example

Run from `code/`. Tuning picks the fastest candidate, the exported table replays without
timing, and two outputs differing at one token are reported with their position.

```python
from selection import Table, bucket, key
from repeatability import compare

assert bucket(700, minimum=16) == 1024 and bucket(5, minimum=16) == 16 and bucket(600, 16, cap=1000) == 1000
k = key("gemm-fp8", (16384, 5120), bucket(700, 16))
assert k == "gemm-fp8:16384x5120:1024"

times = {0: 0.31, 1: 0.19, 2: 0.19, 3: None}          # candidate 3 cannot run
tuning = Table(tuning=True)
assert tuning.choose(k, 4, times.get) == (1, "tuned")
table = tuning.export()
assert table == {k: 1}

replay = Table(table)                                    # every process: no timing
assert replay.choose(k, 4, lambda i: 1 / 0) == (1, "recorded")
assert Table().choose("other:1:16", 4) == (0, "fallback")

report = compare([{"case-1": [5, 7, 9]}, {"case-1": [5, 7, 9]}, {"case-1": [5, 8, 9]}])
assert not report["identical"]
assert report["differences"] == [{"path": "$.case-1[1]", "reason": "value", "index": 1, "run": 2}]

from replay import ExecutionShape, ReplayOffer, select_replay
shape = ExecutionShape("decode", 4, 2, ("representation-v1", "backend-v1", 7))
offers = (ReplayOffer("ordinary", "eager", shape, 100),
          ReplayOffer("captured", "full", shape, 20, resident_bytes=8, setup_ns=500))
short = select_replay(shape, offers, modes={"eager", "full"}, free_bytes=16)
long = select_replay(shape, offers, modes={"eager", "full"}, free_bytes=16, expected_uses=10)
assert short.offer.mode == "eager" and long.offer.mode == "full"
```

## Adapting it

Preserve: recorded indices replay exactly and timing runs only when tuning is requested;
candidate order is stable per library version and device; a bucket's choice is used at
another width only if the backend accepts it; repeatability runs are separate processes
comparing complete outputs. Key format, bucket bounds, candidate sources, timing method,
warm-up widths and the compared output format are free.

## Costs and validation

Tuning costs one timed run per candidate per bucket; replay costs a map lookup per new
exact shape. A table from another library version or device can index a different
candidate and the range check catches only a shorter list, so record the versions.
Repeatability runs cost a full process each and detect, but do not locate,
nondeterminism.

[Python fixtures](../../tests/test_techniques.py) cover buckets, keys, validation, tuning
ties, unavailable candidates, replay without timing, out-of-range indices, difference
reporting and the runner in fresh processes. The
[native fixture](../../tests/inference/selection_tests.cpp) covers the same rules, the
JSON round trip and malformed tables. [Portable policy fixtures](../../tests/test_portable_mechanisms.py)
check capture amortization, memory limits, padding permission, partial fallback and
stale identities. The speed and determinism of each candidate are the engine's
measurements.
