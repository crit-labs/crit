# Record implementation choices instead of timing them at load

When an engine picks among interchangeable implementations of one operation (kernel
variants, tile shapes, a math library's algorithms for a shape), tune once, record the
choices in the frozen configuration and replay them. Picking by timing at load resolves
near-ties differently from process to process; different implementations round
differently, so with greedy decoding two processes of the same package generate
different text, and a single-process repeat check cannot see it.

## When to use it

Use it whenever several candidates exist for an operation. It is unnecessary when every
candidate is bitwise identical for the engine's inputs or only one is ever used, and optional
in quality mode `speed`: there Eval passes an output difference that parts at a
near-tie ([forks](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#forks-in-quality-mode-speed)), which is what two processes' near-tie choices produce.
A prompt GEMM takes one choice for every row count. A prompt's rows never depend on how long
the prompt is, and [crit smoke](../../smoke/README.md#tolerances)'s `causal` tests hold every
engine to it: a kernel that cuBLAS's heuristics pick by row count, or a padded bucket with a
choice of its own, moves a short prompt's rows (on a Qwen3.8-27B NVFP4 engine on an RTX 5090, one
bf16 ulp in one small GEMM moved a 17-token prompt's rows by 0.25 nats on average). Key prompt
operations without a width bucket: record one split-K-free
algorithm per GEMM shape and run it for every row count, which gives a row the same bits at any
prompt length at about the heuristics' speed. `torch._scaled_mm` and `torch.matmul` take no
algorithm, so call [pinned_gemm](code/pinned_gemm.py): record each prompt GEMM shape once per device
and cuBLASLt version (`record(n, k, format)`) into a `GemmTable` kept in the frozen configuration,
load it at engine load and replay with `pinned_mm(a, b, table.entry(n, k, format))`; a table from
another device or cuBLASLt version is refused, so record it again there. A row's bits only have
to be the same at every length, not to come from one algorithm, so `record` also keeps, per
power-of-two bucket of up to 64 rows, a faster split-K-free algorithm whose output equals the
entry's bit for bit (in the cases below, the same cuBLASLt algorithm with another tile or stage
count). Check the result with `crit smoke --only causal`. On an RTX 5090 (cuBLASLt 130101)
with random BF16 and FP8 inputs, the pinned GEMMs of shapes (N, K) = (5120, 5120), (17408, 5120)
and (5120, 17408) took 0.82 to 1.24 times the time of torch's heuristic-picked kernels at 128 to
4,096 rows and, at 2 to 64 rows, 0.92 to 1.43 times in BF16 and 0.98 to 1.36 times in FP8 for
(17408, 5120), whose 17-row product a bucket's algorithm took from 111 to 59 µs. In FP8 at
(5120, 5120) and (5120, 17408) the faster short-prompt candidates give other bits, so the pinned
GEMM stayed 1.5 to 4.5 times slower there. Fixed row slabs are exact too but slow: 128-row slabs took
13% of the Qwen3.8-27B engine's prefill on 512-token prompts.

A format is `INPUTS[-SCALE]:OUTPUT` with FP32 accumulation, output `bf16`, `fp16` or `fp32`:

| Inputs | Scales | cuBLASLt needs |
| --- | --- | --- |
| `fp16`, `bf16`, `fp32` (no TF32) | none | any CUDA GPU |
| `fp8e4m3`, `fp8e5m2`, pairs such as `fp8e4m3xfp8e5m2` | none, `-tensorscale` or `-rowscale` | compute capability 8.9 or later |
| `mxfp8` (e4m3), `mxfp4` (e2m1) | one UE8M0 scale per 32 elements along K | compute capability 10.0 or later (Blackwell) |
| `nvfp4` (e2m1) | one UE4M3 scale per 16 elements, and an FP32 global scale passed as `alpha` | compute capability 10.0 or later (Blackwell) |

A pair names a's type, then b's; e5m2 with e5m2 is not offered. `-tensorscale` takes one FP32
scalar per operand and `-rowscale` FP32 vectors over a's rows and b's output channels, for
row-scaled heads and dynamic per-token FP8, where cuBLASLt offers outer-vector scaling.
FP4 is packed two per byte (the first element in the low nibble). Block scales are passed in
cuBLASLt's blocked layout (128-row by 4-column tiles, rows padded to 128): `blocked_scales()` lays
out row-major scales for any row count, so a prompt prefix `a[:m]` takes
`blocked_scales(row_scales[:m])`. `bf16` and `fp8` still name `bf16:bf16` and
`fp8e4m3-tensorscale:bf16` under their old keys, so existing tables load. When cuBLASLt offers no
algorithm for a format on the device, `record` raises `UnsupportedFormat` naming the format, the
device and the cuBLASLt version. On the RTX 5090 with cuBLASLt 130101 that is the case for
`-rowscale`, e5m2 with e5m2 and `mxfp4`; the other formats are row-exact there. Integer GEMMs need
no pin: int32 accumulation is exact in any order, so any algorithm gives a row the same bits.
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
- [pinned_gemm.py](code/pinned_gemm.py) and [pinned_gemm.cpp](code/pinned_gemm.cpp): prompt GEMMs with
  one recorded split-K-free cuBLASLt algorithm for every row count, for each format above: a [M, K]
  times b [N, K] transposed. `record(n, k, format, rows)` keeps the row-exact candidates and takes the fastest by mean time per
  row, and `short_buckets` adds per short row bucket (`bucket()` widths up to 64) a faster
  algorithm with the same bits; an entry without buckets runs its algorithm everywhere.
  `GemmTable` keys entries with `key()` (width bucket 0: every row count), stores the
  cuBLASLt version and device and refuses others; `pinned_mm(a, b, entry, scale_a, scale_b, alpha)`
  replays, checking each new row count once and raising for one the algorithm cannot run.
  Importing needs no CUDA; the extension builds once with `torch.utils.cpp_extension`.

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
stale identities. [Pinned GEMM tests](tests/test_pinned_gemm.py) check the table's keys, file
round trip, refusals, format parsing and the blocked scale layout's padding without CUDA, and with
CUDA, for every format the GPU takes (the others are skipped with the reason), that every row is
bit-identical across prompt lengths and matches `torch.matmul` or `torch._scaled_mm` where torch
has the format, else the FP32 dequantized product. The speed and determinism of each candidate are the engine's measurements.
