# CUDA services

[cuda.cpp](cuda.cpp) supplies explicit CUDA allocations, pinned host ownership,
streams/events, ordering and release services through
[cuda.h](../../../include/crit/cuda.h). Two optional services build on it: a
[graph capture harness](#graph-capture) and [shared allocations with mapped
sources](#shared-allocations-and-mapped-sources). Their host protocols compile
into `crit::core` without CUDA; only thin CUDA calls live here. Model kernels
live outside the core. The backend is opt-in through `CRIT_CORE_CUDA` and
requires CUDA Toolkit 13.1. See the [Core build guide](../../../README.md) and
[ownership contracts](../../../contracts/README.md).

Rules that apply everywhere:

- `CudaEvent::order` needs a recording that belongs to the current operation; a
  wait observes already recorded work, not a future recording. Pooled event
  storage needs job ownership through release, not a modulo index.
- Keep staging and mapping owners through compute/transfer settlement, and report
  actual device aliases at registration (see [`MappedSource`](#shared-allocations-and-mapped-sources)
  and [expert residency](../../../../inference/expert-residency/README.md#with-other-inference-entries)).
- `upload()` and `download()` synchronize their stream and are for loading only,
  never per-step work.

## Spare device memory

`spare_device_bytes(margin)` reports the smaller of free device memory and the
remaining device budget, less a margin; `allocate_spare(bytes, margin)` allocates
only when that covers it. Place optional speedups there after the required state,
largest gain first. Under WDDM (Windows, WSL2) `cudaMemGetInfo` and the driver
(`nvidia-smi`) disagree on free memory. On an RTX 5090 under WSL2, nvidia-smi kept
showing about 0.75 GiB free at peak whatever the margin, so allocations past that
landed in shared system memory without an error. Size the margin from driver
readings at peak, not from a margin of zero. `CudaBuffer::device_address()` gives kernels the device
mapping of pinned host memory, so a gathered table such as token embeddings can
stay on the host.

## Speculative rounds

Run draft, verify and accept on the device, then issue exactly one
`readback_async(host_dst, offset, device_src, offset, bytes, completion)` for the
accepted tokens and return `&completion` as `Progress::readiness`. The copy runs
on the compute queue after the producing kernels, needs an owned pinned host
destination and adds no host fence.

For `device_timings()`, `backend.stage_timer(capacity)` gives a `CudaStageTimer`:
`begin(stage, index)`/`end()` around graph launches, `collect()` after readiness,
`reset()` before the next step.

## Device time by category

Without a device profiler, attribute device time with `CudaCategoryTimer`
([category_timer.cpp](category_timer.cpp)), never with a host synchronization
after each operation, which measures launch latency and host round trips instead.

`CudaCategoryTimer(capacity, device)` preallocates event pairs. Call
`start(stream)`, then `begin(category)`/`end()` around each category of work,
then `finish(categories)`, which waits once and returns `CudaCategoryTimes`:
per-category sums, the GPU span and host issue time. Categories do not nest;
dropped pairs are counted and mark the result incomplete. Keep timers out of
graphs replayed in production.

Reading it: if host issue time approaches the GPU span, launch rate is the bound
and fewer launches or a captured graph help more than faster kernels. Span minus
the category sums is idle time between kernels. To judge one operator, capture it
alone in a graph and replay it many times.

## Graph capture

Core ships the harness for CUDA engines that replay fixed topology; CPU engines
skip it. Graphs pay off when repeated topology and stable bindings amortize
capture and launch overhead; changing shapes or cheap ordinary launches can make
the added copies and memory unprofitable, so measure.

The host protocol is [device_graph.h](../../../include/crit/device_graph.h)
([device_graph.cpp](../device_graph.cpp)), part of `crit::core`:

| Type | Role |
| --- | --- |
| `GraphKey` | Names every captured assumption: phase, ordered rows, physical addresses with optional per-binding generations, shapes, representation, allocation generation, device and stream. Extend it for any other assumption. |
| `GraphCache` | At most `capacity` graphs; an identical key must name identical owners. `replay` requires settled prerequisites; a failed launch blocks replay and eviction until backend recovery; `evict` destroys only settled graphs. |
| `StagedGraphPlan` | Validates staged geometry (f16/bf16/f32, at most 64 phases). `UnsupportedLayout` is a pre-submission rejection: use another slot or an eager path. |
| `StagedGraphState` | Replay state machine: prepare, launch/bridge/launch in phase order, finish; one replay in flight; any failure poisons replay; only a successful drain closes. |

[cuda_graph.h](../../../include/crit/cuda_graph.h) ([cuda_graph.cpp](cuda_graph.cpp))
adds `CudaStagedGraph` in `crit::cuda`: it binds row geometry to one slot,
allocates padded device staging and captures each phase. Replay zero-fills
staging, copies active rows in, launches each phase with an eager bridge between,
copies rows out and records completion; `ready()`/`wait()` settle it. The
zero-fills and per-row copies sit outside the graphs, so many small rows can make
replay expensive.

Engine obligations: `GraphStage` callbacks are capture-safe and enqueue only on
the supplied stream; closures or `retained_buffers` own every extra weight,
workspace and library handle; output stays private until a successful `wait()`
or `ready()`; calls on one object are serialized, with separate slots for
overlapping work. `close()` drains, destroys captures, then releases owners; a
failed destructor quarantines them (`quarantined_staged_graphs()`). Fixtures:
[graph_tests.cpp](../../../../tests/core/graph_tests.cpp) (host) and
[cuda_graph_tests.cpp](../../../../tests/core/cuda_graph_tests.cpp) (device).

## Shared allocations and mapped sources

[device_memory.h](../../../include/crit/device_memory.h) ([device_memory.cpp](../device_memory.cpp))
is the host protocol, part of `crit::core`. `BufferService` charges each physical
allocation once through the engine's `Resources`; `BufferLease::share()` adds a
lease without charging again, and the last lease settles, releases and drops the
charge. A failed settle retains both for `retry_retained()`; a failed release is
never retried. `LeaseBackend` settlement must cover every reader and writer,
including other streams and CPU workers. Never charge one allocation both here
and through `CudaBuffer`.

`MappedSource` is an immutable address map of one generation: host spans with
their actual device aliases and owners. `resolve` returns one device pointer only
when the extent has no gaps and each alias continues the previous one (host
contiguity does not imply it); otherwise use segmented addressing, a copy or the
original source. Create a new generation when bindings change.

[cuda_memory.h](../../../include/crit/cuda_memory.h) ([cuda_memory.cpp](cuda_memory.cpp))
adds the CUDA hooks: `cuda_lease_backend(backend, memory)` allocates with
`cudaMalloc` or `cudaMallocHost`, settles both streams and frees;
`cuda_map_host(backend, host, bytes)` registers caller-owned host bytes and
reports the actual device alias, unregistering at the last release (a failure
retains it, `quarantined_cuda_mappings()`); `cuda_pin_host_block(backend, block, touch_threads)` registers a filled
[HostBlock](../../memory/README.md#large-host-blocks-and-the-process-pool) once, attaches the registration to it and
touches its pages. Fixtures:
[device_memory_tests.cpp](../../../../tests/core/device_memory_tests.cpp) (host) and
[cuda_memory_tests.cpp](../../../../tests/core/cuda_memory_tests.cpp) (device).
