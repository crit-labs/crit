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

## HIP

`CRIT_CORE_HIP=ON` (instead of `CRIT_CORE_CUDA`) builds these sources on ROCm's HIP runtime as `crit::hip`, found
through `find_package(hip)`. [gpu_runtime.h](../../../include/crit/gpu_runtime.h) maps every CUDA runtime name the
backend uses to its HIP equivalent under `CRIT_CORE_HIP`, which `crit::hip` defines for its consumers, so the
classes keep their names (`CudaBackend`, `cudaStream_t`) and an engine's hipified kernels run on its streams. On
HIP, `launch()` is always an ordinary launch (`programmatic_launch_supported` is false) and the device functions of
[cuda_launch.h](../../../include/crit/cuda_launch.h) are no-ops. The backend, timer and memory fixtures build
against `crit::hip` too; the graph fixtures (cuBLAS) and the linker-injected failures do not. Written against ROCm
6.4 and not yet built on an AMD toolchain.

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

## Programmatic dependent launch

[cuda_launch.h](../../../include/crit/cuda_launch.h) ([cuda_launch.cpp](cuda_launch.cpp))
lets a kernel start while the previous kernel in its stream is still running, on
devices of compute capability 9.0 and later. `programmatic_launch_supported(device)`
(or `(stream)`) is the device query; `launch(kernel, LaunchShape{grid, block, shared},
stream, programmatic, args...)` launches with programmatic stream serialization when
asked and supported, and as an ordinary launch otherwise. In the kernel, do what does
not read the previous kernel's output first (address math, loading weights or constants
into registers or shared memory), then `wait_for_prior_grid()`; the previous kernel may
call `launch_dependents()` once it no longer needs the SMs to itself. Both compile to
nothing below sm_90 and are no-ops for ordinary launches.

The pay-off is the gap between consecutive kernels of a decode step (launch, block
scheduling and ramp-up), which matters most for many small kernels: draft steps, norms
and short GEMVs. Stream capture records the launches as programmatic graph edges, so a
captured decode graph keeps the overlap. Measure it per engine: kernel times in Eval's
kernel profile are exclusive, so an overlapped kernel is charged only from its
predecessor's end ([kernel profile](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#kernel-profile)).
[programmatic_launch_tests.cu](../../../../tests/inference/programmatic_launch_tests.cu)
checks the query, and that a programmatic consumer reads every value a slow producer
wrote, eagerly and from a replayed graph.

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

`GraphCache` destruction checks the same settlement receipts as eviction. Pending,
failed or uncertain work retains the cache's executables, receipts and owners for
process recovery. Capture reserves its ownership bookkeeping before calling the
backend, so a later allocation failure cannot undo retention.
An ambiguous destruction failure also blocks further replay and destruction.

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
Retention bookkeeping is reserved before physical allocation, so final lease
cleanup can retain failures without allocating. If the service is destroyed with
unsettled or uncertain allocations, it retains their backend and charges too.

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

### Unchanged uploads

A decode step uploads small metadata (slot indices, lengths, page tables between page
boundaries) whose values often did not change since the last step, each upload a copy
serialized between the step's launches. `UploadShadow` ([device_memory.h](../../../include/crit/device_memory.h))
keeps the bytes last written through it per destination: `upload(dst, src, copy)`
calls `copy` (`cudaMemcpyAsync` on the engine's stream, say) only when `dst` did not
receive the same bytes last time, and `copied()` / `skipped()` count both. It is for
destinations nothing else writes; `invalidate(dst)` after another write, `forget(dst)`
when freeing it. A captured graph's copies always run on replay, so it serves the
eager uploads between graphs. The fixture (`memory_upload_shadow`) checks the skip
rules on the host; the saving is a few small copies per step and is not measured on a
GPU, so measure a decode step before relying on it.

## Device sets

[cuda_devices.h](../../../include/crit/cuda_devices.h) `CudaDeviceSet(devices)` gives one engine several GPUs: one
`CudaBackend` per member with its own two streams, allocations and resource ledger (`member(i).initialize(...)` with
that device's budget), peer access enabled between every pair of distinct devices that supports it (`peer(reader,
owner)` says whether a member's kernels can read another's memory in place), and stream work between members with no
host fence: `copy_async` copies a device buffer of one member into another's (`cudaMemcpyPeerAsync`) after the source
member's compute work and keeps the source buffer alive until the copy is done; `order(waiter, queue, source, queue)`
makes one member's stream wait for another's work so far; `barrier(queue)` makes every member's stream wait for every
other's (the step between an all-reduce's halves or a pipeline's stages); `category_timer(i, capacity)` times
categories on member i's device. `CudaBackend::owns(buffer)` says which member a buffer belongs to. Every call
binds the devices it acts on only while it runs and leaves the calling thread's current device as it found it (so do
the backend's own calls), since a kernel launch needs its stream's device current: bind a member
(`member(i).bind_thread()`) before launching into its streams. A device listed twice is two members on one device. Partitioning the model, collectives and the schedule stay the engine's.
[cuda_device_set_tests.cpp](../../../../tests/core/cuda_device_set_tests.cpp) checks members, copies, ordering, the
barrier and rejections on a set of the first device twice and, on a machine with two GPUs, of devices 0 and 1; only the
first runs on a one-GPU machine.
