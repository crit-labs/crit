# In-stream host work mailbox

Run host work in the middle of a CUDA stream, including a captured graph, without a host round trip on the
decision path: a device kernel posts a job into mapped pinned memory, a host service thread runs it, and the
stream waits for the answer only where it reads the result. Host-computed experts, host lookups and
host-issued copies that the device decides on mid-graph all fit this shape. The entry supplies the doorbell
protocol, two capturable waits chosen by measurement, an eager fallback, release and failure handling.

## When to use it

Use it when a graph-captured step needs work only the host can do, and the device decides at run time what
that work is (which experts missed, which rows to fetch). Without it the engine either splits the graph at
every such point and synchronizes, or writes its own mailbox. Don't use it when the host work is known before
launch (enqueue it ahead), when the work fits on the device, or when the join would come right after the post
with nothing for the device to do meanwhile and the measured join cost exceeds the work it moves.

## How it works

Each slot has three words and a payload in mapped pinned memory. Per job:

1. **Post (device).** The device writes the payload, fences it to system scope, then writes
   `status = pending`, `done = 0`, a system fence, and `ready = 1`.
2. **Serve (host).** The service thread sees `ready`, clears it, runs the handler on the payload, writes
   `status` and then `done = 1`.
3. **Join (stream).** The stream waits until `done >= 1`, then reads the result with volatile loads.

The device resets `done`, not the host, because the values of stream memory operations are fixed when a graph
is captured: a captured join can only wait for a constant, and only the device runs on every replay. A wait for
a per-replay sequence number can't be captured, and a host-side reset races the next replay. When the device
decides a slot needs no host work this time, it answers the slot itself (`skip`: `status = skipped`,
`done = 1`), so every captured join completes.

**Two capturable joins wait on the same word.**

- `stream_wait` is `cuStreamWaitValue32(done, 1, GEQ)`. The stream's front end holds the wait, so no SM
  is occupied, and cooperative kernels on other streams can still get every SM.
- `device_poll` is a one-thread kernel that polls `done` with `__nanosleep`. It occupies one block slot while
  it waits.

Their cost differs by driver and platform, so neither is a default. `measure_joins(device)` probes support
functionally (eager and captured) and times post plus join in a replayed graph for each. `choice()` takes the
cheaper. The `host_eager` join synchronizes the stream on the host, serves the job on the calling thread and
continues. It works wherever mapped memory does, but it rejects capture and serves as the fallback when no
capturable join is supported.

**Release and failure.** `release()` answers every pending doorbell with `status = released` and
`done = 0x7fffffff`. A sweeper thread then answers later doorbells the same way, even while the service thread
is stuck inside a job, so a stuck window always drains. Use it at shutdown or from a watchdog. Outputs of a
window that saw a release are invalid. `rearm()` waits for a running job to return, clears every word and
accepts work again. If a handler throws, the mailbox fails for good: that doorbell and every later one are
answered at once with `status = failed`, and the handler is never called again. The service answers raised
doorbells in any order, so slices of one layer, out-of-order posts and windows aborted between post and join
never desynchronize it. After an aborted window, call `rearm()` before the next one.

## What the code supplies

| File | Supplies |
| --- | --- |
| [`mailbox.h`](code/mailbox.h), [`mailbox.cu`](code/mailbox.cu) | `Mailbox` over caller-owned mapped bytes (`MappedRegion`, `region_bytes`): slots, service thread, `post`/`skip`/`join` stream operations, `release`, `rearm`, terminal failure; `probe_joins` and `measure_joins` with `JoinCosts::choice()`. Driver entry points are resolved through the runtime, so nothing links the driver library. |
| [`mailbox_device.cuh`](code/mailbox_device.cuh) | `post_job`, `skip_job`, `read_result`, `read_status` for posting and reading from the engine's own kernels |
| [`implementation.py`](code/implementation.py) | Host reference of the protocol with captured programs replayed in order |

## What the engine supplies

- **Mapped bytes**, allocated through Core so they are accounted for: `CudaBackend::allocate(Memory::host,
  region_bytes(slots, payload))` with `device_address()`, or `cuda_map_host`. Keep the owner alive past the
  mailbox.
- **The payload layout and the handler**, which runs on the service thread. A handler that fans out to a
  worker pool owns that pool. A handler must return, because a job that never returns blocks destruction. Its
  outputs must be complete before it returns, since the service writes `done` after it.
- **Posting from its kernels** after every payload write is system-fenced: a fence in each writing thread and
  then a block barrier, or writes from an earlier kernel in stream order. One slot per concurrently
  outstanding job. A slot can be reused for the next job once the stream has joined the previous one.
- **The join choice**, from `measure_joins` on the target machine, ideally re-checked inside the engine's own
  step, because concurrent kernels change both costs. Then a capture with that join, and `release()` from its
  cancellation or watchdog path.
- **Reading results** with volatile loads after the join, and discarding the window's outputs when `status` is
  not `served` or `skipped`.

**Host worker pools fed from the device.** A device-posted job in mapped memory answered by a host pool needs
each worker to keep the job generation it last served in its own variable. A worker that rereads the shared
generation at the top of its loop can see the next job already posted, wait for the one after it, and leave the
pool waiting at the next barrier while the device waits on the reply. Pass the generation into every barrier
rather than reading it there.

## Worked example

Run from `code/`. Three replays of one captured program each stall at the join until the host answers, then
read their own result. A program that raises the doorbell without the device's reset passes its join on a
stale answer. Release completes a stuck window, and a failed handler answers every later doorbell.

```python
from implementation import Mailbox, Stream, SKIPPED, FAILED, RELEASED

box = Mailbox(2, lambda slot, p: p.update(y=3 * p["x"] + 1))
inputs = iter(range(1, 100))
graph = Stream([lambda: box.post(0, x=next(inputs)), lambda: box.joined(0),
                lambda: box.skip(1), lambda: box.joined(1)])
results = []
for _ in range(3):
    graph.launch()
    assert not graph.run()  # stalled on slot 0's join
    assert box.serve_all() == 1  # the host service answers
    assert graph.run()
    results.append(box.slots[0].payload["y"])
assert results == [4, 7, 10] and box.slots[1].status == SKIPPED

# Raising ready without resetting done: done is still 1 from the last replay, so the join passes too early.
stale = Stream([lambda: setattr(box.slots[0], "ready", 1), lambda: box.joined(0)])
stale.launch()
assert stale.run()
box.serve_all()

graph.launch()
assert not graph.run()
box.release()  # a watchdog completes the stuck window; its outputs are invalid
assert graph.run() and box.slots[0].status == RELEASED
assert box.rearm()

def broken(slot, payload):
    raise RuntimeError("host job failed")
box.handler = broken
for _ in range(2):  # the failing window and every later one complete at once
    graph.launch()
    assert not graph.run()
    box.serve_all()
    assert graph.run() and box.slots[0].status == FAILED
assert box.failed and not box.rearm()
```

## Adapting it

Preserve the device-side reset of `done` before `ready`, the system fence between the payload and the
doorbell, the host writing `status` before `done`, one outstanding job per slot, the device answering skipped
slots itself, terminal failure answering, and `rearm()` only after the joining streams are drained. You can
change the number of slots, the payload layout, the polling and back-off policy, posting from a stream
operation or a kernel, and the handler's threading. A different join is fine if it waits for the same word.

With [heterogeneous expert dispatch](../heterogeneous-expert-dispatch/README.md), the device's plan decides
which misses the host computes and posts them. The host share comes from the measured split, and the host must
use the device's arithmetic, as that entry's exact split states. Then the
[contribution](../async-expert-contributions/README.md) rules apply to a join inside the stream: every route
occurrence is answered before the sum, sources stay owned until the host has read them, and a failed or
released answer discards the result instead of publishing a partial one.

## Costs and validation

A post costs a few device writes and two system fences. The join costs what `measure_joins` reports, and that
depends on the platform. The service thread spins while jobs arrive and backs off to short sleeps after
`idle_after`, so leave it a core. For scale only: on one RTX 5090 (driver 596.49, CUDA 13.1), with native
Windows and with WSL2, the fixture measured post plus join at 37-41 µs with `stream_wait` and 2.7-2.8 µs with
`device_poll`. A separate probe showed the stream wait's extra 50-60 µs stays even when the word is already
set and does not hide behind longer device work. On that machine, a join per layer favours the poll kernel. In
one MoE engine there, with about 40 joins per decode pass, the measured choice (the poll kernel) matched the
engine's own spin-kernel handshake with identical output, and the stream wait cost about 8% of plain decode.
On drivers where the front-end wait is cheap, it frees the SM. Measure; don't assume either.

The [CUDA fixture](../../tests/inference/host_mailbox_tests.cpp) runs both capturable joins through 200 replays
of an eight-slot window with a skipped slot. It checks that every replay reads its own answer. It also covers
a release of a window held by a stuck job, `rearm()` refusing while the job runs, and a handler failure on the
fifth job that answers 76 later doorbells without calling the handler. Further cases are an aborted window
whose orphaned doorbells are served, followed by reversed-order posts, and the eager join with its capture
rejection. It passes on native Windows and on WSL2. Engine measurements are not part of the fixture.
