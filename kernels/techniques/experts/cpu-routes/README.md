A routed layer's CPU share joined inside a CUDA stream, captured graphs included. crit's heterogeneous-expert-dispatch
plugin (`inference/heterogeneous-expert-dispatch`) loads it for its in-stream CPU tier, and expert-offload through the
same code. Where it does not build, the plugin's host path serves the CPU tier outside the stream.

- **Post**: one kernel lists the routes the slot table sends to the CPU tier (slot -2) in route order, with their
  expert keys, their tokens in first-use order and those tokens' rows. It writes them into a slot of one of the
  host's handoffs (Core's `crit_plugins.host`) and posts the slot, or answers the slot itself when there is nothing
  for the CPU.
- **Scatter**: after the stream's join, a second kernel copies the CPU's rows into the device part's route rows,
  where the device part's combine sums every route in route order. Where the CPU failed it writes NaN rows (never a
  silent partial sum) and the status word the caller checks.
- **One post and one join a layer**: the next layer's input needs this layer's sum, so a decode step cannot join less
  often. A layer with no CPU routes costs its post kernel and a join that passes at once.

The payload is host-expert-tier's InputLayout: a header {count, tokens, route capacity, token capacity} (uint32),
experts int32 [routes], token int32 [routes], x FP32 [tokens][hidden] and out FP32 [routes][hidden], each part
64-byte aligned. The rows are BF16-rounded FP32, the declared arithmetic's input, so a token's codes are the same
whatever dtype carries it.

It includes Core's `handoff_device.cuh` (`"host_include": true`: `crit_plugins.host.INCLUDE`).

## Interface

```python
from crit_kernels.build import load      # a plugin: crit_plugins.kernels.load_kernel("cpu-routes", folder=...)
cr = load("cpu-routes")
routes = cr.Routes(hidden, routes, tokens, [handoff.words(s) for s in range(slots)])
routes.layout()                           # [experts, token, x, out, bytes]: checked against the tier's
routes.post(slot, ids, slots, x, key_base, cpu_routes, cpu_count, counters)
routes.scatter(slot, y, cpu_routes, cpu_count, status, routes)
```

## Structures

| Structure | What | Exactness |
| --- | --- | --- |
| `post` | one block of 256 threads a call: the CPU routes listed by thread 0 in route order, the tokens' rows copied by every thread, a system-scope fence, then the post or the answer | declared |
| `scatter` | a block a CPU route after the join: the status read once, the CPU's row copied, NaN rows on failure | declared |

No knobs and no bench: the post is one block a layer and the scatter a copy of the CPU's rows; the link and the CPU
set the layer's time.

## Tests

`tests/test_cpu_routes.py`, over a slot in pinned host memory: a post lists the CPU routes, their keys, tokens and
rows, raises ready and counts the job; a post without CPU routes answers the slot itself; a scatter copies a served
slot's rows and writes NaN rows where the CPU failed; captured posts and scatters replay new routes. The full join over
Core's handoff and the CPU tier is heterogeneous-expert-dispatch's and expert-offload's device tests.
