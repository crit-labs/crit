A device expert cache's directory on the device: the plan of one routed layer over a step's routes and the fills that
copy the bundles it admits into the cache's slots. crit's device-expert-cache plugin (`inference/device-expert-cache`)
loads it for its device form, and expert-offload through the same code. Where it does not build, the plugin's torch
ops plan and fill by the same rule (capture-safe, slower).

- **The host twin's decisions**: a plan's slot table, copies, counters and failure bits are those of
  `device_cache.LayerDirectory`, plan for plan. device-expert-cache's tests (`tests/test_device.py`) check the clock,
  the admission limits (quota, quota by miss count, second touch), prompt rings and recent residency, eager and in
  captured rounds.
- **Mechanism, no arithmetic**: no sums, so its level is `declared`. A fill's slots are their store bundles byte for
  byte.
- **One launch each, no host synchronization, no allocation**: a captured round replays it.

## Interface

```python
from crit_kernels.build import load      # a plugin: crit_plugins.kernels.load_kernel("expert-directory", folder=...)
ed = load("expert-directory")
ed.plan(key_slot, slot_key, slot_ref, slot_pin, state, counters, seen, route_counts, quota, admit_quota, layer,
        experts, ids, top, rows, row_counters, ring, out, jobs, job_count, slot_last, victims, recent, pending,
        range_, link)                     # an empty tensor for an absent buffer
ed.fill(source, store, jobs, job_count, blocks, vector, job_first)
ed.fill_program(source, store, jobs, job_count, program, shared, blocks)
```

The directory's tensors and their layout are device-expert-cache's (`code/device_directory.py`).

## Structures

| Structure | What | Exactness |
| --- | --- | --- |
| `plan` | one block of 256 threads a routed layer: distinct experts in route order, hits pinned, misses decided in that order, clock victims by a warp over 32-slot windows, recent residency's victim list, prompt ring slots | declared |
| `fill` | the plan's copies by a fixed grid, 4, 8 or 16 bytes a load (the call's `vector` and `blocks`) | declared |
| `fill-program` | the copies rebuilt from a store in another layout (CSF-compressed NVFP4 scales) | declared |

No compile-time knobs: the fill's grid and load width are call arguments, which the plugin's options set
(`fill_blocks`, `fill_vector`). The bench (`bench.py:Fill`) times the fill of the model's NVFP4 bundles from device
memory against torch's `index_select` and `index_copy_`.

## Tests

`tests/test_expert_directory.py`: a fill copies its bundles at every load width and grid, a range of a longer job
list only, and replays in a CUDA graph; a fresh directory's plan admits the routed experts in route order and lists
their copies.
