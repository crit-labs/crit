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
(`fill_blocks`, `fill_vector`). The bench (`bench.py:Fill`) times the fill of the model's NVFP4 bundles from mapped
host memory (pinned, read at its device address, as an engine's store is; device memory where the platform maps
none) against torch's `index_select` and `index_copy_` over the same mapping.

## Against the link (RTX 5090, PCIe)

A fill from an engine's store reads registered host memory over the link, and the link bounds it. Measured with a
synthetic store of random 2.76 MB bundles (Flash-Next's NVFP4 experts, 2560 x 640) in anonymous host memory mapped
and registered as expert-offload maps it, into device slots, in a CUDA graph with L2 flushed, median of 9; every
configuration's slots checked byte for byte first. us a call (GB/s):

| Bundles | fill, 32 blocks | 128 blocks | 256 blocks | cudaMemcpyAsync a bundle | one contiguous copy | torch index_copy | fill from device memory, 128 blocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 108.2 (25.6) | 107.1 (25.8) | 108.9 (25.4) | 149.4 (18.5) | 144.2 (19.2) | 455.3 (6.1) | 7.3 |
| 2 | 211.4 (26.2) | 211.6 (26.1) | 211.6 (26.1) | 278.1 (19.9) | 274.8 (20.1) | 221.7 (24.9) | 10.9 |
| 4 | 418.5 (26.4) | 416.4 (26.6) | 417.8 (26.5) | 472.8 (23.4) | 437.8 (25.3) | 432.9 (25.6) | 21.0 |
| 8 | 830.1 (26.7) | 830.1 (26.7) | 830.0 (26.7) | 867.1 (25.5) | 815.8 (27.1) | 863.4 (25.6) | 41.6 |
| 16 | 1655.4 (26.7) | 1655.1 (26.7) | 1655.4 (26.7) | 1636.9 (27.0) | 1594.0 (27.8) | 1730.2 (25.6) | 76.4 |

- **The fill is at the link's ceiling.** Every grid from 32 to 256 blocks and every load width (4, 8, 16 bytes: within
  0.5% of each other) copies at the copy engine's rate from the same memory, and faster than the copy engine for one
  to four bundles (it pays no per-copy setup). From device memory the same fill is 15 to 22 times faster: the
  kernel is not what a decode round waits for, the link is. Flash-Next's trace (425 us a call at 32 blocks) is four
  bundles a call at this rate.
- **Four blocks hold the link.** Alone, a fill of four bundles takes 421 us at 4 blocks and 423 us at 8 to 128 (one
  block: 581 us, two: 428). Past 16 bundles a call the link's own rate wanders from 20 to 27 GB/s between runs, the
  copy engine's too, at every grid.
- **What a fill beside work costs the work.** A fill of four bundles forked on a side branch of the graph, beside
  memory-bound BF16 GEMVs over device weights (525 us alone): the work takes 534 us beside 4 fill blocks, 544 beside
  8, 577 beside 16, 638 beside 32, 762 beside 85 (half the SMs, the plugin's default) and 764 beside 128, while
  the fill keeps 25.2 to 25.6 GB/s at every grid. A fill beside a round (recent residency's copies, side swaps)
  holds the link at a few blocks and leaves the SMs to the round; expert-offload's engine measurement on the real
  experts (`fill_blocks` 4, 16, 32, 64: 18.6, 17.0, 16.7, 16.8 ms a round) found the round's end waiting for 4
  blocks' copies, so the grid stays a plugin option measured on the engine, not a kernel default.
- **What does not pay in the kernel.** Grid, load width and read pattern: nothing to gain against a link it already
  saturates. A copy-engine fill: the plan decides the copies on the device, so a host-issued cudaMemcpyAsync cannot
  know their addresses inside a captured round without a host synchronization, and it is no faster at the sizes a
  decode plan copies.
- **What could free the GPU while the link works** (not built here, for the plugins): the copies of a ranked plan
  sit in series between the plan and the layer's experts. Their hit experts need nothing from the fill: grouped-
  experts' route_outputs leaves routes with a negative slot as they were and combine is a pass of its own, so a
  layer could run its hits on the main branch beside the fill (at 4 to 8 blocks), then its misses, then combine,
  every route's bits unchanged. On Flash-Next's decode trace the experts' passes are about 71 us a layer against
  425 us of fill, so this hides at most the hits' share of those 71 us a layer, about 5% of a decode round, less
  the second pass's launches; a prefetch of likely misses pays only where the link is idle, beside the round's
  attention and dense projections.

`bench_fill.py` and `bench_beside.py` in the evidence (wsl-evidence/2026-11-25-improve-crits-weakest-kernels/
1-expert-fill) reproduce these tables.

## Tests

`tests/test_expert_directory.py`: a fill copies its bundles at every load width and grid (more copies than blocks, a
tail shorter than the grid's words, one key into several slots, nothing past the count or for no copies), from mapped
host memory too, a range of a longer job list only, and replays in a CUDA graph, a range's first and count changed
between replays; it refuses a load width or bundle it cannot copy; a fresh directory's plan admits the routed experts
in route order and lists their copies.
