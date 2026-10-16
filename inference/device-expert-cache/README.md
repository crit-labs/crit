# Plan and fill an expert cache on the device

Routed experts live in host memory and a device slot cache holds the hot ones. The
cache directory lives on the device, so a whole decode step, including residency
decisions and miss copies, runs as captured graphs with no host synchronization. A
replay helper compares cache policies offline.

## When to use it

Use it when a host round trip per routed layer costs more than the layer's decode
arithmetic. It needs device-readable host memory for the store (for example mapped
pinned memory) and enough slots for one plan's routes. Skip it when every expert fits
on the device, when the host already overlaps planning with device work, or for large
prefill batches, where a host plan grouped by expert is simpler.

## How it works

Keys are `group * experts + expert` (a group is usually a layer). The directory is
device arrays `key_slot`, `slot_key`, `slot_ref` (clock bit), `slot_pin` (stamp of
the last plan using the slot) and a state block (`hand`, `stamp`, failure flag).

One warp plans the distinct routes of one group (at most 32). It bumps the stamp,
then each hit sets its reference bit and is pinned. Misses are admitted in route
order by a clock: slots pinned by this plan are skipped, referenced slots lose their
bit, and the first other slot is the victim (one ballot per `min(32, slots)` slots,
same victims as a serial scan). The victim is remapped and pinned and a copy job
`store[key] -> slot` is written, or pool-to-slot if the lookahead pool holds the key.
Because hits are pinned first, a miss never evicts a bundle the same layer reads. A
copy kernel moves the bundles and the expert arithmetic reads `uses[i].bundle`.

**Lookahead prefetch.** The next group's router applied to the current input predicts
many of its experts. A prefetch plan tags a pool half (by group parity, so the half
being filled is never the one consumed) with nonresident predictions and writes
store-to-pool copies, run on a side stream. Prefetch never changes residency.

**Offline policy replay.** Each policy starts empty and applies the same ordered
groups of bundle keys, pinning every hit before admitting misses in order. If a bundle
cannot fit after evicting all unpinned entries, admission is bypassed; the miss is
still served and charged. Policies are LRU, LFU (LRU ties), second-chance clock and
farthest-next-use, an offline hindsight reference that is optimal only for
single-demand groups, equal sizes and mandatory admission. Optional `second-touch`
admission retains a bundle only after a demand in a prior group.

A warmup prefix updates state but is not scored. The scored suffix reports hits,
misses, fetch bytes, admissions, bypasses, evictions and the sum of optional
per-bundle `miss_ms`, which is not elapsed time. Traces can also carry
`Capacity(bytes)` (shrink is refused rather than evicting pins), `Decay(n, d)`,
`Pin`/`Release` of resident content, and `Demand(keys, phase, traffic)`. Only decode
demand promotes recency; prefill reuses hits and bypasses misses unless
`prefill_populates=True`. Prefetch and draft traffic classes change accounting
(wasted and unused prefetch bytes), never replacement.

## What the code supplies

- [device_cache.h](code/device_cache.h) / [device_cache.cu](code/device_cache.cu):
  `Directory`, `Use`, `Fill`; `enqueue_plan`, `enqueue_fill` (bundles of any multiple
  of 16 bytes) and `enqueue_prefetch_plan`. All are capturable: inputs and outputs are
  device pointers and counts are read on the device.
- [device_cache.py](code/device_cache.py): `ClockDirectory`, the host reference of the
  same rules (`plan`, `prefetch`), and `simulate` for hit rates of routing traces.
- [replay.py](code/replay.py): `Bundle`, the event types, `replay` and
  `compare_policies`. It does not reproduce the CUDA prefetch pool.

## What the engine supplies

The store (one contiguous bundle per key), slot and pool allocations, an initial
directory (all `-1`, zero bits), the expert arithmetic reading `Use`, the router and
predictor, and stream forks and joins. Per routed layer inside the decode graph:

```text
router -> top-k ids/weights (device)
enqueue_plan(dir, layer, ids, weights, k, uses, fills, fill_count)
enqueue_fill(fills, fill_count, k, bundle_bytes, blocks_per_bundle)
[fork side stream: router(layer+1) on this layer's input -> top-p ids
 enqueue_prefetch_plan(dir, layer+1, ids, p, pool_fills, pool_count); enqueue_fill(...)]
experts(uses) ... ; join side stream before planning layer+1
```

When a phase such as prefill plans on the host, copy the directory across at each
phase change and apply the same clock on both sides. Read `state[2]` when leaving the device phase: 1 means an invalid or duplicate
route, 2 means a plan had more routes than free slots.

## Worked example

```python
from device_cache import ClockDirectory

d = ClockDirectory(slots=4, groups=2, experts=8, pool_slots=2)
print(d.plan(0, [1, 2, 3]).fills)       # three store copies into slots 0, 1, 2
print(d.prefetch(1, [5, 6]))            # [(13, 0), (14, 1)]: pool half 1 tagged
p = d.plan(1, [5, 2])                   # 5 comes from the pool; 2 (key 10) is new too
print(p.hits, p.misses, p.fills)        # 0 2 [(('pool', 0), 13, 3), ('store', 10, 0)]
print(d.plan(0, [1, 2]).hits)           # 1: expert 2 survived, expert 1 was evicted

from replay import Bundle, compare_policies, replay

catalog = {key: Bundle(8, miss_ms=2.0) for key in "abcd"}
trace = tuple((key,) for key in "aabcda")
results = compare_policies(trace, catalog, 8, warmup_groups=2)
assert all(r.demands == 4 and r.peak_resident_bytes <= 8 for r in results)
scan = replay(trace, catalog, 8, admission="second-touch", warmup_groups=2)
assert (scan.hits, scan.misses, scan.fetch_bytes) == (1, 3, 24)
assert scan.estimated_miss_ms == 6.0  # synthetic additive miss cost, not latency
assert scan.bypasses == 3           # every bypass still serves the exact expert
```

## Adapting it

Preserve: every route of one plan pinned against that plan's admissions; complete
copies before the arithmetic reads a slot; a key mapped to at most one slot; the pool
half being consumed never refilled before its plan; the same clock rule on host and
device when both plan. Free to change: replacement policy (if it respects the pins),
bundle layout, copy granularity and blocks per bundle, the predictor and prefetch
width, and mapped reads versus DMA for misses.

For replay, bundle keys must distinguish model content, layer/expert and
representation, sizes must include padding, and each group must be the complete
simultaneous demand set; changing either when converting a router log can fabricate
locality. `peak_resident_bytes` counts only retained bundles, so reserve scratch and
in-flight copies separately. Choose a policy on tuning traces and confirm it on
held-out ones; hindsight results are not an online implementation.

## Costs and validation

A plan costs a few microseconds plus a clock scan of at most four passes over the
slots. Mapped-memory miss copies are bound by the host link and occupy SMs while
waiting, so too many prefetch blocks slow the main stream. Speculative verification of
several tokens multiplies distinct misses because each column routes independently.
Routes beyond the directory set the failure flag and produce null uses; check the flag
before trusting a step's output. A higher hit rate alone does not prove a faster
engine: miss sizes, dispatch, copy contention and admission overhead can reverse it.

[device_expert_cache_tests.cpp](../../tests/inference/device_expert_cache_tests.cpp)
(under `CRIT_INFERENCE_CUDA`) runs 600 random plans with lookahead prefetch on 24 and
72 slots against a host replica, checking slots, counters, bundle bytes,
duplicate-route rejection and invalid launches. The host suite checks
`ClockDirectory` pinning, victims and pool tags.
[Measured policy fixtures](../../tests/test_measured_expert_policies.py) compare replay
clock with `ClockDirectory`, check hindsight against an exhaustive oracle, and cover
sizes, pins, bypass, warmup and the event types. None of these measure
performance.
