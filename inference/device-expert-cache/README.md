# Plan and fill an expert cache on the device

Routed experts live in host memory and a device slot cache holds the hot ones. The
cache directory lives on the device, so a whole decode step, including residency
decisions and miss copies, runs as captured graphs with no host synchronization.
`ExpertCache` is the ready component: one call per routed layer, a warm start from a
residency profile, admission limits that send part of the misses to the CPU at a split
measured on the target, and the telemetry Eval reads. A replay helper compares cache
policies offline. The kernels are one source for CUDA and
HIP (built by hipcc, or with `CRIT_CORE_HIP`, through Core's `crit/gpu_runtime.h`), and
`device_cache.metal` has them for Metal with the directory's arrays bound as buffers. The
HIP and Metal versions are not compiled or run in crit's tests: check them against the
replay on the target.

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

**The host store.** The cache copies whole bundles, so the store holds one contiguous bundle
per key in key order. A checkpoint keeps an expert's matrices and scales as separate tensors,
often over several shards, or stacks a layer's experts in one tensor.
[expert_store.py](code/expert_store.py)'s `build_store` reads them once into one file: every
bundle the same size, each component 16-byte aligned. Beside it goes an index with each
component's offset and the checkpoint's fingerprint, so an unchanged source is not packed again.
Name each component's tensor by a template over `{group}` and `{expert}`; a template without
`{expert}` slices a stacked tensor. Pin the file instead of allocating pinned memory and
copying into it. [expert_store.h](code/expert_store.h)'s `pin_store` registers Core's read-only
mapping of the file in one call where the device allows read-only registration, and otherwise
reads it with several threads into a Core `HostBlock`. `open_store(...).register()` does the
same from Python through torch's cudart. (For scale only: on one machine, `cudaHostAlloc` of a
68 GB store took about 8 minutes under WSL; registering a mapped file takes seconds.)

**Feeding the expert arithmetic.** The routed experts' GEMMs come from a provider's grouped
GEMM (CUTLASS's ptr-array grouped GEMM, `cublasGemmGroupedBatchedEx`, or a block-scaled grouped
GEMM for 4-bit formats, from [Kernels' sources](../../kernels/SOURCES.md)), which takes device
arrays of pointers. `enqueue_operands` writes them from a plan's `Use`s on the device, so a captured
step goes from the plan to the GEMM with no host round trip. For each route it writes a pointer to
each component inside the bundle (the offsets from the store's index), to the route's activation
row and output row, and its routing weight. With `skip_bypassed`, the misses the admission
limits send to the CPU get null operands, and the engine computes them there.

**Lent capacity.** Device memory a request cannot use can hold slots: KV-cache
positions beyond the request's length (revalidated when a request begins) and the
prompt workspace between prompts (invalidated while a prompt chunk runs). The gain
grows with the context the engine is built for and shrinks as requests fill it, so
size it from the engine's own KV capacity and workspace, not a fixed slot count. Do not
reserve a fixed prompt workspace beside the cache: lend the prompt's (prefill) buffers
from cache slots while a prompt runs and return them for decode, so decode never leaves
that memory idle (recon's placement sizes the cache this way). Keep
both kinds out of a prompt's plans while the memory is in use, and never let a lent
slot back a key the current pass reads. Expert residency's
[slot loans](../expert-residency/README.md#lending-owner-memory-to-the-cache) give the owner
that reclaim lifecycle on the host. (For scale only: one engine on a 32 GB GPU
with a 1M-token FP8 KV cache gained about 1,500 and 2,300-2,900 slots next to
3,000-5,000 owned ones.)

**Adapting during decode.** Keep the cache adapting while it decodes: admit the
experts a miss fetches over the link into the cache instead of reading them around it,
and keep the replacement policy running between requests rather than freezing the slots
after a warmup. Set the split between misses computed on the CPU and misses copied over
the link from speeds measured on the target machine, each path alone and both together
(they read the same host memory), and revisit it as the hit rate changes. Do not trade
adaptivity for repeatable Eval numbers: Eval warms each case until consecutive requests
agree before it compares repetitions (crit-eval's `speed.steady_state`), so a cache that
settles is measured at the speed it settles to, and one frozen early is measured cold.

**Admission limits and the CPU/link split.** Two limits keep the link from becoming the
step's bottleneck, and both run inside the plan, so the split needs no host round trip.
`admit_quota` admits at most that many misses per plan in route order (routers list
their experts by weight, so the heaviest misses take the link), and `admit_by_misses`
gives that quota by the plan's miss count. `key_seen` turns on second-touch admission:
a key's first miss is served without being admitted, so a one-off route does not evict a
hot expert. A miss over the limits takes no victim; its `Use` names the store bundle
(`Source::store`), and the engine computes it on the CPU from host memory, or reads it
mapped. Those CPU misses are the split. Derive the quota table from measurements with
[split.py](code/split.py): time one routed layer with k link misses alone, j CPU misses
alone and both at once (`calibrate`). `quota_table` then picks the fastest measured split
per miss count, never interpolating, and `needs_recalibration` says when the hit rate has
moved far enough to measure again. (For scale only: on one RTX 5090 with a PCIe 4.0 link
and a 6-core CPU, about 3 link misses per layer, overlapped with the CPU's, was fastest.)
Under the batch invariance rules the CPU rows must give the device's bits, as
[host expert arithmetic](../host-expert-arithmetic/README.md) does; in quality mode
`as_quantized` they need not, and the CPU may use its own faster rows.

**Starting warm: a residency profile.** A cold cache spends its first requests on misses.
The package may ship a residency or routing profile, a file of the engine's own format
(per-layer route counts, or the slots to fill), that the engine reads at start to fill
the cache before its first request; every file in the package folder is bound to it, so
the profile is part of the candidate's identity. Load it at engine start, never per
request, and keep adapting from it. Routing depends on content, so the profile is a
content-dependent choice under AGENTS.md's rule: build it from a general corpus
unrelated to Eval (never Eval's prompts, pinned continuation, held-out prompts or
look-alike text), state that corpus and a held-out check in several languages and
domains in the proposal, and keep the cache adapting so a profile that misses still
converges. crit names such a corpus: [corpus.json](code/corpus.json) lists public web text in
seven languages and code in three to tune on, plus held-out sources to check on (test splits,
two languages and a programming language the tuning never sees, rows far from the tuning ones).
[corpus.py](code/corpus.py) fetches it through the Hugging Face dataset viewer with the standard
library, caches it and records each dataset's revision. A profile built from it, with its
held-out check, meets the rule without a corpus of the build's own: state `describe(...)` as the
corpus. The engine runs its own forward over each text and logs one plan per routed layer and
token. [residency_profile.py](code/residency_profile.py) builds the profile from those logs.
`build_profile` takes router logs of a tuning corpus (one plan per routed layer and
token), keeps the most routed keys and records the corpus and a held-out check.
`held_out_check` gives cold and preloaded hit rates on at least two held-out traces.
`load_profile` checks the file against the engine's geometry. `enqueue_preload` (or
`ExpertCache::preload`) maps the keys to the first slots, unreferenced, so a profile
entry no route uses is the clock's first victim.

**Lookahead prefetch.** The next group's router applied to the current input predicts
many of its experts. A prefetch plan tags a pool half (by group parity, so the half
being filled is never the one consumed) with nonresident predictions and writes
store-to-pool copies, run on a side stream. Pool slots go to the valid, distinct,
nonresident predictions in prediction order, so a resident prediction takes no slot and
a list longer than the pool fills it. Prefetch never changes residency. Measure it before
keeping it (see Costs): the copies only pay off when they overlap the layer's arithmetic,
and in one engine with the admission split above, lookahead prefetch made decode slower.

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
demand promotes a hit (recency, frequency and the clock bit); prefill reuses hits and
bypasses misses unless `prefill_populates=True`. Prefetch and draft misses are admitted
like decode misses, enter at the current recency and can evict resident bundles;
`protect_decode_groups=n` keeps keys from the last `n` decode groups out of their
eviction. Whether a prefetch or draft is later used changes only accounting (wasted and
unused prefetch bytes).

## What the code supplies

- [device_cache.h](code/device_cache.h) / [device_cache.cu](code/device_cache.cu):
  `Directory` (with the admission limits), `Use`, `Source`, `Fill`; `enqueue_plan`,
  `enqueue_fill` (bundles of any multiple of 16 bytes), `enqueue_prefetch_plan`,
  `enqueue_preload`, and `enqueue_operands` with `OperandSpec` and `Operands` for a provider's
  grouped GEMM. All are capturable: inputs and outputs are device pointers and counts
  are read on the device.
- [runtime.h](code/runtime.h) / [runtime.cu](code/runtime.cu): `ExpertCache`, the ready
  component. It owns the directory and per-layer scratch around the engine's slots, pool
  and store, and offers `init`, `set_admission`, `preload`, `plan_and_fetch` (one
  capturable call per routed layer), `prefetch` and `telemetry`. The telemetry carries the
  `expert_reads` and `expert_hits` that Eval prices the expert tiers with, plus bypassed
  misses and the failure flag.
- [device_cache.py](code/device_cache.py): `ClockDirectory`, the host reference of the
  same rules (`plan` with the admission limits, `prefetch`, `preload`), and `simulate` for
  hit rates of routing traces.
- [residency_profile.py](code/residency_profile.py): `build_profile`, `held_out_check`,
  `write_profile` and `load_profile`.
- [split.py](code/split.py): `SplitTimings`, `calibrate`, `best_split`, `quota_table` and
  `needs_recalibration`.
- [corpus.json](code/corpus.json) / [corpus.py](code/corpus.py): the profile corpus,
  `load_manifest`, `fetch` and `describe`.
- [expert_store.py](code/expert_store.py) / [expert_store.h](code/expert_store.h): `Component`,
  `layout`, `build_store`, `open_store` (`Store` with `bundle`, `component`, `register`) and
  `pin_store` (`PinnedStore`, whose `device` is `Directory::store`).
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

## Recipe

Bring a cache up in this order. Each step names what to measure before the next.

1. **Build the store** with `build_store` and pin it with `pin_store` (or `Store.register`).
   Measure the load time.
2. **Size the slots** from the memory the engine can lend (KV positions beyond the
   request, the prompt workspace), not a fixed count. Measure the free memory after the
   load at the required context.
3. **Serve misses from the store first.** Use `ExpertCache` with no limits and decode with
   every miss admitted. Measure `expert_hits / expert_reads` and the step time against
   the link floor.
4. **Load a profile.** Build it with `residency_profile.py` from routes over `corpus.json`'s
   tuning sources, and check that `held_out_check` raises the hit rate on its held-out ones. Measure the
   first requests' decode speed with the profile and without it.
5. **Measure the split.** Run `split.calibrate` with the engine's own layer timings at the
   hit rate step 3 settles to, then `set_admission(-1, quota_table(...), ...)`. Measure
   decode against step 2's, and calibrate again when `needs_recalibration` says so.
6. **Try second-touch** when hot experts get evicted by one-off routes (`bypassed` rises,
   and the hit rate falls between requests). Keep it only when decode improves.
7. **Only then try lookahead prefetch.** Keep it only when the integrated step gets faster.

## Worked example

```python
from device_cache import ClockDirectory

d = ClockDirectory(slots=4, groups=2, experts=8, pool_slots=2)
print(d.plan(0, [1, 2, 3]).fills)       # three store copies into slots 0, 1, 2
print(d.prefetch(1, [5, 6]))            # [(13, 0), (14, 1)]: pool half 1 tagged
p = d.plan(1, [5, 2])                   # 5 comes from the pool; 2 (key 10) is new too
print(p.hits, p.misses, p.fills)        # 0 2 [(('pool', 0), 13, 3), ('store', 10, 0)]
print(d.plan(0, [1, 2]).hits)           # 1: expert 2 survived, expert 1 was evicted

limited = ClockDirectory(slots=8, groups=1, experts=16, admit_quota=1)
print(limited.plan(0, [4, 5, 6]).slots)  # [0, 'store', 'store']: one link miss, two for the CPU
warm = ClockDirectory(slots=4, groups=1, experts=16)
print(warm.preload([7, 3]), warm.plan(0, [3, 9]).hits)  # [(7, 0), (3, 1)] 1

from split import SplitTimings, quota_table

timings = SplitTimings(link={1: 1.0, 2: 2.5}, cpu={1: 1.5, 2: 3.0}, both={(1, 1): 1.6})
assert quota_table(timings)[2] == 1    # two misses: one over the link, one on the CPU

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
waiting, so too many prefetch blocks slow the main stream. One measurement, on an RTX
5090 with experts in host memory and a decode graph of 4-token verify passes:
- **Prediction worked.** The next layer's router, applied to the hidden state after
  the current layer, caught 75% of misses with 3% extra reads.
- **SM copies did not overlap.** On a side stream they ran only between the pass's
  kernels, so each prefetched expert cost as much as a miss.
- **Late copy-engine copies stalled.** Copies issued from a host thread after the
  prediction arrived too late, and the consuming layer waited for them.
- **The copy engine can overlap.** It ran alongside decode GEMVs at 27 GB/s with 2%
  slowdown, so its transfers must start well before the consuming layer.

Speculative verification of several tokens multiplies distinct misses because each
column routes independently. Routes beyond the directory set the failure flag and
produce null uses; check the flag before trusting a step's output. A higher hit rate alone does not prove a faster
engine: miss sizes, dispatch, copy contention and admission overhead can reverse it.

[device_expert_cache_tests.cpp](../../tests/inference/device_expert_cache_tests.cpp)
(under `CRIT_INFERENCE_CUDA`) runs 600 random plans with lookahead prefetch on 24 and
72 slots against a host replica, checking slots, counters, pool tags, bundle bytes,
prediction lists longer than the pool, a resident prediction taking no pool slot,
duplicate-route rejection and invalid launches. It then runs `ExpertCache` on 20 and 40
slots: a profile preload with a duplicate and an invalid key, then 400 plans under a
quota table and second-touch. It checks slots, bypassed misses naming their store
bundle, copied bytes and the telemetry against the replica, and the provider GEMM operands
`enqueue_operands` writes. All of it passes on an RTX 5090.
[expert_store_tests.cpp](../../tests/inference/expert_store_tests.cpp) pins a store file and reads it
back through the device address `pin_store` returns. The host suite checks
`ClockDirectory` pinning, victims, pool tags, admission limits and preload, plus the
profile builder and the split table.
[Measured policy fixtures](../../tests/test_measured_expert_policies.py) compare replay
clock with `ClockDirectory`, check hindsight against an exhaustive oracle, and cover
sizes, pins, bypass, warmup and the event types. None of these measure
performance.
