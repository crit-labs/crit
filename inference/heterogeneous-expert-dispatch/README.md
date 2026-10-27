# Heterogeneous placement and exact expert dispatch

This entry explores where routed experts execute and where their weight bytes
come from. It supplies static placement feasibility, measured host/device split
planning, occurrence-preserving contribution splitting, one projection that
resolves cached, staged and original sources, and two fused CUDA operations for
the router and shared expert. Each is an optional ingredient.

## When to use it

Use it when routed experts do not all fit on the device and the engine must
decide placement, split work between host and device, or read some experts from a
cache and others from their original source. Choose by actual memory pressure and
operation costs: static placement can be simple and effective, a cache may adapt
better to route locality, and a source-aware projection avoids duplicated work
from splitting an operation.

## How it works

**Static feasibility.** Given ordered indivisible allocation groups, enumerate
host prefixes and device suffixes that fit after fixed device state and headroom.
This answers which placements fit a snapshot; it does not rank speed. Extents
must include padding, repacking and aliases.

**Split execution.** Partition route occurrences between a device path and a host
path, preserving position and weight on both. Device capacity here limits route
occurrences, not distinct experts. Contributions are weighted before combination,
so repeated expert IDs contribute at every occurrence.

**Measured split planning.** Keep protected resident experts on the device, order
unique misses, and consider every CPU-prefix/device-suffix split; all occurrences
of one expert stay together. Each upload is charged once, each occurrence counted
for arithmetic, and splits over the staging bytes or a lane's occurrence limit are
rejected. Each split's estimate uses the median retained sample at the exact
counts: `common + max(cpu, upload + device)` with CPU/device overlap, otherwise
`common + cpu + upload + device`. There is no assumed bandwidth, linear scaling or
interpolation; unmeasured splits stay visible but cannot win, and ties keep the
split with the fewest CPU experts.

**Mixed sources.** Each route uses a protected READY cache slot when available,
else one scratch slot per unique miss while scratch remains, else the exact
original expert as stock fallback. Repeated occurrences share one selection and
staging demand but keep their positions and weights. Scratch cannot overlap a
READY slot. A native backend can resolve a source descriptor per occurrence and
run one projection over one prepared activation, without two graphs or two
activation quantizations.

## What the code supplies

| File | Supplies |
| --- | --- |
| [`implementation.py`](code/implementation.py) | `ResidencyGroup`, `ResidencyPlan`, `prefix_residency_plans`; `Route`, `partition`, `combine`; `ReadySlot`, `MixedRoute`, `MixedSourcePlan`, `mixed_source_plan`; `execute_mixed`, which calls `quantize(activations)` once then `project(encoded, plan)` once |
| [`measured_dispatch.py`](code/measured_dispatch.py) | `CalibrationKey`, `TimingCurve`, `DispatchCalibration`, `plan_dispatch` returning a `DispatchDecision` of every `DispatchCandidate`; optional `HostMemoryCosts`, `admit_host` and `revalidate_memory` |
| [`routing.py`](code/routing.py) | `Routing`, one description of top-k routing; `route`, its host reference for one token; `routing_from_config`, which maps the config.json fields Core parses |
| [fused_moe.h](code/fused_moe.h) | CUDA `entry_cuda` (FP8 router plus shared gate/up in one grid, BF16-rounded `SiLU(gate) * up`) and `route_down_cuda` (routing under a `crit::moe::Routing` beside BF16 shared-down) for batches of 1 to 8 tokens |

`decision.best` is `None` when no feasible measured split exists; the engine then
uses its ordinary exact path. Rejected splits keep their bytes and reasons. The
planner performs no tuning or file writes. `execute_mixed` is an orchestration
boundary, not a hidden kernel.

**Routing.** One `Routing` describes how a token's router logits become expert IDs
and weights, and the CUDA kernel reads the same fields:

| Field | Meaning |
| --- | --- |
| `scoring` | `softmax` or `sigmoid` |
| `score_before_selection` | Score every expert and select on the scores; otherwise select on raw logits and score only the selected (softmax over them, or sigmoid of each) |
| `correction_bias` | A per-expert bias added to the selection value only; weights use the unbiased scores |
| `groups`, `selected_groups`, `group_score_top` | Equal contiguous expert groups, each ranked by the sum of its `group_score_top` best selection values; only the best `selected_groups` groups are selectable |
| `renormalize`, `renormalize_epsilon`, `renormalize_floor` | Divide the selected weights by `max(sum + epsilon, floor)` |
| `scaling` | Multiply every final weight |

Ties keep the lower expert ID, and the lower group index. IDs come back in selection
order. The defaults, softmax over the top-k raw logits, are the kernel's previous
behavior bit for bit. The same fields describe, for example, Mixtral's renormalized
softmax, Qwen-MoE's `norm_topk_prob`, GPT-OSS's and Llama 4's scoring after
selection, and DeepSeek-V3's sigmoid with a correction bias, eight groups ranked by
their two best values, renormalization and 2.5x scaling (GLM-4.5 and Kimi K2 use the
same form with one group). The host fixtures check each against the transformers
router it names.

`routing_from_config` reads Core's `scoring_func`, `norm_topk_prob`,
`routed_scaling_factor`, `n_group`, `topk_group` and `topk_method`. A field the
config omits comes from `defaults`, the value the model's reference code assumes,
and fails when neither gives it, because references disagree (Mixtral always
renormalizes; a Qwen-MoE config without `norm_topk_prob` does not). The engine also
states `score_before_selection`, which no config declares, and whether the
checkpoint has an `e_score_correction_bias` tensor. `topk_method` sets the group
rank: `group_limited_greedy` ranks by the best value, `noaux_tc` by the best two and
requires the bias, `greedy` ignores groups. Not covered: scoring other than softmax
and sigmoid (for example `sqrtsoftplus`), and a renormalization floor stored under a
config key Core does not parse (Ernie 4.5's `moe_norm_min`), which the engine passes
as `renormalize_floor`.

**Native fusion.** The fused routing supports 1 to 1024 experts, up to 32 selected
experts and up to 32 selected groups, and returns original expert IDs, weights under
the `Routing` and an unweighted FP32 shared-expert output. A NaN selection value ranks
as -inf, so a NaN row (a padding row, or a NaN that reached the router) still returns
in-range IDs; the host reference rejects non-finite logits. Its arithmetic is one
example instance: an FP8 router and FP8 shared gate/up in the row codec of
[quantized projections](../quantized-projections/README.md), BF16-rounded SiLU
products and a BF16 shared down projection. Other layouts keep the `Routing`
contract and supply their own projections. The model's shared-expert gate and
routed-expert evaluation are not included. Build with an explicit
`CMAKE_CUDA_ARCHITECTURES` (SM80+).

## What the engine supplies

Quantization, projection, copies, device source resolution and completion
receipts for mixed experts. Every source must produce the same expert values under
the chosen arithmetic; cache and scratch indices are storage locations, never
replacement expert IDs. Protect all READY hits before choosing replacement or
scratch storage, and stage every component in `plan.demands` before projection
reads it. If source formats differ, dispatch compatible decode explicitly, and keep
the original-source fallback executable when scratch runs out.

For the fused CUDA operations: validated FP8/BF16 payloads, finite router
scores and correction bias, the model's `Routing`, owners retained until the stream
settles, and input dependencies and output publication. No host readback or
synchronization is inserted.

For measured dispatch: a complete current `CalibrationKey` (model content,
representation, workload, implementation and hardware topology, including shapes,
route multiplicity, residency, concurrency, CPU threads and transfer settings).
Selection rejects a different key; invalidate samples when inputs change or
timings drift. Measure warmed lanes under representative simultaneous traffic,
with upload samples covering the real copy grouping, not only total bytes. Use
[recorded selection](../recorded-selection/README.md) when a lane-cost model does
not fit.

`HostMemoryCosts` takes explicit peak-byte tables (CPU scratch by occurrence
count, pinned staging by upload bytes, per-expert and per-upload loader bytes and
file pages, common overhead) bound to the same identity. Missing estimates fail
closed; use explicit zeros for resident sources. Compose `admit_host` with
`admit_working_set` from [host-memory-admission](../host-memory-admission/README.md)
and call `revalidate_memory(candidate, callback)` with a fresh observation just
before allocating; on denial, replan or use the exact fallback. No planner result
reserves memory or proves device readiness.

## Worked example

Run from `code/`. One cache hit is reused twice, one miss gets scratch, and the
remaining misses keep their original IDs. No expert is dropped or wrapped.

```python
from implementation import Route, ReadySlot, mixed_source_plan, execute_mixed

routes = (Route(0, 9, 0.5), Route(1, 2, 1.0), Route(2, 9, 0.25),
          Route(3, 7, 1.0), Route(4, 8, 1.0))
plan = mixed_source_plan(routes, {9: ReadySlot(0, 3)},
                         capacity=2, scratch_slots=(1,))
assert plan.demands == ((2, 1),)
assert [(r.source, r.index) for r in plan.routes] == [
    ("cache", 0), ("scratch", 1), ("cache", 0), ("stock", 7), ("stock", 8)]

# A small host projection hook makes source identity and weights observable.
stock = {9: (2.0, 4.0), 2: (1.0, 3.0), 7: (5.0, 1.0), 8: (2.0, 2.0)}
cache, scratch, prepared = {0: stock[9]}, {}, []
def quantize(activation):
    prepared.append(tuple(activation))
    return tuple(activation)  # this fixture deliberately uses plain host values
def project(activation, selected):
    for expert, slot in selected.demands:
        scratch[slot] = stock[expert]
    outputs = []
    for resolved in selected.routes:
        sources = {"cache": cache, "scratch": scratch, "stock": stock}
        weight_row = sources[resolved.source][resolved.index]
        value = sum(a * w for a, w in zip(activation, weight_row))
        outputs.append((resolved.route.position, value * resolved.route.weight))
    return tuple(outputs)

assert execute_mixed(plan, (1.0, 2.0), quantize, project) == (
    (0, 5.0), (1, 7.0), (2, 2.5), (3, 7.0), (4, 6.0))
assert prepared == [(1.0, 2.0)]

# Synthetic calibration: the same host planner accepts any measured hardware.
from measured_dispatch import CalibrationKey, TimingCurve, DispatchCalibration, plan_dispatch

identity = CalibrationKey("weights-v1", "layout-v1", "uniform-row-workload-v1",
                          "kernels-and-config-v1", "cpu-device-topology-v1")
def curve(points):
    return TimingCurve.from_samples({count: (ms,) for count, ms in points.items()})

calibration = DispatchCalibration(identity,
    cpu=curve({0: 0, 1: 4, 2: 8, 3: 12}),
    upload=curve({0: 0, 7: 2, 14: 4, 21: 6}),
    device=curve({0: 0, 1: 1, 2: 2, 3: 3, 4: 4}),
    common_samples_ms=(0.5,), overlap="cpu-device")
work = tuple(Route(i, i, 1.0) for i in range(4))
decision = plan_dispatch(work, {0}, {1: 7, 2: 7, 3: 7}, calibration,
                         identity=identity, staging_budget_bytes=21)
assert tuple(r.expert for r in decision.best.host) == (1,)
assert decision.best.estimated_ms == 7.5
# Byte pressure rules out that split; two misses now execute on the CPU.
tight = plan_dispatch(work, {0}, {1: 7, 2: 7, 3: 7}, calibration,
                      identity=identity, staging_budget_bytes=7)
assert tuple(r.expert for r in tight.best.host) == (1, 2)

# Compose OS admission with exact measured dispatch, using synthetic observations.
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1] / "host-memory-admission" / "code"))
from memory import Observation, admit_working_set
from measured_dispatch import HostMemoryCosts, revalidate_memory

costs = HostMemoryCosts(identity,
    scratch=((0, 0), (1, 16), (2, 32), (3, 48)),
    staging=((0, 0), (7, 7), (14, 14), (21, 21)),
    experts=((1, 0, 1024), (2, 0, 1024), (3, 0, 1024)),
    upload_sources=((1, 0, 0), (2, 0, 0), (3, 0, 0)))
seen = Observation(physical_available=256, commit_available=128)
def admission(**extents):
    return admit_working_set(**extents, observation=seen)

safe = plan_dispatch(work, {0}, {1: 7, 2: 7, 3: 7}, calibration,
                     identity=identity, staging_budget_bytes=21,
                     memory_costs=costs, admit_host=admission)
assert safe.best is not None and not safe.best.host
assert revalidate_memory(safe.best, admission)
seen = Observation(physical_available=10, commit_available=128)
assert not revalidate_memory(safe.best, admission)  # replan before allocating

# Routing for one token, with DeepSeek-V3's form on four groups of two experts
# (synthetic values).
from routing import Routing, route

rule = Routing(scoring="sigmoid", score_before_selection=True, correction_bias=True,
               groups=4, selected_groups=3, group_score_top=2, renormalize=True,
               renormalize_epsilon=1e-20, scaling=2.5)
ids, weights = route((2.0, 1.0, -1.0, 0.0, 3.0, -2.0, 0.5, 0.5), 2, rule,
                     bias=(0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0))
assert ids == (0, 1)  # without the bias, group 2 is selectable and expert 4 ranks first
assert abs(sum(weights) - 2.5) < 1e-12
assert route((1.0, 3.0, 2.0), 2) == ((1, 2), route((3.0, 2.0), 2)[1])
```

## Adapting it

Preserve every route occurrence, exact expert identity, compatible
representation, protected source generations, complete staging and readiness
before use. Placement order, replacement policy, scratch count and where
resolution runs are choices.

Substituting a different resident expert for a missing one changes the routed
computation and is outside `mixed_source_plan`'s exact contract. An approximate
consumer must keep the original routes for comparison and declare its replacement
selection, gate reweighting and quality criteria separately, with the exact
fallback still executable. [Residency](../expert-residency/README.md) supplies
lifetime rules; [asynchronous contributions](../async-expert-contributions/README.md)
supplies a join protocol when splitting pays.

**Exact split.** When host workers compute missed experts, give them the device
kernel's exact per-row operation order, so an expert's contribution is bitwise the
same on either processor and output does not depend on the split, the cache state or
host load. [Host expert arithmetic](../host-expert-arithmetic/README.md) supplies that
for NVFP4: a declared order, host paths matching it bit for bit and the device
function that declares the same one.
For prompt chunks, choose host experts by a fixed function of (layer, expert), never
by what happens to be resident: a cache-dependent split changes a token's arithmetic
between two identical requests. A miss the host computes is not admitted to the
device cache, so the miss rate rises with the host share; weigh that against the
transfer it saves.

**Host work inside a captured decode.** To run the host share mid-graph, post it
through the [host-work mailbox](../host-work-mailbox/README.md), which supplies the
doorbell protocol, measured join choice, release and failure answering, and the
rules for host worker pools fed from the device.

## Costs and validation

Placement feasibility is not a throughput model. Splitting can gain from overlap
but adds transfers, launches and reduction; mixed sources can add irregular access
or slower fallback reads. Measure miss bytes and complete operation time. Host
execution of missed experts reads the same DRAM as concurrent host-to-device
copies, so measure host latency while those copies run. Spin barriers across host
workers stall when one thread is descheduled; leave cores for the launching
thread. Choose the host share of each pass's misses from measurements on the target
machine, never a fixed fraction: host workers add per-expert compute time until they
saturate host DRAM bandwidth, the device pays each miss's transfer at the measured
host-to-device rate (`crit eval probe`), and past the share where host time exceeds
the transfer it replaces, decode slows down. Sweep the share on the mixed suite. Where
the first CPU read of host memory registered with the device is slow (seen under
WSL2), touch every page once at load (Core's `HostBlock::touch`). (For scale only: on
one machine with 6 Ryzen 5 9600 cores, an engine with 3.1 MB NVFP4 experts reached
about 36 GB/s with six threads, near the host's DRAM rate, and gained most with about
60% of its misses on the CPU, while a second engine on the same machine measured its
CPU path slower than the transfer. The same CPU can be worth a large share for one
model and none for another.)

Measured prefix planning is quadratic in route count and prices one
operation, not future cache warmth. Common errors: treating occurrence capacity as
cache capacity, applying weights twice, remapping IDs modulo cache size, and
aliasing scratch with live content.

[CUDA fixtures](../../tests/inference/cuda/kernels.cpp) compare 24 fused cases
with independent host arithmetic and check top-k identities, normalization, ties
and aliases. They also run 12 routing variants (scoring before and after
selection, bias, groups, renormalization, floor, scaling and ties, up to 1024
experts) against expectations that [routing_cases.py](../../tests/inference/cuda/routing_cases.py)
writes from the host reference. [Routing fixtures](../../tests/test_expert_routing.py)
check the host reference against hand-worked cases and, when transformers is
installed, against the Mixtral, Qwen2/Qwen3-MoE, GPT-OSS, Llama 4, DeepSeek-V2/V3,
GLM-4.5, MiniMax-M2, Ernie 4.5 and LongCat-Flash router code, plus the config
mapping. Host fixtures cover placement feasibility, duplicate occurrences,
ordered combination, staging demands, exact fallback and single activation
preparation. [Measured policy fixtures](../../tests/test_measured_expert_policies.py)
cover stale identities, nonlinear curves, overlap, capacity limits, memory
admission and unchanged weighted results across splits. None validates mixed
cache/stock expert kernels, full-model quality or a CPU/GPU speedup.
