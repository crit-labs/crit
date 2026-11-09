# CPU expert tier

Compute a routed layer's CPU share of experts on a pinned worker pool, in the integer block order of
[host-expert-arithmetic](../host-expert-arithmetic/README.md#integer-block-rows), so an expert computed on the CPU
gives exactly the device's bits and the split between the tiers never changes a token. The entry is the ready
component an engine calls instead of writing its own pool, handoff and expert composition: those details decide
whether a CPU tier beats copying misses over the link or loses to it.

## When to use it

Use it when experts do not all fit on the device and `crit eval probe`'s CPU expert lane (`host_expert_gbs`), alone or
beside link copies, beats the link: recon's placement prints both rates and the plain-pass bounds they give. On such a
machine candidate 1 starts with this tier beside the [device expert cache](../device-expert-cache/README.md), not
with every miss copied. A typical desktop reads host memory at about twice its PCIe link's rate.

Don't use it when every expert fits on the device, or when the probed lane and a measured layer give the CPU no share.

## How it works

**The pool.** `ExpertPool` keeps one worker per physical core minus the caller's core (the host loop and the
mailbox's service thread), pinned on Linux to the first hardware thread of each core. A layer is one flat batch: a
task is a row range of one expert, claimed with one compare-and-swap of a word that carries the batch's epoch, so a
worker still leaving an earlier batch claims nothing of the next. The caller drains too. An idle worker spins (with
a pause hint) as long as the recent gaps between batches, 1.5 times the longest of the last 16, at least `spin_us` and
at most `spin_max_us` (3 ms by default, measured on one machine), then sleeps until the next batch: awake through a
decode step's layers, a millisecond or so apart, asleep between steps. A batch published to sleeping workers pays one
wake-up, tens to hundreds of microseconds under a loaded or virtualized OS (the pool counts them). Workers pin to cores
of the process's CPU set after its first, none another pool of the process holds: give each engine process its own
cores (taskset, cgroups) so processes never share one.

**A layer.** `HostExpertTier::compute` groups the layer's CPU routes by expert (up to 8 tokens a chunk) and runs two
pooled phases: gate/up rows for each chunk (`dot_rows_int`, every expert row read once for all of its tokens), a task
taking gate rows k0..k1 and the up rows beside them, so it applies the declared SwiGLU and quantizes its own blocks of
the intermediate; then the down rows into each route's output row. Inputs come quantized (`compute`, as a device
quantized them) or as FP32 token rows (`compute_inputs`), which the tier quantizes per route with that route's expert's
declared quantizer and input global: the same codes. A matrix's post is one number or one per row (a bundle's
1 / global per row times the integer values' factor): a row is computed with post 1 and then multiplied by its own,
which is one rounding, the device's `v * post[n]`. Experts live in a table the engine fills (`set_expert`), views of
bytes it owns: a bundle read in place, rows layout as stored.
Outputs are per route and unweighted: the device multiplies by the routing weight and sums in its own plan order
([async-expert-contributions](../async-expert-contributions/README.md)), so the order of a token's sum never depends
on the split. Any pool size, ISA or store layout gives the same bits.

**The handoff.** The device posts a layer's CPU routes into a [host-work-mailbox](../host-work-mailbox/README.md) slot
laid out as `input_layout(hidden, routes, tokens)`: a header {count, tokens, route capacity, token capacity}, each
route's expert key and token, the tokens' FP32 rows, room for the output rows; the mailbox handler is
`crit_tier_serve` (tier_capi.cpp), which calls `serve_inputs` ([heterogeneous-expert-dispatch](../heterogeneous-expert-dispatch/README.md)'s
CUDA form wires it). Join with the poll kernel where
`measure_joins()` says so (on one RTX 5090 under WSL2 a stream wait cost 50-60 µs more per join than the 2.8 µs poll,
about 5 ms a round at 48 layers), and post before the device's own expert work so the two overlap.

**The split.** Which misses go to the CPU is the device expert cache's admission (`set_admission` with a quota by miss
count from `split.py`): give `split.py` this tier's measured layer times (`stats()`) and the link's, both measured
with the other running, since both read host memory.

**The store.** The tier reads each expert's bundle in place where the bundle store keeps it (the experts slot's
layout: codes row-major as the checkpoint stores them, E4M3 scales, each row's 1 / global, the input scales), so the
experts are not held twice in host memory; a bundle store of another encoding is refused. The rows layout reads at
about 19-21 GB/s on 5 threads of one desktop CPU, the interleaved one (`interleave_matrix`) at 28-30, for a CPU-only
store that can keep that layout.

## The plugin

On the CPU (`devices: ["cpu"]`) the experts slot's device form runs this tier: `load(layer, expert, gate_up, down,
format)` packs a stored `nvfp4` or `int4` expert (`{weight, scale, global_scale, input_scale}` by matrix) as a bundle in
the experts slot's layout (`bundles.py`, the experts plugin's layout() for these formats) and adds it to the tier's
table; `forward(layer, x, ids, weights)` computes every routed (id >= 0) route's unweighted row and sums each token's
routes in route order by FMAs (host-expert-arithmetic's combine, the device experts' declared combine). NVFP4 runs
W4A4 (inputs E2M1 x 2 at the gate/up input_scale, the intermediate at the down's, post 0.25 / global per row), INT4
W4A8 (INT8 per 32). It provides `experts:bundles`: `layout`, `pack` and `forward_bundles` over a CPU uint8 store
[S][bundle_bytes] and a slot table, read in place. The native tier (`cpu_tier.py`: host-expert-arithmetic's
`native.py` builds `tier_capi.cpp`, the tier and the host rows into one library at load; ctypes calls it) runs where it
builds; elsewhere, or with `native: off`, the same order runs in torch (host-expert-arithmetic's `declared.py`), with the
same bits. A token's result never depends on the rows beside it. Its facts name the path (`cpu_expert_rows`), its
telemetry the tier's counters.

## What the code supplies

| File | Supplies |
| --- | --- |
| [`expert_pool.h`](code/expert_pool.h), [`expert_pool.cpp`](code/expert_pool.cpp) | `ExpertPool`, `PoolOptions`, `physical_cores` |
| [`host_tier.h`](code/host_tier.h), [`host_tier.cpp`](code/host_tier.cpp) | `TierExpert`, `TierBatch`, `InputBatch`, `HostExpertTier` (`set_expert`, `compute`, `compute_inputs`, `input_layout`, `serve_inputs`, `stats`), `interleave_matrix` |
| [`implementation.py`](code/implementation.py) | Python model of the epoch-tagged claim, the chunking and the plan-order sum |
| [`tier_capi.cpp`](code/tier_capi.cpp) | A C interface over the tier and its pool for ctypes, and `crit_tier_serve`, the host-work-mailbox handler |
| [`bundles.py`](code/bundles.py) | The nvfp4 / int4 bundle layout and pack the tier reads in place; the declared arithmetic of each format |
| [`cpu_tier.py`](code/cpu_tier.py) | `CpuTier`: the expert table, the native tier or the torch declared order, the handler's address and payload layout |

## What the engine supplies

- The experts in pinned host memory with their scale decoders and post factors, the interleaved copy for the CPU.
- A device kernel that quantizes the layer's input as declared, writes the payload, posts and joins, and sums the
  rows with routing weights in plan order.
- The CPU share from measurement on the target, and a check that the tier's rows match its device kernels there
  (host-expert-arithmetic's CUDA fixture shows how).

## Worked example

Run from `code/`. A worker that read the pool's head in one batch and lands its claim in the next fails the swap
with the epoch in the word and runs the wrong batch's task without it. The tier reads an expert with 10 tokens in
two chunks and keeps one row per route. A host-side partial sum over the CPU's routes changes a token's bits,
which is why the tier returns unweighted rows and the device sums them in plan order.

```python
import random
from implementation import Head, UntaggedHead, bits, chunks, device_sum, f32, host_partial, tier_rows

# A worker reads the head two tasks into batch 1 and stalls; batch 1 drains, batch 2 starts and is two tasks in
# when the worker's swap lands. With the epoch in the word the swap fails; without it the worker runs batch 1's
# function on batch 2's task 2.
for head_type, stale in ((Head, None), (UntaggedHead, 2)):
    head = head_type()
    head.publish(1, 4)
    head.claim(1, head.word), head.claim(1, head.word)
    seen = head.word
    head.claim(1, head.word), head.claim(1, head.word)
    head.publish(2, 4)
    head.claim(2, head.word), head.claim(2, head.word)
    assert head.claim(1, seen) == stale

# 21 routes over 6 experts, 10 of them on expert 2: the tier reads expert 2 twice (8 + 2 tokens), every other
# expert once, and every route keeps its own row.
rng = random.Random(3)
experts = [2] * 10 + [rng.choice([0, 1, 3, 4, 5]) for _ in range(11)]
groups = chunks(experts)
assert [len(r) for e, r in groups if e == 2] == [8, 2] and len(groups) == len(set(experts)) + 1
assert sorted(r for _, rs in groups for r in rs) == list(range(21))
rows = [[f32(rng.gauss(0.0, 1.0)) for _ in range(64)] for _ in experts]
assert tier_rows(experts, lambda e, r: rows[r]) == rows

# A token's 6 experts: the device sums weight x row in its plan order, wherever each row was computed. A host-side
# partial sum of the CPU's 3 routes, added to the device's, reorders that sum: close in value, other bits.
plan = [0, 11, 12, 13, 14, 15]
weights = [f32(rng.random()) for _ in experts]
expect = device_sum(plan, weights, rows)
cpu, gpu = [11, 13, 15], [0, 12, 14]
mixed = [f32(a + b) for a, b in zip(device_sum(gpu, weights, rows), host_partial(cpu, weights, rows))]
assert any(bits(a) != bits(b) for a, b in zip(mixed, expect))
assert all(abs(a - b) <= 1e-5 * max(1.0, abs(b)) for a, b in zip(mixed, expect))
```

## Adapting it

Preserve one declared arithmetic on every processor (host_int's integer rows, `swiglu_declared`, the intermediate's
declared quantizer), unweighted per-route rows summed on the device in plan order, the epoch in the claim word,
and a core left to the caller. You can change rows per task, the worker count and pinning, the payload's route
capacity and the expert's composition, as long as the device declares the same.

## Costs and validation

For scale only: on one Ryzen 5 9600 (6 threads, WSL2, AVX-512 VNNI) a layer of 6 experts of 2.76 MB (hidden 2560, inter
640) took 0.58-0.62, 0.72-0.77 and 0.89 ms at 1, 2 and 3 tokens an expert, 28-30, 23-24 and 20 GB/s of experts (the
interleaved store, quantized inputs), against about 0.11 ms an expert over a 24.5 GB/s link. Bundles read in place
(the rows layout) with FP32 inputs reached 146-154 µs an expert at 3-4 experts and one token on a machine busy with
other work; time the tier where it serves (the dispatch's `quota_table` does). The quantizers find a block's codes by a
binary search over the E4M3 values and the E2M1 midpoints, the codes crit::nvfp4's linear searches give.

`crit_host_expert_tier_tests` prints its own rates, and checks a layer's routes against a serial computation bit for
bit for pools of 0, 2 and 5 workers, scalar and the fastest ISA, both store layouts and both intermediate formats, per-row
posts with FP32 inputs over 36 configurations, the payload, and `N` pools started and stopped (argument, default
300; 3,000 ran 253,500 batches, 183,170 of them published to sleeping workers) with every task run once in its own batch.
The plugin's tests check the device form against the torch declared order and the dense FP64 block, the native tier
against the torch path, a token alone against its batched bits and a store read in place against loaded experts. On an
RTX 5090 the native tier gave the int-block-experts device kernels' bits at hidden 2560 / inter 640 for 1, 3 and 8 tokens,
on random experts and on real NVFP4 checkpoint experts.
