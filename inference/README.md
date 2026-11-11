# crit Inference

Inference holds crit's reusable inference techniques. Each entry explains a
mechanism, supplies actual code and states what an engine must preserve when it
adapts that code. Entries are techniques, not compute kernels: the GEMMs, attention,
normalization and scan kernels an engine runs come from crit's kernel templates in
[Kernels](../kernels/README.md) (built for the engine's device and tuned by `crit kernels sweep`), from
libraries such as cuBLASLt, CUTLASS, FlashInfer and flash-linear-attention, or the
engine writes them. A library plugin that needs a kernel no provider has loads one of
crit's kernels declared in its plugin.json and keeps a torch fallback, and mechanism code that runs on the device (a cache directory, a
mailbox, a round's accept, a CPU SIMD codec) stays in its technique
([kernels and plugins](../docs/PLUGINS.md#kernels-and-plugins)). Agents copy, specialize,
combine or fuse entries; the engine owns that design, and an engine may write every plugin
itself instead of taking an entry.

Use [Core](../core/README.md) for lifecycle, requests and resources,
[Kernels](../kernels/README.md) to find and integrate kernels (external links never
substitute for code an entry claims to supply), and [Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md)
to measure the engine; NLL, perplexity and KL code is in
[Eval measurements](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md). [crit serve](../docs/SERVING.md)
never selects an Inference technique. Engines can adapt artifact-loading,
expert-residency and Core's optional
[CUDA graph and shared-allocation services](../core/src/backends/cuda/README.md#graph-capture)
while preserving content identity, byte ownership and drain-before-release.

## Find the useful entry

Start from the problem, then check what the entry's code supplies and what it
leaves to the engine.

| Need | Entry | Supplies |
| --- | --- | --- |
| Load exact encoded regions | [artifact-loading](artifact-loading/README.md) | Extent/content checks, retained materialization, native chunked reads |
| Check host memory before large allocations | [host-memory-admission](host-memory-admission/README.md) | Physical, commit and cgroup admission; incremental working sets |
| Audit imported model metadata | [model-import-audit](model-import-audit/README.md) | Safetensors header/index checks, tensor roles, shard geometry |
| Reuse serialized state | [checkpoint-storage](checkpoint-storage/README.md) | Complete manifests, prefix coordination, paced transfers, leased tiers |
| Recover speculative recurrence | [recurrent-state-recovery](recurrent-state-recovery/README.md) | Gated-delta verification, accepted-prefix repair, chunked prefill reference; on the device a state pool whose verify windows keep their accepted prefix in captured rounds (deferred, checkpoints or recompute) |
| Share KV and execute attention | [attention-state](attention-state/README.md) | FP32/int8 COW pages for Core's state cache, a host tier for evicted pages, tiled CPU attention, GQA, masks, RoPE; on the device a paged pool with FP8 and INT8 row-scaled pages, round slots for captured rounds and prompt buffers for captured prompt chunks, and the attention slot's shared device form (code/device_attention.py) |
| Partition a long attention sequence | [partitioned-attention](partitioned-attention/README.md) | Interleaved ownership, stable partial reduction, head-major scatter; serves on CUDA: split-KV partitions merged by log-sum-exp on one GPU |
| Execute low-bit projections | [quantized-projections](quantized-projections/README.md) | G32 W4/W8 and dynamic A8 on the CPU, row-scaled FP8 and NVFP4 host codecs; on CUDA stored BF16, FP8, MXFP8 and NVFP4 matrices through cuBLASLt and crit's kernels mma-rows, decode-gemv, fixed-tile-gemm and pinned-gemm, each decode row's bits alone and batched |
| Reuse expert weights, lend cache memory or borrow idle KV and workspace memory | [expert-residency](expert-residency/README.md) | Byte-cache leases, complete bundles, C++ region loans and owner slot loans |
| Plan expert residency inside decode graphs | [device-expert-cache](device-expert-cache/README.md) | Ready `ExpertCache` (the `experts:cache` device form): CUDA directory, miss copies, admission limits and the CPU/link split, a prompt ring, lent KV pages, profile warm start, prefetch, offline policy replay |
| Run routed experts in their stored format on the device | [grouped-experts](grouped-experts/README.md) | Routed SwiGLU experts over bundles in a slot store, every stored format (bf16, fp8, nvfp4, int4, mxfp4), the routes grouped by expert on the device; one function for decode rows and prompt chunks (registry kernel crit-labs/grouped-experts, torch fallback) |
| Read prompts through routed experts kept in host memory | [expert-streamed-prefill](expert-streamed-prefill/README.md) | Chunk size from the measured rate curve, layer double-buffered expert copies, the FP64 reference and the grouped projection's contract (`operations.json`); streamed.py runs a prompt's routed layers through an experts plugin's `forward_bundles` in chunks at the measured knee (no plugin runs it yet) |
| Choose expert execution sources | [heterogeneous-expert-dispatch](heterogeneous-expert-dispatch/README.md) | Placement plans, measured CPU/device splits, one routing description with its host reference; serves on CUDA: the expert cache's CPU routes (slot -2) computed by the CPU tier and joined inside the stream, the split never changing a bit |
| Compute NVFP4 experts on the CPU with the device's bits | [host-expert-arithmetic](host-expert-arithmetic/README.md) | Declared FP32 orders and exact integer block rows, scalar oracles, AVX2/AVX-512 (VNNI) paths, matching CUDA functions, scale hook; serves the projections slot on the CPU |
| Compute a layer's CPU share of experts on a pinned pool | [host-expert-tier](host-expert-tier/README.md) | Ready `HostExpertTier` and `ExpertPool`: integer rows with the device's bits, declared SwiGLU and quantizer, mailbox payload, telemetry; serves the experts slot on the CPU and joins a CUDA engine's -2 routes inside the stream (heterogeneous-expert-dispatch) |
| Join asynchronous contributions | [async-expert-contributions](async-expert-contributions/README.md) | Receipt-based join, source retention, ordered reduction |
| Run device-decided host work inside a captured graph | [host-work-mailbox](host-work-mailbox/README.md) | Device-reset doorbells, measured stream-wait or poll join, release and failure answering |
| Serve a dense model larger than device memory | [dense-layer-offload](dense-layer-offload/README.md) | Measured planner, host AVX2/AVX-512 layers, double-buffered weight streaming, measured plan selection |
| Fetch file-backed experts | [ssd-expert-materialization](ssd-expert-materialization/README.md) | Checked reads, a host tier fed by device evictions, bounded gather, read-ahead, upload handoff |
| Read embedding or n-gram table rows from disk | [lookup-table-rows](lookup-table-rows/README.md) | Random-access advice, exact parallel row reads, hot-row cache; on CUDA tables in registered host memory or files, gathered inside captured graphs |
| Bound prefill, physical batches and scratch | [session-scheduling](session-scheduling/README.md) | Transactional cursor, latency chunk policy, physical caps and workspace sizing |
| Batch independent requests | [concurrent-request-execution](concurrent-request-execution/README.md) | Ready-row compaction, metadata patches, retirement barrier |
| Execute speculative rounds | [mtp-decode](mtp-decode/README.md) | Token-history proposals, greedy/sampled verification, repair, adaptive budgets; copy drafts from device token histories (`draft:device`) |
| Sample tokens on the device, and verify sampled drafts | [token-sampling](token-sampling/README.md) | Core's whole law over the full vocabulary (penalties, top-k/p, min-p), rejection-sampling verify predictions; Core's Sampler as oracle; `rounds:sampler` for captured rounds (draft draws with sparse laws, rejection-sampling predictions, emitted-token log-probabilities) |
| Keep structured output exact under speculation | [token-constraints](token-constraints/README.md) | [Core](../core/src/constraints/README.md) matchers in sampled speculation, private forks and rewinds; serves on CUDA: masks before token-sampling's device law, staged for captured steps and rounds |
| Keep a speculative round on the GPU | [device-speculative-round](device-speculative-round/README.md) | Device accept for greedy and sampled rounds, frontier advance, graphs, one readback per round |
| Verify a tree of drafts in one pass | [draft-trees](draft-trees/README.md) | EAGLE-2 trees, ancestor masks, device tree walk (greedy or sampled), row compaction; device tree drafting and walk in captured rounds (`draft:device`, `verify:device`; its rounds need `attention:tree`) |
| Draft with a published EAGLE-3, DFlash or DSpark drafter | [hidden-state-drafters](hidden-state-drafters/README.md) | Core's reader of every drafter format (one tap convention), catch-up, block input, d2t mapping, confidence cuts |
| Cheapen the draft head | [draft-vocabulary-subset](draft-vocabulary-subset/README.md) | Head row subset, index mapping, CUDA gather-argmax |
| Select compatible execution and retain choices | [recorded-selection](recorded-selection/README.md) | Phase/shape replay policy, recorded choices, repeatability checks |
| Fuse the elementwise work around the matmuls | [fused-norms](fused-norms/README.md) | RMSNorm with a residual, head norms with rotary positions, gated norms, activation products, L2 norms, residual-stream mixes as the `norms` slot: one launch per call of crit's kernel fused-norms, each row's bits independent of its batch; the FP64 host form, the torch form (base torch-norms, and the fallback) and conformance checks for any norms plugin |

Each entry has a Python host reference; the entries that fill a slot also serve on a device as plugins, and some add
C++ or CUDA mechanism code. Each entry states its tested scope and missing native or engine work. Host tests establish
tested semantics, not an integration or a speedup. Before adapting, name the
engine gap, check overlap with existing code and define the comparison that would
justify the cost; the answer may be no new code.

```text
crit inference list
crit inference search "exact fallback" --json
crit inference show heterogeneous-expert-dispatch
crit inference check
```

## Read and adapt an entry

Each technique has a `README.md` (selection, mechanism, supplied code, worked
example, invariants, costs), an `entry.json` retrieval contract, a `code/`
folder and, when it fills an engine slot, the plugin files; a labelled hook
does not prove its backend exists. In `entry.json`,
`agent_guidance.use_when`/`avoid_when` guide selection, `provides`/`engine_supplies`
split responsibility and `preserve`/`adapt` separate invariants from choices.

`category` must be one of the values below; `crit inference check` enforces it and
`crit inference list` groups by it. A new category needs a change to `CATEGORIES`
in [inference.py](../cli/crit/inference.py) and to this table.

| Category | Entries about |
| --- | --- |
| `attention` | KV state, attention kernels and KV storage formats |
| `projections` | Weight formats, projection GEMM/GEMV and fused row epilogues |
| `experts` | Expert residency, caching, dispatch, contributions and file-backed fetch |
| `speculation` | Speculative rounds, drafts and recovery of speculative state |
| `sampling-and-constraints` | Token sampling and constrained output |
| `execution` | Scheduling, batching, graph reuse and recorded execution choices |
| `memory-and-loading` | Artifact loading, import checks, host/device memory and serialized state |

**Operations of a technique: `operations.json`.** crit ships no compute kernels and techniques keep no lists of
them ([kernels and plugins](../docs/PLUGINS.md#kernels-and-plugins)). A technique whose operations run on the device
states each one's contract in `operations.json` beside its README: per operation (`id`, `what`), the technique's CPU
reference to check a kernel against (`reference`: a file of the technique and its symbol), how to check it on the
target before trusting it (`verify`, tolerance included) and, where the contract promises one, what must stay
bit-equal (`invariance`). `crit inference show ID` prints them and `crit inference check` validates the file (and
refuses kernel lists, provider names, kernel source paths and a leftover `kernels.json`). The kernels themselves come
from crit's kernel templates (`kernels/techniques/<family>/<id>/`) and the libraries in [Kernels' libraries](../kernels/README.md#libraries);
a technique's plugin declares crit's kernels it loads in its plugin.json. Run each candidate against the reference on the target GPU and
checkpoint, keep the fastest that passes, and drop one that fails. A technique with no device arithmetic of its own
(bookkeeping, scheduling, host formulas) has none.

**Plugins.** An entry that fills an engine slot is a plugin of the same id
([crit plugins](../docs/PLUGINS.md#the-library)): `plugin.json`, `plugin.py` and
`tests/` beside its README put its own code behind the slot's interface, imported
through the manifest's `sources` (`"code"`), with tests against the slot's base plugin
or the technique's exact reference. Its variants are options, and parts that only work
together are one fused plugin. The base plugins, the plain plugin of each slot, are in
[base/](base/) (chain-verifier, device-placement, eager-graphs, expert-tensors,
experts-reference, greedy-head-drafter, memory-table-rows, projections-dense,
sequential-recurrence, softmax-attention, topk-router, torch-norms). Together they are
crit's library: `crit plugins list` shows them, `crit inference show ID` lists a
technique's plugin and the plugins built on its code, and `crit plugins add
ENGINE/plugins ID...` copies plugins and the code they import into an engine, which then
puts its own device code behind the same interface and keeps the tests. Three entries
only lend their code to plugins (async-expert-contributions, expert-streamed-prefill and
host-work-mailbox, whose join heterogeneous-expert-dispatch runs), and seven are no
plugin, because a model plugin calls them rather than plugs them into a slot:
artifact-loading, model-import-audit, checkpoint-storage, host-memory-admission and
device-speculative-round; or because Core's concurrent runtime does it for every engine:
session-scheduling and concurrent-request-execution. dense-layer-offload is the placement slot's layer planner.

Read the mechanism (units, identities, arithmetic, state transitions) before
porting; the Python object model is a reference, not a required interface. To
compose entries, agree on their shared facts: representation, byte extents,
generation, ownership, completion, visibility and reduction order. A native
adaptation may change placement, layout or batching but must keep sources alive
and ready before reading. Expert-routed engines combine several entries; see
[expert residency](expert-residency/README.md#with-other-inference-entries).

## Writing a technique

A technique README gives a one-paragraph summary, then: when to use it (and when
not); how it works, in enough detail to reconstruct it; what the code supplies
versus the engine; one runnable worked example; the invariants an adaptation must
keep; and costs and validation. It holds exactly one fenced `python` block,
runnable from `code/` with Python 3.11+ and the standard library. The host suite
runs these blocks:

```text
python -B -m unittest discover -s tests -p test_techniques.py -v
crit inference check
```

A technique that fills an engine slot is a plugin too: `plugin.json`, `plugin.py` and
`tests/` beside its README, a host reference built on its `code/` and tested against
the slot's base plugin or the technique's exact reference ([crit
plugins](../docs/PLUGINS.md#the-library)). One that fills no slot lends its code to a
plugin's `sources` or is a service an engine is built on (`ENGINE_SERVICES` in
[cli/crit/plugins.py](../cli/crit/plugins.py)); `tests/test_plugin_library.py` holds
every entry to one of the three and runs every plugin's tests:

```text
crit plugins test
crit plugins test --only ID
```

Host fixtures live under `tests/`. Engine construction follows the
[optimization workflow](../docs/BUILDING_ENGINES.md).

## Build the native code

Inference's C++ and CUDA examples and fixtures build with their own
[CMakeLists.txt](CMakeLists.txt), using Core only through the public SDK
(`crit::core`, `crit::cuda`). By default it builds Core from this checkout:

```sh
cmake -S inference -B build/inference -DCMAKE_BUILD_TYPE=Release
cmake --build build/inference --config Release
ctest --test-dir build/inference -C Release --output-on-failure
```

`-DCRIT_INFERENCE_INSTALLED_CORE=ON` uses an installed Core SDK from
`CMAKE_PREFIX_PATH`. `-DCRIT_INFERENCE_CUDA=ON` adds the CUDA kernels and fixtures
(CUDA Toolkit 13.1+ and a device). The [plugin engine walkthrough](examples/plugin-engine/README.md)
is the smallest engine package Core's loader runs.
