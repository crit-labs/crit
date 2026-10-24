# crit Inference

Inference holds crit's reusable inference techniques. Each entry explains a
mechanism, supplies actual code and states what an engine must preserve when it
adapts that code. Agents copy, specialize, combine or fuse entries; the engine owns
that design, and a complete engine can use crit without any entry.

Use [Core](../core/README.md) for lifecycle, requests and resources,
[Kernels](../kernels/README.md) to find and integrate kernels (external links never
substitute for code an entry claims to supply), and [Eval](../docs/eval/README.md)
to measure the engine; NLL, perplexity and KL code is in
[Eval measurements](../docs/eval/MEASUREMENTS.md). [crit serve](../docs/SERVING.md)
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
| Recover speculative recurrence | [recurrent-state-recovery](recurrent-state-recovery/README.md) | Gated-delta verification, accepted-prefix repair, chunked prefill reference |
| Run gated-delta and Mamba-2 layers on the device | [recurrent-layer-kernels](recurrent-layer-kernels/README.md) | Causal conv1d, chunked two-pass prefills and column-invariant scans with checkpoints |
| Share KV and execute attention | [attention-state](attention-state/README.md) | FP32/int8 COW pages for Core's state cache, tiled CPU attention, GQA, masks, RoPE |
| Evaluate rotation for low-bit KV | [rotated-kv-quantization](rotated-kv-quantization/README.md) | Walsh-Hadamard rotation before int8/int4 KV, host and CUDA |
| Partition a long attention sequence | [partitioned-attention](partitioned-attention/README.md) | Interleaved ownership, stable partial reduction, head-major scatter |
| Execute low-bit projections | [quantized-projections](quantized-projections/README.md) | G32 W4/W8, dynamic A8, row-scaled FP8, NVFP4 codec |
| Read GGUF quantized blocks in projections | [gguf-projections](gguf-projections/README.md) | Q4_0/Q8_0/Q4_K/Q5_K/Q6_K decode equal to gguf-py, CUDA GEMV and dequantization over stored blocks |
| Run FP8 / NVFP4 prompt GEMMs without re-layout | [stored-tile-gemm](stored-tile-gemm/README.md) | Tile layout shared with decode GEMV; W8A8 and W4A4 CUDA GEMMs |
| Run GPTQ, AWQ, compressed-tensors int4 or MXFP4 checkpoints as stored | [w4a16-projections](w4a16-projections/README.md) | Exact conversion to one canonical layout, 4-bit decode GEMV, BF16 tensor-core prompt GEMM (sm_80+) |
| Attend over long prompts without a score matrix | [flash-prefill-attention](flash-prefill-attention/README.md) | Causal GQA BF16 flash kernel, head dims 64/128/256; sliding windows (ring cache), sinks, soft-cap |
| Batch attention over paged KV of many sequences | [paged-attention](paged-attention/README.md) | CUDA decode, verify and varlen prefill over StateCache block tables; BF16/E4M3 pages; windows, sinks, soft-cap, tree masks |
| Fit a long context in an FP8 KV cache | [fp8-kv-attention](fp8-kv-attention/README.md) | E4M3 cache with per-head-vector scales; prompt and decode attention; windows over a ring, sinks, soft-cap, tree masks |
| Attend over a latent KV cache (MLA) | [latent-attention](latent-attention/README.md) | Absorbed MLA over shared [latent ; rope] rows, contiguous or paged, column invariant |
| Normalize and fuse a row epilogue | [rmsnorm-bf16](rmsnorm-bf16/README.md) | CUDA source, borrowed-view interface, host formula |
| Reuse expert weights, lend cache memory or borrow idle KV and workspace memory | [expert-residency](expert-residency/README.md) | Byte-cache leases, complete bundles, C++ region loans and owner slot loans |
| Plan expert residency inside decode graphs | [device-expert-cache](device-expert-cache/README.md) | CUDA directory, miss copies, prefetch, offline policy replay |
| Choose expert execution sources | [heterogeneous-expert-dispatch](heterogeneous-expert-dispatch/README.md) | Placement plans, measured CPU/device splits, router/shared fusion |
| Compute NVFP4 experts on the CPU with the device's bits | [host-expert-arithmetic](host-expert-arithmetic/README.md) | Declared row order, scalar oracle, AVX2/AVX-512 paths, matching CUDA function, scale hook |
| Join asynchronous contributions | [async-expert-contributions](async-expert-contributions/README.md) | Receipt-based join, source retention, ordered reduction |
| Run device-decided host work inside a captured graph | [host-work-mailbox](host-work-mailbox/README.md) | Device-reset doorbells, measured stream-wait or poll join, release and failure answering |
| Serve a dense model larger than device memory | [dense-layer-offload](dense-layer-offload/README.md) | Measured planner, host AVX2/AVX-512 layers, double-buffered weight streaming, measured plan selection |
| Fetch file-backed experts | [ssd-expert-materialization](ssd-expert-materialization/README.md) | Checked reads, bounded gather, read-ahead, upload handoff |
| Bound prefill, physical batches and scratch | [session-scheduling](session-scheduling/README.md) | Transactional cursor, latency chunk policy, physical caps and workspace sizing |
| Batch independent requests | [concurrent-request-execution](concurrent-request-execution/README.md) | Ready-row compaction, metadata patches, retirement barrier |
| Serve concurrent requests with batched steps end to end | [batched-reference-engine](batched-reference-engine/README.md) | Synthetic-model ConcurrentModel: one GEMM per weight per step, StateCache prefix reuse, paged attention, native library for crit serve |
| Execute speculative rounds | [mtp-decode](mtp-decode/README.md) | Token-history proposals, greedy/sampled verification, repair, adaptive budgets |
| Sample tokens on the device, and verify sampled drafts | [token-sampling](token-sampling/README.md) | Core's whole law over the full vocabulary (penalties, top-k/p, min-p), rejection-sampling verify predictions; Core's Sampler as oracle |
| Keep structured output exact under speculation | [token-constraints](token-constraints/README.md) | [Core](../core/src/constraints/README.md) matchers in sampled speculation, private forks and rewinds |
| Keep a speculative round on the GPU | [device-speculative-round](device-speculative-round/README.md) | Device accept for greedy and sampled rounds, frontier advance, graphs, one readback per round |
| Verify a tree of drafts in one pass | [draft-trees](draft-trees/README.md) | EAGLE-2 trees, ancestor masks, device tree walk (greedy or sampled), row compaction |
| Draft with a published EAGLE-3, DFlash or DSpark drafter | [hidden-state-drafters](hidden-state-drafters/README.md) | Core's reader of every drafter format (one tap convention), catch-up, block input, d2t mapping, confidence cuts |
| Cheapen the draft head | [draft-vocabulary-subset](draft-vocabulary-subset/README.md) | Head row subset, index mapping, CUDA gather-argmax |
| Select compatible execution and retain choices | [recorded-selection](recorded-selection/README.md) | Phase/shape replay policy, recorded choices, repeatability checks |

Most code is Python host reference; some entries add C++ or CUDA. Each entry
states its tested scope and missing native or engine work. Host tests establish
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
example, invariants, costs), an `entry.json` retrieval contract and a `code/`
folder; a labelled hook does not prove its backend exists. In `entry.json`,
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
(CUDA Toolkit 13.1+ and a device). The [Core walkthrough](examples/core-projection/README.md)
connects a projection to a Core engine; the CUDA operator suite has its own
[CMakeLists.txt](../tests/inference/cuda/CMakeLists.txt).
