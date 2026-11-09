# crit Core

Core is the engine foundation. [Inference](../inference/README.md) supplies
optional techniques and [Kernels](../kernels/README.md) guides kernel selection;
neither is required to integrate an engine. [Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md)
measures the completed engine through a separate adapter.

Core owns lifecycle and resource infrastructure and ships libraries for request
text and stop strings, sampling, state cache bookkeeping, model files and
config, host-memory admission and optional structured output, plus a Python
package. Model semantics, kernels and their logits, tensor and state layouts,
device work and native tokenization (a `TextFrontend`) belong to the engine.

Core stays model- and hardware-agnostic: it holds no layer arithmetic (attention,
projections, normalization, positional encodings), no kernels or device code beyond
the backends' runtime services (streams, memory, graphs, launches), and no number or
rule that holds only for one model family, dtype or GPU architecture. What batched
or speculative engines share enters Core as bookkeeping (scheduling, admission, state
pages, row layout, acceptance); the arithmetic stays in the engine, an
[Inference](../inference/README.md) entry or a kernel provider ([Kernels](../kernels/README.md)).
Core also carries the protocol crit's own tools speak with an engine, so an engine
needs nothing beyond Core to be measured and served: the stock eval-v1 and
chat-resident-v1 adapters in `crit_core.native`, the package capabilities in
`crit_core.serving`, and the request contexts Eval sends (`ReplayContinuation`,
`PlainDecoding`).

An engine integrates through the public Model/Session/Backend boundary: declare
capabilities and resource needs, advance requests, expose ready progress, commit
accepted output and drain owned work. Its execution strategy (dense or
expert-routed, CPU or accelerator, caches, graph capture) stays internal.

Serving concurrent requests is the normal path. A `ConcurrentModel` runs
[mixed prefill/decode batches](src/runtime/README.md#mixed-token-work-batches) with
reuse-aware admission and an optional [asynchronous batch pipeline](src/runtime/README.md#asynchronous-batch-pipeline);
active capacity goes up to 4096 lanes, subject to memory admission. Beside the
scheduler Core ships what batched engines share: [batched steps](src/batch/README.md)
(the step's row layout, page-reserving admission, speculative rows, batch-invariance
rules) and [step engines](src/runtime/README.md#step-engines), which give an engine's model plugin Core's
scheduler. Every engine Core's loader runs batches: its model plugin declares `max_concurrency`
of 2 or more ([building engines](../docs/BUILDING_ENGINES.md#serve-concurrent-requests)). CUDA and
Inference techniques are opt-in. The SDK covers
generation (with optional log-probabilities, request text and stop strings) and
scoring, with capability-checked [request options](src/runtime/README.md#request-options).
Each integration must validate its own model/backend combination.

Public headers are in `include/crit/` (`core.h` includes its parts by concern
from `include/crit/core/`), C++ sources and their guides in `src/`, the Python
package in `python/` and the [contracts](contracts/README.md) in `contracts/`;
`CMakeLists.txt` builds and installs the SDK.

| Part | Responsibility | Header |
| --- | --- | --- |
| [Runtime](src/runtime/README.md) | Requests, bounded scheduling, readiness, commit/publication, cancellation, retirement | [core.h](include/crit/core.h) |
| [Engine ABI](src/runtime/README.md#engine-abi) | Core's engine C ABI, exported by `crit_core_steps` for the engines Core's loader assembles; Core's Eval and serve adapters in `crit_core.engine` | [native_engine.h](include/crit/native_engine.h) |
| [Step engines](src/runtime/README.md#step-engines) | A `ConcurrentModel` over C step callbacks (`crit_core_steps`), so an engine's model plugin gets Core's scheduler, admission, cancellation and retirement (`crit_core.native.StepEngine`, which Core's loader opens) | [step_engine.h](include/crit/step_engine.h) |
| [Tracing](src/runtime/README.md#lifecycle-tracing) | Bounded lifecycle trace and [operator ranges](src/runtime/README.md#operator-ranges), optional NVTX sink | [trace.h](include/crit/trace.h) |
| [Memory](src/memory/README.md) | Resource budgets, reservations, [host-memory admission](src/memory/README.md#host-memory-admission) | [host_memory.h](include/crit/host_memory.h) |
| [Text](src/text/README.md) | Stop strings, reply text streams, incremental decoding over `detokenize()` | [text.h](include/crit/text.h) |
| [Sampling](src/sampling/README.md) | Counter-based draws, sampling rules, penalties, log-probabilities, greedy acceptance | [sampling.h](include/crit/sampling.h) |
| [State cache](src/state/README.md) | KV page and recurrent slot bookkeeping without touching state bytes | [state.h](include/crit/state.h) |
| [Batched steps](src/batch/README.md) | Step row layout and logits rows, page-reserving admission, speculative rows, batch-invariance rules | [batch.h](include/crit/batch.h) |
| [Files](src/io/README.md) | Strict JSON, read-only files, chunked and mapped reads, SHA-256 | [json.h](include/crit/json.h), [file.h](include/crit/file.h), [sha256.h](include/crit/sha256.h) |
| [Model](src/model/README.md) | Safetensors checkpoints, typed `config.json` and `generation_config.json`, stop ids | [safetensors.h](include/crit/safetensors.h), [model_config.h](include/crit/model_config.h) |
| [Structured output](src/constraints/README.md) | Optional token matchers for choices, JSON and compiled grammars, masked sampling | [constraints.h](include/crit/constraints.h) |
| [Device graphs](src/backends/cuda/README.md#graph-capture) | Optional graph cache, staged geometry and replay state machine; CUDA harness with `CRIT_CORE_CUDA` | [device_graph.h](include/crit/device_graph.h), [cuda_graph.h](include/crit/cuda_graph.h) |
| [Shared device allocations](src/backends/cuda/README.md#shared-allocations-and-mapped-sources) | Optional leases charged once, [fixed buffer plans](src/backends/cuda/README.md#fixed-buffer-plans) (every buffer allocated once, the rest as slots), mapped-source resolution; CUDA hooks with `CRIT_CORE_CUDA` | [device_memory.h](include/crit/device_memory.h), [cuda_memory.h](include/crit/cuda_memory.h) |
| [Programmatic dependent launch](src/backends/cuda/README.md#programmatic-dependent-launch) | Kernel launches that overlap the previous kernel's tail on sm_90+, with a device query and an ordinary fallback; `CRIT_CORE_CUDA` | [cuda_launch.h](include/crit/cuda_launch.h) |
| [CUDA backend](src/backends/cuda/README.md) | Allocation, streams/events, ownership services, [device time by category](src/backends/cuda/README.md#device-time-by-category) | [cuda.h](include/crit/cuda.h) |
| [Python package](python/README.md) | `crit_core`: model files, strict JSON, [config and safetensors](python/crit_core/model/README.md), [text layer](python/crit_core/text/README.md), constraints; `crit_plugins`: the [plugin](../docs/PLUGINS.md) contract an engine loads with | [crit_core](python/crit_core), [crit_plugins](python/crit_plugins) |
| [Contracts](contracts/README.md) | Native ownership and request responsibilities | [CONTRACTS.json](contracts/CONTRACTS.json) |

## Build and use

For a small engine built directly from this checkout, start with the
[Core + Inference walkthrough](../inference/examples/core-projection/README.md).

Use CMake 3.28+ and a C++20 compiler (GCC, Clang or MSVC). Warnings are errors.
CUDA is off by default; enable `CRIT_CORE_CUDA` with Toolkit 13.1+. On AMD GPUs, `CRIT_CORE_HIP` builds the same
backend on ROCm's HIP runtime as `crit::hip` ([HIP](src/backends/cuda/README.md#hip)).
`CRIT_CORE_SANITIZERS` accepts GCC/Clang driver flags only (see
[request release](src/runtime/README.md#request-release) for a known `thread`
false positive).

```sh
cmake -S core -B build/core -DBUILD_TESTING=ON
cmake --build build/core
ctest --test-dir build/core --output-on-failure
cmake --install build/core --prefix build/core-install
```

With multi-configuration generators such as Visual Studio, add `--config Release`
to build/install and `-C Release` to CTest. CUDA fixtures need a device.

Verify the installed headers and library with independent consumers, which also
compile every installed `crit/*.h` header under the strict warnings:

```sh
cmake -S tests/core/concurrency_consumer -B build/consumer -DCMAKE_PREFIX_PATH=/absolute/path/to/build/core-install
cmake --build build/consumer
ctest --test-dir build/consumer --output-on-failure
```

The SDK is version 0.17.2. Include `<crit/core.h>` and link the installed target:

```cmake
find_package(critCore 0.17.2 EXACT CONFIG REQUIRED)
target_link_libraries(your_engine PRIVATE crit::core)
```

The component headers in the table come with the same target; optional CUDA
services use `<crit/cuda.h>`, `<crit/cuda_graph.h>`, `<crit/cuda_memory.h>`,
`<crit/cuda_launch.h>` and `crit::cuda`. Supply `build/core-install` through
`CMAKE_PREFIX_PATH`. See the
[installed consumer example](../tests/core/concurrency_consumer/main.cpp).

Adapters can return an independent `TextFrontend` from `Model::text_frontend()`
for concurrent tokenization and private incremental decoders; it also enables
[request text and stop strings](src/runtime/README.md#request-text-and-stop-strings).
See the [text service contract](src/runtime/README.md#text-services) and
[native fixture](../tests/core/frontend_tests.cpp).

Python adapters and tools use the [`crit_core` package](python/README.md), shipped
in the `crit` wheel; its text layer needs `tokenizers` and `jinja2`, imported
lazily.

For lifecycle diagnostics, enable `Profile::trace` and read `Engine::trace()`
([runtime guide](src/runtime/README.md#lifecycle-tracing)); tracing is off by
default. [Operator ranges](src/runtime/README.md#operator-ranges)
(`TraceConfig::operators`) compile to nothing unless `CRIT_TRACE_ENABLED` is
defined, and `-DCRIT_CORE_NVTX=ON` enables the NVTX sink.

Core builds and tests only Core. Inference's examples build separately with
`cmake -S inference -B build/inference` against this SDK
([Inference](../inference/README.md#build-the-native-code)). Host fixtures do not
establish model arithmetic, CUDA correctness or performance; qualify those in
your [engine project](../docs/BUILDING_ENGINES.md).
