# crit Core

Core is the engine foundation. Every engine is Core plus [plugins](../docs/PLUGINS.md),
one per slot. [Inference](../inference/README.md) supplies the library's plugins and
[Kernels](../kernels/README.md) its kernels; an engine may take them or write its own
plugins. [Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md)
measures the completed engine through a separate adapter.

Core owns lifecycle and resource infrastructure and ships libraries for the
sampling draw, files and optional structured output, plus a Python package (which
holds request text and stop strings). Model semantics, kernels and their logits,
tensor and state layouts and device work belong to the engine.

Core stays model- and hardware-agnostic: it holds no layer arithmetic (attention,
projections, normalization, positional encodings), no kernels or device code beyond
the backends' runtime services (streams, memory, graphs, launches), and no number or
rule that holds only for one model family, dtype or GPU architecture. What batched
or speculative engines share enters Core as bookkeeping (scheduling, admission, state
pages, row layout, acceptance); the arithmetic stays in the engine, an
[Inference](../inference/README.md) entry or a kernel provider ([Kernels](../kernels/README.md)).

A test holds the line: [test_core_layering.py](../tests/test_core_layering.py) reads Core's source and fails when
it imports a module above it, includes a header outside it, or names a library plugin or a model family beyond the
names counted in its allowlist, which may only shrink.
Core also carries the protocol crit's own tools speak with an engine, so an engine
needs nothing beyond Core to be measured and served: the stock eval-v1 and
chat-resident-v1 adapters in `crit_core.native`, the package capabilities in
`crit_core.serving`, and the request contexts Eval sends (`ReplayContinuation`,
`PlainDecoding`).

An engine integrates as a plugin package that Core's loader assembles
([the engine](../docs/PLUGINS.md#the-engine)): its model plugin is Core's step model,
and its plugins declare capabilities and resource needs, advance requests, commit
accepted output and drain owned work. Its execution strategy (dense or
expert-routed, CPU or accelerator, caches, graph capture) lives inside its plugins.

Serving concurrent requests is the only path. The scheduler runs
[mixed prefill/decode batches](src/runtime/README.md#mixed-token-work-batches);
active capacity goes up to 4096 lanes, subject to memory admission. Beside the
scheduler Core ships what batched engines share: [batched steps](python/crit_core/batch.md)
(the step's row layout, page-reserving admission, batch-invariance rules) and [step engines](src/runtime/README.md#step-engines), which give an engine's model plugin Core's
scheduler. Every engine Core's loader runs batches: its model plugin declares `max_concurrency`
of 2 or more ([building engines](../docs/BUILDING_ENGINES.md#serve-concurrent-requests)). CUDA and
Inference techniques are opt-in. The SDK covers
generation (with optional log-probabilities) and
scoring, with capability-checked [request options](src/runtime/README.md#request-options).
Each integration must validate its own model/backend combination.

Public headers are in `include/crit/` (`core.h` includes its parts by concern
from `include/crit/core/`), C++ sources and their guides in `src/`, the Python
package in `python/` and the [contracts](contracts/README.md) in `contracts/`;
`CMakeLists.txt` builds and installs the SDK.

| Part | Responsibility | Header |
| --- | --- | --- |
| [Runtime](src/runtime/README.md) | Requests, bounded scheduling, commit/publication, cancellation, retirement | [core.h](include/crit/core.h) |
| [Engine ABI](src/runtime/README.md#engine-abi) | Core's engine C ABI, exported by `crit_core_steps` for the engines Core's loader assembles; Core's Eval and serve adapters in `crit_core.engine` | [native_engine.h](include/crit/native_engine.h) |
| [Step engines](src/runtime/README.md#step-engines) | Core's `Model` over C step callbacks (`crit_core_steps`), so an engine's model plugin gets Core's scheduler, admission, cancellation and retirement (`crit_core.native.StepEngine`, which Core's loader opens) | [step_engine.h](include/crit/step_engine.h) |
| [Memory](src/memory/README.md) | Resource budgets and reservations | [core/resources.h](include/crit/core/resources.h) |
| [Sampling](src/sampling/README.md) | The counter-based draw (header-only) and the sampling rules a sampler plugin follows | [sampling.h](include/crit/sampling.h) |
| [Files](src/io/README.md) | Strict JSON, read-only files, chunked and mapped reads | [json.h](include/crit/json.h), [file.h](include/crit/file.h) |
| [Structured output](src/constraints/README.md) | Optional token matchers for choices, JSON and compiled grammars, masked sampling | [constraints.h](include/crit/constraints.h) |
| [CUDA backend](src/backends/cuda/README.md) | The launcher of captured decode rounds over the CUDA driver, loaded at run time | [step_engine.h](include/crit/step_engine.h) |
| [Python package](python/README.md) | `crit_core`: model files, strict JSON, [config and safetensors](python/crit_core/model/README.md), [text layer](python/crit_core/text/README.md), constraints; `crit_plugins`: the [plugin](../docs/PLUGINS.md) contract an engine loads with | [crit_core](python/crit_core), [crit_plugins](python/crit_plugins) |
| [Contracts](contracts/README.md) | Native ownership and request responsibilities | [CONTRACTS.json](contracts/CONTRACTS.json) |

## Build and use

For the smallest engine Core's loader runs, start with the
[plugin engine walkthrough](../inference/examples/plugin-engine/README.md).

Use CMake 3.28+ and a C++20 compiler (GCC, Clang or MSVC). Warnings are errors.
Core needs no CUDA toolkit: it loads the device's driver at run time ([CUDA backend](src/backends/cuda/README.md)).
`CRIT_CORE_SANITIZERS` accepts GCC/Clang driver flags only (see
[request release](src/runtime/README.md#request-release) for a known `thread`
false positive).

`crit core build` builds just the runtime engines load, `crit_core_steps`, into `build/core-steps` (from a crit
installed without a checkout: into Core's cache, `CRIT_CORE_CACHE`, default `~/.cache/crit/core/VERSION`, from the
release's tag), where `crit_core.native.step_library()` finds it when `CRIT_CORE_STEP_LIBRARY` names none. The full
build with its tests:

```sh
cmake -S core -B build/core -DBUILD_TESTING=ON
cmake --build build/core
ctest --test-dir build/core --output-on-failure
cmake --install build/core --prefix build/core-install
```

With multi-configuration generators such as Visual Studio, add `--config Release`
to build/install and `-C Release` to CTest.

Verify the installed headers and library with independent consumers, which also
compile every installed `crit/*.h` header under the strict warnings:

```sh
cmake -S tests/core/concurrency_consumer -B build/consumer -DCMAKE_PREFIX_PATH=/absolute/path/to/build/core-install
cmake --build build/consumer
ctest --test-dir build/consumer --output-on-failure
```

The SDK is version 0.20.0. Include `<crit/core.h>` and link the installed target:

```cmake
find_package(critCore 0.20.0 EXACT CONFIG REQUIRED)
target_link_libraries(your_engine PRIVATE crit::core)
```

The component headers in the table come with the same target. Supply `build/core-install` through
`CMAKE_PREFIX_PATH`. See the
[installed consumer example](../tests/core/concurrency_consumer/main.cpp).

Python adapters and tools use the [`crit_core` package](python/README.md), shipped
in the `crit` wheel; its text layer needs `tokenizers` and `jinja2`, imported
lazily.

Core builds and tests only Core. Inference's examples build separately with
`cmake -S inference -B build/inference` against this SDK
([Inference](../inference/README.md#build-the-native-code)). Host fixtures do not
establish model arithmetic, CUDA correctness or performance; qualify those in
your [engine project](../docs/BUILDING_ENGINES.md).
