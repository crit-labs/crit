# crit Kernels

Kernels helps agents find and adapt kernels from external providers for a complete engine. crit
hosts no compute kernels: [SOURCES.md](SOURCES.md) lists where they come from, starting with the
[Hugging Face Kernel Hub](https://huggingface.co/kernels), then libraries such as cuBLASLt, CUTLASS,
FlashInfer and flash-linear-attention. Choose the implementation that fits the engine's operations,
representations and hardware; a kernel written for the candidate is equally valid when none fits.
[Core](../core/README.md) supplies the runtime, [Inference](../inference/README.md) the
techniques and [Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md) the common checks for the integrated engine.

Kernels bundles no kernel runtime. It provides local applicability checks, dependency retention
and integration guidance through the standard-library [crit_kernels](crit_kernels/__init__.py)
package, run as `crit kernels`. It installs or downloads nothing and works without Hugging Face,
PyTorch or any provider library; dependencies chosen for an engine belong in that candidate's
environment and package.

## Find an implementation

1. Describe the operation: formula, input/output types, quantization and scales, dimensions and
   strides, aliases, and any fusion or recurrent state.
2. Record the target: OS, device architecture, driver/toolkit and framework or native ABI. A
   matching model name is insufficient.
3. Browse the Hub and the other providers in [SOURCES.md](SOURCES.md), and inspect each
   candidate's source, builds, license and invocation contract (workspace, streams, output
   readiness). Measure the best one on the engine's own shapes as the yardstick before adopting or
   writing anything.
4. Choose a compatible build, integrate or adapt source, or write the missing operation. Explain
   rejected or untested high-impact options in the candidate's rationale.

Official references: [inspect metadata](https://huggingface.co/docs/kernels/cli-info),
[load and check compatibility](https://huggingface.co/docs/kernels/basic-usage),
[pin revisions](https://huggingface.co/docs/kernels/locking),
[installation and platforms](https://huggingface.co/docs/kernels/installation). The Hub loader
is Python/PyTorch and checks declared architectures, not whether the operation supports your
layout and numerics; the provider recommends WSL for broader Windows coverage.

Three checks catch most mismatches:

- **Native integration.** Most Hub kernels are PyTorch or Triton operators. A native engine
  extracts the CUDA source and build flags, or compiles Triton ahead of time, and calls them on
  its own streams and buffers.
- **The actual device.** A datacenter build can lack the instructions for the target (for
  example consumer Blackwell, sm_120). A declared architecture list is a lead; run it on the device.
- **Exactness across column counts.** When speculative output must equal plain greedy, a kernel
  used for one-token decode and multi-column verify must reduce in the same order for every
  column count. Split counts or tiling that depend on column count or context length change
  greedy output; prefill-only kernels are judged by the quality budget.

Worth searching for before writing: prompt and decode attention, matmuls in the model's weight
formats (FP8, FP4, integer groups), fused gated activations, RMSNorm and rotary embeddings, and
for hybrid models chunked recurrent scans (gated delta, Mamba) and short causal convolutions.
The [kernel profile](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#kernel-profile) shows which cost most.

## Write or adapt a kernel: the workbench

When the engine writes a kernel, or adapts one a provider has, give it a workbench folder:

```sh
crit kernels new down_gemv --op gemv --format fp8 --model MODEL_DIR --root work/kernels
crit eval workbench check work/kernels/down_gemv --python ENGINE_PYTHON
crit eval workbench bench work/kernels/down_gemv --device device.json
crit eval workbench profile work/kernels/down_gemv [--ncu NCU]
```

`new` writes `kernel.json` (the shapes, here every FP8 matrix of the checkpoint, the column counts, the tolerance,
whether the kernel must be column invariant, the yardstick), `kernel.py` (inputs, an FP64 reference, the candidate
and the yardstick: cuBLAS or cuBLASLt for gemv and gemm in bf16 or fp8), crit-eval's `harness.py` and `src/`.
`--format nvfp4` takes NVFP4 weights as compressed-tensors stores them (E2M1 pairs, E4M3 scales per 16 inputs, an
FP32 global divisor) against bf16 activations, the arithmetic of a decode GEMV on stored weights and of a `preserve`
run's prompts; its yardstick is cuBLAS on the weights widened to BF16 once. For gemv and gemm `src/NAME.cu` is the
binding with an empty kernel to write; any other operation (`--op attention`,
`--op scan`, ...) gets the same harness with `kernel.py`'s functions to fill in. `--from DIR` copies an existing
kernel's source, a Hub kernel or a library sample, into `src/` as the starting point; it needs the source's license
file beside it, and a Hub kernel's `torch-ext` binding gets the `registration.h` its macros need. `--starter` writes a
simple, correct kernel in plain CUDA that builds for any NVIDIA GPU instead of the empty one (bf16 and fp8); in a run it is allowed
from the fourth candidate on (`--run RUN_FOLDER`), so the first three candidates find or write their kernels
themselves.

crit-eval measures the folder ([workbench](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#kernel-workbench)):
`check` compares the kernel with the reference on every shape and column count and requires the same bits on a
repeat and, when declared, for each column alone; `bench` times it and the yardstick on inputs cycled past the L2,
against the floor; its `bench.json` is the evidence an `own` lever's `yardstick` cites.

## Inspect and retain

Describe the operation and target in a `kernels_request` and put reviewed declarations in a
local catalog, one JSON file per build variant ([record format](FORMAT.md)). Descriptors are
data: Kernels never imports a kernel, evaluates an expression or runs a build command.

```sh
crit kernels target
crit kernels inventory --root RUN_ROOT/work/provider
crit kernels search gemm --catalog RUN_ROOT/work/catalog
crit kernels inspect --request RUN_ROOT/work/operation.json --catalog RUN_ROOT/work/catalog
crit kernels retain --request RUN_ROOT/work/operation.json --catalog RUN_ROOT/work/catalog --select openblas.sgemm --root RUN_ROOT/work/provider --destination RUN_ROOT/work/dependencies/openblas --reason "Matches this operation and native ABI"
crit kernels verify RUN_ROOT/work/dependencies/openblas
```

`target` observes host OS and architecture; supply the ABI and accelerator/toolchain facts
yourself. `inventory` hashes local files for artifact declarations. `inspect` marks every fact
matched, rejected or unknown; missing facts, uncovered request attributes, unpinned revisions,
missing notices and unreviewed conditions stay unknown.

Only a **declared-compatible** selection can be retained. `retain` checks declared sizes and
hashes, copies the selected files and notices into a **new** directory (default cap 8 GiB,
`--max-bytes` to change) and keeps the request, compared catalog, assessment and rationale.
Existing directories are never overwritten. `verify` rechecks the inventory and applicability;
bind its `dependency_sha256` in the engine's source/package inventories.

These checks prove byte identity and agreement with reviewed declarations, not upstream
authorship, license suitability, machine code, transitive closure, arithmetic or performance.

Examples: [native OpenBLAS](examples/openblas/README.md) (C++20 SGEMM wrapper with LP64/ILP64
selection and retained reference-check evidence) and
[foreign stream](examples/foreign-stream/README.md) (a scoped binding for a native launcher's
thread-local stream slot).

## Integrate and verify

Keep shape-, format- or device-specific fast paths behind explicit applicability checks, and
fall back to another implementation when one does not apply; if none does, report the precise
gap. Loading success does not prove operation support, and parameterized dimensions do not make
device instructions portable. Kernels may be fused with Inference code without a plugin
interface, preserving the shared meaning, representations, ownership, stream ordering and
state. A fusion can change rounding and needs its own numerical and whole-model checks.

Pin dependencies to exact source revisions and build variants, with hashes, notices, build
commands and runtime dependencies; a floating branch or mutable cache is not a sealed
dependency. Build before timing, and never let the loader fetch another build during a
benchmark or request. Third-party research and binaries stay outside published product folders.

Check each operator against a reference across its claimed shapes, formats, tails, strides and
aliases, plus completion ordering and buffer lifetimes. Then validate the complete engine with
[Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md), measuring prefill, decode and speculative verify separately. In
the handoff, say which operations come from backend code, Hub kernels, adapted sources or new
kernels; an upstream benchmark is a lead, not a result. See
[Building engines](../docs/BUILDING_ENGINES.md) for the surrounding process.
