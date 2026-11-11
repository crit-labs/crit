# crit Kernels

Kernels is built like crit itself: a **kernel core** of building blocks and **kernel techniques** made of them, the
way [Core](../core/README.md) holds what every engine needs and [Inference](../inference/README.md) the techniques.

| Layer | What it holds | Where |
| --- | --- | --- |
| Kernel core | Building blocks with no algorithm in them: format decoders (NVFP4, MXFP4/8, FP8, integer groups), tensor-core multiply wrappers per GPU generation, loaders and multi-stage pipelines, reductions (online softmax, crit's fixed-order rule), epilogues | [core/](core/README.md) |
| Kernel techniques | Templates composed from the core: decode GEMVs, row tiles, prompt GEMMs, grouped experts, paged and tiled attention, chunked scans, fused norms. Each has structures (ways of computing its operation) and knobs | `kernels/techniques/<family>/<id>/` |
| On demand | A technique is built for the engine's device and tuned for its model by `crit kernels sweep`, which also times the best library at the same arithmetic beside it | `crit kernels build`, `sweep` |

<a id="salvage"></a>

**How the kernels get better.** Every engineered kernel crit looks at (a library's, a published engine's, one a run
wrote) is mined for mechanisms: a tiling, a pipeline, an MMA layout, a scan algorithm, a decoder trick. A mechanism
every technique can use becomes a core block, so every template that uses the block gains at once; a new way of
computing an operation becomes a structure of its technique, with knobs, so every engine's sweep can pick it. The
sweep's yardstick shows whether a gain is real: the template must match or beat the library it learned from, on the
engine's device and shapes. A new GPU generation or backend is a core addition, not a rewrite of every kernel.

Each technique's speed-only constants (warps, rows a warp owns, tiles, K splits,
pipeline stages, loads in flight) are knobs. They are source in this folder, one `kernels/techniques/<family>/<id>/` each,
built and tuned for the engine's device and model; nothing is downloaded or shipped prebuilt. A
kernel is only ever fast for one GPU and one model's shapes, so crit ships the structure and lets the sweep find the
constants: the know-how that is the same on every GPU lives in the template, what differs per GPU and model is
measured there. Each template is checked against its operation's reference and measured against the best vendor
or open library at the same arithmetic (its README names the library; in a run, `crit eval status --kernels`). An
external engine takes a template into its own code: `crit kernels build ID --tuning FILE` builds it and prints the
package, which it loads with `crit_kernels.build.load`, and `crit kernels sweep` tunes it without any plugin. The
rest of an engine's arithmetic comes from the [libraries](#libraries) below or from the engine. [Core](../core/README.md) supplies the runtime, [Inference](../inference/README.md) the techniques and
[Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md) the common checks for the integrated engine.

<a id="levels"></a>

## Which kernel to use

| Level | Where from | When |
| --- | --- | --- |
| 1 | [crit's kernels](#crits-kernels) | Always, where one covers the operation, swept for the model; a library is only its yardstick |
| 2 | [The libraries](#libraries) | Only where crit has no kernel for the operation |
| 3 | [The workbench](#write-or-adapt-a-kernel-the-workbench) | Where neither fits: the engine writes its own kernel, checked against FP64 and timed against a yardstick |

A level 2 or 3 kernel that holds beyond one model and GPU is a welcome contribution as a crit kernel, so it moves to
level 1 for every engine.

<a id="crits-kernels"></a>

## crit's kernels

| Family | Kernels |
| --- | --- |
| `projections` | [decode-gemv](techniques/projections/decode-gemv/README.md) (CUDA GEMVs and gemv-triton, one Triton GEMV for every weight format), [mma-rows](techniques/projections/mma-rows/README.md), [pinned-gemm](techniques/projections/pinned-gemm/README.md), [fixed-tile-gemm](techniques/projections/fixed-tile-gemm/README.md), [fp8-block-rows](techniques/projections/fp8-block-rows/README.md) |
| `attention` | [paged-kv-attention](techniques/attention/paged-kv-attention/README.md) |
| `recurrence` | [recurrent-scans](techniques/recurrence/recurrent-scans/README.md) |
| `experts` | [grouped-experts](techniques/experts/grouped-experts/README.md), [int-block-experts](techniques/experts/int-block-experts/README.md), [topk-routing](techniques/experts/topk-routing/README.md) |
| `norms` | [fused-norms](techniques/norms/fused-norms/README.md) |
| `starters` | [workbench-starter](starters/workbench-starter/README.md) (the workbench's starter matmul) |

```sh
crit kernels list                                        # every kernel: operations, structures
crit kernels build ID                                    # build for this GPU with the default knobs (cached)
crit kernels sweep ID --model CHECKPOINT --out TUNING    # tune the knobs for a model on this GPU
CRIT_KERNEL_TUNING=TUNING crit serve ...                 # the engine's plugins build with that tuning
```

`crit kernels sweep` builds every knob configuration of each structure a kernel declares (many configurations a
build, picked at run time), checks each against the operation's reference and against crit's row-invariance contract
(a row's bits never depend on the rows beside it), times it on the checkpoint's own shapes with a cold L2, and writes
`TUNING/<ID>.json` (the table per shape and row count) and a report against the default knobs. A knob that changes
the summation order keeps one value a shape for every row count. A library plugin loads a kernel by id
(`crit_plugins.kernels.load_kernel`), built with the engine's tuning or with the default knobs, which reproduce the
kernel as it was hand-tuned, and falls back to its torch path when the kernel cannot be built here (no CUDA toolkit,
an architecture it does not cover). Builds need torch with CUDA, the CUDA toolkit (nvcc) and ninja; they are cached
in `CRIT_KERNEL_CACHE` (default `~/.cache/crit/kernels`).

**Contributing.** A kernel template is welcome where an operation lacks one, and so is a new structure in an
existing kernel (a faster tile shape, another split, a fused path) or a knob value the sweep should try, when it meets
the [contribution criteria](../docs/CONTRIBUTION_CRITERIA.md): every engine's next sweep can then pick it, on every
device it suits. A contribution never adds a constant chosen on one GPU; an engine's own kernel stays in the engine
until it does.
A mechanism mined from an engineered kernel is welcome as a core block or a structure ([how the kernels get
better](#salvage)), with the notice its license asks for where code is adapted.
[TEMPLATES.md](TEMPLATES.md) is the contract: the folder, kernel.json, knobs in CUDA and Triton code, the core, the
sweep's bench.

<a id="libraries"></a>

## Libraries

Libraries have two roles. Where crit has a kernel, a library is only its yardstick (`yardstick` in kernel.json),
never the default path: the library plugins run crit's kernels and keep a library only as an explicit yardstick
option (quantized-projections' `prompt_gemm: pinned_cublaslt` or `provider`, the attention plugins' `flashinfer:
auto`, sequential-recurrence's `fla`). Where crit has no kernel for an operation, an engine takes it from a library
(level 2); the [README](../README.md#kernels-libraries) lists more providers per backend. The yardsticks crit's
kernels are timed against:

| Library | Covers |
| --- | --- |
| [cuBLAS / cuBLASLt](https://docs.nvidia.com/cuda/cublas/) through torch (`matmul`, `_scaled_mm`) | Dense GEMMs in BF16/FP16, FP8 with tensor or row scales, block-scaled NVFP4 and MXFP8 |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | Paged and ragged attention, FP8 KV, norms, fused MoE, sampling |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | Chunked prefill and recurrent decode for gated delta and other linear attention |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention), [CUTLASS](https://github.com/NVIDIA/cutlass) | Prompt attention; GEMM templates for layouts and fusions cuBLASLt does not take |
| torch | Everything else, and the fallback where a kernel cannot build |

Start from the technique: its `operations.json` ([Inference](../inference/README.md#read-and-adapt-an-entry)) says
what each device operation computes, the CPU reference to check a kernel against and how (tolerance, invariance).
Take crit's kernel for the operation where one exists and sweep it for the model; else a library's; else write the
kernel in the workbench, with the library as its yardstick. Three checks catch most mismatches:

- **The actual device.** A datacenter build can lack the instructions for the target (for example consumer
  Blackwell, sm_120). A declared architecture list is a lead; run it on the device.
- **Exactness across column counts.** When speculative output must equal plain greedy, a kernel used for one-token
  decode and multi-column verify must reduce in the same order for every column count. Split counts or tiling that
  depend on column count or context length change greedy output; prefill-only kernels are judged by the quality
  budget. The same holds for the whole decode step even when exactness is only reported: Eval's honesty check
  compares a served reply's log-probabilities with the score call's replay of the same tokens, and a replay drafts and
  verifies at other widths than the reply did. Library GEMMs choose their kernel by shape, so a step whose width
  varies (plain steps, shorter verify passes near the end of a reply) gives the same token other arithmetic. Running
  every decode step at one width, the unused columns as padding, removes the dependence for every kernel in the step
  at once. crit's templates keep a row's bits independent of the row count by contract (order knobs).
- **Native integration.** A native engine calls a library or a built template on its own streams and buffers
  ([foreign stream](examples/foreign-stream/README.md): a scoped binding for a native launcher's thread-local stream
  slot).

The [kernel profile](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#kernel-profile) shows which
operations cost most.

## Write or adapt a kernel: the workbench

When the engine writes a kernel, or adapts one a provider has, give it a workbench folder:

```sh
crit kernels new down_gemv --op gemv --format fp8 --model MODEL_DIR --root work/kernels
crit eval workbench check work/kernels/down_gemv --python ENGINE_PYTHON
crit eval workbench bench work/kernels/down_gemv --device device.json
crit eval workbench profile work/kernels/down_gemv [--ncu NCU]
```

`new` writes `kernel.json` (the shapes, here every FP8 matrix of the checkpoint, the column counts, the tolerance,
whether the kernel must be column invariant, the yardstick), `kernel.py` (inputs, an FP64 reference, the candidate and
the yardstick: cuBLAS or cuBLASLt for gemv and gemm in bf16 or fp8), crit-eval's `harness.py` and `src/`. `--format
nvfp4` takes NVFP4 weights as compressed-tensors stores them (E2M1 pairs, E4M3 scales per 16 inputs, an FP32 global
divisor) against bf16 activations, the arithmetic of a decode GEMV on stored weights and of prompts that keep 16-bit
activations; its yardstick is cuBLAS on the weights widened to BF16 once. `src/` is a torch extension project:
`--from DIR` copies an existing one (one of crit's kernels in `kernels/techniques/<family>/<id>`, a provider's kernel source, a
library sample) with its license file, which it needs beside it, else `kernel-builder init` (on `PATH`, or the
executable `CRIT_KERNEL_BUILDER` names) starts it from kernel-builder's own template; a `torch-ext` binding gets the
`registration.h` its macros need. For gemv and gemm, `kernel.py` builds `src/` with torch's extension builder and calls
its op `run_<format>`: register it in `src/torch-ext/torch_binding.cpp` (bf16 `run_bf16(Tensor x, Tensor w) -> Tensor`;
fp8 `run_fp8(Tensor x, Tensor x_scale, Tensor w, Tensor w_scale) -> Tensor`; nvfp4 `run_nvfp4(Tensor x, Tensor w, Tensor
w_scale, Tensor w_global) -> Tensor`, FP32 out). Any other
operation (`--op attention`, `--op scan`, ...) gets the same harness with `kernel.py`'s functions to fill in. `--starter`
starts from crit's starter matmul, [workbench-starter](starters/workbench-starter/README.md) (bf16 and fp8: simple,
correct, a fixed per-column order, any NVIDIA GPU); in a run it is allowed from the fourth candidate on
(`--run RUN_FOLDER`), so the first three candidates find or write their kernels themselves. A workbench kernel that
proves out belongs in crit's templates: as a structure of the kernel for its operation, its speed constants as knobs
([TEMPLATES.md](TEMPLATES.md)).

crit-eval measures the folder ([workbench](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md#kernel-workbench)):
`check` compares the kernel with the reference on every shape and column count and requires the same bits on a
repeat and, when declared, for each column alone (`--repeatable no` declares a kernel that may round differently on a
repeat, such as atomics or a split the workload chooses: its repeats are then judged within the tolerance, which
suits an engine in quality mode `speed`, where Eval passes output differences that part at near-ties); `bench` times it and the yardstick on inputs cycled past the L2,
against the floor; its `bench.json` is the evidence an `own` lever's `yardstick` cites.

## Integrate and verify

Keep shape-, format- or device-specific fast paths behind explicit applicability checks, and
fall back to another implementation when one does not apply; if none does, report the precise
gap. Loading success does not prove operation support, and parameterized dimensions do not make
device instructions portable. Kernels may be fused with Inference code, inside a slot's plugin
or across slots as one [fused plugin](../docs/PLUGINS.md#when-two-plugins-only-pay-together),
preserving the shared meaning, representations, ownership, stream ordering and state. A fusion
can change rounding and needs its own numerical and whole-model checks.

Pin library dependencies to exact versions in the candidate's package; a floating branch or mutable cache is not a
sealed dependency. Build before timing, and never let the loader fetch another build during a
benchmark or request. Third-party research and binaries stay outside published product folders.

Check each operator against a reference across its claimed shapes, formats, tails, strides and
aliases, plus completion ordering and buffer lifetimes. Then validate the complete engine with
[Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md), measuring prefill, decode and speculative verify separately. In
the handoff, say which operations come from crit's templates (and their tuning), libraries, adapted sources or new
kernels; an upstream benchmark is a lead, not a result. See
[Building engines](../docs/BUILDING_ENGINES.md) for the surrounding process.
