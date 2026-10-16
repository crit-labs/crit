# Native OpenBLAS integration

This example turns a caller-supplied local OpenBLAS library into a retained Kernels dependency
and calls its C ABI from C++20: 64-bit Windows and Linux, LP64 or ILP64 integers, float32
row-major SGEMM, both transpositions and padded leading dimensions. It downloads nothing,
chooses no license and is not a model engine.

## Build and prepare

Needs a repository checkout, Python 3.11+, CMake 3.20+ and a C++20 compiler; build outside
product folders (in an engine project, under `work/`). The synthetic fixtures need no provider.

```sh
cmake -S kernels/examples/openblas -B build/openblas -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/openblas --config Release
ctest --test-dir build/openblas -C Release --output-on-failure
```

Pick a library built for this host, identify its SGEMM, configuration and thread-setting
symbols and review its notices. For an LP64 Linux build:

```sh
python -B kernels/examples/openblas/example.py prepare --library /absolute/path/libopenblas.so --notice /absolute/path/LICENSE --integer-bits 32 --sgemm-symbol cblas_sgemm --config-symbol openblas_get_config --threads-symbol openblas_set_num_threads --variant "reviewed Linux x86_64 LP64 build" --license BSD-3-Clause --work build/openblas-input
```

Use `--integer-bits 64` for `USE64BITINT` builds. Symbol names vary: some NumPy Windows wheels use
`scipy_cblas_sgemm64_`, `scipy_openblas_get_config64_` and `scipy_openblas_set_num_threads64_`;
keep their full bundled notices and declare the licenses of everything bundled. Repeat
`--runtime-file` for companion libraries, `--notice`, `--metadata` for build configuration and
`--runtime-requirement` for external prerequisites. Files are copied without loading, the
primary binary's SHA-256 is pinned, and `--revision git:FULL_COMMIT_HASH` can identify reviewed
source instead. The work directory must be new; inputs are capped at 8 GiB. The recorded host
OS/architecture and ABI are declarations, not facts read from machine code.

## Execute and retain evidence

`check` loads and runs native code (on Windows use `build/openblas/Release/openblas_check.exe`):

```sh
python -B kernels/examples/openblas/example.py check --dependency build/openblas-input/dependency --executable build/openblas/openblas_check --output build/openblas-results
```

It verifies the retained bytes, reserves crit's runtime ownership guard and runs the checker
under the process guardian with a 30-second limit and 1 MiB of diagnostics, then reverifies.
A new results directory keeps diagnostics, launch identity, status, cleanup status and hashes;
a failed check stays failed. If cleanup is uncertain, recover with
`crit serve --recover-runtime` after the owner exits.

The checker runs 648 cases (dimensions `{1,7,31} × {1,11,33} × {1,16,37}`, zero or three
elements of row padding, all transposes, three alpha/beta pairs) against double accumulation
with tolerance `3e-6 + 2e-5 * sum(abs(reference terms))`, and checks input immutability, output
guards and beta-zero with NaN-initialized output, using one provider thread. CTest checks
invalid ABI, shape, extent, alias and symbol handling with **synthetic C ABI fixtures**.

Retain the results, dependency, sources, compiler identity, build logs and executable with the
engine. A pass covers these operator cases only: no throughput, quality or engine qualification.

## Paged attention integration

[attention_gemm.h](attention_gemm.h) adapts `OpenBlas` to Inference's
[paged CPU attention](../../../inference/attention-state/README.md), using SGEMM for QK and for
weights times values. Inference owns packing, masks, RoPE, online normalization, workspace and
KV; the adapter supplies synchronous arithmetic. Pass `OpenBlasAttentionGemm{blas}` as the third
argument to `plan.run(queries, mask, gemm)` and keep the library alive through the call.

```sh
python -B kernels/examples/openblas/example.py check --profile attention --dependency build/openblas-input/dependency --executable build/openblas/kernels_attention_check --output build/openblas-attention-results
```

This runs `kernels_attention_check` (Windows: `Release/kernels_attention_check.exe`) with the
same guards as `--profile gemm` (the default): 2,304 attention comparisons and eight
RoPE/prefill/decode/retained-page comparisons at `2e-5*(1+abs(reference))` per element, and
rejects nonfinite or above-0.001 maximum absolute errors. It writes an
`openblas_attention_validation` record.

## Adapt the wrapper

Construct `crit_kernels::OpenBlas` from [openblas.h](openblas.h) with an absolute path, integer
width and three symbol names; it rejects a `USE64BITINT` mismatch before any matrix call. `gemm`
takes explicit spans, rows/columns and strides, and checks dimensions, overflow, storage bounds,
transpose shapes, finite scalars and disjoint output (over the full spans).

Callers keep buffers alive until the synchronous return, and the wrapper keeps the library until
destruction. OpenBLAS owns its workspace and threads, and its thread setting may affect other
users of the library. With Core, report completion after the call returns. Windows loads the
exact DLL with companion-directory and system search; Linux uses `dlopen(RTLD_NOW | RTLD_LOCAL)`,
so companions resolve through RPATH/RUNPATH and loader configuration. The wrapper is no sandbox.

ABI facts follow OpenBLAS v0.3.30's
[CBLAS header](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.30/cblas.h) and
[common definitions](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.30/common.h); the code is
independently written and nothing from OpenBLAS is vendored.
