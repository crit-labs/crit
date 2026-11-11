# Writing a kernel template

crit's kernels live in `kernels/techniques/<family>/<id>/`: source only, built for the engine's device
(`crit kernels build`), with knobs a sweep tunes for its model (`crit kernels sweep`). This page is the contract
every kernel follows. `projections/decode-gemv` (structure `gemv-nvfp4`) is the worked example.

## Folder

```text
kernels/techniques/<family>/<id>/
  kernel.json        what it is, its operations, sources, structures and their knobs
  bench.py           one bench class per tunable structure (crit_kernels.sweep's interface)
  README.md          what it computes, formats, GPUs, numerics, how it was checked, and the vendor or open library
                     it is measured against at the same arithmetic (a measured number names device, model, workload)
  <sources>          CUDA/C++ sources (a CUDA kernel), named in kernel.json
  torch-ext/
    torch_binding.cpp, torch_binding.h   the ops (TORCH_LIBRARY_EXPAND / REGISTER_EXTENSION, registration.h)
    <package>/       the Python package plugins import: __init__.py and any Triton modules
  tests/             pytest against the operation's reference, loading the kernel with crit_kernels.build.load
```

Families: `projections`, `attention`, `recurrence`, `experts`, `norms`, `starters`.

## The kernel core

A kernel is a technique composed from the core's building blocks ([core/README.md](core/README.md)): include
`crit_kcore/<area>.cuh` in CUDA (the build puts `kernels/core/include` on every kernel's include path) or
`from crit_kcore import <area>` in Triton. Use a core block instead of a private copy: a decoder, an MMA wrapper, a
pipeline or a reduction improved in the core improves every kernel that uses it. A block that more than one technique
needs, or that a new GPU generation changes, belongs in the core with a device test in `core/tests/`; what only one
technique's algorithm does stays in that technique. A change to the core rebuilds every kernel and stales its tuning.

## kernel.json

```json
{
  "schema": "crit-kernel-v1",
  "id": "decode-gemv",
  "family": "projections",
  "title": "one line",
  "package": "decode_gemv",
  "language": "cuda",
  "operations": ["nvfp4-projection"],
  "sources": ["decode_gemv_cuda/decode_gemv.cu", "decode_gemv_cuda/binding.cu", "torch-ext/torch_binding.cpp"],
  "capabilities": ["8.0", "8.6", "8.9", "9.0", "10.0a", "12.0a"],
  "libraries": [],
  "structures": {
    "gemv-nvfp4": {
      "what": "one line",
      "call": "the package function the bench calls",
      "rows": [1, 16],
      "tune_rows": [1, 2, 4, 8, 16],
      "key": ["N", "K"],
      "row_invariant": true,
      "knobs": {
        "WARPS": {"values": [2, 4, 8], "default": 4, "what": "warps a block"},
        "WPR": {"values": [1, 2, 4], "default": 1, "order": true, "what": "..."}
      },
      "default_rows": [{"rows": [3, 4], "knobs": {"ROWS": 4}}],
      "yardstick": {"library": "cuBLAS (torch.matmul) on the weights widened to BF16 once"},
      "bench": "bench.py:GemvNvfp4"
    }
  }
}
```

- `operations`: ids from `inference/*/operations.json` the kernel computes (its reference and tolerance live there).
- `language`: `cuda` (compiled with torch's extension builder for the current GPU) or `triton` (no compile step;
  CUDA sources beside the Triton modules: `cuda` below).
  CUDA is the backend built today; another backend adds its language and capability names.
- `capabilities`: compute capabilities it builds and runs on; an `a` suffix where it uses architecture-specific
  instructions. `crit kernels build ID --arch 80 --arch 86 --arch 89 --arch 90a --arch 100a --arch 120a` checks the
  build for each.
- A **structure** is one way of computing an operation (a CUDA-core GEMV, a tensor-core tile, a split-K reader). Its
  `knobs` are the constants that only change speed (warps, rows a warp owns, tiles, stages, splits, loads in flight,
  Triton's `BLOCK_*`, `num_warps`, `num_stages`). Mark a knob `"order": true` when it changes the summation order: a
  shape then keeps one value of it for every row count, so a row's bits never depend on the rows beside it. A knob a
  per-matrix plan reads (one choice a matrix, whatever the row count) is marked the same way.
- `default` knobs plus `default_rows` (knob overrides by row range, shape independent) must reproduce the kernel's
  behavior before it had knobs: an untuned build runs exactly what crit ran before.
- `yardstick`: the best vendor or open library at the structure's arithmetic, which the sweep times on the same
  shapes beside the tuned kernel (the bench's `yardstick(state)`), or `null` where none computes it; the report then
  gives the kernel's achieved GB/s against the device. A template earns its place by matching or beating it.
- `cuda` (a Triton kernel with CUDA sources beside its Triton modules, `attention/paged-kv-attention`): the sources
  are compiled like a CUDA kernel's (the build's cache and key, its device's architecture or `--arch`'s, its
  `crit_tuning.h` and sweep builds of a chunk of configurations each, `select()`), one shared library a **variant**,
  the compile-time choices the package's code makes a call by (query dtype, page kind, head widths), on the variant's
  first call or ahead of it; the Triton modules load as before. Fields:

  ```json
  "cuda": {
    "sources": ["torch-ext/paged_kv_attention/cuda_window.cu"],
    "structures": ["cuda-window", "cuda-fused", "cuda-prefill"],
    "variant": {"T": [0, 1], "KIND": [0, 1, 2, 3], "D": [64, 128, 256], "DV": [64, 128, 256]},
    "device": {"SMEM": "shared_memory_per_block_optin"},
    "check": [{"T": 0, "KIND": 0, "D": 256, "DV": 256}],
    "flags": []
  }
  ```

  `sources`: paths in the kernel's folder, compiled together with nvcc into one library a variant (`-shared`,
  C++17, `-I` the build's include folder: `crit_tuning.h` and the core headers they use); `structures`: the structures
  whose knobs the sources read from `crit_tuning.h` (the sweep compiles their builds before it holds the GPU);
  `variant`: each field's allowed values, passed as `-DCRIT_VARIANT_<field>=<value>`; `device`: variant fields filled
  from the GPU's properties (torch's names), so a library is built for the device it runs on; `check`: the variants
  `crit kernels build ID --arch ...` compiles for every architecture (the device fields left to the source's
  defaults); `flags`: extra nvcc flags.
- `key`: the shape fields the tuning table is keyed by (a GEMV: `N`, `K`; attention: head dim, heads, page size).
- `rows`: the row (token) counts the structure takes; `tune_rows`: the counts the sweep times (a timed count's pick
  serves the counts above the previous timed count up to it, and the last also serves every count above it);
  `tune_weights`, optional, one weight per timed count (in sorted order) when the order knobs must favour some
  counts, such as long prompt chunks over short ones.
- `alias`: a structure that is another one under an older name. `{"of": S, "weights": [...]}` in the same kernel (the
  kernel's code reads both tables; decode-gemv's gemv-nvfp4 of gemv-cuda). `{"kernel": ID, "of": S}` in another kernel
  (int-block-experts' rows of decode-gemv's rows-int-block): the same knobs as S, and the kernel's Python calls forward
  to ID's package through the generated `_crit_aliases.py` (`from ._crit_aliases import module`; `module("rows")` is
  ID's package). That package is built with this kernel's tuning table for the alias where the table names one, else
  with ID's own tuning (`CRIT_KERNEL_TUNING/ID.json`) or its defaults; the build builds it with this kernel, a build
  with `configs=` (a sweep's) builds ID's sweep build of S with the same configurations, and `select()` on the alias
  selects S there. So an old tuning of the alias still applies, and `crit kernels sweep` of the alias (its own bench,
  calling this kernel's API) tunes the code that runs; S may have a bench of its own for sweeps of ID.

## Knobs in code

CUDA: include `crit_tuning.h` (generated by the build). Per structure `S` (its id, `-` as `_`) it defines
`CRIT_CONFIGS_S(X)`, which expands `X(index, knob values in kernel.json order)` for every configuration the tuning uses
(index 0 is always the defaults), and `crit_lookup_S(key fields..., rows)`, the index for a call. Template the kernel
on its knobs and dispatch:

```cpp
switch (crit_lookup_gemv_nvfp4(N, K, M)) {
#define CASE(index, warps, rows, wpr, pref, stream) \
  case index: return dispatch_rows<warps, rows, wpr, pref, stream, X, O>(M, ...);
  CRIT_CONFIGS_gemv_nvfp4(CASE)
#undef CASE
  default: return false;
}
```

A sweep build defines `CRIT_SWEEP_BUILD`: instantiate only what the bench calls (its dtype, its `tune_rows`), so each of
the sweep's hundreds of builds compiles in seconds.

Triton: the package gets `_crit_tuning.py`; read a call's knobs with
`from ._crit_tuning import knobs; k = knobs("structure-id", {"N": n, "K": k}, rows)` and pass them as `tl.constexpr`
block sizes and `num_warps`/`num_stages`.

A Triton kernel with CUDA sources (kernel.json `cuda`): the package gets `_crit_cuda.py` too, and
`_crit_cuda.library(device, **variant)` returns the variant's library (ctypes, compiled into the build folder on first
use; call it on the caller's stream so CUDA graphs capture its launches). Its sources include `crit_tuning.h` and
dispatch on `CRIT_CONFIGS_S` and `crit_lookup_S` like a CUDA kernel's, for the structures in `cuda.structures`;
`select()` reaches every library of the build, loaded or not yet. Its bench gives `variants(shape)` (the variants a
call on that shape uses) so the sweep compiles them first.

## bench.py

One class per tunable structure, the interface in `crit_kernels/sweep.py`'s docstring: `shapes(model)` (from
`crit_kernels.model.Model`: `matrices()` lists every linear weight with its format; `config`/`get()` the text config),
`inputs(shape, rows, device)` (row 0's input independent of `rows`; cache matrices across calls, the sweep calls it for
every configuration), `run(module, state)`, `reference(state)` (from the operation's reference, cached),
`check(got, want, state)`, `first_row(out)`, optional `bytes(shape, rows)`, `valid(knobs)` and
`valid_for(knobs, shape)` (a configuration that cannot run a shape, such as a split that does not divide K). The sweep
sets `bench.default` to the untuned build before timing, for benches that compare bits with it (so a bench
defines no method of that name).

## Commands

```sh
crit kernels list
crit kernels build ID [--tuning FILE] [--arch 80 ...]
crit kernels sweep ID --model CHECKPOINT --out TUNING_DIR      # or --shapes FILE {structure: [shape, ...]}
```

A plugin loads a kernel by id (`crit_plugins.kernels.load_kernel`), built with the engine's tuning
(`CRIT_KERNEL_TUNING=TUNING_DIR`) or the default knobs, and falls back to its torch path when it cannot be built.
Builds are cached in `CRIT_KERNEL_CACHE` (default `~/.cache/crit/kernels`).
