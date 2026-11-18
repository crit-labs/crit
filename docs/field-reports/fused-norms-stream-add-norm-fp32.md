# Field report: fused-norms' `stream_add_norm` output differs from `stream_add` + `rms_norm` for FP32 streams with per-stream weights (sm_120, Windows)

## Summary

crit 0.19.0's own `inference/fused-norms`, unchanged, fails one of its device tests on an RTX 5060 Ti under Windows:

```text
FAIL: test_the_fused_forms_give_the_separate_calls_bits (test_device.FusedNormsDeviceTests)
AssertionError: stream_add_norm torch.float32 streams of 5120, 4 tokens: output 0 differs from the separate calls' bits
```

- **Which outputs differ:** the normalized output `y` (output 0) differs from `rms_norm` over the streams the separate
  `stream_add` produced. The updated streams (output 1) are identical.
- **Which configuration fails:** the failing configuration is the third one in `conformance.fused_forms`: FP32 streams,
  per-stream mixing weights `weights[t][s]`, no norm weight, FP32 output. The first configuration (FP32 streams,
  per-stream weights, a weight per stream, BF16 output) and the second (BF16 streams) pass.
- **Size of the difference:** each path agrees with an FP64 reference to the same accuracy, so this is a rounding
  difference between the two forms, not a wrong result. It does break the contract the test checks ("each output has
  the bits of the separate calls").

## Environment

| | |
|---|---|
| crit | 0.19.0 (tag `v0.19.0`), `inference/fused-norms` and `kernels/techniques/norms/fused-norms` unchanged |
| GPU | NVIDIA GeForce RTX 5060 Ti 16 GB, compute capability 12.0 (sm_120), driver 610.88 |
| OS | Windows 11 (10.0.26200) |
| Python | 3.14.4 |
| torch | 2.14.1+cu130 (CUDA 13.0) |
| Triton | triton-windows 3.9.0.post30 |
| Kernel record | `kernels.loaded()["fused-norms"]`: `{"source": "built"}`, default knobs, no tuning file |

## Reproduce

From a crit 0.19.0 checkout on a CUDA machine:

```bash
crit plugins test inference --only fused-norms --no-cache
```

An equivalent run without the registry:
1. Copy `inference/fused-norms` and `inference/base/torch-norms` (its reference) into one folder.
2. Run its tests as `crit plugins test` does:

```bash
python -B -m unittest discover -v -s tests -t tests -p "test*.py"
```

Run this from the `fused-norms` folder, with `PYTHONPATH` set to the folder that holds both plugins.

The failure was first seen through `crit run ... serve --check` of an engine package that carries `fused-norms`. The
plugin tests failed there before the serving checks ran.

## Measurements

The `_same` check in `code/conformance.py` was instrumented to print the difference before it fails.

**The failing case** (FP32 streams of 5120 values, 4 tokens × 4 streams, per-stream weights, no norm weight, FP32 output):
- 12,913 of 81,920 output values differ.
- The largest difference is 3 ulp (4.8e-7 absolute).
- 3 of 16 rows are affected: rows 9, 12 and 15, which are token 2 stream 1, token 3 stream 0 and token 3 stream 3.

**Row dependence**, same inputs as the test:

| tokens | rows where fused ≠ separate | separate: rows alone vs together | fused: rows alone vs together | streams equal | max error vs FP64 (fused / separate) |
|---|---|---|---|---|---|
| 1 | 0 / 4 | equal | equal | yes | 2.6e-7 / 2.6e-7 |
| 2 | 0 / 8 | equal | equal | yes | 3.5e-7 / 3.5e-7 |
| 3 | 1 / 12 | equal | equal | yes | 3.5e-7 / 3.7e-7 |
| 4 | 3 / 16 | equal | equal | yes | 3.5e-7 / 4.0e-7 |
| 9 | 7 / 36 | equal | equal | yes | 5.1e-7 / 5.1e-7 |
| 64 | 46 / 256 | equal | equal | yes | 5.6e-7 / 5.1e-7 |

- Each path is row-invariant on its own: a row has the same bits alone and beside other rows.
- The two paths disagree on individual rows, and which rows disagree depends on the data. More tokens only means more
  rows that can disagree; the row count does not cause it. With other random inputs, some rows already differ at 1 token.

**Which configurations disagree** (own probe, widths 2560, 4096, 5120 and 7168, 1 to 64 tokens, both paths called on the same streams):

| configuration | cases where `y` differs | max ulp |
|---|---|---|
| FP32 streams, per-stream weights, no norm weight, FP32 out | 28 / 28 | 4 |
| FP32 streams, per-stream weights, weight per stream, FP32 out | 28 / 28 | 5 |
| FP32 streams, per-stream weights, weight per stream, BF16 out | 0 / 28 (fresh Triton cache; see below) | 0 |
| FP32 streams, plain add (`weights=None`), weight per stream, BF16 out | 0 / 28 | 0 |
| BF16 streams, plain add, one BF16 weight | 0 / 28 | 0 |

The difference appears only with FP32 streams and per-stream mixing weights (`HAS_XW` in `rows_kernel`). With a BF16
output it is mostly hidden by the output rounding.

## Narrowing

What was ruled out:
- **Local changes:** none apply. The checkout had uncommitted Windows build changes in `kernels/crit_kernels/build.py`
  (MSVC and nvcc flags). They do not affect `fused-norms`, which is a Triton kernel (`kernel.json`: `"language": "triton"`).
- **Nondeterminism:** none. Each path gives the same bits when called twice, and the test fails identically on repeated runs.
- **Stale caches:** ruled out. The test fails the same way with a fresh `TRITON_CACHE_DIR`.
- **Launch configuration:** the same for both paths. `stream_add_norm` and `rms_norm` both launch `rows_kernel` with
  `_config(width)` (same `BD`, `HB` and warps), and the default knobs apply (no tuning file).
- **Concurrent GPU use:** not the cause. The failure was first seen while another engine held most of the GPU. It
  reproduces the same way with only an idle embedding service (about 0.7 GB) on it.

**Triton FP fusion** (`TRITON_DEFAULT_FP_FUSION`), each run with a fresh Triton cache:
- **Unset or `1` (contraction on, Triton 3.9's default):** the run fails as above. `y` differs and the streams are equal.
- **`0` (Triton 3.10's announced default):** the run fails earlier: `stream_add_norm torch.float32 streams of 5120, 1 tokens: output 1 differs`.
  - The updated streams now differ from `stream_add`'s in 2,414 of 20,480 values, by up to 256 ulp (9.5e-7 absolute), in all 4 rows.
  - So without contraction, `streams + weights * x` in the fused kernel and in `stream_add` no longer round alike.
  - This will matter once Triton 3.10 changes the default.

**Likely cause (not proven):**
- In the fused path, `rows_kernel` with `HAS_XW` and `HAS_RES` computes `x * xw`, adds the residual, stores the sum and
  normalizes the value in registers.
- The separate `rms_norm` loads the stored FP32 rows.
- The stored values are bit-identical, so the difference has to come from how the compiler contracts or orders the
  arithmetic around the sum of squares in the two specializations of the same kernel. Each specialization rounds
  differently, but each one consistently.

What could not be checked here:
- whether it reproduces on Linux with upstream Triton;
- whether it reproduces on other architectures than sm_120.

## Possible directions

- Make the fused form compute its norm from the stored value explicitly, for example by forcing the rounding barrier
  that `s.to(r_ptr.dtype.element_ty)` provides for 16-bit streams, so both paths reduce the same values the same way.
- Pin the contraction for both paths with `enable_fp_fusion` on the launch, so the bit contract does not depend on
  Triton's default. This also covers the stream mismatch that appears with fusion disabled.
- Run `fused-norms`' device tests on a Windows or sm_120 GPU job, if CI does not cover that combination yet.

## Impact

- No numerical risk: both forms are equally close to FP64.
- An engine package that carries `fused-norms` cannot pass `crit run ... serve --check` on this machine, because the plugin tests run first.
- The workaround used: leave `fused-norms` out of a package whose engine does not select it.
