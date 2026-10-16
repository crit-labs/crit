# BF16 row normalization

A native CUDA row-wise RMS normalization over contiguous BF16 input, gain and output,
with plain gain, unit-offset gain and a fused SiLU gate, plus a Python formula reference
and a standalone CUDA example. It is an optional arithmetic component; Core prescribes
no normalization or tensor layout.

## When to use it

Use it when the engine's normalization matches one of the three modes over contiguous
BF16 rows. The example establishes operator behavior for its tested shapes and values;
model quality and performance need separate evidence, and width, row count, reduction
structure and fusion decide which implementation is best for an engine.

## How it works

For row width W, `r = 1 / sqrt(sum(x[j]^2) / W + epsilon)`, then:

| Mode | Output |
| --- | --- |
| Plain | `y[j] = x[j] * r * gain[j]` |
| Unit offset | `y[j] = x[j] * r * (1 + gain[j])` |
| Gated | `y[j] = x[j] * r * gain[j] * gate[j] / (1 + exp(-gate[j]))` |

One 256-thread block handles a row: threads read strided columns, convert to FP32 and
accumulate squares, a shared 256-float tree reduction gives the sum, and the epilogue
uses `rsqrtf`, applies gain and optional gate, and rounds to BF16 (nearest). One launch,
no allocation or workspace.

**Fold split partials.** A narrow projection feeding a residual sum can be split along
its reduction axis to fill a wide device, but atomically accumulating the splits makes
the sum depend on completion order. Instead the producer writes one partial per split,
and the normalization adds them in split order, writes the updated residual and
normalizes it in one pass (`fold_and_normalize` in [reference.py](code/reference.py)).
That is deterministic, with no atomics or extra launch, at one extra read per partial.
The native kernel here does not implement the fold.

## What the code supplies

| Source | What it implements |
| --- | --- |
| [`include/crit/rmsnorm.h`](code/include/crit/rmsnorm.h) | Borrowed views, modes, parameters, execution context, `validate` and `enqueue` |
| [`interface.cpp`](code/interface.cpp) | Shape, representation, stride, alignment, epsilon, workspace and alias checks |
| [`implementation.cu`](code/implementation.cu) | The reduction and normalization/gated epilogue kernel |
| [`reference.py`](code/reference.py) | Host formula and `fold_and_normalize` with shape checks |
| [`CMakeLists.txt`](code/CMakeLists.txt) | Static library `crit::rmsnorm`, C++/CUDA 20 |
| [`example.cpp`](code/example.cpp) | Owned stream/storage, independent BF16 oracle, all modes, aliases, metadata rejection |

The native domain is width 1..8192, rows 1..49152, two-byte aligned BF16 views with row
stride exactly `2 * width`, a one-row gain, a gate of input shape exactly in gated mode,
positive finite FP32 epsilon, and null zero-byte workspace.

## What the engine supplies

Borrowed device views and a caller stream from its resource services. `validate` checks
metadata and aliases, not that pointers are accessible device memory or values finite.
Exact output/input and output/gate aliases are admitted (the reduction finishes before
writes); partial overlap is rejected, and output must not overlap gain. `enqueue`
success means enqueued, not completed: retain all views until physical completion and
use backend readiness for later operations. A launch error still needs normal
settlement. Build (CMake 3.28+, CUDA toolkit, outputs outside Inference folders) and run
the native example from the repository root with the GPU's real architecture:

```sh
cmake -S inference/rmsnorm-bf16/code -B build/rmsnorm -DCRIT_RMSNORM_EXAMPLE=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/rmsnorm --config Release
```

Then run `build/rmsnorm/crit_rmsnorm_example` (or
`build/rmsnorm/Release/crit_rmsnorm_example.exe` with Visual Studio).

## Worked example

Run from `code/`. It checks the gain conventions and gate formula; host floats do not
reproduce native BF16 arithmetic.

```python
from math import isclose, sqrt
from reference import normalize

plain = normalize(((1.0, 1.0),), (2.0, 2.0), epsilon=1.0)
offset = normalize(((1.0, 1.0),), (1.0, 1.0), epsilon=1.0, mode="unit_offset")
assert plain == offset
assert all(isclose(value, sqrt(2.0)) for value in plain[0])
gated = normalize(((1.0, 1.0),), (2.0, 2.0), epsilon=1.0,
                  mode="gated", gate=((0.0, 0.0),))
assert gated == ((0.0, 0.0),)
try:
    normalize(((1.0, 1.0),), (1.0,), epsilon=1.0)
except ValueError:
    pass
else:
    raise AssertionError("incompatible gain shape accepted")
```

## Adapting it

Preserve the gain convention, epsilon placement, shape and stride contract, arithmetic
profile, borrowed-storage lifetime and, when folding partials, the fixed split order.
Thread count, reduction method, layouts and fusion are free; extending a shape limit
means revisiting validation and the kernel, and vectorization must keep admitted aliases
safe. Fusing with [quantized projection](../quantized-projections/README.md) removes an
intermediate write but the engine must choose where rounding and activation quantization
happen; byte-identical output is not automatically required.

## Costs and validation

O(RW) work for R rows: input is read twice, gain and gate once, output written once;
1024 bytes of static shared memory per block. Small widths idle threads and few rows
limit parallelism. Watch for confusing unit-offset and plain gain, accepting
noncontiguous views, misplaced epsilon, early buffer release, and treating a launch
check as success.

Host fixtures check the formula modes, validation and the fold's order and shapes. The
native example compares bounded finite BF16 inputs with an independent double formula
within one BF16 rounding step across 91 combinations of modes, widths, row counts and
in-place outputs (up to width 8192 and 49152 rows), plus aliases, tails and invalid
metadata; it runs in the optional [CUDA operator suite](../../tests/inference/cuda/CMakeLists.txt).
Extreme magnitudes can overflow FP32 squares and are not qualified.
