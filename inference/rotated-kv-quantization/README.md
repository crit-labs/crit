# Rotate keys and values before low-bit KV quantization

When the KV cache is stored as low-bit integers (int8 or int4 with a scale per group), a
few large channels coarsen every other value in their group. A fixed orthogonal rotation
spreads that energy and lowers group maxima, which can improve quantization accuracy.
Rotation preserves unquantized attention exactly; the quantized result is still an
approximation, and rotation can also increase error.

## When to use it

Use it with a low-bit KV cache whose channel magnitudes are uneven, and keep it only when
measured quality, memory and cost justify it. It is unnecessary for BF16/FP16 caches or
already even channels, and adds a transform per stored row, query and output.

## How it works

Let `R = H * diag(s)`, with `H` the normalized Walsh-Hadamard matrix of the head
dimension (a power of two; orthonormal, symmetric) and `s` seeded `+1`/`-1` signs. Then:

```text
(R q) . (R k) = q . k                        scores are unchanged
R^T * sum_i p_i (R v_i) = sum_i p_i v_i      outputs are unchanged
```

The engine stores `quantize(R k)` and `quantize(R v)`, scores `R q` against stored keys,
accumulates weighted stored values in the rotated basis and applies `R^T = diag(s) * H`
to the result. Apply `R` after RoPE and key normalization, which act on the original
channels, and rotate the query at the same point. Unrotate the output before the output
projection, or fold `R^T` into its weights offline.

Each group of 32 rotated values gets one FP32 scale `amax / qmax` (`qmax = 127` for
int8, `7` for int4) and codes rounded to nearest even. Signs and head dimension are part
of the cache representation: a cache written with one `s` cannot be read with another.

## What the code supplies

- [rotation.py](code/rotation.py): `signs(dim, seed)`, normalized `fwht`,
  `rotate`/`unrotate`, per-group `quantize`/`dequantize` (4 or 8 bits), FP64 `attention`
  and `compressed_attention(q, keys, values, s, bits)` to compare rotated and unrotated
  storage.
- [rotated_kv.h](code/rotated_kv.h) / [rotated_kv.cu](code/rotated_kv.cu): reference
  CUDA kernels (one block per row, shared-memory transform) `enqueue_rotate_quantize`
  (int8 codes and per-32 scales), `enqueue_rotate` (queries), `enqueue_unrotate`
  (outputs) and `enqueue_dequantize`, in place allowed, dimensions 32 to 1024 (powers of
  two).

## What the engine supplies

The attention kernel that dequantizes stored rows on the fly, int4 packing if it stores
two codes per byte, the cache layout and its representation identity. On append,
rotate-quantize new key and value rows; per step, rotate the query once and unrotate the
output (or use a folded projection). Record seed, dimension, bit width and group size in
the representation key, and measure quality with Eval's
[full-vocabulary comparison](../../docs/eval/MEASUREMENTS.md#full-vocabulary-comparison)
against both the unquantized cache and unrotated storage at the same bits and group.

## Worked example

Run from `code/`. This key set with one large channel loses more to int8 storage than
the same keys rotated first; it shows a possible benefit, not a universal ordering.

```python
import random
from rotation import attention, compressed_attention, rotate, signs, unrotate

dim = 64
s = signs(dim, seed=7)
rng = random.Random(1)
keys = [[rng.gauss(0, 1) * (30 if j == 5 else 1) for j in range(dim)] for _ in range(16)]
q = [rng.gauss(0, 0.1) for _ in range(dim)]

rq, rk = rotate(q, s), rotate(keys[0], s)
assert abs(sum(a * b for a, b in zip(rq, rk)) - sum(a * b for a, b in zip(q, keys[0]))) < 1e-9
assert max(abs(a - b) for a, b in zip(unrotate(rotate(q, s), s), q)) < 1e-12

exact = attention(q, keys, keys)
def error(out):
    return sum((a - b) ** 2 for a, b in zip(out, exact))
plain, rotated = compressed_attention(q, keys, keys, bits=8), compressed_attention(q, keys, keys, s, bits=8)
assert error(rotated) < error(plain)
assert error(compressed_attention(q, keys, keys, s, bits=4)) > error(rotated)   # 4-bit costs more
```

Counterexample: `[7, 1, 0, ..., 0]` at dimension 128, int4/group32, sign seed 0 is
stored exactly unrotated; rotation cuts its maximum from 7 to about 0.707 yet raises
reconstruction MSE to about 0.000319. Smaller scales alone do not prove lower error.

## Adapting it

Preserve: one `R` for queries, keys, values and outputs with the output rotated back;
rotation after position encoding and key normalization; signs, dimension, bits and group
size in the representation; symmetric codes with one FP32 scale per 32 values, ties to
even. Free: the seed, folding `R^T` into the output projection (and `R` into the key and
value projections when no position encoding sits between), int4 packing, group size, and
fusing rotate-quantize into append and dequantization into attention.

## Costs and validation

The transform is `D log2 D` additions per vector. Storage is one byte per value (half
packed) plus 4 bytes per 32 values. A mismatched rotation, or a forgotten query rotation
or output unrotation, gives wrong attention with no error, so tie the representation key
to the rotation. Int4 loses noticeably more than int8.

[Python fixtures](../../tests/test_techniques.py) cover self-inverse transforms,
dot-product preservation, sign determinism, quantization bounds, invalid inputs, the
outlier int8 benefit and int4 counterexamples. The
[CUDA fixture](../../tests/inference/rotated_kv_tests.cpp) (under `CRIT_INFERENCE_CUDA`)
checks 64 rows of dimension 128 against an FP64 Walsh-Hadamard: rotation, inverse, in
place, codes within half a step, rejected launches and lower rotated int8 error. No
model-quality or throughput claim.
