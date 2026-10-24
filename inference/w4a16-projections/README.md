# Run published 4-bit checkpoints with BF16 activations

Run the 4-bit checkpoints most models are published in (GPTQ, AWQ and compressed-tensors
int4 groups, and MXFP4 as gpt-oss ships it) as stored, with BF16 activations: convert each
weight once to one canonical layout, then project with a decode GEMV that reads 4 bits per
weight and a prompt GEMM on BF16 tensor cores. This is the 4-bit path for checkpoints that
are not NVFP4 and for GPUs without FP4 tensor cores (sm_80 and later).

## When to use it

When the run must use the published weights as they are (`weights_policy` other than
`transformable`) and they are GPTQ, AWQ, compressed-tensors `pack-quantized` int4 or MXFP4;
or on Ampere and Ada GPUs, where crit's FP4 path ([stored-tile-gemm](../stored-tile-gemm/README.md))
cannot run. Decode at batch 1 is bound by weight bytes, so 4-bit codes read about a quarter
of BF16's bytes (Marlin, the reference W4A16 kernel, reports close to 4x over FP16 up to
16 to 32 tokens on Ampere). Not for NVFP4 checkpoints on sm_120 (stored-tile-gemm runs them on
FP4 tensor cores), and not when activations should be quantized too (W4A4, W8A8).

## How it works

**Canonical layout.** For an `N x K` weight: codes `[N][K / 2]` bytes with the even column
in the low nibble; int4 groups add FP32 scales and uint8 zeros `[N][K / group]`
(`value = (q - zero) * scale`, groups of 32, 64 or 128); MXFP4 adds one E8M0 exponent per
row and 32 columns (`value = e2m1(q) * 2^(e - 127)`). Converting moves and widens bytes only:
every represented value is unchanged.

**Formats.** GPTQ packs eight columns per int32 down K (`qweight [K / 8][N]`) and eight
rows per int32 across N for zeros (`qzeros [G][N / 8]`); `checkpoint_format: "gptq"` stores
`zero - 1`, `"gptq_v2"` the zero. With act-order (`g_idx`), columns of one group are
scattered: the converter sorts columns by group (stably) and returns `perm`, and the
engine gathers activations `x'[j] = x[perm[j]]` before the projection, so every group is
contiguous again. AWQ packs eight columns per int32 across N (`qweight [K][N / 8]`) in the
order `0, 2, 4, 6, 1, 3, 5, 7`. compressed-tensors `pack-quantized` packs down K per row
(`weight_packed [N][K / 8]`), symmetric with zero 8 unless `weight_zero_point` is given.
MXFP4 blocks `[N][K / 32][16]` and scales `[N][K / 32]` are canonical already. See
[w4.py](code/w4.py) for each unpacking.

**Decode GEMV (1 to 8 tokens).** A warp owns an output row; each lane reads 16 code bytes
(32 weights) per step, dequantizes in FP32 registers and accumulates every token in a fixed
order, reduced to lane 0 by a fixed shuffle tree. A token's result does not depend on how
many tokens share the call, so verify columns equal decode bit for bit.

**Prompt GEMM.** A block owns 64 rows by 64 tokens; per 64-column step the weights are
dequantized to BF16 into a swizzled shared tile (the `mma.m16n8k16` A operand) and the
activations staged beside it, then four warps run the tensor cores. Weights round to BF16
only here (prompts tolerate it; decode never sees it).

## What the code supplies

| File | Supplies |
| --- | --- |
| [w4a16.h](code/w4a16.h) / [w4a16.cu](code/w4a16.cu) | `Weights` (int4 groups or MXFP4), `enqueue_gemv`, `enqueue_gemm`, `enqueue_gather_columns`; converters `enqueue_from_gptq` (act-order permutation, zero offset), `enqueue_from_awq`, `enqueue_from_compressed` |
| [w4.py](code/w4.py) | Host reference: every format's unpacking, the canonical layout and the projection |

## What the engine supplies

The checkpoint tensors on the device (once, for conversion), canonical buffers for every
quantized linear layer, the act-order gather into a scratch activation row, fusion of the
output (`add` accumulates, for example into the residual), and the choice of GEMV or GEMM by
token count. Non-quantized layers (embeddings, the head, norms) stay in their own format.

## Worked example

From `code/`: one 2 x 16 weight packed as GPTQ (with act-order and the v1 zero offset), AWQ
and compressed-tensors unpacks to the same represented weights, and the act-order
permutation leaves the projection unchanged.

```python
import random
from w4 import AWQ_ORDER, from_awq, from_compressed, from_gptq, int4_weights, mxfp4_weights, project

rng = random.Random(1)
N, K, group = 8, 16, 8
q = [[rng.randrange(16) for _ in range(K)] for _ in range(N)]
z = [[rng.randrange(1, 16) for _ in range(K // group)] for _ in range(N)]
s = [[rng.choice([0.01, 0.02, 0.05]) for _ in range(K // group)] for _ in range(N)]
want = [[(q[n][k] - z[n][k // group]) * s[n][k // group] for k in range(K)] for n in range(N)]

# GPTQ, act-order: column k belongs to group g_idx[k]; zeros stored minus one ("gptq" format).
g_idx = [k % 2 for k in range(K)]                        # even columns in group 0, odd in group 1
cols = [k for k in range(K) if k % 2 == 0] + [k for k in range(K) if k % 2]
qweight = [[0] * N for _ in range(K // 8)]
for n in range(N):
    for j, k in enumerate(cols):                         # canonical column j is checkpoint column k
        qweight[k // 8][n] |= q[n][j] << (4 * (k % 8))
qzeros = [[sum((z[n][g] - 1) << (4 * (n % 8)) for n in range(8))] for g in range(2)]
scales = [[s[n][g] for n in range(N)] for g in range(2)]
codes, sc, zr, perm = from_gptq(qweight, qzeros, scales, K, N, group, g_idx=g_idx, zero_offset=1)
assert perm == cols and int4_weights(codes, sc, zr) == want
x = [rng.gauss(0, 1) for _ in range(K)]
y = project(int4_weights(codes, sc, zr), x, perm)
assert all(abs(a - b) < 1e-12 for a, b in zip(y, [sum(want[n][j] * x[cols[j]] for j in range(K)) for n in range(N)]))

# AWQ packs eight rows' codes per word in the order 0, 2, 4, 6, 1, 3, 5, 7.
awq = [[sum(q[AWQ_ORDER[i]][k] << (4 * i) for i in range(8))] for k in range(K)]
awq_z = [[sum(z[AWQ_ORDER[i]][g] << (4 * i) for i in range(8))] for g in range(2)]
assert int4_weights(*from_awq(awq, awq_z, scales, K, N, group)[:3]) == want

# compressed-tensors: eight columns per word down each row.
packed = [[sum(q[n][8 * w + i] << (4 * i) for i in range(8)) for w in range(K // 8)] for n in range(N)]
assert int4_weights(*from_compressed(packed, s, K, N, group, zero_point=z)[:3]) == want
assert mxfp4_weights([[bytes([0x21] + [0] * 15)]], [[127 + 2]], 32, 1)[0][:2] == [2.0, 4.0]   # codes 1, 2 (0.5, 1.0) times 2^2
```

## Adapting it

Preserve: every converter's exact bit layout (a format's packing order, zero convention
and act-order permutation are the format's, not a choice), the canonical values, and the
GEMV's per-token reduction order. Adaptable: Marlin-style pipelining (asynchronous copies,
weights pre-shuffled into fragment order at load, as stored-tile-gemm does), split-K for
small grids, FP16 activations, fusing the act-order gather into the previous layer's output,
and keeping FP16 scales.

## Costs and validation

The GEMV reads `K / 2` code bytes per row plus scales; the GEMM dequantizes each weight tile
once per 64 tokens. Both are reference speed: measure against the bytes or FLOPs floor
before relying on them.

[w4a16_tests.cpp](../../tests/inference/w4a16_tests.cpp) packs one random int4 matrix as
GPTQ (act-order, both zero conventions), AWQ and compressed-tensors and checks each converts
to the canonical bytes, scales, zeros and permutation; checks the GEMV for 1 to 8 tokens
and the GEMM against FP64 products for groups of 32, 64 and 128 and for MXFP4; checks that a
token alone equals the same token in a batch; and checks rejections.
[test_w4a16.py](../../tests/test_w4a16.py) checks the reference. No speed is claimed.
