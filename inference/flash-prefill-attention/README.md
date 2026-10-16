# Flash prefill attention

A flash-style tensor-core kernel for the attention of a prompt chunk: it walks the
visible keys in tiles, keeps a running row maximum and sum in registers and accumulates
the output directly, instead of writing the full score matrix `S = Q K^T` to memory. At
long context that matrix dominates (a 2,048-token chunk over 14K keys with 24 heads is
2.8 GB of FP32 scores per layer). In one measured engine (head dimension 256, 6 query
heads per key head) attention at 12K context fell from 148 to 86 ms per 2,048-token
chunk, as accurate as the score-matrix path against an FP32 reference.

## When to use it

Use it for prompt chunks once the context is long; for short prompts either formulation
is fine. Decode (one query per head) splits over keys instead; see
[partitioned-attention](../partitioned-attention/README.md).

## How it works

**Causal chunk.** Query `t` sits at absolute position `pos0 + t` and sees keys
`0 .. pos0 + t`. The cache holds `pos0 + T` positions, and head `h` reads key/value head
`h / (heads / kv_heads)`.

**Online softmax.** Per key tile with scores `s_j`: `m' = max(m, max s_j)`,
`c = exp(m - m')`, `p_j = exp(s_j - m')`, `l' = l c + sum p_j`, `o' = o c + sum p_j v_j`;
the output is `o / l` after the last tile. This equals the direct softmax in exact
arithmetic, in one pass over the keys.

**Tensor cores.** A block of 4 warps owns 64 queries of one head (16 per warp) and walks
keys in tiles of 32. `S` comes from `mma.m16n8k16` with Q and K fragments from
`ldmatrix`. Probabilities are rounded to BF16 and the `S` accumulator layout is reused
directly as the A fragment of `P V`; the row sums add the same rounded values, so weights
stay normalized. V fragments come from `ldmatrix.trans`.

**Memory.** Q, K and V tiles sit in shared memory with a 16-byte XOR swizzle (chunk `c`
of row `r` at `c ^ (r & 7)`), making every `ldmatrix` conflict free. The next K/V tile
loads by `cp.async` while the current one is used (two stages). Output accumulators take
`head_dim / 2` registers per thread, which limits occupancy at head dimension 256.

## What the code supplies

- [flash_prefill.h](code/flash_prefill.h) / [flash_prefill.cu](code/flash_prefill.cu):
  `Problem` (BF16 Q `[T][q_stride]`, contiguous BF16 K and V caches
  `[pos0 + T][kv_stride]`, FP32 output, `T`, `pos0`, heads, key/value heads, head
  dimension, score scale), `enqueue(Problem, stream)` for head dimensions 64, 128 and
  256, and `shared_bytes(head_dim)`.
- [reference.py](code/reference.py): `attention` (direct definition), `flash_attention`
  (tiled online softmax, optionally rounding probabilities), `bf16` and
  `max_difference`.

## What the engine supplies

Q after normalization and rotary embedding, the KV cache with the chunk's own keys and
values written before the call, the following gate or projection, and any paged or
quantized cache (this kernel reads contiguous BF16 rows). Pass
`scale = 1 / sqrt(head_dim)`, or 1 if Q is pre-scaled. Rows must be 16-byte aligned. The
call is asynchronous on the caller's stream, borrows every buffer until the stream
reaches it, and sets the dynamic shared-memory limit on first use. Unsupported geometry
returns `cudaErrorInvalidValue`.

## Worked example

The tiled online softmax equals the direct definition; rounding probabilities to BF16,
as the tensor-core path does, moves the result by about 1e-3.

```python
import random
import reference as ref

rng = random.Random(0)
T, pos0, heads, kv_heads, dim = 5, 70, 4, 2, 8
q = [[[rng.gauss(0, 1) for _ in range(dim)] for _ in range(heads)] for _ in range(T)]
k = [[[rng.gauss(0, 1) for _ in range(dim)] for _ in range(kv_heads)] for _ in range(pos0 + T)]
v = [[[rng.gauss(0, 1) for _ in range(dim)] for _ in range(kv_heads)] for _ in range(pos0 + T)]
direct = ref.attention(q, k, v, pos0, heads, kv_heads, scale=dim ** -0.5)
tiled = ref.flash_attention(q, k, v, pos0, heads, kv_heads, scale=dim ** -0.5, key_tile=32)
rounded = ref.flash_attention(q, k, v, pos0, heads, kv_heads, scale=dim ** -0.5, round_p=ref.bf16)
print("online softmax vs direct:", ref.max_difference(direct, tiled))
print("with BF16 probabilities:", ref.max_difference(direct, rounded))
assert ref.max_difference(direct, tiled) < 1e-12 and ref.max_difference(direct, rounded) < 1e-2
try:
    ref.attention(q, k[:-1], v[:-1], pos0, heads, kv_heads, 1.0)
except ValueError as error:
    print("short cache rejected:", error)
```

## Adapting it

Preserve the causal rule, the grouping of query heads onto key/value heads, and the
normalization: the row sum adds exactly the probabilities used in `P V`, and a fully
masked tile leaves a row unchanged (correction 1, probabilities 0). Free: query and key
tile sizes, stage count, head dimensions, probability format (FP16, or FP8 with its own
error), sharing a K/V tile across a group's query heads, and the cache layout (paged
caches need per-tile indirection, quantized caches dequantization in the load).

## Costs and validation

Work is `4 x heads x head_dim` FLOPs per visible query-key pair at BF16 tensor rates; K/V
tiles are re-read once per 64-query block and head. Shared memory is about
`(64 + 4 x 32) x head_dim x 2` bytes per block (96 KB at 256). BF16 probabilities add
about 1e-3 relative error against FP32; measure whole-model quality.

[reference.py](code/reference.py) checks the online softmax on the host. The
[CUDA test](../../tests/inference/cuda/flash_prefill_tests.cu) compares the kernel with
an FP64 softmax for head dimensions 64, 128 and 256, grouped and ungrouped heads,
single-token and multi-tile chunks and a later chunk over earlier keys; the largest error
was 1.3e-3 of the output magnitude on an sm_120 device. Measure speed on the engine's own
prompts.
