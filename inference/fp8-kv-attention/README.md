# Attention over an FP8 KV cache

An E4M3 KV cache with one FP32 scale per head vector, plus prompt and decode
attention kernels that read it. At one byte per element it holds twice as many
tokens as BF16 and halves cache reads at long context.

## When to use it

When the KV cache decides how much context fits. Decode launch shape does not
depend on capacity, so an engine built for a huge context (`crit eval run
--context`) is as fast on short requests, and one CUDA graph serves every length.
Not for a cache that is small next to the weights, a paged or shared cache (adapt
the loads or use [attention-state](../attention-state/README.md)), or models
needing finer scales.

## How it works

**Storage.** `scale = max |x| / 448`, `code = e4m3(x / scale)` with saturation;
codes `[position][kv_heads][head_dim]`, scales `[position][kv_heads]`. `write`
runs after key normalization and rotary embedding.

**Reading.** Both kernels prefetch the next 32-key tile into registers and
dequantize into a swizzled shared stage, so the BF16 tensor-core path of
[flash-prefill-attention](../flash-prefill-attention/README.md) runs unchanged.
Prompt blocks own 128 queries of one head. Decode uses a fixed `blocks` per KV
head, each merging 128-key pieces in key order, then a fixed-order combine; a
multi-column verify is one launch.

**Column invariance.** Tiles after a row's position leave it bit for bit
unchanged and the combine order is fixed, so a decode column alone gives exactly
its bits inside a multi-column verify.

## Worked example

From `code/`:

```python
from reference import decode, dequantize, quantize

keys = [0.25, -1.5, 3.0, 0.0]
codes, scale = quantize(keys)
assert scale == 3.0 / 448 and decode(codes[2]) * scale == 3.0  # the largest element maps to 448
read = dequantize(codes, scale)                                  # what the attention kernels read
assert all(abs(x - y) <= abs(x) / 15 + scale * 0.002 for x, y in zip(keys, read))
```

The CUDA interface ([fp8_kv_attention.h](code/fp8_kv_attention.h),
[fp8_kv_attention.cu](code/fp8_kv_attention.cu)):

```cpp
namespace kv = crit::fp8_kv;
kv::Cache cache{k_codes, v_codes, k_scales, v_scales, kv_heads, head_dim};
kv::write(cache, k_rows, v_rows, kv_heads * head_dim, T, pos0, stream);  // chunk's keys and values first
kv::prefill({q, heads * head_dim, cache, out, heads * head_dim, T, pos0, heads, scale}, stream);
// decode / verify inside a captured graph: the position comes from device memory
kv::decode({q, heads * head_dim, cache, out, heads * head_dim, columns, 0, heads, device_pos0, scale,
            blocks, partial}, stream);
```

`partial` holds `decode_partial_floats(kv_heads, head_dim, blocks)` floats; about
the SM count divided by KV heads is a good `blocks`.

## Validation

`tests/inference/cuda/fp8_kv_attention_tests.cu` checks head dimensions 64, 128
and 256 against an FP64 softmax over the dequantized cache (at most 1.5e-3 of the
output magnitude on sm_120) and decode columns bit for bit against multi-column
calls. In one measured engine (Qwen3.8-27B, RTX 5090) the FP8 cache raised the
context that fits from about 122K to 244,736 tokens at 0.2% of the quality budget.
