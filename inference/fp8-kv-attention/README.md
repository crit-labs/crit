# Attention over an FP8 KV cache

An E4M3 KV cache with one FP32 scale per head vector, plus prompt and decode
attention kernels that read it. At one byte per element it holds twice as many
tokens as BF16 and halves cache reads at long context.

## When to use it

When the KV cache decides how much context fits. Decode launch shape does not
depend on capacity, so an engine built for a huge context (`crit eval run
--context`) is as fast on short requests, and one CUDA graph serves every length.
Not for a cache that is small next to the weights, a paged or shared cache (use
[paged-attention](../paged-attention/README.md), which reads the same E4M3
representation through block tables), or models needing finer scales.

## How it works

**Storage.** `scale = max |x| / 448`, `code = e4m3(x / scale)` with saturation;
codes `[position][kv_heads][head_dim]`, scales `[position][kv_heads]`. `write`
runs after key normalization and rotary embedding, on BF16 rows or on FP32 rows
(rounded once, straight to E4M3), at a host position or at one read from device
memory, so a captured decode step writes its own keys; such a write skips rows
outside the cache.

**Reading.** Both kernels prefetch the next 32-key tile into registers and
dequantize into a swizzled shared stage, so the BF16 tensor-core path of
[flash-prefill-attention](../flash-prefill-attention/README.md) runs unchanged.
Prompt blocks own 128 queries of one head. Decode uses a fixed `blocks` per KV head,
each merging 128-key pieces in key order, then a fixed-order combine (block weights
computed in parallel, the denominator and each dimension's sum taken in block order
with the partial outputs loaded 16 at a time); a multi-column verify is one launch.
A decode block runs two key streams of four warps, as
[paged-attention](../paged-attention/README.md)'s decode does: they take alternate
tiles of the block's pieces with their own stage and softmax state and synchronize
only within themselves, so one stream's dequantization and softmax overlap the
other's tensor-core work, and the block merges stream 0 then stream 1 into its one
partial result. That takes `decode_shared_bytes(head_dim, 2)` of shared memory (96
KB at head dimension 256); where the device's per-block limit is smaller the block
runs one stream. `Decode::streams` forces one or two. A warp skips the per-key mask
on tiles all its rows see whole and the output rescale when none of its row maxima
moved (decode rows no column uses see every key, so they never force the mask); both
shortcuts leave every bit unchanged.

**E4M3 prompt attention (opt-in).** `prefill_fp8` takes the same `Prefill` and
runs Q·Kᵀ and P·V on E4M3 tensor cores: queries are quantized per row and head,
key and value codes are used as stored, and each probability is multiplied by
its key's value scale over the largest value scale seen so far before rounding
to E4M3 (the accumulator is kept in those units and rescaled only when a row
maximum or that scale grows). V's B fragments come from `ldmatrix.trans` on b16
code pairs re-packed with byte permutes. 64-key tiles are double buffered with
`cp.async`. It trades BF16 rounding for E4M3 in queries and probabilities, so
measure whole-model quality before adopting it. E4M3 `mma.sync` exists from sm_89 on:
the file also builds for sm_80 and sm_86, where `write`, `prefill` and `decode` run
and `prefill_fp8` returns `cudaErrorNotSupported` without launching. It does the same
whenever the code the runtime loads for the device was built for an older target
(for example sm_80 PTX JIT-compiled on a newer GPU), so an engine can fall back to
`prefill`.

**Column invariance.** Tiles after a row's position leave it bit for bit
unchanged, and the tile-to-stream assignment, the merge and the combine order are
fixed, so a decode column alone gives exactly its bits inside a multi-column verify.
One and two streams round differently, so calls that must agree use the same stream
count (the automatic choice is fixed per device and head dimension).

## Sliding windows, sinks and soft-caps

Several model families attend over a sliding window on some layers, add a learned
per-head sink logit to the softmax, or cap logits; the prompt and decode kernels support all three without
changing its arithmetic elsewhere.

- **Window** (`window > 0`): a query at position `p` sees keys `p - window + 1 .. p`. The
  block starts at the tile holding its oldest query's first key and never reads earlier
  keys, so the work follows the window rather than the context.
- **Ring** (`ring > 0`, contiguous caches): position `j` is row `j % ring`, so a window
  layer keeps `ring >= window + T - 1` rows instead of the whole context; a launch reads
  only rows its own writes leave intact.
- **Sinks** (`sinks[h]`, scaled-score units): `exp(sink)` joins each row's denominator
  after the last tile and adds no value, as in models that learn an attention sink.
- **Soft-cap** (`softcap > 0`): `score = softcap * tanh(score / softcap)` after the scale.

**Trees** (`Decode::tree`): column `t` of a verify sees the round's column `j` only when
bit `j` of `tree[t]` is set (its ancestors and itself, in topological order), for
speculative token trees; with a window, a column's window is measured from its depth
(`popcount - 1`), so trees need `window > 64`.

Masked keys contribute exact zeros, so a window, a ring, a tree mask and skipped tiles
change no bits of the visible keys' contribution: a windowed column decoded alone still
equals the same column in a verify.

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
kv::Cache cache{k_codes, v_codes, k_scales, v_scales, kv_heads, head_dim, /*ring*/ 0, capacity};
kv::write(cache, k_rows, v_rows, kv_heads * head_dim, T, pos0, stream);  // chunk's keys and values first
kv::prefill({q, heads * head_dim, cache, out, heads * head_dim, T, pos0, heads, scale}, stream);
// or kv::prefill_fp8 with the same problem (out_stride a multiple of 4)
// decode / verify inside a captured graph: the position comes from device memory
kv::write(cache, k_f32, v_f32, kv_heads * head_dim, columns, device_pos0, stream);  // BF16 or FP32 rows
kv::decode({q, heads * head_dim, cache, out, heads * head_dim, columns, 0, heads, device_pos0, scale,
            blocks, partial}, stream);
```

`partial` holds `decode_partial_floats(kv_heads, head_dim, blocks)` floats; about
the SM count divided by KV heads is a good `blocks`. `capacity` (rows the arrays hold,
0 = unchecked) guards the buffers: a host position past it is rejected, and a decode
whose device position is negative or puts a column at or past it reads nothing and
writes NaN, so a wrong position in device memory shows in the output instead of
reading past the cache.

## Validation

`tests/inference/cuda/fp8_kv_attention_tests.cu` checks head dimensions 64, 128 and
256 against an FP64 softmax over the dequantized cache (at most 1.5e-3 of the output
magnitude on sm_120) and decode columns bit for bit against multi-column calls,
including every verify width `T` from 1 to `64 / (heads / kv_heads)` at 2,048,
3,001, 7,168 and 16,384 keys, with and without a window, for 1, 5 and SM-count /
`kv_heads` blocks, and up to 16 columns at 65,537 keys, for one and two key streams
(the automatic choice gives the bits of the count it picks), and that device
positions outside a cache's `capacity` give NaN rows while a fitting one keeps the
uncapped cache's bits. FP32 writes equal the rule computed on the host,
device-position writes equal host-position ones and leave rows outside the cache
(guard rows on both sides) untouched, and a captured write, decode and position
advance replays equal to eager calls bit for bit. In one measured engine
(Qwen3.8-27B, RTX 5090) the FP8 cache raised the context that fits from about 122K
to 244,736 tokens at 0.2% of the quality budget.
[attention_variants_tests.cu](../../tests/inference/cuda/attention_variants_tests.cu)
checks windows over a ring (with NaN codes in every row a launch must not read),
sinks, soft-caps and a tree verify against an FP64 reference for prompt,
`prefill_fp8` and decode (one and two streams), and windowed columns alone against
the verify bit for bit. Both test files build for sm_80, sm_86, sm_89, sm_90, sm_100a
and sm_120; on code built below sm_89 they expect `prefill_fp8` to return `cudaErrorNotSupported`
(checked by running sm_80 PTX on an RTX 5090, where the BF16 paths pass as well).
`prefill_fp8` passes the same cases within 7.3e-2 of the output magnitude (E4M3
rounding of queries and probabilities) and, on the RTX 5090 with 4,096 queries over
131K keys, runs in 35.7 ms instead of 73.1 ms at head dimension 256 and 29.7 ms
instead of 54.9 ms at 128. In one engine it cost 2 points of the prefill quality
budget (16.7% to 18.8%). Two key streams decode one column over 262,144 keys in 356
us instead of 494 us on the RTX 5090 (head dimension 256, 24/4 heads, 42 blocks, L2
flushed), 1,530 GB/s of KV read.
