// Attention over an FP8 KV cache: writing keys and values as E4M3 codes with one scale per position and key/value
// head, causal prompt attention (flash style, BF16 tensor cores) and decode attention whose launch and work do not
// depend on the cache's capacity.
//
// Cache layout: codes [position][kv_heads][head_dim] (one byte each), scales [position][kv_heads] (FP32). The value
// of an element is code * scale with scale = max |x| over the head's vector / 448 (1 for an all-zero vector).
// Kernels dequantize tiles to BF16 in shared memory: code * scale is rounded once, to BF16.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace crit::fp8_kv {

struct Cache {
    std::uint8_t* k = nullptr;  // [capacity][kv_heads][head_dim]
    std::uint8_t* v = nullptr;
    float* k_scale = nullptr;   // [capacity][kv_heads]
    float* v_scale = nullptr;
    int kv_heads = 0, head_dim = 0;
};

// Quantize T rows of keys and values (BF16, row t at k + t * stride, head g at column g * head_dim) into positions
// pos0 .. pos0 + T - 1 of the cache.
cudaError_t write(const Cache& cache, const __nv_bfloat16* k, const __nv_bfloat16* v, int stride, int T, int pos0,
                  cudaStream_t stream);

// Causal prompt attention: query t (absolute position pos0 + t, row t at q + t * q_stride, head h at column
// h * head_dim) sees cache positions 0 .. pos0 + t, which must already be written. out[t][h] in FP32.
struct Prefill {
    const __nv_bfloat16* q = nullptr;
    int q_stride = 0;
    Cache cache;
    float* out = nullptr;
    int out_stride = 0;
    int T = 0, pos0 = 0, heads = 0;
    float scale = 0.f;  // multiplies q . k before the softmax
};
cudaError_t prefill(const Prefill& problem, cudaStream_t stream);
std::size_t prefill_shared_bytes(int head_dim);

// The same problem on E4M3 tensor cores: queries quantized per row and head, key and value codes used as stored,
// probabilities rounded to E4M3 with the per-key value scale folded in (over the largest value scale seen so far).
// About twice prefill's throughput at long contexts with larger rounding error (E4M3 instead of BF16 queries and
// probabilities), so an engine opts in after measuring whole-model quality. out_stride must be a multiple of 4.
cudaError_t prefill_fp8(const Prefill& problem, cudaStream_t stream);
std::size_t prefill_fp8_shared_bytes(int head_dim);

// Decode attention for T query columns at positions pos0 + t over keys 0 .. pos0 + T - 1. The grid is fixed: blocks
// per key/value head, each merging 128-key pieces b, b + blocks, ... in key order, so neither the launch nor the work
// depends on the cache's capacity, and a CUDA graph captured once serves every length. A tile after a column's position
// leaves that column's state bit for bit unchanged, so each column's result depends only on its own position: one
// column decoded alone equals the same column in a multi-column verify. T * (heads / kv_heads) <= 64.
// pos0 comes from *pos0_device when it is set (device-driven loops), else from pos0.
struct Decode {
    const __nv_bfloat16* q = nullptr;
    int q_stride = 0;
    Cache cache;
    float* out = nullptr;
    int out_stride = 0;
    int T = 0, pos0 = 0, heads = 0;
    const int* pos0_device = nullptr;
    float scale = 0.f;
    int blocks = 0;            // per key/value head, 1 .. 1024; about the SM count / kv_heads (or twice that) is typical
    float* partial = nullptr;  // decode_partial_floats(...) floats of scratch
};
std::size_t decode_partial_floats(int kv_heads, int head_dim, int blocks);
cudaError_t decode(const Decode& problem, cudaStream_t stream);

}  // namespace crit::fp8_kv
