// Attention over an FP8 KV cache: writing keys and values as E4M3 codes with one scale per position and key/value
// head, causal prompt attention (flash style, BF16 tensor cores) and decode attention whose launch and work do not
// depend on the cache's capacity.
//
// Cache layout: codes [row][kv_heads][head_dim] (one byte each), scales [row][kv_heads] (FP32), with position p at
// row p, or at row p % ring for a sliding-window layer's ring buffer. The value of an element is code * scale with
// scale = max |x| over the head's vector / 448 (1 for an all-zero vector). Kernels dequantize tiles to BF16 in shared
// memory: code * scale is rounded once, to BF16.
//
// Every problem also takes the attention variants several model families use: a sliding window (a query at position p
// sees keys p - window + 1 .. p, and keys before the oldest query's window are never read, so a ring of window +
// columns - 1 rows holds a window layer), per-head sink logits that join each row's softmax denominator, and a logit
// soft-cap (score = cap * tanh(score / cap) after the scale). Decode and verify take optional tree masks.
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
    int ring = 0;               // > 0: position p lives at row p % ring (needs a window; ring >= window + rows - 1)
    // > 0: rows the arrays hold (0: unchecked). A linear cache then holds positions 0 .. capacity - 1, and a ring needs
    // ring <= capacity. Launches whose host position passes the capacity are rejected; a decode whose device position
    // is negative or puts a column past it reads nothing and writes NaN (a wrong position in device memory cannot read
    // past the buffers).
    int capacity = 0;
};

// Quantize T rows of keys and values (BF16, row t at k + t * stride, head g at column g * head_dim) into positions
// pos0 .. pos0 + T - 1 of the cache.
cudaError_t write(const Cache& cache, const __nv_bfloat16* k, const __nv_bfloat16* v, int stride, int T, int pos0,
                  cudaStream_t stream);
// The same with pos0 read from device memory when the kernel runs (*pos0_device), so a captured graph can write each
// step's rows: a row whose position lies outside the cache (negative, or at or past a linear cache's capacity) is
// skipped and nothing else changes.
cudaError_t write(const Cache& cache, const __nv_bfloat16* k, const __nv_bfloat16* v, int stride, int T, const int* pos0_device,
                  cudaStream_t stream);
// FP32 rows (for example straight from an FP32 norm and rotary embedding): the same rule on the FP32 values, so each
// element is rounded once (x / scale to E4M3) instead of to BF16 first.
cudaError_t write(const Cache& cache, const float* k, const float* v, int stride, int T, int pos0, cudaStream_t stream);
cudaError_t write(const Cache& cache, const float* k, const float* v, int stride, int T, const int* pos0_device, cudaStream_t stream);

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
    int window = 0;               // > 0: sliding window of `window` keys, the query's own included
    const float* sinks = nullptr; // [heads] sink logits in scaled-score units; null: none
    float softcap = 0.f;          // > 0: score = softcap * tanh(score / softcap)
};
cudaError_t prefill(const Prefill& problem, cudaStream_t stream);
std::size_t prefill_shared_bytes(int head_dim);

// The same problem on E4M3 tensor cores: queries quantized per row and head, key and value codes used as stored,
// probabilities rounded to E4M3 with the per-key value scale folded in (over the largest value scale seen so far).
// About twice prefill's throughput at long contexts with larger rounding error (E4M3 instead of BF16 queries and
// probabilities), so an engine opts in after measuring whole-model quality. out_stride must be a multiple of 4.
// Needs E4M3 mma (sm_89+) in the code the runtime loads for the device: on older devices, or from a build for older
// targets only, it returns cudaErrorNotSupported and launches nothing (the other kernels build and run from sm_80).
cudaError_t prefill_fp8(const Prefill& problem, cudaStream_t stream);
std::size_t prefill_fp8_shared_bytes(int head_dim);

// Decode attention for T query columns at positions pos0 + t over keys 0 .. pos0 + T - 1. The grid is fixed: blocks
// per key/value head, each merging 128-key pieces b, b + blocks, ... in key order, so neither the launch nor the work
// depends on the cache's capacity, and a CUDA graph captured once serves every length. A tile after a column's position
// leaves that column's state bit for bit unchanged, so each column's result depends only on its own position: one
// column decoded alone equals the same column in a multi-column verify. T * (heads / kv_heads) <= 64.
// pos0 comes from *pos0_device when it is set (device-driven loops), else from pos0. A negative device position, or one
// that puts a column past the cache's capacity, gives NaN in every output of the call and reads no cache row.
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
    int window = 0;               // as Prefill
    const float* sinks = nullptr;
    float softcap = 0.f;
    // Token trees: tree[t] has bit j set when column t sees column j (the key at position pos0 + j); a column sees
    // exactly its ancestors and itself, in topological order. Null: the causal chain. With a window, column t's window
    // is measured from its depth (popcount - 1); window > 64 then.
    const std::uint64_t* tree = nullptr;
    // Key streams per decode block: 0 picks 2 where decode_shared_bytes(head_dim, 2) fits the device's per-block
    // shared memory (cudaDevAttrMaxSharedMemoryPerBlockOptin), else 1; 1 or 2 forces it. Two streams of four warps take
    // alternate tiles of the block's pieces and merge at the end, so their softmax and dequantization overlap each
    // other's tensor-core work. A column's bits depend on its position and on the stream count, so keep the count
    // fixed across the calls that must agree (the automatic choice is fixed per device and head dimension).
    int streams = 0;
};
std::size_t decode_shared_bytes(int head_dim, int streams);  // dynamic shared memory of a decode block
std::size_t decode_partial_floats(int kv_heads, int head_dim, int blocks);
cudaError_t decode(const Decode& problem, cudaStream_t stream);

}  // namespace crit::fp8_kv
