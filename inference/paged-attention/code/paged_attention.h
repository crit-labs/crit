// Attention for a ragged batch of sequences whose keys and values live in fixed-size pages addressed by block
// tables (Core's StateCache: one PageId per page_tokens rows, shared prefixes, copy-on-write tails).
//
// Page layout, per attention layer: row r of page p is physical row p * page_tokens + r, and physical rows are laid
// out [row][kv_heads][head_dim]. BF16 pages store values; E4M3 pages store one-byte codes with one FP32 scale per
// row and key/value head ([row][kv_heads]), value = code * scale with scale = max |x| over the head vector / 448
// (1 for an all-zero vector), the representation of fp8-kv-attention. Kernels round every key and value to BF16
// (E4M3: code * scale rounded once) and run Q K^T and P V on BF16 tensor cores with an FP32 online softmax.
//
// A batch is described by device arrays, so its launch shape depends only on host counts (sequences, blocks, the
// largest query count), never on lengths or page ids: a CUDA graph captured once serves every length and table.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace crit::paged_attention {

enum class Format : int { bf16 = 0, e4m3 = 1 };

struct Pages {
    void* k = nullptr;          // [pages * page_tokens][kv_heads][head_dim]: BF16 values or E4M3 codes
    void* v = nullptr;
    float* k_scale = nullptr;   // E4M3 only: [pages * page_tokens][kv_heads]
    float* v_scale = nullptr;
    Format format = Format::bf16;
    int page_tokens = 0, kv_heads = 0, head_dim = 0;
};

// Sequence s reads keys 0 .. kv_lens[s] - 1 through tables[s * table_stride + key / page_tokens]. Its queries are rows
// q_start[s] .. q_start[s + 1] - 1 of q and out; with n of them, query i sits at position kv_lens[s] - n + i and sees
// keys 0 .. that position (its own key included, so a step writes its keys first). A sequence with no queries is an
// idle slot (padding for a captured batch size) and writes nothing.
struct Batch {
    int seqs = 0;                          // host count; a grid dimension
    const std::int32_t* tables = nullptr;  // [seqs][table_stride] page ids
    int table_stride = 0;
    const std::int32_t* kv_lens = nullptr; // [seqs]
    const std::int32_t* q_start = nullptr; // [seqs + 1], nondecreasing
};

// Store `rows` rows of keys and values (BF16, row t at k + t * stride, head g at column g * head_dim) at physical rows
// slots[t] = page * page_tokens + offset (Core's Round::kv); a negative slot skips its row. E4M3 pages quantize each
// head vector with its own scale.
cudaError_t write(const Pages& pages, const __nv_bfloat16* k, const __nv_bfloat16* v, int stride, int rows,
                  const std::int32_t* slots, cudaStream_t stream);

// Decode and speculative verify: up to max_columns queries per sequence. Each (sequence, key/value head) runs `blocks`
// blocks; block b merges 128-key pieces b, b + blocks, ... in key order, and a combine adds the blocks' partial
// results in block order, so results are deterministic and depend only on a query's position and the batch's blocks:
// a column decoded alone equals the same column inside a multi-column verify, and a sequence's result does not depend
// on the other sequences in the batch. max_columns * (heads / kv_heads) <= 64.
struct Decode {
    const __nv_bfloat16* q = nullptr;  // [rows][q_stride]; head h at column h * head_dim
    int q_stride = 0;
    Pages pages;
    Batch batch;
    float* out = nullptr;              // [rows][out_stride]; FP32
    int out_stride = 0;
    int heads = 0;
    float scale = 0.f;                 // multiplies q . k before the softmax
    int max_columns = 1;
    int blocks = 1;                    // per sequence and key/value head, 1 .. 1024
    float* partial = nullptr;          // decode_partial_floats(...) floats of scratch
};
std::size_t decode_partial_floats(int seqs, int kv_heads, int head_dim, int blocks, int max_columns, int heads);
cudaError_t decode(const Decode& problem, cudaStream_t stream);

// Varlen causal prefill: every sequence's queries (ragged counts) attend over its paged prefix and its own new keys,
// which must already be written. Blocks own 128 queries of one head of one sequence, the latest first.
struct Prefill {
    const __nv_bfloat16* q = nullptr;
    int q_stride = 0;
    Pages pages;
    Batch batch;
    float* out = nullptr;
    int out_stride = 0;
    int heads = 0;
    float scale = 0.f;
    int max_queries = 0;               // the largest query count of a sequence; sets the grid
};
cudaError_t prefill(const Prefill& problem, cudaStream_t stream);
std::size_t prefill_shared_bytes(int head_dim);
std::size_t decode_shared_bytes(int head_dim);

}  // namespace crit::paged_attention
