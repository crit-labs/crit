#pragma once
// The batched reference engine's paged attention on CUDA (target crit_paged_attention): the KV write, varlen prefill
// and decode/verify kernels, the device page store behind a StateCache and the device batch tables. paged_reference.h
// gives the page layout, the visibility rules and the host reference these kernels follow.
//
// Kernels round every key and value to BF16 (E4M3: code * scale rounded once) and run Q K^T and P V on BF16 tensor
// cores (sm_80 or later) with an FP32 online softmax; probabilities are rounded to BF16 and the row sums use the
// rounded values. Head dimensions 64, 128 and 256.
//
// A batch is described by device arrays, so its launch shape depends only on host counts (sequences, blocks, the
// largest query count), never on lengths or page ids: a CUDA graph captured once serves every length and table.
#include "paged_reference.h"

#include <crit/state.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace crit::paged_attention {

inline void check(cudaError_t error, const char* what) {
    if(error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}

struct Pages {
    void* k = nullptr;          // [pages * page_tokens][kv_heads][head_dim]: BF16 values or E4M3 codes
    void* v = nullptr;
    float* k_scale = nullptr;   // E4M3 only: [pages * page_tokens][kv_heads]
    float* v_scale = nullptr;
    Format format = Format::bf16;
    int page_tokens = 0, kv_heads = 0, head_dim = 0;
};

// Sequence s reads keys 0 .. kv_lens[s] - 1 through tables[s * table_stride + key / page_tokens]. Its queries are rows
// q_start[s] .. q_start[s + 1] - 1 of q and out (paged_reference.h's visibility). A sequence with no queries is an
// idle slot (padding for a captured batch size) and writes nothing.
//
// These counts live on the device, so the launch cannot check them. A sequence with queries that breaks them (more
// queries than the launch's max_columns or max_queries, more queries than kv_lens[s], or kv_lens[s] beyond the
// table_stride * page_tokens keys its table addresses) is not attended: the kernels read none of its keys, write NaN
// to every head of each of its query rows and set *fault to 1 when fault is set. The launch still returns cudaSuccess;
// Tables rejects more queries than keys and short tables on the host, and its max_queries() bounds the launch's.
struct Batch {
    int seqs = 0;                          // host count; a grid dimension
    const std::int32_t* tables = nullptr;  // [seqs][table_stride] page ids
    int table_stride = 0;
    const std::int32_t* kv_lens = nullptr; // [seqs]
    const std::int32_t* q_start = nullptr; // [seqs + 1], nondecreasing
    std::int32_t* fault = nullptr;         // optional: set to 1 (never cleared) when a sequence breaks the counts above
};

// Store `rows` rows of keys and values (BF16, row t at k + t * stride, head g at column g * head_dim) at physical rows
// slots[t] = page * page_tokens + offset (Core's Round::kv); a negative slot skips its row. E4M3 pages quantize each
// head vector with its own scale, as HostPages::write does.
cudaError_t write(const Pages& pages, const __nv_bfloat16* k, const __nv_bfloat16* v, int stride, int rows,
                  const std::int32_t* slots, cudaStream_t stream);

// Decode and speculative verify: up to max_columns queries per sequence (a sequence with more is not attended; see
// Batch). Each (sequence, key/value head) runs `blocks` blocks; block b merges 128-key pieces b, b + blocks, ... in
// key order, and a combine adds the blocks' partial results in block order, so results are deterministic and depend
// only on a query's position and the batch's blocks: a column decoded alone equals the same column inside a
// multi-column verify, and a sequence's result does not depend on the other sequences in the batch.
// max_columns * (heads / kv_heads) <= 64.
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
    int window = 0;                    // > 0: sliding window of `window` keys, the query's own included
    const float* sinks = nullptr;      // [heads] sink logits in scaled-score units; null: none
    float softcap = 0.f;               // > 0: score = softcap * tanh(score / softcap)
    // Token trees ([rows]): paged_reference.h's tree masks; null: the causal chain. With a window, window > 64.
    const std::uint64_t* tree = nullptr;
};
std::size_t decode_partial_floats(int seqs, int kv_heads, int head_dim, int blocks, int max_columns, int heads);
cudaError_t decode(const Decode& problem, cudaStream_t stream);

// Varlen causal prefill: every sequence's queries (ragged counts) attend over its paged prefix and its own new keys,
// which must already be written. Blocks own 128 queries of one head of one sequence, the latest first; keys are
// walked in absolute 32-key tiles, so prompt chunk boundaries change no bits.
struct Prefill {
    const __nv_bfloat16* q = nullptr;
    int q_stride = 0;
    Pages pages;
    Batch batch;
    float* out = nullptr;
    int out_stride = 0;
    int heads = 0;
    float scale = 0.f;
    int max_queries = 0;               // the largest query count of a sequence; sets the grid (see Batch for more)
    int window = 0;                    // as Decode
    const float* sinks = nullptr;
    float softcap = 0.f;
};
cudaError_t prefill(const Prefill& problem, cudaStream_t stream);
std::size_t prefill_shared_bytes(int head_dim);
std::size_t decode_shared_bytes(int head_dim);

// Pages of `layers` attention layers sharing one PageId space, in device memory: layer l's keys at k + l * layer_rows
// rows, laid out as HostPages. copy_rows enqueues one 2-D copy per array on `stream` (all layers at once), so it is
// ordered before later kernels on that stream; the cache calls it synchronously from begin/try_begin. Allocates with
// cudaMalloc: reserve bytes() in Core Resources (Memory::device) for as long as the pages live.
class DevicePages final : public core::StateStorage {
public:
    DevicePages(int layers, int pages, int page_tokens, int kv_heads, int head_dim, Format format, cudaStream_t stream);
    ~DevicePages() override;
    DevicePages(const DevicePages&) = delete;
    DevicePages& operator=(const DevicePages&) = delete;
    Pages layer(int l) const;
    int pages() const noexcept;
    int page_tokens() const noexcept;
    std::size_t bytes() const noexcept;
    void copy_rows(core::PageId from, core::PageId to, std::uint32_t rows) override;
    void copy_slot(std::uint32_t, core::SlotId, core::SlotId) override;   // logic_error: no recurrent state slots
private:
    int layers_, pages_, page_tokens_, kv_heads_, head_dim_;
    Format format_;
    cudaStream_t stream_;
    std::byte* k_ = nullptr;
    std::byte* v_ = nullptr;
    float* k_scale_ = nullptr;
    float* v_scale_ = nullptr;
    std::size_t row_bytes_ = 0, layer_rows_ = 0;
    void release() noexcept;
    void copy(std::byte* base, std::size_t row_bytes, core::PageId from, core::PageId to, std::uint32_t rows);
};

// One step's batch tables (paged_reference.h's Tables) and the device buffer the kernels read. The host side is
// pageable, so upload() returns once the bytes are staged and the builder can be refilled at once; keep the device
// buffer unchanged until kernels reading the previous upload have run (one stream orders this). Graphs capture kernels
// that read the device buffer; upload before each replay.
class BatchTables {
public:
    BatchTables(int max_seqs, int max_pages, int max_rows);
    ~BatchTables();
    BatchTables(const BatchTables&) = delete;
    BatchTables& operator=(const BatchTables&) = delete;
    void clear() noexcept;
    void add(std::span<const core::PageId> table, std::uint32_t kv_len, int queries, int page_tokens = 0);
    void add(const core::StateCache&, core::SequenceId);
    void add_idle();
    void add_slots(std::span<const core::RowSlot> rows, int page_tokens);
    int seqs() const noexcept;
    int queries() const noexcept;
    int rows() const noexcept;
    int max_queries() const noexcept;
    const Tables& host() const noexcept;
    // The device slots array, for write().
    const std::int32_t* device_slots() const noexcept;
    // Copies the used part of each array (tables of the added sequences, lengths, offsets, slots) on `stream`.
    Batch upload(cudaStream_t stream);
private:
    Tables host_;
    std::int32_t* device_ = nullptr;
};

}  // namespace crit::paged_attention
