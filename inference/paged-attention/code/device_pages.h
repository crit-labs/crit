// Device KV pages behind Core's StateCache, and the per-step batch description the attention kernels read.
//
// The StateCache owns sequences, rounds, block tables, copy-on-write decisions, prefix reuse and retirement; DevicePages
// holds the bytes of every attention layer and performs the copies the cache requests, on the engine's stream.
// BatchTables packs block tables, lengths, query offsets and write slots for one step into one device buffer.
#pragma once
#include "paged_attention.h"

#include <crit/state.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace crit::paged_attention {

inline void check(cudaError_t error, const char* what) {
    if(error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}

// Pages of `layers` attention layers sharing one PageId space: layer l's keys at k + l * layer_rows rows. copy_rows
// enqueues one 2-D copy per array on `stream` (all layers at once), so it is ordered before later kernels on that
// stream; the cache calls it synchronously from begin/try_begin.
class DevicePages final : public crit::core::StateStorage {
    int layers_, pages_, page_tokens_, kv_heads_, head_dim_;
    Format format_;
    cudaStream_t stream_;
    std::byte* k_ = nullptr;
    std::byte* v_ = nullptr;
    float* k_scale_ = nullptr;
    float* v_scale_ = nullptr;
    std::size_t row_bytes_ = 0, layer_rows_ = 0;

    void release() noexcept {
        cudaFree(k_); cudaFree(v_); cudaFree(k_scale_); cudaFree(v_scale_);
        k_ = v_ = nullptr; k_scale_ = v_scale_ = nullptr;
    }
    void copy(std::byte* base, std::size_t row_bytes, crit::core::PageId from, crit::core::PageId to, std::uint32_t rows) {
        const std::size_t page_bytes = row_bytes * page_tokens_, pitch = row_bytes * layer_rows_;
        check(cudaMemcpy2DAsync(base + page_bytes * to, pitch, base + page_bytes * from, pitch, row_bytes * rows, layers_,
                                cudaMemcpyDeviceToDevice, stream_), "page copy");
    }
public:
    DevicePages(int layers, int pages, int page_tokens, int kv_heads, int head_dim, Format format, cudaStream_t stream)
        : layers_(layers), pages_(pages), page_tokens_(page_tokens), kv_heads_(kv_heads), head_dim_(head_dim), format_(format), stream_(stream) {
        if(layers <= 0 || pages <= 0 || page_tokens <= 0 || kv_heads <= 0 || (head_dim != 64 && head_dim != 128 && head_dim != 256))
            throw std::invalid_argument("Positive page geometry and head dimension 64, 128 or 256 required");
        layer_rows_ = static_cast<std::size_t>(pages) * page_tokens;
        row_bytes_ = static_cast<std::size_t>(kv_heads) * head_dim * (format == Format::bf16 ? 2 : 1);
        const std::size_t bytes = row_bytes_ * layer_rows_ * layers;
        try {
            check(cudaMalloc(&k_, bytes), "KV page allocation");
            check(cudaMalloc(&v_, bytes), "KV page allocation");
            if(format == Format::e4m3) {
                const std::size_t scales = static_cast<std::size_t>(kv_heads) * layer_rows_ * layers;
                check(cudaMalloc(&k_scale_, scales * sizeof(float)), "KV scale allocation");
                check(cudaMalloc(&v_scale_, scales * sizeof(float)), "KV scale allocation");
            }
        } catch(...) { release(); throw; }
    }
    ~DevicePages() override { release(); }
    DevicePages(const DevicePages&) = delete;
    DevicePages& operator=(const DevicePages&) = delete;

    Pages layer(int l) const {
        if(l < 0 || l >= layers_) throw std::out_of_range("KV layer outside the pages");
        const std::size_t rows = layer_rows_ * l, scales = rows * kv_heads_;
        return {k_ + row_bytes_ * rows, v_ + row_bytes_ * rows, k_scale_ ? k_scale_ + scales : nullptr,
                v_scale_ ? v_scale_ + scales : nullptr, format_, page_tokens_, kv_heads_, head_dim_};
    }
    int pages() const noexcept { return pages_; }
    int page_tokens() const noexcept { return page_tokens_; }
    std::size_t bytes() const noexcept {
        return row_bytes_ * layer_rows_ * layers_ * 2 + (k_scale_ ? sizeof(float) * kv_heads_ * layer_rows_ * layers_ * 2 : 0);
    }
    void copy_rows(crit::core::PageId from, crit::core::PageId to, std::uint32_t rows) override {
        if(from >= static_cast<std::uint32_t>(pages_) || to >= static_cast<std::uint32_t>(pages_) || rows > static_cast<std::uint32_t>(page_tokens_))
            throw std::out_of_range("KV page copy outside the pages");
        copy(k_, row_bytes_, from, to, rows);
        copy(v_, row_bytes_, from, to, rows);
        if(k_scale_) {
            copy(reinterpret_cast<std::byte*>(k_scale_), sizeof(float) * kv_heads_, from, to, rows);
            copy(reinterpret_cast<std::byte*>(v_scale_), sizeof(float) * kv_heads_, from, to, rows);
        }
    }
    void copy_slot(std::uint32_t, crit::core::SlotId, crit::core::SlotId) override {
        throw std::logic_error("DevicePages holds no recurrent state slots");
    }
};

// One step's batch: up to max_seqs sequences, block tables of up to max_pages pages and max_rows new rows. The host side
// is pageable, so upload() returns once the bytes are staged and the builder can be refilled at once; keep the device
// buffer unchanged until kernels reading the previous upload have run (one stream orders this). Graphs capture kernels
// that read the device buffer; upload before each replay.
class BatchTables {
    int max_seqs_, max_pages_, max_rows_;
    std::vector<std::int32_t> host_;
    std::int32_t* device_ = nullptr;
    int seqs_ = 0, queries_ = 0, rows_ = 0, max_queries_ = 0;
    std::int32_t* tables() { return host_.data(); }
    std::int32_t* lens() { return host_.data() + static_cast<std::size_t>(max_seqs_) * max_pages_; }
    std::int32_t* starts() { return lens() + max_seqs_; }
    std::int32_t* slots() { return starts() + max_seqs_ + 1; }
public:
    BatchTables(int max_seqs, int max_pages, int max_rows) : max_seqs_(max_seqs), max_pages_(max_pages), max_rows_(max_rows) {
        if(max_seqs <= 0 || max_pages <= 0 || max_rows < 0) throw std::invalid_argument("Positive batch bounds required");
        host_.assign(static_cast<std::size_t>(max_seqs) * max_pages + 2 * static_cast<std::size_t>(max_seqs) + 1 + max_rows, 0);
        check(cudaMalloc(&device_, host_.size() * sizeof(std::int32_t)), "batch table allocation");
        clear();
    }
    ~BatchTables() { cudaFree(device_); }
    BatchTables(const BatchTables&) = delete;
    BatchTables& operator=(const BatchTables&) = delete;

    void clear() { seqs_ = queries_ = rows_ = max_queries_ = 0; starts()[0] = 0; }
    // A sequence holding kv_len rows through `table`, whose newest `queries` rows are queried. With page_tokens > 0 the
    // table must also address every row (kv_len <= table.size() * page_tokens). Pass max_queries() or more as the
    // launch's max_columns (decode) or max_queries (prefill): the kernels do not attend a sequence beyond it (Batch).
    void add(std::span<const crit::core::PageId> table, std::uint32_t kv_len, int queries, int page_tokens = 0) {
        if(seqs_ == max_seqs_) throw std::length_error("Batch sequence bound reached");
        if(table.size() > static_cast<std::size_t>(max_pages_) || queries < 0 || static_cast<std::uint32_t>(queries) > kv_len ||
           kv_len > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) || page_tokens < 0 ||
           (page_tokens > 0 && kv_len > table.size() * static_cast<std::uint64_t>(page_tokens)))
            throw std::invalid_argument("Block table, length and queries must fit the batch");
        std::int32_t* row = tables() + static_cast<std::size_t>(seqs_) * max_pages_;
        for(std::size_t i = 0; i < table.size(); ++i) row[i] = static_cast<std::int32_t>(table[i]);
        lens()[seqs_] = static_cast<std::int32_t>(kv_len);
        queries_ += queries;
        starts()[++seqs_] = queries_;
        if(queries > max_queries_) max_queries_ = queries;
    }
    // A sequence's committed and staged rows, querying its staged rows (a round's new tokens).
    void add(const crit::core::StateCache& cache, crit::core::SequenceId sequence) {
        const auto staged = cache.staged(sequence);
        add(cache.block_table(sequence), cache.committed(sequence) + staged, static_cast<int>(staged),
            static_cast<int>(cache.config().page_tokens));
    }
    // Padding for a captured batch size: no queries, no keys.
    void add_idle() { add({}, 0, 0); }
    // Physical rows (page * page_tokens + offset) where new rows go, in query-row order; Round::kv of each sequence.
    void add_slots(std::span<const crit::core::RowSlot> rows, int page_tokens) {
        if(rows_ + rows.size() > static_cast<std::size_t>(max_rows_)) throw std::length_error("Batch row bound reached");
        for(const auto slot : rows) slots()[rows_++] = static_cast<std::int32_t>(slot.page * static_cast<std::uint32_t>(page_tokens) + slot.offset);
    }
    int seqs() const noexcept { return seqs_; }
    int queries() const noexcept { return queries_; }
    int rows() const noexcept { return rows_; }
    int max_queries() const noexcept { return max_queries_; }
    const std::int32_t* device_slots() const noexcept {
        return device_ + static_cast<std::size_t>(max_seqs_) * max_pages_ + 2 * static_cast<std::size_t>(max_seqs_) + 1;
    }
    // Copies the used part of each array (tables of the added sequences, lengths, offsets, slots).
    Batch upload(cudaStream_t stream) {
        const std::size_t lens_at = static_cast<std::size_t>(max_seqs_) * max_pages_, starts_at = lens_at + max_seqs_;
        const auto put = [&](std::size_t at, std::size_t count) {
            if(count) check(cudaMemcpyAsync(device_ + at, host_.data() + at, count * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream), "batch upload");
        };
        put(0, static_cast<std::size_t>(seqs_) * max_pages_);
        put(lens_at, seqs_);
        put(starts_at, static_cast<std::size_t>(seqs_) + 1);
        put(starts_at + max_seqs_ + 1, rows_);
        return {seqs_, device_, max_pages_, device_ + lens_at, device_ + starts_at};
    }
};

}  // namespace crit::paged_attention
