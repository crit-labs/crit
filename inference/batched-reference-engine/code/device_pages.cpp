// Device KV pages behind Core's StateCache, and the per-step device batch tables (paged_attention.h).
//
// The StateCache owns sequences, rounds, block tables, copy-on-write decisions, prefix reuse and retirement; DevicePages
// holds the bytes of every attention layer and performs the copies the cache requests, on the engine's stream.
// BatchTables uploads one step's packed Tables into one device buffer.
#include "paged_attention.h"

#include <stdexcept>

namespace crit::paged_attention {

DevicePages::DevicePages(int layers, int pages, int page_tokens, int kv_heads, int head_dim, Format format, cudaStream_t stream)
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
DevicePages::~DevicePages() { release(); }
void DevicePages::release() noexcept {
    cudaFree(k_); cudaFree(v_); cudaFree(k_scale_); cudaFree(v_scale_);
    k_ = v_ = nullptr; k_scale_ = v_scale_ = nullptr;
}
void DevicePages::copy(std::byte* base, std::size_t row_bytes, core::PageId from, core::PageId to, std::uint32_t rows) {
    const std::size_t page_bytes = row_bytes * page_tokens_, pitch = row_bytes * layer_rows_;
    check(cudaMemcpy2DAsync(base + page_bytes * to, pitch, base + page_bytes * from, pitch, row_bytes * rows, layers_,
                            cudaMemcpyDeviceToDevice, stream_), "page copy");
}
Pages DevicePages::layer(int l) const {
    if(l < 0 || l >= layers_) throw std::out_of_range("KV layer outside the pages");
    const std::size_t rows = layer_rows_ * l, scales = rows * kv_heads_;
    return {k_ + row_bytes_ * rows, v_ + row_bytes_ * rows, k_scale_ ? k_scale_ + scales : nullptr,
            v_scale_ ? v_scale_ + scales : nullptr, format_, page_tokens_, kv_heads_, head_dim_};
}
int DevicePages::pages() const noexcept { return pages_; }
int DevicePages::page_tokens() const noexcept { return page_tokens_; }
std::size_t DevicePages::bytes() const noexcept {
    return row_bytes_ * layer_rows_ * layers_ * 2 + (k_scale_ ? sizeof(float) * kv_heads_ * layer_rows_ * layers_ * 2 : 0);
}
void DevicePages::copy_rows(core::PageId from, core::PageId to, std::uint32_t rows) {
    if(from >= static_cast<std::uint32_t>(pages_) || to >= static_cast<std::uint32_t>(pages_) || rows > static_cast<std::uint32_t>(page_tokens_))
        throw std::out_of_range("KV page copy outside the pages");
    copy(k_, row_bytes_, from, to, rows);
    copy(v_, row_bytes_, from, to, rows);
    if(k_scale_) {
        copy(reinterpret_cast<std::byte*>(k_scale_), sizeof(float) * kv_heads_, from, to, rows);
        copy(reinterpret_cast<std::byte*>(v_scale_), sizeof(float) * kv_heads_, from, to, rows);
    }
}
void DevicePages::copy_slot(std::uint32_t, core::SlotId, core::SlotId) {
    throw std::logic_error("DevicePages holds no recurrent state slots");
}

BatchTables::BatchTables(int max_seqs, int max_pages, int max_rows) : host_(max_seqs, max_pages, max_rows) {
    check(cudaMalloc(&device_, host_.words().size() * sizeof(std::int32_t)), "batch table allocation");
}
BatchTables::~BatchTables() { cudaFree(device_); }
void BatchTables::clear() noexcept { host_.clear(); }
void BatchTables::add(std::span<const core::PageId> table, std::uint32_t kv_len, int queries, int page_tokens) {
    host_.add(table, kv_len, queries, page_tokens);
}
void BatchTables::add(const core::StateCache& cache, core::SequenceId sequence) { host_.add(cache, sequence); }
void BatchTables::add_idle() { host_.add_idle(); }
void BatchTables::add_slots(std::span<const core::RowSlot> rows, int page_tokens) { host_.add_slots(rows, page_tokens); }
int BatchTables::seqs() const noexcept { return host_.seqs(); }
int BatchTables::queries() const noexcept { return host_.queries(); }
int BatchTables::rows() const noexcept { return host_.rows(); }
int BatchTables::max_queries() const noexcept { return host_.max_queries(); }
const Tables& BatchTables::host() const noexcept { return host_; }
const std::int32_t* BatchTables::device_slots() const noexcept { return device_ + host_.slots_at(); }
Batch BatchTables::upload(cudaStream_t stream) {
    const auto words = host_.words();
    const auto put = [&](std::size_t at, std::size_t count) {
        if(count) check(cudaMemcpyAsync(device_ + at, words.data() + at, count * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream), "batch upload");
    };
    const auto seqs = static_cast<std::size_t>(host_.seqs());
    put(0, seqs * host_.max_pages());
    put(host_.lens_at(), seqs);
    put(host_.starts_at(), seqs + 1);
    put(host_.slots_at(), static_cast<std::size_t>(host_.rows()));
    return {host_.seqs(), device_, host_.max_pages(), device_ + host_.lens_at(), device_ + host_.starts_at()};
}

}  // namespace crit::paged_attention
