#include "paged_reference.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace crit::paged_attention {
namespace {
bool valid_head_dim(int head_dim) noexcept { return head_dim == 64 || head_dim == 128 || head_dim == 256; }
}

std::uint16_t to_bf16(float x) noexcept {
    const auto u = std::bit_cast<std::uint32_t>(x);
    if((u & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((u >> 16) | 0x40u);
    return std::uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(std::uint16_t b) noexcept { return std::bit_cast<float>(std::uint32_t(b) << 16); }

std::uint8_t to_e4m3(float x) noexcept {
    if(std::isnan(x)) return 0x7f;
    const std::uint8_t sign = std::signbit(x) ? 0x80 : 0x00;
    const double a = std::fabs(double(x));
    if(a >= 448.0) return std::uint8_t(sign | 0x7e);
    // Below the smallest normal (2^-6) the step is 2^-9; a count of 8 is the smallest normal's code.
    if(a < 0x1p-6) return std::uint8_t(sign | unsigned(std::nearbyint(a * 512.0)));
    int e = 0;
    const double m = std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1): exponent e - 1, mantissa 2m - 1
    unsigned field = unsigned(e - 1 + 7), q = unsigned(std::nearbyint((2.0 * m - 1.0) * 8.0));
    if(q == 8) { q = 0; ++field; }
    return std::uint8_t(sign | std::min(0x7eu, (field << 3) | q));
}
float from_e4m3(std::uint8_t b) noexcept {
    if((b & 0x7f) == 0x7f) return std::numeric_limits<float>::quiet_NaN();
    const int exponent = (b >> 3) & 15, mantissa = b & 7;
    const float magnitude = exponent ? std::ldexp(1.f + float(mantissa) / 8.f, exponent - 7) : std::ldexp(float(mantissa) / 8.f, -6);
    return b & 0x80 ? -magnitude : magnitude;
}
float e4m3_scale(std::span<const float> head) noexcept {
    float largest = 0.f;
    for(const float x:head) largest = std::max(largest, std::fabs(x));
    return largest > 0.f ? largest / 448.f : 1.f;
}

Tables::Tables(int max_seqs, int max_pages, int max_rows) : max_seqs_(max_seqs), max_pages_(max_pages), max_rows_(max_rows) {
    if(max_seqs <= 0 || max_pages <= 0 || max_rows < 0) throw std::invalid_argument("Positive batch bounds required");
    words_.assign(std::size_t(max_seqs) * max_pages + 2 * std::size_t(max_seqs) + 1 + std::size_t(max_rows), 0);
}
void Tables::clear() noexcept {
    seqs_ = queries_ = rows_ = max_queries_ = 0;
    words_[starts_at()] = 0;
}
void Tables::add(std::span<const core::PageId> table, std::uint32_t kv_len, int queries, int page_tokens) {
    if(seqs_ == max_seqs_) throw std::length_error("Batch sequence bound reached");
    if(table.size() > std::size_t(max_pages_) || queries < 0 || std::uint32_t(queries) > kv_len ||
       kv_len > std::uint32_t(std::numeric_limits<std::int32_t>::max()) || page_tokens < 0 ||
       (page_tokens > 0 && kv_len > table.size() * std::uint64_t(page_tokens)))
        throw std::invalid_argument("Block table, length and queries must fit the batch");
    auto* row = words_.data() + std::size_t(seqs_) * max_pages_;
    for(std::size_t i = 0; i < table.size(); ++i) row[i] = std::int32_t(table[i]);
    words_[lens_at() + std::size_t(seqs_)] = std::int32_t(kv_len);
    queries_ += queries;
    words_[starts_at() + std::size_t(++seqs_)] = queries_;
    max_queries_ = std::max(max_queries_, queries);
}
void Tables::add(const core::StateCache& cache, core::SequenceId sequence) {
    const auto staged = cache.staged(sequence);
    add(cache.block_table(sequence), cache.committed(sequence) + staged, int(staged), int(cache.config().page_tokens));
}
void Tables::add_idle() { add({}, 0, 0); }
void Tables::add_slots(std::span<const core::RowSlot> rows, int page_tokens) {
    if(rows_ + rows.size() > std::size_t(max_rows_)) throw std::length_error("Batch row bound reached");
    for(const auto slot:rows) words_[slots_at() + std::size_t(rows_++)] = std::int32_t(slot.page * std::uint32_t(page_tokens) + slot.offset);
}
int Tables::seqs() const noexcept { return seqs_; }
int Tables::queries() const noexcept { return queries_; }
int Tables::rows() const noexcept { return rows_; }
int Tables::max_queries() const noexcept { return max_queries_; }
int Tables::max_seqs() const noexcept { return max_seqs_; }
int Tables::max_pages() const noexcept { return max_pages_; }
int Tables::max_rows() const noexcept { return max_rows_; }
std::span<const std::int32_t> Tables::words() const noexcept { return words_; }
std::size_t Tables::lens_at() const noexcept { return std::size_t(max_seqs_) * max_pages_; }
std::size_t Tables::starts_at() const noexcept { return lens_at() + std::size_t(max_seqs_); }
std::size_t Tables::slots_at() const noexcept { return starts_at() + std::size_t(max_seqs_) + 1; }

HostPages::HostPages(int layers, int pages, int page_tokens, int kv_heads, int head_dim, Format format)
    : layers_(layers), pages_(pages), page_tokens_(page_tokens), kv_heads_(kv_heads), head_dim_(head_dim), format_(format) {
    if(layers <= 0 || pages <= 0 || page_tokens <= 0 || kv_heads <= 0 || !valid_head_dim(head_dim) ||
       (format != Format::bf16 && format != Format::e4m3))
        throw std::invalid_argument("Positive page geometry, head dimension 64, 128 or 256 and a known format required");
    layer_rows_ = std::size_t(pages) * page_tokens;
    element_bytes_ = format == Format::bf16 ? 2 : 1;
    const auto elements = layer_rows_ * kv_heads * head_dim * layers;
    k_.assign(elements * element_bytes_, std::byte{0});
    v_.assign(elements * element_bytes_, std::byte{0});
    if(format == Format::e4m3) {
        k_scale_.assign(layer_rows_ * kv_heads * layers, 1.f);
        v_scale_.assign(k_scale_.size(), 1.f);
    }
}
void HostPages::write(int layer, std::span<const std::uint16_t> k, std::span<const std::uint16_t> v, int stride, int rows,
                      std::span<const std::int32_t> slots) {
    const int width = kv_heads_ * head_dim_;
    if(layer < 0 || layer >= layers_) throw std::out_of_range("KV layer outside the pages");
    if(rows < 0 || stride < width || slots.size() < std::size_t(rows) ||
       (rows && (k.size() < std::size_t(rows - 1) * stride + width || v.size() < std::size_t(rows - 1) * stride + width)))
        throw std::invalid_argument("Key and value rows must cover the written rows");
    std::vector<float> head(static_cast<std::size_t>(head_dim_), 0.f);
    const auto store = [&](std::vector<std::byte>& bytes, std::vector<float>& scales, std::span<const std::uint16_t> src,
                           std::size_t t, std::size_t row) {
        for(int g = 0; g < kv_heads_; ++g) {
            const std::size_t at = ((std::size_t(layer) * layer_rows_ + row) * kv_heads_ + g) * head_dim_;
            const auto* in = src.data() + t * std::size_t(stride) + std::size_t(g) * head_dim_;
            if(format_ == Format::bf16) {
                std::memcpy(bytes.data() + at * 2, in, std::size_t(head_dim_) * 2);
                continue;
            }
            for(int d = 0; d < head_dim_; ++d) head[std::size_t(d)] = from_bf16(in[d]);
            const float scale = e4m3_scale(head);
            for(int d = 0; d < head_dim_; ++d) bytes[at + std::size_t(d)] = std::byte{to_e4m3(head[std::size_t(d)] / scale)};
            scales[(std::size_t(layer) * layer_rows_ + row) * kv_heads_ + g] = scale;
        }
    };
    for(int t = 0; t < rows; ++t) {
        const auto slot = slots[std::size_t(t)];
        if(slot < 0) continue;
        if(std::size_t(slot) >= layer_rows_) throw std::out_of_range("KV write slot outside the pages");
        store(k_, k_scale_, k, std::size_t(t), std::size_t(slot));
        store(v_, v_scale_, v, std::size_t(t), std::size_t(slot));
    }
}
namespace {
float element(const std::vector<std::byte>& bytes, const std::vector<float>& scales, Format format, std::size_t index,
              std::size_t scale_index) {
    if(format == Format::bf16) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes.data() + index * 2, 2);
        return from_bf16(bits);
    }
    return from_bf16(to_bf16(from_e4m3(std::to_integer<std::uint8_t>(bytes[index])) * scales[scale_index]));
}
}
float HostPages::key(int layer, std::size_t row, int g, int d) const {
    const auto head = (std::size_t(layer) * layer_rows_ + row) * kv_heads_ + g;
    return element(k_, k_scale_, format_, head * head_dim_ + std::size_t(d), head);
}
float HostPages::value(int layer, std::size_t row, int g, int d) const {
    const auto head = (std::size_t(layer) * layer_rows_ + row) * kv_heads_ + g;
    return element(v_, v_scale_, format_, head * head_dim_ + std::size_t(d), head);
}
std::span<const std::byte> HostPages::keys(int layer) const {
    if(layer < 0 || layer >= layers_) throw std::out_of_range("KV layer outside the pages");
    const auto bytes = layer_rows_ * kv_heads_ * head_dim_ * element_bytes_;
    return std::span<const std::byte>(k_).subspan(std::size_t(layer) * bytes, bytes);
}
std::span<const std::byte> HostPages::values(int layer) const {
    if(layer < 0 || layer >= layers_) throw std::out_of_range("KV layer outside the pages");
    const auto bytes = layer_rows_ * kv_heads_ * head_dim_ * element_bytes_;
    return std::span<const std::byte>(v_).subspan(std::size_t(layer) * bytes, bytes);
}
std::span<const float> HostPages::key_scales(int layer) const {
    if(layer < 0 || layer >= layers_) throw std::out_of_range("KV layer outside the pages");
    if(k_scale_.empty()) return {};
    return std::span<const float>(k_scale_).subspan(std::size_t(layer) * layer_rows_ * kv_heads_, layer_rows_ * kv_heads_);
}
std::span<const float> HostPages::value_scales(int layer) const {
    if(layer < 0 || layer >= layers_) throw std::out_of_range("KV layer outside the pages");
    if(v_scale_.empty()) return {};
    return std::span<const float>(v_scale_).subspan(std::size_t(layer) * layer_rows_ * kv_heads_, layer_rows_ * kv_heads_);
}
int HostPages::layers() const noexcept { return layers_; }
int HostPages::pages() const noexcept { return pages_; }
int HostPages::page_tokens() const noexcept { return page_tokens_; }
int HostPages::kv_heads() const noexcept { return kv_heads_; }
int HostPages::head_dim() const noexcept { return head_dim_; }
Format HostPages::format() const noexcept { return format_; }
void HostPages::copy_rows(core::PageId from, core::PageId to, std::uint32_t rows) {
    if(from >= std::uint32_t(pages_) || to >= std::uint32_t(pages_) || rows > std::uint32_t(page_tokens_))
        throw std::out_of_range("KV page copy outside the pages");
    const auto row_elements = std::size_t(kv_heads_) * head_dim_;
    for(int l = 0; l < layers_; ++l) {
        const auto base = std::size_t(l) * layer_rows_;
        const auto src = base + std::size_t(from) * page_tokens_, dst = base + std::size_t(to) * page_tokens_;
        for(auto* bytes:{&k_, &v_})
            std::memmove(bytes->data() + dst * row_elements * element_bytes_, bytes->data() + src * row_elements * element_bytes_,
                         rows * row_elements * element_bytes_);
        if(format_ == Format::e4m3)
            for(auto* scales:{&k_scale_, &v_scale_})
                std::copy_n(scales->begin() + std::ptrdiff_t(src * kv_heads_), rows * std::size_t(kv_heads_),
                            scales->begin() + std::ptrdiff_t(dst * kv_heads_));
    }
}
void HostPages::copy_slot(std::uint32_t, core::SlotId, core::SlotId) {
    throw std::logic_error("HostPages holds no recurrent state slots");
}

std::vector<float> attend(const HostPages& pages, const Tables& tables, const Problem& p) {
    const int hd = pages.head_dim(), kv_heads = pages.kv_heads(), queries = tables.queries();
    if(p.layer < 0 || p.layer >= pages.layers() || p.heads <= 0 || p.heads % kv_heads || p.q_stride < std::size_t(p.heads) * hd ||
       p.window < 0 || !(p.softcap >= 0.f) || (!p.sinks.empty() && p.sinks.size() != std::size_t(p.heads)) ||
       (!p.tree.empty() && p.tree.size() != std::size_t(queries)) || (!p.tree.empty() && p.window > 0 && p.window <= 64) ||
       (queries && p.q.size() < std::size_t(queries - 1) * p.q_stride + std::size_t(p.heads) * hd))
        throw std::invalid_argument("Invalid paged attention problem");
    const int group = p.heads / kv_heads, page_tokens = pages.page_tokens();
    const auto words = tables.words();
    const std::size_t width = std::size_t(p.heads) * hd;
    std::vector<float> out(std::size_t(queries) * width, 0.f);
    std::vector<double> scores;
    std::vector<int> keys;
    for(int s = 0; s < tables.seqs(); ++s) {
        const auto* table = words.data() + std::size_t(s) * tables.max_pages();
        const int kv_len = words[tables.lens_at() + std::size_t(s)];
        const int q0 = words[tables.starts_at() + std::size_t(s)], n = words[tables.starts_at() + std::size_t(s) + 1] - q0;
        if(!n) continue;
        if(std::int64_t(kv_len) > std::int64_t(tables.max_pages()) * page_tokens)
            throw std::invalid_argument("A sequence's keys reach past its block table");
        for(int page = 0; page * page_tokens < kv_len; ++page)
            if(table[page] < 0 || table[page] >= pages.pages()) throw std::invalid_argument("A block table names a page outside the pages");
        const int base = kv_len - n;
        for(int i = 0; i < n; ++i) {
            const int pos = base + i, row = q0 + i;
            const std::uint64_t mask = p.tree.empty() ? ~0ull : p.tree[std::size_t(row)];
            const int depth = p.tree.empty() ? i : std::popcount(mask) - 1;
            const int lo = p.window > 0 ? std::max(0, base + depth - p.window + 1) : 0;
            keys.clear();
            for(int key = lo; key <= pos; ++key)
                if(key < base || key - base >= 64 || ((mask >> (key - base)) & 1ull)) keys.push_back(key);
            for(int h = 0; h < p.heads; ++h) {
                const int g = h / group;
                const float* q = p.q.data() + std::size_t(row) * p.q_stride + std::size_t(h) * hd;
                const double sink = p.sinks.empty() ? -std::numeric_limits<double>::infinity() : double(p.sinks[std::size_t(h)]);
                double largest = sink;
                scores.clear();
                for(const int key:keys) {
                    const auto physical = std::size_t(table[key / page_tokens]) * page_tokens + std::size_t(key % page_tokens);
                    double dot = 0;
                    for(int d = 0; d < hd; ++d) dot += double(q[d]) * pages.key(p.layer, physical, g, d);
                    double score = dot * p.scale;
                    if(p.softcap > 0.f) score = p.softcap * std::tanh(score / p.softcap);
                    scores.push_back(score);
                    largest = std::max(largest, score);
                }
                if(keys.empty()) continue;  // no key: zero, with or without a sink
                double total = p.sinks.empty() ? 0.0 : std::exp(sink - largest);
                for(auto& score:scores) total += score = std::exp(score - largest);
                float* o = out.data() + std::size_t(row) * width + std::size_t(h) * hd;
                for(int d = 0; d < hd; ++d) {
                    double sum = 0;
                    for(std::size_t j = 0; j < keys.size(); ++j) {
                        const auto key = keys[j];
                        const auto physical = std::size_t(table[key / page_tokens]) * page_tokens + std::size_t(key % page_tokens);
                        sum += scores[j] * pages.value(p.layer, physical, g, d);
                    }
                    o[d] = float(sum / total);
                }
            }
        }
    }
    return out;
}
}
