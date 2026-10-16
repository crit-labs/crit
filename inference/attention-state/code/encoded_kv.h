#pragma once
// Host G32 INT8 KV pages (FP32 scale per 32 values) addressed by Core's state
// cache page ids, the encoded counterpart of state_pages.h. The StateCache owns
// sequences, rounds, copy-on-write, rewind, forks and retirement; this storage
// encodes rows, performs the copies the cache requests and decodes for attention.
#include "grouped_int.h"
#include "state_pages.h"
#include <crit/state.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace crit::techniques {
class EncodedPages;

// Same validity rules as StatePagesView. Each attention plan owns its copy and a
// bounded one-page FP32 decode buffer.
class EncodedPagesView {
  friend class EncodedPages;
  const EncodedPages* storage_ = nullptr;
  std::vector<crit::core::PageId> table_;
  std::size_t rows_ = 0;
  mutable std::vector<float> scratch_;
  quant::Rows source(std::size_t index, std::size_t rows) const;
public:
  using Block = StatePagesView::Block;
  std::size_t rows() const noexcept { return rows_; }
  std::size_t pages() const noexcept { return table_.size(); }
  std::size_t metadata_bytes() const noexcept { return table_.size() * sizeof(crit::core::PageId); }
  std::size_t scratch_bytes() const;
  // Block geometry without decoding; values is empty.
  Block block_metadata(std::size_t index) const;
  void gather_head(std::size_t page, std::size_t first, std::size_t rows, std::size_t head, std::size_t kw,
                   std::size_t vw, std::span<float> keys, std::span<float> values) const;
  // Decoded span belongs to this view and is valid until its next block() call.
  Block block(std::size_t index) const;
};

class EncodedPages final : public crit::core::StateStorage {
  friend class EncodedPagesView;
  std::size_t pages_, page_rows_, key_width_, value_width_, row_width_, page_values_;
  std::vector<std::uint8_t> codes_;
  std::vector<float> scales_;
  std::size_t at(crit::core::PageId page, std::size_t offset) const {
    return static_cast<std::size_t>(page) * page_values_ + offset * row_width_;
  }
  void check_page(crit::core::PageId page) const {
    if (page >= pages_) throw std::out_of_range("Encoded KV page outside the storage");
  }
  // Validates and encodes every row before any page changes.
  quant::PackedRows encode(std::span<const float> keys, std::span<const float> values) const {
    if (keys.size() % key_width_) throw std::invalid_argument("Encoded KV key extent differs");
    const auto n = keys.size() / key_width_;
    if (!n || values.size() != quant::extent(n, value_width_)) throw std::invalid_argument("Encoded KV row extent differs");
    std::vector<float> joined(quant::extent(n, row_width_));
    for (std::size_t i = 0; i < n; ++i) {
      std::copy_n(keys.data() + i * key_width_, key_width_, joined.data() + i * row_width_);
      std::copy_n(values.data() + i * value_width_, value_width_, joined.data() + i * row_width_ + key_width_);
    }
    return quant::quantize(joined, n, row_width_, 8);
  }
  void store(std::span<const crit::core::RowSlot> slots, const quant::PackedRows& packed) noexcept {
    for (std::size_t i = 0; i < slots.size(); ++i) {
      const auto to = at(slots[i].page, slots[i].offset);
      std::copy_n(packed.codes.data() + i * row_width_, row_width_, codes_.data() + to);
      std::copy_n(packed.scales.data() + i * row_width_ / 32, row_width_ / 32, scales_.data() + to / 32);
    }
  }
public:
  // page_rows must equal the cache's StateCacheConfig::page_tokens and pages its page count.
  EncodedPages(std::uint32_t pages, std::uint32_t page_rows, std::size_t key_width, std::size_t value_width)
      : pages_(pages), page_rows_(page_rows), key_width_(key_width), value_width_(value_width),
        row_width_(key_width + value_width), page_values_(0) {
    if (!pages || !page_rows || !key_width || !value_width || key_width % 32 || value_width % 32 ||
        key_width > std::numeric_limits<std::size_t>::max() - value_width)
      throw std::invalid_argument("Positive complete G32 KV rows required");
    page_values_ = quant::extent(page_rows_, row_width_);
    codes_.resize(quant::extent(pages_, page_values_));
    scales_.resize(codes_.size() / 32);
  }
  std::size_t page_rows() const noexcept { return page_rows_; }
  std::size_t reserved_bytes() const noexcept { return codes_.size() + scales_.size() * sizeof(float); }
  // Encodes token-major rows where slots point (one slot per row); all or none.
  void write_rows(std::span<const crit::core::RowSlot> slots, std::span<const float> keys, std::span<const float> values) {
    const auto packed = encode(keys, values);
    if (packed.rows != slots.size()) throw std::invalid_argument("Encoded KV rows and slots differ");
    for (auto slot : slots) {
      check_page(slot.page);
      if (slot.offset >= page_rows_) throw std::out_of_range("Encoded KV row outside its page");
    }
    store(slots, packed);
  }
  // Encodes the rows, then begins a round of that many rows on `sequence` and
  // stores them where it points; a rejected row or begin changes nothing.
  crit::core::Round stage(crit::core::StateCache& cache, crit::core::SequenceId sequence,
                          std::span<const float> keys, std::span<const float> values) {
    const auto packed = encode(keys, values);
    if (packed.rows > std::numeric_limits<std::uint32_t>::max()) throw std::length_error("Encoded KV round too large");
    auto round = cache.begin(sequence, static_cast<std::uint32_t>(packed.rows));
    store(round.kv, packed);
    return round;
  }
  void copy_rows(crit::core::PageId from, crit::core::PageId to, std::uint32_t rows) override {
    check_page(from);
    check_page(to);
    if (rows > page_rows_) throw std::out_of_range("Encoded KV copy exceeds a page");
    const auto n = static_cast<std::size_t>(rows) * row_width_;
    std::copy_n(codes_.data() + at(from, 0), n, codes_.data() + at(to, 0));
    std::copy_n(scales_.data() + at(from, 0) / 32, n / 32, scales_.data() + at(to, 0) / 32);
  }
  void copy_slot(std::uint32_t, crit::core::SlotId, crit::core::SlotId) override {
    throw std::logic_error("EncodedPages holds no recurrent state slots");
  }
  // rows: the sequence's committed rows, plus its staged rows when a round should read them.
  EncodedPagesView view(std::span<const crit::core::PageId> block_table, std::size_t rows) const {
    if (rows > quant::extent(block_table.size(), page_rows_)) throw std::out_of_range("Encoded KV rows exceed the block table");
    for (auto page : block_table) check_page(page);
    EncodedPagesView result;
    result.storage_ = this;
    result.rows_ = rows;
    result.table_.assign(block_table.begin(), block_table.begin() +
        static_cast<std::ptrdiff_t>(rows / page_rows_ + (rows % page_rows_ != 0)));
    return result;
  }
  EncodedPagesView view(const crit::core::StateCache& cache, crit::core::SequenceId sequence, bool include_staged = false) const {
    return view(cache.block_table(sequence),
                std::size_t{cache.committed(sequence)} + (include_staged ? cache.staged(sequence) : 0));
  }
};

inline std::size_t EncodedPagesView::scratch_bytes() const {
  return storage_ ? quant::extent(storage_->page_values_, sizeof(float)) : 0;
}
inline EncodedPagesView::Block EncodedPagesView::block_metadata(std::size_t index) const {
  if (index >= table_.size()) throw std::out_of_range("Encoded KV block outside view");
  const auto& s = *storage_;
  return {table_[index], 0, std::min(s.page_rows_, rows_ - index * s.page_rows_), s.row_width_, s.key_width_, s.value_width_, {}};
}
inline quant::Rows EncodedPagesView::source(std::size_t index, std::size_t rows) const {
  const auto& s = *storage_;
  const auto at = s.at(table_[index], 0);
  return quant::Rows(std::span(s.codes_).subspan(at, rows * s.row_width_),
                     std::span(s.scales_).subspan(at / 32, rows * s.row_width_ / 32), rows, s.row_width_, 8);
}
inline void EncodedPagesView::gather_head(std::size_t page, std::size_t first, std::size_t rows, std::size_t head,
                                          std::size_t kw, std::size_t vw, std::span<float> keys, std::span<float> values) const {
  const auto block = block_metadata(page);
  const auto& s = *storage_;
  const auto ks = quant::extent(head, kw), vs = quant::extent(head, vw);
  if (!kw || !vw || first > block.rows || rows > block.rows - first || ks > s.key_width_ || kw > s.key_width_ - ks ||
      vs > s.value_width_ || vw > s.value_width_ - vs || keys.size() != quant::extent(rows, kw) || values.size() != quant::extent(rows, vw))
    throw std::out_of_range("Encoded KV head gather geometry");
  const auto encoded = source(page, block.rows);
  for (std::size_t i = 0; i < rows; ++i) {
    encoded.row_range(first + i, ks, keys.subspan(i * kw, kw));
    encoded.row_range(first + i, s.key_width_ + vs, values.subspan(i * vw, vw));
  }
}
inline EncodedPagesView::Block EncodedPagesView::block(std::size_t index) const {
  auto result = block_metadata(index);
  const auto encoded = source(index, result.rows);
  scratch_.resize(result.rows * result.row_stride);
  for (std::size_t i = 0; i < result.rows; ++i) encoded.row(i, std::span(scratch_).subspan(i * result.row_stride, result.row_stride));
  result.values = scratch_;
  return result;
}
} // namespace crit::techniques
