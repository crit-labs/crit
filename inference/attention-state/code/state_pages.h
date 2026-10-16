#pragma once
// Host FP32 KV pages addressed by Core's state cache page ids (crit/state.h). The
// StateCache owns sequences, rounds, block tables, copy-on-write decisions, prefix
// reuse and page retirement; this storage holds the rows, performs the copies the
// cache requests and gives attention.h a view of one sequence's pages. Inference keeps
// no second page ledger: forks, rollback, rewind and prefix reuse are StateCache
// calls, and a reader stays valid by the cache's epochs, not by reference counts.
#include <crit/state.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace crit::techniques {
class StatePages;

// Committed and staged rows of one sequence, read through a copy of its block
// table. A commit, discard, rewind or close releases pages at the cache's epoch();
// they stay readable until a retire() covering that epoch lets the cache hand
// them out again, so a reader is the engine's in-flight work for that epoch. A
// later round of the same sequence can also write an unshared tail page in place
// (rows past a rewind), so read before the sequence stages over the viewed rows.
class StatePagesView {
  friend class StatePages;
  const StatePages* storage_ = nullptr;
  std::vector<crit::core::PageId> table_;
  std::size_t rows_ = 0;
public:
  struct Block {
    std::size_t slot;
    std::uint64_t generation;      // always 0: reuse is ordered by the cache's epochs
    std::size_t rows, row_stride, key_width, value_width;
    std::span<const float> values; // Token-major [K | V], valid rows only.
  };
  std::size_t rows() const noexcept { return rows_; }
  std::size_t pages() const noexcept { return table_.size(); }
  // Plan metadata a copy of this view holds: its block table.
  std::size_t metadata_bytes() const noexcept { return table_.size() * sizeof(crit::core::PageId); }
  Block block(std::size_t i) const;
  std::span<const float> key(std::size_t row) const;
  std::span<const float> value(std::size_t row) const;
  // The first `rows` rows, holding only the pages they touch.
  StatePagesView prefix(std::size_t rows) const;
};

class StatePages final : public crit::core::StateStorage {
  std::size_t pages_, page_rows_, key_width_, value_width_, row_width_;
  std::vector<float> data_;
  static std::size_t checked(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::length_error("KV extent overflow");
    return a * b;
  }
  float* row(crit::core::PageId page, std::size_t offset) {
    return data_.data() + (static_cast<std::size_t>(page) * page_rows_ + offset) * row_width_;
  }
  void check_page(crit::core::PageId page) const {
    if (page >= pages_) throw std::out_of_range("KV page outside the storage");
  }
  // Rows in token-major keys/values, after checking geometry and finiteness.
  std::size_t checked_rows(std::span<const float> keys, std::span<const float> values) const {
    if (keys.size() % key_width_ || values.size() != checked(keys.size() / key_width_, value_width_))
      throw std::invalid_argument("KV row geometry differs");
    for (auto input : {keys, values})
      if (std::any_of(input.begin(), input.end(), [](float v) { return !std::isfinite(v); }))
        throw std::invalid_argument("Finite KV values required");
    return keys.size() / key_width_;
  }
  void store(std::span<const crit::core::RowSlot> slots, std::span<const float> keys, std::span<const float> values) noexcept {
    for (std::size_t i = 0; i < slots.size(); ++i) {
      float* out = row(slots[i].page, slots[i].offset);
      std::copy_n(keys.data() + i * key_width_, key_width_, out);
      std::copy_n(values.data() + i * value_width_, value_width_, out + key_width_);
    }
  }
public:
  // page_rows must equal the cache's StateCacheConfig::page_tokens and pages its page count.
  StatePages(std::uint32_t pages, std::uint32_t page_rows, std::size_t key_width, std::size_t value_width)
      : pages_(pages), page_rows_(page_rows), key_width_(key_width), value_width_(value_width),
        row_width_(key_width + value_width) {
    if (!pages || !page_rows || !key_width || !value_width || key_width > std::numeric_limits<std::size_t>::max() - value_width)
      throw std::invalid_argument("Positive bounded KV page geometry required");
    const auto values = checked(checked(pages_, page_rows_), row_width_);
    static_cast<void>(checked(values, sizeof(float)));
    data_.resize(values);
  }
  std::size_t page_rows() const noexcept { return page_rows_; }
  std::size_t key_width() const noexcept { return key_width_; }
  std::size_t value_width() const noexcept { return value_width_; }
  std::size_t reserved_bytes() const noexcept { return data_.size() * sizeof(float); }
  // Stores one staged row where Round::kv points.
  void write(crit::core::RowSlot slot, std::span<const float> key, std::span<const float> value) {
    if (key.size() != key_width_) throw std::invalid_argument("KV row geometry differs");
    write_rows(std::span(&slot, 1), key, value);
  }
  // Stores token-major rows where slots point (one slot per row): every row is
  // checked before any is written.
  void write_rows(std::span<const crit::core::RowSlot> slots, std::span<const float> keys, std::span<const float> values) {
    if (checked_rows(keys, values) != slots.size()) throw std::invalid_argument("KV rows and slots differ");
    for (auto slot : slots) {
      check_page(slot.page);
      if (slot.offset >= page_rows_) throw std::out_of_range("KV row outside its page");
    }
    store(slots, keys, values);
  }
  // Begins a round of keys.size() / key_width rows on `sequence` and writes them
  // where it points. Rows are checked first, so a rejected row, like a rejected
  // begin, leaves the cache unchanged. The caller commits or discards the round.
  crit::core::Round stage(crit::core::StateCache& cache, crit::core::SequenceId sequence,
                          std::span<const float> keys, std::span<const float> values) {
    const auto count = checked_rows(keys, values);
    if (!count || count > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("Positive bounded KV row count required");
    auto round = cache.begin(sequence, static_cast<std::uint32_t>(count));
    store(round.kv, keys, values);
    return round;
  }
  void copy_rows(crit::core::PageId from, crit::core::PageId to, std::uint32_t rows) override {
    check_page(from);
    check_page(to);
    if (rows > page_rows_) throw std::out_of_range("KV copy exceeds a page");
    std::copy_n(row(from, 0), static_cast<std::size_t>(rows) * row_width_, row(to, 0));
  }
  void copy_slot(std::uint32_t, crit::core::SlotId, crit::core::SlotId) override {
    throw std::logic_error("StatePages holds no recurrent state slots");
  }
  // rows: the sequence's committed rows, plus its staged rows when a round should read them.
  StatePagesView view(std::span<const crit::core::PageId> block_table, std::size_t rows) const {
    if (rows > checked(block_table.size(), page_rows_)) throw std::out_of_range("KV rows exceed the block table");
    for (auto page : block_table) check_page(page);
    StatePagesView result;
    result.storage_ = this;
    result.rows_ = rows;
    result.table_.assign(block_table.begin(), block_table.begin() +
        static_cast<std::ptrdiff_t>(rows / page_rows_ + (rows % page_rows_ != 0)));
    return result;
  }
  // One sequence's committed rows, plus its staged rows when include_staged.
  StatePagesView view(const crit::core::StateCache& cache, crit::core::SequenceId sequence, bool include_staged = false) const {
    return view(cache.block_table(sequence),
                std::size_t{cache.committed(sequence)} + (include_staged ? cache.staged(sequence) : 0));
  }
  friend class StatePagesView;
};

inline StatePagesView::Block StatePagesView::block(std::size_t i) const {
  if (i >= table_.size()) throw std::out_of_range("KV page outside view");
  const auto& s = *storage_;
  const auto rows = std::min(s.page_rows_, rows_ - i * s.page_rows_);
  const auto* values = s.data_.data() + static_cast<std::size_t>(table_[i]) * s.page_rows_ * s.row_width_;
  return {table_[i], 0, rows, s.row_width_, s.key_width_, s.value_width_, {values, rows * s.row_width_}};
}
inline std::span<const float> StatePagesView::key(std::size_t row) const {
  if (row >= rows_) throw std::out_of_range("KV row outside view");
  const auto& s = *storage_;
  return block(row / s.page_rows_).values.subspan((row % s.page_rows_) * s.row_width_, s.key_width_);
}
inline std::span<const float> StatePagesView::value(std::size_t row) const {
  if (row >= rows_) throw std::out_of_range("KV row outside view");
  const auto& s = *storage_;
  return block(row / s.page_rows_).values.subspan((row % s.page_rows_) * s.row_width_ + s.key_width_, s.value_width_);
}
inline StatePagesView StatePagesView::prefix(std::size_t rows) const {
  if (rows > rows_) throw std::out_of_range("KV prefix exceeds view");
  StatePagesView result;
  result.storage_ = storage_;
  result.rows_ = rows;
  if (rows) {
    const auto page_rows = storage_->page_rows_;
    result.table_.assign(table_.begin(), table_.begin() + static_cast<std::ptrdiff_t>(rows / page_rows + (rows % page_rows != 0)));
  }
  return result;
}
} // namespace crit::techniques
