#pragma once
// A bounded lower tier for the prefix pages Core's StateCache evicts (crit/state.h). StateStorage::evicted names an
// evicted page, its chain hash and tokens, the slots holding its components' state and its linked window pages;
// admit() gives the host slot to copy them into, before the cache hands the ids out again. continuation() finds the
// kept pages that continue a prompt past what StateCache::attach reused, verified by chain hash and exact tokens, so
// the engine copies them back in one restore round instead of recomputing them. This ledger moves no bytes and
// allocates nothing after construction: the engine owns the host slots (pinned memory for a device engine) and the
// copies, enqueued on its ordered queue so a spill reads the page before any later write.
#include <crit/state.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace crit::techniques {

class HostPageTier {
public:
  static constexpr std::uint32_t no_slot = std::numeric_limits<std::uint32_t>::max();

  // `slots` host page slots for pages of the cache `config` describes (its page_tokens, identity, hash, components
  // and window groups). A tier for a cache without the prefix cache or with no slots keeps nothing.
  HostPageTier(std::uint32_t slots, const crit::core::StateCacheConfig& config)
      : slots_(config.prefix_cache && config.pages ? slots : 0), pt_(config.page_tokens),
        components_(static_cast<std::uint32_t>(config.components.size())),
        windows_(static_cast<std::uint32_t>(config.windows.size())), identity_(config.identity),
        hash_(config.hash ? config.hash : crit::core::chain_hash) {
    if (slots_ && !pt_) throw std::invalid_argument("host page tier needs page_tokens");
    std::size_t buckets = 1;
    while (buckets < std::size_t(slots_) * 2) buckets *= 2;
    mask_ = slots_ ? buckets - 1 : 0;
    index_.assign(slots_ ? buckets : 0, no_slot);
    hashes_.assign(slots_, 0);
    scopes_.assign(slots_, 0);
    tokens_.assign(std::size_t(slots_) * pt_, 0);
    state_.assign(std::size_t(slots_) * components_, 0);
    window_rows_.assign(std::size_t(slots_) * windows_, 0);
    prev_.assign(slots_, no_slot);
    next_.assign(slots_, no_slot);
  }

  std::uint32_t slots() const noexcept { return slots_; }
  std::uint32_t kept() const noexcept { return kept_; }
  std::uint64_t admitted() const noexcept { return admitted_; }
  std::uint64_t dropped() const noexcept { return dropped_; }
  // Whether `slot` holds component c's state after its last row, and window group w's rows.
  bool state(std::uint32_t slot, std::uint32_t c) const { return state_.at(std::size_t(slot) * components_ + c) != 0; }
  bool window_rows(std::uint32_t slot, std::uint32_t w) const { return window_rows_.at(std::size_t(slot) * windows_ + w) != 0; }

  // From StateStorage::evicted: the slot to copy the page's rows into, and its component state and window rows where
  // the page names them (snapshots[c] != no_slot, windows[w] != no_page). The least recently used page leaves when
  // the tier is full. nullopt: no slots, or the tier already holds this page (same hash and tokens).
  std::optional<std::uint32_t> admit(const crit::core::EvictedPage& page) noexcept {
    if (!slots_ || page.tokens.size() != pt_) return std::nullopt;
    if (const auto held = find(page.hash, page.tokens, page.scope); held != no_slot) {
      touch(held);
      return std::nullopt;
    }
    std::uint32_t slot;
    if (kept_ < slots_) slot = kept_++;
    else {
      slot = head_;
      remove(slot);
      ++dropped_;
    }
    hashes_[slot] = page.hash;
    scopes_[slot] = page.scope;
    for (std::uint32_t i = 0; i < pt_; ++i) tokens_[std::size_t(slot) * pt_ + i] = page.tokens[i];
    for (std::uint32_t c = 0; c < components_; ++c)
      state_[std::size_t(slot) * components_ + c] = c < page.snapshots.size() && page.snapshots[c] != crit::core::no_slot;
    for (std::uint32_t w = 0; w < windows_; ++w)
      window_rows_[std::size_t(slot) * windows_ + w] = w < page.windows.size() && page.windows[w] != crit::core::no_page;
    insert(slot);
    ++admitted_;
    return slot;
  }

  // The kept pages that continue `prompt` after its first `from` rows (a multiple of page_tokens, as attach() and
  // committed rounds leave it), in order: at most min(prompt.size(), max_reuse) rows and `max_rows` rows (the cache's
  // max_round_rows, so they fit one round), each verified by its chain hash and exact tokens. Every page holds every
  // window group's rows (a window page a restore round leaves unwritten would be linked for reuse), and the run ends
  // at the last page holding every component's state. Restore them in one round of pages x page_tokens rows
  // (RoundStates::final_row): write each page's rows where Round::kv and Round::window_kv point, the last page's
  // state into scratch[0] (in_place components: into the live slot), and commit the pages' tokens. Marks them used.
  // `scope` is the sequence's reuse scope (StateCache::open): only pages written under it match.
  std::vector<std::uint32_t> continuation(std::span<const crit::core::Token> prompt, std::uint32_t from,
                                          std::uint32_t max_reuse, std::uint32_t max_rows, std::uint64_t scope = 0) {
    std::vector<std::uint32_t> out;
    if (!kept_ || from % pt_) return out;
    const auto limit = std::min<std::size_t>({prompt.size(), max_reuse, std::size_t(from) + max_rows});
    std::uint64_t h = scope;
    std::size_t row = 0;
    for (; row + pt_ <= from && row + pt_ <= prompt.size(); row += pt_) h = hash_(h, identity_, prompt.subspan(row, pt_));
    if (row != from) return out;
    for (; row + pt_ <= limit; row += pt_) {
      const auto tokens = prompt.subspan(row, pt_);
      h = hash_(h, identity_, tokens);
      const auto slot = find(h, tokens, scope);
      if (slot == no_slot) break;
      bool windows = true;
      for (std::uint32_t w = 0; w < windows_; ++w) windows = windows && window_rows(slot, w);
      if (!windows) break;
      out.push_back(slot);
    }
    if (components_)
      while (!out.empty()) {
        bool complete = true;
        for (std::uint32_t c = 0; c < components_; ++c) complete = complete && state(out.back(), c);
        if (complete) break;
        out.pop_back();
      }
    for (auto slot : out) touch(slot);
    return out;
  }

private:
  std::uint32_t slots_, pt_, components_, windows_;
  std::uint64_t identity_;
  crit::core::PageHash hash_;
  std::size_t mask_ = 0;
  std::vector<std::uint32_t> index_;           // open addressing by hash, linear probing, no tombstones
  std::vector<std::uint64_t> hashes_, scopes_;
  std::vector<crit::core::Token> tokens_;      // slots x page_tokens
  std::vector<std::uint8_t> state_, window_rows_;
  std::vector<std::uint32_t> prev_, next_;     // least recently used first
  std::uint32_t head_ = no_slot, tail_ = no_slot, kept_ = 0;
  std::uint64_t admitted_ = 0, dropped_ = 0;

  bool same(std::uint32_t slot, std::span<const crit::core::Token> tokens) const noexcept {
    for (std::uint32_t i = 0; i < pt_; ++i)
      if (tokens_[std::size_t(slot) * pt_ + i] != tokens[i]) return false;
    return true;
  }
  std::uint32_t find(std::uint64_t h, std::span<const crit::core::Token> tokens, std::uint64_t scope) const noexcept {
    if (!slots_) return no_slot;
    for (auto i = std::size_t(h) & mask_; index_[i] != no_slot; i = (i + 1) & mask_)
      if (hashes_[index_[i]] == h && scopes_[index_[i]] == scope && same(index_[i], tokens)) return index_[i];
    return no_slot;
  }
  void link_tail(std::uint32_t slot) noexcept {
    prev_[slot] = tail_;
    next_[slot] = no_slot;
    if (tail_ != no_slot) next_[tail_] = slot;
    else head_ = slot;
    tail_ = slot;
  }
  void unlink(std::uint32_t slot) noexcept {
    if (prev_[slot] != no_slot) next_[prev_[slot]] = next_[slot];
    else head_ = next_[slot];
    if (next_[slot] != no_slot) prev_[next_[slot]] = prev_[slot];
    else tail_ = prev_[slot];
  }
  void touch(std::uint32_t slot) noexcept {
    unlink(slot);
    link_tail(slot);
  }
  void insert(std::uint32_t slot) noexcept {
    auto i = std::size_t(hashes_[slot]) & mask_;
    while (index_[i] != no_slot) i = (i + 1) & mask_;
    index_[i] = slot;
    link_tail(slot);
  }
  // Backward-shift deletion keeps every probe run unbroken.
  void remove(std::uint32_t slot) noexcept {
    auto i = std::size_t(hashes_[slot]) & mask_;
    while (index_[i] != slot) i = (i + 1) & mask_;
    index_[i] = no_slot;
    for (auto j = (i + 1) & mask_; index_[j] != no_slot; j = (j + 1) & mask_) {
      const auto home = std::size_t(hashes_[index_[j]]) & mask_;
      if (i <= j ? (i < home && home <= j) : (i < home || home <= j)) continue;
      index_[i] = index_[j];
      index_[j] = no_slot;
      i = j;
    }
    unlink(slot);
  }
};

}  // namespace crit::techniques
