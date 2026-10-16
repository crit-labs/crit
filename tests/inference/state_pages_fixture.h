#pragma once
#include "state_pages.h"
#include <crit/core.h>
#include <crit/state.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// One Core StateCache over Inference StatePages for attention fixtures: the cache
// keeps sequences, rounds, forks and retirement, the storage keeps FP32 rows.
// Fixture arithmetic is synchronous, so settle() may retire every release.
struct PagedState {
  crit::core::Resources resources;
  crit::techniques::StatePages pages;
  crit::core::StateCache cache;
  static crit::core::StateCacheConfig config(std::uint32_t page_count, std::uint32_t page_rows,
                                             std::uint32_t max_tokens, std::uint32_t sequences, bool prefix) {
    crit::core::StateCacheConfig result;
    result.pages = page_count; result.page_tokens = page_rows; result.sequences = sequences;
    result.max_tokens = max_tokens; result.max_round_rows = max_tokens;
    result.prefix_cache = prefix; result.identity = prefix ? 0xa77e : 0;
    return result;
  }
  PagedState(std::uint32_t page_count, std::uint32_t page_rows, std::size_t key_width, std::size_t value_width,
             std::uint32_t max_tokens, std::uint32_t sequences = 8, bool prefix = false)
      : resources(std::size_t{1} << 28, 0), pages(page_count, page_rows, key_width, value_width),
        cache(config(page_count, page_rows, max_tokens, sequences, prefix), pages, resources) {}
  crit::core::SequenceId open() { return cache.open(); }
  // Stages and commits token-major rows; tokens default to zeros (only prefix reuse reads them).
  void append(crit::core::SequenceId sequence, std::span<const float> keys, std::span<const float> values,
              std::span<const crit::core::Token> tokens = {}) {
    if (keys.empty() && values.empty()) return;
    const auto round = pages.stage(cache, sequence, keys, values);
    if (tokens.empty()) cache.commit(sequence, round.id, std::vector<crit::core::Token>(round.rows, 0));
    else cache.commit(sequence, round.id, tokens);
  }
  crit::techniques::StatePagesView view(crit::core::SequenceId sequence, bool include_staged = false) const {
    return pages.view(cache, sequence, include_staged);
  }
  void settle() { cache.retire(cache.epoch()); }
  std::uint32_t free_pages() const {
    const auto stats = cache.stats();
    return stats.free_pages + stats.cached_pages;
  }
};
