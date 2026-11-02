#include "host_page_tier.h"
#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// The host page tier over Core's StateCache. A fake engine keeps one value per KV row, window row and recurrent slot:
// the digest of the whole token prefix it was computed for, so a restored page that holds another prefix's rows, a
// restore that skips a window row or a state restored for the wrong page shows up as a wrong value. Pages the cache
// evicts are copied into the tier's host slots; a later prompt restores them in rounds instead of recomputing them.
namespace cc = crit::core;
using namespace crit::techniques;
namespace {
void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
using Value = std::uint64_t;
using Tokens = std::vector<cc::Token>;
Value digest(Value previous, cc::Token token) {
  auto z = previous ^ (std::uint64_t(std::uint32_t(token)) + 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 29)) * 0xbf58476d1ce4e5b9ull;
  return z ^ (z >> 32);
}
std::vector<Value> digests(const Tokens& tokens) {
  std::vector<Value> out;
  Value value = 0;
  for (auto token : tokens) out.push_back(value = digest(value, token));
  return out;
}
Tokens range(cc::Token first, cc::Token count) {
  Tokens out;
  for (cc::Token i = 0; i < count; ++i) out.push_back(first + i);
  return out;
}
Tokens join(Tokens a, const Tokens& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}
std::uint64_t constant_hash(std::uint64_t, std::uint64_t, std::span<const cc::Token>) noexcept { return 7; }

// Engine storage: device rows, window rows and slots, and the host tier's copies of evicted pages.
struct Engine final : cc::StateStorage {
  cc::StateCacheConfig config;
  std::uint32_t pt;
  std::vector<Value> rows;                       // pages x page_tokens
  std::vector<std::vector<Value>> windows, slots;
  HostPageTier tier;
  std::vector<Value> host_rows, host_state;      // tier slots x page_tokens, tier slots x components
  std::vector<std::vector<Value>> host_windows;  // per group: tier slots x page_tokens
  cc::Resources resources{std::size_t{1} << 26, 0};
  std::optional<cc::StateCache> cache;
  std::uint64_t spills = 0, restored_rows = 0, computed_rows = 0;

  Engine(const cc::StateCacheConfig& c, std::uint32_t tier_slots)
      : config(c), pt(c.page_tokens), rows(std::size_t(c.pages) * c.page_tokens, 0), tier(tier_slots, c),
        host_rows(std::size_t(tier_slots) * c.page_tokens, 0), host_state(std::size_t(tier_slots) * c.components.size(), 0) {
    for (const auto& group : c.windows) windows.emplace_back(std::size_t(group.pages) * pt, 0);
    for (const auto& component : c.components) slots.emplace_back(component.slots, 0);
    for (std::size_t w = 0; w < c.windows.size(); ++w) host_windows.emplace_back(std::size_t(tier_slots) * pt, 0);
    cache.emplace(c, *this, resources);
  }
  void copy_rows(cc::PageId from, cc::PageId to, std::uint32_t n) override {
    for (std::uint32_t i = 0; i < n; ++i) rows[std::size_t(to) * pt + i] = rows[std::size_t(from) * pt + i];
  }
  void copy_window_rows(std::uint32_t w, cc::PageId from, cc::PageId to, std::uint32_t n) override {
    for (std::uint32_t i = 0; i < n; ++i) windows[w][std::size_t(to) * pt + i] = windows[w][std::size_t(from) * pt + i];
  }
  void copy_slot(std::uint32_t c, cc::SlotId from, cc::SlotId to) override { slots[c][to] = slots[c][from]; }
  void evicted(const cc::EvictedPage& page) noexcept override {
    const auto slot = tier.admit(page);
    if (!slot) return;
    ++spills;
    for (std::uint32_t i = 0; i < pt; ++i) host_rows[std::size_t(*slot) * pt + i] = rows[std::size_t(page.page) * pt + i];
    for (std::size_t c = 0; c < page.snapshots.size(); ++c)
      if (page.snapshots[c] != cc::no_slot) host_state[*slot * slots.size() + c] = slots[c][page.snapshots[c]];
    for (std::size_t w = 0; w < page.windows.size(); ++w)
      if (page.windows[w] != cc::no_page)
        for (std::uint32_t i = 0; i < pt; ++i)
          host_windows[w][std::size_t(*slot) * pt + i] = windows[w][std::size_t(page.windows[w]) * pt + i];
  }

  // The values the model computes for rows [first, first + n) of `prompt`, written where the round points; the
  // state after the round's last row goes where its commit mode reads it.
  void write_computed(cc::SequenceId seq, const cc::Round& r, const Tokens& prompt) {
    const auto values = digests(prompt);
    for (std::uint32_t i = 0; i < r.rows; ++i) {
      rows[std::size_t(r.kv[i].page) * pt + r.kv[i].offset] = values[r.first_row + i];
      for (std::size_t w = 0; w < windows.size(); ++w) {
        const auto slot = r.window_kv[w * r.rows + i];
        windows[w][std::size_t(slot.page) * pt + slot.offset] = values[r.first_row + i];
      }
    }
    write_state(seq, r, values[r.first_row + r.rows - 1]);
  }
  void write_state(cc::SequenceId seq, const cc::Round& r, Value value) {
    for (std::size_t c = 0; c < slots.size(); ++c) {
      if (config.components[c].commit == cc::RecurrentCommit::in_place) slots[c][cache->live_slot(seq, std::uint32_t(c))] = value;
      else slots[c][r.scratch[c * r.rows]] = value;
    }
  }
  // Prefills `prompt` in chunks of at most `chunk` rows, cut where snapshots belong, after restoring what the tier
  // holds; returns the rows restored from the tier.
  std::uint64_t prefill(cc::SequenceId seq, const Tokens& prompt, std::uint32_t chunk, bool use_tier = true) {
    const auto max_reuse = std::uint32_t(prompt.size() - 1);
    const auto before = restored_rows;
    cache->attach(seq, prompt, max_reuse);
    while (use_tier) {
      const auto from = cache->committed(seq);
      const auto pages = tier.continuation(prompt, from, max_reuse, config.max_round_rows);
      if (pages.empty()) break;
      const auto n = std::uint32_t(pages.size()) * pt;
      const auto r = cache->begin(seq, n, cc::RoundStates::final_row);
      for (std::uint32_t i = 0; i < n; ++i) {
        const auto slot = pages[i / pt];
        rows[std::size_t(r.kv[i].page) * pt + r.kv[i].offset] = host_rows[std::size_t(slot) * pt + i % pt];
        for (std::size_t w = 0; w < windows.size(); ++w) {
          const auto at = r.window_kv[w * n + i];
          windows[w][std::size_t(at.page) * pt + at.offset] = host_windows[w][std::size_t(slot) * pt + i % pt];
        }
      }
      if (!slots.empty()) write_state(seq, r, host_state[pages.back() * slots.size()]);
      cache->commit(seq, r.id, std::span<const cc::Token>(prompt).subspan(from, n));
      restored_rows += n;
    }
    while (cache->committed(seq) < prompt.size()) {
      const auto from = cache->committed(seq);
      const auto want = std::min<std::uint32_t>(chunk, std::uint32_t(prompt.size()) - from);
      const auto n = cache->snapshot_rows(seq, prompt, want);
      const auto r = cache->begin(seq, n, cc::RoundStates::final_row);
      write_computed(seq, r, prompt);
      cache->commit(seq, r.id, std::span<const cc::Token>(prompt).subspan(from, n));
      computed_rows += n;
    }
    return restored_rows - before;
  }
  // Every committed row, live window row and live state of `seq` holds what the model computes for `prompt`.
  void check(cc::SequenceId seq, const Tokens& prompt) {
    const auto values = digests(prompt);
    const auto table = cache->block_table(seq);
    require(cache->committed(seq) == prompt.size(), "sequence length differs");
    for (std::size_t r = 0; r < prompt.size(); ++r)
      require(rows[std::size_t(table[r / pt]) * pt + r % pt] == values[r], "a KV row holds another prefix's value");
    for (std::size_t w = 0; w < windows.size(); ++w) {
      const auto window = cache->window_table(seq, std::uint32_t(w));
      const auto first = prompt.size() + 1 > config.windows[w].window ? prompt.size() + 1 - config.windows[w].window : 0;
      for (auto r = first; r < prompt.size(); ++r)
        require(windows[w][std::size_t(window[r / pt]) * pt + r % pt] == values[r], "a window row holds another value");
    }
    for (std::size_t c = 0; c < slots.size(); ++c)
      require(slots[c][cache->live_slot(seq, std::uint32_t(c))] == values.back(), "the live state differs");
  }
};

cc::StateCacheConfig hybrid_config(std::uint32_t pages) {
  cc::StateCacheConfig c;
  c.pages = pages; c.page_tokens = 4; c.sequences = 2; c.max_tokens = 40; c.max_round_rows = 16;
  c.prefix_cache = true; c.identity = 0x7e11;
  cc::StateComponent component;
  component.commit = cc::RecurrentCommit::recompute; component.slots = 2 + 1 + 6; component.snapshots = 6;
  c.components = {component};
  c.windows = {{20, 6}};
  return c;
}

// A conversation's pages are evicted by other prompts, spilled to the tier, and restored for its next turn: every
// row, window row and the state equal what recomputing them gives, and the next turn computes only its new rows.
void restore() {
  Engine engine(hybrid_config(10), 16);
  auto& cache = *engine.cache;
  const Tokens first = range(1, 24);
  auto a = cache.open();
  require(engine.prefill(a, first, 16) == 0, "nothing to restore on a cold tier");
  engine.check(a, first);
  cache.close(a);
  cache.retire(cache.epoch());
  for (cc::Token other = 0; other < 2; ++other) {   // two other prompts take the pages back
    const Tokens prompt = range(100 + 50 * other, 32);
    auto b = cache.open();
    engine.prefill(b, prompt, 16);
    engine.check(b, prompt);
    cache.close(b);
    cache.retire(cache.epoch());
  }
  require(engine.spills > 0 && engine.tier.kept() > 0, "evicted pages were not spilled");
  require(cache.reusable(join(first, range(500, 4)), 27) == 0, "the first conversation is still on the device");
  const auto next_turn = join(first, range(500, 4));
  auto c = cache.open();
  const auto computed = engine.computed_rows;
  const auto restored = engine.prefill(c, next_turn, 16);
  require(restored == 24, "the tier restores every page of the earlier prompt");
  require(engine.computed_rows - computed == 4, "only the new turn's rows are computed");
  engine.check(c, next_turn);
  // The restored pages are indexed again, snapshot included: a sibling reuses them from the device.
  cache.close(c);
  cache.retire(cache.epoch());
  require(cache.reusable(join(first, range(700, 4)), 27) == 24, "restored pages are reusable from the device");
}

// The run ends at the last page holding the state, never past a page without window rows, within max_rows rows and
// max_reuse; a page is verified by its tokens even when every hash collides; capacity drops the least recently used.
void continuation_rules() {
  Engine engine(hybrid_config(10), 16);
  auto& tier = engine.tier;
  std::vector<cc::SlotId> snap{3};
  std::vector<cc::PageId> win{2};
  auto page = [&](const Tokens& prompt, std::size_t index, bool state, bool window) {
    std::uint64_t h = 0;
    for (std::size_t i = 0; i <= index; ++i)
      h = cc::chain_hash(h, engine.config.identity, std::span<const cc::Token>(prompt).subspan(i * 4, 4));
    snap[0] = state ? 3 : cc::no_slot;
    win[0] = window ? 2 : cc::no_page;
    cc::EvictedPage e;
    e.page = 0; e.hash = h; e.tokens = std::span<const cc::Token>(prompt).subspan(index * 4, 4);
    e.snapshots = snap; e.windows = win;
    return tier.admit(e);
  };
  const Tokens p = range(1, 24);
  require(page(p, 0, false, true) && page(p, 1, true, true) && page(p, 2, false, true) && page(p, 3, true, false) &&
          page(p, 4, true, true), "admissions");
  require(!page(p, 1, true, true), "a page the tier holds is not admitted twice");
  require(tier.continuation(p, 0, 23, 16).size() == 2, "the run ends at the last page with state before a page without window rows");
  require(tier.continuation(p, 8, 23, 16).empty(), "page 2 has no state and page 3 no window rows");
  require(tier.continuation(p, 4, 23, 3).empty(), "max_rows bounds the run");
  require(tier.continuation(p, 4, 7, 16).empty(), "max_reuse bounds the run");
  require(tier.continuation(p, 2, 23, 16).empty(), "a start inside a page restores nothing");
  Tokens other = p;
  other[1] = 99;   // same later pages, another first page: another chain
  require(tier.continuation(other, 4, 23, 16).empty(), "a page is found by the whole prefix, not its own tokens");

  auto colliding = hybrid_config(10);
  colliding.hash = constant_hash;
  colliding.components.clear(); colliding.windows.clear();
  HostPageTier same(4, colliding);
  cc::EvictedPage e;
  e.hash = 7;
  const Tokens q = range(1, 8), r = range(50, 8);
  e.tokens = std::span<const cc::Token>(q).first(4);
  require(same.admit(e).has_value(), "first page admitted");
  e.tokens = std::span<const cc::Token>(r).first(4);
  require(same.admit(e).has_value(), "a colliding page with other tokens is another entry");
  require(same.continuation(q, 0, 7, 16).size() == 1 && same.continuation(r, 0, 7, 16).size() == 1,
          "each prompt finds its own page under one hash");
  require(same.continuation(q, 0, 7, 16, 5).empty(), "a page written under another reuse scope does not match");
  e.scope = 5;
  e.tokens = std::span<const cc::Token>(q).first(4);
  require(same.admit(e).has_value() && same.continuation(q, 0, 7, 16, 5).size() == 1, "the scope's own copy matches");
  e.scope = 0;
  const Tokens t = range(200, 8);
  e.tokens = std::span<const cc::Token>(t).first(4);
  require(same.continuation(t, 0, 7, 16).empty(), "no page holds these tokens");

  HostPageTier small(2, colliding);
  for (cc::Token k = 0; k < 3; ++k) {
    const Tokens s = range(1000 + 10 * k, 4);
    e.tokens = s;
    require(small.admit(e).has_value(), "admission into a full tier drops the least recently used page");
  }
  require(small.kept() == 2 && small.dropped() == 1 && small.admitted() == 3, "capacity accounting");
  require(small.continuation(range(1000, 5), 0, 4, 16).empty() && small.continuation(range(1020, 5), 0, 4, 16).size() == 1,
          "the oldest page left");
  HostPageTier none(0, colliding);
  require(!none.admit(e).has_value() && none.continuation(range(1020, 5), 0, 4, 16).empty(), "a tier without slots keeps nothing");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("Expected a test case");
    const std::string test = argv[1];
    if (test == "restore") restore();
    else if (test == "rules") continuation_rules();
    else throw std::invalid_argument("Unknown test case");
    std::cout << "Host page tier " << test << " passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
