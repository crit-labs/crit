#include "attention.h"
#include "state_pages_fixture.h"
#include <array>
#include <iostream>
#include <optional>
#include <random>
#include <string>

// FP32 KV pages under Core's StateCache. Core's own tests (tests/core/state_tests.cpp)
// cover the bookkeeping: rounds, ref counts, forks, prefix hashing, eviction,
// epochs and failure atomicity. These cases cover what Inference adds: the rows
// StatePages stores and copies where the cache points, the views attention reads
// and the paged scalar attention reference.
namespace cc = crit::core;
using namespace crit::techniques;
void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template<class F> void rejects(F action) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error("Expected rejection");
}
void equal(const StatePagesView& view, std::initializer_list<float> values) {
  require(view.rows() == values.size(), "Wrong visible row count");
  std::size_t i = 0;
  for (auto value : values) require(view.value(i++)[0] == value, "Visible value differs");
}
void transactions() {
  PagedState kv(5, 2, 1, 1, 6);
  require(kv.pages.reserved_bytes() == 5 * 2 * 2 * sizeof(float), "Incorrect arena bytes");
  const auto state = kv.open();
  auto round = kv.pages.stage(kv.cache, state, std::array{1.f, 2.f, 3.f}, std::array{10.f, 20.f, 30.f});
  require(kv.view(state).rows() == 0 && kv.cache.staged(state) == 3, "Staged rows leaked into the committed view");
  equal(kv.view(state, true), {10, 20, 30});
  kv.cache.commit(state, round.id, std::array<cc::Token, 1>{1}); // Accept one row.
  equal(kv.view(state), {10});
  kv.settle();
  require(kv.cache.stats().live_pages == 1, "Rejected suffix still owned by the sequence");
  const auto original = kv.view(state);
  const auto branch = kv.cache.fork(state);
  require(kv.cache.stats().live_pages == 1, "Fork copied shared data eagerly");
  const auto page = original.block(0).slot;
  require(kv.cache.pages_needed(branch, 3) == 2, "COW and suffix admission count differs");
  round = kv.pages.stage(kv.cache, branch, std::array{8.f, 9.f, 4.f}, std::array{80.f, 90.f, 40.f});
  const auto copied = kv.view(branch, true);
  require(copied.block(0).slot != page && copied.key(0)[0] == 1.f && copied.value(0)[0] == 10.f,
          "Shared tail was not copied through copy_rows before the branch wrote");
  equal(original, {10}); equal(kv.view(state), {10});
  kv.cache.commit(branch, round.id, std::array<cc::Token, 2>{8, 9}); equal(kv.view(branch), {10, 80, 90});
  kv.cache.rewind(branch, 1);
  round = kv.pages.stage(kv.cache, branch, std::array{5.f}, std::array{50.f});
  kv.cache.commit(branch, round.id, std::array<cc::Token, 1>{5});
  equal(kv.view(branch), {10, 50});
  round = kv.pages.stage(kv.cache, state, std::array{2.f}, std::array{20.f});
  kv.cache.discard(state, round.id); equal(kv.view(state), {10});
  equal(kv.view(branch), {10, 50});
  const auto prefix = kv.view(branch).prefix(1);
  require(prefix.pages() == 1, "A prefix view holds pages past its rows");
  equal(prefix, {10});
  rejects([&] { prefix.key(1); });
  rejects([&] { kv.view(branch).prefix(3); });
}
// Prefix reuse is Core's: attach() references verified full pages. The reused
// rows must be the stored K/V and attention over them must match dense.
void prefixes() {
  PagedState kv(12, 2, 1, 1, 10, 4, true);
  const std::array<cc::Token, 5> prompt{1, 2, 3, 4, 5};
  const std::array<float, 5> keys{.1f, .2f, .3f, .4f, .5f}, values{10, 20, 30, 40, 50};
  const auto first = kv.open();
  kv.append(first, keys, values, prompt);
  kv.cache.close(first); kv.settle();
  require(kv.cache.stats().cached_pages == 2, "Full prompt pages were not indexed");
  const auto second = kv.open();
  const auto reused = kv.cache.attach(second, prompt, static_cast<std::uint32_t>(prompt.size() - 1));
  require(reused == 4 && kv.cache.committed(second) == 4, "Only verified full pages are reused");
  equal(kv.view(second), {10, 20, 30, 40});
  for (std::size_t row = 0; row < 4; ++row) require(kv.view(second).key(row)[0] == keys[row], "Reused key differs");
  kv.append(second, std::span<const float>(keys).last(1), std::span<const float>(values).last(1),
            std::span<const cc::Token>(prompt).last(1));
  const std::array<float, 1> query{2};
  const auto actual = paged_attention(kv.view(second), query, 1);
  double denominator = 0, expected = 0;
  for (std::size_t row = 0; row < 5; ++row) {
    const double weight = std::exp(static_cast<double>(query[0]) * keys[row]);
    denominator += weight; expected += weight * values[row];
  }
  require(std::abs(actual[0] - expected / denominator) < 1e-5, "Attention over reused pages differs from dense");
  const auto other = kv.open();
  require(kv.cache.attach(other, std::array<cc::Token, 3>{1, 9, 9}, 2) == 0, "Reused a nonmatching prefix");
  require(kv.cache.attach(kv.open(), std::array<cc::Token, 4>{1, 2, 9, 9}, 3) == 2, "Partial prefix reuse differs");
}
void numerics() {
  std::mt19937 random(9182);
  std::uniform_real_distribution<float> sample(-2, 2);
  for (std::uint32_t page_rows : {1u, 2u, 7u, 32u}) for (std::size_t count : {1, 3, 16, 39}) {
    PagedState kv(50, page_rows, 5, 3, static_cast<std::uint32_t>(count + 2));
    const auto state = kv.open();
    std::vector<float> keys(count * 5), values(count * 3);
    std::array<float, 5> query;
    for (auto& x : keys) x = sample(random);
    for (auto& x : values) x = sample(random);
    for (auto& x : query) x = sample(random);
    kv.append(state, keys, values);
    for (std::size_t visible = 1; visible <= count; ++visible) {
      auto actual = paged_attention(kv.view(state).prefix(visible), query, .7);
      std::vector<double> scores(visible);
      for (std::size_t row = 0; row < visible; ++row)
        for (std::size_t col = 0; col < 5; ++col) scores[row] += static_cast<double>(query[col]) * keys[row * 5 + col] * .7;
      const auto top = *std::max_element(scores.begin(), scores.end());
      double denom = 0;
      for (auto& s : scores) { s = std::exp(s - top); denom += s; }
      for (std::size_t col = 0; col < 3; ++col) {
        double expected = 0;
        for (std::size_t row = 0; row < visible; ++row) expected += scores[row] / denom * values[row * 3 + col];
        require(std::abs(actual[col] - expected) < 2e-6, "Paged attention differs from independent dense reference");
      }
    }
    const auto original = kv.view(state);
    const auto branch = kv.cache.fork(state);
    const auto round = kv.pages.stage(kv.cache, branch, std::array{1.f, 1.f, 1.f, 1.f, 1.f}, std::array{999.f, 999.f, 999.f});
    kv.cache.discard(branch, round.id);
    require(paged_attention(kv.view(branch), query, .7) == paged_attention(original, query, .7), "Rollback changed attention");
  }
}
void failures() {
  rejects([] { StatePages p(1, 0, 1, 1); });
  rejects([] { StatePages p(1, 1, 0, 1); });
  rejects([] { StatePages p(1, 1, 1, std::numeric_limits<std::size_t>::max()); });
  rejects([] { StatePages p(std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max(),
                            std::numeric_limits<std::size_t>::max() / 4, 1); });
  PagedState kv(5, 2, 1, 1, 5);
  const auto state = kv.open();
  kv.append(state, std::array{1.f}, std::array{10.f});
  const auto snapshot = kv.view(state);
  const auto live = kv.cache.stats().live_pages;
  const auto unchanged = [&] {
    require(kv.cache.committed(state) == 1 && kv.cache.staged(state) == 0 && kv.cache.stats().live_pages == live,
            "Rejected rows changed the cache");
    equal(kv.view(state), {10});
  };
  rejects([&] { kv.pages.stage(kv.cache, state, std::array{std::numeric_limits<float>::infinity()}, std::array{1.f}); });
  unchanged();
  rejects([&] { kv.pages.stage(kv.cache, state, std::array{1.f, 2.f}, std::array{1.f}); });
  unchanged();
  rejects([&] { kv.pages.stage(kv.cache, state, std::span<const float>{}, std::span<const float>{}); });
  rejects([&] { kv.pages.stage(kv.cache, state, std::vector<float>(5, 1), std::vector<float>(5, 1)); }); // Past max_tokens.
  unchanged();
  rejects([&] { kv.pages.write({5, 0}, std::array{1.f}, std::array{1.f}); });
  rejects([&] { kv.pages.write({0, 2}, std::array{1.f}, std::array{1.f}); });
  rejects([&] { kv.pages.write_rows(std::array<cc::RowSlot, 1>{{{0, 0}}}, std::array{1.f, 2.f}, std::array{1.f, 2.f}); });
  rejects([&] { kv.pages.copy_rows(0, 5, 1); });
  rejects([&] { kv.pages.copy_rows(0, 1, 3); });
  rejects([&] { kv.pages.copy_slot(0, 0, 1); });
  rejects([&] { kv.pages.view(kv.cache.block_table(state), 3); });
  rejects([&] { snapshot.block(1); });
  rejects([&] { paged_attention(snapshot, std::array{1.f}, 0); });
  rejects([&] { paged_attention(snapshot, std::array{1.f, 1.f}, 1); });
  rejects([&] { paged_attention(snapshot.prefix(0), std::array{1.f}, 1); });
  rejects([&] { paged_attention(snapshot, std::array{2.f}, std::numeric_limits<double>::max()); });
  unchanged();
}
// Random rounds, partial commits, discards, rewinds, forks and closes against
// a value oracle: every view reads exactly its sequence's rows (so every
// copy-on-write went through copy_rows), and a view taken before an operation
// still reads its rows until retire() covers that epoch.
void randomized() {
  struct State { cc::SequenceId id; std::vector<float> committed, staged; std::optional<std::uint64_t> round; };
  struct Read { StatePagesView view; std::vector<float> values; };
  PagedState kv(24, 3, 1, 1, 24, 8);
  std::vector<State> states;
  std::mt19937 random(715982);
  const auto check = [](const StatePagesView& view, std::span<const float> values) {
    require(view.rows() == values.size(), "Randomized visible length differs");
    for (std::size_t i = 0; i < values.size(); ++i)
      require(view.key(i)[0] == values[i] && view.value(i)[0] == -values[i], "Randomized rows differ from the oracle");
  };
  std::size_t staged = 0, copied = 0;
  for (int step = 0; step < 5000; ++step) {
    if (states.empty()) states.push_back({kv.open(), {}, {}, std::nullopt});
    auto& state = states[random() % states.size()];
    std::vector<Read> readers;
    for (int i = 0; i < 2; ++i) {
      const auto& read = states[random() % states.size()];
      const bool pending = random() % 2 != 0;
      auto values = read.committed;
      if (pending) values.insert(values.end(), read.staged.begin(), read.staged.end());
      readers.push_back({kv.view(read.id, pending), std::move(values)});
    }
    switch (random() % 7) {
    case 0:
      if (!state.round) {
        const auto count = 1 + random() % 5;
        if (state.committed.size() + count > 24) break;
        std::vector<float> keys(count), values(count);
        for (std::size_t i = 0; i < count; ++i) { keys[i] = static_cast<float>(1 + random() % 1000); values[i] = -keys[i]; }
        const auto tail = state.committed.size() / 3;
        const auto partial = state.committed.size() % 3 != 0;
        const auto before = partial ? kv.cache.block_table(state.id)[tail] : cc::no_page;
        try {
          const auto round = kv.pages.stage(kv.cache, state.id, keys, values);
          state.round = round.id; state.staged = keys; ++staged;
          copied += partial && kv.cache.block_table(state.id)[tail] != before;
        } catch (const cc::CapacityExhausted&) {}
      }
      break;
    case 1:
      if (state.round) {
        const auto accept = random() % (state.staged.size() + 1);
        kv.cache.commit(state.id, *state.round, std::vector<cc::Token>(accept, 0));
        state.committed.insert(state.committed.end(), state.staged.begin(), state.staged.begin() + static_cast<std::ptrdiff_t>(accept));
        state.staged.clear(); state.round.reset();
      }
      break;
    case 2:
      if (state.round) { kv.cache.discard(state.id, *state.round); state.staged.clear(); state.round.reset(); }
      break;
    case 3:
      if (!state.round) {
        const auto count = random() % (state.committed.size() + 1);
        kv.cache.rewind(state.id, static_cast<std::uint32_t>(count)); state.committed.resize(count);
      }
      break;
    case 4:
      if (!state.round && states.size() < 8) {
        State next{kv.cache.fork(state.id), state.committed, {}, std::nullopt};
        states.push_back(std::move(next));
      }
      break;
    case 5:
      if (states.size() < 8) states.push_back({kv.open(), {}, {}, std::nullopt});
      break;
    default:
      if (states.size() > 1) {
        const auto victim = states.begin() + static_cast<std::ptrdiff_t>(random() % states.size());
        kv.cache.close(victim->id); states.erase(victim);
      }
    }
    for (const auto& reader : readers) check(reader.view, reader.values);
    readers.clear();
    kv.settle();
    for (const auto& current : states) {
      check(kv.view(current.id), current.committed);
      auto all = current.committed;
      all.insert(all.end(), current.staged.begin(), current.staged.end());
      check(kv.view(current.id, true), all);
    }
  }
  require(staged > 300 && copied > 10, "Randomized walk omitted rounds or copy-on-write");
  for (const auto& state : states) kv.cache.close(state.id);
  kv.settle();
  require(kv.cache.stats().live_pages == 0, "Randomized teardown leaked pages");
}
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("Expected a test case");
    const std::string test = argv[1];
    if (test == "transactions") transactions();
    else if (test == "prefixes") prefixes();
    else if (test == "numerics") numerics();
    else if (test == "failures") failures();
    else if (test == "randomized") randomized();
    else throw std::invalid_argument("Unknown test case");
    std::cout << "State pages " << test << " passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
