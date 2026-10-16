#include "batch_attention.h"
#include "state_pages.h"
#include <crit/core.h>
#include <crit/state.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <optional>
#include <utility>

// Synthetic one-layer attention, deliberately independent of any checkpoint or
// device. Core's StateCache keeps every sequence's pages, rounds and retirement;
// StatePages holds the FP32 rows. The dense oracle below checks the complete
// streaming integration.
namespace cc = crit::core;
namespace ct = crit::techniques;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::array<float, 2> key(cc::Token t) { return {static_cast<float>(t + 1) / 16.f, 1.f}; }
std::array<float, 2> value(cc::Token t) { return {std::sin(static_cast<float>(t) + .5f), std::cos(static_cast<float>(t) + .5f)}; }
std::array<float, 2> query(cc::Token t) { return {1.f, static_cast<float>(t + 1) / 16.f}; }
cc::Token predict(std::span<const float> attention) {
  double best = -1e100; cc::Token next = 0;
  for (cc::Token t = 0; t < 16; ++t) {
    const auto v = value(t);
    const double score = static_cast<double>(attention[0]) * v[0] + static_cast<double>(attention[1]) * v[1];
    if (score > best) { best = score; next = t; }
  }
  return next;
}
std::vector<cc::Token> dense(std::vector<cc::Token> history, std::size_t count) {
  std::vector<cc::Token> output;
  while (output.size() < count) {
    const auto q = query(history.back());
    double denominator = 0; std::array<double, 2> sum{};
    for (auto t : history) {
      const auto k = key(t), v = value(t);
      const double weight = std::exp((static_cast<double>(q[0]) * k[0] + static_cast<double>(q[1]) * k[1]) * .5);
      denominator += weight;
      for (std::size_t d = 0; d < 2; ++d) sum[d] += weight * v[d];
    }
    const std::array<float, 2> attention{static_cast<float>(sum[0] / denominator), static_cast<float>(sum[1] / denominator)};
    const auto next = predict(attention); output.push_back(next); history.push_back(next);
  }
  return output;
}
struct Stats {
  std::atomic_bool start{false};
  std::size_t largest_batch = 0, finished = 0;
  bool mixed = false, refilled = false;
};
struct Backend final : cc::Backend {
  void initialize(cc::Resources&) override {}
  void bind_thread() override {}
  void drain() override {}
};
struct Receipt final : cc::Retirement { bool ready() override { return true; } };
struct Session final : cc::ConcurrentSession {
  cc::RequestView request;
  cc::StateCache& cache;
  cc::SequenceId sequence;
  std::optional<cc::Round> round;
  std::vector<cc::Token> inputs;   // tokens of the staged rows, committed with them
  std::vector<float> queries;
  std::vector<std::size_t> rows;
  std::vector<cc::Token> candidate;
  cc::Token last = 0;
  Session(cc::RequestView r, cc::StateCache& c) : request(r), cache(c), sequence(c.open()) {}
  ~Session() override { cache.close(sequence); }
  void close(cc::Finish) noexcept override { cache.close(sequence); }   // false, no change, once closed
};
cc::StateCacheConfig cache_config() {
  cc::StateCacheConfig config;
  config.pages = 512; config.page_tokens = 4; config.sequences = 32;
  config.max_tokens = 64; config.max_round_rows = 32;
  return config;
}
// fork() shares committed pages; the child's first row in the shared partial tail
// page makes the cache copy it through StatePages::copy_rows, so the parent's rows
// stay as they were.
void check_copy_on_write() {
  cc::Resources resources(std::size_t{1} << 20, 0);
  ct::StatePages store(8, 4, 2, 2);
  auto config = cache_config();
  config.pages = 8; config.sequences = 2; config.max_tokens = 16; config.max_round_rows = 4;
  cc::StateCache states(config, store, resources);
  const std::array<cc::Token, 3> prompt{1, 2, 3};
  const auto parent = states.open();
  auto round = states.begin(parent, 3);
  for (std::size_t i = 0; i < prompt.size(); ++i) store.write(round.kv[i], key(prompt[i]), value(prompt[i]));
  states.commit(parent, round.id, prompt);
  const auto child = states.fork(parent);
  round = states.begin(child, 1);
  store.write(round.kv[0], key(9), value(9));
  states.commit(child, round.id, std::array<cc::Token, 1>{9});
  const auto a = store.view(states.block_table(parent), 3).block(0);
  const auto b = store.view(states.block_table(child), 4).block(0);
  require(a.slot != b.slot && a.rows == 3 && b.rows == 4, "a shared partial tail is copied before the child writes");
  require(std::equal(a.values.begin(), a.values.end(), b.values.begin()), "copied rows differ");
  require(b.values[12] == key(9)[0] && b.values[15] == value(9)[1], "the child's row is in its own page");
}
struct Model final : cc::ConcurrentModel {
  std::shared_ptr<Stats> stats;
  ct::StatePages pages{512, 4, 2, 2};
  cc::StateCache cache;
  cc::Reservation arena, scratch;
  Model(std::shared_ptr<Stats> s, cc::Resources& resources) : stats(std::move(s)),
      cache(cache_config(), pages, resources),
      arena(resources.reserve(cc::Memory::host, pages.reserved_bytes())),
      scratch(resources.reserve(cc::Memory::host, 2 * 1024 * 1024)) {}
  cc::ModelInfo info() const override {
    cc::ModelInfo result{"synthetic.paged.batch", 16, 64, 8, {}, true, false, 32};
    result.mixed_batches = true; return result;
  }
  cc::Admission admission(const cc::RequestView&) override {
    return stats->start ? cc::Admission::ready : cc::Admission::deferred;
  }
  std::unique_ptr<cc::ConcurrentSession> open(const cc::RequestView& request) override {
    if (request.key.generation > 1 && stats->finished < 16) stats->refilled = true;
    return std::make_unique<Session>(request, cache);
  }
  std::vector<cc::RowProgress> execute(std::span<const cc::WorkItem> items) override {
    std::vector<ct::AttentionBatchSpec<ct::StatePagesView>> specs;
    std::vector<ct::AttentionBatchInput> inputs;
    std::vector<cc::RowProgress> progress;
    bool prefill = false, decode = false;
    for (const auto& item : items) {
      auto& s = static_cast<Session&>(*item.session);
      const auto start = cache.committed(s.sequence);
      const std::size_t count = item.budget.prefill_complete ? 1 : item.budget.prefill_tokens;
      require(count <= item.budget.token_work, "token grant exceeded");
      prefill |= !item.budget.prefill_complete; decode |= item.budget.prefill_complete;
      s.rows.resize(count); s.queries.clear(); s.candidate.clear(); s.inputs.clear();
      s.round = cache.begin(s.sequence, static_cast<std::uint32_t>(count));   // pages for every staged row, or no change
      for (std::size_t i = 0; i < count; ++i) {
        const auto token = item.budget.prefill_complete ? s.last : s.request.prompt[start + i];
        const auto k = key(token), v = value(token), q = query(token);
        pages.write(s.round->kv[i], k, v);
        s.inputs.push_back(token);
        s.queries.insert(s.queries.end(), q.begin(), q.end()); s.rows[i] = start + i;
      }
      const ct::AttentionRequestKey identity{item.key.request, item.key.generation};
      const auto rows = std::size_t{cache.committed(s.sequence)} + cache.staged(s.sequence);
      specs.push_back({identity, pages.view(cache.block_table(s.sequence), rows), {1, 1, 2, 2, .5}, s.rows});
      inputs.push_back({identity, {s.queries, count, 1, 2, 2, 2}});
    }
    stats->largest_batch = std::max(stats->largest_batch, items.size()); stats->mixed |= prefill && decode;
    ct::BasicAttentionBatch<ct::StatePagesView> batch(specs, 4, 4, {32, 256, 256 * 1024, 256 * 1024, 256 * 1024});
    std::reverse(inputs.begin(), inputs.end()); // Identity, not input order, selects state.
    const auto results = batch.run(inputs);
    for (const auto& item : items) {
      auto& s = static_cast<Session&>(*item.session);
      const bool done = item.budget.prefill_complete ||
          std::size_t{cache.committed(s.sequence)} + cache.staged(s.sequence) == s.request.prompt.size();
      const auto it = std::find_if(results.begin(), results.end(), [&](const auto& result) { return result.key.request == item.key.request; });
      if (done) s.candidate.push_back(predict(std::span<const float>(it->attention.values).last(2)));
      cc::Progress p{item.budget.prefill_complete ? 0 : item.budget.prefill_tokens, done, s.candidate};
      p.token_work = static_cast<std::uint32_t>(s.rows.size()); progress.push_back({item.key, p});
    }
    return progress;
  }
  cc::RowProgress prefill(const cc::WorkItem& item) override {
    auto mixed = item; mixed.budget.token_work = mixed.budget.prefill_tokens;
    return execute(std::span<const cc::WorkItem>(&mixed, 1)).front();
  }
  std::vector<cc::RowProgress> decode(std::span<const cc::WorkItem> items) override {
    std::vector<cc::WorkItem> mixed(items.begin(), items.end());
    for (auto& item : mixed) item.budget.token_work = 1;
    return execute(mixed);
  }
  void commit(std::span<const cc::CommitDecision> decisions) override {
    for (const auto& decision : decisions) {
      auto& s = static_cast<Session&>(*decision.session);
      if (!s.round) continue;
      const auto round = std::exchange(s.round, std::nullopt)->id;
      if (decision.cancelled) cache.discard(s.sequence, round);
      else { cache.commit(s.sequence, round, s.inputs); if (decision.accepted) s.last = s.candidate.front(); }
    }
    cache.retire(cache.epoch());   // all arithmetic is synchronous, so released pages are reusable now
  }
  std::unique_ptr<cc::Retirement> retire(cc::RequestKey, cc::ConcurrentSession& session, cc::Finish) override {
    cache.close(static_cast<Session&>(session).sequence);
    cache.retire(cache.epoch());
    ++stats->finished; return std::make_unique<Receipt>();
  }
  void quiesce() override {} // All example arithmetic completes synchronously.
  std::vector<cc::Token> tokenize(std::string_view) override { throw std::logic_error("Synthetic token-only example"); }
  std::string detokenize(std::span<const cc::Token>) override { throw std::logic_error("Synthetic token-only example"); }
};
}
int main() {
  try {
    check_copy_on_write();
    auto stats = std::make_shared<Stats>();
    cc::Profile p{64, 2, 8 * 1024 * 1024, 1024, 16, 16};
    p.max_batch_tokens = 32; p.batch_policy = cc::BatchPolicy::decode_first;
    cc::Engine engine(p, std::make_shared<Backend>(), [stats](const auto&, cc::Services services) {
      return std::make_unique<Model>(stats, services.resources);
    });
    std::vector<cc::Handle> handles; std::vector<std::vector<cc::Token>> expected;
    for (int i = 0; i < 24; ++i) {
      std::vector<cc::Token> prompt(static_cast<std::size_t>(i % 3 == 0 ? 1 : 15), i % 16);
      const auto output = static_cast<std::uint32_t>(i % 3 == 0 ? 1 : 10);
      expected.push_back(dense(prompt, output)); handles.push_back(engine.submit(prompt, output));
    }
    stats->start = true;
    for (std::size_t i = 0; i < handles.size(); ++i) require(handles[i].wait().tokens == expected[i], "batched output differs from dense oracle");
    require(stats->largest_batch == 16 && stats->mixed && stats->refilled, "continuous batch coverage incomplete");
    std::cout << "24 ragged requests match dense attention; 16 active lanes, mixed phases and lane refill verified\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
