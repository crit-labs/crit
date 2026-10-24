#include "speculative.h"
#include "speculative_attention_fixture.h"
#include <crit/core.h>
#include <array>
#include <iostream>
#include <random>

using namespace crit::techniques;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class Function> void rejects(Function function) {
  try { function(); } catch (const std::exception&) { return; }
  throw std::runtime_error("Expected speculative rejection");
}
struct Faults {
  int live = 0;
  bool fork = false, verify = false, rewind = false, repair = false, bad_count = false,
       bad_endpoint = false, bad_history = false, null_fork = false, alias_fork = false, bad_fork_logits = false, bad_rewind_logits = false;
  std::atomic_bool* cancel_verify = nullptr;
  std::atomic_bool* cancel_repair = nullptr;
};
class Cycle final : public GreedyTargetState {
  std::vector<SpecToken> tokens_;
  std::shared_ptr<Faults> faults_;
  bool stale_logits_ = false;
public:
  Cycle(std::span<const SpecToken> prompt, std::shared_ptr<Faults> faults)
      : tokens_(prompt.begin(), prompt.end()), faults_(std::move(faults)) { ++faults_->live; }
  Cycle(const Cycle& other) : tokens_(other.tokens_), faults_(other.faults_), stale_logits_(other.stale_logits_) { ++faults_->live; }
  ~Cycle() override { --faults_->live; }
  std::unique_ptr<GreedyTargetState> fork() const override {
    if (faults_->fork) throw std::bad_alloc();
    if (faults_->null_fork) return {};
    if (faults_->alias_fork) return std::unique_ptr<GreedyTargetState>(const_cast<Cycle*>(this));
    auto result = std::make_unique<Cycle>(*this);
    result->stale_logits_ = faults_->bad_fork_logits;
    return result;
  }
  std::span<const SpecToken> tokens() const noexcept override { return tokens_; }
  SpecToken next_token() const override { return (tokens_.back() + (stale_logits_ ? 2 : 1)) % 5; }
  void append(SpecToken token) override {
    tokens_.push_back(token);
    if (faults_->cancel_repair) *faults_->cancel_repair = true;
    if (faults_->repair) throw std::bad_alloc();
  }
  void rewind(std::size_t length) override {
    if (!length || length > tokens_.size()) throw std::out_of_range("Cycle rewind");
    tokens_.resize(length);
    stale_logits_ = faults_->bad_rewind_logits;
    if (faults_->rewind) throw std::runtime_error("Failed after rewind mutation");
  }
  std::vector<SpecToken> verify(std::span<const SpecToken> drafts) override {
    std::vector<SpecToken> result{next_token()};
    for (auto draft : drafts) { tokens_.push_back(draft); result.push_back(next_token()); }
    if (faults_->cancel_verify) *faults_->cancel_verify = true;
    if (faults_->verify) throw std::runtime_error("Failed after verification mutation");
    if (faults_->bad_count) result.pop_back();
    if (faults_->bad_endpoint) result.front() = (result.front() + 1) % 5;
    if (faults_->bad_history) tokens_.front() = (tokens_.front() + 1) % 5;
    return result;
  }
};
std::vector<SpecToken> perfect(std::span<const SpecToken> history, std::size_t limit) {
  std::vector<SpecToken> result;
  auto token = history.back();
  while (result.size() < limit) { token = (token + 1) % 5; result.push_back(token); }
  return result;
}
void transactions() {
  const std::array<SpecToken, 2> prompt{0, 1};
  auto faults = std::make_shared<Faults>();
  GreedySpeculator loop(std::make_unique<Cycle>(prompt, faults), {32, 6, 5});
  auto good = perfect;
  auto wrong = [](auto, std::size_t) { return std::vector<SpecToken>{2, 3, 0, 1}; };
  auto round = loop.prepare(10, 4, wrong);
  require(round.accepted_drafts == 2 && std::vector<SpecToken>(round.tokens.begin(), round.tokens.end()) == std::vector<SpecToken>({2, 3, 4}), "Mismatch correction differs");
  require(loop.history().size() == 2 && faults->live == 2, "Prepare changed committed state or lost branch ownership");
  rejects([&] { loop.prepare(10, 2, good); });
  rejects([&] { loop.commit(round.id + 1, 1); });
  rejects([&] { loop.commit(round.id, 4); });
  auto committed = loop.commit(round.id, round.tokens.size());
  require(committed.accepted_drafts == 2 && committed.correction_or_bonus && !loop.pending() && faults->live == 1, "Correction was not materialized before publication");
  rejects([&] { loop.commit(round.id, 1); });
  round = loop.prepare(8, 3, good);
  require(round.accepted_drafts == 3 && round.tokens.size() == 4 && round.tokens.front() == 0, "Repair did not restore next prediction");
  committed = loop.commit(round.id, 2);
  require(committed.tokens == std::vector<SpecToken>({0, 1}) && !committed.correction_or_bonus && loop.history().size() == 7, "Partial acceptance kept rejected tail/bonus");
  round = loop.prepare(2, 0, good);
  require(round.proposed == 0 && round.tokens.size() == 1 && round.tokens[0] == 2, "Zero draft budget is not ordinary decode");
  loop.commit(round.id, 0);
  require(loop.history().size() == 7 && faults->live == 1, "Zero acceptance changed state");
  round = loop.prepare(2, 2, good); loop.discard(round.id);
  rejects([&] { loop.discard(round.id); });
  round = loop.prepare(2, 2, good);
  require(loop.commit(round.id, 2).tokens == std::vector<SpecToken>({2, 3}), "Output budget allowed a bonus past its end");

  GreedySpeculator stop_loop(std::make_unique<Cycle>(prompt, faults), {32, 6, 5}, std::array<SpecToken, 1>{3});
  round = stop_loop.prepare(20, 6, good);
  require(round.proposed == 2 && round.tokens.size() == 2, "Draft continued past stop");
  require(stop_loop.commit(round.id, 2).stopped && stop_loop.finished(), "Accepted stop did not finish");
  rejects([&] { stop_loop.prepare(2, 1, good); });
  GreedySpeculator bounded(std::make_unique<Cycle>(prompt, faults), {3, 6, 5});
  round = bounded.prepare(20, 6, good);
  require(round.tokens.size() == 1 && round.proposed == 1, "Context allowance not enforced");
  bounded.commit(round.id, 1); require(bounded.finished(), "Context end did not finish");
}
void failures() {
  const std::array<SpecToken, 2> prompt{0, 1};
  auto faults = std::make_shared<Faults>();
  GreedySpeculator loop(std::make_unique<Cycle>(prompt, faults), {32, 6, 5});
  auto good = perfect;
  auto unchanged = [&] {
    require(!loop.pending() && faults->live == 1 && std::equal(loop.history().begin(), loop.history().end(), prompt.begin(), prompt.end()), "Failure leaked branch or committed partial state");
  };
  for (auto flag : {&Faults::fork, &Faults::verify, &Faults::bad_count, &Faults::bad_endpoint,
                    &Faults::bad_history, &Faults::null_fork, &Faults::alias_fork, &Faults::bad_fork_logits}) {
    faults.get()->*flag = true;
    rejects([&] { loop.prepare(6, 3, good); }); unchanged();
    faults.get()->*flag = false;
  }
  auto bad_token = [](auto, auto) { return std::vector<SpecToken>{-1}; };
  auto too_many = [](auto, auto) { return std::vector<SpecToken>(8, 1); };
  auto no_memory = [](auto, auto) -> std::vector<SpecToken> { throw std::bad_alloc(); };
  rejects([&] { loop.prepare(6, 3, bad_token); }); unchanged();
  rejects([&] { loop.prepare(6, 3, too_many); }); unchanged();
  rejects([&] { loop.prepare(6, 3, no_memory); }); unchanged();
  for (auto flag : {&Faults::rewind, &Faults::repair, &Faults::bad_rewind_logits}) {
    const auto round = loop.prepare(6, 3, good);
    faults.get()->*flag = true;
    rejects([&] { loop.commit(round.id, round.tokens.size()); }); unchanged();
    rejects([&] { loop.commit(round.id, 1); });
    faults.get()->*flag = false;
  }
  std::atomic_bool cancelled{true};
  rejects([&] { loop.prepare(6, 3, good, &cancelled); }); unchanged();
  cancelled = false; faults->cancel_verify = &cancelled;
  rejects([&] { loop.prepare(6, 3, good, &cancelled); }); unchanged();
  faults->cancel_verify = nullptr; cancelled = false;
  auto round = loop.prepare(6, 3, good); cancelled = true;
  rejects([&] { loop.commit(round.id, 4, &cancelled); }); unchanged();
  cancelled = false; round = loop.prepare(6, 3, good); faults->cancel_repair = &cancelled;
  rejects([&] { loop.commit(round.id, 4, &cancelled); }); unchanged();
  faults->cancel_repair = nullptr;
  auto reentrant = [&](auto history, auto limit) {
    rejects([&] { loop.prepare(6, 3, good); }); return perfect(history, limit);
  };
  round = loop.prepare(6, 3, reentrant); loop.commit(round.id, 4);
  require(loop.history().back() == 0, "Successful retry after failures differs");
}
void broken_fork() {
  struct Shared final : GreedyTargetState {
    std::shared_ptr<std::vector<SpecToken>> history = std::make_shared<std::vector<SpecToken>>(std::initializer_list<SpecToken>{0, 1});
    std::unique_ptr<GreedyTargetState> fork() const override { return std::make_unique<Shared>(*this); }
    std::span<const SpecToken> tokens() const noexcept override { return *history; }
    SpecToken next_token() const override { return (history->back() + 1) % 5; }
    void append(SpecToken token) override { history->push_back(token); }
    void rewind(std::size_t length) override { history->resize(length); }
  };
  GreedySpeculator loop(std::make_unique<Shared>(), {16, 4, 5});
  auto propose = perfect;
  rejects([&] { loop.prepare(8, 4, propose); });
  require(loop.poisoned() && !loop.pending() && loop.history().size() == 2, "Mutable shared fork was not poisoned before publication");
  rejects([&] { loop.prepare(8, 0, propose); });
}
void copy_proposals() {
  PromptCopyProposer copy(2, 32);
  const std::array<SpecToken, 6> history{1, 2, 3, 4, 1, 2};
  require(copy(history, 3) == std::vector<SpecToken>({3, 4, 1}), "Known continuation not proposed");
  require(copy(history, 0).empty() && copy(std::array<SpecToken, 2>{1, 2}, 3).empty(), "Empty proposal bounds differ");
  PromptCopyProposer bounded(2, 1);
  require(bounded(history, 3).empty(), "Copy scanned outside lookback bound");
  rejects([] { PromptCopyProposer invalid(0, 1); });
  rejects([] { PromptCopyProposer invalid(1, 0); });
  const std::array<SpecToken, 8> repeated{0, 1, 2, 3, 4, 0, 1, 2};
  auto faults = std::make_shared<Faults>();
  GreedySpeculator loop(std::make_unique<Cycle>(repeated, faults), {64, 8, 5});
  std::size_t count = 0, accepted_drafts = 0, rounds = 0;
  while (count < 32) {
    const auto round = loop.prepare(32 - count, 8, copy);
    const auto committed = loop.commit(round.id, round.tokens.size());
    for (auto token : committed.tokens) require(token == static_cast<SpecToken>((count++ + 3) % 5), "Copy loop diverged from greedy baseline");
    accepted_drafts += committed.accepted_drafts; ++rounds;
  }
  require(accepted_drafts > 0 && rounds < count, "Loop never accepted copied drafts");
}
void accepted_then_zero_draft() {
  // Interleave independent rows with different accepted frontiers around a KV
  // page boundary. Empty proposals and an explicit zero budget must both start
  // from the last COMMITTED state, including after partial draft acceptance.
  for (std::size_t prefix : {std::size_t{3}, std::size_t{4}, std::size_t{5}}) {
    auto weights = std::make_shared<AttentionWeights>();
    std::array<std::unique_ptr<AttentionTarget>, 2> ordinary;
    std::array<std::unique_ptr<GreedySpeculator>, 2> loops;
    for (std::size_t row = 0; row < loops.size(); ++row) {
      std::vector<SpecToken> prompt(prefix + row);
      for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = static_cast<SpecToken>((i + row) % 5);
      ordinary[row] = std::make_unique<AttentionTarget>(weights, prompt);
      loops[row] = std::make_unique<GreedySpeculator>(
          std::make_unique<AttentionTarget>(weights, prompt), SpeculationLimits{64, 4, 5});
    }
    for (std::size_t iteration = 0; iteration < 4; ++iteration) {
      for (std::size_t row : {std::size_t{1}, std::size_t{0}}) {
        auto& baseline = *ordinary[row];
        auto& loop = *loops[row];
        auto oracle_proposer = [&](auto, std::size_t limit) {
          auto branch = baseline.fork();
          std::vector<SpecToken> drafts;
          while (drafts.size() < limit) {
            drafts.push_back(branch->next_token()); branch->append(drafts.back());
          }
          return drafts;
        };
        auto check = [&](const SpeculativeCommit& result) {
          for (auto token : result.tokens) {
            require(token == baseline.next_token(), "Zero-draft transition resumed from a stale frontier");
            baseline.append(token);
          }
          require(std::equal(loop.history().begin(), loop.history().end(), baseline.tokens().begin(), baseline.tokens().end()),
                  "Interleaved speculative row history differs from its sequential target");
        };
        auto round = loop.prepare(8, 4, oracle_proposer);
        require(round.accepted_drafts == 4, "Transition fixture did not accept real drafts");
        // Row zero commits a partial accepted path; row one includes the bonus.
        check(loop.commit(round.id, row == 0 ? 2 : 5));
        auto empty = [](auto, std::size_t) { return std::vector<SpecToken>{}; };
        round = loop.prepare(2, iteration % 2 ? 4 : 0, empty);
        require(round.proposed == 0 && round.accepted_drafts == 0 && round.tokens.size() == 1,
                "Empty draft row was not ordinary decode");
        check(loop.commit(round.id, 1));
      }
    }
  }
}
void attention() {
  accepted_then_zero_draft();
  std::mt19937 rng(88193);
  std::size_t accepted_total = 0, correction_total = 0;
  for (int trial = 0; trial < 100; ++trial) {
    auto weights = std::make_shared<AttentionWeights>();
    std::vector<SpecToken> prompt(1 + rng() % 8);
    for (auto& token : prompt) token = static_cast<SpecToken>(rng() % 5);
    AttentionTarget ordinary(weights, prompt);
    auto state = std::make_unique<AttentionTarget>(weights, prompt);
    auto* original = state.get();
    const auto retained_prefix = original->view();
    GreedySpeculator loop(std::move(state), {64, 8, 5});
    auto propose = [&](auto history, std::size_t limit) {
      if (trial % 3 == 0) return PromptCopyProposer(1, 32)(history, limit);
      if (trial % 3 == 1) {
        auto branch = ordinary.fork();
        std::vector<SpecToken> result;
        while (result.size() < limit) { const auto token = branch->next_token(); result.push_back(token); branch->append(token); }
        return result; // Oracle proposer tests all-accepted branches, not speed.
      }
      std::vector<SpecToken> result(limit);
      for (auto& token : result) token = static_cast<SpecToken>(rng() % 5);
      return result;
    };
    std::size_t produced = 0;
    while (produced < 24) {
      const auto round = loop.prepare(24 - produced, 1 + rng() % 8, propose);
      const auto accept = std::min(round.tokens.size(), std::size_t{1} + rng() % 8);
      const auto committed = loop.commit(round.id, accept);
      for (auto token : committed.tokens) {
        require(token == ordinary.next_token(), "Paged speculative output differs from ordinary greedy target");
        ordinary.append(token); ++produced;
      }
      require(std::equal(loop.history().begin(), loop.history().end(), ordinary.tokens().begin(), ordinary.tokens().end()), "Materialized continuation history differs");
      accepted_total += committed.accepted_drafts; correction_total += committed.correction_or_bonus ? 1 : 0;
    }
    require(retained_prefix.rows() == prompt.size(), "Speculative branch changed a pinned prefix");
  }
  require(accepted_total > 0 && correction_total > 0, "Attention test omitted acceptance or repair");
  std::cout << "100 paged-attention sequences, 2400 ordinary-greedy-equivalent outputs; accepted drafts="
            << accepted_total << ", correction/bonus commits=" << correction_total << '\n';
  auto weights = std::make_shared<AttentionWeights>();
  GreedySpeculator loop(std::make_unique<AttentionTarget>(weights, std::array<SpecToken, 1>{1}), {64, 4, 5});
  // Fill the arena after construction: the first speculative COW tail cannot
  // reserve a page. Parent state must still admit a round after pressure clears.
  auto& kv = weights->kv;
  const auto pressure = kv.open();
  std::vector<float> rows(kv.free_pages() * 4 * 4, 0);
  kv.append(pressure, rows, rows);
  auto propose = [](auto, std::size_t limit) { return std::vector<SpecToken>(limit, 1); };
  rejects([&] { loop.prepare(4, 4, propose); });
  require(!loop.pending() && loop.history().size() == 1 && kv.free_pages() == 0, "Failed speculative admission changed parent or arena ownership");
  kv.cache.close(pressure); kv.settle();
  const auto round = loop.prepare(4, 4, propose);
  require(!loop.commit(round.id, round.tokens.size()).tokens.empty(), "Speculation failed after page pressure cleared");
}

namespace core_fixture {
using namespace crit::core;
struct Host final : Backend { void bind_thread() override {} void drain() override {} };
struct Turn final : Session {
  std::shared_ptr<AttentionWeights> weights;
  std::vector<Token> prompt;
  Reservation memory;
  std::unique_ptr<GreedySpeculator> loop;
  PromptCopyProposer proposer{1, 32};
  std::uint64_t pending_id = 0;
  Turn(std::shared_ptr<AttentionWeights> model, std::span<const Token> input, Resources& resources)
      : weights(std::move(model)), prompt(input.begin(), input.end()), memory(resources.reserve(Memory::host, 65536)) {}
  Progress advance(WorkBudget budget) override {
    if (!loop) {
      require(budget.prefill_tokens == prompt.size(), "Fixture expects one bounded prefill chunk");
      loop = std::make_unique<GreedySpeculator>(std::make_unique<AttentionTarget>(weights, prompt), SpeculationLimits{64, 8, 5});
      return {budget.prefill_tokens, true, {}, nullptr};
    }
    const auto round = loop->prepare(budget.output_tokens, 8, proposer); pending_id = round.id;
    return {0, true, round.tokens, nullptr};
  }
  void commit(std::uint32_t accepted, bool) override {
    if (pending_id) { loop->commit(pending_id, accepted); pending_id = 0; }
  }
  void close(Finish) noexcept override { loop.reset(); }
};
struct ModelAdapter final : Model {
  std::shared_ptr<AttentionWeights> weights;
  Resources& resources;
  Reservation arena;
  ModelAdapter(std::shared_ptr<AttentionWeights> model, Resources& r)
      : weights(std::move(model)), resources(r), arena(r.reserve(Memory::host, weights->kv.pages.reserved_bytes() + 65536)) {}
  ModelInfo info() const override { return {"native-speculative-attention-fixture", 5, 64, 64, {}}; }
  std::unique_ptr<Session> open(std::span<const Token> prompt) override { return std::make_unique<Turn>(weights, prompt, resources); }
  std::vector<Token> tokenize(std::string_view) override { return {0, 1}; }
  std::string detokenize(std::span<const Token>) override { return "synthetic"; }
};
void run() {
  auto weights = std::make_shared<AttentionWeights>();
  Profile profile; profile.context = profile.prefill_chunk = 64; profile.host_limit = 1 << 20; profile.device_limit = 1;
  Engine engine(profile, std::make_shared<Host>(), [&](const Profile&, Services services) { return std::make_unique<ModelAdapter>(weights, services.resources); });
  for (const auto& prompt : {std::vector<Token>{0, 1}, std::vector<Token>{4, 3, 2, 1, 0}}) {
    std::vector<Token> expected;
    {
      AttentionTarget target(weights, prompt);
      for (int i = 0; i < 20; ++i) { expected.push_back(target.next_token()); target.append(expected.back()); }
    }
    const auto result = engine.submit(prompt, 20).wait();
    require(result.tokens == expected && result.finish == Finish::length, "Core publication differs from ordinary native target");
    require(weights->kv.cache.stats().live_pages == 0, "Core retirement retained speculative KV");
    RequestOptions options; options.stop_tokens = std::vector<Token>{expected[0]};
    const auto stopped = engine.submit(prompt, 20, options).wait();
    require(stopped.tokens == std::vector<Token>{expected[0]} && stopped.finish == Finish::stop, "Core accepted prefix did not truncate speculative stop");
    require(weights->kv.cache.stats().live_pages == 0, "Stopped request retained speculative KV");
  }
  require(engine.usage(Memory::host).live == weights->kv.pages.reserved_bytes() + 65536, "Session scratch reservation survived retirement");
  weights->fail_repair = true;
  std::vector<Token> published;
  rejects([&] { engine.submit({0, 1}, 4).wait([&](Token token) { published.push_back(token); }); });
  require(published.empty() && weights->kv.cache.stats().live_pages == 0,
          "Core published failed repaired output or leaked its KV");
}
} // namespace core_fixture
// Greedy output under every way of proposing drafts equals plain greedy decoding token for token: no speculation
// (draft budget 0), prompt copying (windows 1 and 3), a self-draft that predicts exactly what the target will
// (MTP-like, every draft accepted), a separate stateful drafter model (its own sequence, reconciled with the committed
// history before each proposal, agreeing only sometimes), a drafter that is always wrong, and random drafts; each with
// and without a stop token, over draft budgets 1 .. 8. Methods with mixed agreement must both accept and correct.
void matrix() {
  enum class Method { none, copy1, copy3, self, drafter, wrong, random };
  const std::array<std::pair<Method, const char*>, 7> methods{{{Method::none, "none"}, {Method::copy1, "copy window 1"},
      {Method::copy3, "copy window 3"}, {Method::self, "self draft"}, {Method::drafter, "drafter model"},
      {Method::wrong, "always wrong"}, {Method::random, "random"}}};
  for (const auto& entry : methods) {
    const Method method = entry.first;
    const char* const name = entry.second;
    std::mt19937 prompts(4242), rng(977);  // the same prompts and stops for every method
    std::size_t produced_total = 0, accepted_total = 0, corrections = 0, proposed_total = 0, drafter_rewinds = 0;
    for (int trial = 0; trial < 24; ++trial) {
      auto weights = std::make_shared<AttentionWeights>();
      std::vector<SpecToken> prompt(2 + prompts() % 7);
      for (auto& token : prompt) token = static_cast<SpecToken>(prompts() % 5);
      // Plain greedy reference, truncated after the first stop token when one is set.
      std::vector<SpecToken> expected;
      {
        AttentionTarget plain(weights, prompt);
        for (int i = 0; i < 24; ++i) { expected.push_back(plain.next_token()); plain.append(expected.back()); }
      }
      std::vector<SpecToken> stops;
      if (trial % 2) {
        stops.push_back(expected[std::min<std::size_t>(expected.size() - 1, 3 + prompts() % 12)]);
        expected.resize(static_cast<std::size_t>(std::find(expected.begin(), expected.end(), stops[0]) - expected.begin()) + 1);
      }
      GreedySpeculator loop(std::make_unique<AttentionTarget>(weights, prompt), {64, 8, 5}, stops);
      AttentionTarget oracle(weights, prompt);  // the committed history's own continuation, for self and wrong drafts
      // The drafter model sees the history without its first token, so it agrees with the target only sometimes.
      std::unique_ptr<AttentionTarget> drafter;
      auto propose = [&](std::span<const SpecToken> history, std::size_t limit) {
        std::vector<SpecToken> result;
        switch (method) {
        case Method::none: break;
        case Method::copy1: result = PromptCopyProposer(1, 32)(history, limit); break;
        case Method::copy3: result = PromptCopyProposer(3, 32)(history, limit); break;
        case Method::self: case Method::wrong: {
          auto branch = oracle.fork();
          while (result.size() < limit) {
            const auto token = branch->next_token();
            result.push_back(method == Method::self ? token : static_cast<SpecToken>((token + 1) % 5));
            branch->append(token);
          }
          break;
        }
        case Method::drafter: {
          const auto view = history.subspan(1);
          if (!drafter) drafter = std::make_unique<AttentionTarget>(weights, view);
          const auto have = drafter->tokens();
          const auto common = static_cast<std::size_t>(std::mismatch(have.begin(), have.end(), view.begin(), view.end()).first - have.begin());
          // The drafter keeps its own drafts in its sequence, as a draft model's KV does; a rejected suffix differs from
          // the committed history and is rewound here before the accepted tokens are appended.
          if (common == 0) drafter = std::make_unique<AttentionTarget>(weights, view);
          else {
            if (common < have.size()) { drafter->rewind(common); ++drafter_rewinds; }
            for (auto token : view.subspan(common)) drafter->append(token);
          }
          while (result.size() < limit) { const auto token = drafter->next_token(); result.push_back(token); drafter->append(token); }
          break;
        }
        case Method::random:
          result.resize(limit);
          for (auto& token : result) token = static_cast<SpecToken>(rng() % 5);
          break;
        }
        return result;
      };
      std::vector<SpecToken> got;
      while (got.size() < expected.size() && !loop.finished()) {
        const auto budget = method == Method::none ? std::size_t{0} : std::size_t{1} + rng() % 8;
        const auto round = loop.prepare(24 - got.size(), budget, propose);
        const auto committed = loop.commit(round.id, round.tokens.size());
        for (auto token : committed.tokens) { got.push_back(token); oracle.append(token); }
        accepted_total += committed.accepted_drafts; proposed_total += committed.proposed;
        corrections += committed.accepted_drafts < committed.proposed ? 1 : 0;
      }
      if (got != expected) throw std::runtime_error(std::string(name) + ": speculative output differs from plain greedy decoding");
      if (!stops.empty() && !loop.finished()) throw std::runtime_error(std::string(name) + ": the stop token did not finish the loop");
      produced_total += got.size();
    }
    if (method == Method::none) require(proposed_total == 0, "draft budget 0 proposed drafts");
    if (method == Method::self) require(accepted_total == proposed_total && accepted_total > 0, "self drafts were rejected");
    if (method == Method::wrong) require(accepted_total == 0 && proposed_total > 0, "wrong drafts were accepted");
    if (method == Method::drafter) require(drafter_rewinds > 0, "the drafter model never rewound a rejected draft");
    if (method == Method::drafter || method == Method::random || method == Method::copy1 || method == Method::copy3)
      require(accepted_total > 0 && corrections > 0, "a mixed-agreement method did not both accept and correct");
    std::cout << name << ": " << produced_total << " tokens equal plain greedy decoding (" << accepted_total << " of "
              << proposed_total << " drafts accepted)\n";
  }
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("Choose transactions, failures, copy, attention, matrix or core");
    const std::string mode = argv[1];
    if (mode == "transactions") transactions();
    else if (mode == "failures") { failures(); broken_fork(); }
    else if (mode == "copy") copy_proposals();
    else if (mode == "attention") attention();
    else if (mode == "matrix") matrix();
    else if (mode == "core") core_fixture::run();
    else throw std::invalid_argument("Unknown speculative fixture");
    std::cout << "Native greedy speculative " << mode << " passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
