// Device speculative round fixtures: greedy accept against GreedySpeculator,
// graph replay against eager execution, one readback per round, RoundBudget.
#include "device_round.h"
#include "device_round_fixture.h"
#include "round_budget.h"
#include "speculative.h"
#include <crit/cuda.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace crit::core;
namespace dr = crit::techniques::device_round;
namespace fx = crit::tests::device_round;

#ifdef CRIT_TEST_LINKER_WRAP
namespace {
std::atomic<std::uint64_t> d2h_copies{0}, host_syncs{0};
}
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* d, const void* s, std::size_t n, cudaMemcpyKind k, cudaStream_t st) {
    if(k == cudaMemcpyDeviceToHost) ++d2h_copies;
    return __real_cudaMemcpyAsync(d, s, n, k, st);
}
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t s) { ++host_syncs; return __real_cudaStreamSynchronize(s); }
extern "C" cudaError_t __real_cudaEventSynchronize(cudaEvent_t);
extern "C" cudaError_t __wrap_cudaEventSynchronize(cudaEvent_t e) { ++host_syncs; return __real_cudaEventSynchronize(e); }
extern "C" cudaError_t __real_cudaDeviceSynchronize();
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() { ++host_syncs; return __real_cudaDeviceSynchronize(); }
#endif

namespace {
void check(cudaError_t s, const char* what = "CUDA") { if(s != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s)); }
void require(bool c, const std::string& message) { if(!c) throw std::runtime_error(message); }
template<class F> void rejects(F f, const std::string& text) {
    try { f(); } catch(const std::exception& e) {
        require(std::string(e.what()).find(text) != std::string::npos, std::string("unexpected rejection: ") + e.what());
        return;
    }
    throw std::runtime_error("expected rejection did not occur: " + text);
}
struct Fixture {
    Resources resources{1 << 24, 1 << 26};
    std::shared_ptr<CudaBackend> backend = std::make_shared<CudaBackend>();
    Fixture() { backend->initialize(resources); }
};
template<class T> std::vector<T> download(const CudaBuffer& b, std::size_t count) {
    std::vector<T> out(count);
    b.download(std::as_writable_bytes(std::span(out)), 0, Queue::compute);
    return out;
}
template<class T> void upload(CudaBuffer& b, const std::vector<T>& values) {
    b.upload(std::as_bytes(std::span(values)), 0, Queue::compute);
}

// Host target whose greedy prediction is a hash of the whole history.
class HashHistoryTarget final : public crit::techniques::GreedyTargetState {
    std::vector<crit::techniques::SpecToken> tokens_;
    std::uint32_t vocabulary_;
public:
    HashHistoryTarget(std::vector<crit::techniques::SpecToken> t, std::uint32_t v) : tokens_(std::move(t)), vocabulary_(v) {}
    std::unique_ptr<GreedyTargetState> fork() const override { return std::make_unique<HashHistoryTarget>(tokens_, vocabulary_); }
    std::span<const crit::techniques::SpecToken> tokens() const noexcept override { return tokens_; }
    crit::techniques::SpecToken next_token() const override {
        std::uint32_t s = 17;
        for(auto t : tokens_) s = fx::step_state_host(s, t);
        return crit::techniques::SpecToken(fx::mix(s) % vocabulary_);
    }
    void append(crit::techniques::SpecToken t) override { tokens_.push_back(t); }
    void rewind(std::size_t n) override { tokens_.resize(n); }
};

// Randomized rows through the batched accept kernel, each compared with the
// tokens a host GreedySpeculator prepares for the same drafts and target.
void accept() {
    Fixture f;
    constexpr std::uint32_t vocabulary = 6, rows = 4000, max_k = 4;
    std::mt19937 random(1234);
    std::vector<std::int32_t> stops = {2};
    struct Case { std::uint32_t k, budget; std::vector<std::int32_t> drafts, predictions, expected; std::size_t matched; };
    std::vector<Case> cases;
    std::size_t all_match = 0, first_mismatch = 0, stopped = 0, budget_cut = 0;
    std::vector<std::uint32_t> per_k(max_k + 1, 0);
    for(std::uint32_t r = 0; r < rows; ++r) {
        Case c;
        c.k = 1 + random() % max_k;
        c.budget = random() % (c.k + 3); // includes 0 and budgets above K+1
        std::vector<crit::techniques::SpecToken> prompt(1 + random() % 5);
        for(auto& t : prompt) t = std::int32_t(random() % vocabulary);
        HashHistoryTarget scratch(prompt, vocabulary);
        const double match = std::uniform_real_distribution<double>(0.2, 1.0)(random);
        for(std::uint32_t i = 0; i < c.k; ++i) {
            const auto p = scratch.next_token();
            c.predictions.push_back(p);
            const auto d = std::uniform_real_distribution<double>(0, 1)(random) < match ? p : std::int32_t(random() % vocabulary);
            c.drafts.push_back(d);
            scratch.append(d);
        }
        c.predictions.push_back(scratch.next_token());
        if(c.budget) {
            crit::techniques::GreedySpeculator host(std::make_unique<HashHistoryTarget>(prompt, vocabulary),
                {1024, c.k, vocabulary}, stops);
            auto propose = [&](std::span<const crit::techniques::SpecToken>, std::size_t limit) {
                return std::vector<crit::techniques::SpecToken>(c.drafts.begin(), c.drafts.begin() + std::ptrdiff_t(limit));
            };
            const auto round = host.prepare(c.budget, c.k, propose);
            c.expected.assign(round.tokens.begin(), round.tokens.end());
            c.matched = round.accepted_drafts;
            host.discard(round.id);
        }
        ++per_k[c.k];
        if(c.matched == c.k && c.budget > c.k) ++all_match;
        if(c.matched == 0 && c.drafts[0] != c.predictions[0] && c.budget) ++first_mismatch;
        if(std::find(c.expected.begin(), c.expected.end(), stops[0]) != c.expected.end()) ++stopped;
        if(c.budget && c.expected.size() == c.budget && c.budget <= c.k) ++budget_cut;
        cases.push_back(std::move(c));
    }
    require(all_match > 50 && first_mismatch > 50 && stopped > 50 && budget_cut > 50, "randomized categories not covered");
    for(std::uint32_t k = 1; k <= max_k; ++k) require(per_k[k] > 100, "draft width not covered");
    // Rows share the widest layout: K = max_k; narrower cases pad with drafts that
    // can never be reached because the budget bounds them (budget <= k + 2).
    // Use per-K launches instead so each case runs at its own width.
    for(std::uint32_t k = 1; k <= max_k; ++k) {
        std::vector<const Case*> selected;
        for(const auto& c : cases) if(c.k == k) selected.push_back(&c);
        const auto n = std::uint32_t(selected.size());
        std::vector<std::int32_t> drafts, predictions, control(n * dr::control::fields, 0);
        for(std::uint32_t i = 0; i < n; ++i) {
            drafts.insert(drafts.end(), selected[i]->drafts.begin(), selected[i]->drafts.end());
            predictions.insert(predictions.end(), selected[i]->predictions.begin(), selected[i]->predictions.end());
            control[i * dr::control::fields + dr::control::frontier] = std::int32_t(10 + i);
            control[i * dr::control::fields + dr::control::last_token] = std::int32_t(i % vocabulary);
            control[i * dr::control::fields + dr::control::output_budget] = std::int32_t(selected[i]->budget);
        }
        auto d_drafts = f.backend->allocate(Memory::device, drafts.size() * 4), d_pred = f.backend->allocate(Memory::device, predictions.size() * 4);
        auto d_control = f.backend->allocate(Memory::device, control.size() * 4), d_result = f.backend->allocate(Memory::device, n * dr::result_width(k) * 4);
        auto d_stops = f.backend->allocate(Memory::device, 4);
        upload(d_drafts, drafts); upload(d_pred, predictions); upload(d_control, control); upload(d_stops, stops);
        dr::AcceptRows rowsv{static_cast<const std::int32_t*>(d_drafts.data()), static_cast<const std::int32_t*>(d_pred.data()),
                             static_cast<const std::int32_t*>(d_stops.data()), 1, static_cast<std::int32_t*>(d_control.data()),
                             static_cast<std::int32_t*>(d_result.data()), n, k};
        check(dr::enqueue_greedy_accept(rowsv, f.backend->stream()), "accept");
        auto result = download<std::int32_t>(d_result, n * dr::result_width(k));
        auto after = download<std::int32_t>(d_control, control.size());
        for(std::uint32_t i = 0; i < n; ++i) {
            const auto& c = *selected[i];
            const auto* row = result.data() + i * dr::result_width(k);
            const auto* ctl = after.data() + i * dr::control::fields;
            require(row[0] == std::int32_t(c.expected.size()), "accept count differs from GreedySpeculator");
            require(std::equal(c.expected.begin(), c.expected.end(), row + 1), "accepted tokens differ from GreedySpeculator");
            for(auto j = c.expected.size(); j < k + 1; ++j) require(row[1 + j] == -1, "unused result slot not cleared");
            if(c.budget) require(ctl[dr::control::n_matched] == std::int32_t(c.matched), "accepted draft count differs");
            require(ctl[dr::control::frontier] == std::int32_t(10 + i + c.expected.size()), "device frontier not advanced by n_emitted");
            require(ctl[dr::control::base_frontier] == std::int32_t(10 + i), "round base frontier");
            const auto last = c.expected.empty() ? std::int32_t(i % vocabulary) : c.expected.back();
            require(ctl[dr::control::last_token] == last, "pending token differs");
            require(ctl[dr::control::commit_tokens] == row[0], "history count differs from the emitted count");
        }
        // The readback block: the packed result, n_matched and the frontier, then each emitted column's
        // log-probability rows, -1 / -inf beyond the count; and the block without log-probabilities.
        {
            constexpr std::uint32_t top = 3;
            std::vector<float> lp(n * (k + 1)), tv(n * (k + 1) * top);
            std::vector<std::int32_t> ti(n * (k + 1) * top);
            std::uniform_real_distribution<float> value(-9.f, 0.f);
            for(auto& x : lp) x = value(random);
            for(auto& x : tv) x = value(random);
            for(auto& x : ti) x = std::int32_t(random() % vocabulary);
            auto d_lp = f.backend->allocate(Memory::device, lp.size() * 4), d_tv = f.backend->allocate(Memory::device, tv.size() * 4);
            auto d_ti = f.backend->allocate(Memory::device, ti.size() * 4);
            upload(d_lp, lp); upload(d_tv, tv); upload(d_ti, ti);
            const float minus_inf = -std::numeric_limits<float>::infinity();
            for(const bool with : {false, true}) {
                const dr::PackedLayout layout{k, with ? top : 0u, with};
                auto d_block = f.backend->allocate(Memory::device, n * layout.words() * 4);
                const dr::PackRows pack{static_cast<const std::int32_t*>(d_control.data()), static_cast<const std::int32_t*>(d_result.data()),
                                        static_cast<std::int32_t*>(d_block.data()), with ? static_cast<const float*>(d_lp.data()) : nullptr,
                                        with ? static_cast<const std::int32_t*>(d_ti.data()) : nullptr,
                                        with ? static_cast<const float*>(d_tv.data()) : nullptr, n, k, with ? top : 0u};
                check(dr::enqueue_pack(pack, f.backend->stream()), "pack");
                const auto block = download<std::int32_t>(d_block, n * layout.words());
                for(std::uint32_t i = 0; i < n; ++i) {
                    const auto* b = block.data() + i * layout.words();
                    const auto* row = result.data() + i * dr::result_width(k);
                    const auto* ctl = after.data() + i * dr::control::fields;
                    require(std::equal(row, row + dr::result_width(k), b), "the block's packed result");
                    require(b[layout.matched()] == ctl[dr::control::n_matched] && b[layout.frontier()] == ctl[dr::control::frontier],
                            "the block's matched count and frontier");
                    const auto emitted = std::uint32_t(row[0]);
                    for(std::uint32_t c = 0; with && c <= k; ++c) {
                        float got;
                        std::memcpy(&got, b + layout.logprob() + c, 4);
                        require(got == (c < emitted ? lp[i * (k + 1) + c] : minus_inf), "the block's log-probability");
                        for(std::uint32_t a = 0; a < top; ++a) {
                            const auto at = (std::size_t(i) * (k + 1) + c) * top + a;
                            float alt;
                            std::memcpy(&alt, b + layout.top_logprobs() + c * top + a, 4);
                            require(b[layout.top_ids() + c * top + a] == (c < emitted ? ti[at] : -1) &&
                                    alt == (c < emitted ? tv[at] : minus_inf), "the block's ranked alternatives");
                        }
                    }
                }
            }
        }
        // Core truncation applied on the device: frontier/base/last/count follow m.
        for(std::uint32_t i = 0; i < n; i += 7) {
            const auto emitted = std::uint32_t(selected[i]->expected.size());
            const auto m = emitted ? std::uint32_t(random() % (emitted + 1)) : 0u;
            check(dr::enqueue_truncate(static_cast<std::int32_t*>(d_control.data()), static_cast<std::int32_t*>(d_result.data()), k, i, m, f.backend->stream()));
            const auto ctl = download<std::int32_t>(d_control, control.size());
            const auto res = download<std::int32_t>(d_result, n * dr::result_width(k));
            const auto* c = ctl.data() + i * dr::control::fields;
            require(c[dr::control::frontier] == std::int32_t(10 + i + m) && c[dr::control::n_emitted] == std::int32_t(m) &&
                    res[i * dr::result_width(k)] == std::int32_t(m), "device truncation count/frontier");
            require(c[dr::control::last_token] == (m ? selected[i]->expected[m - 1] : std::int32_t(i % vocabulary)), "device truncation pending token");
            require(c[dr::control::commit_tokens] == std::int32_t(m), "device truncation history count");
        }
        // A seed: the one-column round at frontier - 1 that emitted the prefill's token, nothing for the history.
        check(dr::enqueue_seed(static_cast<std::int32_t*>(d_control.data()), static_cast<std::int32_t*>(d_result.data()), k, 0,
                               40, 3, 5, f.backend->stream()));
        const auto seeded = download<std::int32_t>(d_control, dr::control::fields);
        const auto seeded_result = download<std::int32_t>(d_result, dr::result_width(k));
        require(seeded[dr::control::frontier] == 40 && seeded[dr::control::last_token] == 3 && seeded[dr::control::base_frontier] == 39 &&
                seeded[dr::control::base_last] == 5 && seeded[dr::control::n_emitted] == 1 && seeded[dr::control::n_matched] == 0 &&
                seeded[dr::control::commit_tokens] == 0 && seeded_result[0] == 1 && seeded_result[1] == 3 &&
                std::all_of(seeded_result.begin() + 2, seeded_result.end(), [](std::int32_t t) { return t == -1; }), "seeded round state");
        require(dr::enqueue_seed(static_cast<std::int32_t*>(d_control.data()), static_cast<std::int32_t*>(d_result.data()), k, 0,
                                 0, 3, 5, f.backend->stream()) == cudaErrorInvalidValue, "a seed needs a materialized position");
    }
    std::cout << "accept rows " << rows << " all-match " << all_match << " first-mismatch " << first_mismatch
              << " stop " << stopped << " budget " << budget_cut << '\n';
}

// The control field the sampling history's commit takes its count from (token-sampling's enqueue_commit).
constexpr std::uint32_t history_count = dr::control::commit_tokens;

struct HashOptions {
    std::uint32_t max_stops = 0;
    bool seeded = false, logprobs = false, zero_copy = false;
    std::vector<std::int32_t> replay = {}; // forced: verify's predictions come from this continuation
};
// A synthetic engine bound to one controller. Its verify step first takes the previous round's kept tokens into a
// sampling-history stand-in (log), as a sampled engine's first Forward kernel does. Seeded: the prefill materializes
// the whole prompt and produces the first output itself (in the history at request start), and the round starts as
// a one-column round for a stateful drafter's catch-up.
struct HashEngine {
    CudaBuffer state, start, checkpoints, history, prompt, log, log_count, continuation;
    fx::HashTarget target;
    std::unique_ptr<dr::DeviceSpeculativeRound> round;
    std::int32_t first = -1; // seeded: the prefill's output
    cudaStream_t stream = nullptr;
    HashEngine(Fixture& f, std::uint32_t k, std::uint32_t vocabulary, std::uint32_t context, bool graphs,
               std::vector<std::int32_t> stops, const std::vector<std::int32_t>& prompt_tokens, HashOptions options = {}) {
        const bool seeded = options.seeded;
        state = f.backend->allocate(Memory::device, 4); start = f.backend->allocate(Memory::device, 4);
        checkpoints = f.backend->allocate(Memory::device, (k + 1) * 4); history = f.backend->allocate(Memory::device, context * 4);
        prompt = f.backend->allocate(Memory::device, prompt_tokens.size() * 4);
        log = f.backend->allocate(Memory::device, context * 4); log_count = f.backend->allocate(Memory::device, 4);
        upload(state, std::vector<std::uint32_t>{17});
        upload(prompt, prompt_tokens);
        target = {static_cast<std::uint32_t*>(state.data()), static_cast<std::uint32_t*>(start.data()),
                  static_cast<std::uint32_t*>(checkpoints.data()), static_cast<std::int32_t*>(history.data()), vocabulary, 3, 5};
        auto t = target;
        auto* l = static_cast<std::int32_t*>(log.data());
        auto* lc = static_cast<std::int32_t*>(log_count.data());
        const auto materialized = std::uint32_t(prompt_tokens.size() - (seeded ? 0 : 1));
        continuation = f.backend->allocate(Memory::device, std::max<std::size_t>(options.replay.size(), 1) * 4);
        if(!options.replay.empty()) upload(continuation, options.replay);
        const auto* replay = options.replay.empty() ? nullptr : static_cast<const std::int32_t*>(continuation.data());
        // The first output sits at the start frontier (the pending token's position), so outputs = frontier - base.
        const auto base = std::int32_t(materialized) - (seeded ? 1 : 0), length = std::int32_t(options.replay.size());
        dr::RoundSteps steps{
            [t](cudaStream_t s, const dr::RoundView& v) { check(fx::enqueue_draft(t, v.control, v.drafts, v.draft_tokens, s)); },
            [t, l, lc, vocabulary, replay, base, length](cudaStream_t s, const dr::RoundView& v) {
                check(fx::enqueue_take(v.control + history_count, v.result + 1, l, lc, s));
                check(fx::enqueue_verify(t, v.control, v.drafts, v.predictions, v.draft_tokens, s));
                if(replay) check(fx::enqueue_forced(v.control, replay, base, length, v.predictions, v.draft_tokens, s));
                if(v.logprob)
                    check(fx::enqueue_logprobs(v.predictions, v.draft_tokens, v.logprob, v.top_ids, v.top_logprobs, v.top, vocabulary, s));
            },
            [t](cudaStream_t s, const dr::RoundView& v) { check(fx::enqueue_commit_state(t, v.control, s)); }};
        dr::RoundConfig config{k, vocabulary, context, std::move(stops), graphs};
        config.max_stops = options.max_stops;
        config.logprobs = options.logprobs;
        config.top_logprobs = options.logprobs ? 3 : 0;
        config.zero_copy = options.zero_copy;
        round = std::make_unique<dr::DeviceSpeculativeRound>(f.backend, std::move(config), std::move(steps));
        check(fx::enqueue_prefill(target, static_cast<const std::int32_t*>(prompt.data()), materialized, f.backend->stream()));
        stream = f.backend->stream();
        std::vector<std::int32_t> initial;
        if(seeded) {
            std::uint32_t s = 17;
            for(auto token : prompt_tokens) s = fx::step_state_host(s, token);
            first = std::int32_t(fx::mix(s) % vocabulary);
            initial.push_back(first);
            round->seed(materialized, first, prompt_tokens.back());
        } else {
            round->start(materialized, prompt_tokens.back());
        }
        initial.resize(context, 0);
        upload(log, initial);
        upload(log_count, std::vector<std::int32_t>{seeded ? 1 : 0});
    }
    // The sampling history after the last round's tokens were taken in (the next round's first kernel).
    std::vector<std::int32_t> taken() {
        const auto v = round->view();
        check(fx::enqueue_take(v.control + history_count, v.result + 1, static_cast<std::int32_t*>(log.data()),
                               static_cast<std::int32_t*>(log_count.data()), stream));
        const auto n = download<std::int32_t>(log_count, 1)[0];
        return download<std::int32_t>(log, std::size_t(n));
    }
};
std::vector<std::int32_t> ordinary(const std::vector<std::int32_t>& prompt, std::uint32_t vocabulary, std::size_t count) {
    std::uint32_t s = 17;
    for(std::size_t i = 0; i + 1 < prompt.size(); ++i) s = fx::step_state_host(s, prompt[i]);
    auto last = prompt.back();
    std::vector<std::int32_t> out;
    while(out.size() < count) { s = fx::step_state_host(s, last); last = std::int32_t(fx::mix(s) % vocabulary); out.push_back(last); }
    return out;
}
// Run rounds the way a Core session would: launch, wait readiness, copy, commit.
// widths: the draft width of round i is widths[i % size] (empty: the configured width).
std::vector<std::int32_t> run(HashEngine& e, std::size_t count, std::vector<std::uint32_t>* per_round = nullptr,
                              std::uint32_t truncate_every = 0, const std::vector<std::uint32_t>& widths = {}) {
    std::vector<std::int32_t> out;
    std::uint32_t round = 0;
    while(out.size() < count) {
        const auto allowance = std::uint32_t(count - out.size());
        auto* ready = widths.empty() ? e.round->launch(allowance) : e.round->launch(allowance, widths[round % widths.size()]);
        ready->wait();
        const auto tokens = e.round->tokens();
        auto accepted = std::uint32_t(tokens.size());
        if(truncate_every && (round + 1) % truncate_every == 0 && accepted > 1) accepted = 1 + accepted / 3; // emulate a Core stop cut
        ++round;
        out.insert(out.end(), tokens.begin(), tokens.begin() + accepted);
        if(per_round) per_round->push_back(accepted);
        e.round->commit(accepted);
    }
    return out;
}

// Every round's tokens, accepted count, matched count and log-probability rows, with a Core-style cut every
// `truncate_every` rounds.
struct Trace {
    std::vector<std::int32_t> tokens;
    std::vector<std::uint32_t> accepted, matched;
    std::vector<float> logprobs;
    std::vector<TokenLogprob> top;
    bool operator==(const Trace& o) const {
        return tokens == o.tokens && accepted == o.accepted && matched == o.matched && logprobs == o.logprobs &&
               top.size() == o.top.size() && std::equal(top.begin(), top.end(), o.top.begin(), [](const TokenLogprob& a, const TokenLogprob& b) {
                   return a.token == b.token && a.logprob == b.logprob;
               });
    }
};
Trace record(HashEngine& e, std::size_t count, std::uint32_t truncate_every) {
    Trace t;
    while(t.tokens.size() < count) {
        e.round->launch(std::uint32_t(count - t.tokens.size()))->wait();
        const auto tokens = e.round->tokens();
        auto accepted = std::uint32_t(tokens.size());
        if(truncate_every && (t.accepted.size() + 1) % truncate_every == 0 && accepted > 1) accepted = 1 + accepted / 3;
        t.tokens.insert(t.tokens.end(), tokens.begin(), tokens.begin() + accepted);
        t.accepted.push_back(accepted);
        t.matched.push_back(e.round->matched());
        const auto lp = e.round->logprobs();
        const auto top = e.round->top_logprobs();
        const auto width = lp.empty() ? 0 : top.size() / lp.size();
        t.logprobs.insert(t.logprobs.end(), lp.begin(), lp.begin() + std::ptrdiff_t(lp.empty() ? 0 : tokens.size()));
        t.top.insert(t.top.end(), top.begin(), top.begin() + std::ptrdiff_t(tokens.size() * width));
        e.round->commit(accepted);
    }
    return t;
}

void graphs() {
    Fixture f;
    const std::vector<std::int32_t> prompt = {1, 4, 2, 7, 3};
    constexpr std::uint32_t vocabulary = 11, context = 512, count = 120;
    const auto expected = ordinary(prompt, vocabulary, count);
    for(std::uint32_t k = 0; k <= 4; ++k) {
        HashEngine graphed(f, k, vocabulary, context, true, {}, prompt), eager(f, k, vocabulary, context, false, {}, prompt);
        std::vector<std::uint32_t> a, b;
        const auto g = run(graphed, count, &a), e = run(eager, count, &b);
        require(graphed.round->captured() && !eager.round->captured(), "graph mode selection");
        require(g == e && a == b, "graph replay differs from eager execution");
        require(g == expected, "device rounds differ from ordinary greedy decoding");
        require(download<std::uint32_t>(graphed.state, 1) == download<std::uint32_t>(eager.state, 1), "graph/eager device state differs");
        require(graphed.round->frontier() == prompt.size() - 1 + count, "host frontier mirror");
        // A truncated commit (Core stop cut) is applied to device state, so the
        // continuation still equals ordinary decoding token for token.
        HashEngine cut(f, k, vocabulary, context, true, {}, prompt);
        require(run(cut, count, nullptr, 2) == expected, "device truncation corrupted continuation");
        const auto timings = graphed.round->timings();
        require(timings.size() == 2 && timings[0].stage == DeviceStage::forward && timings[1].stage == DeviceStage::accept,
                "round stage timings");
    }
    // The sampling history takes in exactly the tokens Core kept, with Core truncations between rounds: after start()
    // and after a one-column seed for a stateful drafter, whose token the prefill already put in the history.
    for(const bool seeded : {false, true})
        for(const std::uint32_t k : {0u, 3u}) {
            HashEngine e(f, k, vocabulary, context, true, {}, prompt, {.seeded = seeded});
            auto out = run(e, count, nullptr, 2);
            if(seeded) out.insert(out.begin(), e.first);
            require(std::equal(out.begin(), out.end(), ordinary(prompt, vocabulary, out.size()).begin()),
                    "seeded rounds differ from ordinary greedy decoding");
            const auto taken = e.taken();
            require(taken == out, std::string(seeded ? "seeded" : "started") + " K=" + std::to_string(k) +
                    ": the history took in " + std::to_string(taken.size()) + " tokens for " + std::to_string(out.size()) + " published");
        }
    // Draft widths chosen per round over one controller sized for the widest: one graph pair per width used, graph
    // replay equal to eager execution, ordinary greedy output with and without Core truncations, and the history
    // after seeded rounds whose width shrinks below the previous round's emitted count.
    {
        constexpr std::uint32_t widest = 6;
        std::mt19937 pick(9);
        std::vector<std::uint32_t> widths(64);
        for(auto& w : widths) w = pick() % (widest + 1);
        std::size_t narrower = 0;
        for(std::size_t i = 1; i < widths.size(); ++i) narrower += widths[i] < widths[i - 1];
        for(const std::uint32_t truncate_every : {0u, 2u}) {
            HashEngine graphed(f, widest, vocabulary, context, true, {}, prompt), eager(f, widest, vocabulary, context, false, {}, prompt);
            std::vector<std::uint32_t> a, b;
            const auto g = run(graphed, count, &a, truncate_every, widths), e = run(eager, count, &b, truncate_every, widths);
            require(g == e && a == b, "variable-width graph replay differs from eager execution");
            require(g == expected, "variable-width rounds differ from ordinary greedy decoding");
            require(download<std::uint32_t>(graphed.state, 1) == download<std::uint32_t>(eager.state, 1), "variable-width device state");
            for(std::uint32_t k = 0; k <= widest; ++k) {
                const bool used = std::find(widths.begin(), widths.begin() + std::ptrdiff_t(std::min(a.size(), widths.size())), k) !=
                                  widths.begin() + std::ptrdiff_t(std::min(a.size(), widths.size()));
                require(graphed.round->captured(k) == used && !eager.round->captured(k), "one graph pair per width used");
            }
            HashEngine seeded(f, widest, vocabulary, context, true, {}, prompt, {.seeded = true});
            auto out = run(seeded, count, nullptr, truncate_every, widths);
            out.insert(out.begin(), seeded.first);
            require(std::equal(out.begin(), out.end(), ordinary(prompt, vocabulary, out.size()).begin()) && seeded.taken() == out,
                    "seeded variable-width rounds or their history");
        }
        require(narrower > 20, "width sequence covers narrower rounds");
        HashEngine bounded(f, 2, vocabulary, context, true, {}, prompt);
        rejects([&]{ bounded.round->launch(4, 3); }, "widest width");
        rejects([&]{ bounded.round->prepare(3); }, "widest width");
    }
    // The readback block carries each emitted token's log-probability rows and the matched count into Core's spans;
    // set_top_logprobs narrows the alternatives per token to Core's requested count.
    for(const bool graphed : {true, false}) {
        HashEngine e(f, 4, vocabulary, context, graphed, {}, prompt, {.logprobs = true});
        std::vector<std::int32_t> out;
        std::uint32_t matched = 0, round = 0;
        while(out.size() < count) {
            if(round == 5) e.round->set_top_logprobs(2);
            const auto lp = e.round->logprobs();
            const auto top = e.round->top_logprobs();
            const std::uint32_t width = round < 5 ? 3 : 2;
            require(lp.size() == 5 && top.size() == 5 * width, "logprob spans cover the widest round");
            e.round->launch(std::uint32_t(count - out.size()))->wait();
            const auto tokens = e.round->tokens();
            for(std::size_t c = 0; c < tokens.size(); ++c) {
                require(lp[c] == fx::logprob_of(tokens[c]), "an emitted token's log-probability");
                for(std::uint32_t a = 0; a < width; ++a)
                    require(top[c * width + a].token == std::int32_t((std::uint32_t(tokens[c]) + a) % vocabulary) &&
                            top[c * width + a].logprob == fx::logprob_of(tokens[c]) - float(a), "an emitted token's alternatives");
            }
            matched += e.round->matched();
            out.insert(out.end(), tokens.begin(), tokens.end());
            e.round->commit(std::uint32_t(tokens.size()));
            ++round;
        }
        require(out == expected && matched > count / 4, "rounds with log-probabilities");
        rejects([&]{ e.round->set_top_logprobs(4); }, "top_logprobs");
    }
    {
        dr::RoundConfig bad{2, vocabulary, context, {}};
        bad.top_logprobs = 2;
        rejects([&]{ dr::DeviceSpeculativeRound(f.backend, bad, {[](cudaStream_t, const dr::RoundView&) {},
                                                                 [](cudaStream_t, const dr::RoundView&) {},
                                                                 [](cudaStream_t, const dr::RoundView&) {}}); }, "top_logprobs");
    }
    // Zero copy (the allowance read from, and the readback block written to, mapped pinned memory) emits what the
    // copying rounds emit, in graph and eager mode, with truncations, log-probabilities and a seeded history.
    for(const bool graphed : {true, false})
        for(const std::uint32_t k : {0u, 3u}) {
            HashEngine copy(f, k, vocabulary, context, graphed, {}, prompt, {.seeded = true, .logprobs = true});
            HashEngine zero(f, k, vocabulary, context, graphed, {}, prompt, {.seeded = true, .logprobs = true, .zero_copy = true});
            const auto a = record(copy, count, 3), b = record(zero, count, 3);
            require(a == b, "zero-copy rounds differ from copying rounds");
            require(download<std::uint32_t>(copy.state, 1) == download<std::uint32_t>(zero.state, 1) && copy.taken() == zero.taken(),
                    "zero-copy device state or history differs");
            auto out = a.tokens;
            out.insert(out.begin(), copy.first);
            require(std::equal(out.begin(), out.end(), ordinary(prompt, vocabulary, out.size()).begin()), "zero-copy rounds");
            require(zero.round->captured() == graphed, "zero-copy capture");
        }
    // Forced replay: a verify step whose predictions are the continuation's next tokens makes the unchanged accept
    // emit exactly the continuation, drafts kept while they equal it, with the allowance capped at what is left.
    {
        // Greedy text with every ninth token replaced, continuing greedily from the replaced history (as another
        // engine's pinned text departs from this target's and then follows its own greedy path).
        std::vector<std::int32_t> replay;
        std::uint32_t s = 17;
        for(std::size_t i = 0; i + 1 < prompt.size(); ++i) s = fx::step_state_host(s, prompt[i]);
        for(auto last = prompt.back(); replay.size() < count; replay.push_back(last)) {
            s = fx::step_state_host(s, last);
            last = std::int32_t(fx::mix(s) % vocabulary);
            if(replay.size() % 9 == 4) last = (last + 1) % std::int32_t(vocabulary);
        }
        require(replay != expected, "the continuation departs from greedy decoding");
        for(const std::uint32_t k : {0u, 4u}) {
            HashEngine graphed(f, k, vocabulary, context, true, {}, prompt, {.replay = replay});
            HashEngine eager(f, k, vocabulary, context, false, {}, prompt, {.replay = replay});
            const auto t = record(graphed, replay.size(), 0);
            std::uint32_t matched = 0;
            for(auto m : t.matched) matched += m;
            require(t.tokens == replay, "a forced round did not emit its continuation");
            require(record(eager, replay.size(), 0) == t, "forced graph replay differs from eager execution");
            require(k == 0 || matched * 3 > replay.size(), "forced rounds accepted too few drafts");
            std::cout << "forced replay K=" << k << ": " << t.matched.size() << " rounds, " << matched << " drafts accepted\n";
        }
    }
    HashEngine reseed(f, 2, vocabulary, context, true, {}, prompt, {.seeded = true});
    rejects([&]{ reseed.round->seed(0, 1, 1); }, "Seed frontier");
    rejects([&]{ reseed.round->seed(4, std::int32_t(vocabulary), 1); }, "vocabulary");
    // Device stop tokens end the round after the stop, like GreedySpeculator.
    HashEngine stopping(f, 4, vocabulary, context, true, {expected[7]}, prompt);
    const auto out = run(stopping, 8);
    const auto first = std::find(out.begin(), out.end(), expected[7]);
    require(first != out.end() && std::equal(out.begin(), first + 1, expected.begin()), "device stop handling");
    // Per-request stops replace the device list without recapturing the round.
    HashEngine restop(f, 4, vocabulary, context, true, {}, prompt, {.max_stops = 1});
    restop.round->set_stops(std::vector<std::int32_t>{expected[7]});
    const auto replaced = run(restop, 8);
    const auto at = std::find(replaced.begin(), replaced.end(), expected[7]);
    require(at != replaced.end() && std::equal(replaced.begin(), at + 1, expected.begin()), "replaced device stop list");
    restop.round->set_stops({});
    rejects([&]{ restop.round->set_stops(std::vector<std::int32_t>{1, 2}); }, "capacity");
    // A round Core cancels after launch is never committed: abandon() settles its
    // readback, leaves nothing pending and requires a fresh start().
    HashEngine cancelled(f, 3, vocabulary, context, true, {}, prompt);
    run(cancelled, 10);
    cancelled.round->launch(8);
    rejects([&]{ cancelled.round->set_stops({}); }, "pending round");
    cancelled.round->abandon();
    require(!cancelled.round->pending(), "abandoned round still pending");
    cancelled.round->set_stops({});
    rejects([&]{ cancelled.round->launch(8); }, "Start the device frontier");
}

void readback() {
    Fixture f;
    const std::vector<std::int32_t> prompt = {3, 1, 4};
    for(const bool zero_copy : {false, true}) {
        HashEngine e(f, 3, 9, 512, true, {}, prompt, {.zero_copy = zero_copy});
#ifdef CRIT_TEST_LINKER_WRAP
        d2h_copies = 0;
#endif
        std::uint64_t launch_syncs = 0;
        const auto expected = ordinary(prompt, 9, 200);
        std::vector<std::int32_t> out;
        std::uint64_t rounds = 0;
        while(out.size() < expected.size()) {
#ifdef CRIT_TEST_LINKER_WRAP
            const auto before = host_syncs.load();
#endif
            auto* ready = e.round->launch(std::uint32_t(expected.size() - out.size()));
#ifdef CRIT_TEST_LINKER_WRAP
            launch_syncs += host_syncs.load() - before;
#endif
            ready->wait();
            const auto tokens = e.round->tokens();
            out.insert(out.end(), tokens.begin(), tokens.end());
            e.round->commit(std::uint32_t(tokens.size()));
            ++rounds;
        }
        require(out == expected, "readback fixture output");
        require(e.round->readbacks() == rounds && e.round->rounds() == rounds, "controller readback count");
#ifdef CRIT_TEST_LINKER_WRAP
        // Counted at the CUDA API, including calls captured into graphs on the first launch; zero copy writes the block
        // in place and copies nothing.
        const std::uint64_t want = zero_copy ? 0 : rounds;
        require(d2h_copies.load() == want, "expected " + std::to_string(want) + " device-to-host copies, got " +
                std::to_string(d2h_copies.load()) + " for " + std::to_string(rounds) + " rounds");
        require(launch_syncs == 0, "launch() performed a host synchronization");
        std::cout << (zero_copy ? "zero copy: " : "") << "rounds " << rounds << " D2H copies " << d2h_copies.load()
                  << " launch syncs " << launch_syncs << '\n';
#else
        static_cast<void>(launch_syncs);
        std::cout << "rounds " << rounds << " controller readbacks " << e.round->readbacks() << " (API count needs --wrap)\n";
#endif
    }
}

// Not a CTest case (crit_device_round_tests timing): wall time per round of the hash fixture's rounds in graphs,
// copying (allowance copy node, readback copy) against zero copy, median of 7 batches of 500 rounds, alternating.
void timing() {
    Fixture f;
    const std::vector<std::int32_t> prompt = {3, 1, 4};
    for(const std::uint32_t k : {0u, 4u}) {
        double median[2] = {};
        std::vector<double> samples[2];
        HashEngine engines[2] = {HashEngine(f, k, 9, 1 << 16, true, {}, prompt), HashEngine(f, k, 9, 1 << 16, true, {}, prompt, {.zero_copy = true})};
        for(auto& e : engines) record(e, 200, 0); // capture and warm up
        for(int batch = 0; batch < 7; ++batch)
            for(int mode = 0; mode < 2; ++mode) {
                auto& e = engines[mode];
                if(e.round->frontier() + 600 * (k + 1) > (1u << 16)) e.round->start(2, 4);
                const auto begin = std::chrono::steady_clock::now();
                for(int r = 0; r < 500; ++r) {
                    e.round->launch(k + 1)->wait();
                    e.round->commit(std::uint32_t(e.round->tokens().size()));
                }
                samples[mode].push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / 500);
            }
        for(int mode = 0; mode < 2; ++mode) {
            std::sort(samples[mode].begin(), samples[mode].end());
            median[mode] = samples[mode][samples[mode].size() / 2];
        }
        std::cout << "K=" << k << " us per round: copy " << median[0] << " (min " << samples[0].front() << "), zero copy "
                  << median[1] << " (min " << samples[1].front() << ")\n";
    }
}

void budget() {
    dr::RoundBudget b;
    const DeviceTiming step[] = {{DeviceStage::forward, 0, 1000}};
    const DeviceTiming fast[] = {{DeviceStage::forward, 0, 1100}, {DeviceStage::accept, 0, 100}};
    b.observe_step(step); b.observe_round(fast, 3);
    auto r = b.report();
    require(r.pass && r.ratio == 1.2 && r.tokens_per_round == 3 && r.speedup == 2.5, "passing round budget");
    dr::RoundBudget slow;
    const DeviceTiming heavy[] = {{DeviceStage::forward, 0, 2000}, {DeviceStage::accept, 0, 500}};
    slow.observe_step(step); slow.observe_round(heavy, 4);
    r = slow.report();
    require(!r.pass && r.ratio == 2.5 && r.threshold == 1.3, "failing round budget");
    bool rejected = false;
    try { dr::RoundBudget().report(); } catch(const std::logic_error&) { rejected = true; }
    require(rejected, "empty budget report");
}
}

int main(int argc, char** argv) {
    try {
        if(argc != 2) throw std::invalid_argument("test group required");
        const std::pair<const char*, void(*)()> groups[] = {{"accept", accept}, {"graphs", graphs}, {"readback", readback}, {"budget", budget},
                                                            {"timing", timing}};
        for(auto [name, test] : groups) if(argv[1] == std::string(name)) { test(); std::cout << "PASS " << name << '\n'; return 0; }
        throw std::invalid_argument("unknown test group");
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
