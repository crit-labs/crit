// Device speculative round fixtures: greedy accept against GreedySpeculator,
// graph replay against eager execution, one readback per round, RoundBudget.
#include "device_round.h"
#include "device_round_fixture.h"
#include "round_budget.h"
#include "speculative.h"
#include <crit/cuda.h>
#include <algorithm>
#include <atomic>
#include <cstring>
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
        }
    }
    std::cout << "accept rows " << rows << " all-match " << all_match << " first-mismatch " << first_mismatch
              << " stop " << stopped << " budget " << budget_cut << '\n';
}

// A synthetic engine bound to one controller.
struct HashEngine {
    CudaBuffer state, start, checkpoints, history, prompt;
    fx::HashTarget target;
    std::unique_ptr<dr::DeviceSpeculativeRound> round;
    HashEngine(Fixture& f, std::uint32_t k, std::uint32_t vocabulary, std::uint32_t context, bool graphs,
               std::vector<std::int32_t> stops, const std::vector<std::int32_t>& prompt_tokens, std::uint32_t max_stops = 0) {
        state = f.backend->allocate(Memory::device, 4); start = f.backend->allocate(Memory::device, 4);
        checkpoints = f.backend->allocate(Memory::device, (k + 1) * 4); history = f.backend->allocate(Memory::device, context * 4);
        prompt = f.backend->allocate(Memory::device, prompt_tokens.size() * 4);
        upload(state, std::vector<std::uint32_t>{17});
        upload(prompt, prompt_tokens);
        target = {static_cast<std::uint32_t*>(state.data()), static_cast<std::uint32_t*>(start.data()),
                  static_cast<std::uint32_t*>(checkpoints.data()), static_cast<std::int32_t*>(history.data()), vocabulary, 3, 5};
        auto t = target;
        dr::RoundSteps steps{
            [t](cudaStream_t s, const dr::RoundView& v) { check(fx::enqueue_draft(t, v.control, v.drafts, v.draft_tokens, s)); },
            [t](cudaStream_t s, const dr::RoundView& v) { check(fx::enqueue_verify(t, v.control, v.drafts, v.predictions, v.draft_tokens, s)); },
            [t](cudaStream_t s, const dr::RoundView& v) { check(fx::enqueue_commit_state(t, v.control, s)); }};
        dr::RoundConfig config{k, vocabulary, context, std::move(stops), graphs};
        config.max_stops = max_stops;
        round = std::make_unique<dr::DeviceSpeculativeRound>(f.backend, std::move(config), std::move(steps));
        const auto materialized = std::uint32_t(prompt_tokens.size() - 1);
        check(fx::enqueue_prefill(target, static_cast<const std::int32_t*>(prompt.data()), materialized, f.backend->stream()));
        round->start(materialized, prompt_tokens.back());
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
std::vector<std::int32_t> run(HashEngine& e, std::size_t count, std::vector<std::uint32_t>* per_round = nullptr,
                              std::uint32_t truncate_every = 0) {
    std::vector<std::int32_t> out;
    std::uint32_t round = 0;
    while(out.size() < count) {
        auto* ready = e.round->launch(std::uint32_t(count - out.size()));
        ready->wait();
        const auto tokens = e.round->tokens();
        auto accepted = std::uint32_t(tokens.size());
        if(truncate_every && ++round % truncate_every == 0 && accepted > 1) accepted = 1 + accepted / 3; // emulate a Core stop cut
        out.insert(out.end(), tokens.begin(), tokens.begin() + accepted);
        if(per_round) per_round->push_back(accepted);
        e.round->commit(accepted);
    }
    return out;
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
    // Device stop tokens end the round after the stop, like GreedySpeculator.
    HashEngine stopping(f, 4, vocabulary, context, true, {expected[7]}, prompt);
    const auto out = run(stopping, 8);
    const auto first = std::find(out.begin(), out.end(), expected[7]);
    require(first != out.end() && std::equal(out.begin(), first + 1, expected.begin()), "device stop handling");
    // Per-request stops replace the device list without recapturing the round.
    HashEngine restop(f, 4, vocabulary, context, true, {}, prompt, 1);
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
    HashEngine e(f, 3, 9, 512, true, {}, prompt);
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
    // Counted at the CUDA API, including calls captured into graphs on the first launch.
    require(d2h_copies.load() == rounds, "expected exactly one device-to-host copy per round, got " +
            std::to_string(d2h_copies.load()) + " for " + std::to_string(rounds) + " rounds");
    require(launch_syncs == 0, "launch() performed a host synchronization");
    std::cout << "rounds " << rounds << " D2H copies " << d2h_copies.load() << " launch syncs " << launch_syncs << '\n';
#else
    static_cast<void>(launch_syncs);
    std::cout << "rounds " << rounds << " controller readbacks " << e.round->readbacks() << " (API count needs --wrap)\n";
#endif
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
        const std::pair<const char*, void(*)()> groups[] = {{"accept", accept}, {"graphs", graphs}, {"readback", readback}, {"budget", budget}};
        for(auto [name, test] : groups) if(argv[1] == std::string(name)) { test(); std::cout << "PASS " << name << '\n'; return 0; }
        throw std::invalid_argument("unknown test group");
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
