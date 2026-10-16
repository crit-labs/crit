// Composed device speculative rounds through a real Core Engine: a synthetic
// one-attention, one-gated-delta target, a bigram drafter and one advance() per
// round. Checks ordinary-greedy equality, one readback per round and trace stage
// timings, then prints the round/step ratio. Synthetic weights; no speed claim.
#include "device_round.h"
#include "round_budget.h"
#include "toy_model.h"
#include <crit/cuda.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CRIT_TOY_COUNT_COPIES
// Count device-to-host copies at the CUDA API, including captured calls.
std::atomic<std::uint64_t> toy_d2h_copies{0};
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* d, const void* s, std::size_t n, cudaMemcpyKind k, cudaStream_t st) {
    if(k == cudaMemcpyDeviceToHost) ++toy_d2h_copies;
    return __real_cudaMemcpyAsync(d, s, n, k, st);
}
#endif

namespace {
using namespace crit::core;
namespace dr = crit::techniques::device_round;
namespace toy = crit::techniques::device_round::toy;

void check(cudaError_t s, const char* what) {
    if(s != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s));
}
template<class T> CudaBuffer upload(CudaBackend& b, const std::vector<T>& v) {
    auto buffer = b.allocate(Memory::device, v.size() * sizeof(T));
    buffer.upload(std::as_bytes(std::span(v)), 0, Queue::compute); // loading only
    return buffer;
}
CudaBuffer floats(CudaBackend& b, std::size_t count) {
    auto buffer = b.allocate(Memory::device, count * sizeof(float));
    b.fill_async(buffer, 0);
    return buffer;
}

struct ToyWeights {
    std::vector<CudaBuffer> owners;
    toy::Weights view;
    explicit ToyWeights(CudaBackend& b) {
        std::mt19937 random(2026);
        auto make = [&](std::size_t n, float scale) {
            std::normal_distribution<float> d(0, scale);
            std::vector<float> v(n);
            for(auto& x : v) x = d(random);
            owners.push_back(upload(b, v));
            return static_cast<const float*>(owners.back().data());
        };
        const float s = 1.0f / std::sqrt(float(toy::width));
        view.embed = make(toy::vocabulary * toy::width, 1.0f);
        view.wq = make(toy::width * toy::width, s); view.wk = make(toy::width * toy::width, s);
        view.wv = make(toy::width * toy::width, s); view.wo = make(toy::width * toy::width, s);
        view.pq = make(toy::keys * toy::width, s); view.pk = make(toy::keys * toy::width, s);
        view.pv = make(toy::values * toy::width, s);
        view.pa = make(toy::width, s); view.pb = make(toy::width, s);
        view.po = make(toy::width * toy::values, 1.0f / std::sqrt(float(toy::values)));
        view.head = make(toy::vocabulary * toy::width, s);
    }
};

// Per-request device state with stable pointers for graph capture.
struct ToyBuffers {
    CudaBuffer k_cache, v_cache, x1, q, k, v, ld, beta, out, live, start, checkpoints, inputs, predictions, scalars;
    toy::State state;
    explicit ToyBuffers(CudaBackend& b) {
        k_cache = floats(b, toy::context * toy::width); v_cache = floats(b, toy::context * toy::width);
        x1 = floats(b, toy::max_columns * toy::width);
        q = floats(b, toy::max_columns * toy::keys); k = floats(b, toy::max_columns * toy::keys);
        v = floats(b, toy::max_columns * toy::values);
        ld = floats(b, toy::max_columns); beta = floats(b, toy::max_columns);
        out = floats(b, toy::max_columns * toy::values);
        live = floats(b, toy::keys * toy::values); start = floats(b, toy::keys * toy::values);
        checkpoints = floats(b, toy::max_columns * toy::keys * toy::values);
        inputs = b.allocate(Memory::device, toy::context * sizeof(std::int32_t));
        predictions = b.allocate(Memory::device, toy::max_columns * sizeof(std::int32_t));
        scalars = b.allocate(Memory::device, 2 * sizeof(std::int32_t)); // {frontier, count} for eager steps
        auto f = [](CudaBuffer& x) { return static_cast<float*>(x.data()); };
        state = {f(k_cache), f(v_cache), f(x1), f(q), f(k), f(v), f(ld), f(beta), f(out), f(live), f(start), f(checkpoints)};
    }
    std::int32_t* ints(CudaBuffer& x) const { return static_cast<std::int32_t*>(x.data()); }
    // Eager materialization of `count` tokens at `frontier` (prefill or ordinary step).
    void materialize(CudaBackend& b, const toy::Weights& w, const std::int32_t* tokens, std::uint32_t frontier, std::uint32_t count) {
        auto* s = ints(scalars);
        const auto stream = b.stream();
        check(toy::enqueue_set(s, std::int32_t(frontier), stream), "set frontier");
        check(toy::enqueue_set(s + 1, std::int32_t(count), stream), "set count");
        check(toy::enqueue_forward(w, state, tokens, s, count, ints(predictions), stream), "forward");
        check(toy::enqueue_commit(state, s + 1, count, stream), "commit");
    }
};

// Ordinary greedy decoding, one token per step with a synchronous read. Oracle only.
std::vector<std::int32_t> ordinary(CudaBackend& b, const toy::Weights& w, const std::vector<std::int32_t>& prompt, std::size_t count) {
    ToyBuffers buffers(b);
    auto inputs = upload(b, prompt);
    std::uint32_t frontier = 0;
    for(std::size_t i = 0; i + 1 < prompt.size(); i += toy::max_columns) {
        const auto n = std::uint32_t(std::min<std::size_t>(toy::max_columns, prompt.size() - 1 - i));
        buffers.materialize(b, w, static_cast<const std::int32_t*>(inputs.data()) + i, frontier, n);
        frontier += n;
    }
    std::vector<std::int32_t> out;
    auto token = prompt.back();
    auto single = b.allocate(Memory::device, sizeof(std::int32_t));
    while(out.size() < count) {
        single.upload(std::as_bytes(std::span(&token, 1)), 0, Queue::compute);
        buffers.materialize(b, w, static_cast<const std::int32_t*>(single.data()), frontier++, 1);
        buffers.predictions.download(std::as_writable_bytes(std::span(&token, 1)), 0, Queue::compute);
        out.push_back(token);
    }
    return out;
}

struct Stats { std::uint64_t rounds = 0, readbacks = 0, emitted = 0; std::vector<std::uint32_t> per_round; bool captured = false; };

class ToySession final : public Session {
    std::shared_ptr<CudaBackend> backend_;
    const toy::Weights& weights_;
    std::vector<std::int32_t> prompt_;
    CudaBuffer prompt_device_;
    ToyBuffers buffers_;
    std::unique_ptr<dr::DeviceSpeculativeRound> round_;
    Stats& stats_;
    std::uint32_t consumed_ = 0;
    bool decoding_ = false;
public:
    ToySession(std::shared_ptr<CudaBackend> backend, const toy::Weights& w, const CudaBuffer& table, std::span<const Token> prompt,
               std::uint32_t draft_tokens, bool graphs, Stats& stats)
        : backend_(std::move(backend)), weights_(w), prompt_(prompt.begin(), prompt.end()),
          prompt_device_(upload(*backend_, prompt_)), buffers_(*backend_), stats_(stats) {
        auto state = buffers_.state;
        auto* inputs = buffers_.ints(buffers_.inputs);
        const auto* bigram = static_cast<const std::int32_t*>(table.data());
        const toy::Weights* weights = &weights_;
        dr::RoundSteps steps{
            [bigram](cudaStream_t s, const dr::RoundView& v) {
                check(toy::enqueue_bigram_draft(bigram, v.control, v.draft_tokens, v.drafts, s), "draft");
            },
            [weights, state, inputs](cudaStream_t s, const dr::RoundView& v) {
                check(toy::enqueue_round_inputs(v.control, v.drafts, v.draft_tokens, inputs, s), "round inputs");
                check(toy::enqueue_forward(*weights, state, inputs, v.control + dr::control::frontier,
                                           v.draft_tokens + 1, v.predictions, s), "verify");
            },
            [state](cudaStream_t s, const dr::RoundView& v) {
                // Recurrent state from the device-held count; the KV frontier already moved on the device.
                check(toy::enqueue_commit(state, v.control + dr::control::n_emitted, v.draft_tokens + 1, s), "commit state");
            }};
        round_ = std::make_unique<dr::DeviceSpeculativeRound>(backend_,
            dr::RoundConfig{draft_tokens, toy::vocabulary, toy::context, {}, graphs}, std::move(steps));
    }
    Progress advance(WorkBudget budget) override {
        decoding_ = budget.prefill_complete;
        if(!decoding_) {
            // Materialize every prompt token except the last, which becomes the pending input.
            const auto count = std::min<std::uint32_t>(budget.prefill_tokens, std::uint32_t(prompt_.size()) - consumed_);
            const auto end = std::min<std::uint32_t>(consumed_ + count, std::uint32_t(prompt_.size()) - 1);
            for(auto i = consumed_; i < end; i += toy::max_columns) {
                const auto n = std::min<std::uint32_t>(toy::max_columns, end - i);
                buffers_.materialize(*backend_, weights_, static_cast<const std::int32_t*>(prompt_device_.data()) + i, i, n);
            }
            consumed_ += count;
            const bool done = consumed_ == prompt_.size();
            if(done) round_->start(std::uint32_t(prompt_.size() - 1), prompt_.back());
            return {count, done, {}, nullptr};
        }
        // The round count is known only after the readback: return the readiness
        // with deferred tokens; Core waits once and then calls ready_tokens().
        auto* ready = round_->launch(budget.output_tokens);
        stats_.readbacks = round_->readbacks();
        stats_.captured = round_->captured();
        return {0, true, {}, ready, true};
    }
    std::span<const Token> ready_tokens() override {
        const auto tokens = round_->tokens();
        ++stats_.rounds; stats_.emitted += tokens.size();
        stats_.per_round.push_back(std::uint32_t(tokens.size()));
        return tokens;
    }
    std::span<const DeviceTiming> device_timings() override { return decoding_ ? round_->timings() : std::span<const DeviceTiming>{}; }
    void commit(std::uint32_t accepted, bool) override { if(decoding_) round_->commit(accepted); }
    // Core can cancel after advance() launched a round and before commit().
    void close(Finish) noexcept override {
        try { if(round_->pending()) round_->abandon(); } catch(...) {}
    }
};

class ToyModel final : public Model {
    std::shared_ptr<CudaBackend> backend_;
    ToyWeights weights_;
    CudaBuffer table_;
    std::uint32_t draft_tokens_;
    bool graphs_;
public:
    Stats stats;
    ToyModel(std::shared_ptr<CudaBackend> backend, const std::vector<std::int32_t>& table, std::uint32_t k, bool graphs)
        : backend_(std::move(backend)), weights_(*backend_), table_(upload(*backend_, table)), draft_tokens_(k), graphs_(graphs) {}
    ModelInfo info() const override {
        ModelInfo m;
        m.implementation = "device-round-toy";
        m.vocabulary = toy::vocabulary;
        m.context = toy::context;
        m.prefill_chunk = 16;
        return m;
    }
    std::unique_ptr<Session> open(std::span<const Token> prompt) override {
        if(prompt.empty()) throw std::invalid_argument("Nonempty prompt required");
        stats = {};
        return std::make_unique<ToySession>(backend_, weights_.view, table_, prompt, draft_tokens_, graphs_, stats);
    }
    std::vector<Token> tokenize(std::string_view text) override {
        std::vector<Token> t;
        for(unsigned char c : text) t.push_back(Token(c % toy::vocabulary));
        return t;
    }
    std::string detokenize(std::span<const Token> tokens) override { return std::string(tokens.size(), '?'); }
};

struct Run {
    Result result;
    Stats stats;
    TraceSnapshot trace;
};
Run generate(std::uint32_t k, bool graphs, const std::vector<std::int32_t>& table, const std::vector<std::int32_t>& prompt,
             std::uint32_t output, RequestOptions options = {}) {
    Profile profile;
    profile.context = toy::context;
    profile.prefill_chunk = 16;
    profile.host_limit = 1 << 24;
    profile.device_limit = 1 << 26;
    profile.trace.capacity = 4096;
    auto backend = std::make_shared<CudaBackend>();
    ToyModel* model = nullptr;
    Engine engine(profile, backend, [&](const Profile&, Services) {
        auto m = std::make_unique<ToyModel>(backend, table, k, graphs);
        model = m.get();
        return m;
    });
    Run run;
    run.result = engine.submit(prompt, output, std::move(options)).wait();
    run.stats = model->stats;
    run.trace = engine.trace();
    return run;
}
std::vector<std::uint64_t> round_ns(const TraceSnapshot& trace) {
    std::map<std::uint64_t, std::uint64_t> per_step;
    for(const auto& e : trace.events) if(e.phase == TracePhase::device) per_step[e.operation] += e.duration_ns;
    std::vector<std::uint64_t> out;
    for(const auto& [op, ns] : per_step) out.push_back(ns);
    return out;
}
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
}

int main() {
    try {
        constexpr std::uint32_t draft_tokens = 3, output = 48;
        const std::vector<std::int32_t> prompt = {5, 17, 3, 29, 11, 40, 2, 8, 33, 21, 7, 14, 26, 1, 45, 19, 30, 9, 12};
        const std::vector<std::int32_t> other = {44, 6, 23};
        std::vector<std::int32_t> expected, expected_other, table(toy::vocabulary);
        {
            auto backend = std::make_shared<CudaBackend>();
            Resources resources(1 << 24, 1 << 26);
            backend->initialize(resources);
            ToyWeights weights(*backend);
            expected = ordinary(*backend, weights.view, prompt, output);
            expected_other = ordinary(*backend, weights.view, other, output);
            // Toy drafter: a bigram table of transitions seen in an ordinary run on another prompt.
            for(std::uint32_t t = 0; t < toy::vocabulary; ++t) table[t] = std::int32_t(t);
            const std::vector<std::int32_t> warm = {3, 3, 7};
            auto seen = ordinary(*backend, weights.view, warm, 96);
            seen.insert(seen.begin(), warm.back());
            for(std::size_t i = 0; i + 1 < seen.size(); ++i) table[std::size_t(seen[i])] = seen[i + 1];
            backend->drain();
        }
#ifdef CRIT_TOY_COUNT_COPIES
        toy_d2h_copies = 0;
#endif
        auto spec = generate(draft_tokens, true, table, prompt, output);
#ifdef CRIT_TOY_COUNT_COPIES
        const auto copies = toy_d2h_copies.load();
#endif
        require(spec.result.finish == Finish::length, "speculative request did not finish by length");
        require(spec.result.tokens == expected, "speculative output differs from ordinary greedy decoding");
        require(spec.stats.captured, "rounds were not captured into graphs");
        require(spec.stats.readbacks == spec.stats.rounds, "more than one readback per round");
#ifdef CRIT_TOY_COUNT_COPIES
        require(copies == spec.stats.rounds, "CUDA API saw " + std::to_string(copies) + " device-to-host copies for " +
                std::to_string(spec.stats.rounds) + " rounds");
#endif
        std::size_t forward = 0, accept = 0;
        for(const auto& e : spec.trace.events) if(e.phase == TracePhase::device) {
            require(e.duration_ns > 0, "zero device stage duration");
            forward += e.stage == DeviceStage::forward;
            accept += e.stage == DeviceStage::accept;
        }
        require(forward == spec.stats.rounds && accept == spec.stats.rounds, "missing stage timings in the Core trace");

        auto eager = generate(draft_tokens, false, table, prompt, output);
        require(eager.result.tokens == expected && !eager.stats.captured, "eager rounds differ");
        auto second = generate(draft_tokens, true, table, other, output);
        require(second.result.tokens == expected_other, "second prompt differs from ordinary greedy decoding");

        // A request stop the device does not know: Core trims the row and the
        // session applies the truncation to device state.
        const auto stop = expected[12];
        const auto cut = std::size_t(std::find(expected.begin(), expected.end(), stop) - expected.begin()) + 1;
        RequestOptions options;
        options.stop_tokens = std::vector<Token>{stop};
        auto stopped = generate(draft_tokens, true, table, prompt, output, std::move(options));
        require(stopped.result.finish == Finish::stop &&
                stopped.result.tokens == std::vector<std::int32_t>(expected.begin(), expected.begin() + std::ptrdiff_t(cut)),
                "request stop truncation differs");

        // Ordinary step through the same path (K = 0) for the round/step budget.
        auto step = generate(0, true, table, prompt, output);
        require(step.result.tokens == expected, "K=0 device path differs from ordinary greedy decoding");
        dr::RoundBudget budget;
        for(auto ns : round_ns(step.trace)) budget.observe_step_ns(ns);
        // Device events are recorded per decode step in order, one step per round.
        const auto rounds = round_ns(spec.trace);
        require(rounds.size() == spec.stats.per_round.size(), "one device timing group per round");
        for(std::size_t i = 0; i < rounds.size(); ++i) budget.observe_round_ns(rounds[i], spec.stats.per_round[i]);
        const auto report = budget.report();
        std::cout << "tokens " << spec.result.tokens.size() << " rounds " << spec.stats.rounds << " readbacks " << spec.stats.readbacks
#ifdef CRIT_TOY_COUNT_COPIES
                  << " d2h " << copies
#endif
                  << "\nround/step ratio " << report.ratio << " (step " << report.step_ns << " ns, round " << report.round_ns
                  << " ns), tokens/round " << report.tokens_per_round << ", implied speedup " << report.speedup
                  << ", threshold " << report.threshold << ": " << (report.pass ? "pass" : "fail") << '\n';
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
