// Operator ranges (crit/trace.h): scopes, the bounded recorder, thread binding and the
// Engine-owned recorder beside the lifecycle trace. Built twice: with CRIT_TRACE_ENABLED
// (every case but trace_disabled) and without it (trace_disabled only).
#include "crit/core.h"
#include "crit/trace.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#if defined(CRIT_TRACE_NVTX) && (defined(_WINDOWS_) || defined(min) || defined(max) || defined(near))
#error "crit/trace.h exposes the NVTX headers' windows.h to consumers"
#endif

namespace {
using namespace crit::core;
namespace ct = crit::trace;

void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }

#ifdef CRIT_TRACE_ENABLED
void nest(ct::Recorder& recorder, int levels) {
    if(!levels) return;
    CRIT_TRACE_SCOPE(recorder, ct::Work{"level"});
    nest(recorder, levels - 1);
}

void scopes() {
    const auto epoch = std::chrono::steady_clock::now() - std::chrono::milliseconds(5);
    ct::Recorder recorder(8, epoch);
    int calls = 0;
    const auto work = [&](std::string_view stage) { ++calls; return ct::Work{stage, 8, 1, 256, 100, 200, 4096}; };
    {
        CRIT_TRACE_SCOPE(recorder, work("outer"));
        { CRIT_TRACE_SCOPE(recorder, work("inner")); }
    }
    try {
        CRIT_TRACE_SCOPE(recorder, work("unwound"));
        { CRIT_TRACE_SCOPE(recorder, work("completed")); }
        CRIT_TRACE_SCOPE(recorder, work("thrower"));
        throw std::runtime_error("intentional scope unwind");
    } catch(const std::runtime_error&) {}
    const auto snapshot = recorder.snapshot();
    require(calls == 5 && snapshot.enabled && snapshot.ranges.size() == 5, "scope count differs");
    const auto& r = snapshot.ranges;
    require(r[0].name() == "inner" && r[1].name() == "outer" && r[2].name() == "completed" && r[3].name() == "thrower" &&
                r[4].name() == "unwound", "ranges are not in completion order");
    require(r[0].depth == 1 && r[1].depth == 0 && r[2].depth == 1 && r[3].depth == 1 && r[4].depth == 0, "range depth differs");
    require(!r[0].failed && !r[1].failed && !r[2].failed && r[3].failed && r[4].failed, "unwind is not marked failed");
    require(r[1].begin_ns <= r[0].begin_ns && r[0].end_ns <= r[1].end_ns && r[0].begin_ns <= r[0].end_ns,
            "nested range is outside its parent");
    require(r[1].begin_ns >= 5'000'000, "range timestamps ignore the recorder epoch");
    for(std::size_t i = 0; i < r.size(); ++i) require(r[i].sequence == i + 1, "sequence differs");
    require(r[0].tokens == 8 && r[0].batch == 1 && r[0].context == 256 && r[0].useful_bytes_min == 100 &&
                r[0].useful_bytes_max == 200 && r[0].floating_ops == 4096, "work metadata lost");
    require(!snapshot.overwritten && !snapshot.dropped, "unexpected loss");
    ct::Scope null_scope(nullptr, ct::Work{"none"});  // a null sink records nothing
}

void bounds() {
    ct::Recorder ring(3);
    for(int i = 0; i < 5; ++i) { CRIT_TRACE_SCOPE(ring, ct::Work{"step"}); }
    auto snapshot = ring.snapshot();
    require(ring.capacity() == 3 && snapshot.ranges.size() == 3 && snapshot.overwritten == 2 && !snapshot.dropped,
            "ring did not bound retention");
    require(snapshot.ranges[0].sequence == 3 && snapshot.ranges[2].sequence == 5, "ring order corrupted on wrap");

    ct::Recorder deep(128);
    nest(deep, int(ct::max_depth) + 6);
    snapshot = deep.snapshot();
    require(snapshot.ranges.size() == ct::max_depth && snapshot.dropped == 6, "deep ranges not dropped and counted");
    require(snapshot.ranges.front().depth == ct::max_depth - 1 && snapshot.ranges.back().depth == 0,
            "overflow broke shallower nesting");
    deep.end(false);  // unmatched end
    require(deep.snapshot().dropped == 7 && deep.snapshot().ranges.size() == ct::max_depth, "unmatched end not counted");

    const std::string label(200, 'x');
    ct::Recorder truncating(1);
    { CRIT_TRACE_SCOPE(truncating, ct::Work{label}); }
    require(truncating.snapshot().ranges[0].name() == std::string_view(label).substr(0, ct::stage_bytes - 1),
            "long stage name not truncated");

    ct::Recorder empty(0);
    { CRIT_TRACE_SCOPE(empty, ct::Work{"lost"}); }
    snapshot = empty.snapshot();
    require(snapshot.enabled && snapshot.ranges.empty() && snapshot.dropped == 1, "zero capacity range not counted");
}

void binding() {
    require(ct::current_sink() == nullptr, "a thread starts with a bound sink");
    ct::Recorder outer(4), inner(4);
    { CRIT_TRACE_OPERATOR(ct::Work{"unbound"}); }
    {
        const ct::SinkBinding first(&outer);
        { CRIT_TRACE_OPERATOR(ct::Work{"outer"}); }
        {
            const ct::SinkBinding second(&inner);
            require(ct::current_sink() == &inner, "nested binding not current");
            { CRIT_TRACE_OPERATOR(ct::Work{"inner"}); }
            bool elsewhere = true;
            std::thread([&] { elsewhere = ct::current_sink() != nullptr; }).join();
            require(!elsewhere, "binding leaked to another thread");
        }
        require(ct::current_sink() == &outer, "binding not restored");
    }
    require(ct::current_sink() == nullptr, "outer binding not restored");
    const auto a = outer.snapshot(), b = inner.snapshot();
    require(a.ranges.size() == 1 && a.ranges[0].name() == "outer" && b.ranges.size() == 1 && b.ranges[0].name() == "inner",
            "operator ranges reached the wrong sink");
}

void reader() {
    constexpr std::uint64_t total = 20000;
    ct::Recorder recorder(64);
    std::atomic<bool> done{false};
    std::thread writer([&] {
        for(std::uint64_t i = 0; i < total; ++i) { CRIT_TRACE_SCOPE(recorder, ct::Work{"op", i}); }
        done = true;
    });
    std::size_t reads = 0;
    bool ordered = true;
    while(!done || reads < 4) {
        const auto snapshot = recorder.snapshot();
        for(std::size_t i = 1; i < snapshot.ranges.size(); ++i)
            ordered = ordered && snapshot.ranges[i].sequence == snapshot.ranges[i - 1].sequence + 1;
        ordered = ordered && snapshot.ranges.size() <= recorder.capacity();
        ++reads;
    }
    writer.join();
    const auto last = recorder.snapshot();
    require(ordered, "concurrent snapshot order corrupted");
    require(last.ranges.size() == 64 && last.ranges.back().sequence + last.dropped == total &&
                last.overwritten == last.ranges.back().sequence - 64, "concurrent reader lost accounting");
}

struct NullBackend final : Backend {
    void bind_thread() override {}
    void drain() override {}
};
struct TracedSession final : Session {
    std::span<const Token> prompt;
    std::uint32_t frontier = 0;
    Token output = 1;
    explicit TracedSession(std::span<const Token> p) : prompt(p) {}
    Progress advance(WorkBudget budget) override {
        CRIT_TRACE_OPERATOR(ct::Work{"layer", 1, 1});
        { CRIT_TRACE_OPERATOR(ct::Work{"projection", 1, 1, 0, 64, 128, 256}); }
        const auto count = std::min(budget.prefill_tokens, std::uint32_t(prompt.size()) - frontier);
        frontier += count;
        if(frontier != prompt.size()) return {count, false, {}};
        return {count, true, {&output, 1}};
    }
    void commit(std::uint32_t, bool) override {}
    void close(Finish) noexcept override {}
};
struct TracedModel final : Model {
    ModelInfo info() const override { return {"trace.fixture", 64, 32, 4, {5}, true, false}; }
    std::unique_ptr<Session> open(std::span<const Token> prompt) override {
        return std::make_unique<TracedSession>(prompt);
    }
};
std::unique_ptr<Engine> traced_engine(TraceConfig trace, std::size_t host_limit = std::size_t(1) << 22) {
    Profile profile{32, 2, host_limit, 1024};
    profile.trace = trace;
    return std::make_unique<Engine>(profile, std::make_shared<NullBackend>(), [](const Profile&, Services) {
        CRIT_TRACE_OPERATOR(ct::Work{"load"});
        return std::make_unique<TracedModel>();
    });
}
template<class F> void rejects(F&& call, const char* message) {
    bool thrown = false;
    try { call(); } catch(const std::invalid_argument&) { thrown = true; }
    require(thrown, message);
}

void engine() {
    auto e = traced_engine({256, 0, 128});
    require(e->usage(Memory::host).live == 256 * sizeof(TraceEvent) + 128 * sizeof(ct::Range),
            "operator ranges escaped the resource ledger");
    e->submit({1, 1, 1}, 2).wait();
    const auto snapshot = e->trace();
    const auto& ranges = snapshot.operators.ranges;
    require(snapshot.enabled && snapshot.operators.enabled && !snapshot.operators.dropped, "operator ranges disabled");
    require(!ranges.empty() && ranges.front().name() == "load", "construction range missing");
    std::size_t projections = 0;
    for(const auto& range : ranges) {
        if(range.name() == "load") continue;
        require((range.name() == "projection" && range.depth == 1) || (range.name() == "layer" && range.depth == 0),
                "engine range nesting differs");
        if(range.name() != "projection") continue;
        ++projections;
        require(range.useful_bytes_max == 128 && range.floating_ops == 256, "engine range metadata lost");
        // Shared clock: each operator range lies inside one step span of the lifecycle trace.
        const bool inside = std::any_of(snapshot.events.begin(), snapshot.events.end(), [&](const TraceEvent& begin) {
            return begin.phase == TracePhase::step && begin.mark == TraceMark::begin && begin.timestamp_ns <= range.begin_ns &&
                   std::any_of(snapshot.events.begin(), snapshot.events.end(), [&](const TraceEvent& end) {
                       return end.phase == TracePhase::step && end.mark == TraceMark::end &&
                              end.operation == begin.operation && end.timestamp_ns >= range.end_ns;
                   });
        });
        require(inside, "operator range outside every step span");
    }
    require(projections >= 2, "decode operator ranges missing");

    // Operator ranges alone: the lifecycle trace stays disabled and empty.
    auto alone = traced_engine({0, 0, 16});
    alone->submit({1}, 1).wait();
    const auto only = alone->trace();
    require(!only.enabled && only.events.empty() && only.operators.enabled && only.operators.ranges.size() > 1,
            "operator ranges changed the lifecycle trace");

    // Disabled: no engine recorder, and the constructing thread keeps its own sink.
    ct::Recorder own(8);
    {
        const ct::SinkBinding bound(&own);
        auto plain = traced_engine({256, 0, 0});
        plain->submit({1}, 1).wait();
        const auto none = plain->trace();
        require(none.enabled && !none.operators.enabled && none.operators.ranges.empty(), "disabled operators recorded");
        require(plain->usage(Memory::host).live == 256 * sizeof(TraceEvent), "disabled operators reserved memory");
    }
    const auto mine = own.snapshot();
    require(mine.ranges.size() == 1 && mine.ranges[0].name() == "load", "engine replaced the caller's sink");

    rejects([] { traced_engine({0, 0, 65537}); }, "operator capacity above 65536 accepted");
    traced_engine({0, 0, 0}, 64);  // the same limit admits the engine without operator ranges
    bool ceiling = false;
    try { traced_engine({0, 0, 64}, 64); } catch(const std::exception&) { ceiling = true; }
    require(ceiling, "operator ranges exceeded the host limit");
}

#ifdef CRIT_TRACE_NVTX
void nvtx() {
    ct::NvtxSink sink;
    { CRIT_TRACE_SCOPE(sink, ct::Work{"native-smoke", 1, 1, 0, 8, 16, 32}); }
}
#endif
#else
void disabled() {
    ct::Recorder recorder(4);
    int calls = 0;
    const auto work = [&] { ++calls; return ct::Work{"projection"}; };
    (void)work;
    CRIT_TRACE_SCOPE(recorder, work());
    CRIT_TRACE_OPERATOR(work());
    // Neither argument is evaluated, so undeclared names compile.
    CRIT_TRACE_SCOPE(undefined_sink_symbol, undefined_work_symbol);
    CRIT_TRACE_OPERATOR(undefined_work_symbol);
    require(calls == 0 && recorder.snapshot().ranges.empty(), "disabled tracing evaluated its arguments");
}
#endif
}

int main(int argc, char** argv) {
    const std::vector<std::pair<std::string_view, void (*)()>> cases{
#ifdef CRIT_TRACE_ENABLED
        {"trace_scopes", scopes}, {"trace_bounds", bounds}, {"trace_binding", binding}, {"trace_reader", reader},
        {"trace_engine", engine},
#ifdef CRIT_TRACE_NVTX
        {"trace_nvtx", nvtx},
#endif
#else
        {"trace_disabled", disabled},
#endif
    };
    try {
        const std::string_view name = argc > 1 ? argv[1] : "";
        bool found = false;
        for(const auto& [label, run] : cases) if(name == "all" || name == label) {
            run();
            found = true;
            std::printf("crit core trace passed: %.*s\n", int(label.size()), label.data());
        }
        require(found, "unknown trace test case");
    } catch(const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
