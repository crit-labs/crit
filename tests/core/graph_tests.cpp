// Core device graph protocol on the host: graph identity and settled lifetime, staged matrix geometry and the
// staged replay/failure state machine. No device or CUDA call is made.
#include <crit/device_graph.h>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace crit::core;
namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
template<class Error = std::exception, class F> void rejects(F action, const char* message) {
    try { action(); }
    catch(const Error& e) {
        require(std::string(e.what()).find(message) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error(std::string("expected rejection missing: ") + message);
}
struct Receipt final : GraphReceipt {
    bool finished = false, uncertain = false;
    std::exception_ptr failure;
    explicit Receipt(bool done_now = false):finished(done_now) {}
    bool done() override {
        if(uncertain) throw std::system_error(std::make_error_code(std::errc::io_error), "cannot prove completion");
        return finished;
    }
    void result() override { if(failure) std::rethrow_exception(failure); }
};
std::shared_ptr<Receipt> receipt(bool done_now = false) { return std::make_shared<Receipt>(done_now); }
GraphOwner owner() { return std::make_shared<int>(0); }
GraphHandle graph() { return std::make_shared<int>(0); }
GraphKey key() {
    GraphKey value;
    value.phase = "decode";
    value.rows = {{1, 1}, {2, 1}};
    value.addresses = {4096};
    value.shapes = {{2, 4}};
    value.representation = "bf16";
    return value;
}

void graph_key() {
    auto valid = key();
    valid.binding_generations = {3};
    valid.validate();
    std::vector<std::function<void(GraphKey&)>> breaks{
        [](GraphKey& k) { k.phase.clear(); }, [](GraphKey& k) { k.representation.clear(); },
        [](GraphKey& k) { k.rows.clear(); }, [](GraphKey& k) { k.addresses.clear(); },
        [](GraphKey& k) { k.addresses = {0}; }, [](GraphKey& k) { k.rows = {{1, 1}, {1, 1}}; },
        [](GraphKey& k) { k.rows = {{0, 1}}; }, [](GraphKey& k) { k.rows = {{1, 0}}; },
        [](GraphKey& k) { k.shapes.clear(); }, [](GraphKey& k) { k.shapes = {{}}; },
        [](GraphKey& k) { k.shapes = {{2, 0}}; }, [](GraphKey& k) { k.allocation_generation = 0; },
        [](GraphKey& k) { k.binding_generations = {3, 4}; }, [](GraphKey& k) { k.binding_generations = {0}; }};
    for(const auto& change : breaks) {
        auto bad = valid;
        change(bad);
        rejects<std::invalid_argument>([&] { bad.validate(); }, "");
    }
    auto reversed = key();
    std::swap(reversed.rows[0], reversed.rows[1]);
    require(reversed != key() && (reversed < key() || key() < reversed), "row order is part of graph identity");
}
void graph_identity() {
    GraphCache cache(2);
    const auto first = owner();
    int captures = 0;
    const GraphCache::Capture capture = [&](const GraphKey&) { ++captures; return graph(); };
    const auto handle = cache.capture(key(), {first}, capture);
    require(cache.capture(key(), {first}, capture) == handle && captures == 1, "identical key recaptured");
    auto changed = key();
    std::swap(changed.rows[0], changed.rows[1]);
    cache.capture(changed, {first}, capture);
    require(captures == 2 && cache.size() == 2, "changed row order reused a graph");
    rejects<std::logic_error>([&] { cache.capture(key(), {owner()}, capture); }, "new physical owners");
    auto third = key();
    third.phase = "prefill";
    rejects<std::logic_error>([&] { cache.capture(third, {first}, capture); }, "eviction required");
    const auto pending = receipt();
    cache.replay(key(), [&](const GraphHandle&) { return pending; });
    rejects<std::logic_error>([&] { cache.evict(key(), [](const GraphHandle&) {}); }, "retained owners");
    pending->finished = true;
    std::vector<GraphHandle> destroyed;
    cache.evict(key(), [&](const GraphHandle& g) { destroyed.push_back(g); });
    require(destroyed == std::vector<GraphHandle>{handle} && !cache.contains(key()), "settled eviction");
    rejects<std::out_of_range>([&] { cache.replay(key(), [](const GraphHandle&) { return receipt(true); }); },
                               "not cached");
    rejects<std::invalid_argument>([&] { cache.capture(third, {}, capture); }, "physical owners");
}
void graph_generations() {
    auto bound = key();
    bound.addresses = {4096, 8192};
    bound.binding_generations = {3, 7};
    GraphCache cache(2);
    const std::vector<GraphOwner> owners{owner(), owner()};
    int captures = 0;
    const GraphCache::Capture capture = [&](const GraphKey&) { ++captures; return graph(); };
    cache.capture(bound, owners, capture);
    auto rebound = bound;
    rebound.binding_generations = {3, 8};
    cache.capture(rebound, owners, capture);
    require(captures == 2, "a binding generation change must recapture");
}
void graph_prerequisites() {
    GraphCache cache(1);
    cache.capture(key(), {owner()}, [](const GraphKey&) { return graph(); });
    int launches = 0;
    const GraphCache::Launch launch = [&](const GraphHandle&) { ++launches; return receipt(true); };
    const std::vector<std::shared_ptr<GraphReceipt>> unready{receipt()};
    rejects<std::logic_error>([&] { cache.replay(key(), launch, unready); }, "not ready");
    auto failed = receipt(true);
    failed->failure = std::make_exception_ptr(std::runtime_error("upload failed"));
    const std::vector<std::shared_ptr<GraphReceipt>> failed_upload{failed};
    rejects<std::runtime_error>([&] { cache.replay(key(), launch, failed_upload); }, "upload failed");
    const std::vector<std::shared_ptr<GraphReceipt>> missing{nullptr};
    rejects<std::invalid_argument>([&] { cache.replay(key(), launch, missing); }, "receipts");
    require(launches == 0 && !cache.failed(key()), "unready or failed inputs launched a graph");
    const std::vector<std::shared_ptr<GraphReceipt>> ready{receipt(true)};
    cache.replay(key(), launch, ready);
    require(launches == 1, "ready inputs did not launch");
}
void graph_failures() {
    GraphCache cache(1);
    const auto kept = owner();
    rejects<std::runtime_error>([&] {
        cache.capture(key(), {kept}, [](const GraphKey&) -> GraphHandle { throw std::runtime_error("capture failed"); });
    }, "capture failed");
    require(cache.quarantined().size() == 1 && cache.quarantined()[0][0] == kept, "failed capture released owners");
    rejects<std::logic_error>([&] { cache.capture(key(), {kept}, [](const GraphKey&) { return GraphHandle{}; }); },
                              "no graph");
    require(cache.quarantined().size() == 2 && !cache.contains(key()), "null capture result was cached");
    cache.capture(key(), {kept}, [](const GraphKey&) { return graph(); });
    int launches = 0;
    rejects<std::logic_error>([&] { cache.replay(key(), [&](const GraphHandle&) { ++launches; return nullptr; }); },
                              "completion receipt");
    require(cache.failed(key()), "receiptless launch did not block replay");
    rejects<std::logic_error>([&] { cache.replay(key(), [&](const GraphHandle&) { ++launches; return receipt(true); }); },
                              "backend recovery");
    rejects<std::logic_error>([&] { cache.evict(key(), [](const GraphHandle&) {}); }, "retained owners");
    require(launches == 1 && cache.owners(key()).size() == 1 && cache.owners(key())[0] == kept,
            "failed launch released owners or launched again");
    GraphCache thrown(1);
    thrown.capture(key(), {kept}, [](const GraphKey&) { return graph(); });
    rejects<std::runtime_error>([&] {
        thrown.replay(key(), [](const GraphHandle&) -> std::shared_ptr<GraphReceipt> { throw std::runtime_error("submitted"); });
    }, "submitted");
    rejects<std::logic_error>([&] { thrown.evict(key(), [](const GraphHandle&) {}); }, "retained owners");
}
void graph_receipts() {
    GraphCache cache(1);
    const auto kept = owner();
    cache.capture(key(), {kept}, [](const GraphKey&) { return graph(); });
    const auto pending = receipt();
    cache.replay(key(), [&](const GraphHandle&) { return pending; });
    for(int i = 0; i < 10000; ++i) cache.replay(key(), [](const GraphHandle&) { return receipt(true); });
    require(cache.in_flight(key()) == 2, "completed receipts were not pruned during reuse");
    rejects<std::logic_error>([&] { cache.evict(key(), [](const GraphHandle&) {}); }, "retained owners");
    pending->finished = true;
    cache.evict(key(), [](const GraphHandle&) {});
    require(cache.size() == 0, "settled graph was not evicted");
    cache.capture(key(), {kept}, [](const GraphKey&) { return graph(); });
    const auto uncertain = receipt();
    uncertain->uncertain = true;
    cache.replay(key(), [&](const GraphHandle&) { return uncertain; });
    int launches = 0;
    rejects<std::system_error>([&] { cache.replay(key(), [&](const GraphHandle&) { ++launches; return receipt(true); }); },
                               "cannot prove");
    require(launches == 0 && cache.in_flight(key()) == 1 && cache.owners(key())[0] == kept,
            "a failed completion probe dropped receipts or launched");
}
void graph_plan() {
    const MatrixLayout layout{2, 3, 4, MatrixElement::f32};
    StagedGraphPlan plan(layout, layout, 4, 2, 1);
    require(matrix_bytes(layout) == 32 && element_bytes(MatrixElement::bf16) == 2, "matrix extent");
    require(plan.staged_input() == MatrixLayout{4, 3, 3, MatrixElement::f32} && plan.phases() == 2, "staging layout");
    require(plan.input_rows().size() == 2 && plan.input_rows()[1].destination == 12 && plan.input_rows()[1].source == 16 &&
            plan.input_rows()[1].bytes == 12, "input row copy");
    require(plan.output_rows()[1].destination == 16 && plan.output_rows()[1].source == 12, "output row copy");
    auto narrow = layout;
    narrow.stride = 3;
    require(plan.compatible(layout, layout) && !plan.compatible(narrow, layout), "compatibility is exact layout");
    rejects<UnsupportedLayout>([&] { plan.check(narrow, 64, layout, 64); }, "layout differs");
    rejects<UnsupportedLayout>([&] { plan.check(layout, 31, layout, 32); }, "does not cover");
    plan.check(layout, 32, layout, 32);
    const MatrixLayout half{2, 3, 3, MatrixElement::f16};
    StagedGraphPlan mixed(layout, half, 2, 1, 0);
    require(mixed.output_rows()[1].destination == 6 && mixed.output_rows()[1].bytes == 6, "per-layout element size");
    rejects<UnsupportedLayout>([&] { StagedGraphPlan bad({2, 3, 2, MatrixElement::f32}, layout, 2, 1, 0); }, "stride");
    rejects<UnsupportedLayout>([&] { StagedGraphPlan bad({1, 3, 3, MatrixElement::f32}, layout, 2, 1, 0); }, "rows");
    rejects<UnsupportedLayout>([&] { StagedGraphPlan bad(layout, layout, 1, 1, 0); }, "rows");
    rejects<UnsupportedLayout>([&] { StagedGraphPlan bad(layout, layout, 2, 0, 0); }, "phases");
    rejects<UnsupportedLayout>([&] { StagedGraphPlan bad(layout, layout, 2, 65, 64); }, "phases");
    rejects<UnsupportedLayout>([&] { StagedGraphPlan bad(layout, layout, 2, 2, 0); }, "bridge");
    const auto huge = std::size_t(1) << (sizeof(std::size_t) * 8 - 2);
    rejects<UnsupportedLayout>([&] { matrix_bytes({huge, 1, 1, MatrixElement::f32}); }, "overflow");
    rejects<UnsupportedLayout>([&] { StagedGraphPlan bad(layout, layout, huge, 1, 0); }, "overflow");
}
void graph_state() {
    StagedGraphState state;
    std::vector<std::string> order;
    state.capture(3, [&](std::size_t i) { order.push_back("capture" + std::to_string(i)); });
    state.replay(3, [&] { order.push_back("prepare"); },
                 [&](std::size_t i) { order.push_back("launch" + std::to_string(i)); },
                 [&](std::size_t i) { order.push_back("bridge" + std::to_string(i)); },
                 [&] { order.push_back("finish"); });
    require(order == std::vector<std::string>{"capture0", "capture1", "capture2", "prepare", "launch0", "bridge0",
                                              "launch1", "bridge1", "launch2", "finish"}, "phase order");
    require(state.pending(), "submitted replay is not pending");
    const auto nothing = [] {};
    rejects<std::logic_error>([&] { state.replay(1, nothing, [](std::size_t) {}, [](std::size_t) {}, nothing); },
                              "in-flight");
    int settled = 0;
    require(!state.ready([] { return false; }, [&] { ++settled; }) && state.pending() && !settled, "unready probe");
    require(state.ready([] { return true; }, [&] { ++settled; }) && !state.pending() && settled == 1, "ready probe");
    require(state.ready([] { return false; }, [&] { ++settled; }) && settled == 1, "idle state probes nothing");
    state.replay(1, nothing, [](std::size_t) {}, [](std::size_t) {}, nothing);
    state.wait(nothing, [&] { ++settled; });
    require(!state.pending() && settled == 2, "wait settles");
    require(state.close([] {}) && state.closed() && state.close([] { throw std::runtime_error("again"); }), "close");
    rejects<std::logic_error>([&] { state.require_usable(); }, "closed");
}
void graph_state_failures() {
    const auto nothing = [] {};
    StagedGraphState bridge;
    int launches = 0;
    rejects<std::runtime_error>([&] {
        bridge.replay(2, nothing, [&](std::size_t) { ++launches; },
                      [](std::size_t) { throw std::runtime_error("bridge failed after enqueue"); }, nothing);
    }, "bridge failed");
    require(launches == 1 && bridge.failed() && bridge.pending(), "failed bridge did not poison and retain");
    rejects<std::logic_error>([&] { bridge.replay(1, nothing, [](std::size_t) {}, [](std::size_t) {}, nothing); }, "failed");
    rejects<std::logic_error>([&] { bridge.ready([] { return true; }, [] {}); }, "failed");
    rejects<std::logic_error>([&] { bridge.wait([] {}, [] {}); }, "failed");
    require(!bridge.close([] { throw std::runtime_error("drain failed"); }) && !bridge.closed() && bridge.pending(),
            "failed drain released owners");
    require(bridge.close([] {}) && bridge.closed() && !bridge.pending(), "retry after a failed drain");
    StagedGraphState capture;
    rejects<std::runtime_error>([&] {
        capture.capture(2, [](std::size_t i) { if(i == 1) throw std::runtime_error("capture failed"); });
    }, "capture failed");
    require(capture.failed() && !capture.pending(), "capture failure");
    StagedGraphState probe;
    probe.replay(1, nothing, [](std::size_t) {}, [](std::size_t) {}, nothing);
    rejects<std::runtime_error>([&] { probe.ready([]() -> bool { throw std::runtime_error("query failed"); }, [] {}); },
                                "query failed");
    require(probe.failed() && probe.pending(), "failed probe released owners");
    StagedGraphState settle;
    settle.replay(1, nothing, [](std::size_t) {}, [](std::size_t) {}, nothing);
    rejects<std::runtime_error>([&] { settle.wait([] { throw std::runtime_error("fence failed"); }, [] {}); },
                                "fence failed");
    require(settle.failed() && settle.pending(), "failed fence released owners");
}
}

int main(int argc, char** argv) {
    try {
        const std::string test = argc > 1 ? argv[1] : "all";
        const std::vector<std::pair<std::string, std::function<void()>>> tests{
            {"graph_key", graph_key}, {"graph_identity", graph_identity}, {"graph_generations", graph_generations},
            {"graph_prerequisites", graph_prerequisites}, {"graph_failures", graph_failures},
            {"graph_receipts", graph_receipts}, {"graph_plan", graph_plan}, {"graph_state", graph_state},
            {"graph_state_failures", graph_state_failures}};
        bool found = false;
        for(const auto& [name, run] : tests) if(test == "all" || test == name) {
            run();
            found = true;
            std::cout << "crit core graph passed: " << name << '\n';
        }
        require(found, "unknown test");
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
