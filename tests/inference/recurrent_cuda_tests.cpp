// CUDA gated-delta reference against the FP64 host Recovery oracle, for the
// device commit strategies (per-column checkpoints, recompute from inputs and
// deferred replay of the accepted prefix at the next verify).
#include "recurrent_cuda.h"
#include "recurrent_recovery.h"
#include <crit/cuda.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace crit::core;
namespace rc = crit::techniques::recurrent;
namespace dc = crit::techniques::recurrent::cuda;

namespace {
// Stated tolerance: FP32 device versus FP64 host, |gpu - ref| <= 2e-5 + 2e-5 * |ref|.
constexpr double absolute_tolerance = 2e-5, relative_tolerance = 2e-5;
void check(cudaError_t s, const char* what) { if(s != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s)); }
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
struct Fixture {
    Resources resources{1 << 26, 1 << 28};
    std::shared_ptr<CudaBackend> backend = std::make_shared<CudaBackend>();
    Fixture() { backend->initialize(resources); }
    CudaBuffer put(const std::vector<float>& values) {
        auto b = backend->allocate(Memory::device, std::max<std::size_t>(values.size(), 1) * sizeof(float));
        if(!values.empty()) b.upload(std::as_bytes(std::span(values)), 0, Queue::compute);
        return b;
    }
    CudaBuffer zeros(std::size_t count) {
        auto b = backend->allocate(Memory::device, std::max<std::size_t>(count, 1) * sizeof(float));
        backend->fill_async(b, 0);
        return b;
    }
    std::vector<float> get(const CudaBuffer& b, std::size_t count) {
        std::vector<float> out(count);
        b.download(std::as_writable_bytes(std::span(out)), 0, Queue::compute);
        return out;
    }
};
double worst = 0;
void close(float gpu, double ref, const std::string& what) {
    const double error = std::abs(double(gpu) - ref);
    worst = std::max(worst, error / (1 + std::abs(ref)));
    require(std::isfinite(gpu) && error <= absolute_tolerance + relative_tolerance * std::abs(ref),
            what + ": gpu " + std::to_string(gpu) + " ref " + std::to_string(ref));
}

void oracle() {
    Fixture f;
    std::mt19937 random(77);
    std::uniform_real_distribution<float> unit(-1, 1), decay(-2, 0), gate(0, 1);
    std::size_t cases = 0;
    for(std::uint32_t heads : {1u, 3u})
    for(std::uint32_t keys : {1u, 4u, 16u, 64u})
    for(std::uint32_t values : {1u, 8u, 32u, 128u})
    for(std::uint32_t columns : {1u, 3u, 6u}) {
        dc::DeviceGeometry g{heads, keys, values, 1e-6f, 1.0f / std::sqrt(float(keys)), bool(random() & 1), bool(random() & 1)};
        const auto elements = dc::state_elements(g);
        std::vector<float> start(elements), q(columns * heads * keys), k(q.size()), v(columns * heads * values),
            ld(columns * heads), beta(columns * heads);
        for(auto& x : start) x = 0.5f * unit(random);
        for(auto& x : q) x = unit(random);
        for(auto& x : k) x = unit(random);
        for(auto& x : v) x = unit(random);
        for(auto& x : ld) x = decay(random);
        for(auto& x : beta) x = gate(random);
        auto d_start = f.put(start), d_q = f.put(q), d_k = f.put(k), d_v = f.put(v), d_ld = f.put(ld), d_beta = f.put(beta);
        dc::DeviceSteps steps{static_cast<const float*>(d_q.data()), static_cast<const float*>(d_k.data()),
                              static_cast<const float*>(d_v.data()), static_cast<const float*>(d_ld.data()),
                              static_cast<const float*>(d_beta.data())};
        auto d_out_a = f.zeros(columns * heads * values), d_out_b = f.zeros(columns * heads * values);
        auto d_ckpt = f.zeros(columns * elements), d_work = f.zeros(elements);
        const auto stream = f.backend->stream();
        check(dc::enqueue_verify(g, static_cast<const float*>(d_start.data()), steps, columns,
                                 static_cast<float*>(d_out_a.data()), static_cast<float*>(d_ckpt.data()), nullptr, stream), "verify checkpoints");
        check(dc::enqueue_verify(g, static_cast<const float*>(d_start.data()), steps, columns,
                                 static_cast<float*>(d_out_b.data()), nullptr, static_cast<float*>(d_work.data()), stream), "verify recompute");
        const auto out_a = f.get(d_out_a, columns * heads * values), out_b = f.get(d_out_b, columns * heads * values);
        require(std::memcmp(out_a.data(), out_b.data(), out_a.size() * sizeof(float)) == 0, "strategy outputs differ bitwise");
        auto d_count = f.backend->allocate(Memory::device, sizeof(std::int32_t));
        auto d_state_a = f.zeros(elements), d_state_b = f.zeros(elements);
        for(std::uint32_t accepted = 0; accepted <= columns; ++accepted) {
            // The count lives on the device, as a round's control[n_emitted] would.
            const std::int32_t n = std::int32_t(accepted);
            d_count.upload(std::as_bytes(std::span(&n, 1)), 0, Queue::compute);
            check(dc::enqueue_select_accepted(g, static_cast<const float*>(d_start.data()), static_cast<const float*>(d_ckpt.data()),
                                              static_cast<const std::int32_t*>(d_count.data()), columns, static_cast<float*>(d_state_a.data()), stream), "select");
            check(dc::enqueue_recompute_accepted(g, static_cast<const float*>(d_start.data()), steps,
                                                 static_cast<const std::int32_t*>(d_count.data()), columns, static_cast<float*>(d_state_b.data()), stream), "recompute");
            const auto a = f.get(d_state_a, elements), b = f.get(d_state_b, elements);
            require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, "checkpoint and recompute commits differ bitwise");
            if(!accepted) require(std::memcmp(a.data(), start.data(), a.size() * sizeof(float)) == 0, "zero accepted must keep the start state");
            for(std::uint32_t h = 0; h < heads; ++h) {
                const auto per_head = std::size_t(keys) * values;
                rc::Geometry hg{keys, values, double(g.epsilon), double(g.query_scale), g.normalize_keys, g.normalize_queries};
                rc::Snapshot initial{100, std::vector<double>(start.begin() + h * per_head, start.begin() + (h + 1) * per_head), {}};
                rc::Recovery host(hg, initial);
                std::vector<rc::Node> nodes;
                std::vector<std::size_t> path;
                for(std::uint32_t t = 0; t < columns; ++t) {
                    const auto e = std::size_t(t) * heads + h;
                    rc::Step step{{q.begin() + e * keys, q.begin() + (e + 1) * keys}, {k.begin() + e * keys, k.begin() + (e + 1) * keys},
                                  {v.begin() + e * values, v.begin() + (e + 1) * values}, double(ld[e]), double(beta[e])};
                    nodes.push_back({t ? t - 1 : rc::no_parent, std::move(step)});
                    path.push_back(t);
                }
                const auto prepared = host.prepare(nodes, path);
                for(std::uint32_t t = 0; t < columns; ++t)
                    for(std::uint32_t x = 0; x < values; ++x)
                        close(out_a[(std::size_t(t) * heads + h) * values + x], prepared.outputs[std::size_t(t) * values + x], "verify output");
                const auto committed = host.commit(prepared.id, accepted);
                require(committed.state.after_tokens == 100 + accepted, "oracle boundary");
                for(std::size_t i = 0; i < per_head; ++i) close(a[h * per_head + i], committed.state.matrix[i], "accepted state");
            }
        }
        ++cases;
    }
    std::cout << "cases " << cases << " worst scaled error " << worst << " tolerance " << absolute_tolerance << " + "
              << relative_tolerance << "*|ref|\n";
}

// A recorded prefix is applied exactly once: a flush (zero columns) between rounds,
// a repeated flush and a verified round that is never recorded must not replay it
// again. Compared bitwise with recompute commits from separate round-start states.
void deferred_consumed() {
    Fixture f;
    std::mt19937 random(53);
    std::uniform_real_distribution<float> unit(-1, 1), decay(-2, 0), gate(0, 1);
    constexpr std::uint32_t columns = 3;
    const dc::DeviceGeometry g{2, 4, 8, 1e-6f, 0.5f, true, true};
    const auto elements = dc::state_elements(g);
    std::vector<float> start(elements);
    for(auto& x : start) x = 0.5f * unit(random);
    std::vector<CudaBuffer> inputs;
    auto round_steps = [&] {
        std::vector<float> q(columns * g.heads * g.keys), k(q.size()), v(columns * g.heads * g.values), ld(columns * g.heads), beta(ld.size());
        for(auto& x : q) x = unit(random);
        for(auto& x : k) x = unit(random);
        for(auto& x : v) x = unit(random);
        for(auto& x : ld) x = decay(random);
        for(auto& x : beta) x = gate(random);
        for(auto* x : {&q, &k, &v, &ld, &beta}) inputs.push_back(f.put(*x));
        const auto* b = &inputs[inputs.size() - 5];
        return dc::DeviceSteps{static_cast<const float*>(b[0].data()), static_cast<const float*>(b[1].data()),
                               static_cast<const float*>(b[2].data()), static_cast<const float*>(b[3].data()),
                               static_cast<const float*>(b[4].data())};
    };
    inputs.reserve(10); // the returned pointers stay valid
    const auto first = round_steps(), second = round_steps();
    const auto stream = f.backend->stream();
    auto count = [&](std::int32_t n) {
        auto b = f.backend->allocate(Memory::device, sizeof(std::int32_t));
        b.upload(std::as_bytes(std::span(&n, 1)), 0, Queue::compute);
        return b;
    };
    auto two = count(2), one = count(1);
    const auto* two_ptr = static_cast<const std::int32_t*>(two.data());
    const auto* one_ptr = static_cast<const std::int32_t*>(one.data());
    // Oracle: start, then 2 accepted of the first round, then 1 accepted of the second.
    auto s0 = f.put(start), s1 = f.zeros(elements), s2 = f.zeros(elements);
    check(dc::enqueue_recompute_accepted(g, static_cast<const float*>(s0.data()), first, two_ptr, columns, static_cast<float*>(s1.data()), stream), "oracle 1");
    check(dc::enqueue_recompute_accepted(g, static_cast<const float*>(s1.data()), second, one_ptr, columns, static_cast<float*>(s2.data()), stream), "oracle 2");
    const auto after_first = f.get(s1, elements), after_second = f.get(s2, elements);
    auto work = f.zeros(elements), out = f.zeros(columns * g.heads * g.values);
    auto stash_key = f.zeros(columns * g.heads * g.keys), stash_decay = f.zeros(columns * g.heads),
         stash_delta = f.zeros(columns * g.heads * g.values);
    const dc::DeferredStash stash{static_cast<float*>(stash_key.data()), static_cast<float*>(stash_decay.data()),
                                  static_cast<float*>(stash_delta.data()), columns};
    auto pending = f.backend->allocate(Memory::device, sizeof(std::int32_t));
    auto* pending_ptr = static_cast<std::int32_t*>(pending.data());
    for(const bool abandon : {false, true}) {
        auto live = f.put(start);
        auto* live_ptr = static_cast<float*>(live.data());
        f.backend->fill_async(pending, 0);
        auto verify = [&](const dc::DeviceSteps& steps) {
            check(dc::enqueue_verify_deferred(g, live_ptr, steps, columns, pending_ptr, stash, static_cast<float*>(out.data()),
                                              static_cast<float*>(work.data()), stream), "verify deferred");
        };
        auto flush = [&] { check(dc::enqueue_verify_deferred(g, live_ptr, {}, 0, pending_ptr, stash, nullptr, nullptr, stream), "flush"); };
        auto expect = [&](const std::vector<float>& ref, const char* what) {
            const auto l = f.get(live, elements);
            require(std::memcmp(l.data(), ref.data(), l.size() * sizeof(float)) == 0, what);
        };
        verify(first);
        check(dc::enqueue_record_accepted(two_ptr, columns, pending_ptr, stream), "record");
        if(abandon) {
            verify(second); // never recorded: its stash must not replay the first round's count
            flush();
            expect(after_first, "abandoned round replayed the previous prefix");
        } else {
            flush();
            expect(after_first, "flush did not apply the recorded prefix");
            flush();
            expect(after_first, "second flush replayed the prefix");
        }
        verify(second);
        check(dc::enqueue_record_accepted(one_ptr, columns, pending_ptr, stream), "record");
        flush();
        expect(after_second, abandon ? "round after an abandoned round differs from recompute" : "round after a flush differs from recompute");
    }
    std::cout << "deferred prefix applied once across flushes and an abandoned round\n";
}

// The many-count record writes the clamped count into `count` ints of every group,
// `stride` apart, and nothing else (recurrent-layer-kernels' per-block counts).
void record_many() {
    Fixture f;
    const auto stream = f.backend->stream();
    auto pending = f.backend->allocate(Memory::device, 10 * sizeof(std::int32_t)), accepted = f.backend->allocate(Memory::device, sizeof(std::int32_t));
    auto* pending_ptr = static_cast<std::int32_t*>(pending.data());
    const auto* accepted_ptr = static_cast<const std::int32_t*>(accepted.data());
    for(const auto& [n, want] : {std::pair{4, 3}, std::pair{-2, 0}, std::pair{2, 2}}) {
        std::vector<std::int32_t> sentinel(10, -7), got(10);
        pending.upload(std::as_bytes(std::span(sentinel)), 0, Queue::compute);
        accepted.upload(std::as_bytes(std::span(&n, 1)), 0, Queue::compute);
        check(dc::enqueue_record_accepted(accepted_ptr, 3, pending_ptr, 3, 2, 5, stream), "record many");
        pending.download(std::as_writable_bytes(std::span(got)), 0, Queue::compute);
        for(int i = 0; i < 10; ++i)
            require(got[std::size_t(i)] == (i % 5 < 3 ? want : -7), "record many wrote " + std::to_string(got[std::size_t(i)]) + " at " + std::to_string(i));
    }
    require(dc::enqueue_record_accepted(accepted_ptr, 3, pending_ptr, 3, 2, 2, stream) == cudaErrorInvalidValue, "overlapping groups recorded");
    require(dc::enqueue_record_accepted(accepted_ptr, 3, pending_ptr, 0, 1, 0, stream) == cudaErrorInvalidValue, "zero counts recorded");
    std::cout << "many-count record writes every group's counts and nothing else\n";
}

// Deferred commit over many rounds, including reruns with a smaller count after a
// truncation, must reproduce the checkpoint strategy bit for bit.
void deferred() {
    Fixture f;
    std::mt19937 random(91);
    std::uniform_real_distribution<float> unit(-1, 1), decay(-2, 0), gate(0, 1);
    constexpr std::uint32_t capacity = 6, rounds = 12;
    std::size_t checked = 0;
    for(std::uint32_t heads : {1u, 3u})
    for(std::uint32_t keys : {4u, 64u})
    for(std::uint32_t values : {8u, 128u}) {
        dc::DeviceGeometry g{heads, keys, values, 1e-6f, 1.0f / std::sqrt(float(keys)), bool(random() & 1), bool(random() & 1)};
        const auto elements = dc::state_elements(g);
        std::vector<float> start(elements);
        for(auto& x : start) x = 0.5f * unit(random);
        auto reference = f.put(start), live = f.put(start), next = f.zeros(elements), work = f.zeros(elements);
        auto checkpoints = f.zeros(capacity * elements);
        auto out_a = f.zeros(capacity * heads * values), out_c = f.zeros(capacity * heads * values);
        auto stash_key = f.zeros(capacity * heads * keys), stash_decay = f.zeros(capacity * heads),
             stash_delta = f.zeros(capacity * heads * values);
        dc::DeferredStash stash{static_cast<float*>(stash_key.data()), static_cast<float*>(stash_decay.data()),
                                static_cast<float*>(stash_delta.data()), capacity};
        auto count = f.backend->allocate(Memory::device, sizeof(std::int32_t));
        auto pending = f.backend->allocate(Memory::device, sizeof(std::int32_t));
        f.backend->fill_async(pending, 0);
        const auto stream = f.backend->stream();
        auto* pending_ptr = static_cast<std::int32_t*>(pending.data());
        auto* count_ptr = static_cast<std::int32_t*>(count.data());
        for(std::uint32_t round = 0; round < rounds; ++round) {
            const std::uint32_t columns = 1 + std::uint32_t(random() % capacity);
            std::vector<float> q(columns * heads * keys), k(q.size()), v(columns * heads * values), ld(columns * heads), beta(columns * heads);
            for(auto& x : q) x = unit(random);
            for(auto& x : k) x = unit(random);
            for(auto& x : v) x = unit(random);
            for(auto& x : ld) x = decay(random);
            for(auto& x : beta) x = gate(random);
            auto d_q = f.put(q), d_k = f.put(k), d_v = f.put(v), d_ld = f.put(ld), d_beta = f.put(beta);
            dc::DeviceSteps steps{static_cast<const float*>(d_q.data()), static_cast<const float*>(d_k.data()),
                                  static_cast<const float*>(d_v.data()), static_cast<const float*>(d_ld.data()),
                                  static_cast<const float*>(d_beta.data())};
            check(dc::enqueue_verify(g, static_cast<const float*>(reference.data()), steps, columns,
                                     static_cast<float*>(out_a.data()), static_cast<float*>(checkpoints.data()), nullptr, stream), "verify checkpoints");
            check(dc::enqueue_verify_deferred(g, static_cast<float*>(live.data()), steps, columns, pending_ptr, stash,
                                              static_cast<float*>(out_c.data()), static_cast<float*>(work.data()), stream), "verify deferred");
            const auto a = f.get(out_a, columns * heads * values), c = f.get(out_c, columns * heads * values);
            require(std::memcmp(a.data(), c.data(), a.size() * sizeof(float)) == 0, "deferred verify outputs differ bitwise");
            // Commit, then sometimes rerun with a smaller count as after a host truncation.
            std::int32_t accepted = std::int32_t(random() % (columns + 1));
            count.upload(std::as_bytes(std::span(&accepted, 1)), 0, Queue::compute);
            check(dc::enqueue_record_accepted(count_ptr, columns, pending_ptr, stream), "record");
            if(accepted && (random() & 1)) {
                accepted = std::int32_t(random() % std::uint32_t(accepted));
                count.upload(std::as_bytes(std::span(&accepted, 1)), 0, Queue::compute);
                check(dc::enqueue_record_accepted(count_ptr, columns, pending_ptr, stream), "record rerun");
            }
            check(dc::enqueue_select_accepted(g, static_cast<const float*>(reference.data()), static_cast<const float*>(checkpoints.data()),
                                              count_ptr, columns, static_cast<float*>(next.data()), stream), "select");
            std::swap(reference, next);
            ++checked;
        }
        // Zero columns applies only the pending prefix.
        check(dc::enqueue_verify_deferred(g, static_cast<float*>(live.data()), {}, 0, pending_ptr, stash, nullptr, nullptr, stream), "flush");
        const auto r = f.get(reference, elements), l = f.get(live, elements);
        require(std::memcmp(r.data(), l.data(), r.size() * sizeof(float)) == 0, "deferred live state differs bitwise from checkpoint commits");
    }
    std::cout << "deferred rounds " << checked << " bitwise equal to checkpoint commits\n";
    deferred_consumed();
    record_many();
}

void rejects() {
    Fixture f;
    auto buffer = f.zeros(16);
    auto* p = static_cast<float*>(buffer.data());
    dc::DeviceSteps steps{p, p, p, p, p};
    std::int32_t* count = nullptr;
    require(dc::enqueue_verify({1, 2, 0}, p, steps, 1, p, p, nullptr, f.backend->stream()) == cudaErrorInvalidValue, "zero values accepted");
    require(dc::enqueue_verify({1, 2, 2000}, p, steps, 1, p, p, nullptr, f.backend->stream()) == cudaErrorInvalidValue, "too many value columns accepted");
    require(dc::enqueue_verify({1, 2, 2}, p, steps, 1, p, nullptr, nullptr, f.backend->stream()) == cudaErrorInvalidValue, "missing work buffer accepted");
    require(dc::enqueue_select_accepted({1, 2, 2}, p, p, count, 1, p, f.backend->stream()) == cudaErrorInvalidValue, "missing count accepted");
    require(dc::enqueue_recompute_accepted({1, 2, 2}, p, steps, reinterpret_cast<std::int32_t*>(p), 1, p, f.backend->stream()) == cudaErrorInvalidValue,
            "aliased recompute state accepted");
    auto* n = reinterpret_cast<std::int32_t*>(p);
    require(dc::enqueue_verify_deferred({1, 2, 2}, p, steps, 3, n, {p, p, p, 2}, p, p, f.backend->stream()) == cudaErrorInvalidValue,
            "deferred columns above stash capacity accepted");
    require(dc::enqueue_verify_deferred({1, 2, 2}, p, steps, 1, n, {p, p, p, 2}, p, nullptr, f.backend->stream()) == cudaErrorInvalidValue,
            "deferred verify without work buffer accepted");
    require(dc::enqueue_record_accepted(nullptr, 1, n, f.backend->stream()) == cudaErrorInvalidValue, "missing accepted count recorded");
}
}

int main(int argc, char** argv) {
    try {
        if(argc != 2) throw std::invalid_argument("test group required");
        const std::string group = argv[1];
        if(group == "oracle") oracle();
        else if(group == "deferred") deferred();
        else if(group == "rejects") rejects();
        else throw std::invalid_argument("unknown test group");
        std::cout << "PASS " << group << '\n';
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
