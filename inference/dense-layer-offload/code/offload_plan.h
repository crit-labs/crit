// Layer placement for a dense model whose layer weights exceed the device budget: each layer is resident on the
// device, streamed (uploaded into one of two device staging slots every step, overlapping the compute before it), or
// executed on the host from host memory. The planner takes only measurements: the device's read bandwidth and free
// memory (crit eval probe), the pinned upload bandwidth and the host's GEMV rate on this machine, and fixed per-layer
// costs of moving activations; it predicts a step's time for each plan and returns the fastest. Engines then time the
// leading plans for a few real steps and keep the measured fastest (select_measured).
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace crit::dense_offload {

enum class Place : std::uint8_t { device, stream, host };

struct Measurements {
    double device_gbs = 0;         // device bytes read per ns (GB/s) by a decode step's projections
    double upload_gbs = 0;         // pinned host -> device copy bandwidth (GB/s)
    double host_gbs = 0;           // host GEMV weight bytes per ns (GB/s) at the step's batch
    double host_layer_ns = 0;      // fixed cost per run of host layers: activations down and back, one sync
};

struct Plan {
    std::vector<Place> places;     // one per layer
    std::uint64_t device_bytes = 0;   // resident layers plus the staging slots
    std::uint64_t staging_bytes = 0;  // two slots of the largest streamed layer (0 without streaming)
    double predicted_ns = 0;
    const char* name = "";
};

// Predicted step time of a placement: resident layers cost bytes / device_gbs; a streamed layer costs the larger of
// its compute and its upload (the upload of the next streamed layer overlaps the layers before it); consecutive host
// layers cost bytes / host_gbs plus one host_layer_ns per run.
inline double predict(const std::vector<std::uint64_t>& bytes, const std::vector<Place>& places, const Measurements& m) {
    double ns = 0;
    for(std::size_t i = 0; i < bytes.size(); ++i) {
        const double b = static_cast<double>(bytes[i]);
        switch(places[i]) {
        case Place::device: ns += b / m.device_gbs; break;
        case Place::stream: ns += std::max(b / m.device_gbs, b / m.upload_gbs); break;
        case Place::host: ns += b / m.host_gbs + ((i == 0 || places[i - 1] != Place::host) ? m.host_layer_ns : 0.0); break;
        }
    }
    return ns;
}

namespace detail {
// The layers to keep resident within `budget`: largest benefit per byte first (time saved against `other`).
inline std::vector<Place> fill(const std::vector<std::uint64_t>& bytes, std::uint64_t budget, Place other, const Measurements& m) {
    std::vector<std::size_t> order(bytes.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    const auto cost = [&](std::size_t i, Place p) { std::vector<Place> one(1, p); return predict({bytes[i]}, one, m); };
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return (cost(a, other) - cost(a, Place::device)) / static_cast<double>(bytes[a]) >
               (cost(b, other) - cost(b, Place::device)) / static_cast<double>(bytes[b]);
    });
    std::vector<Place> places(bytes.size(), other);
    std::uint64_t used = 0;
    for(const auto i : order)
        if(used + bytes[i] <= budget) { places[i] = Place::device; used += bytes[i]; }
    return places;
}
inline std::uint64_t resident(const std::vector<std::uint64_t>& bytes, const std::vector<Place>& places) {
    std::uint64_t sum = 0;
    for(std::size_t i = 0; i < bytes.size(); ++i) if(places[i] == Place::device) sum += bytes[i];
    return sum;
}
}  // namespace detail

// Candidate plans ordered by predicted time (fastest first): all resident when it fits, host execution of the layers
// that do not fit, and streaming of them through two staging slots taken from the budget.
inline std::vector<Plan> candidates(const std::vector<std::uint64_t>& bytes, std::uint64_t budget, const Measurements& m) {
    if(bytes.empty() || m.device_gbs <= 0 || m.upload_gbs <= 0 || m.host_gbs <= 0)
        throw std::invalid_argument("layer offload planning needs layers and positive measured rates");
    std::vector<Plan> plans;
    const std::uint64_t total = std::accumulate(bytes.begin(), bytes.end(), std::uint64_t{0});
    if(total <= budget) {
        Plan p{std::vector<Place>(bytes.size(), Place::device), total, 0, 0, "resident"};
        p.predicted_ns = predict(bytes, p.places, m);
        plans.push_back(p);
    }
    Plan host{detail::fill(bytes, budget, Place::host, m), 0, 0, 0, "host"};
    host.device_bytes = detail::resident(bytes, host.places);
    host.predicted_ns = predict(bytes, host.places, m);
    if(total > budget) plans.push_back(host);
    // Streaming needs two slots of the largest streamed layer; shrink the resident set until they fit.
    std::uint64_t slot = 0;
    for(int pass = 0; pass < 64; ++pass) {
        if(2 * slot > budget) break;
        Plan stream{detail::fill(bytes, budget - 2 * slot, Place::stream, m), 0, 0, 0, "stream"};
        std::uint64_t largest = 0;
        for(std::size_t i = 0; i < bytes.size(); ++i) if(stream.places[i] == Place::stream) largest = std::max(largest, bytes[i]);
        if(largest > slot) { slot = largest; continue; }
        stream.staging_bytes = largest ? 2 * largest : 0;
        stream.device_bytes = detail::resident(bytes, stream.places) + stream.staging_bytes;
        stream.predicted_ns = predict(bytes, stream.places, m);
        if(largest) plans.push_back(stream);
        break;
    }
    std::stable_sort(plans.begin(), plans.end(), [](const Plan& a, const Plan& b) { return a.predicted_ns < b.predicted_ns; });
    return plans;
}

// Times every candidate predicted within `slack` of the fastest with `run` (which executes `steps` real steps under a
// plan and returns their total measured ns) and returns the measured fastest; predictions only rank.
inline Plan select_measured(const std::vector<Plan>& plans, double slack, int steps, const std::function<double(const Plan&, int)>& run) {
    if(plans.empty()) throw std::invalid_argument("no candidate plan");
    Plan best = plans.front();
    double best_ns = std::numeric_limits<double>::infinity();
    for(const auto& plan : plans) {
        if(plan.predicted_ns > plans.front().predicted_ns * slack) break;
        const double ns = run(plan, steps);
        if(ns < best_ns) { best_ns = ns; best = plan; }
    }
    return best;
}

}  // namespace crit::dense_offload
