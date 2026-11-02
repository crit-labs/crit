// A ready device expert cache around device_cache.h: it owns the directory arrays and the per-group plan scratch,
// and gives an engine one capturable call per routed layer (plan, then copy the admitted misses), a warm start from
// a residency profile, the admission limits and the telemetry Eval reads (expert_reads, expert_hits). The engine
// keeps owning the slots (lent memory included), the pool and the device-visible store, and its streams.
// One source for CUDA and HIP, as device_cache.cu.
#pragma once
#include "device_cache.h"

namespace crit::techniques::device_cache {

struct Geometry {
    int slots = 0, groups = 0, experts = 0, pool_slots = 0;
    std::size_t bundle_bytes = 0;  // multiple of 16
};

// The directory's counters and failure flag. expert_reads and expert_hits are the decode telemetry Eval prices the
// expert tiers with; bypassed counts misses the admission limits sent to the store (computed on the CPU or read
// mapped).
struct Telemetry {
    unsigned long long hits = 0, misses = 0, pooled = 0, prefetched = 0, bypassed = 0;
    int failure = 0;  // state[2]: 0 ok, 1 invalid or duplicate route, 2 more routes than free slots
    unsigned long long expert_reads() const { return hits + misses; }
    unsigned long long expert_hits() const { return hits; }
};

class ExpertCache {
public:
    ExpertCache() = default;
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;
    ~ExpertCache() { release(); }

    // Allocates the directory and clears it on `stream` (every slot empty, no limits). slots_base holds
    // g.slots bundles, pool_base 2 * g.pool_slots (or null without a pool), store one bundle per key.
    cudaError_t init(const Geometry& g, std::uint8_t* slots_base, const std::uint8_t* store, std::uint8_t* pool_base,
                     cudaStream_t stream);
    void release();

    // Admission limits (Directory::admit_quota, admit_by_misses, key_seen): quota -1 admits every miss; table, when
    // given, is a host array of kMaxRoutes + 1 quotas by miss count (split.py quota_table); second_touch admits a key
    // on its second miss. Set between requests, not while a graph that plans is being captured.
    cudaError_t set_admission(int quota, const std::int32_t* table, bool second_touch, cudaStream_t stream);

    // Warm start before the first plan: keys is a host array in a residency profile's order
    // (residency_profile.py load_profile). Copies their bundles and synchronizes `stream`.
    cudaError_t preload(const std::int32_t* keys, int count, int blocks_per_bundle, cudaStream_t stream);

    // One routed layer: plans `count` distinct routes of `group` (device arrays; weights may be null) and copies its
    // admitted misses. No allocation or synchronization, so it can be captured. *uses (device, count entries) stays
    // valid until the next plan of the same group; a Use with Source::store is a bypassed miss.
    cudaError_t plan_and_fetch(int group, const std::int32_t* experts, const float* weights, int count,
                               int blocks_per_bundle, cudaStream_t stream, const Use** uses);

    // Lookahead into the pool half of group's parity, on a side stream; join it before planning `group`. A copy
    // made with SMs on a side stream may only run between the main stream's kernels: measure it before enabling.
    cudaError_t prefetch(int group, const std::int32_t* predicted, int count, int blocks_per_bundle, cudaStream_t side);

    // Reads the counters and the failure flag; synchronizes `stream`, so call it outside capture (between requests,
    // or after a step whose output depends on the flag).
    cudaError_t telemetry(Telemetry& out, cudaStream_t stream) const;

    const Directory& directory() const { return d_; }

private:
    Directory d_{};
    std::uint8_t* seen_ = nullptr;       // [groups * experts] second-touch marks
    std::int32_t* table_ = nullptr;      // [kMaxRoutes + 1]
    Use* uses_ = nullptr;                // [groups * kMaxRoutes]
    Fill* fills_ = nullptr;              // [groups * kMaxRoutes]
    Fill* pool_fills_ = nullptr;         // [2 * pool_slots]
    int* fill_counts_ = nullptr;         // [groups] plans, [groups + 0/1] pool halves
};

} // namespace crit::techniques::device_cache
