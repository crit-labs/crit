// Device-resident expert cache directory: residency lookup, clock replacement,
// miss copies and a lookahead prefetch pool run as CUDA kernels, so a routed
// layer can be planned and filled inside a captured graph without a host round trip.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace crit::techniques::device_cache {

constexpr int kMaxRoutes = 32;  // routes planned per call (one warp)

// Device pointers owned by the engine. Keys are group * experts + expert; the host
// store holds one bundle per key in key order and must be device-visible (mapped
// pinned memory or another device-readable mirror).
struct Directory {
    std::int32_t* key_slot = nullptr;    // [groups * experts] -> slot or -1
    std::int32_t* slot_key = nullptr;    // [slots] -> key or -1
    std::uint8_t* slot_ref = nullptr;    // [slots] clock reference bits
    std::int32_t* slot_pin = nullptr;    // [slots] stamp of the plan that last used the slot
    std::int32_t* state = nullptr;       // [0] clock hand, [1] plan stamp, [2] failure (0 ok, 1 bad route, 2 no victim)
    unsigned long long* counters = nullptr;  // [0] hits, [1] misses, [2] misses served from the pool, [3] prefetched
    std::uint8_t* slots_base = nullptr;  // slots * bundle_bytes
    const std::uint8_t* store = nullptr; // device-visible host store, keys * bundle_bytes
    std::size_t bundle_bytes = 0;        // multiple of 16
    int slots = 0, groups = 0, experts = 0;
    // Optional lookahead pool: two halves (by group parity) of pool_slots bundles, tagged
    // with the key each holds (-1 empty).
    std::uint8_t* pool_base = nullptr;
    std::int32_t* pool_tags = nullptr;   // [2 * pool_slots]
    int pool_slots = 0;
};

// One route's resolved bundle.
struct Use {
    const std::uint8_t* bundle;  // device slot holding the expert (valid after the fill)
    std::int32_t key;            // -1 for a rejected route
    float weight;
};

// One bundle copy; src is the store or a pool slot, dst a cache slot or pool slot.
struct Fill {
    const uint4* src;
    uint4* dst;
};

// Plans `count` (<= kMaxRoutes) distinct routed experts of `group`: hits are pinned
// first, then misses take clock victims (pinned slots are skipped, referenced slots
// lose their bit), uses[i] names each expert's slot and fills/fill_count describe
// the misses' copies. A miss whose key is in the pool half for this group's parity
// is copied from the pool instead of the store. Directory state advances on device.
cudaError_t enqueue_plan(const Directory& d, int group, const std::int32_t* experts, const float* weights, int count,
                         Use* uses, Fill* fills, int* fill_count, cudaStream_t stream);
// Copies fills[0, *fill_count) (at most max_fills), blocks_per_bundle blocks per copy.
cudaError_t enqueue_fill(const Fill* fills, const int* fill_count, int max_fills, std::size_t bundle_bytes,
                         int blocks_per_bundle, cudaStream_t stream);
// Lookahead: tags the pool half for `group` with the predicted experts that are not
// resident and writes their store-to-pool copies. Reads the directory without
// changing residency; run the copies on a side stream and join before planning `group`.
cudaError_t enqueue_prefetch_plan(const Directory& d, int group, const std::int32_t* predicted, int count, Fill* fills,
                                  int* fill_count, cudaStream_t stream);

} // namespace crit::techniques::device_cache
