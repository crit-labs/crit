// Device-resident expert cache directory: residency lookup, clock replacement,
// miss copies and a lookahead prefetch pool run as GPU kernels, so a routed
// layer can be planned and filled inside a captured graph without a host round trip.
// One source for CUDA and HIP: built by hipcc (or with CRIT_CORE_HIP) the runtime names below are HIP's, through
// Core's crit/gpu_runtime.h. Metal has the same kernels in device_cache.metal.
#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__HIPCC__) && !defined(CRIT_CORE_HIP)
#define CRIT_CORE_HIP 1
#endif
#if defined(CRIT_CORE_HIP)
#include "crit/gpu_runtime.h"
#else
#include <cuda_runtime.h>
#endif

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
    // Optional admission limits; with neither set every miss is admitted (copied over the link into a slot).
    // admit_quota >= 0 admits at most that many misses per plan, in route order (routers list their experts by
    // weight, so the heaviest misses take the link); admit_by_misses [kMaxRoutes + 1], when set, gives the quota by
    // the plan's miss count instead (split.py derives it from measured link and CPU times). key_seen
    // [groups * experts] turns on second-touch admission: a key's first miss bypasses and marks it, a later one is
    // admitted. A bypassed miss takes no victim and evicts nothing; its Use names the store bundle (Source::store),
    // which the engine reads mapped or computes on the CPU. With a limit set, counters needs a fifth entry:
    // [4] bypassed misses.
    int admit_quota = -1;
    const std::int32_t* admit_by_misses = nullptr;
    std::uint8_t* key_seen = nullptr;
};

// Where a route's bundle is: a cache slot, or the store for a miss the admission limits bypassed.
enum class Source : std::int32_t { slot = 0, store = 1 };

// One route's resolved bundle.
struct Use {
    const std::uint8_t* bundle;  // device slot holding the expert (valid after the fill), or its store bundle
    std::int32_t key;            // -1 for a rejected route
    float weight;
    Source source = Source::slot;
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
// resident and writes their store-to-pool copies. Pool slots go to the valid, distinct,
// nonresident predictions in prediction order, at most pool_slots of them, so `fills`
// holds pool_slots entries and `count` may exceed the pool. Reads the directory without
// changing residency; run the copies on a side stream and join before planning `group`.
cudaError_t enqueue_prefetch_plan(const Directory& d, int group, const std::int32_t* predicted, int count, Fill* fills,
                                  int* fill_count, cudaStream_t stream);
// Warm start: maps keys[i] (device array, a residency profile's order) to slot i for the first min(count, slots)
// valid, distinct keys and writes their store-to-slot copies (fills holds min(count, slots) entries). Run it on an
// empty directory before the first plan; slots it leaves free and the replacement policy work as usual afterwards.
// Preloaded slots start unreferenced, so a profile entry no route uses is the clock's first victim.
cudaError_t enqueue_preload(const Directory& d, const std::int32_t* keys, int count, Fill* fills, int* fill_count,
                            cudaStream_t stream);

// The operands a provider's grouped GEMM takes for a plan's routes (CUTLASS's ptr-array grouped GEMM,
// cublasGemmGroupedBatchedEx, a block-scaled grouped GEMM): per route r, a pointer to each component inside the
// route's bundle (its slot, or the store for a bypassed miss), to its activation row and to its output row, and its
// routing weight. The expert arithmetic stays the provider's; this only addresses it, on the device, so a captured
// step needs no host round trip between the plan and the GEMM.
constexpr int kMaxComponents = 4;
struct OperandSpec {
    int components = 0;                            // 1 .. kMaxComponents
    std::size_t offsets[kMaxComponents] = {};      // component offsets inside a bundle (expert_store.py's index)
    const std::uint8_t* x = nullptr;               // activations; route r reads row token[r] (row r without token)
    std::size_t x_row_bytes = 0;
    const std::int32_t* token = nullptr;           // optional [count] activation row per route
    std::uint8_t* y = nullptr;                     // outputs; route r writes row r
    std::size_t y_row_bytes = 0;
    bool skip_bypassed = false;                    // null operands for Source::store routes (the CPU computes them)
};
// Device arrays of kMaxRoutes entries each; a rejected (or skipped) route gets null pointers and weight 0.
struct Operands {
    const void** matrix[kMaxComponents] = {};
    const void** x = nullptr;
    void** y = nullptr;
    float* weight = nullptr;
};
cudaError_t enqueue_operands(const Use* uses, int count, const OperandSpec& spec, const Operands& out, cudaStream_t stream);

} // namespace crit::techniques::device_cache
