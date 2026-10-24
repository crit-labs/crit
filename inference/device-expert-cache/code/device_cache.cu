#include "device_cache.h"

namespace crit::techniques::device_cache {
namespace {

constexpr int kFillThreads = 256;

__device__ __forceinline__ const std::uint8_t* pool_source(const Directory& d, int group, int key) {
    if(!d.pool_slots) return nullptr;
    const int half = group & 1;
    const std::int32_t* tags = d.pool_tags + half * d.pool_slots;
    for(int p = 0; p < d.pool_slots; ++p)
        if(tags[p] == key) return d.pool_base + (std::size_t(half) * d.pool_slots + p) * d.bundle_bytes;
    return nullptr;
}

// One warp. Lanes resolve hits in parallel and pin them; misses are then admitted in
// route order by a warp-parallel clock scan that gives the same victims as scanning
// one slot at a time (the first slot neither pinned nor referenced; referenced slots
// passed on the way lose their bit).
__global__ void plan_kernel(Directory d, int group, const std::int32_t* experts, const float* weights, int count, Use* uses,
                            Fill* fills, int* fill_count) {
    __shared__ int s_slot[kMaxRoutes], s_key[kMaxRoutes], s_stamp, s_hand;
    const int r = threadIdx.x;
    if(r == 0) {
        s_stamp = ++d.state[1];
        s_hand = d.state[0];
    }
    __syncwarp();
    const int stamp = s_stamp;
    int key = -1, slot = -1;
    if(r < count) {
        const int e = experts[r];
        if(e < 0 || e >= d.experts || group < 0 || group >= d.groups) d.state[2] = 1;
        else key = group * d.experts + e;
    }
    s_key[r] = key;
    __syncwarp();
    // duplicate routes would be admitted twice: reject them
    if(key >= 0)
        for(int q = 0; q < r; ++q)
            if(s_key[q] == key) { d.state[2] = 1; key = -1; }
    __syncwarp();
    s_key[r] = key;
    if(key >= 0) {
        slot = d.key_slot[key];
        if(slot >= 0) { d.slot_ref[slot] = 1; d.slot_pin[slot] = stamp; }
    }
    s_slot[r] = slot;
    __syncwarp();
    int nfill = 0;
    unsigned long long hits = 0, misses = 0, pooled = 0;
    for(int q = 0; q < count; ++q) {
        if(s_key[q] < 0) continue;
        if(s_slot[q] >= 0) { ++hits; continue; }
        ++misses;
        const int k2 = s_key[q];
        // A window never covers a slot twice (lanes >= slots stay idle), so every lane
        // reads the state the one-at-a-time scan would see.
        const int window = d.slots < 32 ? d.slots : 32;
        int s = -1;
        for(int scanned = 0; scanned < 4 * d.slots && s < 0; scanned += window) {
            __syncwarp();
            const int hand = s_hand;
            const int c = (hand + r) % d.slots;
            const bool active = r < window;
            const bool pinned = active && d.slot_pin[c] == stamp;
            const bool referenced = active && d.slot_ref[c] != 0;
            const unsigned free_mask = __ballot_sync(0xffffffffu, active && !pinned && !referenced);
            const int first = free_mask ? __ffs(free_mask) - 1 : window;
            if(r < first && active && !pinned && referenced) d.slot_ref[c] = 0;
            if(free_mask) s = (hand + first) % d.slots;
            __syncwarp();
            if(r == 0) s_hand = (hand + (free_mask ? first + 1 : window)) % d.slots;
        }
        if(r == 0) {
            if(s < 0) {
                d.state[2] = 2;  // every slot pinned by this plan: fewer slots than routes
                s_slot[q] = -1;
            } else {
                if(d.slot_key[s] >= 0) d.key_slot[d.slot_key[s]] = -1;
                d.slot_key[s] = k2;
                d.key_slot[k2] = s;
                d.slot_ref[s] = 1;
                d.slot_pin[s] = stamp;
                const std::uint8_t* src = pool_source(d, group, k2);
                if(src) ++pooled;
                else src = d.store + std::size_t(k2) * d.bundle_bytes;
                fills[nfill++] = Fill{reinterpret_cast<const uint4*>(src),
                                      reinterpret_cast<uint4*>(d.slots_base + std::size_t(s) * d.bundle_bytes)};
                s_slot[q] = s;
            }
        }
        __syncwarp();
    }
    if(r == 0) {
        d.state[0] = s_hand;
        *fill_count = nfill;  // lane 0 wrote every job
        d.counters[0] += hits;
        d.counters[1] += misses;
        d.counters[2] += pooled;
    }
    __syncwarp();
    if(r < count) {
        const int s = s_slot[r];
        uses[r] = s_key[r] < 0 || s < 0 ? Use{nullptr, -1, 0.f}
                                        : Use{d.slots_base + std::size_t(s) * d.bundle_bytes, s_key[r], weights ? weights[r] : 1.f};
    }
}

__global__ void fill_kernel(const Fill* fills, const int* fill_count, std::size_t words, int blocks_per_bundle) {
    const int job = blockIdx.x / blocks_per_bundle;
    if(job >= *fill_count) return;
    const Fill f = fills[job];
    const std::size_t stride = std::size_t(blocks_per_bundle) * kFillThreads;
    for(std::size_t i = std::size_t(blockIdx.x % blocks_per_bundle) * kFillThreads + threadIdx.x; i < words; i += stride)
        f.dst[i] = f.src[i];
}

__global__ void prefetch_kernel(Directory d, int group, const std::int32_t* predicted, int count, Fill* fills, int* fill_count) {
    if(threadIdx.x != 0) return;
    const int half = group & 1;
    std::int32_t* tags = d.pool_tags + half * d.pool_slots;
    for(int p = 0; p < d.pool_slots; ++p) tags[p] = -1;
    // Pool slots go to the nonresident predictions in prediction order, so resident or
    // invalid predictions do not take a slot.
    int n = 0;
    for(int i = 0; i < count && n < d.pool_slots; ++i) {
        const int e = predicted[i];
        if(e < 0 || e >= d.experts || group < 0 || group >= d.groups) continue;
        const int key = group * d.experts + e;
        if(d.key_slot[key] >= 0) continue;  // resident now (if evicted before use, it is an ordinary miss)
        bool repeated = false;
        for(int q = 0; q < n; ++q) repeated |= tags[q] == key;
        if(repeated) continue;
        tags[n] = key;
        fills[n] = Fill{reinterpret_cast<const uint4*>(d.store + std::size_t(key) * d.bundle_bytes),
                        reinterpret_cast<uint4*>(d.pool_base + (std::size_t(half) * d.pool_slots + n) * d.bundle_bytes)};
        ++n;
    }
    *fill_count = n;
    d.counters[3] += n;
}

bool valid(const Directory& d) {
    return d.key_slot && d.slot_key && d.slot_ref && d.slot_pin && d.state && d.counters && d.slots_base && d.store &&
           d.bundle_bytes && d.bundle_bytes % 16 == 0 && d.slots > 0 && d.groups > 0 && d.experts > 0 &&
           (!d.pool_slots || (d.pool_base && d.pool_tags && d.pool_slots > 0));
}

} // namespace

cudaError_t enqueue_plan(const Directory& d, int group, const std::int32_t* experts, const float* weights, int count, Use* uses,
                         Fill* fills, int* fill_count, cudaStream_t stream) {
    if(!valid(d) || !experts || !uses || !fills || !fill_count || count < 0 || count > kMaxRoutes || count > d.slots)
        return cudaErrorInvalidValue;
    plan_kernel<<<1, 32, 0, stream>>>(d, group, experts, weights, count, uses, fills, fill_count);
    return cudaGetLastError();
}

cudaError_t enqueue_fill(const Fill* fills, const int* fill_count, int max_fills, std::size_t bundle_bytes, int blocks_per_bundle,
                         cudaStream_t stream) {
    if(!fills || !fill_count || max_fills < 0 || !bundle_bytes || bundle_bytes % 16 || blocks_per_bundle < 1)
        return cudaErrorInvalidValue;
    if(max_fills == 0) return cudaSuccess;
    fill_kernel<<<max_fills * blocks_per_bundle, kFillThreads, 0, stream>>>(fills, fill_count, bundle_bytes / 16, blocks_per_bundle);
    return cudaGetLastError();
}

cudaError_t enqueue_prefetch_plan(const Directory& d, int group, const std::int32_t* predicted, int count, Fill* fills,
                                  int* fill_count, cudaStream_t stream) {
    if(!valid(d) || !d.pool_slots || !predicted || !fills || !fill_count || count < 0) return cudaErrorInvalidValue;
    prefetch_kernel<<<1, 32, 0, stream>>>(d, group, predicted, count, fills, fill_count);
    return cudaGetLastError();
}

} // namespace crit::techniques::device_cache
