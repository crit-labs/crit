#include "device_cache.h"

// The planning kernels run as one block of 32 threads. CUDA: a warp, synchronized and voted with the _sync
// intrinsics. HIP: the block lives in lanes 0-31 of one wavefront (32 or 64 wide), so a block barrier is the warp's
// and the low 32 bits of the wavefront's ballot are the block's lanes.
#if defined(CRIT_CORE_HIP)
#define CRIT_CACHE_SYNC() __syncthreads()
#define CRIT_CACHE_BALLOT(predicate) static_cast<unsigned>(__ballot(predicate) & 0xffffffffull)
#else
#define CRIT_CACHE_SYNC() __syncwarp()
#define CRIT_CACHE_BALLOT(predicate) __ballot_sync(0xffffffffu, (predicate))
#endif

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
__device__ __forceinline__ bool limited(const Directory& d) {
    return d.admit_quota >= 0 || d.admit_by_misses || d.key_seen;
}

// Admission limits (Directory::admit_quota, admit_by_misses, key_seen): a miss over the plan's quota, or a key's
// first miss under second-touch, bypasses the cache: no victim, no copy, its Use names the store bundle.
__global__ void plan_kernel(Directory d, int group, const std::int32_t* experts, const float* weights, int count, Use* uses,
                            Fill* fills, int* fill_count) {
    __shared__ int s_slot[kMaxRoutes], s_key[kMaxRoutes], s_bypass[kMaxRoutes], s_stamp, s_hand, s_quota, s_admit;
    const int r = threadIdx.x;
    if(r == 0) {
        s_stamp = ++d.state[1];
        s_hand = d.state[0];
    }
    s_bypass[r] = 0;
    CRIT_CACHE_SYNC();
    const int stamp = s_stamp;
    int key = -1, slot = -1;
    if(r < count) {
        const int e = experts[r];
        if(e < 0 || e >= d.experts || group < 0 || group >= d.groups) d.state[2] = 1;
        else key = group * d.experts + e;
    }
    s_key[r] = key;
    CRIT_CACHE_SYNC();
    // duplicate routes would be admitted twice: reject them
    if(key >= 0)
        for(int q = 0; q < r; ++q)
            if(s_key[q] == key) { d.state[2] = 1; key = -1; }
    CRIT_CACHE_SYNC();
    s_key[r] = key;
    if(key >= 0) {
        slot = d.key_slot[key];
        if(slot >= 0) { d.slot_ref[slot] = 1; d.slot_pin[slot] = stamp; }
    }
    s_slot[r] = slot;
    CRIT_CACHE_SYNC();
    if(r == 0) {
        int miss_count = 0;
        for(int q = 0; q < count; ++q) miss_count += s_key[q] >= 0 && s_slot[q] < 0;
        s_quota = d.admit_by_misses ? d.admit_by_misses[miss_count] : d.admit_quota;
    }
    CRIT_CACHE_SYNC();
    int nfill = 0, admitted = 0;
    unsigned long long hits = 0, misses = 0, pooled = 0, bypassed = 0;
    for(int q = 0; q < count; ++q) {
        if(s_key[q] < 0) continue;
        if(s_slot[q] >= 0) { ++hits; continue; }
        ++misses;
        const int k2 = s_key[q];
        if(r == 0) {
            bool admit = s_quota < 0 || admitted < s_quota;
            if(d.key_seen && !d.key_seen[k2]) {
                d.key_seen[k2] = 1;  // first touch: remembered, served from the store
                admit = false;
            }
            s_admit = admit;
        }
        CRIT_CACHE_SYNC();
        if(!s_admit) {
            ++bypassed;
            if(r == 0) s_bypass[q] = 1;
            CRIT_CACHE_SYNC();
            continue;
        }
        ++admitted;
        // A window never covers a slot twice (lanes >= slots stay idle), so every lane
        // reads the state the one-at-a-time scan would see.
        const int window = d.slots < 32 ? d.slots : 32;
        int s = -1;
        for(int scanned = 0; scanned < 4 * d.slots && s < 0; scanned += window) {
            CRIT_CACHE_SYNC();
            const int hand = s_hand;
            const int c = (hand + r) % d.slots;
            const bool active = r < window;
            const bool pinned = active && d.slot_pin[c] == stamp;
            const bool referenced = active && d.slot_ref[c] != 0;
            const unsigned free_mask = CRIT_CACHE_BALLOT(active && !pinned && !referenced);
            const int first = free_mask ? __ffs(free_mask) - 1 : window;
            if(r < first && active && !pinned && referenced) d.slot_ref[c] = 0;
            if(free_mask) s = (hand + first) % d.slots;
            CRIT_CACHE_SYNC();
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
        CRIT_CACHE_SYNC();
    }
    if(r == 0) {
        d.state[0] = s_hand;
        *fill_count = nfill;  // lane 0 wrote every job
        d.counters[0] += hits;
        d.counters[1] += misses;
        d.counters[2] += pooled;
        if(limited(d)) d.counters[4] += bypassed;
    }
    CRIT_CACHE_SYNC();
    if(r < count) {
        const int s = s_slot[r];
        const float w = weights ? weights[r] : 1.f;
        uses[r] = s_key[r] >= 0 && s_bypass[r] ? Use{d.store + std::size_t(s_key[r]) * d.bundle_bytes, s_key[r], w, Source::store}
                  : s_key[r] < 0 || s < 0      ? Use{nullptr, -1, 0.f}
                                               : Use{d.slots_base + std::size_t(s) * d.bundle_bytes, s_key[r], w};
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

// Warm start from a residency profile: the first valid, distinct keys take slots 0, 1, ... unreferenced.
__global__ void preload_kernel(Directory d, const std::int32_t* keys, int count, Fill* fills, int* fill_count) {
    if(threadIdx.x != 0) return;
    int n = 0;
    for(int i = 0; i < count && n < d.slots; ++i) {
        const int key = keys[i];
        if(key < 0 || key >= d.groups * d.experts || d.key_slot[key] >= 0) continue;
        if(d.slot_key[n] >= 0) d.key_slot[d.slot_key[n]] = -1;
        d.slot_key[n] = key;
        d.key_slot[key] = n;
        d.slot_ref[n] = 0;
        fills[n] = Fill{reinterpret_cast<const uint4*>(d.store + std::size_t(key) * d.bundle_bytes),
                        reinterpret_cast<uint4*>(d.slots_base + std::size_t(n) * d.bundle_bytes)};
        ++n;
    }
    *fill_count = n;
    d.state[0] = n % d.slots;  // the clock starts at the first slot the profile left free
}

// One thread per route: the provider GEMM's pointer arrays from the plan's Uses.
__global__ void operands_kernel(const Use* uses, int count, OperandSpec spec, Operands out) {
    const int r = threadIdx.x;
    if(r >= count) return;
    const Use u = uses[r];
    const bool live = u.key >= 0 && u.bundle && !(spec.skip_bypassed && u.source == Source::store);
    for(int c = 0; c < spec.components; ++c)
        if(out.matrix[c]) out.matrix[c][r] = live ? u.bundle + spec.offsets[c] : nullptr;
    const int row = spec.token ? spec.token[r] : r;
    if(out.x) out.x[r] = live ? spec.x + std::size_t(row) * spec.x_row_bytes : nullptr;
    if(out.y) out.y[r] = live ? spec.y + std::size_t(r) * spec.y_row_bytes : nullptr;
    if(out.weight) out.weight[r] = live ? u.weight : 0.f;
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

cudaError_t enqueue_preload(const Directory& d, const std::int32_t* keys, int count, Fill* fills, int* fill_count,
                            cudaStream_t stream) {
    if(!valid(d) || !keys || !fills || !fill_count || count < 0) return cudaErrorInvalidValue;
    preload_kernel<<<1, 32, 0, stream>>>(d, keys, count, fills, fill_count);
    return cudaGetLastError();
}

cudaError_t enqueue_operands(const Use* uses, int count, const OperandSpec& spec, const Operands& out, cudaStream_t stream) {
    if(!uses || count < 0 || count > kMaxRoutes || spec.components < 1 || spec.components > kMaxComponents ||
       (out.x && (!spec.x || !spec.x_row_bytes)) || (out.y && (!spec.y || !spec.y_row_bytes)))
        return cudaErrorInvalidValue;
    if(count == 0) return cudaSuccess;
    operands_kernel<<<1, kMaxRoutes, 0, stream>>>(uses, count, spec, out);
    return cudaGetLastError();
}

} // namespace crit::techniques::device_cache
