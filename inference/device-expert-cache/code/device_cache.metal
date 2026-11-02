// The device cache directory's kernels (device_cache.cu) in the Metal Shading Language, for Apple GPUs: the same
// plan, fill and prefetch steps, with the Directory's arrays bound as buffers. plan and prefetch run as one
// threadgroup of 32 threads (one SIMD group); fill runs fill_threads threads per block of a bundle.
//
// On Apple silicon the "host store" is ordinary shared memory, so a miss copy moves a bundle within one memory; the
// cache still pays off where the slots are faster to read than the store (a discrete GPU, or a store that is
// file-backed). The engine owns the encoder, the buffers and the dispatch sizes, as it owns the CUDA streams.
//
// The counters are 32-bit atomics here (Metal has no 64-bit ones on every GPU), 64-bit in CUDA and HIP.
// Admission limits (Params admit_quota, has_table, second_touch with the admit_by_misses and key_seen buffers) work
// as device_cache.h describes; a bypassed miss's Use has from_store set and its offset in the store, and counters[4]
// counts bypasses when a limit is set. preload_kernel is the warm start.
//
// Not run on Apple hardware in crit's tests: device_expert_cache_tests checks the CUDA kernels against the Python
// replay (device_cache.py), which states the policy these kernels follow.
#include <metal_stdlib>
using namespace metal;

namespace crit_device_cache {

constant int kMaxRoutes = 32;

struct Params {
    int slots, groups, experts, pool_slots;
    ulong bundle_bytes;
    int group, count;
    int admit_quota;   // -1: every miss admitted
    int has_table;     // admit_by_misses gives the quota by miss count
    int second_touch;  // key_seen marks first misses
};

struct Use {
    ulong bundle_offset;  // byte offset of the expert's slot in slots_base (in the store when from_store); valid when key >= 0
    int key;              // -1 for a rejected route
    float weight;
    uint from_store;      // a miss the admission limits bypassed
};

struct Fill {
    ulong src_offset;  // byte offset in the store, or in the pool when from_pool
    ulong dst_offset;  // byte offset in slots_base, or in the pool when to_pool
    uint from_pool, to_pool;
};

inline int pool_index(device const int* pool_tags, constant Params& p, int key) {
    if(!p.pool_slots) return -1;
    const int half = p.group & 1;
    for(int i = 0; i < p.pool_slots; ++i)
        if(pool_tags[half * p.pool_slots + i] == key) return half * p.pool_slots + i;
    return -1;
}

// One SIMD group of 32 lanes; the steps of device_cache.cu's plan_kernel in the same order.
kernel void plan_kernel(constant Params& p [[buffer(0)]], device int* key_slot [[buffer(1)]],
                        device int* slot_key [[buffer(2)]], device uchar* slot_ref [[buffer(3)]],
                        device int* slot_pin [[buffer(4)]], device int* state [[buffer(5)]],
                        device atomic_uint* counters [[buffer(6)]], device const int* pool_tags [[buffer(7)]],
                        device const int* experts [[buffer(8)]], device const float* weights [[buffer(9)]],
                        device Use* uses [[buffer(10)]], device Fill* fills [[buffer(11)]],
                        device int* fill_count [[buffer(12)]], device const int* admit_by_misses [[buffer(13)]],
                        device uchar* key_seen [[buffer(14)]], uint r [[thread_index_in_simdgroup]]) {
    threadgroup int s_slot[kMaxRoutes], s_key[kMaxRoutes], s_bypass[kMaxRoutes], s_stamp, s_hand, s_quota, s_admit;
    if(r == 0) {
        s_stamp = ++state[1];
        s_hand = state[0];
    }
    s_bypass[r] = 0;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const int stamp = s_stamp;
    int key = -1, slot = -1;
    if(int(r) < p.count) {
        const int e = experts[r];
        if(e < 0 || e >= p.experts || p.group < 0 || p.group >= p.groups) state[2] = 1;
        else key = p.group * p.experts + e;
    }
    s_key[r] = key;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if(key >= 0)
        for(uint q = 0; q < r; ++q)
            if(s_key[q] == key) { state[2] = 1; key = -1; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    s_key[r] = key;
    if(key >= 0) {
        slot = key_slot[key];
        if(slot >= 0) { slot_ref[slot] = 1; slot_pin[slot] = stamp; }
    }
    s_slot[r] = slot;
    simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if(r == 0) {
        int miss_count = 0;
        for(int q = 0; q < p.count; ++q) miss_count += s_key[q] >= 0 && s_slot[q] < 0;
        s_quota = p.has_table ? admit_by_misses[miss_count] : p.admit_quota;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    int nfill = 0, admitted = 0;
    uint hits = 0, misses = 0, pooled = 0, bypassed = 0;
    for(int q = 0; q < p.count; ++q) {
        if(s_key[q] < 0) continue;
        if(s_slot[q] >= 0) { ++hits; continue; }
        ++misses;
        const int k2 = s_key[q];
        if(r == 0) {
            bool admit = s_quota < 0 || admitted < s_quota;
            if(p.second_touch && !key_seen[k2]) {
                key_seen[k2] = 1;
                admit = false;
            }
            s_admit = admit ? 1 : 0;
        }
        simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if(!s_admit) {
            ++bypassed;
            if(r == 0) s_bypass[q] = 1;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        ++admitted;
        const int window = p.slots < 32 ? p.slots : 32;
        int s = -1;
        for(int scanned = 0; scanned < 4 * p.slots && s < 0; scanned += window) {
            simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            const int hand = s_hand;
            const int c = (hand + int(r)) % p.slots;
            const bool active = int(r) < window;
            const bool pinned = active && slot_pin[c] == stamp;
            const bool referenced = active && slot_ref[c] != 0;
            const simd_vote vote = simd_ballot(active && !pinned && !referenced);
            const uint free_mask = uint(static_cast<simd_vote::vote_t>(vote) & 0xffffffffull);
            const int first = free_mask ? int(ctz(free_mask)) : window;
            if(int(r) < first && active && !pinned && referenced) slot_ref[c] = 0;
            if(free_mask) s = (hand + first) % p.slots;
            simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if(r == 0) s_hand = (hand + (free_mask ? first + 1 : window)) % p.slots;
        }
        if(r == 0) {
            if(s < 0) {
                state[2] = 2;
                s_slot[q] = -1;
            } else {
                if(slot_key[s] >= 0) key_slot[slot_key[s]] = -1;
                slot_key[s] = k2;
                key_slot[k2] = s;
                slot_ref[s] = 1;
                slot_pin[s] = stamp;
                const int pi = pool_index(pool_tags, p, k2);
                if(pi >= 0) ++pooled;
                fills[nfill++] = Fill{pi >= 0 ? ulong(pi) * p.bundle_bytes : ulong(k2) * p.bundle_bytes,
                                      ulong(s) * p.bundle_bytes, pi >= 0 ? 1u : 0u, 0u};
                s_slot[q] = s;
            }
        }
        simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    if(r == 0) {
        state[0] = s_hand;
        *fill_count = nfill;
        atomic_fetch_add_explicit(&counters[0], hits, memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[1], misses, memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[2], pooled, memory_order_relaxed);
        if(p.admit_quota >= 0 || p.has_table || p.second_touch)
            atomic_fetch_add_explicit(&counters[4], bypassed, memory_order_relaxed);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if(int(r) < p.count) {
        const int s = s_slot[r];
        const float w = weights ? weights[r] : 1.f;
        uses[r] = s_key[r] >= 0 && s_bypass[r] ? Use{ulong(s_key[r]) * p.bundle_bytes, s_key[r], w, 1u}
                  : s_key[r] < 0 || s < 0      ? Use{0, -1, 0.f, 0u}
                                               : Use{ulong(s) * p.bundle_bytes, s_key[r], w, 0u};
    }
}

// The copies: fill_threads threads per group of blocks_per_bundle threadgroups of one bundle, 16 bytes a step.
kernel void fill_kernel(device const Fill* fills [[buffer(0)]], device const int* fill_count [[buffer(1)]],
                        device const uint4* store [[buffer(2)]], device uint4* pool [[buffer(3)]],
                        device uint4* slots_base [[buffer(4)]], constant ulong& words [[buffer(5)]],
                        constant int& blocks_per_bundle [[buffer(6)]], uint block [[threadgroup_position_in_grid]],
                        uint thread [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    const int job = int(block) / blocks_per_bundle;
    if(job >= *fill_count) return;
    const Fill f = fills[job];
    device const uint4* src = (f.from_pool ? pool : store) + f.src_offset / 16;
    device uint4* dst = (f.to_pool ? pool : slots_base) + f.dst_offset / 16;
    const ulong stride = ulong(blocks_per_bundle) * threads;
    for(ulong i = ulong(int(block) % blocks_per_bundle) * threads + thread; i < words; i += stride) dst[i] = src[i];
}

// Lookahead: tags the pool half for the group with the nonresident predictions, in prediction order.
kernel void prefetch_kernel(constant Params& p [[buffer(0)]], device const int* key_slot [[buffer(1)]],
                            device int* pool_tags [[buffer(2)]], device atomic_uint* counters [[buffer(3)]],
                            device const int* predicted [[buffer(4)]], device Fill* fills [[buffer(5)]],
                            device int* fill_count [[buffer(6)]], uint r [[thread_index_in_simdgroup]]) {
    if(r != 0) return;
    const int half = p.group & 1;
    device int* tags = pool_tags + half * p.pool_slots;
    for(int i = 0; i < p.pool_slots; ++i) tags[i] = -1;
    int n = 0;
    for(int i = 0; i < p.count && n < p.pool_slots; ++i) {
        const int e = predicted[i];
        if(e < 0 || e >= p.experts || p.group < 0 || p.group >= p.groups) continue;
        const int key = p.group * p.experts + e;
        if(key_slot[key] >= 0) continue;
        bool repeated = false;
        for(int q = 0; q < n; ++q) repeated |= tags[q] == key;
        if(repeated) continue;
        tags[n] = key;
        fills[n] = Fill{ulong(key) * p.bundle_bytes, ulong(half * p.pool_slots + n) * p.bundle_bytes, 0u, 1u};
        ++n;
    }
    *fill_count = n;
    atomic_fetch_add_explicit(&counters[3], uint(n), memory_order_relaxed);
}

// Warm start: the first valid, distinct keys of a residency profile take slots 0, 1, ... unreferenced.
kernel void preload_kernel(constant Params& p [[buffer(0)]], device int* key_slot [[buffer(1)]],
                           device int* slot_key [[buffer(2)]], device uchar* slot_ref [[buffer(3)]],
                           device int* state [[buffer(4)]], device const int* keys [[buffer(5)]],
                           device Fill* fills [[buffer(6)]], device int* fill_count [[buffer(7)]],
                           uint r [[thread_index_in_simdgroup]]) {
    if(r != 0) return;
    int n = 0;
    for(int i = 0; i < p.count && n < p.slots; ++i) {
        const int key = keys[i];
        if(key < 0 || key >= p.groups * p.experts || key_slot[key] >= 0) continue;
        if(slot_key[n] >= 0) key_slot[slot_key[n]] = -1;
        slot_key[n] = key;
        key_slot[key] = n;
        slot_ref[n] = 0;
        fills[n] = Fill{ulong(key) * p.bundle_bytes, ulong(n) * p.bundle_bytes, 0u, 0u};
        ++n;
    }
    *fill_count = n;
    state[0] = n % p.slots;
}

}  // namespace crit_device_cache
