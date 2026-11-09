#include "runtime.h"

namespace crit::techniques::device_cache {
namespace {

template <class T>
cudaError_t allocate(T*& p, std::size_t n) {
    return cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T));
}

template <class T>
void free_device(T*& p) {
    if(p) cudaFree(const_cast<void*>(static_cast<const void*>(p)));
    p = nullptr;
}

} // namespace

cudaError_t ExpertCache::init(const Geometry& g, std::uint8_t* slots_base, const std::uint8_t* store,
                              std::uint8_t* pool_base, cudaStream_t stream) {
    release();
    if(g.slots < 1 || g.groups < 1 || g.experts < 1 || g.pool_slots < 0 || !g.bundle_bytes || g.bundle_bytes % 16 ||
       !slots_base || !store || (g.pool_slots && !pool_base))
        return cudaErrorInvalidValue;
    const std::size_t keys = std::size_t(g.groups) * g.experts;
    std::int32_t *key_slot = nullptr, *slot_key = nullptr, *slot_pin = nullptr, *state = nullptr, *pool_tags = nullptr;
    std::uint8_t *slot_ref = nullptr, *key_seen = nullptr;
    unsigned long long* counters = nullptr;
    cudaError_t e = cudaSuccess;
    auto step = [&](cudaError_t r) { if(e == cudaSuccess) e = r; };
    step(allocate(key_slot, keys));
    step(allocate(slot_key, g.slots));
    step(allocate(slot_ref, g.slots));
    step(allocate(slot_pin, g.slots));
    step(allocate(state, 3));
    step(allocate(counters, 5));
    step(allocate(key_seen, keys));
    if(g.pool_slots) step(allocate(pool_tags, 2 * std::size_t(g.pool_slots)));
    step(allocate(table_, kMaxRoutes + 1));
    step(allocate(uses_, std::size_t(g.groups) * kMaxRoutes));
    step(allocate(fills_, std::size_t(g.groups) * kMaxRoutes));
    if(g.pool_slots) step(allocate(pool_fills_, 2 * std::size_t(g.pool_slots)));
    step(allocate(fill_counts_, std::size_t(g.groups) + 2));
    d_.key_slot = key_slot;
    d_.slot_key = slot_key;
    d_.slot_ref = slot_ref;
    d_.slot_pin = slot_pin;
    d_.state = state;
    d_.counters = counters;
    d_.pool_tags = pool_tags;
    seen_ = key_seen;  // second-touch is on only while Directory::key_seen points at it (set_admission)
    if(e != cudaSuccess) {
        release();
        return e;
    }
    d_.slots_base = slots_base;
    d_.store = store;
    d_.pool_base = pool_base;
    d_.bundle_bytes = g.bundle_bytes;
    d_.slots = g.slots;
    d_.groups = g.groups;
    d_.experts = g.experts;
    d_.pool_slots = g.pool_slots;
    // -1 is all ones: empty slots, unmapped keys, unpinned slots, empty pool tags
    step(cudaMemsetAsync(key_slot, 0xff, keys * sizeof(std::int32_t), stream));
    step(cudaMemsetAsync(slot_key, 0xff, g.slots * sizeof(std::int32_t), stream));
    step(cudaMemsetAsync(slot_pin, 0xff, g.slots * sizeof(std::int32_t), stream));
    step(cudaMemsetAsync(slot_ref, 0, g.slots, stream));
    step(cudaMemsetAsync(state, 0, 3 * sizeof(std::int32_t), stream));
    step(cudaMemsetAsync(counters, 0, 5 * sizeof(unsigned long long), stream));
    step(cudaMemsetAsync(key_seen, 0, keys, stream));
    if(pool_tags) step(cudaMemsetAsync(pool_tags, 0xff, 2 * std::size_t(g.pool_slots) * sizeof(std::int32_t), stream));
    step(cudaMemsetAsync(fill_counts_, 0, (std::size_t(g.groups) + 2) * sizeof(int), stream));
    return e;
}

void ExpertCache::release() {
    free_device(d_.key_slot);
    free_device(d_.slot_key);
    free_device(d_.slot_ref);
    free_device(d_.slot_pin);
    free_device(d_.state);
    free_device(d_.counters);
    free_device(d_.pool_tags);
    free_device(seen_);
    free_device(table_);
    free_device(uses_);
    free_device(fills_);
    free_device(pool_fills_);
    free_device(fill_counts_);
    free_device(d_.route_counts);
    free_device(swaps_);
    free_device(swap_fills_);
    free_device(swap_count_);
    if(begun_) cudaEventDestroy(begun_);
    if(copied_) cudaEventDestroy(copied_);
    begun_ = copied_ = nullptr;
    max_swaps_ = pending_ = 0;
    d_ = Directory{};
}

cudaError_t ExpertCache::enable_route_counts(int max_swaps, cudaStream_t stream) {
    if(!d_.state || max_swaps < 1 || d_.route_counts) return cudaErrorInvalidValue;
    const std::size_t keys = std::size_t(d_.groups) * d_.experts;
    std::uint32_t* counts = nullptr;
    cudaError_t e = allocate(counts, keys);
    auto step = [&](cudaError_t r) { if(e == cudaSuccess) e = r; };
    step(allocate(swaps_, std::size_t(max_swaps)));
    step(allocate(swap_fills_, std::size_t(max_swaps)));
    step(allocate(swap_count_, 1));
    step(cudaEventCreateWithFlags(&begun_, cudaEventDisableTiming));
    step(cudaEventCreateWithFlags(&copied_, cudaEventDisableTiming));
    step(cudaMemsetAsync(counts, 0, keys * sizeof(std::uint32_t), stream));
    if(e != cudaSuccess) {
        free_device(counts);
        return e;
    }
    d_.route_counts = counts;
    max_swaps_ = max_swaps;
    return cudaSuccess;
}

cudaError_t ExpertCache::read_route_counts(std::uint32_t* host, cudaStream_t stream) const {
    if(!d_.route_counts || !host) return cudaErrorInvalidValue;
    return cudaMemcpyAsync(host, d_.route_counts, std::size_t(d_.groups) * d_.experts * sizeof(std::uint32_t),
                           cudaMemcpyDeviceToHost, stream);
}

cudaError_t ExpertCache::begin_swaps(const Swap* swaps, int count, int blocks_per_bundle, cudaStream_t main,
                                     cudaStream_t side) {
    if(!d_.route_counts || count < 0 || count > max_swaps_ || pending_ || blocks_per_bundle < 1)
        return cudaErrorInvalidValue;
    if(count == 0) return cudaSuccess;
    if(!swaps) return cudaErrorInvalidValue;
    // A pageable source: the copy returns once the bytes are staged, so the caller may reuse its array.
    cudaError_t e = cudaMemcpyAsync(swaps_, swaps, std::size_t(count) * sizeof(Swap), cudaMemcpyHostToDevice, main);
    if(e == cudaSuccess) e = enqueue_swap_begin(d_, swaps_, count, swap_fills_, swap_count_, main);
    if(e == cudaSuccess) e = cudaEventRecord(begun_, main);
    if(e == cudaSuccess) e = cudaStreamWaitEvent(side, begun_, 0);
    if(e == cudaSuccess) e = enqueue_fill(swap_fills_, swap_count_, count, d_.bundle_bytes, blocks_per_bundle, side);
    if(e == cudaSuccess) e = cudaEventRecord(copied_, side);
    if(e == cudaSuccess) pending_ = count;
    return e;
}

cudaError_t ExpertCache::commit_swaps(cudaStream_t main) {
    if(!d_.route_counts) return cudaErrorInvalidValue;
    if(!pending_) return cudaSuccess;
    cudaError_t e = cudaStreamWaitEvent(main, copied_, 0);
    if(e == cudaSuccess) e = enqueue_swap_commit(d_, swaps_, pending_, main);
    if(e == cudaSuccess) pending_ = 0;
    return e;
}

cudaError_t ExpertCache::set_admission(int quota, const std::int32_t* table, bool second_touch, cudaStream_t stream) {
    if(!d_.state || quota < -1) return cudaErrorInvalidValue;
    d_.admit_quota = quota;
    d_.admit_by_misses = nullptr;
    if(table) {
        for(int m = 0; m <= kMaxRoutes; ++m)
            if(table[m] < -1) return cudaErrorInvalidValue;
        const cudaError_t e = cudaMemcpyAsync(table_, table, (kMaxRoutes + 1) * sizeof(std::int32_t),
                                              cudaMemcpyHostToDevice, stream);
        if(e != cudaSuccess) return e;
        // the copy reads host memory the caller may reuse: complete it before returning
        const cudaError_t s = cudaStreamSynchronize(stream);
        if(s != cudaSuccess) return s;
        d_.admit_by_misses = table_;
    }
    d_.key_seen = second_touch ? seen_ : nullptr;
    return cudaSuccess;
}

cudaError_t ExpertCache::preload(const std::int32_t* keys, int count, int blocks_per_bundle, cudaStream_t stream) {
    if(!d_.state || !keys || count < 0 || blocks_per_bundle < 1) return cudaErrorInvalidValue;
    if(count == 0) return cudaSuccess;
    const int fills = count < d_.slots ? count : d_.slots;
    std::int32_t* device_keys = nullptr;
    Fill* device_fills = nullptr;
    int* device_count = nullptr;
    cudaError_t e = allocate(device_keys, count);
    auto step = [&](cudaError_t r) { if(e == cudaSuccess) e = r; };
    step(allocate(device_fills, fills));
    step(allocate(device_count, 1));
    step(cudaMemcpyAsync(device_keys, keys, std::size_t(count) * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    if(e == cudaSuccess) step(enqueue_preload(d_, device_keys, count, device_fills, device_count, stream));
    if(e == cudaSuccess) step(enqueue_fill(device_fills, device_count, fills, d_.bundle_bytes, blocks_per_bundle, stream));
    step(cudaStreamSynchronize(stream));
    free_device(device_keys);
    free_device(device_fills);
    free_device(device_count);
    return e;
}

cudaError_t ExpertCache::plan_and_fetch(int group, const std::int32_t* experts, const float* weights, int count,
                                        int blocks_per_bundle, cudaStream_t stream, const Use** uses) {
    if(!d_.state || group < 0 || group >= d_.groups || !uses) return cudaErrorInvalidValue;
    Use* out = uses_ + std::size_t(group) * kMaxRoutes;
    Fill* fills = fills_ + std::size_t(group) * kMaxRoutes;
    int* fill_count = fill_counts_ + group;
    cudaError_t e = enqueue_plan(d_, group, experts, weights, count, out, fills, fill_count, stream);
    if(e != cudaSuccess) return e;
    e = enqueue_fill(fills, fill_count, count, d_.bundle_bytes, blocks_per_bundle, stream);
    if(e != cudaSuccess) return e;
    *uses = out;
    return cudaSuccess;
}

cudaError_t ExpertCache::prefetch(int group, const std::int32_t* predicted, int count, int blocks_per_bundle,
                                  cudaStream_t side) {
    if(!d_.state || !d_.pool_slots || group < 0 || group >= d_.groups) return cudaErrorInvalidValue;
    Fill* fills = pool_fills_ + std::size_t(group & 1) * d_.pool_slots;
    int* fill_count = fill_counts_ + d_.groups + (group & 1);
    const cudaError_t e = enqueue_prefetch_plan(d_, group, predicted, count, fills, fill_count, side);
    if(e != cudaSuccess) return e;
    return enqueue_fill(fills, fill_count, d_.pool_slots, d_.bundle_bytes, blocks_per_bundle, side);
}

cudaError_t ExpertCache::telemetry(Telemetry& out, cudaStream_t stream) const {
    if(!d_.state) return cudaErrorInvalidValue;
    unsigned long long counters[5] = {};
    std::int32_t state[3] = {};
    cudaError_t e = cudaMemcpyAsync(counters, d_.counters, sizeof(counters), cudaMemcpyDeviceToHost, stream);
    if(e == cudaSuccess) e = cudaMemcpyAsync(state, d_.state, sizeof(state), cudaMemcpyDeviceToHost, stream);
    if(e == cudaSuccess) e = cudaStreamSynchronize(stream);
    if(e != cudaSuccess) return e;
    out = Telemetry{counters[0], counters[1], counters[2], counters[3], counters[4], state[2]};
    return cudaSuccess;
}

} // namespace crit::techniques::device_cache
