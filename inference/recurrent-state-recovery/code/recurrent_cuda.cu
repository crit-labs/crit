// Reference kernels, not for performance. One block per head and one thread per
// value column v: the gated-delta update of column v depends only on H[:, v], so
// no cross-thread reduction is needed. Every path advances a column through the
// same device function, so verify, select and recompute agree bit for bit.
#include "recurrent_cuda.h"

namespace crit::techniques::recurrent::cuda {
namespace {
__device__ float inverse_norm(const float* x, std::uint32_t n, float epsilon, bool normalize) {
    if(!normalize) return 1.0f;
    float sum = epsilon;
    for(std::uint32_t i = 0; i < n; ++i) sum += x[i] * x[i];
    return 1.0f / sqrtf(sum);
}

// One state element: decayed value plus the rank-1 update. Every strategy (and the
// deferred replay) uses this function so their results agree bit for bit.
__device__ float updated(float h, float decay, float scaled_key, float delta) { return h * decay + scaled_key * delta; }

// Optional capture of the step's normalized key, decay and delta (deferred stash).
struct Capture { float* key; float* decay; float* delta; };

// G = exp(log_decay) * H; delta = beta * (v - k^T G); H' = G + k delta; out = scale * q^T H'.
__device__ void advance_column(const DeviceGeometry& g, const float* in, float* out, const DeviceSteps& s,
                               std::uint32_t t, std::uint32_t h, std::uint32_t v, float* output,
                               const Capture* capture = nullptr) {
    const auto entry = std::size_t(t) * g.heads + h;
    const float* key = s.key + entry * g.keys;
    const float key_scale = inverse_norm(key, g.keys, g.epsilon, g.normalize_keys);
    const float decay = expf(s.log_decay[entry]);
    float residual = s.value[entry * g.values + v];
    for(std::uint32_t k = 0; k < g.keys; ++k) residual -= key[k] * key_scale * (in[k * g.values + v] * decay);
    residual *= s.beta[entry];
    for(std::uint32_t k = 0; k < g.keys; ++k)
        out[k * g.values + v] = updated(in[k * g.values + v], decay, key[k] * key_scale, residual);
    if(capture) {
        capture->delta[entry * g.values + v] = residual;
        for(std::uint32_t k = v; k < g.keys; k += g.values) capture->key[entry * g.keys + k] = key[k] * key_scale;
        if(v == 0) capture->decay[entry] = decay;
    }
    if(output) {
        const float* query = s.query + entry * g.keys;
        const float query_scale = inverse_norm(query, g.keys, g.epsilon, g.normalize_queries);
        float sum = 0.0f;
        for(std::uint32_t k = 0; k < g.keys; ++k) sum += query[k] * query_scale * out[k * g.values + v];
        output[entry * g.values + v] = g.query_scale * sum;
    }
}

__global__ void verify(DeviceGeometry g, const float* start, DeviceSteps s, std::uint32_t columns,
                       float* outputs, float* checkpoints, float* work) {
    const auto h = blockIdx.x, v = threadIdx.x;
    if(v >= g.values) return;
    const auto per_head = std::size_t(g.keys) * g.values, stride = per_head * g.heads;
    const float* in = start + h * per_head;
    if(!checkpoints) { // Strategy B: one running copy, updated in place per column.
        float* running = work + h * per_head;
        for(std::uint32_t k = 0; k < g.keys; ++k) running[k * g.values + v] = in[k * g.values + v];
        in = running;
    }
    for(std::uint32_t t = 0; t < columns; ++t) {
        float* out = checkpoints ? checkpoints + t * stride + h * per_head : work + h * per_head;
        advance_column(g, in, out, s, t, h, v, outputs);
        in = out;
    }
}

__global__ void select_accepted(DeviceGeometry g, const float* start, const float* checkpoints,
                                const std::int32_t* accepted, std::uint32_t columns, float* state) {
    const auto total = std::size_t(g.heads) * g.keys * g.values;
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= total) return;
    auto n = *accepted;
    if(n < 0) n = 0;
    if(std::uint32_t(n) > columns) n = std::int32_t(columns);
    state[i] = n ? checkpoints[std::size_t(n - 1) * total + i] : start[i];
}

__global__ void recompute_accepted(DeviceGeometry g, const float* start, DeviceSteps s,
                                   const std::int32_t* accepted, std::uint32_t columns, float* state) {
    const auto h = blockIdx.x, v = threadIdx.x;
    if(v >= g.values) return;
    const auto per_head = std::size_t(g.keys) * g.values;
    float* running = state + h * per_head;
    const float* in = start + h * per_head;
    for(std::uint32_t k = 0; k < g.keys; ++k) running[k * g.values + v] = in[k * g.values + v];
    auto n = *accepted;
    if(n < 0) n = 0;
    if(std::uint32_t(n) > columns) n = std::int32_t(columns);
    for(std::int32_t t = 0; t < n; ++t) advance_column(g, running, running, s, std::uint32_t(t), h, v, nullptr);
}

__global__ void verify_deferred(DeviceGeometry g, float* state, DeviceSteps s, std::uint32_t columns,
                                const std::int32_t* pending, DeferredStash stash, float* outputs, float* work) {
    const auto h = blockIdx.x, v = threadIdx.x; // blockDim.x == values: every thread reaches the barrier
    const auto per_head = std::size_t(g.keys) * g.values;
    float* live = state + h * per_head;
    auto n = *pending;
    if(n < 0) n = 0;
    if(std::uint32_t(n) > stash.capacity) n = std::int32_t(stash.capacity);
    for(std::int32_t t = 0; t < n; ++t) {
        const auto entry = std::size_t(t) * g.heads + h;
        const float decay = stash.decay[entry], delta = stash.delta[entry * g.values + v];
        const float* key = stash.key + entry * g.keys;
        for(std::uint32_t k = 0; k < g.keys; ++k) live[k * g.values + v] = updated(live[k * g.values + v], decay, key[k], delta);
    }
    __syncthreads(); // every replay read of the stash precedes this round's stash writes
    if(!columns) return;
    float* running = work + h * per_head;
    for(std::uint32_t k = 0; k < g.keys; ++k) running[k * g.values + v] = live[k * g.values + v];
    const Capture capture{stash.key, stash.decay, stash.delta};
    for(std::uint32_t t = 0; t < columns; ++t) advance_column(g, running, running, s, t, h, v, outputs, &capture);
}

__global__ void record_accepted(const std::int32_t* accepted, std::uint32_t columns, std::int32_t* pending) {
    auto n = *accepted;
    if(n < 0) n = 0;
    if(std::uint32_t(n) > columns) n = std::int32_t(columns);
    *pending = n;
}

bool valid(const DeviceGeometry& g) {
    return g.heads && g.keys && g.values && g.values <= 1024 && g.epsilon > 0 && g.query_scale > 0;
}
}

cudaError_t enqueue_verify(const DeviceGeometry& g, const float* start, DeviceSteps s, std::uint32_t columns,
                           float* outputs, float* checkpoints, float* work, cudaStream_t stream) {
    if(!valid(g) || !start || !outputs || (!checkpoints && !work) || !s.query || !s.key || !s.value || !s.log_decay || !s.beta)
        return cudaErrorInvalidValue;
    if(!columns) return cudaSuccess;
    verify<<<g.heads, g.values, 0, stream>>>(g, start, s, columns, outputs, checkpoints, work);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_select_accepted(const DeviceGeometry& g, const float* start, const float* checkpoints,
                                    const std::int32_t* accepted, std::uint32_t columns, float* state, cudaStream_t stream) {
    if(!valid(g) || !start || !accepted || !state || (columns && !checkpoints)) return cudaErrorInvalidValue;
    const auto total = state_elements(g);
    constexpr unsigned threads = 256;
    select_accepted<<<unsigned((total + threads - 1) / threads), threads, 0, stream>>>(g, start, checkpoints, accepted, columns, state);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_recompute_accepted(const DeviceGeometry& g, const float* start, DeviceSteps s,
                                       const std::int32_t* accepted, std::uint32_t columns, float* state, cudaStream_t stream) {
    if(!valid(g) || !start || !accepted || !state || start == state) return cudaErrorInvalidValue;
    if(columns && (!s.key || !s.value || !s.log_decay || !s.beta)) return cudaErrorInvalidValue;
    recompute_accepted<<<g.heads, g.values, 0, stream>>>(g, start, s, accepted, columns, state);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_verify_deferred(const DeviceGeometry& g, float* state, DeviceSteps s, std::uint32_t columns,
                                    const std::int32_t* pending, DeferredStash stash, float* outputs, float* work,
                                    cudaStream_t stream) {
    if(!valid(g) || !state || !pending || !stash.key || !stash.decay || !stash.delta || columns > stash.capacity)
        return cudaErrorInvalidValue;
    if(columns && (!outputs || !work || !s.query || !s.key || !s.value || !s.log_decay || !s.beta)) return cudaErrorInvalidValue;
    verify_deferred<<<g.heads, g.values, 0, stream>>>(g, state, s, columns, pending, stash, outputs, work);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_record_accepted(const std::int32_t* accepted, std::uint32_t columns, std::int32_t* pending, cudaStream_t stream) {
    if(!accepted || !pending) return cudaErrorInvalidValue;
    record_accepted<<<1, 1, 0, stream>>>(accepted, columns, pending);
    return cudaPeekAtLastError();
}
}
