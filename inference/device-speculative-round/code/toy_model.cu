// Reference kernels for the synthetic toy target, not for performance.
#include "toy_model.h"
#include "greedy_accept.h"
#include "recurrent_cuda.h"
#include <cmath>

namespace crit::techniques::device_round::toy {
namespace {
namespace rc = crit::techniques::recurrent::cuda;
constexpr unsigned threads = 128;

__device__ float dot(const float* a, const float* b, std::uint32_t n) {
    float sum = 0.0f;
    for(std::uint32_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

// Columns run in order inside one block; column t attends positions 0..p0+t.
__global__ void attention(Weights w, State s, const std::int32_t* inputs, const std::int32_t* frontier, std::uint32_t columns) {
    __shared__ float x[width], q[width], out[width], scores[context];
    const auto tid = threadIdx.x;
    const auto p0 = std::uint32_t(*frontier);
    const float scale = rsqrtf(float(width));
    for(std::uint32_t t = 0; t < columns; ++t) {
        const auto position = p0 + t;
        if(tid < width) x[tid] = w.embed[std::size_t(inputs[t]) * width + tid];
        __syncthreads();
        if(tid < width) {
            q[tid] = dot(w.wq + tid * width, x, width);
            s.k_cache[std::size_t(position) * width + tid] = dot(w.wk + tid * width, x, width);
            s.v_cache[std::size_t(position) * width + tid] = dot(w.wv + tid * width, x, width);
        }
        __syncthreads();
        for(std::uint32_t p = tid; p <= position; p += blockDim.x) scores[p] = dot(q, s.k_cache + std::size_t(p) * width, width) * scale;
        __syncthreads();
        if(tid == 0) { // Fixed sequential softmax order.
            float m = scores[0];
            for(std::uint32_t p = 1; p <= position; ++p) m = fmaxf(m, scores[p]);
            float sum = 0.0f;
            for(std::uint32_t p = 0; p <= position; ++p) { scores[p] = expf(scores[p] - m); sum += scores[p]; }
            for(std::uint32_t p = 0; p <= position; ++p) scores[p] /= sum;
        }
        __syncthreads();
        if(tid < width) {
            float o = 0.0f;
            for(std::uint32_t p = 0; p <= position; ++p) o += scores[p] * s.v_cache[std::size_t(p) * width + tid];
            out[tid] = o;
        }
        __syncthreads();
        if(tid < width) s.x1[t * width + tid] = x[tid] + dot(w.wo + tid * width, out, width);
        __syncthreads();
    }
}

// Projected, gated gated-delta inputs for every column.
__global__ void gdn_inputs(Weights w, State s) {
    const auto t = blockIdx.x, tid = threadIdx.x;
    const float* x = s.x1 + t * width;
    if(tid < keys) {
        s.q[t * keys + tid] = dot(w.pq + tid * width, x, width);
        s.k[t * keys + tid] = dot(w.pk + tid * width, x, width);
    }
    if(tid < values) s.v[t * values + tid] = dot(w.pv + tid * width, x, width);
    if(tid == 0) {
        const float a = dot(w.pa, x, width), b = dot(w.pb, x, width);
        s.log_decay[t] = -log1pf(expf(a)); // -softplus <= 0
        s.beta[t] = 1.0f / (1.0f + expf(-b));
    }
}

__global__ void head(Weights w, State s, std::int32_t* predictions) {
    __shared__ float x2[width], logits[vocabulary];
    const auto t = blockIdx.x, tid = threadIdx.x;
    if(tid < width) x2[tid] = s.x1[t * width + tid] + dot(w.po + tid * values, s.gdn_out + t * values, values);
    __syncthreads();
    for(std::uint32_t r = tid; r < vocabulary; r += blockDim.x) logits[r] = dot(w.head + r * width, x2, width);
    __syncthreads();
    if(tid == 0) {
        std::uint32_t best = 0;
        for(std::uint32_t r = 1; r < vocabulary; ++r) if(logits[r] > logits[best]) best = r;
        predictions[t] = std::int32_t(best);
    }
}

__global__ void round_inputs(const std::int32_t* control, const std::int32_t* drafts, std::uint32_t k, std::int32_t* inputs) {
    inputs[0] = control[control::last_token];
    for(std::uint32_t i = 0; i < k; ++i) inputs[1 + i] = drafts[i];
}

__global__ void bigram(const std::int32_t* table, const std::int32_t* control, std::uint32_t k, std::int32_t* drafts) {
    auto token = control[control::last_token];
    for(std::uint32_t i = 0; i < k; ++i) { token = table[token]; drafts[i] = token; }
}

__global__ void set(std::int32_t* target, std::int32_t value) { *target = value; }

rc::DeviceGeometry geometry() { return {1, keys, values, 1e-6f, 1.0f / std::sqrt(float(keys)), true, true}; }
rc::DeviceSteps steps(const State& s) { return {s.q, s.k, s.v, s.log_decay, s.beta}; }
}

cudaError_t enqueue_forward(const Weights& w, const State& s, const std::int32_t* inputs, const std::int32_t* frontier,
                            std::uint32_t columns, std::int32_t* predictions, cudaStream_t stream) {
    if(!columns || columns > max_columns) return cudaErrorInvalidValue;
    if(auto e = cudaMemcpyAsync(s.gdn_start, s.gdn_live, keys * values * sizeof(float), cudaMemcpyDeviceToDevice, stream); e != cudaSuccess) return e;
    attention<<<1, threads, 0, stream>>>(w, s, inputs, frontier, columns);
    gdn_inputs<<<columns, threads, 0, stream>>>(w, s);
    if(auto e = cudaPeekAtLastError(); e != cudaSuccess) return e;
    if(auto e = rc::enqueue_verify(geometry(), s.gdn_start, steps(s), columns, s.gdn_out, s.gdn_checkpoints, nullptr, stream); e != cudaSuccess) return e;
    head<<<columns, threads, 0, stream>>>(w, s, predictions);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_commit(const State& s, const std::int32_t* accepted, std::uint32_t columns, cudaStream_t stream) {
    return rc::enqueue_select_accepted(geometry(), s.gdn_start, s.gdn_checkpoints, accepted, columns, s.gdn_live, stream);
}

cudaError_t enqueue_round_inputs(const std::int32_t* control, const std::int32_t* drafts, std::uint32_t k,
                                 std::int32_t* inputs, cudaStream_t stream) {
    round_inputs<<<1, 1, 0, stream>>>(control, drafts, k, inputs);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_bigram_draft(const std::int32_t* table, const std::int32_t* control, std::uint32_t k,
                                 std::int32_t* drafts, cudaStream_t stream) {
    bigram<<<1, 1, 0, stream>>>(table, control, k, drafts);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_set(std::int32_t* target, std::int32_t value, cudaStream_t stream) {
    set<<<1, 1, 0, stream>>>(target, value);
    return cudaPeekAtLastError();
}
}
