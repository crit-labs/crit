// Reference kernel, not for performance. Each subset entry's logit uses the
// same row bytes and the same sequential dot as the full head, so a gathered
// logit equals the full-head logit for that token bit for bit.
#include "subset_argmax.h"
#include <math_constants.h>

namespace crit::techniques::draft_subset {
namespace {
constexpr unsigned threads = 256;

__device__ bool better(float value, std::uint32_t index, float best, std::uint32_t best_index) {
    return value > best || (value == best && index < best_index);
}

__global__ void gather_argmax(GatherArgmax p) {
    __shared__ float values[threads];
    __shared__ std::uint32_t indices[threads];
    const auto b = blockIdx.x;
    const float* state = p.states + std::size_t(b) * p.hidden;
    float best = -CUDART_INF_F;
    std::uint32_t best_index = 0xffffffffu;
    for(std::uint32_t i = threadIdx.x; i < p.count; i += blockDim.x) {
        const auto row = p.rows ? std::size_t(p.rows[i]) : std::size_t(i);
        const float* w = p.weights + row * p.row_stride;
        float sum = 0.0f;
        for(std::uint32_t h = 0; h < p.hidden; ++h) sum += w[h] * state[h];
        if(p.logits) p.logits[std::size_t(b) * p.count + i] = sum;
        if(better(sum, i, best, best_index)) { best = sum; best_index = i; }
    }
    values[threadIdx.x] = best;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for(unsigned step = threads / 2; step; step /= 2) {
        if(threadIdx.x < step && better(values[threadIdx.x + step], indices[threadIdx.x + step], values[threadIdx.x], indices[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + step];
            indices[threadIdx.x] = indices[threadIdx.x + step];
        }
        __syncthreads();
    }
    if(!threadIdx.x) {
        const auto winner = indices[0] == 0xffffffffu ? 0u : indices[0];
        p.tokens[b] = p.rows ? p.rows[winner] : std::int32_t(winner);
    }
}
}

cudaError_t enqueue_gather_argmax(const GatherArgmax& p, cudaStream_t stream) {
    if(!p.weights || !p.states || !p.tokens || !p.count || !p.hidden || p.row_stride < p.hidden) return cudaErrorInvalidValue;
    if(!p.batch) return cudaSuccess;
    gather_argmax<<<p.batch, threads, 0, stream>>>(p);
    return cudaPeekAtLastError();
}
}
