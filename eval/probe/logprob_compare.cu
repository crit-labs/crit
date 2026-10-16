// Reference kernel, not for performance. Each block reduces one row with a fixed
// tree order, so repeated launches give bitwise identical results.
#include "logprob_compare.h"
#include <cmath>

namespace crit::eval::distribution {
namespace {
constexpr unsigned threads = 256;

template <class Op>
__device__ float block_reduce(float value, float* scratch, Op op) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for(unsigned step = threads / 2; step; step /= 2) {
        if(threadIdx.x < step) scratch[threadIdx.x] = op(scratch[threadIdx.x], scratch[threadIdx.x + step]);
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__global__ void row_compare(RowCompare p) {
    __shared__ float scratch[threads];
    const std::size_t r = blockIdx.x;
    const float* row = p.logits + r * p.logits_stride;
    float top = -INFINITY;
    for(std::uint32_t i = threadIdx.x; i < p.valid; i += threads) top = fmaxf(top, row[i]);
    top = block_reduce(top, scratch, [](float a, float b) { return fmaxf(a, b); });
    float sum = 0.0f;
    for(std::uint32_t i = threadIdx.x; i < p.valid; i += threads) sum += expf(row[i] - top);
    sum = block_reduce(sum, scratch, [](float a, float b) { return a + b; });
    const float lse = top + logf(sum);
    if(p.store)
        for(std::uint32_t i = threadIdx.x; i < p.valid; i += threads) p.store[r * p.valid + i] = row[i] - lse;
    if(p.reference) {
        const float* ref = p.reference + r * p.valid;
        float kl = 0.0f;
        for(std::uint32_t i = threadIdx.x; i < p.valid; i += threads) {
            const float q = ref[i];
            kl += expf(q) * (q - (row[i] - lse));
        }
        kl = block_reduce(kl, scratch, [](float a, float b) { return a + b; });
        if(!threadIdx.x) p.kld[r] = fmaxf(kl, 0.0f);
    }
    if(p.targets && !threadIdx.x) {
        const auto t = p.targets[r];
        p.nll[r] = t >= 0 && std::uint32_t(t) < p.valid ? lse - row[t] : NAN;  // out-of-range target: NaN, never a read
    }
}
}

cudaError_t enqueue_row_compare(const RowCompare& p, cudaStream_t stream) {
    if(!p.logits || !p.valid || p.logits_stride < p.valid) return cudaErrorInvalidValue;
    if((p.targets == nullptr) != (p.nll == nullptr) || (p.reference == nullptr) != (p.kld == nullptr)) return cudaErrorInvalidValue;
    if(!p.targets && !p.reference && !p.store) return cudaErrorInvalidValue;  // nothing to produce
    if(!p.rows) return cudaSuccess;
    row_compare<<<p.rows, threads, 0, stream>>>(p);
    return cudaPeekAtLastError();
}
}
