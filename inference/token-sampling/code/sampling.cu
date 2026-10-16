#include "sampling.h"

#include <crit/sampling.h>

#include <cfloat>

namespace crit::techniques::token_sampling {
namespace {

constexpr int kThreads = 256;
constexpr int kNone = 0x7fffffff;

__device__ __forceinline__ bool before(float v, int i, float w, int j) { return v > w || (v == w && i < j); }

__device__ int candidates(const Params* p) {
    return p->temperature > 0.f ? max(1, min(p->top_k > 0 ? p->top_k : kMaxTopK, kMaxTopK)) : 1;
}

// Block-wide best (value, index) in rank order; every thread receives it.
__device__ void block_best(float& v, int& i) {
    __shared__ float sv[32];
    __shared__ int si[32];
#pragma unroll
    for(int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if(before(ov, oi, v, i)) { v = ov; i = oi; }
    }
    __syncthreads();
    if((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = v; si[threadIdx.x >> 5] = i; }
    __syncthreads();
    if(threadIdx.x == 0) {
        for(int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w)
            if(before(sv[w], si[w], v, i)) { v = sv[w]; i = si[w]; }
        sv[0] = v;
        si[0] = i;
    }
    __syncthreads();
    v = sv[0];
    i = si[0];
}

// Stage 1: block b's k best of its slice, in rank order (k block-wide selections).
__global__ void __launch_bounds__(kThreads) local_kernel(const float* __restrict__ logits, int vocab, const Params* p,
                                                         float* cand_values, std::int32_t* cand_ids) {
    const int k = candidates(p);
    const int chunk = (vocab + kStageBlocks - 1) / kStageBlocks;
    const int lo = blockIdx.x * chunk, hi = min(vocab, lo + chunk);
    float pv = FLT_MAX;
    int pi = -1;
    for(int r = 0; r < k; ++r) {
        float v = -FLT_MAX;
        int i = kNone;
        for(int j = lo + threadIdx.x; j < hi; j += kThreads) {
            const float x = logits[j];
            if(before(pv, pi, x, j) && before(x, j, v, i)) { v = x; i = j; }
        }
        block_best(v, i);
        if(threadIdx.x == 0) { cand_values[blockIdx.x * kMaxTopK + r] = v; cand_ids[blockIdx.x * kMaxTopK + r] = i; }
        pv = v;
        pi = i;
    }
}

// Stage 2: global k best from the candidates; then temperature, top-p and one draw.
__global__ void __launch_bounds__(1024) pick_kernel(const float* cand_values, const std::int32_t* cand_ids, Params* p,
                                                    std::int32_t* out) {
    __shared__ float tv[kMaxTopK];
    __shared__ int ti[kMaxTopK];
    const int k = candidates(p);
    float pv = FLT_MAX;
    int pi = -1;
    for(int r = 0; r < k; ++r) {
        float v = -FLT_MAX;
        int i = kNone;
        for(int n = threadIdx.x; n < kStageBlocks * k; n += blockDim.x) {
            const int b = n / k, c = n % k;
            const float x = cand_values[b * kMaxTopK + c];
            const int j = cand_ids[b * kMaxTopK + c];
            if(j != kNone && before(pv, pi, x, j) && before(x, j, v, i)) { v = x; i = j; }
        }
        block_best(v, i);
        if(threadIdx.x == 0) { tv[r] = v; ti[r] = i; }
        pv = v;
        pi = i;
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    if(!(p->temperature > 0.f)) { out[0] = ti[0]; p->position += 1; return; }
    float w[kMaxTopK], total = 0.f;
    int n = 0;
    for(int r = 0; r < k && ti[r] != kNone; ++r, ++n) {
        w[r] = expf((tv[r] - tv[0]) / p->temperature);
        total += w[r];
    }
    const float top_p = p->top_p > 0.f && p->top_p < 1.f ? p->top_p : 1.f;
    int keep = n;
    float acc = 0.f;
    for(int r = 0; r < n; ++r) {
        acc += w[r] / total;
        if(acc >= top_p) { keep = r + 1; break; }
    }
    float kept = 0.f;
    for(int r = 0; r < keep; ++r) kept += w[r];
    // Core's draw: the first kept token whose cumulative weight exceeds u * kept.
    const double target = crit::core::uniform(p->seed, p->position, p->stream, crit::core::DrawPurpose::sample) * kept;
    p->position += 1;
    float c = 0.f;
    int chosen = ti[keep - 1];
    for(int r = 0; r < keep; ++r) {
        c += w[r];
        if(target < c) { chosen = ti[r]; break; }
    }
    out[0] = chosen;
}

} // namespace

cudaError_t enqueue_select(const float* logits, int vocab, Params* params, float* cand_values, std::int32_t* cand_ids,
                           std::int32_t* out, cudaStream_t stream) {
    if(!logits || vocab < 1 || !params || !cand_values || !cand_ids || !out) return cudaErrorInvalidValue;
    local_kernel<<<kStageBlocks, kThreads, 0, stream>>>(logits, vocab, params, cand_values, cand_ids);
    pick_kernel<<<1, 1024, 0, stream>>>(cand_values, cand_ids, params, out);
    return cudaGetLastError();
}

} // namespace crit::techniques::token_sampling
