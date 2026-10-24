// Absorbed latent attention; see latent_attention.h and reference.py. Reference SIMT kernel: a block owns 8 heads of
// one query column and one key split; warp w is head w, lane j scores key j of each 32-key tile and owns the latent
// dimensions lane + 32 i of the output. Tensor-core tiling (FlashMLA-style) is the adaptation.
#include "latent_attention.h"

#include <cfloat>

namespace crit::latent_attention {
namespace {

constexpr int kTile = 32, kHeads = 8, kMaxRank = 512, kMaxWidth = 640;

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__global__ void __launch_bounds__(256) attend_kernel(Problem p) {
    extern __shared__ __nv_bfloat162 smem2[];
    const int W = p.rank + p.rope, stride = (W + 2) / 2;   // padded rows: lanes reading one column hit distinct banks
    __nv_bfloat162* Ks = smem2;                            // [32][stride]
    __nv_bfloat162* Qs = Ks + kTile * stride;              // [8][stride]
    const int split = blockIdx.x, t = blockIdx.y, h0 = blockIdx.z * kHeads;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = h0 + warp;
    const int qpos = p.pos0 + t, nkeys = qpos + 1, R = p.rank / 32;
    for(int e = threadIdx.x; e < kHeads * (W / 2); e += blockDim.x) {
        const int r = e / (W / 2), x = e % (W / 2);
        Qs[r * stride + x] = reinterpret_cast<const __nv_bfloat162*>(p.q + (std::size_t(t) * p.heads + h0 + r) * W)[x];
    }
    float m = neg_inf(), l = 0.f, o[kMaxRank / 32];
#pragma unroll
    for(int i = 0; i < kMaxRank / 32; ++i) o[i] = 0.f;
    for(int tile = split;; tile += p.splits) {
        const int key0 = tile * kTile;
        if(key0 >= nkeys) break;
        __syncthreads();   // the previous tile's readers are done (and Qs is staged before the first)
        for(int e = threadIdx.x; e < kTile * (W / 2); e += blockDim.x) {
            const int j = e / (W / 2), x = e % (W / 2), key = key0 + j;
            __nv_bfloat162 v = __floats2bfloat162_rn(0.f, 0.f);
            if(key < nkeys) {
                const std::size_t row = p.table ? std::size_t(p.table[key / p.page_tokens]) * p.page_tokens + key % p.page_tokens : std::size_t(key);
                v = reinterpret_cast<const __nv_bfloat162*>(p.cache + row * W)[x];
            }
            Ks[j * stride + x] = v;
        }
        __syncthreads();
        // Lane j's score for key key0 + j.
        float s = 0.f;
        for(int x = 0; x < W / 2; ++x) {
            const float2 q = __bfloat1622float2(Qs[warp * stride + x]), k = __bfloat1622float2(Ks[lane * stride + x]);
            s = fmaf(q.x, k.x, s);
            s = fmaf(q.y, k.y, s);
        }
        s = key0 + lane <= qpos ? s * p.scale : neg_inf();
        float mx = s;
#pragma unroll
        for(int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        const float mn = fmaxf(m, mx);
        const float c = m == neg_inf() ? 0.f : __expf(m - mn);
        const float pj = s == neg_inf() ? 0.f : __expf(s - mn);
        float sum = pj;
#pragma unroll
        for(int off = 16; off; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
        sum = __shfl_sync(0xffffffffu, sum, 0);
        l = l * c + sum;
        m = mn;
        const __nv_bfloat16* Kb = reinterpret_cast<const __nv_bfloat16*>(Ks);
#pragma unroll
        for(int i = 0; i < kMaxRank / 32; ++i)
            if(i < R) o[i] *= c;
        for(int j = 0; j < kTile; ++j) {
            const float w = __shfl_sync(0xffffffffu, pj, j);
            if(w == 0.f) continue;
#pragma unroll
            for(int i = 0; i < kMaxRank / 32; ++i)
                if(i < R) o[i] = fmaf(w, __bfloat162float(Kb[j * 2 * stride + lane + 32 * i]), o[i]);
        }
    }
    // Partial results of this split: raw output, max and sum.
    const std::size_t unit = (std::size_t(t) * p.heads + h) * p.splits + split;
    float* po = p.partial + unit * p.rank;
    float* pml = p.partial + std::size_t(p.T) * p.heads * p.splits * p.rank + unit * 2;
#pragma unroll
    for(int i = 0; i < kMaxRank / 32; ++i)
        if(i < R) po[lane + 32 * i] = o[i];
    if(lane == 0) { pml[0] = m; pml[1] = l; }
}

// Block (head, column), one thread per latent dimension: splits weighted l_b e^{m_b - M} and added in split order.
__global__ void combine_kernel(Problem p) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const std::size_t unit0 = (std::size_t(t) * p.heads + h) * p.splits;
    const float* po = p.partial + unit0 * p.rank;
    const float* pml = p.partial + std::size_t(p.T) * p.heads * p.splits * p.rank + unit0 * 2;
    float M = p.sinks ? p.sinks[h] : neg_inf();
    for(int b = 0; b < p.splits; ++b) M = fmaxf(M, pml[2 * b]);
    float den = p.sinks ? __expf(p.sinks[h] - M) : 0.f, num = 0.f;
    for(int b = 0; b < p.splits; ++b) {
        const float mb = pml[2 * b];
        if(mb == neg_inf()) continue;
        const float w = __expf(mb - M);
        den = fmaf(pml[2 * b + 1], w, den);
        num = fmaf(po[std::size_t(b) * p.rank + d], w, num);
    }
    p.out[(std::size_t(t) * p.heads + h) * p.rank + d] = den > 0.f ? num / den : 0.f;
}

}  // namespace

std::size_t partial_floats(int T, int heads, int rank, int splits) {
    return std::size_t(T) * heads * splits * (std::size_t(rank) + 2);
}

cudaError_t enqueue(const Problem& p, cudaStream_t stream) {
    const int W = p.rank + p.rope;
    if(!p.q || !p.cache || !p.out || !p.partial || p.T < 1 || p.T > 65535 || p.pos0 < 0 || p.heads < kHeads || p.heads % kHeads ||
       p.heads / kHeads > 65535 || p.rank < 32 || p.rank > kMaxRank || p.rank % 32 || p.rope < 0 || W > kMaxWidth || W % 8 ||
       p.splits < 1 || p.splits > 256 || (p.table && p.page_tokens < 1))
        return cudaErrorInvalidValue;
    const std::size_t bytes = std::size_t(kTile + kHeads) * ((W + 2) / 2) * sizeof(__nv_bfloat162);
    const cudaError_t attr = cudaFuncSetAttribute(attend_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if(attr != cudaSuccess) return attr;
    attend_kernel<<<dim3(p.splits, p.T, p.heads / kHeads), 256, bytes, stream>>>(p);
    combine_kernel<<<dim3(p.heads, p.T), p.rank, 0, stream>>>(p);
    return cudaGetLastError();
}

}  // namespace crit::latent_attention
