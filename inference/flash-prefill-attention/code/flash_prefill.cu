// Flash-style causal prompt attention on BF16 tensor cores (mma.m16n8k16); see flash_prefill.h.
//
// A block of 4 warps owns 64 queries of one head (16 per warp) and walks the visible keys in tiles of 32:
//   S = Q K^T (16 x 32 per warp, FP32), causal mask, running row max m and sum l, P = exp(S - m) rounded to BF16,
//   O = O * exp(m_old - m) + P V. At the end O / l.
// Q, K and V tiles sit in shared memory with a 16-byte XOR swizzle (chunk c of row r at c ^ (r & 7)) so ldmatrix
// reads are conflict free; the next K/V tile is fetched with cp.async while the current one is used.
#include "flash_prefill.h"

#include <cstdint>

namespace crit::flash_prefill {
namespace {

constexpr int kQ = 64, kK = 32;

__device__ __forceinline__ int swz(int row, int chunk, int chunks) { return row * chunks + (chunk ^ (row & 7)); }
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(std::uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
                                         std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N_)); }

template <int HD>
__global__ void __launch_bounds__(128, 1) flash_kernel(Problem p) {
    constexpr int C = HD / 8;  // 16-byte chunks per row
    extern __shared__ uint4 smem[];
    uint4* Qs = smem;              // [64][C]
    uint4* Ks = Qs + kQ * C;       // [2][32][C]
    uint4* Vs = Ks + 2 * kK * C;   // [2][32][C]
    const int h = blockIdx.y, g = h / (p.heads / p.kv_heads);
    const int qt0 = blockIdx.x * kQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int last_q = min(p.T, qt0 + kQ) - 1;
    const int nkeys = p.causal ? p.pos0 + last_q + 1 : p.pos0 + p.T;
    const int ntiles = (nkeys + kK - 1) / kK;
    // Keys before the block's first window are never read (a ring may already hold newer rows there).
    const int klo = p.window > 0 ? max(0, p.pos0 + qt0 - p.window + 1) : 0, jt0 = klo / kK;
    const __nv_bfloat16* kbase = p.k + static_cast<std::size_t>(g) * HD;
    const __nv_bfloat16* vbase = p.v + static_cast<std::size_t>(g) * HD;
    for(int e = tid; e < kQ * C; e += 128) {
        const int r = e / C, c = e % C;
        uint4* dst = Qs + swz(r, c, C);
        if(qt0 + r < p.T) cp_async16(dst, p.q + static_cast<std::size_t>(qt0 + r) * p.q_stride + static_cast<std::size_t>(h) * HD + c * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    auto load_kv = [&](int tile, int buf) {
        uint4* kd = Ks + buf * kK * C;
        uint4* vd = Vs + buf * kK * C;
        for(int e = tid; e < kK * C; e += 128) {
            const int r = e / C, c = e % C, key = tile * kK + r;
            if(key >= klo && key < nkeys) {
                const std::size_t row = static_cast<std::size_t>(p.ring > 0 ? key % p.ring : key);
                cp_async16(kd + swz(r, c, C), kbase + row * p.kv_stride + c * 8);
                cp_async16(vd + swz(r, c, C), vbase + row * p.kv_stride + c * 8);
            } else {
                kd[swz(r, c, C)] = make_uint4(0u, 0u, 0u, 0u);
                vd[swz(r, c, C)] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };
    load_kv(jt0, jt0 & 1);
    cp_async_commit();
    float o[HD / 8][4];
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;  // rows gl and gl + 8 of this warp
    const int qr0 = warp * 16;
    const int qpos0 = p.pos0 + qt0 + qr0 + gl, qpos1 = qpos0 + 8;
    const int lo0 = p.window > 0 ? qpos0 - p.window + 1 : 0, lo1 = p.window > 0 ? qpos1 - p.window + 1 : 0;
    const int hi0 = p.causal ? qpos0 : nkeys - 1, hi1 = p.causal ? qpos1 : nkeys - 1;  // the last key each row sees
    for(int j = jt0; j < ntiles; ++j) {
        if(j + 1 < ntiles) load_kv(j + 1, (j + 1) & 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const uint4* Kb = Ks + (j & 1) * kK * C;
        const uint4* Vb = Vs + (j & 1) * kK * C;
        float s[4][4];
#pragma unroll
        for(int n = 0; n < 4; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
        for(int ks = 0; ks < HD / 16; ++ks) {
            std::uint32_t a[4];
            ldsm_x4(a, Qs + swz(qr0 + (lane & 15), ks * 2 + (lane >> 4), C));
#pragma unroll
            for(int np = 0; np < 2; ++np) {
                std::uint32_t b[4];
                const int key = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(b, Kb + swz(key, ks * 2 + ((lane >> 3) & 1), C));
                mma_bf16(s[2 * np], a[0], a[1], a[2], a[3], b[0], b[1]);
                mma_bf16(s[2 * np + 1], a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }
        const int kfirst = j * kK;
#pragma unroll
        for(int n = 0; n < 4; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int key = kfirst + n * 8 + 2 * t4 + (i & 1);
                const bool visible = (i >> 1) ? key <= hi1 && key >= lo1 : key <= hi0 && key >= lo0;
                float x = s[n][i] * p.scale;
                if(p.softcap > 0.f) x = p.softcap * tanhf(x / p.softcap);
                s[n][i] = visible ? x : -INFINITY;
            }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for(int n = 0; n < 4; ++n) { mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1])); mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float c0 = mx0 == -INFINITY ? 1.f : __expf(m0 - mx0), c1 = mx1 == -INFINITY ? 1.f : __expf(m1 - mx1);
        m0 = mx0; m1 = mx1;
        float rs0 = 0.f, rs1 = 0.f;
        std::uint32_t pf[4][2];
#pragma unroll
        for(int n = 0; n < 4; ++n) {
            const float e0 = mx0 == -INFINITY ? 0.f : __expf(s[n][0] - mx0), e1 = mx0 == -INFINITY ? 0.f : __expf(s[n][1] - mx0);
            const float e2 = mx1 == -INFINITY ? 0.f : __expf(s[n][2] - mx1), e3 = mx1 == -INFINITY ? 0.f : __expf(s[n][3] - mx1);
            const __nv_bfloat162 p01 = __floats2bfloat162_rn(e0, e1), p23 = __floats2bfloat162_rn(e2, e3);
            rs0 += __low2float(p01) + __high2float(p01);
            rs1 += __low2float(p23) + __high2float(p23);
            pf[n][0] = *reinterpret_cast<const std::uint32_t*>(&p01);
            pf[n][1] = *reinterpret_cast<const std::uint32_t*>(&p23);
        }
        l0 = l0 * c0 + rs0; l1 = l1 * c1 + rs1;
#pragma unroll
        for(int n = 0; n < HD / 8; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
#pragma unroll
        for(int kk = 0; kk < 2; ++kk) {
            const std::uint32_t a0 = pf[2 * kk][0], a1 = pf[2 * kk][1], a2 = pf[2 * kk + 1][0], a3 = pf[2 * kk + 1][1];
            const int key = kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
            for(int dp = 0; dp < HD / 16; ++dp) {
                std::uint32_t b[4];
                ldsm_x4_t(b, Vb + swz(key, dp * 2 + (lane >> 4), C));
                mma_bf16(o[2 * dp], a0, a1, a2, a3, b[0], b[1]);
                mma_bf16(o[2 * dp + 1], a0, a1, a2, a3, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    if(p.sinks) {  // the sink's weight joins each row's denominator
        const float sink = p.sinks[h], M0 = fmaxf(m0, sink), M1 = fmaxf(m1, sink);
        const float c0 = m0 == -INFINITY ? 0.f : __expf(m0 - M0), c1 = m1 == -INFINITY ? 0.f : __expf(m1 - M1);
        l0 = l0 * c0 + __expf(sink - M0);
        l1 = l1 * c1 + __expf(sink - M1);
        i0 = c0 / l0;
        i1 = c1 / l1;
    }
    const int t0r = qt0 + qr0 + gl, t1r = t0r + 8;
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) {
        const int d = n * 8 + 2 * t4;
        if(t0r < p.T)
            *reinterpret_cast<float2*>(p.out + static_cast<std::size_t>(t0r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float2(o[n][0] * i0, o[n][1] * i0);
        if(t1r < p.T)
            *reinterpret_cast<float2*>(p.out + static_cast<std::size_t>(t1r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float2(o[n][2] * i1, o[n][3] * i1);
    }
}

template <int HD>
cudaError_t launch(const Problem& p, cudaStream_t stream) {
    const std::size_t bytes = shared_bytes(HD);
    const cudaError_t attr = cudaFuncSetAttribute(flash_kernel<HD>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if(attr != cudaSuccess) return attr;
    flash_kernel<HD><<<dim3((p.T + kQ - 1) / kQ, p.heads), 128, bytes, stream>>>(p);
    return cudaGetLastError();
}

}  // namespace

std::size_t shared_bytes(int head_dim) { return static_cast<std::size_t>(kQ + 4 * kK) * head_dim * 2; }

cudaError_t enqueue(const Problem& p, cudaStream_t stream) {
    if(!p.q || !p.k || !p.v || !p.out || p.T <= 0 || p.pos0 < 0 || p.heads <= 0 || p.kv_heads <= 0 || p.heads % p.kv_heads ||
       p.q_stride % 8 || p.kv_stride % 8 || p.out_stride % 2 || p.q_stride < p.heads * p.head_dim ||
       p.kv_stride < p.kv_heads * p.head_dim || p.out_stride < p.heads * p.head_dim || p.window < 0 || !(p.softcap >= 0.f) ||
       p.ring < 0 || (p.ring > 0 && (p.window <= 0 || p.ring < p.window + p.T - 1)))
        return cudaErrorInvalidValue;
    switch(p.head_dim) {
    case 64: return launch<64>(p, stream);
    case 128: return launch<128>(p, stream);
    case 256: return launch<256>(p, stream);
    default: return cudaErrorInvalidValue;
    }
}

}  // namespace crit::flash_prefill
