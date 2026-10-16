// FP8 KV cache attention; see fp8_kv_attention.h.
//
// Both attention kernels walk keys in 32-key tiles. The next tile's FP8 codes and scales are fetched into registers
// while the current tile is used; between tiles they are dequantized (code * scale, rounded to BF16) into one shared
// K / V stage with a 16-byte XOR swizzle (chunk c of row r at c ^ (r & 7)), so ldmatrix reads are conflict free.
// S = Q K^T and O = P V run on BF16 mma.m16n8k16 with an FP32 online softmax; probabilities are rounded to BF16 and
// the row sums use the same rounded values. A tile entirely after a row's position leaves that row's maximum, sum and
// output bit for bit unchanged (exp(0) = 1 and p = 0), which the decode kernel's column invariance relies on.
#include "fp8_kv_attention.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace crit::fp8_kv {
namespace {

constexpr int kK = 32;             // keys per tile
constexpr int kPiece = 4 * kK;     // keys per decode piece
constexpr int kDecodeRows = 64;    // decode block rows: columns x query heads of one key/value head

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for(int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ std::uint32_t pair(std::uint32_t two, float sc) {
    const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xFFFFu), __NV_E4M3);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&r));
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x * sc, f.y * sc);
    return *reinterpret_cast<const std::uint32_t*>(&b);
}
// Eight codes (two words, first code in the low byte) times the scale -> eight BF16 values.
__device__ __forceinline__ uint4 dequant8(std::uint32_t w0, std::uint32_t w1, float sc) {
    return make_uint4(pair(w0, sc), pair(w0 >> 16, sc), pair(w1, sc), pair(w1 >> 16, sc));
}
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

// ------------------------------------------------------------------------------------------------ write
template <int HD>
__global__ void __launch_bounds__(HD) write_kernel(Cache c, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                                                   int stride, int pos0) {
    __shared__ float red[2][HD / 32];
    const int t = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
    const float kx = __bfloat162float(k[static_cast<std::size_t>(t) * stride + g * HD + d]);
    const float vx = __bfloat162float(v[static_cast<std::size_t>(t) * stride + g * HD + d]);
    const float ak = warp_max(fabsf(kx)), av = warp_max(fabsf(vx));
    if((d & 31) == 0) { red[0][d >> 5] = ak; red[1][d >> 5] = av; }
    __syncthreads();
    float mk = 0.f, mv = 0.f;
#pragma unroll
    for(int i = 0; i < HD / 32; ++i) { mk = fmaxf(mk, red[0][i]); mv = fmaxf(mv, red[1][i]); }
    const float sk = mk > 0.f ? mk / 448.f : 1.f, sv = mv > 0.f ? mv / 448.f : 1.f;
    const std::size_t pos = static_cast<std::size_t>(pos0) + t, off = (pos * c.kv_heads + g) * HD + d;
    c.k[off] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(kx / sk, __NV_SATFINITE, __NV_E4M3));
    c.v[off] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(vx / sv, __NV_SATFINITE, __NV_E4M3));
    if(d == 0) {
        c.k_scale[pos * c.kv_heads + g] = sk;
        c.v_scale[pos * c.kv_heads + g] = sv;
    }
}

// ------------------------------------------------------------------------------------------------ shared tile machinery
// One 32-key FP8 tile of one key/value head: NT threads own its 16-byte chunks (row e / (HD / 16), chunk e % (HD / 16)).
template <int HD, int NT>
struct Tile {
    static constexpr int C = HD / 8;                 // 16-byte BF16 chunks per row
    static constexpr int CH = kK * HD / 16;          // 16-byte FP8 chunks per tile
    static constexpr int PER = (CH + NT - 1) / NT;
    uint4 kr[PER], vr[PER];
    float ks[PER], vs[PER];

    __device__ __forceinline__ static int swz(int row, int chunk) { return row * C + (chunk ^ (row & 7)); }

    __device__ __forceinline__ void fetch(const Cache& c, int g, int key0, int nkeys, int tid) {
#pragma unroll
        for(int i = 0; i < PER; ++i) {
            const int e = tid + NT * i, key = key0 + e / (HD / 16);
            if(e < CH && key < nkeys) {
                const std::size_t off = (static_cast<std::size_t>(key) * c.kv_heads + g) * HD + (e % (HD / 16)) * 16;
                kr[i] = __ldcg(reinterpret_cast<const uint4*>(c.k + off));
                vr[i] = __ldcg(reinterpret_cast<const uint4*>(c.v + off));
                ks[i] = __ldcg(c.k_scale + static_cast<std::size_t>(key) * c.kv_heads + g);
                vs[i] = __ldcg(c.v_scale + static_cast<std::size_t>(key) * c.kv_heads + g);
            } else {
                kr[i] = vr[i] = make_uint4(0u, 0u, 0u, 0u);
                ks[i] = vs[i] = 0.f;
            }
        }
    }
    __device__ __forceinline__ void stage(uint4* Ks, uint4* Vs, int tid) const {
#pragma unroll
        for(int i = 0; i < PER; ++i) {
            const int e = tid + NT * i;
            if(e >= CH) break;
            const int r = e / (HD / 16), cc = e % (HD / 16);
            Ks[swz(r, 2 * cc)] = dequant8(kr[i].x, kr[i].y, ks[i]);
            Ks[swz(r, 2 * cc + 1)] = dequant8(kr[i].z, kr[i].w, ks[i]);
            Vs[swz(r, 2 * cc)] = dequant8(vr[i].x, vr[i].y, vs[i]);
            Vs[swz(r, 2 * cc + 1)] = dequant8(vr[i].z, vr[i].w, vs[i]);
        }
    }
};

// One warp's 16 query rows against one staged 32-key tile: S = Q K^T, causal mask (key > position: excluded), online
// softmax update of m / l and O = O * c + P V. qpos < 0 marks an unused row.
template <int HD>
struct Rows {
    static constexpr int C = HD / 8;
    float o[HD / 8][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    __device__ __forceinline__ Rows() {
#pragma unroll
        for(int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    }
    __device__ __forceinline__ static int swz(int row, int chunk) { return row * C + (chunk ^ (row & 7)); }

    __device__ __forceinline__ void step(const uint4* Qs, const uint4* Ks, const uint4* Vs, int qr0, int kfirst, int qpos0, int qpos1,
                                         float scale, int lane) {
        const int t4 = lane & 3;
        float s[4][4];
#pragma unroll
        for(int n = 0; n < 4; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
        for(int ks = 0; ks < HD / 16; ++ks) {
            std::uint32_t a[4];
            ldsm_x4(a, Qs + swz(qr0 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
            for(int np = 0; np < 2; ++np) {
                std::uint32_t b[4];
                const int key = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(b, Ks + swz(key, ks * 2 + ((lane >> 3) & 1)));
                mma_bf16(s[2 * np], a[0], a[1], a[2], a[3], b[0], b[1]);
                mma_bf16(s[2 * np + 1], a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }
#pragma unroll
        for(int n = 0; n < 4; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int key = kfirst + n * 8 + 2 * t4 + (i & 1);
                s[n][i] = key > ((i >> 1) ? qpos1 : qpos0) ? -INFINITY : s[n][i] * scale;
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
                ldsm_x4_t(b, Vs + swz(key, dp * 2 + (lane >> 4)));
                mma_bf16(o[2 * dp], a0, a1, a2, a3, b[0], b[1]);
                mma_bf16(o[2 * dp + 1], a0, a1, a2, a3, b[2], b[3]);
            }
        }
    }
    __device__ __forceinline__ void finish_sums() {
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    }
};

// ------------------------------------------------------------------------------------------------ prefill
constexpr int kQ = 128, kPrefillThreads = 256;

template <int HD>
__global__ void __launch_bounds__(kPrefillThreads, 1) prefill_kernel(Prefill p) {
    using T_ = Tile<HD, kPrefillThreads>;
    constexpr int C = HD / 8;
    extern __shared__ uint4 smem[];
    uint4* Qs = smem;
    uint4* Ks = Qs + kQ * C;
    uint4* Vs = Ks + kK * C;
    const int h = blockIdx.y, g = h / (p.heads / p.cache.kv_heads);
    const int qt0 = (gridDim.x - 1 - blockIdx.x) * kQ;  // the latest (heaviest) query tile first
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int last_q = min(p.T, qt0 + kQ) - 1;
    const int nkeys = p.pos0 + last_q + 1;
    const int ntiles = (nkeys + kK - 1) / kK;
    for(int e = tid; e < kQ * C; e += kPrefillThreads) {
        const int r = e / C, c = e % C;
        uint4* dst = Qs + T_::swz(r, c);
        if(qt0 + r < p.T) cp_async16(dst, p.q + static_cast<std::size_t>(qt0 + r) * p.q_stride + static_cast<std::size_t>(h) * HD + c * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;");
    T_ tile;
    tile.fetch(p.cache, g, 0, nkeys, tid);
    tile.stage(Ks, Vs, tid);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    Rows<HD> rows;
    const int qr0 = warp * 16;
    const int qpos0 = p.pos0 + qt0 + qr0 + gl, qpos1 = qpos0 + 8, warp_last = p.pos0 + qt0 + qr0 + 15;
    for(int j = 0; j < ntiles; ++j) {
        const bool more = j + 1 < ntiles;
        if(more) tile.fetch(p.cache, g, (j + 1) * kK, nkeys, tid);
        if(j * kK <= warp_last) rows.step(Qs, Ks, Vs, qr0, j * kK, qpos0, qpos1, p.scale, lane);
        if(!more) break;
        __syncthreads();
        tile.stage(Ks, Vs, tid);
        __syncthreads();
    }
    rows.finish_sums();
    const float i0 = rows.l0 > 0.f ? 1.f / rows.l0 : 0.f, i1 = rows.l1 > 0.f ? 1.f / rows.l1 : 0.f;
    const int t0r = qt0 + qr0 + gl, t1r = t0r + 8;
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) {
        const int d = n * 8 + 2 * t4;
        if(t0r < p.T)
            *reinterpret_cast<float2*>(p.out + static_cast<std::size_t>(t0r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float2(rows.o[n][0] * i0, rows.o[n][1] * i0);
        if(t1r < p.T)
            *reinterpret_cast<float2*>(p.out + static_cast<std::size_t>(t1r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float2(rows.o[n][2] * i1, rows.o[n][3] * i1);
    }
}

// ------------------------------------------------------------------------------------------------ decode
constexpr int kDecodeThreads = 128;

template <int HD>
__global__ void __launch_bounds__(kDecodeThreads, 1) decode_kernel(Decode p) {
    using T_ = Tile<HD, kDecodeThreads>;
    constexpr int C = HD / 8;
    extern __shared__ uint4 smem[];
    uint4* Qs = smem;
    uint4* Ks = Qs + kDecodeRows * C;
    uint4* Vs = Ks + kK * C;
    const int g = blockIdx.x, blk = blockIdx.y, group = p.heads / p.cache.kv_heads;
    const int pos0 = p.pos0_device ? *p.pos0_device : p.pos0;
    const int nkeys = pos0 + p.T;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    float* part = p.partial + (static_cast<std::size_t>(g) * p.blocks + blk) * kDecodeRows * HD;
    float* ml = p.partial + static_cast<std::size_t>(p.cache.kv_heads) * p.blocks * kDecodeRows * HD +
                (static_cast<std::size_t>(g) * p.blocks + blk) * kDecodeRows * 2;
    auto tile_key = [&](int j) { return (blk + (j >> 2) * p.blocks) * kPiece + (j & 3) * kK; };
    if(tile_key(0) >= nkeys) {
        for(int r = tid; r < kDecodeRows; r += kDecodeThreads) { ml[r * 2] = -INFINITY; ml[r * 2 + 1] = 0.f; }
        return;
    }
    const int used = p.T * group;
    for(int e = tid; e < kDecodeRows * C; e += kDecodeThreads) {
        const int r = e / C, c = e % C;
        uint4* dst = Qs + T_::swz(r, c);
        if(r < used) cp_async16(dst, p.q + static_cast<std::size_t>(r / group) * p.q_stride + static_cast<std::size_t>(g * group + r % group) * HD + c * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;");
    T_ tile;
    tile.fetch(p.cache, g, tile_key(0), nkeys, tid);
    tile.stage(Ks, Vs, tid);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    Rows<HD> rows;
    const int qr0 = warp * 16, r0 = qr0 + gl, r1 = r0 + 8;
    const int qpos0 = r0 < used ? pos0 + r0 / group : -1, qpos1 = r1 < used ? pos0 + r1 / group : -1;
    for(int j = 0;; ++j) {
        const int kfirst = tile_key(j), knext = tile_key(j + 1);
        const bool more = knext < nkeys;
        if(more) tile.fetch(p.cache, g, knext, nkeys, tid);
        rows.step(Qs, Ks, Vs, qr0, kfirst, qpos0, qpos1, p.scale, lane);
        if(!more) break;
        __syncthreads();
        tile.stage(Ks, Vs, tid);
        __syncthreads();
    }
    rows.finish_sums();
    const float i0 = rows.l0 > 0.f ? 1.f / rows.l0 : 0.f, i1 = rows.l1 > 0.f ? 1.f / rows.l1 : 0.f;
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) {
        const int dd = n * 8 + 2 * t4;
        *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r0) * HD + dd) = make_float2(rows.o[n][0] * i0, rows.o[n][1] * i0);
        *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r1) * HD + dd) = make_float2(rows.o[n][2] * i1, rows.o[n][3] * i1);
    }
    if(t4 == 0) {
        ml[r0 * 2] = rows.m0; ml[r0 * 2 + 1] = rows.l0;
        ml[r1 * 2] = rows.m1; ml[r1 * 2 + 1] = rows.l1;
    }
}

// Block (column, query head), one thread per dimension: block weights l_b exp(m_b - M), summed in block order.
template <int HD>
__global__ void __launch_bounds__(HD) combine_kernel(Decode p) {
    extern __shared__ float wsp[];
    __shared__ float total;
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, group = p.heads / p.cache.kv_heads, g = h / group;
    const int r = t * group + h % group;
    const int pos0 = p.pos0_device ? *p.pos0_device : p.pos0;
    const int ns = min(p.blocks, (pos0 + p.T + kPiece - 1) / kPiece);  // blocks that hold at least one key
    const float* part = p.partial + static_cast<std::size_t>(g) * p.blocks * kDecodeRows * HD;
    const float* ml = p.partial + static_cast<std::size_t>(p.cache.kv_heads) * p.blocks * kDecodeRows * HD +
                      static_cast<std::size_t>(g) * p.blocks * kDecodeRows * 2;
    for(int b = d; b < ns; b += blockDim.x) wsp[b] = ml[(static_cast<std::size_t>(b) * kDecodeRows + r) * 2];
    __syncthreads();
    if(d == 0) {
        float M = -INFINITY;
        for(int b = 0; b < ns; ++b) M = fmaxf(M, wsp[b]);
        float den = 0.f;
        for(int b = 0; b < ns; ++b) {
            const float m = wsp[b], l = ml[(static_cast<std::size_t>(b) * kDecodeRows + r) * 2 + 1];
            const float w = m == -INFINITY ? 0.f : l * __expf(m - M);
            wsp[b] = w;
            den += w;
        }
        total = den;
    }
    __syncthreads();
    float num = 0.f;
    for(int b = 0; b < ns; ++b) {
        const float w = wsp[b];
        if(w != 0.f) num = fmaf(w, part[(static_cast<std::size_t>(b) * kDecodeRows + r) * HD + d], num);
    }
    p.out[static_cast<std::size_t>(t) * p.out_stride + static_cast<std::size_t>(h) * HD + d] = total > 0.f ? num / total : 0.f;
}

bool valid(const Cache& c) {
    return c.k && c.v && c.k_scale && c.v_scale && c.kv_heads > 0 && (c.head_dim == 64 || c.head_dim == 128 || c.head_dim == 256);
}

template <int HD>
cudaError_t run_prefill(const Prefill& p, cudaStream_t s) {
    const std::size_t bytes = prefill_shared_bytes(HD);
    const cudaError_t attr = cudaFuncSetAttribute(prefill_kernel<HD>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if(attr != cudaSuccess) return attr;
    prefill_kernel<HD><<<dim3((p.T + kQ - 1) / kQ, p.heads), kPrefillThreads, bytes, s>>>(p);
    return cudaGetLastError();
}

template <int HD>
cudaError_t run_decode(const Decode& p, cudaStream_t s) {
    const std::size_t bytes = static_cast<std::size_t>(kDecodeRows + 2 * kK) * HD * 2;
    const cudaError_t attr = cudaFuncSetAttribute(decode_kernel<HD>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if(attr != cudaSuccess) return attr;
    decode_kernel<HD><<<dim3(p.cache.kv_heads, p.blocks), kDecodeThreads, bytes, s>>>(p);
    combine_kernel<HD><<<dim3(p.T, p.heads), HD, sizeof(float) * p.blocks, s>>>(p);
    return cudaGetLastError();
}

}  // namespace

cudaError_t write(const Cache& c, const __nv_bfloat16* k, const __nv_bfloat16* v, int stride, int T, int pos0, cudaStream_t s) {
    if(!valid(c) || !k || !v || T <= 0 || pos0 < 0 || stride < c.kv_heads * c.head_dim) return cudaErrorInvalidValue;
    const dim3 grid(T, c.kv_heads);
    switch(c.head_dim) {
    case 64: write_kernel<64><<<grid, 64, 0, s>>>(c, k, v, stride, pos0); break;
    case 128: write_kernel<128><<<grid, 128, 0, s>>>(c, k, v, stride, pos0); break;
    default: write_kernel<256><<<grid, 256, 0, s>>>(c, k, v, stride, pos0); break;
    }
    return cudaGetLastError();
}

std::size_t prefill_shared_bytes(int head_dim) { return static_cast<std::size_t>(kQ + 2 * kK) * head_dim * 2; }

cudaError_t prefill(const Prefill& p, cudaStream_t s) {
    if(!valid(p.cache) || !p.q || !p.out || p.T <= 0 || p.pos0 < 0 || p.heads <= 0 || p.heads % p.cache.kv_heads ||
       p.q_stride % 8 || p.q_stride < p.heads * p.cache.head_dim || p.out_stride % 2 || p.out_stride < p.heads * p.cache.head_dim)
        return cudaErrorInvalidValue;
    switch(p.cache.head_dim) {
    case 64: return run_prefill<64>(p, s);
    case 128: return run_prefill<128>(p, s);
    default: return run_prefill<256>(p, s);
    }
}

std::size_t decode_partial_floats(int kv_heads, int head_dim, int blocks) {
    return static_cast<std::size_t>(kv_heads) * blocks * kDecodeRows * (head_dim + 2);
}

cudaError_t decode(const Decode& p, cudaStream_t s) {
    if(!valid(p.cache) || !p.q || !p.out || !p.partial || p.T <= 0 || p.heads <= 0 || p.heads % p.cache.kv_heads ||
       p.T * (p.heads / p.cache.kv_heads) > kDecodeRows || p.blocks < 1 || p.blocks > 1024 || (!p.pos0_device && p.pos0 < 0) ||
       p.q_stride % 8 || p.q_stride < p.heads * p.cache.head_dim || p.out_stride < p.heads * p.cache.head_dim)
        return cudaErrorInvalidValue;
    switch(p.cache.head_dim) {
    case 64: return run_decode<64>(p, s);
    case 128: return run_decode<128>(p, s);
    default: return run_decode<256>(p, s);
    }
}

}  // namespace crit::fp8_kv
