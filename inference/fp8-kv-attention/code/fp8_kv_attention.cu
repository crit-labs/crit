// FP8 KV cache attention; see fp8_kv_attention.h.
//
// Both attention kernels walk keys in 32-key tiles. The next tile's FP8 codes and scales are fetched into registers
// while the current tile is used; between tiles they are dequantized (code * scale, rounded to BF16) into one shared
// K / V stage with a 16-byte XOR swizzle (chunk c of row r at c ^ (r & 7)), so ldmatrix reads are conflict free.
// S = Q K^T and O = P V run on BF16 mma.m16n8k16 with an FP32 online softmax; probabilities are rounded to BF16 and
// the row sums use the same rounded values. A tile entirely outside a row's visible keys (after its position, before
// its window, off its tree path) leaves that row's maximum, sum and output bit for bit unchanged (exp(0) = 1 and
// p = 0), which the decode kernel's column invariance relies on: skipping such tiles, or reading keys only from the
// oldest row's window on, changes no bits. Two shortcuts keep the bits too: the per-key mask is skipped where both of a
// thread's rows see the whole tile, and a warp's output rescale where none of its row maxima moved (x * 1 is exact).
#include "fp8_kv_attention.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <climits>
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
__device__ __forceinline__ std::size_t cache_row(const Cache& c, int key) {
    return static_cast<std::size_t>(c.ring > 0 ? key % c.ring : key);
}
// Positions pos0 .. pos0 + T - 1 do not all lie in the cache: pos0 is negative, or a linear cache with a capacity ends
// before pos0 + T.
__host__ __device__ __forceinline__ bool outside(const Cache& c, long long pos0, int T) {
    return pos0 < 0 || (c.ring == 0 && c.capacity > 0 && pos0 + T > c.capacity);
}

// The keys one query row sees: key <= pos (pos < 0: an unused row), key >= lo (sliding window), and keys from `base`
// on only where the row's tree bit (key - base) is set.
struct Visible {
    int pos = -1, lo = 0, base = INT_MAX;
    std::uint64_t tree = ~0ull;
    __device__ __forceinline__ bool operator()(int key) const {
        return key <= pos && key >= lo && (key < base || key - base >= 64 || ((tree >> (key - base)) & 1ull));
    }
    // Every key of the tile kfirst .. kfirst + keys - 1 is visible, so the per-key mask would change nothing.
    __device__ __forceinline__ bool all(int kfirst, int keys) const { return kfirst >= lo && kfirst + keys - 1 <= pos && kfirst + keys - 1 < base; }
};
__device__ __forceinline__ int window_lo(int pos, int window) { return window > 0 ? max(0, pos - window + 1) : 0; }
__device__ __forceinline__ float capped(float x, float cap) { return cap > 0.f ? cap * tanhf(x / cap) : x; }
// Folds a sink logit into a finished row (natural-log units): its weight exp(sink) joins the denominator. Returns the
// factor that turns the row's unnormalized output into its result.
__device__ __forceinline__ float finish_row(float m, float& l, const float* sinks, int h) {
    if(!sinks) return l > 0.f ? 1.f / l : 0.f;
    const float sink = sinks[h], M = fmaxf(m, sink);
    const float c = m == -INFINITY ? 0.f : __expf(m - M);
    l = l * c + __expf(sink - M);
    return c / l;
}

// ------------------------------------------------------------------------------------------------ write
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_float(float x) { return x; }

// Block (row t, key/value head g), one thread per dimension. The position comes from *pos0_device when it is set; a
// row outside the cache is skipped (host positions were checked before the launch).
template <int HD, class In>
__global__ void __launch_bounds__(HD) write_kernel(Cache c, const In* __restrict__ k, const In* __restrict__ v, int stride, int pos0,
                                                   const int* pos0_device) {
    __shared__ float red[2][HD / 32];
    const int t = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
    const long long position = static_cast<long long>(pos0_device ? *pos0_device : pos0) + t;
    if(outside(c, position, 1)) return;
    const float kx = to_float(k[static_cast<std::size_t>(t) * stride + g * HD + d]);
    const float vx = to_float(v[static_cast<std::size_t>(t) * stride + g * HD + d]);
    const float ak = warp_max(fabsf(kx)), av = warp_max(fabsf(vx));
    if((d & 31) == 0) { red[0][d >> 5] = ak; red[1][d >> 5] = av; }
    __syncthreads();
    float mk = 0.f, mv = 0.f;
#pragma unroll
    for(int i = 0; i < HD / 32; ++i) { mk = fmaxf(mk, red[0][i]); mv = fmaxf(mv, red[1][i]); }
    const float sk = mk > 0.f ? mk / 448.f : 1.f, sv = mv > 0.f ? mv / 448.f : 1.f;
    const std::size_t pos = static_cast<std::size_t>(c.ring > 0 ? position % c.ring : position), off = (pos * c.kv_heads + g) * HD + d;
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

    __device__ __forceinline__ void fetch(const Cache& c, int g, int key0, int klo, int nkeys, int tid) {
#pragma unroll
        for(int i = 0; i < PER; ++i) {
            const int e = tid + NT * i, key = key0 + e / (HD / 16);
            if(e < CH && key >= klo && key < nkeys) {
                const std::size_t row = cache_row(c, key);
                const std::size_t off = (row * c.kv_heads + g) * HD + (e % (HD / 16)) * 16;
                kr[i] = __ldcg(reinterpret_cast<const uint4*>(c.k + off));
                vr[i] = __ldcg(reinterpret_cast<const uint4*>(c.v + off));
                ks[i] = __ldcg(c.k_scale + row * c.kv_heads + g);
                vs[i] = __ldcg(c.v_scale + row * c.kv_heads + g);
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

// One warp's 16 query rows against one staged 32-key tile: S = Q K^T, the scale and soft-cap, the rows' visibility
// mask, online softmax update of m / l and O = O * c + P V.
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

    __device__ __forceinline__ void step(const uint4* Qs, const uint4* Ks, const uint4* Vs, int qr0, int kfirst, const Visible& v0,
                                         const Visible& v1, float scale, float softcap, int lane) {
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
        if(v0.all(kfirst, kK) && v1.all(kfirst, kK)) {  // both rows see the whole tile: no per-key mask
#pragma unroll
            for(int n = 0; n < 4; ++n)
#pragma unroll
                for(int i = 0; i < 4; ++i) s[n][i] = capped(s[n][i] * scale, softcap);
        } else {
#pragma unroll
            for(int n = 0; n < 4; ++n)
#pragma unroll
                for(int i = 0; i < 4; ++i) {
                    const int key = kfirst + n * 8 + 2 * t4 + (i & 1);
                    s[n][i] = ((i >> 1) ? v1(key) : v0(key)) ? capped(s[n][i] * scale, softcap) : -INFINITY;
                }
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
        if(__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {  // some row maximum of the warp moved (x * 1 is exact)
#pragma unroll
            for(int n = 0; n < HD / 8; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
        }
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
    const int klo = window_lo(p.pos0 + qt0, p.window), jt0 = klo / kK;  // the block's oldest query sees nothing earlier
    for(int e = tid; e < kQ * C; e += kPrefillThreads) {
        const int r = e / C, c = e % C;
        uint4* dst = Qs + T_::swz(r, c);
        if(qt0 + r < p.T) cp_async16(dst, p.q + static_cast<std::size_t>(qt0 + r) * p.q_stride + static_cast<std::size_t>(h) * HD + c * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;");
    T_ tile;
    tile.fetch(p.cache, g, jt0 * kK, klo, nkeys, tid);
    tile.stage(Ks, Vs, tid);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    Rows<HD> rows;
    const int qr0 = warp * 16;
    Visible v0, v1;
    v0.pos = p.pos0 + qt0 + qr0 + gl;
    v1.pos = v0.pos + 8;
    v0.lo = window_lo(v0.pos, p.window);
    v1.lo = window_lo(v1.pos, p.window);
    const int warp_last = p.pos0 + qt0 + qr0 + 15, warp_lo = window_lo(p.pos0 + qt0 + qr0, p.window);
    for(int j = jt0; j < ntiles; ++j) {
        const bool more = j + 1 < ntiles;
        if(more) tile.fetch(p.cache, g, (j + 1) * kK, klo, nkeys, tid);
        if(j * kK <= warp_last && (j + 1) * kK > warp_lo) rows.step(Qs, Ks, Vs, qr0, j * kK, v0, v1, p.scale, p.softcap, lane);
        if(!more) break;
        __syncthreads();
        tile.stage(Ks, Vs, tid);
        __syncthreads();
    }
    rows.finish_sums();
    const float i0 = finish_row(rows.m0, rows.l0, p.sinks, h), i1 = finish_row(rows.m1, rows.l1, p.sinks, h);
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


// ------------------------------------------------------------------------------------------------ prefill, FP8 tensor cores
// prefill_fp8: S = Q K^T and O = P V on E4M3 mma.m16n8k32. Queries are quantized per row and head (scale max |q| / 448);
// key and value codes are used as stored. The per-key value scale is folded into the probabilities before they are
// rounded to E4M3: P' = p * v_scale[key] * 448 / sigma, where sigma is the largest value scale the block has seen, so
// P' <= 448 and the accumulator is kept in units of sigma / 448 (rescaled only when a row maximum or sigma grows).
// B fragments of V come from ldmatrix.trans on b16 pairs of codes, re-packed with byte permutes; the k index of the
// probability fragment is permuted to match. 64-key tiles are double buffered with cp.async (one barrier per tile)
// and only tiles reaching the diagonal are masked.
constexpr int kK8 = 64;

// E4M3 mma.sync exists from sm_89 on. Code built for an older target keeps every BF16 kernel and traps here instead;
// prefill_fp8 never launches such code (fp8_mma_ready).
__device__ __forceinline__ void mma_fp8(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    (void)c; (void)a; (void)b0; (void)b1;
    __trap();
#else
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ std::uint32_t prmt(std::uint32_t a, std::uint32_t b, std::uint32_t sel) {
    std::uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
__device__ __forceinline__ std::uint32_t fp8x4(float a, float b, float c, float d) {
    const std::uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    const std::uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ void cp_async16_zfill(void* smem, const void* gmem, bool ok) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async4_zfill(void* smem, const void* gmem, bool ok) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(s), "l"(gmem), "r"(ok ? 4 : 0));
}

// Rows of HD one-byte codes in 16-byte chunks, XOR swizzled for conflict-free ldmatrix.
template <int HD>
struct Fp8Layout {
    static constexpr int C = HD / 16;
    static constexpr int kMask = (C < 8 ? C : 8) - 1;
    __device__ __forceinline__ static int swz(int r, int c) { return r * C + (c ^ (r & kMask)); }
    static constexpr std::size_t q_bytes = static_cast<std::size_t>(kQ) * HD;
    static constexpr std::size_t tile_bytes = static_cast<std::size_t>(kK8) * HD;
    static constexpr std::size_t bytes = q_bytes + 4 * tile_bytes + 4 * (kQ + 4 * kK8);
};

template <int HD>
__device__ __forceinline__ void fetch_fp8(uint4* K, uint4* V, float* ks, float* vs, const Cache& c, int g, int key0, int klo,
                                          int nkeys, int tid) {
    using L = Fp8Layout<HD>;
#pragma unroll
    for(int i = 0; i < kK8 * L::C / kPrefillThreads; ++i) {
        const int e = tid + kPrefillThreads * i, r = e / L::C, cc = e % L::C, key = key0 + r;
        const bool ok = key >= klo && key < nkeys;
        const std::size_t off = ((ok ? cache_row(c, key) : 0) * c.kv_heads + g) * HD + cc * 16;
        cp_async16_zfill(K + L::swz(r, cc), c.k + off, ok);
        cp_async16_zfill(V + L::swz(r, cc), c.v + off, ok);
    }
    if(tid < 2 * kK8) {
        const int r = tid % kK8, key = key0 + r;
        const bool ok = key >= klo && key < nkeys;
        const std::size_t si = (ok ? cache_row(c, key) : 0) * c.kv_heads + g;
        if(tid < kK8) cp_async4_zfill(ks + r, c.k_scale + si, ok);
        else cp_async4_zfill(vs + r, c.v_scale + si, ok);
    }
    asm volatile("cp.async.commit_group;");
}

template <int HD>
__global__ void __launch_bounds__(kPrefillThreads, 1) prefill_fp8_kernel(Prefill p) {
    using L = Fp8Layout<HD>;
    extern __shared__ uint4 smem[];
    unsigned char* base = reinterpret_cast<unsigned char*>(smem);
    uint4* Qs = smem;
    // Stage b (0 or 1): keys, values and their scales.
    auto Kb = [&](int b) { return reinterpret_cast<uint4*>(base + L::q_bytes + b * L::tile_bytes); };
    auto Vb = [&](int b) { return reinterpret_cast<uint4*>(base + L::q_bytes + (2 + b) * L::tile_bytes); };
    float* qsc = reinterpret_cast<float*>(base + L::q_bytes + 4 * L::tile_bytes);
    auto ksb = [&](int b) { return qsc + kQ + b * kK8; };
    auto vsb = [&](int b) { return qsc + kQ + (2 + b) * kK8; };
    const int h = blockIdx.y, g = h / (p.heads / p.cache.kv_heads);
    const int qt0 = (gridDim.x - 1 - blockIdx.x) * kQ;  // the latest (heaviest) query tile first
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int last_q = min(p.T, qt0 + kQ) - 1;
    const int nkeys = p.pos0 + last_q + 1;
    const int ntiles = (nkeys + kK8 - 1) / kK8;
    const int klo = window_lo(p.pos0 + qt0, p.window), jt0 = klo / kK8;
    fetch_fp8<HD>(Kb(jt0 & 1), Vb(jt0 & 1), ksb(jt0 & 1), vsb(jt0 & 1), p.cache, g, jt0 * kK8, klo, nkeys, tid);
    // This warp's 16 query rows to E4M3: a lane quantizes 8 values, HD / 8 lanes per row.
    constexpr int kLanes = HD / 8, kRows = 32 / kLanes;
    const int qr0 = warp * 16;
    for(int r0 = 0; r0 < 16; r0 += kRows) {
        const int row = qr0 + r0 + lane / kLanes, seg = lane % kLanes, t = qt0 + row;
        uint4 u = make_uint4(0u, 0u, 0u, 0u);
        if(t < p.T) u = *reinterpret_cast<const uint4*>(p.q + static_cast<std::size_t>(t) * p.q_stride + static_cast<std::size_t>(h) * HD + seg * 8);
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
        float x[8];
        float am = 0.f;
#pragma unroll
        for(int i = 0; i < 4; ++i) {
            const float2 v2 = __bfloat1622float2(b[i]);
            x[2 * i] = v2.x;
            x[2 * i + 1] = v2.y;
            am = fmaxf(am, fmaxf(fabsf(v2.x), fabsf(v2.y)));
        }
#pragma unroll
        for(int o = kLanes / 2; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        const float sc = am > 0.f ? am / 448.f : 1.f;
        const std::uint32_t w0 = fp8x4(x[0] / sc, x[1] / sc, x[2] / sc, x[3] / sc);
        const std::uint32_t w1 = fp8x4(x[4] / sc, x[5] / sc, x[6] / sc, x[7] / sc);
        *(reinterpret_cast<uint2*>(Qs + L::swz(row, seg >> 1)) + (seg & 1)) = make_uint2(w0, w1);
        if(seg == 0) qsc[row] = sc;
    }
    __syncwarp();
    constexpr float kLog2e = 1.4426950408889634f;
    const int qpos0 = p.pos0 + qt0 + qr0 + gl, qpos1 = qpos0 + 8, warp_first = p.pos0 + qt0 + qr0, warp_last = warp_first + 15;
    const int lo0 = window_lo(qpos0, p.window), lo1 = window_lo(qpos1, p.window), warp_lo = window_lo(warp_first, p.window);
    const float cap2 = p.softcap * kLog2e;  // the soft-cap in the kernel's log2 score units
    const float qs0 = qsc[qr0 + gl] * p.scale * kLog2e, qs1 = qsc[qr0 + gl + 8] * p.scale * kLog2e;  // scores in log2 units
    float o[HD / 8][4];
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f, sig = 0.f;
    for(int j = jt0; j < ntiles; ++j) {
        const int buf = j & 1;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if(j + 1 < ntiles) fetch_fp8<HD>(Kb(buf ^ 1), Vb(buf ^ 1), ksb(buf ^ 1), vsb(buf ^ 1), p.cache, g, (j + 1) * kK8, klo, nkeys, tid);
        const int kfirst = j * kK8;
        if(kfirst > warp_last) continue;  // every later tile too; the barriers above still run
        if(kfirst + kK8 <= warp_lo) continue;  // before every row's window
        const uint4* K = Kb(buf);
        const uint4* V = Vb(buf);
        const float* ks = ksb(buf);
        const float* vs = vsb(buf);
        float s[8][4];
#pragma unroll
        for(int n = 0; n < 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
        for(int kc = 0; kc < HD / 32; ++kc) {
            std::uint32_t qa[4];
            ldsm_x4(qa, Qs + L::swz(qr0 + (lane & 15), kc * 2 + (lane >> 4)));
#pragma unroll
            for(int np = 0; np < 4; ++np) {
                std::uint32_t b[4];
                ldsm_x4(b, K + L::swz(np * 16 + (lane & 7) + ((lane >> 4) << 3), kc * 2 + ((lane >> 3) & 1)));
                mma_fp8(s[2 * np], qa, b[0], b[1]);
                mma_fp8(s[2 * np + 1], qa, b[2], b[3]);
            }
        }
        const bool diag = kfirst + kK8 - 1 > warp_first;
        const bool edge = p.window > 0 && kfirst < lo1;  // a tile reaching below some row's window
#pragma unroll
        for(int n = 0; n < 8; ++n) {
            const float k0 = ks[n * 8 + 2 * t4], k1 = ks[n * 8 + 2 * t4 + 1];
            s[n][0] *= qs0 * k0; s[n][1] *= qs0 * k1; s[n][2] *= qs1 * k0; s[n][3] *= qs1 * k1;
            if(cap2 > 0.f) {
#pragma unroll
                for(int i = 0; i < 4; ++i) s[n][i] = cap2 * tanhf(s[n][i] / cap2);
            }
            if(diag || edge) {
                const int key = kfirst + n * 8 + 2 * t4;
                if(key > qpos0 || key < lo0) s[n][0] = -INFINITY;
                if(key + 1 > qpos0 || key + 1 < lo0) s[n][1] = -INFINITY;
                if(key > qpos1 || key < lo1) s[n][2] = -INFINITY;
                if(key + 1 > qpos1 || key + 1 < lo1) s[n][3] = -INFINITY;
            }
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for(int n = 0; n < 8; ++n) { mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1])); mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float nsig = fmaxf(sig, warp_max(fmaxf(vs[lane], vs[lane + 32])));
        const float c0 = mx0 == m0 ? 1.f : (m0 == -INFINITY ? 0.f : ex2(m0 - mx0));
        const float c1 = mx1 == m1 ? 1.f : (m1 == -INFINITY ? 0.f : ex2(m1 - mx1));
        const float ru = nsig == sig ? 1.f : (sig > 0.f ? sig / nsig : 0.f);
        m0 = mx0; m1 = mx1; sig = nsig;
        const float f8 = sig > 0.f ? 448.f / sig : 0.f;
        float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
        for(int n = 0; n < 8; ++n) {
            const float vs0 = vs[n * 8 + 2 * t4] * f8, vs1 = vs[n * 8 + 2 * t4 + 1] * f8;
            const float e0 = mx0 == -INFINITY ? 0.f : ex2(s[n][0] - mx0), e1 = mx0 == -INFINITY ? 0.f : ex2(s[n][1] - mx0);
            const float e2 = mx1 == -INFINITY ? 0.f : ex2(s[n][2] - mx1), e3 = mx1 == -INFINITY ? 0.f : ex2(s[n][3] - mx1);
            rs0 += e0 + e1;
            rs1 += e2 + e3;
            s[n][0] = e0 * vs0; s[n][1] = e1 * vs1; s[n][2] = e2 * vs0; s[n][3] = e3 * vs1;
        }
        l0 = l0 * c0 + rs0; l1 = l1 * c1 + rs1;
        const float o0 = c0 * ru, o1 = c1 * ru;
        if(__any_sync(0xffffffffu, o0 != 1.f || o1 != 1.f)) {
#pragma unroll
            for(int n = 0; n < HD / 8; ++n) { o[n][0] *= o0; o[n][1] *= o0; o[n][2] *= o1; o[n][3] *= o1; }
        }
#pragma unroll
        for(int half = 0; half < 2; ++half) {
            // k index 4 * t4 + i of the probability fragment <-> key {2t4, 2t4 + 1, 8 + 2t4, 9 + 2t4}[i] (+16 for a2, a3).
            const int b0 = half * 4;
            const std::uint32_t af[4] = {fp8x4(s[b0][0], s[b0][1], s[b0 + 1][0], s[b0 + 1][1]),
                                         fp8x4(s[b0][2], s[b0][3], s[b0 + 1][2], s[b0 + 1][3]),
                                         fp8x4(s[b0 + 2][0], s[b0 + 2][1], s[b0 + 3][0], s[b0 + 3][1]),
                                         fp8x4(s[b0 + 2][2], s[b0 + 2][3], s[b0 + 3][2], s[b0 + 3][3])};
            const int key = half * 32 + (lane >> 3) * 8 + (lane & 7);
#pragma unroll
            for(int dp = 0; dp < HD / 16; ++dp) {
                // Transposed b16 tiles of keys 0-7 / 8-15 / 16-23 / 24-31 over dims 16dp..16dp+15: each register holds
                // (key 2t4: dims 2gl, 2gl+1), (key 2t4+1: dims 2gl, 2gl+1); even dims go to tile 2dp, odd to 2dp+1.
                std::uint32_t r[4];
                ldsm_x4_t(r, V + L::swz(key, dp));
                mma_fp8(o[2 * dp], af, prmt(r[0], r[1], 0x6420), prmt(r[2], r[3], 0x6420));
                mma_fp8(o[2 * dp + 1], af, prmt(r[0], r[1], 0x7531), prmt(r[2], r[3], 0x7531));
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float u = sig / 448.f;
    float j0 = l0 > 0.f ? u / l0 : 0.f, j1 = l1 > 0.f ? u / l1 : 0.f;
    if(p.sinks) {  // the sink's weight joins each row's denominator, in the kernel's log2 units
        const float sink = p.sinks[h] * kLog2e, M0 = fmaxf(m0, sink), M1 = fmaxf(m1, sink);
        const float c0 = m0 == -INFINITY ? 0.f : ex2(m0 - M0), c1 = m1 == -INFINITY ? 0.f : ex2(m1 - M1);
        l0 = l0 * c0 + ex2(sink - M0);
        l1 = l1 * c1 + ex2(sink - M1);
        j0 = u * c0 / l0;
        j1 = u * c1 / l1;
    }
    const int t0r = qt0 + qr0 + gl, t1r = t0r + 8;
#pragma unroll
    for(int dp = 0; dp < HD / 16; ++dp) {
        const int d = dp * 16 + 4 * t4;
        if(t0r < p.T)
            *reinterpret_cast<float4*>(p.out + static_cast<std::size_t>(t0r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float4(o[2 * dp][0] * j0, o[2 * dp + 1][0] * j0, o[2 * dp][1] * j0, o[2 * dp + 1][1] * j0);
        if(t1r < p.T)
            *reinterpret_cast<float4*>(p.out + static_cast<std::size_t>(t1r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float4(o[2 * dp][2] * j1, o[2 * dp + 1][2] * j1, o[2 * dp][3] * j1, o[2 * dp + 1][3] * j1);
    }
}

// ------------------------------------------------------------------------------------------------ decode
// A decode block runs S key streams (1 or 2) of four warps each over the same 64 rows: stream s takes tiles s, s + S,
// s + 2S, ... of the block's pieces with its own K / V stage and online-softmax state, and synchronizes only within
// itself, so the streams' phases drift apart and one stream's dequantization and softmax overlap the other's tensor-core
// work (decode rows are few, so one stream leaves the tensor cores idle while tiles load). At the end a block of two
// streams merges stream 0 then stream 1 into its one partial result. With S = 2, warp w runs row group w / 2 of stream
// ((w >> 2) ^ w) & 1: every SM sub-partition (warp w % 4) holds one warp of each stream, and the two warps of a
// one-column decode (row group 0) sit on different sub-partitions. Warps whose 16 rows are all unused skip the tensor
// cores. Tile-to-stream assignment and merge order depend only on key positions, so a column's bits still depend only on
// its own position (and on S, which the launcher fixes per device and head dimension).
constexpr int kStreamThreads = 128;
// A decode row no column uses sees every key, so it never forces the per-key mask on its warp; its state (zero query)
// stays finite and is never written out.
constexpr int kUnusedRowPos = INT_MAX / 2;

template <int S>
__device__ __forceinline__ void stream_sync(int stream) {
    if constexpr(S == 1) {
        __syncthreads();
    } else {
        if(stream) asm volatile("bar.sync 2, %0;" :: "n"(kStreamThreads) : "memory");
        else asm volatile("bar.sync 1, %0;" :: "n"(kStreamThreads) : "memory");
    }
}

template <int HD, int S>
__global__ void __launch_bounds__(S * kStreamThreads, 1) decode_kernel(Decode p) {
    using T_ = Tile<HD, kStreamThreads>;
    constexpr int C = HD / 8;
    extern __shared__ uint4 smem[];
    uint4* Qs = smem;
    const int g = blockIdx.x, blk = blockIdx.y, group = p.heads / p.cache.kv_heads;
    const int pos0 = p.pos0_device ? *p.pos0_device : p.pos0;
    if(outside(p.cache, pos0, p.T)) return;  // nothing read; the combine answers NaN
    const int nkeys = pos0 + p.T;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int stream = S == 2 ? ((warp >> 2) ^ warp) & 1 : 0, rg = S == 2 ? warp >> 1 : warp;  // row group 0..3
    const int stid = rg * 32 + lane;  // thread index within its stream
    uint4* Ks = Qs + kDecodeRows * C + stream * 2 * kK * C;
    uint4* Vs = Ks + kK * C;
    float* part = p.partial + (static_cast<std::size_t>(g) * p.blocks + blk) * kDecodeRows * HD;
    float* ml = p.partial + static_cast<std::size_t>(p.cache.kv_heads) * p.blocks * kDecodeRows * HD +
                (static_cast<std::size_t>(g) * p.blocks + blk) * kDecodeRows * 2;
    // Pieces stay assigned by position (piece P to block P mod blocks); with a window the walk starts at the oldest
    // column's window, so the work follows the window, not the context.
    const int klo = window_lo(pos0, p.window), first = klo / kPiece;
    const int piece0 = first + ((blk - first % p.blocks) % p.blocks + p.blocks) % p.blocks;
    auto tile_key = [&](int j) { return (piece0 + (j >> 2) * p.blocks) * kPiece + (j & 3) * kK; };
    if(tile_key(0) >= nkeys) {
        for(int r = tid; r < kDecodeRows; r += S * kStreamThreads) { ml[r * 2] = -INFINITY; ml[r * 2 + 1] = 0.f; }
        return;
    }
    const int used = p.T * group;
    for(int e = tid; e < kDecodeRows * C; e += S * kStreamThreads) {
        const int r = e / C, c = e % C;
        uint4* dst = Qs + T_::swz(r, c);
        if(r < used) cp_async16(dst, p.q + static_cast<std::size_t>(r / group) * p.q_stride + static_cast<std::size_t>(g * group + r % group) * HD + c * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;");
    T_ tile;
    tile.fetch(p.cache, g, tile_key(stream), klo, nkeys, stid);
    tile.stage(Ks, Vs, stid);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    Rows<HD> rows;
    const int qr0 = rg * 16, r0 = qr0 + gl, r1 = r0 + 8;
    const bool computes = qr0 < used;
    auto visible = [&](int r) {
        Visible v;
        if(r >= used) { v.pos = kUnusedRowPos; return v; }
        const int column = r / group;
        v.pos = pos0 + column;
        int depth = column;
        if(p.tree) {
            v.tree = p.tree[column];
            v.base = pos0;
            depth = __popcll(v.tree) - 1;
        }
        v.lo = window_lo(pos0 + depth, p.window);
        return v;
    };
    const Visible v0 = visible(r0), v1 = visible(r1);
    for(int j = stream; tile_key(j) < nkeys; j += S) {
        const bool more = tile_key(j + S) < nkeys;
        if(more) tile.fetch(p.cache, g, tile_key(j + S), klo, nkeys, stid);
        if(computes) rows.step(Qs, Ks, Vs, qr0, tile_key(j), v0, v1, p.scale, p.softcap, lane);
        if(!more) break;
        stream_sync<S>(stream);
        tile.stage(Ks, Vs, stid);
        stream_sync<S>(stream);
    }
    rows.finish_sums();
    float a0 = 1.f, a1 = 1.f, b0 = 0.f, b1 = 0.f, m0 = rows.m0, m1 = rows.m1, l0 = rows.l0, l1 = rows.l1;
    const float4* other = reinterpret_cast<const float4*>(Qs + kDecodeRows * C);  // stream 1's accumulators
    if constexpr(S == 2) {
        // Stream 1 parks its accumulators ([row group][HD / 8][lane]) in the K / V stages and its (m, l) in the query
        // stage; the stream-0 warp of the same row group (same fragment layout) merges: m = max, l = l_0 a + l_1 b,
        // o = o_0 a + o_1 b with a = exp(m_0 - m), b = exp(m_1 - m).
        float4* park = reinterpret_cast<float4*>(Qs + kDecodeRows * C);
        float4* mlb = reinterpret_cast<float4*>(Qs);
        __syncthreads();  // every warp is done with the stages
        if(stream == 1 && computes) {
#pragma unroll
            for(int n = 0; n < HD / 8; ++n) park[(rg * (HD / 8) + n) * 32 + lane] = make_float4(rows.o[n][0], rows.o[n][1], rows.o[n][2], rows.o[n][3]);
            mlb[rg * 32 + lane] = make_float4(rows.m0, rows.l0, rows.m1, rows.l1);
        }
        __syncthreads();
        if(stream == 1 || !computes) return;
        const float4 x = mlb[rg * 32 + lane];
        m0 = fmaxf(rows.m0, x.x);
        m1 = fmaxf(rows.m1, x.z);
        a0 = rows.m0 == -INFINITY ? 0.f : __expf(rows.m0 - m0);
        b0 = x.x == -INFINITY ? 0.f : __expf(x.x - m0);
        a1 = rows.m1 == -INFINITY ? 0.f : __expf(rows.m1 - m1);
        b1 = x.z == -INFINITY ? 0.f : __expf(x.z - m1);
        l0 = rows.l0 * a0 + x.y * b0;
        l1 = rows.l1 * a1 + x.w * b1;
    }
    if(!computes) return;  // no used row
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for(int n = 0; n < HD / 8; ++n) {
        const int dd = n * 8 + 2 * t4;
        float4 o = make_float4(rows.o[n][0], rows.o[n][1], rows.o[n][2], rows.o[n][3]);
        if constexpr(S == 2) {
            const float4 x = other[(rg * (HD / 8) + n) * 32 + lane];
            o = make_float4(o.x * a0 + x.x * b0, o.y * a0 + x.y * b0, o.z * a1 + x.z * b1, o.w * a1 + x.w * b1);
        }
        if(r0 < used) *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r0) * HD + dd) = make_float2(o.x * i0, o.y * i0);
        if(r1 < used) *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r1) * HD + dd) = make_float2(o.z * i1, o.w * i1);
    }
    if(t4 == 0) {
        if(r0 < used) { ml[r0 * 2] = m0; ml[r0 * 2 + 1] = l0; }
        if(r1 < used) { ml[r1 * 2] = m1; ml[r1 * 2 + 1] = l1; }
    }
}

// Block (column, query head): one thread per dimension plus one warp for the denominator. Block weights
// w_b = l_b exp(m_b - M); M = max m_b is exact in any order, so the weights are computed in parallel. The denominator
// (by the extra warp, off the numerator's path) and each dimension's numerator are summed in block order exactly as
// one thread's loop would, with the partial outputs loaded kCombineBatch at a time.
constexpr int kCombineBatch = 16;

template <int HD>
__global__ void __launch_bounds__(HD + 32) combine_kernel(Decode p) {
    constexpr int NT = HD + 32, kPer = (1024 + NT - 1) / NT;
    extern __shared__ float wsp[];  // [blocks] weights
    __shared__ float total, red[NT / 32];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, group = p.heads / p.cache.kv_heads, g = h / group;
    const int r = t * group + h % group;
    const int pos0 = p.pos0_device ? *p.pos0_device : p.pos0;
    float* out = p.out + static_cast<std::size_t>(t) * p.out_stride + static_cast<std::size_t>(h) * HD + d;
    if(outside(p.cache, pos0, p.T)) {
        if(d < HD) *out = __int_as_float(0x7fffffff);  // NaN: the device position does not fit the cache
        return;
    }
    const int ns = min(p.blocks, (pos0 + p.T + kPiece - 1) / kPiece);  // blocks that hold at least one key
    const float* part = p.partial + static_cast<std::size_t>(g) * p.blocks * kDecodeRows * HD;
    const float* ml = p.partial + static_cast<std::size_t>(p.cache.kv_heads) * p.blocks * kDecodeRows * HD +
                      static_cast<std::size_t>(g) * p.blocks * kDecodeRows * 2;
    float m[kPer], l[kPer];
    float mx = -INFINITY;
#pragma unroll
    for(int i = 0; i < kPer; ++i) {
        const int b = d + i * NT;
        m[i] = b < ns ? ml[(static_cast<std::size_t>(b) * kDecodeRows + r) * 2] : -INFINITY;
        l[i] = b < ns ? ml[(static_cast<std::size_t>(b) * kDecodeRows + r) * 2 + 1] : 0.f;
        mx = fmaxf(mx, m[i]);
    }
    mx = warp_max(mx);
    if((d & 31) == 0) red[d >> 5] = mx;
    __syncthreads();
    float M = p.sinks ? p.sinks[h] : -INFINITY;
#pragma unroll
    for(int i = 0; i < NT / 32; ++i) M = fmaxf(M, red[i]);
#pragma unroll
    for(int i = 0; i < kPer; ++i) {
        const int b = d + i * NT;
        if(b < ns) wsp[b] = m[i] == -INFINITY ? 0.f : l[i] * __expf(m[i] - M);
    }
    __syncthreads();
    if(d == HD) {  // the extra warp: the denominator
        float den = 0.f;
        for(int b = 0; b < ns; ++b) den += wsp[b];
        if(p.sinks) den += __expf(p.sinks[h] - M);  // the sink's weight joins the denominator after the blocks'
        total = den;
    }
    float num = 0.f;
    if(d < HD)
        for(int b0 = 0; b0 < ns; b0 += kCombineBatch) {
            float x[kCombineBatch];
#pragma unroll
            for(int u = 0; u < kCombineBatch; ++u) {
                const int b = b0 + u;
                x[u] = b < ns && wsp[b] != 0.f ? __ldcg(part + (static_cast<std::size_t>(b) * kDecodeRows + r) * HD + d) : 0.f;
            }
#pragma unroll
            for(int u = 0; u < kCombineBatch; ++u) {
                const int b = b0 + u;
                if(b < ns && wsp[b] != 0.f) num = fmaf(wsp[b], x[u], num);
            }
        }
    __syncthreads();
    if(d < HD) *out = total > 0.f ? num / total : 0.f;
}

bool valid(const Cache& c) {
    return c.k && c.v && c.k_scale && c.v_scale && c.kv_heads > 0 && (c.head_dim == 64 || c.head_dim == 128 || c.head_dim == 256) &&
           c.ring >= 0 && c.capacity >= 0 && (c.ring == 0 || c.capacity == 0 || c.ring <= c.capacity);
}
// A ring holds a window layer when every key a launch reads survives the launch's own writes.
bool valid_window(const Cache& c, int window, float softcap, int rows) {
    return window >= 0 && softcap >= 0.f && (c.ring == 0 || (window > 0 && c.ring >= window + rows - 1));
}

// pos0_device set: the kernel reads the position and skips rows outside the cache; else pos0 .. pos0 + T - 1 must fit.
template <class In>
cudaError_t run_write(const Cache& c, const In* k, const In* v, int stride, int T, int pos0, const int* pos0_device, cudaStream_t s) {
    if(!valid(c) || !k || !v || T <= 0 || (!pos0_device && outside(c, pos0, T)) || stride < c.kv_heads * c.head_dim)
        return cudaErrorInvalidValue;
    const dim3 grid(T, c.kv_heads);
    switch(c.head_dim) {
    case 64: write_kernel<64, In><<<grid, 64, 0, s>>>(c, k, v, stride, pos0, pos0_device); break;
    case 128: write_kernel<128, In><<<grid, 128, 0, s>>>(c, k, v, stride, pos0, pos0_device); break;
    default: write_kernel<256, In><<<grid, 256, 0, s>>>(c, k, v, stride, pos0, pos0_device); break;
    }
    return cudaGetLastError();
}

template <int HD>
cudaError_t run_prefill(const Prefill& p, cudaStream_t s) {
    const std::size_t bytes = prefill_shared_bytes(HD);
    const cudaError_t attr = cudaFuncSetAttribute(prefill_kernel<HD>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if(attr != cudaSuccess) return attr;
    prefill_kernel<HD><<<dim3((p.T + kQ - 1) / kQ, p.heads), kPrefillThreads, bytes, s>>>(p);
    return cudaGetLastError();
}

// The code the runtime picked for this device was compiled for sm_89 or later (both its PTX target and its binary), so it
// holds the E4M3 mma rather than the trap: a build for older targets, or such a build JIT- or binary-loaded on a newer
// device, is refused.
template <int HD>
cudaError_t fp8_mma_ready() {
    cudaFuncAttributes attributes{};
    const cudaError_t error = cudaFuncGetAttributes(&attributes, prefill_fp8_kernel<HD>);
    if(error != cudaSuccess) return error;
    return attributes.ptxVersion >= 89 && attributes.binaryVersion >= 89 ? cudaSuccess : cudaErrorNotSupported;
}

template <int HD>
cudaError_t run_prefill_fp8(const Prefill& p, cudaStream_t s) {
    const cudaError_t ready = fp8_mma_ready<HD>();
    if(ready != cudaSuccess) return ready;
    const std::size_t bytes = Fp8Layout<HD>::bytes;
    const cudaError_t attr = cudaFuncSetAttribute(prefill_fp8_kernel<HD>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if(attr != cudaSuccess) return attr;
    prefill_fp8_kernel<HD><<<dim3((p.T + kQ - 1) / kQ, p.heads), kPrefillThreads, bytes, s>>>(p);
    return cudaGetLastError();
}

template <int HD, int S>
cudaError_t run_decode(const Decode& p, cudaStream_t s) {
    const std::size_t bytes = decode_shared_bytes(HD, S);
    const cudaError_t attr = cudaFuncSetAttribute(decode_kernel<HD, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if(attr != cudaSuccess) return attr;
    decode_kernel<HD, S><<<dim3(p.cache.kv_heads, p.blocks), S * kStreamThreads, bytes, s>>>(p);
    combine_kernel<HD><<<dim3(p.T, p.heads), HD + 32, sizeof(float) * p.blocks, s>>>(p);
    return cudaGetLastError();
}
// Two key streams where their shared memory fits the device's per-block limit (it is checked, not assumed), else one.
template <int HD>
cudaError_t run_decode(const Decode& p, cudaStream_t s) {
    int streams = p.streams;
    if(streams == 0) {
        int dev = 0, limit = 0;
        cudaError_t e = cudaGetDevice(&dev);
        if(e == cudaSuccess) e = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if(e != cudaSuccess) return e;
        streams = static_cast<std::size_t>(limit) >= decode_shared_bytes(HD, 2) ? 2 : 1;
    }
    return streams == 2 ? run_decode<HD, 2>(p, s) : run_decode<HD, 1>(p, s);
}

}  // namespace

cudaError_t write(const Cache& c, const __nv_bfloat16* k, const __nv_bfloat16* v, int stride, int T, int pos0, cudaStream_t s) {
    return run_write(c, k, v, stride, T, pos0, nullptr, s);
}
cudaError_t write(const Cache& c, const __nv_bfloat16* k, const __nv_bfloat16* v, int stride, int T, const int* pos0_device, cudaStream_t s) {
    return pos0_device ? run_write(c, k, v, stride, T, 0, pos0_device, s) : cudaErrorInvalidValue;
}
cudaError_t write(const Cache& c, const float* k, const float* v, int stride, int T, int pos0, cudaStream_t s) {
    return run_write(c, k, v, stride, T, pos0, nullptr, s);
}
cudaError_t write(const Cache& c, const float* k, const float* v, int stride, int T, const int* pos0_device, cudaStream_t s) {
    return pos0_device ? run_write(c, k, v, stride, T, 0, pos0_device, s) : cudaErrorInvalidValue;
}

std::size_t prefill_shared_bytes(int head_dim) { return static_cast<std::size_t>(kQ + 2 * kK) * head_dim * 2; }

cudaError_t prefill(const Prefill& p, cudaStream_t s) {
    if(!valid(p.cache) || !p.q || !p.out || p.T <= 0 || outside(p.cache, p.pos0, p.T) || p.heads <= 0 || p.heads % p.cache.kv_heads ||
       p.q_stride % 8 || p.q_stride < p.heads * p.cache.head_dim || p.out_stride % 2 || p.out_stride < p.heads * p.cache.head_dim ||
       !valid_window(p.cache, p.window, p.softcap, p.T))
        return cudaErrorInvalidValue;
    switch(p.cache.head_dim) {
    case 64: return run_prefill<64>(p, s);
    case 128: return run_prefill<128>(p, s);
    default: return run_prefill<256>(p, s);
    }
}

std::size_t prefill_fp8_shared_bytes(int head_dim) {
    switch(head_dim) {
    case 64: return Fp8Layout<64>::bytes;
    case 128: return Fp8Layout<128>::bytes;
    default: return Fp8Layout<256>::bytes;
    }
}

cudaError_t prefill_fp8(const Prefill& p, cudaStream_t s) {
    if(!valid(p.cache) || !p.q || !p.out || p.T <= 0 || outside(p.cache, p.pos0, p.T) || p.heads <= 0 || p.heads % p.cache.kv_heads ||
       p.q_stride % 8 || p.q_stride < p.heads * p.cache.head_dim || p.out_stride % 4 || p.out_stride < p.heads * p.cache.head_dim ||
       !valid_window(p.cache, p.window, p.softcap, p.T))
        return cudaErrorInvalidValue;
    switch(p.cache.head_dim) {
    case 64: return run_prefill_fp8<64>(p, s);
    case 128: return run_prefill_fp8<128>(p, s);
    default: return run_prefill_fp8<256>(p, s);
    }
}

std::size_t decode_shared_bytes(int head_dim, int streams) { return static_cast<std::size_t>(kDecodeRows + 2 * streams * kK) * head_dim * 2; }

std::size_t decode_partial_floats(int kv_heads, int head_dim, int blocks) {
    return static_cast<std::size_t>(kv_heads) * blocks * kDecodeRows * (head_dim + 2);
}

cudaError_t decode(const Decode& p, cudaStream_t s) {
    if(!valid(p.cache) || !p.q || !p.out || !p.partial || p.T <= 0 || p.heads <= 0 || p.heads % p.cache.kv_heads ||
       p.T * (p.heads / p.cache.kv_heads) > kDecodeRows || p.blocks < 1 || p.blocks > 1024 || (!p.pos0_device && outside(p.cache, p.pos0, p.T)) ||
       p.q_stride % 8 || p.q_stride < p.heads * p.cache.head_dim || p.out_stride < p.heads * p.cache.head_dim ||
       !valid_window(p.cache, p.window, p.softcap, p.T) || (p.tree && p.window > 0 && p.window <= 64) || p.streams < 0 || p.streams > 2)
        return cudaErrorInvalidValue;
    switch(p.cache.head_dim) {
    case 64: return run_decode<64>(p, s);
    case 128: return run_decode<128>(p, s);
    default: return run_decode<256>(p, s);
    }
}

}  // namespace crit::fp8_kv
