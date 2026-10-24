// Batched paged attention; see paged_attention.h.
//
// Both attention kernels walk a sequence's keys in 32-key tiles. A tile's rows are gathered through the block table
// (physical row = table[key / page_tokens] * page_tokens + key % page_tokens), so a tile may span pages and pages of
// any size work. The next tile is fetched into registers while the current one is used; between tiles it is stored
// (BF16 pages) or dequantized (E4M3 pages: code * scale rounded to BF16) into one shared K / V stage with a 16-byte
// XOR swizzle (chunk c of row r at c ^ (r & 7)), so ldmatrix reads are conflict free. S = Q K^T and O = P V run on
// BF16 mma.m16n8k16 with an FP32 online softmax; probabilities are rounded to BF16 and the row sums use the rounded
// values. A tile entirely outside a row's visible keys (after its position, before its window, or off its tree path)
// leaves that row's maximum, sum and output bit for bit unchanged (exp(0) = 1 and p = 0), which decode's column
// invariance relies on: skipping such tiles, or fetching keys only from the oldest row's window on, changes no bits.
#include "paged_attention.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <limits>

#include <climits>
#include <cstdint>

namespace crit::paged_attention {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr int kK = 32;          // keys per tile
constexpr int kPiece = 4 * kK;  // keys per decode piece
constexpr int kRows = 64;       // decode rows per block: columns x query heads of one key/value head
constexpr int kQ = 128;         // prefill queries per block
constexpr int kThreads = 256;   // both kernels: every thread loads, warps owning rows run the tensor cores

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
__device__ __forceinline__ std::size_t physical(const std::int32_t* table, int page_tokens, int key) {
    return static_cast<std::size_t>(__ldg(table + key / page_tokens)) * page_tokens + key % page_tokens;
}
__device__ __forceinline__ int swizzle(int row, int chunk, int chunks) { return row * chunks + (chunk ^ (row & 7)); }

// Sequence s with n queries breaks the batch's counts (Batch): more queries than the launch's bound `limit` or than
// its keys, or more keys than its table row addresses.
__device__ __forceinline__ bool broken(const Batch& b, int s, int n, int limit, int page_tokens) {
    if(n == 0) return false;
    const int kv = b.kv_lens[s];
    return n < 0 || n > limit || n > kv || static_cast<long long>(kv) > static_cast<long long>(b.table_stride) * page_tokens;
}
// A broken sequence's answer: NaN in every dimension of head h of its query rows first, first + step, ...; one caller
// per sequence (lead) raises the fault flag.
__device__ __forceinline__ void poison(const Batch& b, float* out, int out_stride, int q0, int n, int first, int step, int h, int HD,
                                       bool lead) {
    for(int r = first; r < n; r += step)
        for(int d = threadIdx.x; d < HD; d += blockDim.x)
            out[static_cast<std::size_t>(q0 + r) * out_stride + static_cast<std::size_t>(h) * HD + d] = kNaN;
    if(lead && threadIdx.x == 0 && b.fault) *b.fault = 1;
}

// The keys one query row sees: key <= pos (pos < 0: an unused row), key >= lo (sliding window), and keys from `base`
// on only where the row's tree bit (key - base) is set.
struct Visible {
    int pos = -1, lo = 0, base = INT_MAX;
    std::uint64_t tree = ~0ull;
    __device__ __forceinline__ bool operator()(int key) const {
        return key <= pos && key >= lo && (key < base || key - base >= 64 || ((tree >> (key - base)) & 1ull));
    }
};
__device__ __forceinline__ int window_lo(int pos, int window) { return window > 0 ? max(0, pos - window + 1) : 0; }
__device__ __forceinline__ float capped(float x, float cap) { return cap > 0.f ? cap * tanhf(x / cap) : x; }
// Folds a sink logit into a finished row: its weight exp(sink) joins the denominator. Returns the factor that turns
// the row's unnormalized output into its result (0 for an empty row without a sink).
__device__ __forceinline__ float finish_row(float m, float& l, const float* sinks, int h) {
    if(!sinks) return l > 0.f ? 1.f / l : 0.f;
    const float sink = sinks[h], M = fmaxf(m, sink);
    const float c = m == -kInf ? 0.f : __expf(m - M);
    l = l * c + __expf(sink - M);
    return c / l;
}

// ------------------------------------------------------------------------------------------------ write
template <int HD, Format F>
__global__ void __launch_bounds__(HD) write_kernel(Pages pg, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                                                   int stride, const std::int32_t* __restrict__ slots) {
    const int t = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
    const int slot = slots[t];
    if(slot < 0) return;
    const __nv_bfloat16 kb = k[static_cast<std::size_t>(t) * stride + g * HD + d];
    const __nv_bfloat16 vb = v[static_cast<std::size_t>(t) * stride + g * HD + d];
    const std::size_t row = static_cast<std::size_t>(slot) * pg.kv_heads + g, off = row * HD + d;
    if constexpr(F == Format::bf16) {
        static_cast<__nv_bfloat16*>(pg.k)[off] = kb;
        static_cast<__nv_bfloat16*>(pg.v)[off] = vb;
    } else {
        __shared__ float red[2][HD / 32];
        const float kx = __bfloat162float(kb), vx = __bfloat162float(vb);
        const float ak = warp_max(fabsf(kx)), av = warp_max(fabsf(vx));
        if((d & 31) == 0) { red[0][d >> 5] = ak; red[1][d >> 5] = av; }
        __syncthreads();
        float mk = 0.f, mv = 0.f;
#pragma unroll
        for(int i = 0; i < HD / 32; ++i) { mk = fmaxf(mk, red[0][i]); mv = fmaxf(mv, red[1][i]); }
        const float sk = mk > 0.f ? mk / 448.f : 1.f, sv = mv > 0.f ? mv / 448.f : 1.f;
        static_cast<std::uint8_t*>(pg.k)[off] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(kx / sk, __NV_SATFINITE, __NV_E4M3));
        static_cast<std::uint8_t*>(pg.v)[off] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(vx / sv, __NV_SATFINITE, __NV_E4M3));
        if(d == 0) { pg.k_scale[row] = sk; pg.v_scale[row] = sv; }
    }
}

// ------------------------------------------------------------------------------------------------ tiles
// One 32-key tile of one key/value head gathered through a block table; NT threads own its 16-byte chunks.
template <int HD, Format F, int NT>
struct Tile;

template <int HD, int NT>
struct Tile<HD, Format::bf16, NT> {
    static constexpr int C = HD / 8;  // 16-byte chunks per row, in pages and in shared memory
    static constexpr int PER = (kK * C + NT - 1) / NT;
    uint4 kr[PER], vr[PER];

    __device__ __forceinline__ void fetch(const Pages& pg, const std::int32_t* table, int g, int key0, int klo, int nkeys, int tid) {
#pragma unroll
        for(int i = 0; i < PER; ++i) {
            const int e = tid + NT * i, key = key0 + e / C;
            if(e < kK * C && key >= klo && key < nkeys) {
                const std::size_t off = (physical(table, pg.page_tokens, key) * pg.kv_heads + g) * HD + (e % C) * 8;
                kr[i] = __ldcg(reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(pg.k) + off));
                vr[i] = __ldcg(reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(pg.v) + off));
            } else {
                kr[i] = vr[i] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }
    __device__ __forceinline__ void stage(uint4* Ks, uint4* Vs, int tid) const {
#pragma unroll
        for(int i = 0; i < PER; ++i) {
            const int e = tid + NT * i;
            if(e >= kK * C) break;
            Ks[swizzle(e / C, e % C, C)] = kr[i];
            Vs[swizzle(e / C, e % C, C)] = vr[i];
        }
    }
};

template <int HD, int NT>
struct Tile<HD, Format::e4m3, NT> {
    static constexpr int C = HD / 8;
    static constexpr int CC = HD / 16;  // 16-byte code chunks per row
    static constexpr int PER = (kK * CC + NT - 1) / NT;
    uint4 kr[PER], vr[PER];
    float ks[PER], vs[PER];

    __device__ __forceinline__ void fetch(const Pages& pg, const std::int32_t* table, int g, int key0, int klo, int nkeys, int tid) {
#pragma unroll
        for(int i = 0; i < PER; ++i) {
            const int e = tid + NT * i, key = key0 + e / CC;
            if(e < kK * CC && key >= klo && key < nkeys) {
                const std::size_t row = physical(table, pg.page_tokens, key) * pg.kv_heads + g, off = row * HD + (e % CC) * 16;
                kr[i] = __ldcg(reinterpret_cast<const uint4*>(static_cast<const std::uint8_t*>(pg.k) + off));
                vr[i] = __ldcg(reinterpret_cast<const uint4*>(static_cast<const std::uint8_t*>(pg.v) + off));
                ks[i] = __ldcg(pg.k_scale + row);
                vs[i] = __ldcg(pg.v_scale + row);
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
            if(e >= kK * CC) break;
            const int r = e / CC, c = e % CC;
            Ks[swizzle(r, 2 * c, C)] = dequant8(kr[i].x, kr[i].y, ks[i]);
            Ks[swizzle(r, 2 * c + 1, C)] = dequant8(kr[i].z, kr[i].w, ks[i]);
            Vs[swizzle(r, 2 * c, C)] = dequant8(vr[i].x, vr[i].y, vs[i]);
            Vs[swizzle(r, 2 * c + 1, C)] = dequant8(vr[i].z, vr[i].w, vs[i]);
        }
    }
};

// One warp's 16 query rows against one staged 32-key tile: S = Q K^T, the scale and soft-cap, the rows' visibility
// mask, online softmax update of m / l and O = O * c + P V.
template <int HD>
struct Rows {
    static constexpr int C = HD / 8;
    float o[HD / 8][4];
    float m0 = -kInf, m1 = -kInf, l0 = 0.f, l1 = 0.f;

    __device__ __forceinline__ Rows() {
#pragma unroll
        for(int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    }

    __device__ __forceinline__ void step(const uint4* Qs, const uint4* Ks, const uint4* Vs, int qr0, int kfirst, const Visible& v0,
                                         const Visible& v1, float scale, float softcap, int lane) {
        const int t4 = lane & 3;
        float s[4][4];
#pragma unroll
        for(int n = 0; n < 4; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
        for(int ks = 0; ks < HD / 16; ++ks) {
            std::uint32_t a[4];
            ldsm_x4(a, Qs + swizzle(qr0 + (lane & 15), ks * 2 + (lane >> 4), C));
#pragma unroll
            for(int np = 0; np < 2; ++np) {
                std::uint32_t b[4];
                const int key = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(b, Ks + swizzle(key, ks * 2 + ((lane >> 3) & 1), C));
                mma_bf16(s[2 * np], a[0], a[1], a[2], a[3], b[0], b[1]);
                mma_bf16(s[2 * np + 1], a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }
#pragma unroll
        for(int n = 0; n < 4; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) {
                const int key = kfirst + n * 8 + 2 * t4 + (i & 1);
                s[n][i] = ((i >> 1) ? v1(key) : v0(key)) ? capped(s[n][i] * scale, softcap) : -kInf;
            }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for(int n = 0; n < 4; ++n) { mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1])); mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float c0 = mx0 == -kInf ? 1.f : __expf(m0 - mx0), c1 = mx1 == -kInf ? 1.f : __expf(m1 - mx1);
        m0 = mx0; m1 = mx1;
        float rs0 = 0.f, rs1 = 0.f;
        std::uint32_t pf[4][2];
#pragma unroll
        for(int n = 0; n < 4; ++n) {
            const float e0 = mx0 == -kInf ? 0.f : __expf(s[n][0] - mx0), e1 = mx0 == -kInf ? 0.f : __expf(s[n][1] - mx0);
            const float e2 = mx1 == -kInf ? 0.f : __expf(s[n][2] - mx1), e3 = mx1 == -kInf ? 0.f : __expf(s[n][3] - mx1);
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
                ldsm_x4_t(b, Vs + swizzle(key, dp * 2 + (lane >> 4), C));
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
template <int HD, Format F>
__global__ void __launch_bounds__(kThreads, 1) prefill_kernel(Prefill p) {
    constexpr int C = HD / 8;
    extern __shared__ uint4 smem[];
    uint4* Qs = smem;
    uint4* Ks = Qs + kQ * C;
    uint4* Vs = Ks + kK * C;
    const int h = blockIdx.y, s = blockIdx.z, g = h / (p.heads / p.pages.kv_heads);
    const int q0 = p.batch.q_start[s], n = p.batch.q_start[s + 1] - q0, qtiles = (n + kQ - 1) / kQ;
    if(broken(p.batch, s, n, p.max_queries, p.pages.page_tokens)) {
        poison(p.batch, p.out, p.out_stride, q0, n, blockIdx.x, gridDim.x, h, HD, blockIdx.x == 0 && h == 0);
        return;
    }
    if(static_cast<int>(blockIdx.x) >= qtiles) return;
    const int qt0 = (qtiles - 1 - static_cast<int>(blockIdx.x)) * kQ;  // a sequence's latest (heaviest) query tile first
    const int pos0 = p.batch.kv_lens[s] - n;
    const std::int32_t* table = p.batch.tables + static_cast<std::size_t>(s) * p.batch.table_stride;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int last_q = min(n, qt0 + kQ) - 1;
    const int nkeys = pos0 + last_q + 1;
    const int ntiles = (nkeys + kK - 1) / kK;
    const int klo = window_lo(pos0 + qt0, p.window), jt0 = klo / kK;  // the block's oldest query sees nothing earlier
    for(int e = tid; e < kQ * C; e += kThreads) {
        const int r = e / C, c = e % C;
        uint4* dst = Qs + swizzle(r, c, C);
        if(qt0 + r < n) cp_async16(dst, p.q + static_cast<std::size_t>(q0 + qt0 + r) * p.q_stride + static_cast<std::size_t>(h) * HD + c * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;");
    Tile<HD, F, kThreads> tile;
    tile.fetch(p.pages, table, g, jt0 * kK, klo, nkeys, tid);
    tile.stage(Ks, Vs, tid);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    Rows<HD> rows;
    const int qr0 = warp * 16;
    Visible v0, v1;
    v0.pos = pos0 + qt0 + qr0 + gl;
    v1.pos = v0.pos + 8;
    v0.lo = window_lo(v0.pos, p.window);
    v1.lo = window_lo(v1.pos, p.window);
    const int warp_last = pos0 + qt0 + qr0 + 15, warp_lo = window_lo(pos0 + qt0 + qr0, p.window);
    for(int j = jt0; j < ntiles; ++j) {
        const bool more = j + 1 < ntiles;
        if(more) tile.fetch(p.pages, table, g, (j + 1) * kK, klo, nkeys, tid);
        if(j * kK <= warp_last && (j + 1) * kK > warp_lo && qt0 + qr0 < n)
            rows.step(Qs, Ks, Vs, qr0, j * kK, v0, v1, p.scale, p.softcap, lane);
        if(!more) break;
        __syncthreads();
        tile.stage(Ks, Vs, tid);
        __syncthreads();
    }
    rows.finish_sums();
    const float i0 = finish_row(rows.m0, rows.l0, p.sinks, h), i1 = finish_row(rows.m1, rows.l1, p.sinks, h);
    const int t0r = qt0 + qr0 + gl, t1r = t0r + 8;
#pragma unroll
    for(int c = 0; c < HD / 8; ++c) {
        const int d = c * 8 + 2 * t4;
        if(t0r < n)
            *reinterpret_cast<float2*>(p.out + static_cast<std::size_t>(q0 + t0r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float2(rows.o[c][0] * i0, rows.o[c][1] * i0);
        if(t1r < n)
            *reinterpret_cast<float2*>(p.out + static_cast<std::size_t>(q0 + t1r) * p.out_stride + static_cast<std::size_t>(h) * HD + d) =
                make_float2(rows.o[c][2] * i1, rows.o[c][3] * i1);
    }
}

// ------------------------------------------------------------------------------------------------ decode
// A decode block runs two key streams: warps 0-3 take tiles 0, 2, 4, ... of the block's pieces and warps 4-7 tiles
// 1, 3, 5, ..., each with its own online softmax over the same rows (decode rows are few, so a single stream leaves
// tensor cores idle while tiles load). Each stream stores its own partial result and the combine merges them.
// Scratch: partial outputs [seqs][kv_heads][blocks][2][R][HD], then (max, sum) pairs [seqs][kv_heads][blocks][2][R][2],
// with R = max_columns * group rows per (sequence, key/value head).
constexpr int kStreams = 2, kStreamThreads = kThreads / kStreams;

template <int HD, Format F>
__global__ void __launch_bounds__(kThreads, 1) decode_kernel(Decode p) {
    constexpr int C = HD / 8;
    extern __shared__ uint4 smem[];
    uint4* Qs = smem;
    const int g = blockIdx.x, blk = blockIdx.y, s = blockIdx.z, group = p.heads / p.pages.kv_heads;
    const int R = p.max_columns * group;
    const int q0 = p.batch.q_start[s], n = p.batch.q_start[s + 1] - q0;
    const bool bad = broken(p.batch, s, n, p.max_columns, p.pages.page_tokens);  // the combine answers it
    const int nkeys = p.batch.kv_lens[s], pos0 = nkeys - n;
    const std::int32_t* table = p.batch.tables + static_cast<std::size_t>(s) * p.batch.table_stride;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int stream = warp / 4, stid = tid % kStreamThreads;
    // Pieces stay assigned by position (piece P to block P mod blocks); with a window the walk starts at the oldest
    // column's window, so the work follows the window, not the context.
    const int klo = window_lo(pos0, p.window), first = klo / kPiece;
    const int piece0 = first + ((blk - first % p.blocks) % p.blocks + p.blocks) % p.blocks;
    uint4* Ks = Qs + kRows * C + stream * 2 * kK * C;
    uint4* Vs = Ks + kK * C;
    const std::size_t unit = ((static_cast<std::size_t>(s) * p.pages.kv_heads + g) * p.blocks + blk) * kStreams + stream;
    float* part = p.partial + unit * R * HD;
    float* ml = p.partial + static_cast<std::size_t>(p.batch.seqs) * p.pages.kv_heads * p.blocks * kStreams * R * HD + unit * R * 2;
    auto tile_key = [&](int j) { return (piece0 + (j >> 2) * p.blocks) * kPiece + (j & 3) * kK; };
    if(n <= 0 || bad || tile_key(0) >= nkeys) {
        for(int r = stid; r < R; r += kStreamThreads) { ml[r * 2] = -kInf; ml[r * 2 + 1] = 0.f; }
        return;
    }
    const int used = n * group;
    for(int e = tid; e < kRows * C; e += kThreads) {
        const int r = e / C, c = e % C;
        uint4* dst = Qs + swizzle(r, c, C);
        if(r < used) cp_async16(dst, p.q + static_cast<std::size_t>(q0 + r / group) * p.q_stride + static_cast<std::size_t>(g * group + r % group) * HD + c * 8);
        else *dst = make_uint4(0u, 0u, 0u, 0u);
    }
    asm volatile("cp.async.commit_group;");
    Tile<HD, F, kStreamThreads> tile;
    tile.fetch(p.pages, table, g, tile_key(stream), klo, nkeys, stid);
    tile.stage(Ks, Vs, stid);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    Rows<HD> rows;
    const int qr0 = (warp % 4) * 16, r0 = qr0 + gl, r1 = r0 + 8;
    const bool computes = qr0 < used;
    auto visible = [&](int r) {
        Visible v;
        if(r >= used) return v;
        const int column = r / group;
        v.pos = pos0 + column;
        int depth = column;
        if(p.tree) {
            v.tree = p.tree[q0 + column];
            v.base = pos0;
            depth = __popcll(v.tree) - 1;
        }
        v.lo = window_lo(pos0 + depth, p.window);
        return v;
    };
    const Visible v0 = visible(r0), v1 = visible(r1);
    for(int i = 0;; i += kStreams) {
        const int kfirst = tile_key(i + stream);
        const bool more = tile_key(i + kStreams) < nkeys;  // the same for both streams
        if(more) tile.fetch(p.pages, table, g, tile_key(i + kStreams + stream), klo, nkeys, stid);
        if(computes && kfirst < nkeys) rows.step(Qs, Ks, Vs, qr0, kfirst, v0, v1, p.scale, p.softcap, lane);
        if(!more) break;
        __syncthreads();
        tile.stage(Ks, Vs, stid);
        __syncthreads();
    }
    rows.finish_sums();
    const float i0 = rows.l0 > 0.f ? 1.f / rows.l0 : 0.f, i1 = rows.l1 > 0.f ? 1.f / rows.l1 : 0.f;
#pragma unroll
    for(int c = 0; c < HD / 8; ++c) {
        const int dd = c * 8 + 2 * t4;
        if(r0 < R) *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r0) * HD + dd) = make_float2(rows.o[c][0] * i0, rows.o[c][1] * i0);
        if(r1 < R) *reinterpret_cast<float2*>(part + static_cast<std::size_t>(r1) * HD + dd) = make_float2(rows.o[c][2] * i1, rows.o[c][3] * i1);
    }
    if(t4 == 0) {
        if(r0 < R) { ml[r0 * 2] = rows.m0; ml[r0 * 2 + 1] = rows.l0; }
        if(r1 < R) { ml[r1 * 2] = rows.m1; ml[r1 * 2 + 1] = rows.l1; }
    }
}

// Block (column, query head, sequence), one thread per dimension: partial weights l_e exp(m_e - M) over the entries
// e = (block, stream) in that order, summed in that order.
template <int HD>
__global__ void __launch_bounds__(HD) combine_kernel(Decode p) {
    extern __shared__ float wsp[];
    __shared__ float total;
    const int c = blockIdx.x, h = blockIdx.y, s = blockIdx.z, d = threadIdx.x;
    const int group = p.heads / p.pages.kv_heads, g = h / group, R = p.max_columns * group;
    const int q0 = p.batch.q_start[s], n = p.batch.q_start[s + 1] - q0;
    if(broken(p.batch, s, n, p.max_columns, p.pages.page_tokens)) {
        poison(p.batch, p.out, p.out_stride, q0, n, c, gridDim.x, h, HD, c == 0 && h == 0);
        return;
    }
    if(c >= n) return;
    const int r = c * group + h % group;
    const int nkeys = p.batch.kv_lens[s];
    const int ns = nkeys > 0 ? min(p.blocks, (nkeys + kPiece - 1) / kPiece) * kStreams : 0;  // entries of blocks holding keys
    const std::size_t unit0 = (static_cast<std::size_t>(s) * p.pages.kv_heads + g) * p.blocks * kStreams;
    const float* part = p.partial + unit0 * R * HD;
    const float* ml = p.partial + static_cast<std::size_t>(p.batch.seqs) * p.pages.kv_heads * p.blocks * kStreams * R * HD + unit0 * R * 2;
    for(int b = d; b < ns; b += blockDim.x) wsp[b] = ml[(static_cast<std::size_t>(b) * R + r) * 2];
    __syncthreads();
    if(d == 0) {
        float M = p.sinks ? p.sinks[h] : -kInf;
        for(int b = 0; b < ns; ++b) M = fmaxf(M, wsp[b]);
        float den = 0.f;
        for(int b = 0; b < ns; ++b) {
            const float m = wsp[b], l = ml[(static_cast<std::size_t>(b) * R + r) * 2 + 1];
            const float w = m == -kInf ? 0.f : l * __expf(m - M);
            wsp[b] = w;
            den += w;
        }
        if(p.sinks) den += __expf(p.sinks[h] - M);  // the sink's weight joins the denominator after the blocks'
        total = den;
    }
    __syncthreads();
    float num = 0.f;
    for(int b = 0; b < ns; ++b) {
        const float w = wsp[b];
        if(w != 0.f) num = fmaf(w, part[(static_cast<std::size_t>(b) * R + r) * HD + d], num);
    }
    p.out[static_cast<std::size_t>(q0 + c) * p.out_stride + static_cast<std::size_t>(h) * HD + d] = total > 0.f ? num / total : 0.f;
}

// ------------------------------------------------------------------------------------------------ launch
bool valid(const Pages& pg) {
    return pg.k && pg.v && pg.page_tokens > 0 && pg.kv_heads > 0 && (pg.head_dim == 64 || pg.head_dim == 128 || pg.head_dim == 256) &&
           (pg.format == Format::bf16 || (pg.format == Format::e4m3 && pg.k_scale && pg.v_scale));
}
bool valid(const Batch& b) { return b.seqs > 0 && b.seqs <= 65535 && b.tables && b.table_stride > 0 && b.kv_lens && b.q_start; }
bool valid_heads(const Pages& pg, int heads, int q_stride, int out_stride) {
    return heads > 0 && heads % pg.kv_heads == 0 && q_stride % 8 == 0 && q_stride >= heads * pg.head_dim && out_stride % 2 == 0 &&
           out_stride >= heads * pg.head_dim;
}

template <int HD, Format F>
cudaError_t run_prefill(const Prefill& p, cudaStream_t s) {
    const std::size_t bytes = prefill_shared_bytes(HD);
    const cudaError_t attr = cudaFuncSetAttribute(prefill_kernel<HD, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if(attr != cudaSuccess) return attr;
    prefill_kernel<HD, F><<<dim3((p.max_queries + kQ - 1) / kQ, p.heads, p.batch.seqs), kThreads, bytes, s>>>(p);
    return cudaGetLastError();
}

template <int HD, Format F>
cudaError_t run_decode(const Decode& p, cudaStream_t s) {
    const std::size_t bytes = decode_shared_bytes(HD);
    const cudaError_t attr = cudaFuncSetAttribute(decode_kernel<HD, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if(attr != cudaSuccess) return attr;
    decode_kernel<HD, F><<<dim3(p.pages.kv_heads, p.blocks, p.batch.seqs), kThreads, bytes, s>>>(p);
    combine_kernel<HD><<<dim3(p.max_columns, p.heads, p.batch.seqs), HD, sizeof(float) * p.blocks * kStreams, s>>>(p);
    return cudaGetLastError();
}

template <template <int, Format> class Run, class P>
cudaError_t dispatch(const P& p, cudaStream_t s) {
    const bool e4m3 = p.pages.format == Format::e4m3;
    switch(p.pages.head_dim) {
    case 64: return e4m3 ? Run<64, Format::e4m3>::go(p, s) : Run<64, Format::bf16>::go(p, s);
    case 128: return e4m3 ? Run<128, Format::e4m3>::go(p, s) : Run<128, Format::bf16>::go(p, s);
    default: return e4m3 ? Run<256, Format::e4m3>::go(p, s) : Run<256, Format::bf16>::go(p, s);
    }
}
template <int HD, Format F> struct PrefillRun { static cudaError_t go(const Prefill& p, cudaStream_t s) { return run_prefill<HD, F>(p, s); } };
template <int HD, Format F> struct DecodeRun { static cudaError_t go(const Decode& p, cudaStream_t s) { return run_decode<HD, F>(p, s); } };

}  // namespace

std::size_t prefill_shared_bytes(int head_dim) { return static_cast<std::size_t>(kQ + 2 * kK) * head_dim * 2; }
std::size_t decode_shared_bytes(int head_dim) { return static_cast<std::size_t>(kRows + 2 * kStreams * kK) * head_dim * 2; }

cudaError_t write(const Pages& pg, const __nv_bfloat16* k, const __nv_bfloat16* v, int stride, int rows, const std::int32_t* slots,
                  cudaStream_t s) {
    if(!valid(pg) || !k || !v || !slots || rows <= 0 || stride < pg.kv_heads * pg.head_dim) return cudaErrorInvalidValue;
    const dim3 grid(rows, pg.kv_heads);
    const bool e4m3 = pg.format == Format::e4m3;
    switch(pg.head_dim) {
    case 64: e4m3 ? write_kernel<64, Format::e4m3><<<grid, 64, 0, s>>>(pg, k, v, stride, slots)
                  : write_kernel<64, Format::bf16><<<grid, 64, 0, s>>>(pg, k, v, stride, slots); break;
    case 128: e4m3 ? write_kernel<128, Format::e4m3><<<grid, 128, 0, s>>>(pg, k, v, stride, slots)
                   : write_kernel<128, Format::bf16><<<grid, 128, 0, s>>>(pg, k, v, stride, slots); break;
    default: e4m3 ? write_kernel<256, Format::e4m3><<<grid, 256, 0, s>>>(pg, k, v, stride, slots)
                  : write_kernel<256, Format::bf16><<<grid, 256, 0, s>>>(pg, k, v, stride, slots); break;
    }
    return cudaGetLastError();
}

cudaError_t prefill(const Prefill& p, cudaStream_t s) {
    if(!valid(p.pages) || !valid(p.batch) || !p.q || !p.out || p.max_queries <= 0 || !valid_heads(p.pages, p.heads, p.q_stride, p.out_stride) ||
       p.window < 0 || !(p.softcap >= 0.f))
        return cudaErrorInvalidValue;
    return dispatch<PrefillRun>(p, s);
}

std::size_t decode_partial_floats(int seqs, int kv_heads, int head_dim, int blocks, int max_columns, int heads) {
    const std::size_t rows = static_cast<std::size_t>(max_columns) * (heads / kv_heads);
    return static_cast<std::size_t>(seqs) * kv_heads * blocks * kStreams * rows * (head_dim + 2);
}

cudaError_t decode(const Decode& p, cudaStream_t s) {
    if(!valid(p.pages) || !valid(p.batch) || !p.q || !p.out || !p.partial || !valid_heads(p.pages, p.heads, p.q_stride, p.out_stride) ||
       p.max_columns < 1 || p.max_columns * (p.heads / p.pages.kv_heads) > kRows || p.blocks < 1 || p.blocks > 1024 ||
       p.window < 0 || !(p.softcap >= 0.f) || (p.tree && p.window > 0 && p.window <= 64))
        return cudaErrorInvalidValue;
    return dispatch<DecodeRun>(p, s);
}

}  // namespace crit::paged_attention
