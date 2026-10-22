// Synthetic transformer kernels; see synthetic_model.h.
#include "synthetic_model.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace crit::batched_engine {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr int kTile = 64;      // GEMM rows, columns and K per tile
constexpr int kThreads = 128;  // GEMM: 4 warps of 16 rows, each covering all 64 columns
constexpr int kChunks = kTile / 8;

__device__ __forceinline__ int swizzle(int row, int chunk) { return row * kChunks + (chunk ^ (row & 7)); }
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}

__device__ __forceinline__ std::uint64_t splitmix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__global__ void fill_kernel(__nv_bfloat16* out, std::size_t count, float bound, std::uint64_t seed) {
    for(std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const double u = static_cast<double>(splitmix(seed ^ splitmix(i)) >> 11) * 0x1.0p-53;
        out[i] = __float2bfloat16(static_cast<float>((2.0 * u - 1.0) * bound));
    }
}

struct GemmArgs {
    const __nv_bfloat16* x;
    const __nv_bfloat16* w;
    float* c;
    float* scratch;
    int rows, n, k, tiles_per_split, splits;
    bool accumulate;
};

// Block (column tile, row tile, split): two cp.async stages of 64 x 64 X and W tiles, BF16 mma.m16n8k16 in K order.
__global__ void __launch_bounds__(kThreads) gemm_kernel(GemmArgs g) {
    extern __shared__ uint4 smem[];
    uint4* xs = smem;                       // [2][64 rows][8 chunks]
    uint4* ws = smem + 2 * kTile * kChunks; // [2][64 rows][8 chunks]
    const int col0 = blockIdx.x * kTile, row0 = blockIdx.y * kTile, split = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gl = lane >> 2, t4 = lane & 3;
    const int ktiles = g.k / kTile, first = split * g.tiles_per_split, last = min(first + g.tiles_per_split, ktiles);
    auto load = [&](int stage, int kt) {
        for(int e = tid; e < kTile * kChunks; e += kThreads) {
            const int r = e / kChunks, c = e % kChunks;
            const bool ok = row0 + r < g.rows;
            cp_async16(xs + stage * kTile * kChunks + swizzle(r, c),
                       g.x + static_cast<std::size_t>(ok ? row0 + r : 0) * g.k + kt * kTile + c * 8, ok);
            cp_async16(ws + stage * kTile * kChunks + swizzle(r, c), g.w + static_cast<std::size_t>(col0 + r) * g.k + kt * kTile + c * 8, true);
        }
        asm volatile("cp.async.commit_group;");
    };
    float acc[8][4];
#pragma unroll
    for(int n = 0; n < 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    if(first < last) load(0, first);
    for(int kt = first; kt < last; ++kt) {
        const int stage = (kt - first) & 1;
        if(kt + 1 < last) {
            load(stage ^ 1, kt + 1);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
        const uint4* xt = xs + stage * kTile * kChunks;
        const uint4* wt = ws + stage * kTile * kChunks;
#pragma unroll
        for(int ks = 0; ks < kTile / 16; ++ks) {
            std::uint32_t a[4];
            ldsm_x4(a, xt + swizzle(warp * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
            for(int np = 0; np < 4; ++np) {
                std::uint32_t b[4];
                ldsm_x4(b, wt + swizzle(np * 16 + (lane & 7) + ((lane >> 4) << 3), ks * 2 + ((lane >> 3) & 1)));
                mma_bf16(acc[2 * np], a, b[0], b[1]);
                mma_bf16(acc[2 * np + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    const int r0 = row0 + warp * 16 + gl, r1 = r0 + 8;
    float* out = g.splits == 1 ? g.c : g.scratch + static_cast<std::size_t>(split) * g.rows * g.n;
    const bool add = g.splits == 1 && g.accumulate;
#pragma unroll
    for(int n = 0; n < 8; ++n) {
        const int col = col0 + n * 8 + 2 * t4;
        if(r0 < g.rows) {
            float2* p = reinterpret_cast<float2*>(out + static_cast<std::size_t>(r0) * g.n + col);
            const float2 old = add ? *p : make_float2(0.f, 0.f);
            *p = make_float2(old.x + acc[n][0], old.y + acc[n][1]);
        }
        if(r1 < g.rows) {
            float2* p = reinterpret_cast<float2*>(out + static_cast<std::size_t>(r1) * g.n + col);
            const float2 old = add ? *p : make_float2(0.f, 0.f);
            *p = make_float2(old.x + acc[n][2], old.y + acc[n][3]);
        }
    }
}

// Sum of the splits in split order, stored or added to c.
__global__ void reduce_kernel(GemmArgs g) {
    const std::size_t total = static_cast<std::size_t>(g.rows) * g.n;
    for(std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < total; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        float sum = 0.f;
        for(int s = 0; s < g.splits; ++s) sum += g.scratch[static_cast<std::size_t>(s) * total + i];
        g.c[i] = g.accumulate ? g.c[i] + sum : sum;
    }
}

__global__ void embed_kernel(const __nv_bfloat16* table, int hidden, const std::int32_t* tokens, float* x) {
    const std::size_t r = blockIdx.x;
    const __nv_bfloat16* row = table + static_cast<std::size_t>(tokens[r]) * hidden;
    for(int d = threadIdx.x; d < hidden; d += blockDim.x) x[r * hidden + d] = __bfloat162float(row[d]);
}

// One block per row; a fixed-shape tree reduction keeps the sum's order independent of everything but the row.
__global__ void rmsnorm_kernel(const float* x, const std::int32_t* index, int hidden, __nv_bfloat16* y) {
    __shared__ float part[32];
    const int i = blockIdx.x, r = index ? index[i] : i;
    const float* row = x + static_cast<std::size_t>(r) * hidden;
    float ss = 0.f;
    for(int d = threadIdx.x; d < hidden; d += blockDim.x) ss = fmaf(row[d], row[d], ss);
#pragma unroll
    for(int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    if(threadIdx.x < 32) {
        float v = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0.f;
#pragma unroll
        for(int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if(threadIdx.x == 0) part[0] = rsqrtf(v / static_cast<float>(hidden) + 1e-6f);
    }
    __syncthreads();
    const float inv = part[0];
    for(int d = threadIdx.x; d < hidden; d += blockDim.x) y[static_cast<std::size_t>(i) * hidden + d] = __float2bfloat16(row[d] * inv);
}

__global__ void qkv_rope_kernel(const float* qkv, const std::int32_t* positions, const float* inv_freq, Shape s, __nv_bfloat16* q,
                                __nv_bfloat16* k, __nv_bfloat16* v) {
    const std::size_t r = blockIdx.x;
    const float* row = qkv + r * static_cast<std::size_t>((s.heads + 2 * s.kv_heads) * s.head_dim);
    const float pos = static_cast<float>(positions[r]);
    const int half = s.head_dim / 2, rotated = (s.heads + s.kv_heads) * half;
    for(int e = threadIdx.x; e < rotated; e += blockDim.x) {
        const int h = e / half, i = e % half;
        float sn, cs;
        sincosf(pos * inv_freq[i], &sn, &cs);
        const float x0 = row[h * s.head_dim + i], x1 = row[h * s.head_dim + i + half];
        const __nv_bfloat16 a = __float2bfloat16(x0 * cs - x1 * sn), b = __float2bfloat16(x0 * sn + x1 * cs);
        if(h < s.heads) {
            q[(r * s.heads + h) * s.head_dim + i] = a;
            q[(r * s.heads + h) * s.head_dim + i + half] = b;
        } else {
            const int g = h - s.heads;
            k[(r * s.kv_heads + g) * s.head_dim + i] = a;
            k[(r * s.kv_heads + g) * s.head_dim + i + half] = b;
        }
    }
    const int vw = s.kv_heads * s.head_dim, v0 = (s.heads + s.kv_heads) * s.head_dim;
    for(int e = threadIdx.x; e < vw; e += blockDim.x) v[r * vw + e] = __float2bfloat16(row[v0 + e]);
}

__global__ void silu_mul_kernel(const float* gu, int ffn, __nv_bfloat16* out) {
    const std::size_t r = blockIdx.x;
    for(int j = threadIdx.x; j < ffn; j += blockDim.x) {
        const float g = gu[r * 2 * ffn + j], u = gu[r * 2 * ffn + ffn + j];
        out[r * ffn + j] = __float2bfloat16(g / (1.f + expf(-g)) * u);
    }
}

__global__ void to_bf16_kernel(const float* x, std::size_t count, __nv_bfloat16* out) {
    for(std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        out[i] = __float2bfloat16(x[i]);
}

__global__ void argmax_kernel(const float* logits, int vocab, std::int32_t* out) {
    __shared__ float best_v[32];
    __shared__ int best_i[32];
    const float* row = logits + static_cast<std::size_t>(blockIdx.x) * vocab;
    float v = -kInf;
    int i = vocab;
    for(int j = threadIdx.x; j < vocab; j += blockDim.x)
        if(row[j] > v) { v = row[j]; i = j; }  // each thread sees increasing j, so ties keep its lowest index
    const auto better = [](float av, int ai, float bv, int bi) { return bv > av || (bv == av && bi < ai); };
#pragma unroll
    for(int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if(better(v, i, ov, oi)) { v = ov; i = oi; }
    }
    if((threadIdx.x & 31) == 0) { best_v[threadIdx.x >> 5] = v; best_i[threadIdx.x >> 5] = i; }
    __syncthreads();
    if(threadIdx.x == 0) {
        for(int w = 1; w < static_cast<int>(blockDim.x / 32); ++w)
            if(better(v, i, best_v[w], best_i[w])) { v = best_v[w]; i = best_i[w]; }
        out[blockIdx.x] = i;
    }
}

int grid_for(std::size_t count) { return static_cast<int>(std::min<std::size_t>((count + 255) / 256, 4096)); }

}  // namespace

bool valid(const Shape& s) {
    return s.vocab > 0 && s.hidden > 0 && s.layers > 0 && s.heads > 0 && s.kv_heads > 0 && s.heads % s.kv_heads == 0 && s.ffn > 0 &&
           (s.head_dim == 64 || s.head_dim == 128 || s.head_dim == 256) && s.hidden % 64 == 0 && s.ffn % 64 == 0 && s.vocab % 64 == 0 &&
           (s.heads * s.head_dim) % 64 == 0 && s.qkv() % 64 == 0 && s.rope_theta > 0.f;
}

cudaError_t fill_uniform(__nv_bfloat16* out, std::size_t count, float bound, std::uint64_t seed, cudaStream_t stream) {
    if(!out || !count) return cudaErrorInvalidValue;
    fill_kernel<<<grid_for(count), 256, 0, stream>>>(out, count, bound, seed);
    return cudaGetLastError();
}

GemmPlan plan_gemm(int n, int k, int multiprocessors) {
    GemmPlan p{n, k, 1, k / kTile};
    const int tiles_n = n / kTile, ktiles = k / kTile;
    // Split K until the column tiles alone cover the multiprocessors once: decode steps have one row tile.
    const int want = std::max(1, std::min(ktiles, multiprocessors / std::max(1, tiles_n)));
    p.tiles_per_split = (ktiles + want - 1) / want;
    p.splits = (ktiles + p.tiles_per_split - 1) / p.tiles_per_split;
    return p;
}

std::size_t gemm_scratch_floats(const GemmPlan& p, int max_rows) {
    return p.splits > 1 ? static_cast<std::size_t>(p.splits) * max_rows * p.n : 0;
}

cudaError_t gemm(const GemmPlan& p, const __nv_bfloat16* x, int rows, const __nv_bfloat16* w, float* c, bool accumulate, float* scratch,
                 cudaStream_t stream) {
    if(!x || !w || !c || rows <= 0 || p.n <= 0 || p.n % kTile || p.k <= 0 || p.k % kTile || (p.splits > 1 && !scratch))
        return cudaErrorInvalidValue;
    const GemmArgs g{x, w, c, scratch, rows, p.n, p.k, p.tiles_per_split, p.splits, accumulate};
    constexpr int bytes = 4 * kTile * kTile * 2;
    gemm_kernel<<<dim3(p.n / kTile, (rows + kTile - 1) / kTile, p.splits), kThreads, bytes, stream>>>(g);
    if(p.splits > 1) reduce_kernel<<<grid_for(static_cast<std::size_t>(rows) * p.n), 256, 0, stream>>>(g);
    return cudaGetLastError();
}

cudaError_t embed(const __nv_bfloat16* table, int hidden, const std::int32_t* tokens, int rows, float* x, cudaStream_t stream) {
    if(!table || !tokens || !x || rows <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    embed_kernel<<<rows, 256, 0, stream>>>(table, hidden, tokens, x);
    return cudaGetLastError();
}

cudaError_t rmsnorm(const float* x, const std::int32_t* index, int rows, int hidden, __nv_bfloat16* y, cudaStream_t stream) {
    if(!x || !y || rows <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, index, hidden, y);
    return cudaGetLastError();
}

cudaError_t qkv_rope(const float* qkv, const std::int32_t* positions, const float* inv_freq, int rows, const Shape& shape, __nv_bfloat16* q,
                     __nv_bfloat16* k, __nv_bfloat16* v, cudaStream_t stream) {
    if(!qkv || !positions || !inv_freq || !q || !k || !v || rows <= 0 || !valid(shape)) return cudaErrorInvalidValue;
    qkv_rope_kernel<<<rows, 256, 0, stream>>>(qkv, positions, inv_freq, shape, q, k, v);
    return cudaGetLastError();
}

cudaError_t silu_mul(const float* gu, int rows, int ffn, __nv_bfloat16* out, cudaStream_t stream) {
    if(!gu || !out || rows <= 0 || ffn <= 0) return cudaErrorInvalidValue;
    silu_mul_kernel<<<rows, 256, 0, stream>>>(gu, ffn, out);
    return cudaGetLastError();
}

cudaError_t to_bf16(const float* x, std::size_t count, __nv_bfloat16* out, cudaStream_t stream) {
    if(!x || !out || !count) return cudaErrorInvalidValue;
    to_bf16_kernel<<<grid_for(count), 256, 0, stream>>>(x, count, out);
    return cudaGetLastError();
}

cudaError_t argmax(const float* logits, int rows, int vocab, std::int32_t* out, cudaStream_t stream) {
    if(!logits || !out || rows <= 0 || vocab <= 0) return cudaErrorInvalidValue;
    argmax_kernel<<<rows, 256, 0, stream>>>(logits, vocab, out);
    return cudaGetLastError();
}

}  // namespace crit::batched_engine
