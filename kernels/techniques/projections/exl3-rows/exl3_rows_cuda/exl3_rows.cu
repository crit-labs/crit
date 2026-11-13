// EXL3 projections (exl3_rows.h): the Hadamard rotations, the tiles decoded into a GEMV, and a decode of the tiles.
#include "exl3_rows.h"
#include "crit_tuning.h"
#include "crit_kcore/exl3.cuh"
#include "crit_kcore/mma.cuh"
#include "crit_kcore/load.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace crit::techniques::exl3_rows {
namespace {

namespace kc = crit::kcore;

// Knobs (kernel.json, structure gemv; crit_tuning.h picks them per out, in and row count):
//   WARPS   warps a block
//   SPLITS  contiguous parts of the input's tiles, one a block row, added in part order (0: 48 warps a
//           multiprocessor, each part at least two stages): changes the sums' order, so one value a matrix (an order knob)
//   STRIPS  neighbouring tile strips a warp decodes (the activations loaded once for them)
//   ROWS    input rows a stage copies
//   STAGES  stages in flight
// All but SPLITS change speed only.

__device__ __forceinline__ float load_value(const void* p, int kind, int64_t i) {
    if (kind == 1) return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
    if (kind == 2) return __half2float(static_cast<const __half*>(p)[i]);
    return static_cast<const float*>(p)[i];
}
__device__ __forceinline__ void store_value(void* p, int kind, int64_t i, float v) {
    if (kind == 1) static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
    else if (kind == 2) static_cast<__half*>(p)[i] = __float2half_rn(v);
    else static_cast<float*>(p)[i] = v;
}

// One block a (row, 128-value block): times pre, the Walsh-Hadamard butterflies at strides 1 to 64 in that order
// (the natural-order Sylvester H), times 1 / sqrt(128) and post.
__global__ void __launch_bounds__(kBlock) rotate_kernel(const void* __restrict__ rows, int in_kind, int64_t ld_in, int L,
                                                        const __half* __restrict__ pre, const __half* __restrict__ post,
                                                        void* __restrict__ out, int out_kind, int64_t ld_out, int parts,
                                                        int64_t part_stride) {
    __shared__ float s[kBlock];
    const int row = blockIdx.y, t = threadIdx.x, c = blockIdx.x * kBlock + t;
    kc::griddep_wait();   // launched as the dependent of the kernel before it (9.0 on): its writes are visible now
    float v = load_value(rows, in_kind, row * ld_in + c);
    for (int p = 1; p < parts; ++p) v += static_cast<const float*>(rows)[p * part_stride + row * ld_in + c];
    if (pre) v *= __half2float(pre[c]);
    s[t] = v;
    __syncthreads();
#pragma unroll
    for (int h = 1; h < kBlock; h <<= 1) {
        float a = 0.f, b = 0.f;
        const bool low = (t & h) == 0;
        if (low) {
            a = s[t];
            b = s[t + h];
        }
        __syncthreads();
        if (low) {
            s[t] = a + b;
            s[t + h] = a - b;
        }
        __syncthreads();
    }
    kc::griddep_launch_dependents();
    v = s[t] * 0.08838834764831845f;   // 1 / sqrt(128)
    if (post) v *= __half2float(post[c]);
    store_value(out, out_kind, row * ld_out + c, v);
}

// A warp's tile: its words into shared memory (lane i loads words i and i + 32).
__device__ __forceinline__ void load_tile(const std::uint16_t* __restrict__ trellis, size_t tile, int count,
                                          uint32_t* words, int lane) {
    const uint32_t* src = reinterpret_cast<const uint32_t*>(trellis) + tile * count;
    for (int i = lane; i < count; i += 32) words[i] = __ldg(src + i);
    __syncwarp();
}

template <int CB>
__global__ void __launch_bounds__(256) decode_kernel(const std::uint16_t* __restrict__ trellis, int tiles,
                                                     int tiles_out, int bits, bool half_bit, __half* __restrict__ q,
                                                     int out) {
    __shared__ uint32_t words[8][64];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tile = blockIdx.x * 8 + warp, ti = tile / tiles_out, tn = tile - ti * tiles_out;
    if (tile >= tiles) return;   // the warp leaves together
    const int count = kc::exl3_words(bits, half_bit);
    load_tile(trellis, static_cast<size_t>(tile), count, words[warp], lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float v = kc::exl3_value<CB>(kc::exl3_state(words[warp], count, bits, half_bit, 8 * lane + j));
        q[static_cast<size_t>(ti * 16 + kc::exl3_row(lane, j)) * out + tn * 16 + kc::exl3_col(lane, j)] =
            __float2half_rn(v);   // exact: v is an FP16 value
    }
}

// One block: CTA = WARPS x STRIPS neighbouring tile strips (16 output columns each) over its split of the input tiles.
// The strips' tiles of one input row lie side by side in memory, a slice of CTA x count words; ROWS slices make a
// stage, copied into shared memory with 16-byte cp.async by every thread, STAGES stages in flight (the classic
// multistage ring: the copies of the next stages overlap the decode and multiply of this one). Warp w decodes its
// STRIPS strips' tiles from shared memory, each tile once: lane L's eight positions are exactly its A fragment of an
// m16n8k16 MMA whose rows are the strip's 16 output columns (positions 8 L, + 1: column L / 4, k 2 (L % 4) + {0, 1}:
// a0; + 2, + 3: k + 8: a2; + 4, + 5: column + 8: a1; + 6, + 7: a3). The activations are the B operand, 8 rows a column
// group (zeros past M), copied into shared memory with the stage and read once for the warp's strips: a tile costs
// one MMA for up to 8 rows, two for up to 16.
// FP16 operands, FP32 accumulators in input-tile order, the same instructions for a row at any M within its group count.
template <int CB, int BITS, bool HALF, int WARPS, int STRIPS, int ROWS, int STAGES, int G>
__global__ void __launch_bounds__(WARPS * 32) tiles_kernel(const std::uint16_t* __restrict__ trellis, int tiles_in,
                                                           int tiles_out, const __half* __restrict__ xh, int M, int in,
                                                           float* __restrict__ yr, int out) {
    constexpr int count = kc::exl3_words(BITS, HALF);
    constexpr int CTA = WARPS * STRIPS, SLICE = CTA * count, STAGE = ROWS * SLICE;
    constexpr int XSTAGE = ROWS * 8 * G * 8;   // a stage's activations: ROWS input rows x 8 G rows x 16 halves (words)
    extern __shared__ uint4 shared_words[];
    uint32_t* const smem = reinterpret_cast<uint32_t*>(shared_words);
    uint32_t* const xsmem = smem + STAGES * STAGE;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int first = blockIdx.x * CTA, strips = tiles_out - first < CTA ? tiles_out - first : CTA;
    const int split = blockIdx.y, ti_lo = split * tiles_in / gridDim.y, ti_hi = (split + 1) * tiles_in / gridDim.y;
    const int stages = (ti_hi - ti_lo + ROWS - 1) / ROWS;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(trellis);
    const int slice_chunks = strips * count / 4;   // a slice's 16-byte chunks that hold this block's tiles

    auto issue = [&](int st) {   // stage st: input rows ti_lo + ROWS st .. into its buffers, tiles and activations
        uint32_t* dst = smem + (st % STAGES) * STAGE;
        for (int i = threadIdx.x; i < STAGE / 4; i += WARPS * 32) {
            const int r = i / (SLICE / 4), chunk = i - r * (SLICE / 4), ti = ti_lo + st * ROWS + r;
            const bool valid = ti < ti_hi && chunk < slice_chunks;
            const uint32_t* src = valid ? words + (static_cast<size_t>(ti) * tiles_out + first) * count + 4 * chunk : words;
            kc::cp_async16(dst + r * SLICE + 4 * chunk, src, valid);
        }
        uint32_t* xdst = xsmem + (st % STAGES) * XSTAGE;
        for (int i = threadIdx.x; i < XSTAGE / 4; i += WARPS * 32) {   // two 16-byte chunks an (input row, row)
            const int r = i / (16 * G), rest = i - r * 16 * G, t = rest >> 1, half8 = rest & 1, ti = ti_lo + st * ROWS + r;
            const bool valid = ti < ti_hi && t < M;
            const __half* src = valid ? xh + static_cast<size_t>(t) * in + ti * 16 + 8 * half8 : xh;
            kc::cp_async16(xdst + (r * 8 * G + t) * 8 + 4 * half8, src, valid);
        }
        kc::cp_async_commit();
    };

    const kc::Exl3Lane at = kc::exl3_lane<BITS, HALF>(lane);
    float c[STRIPS][G][4] = {};

    kc::griddep_wait();   // the rotated activations of the kernel before (launched as its dependent, 9.0 on)
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < stages) issue(st);
        else kc::cp_async_commit();
    }
    for (int st = 0; st < stages; ++st) {
        kc::cp_async_wait<STAGES - 2>();
        __syncthreads();   // stage st landed for every thread, and stage st - 1's buffer is free again
        if (st + STAGES - 1 < stages) issue(st + STAGES - 1);
        else kc::cp_async_commit();
        const uint32_t* buffer = smem + (st % STAGES) * STAGE;
        const uint32_t* xbuffer = xsmem + (st % STAGES) * XSTAGE;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int ti = ti_lo + st * ROWS + r;
            if (ti >= ti_hi) break;
            uint32_t b0[G], b1[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {   // B: row 8 g + gid's k 2 tig, + 1 and + 8, + 9 (zeros past M, copied so)
                const uint32_t* xr = xbuffer + (r * 8 * G + 8 * g + gid) * 8;
                b0[g] = xr[tig];
                b1[g] = xr[4 + tig];
            }
#pragma unroll
            for (int s = 0; s < STRIPS; ++s) {
                const int local = warp * STRIPS + s;
                if (local >= strips) break;   // warp-uniform
                const uint32_t* tile = buffer + r * SLICE + local * count;
                uint32_t w[8];
                kc::exl3_lane_decode<BITS, HALF>(make_uint3(tile[at.w0], tile[at.w1], tile[at.w2]), at.r, w);
                const uint32_t a0 = kc::exl3_pair<CB>(w[0], w[1]), a1 = kc::exl3_pair<CB>(w[4], w[5]);
                const uint32_t a2 = kc::exl3_pair<CB>(w[2], w[3]), a3 = kc::exl3_pair<CB>(w[6], w[7]);
#pragma unroll
                for (int g = 0; g < G; ++g) kc::mma_f16_m16n8k16(c[s][g], a0, a1, a2, a3, b0[g], b1[g]);
            }
        }
    }
    kc::cp_async_wait<0>();
    kc::griddep_launch_dependents();
    yr += static_cast<size_t>(split) * M * out;
#pragma unroll
    for (int s = 0; s < STRIPS; ++s) {
        const int local = warp * STRIPS + s;
        if (local >= strips) break;
#pragma unroll
        for (int g = 0; g < G; ++g) {   // c0, c1: output column gid, rows 8 g + 2 tig, + 1; c2, c3: column gid + 8
            const int n = (first + local) * 16 + gid, t = 8 * g + 2 * tig;
            if (t < M) {
                yr[static_cast<size_t>(t) * out + n] = c[s][g][0];
                yr[static_cast<size_t>(t) * out + n + 8] = c[s][g][2];
            }
            if (t + 1 < M) {
                yr[static_cast<size_t>(t + 1) * out + n] = c[s][g][1];
                yr[static_cast<size_t>(t + 1) * out + n + 8] = c[s][g][3];
            }
        }
    }
}

// Shared bytes a block of this configuration takes.
template <int BITS, bool HALF, int WARPS, int STRIPS, int ROWS, int STAGES, int G>
constexpr int tiles_shared() { return STAGES * ROWS * (WARPS * STRIPS * kc::exl3_words(BITS, HALF) + 8 * G * 8) * 4; }

template <int CB, int BITS, bool HALF, int WARPS, int STRIPS, int ROWS, int STAGES, int G>
cudaError_t launch_one(const Matrix& w, const __half* xh, int M, float* yr, int parts, cudaStream_t stream) {
    constexpr int bytes = tiles_shared<BITS, HALF, WARPS, STRIPS, ROWS, STAGES, G>();
    auto kernel = tiles_kernel<CB, BITS, HALF, WARPS, STRIPS, ROWS, STAGES, G>;
    if constexpr (bytes > 48 * 1024) {
        static bool raised = false;   // the opt-in above 48 KiB, once a process and configuration
        if (!raised) {
            const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
            if (e != cudaSuccess) return e;
            raised = true;
        }
    }
    const int tiles_out = w.out / 16;
    const dim3 grid((tiles_out + WARPS * STRIPS - 1) / (WARPS * STRIPS), parts);
    return kc::launch_programmatic(kernel, grid, dim3(WARPS * 32), bytes, stream, w.trellis, w.in / 16, tiles_out, xh, M,
                                   w.in, yr, w.out);
}

template <int CB, int BITS, bool HALF>
cudaError_t launch_rate(const Matrix& w, const __half* xh, int M, float* yr, int parts, cudaStream_t stream) {
    switch (crit_lookup_gemv(w.out, w.in, M)) {
#define CRIT_EXL3_CASE(index, warps, split_knob, strips, rows, stages)                                       \
    case index:                                                                                              \
        return M > 8 ? launch_one<CB, BITS, HALF, warps, strips, rows, stages, 2>(w, xh, M, yr, parts, stream) \
                     : launch_one<CB, BITS, HALF, warps, strips, rows, stages, 1>(w, xh, M, yr, parts, stream);
        CRIT_CONFIGS_gemv(CRIT_EXL3_CASE)
#undef CRIT_EXL3_CASE
        default:
            return cudaErrorInvalidConfiguration;
    }
}

template <int CB>
cudaError_t launch_tiles(const Matrix& w, const __half* xh, int M, float* yr, int parts, cudaStream_t stream) {
    if (w.half_bit) {
        if constexpr (CB == kc::kExl3Mul1) {
            switch (w.bits) {
                case 1: return launch_rate<CB, 1, true>(w, xh, M, yr, parts, stream);
                case 2: return launch_rate<CB, 2, true>(w, xh, M, yr, parts, stream);
                case 3: return launch_rate<CB, 3, true>(w, xh, M, yr, parts, stream);
                default: return cudaErrorInvalidValue;
            }
        }
        return cudaErrorInvalidValue;
    }
    switch (w.bits) {
        case 1: return launch_rate<CB, 1, false>(w, xh, M, yr, parts, stream);
        case 2: return launch_rate<CB, 2, false>(w, xh, M, yr, parts, stream);
        case 3: return launch_rate<CB, 3, false>(w, xh, M, yr, parts, stream);
        case 4: return launch_rate<CB, 4, false>(w, xh, M, yr, parts, stream);
        case 5: return launch_rate<CB, 5, false>(w, xh, M, yr, parts, stream);
        case 6: return launch_rate<CB, 6, false>(w, xh, M, yr, parts, stream);
        case 7: return launch_rate<CB, 7, false>(w, xh, M, yr, parts, stream);
        case 8: return launch_rate<CB, 8, false>(w, xh, M, yr, parts, stream);
        default: return cudaErrorInvalidValue;
    }
}

bool valid(const Matrix& w) {
    return w.trellis && w.in > 0 && w.out > 0 && w.in % kBlock == 0 && w.out % kBlock == 0 && w.bits >= 1 &&
           w.bits <= 8 && w.codebook >= 0 && w.codebook <= 2 && (!w.half_bit || (w.codebook == 2 && w.bits <= 3)) &&
           reinterpret_cast<std::uintptr_t>(w.trellis) % 4 == 0;
}

}  // namespace

cudaError_t enqueue_rotate(const void* rows, int in_kind, int64_t ld_in, int M, int L, const void* pre,
                           const void* post, void* out, int out_kind, int64_t ld_out, cudaStream_t stream, int parts,
                           int64_t part_stride) {
    if (!rows || !out || M < 0 || L <= 0 || L % kBlock || M > 65535 || parts < 1 || (parts > 1 && in_kind != 0))
        return cudaErrorInvalidValue;
    if (M == 0) return cudaSuccess;
    return kc::launch_programmatic(rotate_kernel, dim3(L / kBlock, M), dim3(kBlock), 0, stream, rows, in_kind, ld_in, L,
                                   static_cast<const __half*>(pre), static_cast<const __half*>(post), out, out_kind,
                                   ld_out, parts, part_stride);
}

// The (block's strips, rows a stage, knob SPLITS) of this matrix's configuration at `rows` rows.
void plan_of(const Matrix& w, int rows, int& cta, int& per_stage, int& knob, int& warps_a_block) {
    cta = 8, per_stage = 4, knob = 0, warps_a_block = 4;
    switch (crit_lookup_gemv(w.out, w.in, rows < 1 ? 1 : rows)) {
#define CRIT_EXL3_PLAN(index, warps, split_knob, strips, stage_rows, stages) \
    case index:                                                              \
        cta = warps * strips, per_stage = stage_rows, knob = split_knob;     \
        warps_a_block = warps;                                               \
        break;
        CRIT_CONFIGS_gemv(CRIT_EXL3_PLAN)
#undef CRIT_EXL3_PLAN
        default:
            break;
    }
}

int splits(const Matrix& w, int rows, int multiprocessors) {
    const int tiles_in = w.in / 16, tiles_out = w.out / 16;
    int cta = 8, per_stage = 4, knob = 0, warps = 4;
    plan_of(w, rows, cta, per_stage, knob, warps);
    const int blocks = (tiles_out + cta - 1) / cta;
    // 0: enough blocks for 48 warps a multiprocessor, each split holding at least two stages' rows
    int s = knob > 0 ? knob : (48 * multiprocessors / warps + blocks - 1) / blocks;
    const int most = tiles_in / (2 * per_stage) > 0 ? tiles_in / (2 * per_stage) : 1;
    if (knob <= 0 && s > most) s = most;
    return s < 1 ? 1 : s > tiles_in ? tiles_in : s > 65535 ? 65535 : s;
}

cudaError_t enqueue_tiles(const Matrix& w, const void* xh, int M, float* yr, int parts, cudaStream_t stream) {
    if (!valid(w) || !xh || !yr || M < 1 || M > kMaxRows || parts < 1 || parts > w.in / 16) return cudaErrorInvalidValue;
    const auto* x = static_cast<const __half*>(xh);
    if (w.codebook == 0) return launch_tiles<kc::kExl3Inst>(w, x, M, yr, parts, stream);
    if (w.codebook == 1) return launch_tiles<kc::kExl3Mcg>(w, x, M, yr, parts, stream);
    return launch_tiles<kc::kExl3Mul1>(w, x, M, yr, parts, stream);
}

cudaError_t enqueue_decode(const Matrix& w, void* q, cudaStream_t stream) {
    if (!valid(w) || !q) return cudaErrorInvalidValue;
    const int tiles_in = w.in / 16, tiles_out = w.out / 16, tiles = tiles_in * tiles_out;
    const int grid = (tiles + 7) / 8;
    auto* out = static_cast<__half*>(q);
    if (w.codebook == 0) decode_kernel<kc::kExl3Inst><<<grid, 256, 0, stream>>>(w.trellis, tiles, tiles_out, w.bits, w.half_bit, out, w.out);
    else if (w.codebook == 1) decode_kernel<kc::kExl3Mcg><<<grid, 256, 0, stream>>>(w.trellis, tiles, tiles_out, w.bits, w.half_bit, out, w.out);
    else decode_kernel<kc::kExl3Mul1><<<grid, 256, 0, stream>>>(w.trellis, tiles, tiles_out, w.bits, w.half_bit, out, w.out);
    return cudaGetLastError();
}

}  // namespace crit::techniques::exl3_rows
