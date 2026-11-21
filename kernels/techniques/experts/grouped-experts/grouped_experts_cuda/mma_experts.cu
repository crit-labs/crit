// Routed SwiGLU experts on the format's own tensor cores (grouped_experts.h, arithmetics "fp4" and "fp8"): one
// mainloop for two MMA atoms, each a decode/prompt structure pair (fp4-decode / fp4-prompt, fp8-decode /
// fp8-prompt), one function of a route and its expert.
//
// - FP4 (nvfp4 bundles with input scales): mma.sync kind::mxf4nvf4 m16n8k64 (the sm_120 family). The routes' gate/up
//   inputs are quantized once per route at the expert's gate/up input scale by crit's declared NVFP4 quantizer
//   (kc::nvfp4_quantize16: E2M1 codes and a UE4M3 block scale a 16-value block, the operands a4_inputs gives the BF16
//   path as e2m1 x decoded scale), into [routes][K / 2] codes and [routes][K / 16] scale bytes; the gate/up epilogue
//   quantizes the intermediate the same way at the down's input scale. The block scales ride in the MMA.
// - FP8 (fp8:channel and fp8:block bundles, W8A8): mma.sync e4m3 m16n8k32 (8.9 on; kind::mxf8f6f4 with unit scales on
//   the sm_120 family, kc::mma_e4m3_m16n8k32_fast). The routes' inputs and the intermediate are quantized per 128
//   values of a row to E4M3 at amax / 448 (crit's declared group quantizer, quantized-projections' fp8_block.h
//   quantize_group), codes [routes][K] and FP32 scales [routes][K / 128]; each 128-k block's sum (four MMAs from zero)
//   is folded into the output's running sum with one fmaf by the activation's scale times the weight's (fp8:block's
//   128 x 128 grid; fp8:channel's row scale multiplies the finished sum).
// - The weights the A operand (16 matrix rows a tile: down's 16 output rows, or gate/up's 8 gate rows over the same 8 up
//   rows, so a lane holds both halves of its SwiGLU), the routes the B operand (8 an n-tile). A lane reads 16 contiguous
//   bytes of each of its rows a 64-byte step (two MMAs: 128 k of FP4, 64 of FP8) and the quad's word transpose gives
//   every lane its fragments. K is cut into k_parts(K) parts at fixed 128-k block boundaries (a function of K
//   alone); every output is one chain a part, over its K in order, and the parts' sums are added in part order
//   (((p0 + p1) + p2) + p3, FP32), whatever its item, its tile or the rows beside it. Decode items run a block's
//   parts on separate warps (more warps on the GPU: one token's grid is small) and add them through shared memory;
//   prompt items run them one after another in a warp, adding each into a second accumulator. The item width (8 for
//   decode rows, 32 or 64 for prompt chunks), the warps, tiles, steps in flight and the staged chunk are speed only,
//   so a prompt item gives every route its decode item's bits.
// - The row factors, crit's declared SwiGLU or GELU's tanh form; the down rows' FP32 outputs unweighted per route;
//   each token's routes summed by combine (route order).
// Other arithmetics than the BF16 path's (the same operands for FP4's W4A4, quantized activations for FP8): not its
// bits.
#include "grouped_experts.h"
#include "crit_tuning.h"
#include "crit_kcore/arch.cuh"
#include "crit_kcore/device.cuh"
#include "crit_kcore/epilogue.cuh"
#include "crit_kcore/formats.cuh"
#include "crit_kcore/load.cuh"
#include "crit_kcore/mma.cuh"
#include "crit_kcore/mma_fp8.cuh"

namespace crit::techniques::grouped_experts {
namespace {

namespace kc = crit::kcore;
using bf16 = __nv_bfloat16;

enum : int { kFp4 = 0, kFp8 = 1 };
template <int A>
constexpr int kStepK = A == kFp4 ? 128 : 64;   // k values a 64-byte step of a row holds (two MMAs)
template <int A>
constexpr int kCodesPerByte = A == kFp4 ? 2 : 1;
constexpr int kGroup = 128;                    // FP8's activation groups and weight blocks: 128 k
constexpr int kCodePad = 16;   // bytes of padding a staged route row of codes: conflict-free fragment reads
constexpr int kScalePad = 8;   // ... and of scales

#if CRIT_KCORE_HAS_BLOCK_SCALED_MMA
constexpr bool kHasFp4 = true;
#else
constexpr bool kHasFp4 = false;
#endif
#if CRIT_KCORE_HAS_FP8_MMA
constexpr bool kHasFp8 = true;
#else
constexpr bool kHasFp8 = false;
#endif

struct MmaPass {
    const std::uint8_t* store;
    Layout layout;
    const int* item_slot;
    const int* item_routes;
    const int* item_count;
    const std::uint8_t* codes;    // the routes' operands: [routes][K / codes a byte]
    const void* scales;           // FP4: UE4M3 [routes][K / 16]; FP8: FP32 [routes][K / 128]
    void* out;                    // gate/up: FP4 the intermediate's codes [routes][inter / 2], FP8 h BF16
                                  // [routes][inter]; down: y FP32 [routes][hidden]
    std::uint8_t* out_scales;     // FP4 gate/up: the intermediate's scales [routes][inter / 16]
    int chunk;                    // k values a prompt item stages a pass (a multiple of 128)
};

__device__ __forceinline__ float input_scale(const std::uint8_t* bundle, const Layout& L, int i) {
    return reinterpret_cast<const float*>(bundle + L.input)[i];
}

// The FP8 weight scale of a matrix row and 128-k block: fp8:block's grid (gate/up holds gate's grid, then up's), 1 for
// fp8:channel (its row scale multiplies the finished sum).
template <bool GATEUP>
__device__ __forceinline__ float fp8_block_scale(const std::uint8_t* scales, const Layout& L, int row, int kb) {
    if(L.format != Format::fp8_block) return 1.0f;
    const int K = GATEUP ? L.hidden : L.inter, I = L.inter;
    const int kblocks = (K + 127) / 128;
    const int brow = GATEUP ? (row < I ? row / 128 : (I + 127) / 128 + (row - I) / 128) : row / 128;
    return __ldg(reinterpret_cast<const float*>(scales) + brow * kblocks + kb);
}

// The 16-value block v of a route at global scale g: two E2M1 codes a byte (8 bytes) and the UE4M3 scale byte.
__device__ __forceinline__ void quantize_block(const float* v, float g, uint2& codes, std::uint8_t& scale) {
    float amax = 0.0f;
#pragma unroll
    for(int i = 0; i < 16; ++i) amax = kc::abs_max(amax, v[i]);
    uint32_t c[16], s;
    kc::nvfp4_quantize16(v, amax, g, c, s);
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for(int i = 0; i < 16; ++i) w[i / 8] |= c[i] << (4 * (i % 8));
    codes = make_uint2(w[0], w[1]);
    scale = static_cast<std::uint8_t>(s);
}

// Item u's routes' FP4 gate/up operands: x [tokens][hidden] BF16 (route r reads token r / top) quantized at the route's
// expert's gate/up input scale. One thread a 16-value block of one route.
__global__ void fp4_inputs_kernel(const bf16* x, int top, const std::uint8_t* store, Layout L, Items it,
                                  std::uint8_t* codes, std::uint8_t* scales) {
    const int u = blockIdx.x;
    if(u >= *it.count) return;
    const int H = L.hidden, blocks = H / 16;
    const int i = blockIdx.y * blockDim.x + threadIdx.x, j = i / blocks, b = i % blocks;
    if(j >= it.width) return;
    const int r = it.routes[static_cast<long long>(u) * it.width + j];
    if(r < 0) return;
    const std::uint8_t* bundle = store + static_cast<long long>(it.slot[u]) * L.bytes;
    const float g = input_scale(bundle, L, 0);
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<long long>(r / top) * H + 16 * b);
    const uint4 a = __ldg(src), c = __ldg(src + 1);
    const uint32_t words[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
    float v[16];
#pragma unroll
    for(int q = 0; q < 8; ++q) {
        const float2 f = __bfloat1622float2(kc::bf162_of(words[q]));
        v[2 * q] = f.x;
        v[2 * q + 1] = f.y;
    }
    uint2 out;
    std::uint8_t s;
    quantize_block(v, g, out, s);
    *reinterpret_cast<uint2*>(codes + static_cast<long long>(r) * (H / 2) + 8 * b) = out;
    scales[static_cast<long long>(r) * blocks + b] = s;
}

// Item u's routes' FP8 operands, crit's declared group quantizer: each 128 values of a row (route r's row r / div of
// rows [..][K] BF16) in E4M3 at scale amax / 448 (codes the nearest, ties to even, saturating; a zero group's scale 0
// and codes 0; a NaN or an infinity makes the scale NaN, which its sums carry). A warp a group, four values a lane.
__global__ void fp8_inputs_kernel(const bf16* rows, int div, int K, Items it, std::uint8_t* codes, float* scales) {
    const int u = blockIdx.x;
    if(u >= *it.count) return;
    const int groups = K / kGroup, warp = (blockIdx.y * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int j = warp / groups, grp = warp % groups;
    if(j >= it.width) return;
    const int r = it.routes[static_cast<long long>(u) * it.width + j];
    if(r < 0) return;
    const uint2 w = __ldg(reinterpret_cast<const uint2*>(rows + static_cast<long long>(r / div) * K + kGroup * grp + 4 * lane));
    const float2 lo = __bfloat1622float2(kc::bf162_of(w.x)), hi = __bfloat1622float2(kc::bf162_of(w.y));
    const float v[4] = {lo.x, lo.y, hi.x, hi.y};
    float amax = 0.0f;
#pragma unroll
    for(int i = 0; i < 4; ++i) amax = kc::abs_max(amax, v[i]);
#pragma unroll
    for(int o = 1; o < 32; o <<= 1) amax = kc::abs_max(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = __fdiv_rn(amax, 448.0f);
    uint32_t packed = 0u;
#pragma unroll
    for(int i = 0; i < 4; ++i)
        packed |= (scale > 0.0f ? kc::e4m3_encode_satfinite(__fdiv_rn(v[i], scale)) & 0xffu : 0u) << (8 * i);
    *reinterpret_cast<uint32_t*>(codes + static_cast<long long>(r) * K + kGroup * grp + 4 * lane) = packed;
    if(lane == 0) scales[static_cast<long long>(r) * groups + grp] = scale;
}

// The item's staged operands, k0 .. k0 + count of `rows` routes, into shared memory asynchronously (zero-filled past
// the routes): codes rows of chunk / codes a byte + kCodePad bytes, then scale rows (FP4 chunk / 16 bytes, FP8 chunk /
// 128 floats) + kScalePad.
template <int A>
__host__ __device__ constexpr int code_row(int chunk) { return chunk / kCodesPerByte<A> + kCodePad; }
template <int A>
__host__ __device__ constexpr int scale_row(int chunk) { return (A == kFp4 ? chunk / 16 : 4 * (chunk / kGroup)) + kScalePad; }
template <int A>
__host__ __device__ constexpr int half_bytes(int chunk) { return 64 * code_row<A>(chunk) + 64 * scale_row<A>(chunk); }

template <int A>
__device__ __forceinline__ void stage(const MmaPass& p, const int* routes, int rows, int K, int k0, int count,
                                      std::uint8_t* half) {
    const int cw = code_row<A>(p.chunk), sw = scale_row<A>(p.chunk);
    std::uint8_t* sbase = half + 64 * cw;   // codes of 64 routes at most, then their scales
    const int rowb = K / kCodesPerByte<A>, per = count / kCodesPerByte<A> / 16;   // 16-byte pieces of a route's codes
    for(int i = threadIdx.x; i < rows * per; i += blockDim.x) {
        const int j = i / per, c = (i % per) * 16, r = routes[j];
        const std::uint8_t* src = p.codes + static_cast<long long>(r >= 0 ? r : 0) * rowb + k0 / kCodesPerByte<A> + c;
        kc::cp_async16(half + j * cw + c, src, r >= 0);
    }
    const std::uint8_t* scales = static_cast<const std::uint8_t*>(p.scales);
    const int srow = A == kFp4 ? K / 16 : 4 * (K / kGroup), s0 = A == kFp4 ? k0 / 16 : 4 * (k0 / kGroup);
    const int sper = count / kGroup;   // FP4: 8-byte pieces (128 k); FP8: 4-byte pieces (one float a 128 k)
    for(int i = threadIdx.x; i < rows * sper; i += blockDim.x) {
        const int j = i / sper, c = i % sper, r = routes[j];
        const std::uint8_t* src = scales + static_cast<long long>(r >= 0 ? r : 0) * srow + s0;
        if constexpr(A == kFp4) kc::cp_async_ca<8>(sbase + j * sw + 8 * c, src + 8 * c, r >= 0);
        else kc::cp_async_ca<4>(sbase + j * sw + 4 * c, src + 4 * c, r >= 0);
    }
    kc::cp_async_commit();
}

// The parts K is cut into (the same for a pair's decode and prompt structures, so their sums are one expression):
// 4 from 2,048 k, 2 from 512, else 1; part q covers the 128-k blocks [q * blocks / parts, (q + 1) * blocks / parts).
__host__ __device__ constexpr int k_parts(int K) { return K >= 2048 ? 4 : K >= 512 ? 2 : 1; }
// A decode block's warp groups (row tiles of WARPS warps' worth, each run by k_parts warps, one a part).
__host__ __device__ constexpr int decode_groups(int warps, int parts) { return warps / parts > 0 ? warps / parts : 1; }

template <int DEPTH, int ROWS>
struct WeightRing {
    uint4 w[DEPTH][ROWS];       // a step's 16 bytes of each of the lane's rows
    uint2 s[DEPTH][ROWS / 2];   // FP4: the step's two scale words of the lane's scale row a tile
};

// One block: item blockIdx.y, WARPS warps of TILES MMA row tiles each. NT n-tiles of 8 routes (1: decode items, the
// routes' operands read straight from global memory; 4 or 8: prompt items, staged a chunk at a time). FETCH (decode
// items): 0, a lane loads 16 contiguous bytes of a step's row and a quad transpose gives it its fragment words; 1, a
// lane loads its four fragment words directly (bytes 16 s + 4 t4 of the step, s = 0..3): the same registers, so the
// same MMAs and bits, with four 4-byte loads in place of one 16-byte load and the transpose's shuffles.
template <int A, int NT, int TILES, int WARPS, int DEPTH, bool GATEUP, int FETCH = 0>
__global__ void __launch_bounds__((NT == 1 && WARPS < 4 ? 4 : WARPS) * 32) mma_kernel(MmaPass p) {
    constexpr bool kHas = A == kFp4 ? kHasFp4 : kHasFp8;
    if constexpr(!kHas) {
        (void)p;
        __trap();
    } else {
    constexpr int R = 8 * NT, ROWS = 2 * TILES, STEP = kStepK<A>;
    constexpr bool kWords = NT == 1 && FETCH == 1;   // fragment words loaded directly
    static_assert(FETCH == 0 || NT == 1, "FETCH 1 is a decode item's");
    constexpr bool kFp8Fold = A == kFp8;
    static_assert(!(A == kFp4 && GATEUP) || TILES % 2 == 0, "FP4's intermediate blocks of 16 rows are tile pairs");
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int routes[R];
    const int u = blockIdx.y;
    if(u >= *p.item_count) return;
    for(int j = threadIdx.x; j < R; j += blockDim.x) routes[j] = p.item_routes[static_cast<long long>(u) * R + j];
    __syncthreads();
    int active = 0;
#pragma unroll
    for(int n = 0; n < NT; ++n)
        if(routes[8 * n] >= 0) active = n + 1;
    const Layout& L = p.layout;
    const std::uint8_t* bundle = p.store + static_cast<long long>(p.item_slot[u]) * L.bytes;
    const int H = L.hidden, I = L.inter;
    const int K = GATEUP ? H : I, M = GATEUP ? I : H;
    const std::uint8_t* wcodes = bundle + (GATEUP ? L.gate_up : L.down);
    const std::uint8_t* wscales = bundle + (GATEUP ? L.gate_up_scale : L.down_scale);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    // The K parts: a decode block's warps are groups x parts (warp = group * P + part), a prompt warp runs every part.
    constexpr bool kSplitWarps = NT == 1;
    const int P = k_parts(K), G = kSplitWarps ? decode_groups(WARPS, P) : WARPS;
    const int part = kSplitWarps ? warp % P : 0, grp = kSplitWarps ? warp / P : warp;
    constexpr int SB = A == kFp4 ? 1 : 2;   // steps a 128-k block
    auto part_step = [&](int q) { return q * (K / kGroup) / P * SB; };
    const int r0 = (blockIdx.x * G + grp) * (GATEUP ? 8 : 16) * TILES;
    const bool live = r0 < M;
    int rows[ROWS];
#pragma unroll
    for(int t = 0; t < TILES; ++t) {
        rows[2 * t] = GATEUP ? r0 + 8 * t + g : r0 + 16 * t + g;            // A row g
        rows[2 * t + 1] = GATEUP ? I + r0 + 8 * t + g : r0 + 16 * t + 8 + g;  // A row g + 8
    }
    const int steps = K / STEP, rowb = K / kCodesPerByte<A>, rows_s = K / 16;
    const int sb = kSplitWarps ? part_step(part) : 0, se = kSplitWarps ? part_step(part + 1) : steps;
    // The lane's 16 bytes of a step of each row, and (FP4) the scale words of the row it gives the MMA (A row g + 8
    // (t4 & 1)).
    auto fetch = [&](WeightRing<DEPTH, ROWS>& ring, int slot, int st) {
#pragma unroll
        for(int r = 0; r < ROWS; ++r) {
            const std::uint8_t* at = wcodes + static_cast<long long>(rows[r]) * rowb + 64 * st;
            if constexpr(kWords) {
                const auto* w = reinterpret_cast<const std::uint32_t*>(at + 4 * t4);
                ring.w[slot][r] = make_uint4(__ldg(w), __ldg(w + 4), __ldg(w + 8), __ldg(w + 12));
            } else {
                ring.w[slot][r] = __ldg(reinterpret_cast<const uint4*>(at + 16 * t4));
            }
        }
        if constexpr(A == kFp4) {
#pragma unroll
            for(int t = 0; t < TILES; ++t)
                ring.s[slot][t] = __ldg(reinterpret_cast<const uint2*>(
                    wscales + static_cast<long long>(rows[2 * t + (t4 & 1)]) * rows_s + 8 * st));
        }
    };
    float acc[TILES][NT][4], part_sum[kFp8Fold ? TILES : 1][kFp8Fold ? NT : 1][4];
    float total[kSplitWarps ? 1 : TILES][kSplitWarps ? 1 : NT][4];   // prompt items: the parts added so far
#pragma unroll
    for(int t = 0; t < TILES; ++t)
#pragma unroll
        for(int n = 0; n < NT; ++n)
#pragma unroll
            for(int i = 0; i < 4; ++i) acc[t][n][i] = 0.0f;
    if constexpr(kFp8Fold) {   // FP8: each 128-k block's sum from zero
#pragma unroll
        for(int t = 0; t < TILES; ++t)
#pragma unroll
            for(int n = 0; n < NT; ++n)
#pragma unroll
                for(int i = 0; i < 4; ++i) part_sum[t][n][i] = 0.0f;
    }

    constexpr bool kDirect = NT == 1;
    // Decode items: the route's codes (and FP4 scales) straight from global memory, as deep as the weights.
    const std::uint8_t* bcodes = nullptr;
    const std::uint8_t* bscales = nullptr;
    if constexpr(kDirect) {
        const int r = routes[g];
        if(r >= 0) {
            bcodes = p.codes + static_cast<long long>(r) * rowb + (kWords ? 4 : 16) * t4;
            if constexpr(A == kFp4) bscales = static_cast<const std::uint8_t*>(p.scales) + static_cast<long long>(r) * rows_s;
        }
    }
    const int cw = code_row<A>(p.chunk), sw = scale_row<A>(p.chunk), hb = half_bytes<A>(p.chunk);
    WeightRing<DEPTH, ROWS> ring;
    uint4 bring[kDirect ? DEPTH : 1];
    uint2 sring[kDirect && A == kFp4 ? DEPTH : 1];
    auto fetch_b = [&](int slot, int st) {
        if constexpr(kDirect) {
            if(bcodes) {
                if constexpr(kWords) {
                    const auto* w = reinterpret_cast<const std::uint32_t*>(bcodes + 64 * st);
                    bring[slot] = make_uint4(__ldg(w), __ldg(w + 4), __ldg(w + 8), __ldg(w + 12));
                } else {
                    bring[slot] = __ldg(reinterpret_cast<const uint4*>(bcodes + 64 * st));
                }
                if constexpr(A == kFp4) sring[slot] = __ldg(reinterpret_cast<const uint2*>(bscales + 8 * st));
            } else {
                bring[slot] = make_uint4(0u, 0u, 0u, 0u);
                if constexpr(A == kFp4) sring[slot] = make_uint2(0u, 0u);
            }
        }
    };
    // FP8: the end of a 128-k block (two steps): each output's block sum folded into its running sum by the activation's
    // scale times the weight's, in block order.
    auto fold = [&](int kb, const std::uint8_t* half, int kk) {
        if constexpr(kFp8Fold) {
            float ws[TILES][2];
#pragma unroll
            for(int t = 0; t < TILES; ++t)
#pragma unroll
                for(int h = 0; h < 2; ++h) ws[t][h] = fp8_block_scale<GATEUP>(wscales, L, rows[2 * t + h], kb);
#pragma unroll
            for(int n = 0; n < NT; ++n) {
                if(n >= active) break;
                float as[2];
#pragma unroll
                for(int c = 0; c < 2; ++c) {
                    const int j = 8 * n + 2 * t4 + c, r = routes[j];
                    if constexpr(kDirect)
                        as[c] = r >= 0 ? __ldg(static_cast<const float*>(p.scales) + static_cast<long long>(r) * (K / kGroup) + kb)
                                       : 0.0f;
                    else
                        as[c] = *reinterpret_cast<const float*>(half + 64 * cw + j * sw + 4 * (kk / kGroup));
                }
#pragma unroll
                for(int t = 0; t < TILES; ++t)
#pragma unroll
                    for(int i = 0; i < 4; ++i) {
                        acc[t][n][i] = __fmaf_rn(part_sum[t][n][i], __fmul_rn(as[i & 1], ws[t][i >> 1]),
                                                 acc[t][n][i]);
                        part_sum[t][n][i] = 0.0f;
                    }
            }
        }
    };
    if(live) {
#pragma unroll
        for(int d = 0; d < DEPTH; ++d)
            if(sb + d < se) {
                fetch(ring, d, sb + d);
                fetch_b(d, sb + d);
            }
    }
    int done = 0, part_end = part_step(1);   // prompt items: the parts added into total, the end of the current one
    if constexpr(!kDirect) stage<A>(p, routes, 8 * active, K, 0, K < p.chunk ? K : p.chunk, smem);
    for(int s0 = sb; s0 < se; s0 += DEPTH) {
#pragma unroll
        for(int d = 0; d < DEPTH; ++d) {
            const int st = s0 + d;
            if(st >= se) break;
            const int k = STEP * st, kk = k % (kDirect ? K : p.chunk);
            if constexpr(!kDirect) {
                if(kk == 0) {   // chunk k / chunk landed; every warp is done with the other half
                    kc::cp_async_wait_all();
                    __syncthreads();
                    const int next = k + p.chunk;
                    if(next < K)
                        stage<A>(p, routes, 8 * active, K, next, K - next < p.chunk ? K - next : p.chunk,
                                 smem + ((next / p.chunk) & 1) * hb);
                }
            }
            if(!live) continue;
            uint32_t a[ROWS][4];
#pragma unroll
            for(int r = 0; r < ROWS; ++r) {
                if constexpr(kWords) {
                    a[r][0] = ring.w[d][r].x, a[r][1] = ring.w[d][r].y, a[r][2] = ring.w[d][r].z, a[r][3] = ring.w[d][r].w;
                } else {
                    kc::quad_transpose(ring.w[d][r], t4, lane, a[r]);
                }
            }
            uint2 sa[TILES];
            if constexpr(A == kFp4) {
#pragma unroll
                for(int t = 0; t < TILES; ++t) sa[t] = ring.s[d][t];
            }
            uint32_t bd[4] = {0u, 0u, 0u, 0u};
            uint2 sbd = make_uint2(0u, 0u);
            if constexpr(kDirect) {
                if constexpr(kWords) bd[0] = bring[d].x, bd[1] = bring[d].y, bd[2] = bring[d].z, bd[3] = bring[d].w;
                else kc::quad_transpose(bring[d], t4, lane, bd);
                if constexpr(A == kFp4) sbd = sring[d];
            }
            if(st + DEPTH < se) {
                fetch(ring, d, st + DEPTH);
                fetch_b(d, st + DEPTH);
            }
            const std::uint8_t* half = kDirect ? nullptr : smem + ((k / p.chunk) & 1) * hb;
#pragma unroll
            for(int n = 0; n < NT; ++n) {
                if(n >= active) break;
                uint32_t b[4];
                uint2 sb = make_uint2(0u, 0u);
                if constexpr(kDirect) {
#pragma unroll
                    for(int i = 0; i < 4; ++i) b[i] = bd[i];
                    sb = sbd;
                } else {
                    const std::uint8_t* row = half + (8 * n + g) * cw + kk / kCodesPerByte<A>;
#pragma unroll
                    for(int i = 0; i < 4; ++i) b[i] = *reinterpret_cast<const uint32_t*>(row + 16 * i + 4 * t4);
                    if constexpr(A == kFp4) sb = *reinterpret_cast<const uint2*>(half + 64 * cw + (8 * n + g) * sw + kk / 16);
                }
#pragma unroll
                for(int m = 0; m < 2; ++m)
#pragma unroll
                    for(int t = 0; t < TILES; ++t) {
                        if constexpr(A == kFp4)
                            kc::mma_nvf4_m16n8k64(acc[t][n], a[2 * t][2 * m], a[2 * t + 1][2 * m], a[2 * t][2 * m + 1],
                                                  a[2 * t + 1][2 * m + 1], b[2 * m], b[2 * m + 1],
                                                  m ? sa[t].y : sa[t].x, m ? sb.y : sb.x);
                        else
                            kc::mma_e4m3_m16n8k32_fast(part_sum[t][n], a[2 * t][2 * m], a[2 * t + 1][2 * m],
                                                       a[2 * t][2 * m + 1], a[2 * t + 1][2 * m + 1], b[2 * m],
                                                       b[2 * m + 1]);
                    }
            }
            if constexpr(kFp8Fold) {
                if(st & 1) fold(st / 2, half, kk);   // a 128-k block is two steps (K a multiple of 128)
            }
            if constexpr(!kSplitWarps) {
                if(st + 1 == part_end) {   // a part ends: its chain added into total, in part order; the next from zero
#pragma unroll
                    for(int t = 0; t < TILES; ++t)
#pragma unroll
                        for(int n = 0; n < NT; ++n)
#pragma unroll
                            for(int i = 0; i < 4; ++i) {
                                total[t][n][i] = done ? __fadd_rn(total[t][n][i], acc[t][n][i]) : acc[t][n][i];
                                acc[t][n][i] = 0.0f;
                            }
                    ++done;
                    part_end = part_step(done + 1);
                }
            }
        }
    }
    if constexpr(kSplitWarps) {
        // Decode items: the parts' sums, each warp's, added in part order by the group's part-0 warp.
        if(P > 1) {
            float* parts = reinterpret_cast<float*>(smem);   // [G][P - 1][TILES * 4][32]
            if(part > 0) {
#pragma unroll
                for(int t = 0; t < TILES; ++t)
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                        parts[((grp * (P - 1) + part - 1) * TILES * 4 + 4 * t + i) * 32 + lane] = acc[t][0][i];
            }
            __syncthreads();
            if(part > 0) return;
            for(int q = 1; q < P; ++q)
#pragma unroll
                for(int t = 0; t < TILES; ++t)
#pragma unroll
                    for(int i = 0; i < 4; ++i)
                        acc[t][0][i] = __fadd_rn(acc[t][0][i],
                                                 parts[((grp * (P - 1) + q - 1) * TILES * 4 + 4 * t + i) * 32 + lane]);
        }
    } else {
#pragma unroll
        for(int t = 0; t < TILES; ++t)
#pragma unroll
            for(int n = 0; n < NT; ++n)
#pragma unroll
                for(int i = 0; i < 4; ++i) acc[t][n][i] = total[t][n][i];
    }
    if(!live) return;
    if constexpr(GATEUP) {
        // c0, c1: gate row rows[2t] (inter r0 + 8t + g) for routes 2 t4, 2 t4 + 1; c2, c3: its up row, same routes.
        float fg[TILES], fu[TILES];
        if constexpr(A == kFp4) {
            const float* inverse = reinterpret_cast<const float*>(bundle + L.gate_up_inverse);
            const float in_rcp = input_scale(bundle, L, 2);
#pragma unroll
            for(int t = 0; t < TILES; ++t) {
                fg[t] = __fmul_rn(inverse[rows[2 * t]], in_rcp);
                fu[t] = __fmul_rn(inverse[rows[2 * t + 1]], in_rcp);
            }
        } else {
            const bool channel = L.format == Format::fp8_channel;
#pragma unroll
            for(int t = 0; t < TILES; ++t) {
                fg[t] = channel ? reinterpret_cast<const float*>(wscales)[rows[2 * t]] : 1.0f;
                fu[t] = channel ? reinterpret_cast<const float*>(wscales)[rows[2 * t + 1]] : 1.0f;
            }
        }
#pragma unroll
        for(int n = 0; n < NT; ++n) {
            if(n >= active) break;
#pragma unroll
            for(int c = 0; c < 2; ++c) {
                float hv[TILES];
#pragma unroll
                for(int t = 0; t < TILES; ++t)
                    hv[t] = L.activation == 1
                        ? kc::gelu_tanh_mul(__fmul_rn(acc[t][n][c], fg[t]), __fmul_rn(acc[t][n][2 + c], fu[t]))
                        : kc::swiglu_declared(__fmul_rn(acc[t][n][c], fg[t]), __fmul_rn(acc[t][n][2 + c], fu[t]));
                const int r = routes[8 * n + 2 * t4 + c];
                if constexpr(A == kFp8) {   // h in BF16: the group quantizer reads it next
                    bf16* h = reinterpret_cast<bf16*>(p.out);
                    if(r >= 0) {
#pragma unroll
                        for(int t = 0; t < TILES; ++t)
                            h[static_cast<long long>(r) * I + r0 + 8 * t + g] = __float2bfloat16_rn(hv[t]);
                    }
                    continue;
                } else {
                    std::uint8_t* hc = reinterpret_cast<std::uint8_t*>(p.out);
                    const float g_dn = input_scale(bundle, L, 1);
#pragma unroll
                    for(int q = 0; q < TILES / 2; ++q) {
                        // tiles 2q and 2q + 1: inter rows r0 + 16q .. + 15, one block of the down's input
                        float amax = kc::abs_max(kc::abs_max(0.0f, hv[2 * q]), hv[2 * q + 1]);
#pragma unroll
                        for(int o = 4; o < 32; o <<= 1) amax = kc::abs_max(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                        uint32_t code2[2], s;
                        kc::nvfp4_quantize_pair(hv[2 * q], hv[2 * q + 1], amax, g_dn, code2, s);
                        // inter rows i (lane g) and i + 1 (lane g + 1, lane ^ 4) of each tile: one byte, low nibble first
#pragma unroll
                        for(int h = 0; h < 2; ++h) {
                            const uint32_t other = __shfl_xor_sync(0xffffffffu, code2[h], 4);
                            if(r >= 0 && !(g & 1))
                                hc[static_cast<long long>(r) * (I / 2) + (r0 + 16 * q + 8 * h + g) / 2] =
                                    static_cast<std::uint8_t>(code2[h] | (other << 4));
                        }
                        if(r >= 0 && g == 0)
                            p.out_scales[static_cast<long long>(r) * (I / 16) + (r0 + 16 * q) / 16] =
                                static_cast<std::uint8_t>(s);
                    }
                }
            }
        }
    } else {
        float* y = reinterpret_cast<float*>(p.out);
        float f[ROWS];
        if constexpr(A == kFp4) {
            const float* inverse = reinterpret_cast<const float*>(bundle + L.down_inverse);
            const float in_rcp = input_scale(bundle, L, 3);
#pragma unroll
            for(int r = 0; r < ROWS; ++r) f[r] = __fmul_rn(inverse[rows[r]], in_rcp);
        } else {
            const bool channel = L.format == Format::fp8_channel;
#pragma unroll
            for(int r = 0; r < ROWS; ++r) f[r] = channel ? reinterpret_cast<const float*>(wscales)[rows[r]] : 1.0f;
        }
#pragma unroll
        for(int n = 0; n < NT; ++n) {
            if(n >= active) break;
#pragma unroll
            for(int t = 0; t < TILES; ++t)
#pragma unroll
                for(int i = 0; i < 4; ++i) {
                    const int r = routes[8 * n + 2 * t4 + (i & 1)];
                    const int row = rows[2 * t + (i >> 1)];
                    if(r >= 0) y[static_cast<long long>(r) * H + row] = __fmul_rn(acc[t][n][i], f[2 * t + (i >> 1)]);
                }
        }
    }
    }
}

__global__ void probe_kernel(int* found) { found[0] = kHasFp4 ? 1 : 0, found[1] = kHasFp8 ? 1 : 0; }

// ----------------------------------------------------------------------------------------------- launches

template <int A, int NT, bool GATEUP, int WARPS, int TILES, int DEPTH, int CHUNK, int FETCH = 0>
cudaError_t launch(const MmaPass& base, const Items& items, cudaStream_t stream) {
    MmaPass p = base;
    const int K = GATEUP ? p.layout.hidden : p.layout.inter, M = GATEUP ? p.layout.inter : p.layout.hidden;
    p.chunk = NT == 1 ? K : (K < CHUNK ? K : CHUNK);
    // Decode items: groups x parts warps a block (k_parts), the parts' sums through shared memory; prompt items: WARPS.
    const int parts = k_parts(K), groups = NT == 1 ? decode_groups(WARPS, parts) : WARPS;
    const int threads = NT == 1 ? groups * parts * 32 : WARPS * 32;
    const int rows_per_block = groups * (GATEUP ? 8 : 16) * TILES;
    const dim3 grid((M + rows_per_block - 1) / rows_per_block, items.capacity);
    const size_t bytes = NT == 1 ? static_cast<size_t>(groups) * (parts - 1) * TILES * 4 * 32 * sizeof(float)
                                 : 2 * static_cast<size_t>(half_bytes<A>(p.chunk));
    if(bytes > 48 * 1024) {
        static kc::PerDevice<cudaError_t> optin;
        if(const cudaError_t e =
               kc::shared_memory_optin(optin, mma_kernel<A, NT, TILES, WARPS, DEPTH, GATEUP, FETCH>, int(bytes));
           e != cudaSuccess)
            return e;
    }
    mma_kernel<A, NT, TILES, WARPS, DEPTH, GATEUP, FETCH><<<grid, threads, bytes, stream>>>(p);
    return cudaGetLastError();
}

#ifdef CRIT_SWEEP_BUILD
constexpr bool kMmaTuning = CRIT_CONFIG_COUNT_fp4_decode > 1 || CRIT_CONFIG_COUNT_fp4_prompt > 1 ||
                            CRIT_CONFIG_COUNT_fp8_decode > 1 || CRIT_CONFIG_COUNT_fp8_prompt > 1;
constexpr bool kBuiltFp4Decode = !kMmaTuning || CRIT_CONFIG_COUNT_fp4_decode > 1;
constexpr bool kBuiltFp4Prompt = !kMmaTuning || CRIT_CONFIG_COUNT_fp4_prompt > 1;
constexpr bool kBuiltFp8Decode = !kMmaTuning || CRIT_CONFIG_COUNT_fp8_decode > 1;
constexpr bool kBuiltFp8Prompt = !kMmaTuning || CRIT_CONFIG_COUNT_fp8_prompt > 1;
#else
constexpr bool kBuiltFp4Decode = true, kBuiltFp4Prompt = true, kBuiltFp8Decode = true, kBuiltFp8Prompt = true;
#endif

template <bool BUILT, int A, int NT, bool GATEUP, int WARPS, int TILES, int DEPTH, int CHUNK, int FETCH = 0>
cudaError_t built(const MmaPass& p, const Items& items, cudaStream_t stream) {
    if constexpr(!BUILT) {
        (void)p, (void)items, (void)stream;
        return cudaErrorNotSupported;
    } else {
        return launch<A, NT, GATEUP, WARPS, TILES, DEPTH, CHUNK, FETCH>(p, items, stream);
    }
}

// FP4's gate/up tiles come in pairs (the intermediate's 16-row blocks): one tile a warp runs two.
template <int A, bool GATEUP, int TILES>
constexpr int tiles_of() { return A == kFp4 && GATEUP && TILES == 1 ? 2 : TILES; }

template <bool GATEUP>
cudaError_t dispatch_fp4(const MmaPass& p, const Items& items, int tokens, cudaStream_t stream) {
    const int H = p.layout.hidden, I = p.layout.inter, rows = tokens > 0 ? tokens : 0;
    switch(items.width) {
        case 8:
            switch(crit_lookup_fp4_decode(GATEUP ? 1 : 0, H, I, rows)) {
#define CRIT_GX_FP4D(index, warps, tiles, depth, fetch) \
    case index: return built<kBuiltFp4Decode, kFp4, 1, GATEUP, warps, tiles_of<kFp4, GATEUP, tiles>(), depth, 128, fetch>(p, items, stream);
                CRIT_CONFIGS_fp4_decode(CRIT_GX_FP4D)
#undef CRIT_GX_FP4D
                default: return cudaErrorInvalidConfiguration;
            }
        case 32:
        case 64:
            switch(crit_lookup_fp4_prompt(GATEUP ? 1 : 0, H, I, rows)) {
#define CRIT_GX_FP4P(index, warps, tiles, depth, chunk)                                                                   \
    case index:                                                                                                           \
        return items.width == 32                                                                                          \
            ? built<kBuiltFp4Prompt, kFp4, 4, GATEUP, warps, tiles_of<kFp4, GATEUP, tiles>(), depth, chunk>(p, items, stream) \
            : built<kBuiltFp4Prompt, kFp4, 8, GATEUP, warps, tiles_of<kFp4, GATEUP, tiles>(), depth, chunk>(p, items, stream);
                CRIT_CONFIGS_fp4_prompt(CRIT_GX_FP4P)
#undef CRIT_GX_FP4P
                default: return cudaErrorInvalidConfiguration;
            }
        default:
            return cudaErrorInvalidValue;
    }
}

template <bool GATEUP>
cudaError_t dispatch_fp8(const MmaPass& p, const Items& items, int tokens, cudaStream_t stream) {
    const int f = static_cast<int>(p.layout.format), H = p.layout.hidden, I = p.layout.inter;
    const int rows = tokens > 0 ? tokens : 0;
    switch(items.width) {
        case 8:
            switch(crit_lookup_fp8_decode(f, GATEUP ? 1 : 0, H, I, rows)) {
#define CRIT_GX_FP8D(index, warps, tiles, depth, fetch) \
    case index: return built<kBuiltFp8Decode, kFp8, 1, GATEUP, warps, tiles, depth, 128, fetch>(p, items, stream);
                CRIT_CONFIGS_fp8_decode(CRIT_GX_FP8D)
#undef CRIT_GX_FP8D
                default: return cudaErrorInvalidConfiguration;
            }
        case 32:
        case 64:
            switch(crit_lookup_fp8_prompt(f, GATEUP ? 1 : 0, H, I, rows)) {
#define CRIT_GX_FP8P(index, warps, tiles, depth, chunk)                                                    \
    case index:                                                                                            \
        return items.width == 32 ? built<kBuiltFp8Prompt, kFp8, 4, GATEUP, warps, tiles, depth, chunk>(p, items, stream) \
                                 : built<kBuiltFp8Prompt, kFp8, 8, GATEUP, warps, tiles, depth, chunk>(p, items, stream);
                CRIT_CONFIGS_fp8_prompt(CRIT_GX_FP8P)
#undef CRIT_GX_FP8P
                default: return cudaErrorInvalidConfiguration;
            }
        default:
            return cudaErrorInvalidValue;
    }
}

bool common_layout(const Layout& L) {
    return L.bytes > 0 && L.bytes % 16 == 0 && L.hidden % 128 == 0 && L.inter % 128 == 0 && L.hidden >= 128 &&
           L.inter >= 128 && L.gate_up >= 0 && L.down >= 0 && L.gate_up_scale >= 0 && L.down_scale >= 0 &&
           (L.activation == 0 || L.activation == 1) && L.gate_up % 16 == 0 && L.down % 16 == 0;
}
bool fp4_layout(const Layout& L) {
    return L.format == Format::nvfp4 && common_layout(L) && L.gate_up_inverse >= 0 && L.down_inverse >= 0 &&
           L.input >= 0 && L.gate_up_scale % 8 == 0 && L.down_scale % 8 == 0;
}
bool fp8_layout(const Layout& L) {
    return (L.format == Format::fp8_channel || L.format == Format::fp8_block) && common_layout(L) &&
           L.gate_up_scale % 4 == 0 && L.down_scale % 4 == 0;
}
bool items_ok(const Items& it) {
    return it.slot && it.routes && it.count && (it.width == 8 || it.width == 32 || it.width == 64) && it.capacity >= 0;
}

}  // namespace

// found[0]: FP4 MMAs, found[1]: FP8 MMAs in this build's code for the current device.
static int probe(cudaStream_t stream) {
    int* found = nullptr;
    int host[2] = {0, 0};
    if(cudaMallocAsync(&found, 2 * sizeof(int), stream) != cudaSuccess) return 0;
    probe_kernel<<<1, 1, 0, stream>>>(found);
    const bool ok = cudaMemcpyAsync(host, found, 2 * sizeof(int), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                    cudaStreamSynchronize(stream) == cudaSuccess;
    cudaFreeAsync(found, stream);
    return ok ? host[0] | (host[1] << 1) : 0;
}

bool fp4_available(cudaStream_t stream) { return (probe(stream) & 1) != 0; }
bool fp8_available(cudaStream_t stream) { return (probe(stream) & 2) != 0; }

cudaError_t enqueue_fp4_inputs(const bf16* x, int top, const std::uint8_t* store, const Layout& layout,
                               const Items& items, std::uint8_t* codes, std::uint8_t* scales, cudaStream_t stream) {
    if(!x || top < 1 || !store || !codes || !scales || !fp4_layout(layout) || !items_ok(items))
        return cudaErrorInvalidValue;
    if(items.capacity == 0) return cudaSuccess;
    constexpr int threads = 128;
    const int per = items.width * (layout.hidden / 16);
    const dim3 grid(items.capacity, (per + threads - 1) / threads);
    fp4_inputs_kernel<<<grid, threads, 0, stream>>>(x, top, store, layout, items, codes, scales);
    return cudaGetLastError();
}

cudaError_t enqueue_fp4_gate_up(const std::uint8_t* codes, const std::uint8_t* scales, const std::uint8_t* store,
                                const Layout& layout, const Items& items, std::uint8_t* h_codes,
                                std::uint8_t* h_scales, cudaStream_t stream, int tokens) {
    if(!codes || !scales || !store || !h_codes || !h_scales || !fp4_layout(layout) || !items_ok(items))
        return cudaErrorInvalidValue;
    if(items.capacity == 0) return cudaSuccess;
    const MmaPass p{store, layout, items.slot, items.routes, items.count, codes, scales, h_codes, h_scales, 0};
    return dispatch_fp4<true>(p, items, tokens, stream);
}

cudaError_t enqueue_fp4_down(const std::uint8_t* h_codes, const std::uint8_t* h_scales, const std::uint8_t* store,
                             const Layout& layout, const Items& items, float* y, cudaStream_t stream, int tokens) {
    if(!h_codes || !h_scales || !store || !y || !fp4_layout(layout) || !items_ok(items)) return cudaErrorInvalidValue;
    if(items.capacity == 0) return cudaSuccess;
    const MmaPass p{store, layout, items.slot, items.routes, items.count, h_codes, h_scales, y, nullptr, 0};
    return dispatch_fp4<false>(p, items, tokens, stream);
}

cudaError_t enqueue_fp8_inputs(const bf16* rows, int div, int K, const Items& items, std::uint8_t* codes, float* scales,
                               cudaStream_t stream) {
    if(!rows || div < 1 || K < 128 || K % 128 || !codes || !scales || !items_ok(items)) return cudaErrorInvalidValue;
    if(items.capacity == 0) return cudaSuccess;
    constexpr int threads = 256;   // eight groups a block
    const int warps = items.width * (K / kGroup);
    const dim3 grid(items.capacity, (warps * 32 + threads - 1) / threads);
    fp8_inputs_kernel<<<grid, threads, 0, stream>>>(rows, div, K, items, codes, scales);
    return cudaGetLastError();
}

cudaError_t enqueue_fp8_gate_up(const std::uint8_t* codes, const float* scales, const std::uint8_t* store,
                                const Layout& layout, const Items& items, bf16* h, cudaStream_t stream, int tokens) {
    if(!codes || !scales || !store || !h || !fp8_layout(layout) || !items_ok(items)) return cudaErrorInvalidValue;
    if(items.capacity == 0) return cudaSuccess;
    const MmaPass p{store, layout, items.slot, items.routes, items.count, codes, scales, h, nullptr, 0};
    return dispatch_fp8<true>(p, items, tokens, stream);
}

cudaError_t enqueue_fp8_down(const std::uint8_t* h_codes, const float* h_scales, const std::uint8_t* store,
                             const Layout& layout, const Items& items, float* y, cudaStream_t stream, int tokens) {
    if(!h_codes || !h_scales || !store || !y || !fp8_layout(layout) || !items_ok(items)) return cudaErrorInvalidValue;
    if(items.capacity == 0) return cudaSuccess;
    const MmaPass p{store, layout, items.slot, items.routes, items.count, h_codes, h_scales, y, nullptr, 0};
    return dispatch_fp8<false>(p, items, tokens, stream);
}

}  // namespace crit::techniques::grouped_experts
