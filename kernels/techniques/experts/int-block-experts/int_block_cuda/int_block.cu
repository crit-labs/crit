// Integer block expert projections (int_block.h). The device arithmetic is crit's host_int_device.cuh, operation for
// operation: exact integer block sums, then one rounded product and one FMA per block on lane b % 16, the halving
// tree, post. Explicit round-to-nearest intrinsics keep the compiler from contracting anything. The code decoder, the
// block MMAs, the step, the tree's fold, the E4M3 decoder and SwiGLU are crit's kernel core blocks
// (blockscaled_mma.cuh, mma.cuh, reduce.cuh, formats.cuh, epilogue.cuh), shared by every structure here.
#include "int_block.h"
#include "crit_tuning.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <mutex>
#include <vector>
#include <cuda_fp8.h>

#include "crit_kcore/blockscaled_mma.cuh"
#include "crit_kcore/blockscaled_order.cuh"
#include "crit_kcore/blockscaled_stage.cuh"
#include "crit_kcore/declared_rows.cuh"
#include "crit_kcore/epilogue.cuh"
#include "crit_kcore/formats.cuh"
#include "crit_kcore/load.cuh"
#include "crit_kcore/tile.cuh"

namespace crit::techniques::int_block {
namespace {

namespace kc = crit::kcore;
namespace bs = crit::kcore::blockscaled;

// E4M3 scale values, decoded once per launch into shared memory by e4m3_table.
__device__ inline float e4m3(unsigned byte) { return kc::e4m3_value(byte); }
__device__ inline float e2m1(unsigned code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (code & 8u) ? -magnitude[code & 7u] : magnitude[code & 7u];
}
__device__ inline unsigned e2m1_encode(float x) {
    const float m = fminf(fabsf(x), 6.0f);
    const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    unsigned code = 0;
    while(code < 7 && (m > mids[code] || (m == mids[code] && (code % 2) == 1))) ++code;
    return code | (x < 0.0f && code ? 8u : 0u);
}
__device__ inline unsigned e4m3_encode(float x) {
    const float m = fminf(fabsf(x), 448.0f);
    unsigned best = 0;
    for(unsigned i = 1; i < 0x7Fu; ++i) {
        const float d = fabsf(m - e4m3(i)), bd = fabsf(m - e4m3(best));
        if(d < bd || (d == bd && i % 2 == 0 && best % 2 == 1)) best = i;
    }
    return best | (x < 0.0f && best ? 0x80u : 0u);
}
// ---------------------------------------------------------------------------------------------------- decode rows

// The decode rows' knobs (kernel.json, structure rows; crit_tuning.h picks them per matrix shape and row count):
//   THREADS  threads a block (a half-warp per output row)
//   ROWS     output rows a half-warp owns, each token's activation block read once for all of them
//   STREAM   read codes with L1::no_allocate (they are read once)
//   UNROLL   blocks a lane loads before it computes them (loads in flight)
// None changes an output's operations or their order: lane b % 16 takes its blocks in ascending order, the halving
// tree, post. Every configuration gives the declared order's bits.
__device__ inline void e4m3_table(float* table) {
    for(int i = int(threadIdx.x); i < 256; i += int(blockDim.x)) table[i] = e4m3(unsigned(i));
    __syncthreads();
}

// The rows are the kernel core's declared integer rows (declared_rows.cuh's int_rows_kernel, on multiply.cuh's
// DeclaredInt): the operations this file's rows kernel ran, so the same bits.

// ---------------------------------------------------------------------------------------------------- prompt GEMM

// The prompt GEMM's knobs (kernel.json, structure gemm; crit_tuning.h picks them per matrix shape and token count):
//   WARPS   warps a block
//   TILES   m16n8k16 tiles a warp runs side by side on one A fragment (8 weight rows each)
//   TOKZ    16-token tiles a block takes along grid.z, one after another
// None changes an output's operations or their order (each block's int8 sums exact, lane by lane in kLaneOrder).

// A warp computes 16 tokens x 8 TILES weight rows (TILES m16n8k16 tiles sharing the A fragment) per pass over the
// blocks,
// lane by lane in kLaneOrder16: lane j's blocks j, j + 16, ... in order, each block's exact int8 sums from the tensor
// cores and the FP32 step on the CUDA cores, then the lane folded into the tree.
template <int WARPS, int TILES>
__global__ void gemm_kernel(const Matrix* matrices, const GemmJob* jobs, Acts acts, float* out, int out_ld) {
    constexpr int kTiles = TILES, kGemmRows = 8 * kTiles * WARPS;
    __shared__ float scale_of[256];
    e4m3_table(scale_of);
    const GemmJob job = jobs[blockIdx.y];
    const Matrix m = matrices[job.matrix];
    const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    const int g = lane / 4, tig = lane % 4;
    const int base = int(blockIdx.x) * kGemmRows + warp * 8 * kTiles;
    if(base >= m.rows) return;
    const int blocks = m.cols / 16, ratio = acts.block / 16;
    const bs::NibbleTable table = bs::nibble_table(m.int4 != 0);
    const std::uint8_t* wcodes[kTiles];
    const std::uint8_t* wscale[kTiles][2];
    bool wlive[kTiles], olive[kTiles][2];
#pragma unroll
    for(int n = 0; n < kTiles; ++n) {
        const int wrow = base + 8 * n + g;   // the B fragment's row
        wlive[n] = wrow < m.rows;
        wcodes[n] = m.codes + std::size_t(wlive[n] ? wrow : 0) * std::size_t(m.cols / 2);
#pragma unroll
        for(int c = 0; c < 2; ++c) {
            const int orow = base + 8 * n + 2 * tig + c;   // this thread's output rows
            olive[n][c] = orow < m.rows;
            wscale[n][c] = m.scales + std::size_t(olive[n][c] ? orow : 0) * std::size_t(blocks);
        }
    }
    for(int tile_t = 16 * int(blockIdx.z); tile_t < job.count; tile_t += 16 * int(gridDim.z)) {
        const int t0 = tile_t + g, t1 = t0 + 8;
        const bool live0 = t0 < job.count, live1 = t1 < job.count;
        const std::int8_t* x0 = acts.values + std::size_t(job.first + (live0 ? t0 : 0)) * acts.ld;
        const std::int8_t* x1 = acts.values + std::size_t(job.first + (live1 ? t1 : 0)) * acts.ld;
        const float* xs0 = acts.scales + std::size_t(job.first + (live0 ? t0 : 0)) * acts.lds;
        const float* xs1 = acts.scales + std::size_t(job.first + (live1 ? t1 : 0)) * acts.lds;
        // Outputs o = (n, token half h, column c) at o = 4n + 2h + c; x[o][..]: the tree's partial sums.
        float x[4 * kTiles][4];
#pragma unroll
        for(int step = 0; step < 16; ++step) {
            const int j = kc::kLaneOrder16[step];
            float acc[4 * kTiles];
#pragma unroll
            for(int o = 0; o < 4 * kTiles; ++o) acc[o] = 0.0f;
            for(int b = j; b < blocks; b += 16) {
                const int a0 = live0 ? *reinterpret_cast<const int*>(x0 + 16 * b + 4 * tig) : 0;
                const int a1 = live1 ? *reinterpret_cast<const int*>(x1 + 16 * b + 4 * tig) : 0;
                const float as0 = xs0[b / ratio], as1 = xs1[b / ratio];
#pragma unroll
                for(int n = 0; n < kTiles; ++n) {
                    const int bw = wlive[n] ? int(bs::nibbles_to_s8x4(*reinterpret_cast<const unsigned short*>(
                                                  wcodes[n] + (16 * b + 4 * tig) / 2), table)) : 0;
                    int c0, c1, c2, c3;
                    kc::mma_s8_m16n8k16_zero(a0, a1, bw, c0, c1, c2, c3);
                    const float w0 = scale_of[wscale[n][0][b]], w1 = scale_of[wscale[n][1][b]];
                    acc[4 * n + 0] = bs::step(float(c0), w0, as0, acc[4 * n + 0]);
                    acc[4 * n + 1] = bs::step(float(c1), w1, as0, acc[4 * n + 1]);
                    acc[4 * n + 2] = bs::step(float(c2), w0, as1, acc[4 * n + 2]);
                    acc[4 * n + 3] = bs::step(float(c3), w1, as1, acc[4 * n + 3]);
                }
            }
#pragma unroll
            for(int o = 0; o < 4 * kTiles; ++o) kc::fold16(step, acc[o], x[o]);
        }
#pragma unroll
        for(int n = 0; n < kTiles; ++n)
#pragma unroll
            for(int h = 0; h < 2; ++h)
#pragma unroll
                for(int c = 0; c < 2; ++c) {
                    const int t = h ? t1 : t0;
                    if((h ? live1 : live0) && olive[n][c])
                        out[std::size_t(job.first + t) * out_ld + base + 8 * n + 2 * tig + c] =
                            __fmul_rn(x[4 * n + 2 * h + c][0], m.post);
                }
    }
}

// ---------------------------------------------------------------------------------------------------- staged GEMM

// The prompt GEMM with shared-memory staging (structure gemm-staged), from crit's kernel core blocks. A CTA computes
// BM weight rows x BN tokens. The 16-element blocks are visited lane by lane in the tree's order, G lanes and RUN
// blocks of each a visit (blockscaled_order.cuh's GroupVisits); each visit is one stage of a STAGES-deep cp.async ring
// (load.cuh): its blocks' weight codes, token values, weight scale words and token scales, all asynchronous, the
// multistage mainloop CUTLASS uses (BSD-3-Clause; the mechanism, no code). One stage ahead of the MMAs the CTA decodes
// the stage's codes to int8 (nibbles_to_s8x4) and its E4M3 scales to FP32, once a CTA; then a warp computes BM / WM
// rows x BN / WN tokens, its fragments by ldmatrix: one s8 m16n8k16 MMA a 16 x 8 tile and block onto the FP32 bias
// (block_sum_s8: the exact block sum is one subtraction), the declared step on its lane, the G lanes folded into the
// tree when their last block is done. Every output has the declared order's bits: the same as `gemm`, `rows` and the
// CPU rows, whatever the knobs.
template <int BM, int BN, int WM, int WN, int G, int RUN, int STAGES>
struct Staged {
    static_assert(STAGES >= 3, "a stage is decoded one iteration before its MMAs");
    static constexpr int kThreads = 32 * WM * WN, kWarpRows = BM / WM, kWarpTokens = BN / WN;
    static constexpr int kMT = kWarpRows / 16, kNT = kWarpTokens / 8, kOut = 4 * kMT * kNT;
    static_assert(kMT >= 1 && kNT >= 1 && kWarpRows % 16 == 0 && kWarpTokens % 8 == 0, "warp tiles of 16 x 8");
    static constexpr int kRowsPer = (BM + kThreads - 1) / kThreads, kTokensPer = (BN + kThreads - 1) / kThreads;
    // A block of a stage: weight codes [BM][8] as copied, weight values [BM][16] int8 decoded, token values [BN][16],
    // the weight scales' words [BM] as copied, the token scales [BN], the weight scales decoded [BM] (pair_slot).
    static constexpr int kCodes = 0, kA = kCodes + 8 * BM, kB = kA + 16 * BM, kWraw = kB + 16 * BN;
    static constexpr int kAs = kWraw + 4 * BM, kWs = kAs + 4 * BN, kSub = kWs + 4 * BM;
    static constexpr int kStage = G * RUN * kSub, kSmem = STAGES * kStage;
    // CTAs an SM can hold by registers (the launch bound): an output holds G running lanes and the partial sums still
    // needed (5 registers, 6 with G = 4), plus about 48 a thread of fragments and addresses.
    static constexpr int kRegs = (kOut * (G == 4 ? 6 : 5) + 48 + 7) / 8 * 8;
    static constexpr int kFit = 65536 / (kThreads * kRegs), kMinBlocks = kFit < 1 ? 1 : kFit > 4 ? 4 : kFit;
};

template <int BM, int BN, int WM, int WN, int G, int RUN, int STAGES>
__global__ void __launch_bounds__(32 * WM * WN, (Staged<BM, BN, WM, WN, G, RUN, STAGES>::kMinBlocks))
    staged_kernel(const Matrix* matrices, const GemmJob* jobs, Acts acts, float* out, int out_ld) {
    using S = Staged<BM, BN, WM, WN, G, RUN, STAGES>;
    using Visits = bs::GroupVisits<G, RUN>;
    extern __shared__ __align__(16) unsigned char smem[];
    const GemmJob job = jobs[blockIdx.y];
    const Matrix m = matrices[job.matrix];
    const int row0 = int(blockIdx.x) * BM;
    if(row0 >= m.rows || job.count <= 0) return;
    const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32, g = lane / 4, q = lane % 4;
    const int wr = (warp / WN) * S::kWarpRows, wt = (warp % WN) * S::kWarpTokens;
    const int blocks = m.cols / 16, shift = acts.block == 32 ? 1 : 0, last = job.first + job.count - 1;
    const bs::NibbleTable table = bs::nibble_table(m.int4 != 0);

    for(int tile = BN * int(blockIdx.z); tile < job.count; tile += BN * int(gridDim.z)) {
        // This thread's rows: weight rows (codes and scales: cols a multiple of 64, so a scale row is word aligned)
        // and tokens (values and scales).
        const std::uint8_t* w_codes[S::kRowsPer];
        const std::uint8_t* w_scales[S::kRowsPer];
#pragma unroll
        for(int u = 0; u < S::kRowsPer; ++u) {
            const std::size_t row = std::size_t(min(row0 + tid + u * S::kThreads, m.rows - 1));
            w_codes[u] = m.codes + row * std::size_t(m.cols / 2);
            w_scales[u] = m.scales + row * std::size_t(blocks);
        }
        const std::int8_t* t_values[S::kTokensPer];
        const float* t_scales[S::kTokensPer];
#pragma unroll
        for(int u = 0; u < S::kTokensPer; ++u) {
            const std::size_t t = std::size_t(min(job.first + tile + tid + u * S::kThreads, last));
            t_values[u] = acts.values + t * std::size_t(acts.ld);
            t_scales[u] = acts.scales + t * std::size_t(acts.lds);
        }
        // A visit's copies into stage `slot` (a lane without a block copies the first lane's: never used).
        auto issue = [&](int slot, const Visits& it) {
#pragma unroll
            for(int i = 0; i < G * RUN; ++i) {
                const int s = i % G, r = i / G;
                const int b = it.block(s, r) >= 0 ? it.block(s, r) : it.block(0);
                unsigned char* sub = smem + slot * S::kStage + i * S::kSub;
#pragma unroll
                for(int u = 0; u < S::kRowsPer; ++u) {
                    const int r = tid + u * S::kThreads;
                    if(BM % S::kThreads == 0 || r < BM) {
                        kc::cp_async_ca<8>(sub + S::kCodes + 8 * r, w_codes[u] + 8 * b);
                        kc::cp_async_ca<4>(sub + S::kWraw + 4 * r, w_scales[u] + (b & ~3));
                    }
                }
#pragma unroll
                for(int u = 0; u < S::kTokensPer; ++u) {
                    const int r = tid + u * S::kThreads;
                    if(BN % S::kThreads == 0 || r < BN) {
                        kc::cp_async16(sub + S::kB + 16 * r, t_values[u] + 16 * b);
                        kc::cp_async_ca<4>(sub + S::kAs + 4 * r, t_scales[u] + (b >> shift));
                    }
                }
            }
        };
        // A landed stage's weight codes to int8 values and its scales to FP32, by the threads that copied them.
        auto decode = [&](int slot, const Visits& it) {
#pragma unroll
            for(int i = 0; i < G * RUN; ++i) {
                const int b = it.block(i % G, i / G);
                if(b < 0) continue;
                unsigned char* sub = smem + slot * S::kStage + i * S::kSub;
                float* ws = reinterpret_cast<float*>(sub + S::kWs);
#pragma unroll
                for(int u = 0; u < S::kRowsPer; ++u) {
                    const int r = tid + u * S::kThreads;
                    if(BM % S::kThreads == 0 || r < BM) {
                        const uint2 c = *reinterpret_cast<const uint2*>(sub + S::kCodes + 8 * r);
                        *reinterpret_cast<uint4*>(sub + S::kA + 16 * r) =
                            make_uint4(bs::nibbles_to_s8x4(c.x & 0xffffu, table),
                                       bs::nibbles_to_s8x4(c.x >> 16, table),
                                       bs::nibbles_to_s8x4(c.y & 0xffffu, table),
                                       bs::nibbles_to_s8x4(c.y >> 16, table));
                        const unsigned word = *reinterpret_cast<const unsigned*>(sub + S::kWraw + 4 * r);
                        ws[bs::pair_slot(r)] = kc::e4m3_value((word >> (8 * (b & 3))) & 0xffu);
                    }
                }
            }
        };

        Visits producer(blocks), decoder(blocks);
#pragma unroll
        for(int s = 0; s < STAGES - 1; ++s) {
            if(!producer.done()) {
                issue(s, producer);
                producer.next();
            }
            kc::cp_async_commit();
        }
        kc::cp_async_wait<STAGES - 2>();
        __syncthreads();
        decode(0, decoder);
        decoder.next();

        float acc[G][S::kOut], x[S::kOut][4];
        int v = 0;
        // G lanes at a time in the tree's order: their blocks (a lane past the last block has none: the tree then
        // takes +0.0f), RUN of each a visit of the ring, then the lanes folded into the partial sums.
#pragma unroll
        for(int at = 0; at < 16; at += G) {
#pragma unroll
            for(int s = 0; s < G; ++s)
#pragma unroll
                for(int o = 0; o < S::kOut; ++o) acc[s][o] = 0.0f;
            const int count = Visits::count(at, blocks);
            for(int k0 = 0; k0 < count; k0 += RUN, ++v) {
                kc::cp_async_wait<STAGES - 3>();
                __syncthreads();
                if(!decoder.done()) {
                    decode((v + 1) % STAGES, decoder);
                    decoder.next();
                }
                if(!producer.done()) {
                    issue((v + STAGES - 1) % STAGES, producer);
                    producer.next();
                }
                kc::cp_async_commit();
                // One block of each lane at a time (not unrolled: the fragments of one in registers at once).
#pragma unroll 1
                for(int r = 0; r < RUN; ++r)
#pragma unroll
                for(int s = 0; s < G; ++s) {
                    if(bs::lane_at(at + s) + 16 * (k0 + r) >= blocks) continue;
                    const unsigned char* sub = smem + (v % STAGES) * S::kStage + (r * G + s) * S::kSub;
                    const float* ws = reinterpret_cast<const float*>(sub + S::kWs);
                    const float* as = reinterpret_cast<const float*>(sub + S::kAs);
                    uint32_t fa[S::kMT][2], fb[S::kNT];
                    float2 w[S::kMT], a[S::kNT];
#pragma unroll
                    for(int mt = 0; mt < S::kMT; ++mt) {
                        kc::ldmatrix_x2(fa[mt][0], fa[mt][1], sub + S::kA + 16 * (wr + 16 * mt + lane % 16));
                        w[mt] = *reinterpret_cast<const float2*>(ws + bs::pair_slot(wr + 16 * mt + g));
                    }
                    if constexpr(S::kNT % 4 == 0) {
#pragma unroll
                        for(int nt = 0; nt < S::kNT; nt += 4)
                            kc::ldmatrix_x4(fb[nt], fb[nt + 1], fb[nt + 2], fb[nt + 3],
                                            sub + S::kB + 16 * (wt + 8 * nt + lane));
                    } else if constexpr(S::kNT == 2) {
                        kc::ldmatrix_x2(fb[0], fb[1], sub + S::kB + 16 * (wt + lane % 16));
                    } else {
                        kc::ldmatrix_x1(fb[0], sub + S::kB + 16 * (wt + lane % 8));
                    }
#pragma unroll
                    for(int nt = 0; nt < S::kNT; ++nt)
                        a[nt] = *reinterpret_cast<const float2*>(as + wt + 8 * nt + 2 * q);
                    // The block's MMAs first, then their steps: the sums of all tiles in flight together.
                    int d[S::kMT][S::kNT][4];
#pragma unroll
                    for(int mt = 0; mt < S::kMT; ++mt)
#pragma unroll
                        for(int nt = 0; nt < S::kNT; ++nt) bs::block_sum_s8(fa[mt][0], fa[mt][1], fb[nt], d[mt][nt]);
#pragma unroll
                    for(int mt = 0; mt < S::kMT; ++mt)
#pragma unroll
                        for(int nt = 0; nt < S::kNT; ++nt) {
                            float* c = acc[s] + 4 * (mt * S::kNT + nt);
                            c[0] = bs::step(bs::biased_sum(d[mt][nt][0]), w[mt].x, a[nt].x, c[0]);
                            c[1] = bs::step(bs::biased_sum(d[mt][nt][1]), w[mt].x, a[nt].y, c[1]);
                            c[2] = bs::step(bs::biased_sum(d[mt][nt][2]), w[mt].y, a[nt].x, c[2]);
                            c[3] = bs::step(bs::biased_sum(d[mt][nt][3]), w[mt].y, a[nt].y, c[3]);
                        }
                }
            }
#pragma unroll
            for(int s = 0; s < G; ++s)
#pragma unroll
                for(int o = 0; o < S::kOut; ++o) kc::fold16(at + s, acc[s][o], x[o]);
        }
#pragma unroll
        for(int mt = 0; mt < S::kMT; ++mt)
#pragma unroll
            for(int nt = 0; nt < S::kNT; ++nt)
#pragma unroll
                for(int o = 0; o < 4; ++o) {
                    const int r = row0 + wr + 16 * mt + g + 8 * (o / 2), t = tile + wt + 8 * nt + 2 * q + o % 2;
                    if(r < m.rows && t < job.count)
                        out[std::size_t(job.first + t) * std::size_t(out_ld) + std::size_t(r)] =
                            __fmul_rn(x[4 * (mt * S::kNT + nt) + o][0], m.post);
                }
        __syncthreads();   // the next token tile's prologue refills the stages
    }
}

// ---------------------------------------------------------------------------------------------------- native GEMM

// The prompt GEMM accumulated in the tensor cores (structure gemm-native): not the declared order. Its activations
// are the E2M1 x 2 quantizer's values packed as NVFP4 (quantize_nvfp4: the same codes, two a byte, and each block's
// E4M3 scale byte before the division by the global divisor). A CTA computes TM tokens x TN weight rows (tokens the
// MMA's A operand), a warp TM / WM x TN / WN, the K loop in BK-byte stages (2 BK values) of a STAGES-deep cp.async ring
// of swizzled tiles (kernel core tile.cuh: KTileCopy, ldsm_a, ldsm_b2, copy_rows, store_tile), persistent CTAs over
// the tiles: the multistage mainloop and tile scheduling of CUTLASS (BSD-3-Clause; the mechanisms and the
// shared-memory layouts, written anew). Two arithmetic paths, chosen by the call (enqueue_gemm_native's int4):
//   native  (sm_120a, E2M1 weights, non-negative scales as NVFP4's): kind::mxf4nvf4 block-scaled MMAs (mma.cuh's mma_nvf4_m16n8k64, UE4M3 scales per
//           16 on both operands), the tensor core applying the scales and summing into its FP32 accumulator;
//   generic (INT4 weights, and every architecture without block-scaled MMAs): each 16-block's exact integer sum on
//           the s8 MMA (blockscaled_mma.cuh's block_sum_s8 on codes widened by nibbles_to_s8x4), then
//           acc += sum * (weight scale * activation scale) on the CUDA cores.
// An output's sum depends on the CTA tile, so its bits may change with the knobs and the token count: row_invariant
// false. Within FP32 accumulation of the FP64 product.
// Two adjacent outputs (columns col, col + 1; the second only when `two`) of row `row`: times the column scales when
// given, then FP32 or BF16 (one rounding to nearest even), in one store where the alignment allows.
__device__ __forceinline__ void store_pair(const NativeOut& o, std::size_t row, int col, float v0, float v1, bool two) {
    if(o.col) {
        v0 = __fmul_rn(v0, o.col[col]);
        if(two) v1 = __fmul_rn(v1, o.col[col + 1]);
    }
    const std::size_t at = row * std::size_t(o.ld) + std::size_t(col);
    if(o.bf16) {
        __nv_bfloat16* p = static_cast<__nv_bfloat16*>(o.data) + at;
        if(two && at % 2 == 0) {
            *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(v0, v1);
        } else {
            p[0] = __float2bfloat16_rn(v0);
            if(two) p[1] = __float2bfloat16_rn(v1);
        }
    } else {
        float* p = static_cast<float*>(o.data) + at;
        if(two && at % 2 == 0) {
            *reinterpret_cast<float2*>(p) = make_float2(v0, v1);
        } else {
            p[0] = v0;
            if(two) p[1] = v1;
        }
    }
}

template <int TM, int TN, int WM, int WN, int BK, int STAGES, int GROUP>
struct Native {
    static constexpr int kThreads = 32 * WM * WN, kWTM = TM / WM, kWTN = TN / WN;
    static constexpr int kMT = kWTM / 16, kNT = kWTN / 8, kKS = BK / 32, kSB = BK / 8;
    static_assert(kWTM % 16 == 0 && kWTN % 16 == 0 && kMT >= 1, "warp tiles of whole 16-token, 16-row tiles");
    static_assert(BK == 64 || BK == 128, "64 or 128 bytes (128 or 256 values) of K a stage");
    static_assert(STAGES >= 2, "a ring of two stages or more");
    // A stage: tokens' codes [TM][BK], weight codes [TN][BK] (swizzled tiles), token scale bytes [TM][kSB], weight
    // scale bytes [TN][kSB].
    static constexpr int kA = 0, kB = TM * BK, kSa = kB + TN * BK, kSw = kSa + TM * kSB;
    static constexpr int kStage = (kSw + TN * kSB + 127) / 128 * 128, kSmem = STAGES * kStage;
    // Rows of the accumulator tile the epilogue stages in shared memory at a time (tile.cuh's store_tile).
    static constexpr int kChunk = crit::kcore::tile::store_chunk(TM, TN, kSmem);
    static_assert(kChunk * (TN + 4) * 4 <= kSmem, "the epilogue's 16 rows within the ring's shared memory");
};

// A CTA's tiles of one job, persistent: tiles blockIdx.x, + gridDim.x, ... in tile_of's grouped order. Each tile's
// first STAGES - 1 stages are issued before the previous tile's epilogue (EPI 0: the accumulators stored from
// registers while the next tile's copies fly) or right after it (EPI 1: through shared memory, tile.cuh's store_tile,
// whole rows a store).
template <class C, bool NATIVE, int TM, int TN, int WM, int BK, int STAGES, int EPI>
__device__ __forceinline__ void native_tiles(const Matrix& m, const ActsFp4& acts, const GemmJob& job, int tiles_m,
                                             int tiles_n, int group, unsigned char* smem, const NativeOut& out) {
    namespace kt = crit::kcore::tile;
    constexpr int T = C::kThreads, MT = C::kMT, NT = C::kNT, SB = C::kSB, KS = C::kKS;
    const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32, gid = lane >> 2, tig = lane & 3;
    const int wm0 = (warp % WM) * C::kWTM, wn0 = (warp / WM) * C::kWTN;
    const int kbytes = m.cols / 2, sbytes = m.cols / 16, KT = (kbytes + BK - 1) / BK, total = tiles_m * tiles_n;
    // Native sums are 4 x (both operands' E2M1 x 2 integers), generic ones their integers; times post / global.
    const float scale = (NATIVE ? 4.0f : 1.0f) * m.post / acts.global;

    struct Tile {
        int n0, tokens, rows;
        std::size_t token0;
        kt::KTileCopy<TM, BK, T> a;
        kt::KTileCopy<TN, BK, T> b;
        const unsigned char *a_scales, *w_scales;
        int a_width, w_width;
    };
    auto set = [&](Tile& s, int linear) {
        int tm, tn;
        kt::tile_of(linear, tiles_m, tiles_n, group, tm, tn);
        s.n0 = tn * TN;
        s.tokens = min(TM, job.count - tm * TM);
        s.rows = min(TN, m.rows - s.n0);
        s.token0 = std::size_t(job.first + tm * TM);
        s.a.init(acts.codes + s.token0 * std::size_t(acts.ld), std::size_t(acts.ld), s.tokens, tid);
        s.b.init(m.codes + std::size_t(s.n0) * std::size_t(kbytes), std::size_t(kbytes), s.rows, tid);
        s.a_scales = acts.scales + s.token0 * std::size_t(acts.lds);
        s.w_scales = m.scales + std::size_t(s.n0) * std::size_t(sbytes);
        s.a_width = kt::row_copy_width(s.a_scales, std::size_t(acts.lds), SB);
        s.w_width = kt::row_copy_width(s.w_scales, std::size_t(sbytes), SB);
    };
    auto issue = [&](const Tile& s, int k_tile) {
        unsigned char* stage = smem + (k_tile % STAGES) * C::kStage;
        const int k0 = k_tile * BK;
        s.a.issue(stage + C::kA, k0, kbytes);
        s.b.issue(stage + C::kB, k0, kbytes);
        kt::copy_rows<TM, SB, T>(stage + C::kSa, s.a_scales, std::size_t(acts.lds), s.tokens, k0 / 8, sbytes,
                                 s.a_width, tid);
        kt::copy_rows<TN, SB, T>(stage + C::kSw, s.w_scales, std::size_t(sbytes), s.rows, k0 / 8, sbytes, s.w_width,
                                 tid);
    };
    auto prologue = [&](const Tile& s) {
#pragma unroll
        for(int st = 0; st < STAGES - 1; ++st) {
            if(st < KT) issue(s, st);
            kc::cp_async_commit();
        }
    };
    // A k-step's fragments (32 bytes of every row: 64 values) and its scale words (4 blocks a row).
    auto fragments = [&](const unsigned char* stage, int ks, uint32_t (&a)[MT][4], uint32_t (&b)[NT][2],
                         uint32_t (&sa)[MT], uint32_t (&sb)[NT]) {
#pragma unroll
        for(int i = 0; i < MT; ++i) {
            kt::ldsm_a<BK>(stage + C::kA, wm0 + 16 * i, ks, lane, a[i]);
            sa[i] = *reinterpret_cast<const uint32_t*>(stage + C::kSa + (wm0 + 16 * i + gid + 8 * (tig & 1)) * SB +
                                                       4 * ks);
        }
#pragma unroll
        for(int j = 0; j < NT; j += 2) {
            uint32_t four[4];
            kt::ldsm_b2<BK>(stage + C::kB, wn0 + 8 * j, ks, lane, four);
            b[j][0] = four[0], b[j][1] = four[1], b[j + 1][0] = four[2], b[j + 1][1] = four[3];
        }
#pragma unroll
        for(int j = 0; j < NT; ++j)
            sb[j] = *reinterpret_cast<const uint32_t*>(stage + C::kSw + (wn0 + 8 * j + gid) * SB + 4 * ks);
    };
    [[maybe_unused]] const bs::NibbleTable weights = bs::nibble_table(m.int4 != 0), values = bs::nibble_table(false);

    int linear = int(blockIdx.x);
    Tile cur;
    set(cur, linear);
    prologue(cur);
    for(;;) {
        float acc[MT][NT][4];
#pragma unroll
        for(int i = 0; i < MT; ++i)
#pragma unroll
            for(int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        if constexpr(NATIVE) {
            // CUTLASS's multistage order (BSD-3-Clause; the mechanism): a k-step's fragments are read while the
            // previous k-step's MMAs run, and the next stage's wait and barrier come before the last k-step of a stage,
            // so its MMAs cover the first reads of the next stage; a stage's copies are issued after its first k-step.
            uint32_t a[2][MT][4], b[2][NT][2], sa[2][MT], sb[2][NT];
            kc::cp_async_wait<STAGES - 2>();
            __syncthreads();
            fragments(smem, 0, a[0], b[0], sa[0], sb[0]);
            for(int k_tile = 0; k_tile < KT; ++k_tile) {
                const unsigned char* stage = smem + (k_tile % STAGES) * C::kStage;
                const unsigned char* next = smem + ((k_tile + 1) % STAGES) * C::kStage;
#pragma unroll
                for(int ks = 0; ks < KS; ++ks) {
                    const int f = ks & 1, g = (ks + 1) & 1;
                    if(ks == KS - 1) {
                        kc::cp_async_wait<STAGES - 2>();
                        __syncthreads();
                        if(k_tile + 1 < KT) fragments(next, 0, a[g], b[g], sa[g], sb[g]);
                    } else {
                        fragments(stage, ks + 1, a[g], b[g], sa[g], sb[g]);
                    }
                    if(ks == 0) {
                        if(k_tile + STAGES - 1 < KT) issue(cur, k_tile + STAGES - 1);
                        kc::cp_async_commit();
                    }
#pragma unroll
                    for(int i = 0; i < MT; ++i)
#pragma unroll
                        for(int j = 0; j < NT; ++j)
                            kc::mma_nvf4_m16n8k64(acc[i][j], a[f][i][0], a[f][i][1], a[f][i][2], a[f][i][3], b[f][j][0],
                                                  b[f][j][1], sa[f][i], sb[f][j]);
                }
            }
        } else {
            for(int k_tile = 0; k_tile < KT; ++k_tile) {
                kc::cp_async_wait<STAGES - 2>();
                __syncthreads();
                if(k_tile + STAGES - 1 < KT) issue(cur, k_tile + STAGES - 1);
                kc::cp_async_commit();
                const unsigned char* stage = smem + (k_tile % STAGES) * C::kStage;
                const unsigned char* sa_row = stage + C::kSa;
                const unsigned char* sw_row = stage + C::kSw;
                // A 16-value block is 8 bytes of a row: thread tig's 4 values are bytes 2 tig, 2 tig + 1 of it.
                auto pair = [&](const unsigned char* tile, int row, int byte) {
                    return unsigned(*reinterpret_cast<const uint16_t*>(tile + kt::offset<BK>(row, byte / 16) + byte % 16));
                };
#pragma unroll 1
                for(int blk = 0; blk < BK / 8; ++blk) {
                    const int byte = 8 * blk + 2 * tig;
                    uint32_t a[MT][2], b[NT];
                    float xa[MT][2], xw[NT][2];
#pragma unroll
                    for(int i = 0; i < MT; ++i) {
                        const int r = wm0 + 16 * i + gid;
                        a[i][0] = bs::nibbles_to_s8x4(pair(stage + C::kA, r, byte), values);
                        a[i][1] = bs::nibbles_to_s8x4(pair(stage + C::kA, r + 8, byte), values);
                        xa[i][0] = kc::e4m3_value(sa_row[r * SB + blk]);
                        xa[i][1] = kc::e4m3_value(sa_row[(r + 8) * SB + blk]);
                    }
#pragma unroll
                    for(int j = 0; j < NT; ++j) {
                        const int c = wn0 + 8 * j;
                        b[j] = bs::nibbles_to_s8x4(pair(stage + C::kB, c + gid, byte), weights);
                        xw[j][0] = kc::e4m3_value(sw_row[(c + 2 * tig) * SB + blk]);
                        xw[j][1] = kc::e4m3_value(sw_row[(c + 2 * tig + 1) * SB + blk]);
                    }
#pragma unroll
                    for(int i = 0; i < MT; ++i)
#pragma unroll
                        for(int j = 0; j < NT; ++j) {
                            int d[4];
                            bs::block_sum_s8(a[i][0], a[i][1], b[j], d);
                            acc[i][j][0] = fmaf(bs::biased_sum(d[0]), xa[i][0] * xw[j][0], acc[i][j][0]);
                            acc[i][j][1] = fmaf(bs::biased_sum(d[1]), xa[i][0] * xw[j][1], acc[i][j][1]);
                            acc[i][j][2] = fmaf(bs::biased_sum(d[2]), xa[i][1] * xw[j][0], acc[i][j][2]);
                            acc[i][j][3] = fmaf(bs::biased_sum(d[3]), xa[i][1] * xw[j][1], acc[i][j][3]);
                        }
                }
            }
        }
        const int next = linear + int(gridDim.x);
        const bool more = next < total;
        Tile nx;
        // EPI 1 stages FP32 rows through shared memory; BF16 outputs and column scales take the register stores.
        const bool staged = EPI == 1 && !out.bf16 && !out.col;
        if(!staged) {
            if(more) {
                __syncthreads();   // every warp's last reads of the ring are done
                set(nx, next);
                prologue(nx);
            }
#pragma unroll
            for(int i = 0; i < MT; ++i)
#pragma unroll
                for(int h = 0; h < 2; ++h) {
                    const int t = wm0 + 16 * i + gid + 8 * h;
                    if(t >= cur.tokens) continue;
                    const std::size_t row = cur.token0 + std::size_t(t);
#pragma unroll
                    for(int j = 0; j < NT; ++j) {
                        const int r = wn0 + 8 * j + 2 * tig;
                        if(r < cur.rows)
                            store_pair(out, row, cur.n0 + r, acc[i][j][2 * h] * scale, acc[i][j][2 * h + 1] * scale,
                                       r + 1 < cur.rows);
                    }
                }
        } else {
            float* tile_out = static_cast<float*>(out.data) + cur.token0 * std::size_t(out.ld) + std::size_t(cur.n0);
            kc::cp_async_wait<0>();
            kt::store_tile<TM, TN, MT, NT, C::kChunk, T>(acc, scale, wm0, wn0, lane, reinterpret_cast<float*>(smem),
                                                         tile_out, std::size_t(out.ld), cur.tokens, cur.rows, tid);
            if(more) {
                __syncthreads();   // every thread's reads of the staged rows are done
                set(nx, next);
                prologue(nx);
            }
        }
        if(!more) break;
        cur = nx;
        linear = next;
    }
    kc::cp_async_wait<0>();
}

// NATIVE: the caller states that every matrix holds E2M1 weights; on sm_120a the kernel then holds the block-scaled
// path alone (an INT4 matrix, contrary to the statement, gets NaN outputs: the generic path beside it would cost the
// native one registers). Elsewhere, and with NATIVE false, the generic path.
template <int TM, int TN, int WM, int WN, int BK, int STAGES, int GROUP, int EPI, bool NATIVE>
__global__ void __launch_bounds__(32 * WM * WN, 1)
    native_kernel(const Matrix* matrices, const GemmJob* jobs, ActsFp4 acts, NativeOut out) {
    using C = Native<TM, TN, WM, WN, BK, STAGES, GROUP>;
    extern __shared__ __align__(128) unsigned char native_smem[];
    const GemmJob job = jobs[blockIdx.y];
    const Matrix m = matrices[job.matrix];
    const int tiles_m = (job.count + TM - 1) / TM, tiles_n = (m.rows + TN - 1) / TN;
    if(job.count <= 0 || int(blockIdx.x) >= tiles_m * tiles_n) return;
    if constexpr(NATIVE && CRIT_KCORE_HAS_BLOCK_SCALED_MMA) {
        if(m.int4) {
            if(blockIdx.x == 0)
                for(int i = int(threadIdx.x); i < job.count * m.rows; i += C::kThreads)
                    store_pair(out, std::size_t(job.first + i / m.rows), i % m.rows, __int_as_float(0x7fc00000), 0.0f,
                               false);
            return;
        }
        native_tiles<C, true, TM, TN, WM, BK, STAGES, EPI>(m, acts, job, tiles_m, tiles_n, GROUP, native_smem, out);
    } else {
        native_tiles<C, false, TM, TN, WM, BK, STAGES, EPI>(m, acts, job, tiles_m, tiles_n, GROUP, native_smem, out);
    }
}

// ---------------------------------------------------------------------------------------------------- pipelined native GEMM

// gemm-native's block-scaled path on sm_120a (E2M1 weights): native_tiles' arithmetic (kind::mxf4nvf4 m16n8k64 MMAs,
// both operands' UE4M3 scales per 16 applied by the tensor core, FP32 accumulation in increasing K) with the data
// movement of CUTLASS's warp-specialized TMA GEMMs for sm_90 / sm_120 (BSD-3-Clause; the mechanisms, written anew from
// the PTX ISA): a ring of STAGES shared-memory stages, each filled by two TMA tile copies (the activation codes' and
// the matrix's tensor maps: TM or TN rows x BK bytes a box, swizzled as tile.cuh's offset<BK>, rows and K past the
// tensors zero-filled) and one bulk copy a 128-row tile of tiled scale bytes, completing on the stage's full mbarrier.
// A producer warp issues each stage once the WM x WN consumer warps have released it (its empty mbarrier); the
// consumers wait on the full one, read their fragments by ldmatrix at fixed offsets (the next k-step's while the
// current one's MMAs run, the next stage's first while a stage's last MMAs run) and release it. Persistent CTAs take
// their tiles in tile_of's grouped order, and the producer loads the next tile's first stages while the consumers store
// the last one. Without tensor maps (a table without its host record) the producer's lanes copy 16-byte chunks by
// cp.async; row-major scale bytes go in 4-byte cp.async pieces (the lanes' noinc arrivals on the full barrier). The
// tiles past the last whole wave may be shared out over the CTAs k-iteration by k-iteration (stream-K, WsSplit). Not
// the declared order; an output's bits depend on the CTA tile and its K segments (fixed by the configuration,
// max_count, max_rows and the device's SM count), never on a job's own token count, which CTA takes a tile or which
// segment finishes last.
namespace ws {

__device__ __forceinline__ void cp_async_arrive_noinc(uint64_t* bar) {
#if CRIT_KCORE_HAS_CP_ASYNC
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(kc::smem_u32(bar)) : "memory");
#endif
}
__device__ __forceinline__ void consumers_sync(int threads) {
    asm volatile("bar.sync 1, %0;\n" ::"r"(threads) : "memory");
}
// Until at most N of this thread's committed bulk stores still read their shared memory.
template <int N>
__device__ __forceinline__ void store_wait() {
#if CRIT_KCORE_HAS_TMA
    asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
#endif
}
// A byte's place in a staging box of RB-byte rows, in TMA's 32B / 64B / 128B swizzle (load.cuh's swizzles).
template <int RB>
__device__ __forceinline__ uint32_t store_swizzle(uint32_t byte) {
    if constexpr(RB == 128) return kc::swizzle_128b(byte);
    else if constexpr(RB == 64) return kc::swizzle_64b(byte);
    else return kc::swizzle_32b(byte);
}
// Row r's scale word of k-step ks (four 16-blocks) in a stage's scale area: 128-row tiles of KS k-steps, each k-step a
// 512-byte block in the tiled layout (row % 32) * 16 + (row % 128 / 32) * 4.
template <int KS>
__device__ __forceinline__ uint32_t scale_at(int r, int ks) {
    return uint32_t(((r >> 7) * KS + ks) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4);
}

}  // namespace ws

template <int TM, int TN, int WM, int WN, int BK, int STAGES, int EPI = 0>
struct Ws {
    static constexpr int kConsumers = WM * WN, kThreads = 32 * (kConsumers + 1);
    static constexpr int kWTM = TM / WM, kWTN = TN / WN, kMT = kWTM / 16, kNT = kWTN / 8, kKS = BK / 32;
    static_assert(kWTM % 16 == 0 && kWTN % 16 == 0 && kMT >= 1, "warp tiles of whole 16-token, 16-row tiles");
    static_assert(BK == 64 || BK == 128, "64 or 128 bytes (128 or 256 values) of K a stage");
    static_assert(TM % 64 == 0 && TN % 64 == 0, "CTA tiles of 64, 128, 192 or 256");
    static_assert(kWTM <= 128 && kWTN <= 128, "a warp's scale rows within one 128-row tile");
    // A stage: token codes [TM][BK], weight codes [TN][BK] (swizzled tiles), then each operand's scale words as
    // 128-row tiles.
    // 128-row scale tiles a stage holds: a tile starts at a multiple of 64 rows of its 128-row scale tile.
    static constexpr int kTA = TM % 128 ? (TM + 64 + 127) / 128 : TM > 128 ? TM / 128 : 1;
    static constexpr int kTB = TN % 128 ? (TN + 64 + 127) / 128 : TN > 128 ? TN / 128 : 1;
    // More than 8 consumer warps (128 registers a thread): the B fragments streamed two n8 tiles at a time.
    static constexpr bool kStream = kConsumers > 8;
    static constexpr int kA = 0, kB = TM * BK, kSa = kB + TN * BK, kSw = kSa + kTA * kKS * 512;
    static constexpr int kStage = (kSw + kTB * kKS * 512 + 1023) / 1024 * 1024;   // TMA swizzle: 1024-aligned
    // EPI 1 / 2: one or two BF16 staging boxes a consumer warp for its TMA stores, 16 rows x its kWTN columns each
    // (rows of kWTN x 2 bytes in TMA's matching swizzle), 1024-aligned.
    static constexpr int kBox = 16 * kWTN * 2, kBoxAt = (kBox + 1023) / 1024 * 1024;
    static constexpr int kEpi = EPI ? kConsumers * EPI * kBoxAt : 0;
    static constexpr int kFit = (101376 - 1024 - 256 - kEpi) / kStage;   // 99 KB less the alignment and the barriers
    static_assert(kFit >= 1, "one stage within a CTA's shared memory");
    static constexpr int kStages = STAGES < kFit ? STAGES : kFit;   // the knob, or as many as fit
    static constexpr int kOut = kStages * kStage, kBar = kOut + kEpi, kSmem = 1024 + kBar + 2 * kStages * 8 + 16;
    static_assert(!EPI || kWTN == 16 || kWTN == 32 || kWTN == 64, "store rows of 32, 64 or 128 bytes");
};

// A launch's tile plan (chosen on the host from max_count and max_rows, so a row's bits never depend on a job's own
// count): tiles in tile_of's order of a tiles_m x tiles_n grid; the first `whole` taken whole by the persistent CTAs,
// the `tail` after them shared out k-iteration by k-iteration over `ctas` CTAs (stream-K: CTA b takes k-iterations
// [b W / ctas, (b + 1) W / ctas) of the tail's W = tail x KT, a segment of one tile or the end of one and the start of
// the next). A tile in segments has each segment's FP32 partial in the workspace (segs slots a tile) and the last
// segment to finish (a per-tile counter) adds them in segment order and stores the tile.
struct WsSplit {
    int tiles_m = 0, tiles_n = 0;
    int whole = 0, tail = 0, ctas = 0, segs = 1;
    float* partial = nullptr;  // [tail][segs][MT NT 4][consumer threads] FP32
    int* counters = nullptr;   // [tail], zero
};

// Tensor maps of one launch (built on the host): the activation codes' and each matrix's, boxes of TM / TN rows x BK
// bytes; count 0 where the host has no matrix pointers (the copies then go by cp.async).
constexpr int kMaxMaps = 4;
struct alignas(64) WsMaps {
    CUtensorMap a;
    CUtensorMap w[kMaxMaps];
    CUtensorMap out;   // EPI 1 / 2: the BF16 output, boxes of 16 rows x a warp tile's columns
    int count;
    int out_ok;
};

template <int TM, int TN, int WM, int WN, int BK, int STAGES, int GROUP, int EPI>
__global__ void __launch_bounds__(32 * (WM * WN + 1), 1)
    ws_kernel(const Matrix* matrices, const GemmJob* jobs, ActsFp4 acts, NativeOut out, WsSplit split,
              const __grid_constant__ WsMaps maps) {
    using C = Ws<TM, TN, WM, WN, BK, STAGES, EPI>;
    constexpr int MT = C::kMT, NT = C::kNT, KS = C::kKS, ST = C::kStages, NC = C::kConsumers;
    extern __shared__ __align__(128) unsigned char ws_smem[];
    const GemmJob job = jobs[blockIdx.y];
    const Matrix m = matrices[job.matrix];
    const int tiles_m = split.tiles_m, tiles_n = split.tiles_n;
    if(job.count <= 0 || (int(blockIdx.x) >= split.whole && int(blockIdx.x) >= split.ctas)) return;
#if CRIT_KCORE_HAS_BLOCK_SCALED_MMA && CRIT_KCORE_HAS_TMA
    unsigned char* ws_base = ws_smem + ((1024u - (kc::smem_u32(ws_smem) & 1023u)) & 1023u);
    uint64_t* full = reinterpret_cast<uint64_t*>(ws_base + C::kBar);
    uint64_t* empty = full + ST;
    int* last = reinterpret_cast<int*>(empty + ST);
    const bool a_tiled = acts.tiled & kTiledActs, w_tiled = acts.tiled & kTiledWeights;
    const bool tma = job.matrix < maps.count;
    const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
    // The issuing lanes' cp.async arrivals: without tensor maps, or for row-major scale bytes.
    const bool pieces = !tma || !a_tiled || !w_tiled;
    if(tid == 0) {
        for(int s = 0; s < ST; ++s) {
            kc::mbarrier_init(full + s, pieces ? (tma ? 33 : 32) : 1);
            kc::mbarrier_init(empty + s, NC);
        }
        kc::fence_barrier_init();
    }
    __syncthreads();
    const int kbytes = m.cols / 2, KT = (kbytes + BK - 1) / BK;
    const int a_ct = acts.lds / 4, w_ct = m.cols / 64;   // 4-block column tiles a row tile (tiled layouts)
    // A unit of work: k-tiles kt0 .. kt1 of a tile, segment seg of nseg (nseg 1: the whole tile); empty when the tile
    // is past the job's tokens or the matrix's rows (planned for max_count and max_rows).
    struct Unit {
        int tile, seg, nseg, tokens, rows, n0, kt0, kt1, a_off, w_off;
        std::size_t token0;
        bool empty;
    };
    const long long tail_work = (long long)split.tail * KT;
    auto cta_of = [&](long long k) { return int((k * split.ctas + split.ctas - 1) / tail_work); };   // tail k-iteration k's
    auto unit = [&](int tile, int kt0, int kt1) {
        Unit t;
        t.tile = tile;
        t.kt0 = kt0;
        t.kt1 = kt1;
        t.seg = 0;
        t.nseg = 1;
        if(tile >= split.whole) {
            const long long k0 = (long long)(tile - split.whole) * KT;
            const int first = cta_of(k0);
            t.seg = int(blockIdx.x) - first;
            t.nseg = cta_of(k0 + KT - 1) - first + 1;
        }
        int tm, tn;
        kc::tile::tile_of(tile, tiles_m, tiles_n, GROUP, tm, tn);
        t.n0 = tn * TN;
        t.tokens = min(TM, job.count - tm * TM);
        t.rows = min(TN, m.rows - t.n0);
        t.empty = t.tokens <= 0 || t.rows <= 0;
        t.token0 = std::size_t(job.first + tm * TM);
        t.a_off = a_tiled ? int(t.token0 & 127) : 0;
        t.w_off = w_tiled ? (t.n0 & 127) : 0;
        return t;
    };
    // This CTA's units in order: its whole tiles (gridDim.x apart), then its stretch of the tail; empty ones skipped.
    struct Walk {
        int w;
        long long k, hi;
    };
    auto walk_start = [&](Walk& v) {
        v.w = int(blockIdx.x);
        v.k = v.hi = 0;
        if(int(blockIdx.x) < split.ctas) {
            v.k = (long long)blockIdx.x * tail_work / split.ctas;
            v.hi = ((long long)blockIdx.x + 1) * tail_work / split.ctas;
        }
    };
    auto walk_next = [&](Walk& v, Unit& t) {
        for(;;) {
            if(v.w < split.whole) {
                t = unit(v.w, 0, KT);
                v.w += int(gridDim.x);
            } else if(v.k < v.hi) {
                const int tile = split.whole + int(v.k / KT), kt0 = int(v.k % KT);
                const int kt1 = int(min((long long)KT, kt0 + (v.hi - v.k)));
                t = unit(tile, kt0, kt1);
                v.k += kt1 - kt0;
            } else {
                return false;
            }
            if(!t.empty) return true;
        }
    };
    // One stage's copies (k-tile kt of unit t into slot s), by one warp: lane 0 the tensor-map and bulk copies and
    // their byte count, the lanes the cp.async chunks and pieces (and their arrivals).
    auto issue = [&](int s, const Unit& t, int kt) {
        unsigned char* stage = ws_base + s * C::kStage;
        const int k0 = kt * BK, ksv = min(KS, (kbytes - k0) / 32);
        const int a_tiles = (t.a_off + t.tokens + 127) / 128, w_tiles = (t.w_off + t.rows + 127) / 128;
        if(tma) {
            if(lane == 0) {
                const uint32_t bytes = uint32_t((TM + TN) * BK + (a_tiled ? a_tiles * ksv * 512 : 0) +
                                                (w_tiled ? w_tiles * ksv * 512 : 0));
                kc::fence_proxy_async();
                kc::mbarrier_arrive_expect_tx(full + s, bytes);
                kc::tma_load_2d(stage + C::kA, &maps.a, k0, int(t.token0), full + s);
                kc::tma_load_2d(stage + C::kB, &maps.w[job.matrix], k0, t.n0, full + s);
                if(a_tiled)
                    for(int tl = 0; tl < a_tiles; ++tl)
                        kc::tma_load_1d(stage + C::kSa + tl * KS * 512,
                                        acts.scales + (std::size_t(int(t.token0 >> 7) + tl) * a_ct + k0 / 32) * 512,
                                        uint32_t(ksv * 512), full + s);
                if(w_tiled)
                    for(int tl = 0; tl < w_tiles; ++tl)
                        kc::tma_load_1d(stage + C::kSw + tl * KS * 512,
                                        m.scales + (std::size_t((t.n0 >> 7) + tl) * w_ct + k0 / 32) * 512,
                                        uint32_t(ksv * 512), full + s);
            }
        } else {
            constexpr int CH = BK / 16;
            const unsigned char* a_src = acts.codes + t.token0 * std::size_t(acts.ld);
            const unsigned char* w_src = m.codes + std::size_t(t.n0) * std::size_t(kbytes);
            for(int idx = lane; idx < TM * CH; idx += 32) {
                const int r = idx / CH, c = idx % CH;
                const bool ok = r < t.tokens && k0 + 16 * c < kbytes;
                kc::cp_async16(stage + C::kA + kc::tile::offset<BK>(r, c),
                               ok ? a_src + std::size_t(r) * std::size_t(acts.ld) + k0 + 16 * c : a_src, ok);
            }
            for(int idx = lane; idx < TN * CH; idx += 32) {
                const int r = idx / CH, c = idx % CH;
                const bool ok = r < t.rows && k0 + 16 * c < kbytes;
                kc::cp_async16(stage + C::kB + kc::tile::offset<BK>(r, c),
                               ok ? w_src + std::size_t(r) * std::size_t(kbytes) + k0 + 16 * c : w_src, ok);
            }
            if(a_tiled)
                for(int idx = lane; idx < a_tiles * ksv * 32; idx += 32) {
                    const int tl = idx / (ksv * 32), ks = (idx / 32) % ksv, piece = idx % 32;
                    kc::cp_async16(stage + C::kSa + (tl * KS + ks) * 512 + piece * 16,
                                   acts.scales + (std::size_t(int(t.token0 >> 7) + tl) * a_ct + k0 / 32 + ks) * 512 +
                                       piece * 16);
                }
            if(w_tiled)
                for(int idx = lane; idx < w_tiles * ksv * 32; idx += 32) {
                    const int tl = idx / (ksv * 32), ks = (idx / 32) % ksv, piece = idx % 32;
                    kc::cp_async16(stage + C::kSw + (tl * KS + ks) * 512 + piece * 16,
                                   m.scales + (std::size_t((t.n0 >> 7) + tl) * w_ct + k0 / 32 + ks) * 512 + piece * 16);
                }
        }
        if(!a_tiled)
            for(int p = lane; p < t.tokens * ksv; p += 32) {
                const int r = p / ksv, ks = p % ksv;
                kc::cp_async_ca<4>(stage + C::kSa + ws::scale_at<KS>(r, ks),
                                   acts.scales + (t.token0 + r) * std::size_t(acts.lds) + k0 / 8 + 4 * ks);
            }
        if(!w_tiled)
            for(int p = lane; p < t.rows * ksv; p += 32) {
                const int r = p / ksv, ks = p % ksv;
                kc::cp_async_ca<4>(stage + C::kSw + ws::scale_at<KS>(r, ks),
                                   m.scales + std::size_t(t.n0 + r) * std::size_t(m.cols / 16) + k0 / 8 + 4 * ks);
            }
        if(pieces) ws::cp_async_arrive_noinc(full + s);
    };
    {
        if(warp == NC) {
            // The producer: every stage of this CTA's units, in the consumers' order.
            Walk v;
            walk_start(v);
            Unit t;
            int it = 0;
            while(walk_next(v, t))
                for(int kt = t.kt0; kt < t.kt1; ++kt, ++it) {
                    const int s = it % ST;
                    if(it >= ST) kc::mbarrier_wait(empty + s, uint32_t((it / ST - 1) & 1));
                    issue(s, t, kt);
                }
            kc::cp_async_wait<0>();
            return;
        }
    }
    auto release = [&](int it) {
        __syncwarp();
        if(lane == 0) kc::mbarrier_arrive(empty + it % ST);
    };

    // The MMAs. Each lane's fragment and scale addresses are fixed offsets of a stage (one per k-step, the tiles' rows
    // a constant apart: 16 rows keep tile.cuh's swizzle, and a warp's scale rows stay in one 128-row tile).
    const int gid = lane >> 2, tig = lane & 3, q = lane >> 3;
    const int wm0 = (warp % WM) * C::kWTM, wn0 = (warp / WM) * C::kWTN;
    const float scale = 4.0f * m.post / acts.global;   // sums of both operands' E2M1 x 2 integers, times post / global
    uint32_t a_at[KS], b_at[KS];
#pragma unroll
    for(int ks = 0; ks < KS; ++ks) {
        a_at[ks] = uint32_t(C::kA) + kc::tile::offset<BK>(wm0 + (lane & 7) + 8 * (q & 1), 2 * ks + (q >> 1));
        b_at[ks] = uint32_t(C::kB) + kc::tile::offset<BK>(wn0 + (lane & 7) + 8 * (q >> 1), 2 * ks + (q & 1));
    }
    constexpr auto delta = [](int d) { return (d % 32) * 16 + (d / 32) * 4; };   // scale rows d apart (scale_at)
    constexpr bool STREAM = C::kStream;
    uint32_t fa[STREAM ? 1 : 2][MT][4], fb[STREAM ? 1 : 2][NT][2], sa[STREAM ? 1 : 2][MT], sb[STREAM ? 1 : 2][NT];
    uint32_t qa[STREAM ? 2 : 1][MT][4], qsa[STREAM ? 2 : 1][MT], qb[2][4], qsb[2][2];
    auto load_a = [&](const unsigned char* stage, int ks, uint32_t sa_at, int buf) {
#pragma unroll
        for(int i = 0; i < MT; ++i) {
            kc::ldmatrix_x4(qa[buf][i][0], qa[buf][i][1], qa[buf][i][2], qa[buf][i][3], stage + a_at[ks] + 16 * i * BK);
            qsa[buf][i] = *reinterpret_cast<const uint32_t*>(stage + sa_at + ks * 512 + delta(16 * i));
        }
    };
    auto load_pair = [&](const unsigned char* stage, int ks, int pr, uint32_t sb_at, int buf) {
        kc::ldmatrix_x4(qb[buf][0], qb[buf][1], qb[buf][2], qb[buf][3], stage + b_at[ks] + 16 * pr * BK);
        qsb[buf][0] = *reinterpret_cast<const uint32_t*>(stage + sb_at + ks * 512 + delta(16 * pr));
        qsb[buf][1] = *reinterpret_cast<const uint32_t*>(stage + sb_at + ks * 512 + delta(16 * pr + 8));
    };
    auto fragments = [&](const unsigned char* stage, int ks, uint32_t sa_at, uint32_t sb_at, uint32_t (&a)[MT][4],
                         uint32_t (&b)[NT][2], uint32_t (&xa)[MT], uint32_t (&xb)[NT]) {
#pragma unroll
        for(int i = 0; i < MT; ++i) {
            kc::ldmatrix_x4(a[i][0], a[i][1], a[i][2], a[i][3], stage + a_at[ks] + 16 * i * BK);
            xa[i] = *reinterpret_cast<const uint32_t*>(stage + sa_at + ks * 512 + delta(16 * i));
        }
#pragma unroll
        for(int j = 0; j < NT; j += 2)
            kc::ldmatrix_x4(b[j][0], b[j][1], b[j + 1][0], b[j + 1][1], stage + b_at[ks] + 8 * j * BK);
#pragma unroll
        for(int j = 0; j < NT; ++j) xb[j] = *reinterpret_cast<const uint32_t*>(stage + sb_at + ks * 512 + delta(8 * j));
    };
    int it = 0, stores = 0;
    Walk walk;
    walk_start(walk);
    Unit t;
    while(walk_next(walk, t)) {
        const uint32_t sa_at = uint32_t(C::kSa) + ws::scale_at<KS>(t.a_off + wm0 + gid + 8 * (tig & 1), 0);
        const uint32_t sb_at = uint32_t(C::kSw) + ws::scale_at<KS>(t.w_off + wn0 + gid, 0);
        float acc[MT][NT][4];
#pragma unroll
        for(int i = 0; i < MT; ++i)
#pragma unroll
            for(int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        if constexpr(C::kStream) {
            // A k-step's A fragments read one k-step ahead (at its first pair), its B fragments two n8 tiles at a
            // time, the next pair's read while a pair's MMAs run.
            for(int kt = t.kt0; kt < t.kt1; ++kt, ++it) {
                const unsigned char* stage = ws_base + (it % ST) * C::kStage;
                if(ST == 1 || kt == t.kt0) {
                    kc::mbarrier_wait(full + it % ST, uint32_t((it / ST) & 1));
                    load_a(stage, 0, sa_at, 0);
                    load_pair(stage, 0, 0, sb_at, 0);
                }
                const int ksv = min(KS, (kbytes - kt * BK) / 32);
                const unsigned char* next = stage;
#pragma unroll
                for(int ks = 0; ks < KS; ++ks) {
                    if(ks < ksv) {
                        const int f = ks & 1;
                        const bool more = ks + 1 < ksv, cross = !more && ST > 1 && kt + 1 < t.kt1;
#pragma unroll
                        for(int pr = 0; pr < NT / 2; ++pr) {
                            const int cb = pr & 1;
                            if(pr == 0) {
                                if(more) {
                                    load_a(stage, ks + 1, sa_at, f ^ 1);
                                } else if(cross) {
                                    const int n = it + 1;
                                    kc::mbarrier_wait(full + n % ST, uint32_t((n / ST) & 1));
                                    next = ws_base + (n % ST) * C::kStage;
                                    load_a(next, 0, sa_at, f ^ 1);
                                }
                            }
                            if(pr + 1 < NT / 2) load_pair(stage, ks, pr + 1, sb_at, cb ^ 1);
                            else if(more) load_pair(stage, ks + 1, 0, sb_at, 0);
                            else if(cross) load_pair(next, 0, 0, sb_at, 0);
#pragma unroll
                            for(int i = 0; i < MT; ++i)
#pragma unroll
                                for(int h = 0; h < 2; ++h)
                                    kc::mma_nvf4_m16n8k64(acc[i][2 * pr + h], qa[f][i][0], qa[f][i][1], qa[f][i][2],
                                                          qa[f][i][3], qb[cb][2 * h], qb[cb][2 * h + 1], qsa[f][i],
                                                          qsb[cb][h]);
                        }
                    }
                }
                release(it);
            }
        } else {
            for(int kt = t.kt0; kt < t.kt1; ++kt, ++it) {
                const unsigned char* stage = ws_base + (it % ST) * C::kStage;
                if(ST == 1 || kt == t.kt0) {
                    kc::mbarrier_wait(full + it % ST, uint32_t((it / ST) & 1));
                    fragments(stage, 0, sa_at, sb_at, fa[0], fb[0], sa[0], sb[0]);
                }
                const int ksv = min(KS, (kbytes - kt * BK) / 32);
    #pragma unroll
                for(int ks = 0; ks < KS; ++ks) {
                    if(ks < ksv) {
                        const int f = ks & 1;
                        if(ks + 1 < ksv) {
                            fragments(stage, ks + 1, sa_at, sb_at, fa[f ^ 1], fb[f ^ 1], sa[f ^ 1], sb[f ^ 1]);
                        } else if(ST > 1 && kt + 1 < t.kt1) {   // (a one-stage ring cannot hold the next stage yet)
                            const int n = it + 1;
                            kc::mbarrier_wait(full + n % ST, uint32_t((n / ST) & 1));
                            fragments(ws_base + (n % ST) * C::kStage, 0, sa_at, sb_at, fa[f ^ 1], fb[f ^ 1], sa[f ^ 1],
                                      sb[f ^ 1]);
                        }
    #pragma unroll
                        for(int i = 0; i < MT; ++i)
    #pragma unroll
                            for(int j = 0; j < NT; ++j)
                                kc::mma_nvf4_m16n8k64(acc[i][j], fa[f][i][0], fa[f][i][1], fa[f][i][2], fa[f][i][3],
                                                      fb[f][j][0], fb[f][j][1], sa[f][i], sb[f][j]);
                    }
                }
                release(it);
            }
        }
        if(t.nseg > 1) {
            // A tile in segments: this segment's partial to the workspace (this thread's registers, coalesced over the
            // lanes); the last segment to finish adds them in segment order (0 + p0 + p1 + ...: the same bits whichever
            // finishes last) and stores the tile.
            constexpr int R = MT * NT * 4, CT = 32 * NC;
            const int tau = t.tile - split.whole;
            float* base = split.partial + std::size_t(tau) * std::size_t(split.segs) * std::size_t(CT * R);
            float* mine = base + std::size_t(t.seg) * std::size_t(CT * R);
#pragma unroll
            for(int i = 0; i < MT; ++i)
#pragma unroll
                for(int j = 0; j < NT; ++j)
#pragma unroll
                    for(int e = 0; e < 4; ++e) __stcg(mine + std::size_t((i * NT + j) * 4 + e) * CT + tid, acc[i][j][e]);
            __threadfence();
            ws::consumers_sync(CT);
            if(tid == 0) {
                const int done = atomicAdd(split.counters + tau, 1);
                *last = done == t.nseg - 1;
            }
            ws::consumers_sync(CT);
            const bool finish = *last != 0;
            ws::consumers_sync(CT);   // every consumer has read the flag before the next unit's may change it
            if(!finish) continue;
            __threadfence();
#pragma unroll
            for(int i = 0; i < MT; ++i)
#pragma unroll
                for(int j = 0; j < NT; ++j)
#pragma unroll
                    for(int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
            for(int g = 0; g < t.nseg; ++g) {
                const float* p = base + std::size_t(g) * std::size_t(CT * R);
#pragma unroll
                for(int i = 0; i < MT; ++i)
#pragma unroll
                    for(int j = 0; j < NT; ++j)
#pragma unroll
                        for(int e = 0; e < 4; ++e)
                            acc[i][j][e] += __ldcg(p + std::size_t((i * NT + j) * 4 + e) * CT + tid);
            }
        }
        if constexpr(EPI >= 1) {
            if(maps.out_ok && t.tokens == TM && t.rows == TN) {
                // A whole tile: each warp's 16-row slices in BF16 into its staging box, one TMA store each by its lane
                // 0 (which runs while the warp goes on); EPI 2 alternates two boxes, so a slice waits only for the
                // store before last.
                constexpr int RB = C::kWTN * 2;
#pragma unroll
                for(int i = 0; i < MT; ++i) {
                    unsigned char* box = ws_base + C::kOut + (warp * EPI + (EPI == 2 ? (stores & 1) : 0)) * C::kBoxAt;
                    if(lane == 0) ws::store_wait<EPI - 1>();
                    __syncwarp();
#pragma unroll
                    for(int h = 0; h < 2; ++h) {
                        const int r = gid + 8 * h;
#pragma unroll
                        for(int j = 0; j < NT; ++j) {
                            const int c = 8 * j + 2 * tig;
                            float v0 = acc[i][j][2 * h] * scale, v1 = acc[i][j][2 * h + 1] * scale;
                            const int col = t.n0 + wn0 + c;
                            if(out.col) v0 = __fmul_rn(v0, out.col[col]), v1 = __fmul_rn(v1, out.col[col + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(box + ws::store_swizzle<RB>(uint32_t(r * RB + c * 2))) =
                                __floats2bfloat162_rn(v0, v1);
                        }
                    }
                    kc::fence_proxy_async();
                    __syncwarp();
                    if(lane == 0) {
                        kc::tma_store_2d(&maps.out, box, t.n0 + wn0, int(t.token0) + wm0 + 16 * i);
                        kc::tma_store_commit();
                    }
                    ++stores;
                }
                continue;
            }
        }
#pragma unroll
        for(int i = 0; i < MT; ++i)
#pragma unroll
            for(int h = 0; h < 2; ++h) {
                const int r = wm0 + 16 * i + gid + 8 * h;
                if(r >= t.tokens) continue;
                const std::size_t row = t.token0 + std::size_t(r);
#pragma unroll
                for(int j = 0; j < NT; ++j) {
                    const int c = wn0 + 8 * j + 2 * tig;
                    if(c < t.rows)
                        store_pair(out, row, t.n0 + c, acc[i][j][2 * h] * scale, acc[i][j][2 * h + 1] * scale,
                                   c + 1 < t.rows);
                }
            }
    }
    kc::cp_async_wait<0>();
    if constexpr(EPI >= 1) {
        if(lane == 0) ws::store_wait<0>();
    }
#else
    (void)tiles_m, (void)tiles_n, (void)acts, (void)out, (void)m, (void)maps;
    __trap();
#endif
}

// ---------------------------------------------------------------------------------------------------- quantizers

__global__ void quantize_e2m1x2_kernel(const float* x, int ld_x, int rows, int cols, float global, Acts acts) {
    const int blocks = cols / 16;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= rows * blocks) return;
    const int row = i / blocks, b = i % blocks;
    const float* v = x + std::size_t(row) * ld_x + 16 * b;
    float amax = 0.0f;
    for(int k = 0; k < 16; ++k) amax = fmaxf(amax, fabsf(v[k]));
    const float wanted = __fdiv_rn(__fmul_rn(amax, global), 6.0f);
    const float decoded = e4m3(e4m3_encode(isfinite(wanted) ? wanted : 448.0f));
    std::int8_t* out = const_cast<std::int8_t*>(acts.values) + std::size_t(row) * acts.ld + 16 * b;
    for(int k = 0; k < 16; ++k) {
        const float q = __fdiv_rn(__fmul_rn(v[k], global), decoded);
        out[k] = decoded > 0.0f && q == q ? std::int8_t(2.0f * e2m1(e2m1_encode(q))) : 0;
    }
    const_cast<float*>(acts.scales)[std::size_t(row) * acts.lds + b] = __fdiv_rn(decoded, global);
}

// The E2M1 code quantize_e2m1x2's e2m1_encode gives (round to nearest, ties to the even code, saturating at 6), from
// seven comparisons; x not NaN.
__device__ __forceinline__ unsigned e2m1_encode_fast(float x) {
    const float m = fabsf(x);
    const unsigned code = unsigned(m > 0.25f) + unsigned(m >= 0.75f) + unsigned(m > 1.25f) + unsigned(m >= 1.75f) +
                          unsigned(m > 2.5f) + unsigned(m >= 3.5f) + unsigned(m > 5.0f);
    return code | (x < 0.0f && code ? 8u : 0u);
}
// e4m3_encode's byte for a finite, non-negative x (nearest, ties to the even code, saturating at 448): the hardware
// conversion from 8.9 on, the search before.
__device__ __forceinline__ unsigned e4m3_encode_fast(float x) {
#if CRIT_KCORE_ARCH >= 890
    return unsigned(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
#else
    return e4m3_encode(x);
#endif
}

// A thread a 16-block: quantize_e2m1x2_kernel's arithmetic step for step (the same codes and the same scale before the
// division), FP32 or BF16 input (widened exactly), the codes as one 8-byte store, the scale byte row major or tiled.
template <bool BF16>
__global__ void quantize_nvfp4_kernel(const void* x, int ld_x, int rows, int cols, float global, ActsFp4 acts) {
    const int blocks = cols / 16;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= rows * blocks) return;
    const int row = i / blocks, b = i % blocks;
    float v[16];
    if constexpr(BF16) {
        const __nv_bfloat16* p = static_cast<const __nv_bfloat16*>(x) + std::size_t(row) * ld_x + 16 * b;
        if(reinterpret_cast<std::uintptr_t>(p) % 16 == 0) {
            const uint4 w0 = *reinterpret_cast<const uint4*>(p), w1 = *reinterpret_cast<const uint4*>(p + 8);
            const uint32_t w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for(int k = 0; k < 8; ++k) {
                v[2 * k] = __uint_as_float(w[k] << 16);
                v[2 * k + 1] = __uint_as_float(w[k] & 0xFFFF0000u);
            }
        } else {
#pragma unroll
            for(int k = 0; k < 16; ++k) v[k] = __bfloat162float(p[k]);
        }
    } else {
        const float* p = static_cast<const float*>(x) + std::size_t(row) * ld_x + 16 * b;
        if(reinterpret_cast<std::uintptr_t>(p) % 16 == 0) {
#pragma unroll
            for(int k = 0; k < 4; ++k) {
                const float4 f = reinterpret_cast<const float4*>(p)[k];
                v[4 * k] = f.x, v[4 * k + 1] = f.y, v[4 * k + 2] = f.z, v[4 * k + 3] = f.w;
            }
        } else {
#pragma unroll
            for(int k = 0; k < 16; ++k) v[k] = p[k];
        }
    }
    float amax = 0.0f;
#pragma unroll
    for(int k = 0; k < 16; ++k) amax = fmaxf(amax, fabsf(v[k]));
    const float wanted = __fdiv_rn(__fmul_rn(amax, global), 6.0f);
    const unsigned byte = e4m3_encode_fast(isfinite(wanted) ? wanted : 448.0f);
    const float decoded = e4m3(byte);
    uint32_t packed[2] = {0u, 0u};
#pragma unroll
    for(int k = 0; k < 16; ++k) {
        const float q = __fdiv_rn(__fmul_rn(v[k], global), decoded);
        const unsigned c = decoded > 0.0f && q == q ? e2m1_encode_fast(q) : 0u;
        packed[k / 8] |= c << (4 * (k % 8));
    }
    std::uint8_t* out = const_cast<std::uint8_t*>(acts.codes) + std::size_t(row) * acts.ld + 8 * b;
    if(reinterpret_cast<std::uintptr_t>(out) % 8 == 0) {
        *reinterpret_cast<uint2*>(out) = make_uint2(packed[0], packed[1]);
    } else {
        for(int k = 0; k < 8; ++k) out[k] = std::uint8_t(packed[k / 4] >> (8 * (k % 4)));
    }
    std::uint8_t* scales = const_cast<std::uint8_t*>(acts.scales);
    if(acts.tiled & kTiledActs)
        scales[(std::size_t(row >> 7) * std::size_t(acts.lds / 4) + std::size_t(b >> 2)) * 512 + (row & 31) * 16 +
               ((row >> 5) & 3) * 4 + (b & 3)] = std::uint8_t(byte);
    else
        scales[std::size_t(row) * acts.lds + b] = std::uint8_t(byte);
}

__global__ void quantize_int8_kernel(const float* x, int ld_x, int rows, int cols, Acts acts) {
    const int blocks = cols / acts.block;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= rows * blocks) return;
    const int row = i / blocks, b = i % blocks;
    const float* v = x + std::size_t(row) * ld_x + acts.block * b;
    float amax = 0.0f;
    for(int k = 0; k < acts.block; ++k) amax = fmaxf(amax, fabsf(v[k]));
    const float d = __fdiv_rn(amax, 127.0f);
    std::int8_t* out = const_cast<std::int8_t*>(acts.values) + std::size_t(row) * acts.ld + acts.block * b;
    for(int k = 0; k < acts.block; ++k) {
        float q = d > 0.0f ? rintf(__fdiv_rn(v[k], d)) : 0.0f;
        q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
        out[k] = std::int8_t(q);
    }
    const_cast<float*>(acts.scales)[std::size_t(row) * acts.lds + b] = d;
}

__global__ void swiglu_kernel(const float* gate_up, int inter, int rows, float* out) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= rows * inter) return;
    const int row = i / inter, k = i % inter;
    const float gate = gate_up[std::size_t(row) * 2 * inter + k], up = gate_up[std::size_t(row) * 2 * inter + inter + k];
    out[std::size_t(row) * inter + k] = kc::swiglu_declared(gate, up);
}

bool bad_acts(const Acts& a) { return !a.values || !a.scales || (a.block != 16 && a.block != 32) || a.ld % 4; }

}  // namespace

namespace {

template <int THREADS, int ROWS, int STREAM, int UNROLL>
void launch_rows(const Matrix* matrices, const RowJob* jobs, int job_count, int max_rows, const Acts& acts, float* out,
                 int out_ld, cudaStream_t stream) {
    constexpr int per_cta = THREADS / 16 * ROWS;
    const dim3 grid(unsigned((max_rows + per_cta - 1) / per_cta), unsigned(job_count));
    kc::declared::int_rows_kernel<THREADS, ROWS, STREAM, UNROLL, kMaxJobTokens, Matrix, Acts, RowJob>
        <<<grid, THREADS, 0, stream>>>(matrices, jobs, acts, out, out_ld);
}

template <int WARPS, int TILES, int TOKZ>
void launch_gemm(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                 const Acts& acts, float* out, int out_ld, cudaStream_t stream) {
    constexpr int rows_per_cta = 8 * TILES * WARPS;
    // Token tiles of 16 across grid.z (TOKZ a block), so a long chunk spreads over every SM.
    const int tiles = ((max_count + 15) / 16 + TOKZ - 1) / TOKZ;
    const dim3 grid(unsigned((max_rows + rows_per_cta - 1) / rows_per_cta), unsigned(job_count),
                    unsigned(tiles < 65535 ? tiles : 65535));
    gemm_kernel<WARPS, TILES><<<grid, 32 * WARPS, 0, stream>>>(matrices, jobs, acts, out, out_ld);
}

template <int BM, int BN, int WM, int WN, int G, int RUN, int STAGES>
cudaError_t launch_staged(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                          const Acts& acts, float* out, int out_ld, cudaStream_t stream) {
    using S = Staged<BM, BN, WM, WN, G, RUN, STAGES>;
    auto kernel = staged_kernel<BM, BN, WM, WN, G, RUN, STAGES>;
    static const cudaError_t attribute =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSmem);
    if(attribute != cudaSuccess) {   // more shared memory than the device gives a CTA
        (void)cudaGetLastError();
        return attribute;
    }
    const int tiles = (max_count + BN - 1) / BN;
    const dim3 grid(unsigned((max_rows + BM - 1) / BM), unsigned(job_count), unsigned(tiles < 65535 ? tiles : 65535));
    kernel<<<grid, S::kThreads, S::kSmem, stream>>>(matrices, jobs, acts, out, out_ld);
    return cudaSuccess;
}

template <int TM, int TN, int WM, int WN, int BK, int STAGES, int GROUP, int EPI, bool NATIVE>
cudaError_t launch_native(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                          const ActsFp4& acts, const NativeOut& out, cudaStream_t stream) {
    using C = Native<TM, TN, WM, WN, BK, STAGES, GROUP>;
    auto kernel = native_kernel<TM, TN, WM, WN, BK, STAGES, GROUP, EPI, NATIVE>;
    static const cudaError_t attribute =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem);
    if(attribute != cudaSuccess) {   // more shared memory than the device gives a CTA
        (void)cudaGetLastError();
        return attribute;
    }
    // Persistent CTAs: at most as many a job as the device holds at once, each taking tiles gridDim.x apart.
    static const int resident = [kernel] {
        int device = 0, sms = 0, per = 0;
        (void)cudaGetDevice(&device);
        (void)cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, C::kThreads, C::kSmem);
        (void)cudaGetLastError();
        return sms * (per > 0 ? per : 1);
    }();
    const long long tiles = (long long)((max_count + TM - 1) / TM) * ((max_rows + TN - 1) / TN);
    if(tiles > 0x7fffffffLL) return cudaErrorInvalidValue;
    const dim3 grid{unsigned(tiles < resident ? tiles : resident), unsigned(job_count), 1u};
    kernel<<<grid, C::kThreads, C::kSmem, stream>>>(matrices, jobs, acts, out);
    return cudaSuccess;
}

int device_sms() {
    int device = 0, sms = 0;
    (void)cudaGetDevice(&device);
    if(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) (void)cudaGetLastError();
    return sms > 0 ? sms : 1;
}

// Whether the current device runs the warp-specialized block-scaled path (compute capability 12.x: crit builds it as
// sm_120a / sm_121a, whose block-scaled MMAs the path needs).
bool ws_device() {
    static int major[64] = {};
    int device = 0;
    if(cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return false;
    if(!major[device] && cudaDeviceGetAttribute(&major[device], cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        return false;
    return major[device] == 12;
}

// A launch's tile plan (WsSplit) and workspace bytes, from max_count, max_rows, K and the CTAs the device holds at once
// (`resident`): SPLIT 1 takes every tile whole; SPLIT 0 (stream-K) shares the tiles past the last whole wave over the
// CTAs, each 8 k-tiles or more, where that gives more CTAs than tiles (one job a launch; several take tiles whole).
// SPLIT 2 shares every launch's last partial wave; SPLIT 0 only where the tiles fill at most half the CTAs and K is
// 64 stages or more: elsewhere the segments'
// partials cost more than the idle CTAs (RTX 5090: 5120 x 17408 at 128 tokens 65 us shared, 76 whole; 10240 x 5120 at
// 128 tokens 48 shared, 34 whole; 5120 x 17408 at 512 tokens 115 shared, 102 whole).
struct WsPlan {
    WsSplit split;
    int grid = 0;
    std::size_t counter_bytes = 0, bytes = 0;
};
WsPlan ws_plan(int tm, int tn, int bk, int split_knob, int job_count, int max_rows, int max_count, int kbytes,
               int resident) {
    WsPlan p;
    WsSplit& s = p.split;
    s.tiles_m = (max_count + tm - 1) / tm;
    s.tiles_n = (max_rows + tn - 1) / tn;
    const int tiles = s.tiles_m * s.tiles_n, kt = (kbytes + bk - 1) / bk;
    s.whole = tiles;
    const bool share = (split_knob == 0 && 2 * tiles <= resident && kt >= 64) || (split_knob == 2 && tiles % resident);
    if(job_count == 1 && resident > 0 && share) {
        const int tail = tiles % resident;
        const long long work = (long long)tail * kt;
        const int ctas = int(work / 8 < resident ? work / 8 : resident);
        if(ctas > tail) {
            s.whole = tiles - tail;
            s.tail = tail;
            s.ctas = ctas;
            s.segs = (ctas + tail - 1) / tail + 1;
            p.counter_bytes = (std::size_t(tail) * sizeof(int) + 15) / 16 * 16;
            p.bytes = p.counter_bytes + std::size_t(tail) * s.segs * std::size_t(tm) * tn * sizeof(float);
        }
    }
    p.grid = s.whole > 0 ? (s.whole < resident ? s.whole : resident) : s.ctas;
    if(s.ctas > p.grid) p.grid = s.ctas;
    return p;
}

// A tensor map over a row-major byte matrix (rows x row_bytes, rows row_bytes apart): boxes of box_rows x BK bytes,
// swizzled as tile.cuh's offset<BK> (TMA's 64B / 128B modes), out-of-bounds bytes zero. Weight matrices' maps are kept
// (keyed by everything the map holds), the activations' made per launch.
template <int BK>
bool byte_map(CUtensorMap* map, const void* base, int rows, int row_bytes, int box_rows) {
    return kc::make_tensor_map_2d(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, base, uint64_t(rows), uint64_t(row_bytes),
                                  uint64_t(row_bytes), uint32_t(box_rows), uint32_t(BK),
                                  BK == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B) == CUDA_SUCCESS;
}
template <int BK, int TN>
bool weight_map(CUtensorMap* map, const HostMatrix& h) {
    struct Key {
        const void* codes;
        int rows, cols;
    };
    struct Entry {
        Key key;
        CUtensorMap map;
    };
    static std::mutex lock;
    static std::vector<Entry> cache;
    std::lock_guard<std::mutex> hold(lock);
    for(const Entry& e : cache)
        if(e.key.codes == h.codes && e.key.rows == h.rows && e.key.cols == h.cols) {
            *map = e.map;
            return true;
        }
    Entry e{{h.codes, h.rows, h.cols}, {}};
    if(!byte_map<BK>(&e.map, h.codes, h.rows, h.cols / 2, TN)) return false;
    if(cache.size() >= 4096) cache.erase(cache.begin());
    cache.push_back(e);
    *map = e.map;
    return true;
}

// The configuration's kernel with its shared memory set, and the CTAs the device holds at once (0: it cannot run).
template <int TM, int TN, int WM, int WN, int BK, int STAGES, int GROUP, int EPI>
int ws_resident() {
    using C = Ws<TM, TN, WM, WN, BK, STAGES, EPI>;
    static const int resident = [] {
        auto kernel = ws_kernel<TM, TN, WM, WN, BK, STAGES, GROUP, EPI>;
        if(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem) != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
        int per = 0;
        (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, C::kThreads, C::kSmem);
        (void)cudaGetLastError();
        return device_sms() * (per > 0 ? per : 1);
    }();
    return resident;
}

template <int TM, int TN, int WM, int WN, int BK, int STAGES, int GROUP, int EPI>
cudaError_t launch_ws(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                      const HostMatrix* host, int host_count, const ActsFp4& acts, const NativeOut& out, int split_knob,
                      void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
    using C = Ws<TM, TN, WM, WN, BK, STAGES, EPI>;
    auto kernel = ws_kernel<TM, TN, WM, WN, BK, STAGES, GROUP, EPI>;
    const int resident = ws_resident<TM, TN, WM, WN, BK, STAGES, GROUP, EPI>();
    if(resident <= 0) return cudaErrorInvalidConfiguration;
    const WsPlan plan = ws_plan(TM, TN, BK, split_knob, job_count, max_rows, max_count, acts.ld, resident);
    WsSplit split = plan.split;
    if(plan.bytes) {
        if(!workspace || workspace_bytes < plan.bytes || reinterpret_cast<std::uintptr_t>(workspace) % 16)
            return cudaErrorInvalidValue;
        split.counters = static_cast<int*>(workspace);
        split.partial = reinterpret_cast<float*>(static_cast<unsigned char*>(workspace) + plan.counter_bytes);
        const cudaError_t e = cudaMemsetAsync(workspace, 0, plan.counter_bytes, stream);
        if(e != cudaSuccess) return e;
    }
    WsMaps maps{};
    maps.count = 0;
    if(host && host_count > 0 && host_count <= kMaxMaps && acts.rows > 0 &&
       byte_map<BK>(&maps.a, acts.codes, acts.rows, acts.ld, TM)) {
        int made = 0;
        while(made < host_count && host[made].codes && host[made].cols % 64 == 0 &&
              reinterpret_cast<std::uintptr_t>(host[made].codes) % 16 == 0 &&
              weight_map<BK, TN>(&maps.w[made], host[made]))
            ++made;
        maps.count = made == host_count ? made : 0;
    }
    maps.out_ok = 0;
    if(EPI >= 1 && out.bf16 && out.rows > 0 && out.ld % 8 == 0 && reinterpret_cast<std::uintptr_t>(out.data) % 16 == 0)
        maps.out_ok = kc::make_tensor_map_2d(&maps.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, out.data, uint64_t(out.rows),
                                             uint64_t(out.ld), uint64_t(out.ld) * 2, 16u, uint32_t(C::kWTN),
                                             C::kWTN == 64   ? CU_TENSOR_MAP_SWIZZLE_128B
                                             : C::kWTN == 32 ? CU_TENSOR_MAP_SWIZZLE_64B
                                                             : CU_TENSOR_MAP_SWIZZLE_32B) == CUDA_SUCCESS;
    if((long long)split.tiles_m * split.tiles_n > 0x7fffffffLL || plan.grid <= 0) return cudaErrorInvalidValue;
    const dim3 grid{unsigned(plan.grid), unsigned(job_count), 1u};
    kernel<<<grid, C::kThreads, C::kSmem, stream>>>(matrices, jobs, acts, out, split, maps);
    return cudaSuccess;
}

}  // namespace

// Each enqueue runs the configuration crit_tuning.h's table names for the matrix shape (the largest matrix's rows, the
// activations' width) and row count (0, the defaults, where it names none).
cudaError_t enqueue_rows(const Matrix* matrices, const RowJob* jobs, int job_count, int max_rows, const Acts& acts,
                         float* out, int out_ld, cudaStream_t stream) {
    if(!matrices || !jobs || job_count < 0 || max_rows < 0 || !out || bad_acts(acts)) return cudaErrorInvalidValue;
    if(job_count == 0 || max_rows == 0) return cudaSuccess;
    switch(crit_lookup_rows(max_rows, acts.ld, acts.rows > 0 ? acts.rows : job_count)) {
#define CRIT_IB_ROWS(index, threads, rows, stream_, unroll)                                                     \
    case index:                                                                                                  \
        launch_rows<threads, rows, stream_, unroll>(matrices, jobs, job_count, max_rows, acts, out, out_ld, stream); \
        break;
        CRIT_CONFIGS_rows(CRIT_IB_ROWS)
#undef CRIT_IB_ROWS
        default:
            return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}

cudaError_t enqueue_gemm(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                         const Acts& acts, float* out, int out_ld, cudaStream_t stream) {
    if(!matrices || !jobs || job_count < 0 || max_rows < 0 || max_count < 0 || !out || bad_acts(acts))
        return cudaErrorInvalidValue;
    if(job_count == 0 || max_rows == 0 || max_count == 0) return cudaSuccess;
    switch(crit_lookup_gemm(max_rows, acts.ld, max_count)) {
#define CRIT_IB_GEMM(index, warps, tiles, tokz)                                                                     \
    case index:                                                                                                      \
        launch_gemm<warps, tiles, tokz>(matrices, jobs, job_count, max_rows, max_count, acts, out, out_ld, stream); \
        break;
        CRIT_CONFIGS_gemm(CRIT_IB_GEMM)
#undef CRIT_IB_GEMM
        default:
            return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}

cudaError_t enqueue_gemm_staged(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows,
                                int max_count, const Acts& acts, float* out, int out_ld, cudaStream_t stream) {
    if(!matrices || !jobs || job_count < 0 || max_rows < 0 || max_count < 0 || !out || bad_acts(acts) ||
       acts.ld % 16 || reinterpret_cast<std::uintptr_t>(acts.values) % 16)
        return cudaErrorInvalidValue;
    if(job_count == 0 || max_rows == 0 || max_count == 0) return cudaSuccess;
    cudaError_t e = cudaErrorInvalidConfiguration;
    switch(crit_lookup_gemm_staged(max_rows, acts.ld, max_count)) {
#define CRIT_IB_STAGED(index, bm, bn, wm, wn, group, run, stages)                                                \
    case index:                                                                                                \
        e = launch_staged<bm, bn, wm, wn, group, run, stages>(matrices, jobs, job_count, max_rows, max_count, acts, \
                                                              out, out_ld, stream);                            \
        break;
        CRIT_CONFIGS_gemm_staged(CRIT_IB_STAGED)
#undef CRIT_IB_STAGED
        default:
            break;
    }
    return e != cudaSuccess ? e : cudaGetLastError();
}

cudaError_t enqueue_quantize_e2m1x2(const float* x, int ld_x, int rows, int cols, float global, const Acts& acts,
                                    cudaStream_t stream) {
    if(!x || rows < 0 || cols % 16 || !(global > 0.0f) || bad_acts(acts)) return cudaErrorInvalidValue;
    const int n = rows * (cols / 16);
    if(n == 0) return cudaSuccess;
    quantize_e2m1x2_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, ld_x, rows, cols, global, acts);
    return cudaGetLastError();
}

namespace {
bool bad_native(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows, int max_count,
                const ActsFp4& acts) {
    return !matrices || !jobs || job_count < 0 || max_rows < 0 || max_count < 0 || !acts.codes || !acts.scales ||
           acts.ld % 16 || reinterpret_cast<std::uintptr_t>(acts.codes) % 16 || !(acts.global > 0.0f) ||
           ((acts.tiled & kTiledActs) && (acts.lds % 4 || reinterpret_cast<std::uintptr_t>(acts.scales) % 16));
}
}  // namespace

std::size_t gemm_native_workspace(int job_count, int max_rows, int max_count, const ActsFp4& acts, bool int4) {
    if(int4 || !ws_device() || job_count <= 0 || max_rows <= 0 || max_count <= 0) return 0;
    switch(crit_lookup_gemm_native(max_rows, 2 * acts.ld, max_count)) {
#define CRIT_IB_PLAN(index, tm, tn, wm, wn, bk, stages, group, epi, split_) \
    case index:                                                              \
        return ws_plan(tm, tn, bk, split_, job_count, max_rows, max_count, acts.ld,                        \
                       ws_resident<tm, tn, wm, wn, bk, stages, group, epi>()).bytes;
        CRIT_CONFIGS_gemm_native(CRIT_IB_PLAN)
#undef CRIT_IB_PLAN
        default:
            return 0;
    }
}

cudaError_t enqueue_gemm_native(const Matrix* matrices, const GemmJob* jobs, int job_count, int max_rows,
                                int max_count, const ActsFp4& acts, bool int4, const NativeOut& out, void* workspace,
                                std::size_t workspace_bytes, cudaStream_t stream, const HostMatrix* host,
                                int host_count) {
    if(bad_native(matrices, jobs, job_count, max_rows, max_count, acts) || !out.data || out.ld < 0)
        return cudaErrorInvalidValue;
    if(job_count == 0 || max_rows == 0 || max_count == 0) return cudaSuccess;
    const bool ws = !int4 && ws_device();
    // Tiled scale bytes: the warp-specialized path's copies only.
    if(!ws && acts.tiled) return cudaErrorInvalidValue;
    // INT4 weights: the generic path in one fixed configuration (the fallback; the tuned configurations are the
    // block-scaled path's, and the generic path's on architectures without block-scaled MMAs).
    if(int4) {
        const cudaError_t e = launch_native<64, 128, 2, 2, 128, 3, 8, 0, false>(matrices, jobs, job_count, max_rows,
                                                                             max_count, acts, out, stream);
        return e != cudaSuccess ? e : cudaGetLastError();
    }
    cudaError_t e = cudaErrorInvalidConfiguration;
    switch(crit_lookup_gemm_native(max_rows, 2 * acts.ld, max_count)) {
#define CRIT_IB_NATIVE(index, tm, tn, wm, wn, bk, stages, group, epi, split_)                                        \
    case index:                                                                                                    \
        if(ws)                                                                                                     \
            e = launch_ws<tm, tn, wm, wn, bk, stages, group, epi>(                                                    \
                matrices, jobs, job_count, max_rows, max_count, host, host_count, acts, out, split_, workspace,    \
                workspace_bytes, stream);                                                                          \
        else                                                                                                       \
            e = launch_native<tm, tn, wm, wn, bk, (stages < 3 ? 3 : stages > 4 ? 4 : stages), group, epi, true>(  \
                matrices, jobs, job_count, max_rows, max_count, acts, out, stream);                                \
        break;
        CRIT_CONFIGS_gemm_native(CRIT_IB_NATIVE)
#undef CRIT_IB_NATIVE
        default:
            break;
    }
    return e != cudaSuccess ? e : cudaGetLastError();
}

cudaError_t enqueue_quantize_nvfp4(const void* x, bool bf16, int ld_x, int rows, int cols, float global,
                                   const ActsFp4& acts, cudaStream_t stream) {
    if(!x || rows < 0 || cols % 16 || !(global > 0.0f) || !acts.codes || !acts.scales ||
       ((acts.tiled & kTiledActs) && acts.lds % 4))
        return cudaErrorInvalidValue;
    const int n = rows * (cols / 16);
    if(n == 0) return cudaSuccess;
    if(bf16)
        quantize_nvfp4_kernel<true><<<(n + 255) / 256, 256, 0, stream>>>(x, ld_x, rows, cols, global, acts);
    else
        quantize_nvfp4_kernel<false><<<(n + 255) / 256, 256, 0, stream>>>(x, ld_x, rows, cols, global, acts);
    return cudaGetLastError();
}

cudaError_t enqueue_quantize_int8(const float* x, int ld_x, int rows, int cols, const Acts& acts, cudaStream_t stream) {
    if(!x || rows < 0 || bad_acts(acts) || cols % acts.block) return cudaErrorInvalidValue;
    const int n = rows * (cols / acts.block);
    if(n == 0) return cudaSuccess;
    quantize_int8_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, ld_x, rows, cols, acts);
    return cudaGetLastError();
}

cudaError_t enqueue_swiglu(const float* gate_up, int inter, int rows, float* out, cudaStream_t stream) {
    if(!gate_up || !out || inter < 0 || rows < 0) return cudaErrorInvalidValue;
    const int n = rows * inter;
    if(n == 0) return cudaSuccess;
    swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gate_up, inter, rows, out);
    return cudaGetLastError();
}

}  // namespace crit::techniques::int_block
