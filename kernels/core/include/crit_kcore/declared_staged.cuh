// crit's kernel core: the declared block order's staged prompt GEMM, its mainloop once for every format
// (kernels/core/README.md, "declared_staged").
//
// A CTA computes BM weight rows x BN tokens. The blocks are visited lane by lane in the tree's order, G lanes and RUN
// blocks of each a visit (blockscaled_order.cuh's GroupVisits); each visit is one stage of a STAGES-deep cp.async ring
// (load.cuh), the multistage mainloop CUTLASS uses (BSD-3-Clause; the mechanism, no code). One stage ahead of the MMAs
// the CTA prepares the stage (the technique's decode: weight scales to FP32 in pair_slot order, codes to the MMA's
// operand type); then a warp computes BM / WM rows x BN / WN tokens: the technique's block sums of its 16 x 8 tiles
// (one MMA a tile from a zero accumulator, or onto an exact-sum bias), the declared step on its lane
// (blockscaled_mma.cuh's step: lane = fma(sum, weight scale x activation scale, lane)), the G lanes folded into the
// tree (reduce.cuh's fold16) when their last block is done. Every output has the declared order's bits, whatever the
// knobs: the same as the format's declared rows.
//
// A technique supplies the stage layout (StagedShape's SUB: bytes of one block's sub-stage, with the decoded weight
// scales at WS and the token scales at AS) and three device callables:
//   issue(unsigned char* sub, int block)   this thread's asynchronous copies of `block` into a sub-stage
//   decode(unsigned char* sub, int block)  this thread's conversions of a landed sub-stage (between the barriers)
//   sums(const unsigned char* sub, float (&d)[kMT][kNT][4])  the warp's block sums of its tiles, as FP32 values
#pragma once
#include "crit_kcore/blockscaled_mma.cuh"
#include "crit_kcore/blockscaled_order.cuh"
#include "crit_kcore/blockscaled_stage.cuh"
#include "crit_kcore/load.cuh"
#include "crit_kcore/reduce.cuh"

#include <cstddef>

namespace crit::kcore {

template <int BM, int BN, int WM, int WN, int G, int RUN, int STAGES, int SUB, int WS, int AS>
struct StagedShape {
    static_assert(STAGES >= 3, "a stage is decoded one iteration before its MMAs");
    static constexpr int kThreads = 32 * WM * WN, kWarpRows = BM / WM, kWarpTokens = BN / WN;
    static constexpr int kMT = kWarpRows / 16, kNT = kWarpTokens / 8, kOut = 4 * kMT * kNT;
    static_assert(kMT >= 1 && kNT >= 1 && kWarpRows % 16 == 0 && kWarpTokens % 8 == 0, "warp tiles of 16 x 8");
    static constexpr int kRowsPer = (BM + kThreads - 1) / kThreads, kTokensPer = (BN + kThreads - 1) / kThreads;
    static constexpr int kSub = SUB, kWs = WS, kAs = AS, kStage = G * RUN * kSub, kSmem = STAGES * kStage;
    static constexpr int kG = G, kRun = RUN, kStages = STAGES;
    // CTAs an SM can hold by registers (the launch bound): an output holds G running lanes and the partial sums still
    // needed (5 registers, 6 with G = 4), plus about 48 a thread of fragments and addresses.
    static constexpr int kRegs = (kOut * (G == 4 ? 6 : 5) + 48 + 7) / 8 * 8;
    static constexpr int kFit = 65536 / (kThreads * kRegs), kMinBlocks = kFit < 1 ? 1 : kFit > 4 ? 4 : kFit;
};

// One token tile's declared sums: x[o][0] the output o of this thread (o = 4 (mt kNT + nt) + c, c the MMA accumulator
// register), after `blocks` blocks of K. wr, wt: the warp's first row and token in the CTA tile; g, q: the lane's group
// and thread in its quad. The caller's barrier after the call frees the stages for the next tile.
template <class S, class Issue, class Decode, class Sums>
__device__ __forceinline__ void declared_staged(unsigned char* smem, int blocks, int wr, int wt, int g, int q,
                                                Issue issue_block, Decode decode_block, Sums sums,
                                                float (&x)[S::kOut][4]) {
    constexpr int G = S::kG, RUN = S::kRun, STAGES = S::kStages;
    using Visits = blockscaled::GroupVisits<G, RUN>;
    // A visit's copies into stage `slot` (a lane without a block copies the first lane's: never used).
    auto issue = [&](int slot, const Visits& it) {
#pragma unroll
        for(int i = 0; i < G * RUN; ++i) {
            const int s = i % G, r = i / G;
            const int b = it.block(s, r) >= 0 ? it.block(s, r) : it.block(0);
            issue_block(smem + slot * S::kStage + i * S::kSub, b);
        }
    };
    auto decode = [&](int slot, const Visits& it) {
#pragma unroll
        for(int i = 0; i < G * RUN; ++i) {
            const int b = it.block(i % G, i / G);
            if(b < 0) continue;
            decode_block(smem + slot * S::kStage + i * S::kSub, b);
        }
    };

    Visits producer(blocks), decoder(blocks);
#pragma unroll
    for(int s = 0; s < STAGES - 1; ++s) {
        if(!producer.done()) {
            issue(s, producer);
            producer.next();
        }
        cp_async_commit();
    }
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    decode(0, decoder);
    decoder.next();

    float acc[G][S::kOut];
    int v = 0;
    // G lanes at a time in the tree's order: their blocks (a lane past the last block has none: the tree then takes
    // +0.0f), RUN of each a visit of the ring, then the lanes folded into the partial sums.
#pragma unroll
    for(int at = 0; at < 16; at += G) {
#pragma unroll
        for(int s = 0; s < G; ++s)
#pragma unroll
            for(int o = 0; o < S::kOut; ++o) acc[s][o] = 0.0f;
        const int count = Visits::count(at, blocks);
        for(int k0 = 0; k0 < count; k0 += RUN, ++v) {
            cp_async_wait<STAGES - 3>();
            __syncthreads();
            if(!decoder.done()) {
                decode((v + 1) % STAGES, decoder);
                decoder.next();
            }
            if(!producer.done()) {
                issue((v + STAGES - 1) % STAGES, producer);
                producer.next();
            }
            cp_async_commit();
            // One block of each lane at a time (not unrolled: the fragments of one in registers at once).
#pragma unroll 1
            for(int r = 0; r < RUN; ++r)
#pragma unroll
            for(int s = 0; s < G; ++s) {
                if(blockscaled::lane_at(at + s) + 16 * (k0 + r) >= blocks) continue;
                const unsigned char* sub = smem + (v % STAGES) * S::kStage + (r * G + s) * S::kSub;
                const float* ws = reinterpret_cast<const float*>(sub + S::kWs);
                const float* as = reinterpret_cast<const float*>(sub + S::kAs);
                float2 w[S::kMT], a[S::kNT];
#pragma unroll
                for(int mt = 0; mt < S::kMT; ++mt)
                    w[mt] = *reinterpret_cast<const float2*>(ws + blockscaled::pair_slot(wr + 16 * mt + g));
#pragma unroll
                for(int nt = 0; nt < S::kNT; ++nt) a[nt] = *reinterpret_cast<const float2*>(as + wt + 8 * nt + 2 * q);
                // The block's MMAs first, then their steps: the sums of all tiles in flight together.
                float d[S::kMT][S::kNT][4];
                sums(sub, d);
#pragma unroll
                for(int mt = 0; mt < S::kMT; ++mt)
#pragma unroll
                    for(int nt = 0; nt < S::kNT; ++nt) {
                        float* c = acc[s] + 4 * (mt * S::kNT + nt);
                        c[0] = blockscaled::step(d[mt][nt][0], w[mt].x, a[nt].x, c[0]);
                        c[1] = blockscaled::step(d[mt][nt][1], w[mt].x, a[nt].y, c[1]);
                        c[2] = blockscaled::step(d[mt][nt][2], w[mt].y, a[nt].x, c[2]);
                        c[3] = blockscaled::step(d[mt][nt][3], w[mt].y, a[nt].y, c[3]);
                    }
            }
        }
#pragma unroll
        for(int s = 0; s < G; ++s)
#pragma unroll
            for(int o = 0; o < S::kOut; ++o) fold16(at + s, acc[s][o], x[o]);
    }
}

// The tile's outputs, out[first + t][r] = x x post for the CTA's rows r < rows and tokens t < count (row0, tile: the
// CTA tile's first row and token).
template <class S>
__device__ __forceinline__ void declared_staged_store(const float (&x)[S::kOut][4], float* out, int out_ld, int row0,
                                                      int tile, int rows, int first, int count, float post, int wr,
                                                      int wt, int g, int q) {
#pragma unroll
    for(int mt = 0; mt < S::kMT; ++mt)
#pragma unroll
        for(int nt = 0; nt < S::kNT; ++nt)
#pragma unroll
            for(int o = 0; o < 4; ++o) {
                const int r = row0 + wr + 16 * mt + g + 8 * (o / 2), t = tile + wt + 8 * nt + 2 * q + o % 2;
                if(r < rows && t < count)
                    out[std::size_t(first + t) * std::size_t(out_ld) + std::size_t(r)] =
                        __fmul_rn(x[4 * (mt * S::kNT + nt) + o][0], post);
            }
}

}  // namespace crit::kcore
