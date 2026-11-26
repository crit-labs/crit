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
// The visits are planned once a CTA (declared_staged_plan: each visit's first block in a table at the front of the
// shared memory, the producer and the decoder read theirs), a block's sums and steps go one 16-row tile at a time (the
// tile's block sums live only until their steps), and the ring's slots advance by compares: the mainloop's own work
// a block is the technique's copies, its fragment reads, the MMAs and the steps.
//
// A technique supplies the stage layout (StagedShape's SUB: bytes of one block's sub-stage, with the decoded weight
// scales at WS and the token scales at AS; a visit's stage is its G RUN sub-stages, slot i = G r + s the block
// staged_block(first, i), and EXTRA bytes after them for what the technique keeps a visit at once, such as each
// token's blocks of the visit side by side) and four device callables:
//   issue(unsigned char* stage, int first)   this thread's asynchronous copies of a visit (first: its first block)
//   decode(unsigned char* stage, int first)  this thread's conversions of a landed visit (between the barriers)
//   tokens(const unsigned char* stage, int i)  the warp's token fragments of slot i's block (the technique's type)
//   sums(const unsigned char* stage, int i, int mt, const T& tokens, float (&d)[kNT][4])  row tile mt's block sums
#pragma once
#include "crit_kcore/blockscaled_mma.cuh"
#include "crit_kcore/blockscaled_order.cuh"
#include "crit_kcore/blockscaled_stage.cuh"
#include "crit_kcore/load.cuh"
#include "crit_kcore/reduce.cuh"

#include <cstddef>
#include <cstdint>

namespace crit::kcore {

template <int BM, int BN, int WM, int WN, int G, int RUN, int STAGES, int SUB, int WS, int AS, int EXTRA = 0>
struct StagedShape {
    static_assert(STAGES >= 3, "a stage is decoded one iteration before its MMAs");
    static constexpr int kThreads = 32 * WM * WN, kWarpRows = BM / WM, kWarpTokens = BN / WN;
    static constexpr int kMT = kWarpRows / 16, kNT = kWarpTokens / 8, kOut = 4 * kMT * kNT;
    static_assert(kMT >= 1 && kNT >= 1 && kWarpRows % 16 == 0 && kWarpTokens % 8 == 0, "warp tiles of 16 x 8");
    static constexpr int kRowsPer = (BM + kThreads - 1) / kThreads, kTokensPer = (BN + kThreads - 1) / kThreads;
    // The plan: one 16-bit first block a visit, at most kVisits (a dot of up to 16 (kVisits / (16 / G) - 1) RUN
    // blocks), in front of the ring.
    static constexpr int kVisits = 1024, kPlan = 2 * kVisits;
    static constexpr int kSub = SUB, kWs = WS, kAs = AS, kStage = G * RUN * kSub + EXTRA, kSmem = kPlan + STAGES * kStage;
    static constexpr int kG = G, kRun = RUN, kStages = STAGES;
    // CTAs an SM can hold by registers (the launch bound): an output holds G running lanes and the partial sums still
    // needed (5 registers, 6 with G = 4), plus about 48 a thread of fragments and addresses.
    static constexpr int kRegs = (kOut * (G == 4 ? 6 : 5) + 48 + 7) / 8 * 8;
    static constexpr int kFit = 65536 / (kThreads * kRegs), kMinBlocks = kFit < 1 ? 1 : kFit > 4 ? 4 : kFit;
    // The most blocks a dot may have (the plan's size), for the launch's check.
    static constexpr int kMaxBlocks = 16 * (kVisits / (16 / G) - 1) * RUN;
};

// Slot i of the visit whose first block is `first`: block first + lane_at(i % G) + 16 (i / G), that lane's block
// (i % G) and of its run (i / G); at or past `blocks` the lane has no such block.
template <class S>
__device__ __forceinline__ int staged_block(int first, int i) {
    return first + blockscaled::lane_at(i % S::kG) + 16 * (i / S::kG);
}

// The visits of a group of G lanes starting at tree step `at`: the first lane (the one with the most blocks) has
// count() blocks, RUN a visit.
template <int G, int RUN>
__host__ __device__ constexpr int group_visits(int at, int blocks) {
    return (blockscaled::GroupVisits<G, RUN>::count(at, blocks) + RUN - 1) / RUN;
}

// The plan of a dot of `blocks` blocks: visit v's first block (lane lane_at(at) block 16 k0, its group's first lane;
// slot (s, r) of the visit is that + lane_at(s) + 16 r, where below `blocks`), in the consumer's order, written by the
// CTA's threads; then a barrier. Returns the visits.
template <class S>
__device__ __forceinline__ int declared_staged_plan(unsigned char* smem, int blocks) {
    constexpr int G = S::kG, RUN = S::kRun;
    std::int16_t* plan = reinterpret_cast<std::int16_t*>(smem);
    int total = 0;
#pragma unroll
    for(int at = 0; at < 16; at += G) {
        const int n = group_visits<G, RUN>(at, blocks);
        for(int k = int(threadIdx.x); k < n; k += S::kThreads)
            plan[total + k] = std::int16_t(blockscaled::lane_at(at) + 16 * RUN * k);
        total += n;
    }
    __syncthreads();
    return total;
}

// One token tile's declared sums: x[o][0] the output o of this thread (o = 4 (mt kNT + nt) + c, c the MMA accumulator
// register), after `blocks` blocks of K (`visits` of them, declared_staged_plan's). wr, wt: the warp's first row and
// token in the CTA tile; g, q: the lane's group and thread in its quad. The caller's barrier after the call frees the
// stages for the next tile.
//
// uniform: every row of the CTA has the same weight scale (the technique's decode writes it at ws[0]): a block's scale
// products are then one a token (the same values the per-output products would round to), not one an output.
template <class S, class Issue, class Decode, class Tokens, class Sums>
__device__ __forceinline__ void declared_staged(unsigned char* smem, int blocks, int visits, bool uniform, int wr,
                                                int wt, int g, int q, Issue issue_block, Decode decode_block,
                                                Tokens tokens, Sums sums, float (&x)[S::kOut][4]) {
    constexpr int G = S::kG, RUN = S::kRun, STAGES = S::kStages;
    const std::int16_t* plan = reinterpret_cast<const std::int16_t*>(smem);
    unsigned char* ring = smem + S::kPlan;
    auto issue = [&](int slot, int v) { issue_block(ring + slot * S::kStage, int(plan[v])); };
    auto decode = [&](int slot, int v) { decode_block(ring + slot * S::kStage, int(plan[v])); };

#pragma unroll
    for(int s = 0; s < STAGES - 1; ++s) {
        if(s < visits) issue(s, s);
        cp_async_commit();
    }
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    decode(0, 0);

    // The ring's slots of the visit computed (slot), decoded (slot + 1) and issued (slot - 1), modulo STAGES.
    int slot = 0, v = 0;
    float acc[G][S::kOut];
    // G lanes at a time in the tree's order: their blocks (a lane past the last block has none: the tree then takes
    // +0.0f), RUN of each a visit of the ring, then the lanes folded into the partial sums.
#pragma unroll
    for(int at = 0; at < 16; at += G) {
#pragma unroll
        for(int s = 0; s < G; ++s)
#pragma unroll
            for(int o = 0; o < S::kOut; ++o) acc[s][o] = 0.0f;
        const int n = group_visits<G, RUN>(at, blocks);
        for(int k = 0; k < n; ++k, ++v) {
            const int next = slot + 1 == STAGES ? 0 : slot + 1, prev = slot == 0 ? STAGES - 1 : slot - 1;
            cp_async_wait<STAGES - 3>();
            __syncthreads();
            if(v + 1 < visits) decode(next, v + 1);
            if(v + STAGES - 1 < visits) issue(prev, v + STAGES - 1);
            cp_async_commit();
            const unsigned char* stage = ring + slot * S::kStage;
            slot = next;
            // One block of each lane at a time (not unrolled: the fragments of one in registers at once).
#pragma unroll 1
            for(int r = 0; r < RUN; ++r)
#pragma unroll
            for(int s = 0; s < G; ++s) {
                if(blockscaled::lane_at(at + s) + 16 * (RUN * k + r) >= blocks) continue;
                const int i = r * G + s;
                const unsigned char* sub = stage + i * S::kSub;
                const float* ws = reinterpret_cast<const float*>(sub + S::kWs);
                const float* as = reinterpret_cast<const float*>(sub + S::kAs);
                float2 a[S::kNT];
#pragma unroll
                for(int nt = 0; nt < S::kNT; ++nt) a[nt] = *reinterpret_cast<const float2*>(as + wt + 8 * nt + 2 * q);
                const auto fb = tokens(stage, i);
                // A row tile at a time: its MMAs, then their steps.
                if(uniform) {   // the products weight scale x token scale, one a token: step()'s, rounded once
                    const float w = ws[0];
#pragma unroll
                    for(int nt = 0; nt < S::kNT; ++nt) a[nt] = make_float2(__fmul_rn(w, a[nt].x), __fmul_rn(w, a[nt].y));
#pragma unroll
                    for(int mt = 0; mt < S::kMT; ++mt) {
                        float d[S::kNT][4];
                        sums(stage, i, mt, fb, d);
#pragma unroll
                        for(int nt = 0; nt < S::kNT; ++nt) {
                            float* c = acc[s] + 4 * (mt * S::kNT + nt);
                            c[0] = __fmaf_rn(d[nt][0], a[nt].x, c[0]);
                            c[1] = __fmaf_rn(d[nt][1], a[nt].y, c[1]);
                            c[2] = __fmaf_rn(d[nt][2], a[nt].x, c[2]);
                            c[3] = __fmaf_rn(d[nt][3], a[nt].y, c[3]);
                        }
                    }
                    continue;
                }
#pragma unroll
                for(int mt = 0; mt < S::kMT; ++mt) {
                    const float2 w = *reinterpret_cast<const float2*>(ws + blockscaled::pair_slot(wr + 16 * mt + g));
                    float d[S::kNT][4];
                    sums(stage, i, mt, fb, d);
#pragma unroll
                    for(int nt = 0; nt < S::kNT; ++nt) {
                        float* c = acc[s] + 4 * (mt * S::kNT + nt);
                        c[0] = blockscaled::step(d[nt][0], w.x, a[nt].x, c[0]);
                        c[1] = blockscaled::step(d[nt][1], w.x, a[nt].y, c[1]);
                        c[2] = blockscaled::step(d[nt][2], w.y, a[nt].x, c[2]);
                        c[3] = blockscaled::step(d[nt][3], w.y, a[nt].y, c[3]);
                    }
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
