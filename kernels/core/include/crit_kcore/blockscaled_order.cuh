// crit's kernel core: the declared block order of a block-scaled dot (kernels/core/README.md, "blockscaled").
//
// A block-scaled projection in crit's declared order sums each block of the dot exactly or on the tensor cores, then
// takes one FP32 step per block on lane b % 16 (lane = fma(block sum, weight scale * activation scale, lane), the
// product rounded), the blocks of a lane in increasing order, and reduces the 16 lanes by the halving tree
// ((l0 + l8) + (l4 + l12)) + ((l2 + l10) + (l6 + l14)), then the same over the odd lanes, then their sum.
//
// A GEMM that keeps 16 lanes an output runs out of registers. These blocks let it hold one running lane and four
// partial sums instead: it visits the blocks lane by lane in the order the tree consumes the lanes (lane lane_at(at)
// at step at; GroupVisits walks that order for a pipeline's producer) and folds each finished lane into the partial
// sums with reduce.cuh's fold16 (the same tree). An empty lane (fewer than 16 blocks) folds as +0.0f. The bits are
// those of 16 lanes kept to the end. Unroll the 16 steps so `at` is a constant: fold16 is then a few adds, and only
// the partial sums still needed live.
#pragma once
#include "crit_kcore/reduce.cuh"

namespace crit::kcore::blockscaled {

// The lane the halving tree consumes at step `at` (0..15), reduce.cuh's kLaneOrder16 as arithmetic (a bit reversal),
// for host code and run-time steps: {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}.
__host__ __device__ constexpr int lane_at(int at) {
    return ((at & 1) << 3) | ((at & 2) << 1) | ((at & 4) >> 1) | ((at & 8) >> 3);
}

// The blocks of a dot of `blocks` blocks in lane-major visiting order, G lanes at a time (G = 1, 2 or 4: tree steps
// at .. at + G - 1, siblings in the tree, so folding them together needs no more partial sums than one at a time) and
// RUN consecutive blocks of each lane a visit: visit k0 of a group holds blocks lane_at(at + s) + 16 (k0 + r) of its
// lanes s, r < RUN (block(s, r), -1 where that lane has fewer blocks), the visits of a group in increasing k0, so every
// lane still takes its blocks in increasing order. Groups whose lanes have no block (blocks < 16) are skipped. A
// pipeline's producer and its decoder each keep one and advance it once a visit; the consumer runs the same order
// with `at` unrolled (count(): the group's blocks a lane, those of its first lane, which has the most).
template<int G, int RUN = 1>
struct GroupVisits {
    static_assert(G == 1 || G == 2 || G == 4, "1, 2 or 4 lanes a visit");
    static_assert(RUN >= 1, "at least one block of each lane a visit");
    int blocks = 0, at = 0, k0 = 0;

    __device__ __forceinline__ explicit GroupVisits(int blocks_) : blocks(blocks_), at(0), k0(0) {}
    __host__ __device__ static constexpr int count(int at, int blocks) {
        return lane_at(at) < blocks ? (blocks - lane_at(at) + 15) / 16 : 0;
    }
    __device__ __forceinline__ bool done() const { return at >= 16; }
    __device__ __forceinline__ int block(int s, int r = 0) const {
        const int b = lane_at(at + s) + 16 * (k0 + r);
        return b < blocks ? b : -1;
    }
    __device__ __forceinline__ void next() {
        k0 += RUN;
        if(k0 >= count(at, blocks)) {
            k0 = 0;
            do at += G;
            while(at < 16 && count(at, blocks) == 0);
        }
    }
};

// The visit order of a dot's blocks: a layout of a row's blocks (activation codes the staged GEMM reads, VISIT) in
// which a visit of sibling lanes is one contiguous piece. The tree's steps in pairs (2p, 2p + 1: sibling lanes l and
// l + 8), each pair's blocks interleaved by k (step 2p's block k, then step 2p + 1's), the pairs in the tree's order.
// A visit of G = 2 lanes and RUN blocks of each is then 64 RUN contiguous bytes of an E4M3 row, where the plain
// layout's are 32-byte pieces 256 bytes apart (L2 serves 64-byte pieces at about twice the rate of 32-byte ones on an
// RTX 5090). position(b): where block b of the plain layout sits; a bijection of 0 .. blocks - 1.
struct VisitOrder {
    int c = 0, mask = 0;   // blocks / 16, and the tree steps whose lane has one more block (bit at)
    __host__ __device__ explicit VisitOrder(int blocks) : c(blocks >> 4), mask(0) {
        const int rem = blocks & 15;
        for(int at = 0; at < 16; ++at)
            if(lane_at(at) < rem) mask |= 1 << at;
    }
    __host__ __device__ int position(int b) const {
        const int at = lane_at(b & 15), pair = at >> 1;
        const unsigned before = unsigned(mask) & ((1u << (2 * pair)) - 1u);
#ifdef __CUDA_ARCH__
        const int extra = __popc(before);
#else
        const int extra = __builtin_popcount(before);
#endif
        return 2 * c * pair + extra + 2 * (b >> 4) + (at & 1);
    }
};

}  // namespace crit::kcore::blockscaled
