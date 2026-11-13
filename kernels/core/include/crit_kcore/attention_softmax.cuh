// crit's kernel core: the online softmax of a flash-style attention tile held in mma.sync accumulators
// (kernels/core/README.md, "attention_softmax"), with no attention algorithm in it (which positions a warp reads, in
// what order, and where its state goes are the kernel's).
//
// A warp owns 16 query rows of a tile; a lane holds rows gid and gid + 8 (h = 0, 1) of every m16 x n8 accumulator
// (mma.cuh's C layout: c[i] row gid + 8 (i >> 1), column 2 tig + (i & 1)). Its state a row: the running maximum m
// (base 2: the scores carry log2(e), folded into the softmax scale), the lane's part of the running sum l (the lane's
// own columns, added in a fixed order; row_total() adds the quad's four parts at the end) and the unnormalized output
// (the kernel's accumulators, rescaled by alpha each block). Every step is written out (__fadd_rn, __fmul_rn, fmaf,
// ex2.approx): no contraction the compiler may or may not make, so a row's bits depend on its own scores and the
// block boundaries only, never on the rows beside it, the warps or the tile.
//
// Mechanisms after FlashAttention-2 and FlashInfer (Apache-2.0): exp2 with the scale folded in, a -inf-safe maximum,
// the scores' accumulators reused as the next product's A operand, per-lane partial sums reduced once at the end.
#pragma once
#include "crit_kcore/formats.cuh"

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace crit::kcore {

constexpr float kAttnLog2e = 1.4426950408889634f;   // natural-log scores times this are base-2 scores
constexpr float kAttnLn2 = 0.6931471805599453f;     // base-2 log-sum-exps times this are natural ones

// 2^x by the hardware approximation (ex2.approx.ftz: -inf -> +0, the same bits on every GPU from sm_80).
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// A fresh row state: no position seen.
__device__ __forceinline__ void softmax_rows_init(float (&m)[2], float (&l)[2]) {
    m[0] = m[1] = -INFINITY;
    l[0] = l[1] = 0.0f;
}

// One block of base-2 scores s (NT accumulators of m16 x n8, masked ones -inf) into the running state: s becomes the
// probabilities relative to the new maximum, alpha[h] the factor row h's running output takes (0 for a row that had
// seen nothing, exactly 1 where the maximum stays: a block a row does not see changes none of its bits). A row that has seen no position yet keeps -inf and p = 0 (no NaN). The
// maximum crosses the quad by two butterflies (exact); the lane's sum adds its p in the order t, then i.
template <int NT>
__device__ __forceinline__ void softmax_step_m16(float (&s)[NT][4], float (&m)[2], float (&l)[2], float (&alpha)[2]) {
#pragma unroll
    for(int h = 0; h < 2; ++h) {
        float bm = -INFINITY;
#pragma unroll
        for(int t = 0; t < NT; ++t) bm = fmaxf(bm, fmaxf(s[t][2 * h], s[t][2 * h + 1]));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
        const float m_new = fmaxf(m[h], bm);
        const float m_sub = m_new == -INFINITY ? 0.0f : m_new;
        alpha[h] = m_new == m[h] ? 1.0f : ex2(__fsub_rn(m[h], m_sub));
        float sum = 0.0f;
#pragma unroll
        for(int t = 0; t < NT; ++t) {
#pragma unroll
            for(int i = 0; i < 2; ++i) {
                const float p = ex2(__fsub_rn(s[t][2 * h + i], m_sub));
                s[t][2 * h + i] = p;
                sum = __fadd_rn(sum, p);
            }
        }
        l[h] = fmaf(l[h], alpha[h], sum);
        m[h] = m_new;
    }
}

// The row's sum from the quad's four lane parts (lanes 4 gid .. 4 gid + 3): (x0 + x1) + (x2 + x3) on every lane of
// the quad (IEEE addition commutes, so all four get the same bits).
__device__ __forceinline__ float softmax_row_total(float l) {
    l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 1));
    return __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 2));
}

// Two m16 x n8 accumulators of probabilities (positions 8 t .. 8 t + 15) as the 16-bit A operand of the next
// m16n8k16 product (k = those positions), rounded to nearest: the FlashAttention-2 register reuse.
template <typename T>
__device__ __forceinline__ void probabilities_to_a(const float (&p0)[4], const float (&p1)[4], uint32_t (&a)[4]) {
    if constexpr(std::is_same_v<T, __half>) {
        a[0] = u32_of(__floats2half2_rn(p0[0], p0[1]));
        a[1] = u32_of(__floats2half2_rn(p0[2], p0[3]));
        a[2] = u32_of(__floats2half2_rn(p1[0], p1[1]));
        a[3] = u32_of(__floats2half2_rn(p1[2], p1[3]));
    } else {
        a[0] = pack_bf16x2(p0[0], p0[1]);
        a[1] = pack_bf16x2(p0[2], p0[3]);
        a[2] = pack_bf16x2(p1[0], p1[1]);
        a[3] = pack_bf16x2(p1[2], p1[3]);
    }
}

// A partial state (maximum m, sum l, base 2) into a running one (m_run, l_run), in the caller's order of partials:
// output = fmaf(output, keep, partial * take) (fold_output), the sum fmaf(l_run, keep, l * take). A state with no
// position seen (m -inf, l 0) changes no bit of the running one; folding the same partials in the same order gives
// the same bits wherever it runs (a split launch's merge and a fused program alike).
struct SoftmaxFold2 {
    float m, l, keep, take;
};
__device__ __forceinline__ SoftmaxFold2 softmax_fold2(float m_run, float l_run, float m, float l) {
    const float m_new = fmaxf(m_run, m);
    const float m_sub = m_new == -INFINITY ? 0.0f : m_new;
    const float keep = m_run == m_new ? 1.0f : ex2(__fsub_rn(m_run, m_sub));
    const float take = m == m_new ? 1.0f : ex2(__fsub_rn(m, m_sub));
    return SoftmaxFold2{m_new, fmaf(l_run, keep, __fmul_rn(l, take)), keep, take};
}
__device__ __forceinline__ float fold_output(float out, float o, const SoftmaxFold2& f) {
    return fmaf(out, f.keep, __fmul_rn(o, f.take));
}

// The normalized output acc / l, 0 where the row saw no position (l = 0).
// 0 where a row saw no position (l = 0); a NaN sum (a NaN score) stays NaN.
__device__ __forceinline__ float softmax_finish(float acc, float l) { return l != 0.0f ? __fdiv_rn(acc, l) : 0.0f; }

// A sink logit s (natural log, in the scaled scores' units; -inf: none) into a row's final base-2 state as one more
// position that carries no value: the output is fmul(acc, keep) before softmax_finish, the sum and maximum the
// returned ones (softmax_lse2 then counts the sink). Folded once, after every partial state.
__device__ __forceinline__ SoftmaxFold2 softmax_sink2(float m_run, float l_run, float s) {
    return softmax_fold2(m_run, l_run, __fmul_rn(s, 1.4426950408889634f), 1.0f);
}

// The natural-log log-sum-exp of a row's scores from its base-2 maximum and sum (-inf where it saw no position): one
// FMA, written out.
__device__ __forceinline__ float softmax_lse2(float m, float l) {
    return l != 0.0f ? fmaf(m, kAttnLn2, logf(l)) : -INFINITY;
}

}  // namespace crit::kcore
