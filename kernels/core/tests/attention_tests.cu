// crit_kcore/attention_softmax.cuh on the device: a warp's online softmax over blocks of mma accumulators (step, row
// totals) against float64, a row's bits the same wherever its row sits in the tile (rows permuted between runs, bit
// for bit), rows that see nothing (no NaN, sum 0), partial states folded in order (an empty state changes no bit),
// finish and lse, and the probabilities reused as an m16n8k16 A operand.
#include "check.cuh"
#include "crit_kcore/attention_softmax.cuh"
#include "crit_kcore/mma.cuh"

#include <cmath>
#include <random>

namespace kc = crit::kcore;
using check::bits;
using check::report;

namespace {

constexpr int NT = 4, BLOCKS = 3, N = 8 * NT * BLOCKS;   // 16 rows x 96 positions in three blocks of 32

// One warp: scores s [16][N] (base 2, -inf: not seen; row order `perm`: tile row i holds row perm[i]) -> each row's
// maximum and total sum, and the probabilities of the last block (relative to the final maximum of that block).
__global__ void online_kernel(const float* s, const int* perm, float* m_out, float* l_out) {
    const int lane = threadIdx.x, gid = lane >> 2, tig = lane & 3;
    float m[2], l[2];
    kc::softmax_rows_init(m, l);
    for(int blk = 0; blk < BLOCKS; ++blk) {
        float sc[NT][4];
        for(int t = 0; t < NT; ++t)
            for(int i = 0; i < 4; ++i) {
                const int row = perm[gid + 8 * (i >> 1)], col = blk * 8 * NT + 8 * t + 2 * tig + (i & 1);
                sc[t][i] = s[row * N + col];
            }
        float alpha[2];
        kc::softmax_step_m16<NT>(sc, m, l, alpha);
    }
    for(int h = 0; h < 2; ++h) {
        const float total = kc::softmax_row_total(l[h]);
        if(tig == 0) {
            m_out[perm[gid + 8 * h]] = m[h];
            l_out[perm[gid + 8 * h]] = total;
        }
    }
}

// fold2 of n partial (m, l) pairs in order, finish and lse of the result; and an empty state folded in.
__global__ void fold_kernel(const float* m, const float* l, const float* o, int n, float* out) {
    float m_run = -INFINITY, l_run = 0.0f, acc = 0.0f;
    for(int i = 0; i < n; ++i) {
        const kc::SoftmaxFold2 f = kc::softmax_fold2(m_run, l_run, m[i], l[i]);
        acc = kc::fold_output(acc, o[i], f);
        m_run = f.m, l_run = f.l;
    }
    const kc::SoftmaxFold2 e = kc::softmax_fold2(m_run, l_run, -INFINITY, 0.0f);
    out[0] = m_run, out[1] = l_run, out[2] = acc;
    out[3] = e.m, out[4] = e.l, out[5] = kc::fold_output(acc, 0.0f, e);
    out[6] = kc::softmax_finish(acc, l_run), out[7] = kc::softmax_lse2(m_run, l_run);
    out[8] = kc::softmax_finish(1.0f, 0.0f), out[9] = kc::softmax_lse2(-INFINITY, 0.0f);
}

// Probabilities p [16][16] as two m16 x n8 accumulators, reused as the A operand of p x b (b [16][8] BF16).
__global__ void reuse_kernel(const float* p, const __nv_bfloat16* b, float* c) {
    const int lane = threadIdx.x, gid = lane >> 2, tig = lane & 3;
    float p0[4], p1[4];
    for(int i = 0; i < 4; ++i) {
        const int row = gid + 8 * (i >> 1), col = 2 * tig + (i & 1);
        p0[i] = p[row * 16 + col];
        p1[i] = p[row * 16 + 8 + col];
    }
    uint32_t a[4];
    kc::probabilities_to_a<__nv_bfloat16>(p0, p1, a);
    auto bb = [&](int k) { return __bfloat16_as_ushort(b[k * 8 + gid]); };
    const uint32_t b0 = bb(2 * tig) | (uint32_t(bb(2 * tig + 1)) << 16);
    const uint32_t b1 = bb(8 + 2 * tig) | (uint32_t(bb(8 + 2 * tig + 1)) << 16);
    float acc[4] = {0, 0, 0, 0};
    kc::mma_bf16_m16n8k16(acc, a[0], a[1], a[2], a[3], b0, b1);
    for(int i = 0; i < 4; ++i) c[kc::mma_c_row(lane, i) * 8 + kc::mma_c_col(lane, i)] = acc[i];
}

void test_online() {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::vector<float> s(16 * N);
    for(int r = 0; r < 16; ++r)
        for(int c = 0; c < N; ++c) {
            float v = nd(rng) + (r == 3 ? 40.0f : 0.0f);
            if(r == 5) v = -INFINITY;                              // a row that sees nothing
            if(r == 7 && c >= 40) v = -INFINITY;                   // a row whose last blocks are not seen
            if(r == 9 && c < 70) v = -INFINITY;                    // a row that sees only its last block
            if((r * 31 + c * 7) % 11 == 0) v = -INFINITY;          // scattered masks
            s[r * N + c] = v;
        }
    std::vector<int> ident(16), perm(16);
    for(int i = 0; i < 16; ++i) ident[i] = i, perm[i] = (i * 5 + 3) % 16;
    float* ds = check::dev(s);
    int* dp = check::dev(ident);
    int* dq = check::dev(perm);
    float *m0 = check::dev<float>(16), *l0 = check::dev<float>(16), *m1 = check::dev<float>(16),
          *l1 = check::dev<float>(16);
    online_kernel<<<1, 32>>>(ds, dp, m0, l0);
    online_kernel<<<1, 32>>>(ds, dq, m1, l1);
    const auto hm0 = check::get(m0, 16), hl0 = check::get(l0, 16), hm1 = check::get(m1, 16), hl1 = check::get(l1, 16);
    long long bad = 0, moved = 0;
    for(int r = 0; r < 16; ++r) {
        moved += bits(hm0[r]) != bits(hm1[r]) || bits(hl0[r]) != bits(hl1[r]);
        double mx = -INFINITY, sum = 0.0;
        for(int c = 0; c < N; ++c) mx = std::fmax(mx, double(s[r * N + c]));
        for(int c = 0; c < N; ++c)
            if(std::isfinite(s[r * N + c])) sum += std::exp2(double(s[r * N + c]) - mx);
        if(r == 5) {
            bad += !(hm0[r] == -INFINITY && hl0[r] == 0.0f);
        } else {
            bad += hm0[r] != float(mx) || std::fabs(hl0[r] - sum) > 2e-5 * sum;
        }
    }
    report("softmax_step_m16 + softmax_row_total against float64 (empty and partly masked rows)", bad);
    report("a row's maximum and sum have the same bits wherever it sits in the tile", moved);
}

void test_fold() {
    const std::vector<float> m = {3.0f, -INFINITY, 7.5f, 1.25f}, l = {2.0f, 0.0f, 1.5f, 4.0f}, o = {0.7f, 0.0f, -2.0f, 3.0f};
    float* dm = check::dev(m);
    float* dl = check::dev(l);
    float* d_o = check::dev(o);
    float* out = check::dev<float>(10);
    fold_kernel<<<1, 1>>>(dm, dl, d_o, 4, out);
    const auto h = check::get(out, 10);
    double mx = 7.5, sum = 0.0, acc = 0.0;
    for(int i = 0; i < 4; ++i)
        if(std::isfinite(m[i])) sum += l[i] * std::exp2(m[i] - mx), acc += o[i] * std::exp2(m[i] - mx);
    long long bad = h[0] != 7.5f || std::fabs(h[1] - sum) > 1e-5 * sum || std::fabs(h[2] - acc) > 1e-5 * std::fabs(acc);
    report("softmax_fold2 + fold_output against float64", bad);
    report("an empty state folded in changes no bit", bits(h[3]) != bits(h[0]) || bits(h[4]) != bits(h[1]) ||
                                                           bits(h[5]) != bits(h[2]));
    const double lse = mx * std::log(2.0) + std::log(sum);
    report("softmax_finish and softmax_lse2 (and -inf, 0 for a row that saw nothing)",
           std::fabs(h[6] - acc / sum) > 1e-5 || std::fabs(h[7] - lse) > 1e-5 || h[8] != 0.0f || h[9] != -INFINITY);
}

void test_reuse() {
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> p(256);
    std::vector<__nv_bfloat16> b(128);
    for(auto& x : p) x = u(rng);
    for(auto& x : b) x = __float2bfloat16(u(rng) - 0.5f);
    float* dp = check::dev(p);
    __nv_bfloat16* db = check::dev(b);
    float* dc = check::dev<float>(128);
    reuse_kernel<<<1, 32>>>(dp, db, dc);
    const auto c = check::get(dc, 128);
    long long bad = 0;
    for(int r = 0; r < 16; ++r)
        for(int n = 0; n < 8; ++n) {
            double want = 0.0;
            for(int k = 0; k < 16; ++k)
                want += double(__bfloat162float(__float2bfloat16(p[r * 16 + k]))) * __bfloat162float(b[k * 8 + n]);
            bad += std::fabs(c[r * 8 + n] - want) > 1e-5;
        }
    report("probabilities_to_a: two accumulators as the next m16n8k16 A operand", bad);
}

}  // namespace

int main() {
    check::flags();
    test_online();
    test_fold();
    test_reuse();
    return check::finish();
}
