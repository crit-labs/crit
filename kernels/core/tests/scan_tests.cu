// scan_columns.cuh and scan_chunk.cuh on the device: the column sums (4, 8, 16 and 32 values) against a host model of
// the 32-lane butterfly (bit for bit), the exact decay against the host's FP64 exp rounded once, the unit lower
// triangular inverse against an FP64 solve, and the transposed state tile: loaded from [dk][dv], used as the A operand
// of S^T K^T and updated by S^T += D^T K, then stored, against FP64 products of the same FP16-rounded operands.
#include "check.cuh"
#include "crit_kcore/scan_chunk.cuh"
#include "crit_kcore/scan_columns.cuh"

#include <cmath>
#include <random>

using namespace crit::kcore;

// 32 lanes x 8 values in, 32 x 8 sums out (every lane's copy), and 32 x 4 of column_sum4.
__global__ void sums_kernel(const float* in, float* out8, float* out4) {
    const int lane = threadIdx.x;
    float x[8], y[4];
    for(int i = 0; i < 8; ++i) x[i] = in[lane * 8 + i];
    for(int i = 0; i < 4; ++i) y[i] = in[lane * 8 + i];
    scan::column_sum8(x);
    scan::column_sum4(y);
    for(int i = 0; i < 8; ++i) out8[lane * 8 + i] = x[i];
    for(int i = 0; i < 4; ++i) out4[lane * 4 + i] = y[i];
}

// column_sum<N> for 8, 16 and 32 values from the same 32 x 32 inputs (lane l's value i at in[l * 32 + i]).
template <int N>
__global__ void sums_n_kernel(const float* in, float* out) {
    const int lane = threadIdx.x;
    float x[N];
    for(int i = 0; i < N; ++i) x[i] = in[lane * 32 + i];
    scan::column_sum(x);
    for(int i = 0; i < N; ++i) out[lane * N + i] = x[i];
}

__global__ void decay_kernel(const float* x, float* e, float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        e[i] = scan::decay_exact(double(x[i]));
        u[i] = scan::delta_update(x[i], e[i], 0.5f, x[i]);
    }
}

constexpr int kN = 64, kStride = 68;
__global__ void inverse_kernel(const float* A, float* T) {
    __shared__ __align__(16) float As[kN * kStride];
    for(int e = threadIdx.x; e < kN * kN; e += blockDim.x) As[(e / kN) * kStride + e % kN] = A[e];
    __syncthreads();
    float t[kN];
    scan::unit_lower_inverse_column<kN>(As, kStride, threadIdx.x, t);
    for(int i = 0; i < kN; ++i) T[i * kN + threadIdx.x] = t[i];
}

// One warp: S [DK][16] (dv 16, v0 0), K [16 tokens][DK], D [16 tokens][16 values]. X^T = S^T K^T (16 x 16), then
// S^T += D^T K (keys as columns), S stored back.
constexpr int kDK = 32;
__global__ void state_kernel(float* S, const float* K, const float* D, float* X) {
    const int lane = threadIdx.x, gid = lane >> 2, tig = lane & 3;
    scan::TransposedState<kDK> st;
    st.load(S, 16, 0, lane, true);
    auto pair = [&](float lo, float hi) { return scan::f16x2(lo, hi); };
    float x[2][4] = {};
    for(int ks = 0; ks < kDK / 16; ++ks) {
        uint32_t a[4];
        st.a_operand(ks, a);
        for(int nt = 0; nt < 2; ++nt) {   // B (k = keys, n = tokens): b0 keys 16 ks + 2 tig .., token 8 nt + gid
            const float* kr = K + (8 * nt + gid) * kDK + 16 * ks + 2 * tig;
            mma_f16_m16n8k16(x[nt], a[0], a[1], a[2], a[3], pair(kr[0], kr[1]), pair(kr[8], kr[9]));
        }
    }
    for(int nt = 0; nt < 2; ++nt)
        for(int i = 0; i < 4; ++i) X[mma_c_row(lane, i) * 16 + 8 * nt + mma_c_col(lane, i)] = x[nt][i];
    // D^T as A (rows values, k tokens): the accumulator layout of D^T, built here from memory
    float lo[4], hi[4];
    for(int i = 0; i < 4; ++i) {
        const int v = mma_c_row(lane, i), t = mma_c_col(lane, i);
        lo[i] = D[t * 16 + v];
        hi[i] = D[(t + 8) * 16 + v];
    }
    uint32_t a[4];
    scan::f16_a_of_accumulators(a, lo, hi);
    for(int kt = 0; kt < kDK / 8; ++kt) {   // B (k = tokens, n = keys): b0 tokens 2 tig .., key 8 kt + gid
        const float* kc = K + 8 * kt + gid;
        mma_f16_m16n8k16(st.s[kt], a[0], a[1], a[2], a[3], pair(kc[(2 * tig) * kDK], kc[(2 * tig + 1) * kDK]),
                         pair(kc[(2 * tig + 8) * kDK], kc[(2 * tig + 9) * kDK]));
    }
    st.store(S, 16, 0, lane);
}

static float h16(float v) { return __half2float(__float2half_rn(v)); }

int main() {
    check::flags();
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    {   // column sums: the 32-lane butterfly's bits on every value
        std::vector<float> v(256);
        for(auto& x : v) x = u(rng) * std::ldexp(1.0f, int(rng() % 16) - 8);
        float *d = check::dev(v), *o8 = check::dev<float>(256), *o4 = check::dev<float>(128);
        sums_kernel<<<1, 32>>>(d, o8, o4);
        auto O8 = check::get(o8, 256), O4 = check::get(o4, 128);
        long long bad8 = 0, bad4 = 0;
        for(int i = 0; i < 8; ++i) {
            std::vector<float> lanes(32);
            for(int l = 0; l < 32; ++l) lanes[l] = v[l * 8 + i];
            for(int d2 = 16; d2; d2 >>= 1) {
                std::vector<float> next(32);
                for(int l = 0; l < 32; ++l) next[l] = lanes[l] + lanes[l ^ d2];
                lanes = next;
            }
            for(int l = 0; l < 32; ++l) {
                bad8 += !check::same(O8[l * 8 + i], lanes[0]);
                if(i < 4) bad4 += !check::same(O4[l * 4 + i], lanes[0]);
            }
        }
        check::report("column_sum8", bad8);
        check::report("column_sum4", bad4);
    }
    {   // column_sum<N>: the same butterfly bits for 8, 16 and 32 values
        std::vector<float> v(1024);
        for(auto& x : v) x = u(rng) * std::ldexp(1.0f, int(rng() % 16) - 8);
        float* d = check::dev(v);
        float *o8 = check::dev<float>(256), *o16 = check::dev<float>(512), *o32 = check::dev<float>(1024);
        sums_n_kernel<8><<<1, 32>>>(d, o8);
        sums_n_kernel<16><<<1, 32>>>(d, o16);
        sums_n_kernel<32><<<1, 32>>>(d, o32);
        auto O8 = check::get(o8, 256), O16 = check::get(o16, 512), O32 = check::get(o32, 1024);
        long long bad = 0;
        for(int i = 0; i < 32; ++i) {
            std::vector<float> lanes(32);
            for(int l = 0; l < 32; ++l) lanes[l] = v[l * 32 + i];
            for(int d2 = 16; d2; d2 >>= 1) {
                std::vector<float> next(32);
                for(int l = 0; l < 32; ++l) next[l] = lanes[l] + lanes[l ^ d2];
                lanes = next;
            }
            for(int l = 0; l < 32; ++l) {
                if(i < 8) bad += !check::same(O8[l * 8 + i], lanes[0]);
                if(i < 16) bad += !check::same(O16[l * 16 + i], lanes[0]);
                bad += !check::same(O32[l * 32 + i], lanes[0]);
            }
        }
        check::report("column_sum<8, 16, 32>", bad);
    }
    {   // decays and the update expression
        const int n = 4096;
        std::vector<float> x(n);
        for(auto& e : x) e = -std::fabs(u(rng)) * std::ldexp(1.0f, int(rng() % 24) - 20);
        float *dx = check::dev(x), *de = check::dev<float>(n), *du = check::dev<float>(n);
        decay_kernel<<<(n + 255) / 256, 256>>>(dx, de, du, n);
        auto E = check::get(de, n), U = check::get(du, n);
        long long bad = 0, badu = 0;
        for(int i = 0; i < n; ++i) {
            const float want = static_cast<float>(std::exp(double(x[i])));
            bad += !check::same(E[i], want);
            badu += !check::same(U[i], std::fmaf(0.5f, x[i], x[i] * want));
        }
        check::report("decay_exact", bad);
        check::report("delta_update", badu);
    }
    {   // the inverse of I + A, A strictly lower with entries up to 0.3
        std::vector<float> A(kN * kN, 0.f);
        for(int i = 0; i < kN; ++i)
            for(int j = 0; j < i; ++j) A[i * kN + j] = 0.3f * u(rng);
        float *dA = check::dev(A), *dT = check::dev<float>(kN * kN);
        inverse_kernel<<<1, kN>>>(dA, dT);
        auto T = check::get(dT, kN * kN);
        std::vector<double> want(kN * kN, 0.0);   // forward substitution in FP64, column by column
        for(int j = 0; j < kN; ++j)
            for(int i = 0; i < kN; ++i) {
                double acc = i == j ? 1.0 : 0.0;
                for(int m = 0; m < i; ++m) acc -= double(A[i * kN + m]) * want[m * kN + j];
                want[i * kN + j] = acc;
            }
        long long bad = 0;
        for(int e = 0; e < kN * kN; ++e) bad += std::fabs(T[e] - want[e]) > 1e-5 * (1.0 + std::fabs(want[e]));
        for(int i = 0; i < kN; ++i)
            for(int j = i + 1; j < kN; ++j) bad += T[i * kN + j] != 0.f;
        check::report("unit_lower_inverse_column", bad);
    }
    {   // the transposed state tile
        std::vector<float> S(kDK * 16), K(16 * kDK), D(16 * 16);
        for(auto& x : S) x = u(rng);
        for(auto& x : K) x = u(rng);
        for(auto& x : D) x = u(rng);
        float *dS = check::dev(S), *dK = check::dev(K), *dD = check::dev(D), *dX = check::dev<float>(256);
        state_kernel<<<1, 32>>>(dS, dK, dD, dX);
        auto X = check::get(dX, 256), S2 = check::get(dS, kDK * 16);
        long long bad = 0, bads = 0;
        for(int v = 0; v < 16; ++v)
            for(int t = 0; t < 16; ++t) {   // X^T[v][t] = sum_key S[key][v] K[t][key]
                double want = 0.0;
                for(int key = 0; key < kDK; ++key) want += double(h16(S[key * 16 + v])) * h16(K[t * kDK + key]);
                bad += std::fabs(X[v * 16 + t] - want) > 1e-4;
            }
        for(int key = 0; key < kDK; ++key)
            for(int v = 0; v < 16; ++v) {   // S'[key][v] = S[key][v] + sum_t D[t][v] K[t][key]
                double want = S[key * 16 + v];
                for(int t = 0; t < 16; ++t) want += double(h16(D[t * 16 + v])) * h16(K[t * kDK + key]);
                bads += std::fabs(S2[key * 16 + v] - want) > 1e-4;
            }
        check::report("TransposedState as A operand", bad);
        check::report("TransposedState update", bads);
    }
    return check::finish();
}
