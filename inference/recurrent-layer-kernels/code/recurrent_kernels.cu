// Recurrent-layer kernels; see recurrent_kernels.h, conv.py and ssd.py (and recurrent-state-recovery's chunked_prefill.py
// for the gated-delta math). FP32 SIMT throughout: the chunked prefills are matrix products a tensor-core
// adaptation can take over (keeping FP32-equivalent accuracy, for example split TF32).
#include "recurrent_kernels.h"

#include <cstdint>

namespace crit::recurrent_kernels {
namespace {

constexpr int kChunk = 64, kThreads = 256, kMaxConv = 8;

__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }

// Warp sum to lane 0, then broadcast: every lane uses lane 0's value, so one token's update is one expression.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for(int o = 16; o; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
    return __shfl_sync(0xffffffffu, x, 0);
}

// ---------------------------------------------------------------------------------------------- convolution
__global__ void conv_kernel(Conv c, int rows) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if(ch >= c.channels) return;
    const int W = c.width;
    float win[kMaxConv - 1], w[kMaxConv];
#pragma unroll
    for(int i = 0; i < kMaxConv - 1; ++i) win[i] = i < W - 1 ? c.state[std::size_t(ch) * (W - 1) + i] : 0.f;
#pragma unroll
    for(int i = 0; i < kMaxConv; ++i) w[i] = i < W ? c.weight[std::size_t(ch) * W + i] : 0.f;
    const float b = c.bias ? c.bias[ch] : 0.f;
    for(int t = 0; t < rows; ++t) {
        const float xt = c.x[std::size_t(t) * c.x_stride + ch];
        float acc = b;
#pragma unroll
        for(int i = 0; i < kMaxConv - 1; ++i)
            if(i < W - 1) acc = fmaf(w[i], win[i], acc);
#pragma unroll
        for(int i = 0; i < kMaxConv; ++i)
            if(i == W - 1) acc = fmaf(w[i], xt, acc);
        c.out[std::size_t(t) * c.out_stride + ch] = c.silu ? silu(acc) : acc;
#pragma unroll
        for(int i = 0; i < kMaxConv - 2; ++i)
            if(i < W - 2) win[i] = win[i + 1];
#pragma unroll
        for(int i = 0; i < kMaxConv - 1; ++i)
            if(i == W - 2) win[i] = xt;
        if(c.checkpoints)
            for(int i = 0; i < W - 1; ++i) c.checkpoints[(std::size_t(t) * c.channels + ch) * (W - 1) + i] = win[i];
    }
    for(int i = 0; i < W - 1; ++i) c.state[std::size_t(ch) * (W - 1) + i] = win[i];
}

// ---------------------------------------------------------------------------------------------- gated delta
struct DeltaScratch { float* tb; float* w; float* p; float* g; };

template <int DK>
DeltaScratch delta_split(float* scratch, int v_heads, int chunks) {
    const std::size_t blocks = std::size_t(v_heads) * chunks;
    DeltaScratch s;
    s.tb = scratch;
    s.w = s.tb + blocks * kChunk * kChunk;
    s.p = s.w + blocks * kChunk * DK;
    s.g = s.p + blocks * kChunk * kChunk;
    return s;
}

// Pass 1, block per (value head, chunk): L, P, T = (I + L)^-1, Tb = T diag(beta), W = T diag(beta e^G) K, G.
// Missing tokens of a partial chunk have beta = 0, k = q = 0 and log decay 0: no update, no output.
template <int DK>
__global__ void __launch_bounds__(kThreads) delta_prepare_kernel(DeltaShape d, DeltaInputs in, int rows, DeltaScratch cs) {
    extern __shared__ float4 smem4[];
    float* Ks = reinterpret_cast<float*>(smem4);   // [64][DK]
    float* Qs = Ks + kChunk * DK;                   // [64][DK], scaled
    float* Lm = Qs + kChunk * DK;                   // [64][64]
    float* Tm = Lm + kChunk * kChunk;               // [64][64]
    __shared__ float G[kChunk], B[kChunk];
    const int h = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
    const int kh = h / (d.v_heads / d.k_heads), t0 = c * kChunk, n = min(kChunk, rows - t0);
    for(int e = tid; e < kChunk * DK; e += kThreads) {
        const int i = e / DK, x = e % DK;
        const std::size_t at = std::size_t(t0 + i) * in.qk_stride + std::size_t(kh) * DK + x;
        Ks[e] = i < n ? in.k[at] : 0.f;
        Qs[e] = i < n ? in.q[at] * in.scale : 0.f;
    }
    if(tid < kChunk) {
        const bool live = tid < n;
        G[tid] = live ? in.log_decay[std::size_t(t0 + tid) * in.gate_stride + h] : 0.f;
        B[tid] = live ? in.beta[std::size_t(t0 + tid) * in.gate_stride + h] : 0.f;
    }
    __syncthreads();
    if(tid == 0)
        for(int i = 1; i < kChunk; ++i) G[i] += G[i - 1];
    __syncthreads();
    const std::size_t blk = std::size_t(h) * gridDim.y + c;
    {   // L and P: thread owns row i = tid / 4 and 16 columns.
        const int i = tid / 4, j0 = (tid % 4) * 16;
        float a[16], pq[16];
#pragma unroll
        for(int j = 0; j < 16; ++j) a[j] = pq[j] = 0.f;
        for(int x = 0; x < DK; ++x) {
            const float ki = Ks[i * DK + x], qi = Qs[i * DK + x];
#pragma unroll
            for(int j = 0; j < 16; ++j) {
                const float kj = Ks[(j0 + j) * DK + x];
                a[j] = fmaf(ki, kj, a[j]);
                pq[j] = fmaf(qi, kj, pq[j]);
            }
        }
        float* P = cs.p + blk * kChunk * kChunk;
#pragma unroll
        for(int j = 0; j < 16; ++j) {
            const int jj = j0 + j;
            Lm[i * kChunk + jj] = jj < i ? B[i] * expf(G[i] - G[jj]) * a[j] : 0.f;
            P[i * kChunk + jj] = jj <= i ? expf(G[i] - G[jj]) * pq[j] : 0.f;
        }
    }
    __syncthreads();
    // T = (I + L)^-1 by forward substitution, one row at a time over 64 column threads.
    for(int i = 0; i < kChunk; ++i) {
        if(tid < kChunk) {
            const int j = tid;
            float v = i == j ? 1.f : 0.f;
            for(int m = j; m < i; ++m) v = fmaf(-Lm[i * kChunk + m], Tm[m * kChunk + j], v);
            Tm[i * kChunk + j] = j <= i ? v : 0.f;
        }
        __syncthreads();
    }
    {   // Tb = T diag(beta); W = T diag(beta e^G) K: thread owns row t = tid / 4 and DK / 4 columns of W.
        constexpr int CW = DK / 4;
        const int t = tid / 4, c0 = (tid % 4) * CW;
        float* Tb = cs.tb + blk * kChunk * kChunk;
        for(int j = tid % 4; j < kChunk; j += 4) Tb[t * kChunk + j] = Tm[t * kChunk + j] * B[j];
        float wv[CW];
#pragma unroll
        for(int x = 0; x < CW; ++x) wv[x] = 0.f;
        for(int i = 0; i <= t; ++i) {
            const float f = Tm[t * kChunk + i] * B[i] * expf(G[i]);
#pragma unroll
            for(int x = 0; x < CW; ++x) wv[x] = fmaf(f, Ks[i * DK + c0 + x], wv[x]);
        }
        float* Wm = cs.w + blk * kChunk * DK;
#pragma unroll
        for(int x = 0; x < CW; ++x) Wm[t * DK + c0 + x] = wv[x];
    }
    if(tid < kChunk) cs.g[blk * kChunk + tid] = G[tid];
}

// Pass 2, block per (value head, 64 value columns), chunks in order with the state slice S[DK][64] in shared memory.
template <int DK>
__global__ void __launch_bounds__(kThreads) delta_scan_kernel(DeltaShape d, DeltaInputs in, int rows, int chunks,
                                                              DeltaScratch cs, float* state, float* out, std::size_t out_stride) {
    extern __shared__ float4 smem4[];
    float* S = reinterpret_cast<float*>(smem4);   // [DK][64]
    float* U = S + DK * 64;                        // [64][64]
    float* V = U + kChunk * 64;                    // [64][64]
    float* Ks = V + kChunk * 64;                   // [64][DK]
    const int h = blockIdx.x, jb = blockIdx.y * 64, tid = threadIdx.x;
    const int kh = h / (d.v_heads / d.k_heads);
    float* st = state + std::size_t(h) * DK * d.dv;
    for(int e = tid; e < DK * 64; e += kThreads) S[e] = st[std::size_t(e / 64) * d.dv + jb + e % 64];
    for(int c = 0; c < chunks; ++c) {
        const int t0 = c * kChunk, n = min(kChunk, rows - t0);
        const std::size_t blk = std::size_t(h) * chunks + c;
        const float* Tb = cs.tb + blk * kChunk * kChunk;
        const float* Wm = cs.w + blk * kChunk * DK;
        const float* P = cs.p + blk * kChunk * kChunk;
        const float* G = cs.g + blk * kChunk;
        __syncthreads();  // the previous chunk's readers of V, Ks and U are done
        for(int e = tid; e < kChunk * 64; e += kThreads) {
            const int i = e / 64, j = e % 64;
            V[e] = i < n ? in.v[std::size_t(t0 + i) * in.v_stride + std::size_t(h) * d.dv + jb + j] : 0.f;
        }
        for(int e = tid; e < kChunk * DK; e += kThreads) {
            const int i = e / DK, x = e % DK;
            Ks[e] = i < n ? in.k[std::size_t(t0 + i) * in.qk_stride + std::size_t(kh) * DK + x] : 0.f;
        }
        __syncthreads();
        const int t = tid / 4, j0 = (tid % 4) * 16;
        {   // U = Tb V - W S
            float u[16];
#pragma unroll
            for(int j = 0; j < 16; ++j) u[j] = 0.f;
            for(int i = 0; i <= t; ++i) {
                const float f = Tb[t * kChunk + i];
#pragma unroll
                for(int j = 0; j < 16; ++j) u[j] = fmaf(f, V[i * 64 + j0 + j], u[j]);
            }
            for(int x = 0; x < DK; ++x) {
                const float f = Wm[t * DK + x];
#pragma unroll
                for(int j = 0; j < 16; ++j) u[j] = fmaf(-f, S[x * 64 + j0 + j], u[j]);
            }
#pragma unroll
            for(int j = 0; j < 16; ++j) U[t * 64 + j0 + j] = u[j];
        }
        __syncthreads();
        if(t < n) {   // o_t = e^{G_t} (s q_t) S + sum_{i <= t} P_ti U_i
            float ov[16], qs[16];
#pragma unroll
            for(int j = 0; j < 16; ++j) qs[j] = ov[j] = 0.f;
            const float* qr = in.q + std::size_t(t0 + t) * in.qk_stride + std::size_t(kh) * DK;
            for(int x = 0; x < DK; ++x) {
                const float f = qr[x] * in.scale;
#pragma unroll
                for(int j = 0; j < 16; ++j) qs[j] = fmaf(f, S[x * 64 + j0 + j], qs[j]);
            }
            for(int i = 0; i <= t; ++i) {
                const float f = P[t * kChunk + i];
#pragma unroll
                for(int j = 0; j < 16; ++j) ov[j] = fmaf(f, U[i * 64 + j0 + j], ov[j]);
            }
            const float eg = expf(G[t]);
            float* orow = out + std::size_t(t0 + t) * out_stride + std::size_t(h) * d.dv + jb + j0;
#pragma unroll
            for(int j = 0; j < 16; ++j) orow[j] = fmaf(eg, qs[j], ov[j]);
        }
        __syncthreads();  // every reader of the old S is done
        {   // S' = e^{G_C} S + sum_i e^{G_C - G_i} k_i U_i^T: TPR threads per key row
            constexpr int TPR = kThreads / DK, CPT = 64 / TPR;
            const int r = tid / TPR, c0 = (tid % TPR) * CPT;
            const float gC = G[kChunk - 1], egc = expf(gC);
            float sv[CPT];
#pragma unroll
            for(int j = 0; j < CPT; ++j) sv[j] = S[r * 64 + c0 + j] * egc;
            for(int i = 0; i < kChunk; ++i) {
                const float f = Ks[i * DK + r] * expf(gC - G[i]);
#pragma unroll
                for(int j = 0; j < CPT; ++j) sv[j] = fmaf(f, U[i * 64 + c0 + j], sv[j]);
            }
#pragma unroll
            for(int j = 0; j < CPT; ++j) S[r * 64 + c0 + j] = sv[j];
        }
    }
    __syncthreads();
    for(int e = tid; e < DK * 64; e += kThreads) st[std::size_t(e / 64) * d.dv + jb + e % 64] = S[e];
}

// A lane's 4 value columns of key row r: one 16-byte access when the base is 16-byte aligned (j0 is a multiple of 4
// and dv of 64), else 4 scalar ones. Only the memory access differs; the values are the same.
__device__ __forceinline__ void load4(const float* p, bool wide, float (&x)[4]) {
    if(wide) { const float4 v = *reinterpret_cast<const float4*>(p); x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w; }
    else for(int j = 0; j < 4; ++j) x[j] = p[j];
}
__device__ __forceinline__ void store4(float* p, bool wide, const float (&x)[4]) {
    if(wide) *reinterpret_cast<float4*>(p) = make_float4(x[0], x[1], x[2], x[3]);
    else for(int j = 0; j < 4; ++j) p[j] = x[j];
}
__device__ __forceinline__ bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; }

// Column scan, block per (value head, 32 value columns): warp w owns 4 value columns, lane l key rows l * R .. + R - 1.
template <int DK>
__global__ void __launch_bounds__(kThreads) delta_columns_kernel(DeltaShape d, DeltaInputs in, int rows, const float* state_in,
                                                                 float* state_out, float* checkpoints, float* out,
                                                                 std::size_t out_stride) {
    constexpr int R = DK / 32;
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j0 = blockIdx.y * 32 + warp * 4, kh = h / (d.v_heads / d.k_heads);
    const std::size_t head = std::size_t(h) * DK * d.dv;
    float S[R][4];
    const bool wide_in = aligned16(state_in), wide_out = aligned16(state_out), wide_ck = checkpoints && aligned16(checkpoints);
#pragma unroll
    for(int r = 0; r < R; ++r) load4(state_in + head + std::size_t(lane * R + r) * d.dv + j0, wide_in, S[r]);
    for(int t = 0; t < rows; ++t) {
        const float* kr = in.k + std::size_t(t) * in.qk_stride + std::size_t(kh) * DK + lane * R;
        const float* qr = in.q + std::size_t(t) * in.qk_stride + std::size_t(kh) * DK + lane * R;
        const float* vr = in.v + std::size_t(t) * in.v_stride + std::size_t(h) * d.dv + j0;
        const float e = expf(in.log_decay[std::size_t(t) * in.gate_stride + h]);
        const float b = in.beta[std::size_t(t) * in.gate_stride + h];
        float k[R], q[R], qk = 0.f, kS[4] = {0.f, 0.f, 0.f, 0.f}, qS[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for(int r = 0; r < R; ++r) {
            k[r] = kr[r];
            q[r] = qr[r] * in.scale;
            qk = fmaf(q[r], k[r], qk);
#pragma unroll
            for(int j = 0; j < 4; ++j) {
                kS[j] = fmaf(k[r], S[r][j], kS[j]);
                qS[j] = fmaf(q[r], S[r][j], qS[j]);
            }
        }
        qk = warp_sum(qk);
        float delta[4], o[4];
#pragma unroll
        for(int j = 0; j < 4; ++j) {
            kS[j] = warp_sum(kS[j]);
            qS[j] = warp_sum(qS[j]);
            delta[j] = b * (vr[j] - e * kS[j]);
            o[j] = fmaf(e, qS[j], qk * delta[j]);
        }
#pragma unroll
        for(int r = 0; r < R; ++r)
#pragma unroll
            for(int j = 0; j < 4; ++j) S[r][j] = fmaf(k[r], delta[j], S[r][j] * e);   // the one update expression
        if(lane == 0)
#pragma unroll
            for(int j = 0; j < 4; ++j) out[std::size_t(t) * out_stride + std::size_t(h) * d.dv + j0 + j] = o[j];
        if(checkpoints) {
            float* ck = checkpoints + std::size_t(t) * d.v_heads * DK * d.dv + head;
#pragma unroll
            for(int r = 0; r < R; ++r) store4(ck + std::size_t(lane * R + r) * d.dv + j0, wide_ck, S[r]);
        }
    }
#pragma unroll
    for(int r = 0; r < R; ++r) store4(state_out + head + std::size_t(lane * R + r) * d.dv + j0, wide_out, S[r]);
}

// ---------------------------------------------------------------------------------------------- Mamba-2
struct SsdScratch { float* m; float* a; };

// Pass 1, block per (head, chunk): a_t = sum_{s <= t} dt_s A (inclusive) and M_ts = (C_t . B_s) e^{a_t - a_s} dt_s.
template <int N>
__global__ void __launch_bounds__(kThreads) ssd_prepare_kernel(SsdShape d, SsdInputs in, int rows, SsdScratch cs) {
    extern __shared__ float4 smem4[];
    float* Bs = reinterpret_cast<float*>(smem4);   // [64][N]
    float* Cs = Bs + kChunk * N;                    // [64][N]
    __shared__ float a[kChunk], dts[kChunk];
    const int h = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
    const int g = h / (d.heads / d.groups), t0 = c * kChunk, n = min(kChunk, rows - t0);
    for(int e = tid; e < kChunk * N; e += kThreads) {
        const int i = e / N, x = e % N;
        const std::size_t at = std::size_t(t0 + i) * in.bc_stride + std::size_t(g) * N + x;
        Bs[e] = i < n ? in.B[at] : 0.f;
        Cs[e] = i < n ? in.C[at] : 0.f;
    }
    if(tid < kChunk) {
        dts[tid] = tid < n ? in.dt[std::size_t(t0 + tid) * in.dt_stride + h] : 0.f;
        a[tid] = dts[tid] * in.A[h];
    }
    __syncthreads();
    if(tid == 0)
        for(int i = 1; i < kChunk; ++i) a[i] += a[i - 1];
    __syncthreads();
    const std::size_t blk = std::size_t(h) * gridDim.y + c;
    const int t = tid / 4, s0 = (tid % 4) * 16;
    float cb[16];
#pragma unroll
    for(int j = 0; j < 16; ++j) cb[j] = 0.f;
    for(int x = 0; x < N; ++x) {
        const float ct = Cs[t * N + x];
#pragma unroll
        for(int j = 0; j < 16; ++j) cb[j] = fmaf(ct, Bs[(s0 + j) * N + x], cb[j]);
    }
    float* M = cs.m + blk * kChunk * kChunk;
#pragma unroll
    for(int j = 0; j < 16; ++j) {
        const int s = s0 + j;
        M[t * kChunk + s] = s <= t ? cb[j] * expf(a[t] - a[s]) * dts[s] : 0.f;
    }
    if(tid < kChunk) cs.a[blk * kChunk + tid] = a[tid];
}

// Pass 2, block per (head, 32 rows of head_dim), chunks in order with H[32][N] in shared memory:
//   y_t = e^{a_t} H C_t + sum_{s <= t} M_ts x_s + D x_t;  H' = e^{a_Q} H + sum_s e^{a_Q - a_s} dt_s x_s B_s^T.
template <int N>
__global__ void __launch_bounds__(kThreads) ssd_scan_kernel(SsdShape d, SsdInputs in, int rows, int chunks, SsdScratch cs,
                                                            float* state, float* out, std::size_t out_stride) {
    extern __shared__ float4 smem4[];
    float* H = reinterpret_cast<float*>(smem4);   // [32][N]
    float* Bs = H + 32 * N;                        // [64][N]
    float* Cs = Bs + kChunk * N;                   // [64][N]
    float* Xs = Cs + kChunk * N;                   // [64][32]
    __shared__ float dts[kChunk];
    const int h = blockIdx.x, p0 = blockIdx.y * 32, tid = threadIdx.x;
    const int g = h / (d.heads / d.groups);
    const float Dh = in.D ? in.D[h] : 0.f;
    float* st = state + std::size_t(h) * d.head_dim * N;
    for(int e = tid; e < 32 * N; e += kThreads) H[e] = st[std::size_t(p0 + e / N) * N + e % N];
    for(int c = 0; c < chunks; ++c) {
        const int t0 = c * kChunk, n = min(kChunk, rows - t0);
        const std::size_t blk = std::size_t(h) * chunks + c;
        const float* M = cs.m + blk * kChunk * kChunk;
        const float* a = cs.a + blk * kChunk;
        __syncthreads();
        for(int e = tid; e < kChunk * N; e += kThreads) {
            const int i = e / N, x = e % N;
            const std::size_t at = std::size_t(t0 + i) * in.bc_stride + std::size_t(g) * N + x;
            Bs[e] = i < n ? in.B[at] : 0.f;
            Cs[e] = i < n ? in.C[at] : 0.f;
        }
        for(int e = tid; e < kChunk * 32; e += kThreads) {
            const int i = e / 32, p = e % 32;
            Xs[e] = i < n ? in.x[std::size_t(t0 + i) * in.x_stride + std::size_t(h) * d.head_dim + p0 + p] : 0.f;
        }
        if(tid < kChunk) dts[tid] = tid < n ? in.dt[std::size_t(t0 + tid) * in.dt_stride + h] : 0.f;
        __syncthreads();
        {   // outputs: thread owns token t = tid / 4 and 8 rows p
            const int t = tid / 4, q0 = (tid % 4) * 8;
            if(t < n) {
                float y[8], hc[8];
#pragma unroll
                for(int j = 0; j < 8; ++j) y[j] = hc[j] = 0.f;
                for(int x = 0; x < N; ++x) {
                    const float ct = Cs[t * N + x];
#pragma unroll
                    for(int j = 0; j < 8; ++j) hc[j] = fmaf(H[(q0 + j) * N + x], ct, hc[j]);
                }
                for(int s = 0; s <= t; ++s) {
                    const float m = M[t * kChunk + s];
#pragma unroll
                    for(int j = 0; j < 8; ++j) y[j] = fmaf(m, Xs[s * 32 + q0 + j], y[j]);
                }
                const float ea = expf(a[t]);
                float* orow = out + std::size_t(t0 + t) * out_stride + std::size_t(h) * d.head_dim + p0 + q0;
#pragma unroll
                for(int j = 0; j < 8; ++j) orow[j] = fmaf(Dh, Xs[t * 32 + q0 + j], fmaf(ea, hc[j], y[j]));
            }
        }
        __syncthreads();  // every reader of the old H is done
        {   // state: thread owns row p = tid / 8 and N / 8 columns
            constexpr int CPT = N / 8;
            const int p = tid / 8, n0 = (tid % 8) * CPT;
            const float aQ = a[kChunk - 1], eq = expf(aQ);
            float hv[CPT];
#pragma unroll
            for(int j = 0; j < CPT; ++j) hv[j] = H[p * N + n0 + j] * eq;
            for(int s = 0; s < kChunk; ++s) {
                const float f = expf(aQ - a[s]) * dts[s] * Xs[s * 32 + p];
#pragma unroll
                for(int j = 0; j < CPT; ++j) hv[j] = fmaf(f, Bs[s * N + n0 + j], hv[j]);
            }
#pragma unroll
            for(int j = 0; j < CPT; ++j) H[p * N + n0 + j] = hv[j];
        }
    }
    __syncthreads();
    for(int e = tid; e < 32 * N; e += kThreads) st[std::size_t(p0 + e / N) * N + e % N] = H[e];
}

// Column scan, block per (head, 8 rows of head_dim): warp w owns row p0 + w, lane l state columns l * R .. + R - 1.
template <int N>
__global__ void __launch_bounds__(kThreads) ssd_columns_kernel(SsdShape d, SsdInputs in, int rows, const float* state_in,
                                                               float* state_out, float* checkpoints, float* out,
                                                               std::size_t out_stride) {
    constexpr int R = N / 32;
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p = blockIdx.y * 8 + warp, g = h / (d.heads / d.groups);
    const std::size_t row = (std::size_t(h) * d.head_dim + p) * N;
    const float A = in.A[h], Dh = in.D ? in.D[h] : 0.f;
    float H[R];
#pragma unroll
    for(int r = 0; r < R; ++r) H[r] = state_in[row + lane * R + r];
    for(int t = 0; t < rows; ++t) {
        const float x = in.x[std::size_t(t) * in.x_stride + std::size_t(h) * d.head_dim + p];
        const float dt = in.dt[std::size_t(t) * in.dt_stride + h], dA = expf(dt * A), dx = dt * x;
        const float* Br = in.B + std::size_t(t) * in.bc_stride + std::size_t(g) * N + lane * R;
        const float* Cr = in.C + std::size_t(t) * in.bc_stride + std::size_t(g) * N + lane * R;
        float y = 0.f;
#pragma unroll
        for(int r = 0; r < R; ++r) {
            H[r] = fmaf(dx, Br[r], H[r] * dA);   // the one update expression
            y = fmaf(H[r], Cr[r], y);
        }
        y = warp_sum(y);
        if(lane == 0) out[std::size_t(t) * out_stride + std::size_t(h) * d.head_dim + p] = fmaf(Dh, x, y);
        if(checkpoints)
#pragma unroll
            for(int r = 0; r < R; ++r) checkpoints[std::size_t(t) * d.heads * d.head_dim * N + row + lane * R + r] = H[r];
    }
#pragma unroll
    for(int r = 0; r < R; ++r) state_out[row + lane * R + r] = H[r];
}

bool valid(const DeltaShape& d) {
    return d.k_heads > 0 && d.v_heads > 0 && d.v_heads % d.k_heads == 0 && (d.dk == 64 || d.dk == 128) && d.dv > 0 && d.dv % 64 == 0;
}
bool valid(const DeltaInputs& in) { return in.q && in.k && in.v && in.log_decay && in.beta; }
bool valid(const SsdShape& d) {
    return d.heads > 0 && d.groups > 0 && d.heads % d.groups == 0 && (d.d_state == 64 || d.d_state == 128) && d.head_dim > 0 &&
           d.head_dim % 32 == 0;
}
bool valid(const SsdInputs& in) { return in.x && in.dt && in.B && in.C && in.A; }

template <class K>
cudaError_t big_shared(K kernel, std::size_t bytes) {
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
}

template <int DK>
cudaError_t delta_prefill(const DeltaShape& d, const DeltaInputs& in, int rows, float* state, float* scratch, float* out,
                          std::size_t out_stride, cudaStream_t stream) {
    const int chunks = (rows + kChunk - 1) / kChunk;
    const DeltaScratch cs = delta_split<DK>(scratch, d.v_heads, chunks);
    const std::size_t prep = std::size_t(2 * kChunk * DK + 2 * kChunk * kChunk) * sizeof(float);
    const std::size_t scan = std::size_t(DK * 64 + 2 * kChunk * 64 + kChunk * DK) * sizeof(float);
    cudaError_t e = big_shared(delta_prepare_kernel<DK>, prep);
    if(e == cudaSuccess) e = big_shared(delta_scan_kernel<DK>, scan);
    if(e != cudaSuccess) return e;
    delta_prepare_kernel<DK><<<dim3(d.v_heads, chunks), kThreads, prep, stream>>>(d, in, rows, cs);
    delta_scan_kernel<DK><<<dim3(d.v_heads, d.dv / 64), kThreads, scan, stream>>>(d, in, rows, chunks, cs, state, out, out_stride);
    return cudaGetLastError();
}

template <int N>
cudaError_t ssd_prefill(const SsdShape& d, const SsdInputs& in, int rows, float* state, float* scratch, float* out,
                        std::size_t out_stride, cudaStream_t stream) {
    const int chunks = (rows + kChunk - 1) / kChunk;
    const std::size_t blocks = std::size_t(d.heads) * chunks;
    const SsdScratch cs{scratch, scratch + blocks * kChunk * kChunk};
    const std::size_t prep = std::size_t(2 * kChunk * N) * sizeof(float);
    const std::size_t scan = std::size_t(32 * N + 2 * kChunk * N + kChunk * 32) * sizeof(float);
    cudaError_t e = big_shared(ssd_prepare_kernel<N>, prep);
    if(e == cudaSuccess) e = big_shared(ssd_scan_kernel<N>, scan);
    if(e != cudaSuccess) return e;
    ssd_prepare_kernel<N><<<dim3(d.heads, chunks), kThreads, prep, stream>>>(d, in, rows, cs);
    ssd_scan_kernel<N><<<dim3(d.heads, d.head_dim / 32), kThreads, scan, stream>>>(d, in, rows, chunks, cs, state, out, out_stride);
    return cudaGetLastError();
}

}  // namespace

cudaError_t enqueue_conv(const Conv& c, int rows, cudaStream_t stream) {
    if(!rows) return cudaSuccess;
    if(!c.x || !c.weight || !c.state || !c.out || c.channels < 1 || c.width < 2 || c.width > kMaxConv || rows < 0 ||
       c.x_stride < std::size_t(c.channels) || c.out_stride < std::size_t(c.channels))
        return cudaErrorInvalidValue;
    conv_kernel<<<(c.channels + 127) / 128, 128, 0, stream>>>(c, rows);
    return cudaGetLastError();
}

std::size_t delta_scratch_floats(const DeltaShape& d, int rows) {
    const std::size_t chunks = std::size_t((rows + kChunk - 1) / kChunk);
    return std::size_t(d.v_heads) * chunks * (2 * kChunk * kChunk + std::size_t(kChunk) * d.dk + kChunk);
}

cudaError_t enqueue_delta_prefill(const DeltaShape& d, const DeltaInputs& in, int rows, float* state, float* scratch, float* out,
                                  std::size_t out_stride, cudaStream_t stream) {
    if(!rows) return cudaSuccess;
    if(!valid(d) || !valid(in) || !state || !scratch || !out || rows < 0 || d.v_heads > 65535 ||
       out_stride < std::size_t(d.v_heads) * d.dv)
        return cudaErrorInvalidValue;
    return d.dk == 64 ? delta_prefill<64>(d, in, rows, state, scratch, out, out_stride, stream)
                      : delta_prefill<128>(d, in, rows, state, scratch, out, out_stride, stream);
}

cudaError_t enqueue_delta_columns(const DeltaShape& d, const DeltaInputs& in, int rows, const float* state_in, float* state_out,
                                  float* checkpoints, float* out, std::size_t out_stride, cudaStream_t stream) {
    if(!rows) return cudaSuccess;
    if(!valid(d) || !valid(in) || !state_in || !state_out || !out || rows < 0 || out_stride < std::size_t(d.v_heads) * d.dv)
        return cudaErrorInvalidValue;
    const dim3 grid(d.v_heads, d.dv / 32);
    if(d.dk == 64) delta_columns_kernel<64><<<grid, kThreads, 0, stream>>>(d, in, rows, state_in, state_out, checkpoints, out, out_stride);
    else delta_columns_kernel<128><<<grid, kThreads, 0, stream>>>(d, in, rows, state_in, state_out, checkpoints, out, out_stride);
    return cudaGetLastError();
}

std::size_t ssd_scratch_floats(const SsdShape& d, int rows) {
    const std::size_t chunks = std::size_t((rows + kChunk - 1) / kChunk);
    return std::size_t(d.heads) * chunks * (kChunk * kChunk + kChunk);
}

cudaError_t enqueue_ssd_prefill(const SsdShape& d, const SsdInputs& in, int rows, float* state, float* scratch, float* out,
                                std::size_t out_stride, cudaStream_t stream) {
    if(!rows) return cudaSuccess;
    if(!valid(d) || !valid(in) || !state || !scratch || !out || rows < 0 || d.heads > 65535 ||
       out_stride < std::size_t(d.heads) * d.head_dim)
        return cudaErrorInvalidValue;
    return d.d_state == 64 ? ssd_prefill<64>(d, in, rows, state, scratch, out, out_stride, stream)
                           : ssd_prefill<128>(d, in, rows, state, scratch, out, out_stride, stream);
}

cudaError_t enqueue_ssd_columns(const SsdShape& d, const SsdInputs& in, int rows, const float* state_in, float* state_out,
                                float* checkpoints, float* out, std::size_t out_stride, cudaStream_t stream) {
    if(!rows) return cudaSuccess;
    if(!valid(d) || !valid(in) || !state_in || !state_out || !out || rows < 0 || out_stride < std::size_t(d.heads) * d.head_dim)
        return cudaErrorInvalidValue;
    const dim3 grid(d.heads, d.head_dim / 8);
    if(d.d_state == 64) ssd_columns_kernel<64><<<grid, kThreads, 0, stream>>>(d, in, rows, state_in, state_out, checkpoints, out, out_stride);
    else ssd_columns_kernel<128><<<grid, kThreads, 0, stream>>>(d, in, rows, state_in, state_out, checkpoints, out, out_stride);
    return cudaGetLastError();
}

}  // namespace crit::recurrent_kernels
