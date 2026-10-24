// Recurrent-layer kernels against FP64 token-by-token references: the causal convolution (outputs, carried state,
// per-token checkpoints), gated delta and Mamba-2 column scans (outputs, state, checkpoints, and a verify of several
// columns equal bit for bit to the same columns one at a time), and both chunked prefills over a prompt that spans
// several chunks and ends in a partial one.
#include "recurrent_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace rk = crit::recurrent_kernels;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

template <class T> struct Buffer {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Buffer(std::size_t count) : n(count) { ok(cudaMalloc(&p, sizeof(T) * std::max<std::size_t>(count, 1)), "malloc"); ok(cudaMemset(p, 0, sizeof(T) * std::max<std::size_t>(count, 1)), "memset"); }
    explicit Buffer(const std::vector<T>& h) : Buffer(h.size()) { put(h); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& h) { ok(cudaMemcpy(p, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> get() const { std::vector<T> h(n); ok(cudaMemcpy(h.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost), "download"); return h; }
};

double relative(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0., top = 0.;
    for(std::size_t i = 0; i < want.size(); ++i) {
        if(!std::isfinite(got[i])) return INFINITY;
        worst = std::max(worst, std::abs(double(got[i]) - want[i]));
        top = std::max(top, std::abs(want[i]));
    }
    return worst / std::max(top, 1e-30);
}

std::vector<float> normal(std::mt19937& rng, std::size_t n, float s = 1.f) {
    std::normal_distribution<float> d(0.f, s);
    std::vector<float> v(n);
    for(auto& x : v) x = d(rng);
    return v;
}

void conv_case(std::mt19937& rng) {
    constexpr int C = 300, T = 37, W = 4;
    const auto x = normal(rng, std::size_t(T) * C), w = normal(rng, std::size_t(C) * W, 0.5f), b = normal(rng, C, 0.1f);
    const auto s0 = normal(rng, std::size_t(C) * (W - 1));
    Buffer<float> dx(x), dw(w), db(b), ds(s0), dout(std::size_t(T) * C), dck(std::size_t(T) * C * (W - 1));
    rk::Conv c;
    c.x = dx.p; c.x_stride = C; c.channels = C; c.width = W; c.weight = dw.p; c.bias = db.p; c.state = ds.p; c.out = dout.p;
    c.out_stride = C; c.checkpoints = dck.p;
    ok(rk::enqueue_conv(c, T, nullptr), "conv");
    const auto out = dout.get(), st = ds.get(), ck = dck.get();
    std::vector<double> want(out.size()), wck(ck.size()), wst(st.size());
    for(int ch = 0; ch < C; ++ch) {
        std::vector<double> win(s0.begin() + std::ptrdiff_t(ch * (W - 1)), s0.begin() + std::ptrdiff_t((ch + 1) * (W - 1)));
        for(int t = 0; t < T; ++t) {
            const double xt = x[std::size_t(t) * C + std::size_t(ch)];
            double acc = b[std::size_t(ch)];
            for(int i = 0; i < W - 1; ++i) acc += double(w[std::size_t(ch) * W + std::size_t(i)]) * win[std::size_t(i)];
            acc += double(w[std::size_t(ch) * W + W - 1]) * xt;
            want[std::size_t(t) * C + std::size_t(ch)] = acc / (1. + std::exp(-acc));
            win.erase(win.begin());
            win.push_back(xt);
            for(int i = 0; i < W - 1; ++i) wck[(std::size_t(t) * C + std::size_t(ch)) * (W - 1) + std::size_t(i)] = win[std::size_t(i)];
        }
        for(int i = 0; i < W - 1; ++i) wst[std::size_t(ch) * (W - 1) + std::size_t(i)] = win[std::size_t(i)];
    }
    require(relative(out, want) < 1e-5 && relative(st, wst) == 0. && relative(ck, wck) == 0., "convolution differs from the reference");
    std::printf("conv: outputs, state and checkpoints match\n");
}

// Gated delta, FP64 token by token: delta = b (v - e S^T k), S' = e S + k delta^T, out = e S^T (s q) + ((s q) . k) delta.
void delta_reference(const rk::DeltaShape& d, const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                     const std::vector<float>& l, const std::vector<float>& b, float scale, int T, std::vector<double>& S,
                     std::vector<double>& out) {
    const std::size_t qs = std::size_t(d.k_heads) * d.dk, vs = std::size_t(d.v_heads) * d.dv;
    out.assign(std::size_t(T) * vs, 0.);
    for(int t = 0; t < T; ++t)
        for(int h = 0; h < d.v_heads; ++h) {
            const int kh = h / (d.v_heads / d.k_heads);
            double* Sh = S.data() + std::size_t(h) * d.dk * d.dv;
            const float* kr = k.data() + std::size_t(t) * qs + std::size_t(kh) * d.dk;
            const float* qr = q.data() + std::size_t(t) * qs + std::size_t(kh) * d.dk;
            const double e = std::exp(double(l[std::size_t(t) * d.v_heads + std::size_t(h)])), beta = b[std::size_t(t) * d.v_heads + std::size_t(h)];
            double qk = 0.;
            for(int x = 0; x < d.dk; ++x) qk += double(qr[x]) * scale * kr[x];
            for(int j = 0; j < d.dv; ++j) {
                double kS = 0., qS = 0.;
                for(int x = 0; x < d.dk; ++x) { kS += double(kr[x]) * Sh[std::size_t(x) * d.dv + j]; qS += double(qr[x]) * scale * Sh[std::size_t(x) * d.dv + j]; }
                const double delta = beta * (double(v[std::size_t(t) * vs + std::size_t(h) * d.dv + std::size_t(j)]) - e * kS);
                out[std::size_t(t) * vs + std::size_t(h) * d.dv + std::size_t(j)] = e * qS + qk * delta;
                for(int x = 0; x < d.dk; ++x) Sh[std::size_t(x) * d.dv + j] = e * Sh[std::size_t(x) * d.dv + j] + double(kr[x]) * delta;
            }
        }
}

void delta_case(int dk, std::mt19937& rng) {
    rk::DeltaShape d{2, 4, dk, 128};
    constexpr int T = 300;
    const std::size_t qs = std::size_t(d.k_heads) * dk, vs = std::size_t(d.v_heads) * d.dv, ss = std::size_t(d.v_heads) * dk * d.dv;
    auto q = normal(rng, T * qs), k = normal(rng, T * qs);
    for(std::size_t r = 0; r < std::size_t(T) * d.k_heads; ++r) {   // unit keys and queries, as the engine normalizes them
        double nk = 0., nq = 0.;
        for(int x = 0; x < dk; ++x) { nk += double(k[r * dk + std::size_t(x)]) * k[r * dk + std::size_t(x)]; nq += double(q[r * dk + std::size_t(x)]) * q[r * dk + std::size_t(x)]; }
        for(int x = 0; x < dk; ++x) { k[r * dk + std::size_t(x)] = float(k[r * dk + std::size_t(x)] / std::sqrt(nk)); q[r * dk + std::size_t(x)] = float(q[r * dk + std::size_t(x)] / std::sqrt(nq)); }
    }
    const auto v = normal(rng, T * vs);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::vector<float> l(std::size_t(T) * d.v_heads), b(l.size());
    for(auto& x : l) x = -0.3f * unit(rng);
    for(auto& x : b) x = unit(rng);
    const auto s0 = normal(rng, ss, 0.1f);
    const float scale = 1.f / std::sqrt(float(dk));
    Buffer<float> dq(q), dkk(k), dv(v), dl(l), db(b);
    rk::DeltaInputs in;
    in.q = dq.p; in.k = dkk.p; in.qk_stride = qs; in.v = dv.p; in.v_stride = vs; in.log_decay = dl.p; in.beta = db.p;
    in.gate_stride = std::size_t(d.v_heads); in.scale = scale;
    std::vector<double> S(s0.begin(), s0.end()), want;
    delta_reference(d, q, k, v, l, b, scale, T, S, want);
    // Prefill, chunked.
    Buffer<float> state(s0), scratch(rk::delta_scratch_floats(d, T)), out(T * vs);
    ok(rk::enqueue_delta_prefill(d, in, T, state.p, scratch.p, out.p, vs, nullptr), "delta prefill");
    const double e1 = relative(out.get(), want), e2 = relative(state.get(), S);
    std::printf("gated delta dk %d prefill: outputs %.2e, state %.2e of max\n", dk, e1, e2);
    require(e1 < 2e-3 && e2 < 2e-3, "chunked gated-delta prefill differs from the recurrence");
    // Column scan over the first 5 tokens, with checkpoints, then the same tokens one at a time.
    constexpr int Cn = 5;
    std::vector<double> S5(s0.begin(), s0.end()), want5;
    delta_reference(d, q, k, v, l, b, scale, Cn, S5, want5);
    Buffer<float> start(s0), sout(ss), ck(std::size_t(Cn) * ss), o5(Cn * vs);
    ok(rk::enqueue_delta_columns(d, in, Cn, start.p, sout.p, ck.p, o5.p, vs, nullptr), "delta columns");
    const auto got5 = o5.get(), cks = ck.get();
    require(relative(got5, want5) < 1e-4 && relative(sout.get(), S5) < 1e-4, "gated-delta column scan differs from the recurrence");
    Buffer<float> live(s0), o1(vs);
    for(int t = 0; t < Cn; ++t) {
        rk::DeltaInputs one = in;
        one.q = dq.p + std::size_t(t) * qs; one.k = dkk.p + std::size_t(t) * qs; one.v = dv.p + std::size_t(t) * vs;
        one.log_decay = dl.p + std::size_t(t) * d.v_heads; one.beta = db.p + std::size_t(t) * d.v_heads;
        ok(rk::enqueue_delta_columns(d, one, 1, live.p, live.p, nullptr, o1.p, vs, nullptr), "delta one column");
        const auto a = o1.get(), st = live.get();
        for(std::size_t i = 0; i < vs; ++i) require(a[i] == got5[std::size_t(t) * vs + i], "a gated-delta column alone differs from the verify column");
        for(std::size_t i = 0; i < ss; ++i) require(st[i] == cks[std::size_t(t) * ss + i], "a checkpoint differs from the state after its column");
    }
    std::printf("gated delta dk %d columns: recurrence, checkpoints and column invariance hold\n", dk);
}

void ssd_reference(const rk::SsdShape& d, const std::vector<float>& x, const std::vector<float>& dt, const std::vector<float>& B,
                   const std::vector<float>& C, const std::vector<float>& A, const std::vector<float>& D, int T, std::vector<double>& H,
                   std::vector<double>& out) {
    const std::size_t xs = std::size_t(d.heads) * d.head_dim, bs = std::size_t(d.groups) * d.d_state;
    out.assign(std::size_t(T) * xs, 0.);
    for(int t = 0; t < T; ++t)
        for(int h = 0; h < d.heads; ++h) {
            const int g = h / (d.heads / d.groups);
            const double dth = dt[std::size_t(t) * d.heads + std::size_t(h)], dA = std::exp(dth * A[std::size_t(h)]);
            for(int p = 0; p < d.head_dim; ++p) {
                const double xp = x[std::size_t(t) * xs + std::size_t(h) * d.head_dim + std::size_t(p)];
                double y = 0.;
                for(int n = 0; n < d.d_state; ++n) {
                    double& hv = H[(std::size_t(h) * d.head_dim + std::size_t(p)) * d.d_state + std::size_t(n)];
                    hv = dA * hv + dth * xp * B[std::size_t(t) * bs + std::size_t(g) * d.d_state + std::size_t(n)];
                    y += hv * C[std::size_t(t) * bs + std::size_t(g) * d.d_state + std::size_t(n)];
                }
                out[std::size_t(t) * xs + std::size_t(h) * d.head_dim + std::size_t(p)] = y + double(D[std::size_t(h)]) * xp;
            }
        }
}

void ssd_case(int N, std::mt19937& rng) {
    rk::SsdShape d{4, 2, 64, N};
    constexpr int T = 300;
    const std::size_t xs = std::size_t(d.heads) * d.head_dim, bs = std::size_t(d.groups) * N, ss = std::size_t(d.heads) * d.head_dim * N;
    const auto x = normal(rng, T * xs), B = normal(rng, T * bs, 0.3f), C = normal(rng, T * bs, 0.3f);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::vector<float> dt(std::size_t(T) * d.heads), A(d.heads), D(d.heads);
    for(auto& v : dt) v = 0.01f + 0.2f * unit(rng);
    for(auto& v : A) v = -0.5f - 1.5f * unit(rng);
    for(auto& v : D) v = unit(rng);
    const auto h0 = normal(rng, ss, 0.1f);
    Buffer<float> dx(x), ddt(dt), dB(B), dC(C), dA(A), dD(D);
    rk::SsdInputs in;
    in.x = dx.p; in.x_stride = xs; in.dt = ddt.p; in.dt_stride = std::size_t(d.heads); in.B = dB.p; in.C = dC.p; in.bc_stride = bs;
    in.A = dA.p; in.D = dD.p;
    std::vector<double> H(h0.begin(), h0.end()), want;
    ssd_reference(d, x, dt, B, C, A, D, T, H, want);
    Buffer<float> state(h0), scratch(rk::ssd_scratch_floats(d, T)), out(T * xs);
    ok(rk::enqueue_ssd_prefill(d, in, T, state.p, scratch.p, out.p, xs, nullptr), "ssd prefill");
    const double e1 = relative(out.get(), want), e2 = relative(state.get(), H);
    std::printf("mamba-2 d_state %d prefill: outputs %.2e, state %.2e of max\n", N, e1, e2);
    require(e1 < 2e-3 && e2 < 2e-3, "chunked Mamba-2 prefill differs from the recurrence");
    constexpr int Cn = 5;
    std::vector<double> H5(h0.begin(), h0.end()), want5;
    ssd_reference(d, x, dt, B, C, A, D, Cn, H5, want5);
    Buffer<float> start(h0), sout(ss), ck(std::size_t(Cn) * ss), o5(Cn * xs);
    ok(rk::enqueue_ssd_columns(d, in, Cn, start.p, sout.p, ck.p, o5.p, xs, nullptr), "ssd columns");
    const auto got5 = o5.get(), cks = ck.get();
    require(relative(got5, want5) < 1e-4 && relative(sout.get(), H5) < 1e-4, "Mamba-2 column scan differs from the recurrence");
    Buffer<float> live(h0), o1(xs);
    for(int t = 0; t < Cn; ++t) {
        rk::SsdInputs one = in;
        one.x = dx.p + std::size_t(t) * xs; one.dt = ddt.p + std::size_t(t) * d.heads; one.B = dB.p + std::size_t(t) * bs; one.C = dC.p + std::size_t(t) * bs;
        ok(rk::enqueue_ssd_columns(d, one, 1, live.p, live.p, nullptr, o1.p, xs, nullptr), "ssd one column");
        const auto a = o1.get(), st = live.get();
        for(std::size_t i = 0; i < xs; ++i) require(a[i] == got5[std::size_t(t) * xs + i], "a Mamba-2 column alone differs from the verify column");
        for(std::size_t i = 0; i < ss; ++i) require(st[i] == cks[std::size_t(t) * ss + i], "a Mamba-2 checkpoint differs from the state after its column");
    }
    std::printf("mamba-2 d_state %d columns: recurrence, checkpoints and column invariance hold\n", N);
}
}  // namespace

int main() {
    int count = 0;
    if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::printf("no CUDA device; skipped\n"); return 0; }
    try {
        std::mt19937 rng(29);
        conv_case(rng);
        for(const int dk : {64, 128}) delta_case(dk, rng);
        for(const int n : {64, 128}) ssd_case(n, rng);
        rk::Conv bad;
        require(rk::enqueue_conv(bad, 3, nullptr) == cudaErrorInvalidValue, "an empty convolution is rejected");
        rk::DeltaShape wrong{2, 4, 96, 128};
        rk::DeltaInputs none;
        require(rk::enqueue_delta_columns(wrong, none, 1, nullptr, nullptr, nullptr, nullptr, 0, nullptr) == cudaErrorInvalidValue,
                "an unsupported key width is rejected");
        ok(cudaDeviceSynchronize(), "sync");
        std::printf("recurrent kernels passed\n");
        return 0;
    } catch(const std::exception& e) {
        std::printf("recurrent kernels FAILED: %s\n", e.what());
        return 1;
    }
}
