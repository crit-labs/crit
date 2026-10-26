// Recurrent-layer kernels against FP64 token-by-token references: the causal convolution (outputs, carried state,
// per-token checkpoints, the prompt form equal to the row-sequential form), gated delta and Mamba-2 column scans
// (outputs, state, checkpoints, and a verify of several columns equal bit for bit to the same columns one at a time),
// and both chunked prefills over a prompt that spans several chunks and ends in a partial one.
#include "recurrent_cuda.h"   // recurrent-state-recovery: enqueue_record_accepted
#include "recurrent_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace rk = crit::recurrent_kernels;
namespace rc = crit::techniques::recurrent::cuda;

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

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
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
    Buffer<float> dx(x), dw(w), db(b), ds(s0), dso(s0.size()), dout(std::size_t(T) * C), dck(std::size_t(T) * C * (W - 1));
    rk::Conv c;   // the round-start window stays in state_in; the window after the rows goes to state_out
    c.x = dx.p; c.x_stride = C; c.channels = C; c.width = W; c.weight = dw.p; c.bias = db.p; c.state_in = ds.p;
    c.state_out = dso.p; c.out = dout.p; c.out_stride = C; c.checkpoints = dck.p;
    ok(rk::enqueue_conv(c, T, nullptr), "conv");
    const auto out = dout.get(), st = dso.get(), ck = dck.get();
    require(same_bits(ds.get(), s0), "the convolution wrote its state_in");
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

// The prompt convolution out of place (a thread per channel and 32 rows) equals the row-sequential kernel (in place: x
// is the output buffer) bit for bit: outputs, checkpoints and the final window, for widths 2 to 8, with and without
// bias and SiLU, rows that do and do not fill the last segment, and state_out the same buffer as state_in or another.
void conv_parallel_case(std::mt19937& rng) {
    int cases = 0;
    for(const int W : {2, 3, 4, 5, 8})
        for(const int C : {1, 300, 1029})
            for(const int T : {33, 64, 95, 300}) {
                const bool act = (W + T) % 2 == 0, with_bias = C != 300, alias = (C + T) % 2 == 1, with_ck = T <= 95;
                const auto x = normal(rng, std::size_t(T) * C), w = normal(rng, std::size_t(C) * W, 0.5f), b = normal(rng, C, 0.1f);
                const auto s0 = normal(rng, std::size_t(C) * (W - 1));
                const std::size_t ckn = with_ck ? std::size_t(T) * C * (W - 1) : 0;
                Buffer<float> dx(x), dw(w), db(b), in0(s0), out0(s0.size()), y(std::size_t(T) * C), ck0(ckn);
                Buffer<float> seq(x), in1(s0), ck1(ckn);
                rk::Conv c;
                c.x = dx.p; c.x_stride = C; c.channels = C; c.width = W; c.weight = dw.p; c.bias = with_bias ? db.p : nullptr;
                c.silu = act; c.state_in = in0.p; c.state_out = alias ? in0.p : out0.p; c.out = y.p; c.out_stride = C;
                c.checkpoints = with_ck ? ck0.p : nullptr;
                ok(rk::enqueue_conv(c, T, nullptr), "conv, out of place");
                rk::Conv r = c;   // in place: the row-sequential kernel
                r.x = seq.p; r.out = seq.p; r.state_in = in1.p; r.state_out = in1.p; r.checkpoints = with_ck ? ck1.p : nullptr;
                ok(rk::enqueue_conv(r, T, nullptr), "conv, in place");
                const auto window = (alias ? in0 : out0).get();
                require(same_bits(y.get(), seq.get()) && same_bits(window, in1.get()) && (!with_ck || same_bits(ck0.get(), ck1.get())),
                        "the prompt convolution differs from the row-sequential one (width " + std::to_string(W) + ", " +
                            std::to_string(C) + " channels, " + std::to_string(T) + " rows)");
                if(!alias) require(same_bits(in0.get(), s0), "the prompt convolution wrote its state_in");
                ++cases;
            }
    {   // a long prompt
        constexpr int C = 1024, W = 4, T = 8192;
        const auto x = normal(rng, std::size_t(T) * C), w = normal(rng, std::size_t(C) * W, 0.5f), s0 = normal(rng, std::size_t(C) * (W - 1));
        Buffer<float> dx(x), dw(w), state(s0), y(std::size_t(T) * C), seq(x), state1(s0);
        rk::Conv c;
        c.x = dx.p; c.x_stride = C; c.channels = C; c.width = W; c.weight = dw.p; c.state_in = state.p; c.state_out = state.p;
        c.out = y.p; c.out_stride = C;
        ok(rk::enqueue_conv(c, T, nullptr), "conv, out of place");
        rk::Conv r = c;
        r.x = seq.p; r.out = seq.p; r.state_in = state1.p; r.state_out = state1.p;
        ok(rk::enqueue_conv(r, T, nullptr), "conv, in place");
        require(same_bits(y.get(), seq.get()) && same_bits(state.get(), state1.get()),
                "the 8192-row prompt convolution differs from the row-sequential one");
        ++cases;
    }
    std::printf("conv: the prompt form equals the row-sequential form bit for bit (%d cases)\n", cases);
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
    // Every verify width 1 .. 16: each column of an n-column scan equals the same token scanned alone, bit for bit.
    constexpr int Wn = 16;
    Buffer<float> serial(s0), on(std::size_t(Wn) * vs);
    std::vector<float> alone(std::size_t(Wn) * vs);
    for(int t = 0; t < Wn; ++t) {
        rk::DeltaInputs one = in;
        one.q = dq.p + std::size_t(t) * qs; one.k = dkk.p + std::size_t(t) * qs; one.v = dv.p + std::size_t(t) * vs;
        one.log_decay = dl.p + std::size_t(t) * d.v_heads; one.beta = db.p + std::size_t(t) * d.v_heads;
        ok(rk::enqueue_delta_columns(d, one, 1, serial.p, serial.p, nullptr, o1.p, vs, nullptr), "delta one column");
        const auto a = o1.get();
        std::copy(a.begin(), a.end(), alone.begin() + std::ptrdiff_t(std::size_t(t) * vs));
    }
    for(int n = 1; n <= Wn; ++n) {
        ok(rk::enqueue_delta_columns(d, in, n, start.p, sout.p, nullptr, on.p, vs, nullptr), "delta columns");
        const auto got = on.get();
        require(std::memcmp(got.data(), alone.data(), std::size_t(n) * vs * sizeof(float)) == 0,
                "a gated-delta verify of " + std::to_string(n) + " columns differs from its columns scanned alone");
    }
    std::printf("gated delta dk %d columns: recurrence, checkpoints and column invariance hold (widths 1..%d)\n", dk, Wn);
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

// The checkpoint contract a speculative round's rollback rests on (Core's column_checkpoints): after a T-column scan,
// checkpoint i is bit for bit the final state of a separate scan over the first i + 1 tokens from the same start, for
// gated delta (dk 64 and 128, 16 to 64 value heads), the causal convolution and Mamba-2, T in {1, 2, 4, 8}, three
// seeds each; a checkpoint plane past the round keeps its sentinel bytes (the scan writes exactly T planes).
template <class Scan> void checkpoint_planes(const char* what, std::size_t plane, Scan scan) {
    for(const int T : {1, 2, 4, 8}) {
        Buffer<float> planes(std::size_t(T + 1) * plane);
        ok(cudaMemset(planes.p, 0x3e, sizeof(float) * planes.n), "sentinel");
        scan(T, planes.p, nullptr);
        const auto all = planes.get();
        for(int n = 1; n <= T; ++n) {
            Buffer<float> final_state(plane);
            scan(n, nullptr, final_state.p);
            const auto got = final_state.get();
            require(std::memcmp(got.data(), all.data() + std::size_t(n - 1) * plane, plane * sizeof(float)) == 0,
                    std::string(what) + ": checkpoint " + std::to_string(n - 1) + " of " + std::to_string(T) +
                        " differs from the state after " + std::to_string(n) + " tokens");
        }
        const auto* tail = reinterpret_cast<const unsigned char*>(all.data() + std::size_t(T) * plane);
        for(std::size_t i = 0; i < plane * sizeof(float); ++i) require(tail[i] == 0x3e, std::string(what) + ": a plane past the round was written");
    }
}

void checkpoint_contract(std::mt19937& rng) {
    int cells = 0;
    for(int seed = 0; seed < 3; ++seed) {
        for(const int dk : {64, 128})
            for(const int v_heads : {16, 32, 48, 64}) {
                rk::DeltaShape d{v_heads / 4, v_heads, dk, dk};
                constexpr int T = 8;
                const std::size_t qs = std::size_t(d.k_heads) * dk, vs = std::size_t(d.v_heads) * d.dv, ss = std::size_t(d.v_heads) * dk * d.dv;
                auto q = normal(rng, T * qs, 0.125f), k = normal(rng, T * qs, 0.125f);
                const auto v = normal(rng, T * vs), s0 = normal(rng, ss, 0.1f);
                std::uniform_real_distribution<float> unit(0.f, 1.f);
                std::vector<float> l(std::size_t(T) * d.v_heads), b(l.size());
                for(auto& x : l) x = -0.3f * unit(rng);
                for(auto& x : b) x = unit(rng);
                Buffer<float> dq(q), dkk(k), dv(v), dl(l), db(b), start(s0), out(T * vs), scratch_state(ss);
                rk::DeltaInputs in;
                in.q = dq.p; in.k = dkk.p; in.qk_stride = qs; in.v = dv.p; in.v_stride = vs; in.log_decay = dl.p; in.beta = db.p;
                in.gate_stride = std::size_t(d.v_heads); in.scale = 1.f / std::sqrt(float(dk));
                checkpoint_planes("gated delta", ss, [&](int n, float* planes, float* final_state) {
                    ok(rk::enqueue_delta_columns(d, in, n, start.p, final_state ? final_state : scratch_state.p, planes, out.p, vs, nullptr), "delta columns");
                });
                ++cells;
            }
        {
            constexpr int C = 1024, W = 4, T = 8;
            const auto x = normal(rng, std::size_t(T) * C), w = normal(rng, std::size_t(C) * W, 0.5f), s0 = normal(rng, std::size_t(C) * (W - 1));
            Buffer<float> dx(x), dw(w), start(s0), scratch_state(std::size_t(C) * (W - 1)), out(std::size_t(T) * C);
            checkpoint_planes("conv", std::size_t(C) * (W - 1), [&](int n, float* planes, float* final_state) {
                rk::Conv c;
                c.x = dx.p; c.x_stride = C; c.channels = C; c.width = W; c.weight = dw.p; c.state_in = start.p;
                c.state_out = final_state ? final_state : scratch_state.p; c.out = out.p; c.out_stride = C; c.checkpoints = planes;
                ok(rk::enqueue_conv(c, n, nullptr), "conv");
            });
            ++cells;
        }
        for(const int N : {64, 128}) {
            rk::SsdShape d{16, 2, 64, N};
            constexpr int T = 8;
            const std::size_t xs = std::size_t(d.heads) * d.head_dim, bs = std::size_t(d.groups) * N, ss = std::size_t(d.heads) * d.head_dim * N;
            const auto x = normal(rng, T * xs), B = normal(rng, T * bs, 0.3f), Cm = normal(rng, T * bs, 0.3f), h0 = normal(rng, ss, 0.1f);
            std::uniform_real_distribution<float> unit(0.f, 1.f);
            std::vector<float> dt(std::size_t(T) * d.heads), A(d.heads);
            for(auto& v : dt) v = 0.01f + 0.2f * unit(rng);
            for(auto& v : A) v = -0.5f - 1.5f * unit(rng);
            Buffer<float> dx(x), ddt(dt), dB(B), dC(Cm), dA(A), start(h0), out(T * xs), scratch_state(ss);
            rk::SsdInputs in;
            in.x = dx.p; in.x_stride = xs; in.dt = ddt.p; in.dt_stride = std::size_t(d.heads); in.B = dB.p; in.C = dC.p; in.bc_stride = bs;
            in.A = dA.p;
            checkpoint_planes("mamba-2", ss, [&](int n, float* planes, float* final_state) {
                ok(rk::enqueue_ssd_columns(d, in, n, start.p, final_state ? final_state : scratch_state.p, planes, out.p, xs, nullptr), "ssd columns");
            });
            ++cells;
        }
    }
    // State and checkpoint pointers off a 16-byte boundary take the scalar accesses and give the same bits.
    {
        rk::DeltaShape d{2, 4, 64, 64};
        constexpr int T = 4;
        const std::size_t qs = 128, vs = 256, ss = std::size_t(4) * 64 * 64;
        const auto q = normal(rng, T * qs, 0.125f), k = normal(rng, T * qs, 0.125f), v = normal(rng, T * vs), s0 = normal(rng, ss, 0.1f);
        std::vector<float> l(std::size_t(T) * 4, -0.2f), b(l.size(), 0.5f);
        Buffer<float> dq(q), dkk(k), dv(v), dl(l), db(b), start(s0), out(T * vs), shifted_start(ss + 1), so(ss), sso(ss + 1), ck(T * ss), sck(T * ss + 1);
        ok(cudaMemcpy(shifted_start.p + 1, start.p, sizeof(float) * ss, cudaMemcpyDeviceToDevice), "shift");
        rk::DeltaInputs in;
        in.q = dq.p; in.k = dkk.p; in.qk_stride = qs; in.v = dv.p; in.v_stride = vs; in.log_decay = dl.p; in.beta = db.p; in.gate_stride = 4;
        ok(rk::enqueue_delta_columns(d, in, T, start.p, so.p, ck.p, out.p, vs, nullptr), "aligned");
        ok(rk::enqueue_delta_columns(d, in, T, shifted_start.p + 1, sso.p + 1, sck.p + 1, out.p, vs, nullptr), "misaligned");
        const auto a = ck.get(), m = sck.get(), fa = so.get(), fm = sso.get();
        require(std::memcmp(a.data(), m.data() + 1, a.size() * sizeof(float)) == 0 && std::memcmp(fa.data(), fm.data() + 1, fa.size() * sizeof(float)) == 0,
                "misaligned state or checkpoint pointers change the bits");
    }
    std::printf("checkpoint contract: %d shapes x T in {1, 2, 4, 8}, every checkpoint equals the shorter scan's state\n", cells);
}

// Long decode with slow decays (e^l within about 1e-6 .. 1e-3 of 1): the state after 8192 tokens against FP64. A decay
// applied token after token compounds any bias in its rounding, so a column scan's error must stay within twice that of
// a host FP32 recurrence with correctly rounded decays and the scan's update expression; CUDA's expf, which rounds up
// more often than down, gave 8x on these inputs. One call of 8192 columns is 8192 decode steps (column invariance).
void drift_case(std::mt19937& rng) {
    constexpr int T = 8192;
    std::string drifted;
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    {
        rk::DeltaShape d{1, 4, 128, 64};
        const int dk = d.dk;
        const std::size_t qs = std::size_t(dk), vs = std::size_t(d.v_heads) * d.dv, ss = std::size_t(d.v_heads) * dk * d.dv;
        auto q = normal(rng, T * qs), k = normal(rng, T * qs);
        for(std::size_t r = 0; r < std::size_t(T); ++r) {
            double nk = 0., nq = 0.;
            for(int x = 0; x < dk; ++x) { nk += double(k[r * qs + std::size_t(x)]) * k[r * qs + std::size_t(x)]; nq += double(q[r * qs + std::size_t(x)]) * q[r * qs + std::size_t(x)]; }
            for(int x = 0; x < dk; ++x) { k[r * qs + std::size_t(x)] = float(k[r * qs + std::size_t(x)] / std::sqrt(nk)); q[r * qs + std::size_t(x)] = float(q[r * qs + std::size_t(x)] / std::sqrt(nq)); }
        }
        const auto v = normal(rng, T * vs), s0 = normal(rng, ss);
        std::vector<float> l(std::size_t(T) * d.v_heads), b(l.size());
        for(std::size_t i = 0; i < l.size(); ++i) l[i] = float(-1e-6 * std::pow(10., double(i % 4)) * (1. + 0.5 * unit(rng)));   // head h: ~10^(h-6)
        for(auto& x : b) x = 0.1f * unit(rng);
        std::vector<double> S(s0.begin(), s0.end()), want;
        delta_reference(d, q, k, v, l, b, 1.f, T, S, want);
        std::vector<float> S32(s0);   // FP32, correctly rounded decays, fmaf(k, delta, S * e)
        for(int t = 0; t < T; ++t)
            for(int h = 0; h < d.v_heads; ++h) {
                const float e = float(std::exp(double(l[std::size_t(t) * d.v_heads + std::size_t(h)]))), beta = b[std::size_t(t) * d.v_heads + std::size_t(h)];
                const float* kr = k.data() + std::size_t(t) * qs;
                float* Sh = S32.data() + std::size_t(h) * dk * d.dv;
                for(int j = 0; j < d.dv; ++j) {
                    float kS = 0.f;
                    for(int x = 0; x < dk; ++x) kS = std::fma(kr[x], Sh[std::size_t(x) * d.dv + std::size_t(j)], kS);
                    const float delta = beta * std::fma(-e, kS, v[std::size_t(t) * vs + std::size_t(h) * d.dv + std::size_t(j)]);
                    for(int x = 0; x < dk; ++x) Sh[std::size_t(x) * d.dv + std::size_t(j)] = std::fma(kr[x], delta, Sh[std::size_t(x) * d.dv + std::size_t(j)] * e);
                }
            }
        Buffer<float> dq(q), dkk(k), dv(v), dl(l), db(b), state(s0), out(T * vs);
        rk::DeltaInputs in;
        in.q = dq.p; in.k = dkk.p; in.qk_stride = qs; in.v = dv.p; in.v_stride = vs; in.log_decay = dl.p; in.beta = db.p;
        in.gate_stride = std::size_t(d.v_heads);
        ok(rk::enqueue_delta_columns(d, in, T, state.p, state.p, nullptr, out.p, vs, nullptr), "delta columns");
        const double got = relative(state.get(), S), floor32 = relative(S32, S);
        std::printf("gated delta, %d decode steps with slow decays: state %.2e of max (FP32 recurrence %.2e)\n", T, got, floor32);
        if(got > 2. * floor32) drifted += " gated delta";
    }
    {
        rk::SsdShape d{4, 1, 32, 64};
        const int N = d.d_state;
        const std::size_t xs = std::size_t(d.heads) * d.head_dim, bs = std::size_t(N), ss = std::size_t(d.heads) * d.head_dim * N;
        const auto x = normal(rng, T * xs), B = normal(rng, T * bs, 0.3f), C = normal(rng, T * bs, 0.3f), h0 = normal(rng, ss);
        std::vector<float> dt(std::size_t(T) * d.heads), A(d.heads), D(d.heads, 0.f);
        for(int h = 0; h < d.heads; ++h) A[std::size_t(h)] = float(-1e-5 * std::pow(10., h));
        for(auto& v : dt) v = 0.05f + 0.1f * unit(rng);   // dt A: head h ~10^(h-6)
        std::vector<double> H(h0.begin(), h0.end()), want;
        ssd_reference(d, x, dt, B, C, A, D, T, H, want);
        std::vector<float> H32(h0);
        for(int t = 0; t < T; ++t)
            for(int h = 0; h < d.heads; ++h) {
                const float dth = dt[std::size_t(t) * d.heads + std::size_t(h)], dA = float(std::exp(double(dth) * A[std::size_t(h)]));
                for(int p = 0; p < d.head_dim; ++p) {
                    const float dx = dth * x[std::size_t(t) * xs + std::size_t(h) * d.head_dim + std::size_t(p)];
                    for(int n = 0; n < N; ++n) {
                        float& hv = H32[(std::size_t(h) * d.head_dim + std::size_t(p)) * N + std::size_t(n)];
                        hv = std::fma(dx, B[std::size_t(t) * bs + std::size_t(n)], hv * dA);
                    }
                }
            }
        Buffer<float> dx(x), ddt(dt), dB(B), dC(C), dA(A), state(h0), out(T * xs);
        rk::SsdInputs in;
        in.x = dx.p; in.x_stride = xs; in.dt = ddt.p; in.dt_stride = std::size_t(d.heads); in.B = dB.p; in.C = dC.p; in.bc_stride = bs;
        in.A = dA.p;
        ok(rk::enqueue_ssd_columns(d, in, T, state.p, state.p, nullptr, out.p, xs, nullptr), "ssd columns");
        const double got = relative(state.get(), H), floor32 = relative(H32, H);
        std::printf("mamba-2, %d decode steps with slow decays: state %.2e of max (FP32 recurrence %.2e)\n", T, got, floor32);
        if(got > 2. * floor32) drifted += " Mamba-2";
    }
    require(drifted.empty(), "column scans drift from the recurrence over a long decode:" + drifted);
}

// The deferred commit (recurrent-state-recovery's Strategy C) on the column scan and the convolution against plain
// decoding, bit for bit: rounds of 1 to `capacity` columns whose first n columns are the true next tokens and the rest
// drafts from elsewhere, recorded with n (0 and every column included), never recorded (abandoned) or recorded again
// with a smaller count (a host truncation). Every accepted column's outputs must equal the same token decoded alone from
// the plain state and window; after a flush (sometimes repeated) the live state and window must equal the plain ones,
// also across a prompt chunk run on both paths. Two layers share one record call (their control blocks consecutive).
void deferred_case(const rk::DeltaShape& d, int width, int capacity, int rounds, std::mt19937& rng) {
    constexpr int P = 2048, layers = 2;
    const int channels = 160 + width;   // two convolution blocks, the second partial
    const std::size_t qs = std::size_t(d.k_heads) * d.dk, vs = std::size_t(d.v_heads) * d.dv, ss = std::size_t(d.v_heads) * d.dk * d.dv;
    const std::size_t ws = std::size_t(channels) * (width - 1);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    auto q = normal(rng, P * qs), k = normal(rng, P * qs);
    for(std::size_t r = 0; r < std::size_t(P) * d.k_heads; ++r) {
        double nk = 0.;
        for(int x = 0; x < d.dk; ++x) nk += double(k[r * d.dk + std::size_t(x)]) * k[r * d.dk + std::size_t(x)];
        for(int x = 0; x < d.dk; ++x) k[r * d.dk + std::size_t(x)] = float(k[r * d.dk + std::size_t(x)] / std::sqrt(nk));
    }
    const auto v = normal(rng, P * vs), xs = normal(rng, std::size_t(P) * channels);
    std::vector<float> l(std::size_t(P) * d.v_heads), b(l.size());
    for(auto& x : l) x = -0.3f * unit(rng);
    for(auto& x : b) x = unit(rng);
    Buffer<float> dq(q), dkk(k), dv(v), dl(l), db(b), dx(xs);
    auto pool = [&](int t) {   // the true token stream from position t
        rk::DeltaInputs in;
        in.q = dq.p + std::size_t(t) * qs; in.k = dkk.p + std::size_t(t) * qs; in.qk_stride = qs; in.v = dv.p + std::size_t(t) * vs;
        in.v_stride = vs; in.log_decay = dl.p + std::size_t(t) * d.v_heads; in.beta = db.p + std::size_t(t) * d.v_heads;
        in.gate_stride = std::size_t(d.v_heads); in.scale = 1.f / std::sqrt(float(d.dk));
        return in;
    };
    const int conv_slots = rk::conv_deferred_slots(channels), delta_slots = rk::delta_deferred_slots(d);
    const int control_ints = conv_slots + 2 * delta_slots;   // per layer: conv counts, delta counts, delta parities
    Buffer<std::int32_t> control(std::size_t(layers) * control_ints), count(1);
    struct Layer {
        Buffer<float> w, plain_window, plain_state, window, state, conv_stash, stash;
        rk::ConvDeferred cd;
        rk::DeltaDeferred dd;
    };
    std::vector<std::unique_ptr<Layer>> L;
    for(int i = 0; i < layers; ++i) {
        const auto w0 = normal(rng, std::size_t(channels) * width, 0.5f), win0 = normal(rng, ws), s0 = normal(rng, ss, 0.1f);
        L.push_back(std::unique_ptr<Layer>(new Layer{Buffer<float>(w0), Buffer<float>(win0), Buffer<float>(s0), Buffer<float>(win0),
                                                     Buffer<float>(s0), Buffer<float>(std::size_t(capacity) * channels),
                                                     Buffer<float>(rk::delta_stash_floats(d, capacity)), {}, {}}));
        Layer& y = *L.back();
        std::int32_t* c = control.p + std::size_t(i) * control_ints;
        y.cd.window = y.window.p; y.cd.rows_stash = y.conv_stash.p; y.cd.capacity = capacity; y.cd.pending = c;
        y.dd.stash = y.stash.p; y.dd.capacity = capacity; y.dd.control = c + conv_slots;
    }
    auto conv_of = [&](Layer& y, const float* x, float* out) {
        rk::Conv c;
        c.x = x; c.x_stride = std::size_t(channels); c.channels = channels; c.width = width; c.weight = y.w.p; c.silu = true;
        c.state_in = y.plain_window.p; c.state_out = y.plain_window.p; c.out = out; c.out_stride = std::size_t(channels);
        return c;
    };
    auto record = [&](int n, int columns) {
        count.put(std::vector<std::int32_t>{n});
        ok(rc::enqueue_record_accepted(count.p, std::uint32_t(columns), control.p, std::uint32_t(conv_slots + delta_slots), layers,
                                       std::size_t(control_ints), nullptr), "record");
    };
    auto flush = [&](Layer& y) {
        rk::Conv c;
        c.channels = channels; c.width = width;
        ok(rk::enqueue_conv_deferred(c, y.cd, 0, nullptr), "conv flush");
        ok(rk::enqueue_delta_columns_deferred(d, rk::DeltaInputs{}, 0, y.state.p, y.dd, nullptr, 0, nullptr), "delta flush");
    };
    int mismatched = 0, compared = 0, pos = 0, columns = 0, zero = 0, full = 0, abandoned = 0, truncated = 0, flushes = 0, prompts = 0;
    auto compare = [&] {
        for(auto& y : L) {
            mismatched += !same_bits(y->window.get(), y->plain_window.get()) || !same_bits(y->state.get(), y->plain_state.get());
            ++compared;
        }
    };
    Buffer<float> rq(std::size_t(capacity) * qs), rkk(std::size_t(capacity) * qs), rv(std::size_t(capacity) * vs), rl(std::size_t(capacity) * d.v_heads),
        rb(std::size_t(capacity) * d.v_heads), rx(std::size_t(capacity) * channels), conv_out(std::size_t(capacity) * channels),
        delta_out(std::size_t(capacity) * vs), one_conv{std::size_t(channels)}, one_out(vs);
    std::vector<float> hq(rq.n), hk(rkk.n), hv(rv.n), hl(rl.n), hb(rb.n), hx(rx.n);
    Buffer<float> chunk_out(64 * std::max<std::size_t>(vs, std::size_t(channels))), scratch(rk::delta_scratch_floats(d, 64));
    for(int r = 0; r < rounds; ++r) {
        const int w = 1 + int(rng() % unsigned(capacity));
        const unsigned pick = rng() % 100;
        const int n = pick < 15 ? 0 : pick < 30 ? w : int(rng() % unsigned(w + 1));
        if(pos + w + 64 >= P) break;
        for(int i = 0; i < w; ++i) {   // the round's rows: n true tokens, then drafts from elsewhere in the pool
            const int t = i < n ? pos + i : (pos + i + 389 + int(rng() % 200)) % P;
            std::copy_n(q.begin() + std::ptrdiff_t(std::size_t(t) * qs), qs, hq.begin() + std::ptrdiff_t(std::size_t(i) * qs));
            std::copy_n(k.begin() + std::ptrdiff_t(std::size_t(t) * qs), qs, hk.begin() + std::ptrdiff_t(std::size_t(i) * qs));
            std::copy_n(v.begin() + std::ptrdiff_t(std::size_t(t) * vs), vs, hv.begin() + std::ptrdiff_t(std::size_t(i) * vs));
            std::copy_n(l.begin() + std::ptrdiff_t(std::size_t(t) * d.v_heads), d.v_heads, hl.begin() + std::ptrdiff_t(std::size_t(i) * d.v_heads));
            std::copy_n(b.begin() + std::ptrdiff_t(std::size_t(t) * d.v_heads), d.v_heads, hb.begin() + std::ptrdiff_t(std::size_t(i) * d.v_heads));
            std::copy_n(xs.begin() + std::ptrdiff_t(std::size_t(t) * channels), channels, hx.begin() + std::ptrdiff_t(std::size_t(i) * channels));
        }
        rq.put(hq); rkk.put(hk); rv.put(hv); rl.put(hl); rb.put(hb); rx.put(hx);
        rk::DeltaInputs in = pool(0);
        in.q = rq.p; in.k = rkk.p; in.v = rv.p; in.log_decay = rl.p; in.beta = rb.p;
        std::vector<std::vector<float>> conv_rows(layers), delta_rows(layers);
        for(int i = 0; i < layers; ++i) {
            Layer& y = *L[std::size_t(i)];
            ok(rk::enqueue_conv_deferred(conv_of(y, rx.p, conv_out.p), y.cd, w, nullptr), "conv deferred");
            ok(rk::enqueue_delta_columns_deferred(d, in, w, y.state.p, y.dd, delta_out.p, vs, nullptr), "delta deferred");
            conv_rows[std::size_t(i)] = conv_out.get();
            delta_rows[std::size_t(i)] = delta_out.get();
        }
        columns += w;
        int accepted = n;
        if(rng() % 10 == 0) {
            accepted = 0;
            ++abandoned;
        } else {
            record(n, w);
            if(n > 0 && rng() % 5 == 0) {
                accepted = int(rng() % unsigned(n));
                record(accepted, w);
                ++truncated;
            }
            zero += n == 0;
            full += n == w;
        }
        for(int t = 0; t < accepted; ++t)   // plain decoding of the accepted tokens
            for(int i = 0; i < layers; ++i) {
                Layer& y = *L[std::size_t(i)];
                ok(rk::enqueue_conv(conv_of(y, dx.p + std::size_t(pos + t) * channels, one_conv.p), 1, nullptr), "conv");
                ok(rk::enqueue_delta_columns(d, pool(pos + t), 1, y.plain_state.p, y.plain_state.p, nullptr, one_out.p, vs, nullptr), "delta");
                const auto a = one_conv.get(), o = one_out.get();
                mismatched += std::memcmp(a.data(), conv_rows[std::size_t(i)].data() + std::size_t(t) * channels, a.size() * sizeof(float)) != 0;
                mismatched += std::memcmp(o.data(), delta_rows[std::size_t(i)].data() + std::size_t(t) * vs, o.size() * sizeof(float)) != 0;
                compared += 2;
            }
        pos += accepted;
        if(rng() % 4 == 0) {
            const bool twice = rng() % 3 == 0;
            for(auto& y : L) {
                flush(*y);
                if(twice) flush(*y);
            }
            ++flushes;
            compare();
            if(rng() % 4 == 0) {   // a prompt chunk on both paths: the prompt convolution and the chunked prefill
                const int m = 33 + int(rng() % 31);
                for(auto& y : L)
                    for(const bool plain : {true, false}) {
                        rk::Conv c = conv_of(*y, dx.p + std::size_t(pos) * channels, chunk_out.p);
                        if(!plain) { c.state_in = y->window.p; c.state_out = y->window.p; }
                        ok(rk::enqueue_conv(c, m, nullptr), "prompt conv");
                        ok(rk::enqueue_delta_prefill(d, pool(pos), m, plain ? y->plain_state.p : y->state.p, scratch.p, chunk_out.p, vs, nullptr),
                           "prefill");
                    }
                pos += m;
                ++prompts;
                compare();
            }
        }
    }
    for(auto& y : L) flush(*y);
    compare();
    require(mismatched == 0, "the deferred commit differs from plain decoding in " + std::to_string(mismatched) + " of " +
                                 std::to_string(compared) + " comparisons (dk " + std::to_string(d.dk) + ", dv " + std::to_string(d.dv) + ")");
    std::printf("deferred commit k%d v%d dk%d dv%d, width %d, capacity %d: %d columns, %d positions decoded (%d rounds recorded 0, %d full, "
                "%d abandoned, %d truncated), %d flushes, %d prompt chunks: bit for bit plain decoding\n",
                d.k_heads, d.v_heads, d.dk, d.dv, width, capacity, columns, pos, zero, full, abandoned, truncated, flushes, prompts);
}

void deferred_rejects() {
    rk::DeltaShape d{1, 2, 64, 64};
    Buffer<float> f(rk::delta_stash_floats(d, 4) + 64 * 64 * 2);
    Buffer<std::int32_t> c(64);
    rk::DeltaDeferred dd;
    dd.stash = f.p; dd.capacity = 4; dd.control = c.p;
    rk::DeltaInputs none;
    require(rk::enqueue_delta_columns_deferred(d, none, 5, f.p, dd, f.p, 128, nullptr) == cudaErrorInvalidValue, "a verify above the stash capacity is accepted");
    require(rk::enqueue_delta_columns_deferred(d, none, 1, f.p, dd, f.p, 128, nullptr) == cudaErrorInvalidValue, "a verify without inputs is accepted");
    dd.capacity = 0;
    require(rk::enqueue_delta_columns_deferred(d, none, 0, f.p, dd, nullptr, 0, nullptr) == cudaErrorInvalidValue, "a stash of capacity 0 is accepted");
    rk::ConvDeferred cd;
    cd.window = f.p; cd.rows_stash = f.p; cd.capacity = 2; cd.pending = c.p;
    rk::Conv conv;
    conv.channels = 8; conv.width = 4;
    require(rk::enqueue_conv_deferred(conv, cd, 3, nullptr) == cudaErrorInvalidValue, "a deferred convolution above the stash capacity is accepted");
    require(rk::delta_deferred_slots(rk::DeltaShape{1, 2, 96, 64}) == 0 && rk::delta_stash_floats(d, 0) == 0, "sizes of an invalid shape");
    require(rc::enqueue_record_accepted(c.p, 1, c.p, 4, 2, 3, nullptr) == cudaErrorInvalidValue, "a record with overlapping groups is accepted");
}

}  // namespace

int main() {
    int count = 0;
    if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::printf("no CUDA device; skipped\n"); return 0; }
    try {
        std::mt19937 rng(29);
        conv_case(rng);
        conv_parallel_case(rng);
        for(const int dk : {64, 128}) delta_case(dk, rng);
        for(const int n : {64, 128}) ssd_case(n, rng);
        checkpoint_contract(rng);
        drift_case(rng);
        for(const rk::DeltaShape d : {rk::DeltaShape{2, 4, 64, 64}, rk::DeltaShape{1, 3, 128, 128}, rk::DeltaShape{2, 8, 128, 64},
                                      rk::DeltaShape{4, 4, 64, 128}})
            deferred_case(d, 2 + int(rng() % 7u), d.dk == 64 ? 9 : 16, 160, rng);
        deferred_rejects();
        rk::Conv bad;
        require(rk::enqueue_conv(bad, 3, nullptr) == cudaErrorInvalidValue, "an empty convolution is rejected");
        {
            Buffer<float> x(64 * 8), w(8 * 4), st(8 * 3);
            rk::Conv c;
            c.x = x.p; c.x_stride = 8; c.channels = 8; c.width = 4; c.weight = w.p; c.state_in = st.p; c.out = x.p; c.out_stride = 8;
            require(rk::enqueue_conv(c, 3, nullptr) == cudaErrorInvalidValue, "a convolution without state_out is rejected");
            c.state_out = st.p; c.x_stride = 16;
            require(rk::enqueue_conv(c, 3, nullptr) == cudaErrorInvalidValue, "an in-place convolution with two strides is rejected");
        }
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
