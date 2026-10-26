// Attention variants in the three CUDA attention entries against an FP64 reference over the values the kernels read:
// sliding windows (with a ring buffer for the contiguous caches), per-head sinks, logit soft-caps and tree masks in
// decode/verify. Keys before the oldest query's window hold NaN (BF16 NaN, E4M3 code 0x7f, or a NaN page in a paged
// table), so any read of them shows up in the output. Windowed decode columns alone equal the same columns in a
// verify bit for bit, and invalid window, ring and tree problems are rejected.
#include "flash_prefill.h"
#include "fp8_kv_attention.h"
#include "paged_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fp = crit::flash_prefill;
namespace kv = crit::fp8_kv;
namespace pa = crit::paged_attention;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

template <class T> struct Buffer {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Buffer(std::size_t count) : n(count) { ok(cudaMalloc(&p, sizeof(T) * std::max<std::size_t>(count, 1)), "malloc"); }
    explicit Buffer(const std::vector<T>& h) : Buffer(h.size()) { put(h); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& h) { ok(cudaMemcpy(p, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> get() const { std::vector<T> h(n); ok(cudaMemcpy(h.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost), "download"); return h; }
};

float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
float e4m3(std::uint8_t b) {
    const int sign = b >> 7, exponent = (b >> 3) & 15, mantissa = b & 7;
    const float magnitude = exponent ? std::ldexp(1.f + static_cast<float>(mantissa) / 8.f, exponent - 7)
                                     : std::ldexp(static_cast<float>(mantissa) / 8.f, -6);
    return sign ? -magnitude : magnitude;
}
const __nv_bfloat16 kNaN = __float2bfloat16(std::nanf(""));
// The target of the code the runtime loads for this test, built with the kernels' flags: prefill_fp8 runs from sm_89
// code on and returns cudaErrorNotSupported below it (a build for older targets, or such code JIT-loaded on a newer GPU).
__global__ void loaded_arch(int* arch) {
#ifdef __CUDA_ARCH__
    *arch = __CUDA_ARCH__;
#endif
}
bool fp8_mma_loaded() {
    int* d = nullptr;
    int arch = 0;
    if(cudaMalloc(&d, sizeof(int)) != cudaSuccess) return false;
    loaded_arch<<<1, 1>>>(d);
    const bool ran = cudaMemcpy(&arch, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d);
    return ran && arch >= 890;
}

// One query row's visible keys, as the kernels define them.
struct Row {
    int pos = -1, lo = 0, base = INT_MAX;
    std::uint64_t tree = ~0ull;
    bool sees(int key) const { return key <= pos && key >= lo && (key < base || key - base >= 64 || ((tree >> (key - base)) & 1ull)); }
};
int window_lo(int pos, int window) { return window > 0 ? std::max(0, pos - window + 1) : 0; }

// FP64 attention of one query head over keys key(j, d) / value(j, d) of its key/value head.
template <class K, class V>
std::vector<double> reference(const std::vector<float>& q, int hd, int nkeys, const Row& row, double scale, double softcap,
                              const float* sink, K key, V value) {
    std::vector<double> s, w;
    std::vector<int> keys;
    double top = sink ? double(*sink) : -1e300;
    for(int j = 0; j < nkeys; ++j) {
        if(!row.sees(j)) continue;
        double acc = 0.;
        for(int d = 0; d < hd; ++d) acc += double(q[std::size_t(d)]) * key(j, d);
        acc *= scale;
        if(softcap > 0.) acc = softcap * std::tanh(acc / softcap);
        keys.push_back(j);
        s.push_back(acc);
        top = std::max(top, acc);
    }
    double total = sink ? std::exp(double(*sink) - top) : 0.;
    for(double x : s) { w.push_back(std::exp(x - top)); total += w.back(); }
    std::vector<double> out(std::size_t(hd), 0.);
    for(std::size_t i = 0; i < keys.size(); ++i)
        for(int d = 0; d < hd; ++d) out[std::size_t(d)] += w[i] * value(keys[i], d);
    for(auto& x : out) x /= total;
    return out;
}

// Largest |got - want| over the largest |want|; NaN anywhere fails.
double relative(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0., top = 0.;
    for(std::size_t i = 0; i < want.size(); ++i) {
        if(!std::isfinite(got[i])) return INFINITY;
        worst = std::max(worst, std::abs(double(got[i]) - want[i]));
        top = std::max(top, std::abs(want[i]));
    }
    return worst / std::max(top, 1e-30);
}

std::vector<float> rounded(std::mt19937& rng, std::size_t n) {
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> v(n);
    for(auto& x : v) x = bf(normal(rng));
    return v;
}
std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& v) {
    std::vector<__nv_bfloat16> b(v.size());
    for(std::size_t i = 0; i < v.size(); ++i) b[i] = __float2bfloat16(v[i]);
    return b;
}

// ---------------------------------------------------------------------------------------------- flash prefill
void flash_case(int hd, int ring, std::mt19937& rng) {
    const int heads = 8, kv_heads = 2, group = heads / kv_heads, T = 40, pos0 = 300, window = 100;
    const int qs = heads * hd, kvs = kv_heads * hd, L = pos0 + T, rows = ring > 0 ? ring : L;
    const int klo = window_lo(pos0, window);
    const auto q = rounded(rng, std::size_t(T) * qs);
    const auto kf = rounded(rng, std::size_t(L) * kvs), vf = rounded(rng, std::size_t(L) * kvs);
    std::vector<__nv_bfloat16> kb(std::size_t(rows) * kvs, kNaN), vb(kb.size(), kNaN);
    for(int j = klo; j < L; ++j) {
        const std::size_t r = std::size_t(ring > 0 ? j % ring : j);
        for(int e = 0; e < kvs; ++e) {
            kb[r * kvs + std::size_t(e)] = __float2bfloat16(kf[std::size_t(j) * kvs + std::size_t(e)]);
            vb[r * kvs + std::size_t(e)] = __float2bfloat16(vf[std::size_t(j) * kvs + std::size_t(e)]);
        }
    }
    std::vector<float> sinks(heads);
    for(int h = 0; h < heads; ++h) sinks[std::size_t(h)] = 0.5f * float(h) - 1.f;
    Buffer<__nv_bfloat16> dq(to_bf16(q)), dk(kb), dv(vb);
    Buffer<float> dout(std::size_t(T) * qs), dsinks(sinks);
    fp::Problem p{};
    p.q = dq.p; p.q_stride = qs; p.k = dk.p; p.v = dv.p; p.kv_stride = kvs; p.out = dout.p; p.out_stride = qs;
    p.T = T; p.pos0 = pos0; p.heads = heads; p.kv_heads = kv_heads; p.head_dim = hd; p.scale = 1.f / std::sqrt(float(hd));
    p.window = window; p.ring = ring; p.sinks = dsinks.p; p.softcap = 20.f;
    ok(fp::enqueue(p, nullptr), "flash prefill");
    const auto got = dout.get();
    std::vector<double> want;
    std::vector<float> gotv;
    for(int t = 0; t < T; ++t)
        for(int h = 0; h < heads; ++h) {
            const int g = h / group;
            Row row;
            row.pos = pos0 + t;
            row.lo = window_lo(row.pos, window);
            const std::vector<float> qv(q.begin() + std::ptrdiff_t(std::size_t(t) * qs + std::size_t(h) * hd),
                                        q.begin() + std::ptrdiff_t(std::size_t(t) * qs + std::size_t(h + 1) * hd));
            const auto r = reference(qv, hd, L, row, p.scale, p.softcap, &sinks[std::size_t(h)],
                                     [&](int j, int d) { return double(kf[std::size_t(j) * kvs + std::size_t(g * hd + d)]); },
                                     [&](int j, int d) { return double(vf[std::size_t(j) * kvs + std::size_t(g * hd + d)]); });
            want.insert(want.end(), r.begin(), r.end());
            gotv.insert(gotv.end(), got.begin() + std::ptrdiff_t(std::size_t(t) * qs + std::size_t(h) * hd),
                        got.begin() + std::ptrdiff_t(std::size_t(t) * qs + std::size_t(h + 1) * hd));
        }
    const double err = relative(gotv, want);
    std::printf("flash prefill hd %d ring %d: %.2e of max |out|\n", hd, ring, err);
    require(err < 1e-2, "flash prefill window/sink/softcap differs from the reference (or read a key before the window)");
    fp::Problem bad = p;
    bad.ring = window + T - 2;
    require(fp::enqueue(bad, nullptr) == cudaErrorInvalidValue, "a ring shorter than window + T - 1 is rejected");
    bad = p;
    bad.window = 0;
    require(ring == 0 || fp::enqueue(bad, nullptr) == cudaErrorInvalidValue, "a ring without a window is rejected");
}

// ---------------------------------------------------------------------------------------------- FP8 cache
void fp8_case(int hd, std::mt19937& rng) {
    const int heads = 8, kv_heads = 2, group = heads / kv_heads, T = 40, pos0 = 300, window = 100;
    const int ring = window + T - 1 + 11, qs = heads * hd, kvs = kv_heads * hd, L = pos0 + T;
    const int klo = window_lo(pos0, window);
    Buffer<std::uint8_t> ck(std::size_t(ring) * kvs), cv(std::size_t(ring) * kvs);
    Buffer<float> sk(std::size_t(ring) * kv_heads), sv(std::size_t(ring) * kv_heads);
    ok(cudaMemset(ck.p, 0x7f, ck.n), "poison");   // E4M3 NaN in every row the writes below do not reach
    ok(cudaMemset(cv.p, 0x7f, cv.n), "poison");
    std::vector<float> ones(sk.n, 1.f);
    sk.put(ones);
    sv.put(ones);
    kv::Cache cache{ck.p, cv.p, sk.p, sv.p, kv_heads, hd, ring};
    const auto kf = rounded(rng, std::size_t(L - klo) * kvs), vf = rounded(rng, std::size_t(L - klo) * kvs);
    Buffer<__nv_bfloat16> dk(to_bf16(kf)), dv(to_bf16(vf));
    ok(kv::write(cache, dk.p, dv.p, kvs, L - klo, klo, nullptr), "fp8 write");
    // What the kernels read: code * scale rounded to BF16, at row position % ring.
    const auto codes_k = ck.get(), codes_v = cv.get();
    const auto scales_k = sk.get(), scales_v = sv.get();
    auto key = [&](int g, int j, int d) {
        const std::size_t r = std::size_t(j % ring);
        return double(bf(e4m3(codes_k[(r * kv_heads + std::size_t(g)) * hd + std::size_t(d)]) * scales_k[r * kv_heads + std::size_t(g)]));
    };
    auto value = [&](int g, int j, int d) {
        const std::size_t r = std::size_t(j % ring);
        return double(bf(e4m3(codes_v[(r * kv_heads + std::size_t(g)) * hd + std::size_t(d)]) * scales_v[r * kv_heads + std::size_t(g)]));
    };
    std::vector<float> sinks(heads);
    for(int h = 0; h < heads; ++h) sinks[std::size_t(h)] = 1.f - 0.3f * float(h);
    Buffer<float> dsinks(sinks);
    const auto q = rounded(rng, std::size_t(T) * qs);
    Buffer<__nv_bfloat16> dq(to_bf16(q));
    const float scale = 1.f / std::sqrt(float(hd)), softcap = 25.f;

    auto check_rows = [&](const std::vector<float>& got, int columns, int first, const std::vector<Row>& rows, int nkeys,
                          const float* sink_values, const char* what, double tolerance) {
        std::vector<double> want;
        std::vector<float> gotv;
        for(int t = 0; t < columns; ++t)
            for(int h = 0; h < heads; ++h) {
                const int g = h / group;
                const std::vector<float> qv(q.begin() + std::ptrdiff_t(std::size_t(first + t) * qs + std::size_t(h) * hd),
                                            q.begin() + std::ptrdiff_t(std::size_t(first + t) * qs + std::size_t(h + 1) * hd));
                const auto r = reference(qv, hd, nkeys, rows[std::size_t(t)], scale, softcap, sink_values ? sink_values + h : nullptr,
                                         [&](int j, int d) { return key(g, j, d); }, [&](int j, int d) { return value(g, j, d); });
                want.insert(want.end(), r.begin(), r.end());
                gotv.insert(gotv.end(), got.begin() + std::ptrdiff_t(std::size_t(t) * qs + std::size_t(h) * hd),
                            got.begin() + std::ptrdiff_t(std::size_t(t) * qs + std::size_t(h + 1) * hd));
            }
        const double err = relative(gotv, want);
        std::printf("fp8 %s hd %d: %.2e of max |out|\n", what, hd, err);
        require(err < tolerance, std::string("fp8 ") + what + " differs from the reference (or read a key before the window)");
    };

    // Prompt attention, BF16 and E4M3 tensor-core paths.
    std::vector<Row> prompt(static_cast<std::size_t>(T));
    for(int t = 0; t < T; ++t) { prompt[std::size_t(t)].pos = pos0 + t; prompt[std::size_t(t)].lo = window_lo(pos0 + t, window); }
    Buffer<float> dout(std::size_t(T) * qs);
    kv::Prefill pf;
    pf.q = dq.p; pf.q_stride = qs; pf.cache = cache; pf.out = dout.p; pf.out_stride = qs; pf.T = T; pf.pos0 = pos0;
    pf.heads = heads; pf.scale = scale; pf.window = window; pf.sinks = dsinks.p; pf.softcap = softcap;
    ok(kv::prefill(pf, nullptr), "fp8 prefill");
    check_rows(dout.get(), T, 0, prompt, L, sinks.data(), "prefill", 1e-2);
    if(fp8_mma_loaded()) {
        ok(kv::prefill_fp8(pf, nullptr), "fp8 prefill_fp8");
        check_rows(dout.get(), T, 0, prompt, L, sinks.data(), "prefill_fp8", 1.2e-1);
    } else {
        require(kv::prefill_fp8(pf, nullptr) == cudaErrorNotSupported, "prefill_fp8 without E4M3 mma is refused");
    }

    // Decode and verify over the last three positions, for one and two key streams per decode block: a tree whose
    // third column hangs off the first.
    constexpr int C = 3;
    const int dpos0 = L - C, blocks = 7;
    const std::vector<std::uint64_t> tree = {0b001, 0b011, 0b101};
    Buffer<std::uint64_t> dtree(tree);
    Buffer<float> partial(kv::decode_partial_floats(kv_heads, hd, blocks)), dec(std::size_t(C) * qs);
    Buffer<__nv_bfloat16> dqd(to_bf16(std::vector<float>(q.end() - std::ptrdiff_t(std::size_t(C) * qs), q.end())));
    kv::Decode d;
    for(const int streams : {1, 2}) {
        d = kv::Decode{};
        d.q = dqd.p; d.q_stride = qs; d.cache = cache; d.out = dec.p; d.out_stride = qs; d.T = C; d.pos0 = dpos0; d.heads = heads;
        d.scale = scale; d.blocks = blocks; d.partial = partial.p; d.window = window; d.sinks = dsinks.p; d.softcap = softcap;
        d.tree = dtree.p;
        d.streams = streams;
        ok(kv::decode(d, nullptr), "fp8 tree verify");
        std::vector<Row> nodes(C);
        for(int c = 0; c < C; ++c) {
            nodes[std::size_t(c)].pos = dpos0 + c;
            nodes[std::size_t(c)].base = dpos0;
            nodes[std::size_t(c)].tree = tree[std::size_t(c)];
            const int depth = std::popcount(tree[std::size_t(c)]) - 1;
            nodes[std::size_t(c)].lo = window_lo(dpos0 + depth, window);
        }
        check_rows(dec.get(), C, T - C, nodes, L, sinks.data(), streams == 1 ? "tree verify (1 stream)" : "tree verify (2 streams)", 1e-2);
        // Chain verify without sinks: each windowed column alone equals the same column in the verify, bit for bit.
        d.tree = nullptr;
        d.sinks = nullptr;
        ok(kv::decode(d, nullptr), "fp8 chain verify");
        const auto chain = dec.get();
        for(int c = 0; c < C; ++c) {
            Buffer<__nv_bfloat16> one(to_bf16(std::vector<float>(q.end() - std::ptrdiff_t(std::size_t(C - c) * qs),
                                                                  q.end() - std::ptrdiff_t(std::size_t(C - c - 1) * qs))));
            Buffer<float> out1{std::size_t(qs)};
            kv::Decode single = d;
            single.q = one.p;
            single.out = out1.p;
            single.T = 1;
            single.pos0 = dpos0 + c;
            ok(kv::decode(single, nullptr), "fp8 single column");
            const auto alone = out1.get();
            for(int i = 0; i < qs; ++i)
                require(alone[std::size_t(i)] == chain[std::size_t(c) * qs + std::size_t(i)], "a windowed column alone differs from the verify column");
        }
    }
    kv::Decode bad = d;
    bad.tree = dtree.p;
    bad.window = 64;
    require(kv::decode(bad, nullptr) == cudaErrorInvalidValue, "a tree with a window of 64 or less is rejected");
    bad = d;
    bad.cache.ring = window;
    require(kv::decode(bad, nullptr) == cudaErrorInvalidValue, "a ring shorter than window + T - 1 is rejected");
}

// ---------------------------------------------------------------------------------------------- paged
void paged_case(int hd, pa::Format format, std::mt19937& rng) {
    const int heads = 8, kv_heads = 2, group = heads / kv_heads, page_tokens = 16, kv_len = 600, window = 100;
    const int qs = heads * hd, kvs = kv_heads * hd, npages = (kv_len + page_tokens - 1) / page_tokens;
    const int poison = npages, pages_total = npages + 1;   // page `poison` stands for no_page: never to be read
    const bool e4 = format == pa::Format::e4m3;
    const std::size_t elems = std::size_t(pages_total) * page_tokens * kvs;
    Buffer<std::uint8_t> kbytes(elems * (e4 ? 1 : 2)), vbytes(elems * (e4 ? 1 : 2));
    Buffer<float> ks(std::size_t(pages_total) * page_tokens * kv_heads), vs(ks.n);
    pa::Pages pages{kbytes.p, vbytes.p, e4 ? ks.p : nullptr, e4 ? vs.p : nullptr, format, page_tokens, kv_heads, hd};
    // Every page in its own order (page i holds keys of logical page npages - 1 - i), then NaN in the poison page.
    std::vector<std::int32_t> phys(npages);
    for(int i = 0; i < npages; ++i) phys[std::size_t(i)] = npages - 1 - i;
    const auto kf = rounded(rng, std::size_t(kv_len) * kvs), vf = rounded(rng, std::size_t(kv_len) * kvs);
    std::vector<std::int32_t> slots(kv_len);
    for(int j = 0; j < kv_len; ++j) slots[std::size_t(j)] = phys[std::size_t(j / page_tokens)] * page_tokens + j % page_tokens;
    Buffer<__nv_bfloat16> dk(to_bf16(kf)), dv(to_bf16(vf));
    Buffer<std::int32_t> dslots(slots);
    ok(pa::write(pages, dk.p, dv.p, kvs, kv_len, dslots.p, nullptr), "paged write");
    if(e4) {
        ok(cudaMemset(kbytes.p + std::size_t(poison) * page_tokens * kvs, 0x7f, std::size_t(page_tokens) * kvs), "poison");
        ok(cudaMemset(vbytes.p + std::size_t(poison) * page_tokens * kvs, 0x7f, std::size_t(page_tokens) * kvs), "poison");
    } else {
        std::vector<__nv_bfloat16> nan(std::size_t(page_tokens) * kvs, kNaN);
        ok(cudaMemcpy(kbytes.p + std::size_t(poison) * page_tokens * kvs * 2, nan.data(), nan.size() * 2, cudaMemcpyHostToDevice), "poison");
        ok(cudaMemcpy(vbytes.p + std::size_t(poison) * page_tokens * kvs * 2, nan.data(), nan.size() * 2, cudaMemcpyHostToDevice), "poison");
    }
    const auto kb = kbytes.get(), vb = vbytes.get();
    const auto ksc = ks.get(), vsc = vs.get();
    auto read = [&](const std::vector<std::uint8_t>& bytes, const std::vector<float>& sc, int g, int j, int d) {
        const std::size_t row = std::size_t(slots[std::size_t(j)]) * kv_heads + std::size_t(g);
        if(e4) return double(bf(e4m3(bytes[row * hd + std::size_t(d)]) * sc[row]));
        __nv_bfloat16 x;
        std::copy_n(bytes.data() + (row * hd + std::size_t(d)) * 2, 2, reinterpret_cast<std::uint8_t*>(&x));
        return double(__bfloat162float(x));
    };
    std::vector<float> sinks(heads);
    for(int h = 0; h < heads; ++h) sinks[std::size_t(h)] = 0.25f * float(h);
    Buffer<float> dsinks(sinks);
    const float scale = 1.f / std::sqrt(float(hd)), softcap = 30.f;

    // A table whose pages before `klo` point at the poison page, as Core's window_table leaves no_page there.
    auto table_for = [&](int klo) {
        std::vector<std::int32_t> t(npages);
        for(int i = 0; i < npages; ++i) t[std::size_t(i)] = (i + 1) * page_tokens <= klo ? poison : phys[std::size_t(i)];
        return t;
    };
    auto check = [&](const std::vector<float>& got, const std::vector<float>& q, const std::vector<Row>& rows, const float* sink,
                     const char* what) {
        std::vector<double> want;
        std::vector<float> gotv;
        for(std::size_t t = 0; t < rows.size(); ++t)
            for(int h = 0; h < heads; ++h) {
                const int g = h / group;
                const std::vector<float> qv(q.begin() + std::ptrdiff_t(t * qs + std::size_t(h) * hd), q.begin() + std::ptrdiff_t(t * qs + std::size_t(h + 1) * hd));
                const auto r = reference(qv, hd, kv_len, rows[t], scale, softcap, sink ? sink + h : nullptr,
                                         [&](int j, int d) { return read(kb, ksc, g, j, d); },
                                         [&](int j, int d) { return read(vb, vsc, g, j, d); });
                want.insert(want.end(), r.begin(), r.end());
                gotv.insert(gotv.end(), got.begin() + std::ptrdiff_t(t * qs + std::size_t(h) * hd), got.begin() + std::ptrdiff_t(t * qs + std::size_t(h + 1) * hd));
            }
        const double err = relative(gotv, want);
        std::printf("paged %s hd %d %s: %.2e of max |out|\n", what, hd, e4 ? "e4m3" : "bf16", err);
        require(err < 1e-2, std::string("paged ") + what + " differs from the reference (or read a page before the window)");
    };

    // Varlen prefill of the last 40 positions with a window.
    {
        constexpr int n = 40;
        const auto q = rounded(rng, std::size_t(n) * qs);
        Buffer<__nv_bfloat16> dq(to_bf16(q));
        Buffer<std::int32_t> tables(table_for(window_lo(kv_len - n, window))), lens(std::vector<std::int32_t>{kv_len}),
            starts(std::vector<std::int32_t>{0, n});
        Buffer<float> out(std::size_t(n) * qs);
        pa::Prefill p;
        p.q = dq.p; p.q_stride = qs; p.pages = pages; p.batch = {1, tables.p, npages, lens.p, starts.p}; p.out = out.p;
        p.out_stride = qs; p.heads = heads; p.scale = scale; p.max_queries = n; p.window = window; p.sinks = dsinks.p; p.softcap = softcap;
        ok(pa::prefill(p, nullptr), "paged prefill");
        std::vector<Row> rows(n);
        for(int t = 0; t < n; ++t) { rows[std::size_t(t)].pos = kv_len - n + t; rows[std::size_t(t)].lo = window_lo(kv_len - n + t, window); }
        check(out.get(), q, rows, sinks.data(), "prefill");
    }
    // Tree verify and chain invariance.
    {
        constexpr int C = 3;
        const int blocks = 5, pos0 = kv_len - C;
        const auto q = rounded(rng, std::size_t(C) * qs);
        Buffer<__nv_bfloat16> dq(to_bf16(q));
        Buffer<std::int32_t> tables(table_for(window_lo(pos0, window))), lens(std::vector<std::int32_t>{kv_len}),
            starts(std::vector<std::int32_t>{0, C});
        const std::vector<std::uint64_t> tree = {0b001, 0b011, 0b101};
        Buffer<std::uint64_t> dtree(tree);
        Buffer<float> out(std::size_t(C) * qs), partial(pa::decode_partial_floats(1, kv_heads, hd, blocks, C, heads));
        pa::Decode d;
        d.q = dq.p; d.q_stride = qs; d.pages = pages; d.batch = {1, tables.p, npages, lens.p, starts.p}; d.out = out.p;
        d.out_stride = qs; d.heads = heads; d.scale = scale; d.max_columns = C; d.blocks = blocks; d.partial = partial.p;
        d.window = window; d.sinks = dsinks.p; d.softcap = softcap; d.tree = dtree.p;
        ok(pa::decode(d, nullptr), "paged tree verify");
        std::vector<Row> rows(C);
        for(int c = 0; c < C; ++c) {
            rows[std::size_t(c)].pos = pos0 + c;
            rows[std::size_t(c)].base = pos0;
            rows[std::size_t(c)].tree = tree[std::size_t(c)];
            rows[std::size_t(c)].lo = window_lo(pos0 + std::popcount(tree[std::size_t(c)]) - 1, window);
        }
        check(out.get(), q, rows, sinks.data(), "tree verify");
        d.tree = nullptr;
        ok(pa::decode(d, nullptr), "paged chain verify");
        const auto chain = out.get();
        for(int c = 0; c < C; ++c) {
            Buffer<__nv_bfloat16> one(to_bf16(std::vector<float>(q.begin() + std::ptrdiff_t(std::size_t(c) * qs), q.begin() + std::ptrdiff_t(std::size_t(c + 1) * qs))));
            Buffer<std::int32_t> len1(std::vector<std::int32_t>{pos0 + c + 1}), start1(std::vector<std::int32_t>{0, 1});
            Buffer<float> out1{std::size_t(qs)};
            pa::Decode single = d;
            single.q = one.p;
            single.out = out1.p;
            single.batch.kv_lens = len1.p;
            single.batch.q_start = start1.p;
            ok(pa::decode(single, nullptr), "paged single column");
            const auto alone = out1.get();
            for(int i = 0; i < qs; ++i)
                require(alone[std::size_t(i)] == chain[std::size_t(c) * qs + std::size_t(i)], "a windowed paged column alone differs from the verify column");
        }
        pa::Decode bad = d;
        bad.tree = dtree.p;
        bad.window = 60;
        require(pa::decode(bad, nullptr) == cudaErrorInvalidValue, "a paged tree with a window of 64 or less is rejected");
    }
}
}  // namespace

int main() {
    int count = 0;
    if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::printf("no CUDA device; skipped\n"); return 0; }
    try {
        std::mt19937 rng(17);
        for(const int hd : {64, 128, 256}) {
            flash_case(hd, 0, rng);
            flash_case(hd, 100 + 40 - 1 + 7, rng);
            fp8_case(hd, rng);
            paged_case(hd, pa::Format::bf16, rng);
            paged_case(hd, pa::Format::e4m3, rng);
        }
        ok(cudaDeviceSynchronize(), "sync");
        std::printf("attention variants passed\n");
        return 0;
    } catch(const std::exception& e) {
        std::printf("attention variants FAILED: %s\n", e.what());
        return 1;
    }
}
