// FP8 KV cache attention against FP64 references over the dequantized cache, and decode column invariance (for every
// verify width at 2048 to 16384 keys).
#include "fp8_kv_attention.h"

#include <cuda_fp8.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace kv = crit::fp8_kv;

namespace {

float e4m3(std::uint8_t b) {
    const int sign = b >> 7, exponent = (b >> 3) & 15, mantissa = b & 7;
    const float magnitude = exponent ? std::ldexp(1.f + mantissa / 8.f, exponent - 7) : std::ldexp(mantissa / 8.f, -6);
    return sign ? -magnitude : magnitude;
}
float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
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

struct Device {
    std::vector<void*> owned;
    template <class T> T* alloc(std::size_t n) { void* p = nullptr; cudaMalloc(&p, n * sizeof(T)); cudaMemset(p, 0, n * sizeof(T)); owned.push_back(p); return static_cast<T*>(p); }
    ~Device() { for(void* p : owned) cudaFree(p); }
};

struct Setup {
    int hd, heads, kv_heads, length;
    Device dev;
    kv::Cache cache;
    std::vector<__nv_bfloat16> k, v;
    std::vector<float> kd, vd;  // the cache's values as the kernels see them: bf16(code * scale)
    float worst_write = 0.f;

    Setup(int hd_, int heads_, int kv_heads_, int length_, std::mt19937& rng) : hd(hd_), heads(heads_), kv_heads(kv_heads_), length(length_) {
        const std::size_t n = static_cast<std::size_t>(length) * kv_heads * hd;
        std::normal_distribution<float> normal(0.f, 1.f);
        k.resize(n); v.resize(n);
        for(auto& x : k) x = __float2bfloat16(normal(rng));
        for(auto& x : v) x = __float2bfloat16(2.f * normal(rng));
        cache = {dev.alloc<std::uint8_t>(n), dev.alloc<std::uint8_t>(n), dev.alloc<float>(length * kv_heads), dev.alloc<float>(length * kv_heads), kv_heads, hd};
        auto* dk = dev.alloc<__nv_bfloat16>(n);
        auto* dv = dev.alloc<__nv_bfloat16>(n);
        cudaMemcpy(dk, k.data(), n * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v.data(), n * 2, cudaMemcpyHostToDevice);
        kv::write(cache, dk, dv, kv_heads * hd, length, 0, nullptr);  // positions 0 .. length - 1 in one call
        std::vector<std::uint8_t> kc(n), vc(n);
        std::vector<float> ks(length * kv_heads), vs(length * kv_heads);
        cudaMemcpy(kc.data(), cache.k, n, cudaMemcpyDeviceToHost);
        cudaMemcpy(vc.data(), cache.v, n, cudaMemcpyDeviceToHost);
        cudaMemcpy(ks.data(), cache.k_scale, ks.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(vs.data(), cache.v_scale, vs.size() * 4, cudaMemcpyDeviceToHost);
        kd.resize(n); vd.resize(n);
        for(std::size_t i = 0; i < n; ++i) {
            const std::size_t row = i / hd;
            kd[i] = bf16_round(e4m3(kc[i]) * ks[row]);
            vd[i] = bf16_round(e4m3(vc[i]) * vs[row]);
            // E4M3 keeps 3 mantissa bits: relative error at most 1/16 of the value, plus the subnormal step
            const float bound_k = std::fabs(__bfloat162float(k[i])) / 15.f + ks[row] * 0.002f;
            const float bound_v = std::fabs(__bfloat162float(v[i])) / 15.f + vs[row] * 0.002f;
            worst_write = std::fmax(worst_write, std::fmax(std::fabs(kd[i] - __bfloat162float(k[i])) / bound_k,
                                                           std::fabs(vd[i] - __bfloat162float(v[i])) / bound_v));
        }
    }

    // FP64 attention of query row (heads x hd) at position pos over the dequantized cache
    std::vector<double> attend(const std::vector<__nv_bfloat16>& q, std::size_t row, int pos, float scale) const {
        std::vector<double> out(static_cast<std::size_t>(heads) * hd, 0.0);
        for(int h = 0; h < heads; ++h) {
            const int g = h / (heads / kv_heads);
            std::vector<double> s(pos + 1);
            double mx = -1e300;
            for(int j = 0; j <= pos; ++j) {
                double acc = 0.0;
                for(int d = 0; d < hd; ++d)
                    acc += static_cast<double>(__bfloat162float(q[row * heads * hd + h * hd + d])) * kd[(static_cast<std::size_t>(j) * kv_heads + g) * hd + d];
                s[j] = acc * scale;
                mx = std::fmax(mx, s[j]);
            }
            double sum = 0.0;
            for(auto& x : s) sum += x = std::exp(x - mx);
            for(int d = 0; d < hd; ++d) {
                double acc = 0.0;
                for(int j = 0; j <= pos; ++j) acc += s[j] * vd[(static_cast<std::size_t>(j) * kv_heads + g) * hd + d];
                out[static_cast<std::size_t>(h) * hd + d] = acc / sum;
            }
        }
        return out;
    }
};

double rel_error(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0.0, top = 0.0;
    for(std::size_t i = 0; i < want.size(); ++i) { top = std::fmax(top, std::fabs(want[i])); worst = std::fmax(worst, std::fabs(got[i] - want[i])); }
    return worst / top;
}

// fp8: prefill_fp8 (E4M3 queries and probabilities), checked against the same FP64 reference with a wider bound.
bool prefill_case(int hd, int heads, int kv_heads, int T, int pos0, std::mt19937& rng, bool fp8 = false) {
    Setup s(hd, heads, kv_heads, pos0 + T, rng);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<__nv_bfloat16> q(static_cast<std::size_t>(T) * heads * hd);
    for(auto& x : q) x = __float2bfloat16(normal(rng));
    auto* dq = s.dev.alloc<__nv_bfloat16>(q.size());
    auto* dout = s.dev.alloc<float>(q.size());
    cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    const kv::Prefill p{dq, heads * hd, s.cache, dout, heads * hd, T, pos0, heads, scale};
    const cudaError_t e = fp8 ? kv::prefill_fp8(p, nullptr) : kv::prefill(p, nullptr);
    if(fp8 && !fp8_mma_loaded()) {
        std::printf("prefill_fp8 hd %d: no E4M3 mma in the loaded code, %s\n", hd, e == cudaErrorNotSupported ? "not supported" : "NOT REFUSED");
        return e == cudaErrorNotSupported;
    }
    std::vector<float> got(q.size());
    const bool ran = e == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
                     cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    if(!ran) { std::printf("prefill launch failed: %s\n", cudaGetErrorString(e)); return false; }
    std::vector<double> want;
    for(int t = 0; t < T; ++t) {
        const auto row = s.attend(q, t, pos0 + t, scale);
        want.insert(want.end(), row.begin(), row.end());
    }
    const double rel = rel_error(got, want);
    std::printf("prefill%s hd %d heads %d/%d T %d pos0 %d: %.2e of max |out|; write error %.2f of its bound\n", fp8 ? "_fp8" : "", hd, heads,
                kv_heads, T, pos0, rel, s.worst_write);
    // prefill_fp8 rounds queries and probabilities to E4M3 (3 mantissa bits): a few % of max |out| on random data,
    // where a layout error would be of order 1.
    return rel < (fp8 ? 1e-1 : 1e-2) && s.worst_write <= 1.f;
}

// The stream count decode picks itself (streams = 0): two where their shared memory fits the device's per-block limit.
int auto_streams(int hd) {
    int dev = 0, limit = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return static_cast<std::size_t>(limit) >= kv::decode_shared_bytes(hd, 2) ? 2 : 1;
}

// For one and two key streams: T columns at once against FP64, and each column alone bit for bit equal to its column in
// the multi-column call; the automatic stream count gives the bits of the count it picks.
bool decode_case(int hd, int heads, int kv_heads, int T, int pos0, int blocks, std::mt19937& rng) {
    Setup s(hd, heads, kv_heads, pos0 + T, rng);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<__nv_bfloat16> q(static_cast<std::size_t>(T) * heads * hd);
    for(auto& x : q) x = __float2bfloat16(normal(rng));
    auto* dq = s.dev.alloc<__nv_bfloat16>(q.size());
    auto* dout = s.dev.alloc<float>(q.size());
    auto* dpos = s.dev.alloc<int>(1);
    auto* partial = s.dev.alloc<float>(kv::decode_partial_floats(kv_heads, hd, blocks));
    cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dpos, &pos0, sizeof(int), cudaMemcpyHostToDevice);
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    std::vector<double> want;
    for(int t = 0; t < T; ++t) {
        const auto row = s.attend(q, t, pos0 + t, scale);
        want.insert(want.end(), row.begin(), row.end());
    }
    bool ok = true;
    std::vector<float> by_streams[3];
    for(const int streams : {1, 2, 0}) {
        // all T columns at once, position from device memory
        kv::Decode p{dq, heads * hd, s.cache, dout, heads * hd, T, 0, heads, dpos, scale, blocks, partial};
        p.streams = streams;
        cudaError_t e = kv::decode(p, nullptr);
        std::vector<float>& got = by_streams[streams];
        got.resize(q.size());
        if(e != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) { std::printf("decode launch failed: %s\n", cudaGetErrorString(e)); return false; }
        cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
        if(streams == 0) continue;
        const double rel = rel_error(got, want);
        // each column alone, at its own position: bit for bit the same as inside the multi-column call
        bool invariant = true;
        for(int t = 0; t < T; ++t) {
            kv::Decode one{dq + static_cast<std::size_t>(t) * heads * hd, heads * hd, s.cache, dout, heads * hd, 1, pos0 + t, heads, nullptr, scale, blocks, partial};
            one.streams = streams;
            e = kv::decode(one, nullptr);
            std::vector<float> alone(static_cast<std::size_t>(heads) * hd);
            if(e != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
            cudaMemcpy(alone.data(), dout, alone.size() * 4, cudaMemcpyDeviceToHost);
            invariant = invariant && std::memcmp(alone.data(), got.data() + static_cast<std::size_t>(t) * heads * hd, alone.size() * 4) == 0;
        }
        std::printf("decode hd %d heads %d/%d T %d pos0 %d blocks %d streams %d: %.2e of max |out|, columns invariant %s\n", hd, heads, kv_heads, T, pos0,
                    blocks, streams, rel, invariant ? "yes" : "NO");
        ok = rel < 1e-2 && invariant && ok;
    }
    const bool picked = by_streams[0] == by_streams[auto_streams(hd)];
    if(!picked) std::printf("decode hd %d: the automatic stream count differs from the %d streams it should pick\n", hd, auto_streams(hd));
    return ok && picked;
}

// A long context for both stream counts: every verify width against the columns alone, bit for bit, at blocks 1, 7
// and the SM count / kv_heads, and one verify against FP64.
bool long_case(std::mt19937& rng) {
    const int hd = 128, heads = 4, kv_heads = 1, len = 65537, n_max = 16, stride = heads * hd;
    Setup s(hd, heads, kv_heads, len, rng);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<__nv_bfloat16> bank(static_cast<std::size_t>(n_max) * stride);
    for(auto& x : bank) x = __float2bfloat16(normal(rng));
    auto* dq = s.dev.alloc<__nv_bfloat16>(bank.size());
    auto* dout = s.dev.alloc<float>(bank.size());
    cudaMemcpy(dq, bank.data(), bank.size() * 2, cudaMemcpyHostToDevice);
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    std::vector<double> want;  // the last three positions, rows n_max - 3 .. n_max - 1 of the bank
    for(int t = 0; t < 3; ++t) {
        const auto row = s.attend(bank, n_max - 3 + t, len - 3 + t, scale);
        want.insert(want.end(), row.begin(), row.end());
    }
    bool ok = true;
    int compared = 0;
    double worst = 0.0;
    for(const int streams : {1, 2})
        for(const int blocks : {1, 7, sms / kv_heads}) {
            auto* partial = s.dev.alloc<float>(kv::decode_partial_floats(kv_heads, hd, blocks));
            auto run = [&](int T, int pos0, const __nv_bfloat16* rows, std::vector<float>& got) {
                kv::Decode p{rows, stride, s.cache, dout, stride, T, pos0, heads, nullptr, scale, blocks, partial};
                p.streams = streams;
                got.resize(static_cast<std::size_t>(T) * stride);
                return kv::decode(p, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
                       cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
            };
            std::vector<std::vector<float>> alone(n_max);
            for(int j = 0; j < n_max; ++j)
                if(!run(1, len - n_max + j, dq + static_cast<std::size_t>(j) * stride, alone[j])) return false;
            std::vector<float> verify;
            for(int T = 1; T <= n_max; ++T) {
                if(!run(T, len - T, dq + static_cast<std::size_t>(n_max - T) * stride, verify)) return false;
                for(int t = 0; t < T; ++t, ++compared)
                    if(std::memcmp(alone[n_max - T + t].data(), verify.data() + static_cast<std::size_t>(t) * stride, stride * 4) != 0) {
                        std::printf("long: T %d column %d (streams %d, blocks %d) differs from the column alone\n", T, t, streams, blocks);
                        ok = false;
                    }
            }
            if(!run(3, len - 3, dq + static_cast<std::size_t>(n_max - 3) * stride, verify)) return false;
            worst = std::fmax(worst, rel_error(verify, want));
        }
    std::printf("long: %d keys, T 1..%d for 1 and 2 streams at blocks 1 / 7 / SM count: %d verify columns bitwise equal to columns alone: %s; "
                "worst %.2e of max |out| against FP64\n", len, n_max, compared, ok ? "yes" : "NO", worst);
    return ok && worst < 1e-2;
}

__global__ void advance(int* pos, int by) { *pos += by; }

// A decode loop captured once into a CUDA graph: write the step's key and value rows at the device position, decode the
// step's T columns there, advance the position by T. Replays (new rows and queries before each) equal eager calls with
// host positions on a second cache, outputs and cache bytes bit for bit, across a 128-key piece boundary. It runs first,
// so the launchers' first calls for this head dimension happen inside the capture. fp32: the rows are FP32.
bool graph_case(int hd, int T, bool fp32, std::mt19937& rng) {
    const int heads = 8, kv_heads = 2, start = 1000, steps = 16, cap = start + steps * T, blocks = 5;
    const int stride = heads * hd, width = kv_heads * hd;
    Setup a(hd, heads, kv_heads, cap, rng), b(hd, heads, kv_heads, cap, rng);
    a.cache.capacity = b.cache.capacity = cap;
    {  // the same first `start` positions in both caches
        std::normal_distribution<float> normal(0.f, 1.f);
        std::vector<__nv_bfloat16> rows(static_cast<std::size_t>(start) * width);
        for(auto& x : rows) x = __float2bfloat16(normal(rng));
        auto* d = a.dev.alloc<__nv_bfloat16>(rows.size());
        cudaMemcpy(d, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice);
        if(kv::write(a.cache, d, d, width, start, 0, nullptr) != cudaSuccess || kv::write(b.cache, d, d, width, start, 0, nullptr) != cudaSuccess)
            return false;
    }
    auto* kb = a.dev.alloc<__nv_bfloat16>(static_cast<std::size_t>(T) * width);
    auto* vb = a.dev.alloc<__nv_bfloat16>(static_cast<std::size_t>(T) * width);
    auto* kf = a.dev.alloc<float>(static_cast<std::size_t>(T) * width);
    auto* vf = a.dev.alloc<float>(static_cast<std::size_t>(T) * width);
    auto* q = a.dev.alloc<__nv_bfloat16>(static_cast<std::size_t>(T) * stride);
    auto* out_a = a.dev.alloc<float>(static_cast<std::size_t>(T) * stride);
    auto* out_b = a.dev.alloc<float>(static_cast<std::size_t>(T) * stride);
    auto* part_a = a.dev.alloc<float>(kv::decode_partial_floats(kv_heads, hd, blocks));
    auto* part_b = a.dev.alloc<float>(kv::decode_partial_floats(kv_heads, hd, blocks));
    auto* dpos = a.dev.alloc<int>(1);
    cudaMemcpy(dpos, &start, sizeof(int), cudaMemcpyHostToDevice);
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    auto put_rows = [&](const kv::Cache& c, const int* device_pos, int pos0, cudaStream_t st) {
        if(fp32) return device_pos ? kv::write(c, kf, vf, width, T, device_pos, st) : kv::write(c, kf, vf, width, T, pos0, st);
        return device_pos ? kv::write(c, kb, vb, width, T, device_pos, st) : kv::write(c, kb, vb, width, T, pos0, st);
    };
    cudaStream_t st = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool ok = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess && cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal) == cudaSuccess;
    if(ok) {
        const kv::Decode d{q, stride, a.cache, out_a, stride, T, 0, heads, dpos, scale, blocks, part_a};
        bool captured = put_rows(a.cache, dpos, 0, st) == cudaSuccess && kv::decode(d, st) == cudaSuccess;
        advance<<<1, 1, 0, st>>>(dpos, T);
        captured = captured && cudaGetLastError() == cudaSuccess;
        ok = cudaStreamEndCapture(st, &graph) == cudaSuccess && captured && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
    }
    int differ = 0;
    std::normal_distribution<float> normal(0.f, 1.f);
    for(int step = 0; ok && step < steps; ++step) {
        const int p = start + step * T;
        std::vector<float> rk(static_cast<std::size_t>(T) * width), rv(rk.size());
        std::vector<__nv_bfloat16> bk(rk.size()), bv(rk.size()), hq(static_cast<std::size_t>(T) * stride);
        for(std::size_t i = 0; i < rk.size(); ++i) {
            rk[i] = normal(rng); rv[i] = 2.f * normal(rng);
            bk[i] = __float2bfloat16(rk[i]); bv[i] = __float2bfloat16(rv[i]);
        }
        for(auto& x : hq) x = __float2bfloat16(normal(rng));
        cudaMemcpy(kf, rk.data(), rk.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(vf, rv.data(), rv.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(kb, bk.data(), bk.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(vb, bv.data(), bv.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice);
        ok = cudaGraphLaunch(exec, st) == cudaSuccess && cudaStreamSynchronize(st) == cudaSuccess;
        const kv::Decode d{q, stride, b.cache, out_b, stride, T, p, heads, nullptr, scale, blocks, part_b};
        ok = ok && put_rows(b.cache, nullptr, p, nullptr) == cudaSuccess && kv::decode(d, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        std::vector<float> ga(hq.size()), gb(hq.size());
        cudaMemcpy(ga.data(), out_a, ga.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gb.data(), out_b, gb.size() * 4, cudaMemcpyDeviceToHost);
        differ += std::memcmp(ga.data(), gb.data(), ga.size() * 4) != 0;
    }
    int final_pos = 0;
    cudaMemcpy(&final_pos, dpos, sizeof(int), cudaMemcpyDeviceToHost);
    const std::size_t n = static_cast<std::size_t>(cap) * width, ns = static_cast<std::size_t>(cap) * kv_heads;
    auto bytes = [](const void* p, std::size_t count) { std::vector<std::uint8_t> h(count); cudaMemcpy(h.data(), p, count, cudaMemcpyDeviceToHost); return h; };
    const bool caches = bytes(a.cache.k, n) == bytes(b.cache.k, n) && bytes(a.cache.v, n) == bytes(b.cache.v, n) &&
                        bytes(a.cache.k_scale, ns * 4) == bytes(b.cache.k_scale, ns * 4) && bytes(a.cache.v_scale, ns * 4) == bytes(b.cache.v_scale, ns * 4);
    if(exec) cudaGraphExecDestroy(exec);
    if(graph) cudaGraphDestroy(graph);
    if(st) cudaStreamDestroy(st);
    std::printf("graph hd %d T %d %s rows: %d replays (positions %d..%d) captured and run: %s, %d outputs differ from eager calls, device "
                "position %d (want %d), caches equal: %s\n", hd, T, fp32 ? "FP32" : "BF16", steps, start, cap - 1, ok ? "yes" : "NO", differ,
                final_pos, cap, caches ? "yes" : "NO");
    return ok && differ == 0 && final_pos == cap && caches;
}

// write's other forms. FP32 rows match the rule computed on the host from the FP32 values (scale = max |x| / 448 per
// head vector, code = e4m3(x / scale), one rounding) and, when they hold BF16 values, the BF16 write's bytes. A device
// position writes what the host position writes (linear cache and ring) and skips the rows outside the cache: guard rows
// on both sides of the arrays keep their pattern, as does every row the host write leaves alone.
bool write_case(std::mt19937& rng) {
    const int hd = 128, kv_heads = 2, cap = 64, guard = 8, T = 8, width = kv_heads * hd;
    Device dev;
    const int all = cap + 2 * guard;
    const std::size_t n = static_cast<std::size_t>(all) * width, ns = static_cast<std::size_t>(all) * kv_heads;
    const std::size_t gk = static_cast<std::size_t>(guard) * width, gs = static_cast<std::size_t>(guard) * kv_heads;
    auto fresh = [&](int ring) {  // `cap` rows between guard rows, filled with a pattern no write produces as a whole row
        kv::Cache c{dev.alloc<std::uint8_t>(n) + gk, dev.alloc<std::uint8_t>(n) + gk, dev.alloc<float>(ns) + gs, dev.alloc<float>(ns) + gs,
                    kv_heads, hd, ring, cap};
        cudaMemset(c.k - gk, 0x5a, n); cudaMemset(c.v - gk, 0x5a, n); cudaMemset(c.k_scale - gs, 0x5a, ns * 4); cudaMemset(c.v_scale - gs, 0x5a, ns * 4);
        return c;
    };
    struct Bytes { std::vector<std::uint8_t> k, v, ks, vs; bool operator==(const Bytes&) const = default; };
    auto read = [&](const kv::Cache& c) {  // guard rows included
        Bytes b{std::vector<std::uint8_t>(n), std::vector<std::uint8_t>(n), std::vector<std::uint8_t>(ns * 4), std::vector<std::uint8_t>(ns * 4)};
        cudaDeviceSynchronize();
        cudaMemcpy(b.k.data(), c.k - gk, n, cudaMemcpyDeviceToHost); cudaMemcpy(b.v.data(), c.v - gk, n, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.ks.data(), c.k_scale - gs, ns * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.vs.data(), c.v_scale - gs, ns * 4, cudaMemcpyDeviceToHost);
        return b;
    };
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> fk(static_cast<std::size_t>(T) * width), fv(fk.size()), bkf(fk.size()), bvf(fk.size());
    std::vector<__nv_bfloat16> bk(fk.size()), bv(fk.size());
    for(std::size_t i = 0; i < fk.size(); ++i) {
        fk[i] = 3.f * normal(rng); fv[i] = normal(rng);
        bk[i] = __float2bfloat16(fk[i]); bv[i] = __float2bfloat16(fv[i]);
        bkf[i] = __bfloat162float(bk[i]); bvf[i] = __bfloat162float(bv[i]);
    }
    fk[5] = 0.f;  // zeros, and one all-zero head vector (scale 1)
    for(int d = 0; d < hd; ++d) fv[static_cast<std::size_t>(3) * width + hd + d] = 0.f;
    auto* dfk = dev.alloc<float>(fk.size()); auto* dfv = dev.alloc<float>(fk.size());
    auto* dbkf = dev.alloc<float>(fk.size()); auto* dbvf = dev.alloc<float>(fk.size());
    auto* dbk = dev.alloc<__nv_bfloat16>(fk.size()); auto* dbv = dev.alloc<__nv_bfloat16>(fk.size());
    cudaMemcpy(dfk, fk.data(), fk.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dfv, fv.data(), fv.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbkf, bkf.data(), bkf.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dbvf, bvf.data(), bvf.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbk, bk.data(), bk.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dbv, bv.data(), bv.size() * 2, cudaMemcpyHostToDevice);
    auto* dpos = dev.alloc<int>(1);
    auto at = [&](int p) { cudaMemcpy(dpos, &p, sizeof(int), cudaMemcpyHostToDevice); return dpos; };
    const int pos0 = 5;
    bool ok = true;
    // FP32 rows against the host rule.
    {
        const kv::Cache c = fresh(0);
        ok = kv::write(c, dfk, dfv, width, T, pos0, nullptr) == cudaSuccess && ok;
        Bytes want = read(fresh(0));
        for(int t = 0; t < T; ++t)
            for(int g = 0; g < kv_heads; ++g)
                for(int kind = 0; kind < 2; ++kind) {
                    const float* x = (kind ? fv : fk).data() + static_cast<std::size_t>(t) * width + static_cast<std::size_t>(g) * hd;
                    float m = 0.f;
                    for(int d = 0; d < hd; ++d) m = std::fmax(m, std::fabs(x[d]));
                    const float sc = m > 0.f ? m / 448.f : 1.f;
                    const std::size_t row = static_cast<std::size_t>(guard + pos0 + t) * kv_heads + g;
                    for(int d = 0; d < hd; ++d)
                        (kind ? want.v : want.k)[row * hd + d] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(x[d] / sc, __NV_SATFINITE, __NV_E4M3));
                    std::memcpy((kind ? want.vs : want.ks).data() + row * 4, &sc, 4);
                }
        const bool rule = read(c) == want;
        std::printf("write FP32 rows: codes and scales equal the host rule: %s\n", rule ? "yes" : "NO");
        ok = rule && ok;
    }
    // FP32 rows holding BF16 values: the BF16 write's bytes.
    {
        const kv::Cache a = fresh(0), b = fresh(0);
        ok = kv::write(a, dbk, dbv, width, T, pos0, nullptr) == cudaSuccess && kv::write(b, dbkf, dbvf, width, T, pos0, nullptr) == cudaSuccess && ok;
        const bool same = read(a) == read(b);
        std::printf("write FP32 rows of BF16 values: equal to the BF16 write: %s\n", same ? "yes" : "NO");
        ok = same && ok;
    }
    // Device positions: the host position's bytes, rows outside the cache skipped.
    struct Case { int pos0, ring, first, rows; };  // device pos0 -> host write of rows first .. first + rows - 1 at pos0 + first
    int checked = 0;
    for(const Case cs : {Case{pos0, 0, 0, T}, Case{cap - 3, 0, 0, 3}, Case{-2, 0, 2, T - 2}, Case{30, 16, 0, T}, Case{-5, 16, 5, T - 5}}) {
        for(const bool f32 : {false, true}) {
            const kv::Cache a = fresh(cs.ring), b = fresh(cs.ring);
            const std::size_t off = static_cast<std::size_t>(cs.first) * width;
            const bool ran = (f32 ? kv::write(a, dfk, dfv, width, T, at(cs.pos0), nullptr) : kv::write(a, dbk, dbv, width, T, at(cs.pos0), nullptr)) == cudaSuccess &&
                             (f32 ? kv::write(b, dfk + off, dfv + off, width, cs.rows, cs.pos0 + cs.first, nullptr)
                                  : kv::write(b, dbk + off, dbv + off, width, cs.rows, cs.pos0 + cs.first, nullptr)) == cudaSuccess;
            const bool same = ran && read(a) == read(b);
            if(!same) std::printf("write at device position %d (ring %d, %s): differs from the host write\n", cs.pos0, cs.ring, f32 ? "FP32" : "BF16");
            ok = same && ok;
            ++checked;
        }
    }
    const kv::Cache c = fresh(0);
    ok = kv::write(c, dbk, dbv, width, T, static_cast<const int*>(nullptr), nullptr) == cudaErrorInvalidValue && ok;
    ok = kv::write(c, dfk, dfv, width, T, cap - T + 1, nullptr) == cudaErrorInvalidValue && ok;
    std::printf("write at device positions: %d cases equal to host writes of the rows inside the cache: %s\n", checked, ok ? "yes" : "NO");
    return ok;
}

// Cache::capacity. A decode whose device position is negative or puts a column at or past the capacity answers NaN in
// every output and reads nothing (the rows past the capacity hold finite keys, which a read would turn into finite
// outputs); a position that fits gives the same bits as the cache without a capacity; host positions past the capacity
// are rejected by write, prefill and decode, and a ring longer than the capacity is rejected.
bool capacity_case(std::mt19937& rng) {
    const int hd = 128, heads = 8, kv_heads = 2, cap = 1000, T = 3, blocks = 5, stride = heads * hd;
    Setup s(hd, heads, kv_heads, cap + 300, rng);  // rows cap .. cap + 299 hold finite keys and values too
    kv::Cache capped = s.cache;
    capped.capacity = cap;
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<__nv_bfloat16> q(static_cast<std::size_t>(T) * stride);
    for(auto& x : q) x = __float2bfloat16(normal(rng));
    auto* dq = s.dev.alloc<__nv_bfloat16>(q.size());
    auto* dout = s.dev.alloc<float>(q.size());
    auto* dpos = s.dev.alloc<int>(1);
    auto* partial = s.dev.alloc<float>(kv::decode_partial_floats(kv_heads, hd, blocks));
    cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    auto run = [&](const kv::Cache& c, int pos0, std::vector<float>& got) {
        cudaMemcpy(dpos, &pos0, sizeof(int), cudaMemcpyHostToDevice);
        const kv::Decode d{dq, stride, c, dout, stride, T, 0, heads, dpos, scale, blocks, partial};
        got.assign(q.size(), 0.f);
        return kv::decode(d, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
               cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    };
    std::vector<float> plain, fits;
    bool ok = run(s.cache, cap - T, plain) && run(capped, cap - T, fits) && std::memcmp(plain.data(), fits.data(), plain.size() * 4) == 0;
    for(const float x : fits) ok = ok && std::isfinite(x);
    const bool fitting = ok;
    for(const int pos0 : {cap - T + 1, cap + 100, -1, -T, INT_MAX - 1}) {
        std::vector<float> got;
        ok = run(capped, pos0, got) && ok;
        for(const float x : got) ok = ok && std::isnan(x);
    }
    const bool nan_rows = ok;
    // Host positions.
    kv::Decode host{dq, stride, capped, dout, stride, T, cap - T + 1, heads, nullptr, scale, blocks, partial};
    kv::Prefill prompt{dq, stride, capped, dout, stride, T, cap - T + 1, heads, scale};
    ok = kv::decode(host, nullptr) == cudaErrorInvalidValue && ok;
    ok = kv::prefill(prompt, nullptr) == cudaErrorInvalidValue && ok;
    ok = kv::prefill_fp8(prompt, nullptr) == cudaErrorInvalidValue && ok;
    host.pos0 = prompt.pos0 = cap - T;
    ok = kv::decode(host, nullptr) == cudaSuccess && kv::prefill(prompt, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess && ok;
    auto* rows = s.dev.alloc<__nv_bfloat16>(static_cast<std::size_t>(2) * kv_heads * hd);
    ok = kv::write(capped, rows, rows, kv_heads * hd, 2, cap - 1, nullptr) == cudaErrorInvalidValue && ok;
    ok = kv::write(capped, rows, rows, kv_heads * hd, 1, cap, nullptr) == cudaErrorInvalidValue && ok;
    kv::Cache ring = capped;
    ring.ring = cap + 1;
    host.cache = ring;
    host.window = 100;
    ok = kv::decode(host, nullptr) == cudaErrorInvalidValue && ok;
    std::printf("capacity: a fitting position equals the uncapped cache: %s; positions outside answer NaN: %s; host positions past it "
                "rejected: %s\n", fitting ? "yes" : "NO", nan_rows ? "yes" : "NO", ok ? "yes" : "NO");
    return ok;
}

// Every verify width at long contexts: for each T in 1 .. 64 / group, a T-column decode over keys 0 .. len - 1 equals
// each column decoded alone at its own position, bit for bit, at 2048, 7168 and 16384 keys (and one length off a piece
// boundary), with and without a sliding window, for blocks 1, 5 and the SM count over key/value heads.
bool widths_case(int hd, int heads, int kv_heads, std::mt19937& rng) {
    constexpr int max_len = 16384;
    Setup s(hd, heads, kv_heads, max_len, rng);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int stride = heads * hd, n_max = 64 / (heads / kv_heads);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<__nv_bfloat16> bank(static_cast<std::size_t>(n_max) * stride);
    auto* dq = s.dev.alloc<__nv_bfloat16>(bank.size());
    auto* dout = s.dev.alloc<float>(bank.size());
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    int compared = 0;
    for(const int streams : {1, 2})
    for(const int blocks : {1, 5, sms / kv_heads > 1 ? sms / kv_heads : 1}) {
        auto* partial = s.dev.alloc<float>(kv::decode_partial_floats(kv_heads, hd, blocks));
        auto run = [&](int T, int pos0, int window, const __nv_bfloat16* rows, std::vector<float>& got) {
            kv::Decode p{rows, stride, s.cache, dout, stride, T, pos0, heads, nullptr, scale, blocks, partial, window};
            p.streams = streams;
            got.resize(static_cast<std::size_t>(T) * stride);
            return kv::decode(p, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
                   cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
        };
        for(const int len : {2048, 7168, 16384, 3001})
            for(const int window : {0, 1000}) {
                for(auto& x : bank) x = __float2bfloat16(normal(rng));
                if(cudaMemcpy(dq, bank.data(), bank.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) return false;
                // Row j of the bank is the query at position len - n_max + j.
                std::vector<std::vector<float>> alone(n_max);
                for(int j = 0; j < n_max; ++j)
                    if(!run(1, len - n_max + j, window, dq + static_cast<std::size_t>(j) * stride, alone[j])) return false;
                std::vector<float> verify;
                for(int T = 1; T <= n_max; ++T) {
                    if(!run(T, len - T, window, dq + static_cast<std::size_t>(n_max - T) * stride, verify)) return false;
                    for(int t = 0; t < T; ++t, ++compared)
                        if(std::memcmp(alone[n_max - T + t].data(), verify.data() + static_cast<std::size_t>(t) * stride, stride * 4) != 0) {
                            std::printf("widths hd %d heads %d/%d: T %d column %d at %d keys (window %d, blocks %d, streams %d) differs from the "
                                        "column alone\n", hd, heads, kv_heads, T, t, len, window, blocks, streams);
                            return false;
                        }
                }
            }
    }
    std::printf("widths hd %d heads %d/%d: T 1..%d, 1 and 2 streams, %d verify columns bitwise equal to columns alone\n", hd, heads, kv_heads, n_max,
                compared);
    return true;
}

}  // namespace

int main() {
    int count = 0;
    if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::printf("no CUDA device; skipped\n"); return 0; }
    std::mt19937 rng(7);
    bool ok = true;
    // First: the decode launchers' first calls happen inside these captures.
    ok = graph_case(64, 1, false, rng) && ok;
    ok = graph_case(128, 3, true, rng) && ok;
    ok = graph_case(256, 3, false, rng) && ok;
    for(const int hd : {64, 128, 256}) {
        ok = prefill_case(hd, 8, 2, 1, 0, rng) && ok;
        ok = prefill_case(hd, 8, 2, 150, 0, rng) && ok;
        ok = prefill_case(hd, 6, 1, 70, 300, rng) && ok;   // a later chunk over earlier positions
        ok = decode_case(hd, 8, 2, 4, 0, 3, rng) && ok;
        ok = decode_case(hd, 6, 1, 8, 900, 3, rng) && ok;  // several 128-key pieces per block
    }
    for(const int hd : {64, 128, 256}) {
        ok = prefill_case(hd, 8, 2, 1, 0, rng, true) && ok;
        ok = prefill_case(hd, 8, 2, 150, 0, rng, true) && ok;     // several 64-key tiles, diagonal masking
        ok = prefill_case(hd, 6, 1, 70, 300, rng, true) && ok;    // a later chunk over earlier positions
    }
    ok = prefill_case(128, 4, 4, 33, 7, rng) && ok;       // multi-head (no grouping)
    ok = prefill_case(128, 4, 4, 33, 7, rng, true) && ok;
    ok = decode_case(128, 4, 4, 1, 4000, 7, rng) && ok;
    ok = capacity_case(rng) && ok;
    ok = write_case(rng) && ok;
    ok = widths_case(128, 32, 8, rng) && ok;   // 4 query heads per key/value head: T up to 16
    ok = widths_case(64, 8, 8, rng) && ok;     // no grouping: T up to 64
    ok = widths_case(256, 16, 2, rng) && ok;   // 8 per key/value head: T up to 8
    ok = long_case(rng) && ok;
    const kv::Cache none{};
    ok = kv::prefill(kv::Prefill{}, nullptr) == cudaErrorInvalidValue && ok;
    ok = kv::prefill_fp8(kv::Prefill{}, nullptr) == cudaErrorInvalidValue && ok;
    ok = kv::decode(kv::Decode{}, nullptr) == cudaErrorInvalidValue && ok;
    {
        Setup s(64, 2, 1, 8, rng);
        auto* dq = s.dev.alloc<__nv_bfloat16>(128);
        auto* dout = s.dev.alloc<float>(128);
        auto* partial = s.dev.alloc<float>(kv::decode_partial_floats(1, 64, 1));
        kv::Decode bad{dq, 128, s.cache, dout, 128, 1, 3, 2, nullptr, 0.125f, 1, partial};
        for(const int streams : {-1, 3}) {
            bad.streams = streams;
            ok = kv::decode(bad, nullptr) == cudaErrorInvalidValue && ok;  // only 0, 1 and 2 key streams
        }
    }
    const __nv_bfloat16* no_rows = nullptr;  // write's overloads take BF16 or FP32 rows
    ok = kv::write(none, no_rows, no_rows, 0, 1, 0, nullptr) == cudaErrorInvalidValue && ok;
    std::printf(ok ? "fp8 kv attention passed\n" : "fp8 kv attention FAILED\n");
    return ok ? 0 : 1;
}
