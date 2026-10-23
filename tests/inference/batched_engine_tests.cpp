// The batched synthetic engine: its kernels against host references (the GEMM also bitwise independent of the row
// count), batched requests equal to the same requests run one at a time token for token (greedy and seeded sampling,
// BF16 and E4M3 pages, different prompt chunking), and prefix reuse reported and exact.
#include "engine.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace be = crit::batched_engine;
namespace cc = crit::core;
namespace pa = crit::paged_attention;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
using pa::check;
float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }

template <class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(std::size_t n) { check(cudaMalloc(&p, sizeof(T) * std::max<std::size_t>(n, 1)), "malloc"); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& h) { check(cudaMemcpy(p, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> get(std::size_t n) const { std::vector<T> h(n); check(cudaMemcpy(h.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost), "download"); return h; }
};

std::vector<__nv_bfloat16> random_bf16(std::size_t n, std::mt19937& rng, float scale = 1.f) {
    std::normal_distribution<float> normal(0.f, scale);
    std::vector<__nv_bfloat16> v(n);
    for(auto& x : v) x = __float2bfloat16(normal(rng));
    return v;
}

void case_kernels() {
    std::mt19937 rng(7);
    int sms = 0;
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), "attribute");
    // GEMM against an FP64 host product, with and without split K, stored and accumulated.
    for(const auto& [n, k] : {std::pair{128, 1024}, std::pair{4096, 256}, std::pair{192, 64}}) {
        const auto plan = be::plan_gemm(n, k, sms);
        const int rows = 150;
        const auto x = random_bf16(static_cast<std::size_t>(rows) * k, rng), w = random_bf16(static_cast<std::size_t>(n) * k, rng);
        Buffer<__nv_bfloat16> dx(x.size()), dw(w.size());
        dx.put(x); dw.put(w);
        Buffer<float> c(static_cast<std::size_t>(rows) * n), scratch(be::gemm_scratch_floats(plan, rows) + 1);
        std::vector<float> base(static_cast<std::size_t>(rows) * n);
        for(auto& v : base) v = std::normal_distribution<float>(0.f, 1.f)(rng);
        for(const bool accumulate : {false, true}) {
            c.put(base);
            check(be::gemm(plan, dx.p, rows, dw.p, c.p, accumulate, scratch.p, nullptr), "gemm");
            const auto got = c.get(base.size());
            for(int r = 0; r < rows; ++r)
                for(int j = 0; j < n; ++j) {
                    double ref = accumulate ? base[static_cast<std::size_t>(r) * n + j] : 0.0;
                    double mag = std::fabs(ref);
                    for(int e = 0; e < k; ++e) {
                        const double t = static_cast<double>(__bfloat162float(x[static_cast<std::size_t>(r) * k + e])) * __bfloat162float(w[static_cast<std::size_t>(j) * k + e]);
                        ref += t;
                        mag += std::fabs(t);
                    }
                    require(std::fabs(got[static_cast<std::size_t>(r) * n + j] - ref) <= 1e-5 * mag + 1e-6, "GEMM differs from the host product");
                }
            // A single row gives the same bits as row 0 of the 150-row product.
            c.put(base);
            check(be::gemm(plan, dx.p, 1, dw.p, c.p, accumulate, scratch.p, nullptr), "gemm");
            require(std::memcmp(c.get(n).data(), got.data(), sizeof(float) * n) == 0, "a GEMM row depends on the row count");
        }
        std::cout << "gemm n=" << n << " k=" << k << " splits=" << plan.splits << " exact to the host product within 1e-5\n";
    }
    // RMSNorm, rotary split, SiLU product and BF16 conversion against host formulas (one BF16 rounding).
    const be::Shape s{64, 128, 1, 4, 2, 64, 128, 10000.f, 3};
    const int rows = 5;
    std::vector<float> x(static_cast<std::size_t>(rows) * s.hidden);
    for(auto& v : x) v = std::normal_distribution<float>(0.f, 2.f)(rng);
    Buffer<float> dx(x.size());
    dx.put(x);
    Buffer<__nv_bfloat16> dy(x.size());
    const std::vector<std::int32_t> index{3, 0};
    Buffer<std::int32_t> di(2);
    di.put(index);
    check(be::rmsnorm(dx.p, di.p, 2, s.hidden, dy.p, nullptr), "rmsnorm");
    const auto y = dy.get(2 * static_cast<std::size_t>(s.hidden));
    for(int i = 0; i < 2; ++i) {
        double ss = 0;
        for(int d = 0; d < s.hidden; ++d) ss += static_cast<double>(x[index[i] * s.hidden + d]) * x[index[i] * s.hidden + d];
        const double inv = 1.0 / std::sqrt(ss / s.hidden + 1e-6);
        for(int d = 0; d < s.hidden; ++d) {
            const double want = x[index[i] * s.hidden + d] * inv, got = __bfloat162float(y[i * s.hidden + d]);
            require(std::fabs(got - want) <= std::fabs(want) / 128 + 1e-6, "rmsnorm differs from the host formula");
        }
    }
    std::vector<float> qkv(static_cast<std::size_t>(rows) * s.qkv());
    for(auto& v : qkv) v = std::normal_distribution<float>(0.f, 1.f)(rng);
    const std::vector<std::int32_t> positions{0, 1, 7, 300, 4095};
    std::vector<float> inv(s.head_dim / 2);
    for(std::size_t i = 0; i < inv.size(); ++i) inv[i] = static_cast<float>(std::pow(10000.0, -2.0 * static_cast<double>(i) / s.head_dim));
    Buffer<float> dqkv(qkv.size()), dinv(inv.size());
    Buffer<std::int32_t> dpos(positions.size());
    dqkv.put(qkv); dinv.put(inv); dpos.put(positions);
    Buffer<__nv_bfloat16> q(static_cast<std::size_t>(rows) * s.heads * s.head_dim), k(static_cast<std::size_t>(rows) * s.kv_heads * s.head_dim),
        v(static_cast<std::size_t>(rows) * s.kv_heads * s.head_dim);
    check(be::qkv_rope(dqkv.p, dpos.p, dinv.p, rows, s, q.p, k.p, v.p, nullptr), "rope");
    const auto hq = q.get(static_cast<std::size_t>(rows) * s.heads * s.head_dim), hk = k.get(static_cast<std::size_t>(rows) * s.kv_heads * s.head_dim),
               hv = v.get(static_cast<std::size_t>(rows) * s.kv_heads * s.head_dim);
    const int half = s.head_dim / 2;
    for(int r = 0; r < rows; ++r)
        for(int h = 0; h < s.heads + s.kv_heads; ++h)
            for(int i = 0; i < half; ++i) {
                const double a = static_cast<double>(positions[r]) * inv[i];
                const double x0 = qkv[static_cast<std::size_t>(r) * s.qkv() + h * s.head_dim + i], x1 = qkv[static_cast<std::size_t>(r) * s.qkv() + h * s.head_dim + i + half];
                const double w0 = x0 * std::cos(a) - x1 * std::sin(a), w1 = x0 * std::sin(a) + x1 * std::cos(a);
                const auto& out = h < s.heads ? hq : hk;
                const std::size_t base = h < s.heads ? (static_cast<std::size_t>(r) * s.heads + h) * s.head_dim
                                                     : (static_cast<std::size_t>(r) * s.kv_heads + h - s.heads) * s.head_dim;
                require(std::fabs(__bfloat162float(out[base + i]) - w0) <= std::fabs(w0) / 128 + 1e-3 &&
                        std::fabs(__bfloat162float(out[base + i + half]) - w1) <= std::fabs(w1) / 128 + 1e-3, "rotary embedding differs from the host formula");
            }
    for(int r = 0; r < rows; ++r)
        for(int e = 0; e < s.kv_heads * s.head_dim; ++e)
            require(__bfloat162float(hv[static_cast<std::size_t>(r) * s.kv_heads * s.head_dim + e]) ==
                    bf(qkv[static_cast<std::size_t>(r) * s.qkv() + (s.heads + s.kv_heads) * s.head_dim + e]), "values are copied as BF16");
    std::vector<float> gu(static_cast<std::size_t>(rows) * 2 * s.ffn);
    for(auto& t : gu) t = std::normal_distribution<float>(0.f, 3.f)(rng);
    Buffer<float> dgu(gu.size());
    dgu.put(gu);
    Buffer<__nv_bfloat16> act(static_cast<std::size_t>(rows) * s.ffn);
    check(be::silu_mul(dgu.p, rows, s.ffn, act.p, nullptr), "silu");
    const auto ha = act.get(static_cast<std::size_t>(rows) * s.ffn);
    for(int r = 0; r < rows; ++r)
        for(int j = 0; j < s.ffn; ++j) {
            const double g = gu[static_cast<std::size_t>(r) * 2 * s.ffn + j], u = gu[static_cast<std::size_t>(r) * 2 * s.ffn + s.ffn + j];
            const double want = g / (1.0 + std::exp(-g)) * u;
            require(std::fabs(__bfloat162float(ha[static_cast<std::size_t>(r) * s.ffn + j]) - want) <= std::fabs(want) / 128 + 1e-6, "SiLU product differs");
        }
    // Device argmax equals Core's greedy Sampler, ties included (lowest id).
    const int vocab = 32000, logit_rows = 3;
    std::vector<float> logits(static_cast<std::size_t>(logit_rows) * vocab);
    for(auto& t : logits) t = std::round(std::normal_distribution<float>(0.f, 2.f)(rng) * 4.f) / 4.f;
    logits[vocab + 31000] = logits[vocab + 17] = 40.f;  // a tie in row 1
    logits[2 * vocab + vocab - 1] = 41.f;                // the last id in row 2
    Buffer<float> dl(logits.size());
    dl.put(logits);
    Buffer<std::int32_t> best(logit_rows);
    check(be::argmax(dl.p, logit_rows, vocab, best.p, nullptr), "argmax");
    const auto got_best = best.get(logit_rows);
    cc::Sampler sampler(vocab);
    cc::SamplingParams greedy;
    greedy.temperature = 0.0;
    const cc::TokenHistory empty(std::span<const cc::Token>{});
    for(int r = 0; r < logit_rows; ++r)
        require(got_best[r] == sampler.sample(std::span<const float>(logits).subspan(static_cast<std::size_t>(r) * vocab, vocab), greedy, empty).token,
                "device argmax differs from Core's greedy choice");
    require(got_best[1] == 17 && got_best[2] == vocab - 1, "ties break to the lowest id");
    std::cout << "rmsnorm, rotary, SiLU, conversion and argmax match the host references\n";
}

struct Run {
    std::vector<std::vector<cc::Token>> outputs;
    std::vector<std::uint32_t> reused;
    std::uint32_t largest_batch = 0, mixed_steps = 0;
};

std::vector<std::vector<cc::Token>> prompts() {
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> token(0, 511);
    std::vector<cc::Token> shared(53);
    for(auto& t : shared) t = token(rng);
    std::vector<std::vector<cc::Token>> result;
    for(const int length : {5, 90, 17, 1, 64, 33, 120}) {
        std::vector<cc::Token> p(static_cast<std::size_t>(length));
        for(auto& t : p) t = token(rng);
        result.push_back(std::move(p));
    }
    auto a = shared, b = shared;  // two prompts sharing 53 tokens (three full 16-token pages)
    a.push_back(1); a.push_back(2);
    b.push_back(3);
    result.push_back(a);
    result.push_back(b);
    return result;
}

cc::RequestOptions options_for(std::size_t i) {
    cc::RequestOptions o;
    if(i % 3 == 1) { o.sampling.temperature = 0.8; o.sampling.seed = 1000 + i; o.sampling.top_k = 40; }
    return o;
}

Run run(const be::Config& config, bool concurrent, int repeat_first = 0) {
    auto stats = std::make_shared<be::Stats>();
    auto engine = be::make_engine(config, stats, std::size_t{1} << 30, std::size_t{1} << 30);
    const auto all = prompts();
    Run result;
    result.outputs.resize(all.size());
    result.reused.resize(all.size());
    if(concurrent) {
        std::vector<cc::Handle> handles;
        for(std::size_t i = 0; i < all.size(); ++i) handles.push_back(engine->submit(all[i], 24, options_for(i)));
        for(std::size_t i = 0; i < all.size(); ++i) {
            const auto r = handles[i].wait();
            require(r.finish == cc::Finish::length, "a batched request did not finish");
            result.outputs[i] = r.tokens;
            result.reused[i] = r.timing.reused_prompt_tokens;
        }
    } else {
        for(std::size_t i = 0; i < all.size(); ++i) {
            const auto r = engine->submit(all[i], 24, options_for(i)).wait();
            require(r.finish == cc::Finish::length, "a serial request did not finish");
            result.outputs[i] = r.tokens;
            result.reused[i] = r.timing.reused_prompt_tokens;
        }
    }
    for(int i = 0; i < repeat_first; ++i) {
        const auto r = engine->submit(all.back(), 24, options_for(all.size() - 1)).wait();
        result.outputs.push_back(r.tokens);
        result.reused.push_back(r.timing.reused_prompt_tokens);
    }
    result.largest_batch = stats->largest_batch;
    result.mixed_steps = stats->mixed_steps;
    return result;
}

be::Config small(pa::Format format) {
    be::Config c;
    c.shape = {512, 256, 2, 4, 2, 64, 512, 10000.f, 11};
    c.format = format;
    c.page_tokens = 16;
    c.context = 256;
    return c;
}

void case_batched(pa::Format format) {
    auto serial = small(format);
    serial.max_concurrency = 1;
    serial.max_batch_tokens = 64;
    serial.prefill_chunk = 64;
    serial.prefix_cache = false;
    auto batched = small(format);
    batched.max_concurrency = 9;
    batched.max_batch_tokens = 96;
    batched.prefill_chunk = 40;  // other chunk boundaries than the serial run
    const auto one = run(serial, false);
    const auto many = run(batched, true, 1);
    for(std::size_t i = 0; i < one.outputs.size(); ++i) {
        require(one.outputs[i].size() == 24, "24 tokens per request");
        require(many.outputs[i] == one.outputs[i], "batched request " + std::to_string(i) + " differs from its serial run");
    }
    require(many.largest_batch >= 4 && many.mixed_steps > 0, "requests shared steps, prompt chunks with decode rows");
    // A repeat of the last prompt after the batch reuses the three full pages its prompt shares (53 tokens, 16 per page).
    const auto& repeat = many.outputs.back();
    require(repeat == one.outputs.back(), "a request on reused prefix pages equals its run without reuse");
    require(many.reused.back() == 48 && one.reused.back() == 0, "prefix reuse is reported");
    bool distinct = false;
    for(std::size_t i = 1; i < one.outputs.size(); ++i) distinct |= one.outputs[i] != one.outputs[0];
    require(distinct, "requests produce different text");
    std::cout << (format == pa::Format::bf16 ? "bf16" : "e4m3") << " pages: " << one.outputs.size()
              << " batched requests equal their serial runs token for token; largest batch " << many.largest_batch << ", "
              << many.mixed_steps << " mixed steps, reused " << many.reused.back() << " prompt tokens\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if(cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::cout << "no CUDA device; skipped\n"; return 0; }
        const std::string which = argc > 1 ? argv[1] : "all";
        if(which == "kernels" || which == "all") case_kernels();
        if(which == "batched" || which == "all") case_batched(pa::Format::bf16);
        if(which == "e4m3" || which == "all") case_batched(pa::Format::e4m3);
        std::cout << "batched engine " << which << " passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
