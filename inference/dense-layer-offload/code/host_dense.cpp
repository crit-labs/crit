// Host dense layers; see host_dense.h.
#include "host_dense.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace crit::dense_offload {
namespace detail {
using RowsFn = void (*)(const std::uint16_t*, std::size_t, const float*, std::size_t, int, std::size_t, std::size_t, float*, std::size_t);
void rows_avx2(const std::uint16_t*, std::size_t, const float*, std::size_t, int, std::size_t, std::size_t, float*, std::size_t);
void rows_avx512(const std::uint16_t*, std::size_t, const float*, std::size_t, int, std::size_t, std::size_t, float*, std::size_t);
}  // namespace detail

namespace {
void rows_scalar(const std::uint16_t* w, std::size_t cols, const float* x, std::size_t ldx, int batch, std::size_t row0, std::size_t row1,
                 float* y, std::size_t ldy) {
    for(std::size_t r = row0; r < row1; ++r) {
        float acc[kMaxBatch] = {};
        const std::uint16_t* row = w + r * cols;
        for(std::size_t c = 0; c < cols; ++c) {
            const float v = bf16_to_float(row[c]);
            for(int b = 0; b < batch; ++b) acc[b] = std::fma(v, x[static_cast<std::size_t>(b) * ldx + c], acc[b]);
        }
        for(int b = 0; b < batch; ++b) y[static_cast<std::size_t>(b) * ldy + r] = acc[b];
    }
}

detail::RowsFn rows_for(Isa isa) {
    switch(isa) {
    case Isa::scalar: return rows_scalar;
#if defined(CRIT_HOST_ISA_FILES)
    case Isa::avx2: return detail::rows_avx2;
    case Isa::avx512: return detail::rows_avx512;
#else
    default: break;
#endif
    }
    return nullptr;
}
}  // namespace

float round_bf16(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    if((bits & 0x7f800000u) != 0x7f800000u) bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

void gemv_rows(Isa isa, const std::uint16_t* w, std::size_t cols, const float* x, std::size_t ldx, int batch, std::size_t row0,
               std::size_t row1, float* y, std::size_t ldy) {
    if(!w || !x || !y || cols % 16 || batch < 1 || batch > kMaxBatch || row1 < row0 || ldx < cols)
        throw std::invalid_argument("host GEMV needs BF16 rows of a multiple of 16 and 1 to 8 activation rows");
    if(!supported(isa)) throw std::invalid_argument("host GEMV ISA is not supported by this CPU or build");
    rows_for(isa)(w, cols, x, ldx, batch, row0, row1, y, ldy);
}

ThreadPool::ThreadPool(unsigned threads) {
    if(!threads) threads = std::max(1u, std::thread::hardware_concurrency());
    for(unsigned i = 1; i < threads; ++i) workers_.emplace_back([this, i] { loop(i); });
}
ThreadPool::~ThreadPool() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for(auto& t : workers_) t.join();
}
void ThreadPool::loop(unsigned index) {
    std::uint64_t seen = 0;
    for(;;) {
        const std::function<void(std::size_t, std::size_t)>* job;
        std::size_t n;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if(stop_) return;
            seen = generation_;
            job = job_;
            n = n_;
        }
        const std::size_t parts = size(), begin = n * index / parts, end = n * (index + 1) / parts;
        if(begin < end) (*job)(begin, end);
        std::lock_guard lock(mutex_);
        if(--pending_ == 0) done_.notify_one();
    }
}
void ThreadPool::run(std::size_t n, const std::function<void(std::size_t, std::size_t)>& fn) {
    if(workers_.empty()) { if(n) fn(0, n); return; }
    {
        std::lock_guard lock(mutex_);
        job_ = &fn;
        n_ = n;
        pending_ = unsigned(workers_.size());
        ++generation_;
    }
    wake_.notify_all();
    const std::size_t end = n / size();
    if(end) fn(0, end);
    std::unique_lock lock(mutex_);
    done_.wait(lock, [&] { return pending_ == 0; });
}

void gemv(ThreadPool& pool, Isa isa, const std::uint16_t* w, std::size_t rows, std::size_t cols, const float* x, std::size_t ldx, int batch,
          float* y, std::size_t ldy) {
    if(!w || !x || !y || cols % 16 || batch < 1 || batch > kMaxBatch || ldx < cols) throw std::invalid_argument("invalid host GEMV");
    if(!supported(isa)) throw std::invalid_argument("host GEMV ISA is not supported by this CPU or build");
    const auto fn = rows_for(isa);
    pool.run(rows, [&](std::size_t r0, std::size_t r1) { fn(w, cols, x, ldx, batch, r0, r1, y, ldy); });
}

void host_block(ThreadPool& pool, Isa isa, const BlockShape& s, const BlockWeights& w, HostKv kv, const float* inv_freq, int pos0, int batch,
                float* x) {
    if(batch < 1 || batch > kMaxBatch || pos0 < 0 || pos0 + batch > kv.capacity || s.heads % s.kv_heads)
        throw std::invalid_argument("host block geometry or KV capacity");
    const int hd = s.head_dim, half = hd / 2, group = s.heads / s.kv_heads, qkv = (s.heads + 2 * s.kv_heads) * hd;
    const std::size_t H = static_cast<std::size_t>(s.hidden);
    std::vector<float> xn(H * batch), proj(static_cast<std::size_t>(std::max(qkv, 2 * s.ffn)) * batch);
    std::vector<float> att(static_cast<std::size_t>(s.heads) * hd * batch), act(static_cast<std::size_t>(s.ffn) * batch);
    const auto norm = [&] {
        for(int b = 0; b < batch; ++b) {
            float ss = 0.f;
            for(std::size_t d = 0; d < H; ++d) ss = std::fma(x[b * H + d], x[b * H + d], ss);
            const float inv = 1.f / std::sqrt(ss / static_cast<float>(H) + 1e-6f);
            for(std::size_t d = 0; d < H; ++d) xn[b * H + d] = round_bf16(x[b * H + d] * inv);
        }
    };
    norm();
    gemv(pool, isa, w.qkv, static_cast<std::size_t>(qkv), H, xn.data(), H, batch, proj.data(), static_cast<std::size_t>(qkv));
    // Rotary embedding on Q and K, then the new keys and values into the cache, as BF16 values.
    std::vector<float> q(static_cast<std::size_t>(s.heads) * hd * batch);
    for(int b = 0; b < batch; ++b) {
        const float pos = static_cast<float>(pos0 + b);
        const float* row = proj.data() + static_cast<std::size_t>(b) * qkv;
        for(int h = 0; h < s.heads + s.kv_heads; ++h)
            for(int i = 0; i < half; ++i) {
                const float angle = pos * inv_freq[i], cs = std::cos(angle), sn = std::sin(angle);
                const float x0 = row[h * hd + i], x1 = row[h * hd + i + half];
                const float a = round_bf16(x0 * cs - x1 * sn), c = round_bf16(x0 * sn + x1 * cs);
                if(h < s.heads) { q[(static_cast<std::size_t>(b) * s.heads + h) * hd + i] = a; q[(static_cast<std::size_t>(b) * s.heads + h) * hd + i + half] = c; }
                else {
                    float* k = kv.k + (static_cast<std::size_t>(pos0 + b) * s.kv_heads + (h - s.heads)) * hd;
                    k[i] = a; k[i + half] = c;
                }
            }
        float* v = kv.v + static_cast<std::size_t>(pos0 + b) * s.kv_heads * hd;
        for(int e = 0; e < s.kv_heads * hd; ++e) v[e] = round_bf16(row[(s.heads + s.kv_heads) * hd + e]);
    }
    // Causal grouped-query attention for each (row, head), heads split over the pool.
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    pool.run(static_cast<std::size_t>(batch) * s.heads, [&](std::size_t begin, std::size_t end) {
        std::vector<float> scores(static_cast<std::size_t>(pos0 + batch));
        for(std::size_t job = begin; job < end; ++job) {
            const int b = int(job / s.heads), h = int(job % s.heads), g = h / group, keys = pos0 + b + 1;
            const float* qh = q.data() + (static_cast<std::size_t>(b) * s.heads + h) * hd;
            float top = -INFINITY;
            for(int j = 0; j < keys; ++j) {
                const float* k = kv.k + (static_cast<std::size_t>(j) * s.kv_heads + g) * hd;
                float dot = 0.f;
                for(int d = 0; d < hd; ++d) dot = std::fma(qh[d], k[d], dot);
                scores[j] = dot * scale;
                top = std::max(top, scores[j]);
            }
            float sum = 0.f;
            for(int j = 0; j < keys; ++j) { scores[j] = std::exp(scores[j] - top); sum += scores[j]; }
            float* out = att.data() + (static_cast<std::size_t>(b) * s.heads + h) * hd;
            for(int d = 0; d < hd; ++d) out[d] = 0.f;
            for(int j = 0; j < keys; ++j) {
                const float p = scores[j] / sum;
                const float* v = kv.v + (static_cast<std::size_t>(j) * s.kv_heads + g) * hd;
                for(int d = 0; d < hd; ++d) out[d] = std::fma(p, v[d], out[d]);
            }
            for(int d = 0; d < hd; ++d) out[d] = round_bf16(out[d]);
        }
    });
    const std::size_t qw = static_cast<std::size_t>(s.heads) * hd;
    gemv(pool, isa, w.o, H, qw, att.data(), qw, batch, proj.data(), H);
    for(int b = 0; b < batch; ++b) for(std::size_t d = 0; d < H; ++d) x[b * H + d] += proj[b * H + d];
    norm();
    const std::size_t F = static_cast<std::size_t>(s.ffn);
    gemv(pool, isa, w.gate_up, 2 * F, H, xn.data(), H, batch, proj.data(), 2 * F);
    for(int b = 0; b < batch; ++b)
        for(std::size_t j = 0; j < F; ++j) {
            const float g = proj[b * 2 * F + j], u = proj[b * 2 * F + F + j];
            act[b * F + j] = round_bf16(g / (1.f + std::exp(-g)) * u);
        }
    gemv(pool, isa, w.down, H, F, act.data(), F, batch, proj.data(), H);
    for(int b = 0; b < batch; ++b) for(std::size_t d = 0; d < H; ++d) x[b * H + d] += proj[b * H + d];
}

}  // namespace crit::dense_offload
