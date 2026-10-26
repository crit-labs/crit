// Device token selection against Core's host sampler: greedy ties, top-k 1 and tiny
// top-p reduce to greedy, seeded draws repeat and advance the position, chosen tokens
// equal crit::core::Sampler's away from FP32 rounding of a boundary, frequencies
// follow the tempered distribution, constraint bitmasks mask logits as the host does,
// and device log-probabilities (enqueue_logprobs, and fused into enqueue_sample and
// enqueue_verify) follow Core's log_softmax and top_logprobs.
#include "sampling.h"

#include <crit/sampling.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cc = crit::core;
namespace ts = crit::techniques::token_sampling;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

struct Sampler {
    float* logits = nullptr;
    void* scratch = nullptr;
    std::int32_t* out = nullptr;
    ts::Params* params = nullptr;
    int vocab;
    explicit Sampler(int v) : vocab(v) {
        ok(cudaMalloc(&logits, sizeof(float) * v), "malloc");
        ok(cudaMalloc(&scratch, ts::select_scratch_bytes(v, 1)), "malloc");
        ok(cudaMalloc(&out, sizeof(std::int32_t)), "malloc");
        ok(cudaMalloc(&params, sizeof(ts::Params)), "malloc");
    }
    ~Sampler() { cudaFree(logits); cudaFree(scratch); cudaFree(out); cudaFree(params); }
    void load(const std::vector<float>& row) { ok(cudaMemcpy(logits, row.data(), sizeof(float) * vocab, cudaMemcpyHostToDevice), "upload"); }
    void set(const ts::Params& p) { ok(cudaMemcpy(params, &p, sizeof p, cudaMemcpyHostToDevice), "params"); }
    int draw() {
        ok(ts::enqueue_select(logits, vocab, 1, vocab, params, scratch, out, nullptr), "select");
        int v = -1;
        ok(cudaMemcpy(&v, out, sizeof v, cudaMemcpyDeviceToHost), "download");
        return v;
    }
    std::uint64_t position() {
        ts::Params p;
        ok(cudaMemcpy(&p, params, sizeof p, cudaMemcpyDeviceToHost), "params");
        return p.position;
    }
};

template<class T> struct Buffer {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Buffer(std::size_t count) : n(count) { ok(cudaMalloc(&p, sizeof(T) * std::max<std::size_t>(count, 1)), "malloc"); }
    explicit Buffer(const std::vector<T>& v) : Buffer(v.size()) {
        ok(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice), "upload");
    }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    std::vector<T> get() const {
        std::vector<T> v(n);
        ok(cudaMemcpy(v.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost), "download");
        return v;
    }
};

// Logit rows of different shapes: a moderate or peaked head, flat, a 0.25 grid of ties, a grammar-style mask (2% of
// the tokens left), large magnitudes and all-negative rows.
enum class Shape { moderate, peaked, flat, ties, masked, huge, negative };
const Shape kShapes[] = {Shape::moderate, Shape::peaked, Shape::flat, Shape::ties, Shape::masked, Shape::huge, Shape::negative};
void fill_row(std::mt19937& rng, Shape shape, float* x, int vocab) {
    std::normal_distribution<float> normal(0.f, 1.f);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_int_distribution<int> any(0, vocab - 1);
    for(int t = 0; t < vocab; ++t) {
        const float z = normal(rng);
        switch(shape) {
        case Shape::moderate: case Shape::peaked: x[t] = -2.f + 2.5f * z; break;
        case Shape::flat: x[t] = z; break;
        case Shape::ties: x[t] = std::round(4.f * (-2.f + 2.5f * z)) / 4.f; break;
        case Shape::masked: x[t] = unit(rng) < 0.02f ? -2.f + 3.f * z : -INFINITY; break;
        case Shape::huge: x[t] = 3.0e4f + 2.5f * z; break;
        case Shape::negative: x[t] = -80.f + 3.f * z; break;
        }
    }
    const float head = shape == Shape::huge ? 3.0e4f + 13.f : (shape == Shape::negative ? -66.f : 12.f);
    for(int i = 0; i < 40; ++i) x[any(rng)] = shape == Shape::ties ? std::round(4.f * (11.f + 1.5f * normal(rng))) / 4.f : head + 2.f * normal(rng);
    if(shape == Shape::peaked) x[any(rng)] = 27.f;
}

float ulp(float x) { return std::nextafter(std::abs(x), INFINITY) - std::abs(x); }

// The device's ranked alternatives (ids, lps) against Core's top_logprobs: values within tol, ranked (value desc, id
// asc) and distinct; a different order only where Core's own values are one ulp apart (the FP32 exps move log_sum by
// ~1e-7, which can merge or split such neighbours). Returns whether the ids are Core's exactly.
bool same_ranking(const std::vector<cc::TokenLogprob>& want, const std::int32_t* ids, const float* lps, int n, int vocab,
                  double tol, double& worst) {
    std::vector<int> seen;
    bool same = true;
    for(int i = 0; i < n; ++i) {
        require(ids[i] >= 0 && ids[i] < vocab && !std::isnan(lps[i]), "an alternative outside the vocabulary or NaN");
        if(i) require(lps[i] < lps[i - 1] || (lps[i] == lps[i - 1] && ids[i] > ids[i - 1]), "alternatives not ranked (value desc, id asc)");
        seen.push_back(ids[i]);
        same &= ids[i] == want[std::size_t(i)].token;
        if(std::isfinite(lps[i]) || std::isfinite(want[std::size_t(i)].logprob)) {
            const double d = std::abs(double(lps[i]) - double(want[std::size_t(i)].logprob));
            worst = std::max(worst, d);
            require(d <= tol, "an alternative's log-probability differs from Core's");
        }
    }
    std::sort(seen.begin(), seen.end());
    require(std::adjacent_find(seen.begin(), seen.end()) == seen.end(), "alternatives not distinct");
    if(!same)
        for(int i = 0; i < n; ++i)
            if(ids[i] != want[std::size_t(i)].token)
                require(std::abs(lps[i] - want[std::size_t(i)].logprob) <= ulp(want[std::size_t(i)].logprob),
                        "the ranking differs from Core's away from a one-ulp tie");
    return same;
}

struct LogprobRun {
    std::vector<float> chosen, lps;
    std::vector<std::int32_t> ids;
};
LogprobRun device_logprobs(const std::vector<float>& rows, int count, int ld, int vocab, const std::vector<std::int32_t>& chosen, int n) {
    Buffer<float> x(rows), c(static_cast<std::size_t>(count)), l(static_cast<std::size_t>(count) * std::size_t(std::max(n, 1)));
    Buffer<std::int32_t> ch(chosen), ids(static_cast<std::size_t>(count) * std::size_t(std::max(n, 1)));
    Buffer<unsigned char> scratch(ts::logprob_scratch_bytes(vocab, count, n));
    ok(ts::enqueue_logprobs(x.p, ld, count, vocab, ch.p, n, c.p, n ? ids.p : nullptr, n ? l.p : nullptr, scratch.p, nullptr), "logprobs");
    LogprobRun r;
    r.chosen = c.get();
    r.ids = ids.get();
    r.lps = l.get();
    r.ids.resize(std::size_t(count) * std::size_t(n));
    r.lps.resize(std::size_t(count) * std::size_t(n));
    return r;
}

std::vector<std::int32_t> top_ids(std::span<const float> x, int k) {
    std::vector<std::int32_t> ids(x.size());
    for(std::size_t i = 0; i < x.size(); ++i) ids[i] = std::int32_t(i);
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) { return x[std::size_t(a)] == x[std::size_t(b)] ? a < b : x[std::size_t(a)] > x[std::size_t(b)]; });
    ids.resize(std::size_t(k));
    return ids;
}

void logprob_checks(std::mt19937& rng) {
    constexpr int V = 151936, LD = V + 64;
    // 1. Against log_softmax and top_logprobs: every shape, n = 0, 1, 5, 20, 64, chosen tokens at the top, random, the
    //    last id and rank 50, four rows per launch.
    double worst_chosen = 0, worst_alt = 0;
    int ranked = 0, identical = 0;
    for(const Shape shape : kShapes)
        for(const int n : {0, 1, 5, 20, 64}) {
            constexpr int rows = 4;
            std::vector<float> x(std::size_t(rows) * LD, 0.f);
            std::vector<std::int32_t> chosen(rows);
            for(int r = 0; r < rows; ++r) {
                float* row = x.data() + std::size_t(r) * LD;
                fill_row(rng, shape, row, V);
                const std::span<const float> sp(row, V);
                chosen[std::size_t(r)] = r == 0 ? top_ids(sp, 1)[0] : (r == 1 ? std::int32_t(rng() % V) : (r == 2 ? V - 1 : top_ids(sp, 50)[49]));
            }
            const LogprobRun got = device_logprobs(x, rows, LD, V, chosen, n);
            for(int r = 0; r < rows; ++r) {
                const std::span<const float> sp(x.data() + std::size_t(r) * LD, V);
                std::vector<float> lsm(V);
                cc::log_softmax(sp, lsm);
                const float want = lsm[std::size_t(chosen[std::size_t(r)])], have = got.chosen[std::size_t(r)];
                if(std::isinf(want)) require(have == want, "a -inf chosen log-probability differs");
                else {
                    worst_chosen = std::max(worst_chosen, double(std::abs(have - want)));
                    require(std::abs(have - want) <= 1e-5f, "a chosen log-probability differs from log_softmax");
                }
                if(!n) continue;
                std::vector<cc::TokenLogprob> top(static_cast<std::size_t>(n));
                cc::top_logprobs(sp, top);
                identical += same_ranking(top, got.ids.data() + std::size_t(r) * n, got.lps.data() + std::size_t(r) * n, n, V, 1e-5, worst_alt);
                ++ranked;
            }
        }
    // 2. Collapsing boundaries: raw values one float apart near 2.0 whose log-probabilities near -30 round to one float
    //    in groups, ids ascending with value, so Core's ranking (log-probability desc, id asc) reverses the raw order
    //    inside a group; the rows put the n-th answer inside a group that continues past it, which the raw top n + 1
    //    cannot settle and the tie scan must.
    int collapsed = 0;
    for(const int n : {5, 20, 64})
        for(int extra = 2; extra <= 12; ++extra) {
            std::vector<float> x(LD, 0.f);
            std::normal_distribution<float> normal(0.f, 1.f);
            for(int t = 0; t < V; ++t) x[std::size_t(t)] = std::min(-6.f + 1.5f * normal(rng), 1.f);
            x[100] = 32.f;
            for(int i = 0; i < n - extra; ++i) x[std::size_t(1000 + i)] = 20.f + 0.01f * float(i);
            float v = 2.f;
            for(int i = 0; i < 64; ++i) {
                x[std::size_t(5000 + 7 * i)] = v;
                v = std::nextafter(v, 3.f);
            }
            const std::span<const float> sp(x.data(), V);
            std::vector<cc::TokenLogprob> top(static_cast<std::size_t>(n + 1));
            cc::top_logprobs(sp, top);
            if(top[std::size_t(n - 1)].logprob != top[std::size_t(n)].logprob) continue;
            const auto raw = top_ids(sp, n);
            bool differs = false;
            for(int i = 0; i < n; ++i) differs |= raw[std::size_t(i)] != top[std::size_t(i)].token;
            if(!differs) continue;
            const LogprobRun got = device_logprobs(x, 1, LD, V, {100}, n);
            top.resize(std::size_t(n));
            require(same_ranking(top, got.ids.data(), got.lps.data(), n, V, 1e-5, worst_alt), "a collapsed boundary is not resolved as Core resolves it");
            ++collapsed;
            break;
        }
    require(collapsed == 3, "could not build the collapsed-boundary rows");
    // 3. Rows are independent of the launch: rows 1..8 of one launch equal single-row launches bit for bit.
    {
        constexpr int M = 8, n = 20;
        std::vector<float> x(std::size_t(M) * LD, 0.f);
        std::vector<std::int32_t> chosen(M);
        for(int r = 0; r < M; ++r) {
            fill_row(rng, kShapes[r % 7], x.data() + std::size_t(r) * LD, V);
            chosen[std::size_t(r)] = std::int32_t(rng() % V);
        }
        const LogprobRun all = device_logprobs(x, M, LD, V, chosen, n);
        for(int r = 0; r < M; ++r) {
            const std::vector<float> one(x.begin() + std::ptrdiff_t(r) * LD, x.begin() + std::ptrdiff_t(r + 1) * LD);
            const LogprobRun single = device_logprobs(one, 1, LD, V, {chosen[std::size_t(r)]}, n);
            require(std::memcmp(&single.chosen[0], &all.chosen[std::size_t(r)], sizeof(float)) == 0 &&
                    std::memcmp(single.ids.data(), all.ids.data() + std::size_t(r) * n, sizeof(std::int32_t) * n) == 0 &&
                    std::memcmp(single.lps.data(), all.lps.data() + std::size_t(r) * n, sizeof(float) * n) == 0,
                    "a row's log-probabilities depend on the launch's other rows");
        }
    }
    // 4. Fused into enqueue_sample (fast rows with and without processing, greedy, open and full-law rows) and into
    //    enqueue_verify (sampled drafts with sparse laws): equal to enqueue_logprobs on the same rows and tokens, bit
    //    for bit.
    {
        constexpr int R = 6, n = 20;
        std::vector<float> x(std::size_t(R) * LD, 0.f);
        for(int r = 0; r < R; ++r) fill_row(rng, kShapes[r], x.data() + std::size_t(r) * LD, V);
        std::vector<std::int32_t> counts(V, 0);
        for(int i = 0; i < 40; ++i) counts[rng() % V] = 1 + int(rng() % 3);
        const auto head = top_ids(std::span<const float>(x.data(), V), 10);
        for(int i = 0; i < 10; i += 2) counts[std::size_t(head[std::size_t(i)])] = 2;
        Buffer<std::int32_t> d_counts(counts);
        std::vector<ts::Request> reqs(R);
        for(int r = 0; r < R; ++r) {
            auto& q = reqs[std::size_t(r)];
            q.temperature = r == 3 ? 0.0 : 1.0;
            q.top_p = 0.95;
            q.top_k = r == 4 ? 100 : (r == 5 ? 0 : 20);   // row 4 takes the full law, row 5 is open
            q.seed = 11 + std::uint64_t(r);
            if(r == 2 || r == 5) { q.presence = 0.6; q.frequency = 0.2; q.repetition = 1.1; q.counts = d_counts.p; }
        }
        Buffer<float> d_x(x), scratch(ts::scratch_floats(V, R)), chosen(static_cast<std::size_t>(R)), lps(static_cast<std::size_t>(R) * n);
        Buffer<ts::Request> d_req(reqs);
        Buffer<std::int32_t> out(static_cast<std::size_t>(R)), ids(static_cast<std::size_t>(R) * n);
        Buffer<ts::SparseLaw> laws(static_cast<std::size_t>(R));
        ts::Rows rows;
        rows.logits = d_x.p; rows.stride = LD; rows.vocab = V; rows.requests = R; rows.request = d_req.p; rows.scratch = scratch.p;
        ts::Logprobs lp;
        lp.n = n; lp.chosen = chosen.p; lp.top_ids = ids.p; lp.top_logprobs = lps.p;
        ok(ts::enqueue_sample(rows, cc::DrawPurpose::sample, out.p, nullptr, laws.p, lp, nullptr), "sample + logprobs");
        const auto tokens = out.get();
        const LogprobRun ref = device_logprobs(x, R, LD, V, tokens, n);
        require(std::memcmp(chosen.get().data(), ref.chosen.data(), sizeof(float) * R) == 0, "fused chosen log-probabilities differ");
        require(ids.get() == ref.ids && std::memcmp(lps.get().data(), ref.lps.data(), sizeof(float) * R * n) == 0,
                "fused alternatives differ from enqueue_logprobs");
        // Verify: one request, the same six rows as K + 1 = 6 columns of draws from row 0's law.
        constexpr int K = R - 1;
        std::vector<ts::Request> one(1, reqs[1]);
        Buffer<ts::Request> d_one(one);
        Buffer<std::int32_t> drafts(static_cast<std::size_t>(K)), pred(static_cast<std::size_t>(K + 1));
        Buffer<ts::SparseLaw> dlaws(static_cast<std::size_t>(K));
        ts::Rows d = rows;
        d.requests = 1; d.columns = K; d.request = d_one.p;
        ok(ts::enqueue_sample(d, cc::DrawPurpose::draft, drafts.p, nullptr, dlaws.p, ts::Logprobs{}, nullptr), "drafts");
        ts::Verify v;
        v.target = rows;
        v.target.requests = 1; v.target.columns = K + 1; v.target.request = d_one.p;
        v.drafts = drafts.p;
        v.draft_laws = dlaws.p;
        v.predictions = pred.p;
        v.logprobs = lp;
        ok(ts::enqueue_verify(v, nullptr), "verify + logprobs");
        const LogprobRun vref = device_logprobs(x, R, LD, V, pred.get(), n);
        require(std::memcmp(chosen.get().data(), vref.chosen.data(), sizeof(float) * R) == 0 && ids.get() == vref.ids &&
                std::memcmp(lps.get().data(), vref.lps.data(), sizeof(float) * R * n) == 0,
                "verify's fused log-probabilities differ from enqueue_logprobs");
    }
    {
        Buffer<float> x(static_cast<std::size_t>(V));
        Buffer<unsigned char> scratch(ts::logprob_scratch_bytes(V, 1, 64));
        Buffer<std::int32_t> ids(65);
        Buffer<float> l(65);
        require(ts::enqueue_logprobs(x.p, V, 1, V, nullptr, 65, nullptr, ids.p, l.p, scratch.p, nullptr) == cudaErrorInvalidValue, "n above 64");
        require(ts::enqueue_logprobs(x.p, V, 1, V, nullptr, 5, nullptr, nullptr, l.p, scratch.p, nullptr) == cudaErrorInvalidValue, "null ids");
        require(ts::enqueue_logprobs(x.p, V, 1, V, nullptr, 5, nullptr, ids.p, l.p, nullptr, nullptr) == cudaErrorInvalidValue, "null scratch");
        require(ts::logprob_scratch_bytes(V, 1, 65) == 0, "scratch for n above 64");
    }
    std::cout << "log-probabilities: chosen within " << worst_chosen << " of log_softmax; " << ranked << " ranked rows, " << identical
              << " identical to top_logprobs (the rest at Core's one-ulp ties), values within " << worst_alt << "; "
              << collapsed << " collapsed boundaries resolved; launch-independent; fused into sample and verify bit for bit\n";
}

// Core's rules for the device parameters: top_k 0 is the whole vocabulary on both sides.
cc::SamplingParams core_params(const ts::Params& p) {
    cc::SamplingParams q;
    q.temperature = p.temperature > 0.f ? double(p.temperature) : 0.0;
    q.top_p = p.top_p > 0.f && p.top_p < 1.f ? double(p.top_p) : 1.0;
    q.top_k = static_cast<std::uint32_t>(p.top_k > 0 ? p.top_k : 0);
    q.seed = p.seed;
    q.stream = p.stream;
    return q;
}
} // namespace

int main() {
    try {
        constexpr int V = 50000;
        Sampler s(V);
        std::mt19937 rng(11);
        std::normal_distribution<float> n(0.f, 2.f);
        std::vector<float> row(V);

        // greedy: lowest index among tied maxima, also for top-k 1 and a tiny top-p
        for(auto& v : row) v = std::round(n(rng) * 4.f) / 4.f;
        row[777] = row[31337] = 30.f;
        s.load(row);
        s.set(ts::Params{});
        require(s.draw() == 777, "greedy tie breaks to the lowest index");
        require(s.position() == 1, "greedy advances the position");
        cc::Sampler host(V);
        const cc::TokenHistory empty(std::span<const cc::Token>{});
        require(host.sample(row, core_params(ts::Params{}), empty).token == 777, "Core's greedy choice differs");
        ts::Params p;
        p.temperature = 1.5f;
        p.top_k = 1;
        s.set(p);
        require(s.draw() == 777, "top-k 1 is greedy");
        p.top_k = 0;
        p.top_p = 1e-6f;
        s.set(p);
        require(s.draw() == 777, "tiny top-p is greedy");

        // seeded draws: repeatable, position advances, equal to Core's draw
        int mismatches = 0, draws = 0;
        for(int t = 0; t < 40; ++t) {
            for(auto& v : row) v = n(rng);
            s.load(row);
            ts::Params q;
            q.temperature = 0.5f + 0.05f * t;
            q.top_p = t % 2 ? 0.8f : 1.f;
            q.top_k = t % 3 == 0 ? 0 : 20;
            q.seed = 1000 + t;
            q.stream = static_cast<std::uint32_t>(t % 4);
            s.set(q);
            std::vector<int> first;
            for(int c = 0; c < 6; ++c) first.push_back(s.draw());
            require(s.position() == 6, "position advances per draw");
            s.set(q);
            for(int c = 0; c < 6; ++c) require(s.draw() == first[c], "same seed repeats");
            // Core's law (kept tokens in rank order) and its draw at each position.
            const auto rules = core_params(q);
            std::vector<cc::TokenLogprob> law;
            host.distribution(row, rules, empty, 0, law);
            cc::TokenHistory history(std::span<const cc::Token>{});
            for(int c = 0; c < 6; ++c) {
                const int want = host.sample(row, rules, history).token;
                const cc::Token emitted = first[c];
                history.stage(std::span<const cc::Token>(&emitted, 1));
                history.commit(1);
                const double u = cc::uniform(q.seed, static_cast<std::uint64_t>(c), q.stream);
                double margin = 1, upper = 0;
                bool in_pool = false;
                for(const auto& e : law) {
                    upper += std::exp(double(e.logprob));
                    margin = std::min(margin, std::abs(u - upper));
                    in_pool |= e.token == first[c];
                }
                require(in_pool, "draw outside Core's kept set");
                ++draws;
                if(first[c] != want) {
                    // top_k 20 takes the fast path (Core's law in FP64): only exp's last bit can move a draw.
                    require(margin < (q.top_k ? 1e-12 : 1e-4), "draw differs from Core's sampler away from a boundary");
                    ++mismatches;
                }
            }
        }

        // Strided rows with their own parameters in one launch: greedy, fast (top_k 20 and 1), open (top_k 0 with a
        // closing top_p, and none) and full-law (top_k 100) rows each draw Core's token, positions advance per row.
        {
            constexpr int rows = 6, ld = V + 37;
            std::vector<float> block(std::size_t(rows) * ld, 0.f);
            for(int r = 0; r < rows; ++r)
                for(int t = 0; t < V; ++t) block[std::size_t(r) * ld + std::size_t(t)] = (r == 5 ? 0.05f : 2.5f) * n(rng);
            std::vector<ts::Params> ps(rows);
            const double temps[rows] = {0., 1., 0.9, 0.7, 1.0, 1.0}, tops[rows] = {1., 0.9, 1., 0.6, 0.95, 1.};
            const int ks[rows] = {0, 20, 1, 0, 100, 0};
            for(int r = 0; r < rows; ++r) {
                ps[std::size_t(r)].temperature = temps[r];
                ps[std::size_t(r)].top_p = tops[r];
                ps[std::size_t(r)].top_k = ks[r];
                ps[std::size_t(r)].seed = 300 + std::uint64_t(r);
                ps[std::size_t(r)].position = std::uint64_t(r);
            }
            float* d_block = nullptr;
            void* d_scratch = nullptr;
            ts::Params* d_params = nullptr;
            std::int32_t* d_out = nullptr;
            ok(cudaMalloc(&d_block, sizeof(float) * block.size()), "malloc");
            ok(cudaMalloc(&d_scratch, ts::select_scratch_bytes(V, rows)), "malloc");
            ok(cudaMalloc(&d_params, sizeof(ts::Params) * rows), "malloc");
            ok(cudaMalloc(&d_out, sizeof(std::int32_t) * rows), "malloc");
            ok(cudaMemcpy(d_block, block.data(), sizeof(float) * block.size(), cudaMemcpyHostToDevice), "upload");
            ok(cudaMemcpy(d_params, ps.data(), sizeof(ts::Params) * rows, cudaMemcpyHostToDevice), "upload");
            for(int step = 0; step < 3; ++step) {
                ok(ts::enqueue_select(d_block, ld, rows, V, d_params, d_scratch, d_out, nullptr), "select rows");
                std::vector<std::int32_t> got(rows);
                ok(cudaMemcpy(got.data(), d_out, sizeof(std::int32_t) * rows, cudaMemcpyDeviceToHost), "download");
                for(int r = 0; r < rows; ++r) {
                    const std::span<const float> x(block.data() + std::size_t(r) * ld, std::size_t(V));
                    ts::Params at = ps[std::size_t(r)];
                    at.position += std::uint64_t(step);
                    cc::TokenHistory h(std::span<const cc::Token>{});
                    std::vector<cc::Token> filler(at.position, 0);
                    if(!filler.empty()) { h.stage(filler); h.commit(filler.size()); }
                    const auto rules = core_params(at);
                    const int want = host.sample(x, rules, h).token;
                    if(got[std::size_t(r)] == want) continue;
                    // Only the open rows' whole-row weight (1e-6) and the full law (1e-4) round near a boundary.
                    std::vector<cc::TokenLogprob> law;
                    host.distribution(x, rules, h, 0, law);
                    const double u = cc::uniform(at.seed, at.position, at.stream);
                    double margin = 1, upper = 0;
                    for(const auto& e : law) { upper += std::exp(double(e.logprob)); margin = std::min(margin, std::abs(u - upper)); }
                    require(margin < (ks[r] >= 1 && ks[r] <= ts::kMaxTopK ? 1e-12 : 1e-4), "a strided row's draw differs from Core's");
                }
            }
            std::vector<ts::Params> back(rows);
            ok(cudaMemcpy(back.data(), d_params, sizeof(ts::Params) * rows, cudaMemcpyDeviceToHost), "download");
            for(int r = 0; r < rows; ++r) require(back[std::size_t(r)].position == std::uint64_t(r + 3), "a strided row's position");
            cudaFree(d_block);
            cudaFree(d_scratch);
            cudaFree(d_params);
            cudaFree(d_out);
        }

        // frequencies follow softmax(logits / T) over the candidates
        std::fill(row.begin(), row.end(), -100.f);
        const float lg[4] = {1.0f, 0.5f, 0.0f, -0.5f};
        for(int i = 0; i < 4; ++i) row[100 + 7 * i] = lg[i];
        s.load(row);
        ts::Params f;
        f.temperature = 1.f;
        f.top_k = 4;
        f.seed = 99;
        s.set(f);
        std::vector<int> hist(4, 0);
        constexpr int N = 8000;
        for(int c = 0; c < N; ++c) {
            const int t = s.draw();
            require(t >= 100 && (t - 100) % 7 == 0 && (t - 100) / 7 < 4, "draw outside top-k");
            ++hist[(t - 100) / 7];
        }
        double z = 0;
        for(float v : lg) z += std::exp(double(v));
        for(int i = 0; i < 4; ++i) {
            const double pr = std::exp(double(lg[i])) / z, got = double(hist[i]) / N;
            require(std::abs(got - pr) < 4 * std::sqrt(pr * (1 - pr) / N) + 1e-3, "frequency differs from the distribution");
        }

        // Device masks from a constraint's bitmask: exactly the host's -inf pattern on strided rows, and selection on
        // the masked row equal to Core's constrained Sampler (AllowedTokens), ties to the lowest allowed id.
        {
            constexpr int rows = 3, stride = V + 7, words = (V + 31) / 32, mask_stride = words + 3;
            std::vector<float> block(static_cast<std::size_t>(rows) * stride);
            for(auto& x : block) x = std::round(n(rng) * 4.f) / 4.f;
            std::vector<std::uint32_t> mask(static_cast<std::size_t>(rows) * mask_stride);
            std::uniform_int_distribution<std::uint32_t> bits;
            for(auto& w : mask) w = bits(rng) & bits(rng);   // about a quarter of the tokens allowed
            float* d_block = nullptr;
            std::uint32_t* d_mask = nullptr;
            ok(cudaMalloc(&d_block, sizeof(float) * block.size()), "malloc");
            ok(cudaMalloc(&d_mask, sizeof(std::uint32_t) * mask.size()), "malloc");
            ok(cudaMemcpy(d_block, block.data(), sizeof(float) * block.size(), cudaMemcpyHostToDevice), "upload");
            ok(cudaMemcpy(d_mask, mask.data(), sizeof(std::uint32_t) * mask.size(), cudaMemcpyHostToDevice), "upload");
            ok(ts::enqueue_mask(d_block, V, stride, d_mask, mask_stride, rows, nullptr), "mask");
            std::vector<float> masked(block.size());
            ok(cudaMemcpy(masked.data(), d_block, sizeof(float) * masked.size(), cudaMemcpyDeviceToHost), "download");
            for(int r = 0; r < rows; ++r)
                for(int t = 0; t < stride; ++t) {
                    const std::size_t i = static_cast<std::size_t>(r) * stride + t;
                    const bool allowed = t >= V || ((mask[static_cast<std::size_t>(r) * mask_stride + t / 32] >> (t % 32)) & 1u);
                    require(allowed ? masked[i] == block[i] : std::isinf(masked[i]) && masked[i] < 0, "device mask differs from the host's");
                }
            for(int r = 0; r < rows; ++r) {
                std::vector<float> raw(block.begin() + static_cast<std::ptrdiff_t>(r) * stride, block.begin() + static_cast<std::ptrdiff_t>(r) * stride + V);
                std::vector<cc::Token> ids;
                for(int t = 0; t < V; ++t) if((mask[static_cast<std::size_t>(r) * mask_stride + t / 32] >> (t % 32)) & 1u) ids.push_back(t);
                s.load(std::vector<float>(masked.begin() + static_cast<std::ptrdiff_t>(r) * stride, masked.begin() + static_cast<std::ptrdiff_t>(r) * stride + V));
                s.set(ts::Params{});
                cc::SamplingParams greedy;
                greedy.temperature = 0.0;
                require(s.draw() == host.sample(raw, cc::AllowedTokens{ids}, greedy, empty).token, "masked greedy differs from Core's constrained choice");
                ts::Params q;
                q.temperature = 0.8f;
                q.top_k = 30;
                q.seed = 77 + static_cast<std::uint64_t>(r);
                s.set(q);
                const int got = s.draw();
                require(std::binary_search(ids.begin(), ids.end(), got), "a masked draw left the allowed set");
            }
            require(ts::enqueue_mask(d_block, V, V - 1, d_mask, words, 1, nullptr) == cudaErrorInvalidValue, "short logit stride");
            require(ts::enqueue_mask(d_block, V, V, d_mask, words - 1, 1, nullptr) == cudaErrorInvalidValue, "short mask stride");
            cudaFree(d_block);
            cudaFree(d_mask);
        }

        logprob_checks(rng);

        require(ts::enqueue_select(nullptr, V, 1, V, s.params, s.scratch, s.out, nullptr) == cudaErrorInvalidValue, "null logits");
        require(ts::enqueue_select(s.logits, 0, 1, 0, s.params, s.scratch, s.out, nullptr) == cudaErrorInvalidValue, "empty vocab");
        require(ts::enqueue_select(s.logits, V - 1, 1, V, s.params, s.scratch, s.out, nullptr) == cudaErrorInvalidValue, "short stride");
        require(ts::enqueue_select(s.logits, V, 1, V, s.params, nullptr, s.out, nullptr) == cudaErrorInvalidValue, "null scratch");
        std::cout << "token sampling: greedy ties, nucleus draws (" << mismatches << " boundary mismatches of " << draws
                  << "), repeatable seeds and frequencies within bounds\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
