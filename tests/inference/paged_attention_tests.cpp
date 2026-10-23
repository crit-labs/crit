// Batched paged attention against the attention-state CPU reference over Core's StateCache: ragged batches with
// forks (copy-on-write tails), reused prefix pages and rows crossing page boundaries, for BF16 and E4M3 pages and head
// dimensions 64, 128 and 256. Also: E4M3 writes equal a host encoder, decode columns alone equal the same columns in a
// verify, a sequence's result does not depend on its batch, captured graphs replay new tables and lengths, and invalid
// problems are rejected.
#include "device_pages.h"
#include "paged_attention.h"

#include "attention.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cc = crit::core;
namespace pa = crit::paged_attention;
namespace ct = crit::techniques;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
using pa::check;

float e4m3(std::uint8_t b) {
    const int sign = b >> 7, exponent = (b >> 3) & 15, mantissa = b & 7;
    const float magnitude = exponent ? std::ldexp(1.f + static_cast<float>(mantissa) / 8.f, exponent - 7)
                                     : std::ldexp(static_cast<float>(mantissa) / 8.f, -6);
    return sign ? -magnitude : magnitude;
}
float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }

template <class T> struct Buffer {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Buffer(std::size_t count) : n(count) { check(cudaMalloc(&p, sizeof(T) * std::max<std::size_t>(count, 1)), "malloc"); check(cudaMemset(p, 0, sizeof(T) * std::max<std::size_t>(count, 1)), "memset"); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& h) { require(h.size() <= n, "upload too large"); check(cudaMemcpy(p, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> get(std::size_t count) const { std::vector<T> h(count); check(cudaMemcpy(h.data(), p, sizeof(T) * count, cudaMemcpyDeviceToHost), "download"); return h; }
};

// Forwards the cache's copies to the host reference pages and the device pages.
struct Mirror final : cc::StateStorage {
    ct::StatePages& host;
    pa::DevicePages& device;
    Mirror(ct::StatePages& h, pa::DevicePages& d) : host(h), device(d) {}
    void copy_rows(cc::PageId from, cc::PageId to, std::uint32_t rows) override { host.copy_rows(from, to, rows); device.copy_rows(from, to, rows); }
    void copy_slot(std::uint32_t, cc::SlotId, cc::SlotId) override { throw std::logic_error("no slots"); }
};

constexpr int kPageTokens = 16, kPages = 160, kMaxRows = 400, kMaxSeqs = 8;

struct Fixture {
    int hd, heads, kv_heads, width;
    pa::Format format;
    float scale;
    std::mt19937 rng{1234};
    cc::Resources resources{std::size_t{1} << 24, 0};
    ct::StatePages host;
    pa::DevicePages device;
    Mirror mirror;
    cc::StateCache cache;
    pa::BatchTables tables{kMaxSeqs, kPages, kMaxRows};
    Buffer<__nv_bfloat16> q, k, v;
    Buffer<float> out, partial;
    std::vector<float> q_host;  // BF16-rounded queries of the current step, [rows][heads * hd]
    int blocks = 3;
    double worst = 0.0;

    static cc::StateCacheConfig config() {
        cc::StateCacheConfig c;
        c.pages = kPages; c.page_tokens = kPageTokens; c.sequences = kMaxSeqs; c.max_tokens = 1024; c.max_round_rows = kMaxRows;
        c.prefix_cache = true; c.identity = 0xa77e;
        return c;
    }
    Fixture(int hd_, int heads_, int kv_heads_, pa::Format f)
        : hd(hd_), heads(heads_), kv_heads(kv_heads_), width(kv_heads_ * hd_), format(f), scale(1.f / std::sqrt(static_cast<float>(hd_))),
          host(kPages, kPageTokens, static_cast<std::size_t>(kv_heads_) * hd_, static_cast<std::size_t>(kv_heads_) * hd_),
          device(1, kPages, kPageTokens, kv_heads_, hd_, f, nullptr), mirror(host, device), cache(config(), mirror, resources),
          q(static_cast<std::size_t>(kMaxRows) * heads_ * hd_), k(static_cast<std::size_t>(kMaxRows) * kv_heads_ * hd_),
          v(static_cast<std::size_t>(kMaxRows) * kv_heads_ * hd_), out(static_cast<std::size_t>(kMaxRows) * heads_ * hd_),
          partial(pa::decode_partial_floats(kMaxSeqs, kv_heads_, hd_, 8, 4, heads_)) {}

    // Stages `rows` new rows on each sequence (in order), writes their keys and values to the device pages and the host
    // reference, draws queries for them and uploads the batch.
    pa::Batch stage(const std::vector<cc::SequenceId>& seqs, const std::vector<int>& rows, std::vector<cc::Round>& rounds) {
        std::normal_distribution<float> normal(0.f, 1.f);
        tables.clear();
        rounds.clear();
        int total = 0;
        for(int r : rows) total += r;
        std::vector<__nv_bfloat16> kb(static_cast<std::size_t>(total) * width), vb(kb.size()), qb(static_cast<std::size_t>(total) * heads * hd);
        for(auto& x : kb) x = __float2bfloat16(normal(rng));
        for(auto& x : vb) x = __float2bfloat16(2.f * normal(rng));
        q_host.resize(qb.size());
        for(std::size_t i = 0; i < qb.size(); ++i) { qb[i] = __float2bfloat16(normal(rng)); q_host[i] = __bfloat162float(qb[i]); }
        for(std::size_t i = 0; i < seqs.size(); ++i) {
            if(rows[i] == 0) { tables.add_idle(); continue; }
            rounds.push_back(cache.begin(seqs[i], static_cast<std::uint32_t>(rows[i])));
            tables.add_slots(rounds.back().kv, kPageTokens);
            tables.add(cache, seqs[i]);
        }
        k.put(kb); v.put(vb); q.put(qb);
        const pa::Batch batch = tables.upload(nullptr);
        const pa::Pages pages = device.layer(0);
        check(pa::write(pages, k.p, v.p, width, total, tables.device_slots(), nullptr), "write");
        check(cudaDeviceSynchronize(), "write sync");
        // The host reference holds what the kernels read: BF16 values, or bf16(code * scale) for E4M3 pages.
        std::vector<float> keys(kb.size()), values(vb.size());
        std::vector<cc::RowSlot> slots;
        for(const auto& round : rounds) slots.insert(slots.end(), round.kv.begin(), round.kv.end());
        for(std::size_t t = 0; t < slots.size(); ++t) {
            const std::size_t row = static_cast<std::size_t>(slots[t].page) * kPageTokens + slots[t].offset;
            if(format == pa::Format::bf16) {
                for(int e = 0; e < width; ++e) {
                    keys[t * width + e] = __bfloat162float(kb[t * width + e]);
                    values[t * width + e] = __bfloat162float(vb[t * width + e]);
                }
                continue;
            }
            std::vector<std::uint8_t> kc(width), vc(width);
            std::vector<float> ks(kv_heads), vs(kv_heads);
            check(cudaMemcpy(kc.data(), static_cast<std::uint8_t*>(pages.k) + row * width, width, cudaMemcpyDeviceToHost), "codes");
            check(cudaMemcpy(vc.data(), static_cast<std::uint8_t*>(pages.v) + row * width, width, cudaMemcpyDeviceToHost), "codes");
            check(cudaMemcpy(ks.data(), pages.k_scale + row * kv_heads, sizeof(float) * kv_heads, cudaMemcpyDeviceToHost), "scales");
            check(cudaMemcpy(vs.data(), pages.v_scale + row * kv_heads, sizeof(float) * kv_heads, cudaMemcpyDeviceToHost), "scales");
            for(int g = 0; g < kv_heads; ++g) {
                // The write kernel's encoding: scale = max |x| / 448, code = e4m3(x / scale) saturated.
                float mk = 0.f, mv = 0.f;
                for(int d = 0; d < hd; ++d) {
                    mk = std::max(mk, std::fabs(__bfloat162float(kb[t * width + g * hd + d])));
                    mv = std::max(mv, std::fabs(__bfloat162float(vb[t * width + g * hd + d])));
                }
                const float sk = mk > 0.f ? mk / 448.f : 1.f, sv = mv > 0.f ? mv / 448.f : 1.f;
                require(ks[g] == sk && vs[g] == sv, "E4M3 scales differ from the host encoder");
                for(int d = 0; d < hd; ++d) {
                    const int e = g * hd + d;
                    const auto hk = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(__bfloat162float(kb[t * width + e]) / sk, __NV_SATFINITE, __NV_E4M3));
                    const auto hv = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(__bfloat162float(vb[t * width + e]) / sv, __NV_SATFINITE, __NV_E4M3));
                    require(kc[e] == hk && vc[e] == hv, "E4M3 codes differ from the host encoder");
                    keys[t * width + e] = bf16_round(e4m3(kc[e]) * ks[g]);
                    values[t * width + e] = bf16_round(e4m3(vc[e]) * vs[g]);
                }
            }
        }
        host.write_rows(slots, keys, values);
        return batch;
    }

    void commit(const std::vector<cc::SequenceId>& seqs, std::vector<cc::Round>& rounds, int token_base) {
        std::size_t r = 0;
        for(const auto& seq : seqs) {
            if(!cache.staged(seq)) continue;
            std::vector<cc::Token> tokens(rounds[r].rows);
            for(std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = token_base + 10000 * static_cast<int>(r) + static_cast<int>(i);
            cache.commit(seq, rounds[r].id, tokens);
            ++r;
        }
        cache.retire(cache.epoch());
    }

    // Each sequence's staged rows against the CPU reference over its committed and staged rows.
    void compare(const std::vector<cc::SequenceId>& seqs) {
        const auto got = out.get(static_cast<std::size_t>(tables.queries()) * heads * hd);
        int row = 0;
        for(const auto& seq : seqs) {
            const int n = static_cast<int>(cache.staged(seq));
            if(!n) continue;
            const auto view = host.view(cache, seq, true);
            std::vector<std::size_t> positions(n);
            for(int i = 0; i < n; ++i) positions[i] = view.rows() - static_cast<std::size_t>(n) + static_cast<std::size_t>(i);
            ct::PagedAttentionPlan plan(view, {static_cast<std::size_t>(heads), static_cast<std::size_t>(kv_heads), static_cast<std::size_t>(hd),
                                               static_cast<std::size_t>(hd), scale}, positions, 16, 64, std::size_t{1} << 30);
            const std::size_t stride = static_cast<std::size_t>(heads) * hd;
            const ct::HeadTensorView queries{std::span<const float>(q_host).subspan(static_cast<std::size_t>(row) * stride, static_cast<std::size_t>(n) * stride),
                                             static_cast<std::size_t>(n), static_cast<std::size_t>(heads), static_cast<std::size_t>(hd), stride, static_cast<std::size_t>(hd)};
            const auto ref = plan.run(queries);
            double magnitude = 0.0, error = 0.0;
            for(std::size_t i = 0; i < ref.values.size(); ++i) {
                magnitude = std::max(magnitude, std::fabs(static_cast<double>(ref.values[i])));
                error = std::max(error, std::fabs(static_cast<double>(got[static_cast<std::size_t>(row) * stride + i]) - ref.values[i]));
            }
            worst = std::max(worst, error / magnitude);
            // Probabilities are rounded to BF16 (8 significant bits) before P V: up to 2^-9 of each weight.
            require(error <= 0x1p-8 * magnitude, "paged attention differs from the CPU reference: " + std::to_string(error / magnitude));
            row += n;
        }
    }

    pa::Decode decode_problem(const pa::Batch& batch, int max_columns) {
        pa::Decode p;
        p.q = q.p; p.q_stride = heads * hd; p.pages = device.layer(0); p.batch = batch; p.out = out.p; p.out_stride = heads * hd;
        p.heads = heads; p.scale = scale; p.max_columns = max_columns; p.blocks = blocks; p.partial = partial.p;
        return p;
    }
    pa::Prefill prefill_problem(const pa::Batch& batch) {
        pa::Prefill p;
        p.q = q.p; p.q_stride = heads * hd; p.pages = device.layer(0); p.batch = batch; p.out = out.p; p.out_stride = heads * hd;
        p.heads = heads; p.scale = scale; p.max_queries = tables.max_queries();
        return p;
    }
    void run_prefill(const std::vector<cc::SequenceId>& seqs, const std::vector<int>& rows, int token_base) {
        std::vector<cc::Round> rounds;
        const auto batch = stage(seqs, rows, rounds);
        check(pa::prefill(prefill_problem(batch), nullptr), "prefill");
        check(cudaDeviceSynchronize(), "prefill sync");
        compare(seqs);
        commit(seqs, rounds, token_base);
    }
    std::vector<float> run_decode(const std::vector<cc::SequenceId>& seqs, const std::vector<int>& rows, int max_columns, bool keep_staged = false,
                                  std::vector<cc::Round>* kept = nullptr) {
        std::vector<cc::Round> rounds;
        const auto batch = stage(seqs, rows, rounds);
        check(pa::decode(decode_problem(batch, max_columns), nullptr), "decode");
        check(cudaDeviceSynchronize(), "decode sync");
        compare(seqs);
        auto got = out.get(static_cast<std::size_t>(tables.queries()) * heads * hd);
        if(keep_staged && kept) *kept = rounds;
        else commit(seqs, rounds, 5000);
        return got;
    }
};

// Prompt pages for prefix reuse: a 64-row prompt committed and closed; its full pages stay indexed.
struct Scenario {
    Fixture& f;
    cc::SequenceId a, b, c, e;
    explicit Scenario(Fixture& fx) : f(fx) {
        const auto d = f.cache.open();
        a = f.cache.open();
        // Ragged prefill from empty sequences: 64 and 300 rows (300 crosses a 128-query tile and three decode pieces).
        f.run_prefill({d, a}, {64, 300}, 1000);
        f.cache.close(d);
        f.cache.retire(f.cache.epoch());
        c = f.cache.open();
        std::vector<cc::Token> prompt(64 + 9);
        for(std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = 1000 + static_cast<int>(i);
        require(f.cache.attach(c, prompt, static_cast<std::uint32_t>(prompt.size() - 1)) == 64, "the closed prompt's pages are reused");
        require(f.cache.stats().prefix_hits == 4, "four prefix pages hit");
        e = f.cache.open();
        // Prefill over a reused paged prefix (9 new rows after 64 reused), a fresh 130-row chunk and one row.
        b = f.cache.fork(a);
        f.run_prefill({c, e, b}, {9, 130, 1}, 2000);
    }
};

void case_decode(Fixture& f) {
    Scenario s(f);
    // A one-column decode, an idle slot, a 3-column verify on a fork whose tail page is shared (copy-on-write), and a
    // sequence on reused prefix pages.
    f.run_decode({s.a, s.e, s.b, s.c}, {1, 0, 3, 1}, 3);
    // Decode steps across a page boundary (300 + 1 + ... rows) keep matching the reference.
    for(int i = 0; i < 20; ++i) f.run_decode({s.a, s.b, s.c, s.e}, {1, 1, 1, 1}, 1);
}

void case_invariance(Fixture& f) {
    Scenario s(f);
    std::vector<cc::Round> rounds;
    const auto verify = f.run_decode({s.b, s.c}, {3, 1}, 3, true, &rounds);
    const std::size_t stride = static_cast<std::size_t>(f.heads) * f.hd;
    const auto committed = f.cache.committed(s.b);
    const auto table = f.cache.block_table(s.b);
    const auto c_len = f.cache.committed(s.c) + 1;
    const auto c_table = f.cache.block_table(s.c);
    const std::vector<float> c_row(verify.begin() + static_cast<std::ptrdiff_t>(3 * stride), verify.begin() + static_cast<std::ptrdiff_t>(4 * stride));
    for(int col = 0; col < 3; ++col) {
        // Column col of the verify as a lone one-column decode: its query row moved to row 0, keys up to its position.
        f.tables.clear();
        f.tables.add(table, committed + static_cast<std::uint32_t>(col) + 1, 1);
        const auto batch = f.tables.upload(nullptr);
        std::vector<__nv_bfloat16> qrow(stride);
        for(std::size_t i = 0; i < stride; ++i) qrow[i] = __float2bfloat16(f.q_host[col * stride + i]);
        Buffer<__nv_bfloat16> q1(stride);
        q1.put(qrow);
        auto p = f.decode_problem(batch, 1);
        p.q = q1.p;
        check(pa::decode(p, nullptr), "decode");
        const auto alone = f.out.get(stride);
        require(std::memcmp(alone.data(), verify.data() + col * stride, stride * sizeof(float)) == 0, "a decode column alone equals its verify column");
    }
    // Sequence c in a batch of its own (and a larger max_columns) equals its row in the shared batch.
    f.tables.clear();
    f.tables.add_idle();
    f.tables.add(c_table, c_len, 1);
    const auto batch = f.tables.upload(nullptr);
    std::vector<__nv_bfloat16> qrow(stride);
    for(std::size_t i = 0; i < stride; ++i) qrow[i] = __float2bfloat16(f.q_host[3 * stride + i]);
    Buffer<__nv_bfloat16> q1(stride);
    q1.put(qrow);
    auto p = f.decode_problem(batch, 2);
    p.q = q1.p;
    check(pa::decode(p, nullptr), "decode");
    require(std::memcmp(f.out.get(stride).data(), c_row.data(), stride * sizeof(float)) == 0, "a sequence's result does not depend on its batch");
    f.commit({s.b, s.c}, rounds, 7000);
}

void case_graph(Fixture& f) {
    Scenario s(f);
    // Capture one decode for four sequences, then replay it for later steps with new lengths and tables.
    std::vector<cc::Round> rounds;
    auto batch = f.stage({s.a, s.b, s.c, s.e}, {1, 1, 1, 1}, rounds);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    check(pa::decode(f.decode_problem(batch, 1), stream), "captured decode");
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    for(int step = 0; step < 6; ++step) {
        if(step) batch = f.stage({s.a, s.b, s.c, s.e}, {1, 1, 1, 1}, rounds);
        check(cudaGraphLaunch(exec, stream), "replay");
        check(cudaStreamSynchronize(stream), "replay sync");
        f.compare({s.a, s.b, s.c, s.e});
        const auto replayed = f.out.get(static_cast<std::size_t>(4) * f.heads * f.hd);
        check(pa::decode(f.decode_problem(batch, 1), stream), "direct decode");
        check(cudaStreamSynchronize(stream), "direct sync");
        require(replayed == f.out.get(replayed.size()), "graph replay equals a direct launch");
        f.commit({s.a, s.b, s.c, s.e}, rounds, 8000 + step);
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

void case_rejects() {
    Buffer<std::int32_t> ints(64);
    Buffer<std::uint8_t> bytes(1 << 16);
    Buffer<float> floats(1 << 16);
    Buffer<__nv_bfloat16> rows(1 << 12);
    pa::Pages pages{bytes.p, bytes.p, nullptr, nullptr, pa::Format::e4m3, 16, 2, 64};
    const pa::Batch batch{1, ints.p, 4, ints.p, ints.p};
    pa::Decode d;
    d.q = rows.p; d.q_stride = 512; d.pages = pages; d.batch = batch; d.out = floats.p; d.out_stride = 512; d.heads = 8;
    d.scale = 1.f; d.max_columns = 1; d.blocks = 1; d.partial = floats.p;
    require(pa::decode(d, nullptr) == cudaErrorInvalidValue, "E4M3 pages without scales are rejected");
    d.pages.k_scale = d.pages.v_scale = floats.p;
    d.max_columns = 17;
    require(pa::decode(d, nullptr) == cudaErrorInvalidValue, "more than 64 rows per key/value head are rejected");
    d.max_columns = 1;
    d.heads = 7;
    require(pa::decode(d, nullptr) == cudaErrorInvalidValue, "heads not a multiple of key/value heads are rejected");
    d.heads = 8;
    d.pages.head_dim = 96;
    require(pa::decode(d, nullptr) == cudaErrorInvalidValue, "unsupported head dimensions are rejected");
    d.pages.head_dim = 64;
    d.blocks = 0;
    require(pa::decode(d, nullptr) == cudaErrorInvalidValue, "zero blocks are rejected");
    pa::Prefill p;
    p.q = rows.p; p.q_stride = 512; p.pages = d.pages; p.batch = batch; p.out = floats.p; p.out_stride = 512; p.heads = 8; p.scale = 1.f;
    require(pa::prefill(p, nullptr) == cudaErrorInvalidValue, "a prefill without a query bound is rejected");
    p.max_queries = 1;
    p.q_stride = 100;
    require(pa::prefill(p, nullptr) == cudaErrorInvalidValue, "unaligned query strides are rejected");
    require(pa::write(d.pages, rows.p, rows.p, 128, 1, nullptr, nullptr) == cudaErrorInvalidValue, "a write without slots is rejected");
}

using Case = void (*)(Fixture&);
void each_geometry(Case run, const char* name) {
    struct G { int hd, heads, kv_heads; };
    for(const G g : {G{64, 8, 2}, G{128, 4, 4}, G{256, 16, 2}})
        for(const auto format : {pa::Format::bf16, pa::Format::e4m3}) {
            Fixture f(g.hd, g.heads, g.kv_heads, format);
            run(f);
            std::cout << name << " head_dim=" << g.hd << " heads=" << g.heads << "/" << g.kv_heads
                      << (format == pa::Format::bf16 ? " bf16" : " e4m3") << " worst=" << f.worst << '\n';
        }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if(cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::cout << "no CUDA device; skipped\n"; return 0; }
        const std::string which = argc > 1 ? argv[1] : "all";
        if(which == "decode" || which == "all") each_geometry(case_decode, "decode");
        if(which == "invariance" || which == "all") each_geometry(case_invariance, "invariance");
        if(which == "graph" || which == "all") each_geometry(case_graph, "graph");
        if(which == "rejects" || which == "all") case_rejects();
        std::cout << "paged attention " << which << " passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
