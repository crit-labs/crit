// The batched reference engine's paged attention kernels (paged_attention.h) against the host half (paged_reference.h): the KV
// write kernel stores the bytes HostPages::write stores (E4M3 codes and scales included), prefill, decode and verify
// match the FP64 reference over StateCache pages with forks and rows crossing pages (windows, sinks, a soft-cap and
// tree masks, BF16 and E4M3 pages, head dimensions 64 and 128), a verify column equals the same column decoded alone
// bit for bit, a sequence's result does not depend on its batch, and a sequence whose counts break the launch gets NaN
// rows and the fault flag. Needs an sm_80 or later device.
#include "paged_attention.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cc = crit::core;
namespace pa = crit::paged_attention;

namespace {
void require(bool ok, const std::string& what) { if(!ok) throw std::runtime_error(what); }
using pa::check;

template<class T> struct Buffer {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Buffer(std::size_t count) : n(std::max<std::size_t>(count, 1)) {
        check(cudaMalloc(&p, sizeof(T) * n), "malloc");
        check(cudaMemset(p, 0, sizeof(T) * n), "memset");
    }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& h) {
        require(h.size() <= n, "upload too large");
        check(cudaMemcpy(p, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> get(std::size_t count) const {
        std::vector<T> h(count);
        check(cudaMemcpy(h.data(), p, sizeof(T) * count, cudaMemcpyDeviceToHost), "download");
        return h;
    }
};

// Forwards the cache's copies to the host pages and the device pages.
struct Mirror final : cc::StateStorage {
    pa::HostPages& host;
    pa::DevicePages& device;
    Mirror(pa::HostPages& h, pa::DevicePages& d) : host(h), device(d) {}
    void copy_rows(cc::PageId from, cc::PageId to, std::uint32_t rows) override { host.copy_rows(from, to, rows); device.copy_rows(from, to, rows); }
    void copy_slot(std::uint32_t, cc::SlotId, cc::SlotId) override { throw std::logic_error("no slots"); }
};

constexpr int kPageTokens = 16, kPages = 96, kMaxRows = 320, kMaxSeqs = 6, kMaxColumns = 4;

struct Fixture {
    int hd, heads, kv_heads, width;
    pa::Format format;
    std::mt19937 rng{77};
    cc::Resources resources{std::size_t{1} << 24, 0};
    pa::HostPages host;
    pa::DevicePages device;
    Mirror mirror;
    cc::StateCache cache;
    pa::BatchTables tables{kMaxSeqs, kPages, kMaxRows};
    Buffer<std::uint16_t> q, k, v;
    Buffer<float> out, partial;
    Buffer<std::int32_t> fault;
    std::vector<float> q_host;
    std::vector<cc::Round> rounds;

    static cc::StateCacheConfig config() {
        cc::StateCacheConfig c;
        c.pages = kPages; c.page_tokens = kPageTokens; c.sequences = kMaxSeqs; c.max_tokens = 1024; c.max_round_rows = kMaxRows;
        return c;
    }
    Fixture(int head_dim, int h, int g, pa::Format f)
        : hd(head_dim), heads(h), kv_heads(g), width(g * head_dim), format(f), host(1, kPages, kPageTokens, g, head_dim, f),
          device(1, kPages, kPageTokens, g, head_dim, f, nullptr), mirror(host, device), cache(config(), mirror, resources),
          q(std::size_t(kMaxRows) * h * head_dim), k(std::size_t(kMaxRows) * g * head_dim), v(std::size_t(kMaxRows) * g * head_dim),
          out(std::size_t(kMaxRows) * h * head_dim), partial(pa::decode_partial_floats(kMaxSeqs, g, head_dim, 5, kMaxColumns, h)), fault(1) {
        // Start the device store where the host store starts (zero bytes, unit scales), so whole stores compare.
        const auto pages = device.layer(0);
        check(cudaMemset(pages.k, 0, host.keys(0).size()), "zero keys");
        check(cudaMemset(pages.v, 0, host.values(0).size()), "zero values");
        if(f == pa::Format::e4m3) {
            const std::vector<float> ones(host.key_scales(0).size(), 1.f);
            check(cudaMemcpy(pages.k_scale, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice), "unit scales");
            check(cudaMemcpy(pages.v_scale, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice), "unit scales");
        }
    }
    std::uint16_t draw(float sigma) { return pa::to_bf16(std::normal_distribution<float>(0.f, sigma)(rng)); }

    // Stages rows[i] new rows on seqs[i] (zero: an idle slot), writes their keys and values to both page stores and
    // uploads the queries and tables.
    pa::Batch stage(const std::vector<cc::SequenceId>& seqs, const std::vector<int>& rows) {
        tables.clear();
        rounds.clear();
        int total = 0;
        for(const int r:rows) total += r;
        std::vector<std::uint16_t> kb(std::size_t(total) * width), vb(kb.size()), qb(std::size_t(total) * heads * hd);
        for(auto& x:kb) x = draw(1.f);
        for(auto& x:vb) x = draw(2.f);
        for(auto& x:qb) x = draw(1.f);
        q_host.resize(qb.size());
        for(std::size_t i = 0; i < qb.size(); ++i) q_host[i] = pa::from_bf16(qb[i]);
        std::vector<std::int32_t> slots;
        for(std::size_t i = 0; i < seqs.size(); ++i) {
            if(!rows[i]) { tables.add_idle(); continue; }
            rounds.push_back(cache.begin(seqs[i], std::uint32_t(rows[i])));
            tables.add_slots(rounds.back().kv, kPageTokens);
            tables.add(cache, seqs[i]);
            for(const auto slot:rounds.back().kv) slots.push_back(std::int32_t(slot.page * kPageTokens + slot.offset));
        }
        k.put(kb); v.put(vb); q.put(qb);
        auto batch = tables.upload(nullptr);
        batch.fault = fault.p;
        check(pa::write(device.layer(0), reinterpret_cast<const __nv_bfloat16*>(k.p), reinterpret_cast<const __nv_bfloat16*>(v.p),
                        width, total, tables.device_slots(), nullptr), "write");
        host.write(0, kb, vb, width, total, slots);
        check(cudaDeviceSynchronize(), "write sync");
        return batch;
    }
    void commit(const std::vector<cc::SequenceId>& seqs) {
        std::size_t r = 0;
        for(const auto& seq:seqs) {
            if(!cache.staged(seq)) continue;
            std::vector<cc::Token> tokens(rounds[r].rows, 7);
            cache.commit(seq, rounds[r].id, tokens);
            ++r;
        }
        cache.retire(cache.epoch());
    }
    pa::Problem problem(int window = 0, std::span<const float> sinks = {}, float softcap = 0.f, std::span<const std::uint64_t> tree = {}) const {
        pa::Problem p;
        p.q = q_host; p.q_stride = std::size_t(heads) * hd; p.heads = heads; p.scale = 1.f / std::sqrt(float(hd));
        p.window = window; p.sinks = sinks; p.softcap = softcap; p.tree = tree;
        return p;
    }
    // The kernels' rows against the reference; probabilities are rounded to BF16 before P V: up to 2^-9 per weight.
    std::vector<float> compare(const pa::Problem& p, const char* what) {
        const auto want = pa::attend(host, tables.host(), p);
        const auto got = out.get(want.size());
        double magnitude = 0, error = 0;
        for(std::size_t i = 0; i < want.size(); ++i) {
            magnitude = std::max(magnitude, std::fabs(double(want[i])));
            error = std::max(error, std::fabs(double(got[i]) - want[i]));
        }
        require(error <= 0x1p-8 * magnitude, std::string(what) + " differs from the reference: " + std::to_string(error / magnitude));
        return got;
    }
    pa::Decode decode_problem(const pa::Batch& batch, int max_columns) {
        pa::Decode d;
        d.q = reinterpret_cast<const __nv_bfloat16*>(q.p); d.q_stride = heads * hd; d.pages = device.layer(0); d.batch = batch;
        d.out = out.p; d.out_stride = heads * hd; d.heads = heads; d.scale = 1.f / std::sqrt(float(hd)); d.max_columns = max_columns;
        d.blocks = 5; d.partial = partial.p;
        return d;
    }
    pa::Prefill prefill_problem(const pa::Batch& batch) {
        pa::Prefill p;
        p.q = reinterpret_cast<const __nv_bfloat16*>(q.p); p.q_stride = heads * hd; p.pages = device.layer(0); p.batch = batch;
        p.out = out.p; p.out_stride = heads * hd; p.heads = heads; p.scale = 1.f / std::sqrt(float(hd)); p.max_queries = tables.max_queries();
        return p;
    }
};

template<class Run> void each_geometry(Run&& run) {
    for(const auto format:{pa::Format::bf16, pa::Format::e4m3})
        for(const auto [hd, heads, kv_heads]:{std::array<int, 3>{64, 8, 2}, std::array<int, 3>{128, 4, 4}}) {
            Fixture f(hd, heads, kv_heads, format);
            run(f);
        }
}

void case_write() {
    each_geometry([](Fixture& f) {
        const auto a = f.cache.open(), b = f.cache.open();
        f.stage({a, b}, {37, 70});
        f.commit({a, b});
        const auto fork = f.cache.fork(a);
        f.stage({fork, a}, {3, 1});  // copy-on-write of a's shared tail reaches both stores
        const auto pages = f.device.layer(0);
        const auto bytes = f.host.keys(0).size();
        std::vector<std::byte> dk(bytes), dv(bytes);
        check(cudaMemcpy(dk.data(), pages.k, bytes, cudaMemcpyDeviceToHost), "keys");
        check(cudaMemcpy(dv.data(), pages.v, bytes, cudaMemcpyDeviceToHost), "values");
        require(std::memcmp(dk.data(), f.host.keys(0).data(), bytes) == 0 && std::memcmp(dv.data(), f.host.values(0).data(), bytes) == 0,
                "device pages hold the host encoder's bytes");
        if(f.format == pa::Format::e4m3) {
            const auto scales = f.host.key_scales(0).size();
            std::vector<float> ks(scales), vs(scales);
            check(cudaMemcpy(ks.data(), pages.k_scale, scales * sizeof(float), cudaMemcpyDeviceToHost), "scales");
            check(cudaMemcpy(vs.data(), pages.v_scale, scales * sizeof(float), cudaMemcpyDeviceToHost), "scales");
            require(std::memcmp(ks.data(), f.host.key_scales(0).data(), scales * sizeof(float)) == 0 &&
                    std::memcmp(vs.data(), f.host.value_scales(0).data(), scales * sizeof(float)) == 0, "E4M3 scales equal the host encoder's");
        }
        f.commit({fork, a});
    });
}

void case_attention() {
    each_geometry([](Fixture& f) {
        const auto a = f.cache.open(), b = f.cache.open(), c = f.cache.open();
        // Ragged prefill over empty sequences: 37, 150 (two 128-query tiles) and 5 rows.
        auto batch = f.stage({a, b, c}, {37, 150, 5});
        check(pa::prefill(f.prefill_problem(batch), nullptr), "prefill");
        check(cudaDeviceSynchronize(), "prefill sync");
        f.compare(f.problem(), "prefill");
        f.commit({a, b, c});
        const auto fork = f.cache.fork(b);
        std::vector<float> sinks(std::size_t(f.heads));
        for(auto& x:sinks) x = std::normal_distribution<float>(0.f, 2.f)(f.rng);
        Buffer<float> device_sinks(sinks.size());   // the kernels read the sinks on the device
        device_sinks.put(sinks);
        for(int step = 0; step < 6; ++step) {
            // Decode, a verify on the fork (copy-on-write tail), an idle slot, and decode across page boundaries.
            batch = f.stage({a, fork, b, c}, {1, kMaxColumns, 0, 1});
            const int window = step % 3 == 1 ? 40 : 0;
            const float softcap = step % 3 == 2 ? 6.f : 0.f;
            const std::span<const float> sink = step % 2 ? std::span<const float>(sinks) : std::span<const float>();
            auto d = f.decode_problem(batch, kMaxColumns);
            d.window = window; d.softcap = softcap; d.sinks = sink.empty() ? nullptr : device_sinks.p;
            check(pa::decode(d, nullptr), "decode");
            check(cudaDeviceSynchronize(), "decode sync");
            f.compare(f.problem(window, sink, softcap), "decode");
            f.commit({a, fork, b, c});
        }
        // A tree verify: column i's parent is a random earlier column; with a window it must exceed 64 keys.
        batch = f.stage({a, fork}, {1, kMaxColumns});
        std::vector<std::uint64_t> masks{1};
        for(std::uint64_t i = 0; i < std::uint64_t(kMaxColumns); ++i) masks.push_back((i ? masks[1 + f.rng() % i] : 0) | (1ull << i));
        Buffer<std::uint64_t> tree(masks.size());
        tree.put(masks);
        auto d = f.decode_problem(batch, kMaxColumns);
        d.tree = tree.p; d.window = 80;
        check(pa::decode(d, nullptr), "tree decode");
        check(cudaDeviceSynchronize(), "tree sync");
        f.compare(f.problem(80, {}, 0.f, masks), "tree verify");
        f.commit({a, fork});
    });
}

void case_invariance() {
    Fixture f(64, 8, 2, pa::Format::bf16);
    const auto a = f.cache.open(), b = f.cache.open();
    auto batch = f.stage({a, b}, {300, 20});
    check(pa::prefill(f.prefill_problem(batch), nullptr), "prefill");
    f.commit({a, b});
    batch = f.stage({a, b}, {kMaxColumns, 1});
    check(pa::decode(f.decode_problem(batch, kMaxColumns), nullptr), "verify");
    check(cudaDeviceSynchronize(), "verify sync");
    const auto verify = f.compare(f.problem(), "verify");
    const std::size_t stride = std::size_t(f.heads) * f.hd;
    const auto committed = f.cache.committed(a);
    const std::vector<cc::PageId> table(f.cache.block_table(a).begin(), f.cache.block_table(a).end());
    const std::vector<cc::PageId> b_table(f.cache.block_table(b).begin(), f.cache.block_table(b).end());
    const auto b_len = f.cache.committed(b) + 1;
    const auto queries = f.q.get(f.q.n);
    for(std::uint32_t col = 0; col < std::uint32_t(kMaxColumns); ++col) {
        // Column col alone, its query moved to row 0, keys up to its position, at max_columns 1 and the widest.
        for(const int columns:{1, kMaxColumns}) {
            f.tables.clear();
            f.tables.add(table, committed + col + 1, 1, kPageTokens);
            const auto alone = f.tables.upload(nullptr);
            f.q.put(std::vector<std::uint16_t>(queries.begin() + std::ptrdiff_t(col * stride), queries.begin() + std::ptrdiff_t((col + 1) * stride)));
            check(pa::decode(f.decode_problem(alone, columns), nullptr), "decode");
            require(std::memcmp(f.out.get(stride).data(), verify.data() + col * stride, stride * sizeof(float)) == 0,
                    "a verify column equals itself decoded alone");
        }
    }
    // b in a batch of its own, behind an idle slot, equals its row in the shared batch.
    f.tables.clear();
    f.tables.add_idle();
    f.tables.add(b_table, b_len, 1, kPageTokens);
    const auto own = f.tables.upload(nullptr);
    f.q.put(std::vector<std::uint16_t>(queries.begin() + std::ptrdiff_t(kMaxColumns * stride), queries.begin() + std::ptrdiff_t((kMaxColumns + 1) * stride)));
    check(pa::decode(f.decode_problem(own, kMaxColumns), nullptr), "decode");
    require(std::memcmp(f.out.get(stride).data(), verify.data() + kMaxColumns * stride, stride * sizeof(float)) == 0,
            "a sequence's result does not depend on its batch");
    f.commit({a, b});
}

void case_faults() {
    Fixture f(64, 4, 2, pa::Format::bf16);
    const auto a = f.cache.open(), b = f.cache.open();
    f.stage({a, b}, {20, 2});
    f.commit({a, b});
    auto batch = f.stage({a, b}, {1, 3});
    // Three queries on b at max_columns 2: b is not attended (NaN rows, fault set), a still is.
    check(cudaMemset(f.fault.p, 0, sizeof(std::int32_t)), "fault reset");
    auto d = f.decode_problem(batch, 2);
    check(pa::decode(d, nullptr), "decode");
    check(cudaDeviceSynchronize(), "decode sync");
    require(f.fault.get(1)[0] == 1, "a broken sequence raises the fault flag");
    const std::size_t stride = std::size_t(f.heads) * f.hd;
    const auto got = f.out.get(std::size_t(f.tables.queries()) * stride);
    for(std::size_t i = 0; i < got.size(); ++i) require(std::isnan(got[i]) == (i >= stride), "only the broken sequence's rows are NaN");
    // Invalid launches are refused before any kernel runs.
    d.max_columns = 0;
    require(pa::decode(d, nullptr) == cudaErrorInvalidValue, "max_columns zero");
    d = f.decode_problem(batch, 2);
    d.tree = reinterpret_cast<const std::uint64_t*>(f.partial.p);
    d.window = 64;
    require(pa::decode(d, nullptr) == cudaErrorInvalidValue, "tree masks with a window of 64 keys");
    auto p = f.prefill_problem(batch);
    p.heads = 3;
    require(pa::prefill(p, nullptr) == cudaErrorInvalidValue, "heads not a multiple of kv_heads");
    f.commit({a, b});
}
}

int main(int argc, char** argv) {
    const std::array<std::pair<std::string_view, void (*)()>, 4> cases{{
        {"write", case_write}, {"attention", case_attention}, {"invariance", case_invariance}, {"faults", case_faults}}};
    try {
        const std::string_view name = argc > 1 ? argv[1] : "";
        bool found = false;
        for(const auto& [label, run]:cases) if(name == "all" || name == label) {
            run(); found = true;
            std::printf("cuda paged attention passed: %.*s\n", int(label.size()), label.data());
        }
        require(found, "unknown CUDA paged attention test case");
    } catch(const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
