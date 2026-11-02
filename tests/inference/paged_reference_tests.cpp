// The batched reference engine's paged attention, host half (paged_reference.h): the BF16 and E4M3 codecs (every code), the packed batch
// tables, HostPages behind a StateCache (forks with copy-on-write tails, reused prefix pages, rows crossing pages) and
// the FP64 reference against an independent dense oracle over each sequence's logical rows, with grouped heads, sliding
// windows, sinks, a soft-cap and tree masks, for BF16 and E4M3 pages; a query's result does not depend on its batch.
#include "paged_reference.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
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
template<class Error, class F> void rejects_as(F&& f, const std::string& what) {
    try { f(); } catch(const Error&) { return; }
    throw std::runtime_error("expected rejection: " + what);
}

void codecs() {
    // BF16: round to nearest even, every finite value round-trips, NaN stays NaN.
    require(pa::to_bf16(1.f) == 0x3f80 && pa::from_bf16(0x3f80) == 1.f, "bf16 one");
    require(pa::to_bf16(1.f + 0x1p-8f) == 0x3f80 && pa::to_bf16(1.f + 3 * 0x1p-8f) == 0x3f82, "bf16 ties to even");
    require(pa::to_bf16(1.f + 0x1p-8f + 0x1p-20f) == 0x3f81, "bf16 above the tie");
    require(std::isnan(pa::from_bf16(pa::to_bf16(std::numeric_limits<float>::quiet_NaN()))), "bf16 NaN");
    for(std::uint32_t b = 0; b < 0x10000; ++b) {
        const float x = pa::from_bf16(std::uint16_t(b));
        if(!std::isnan(x)) require(pa::to_bf16(x) == b, "bf16 round trip " + std::to_string(b));
    }
    // E4M3: every code round-trips; midpoints round to the even code; saturation at 448; subnormals step 2^-9.
    for(unsigned c = 0; c < 256; ++c) {
        const auto code = std::uint8_t(c);
        if((code & 0x7f) == 0x7f) { require(std::isnan(pa::from_e4m3(code)), "E4M3 NaN codes"); continue; }
        require(pa::to_e4m3(pa::from_e4m3(code)) == code, "E4M3 round trip " + std::to_string(c));
        if((code & 0x7f) < 0x7e) {
            const float mid = (pa::from_e4m3(code) + pa::from_e4m3(std::uint8_t(code + 1))) / 2;
            const auto even = std::uint8_t(code & 1 ? code + 1 : code);
            require(pa::to_e4m3(mid) == even, "E4M3 ties to even at " + std::to_string(c));
            require(pa::to_e4m3(std::nextafter(mid, 0.f)) == code && pa::to_e4m3(std::nextafter(mid, mid * 2)) == code + 1,
                    "E4M3 rounds to the nearer code at " + std::to_string(c));
        }
    }
    require(pa::from_e4m3(0x7e) == 448.f && pa::from_e4m3(0x01) == 0x1p-9f && pa::from_e4m3(0x08) == 0x1p-6f, "E4M3 values");
    for(const float big:{449.f, 463.f, 480.f, 1e30f, std::numeric_limits<float>::infinity()}) {
        require(pa::to_e4m3(big) == 0x7e && pa::to_e4m3(-big) == 0xfe, "E4M3 saturates");
    }
    require(pa::to_e4m3(0x1p-10f) == 0x00 && pa::to_e4m3(3 * 0x1p-10f) == 0x02 && pa::to_e4m3(-0.f) == 0x80, "E4M3 subnormal ties");
    require(pa::to_e4m3(std::numeric_limits<float>::quiet_NaN()) == 0x7f, "E4M3 NaN");
    const std::array<float, 3> zero{}, head{1.f, -896.f, 3.f};
    require(pa::e4m3_scale(zero) == 1.f && pa::e4m3_scale(head) == 2.f, "E4M3 scales");
}

void tables() {
    pa::Tables t(3, 4, 5);
    require(t.lens_at() == 12 && t.starts_at() == 15 && t.slots_at() == 19 && t.words().size() == 24, "packed layout");
    const std::array<cc::PageId, 2> a{7, 3};
    t.add(a, 20, 2, 16);
    t.add_idle();
    const std::array<cc::RowSlot, 2> slots{{{7, 2}, {3, 15}}};
    t.add_slots(slots, 16);
    const auto w = t.words();
    require(w[0] == 7 && w[1] == 3 && w[12] == 20 && w[13] == 0 && w[15] == 0 && w[16] == 2 && w[17] == 2, "tables, lengths, starts");
    require(w[19] == 7 * 16 + 2 && w[20] == 3 * 16 + 15, "slots are physical rows");
    require(t.seqs() == 2 && t.queries() == 2 && t.rows() == 2 && t.max_queries() == 2, "counts");
    rejects_as<std::invalid_argument>([&] { t.add(a, 33, 1, 16); }, "a table that does not address every row");
    rejects_as<std::invalid_argument>([&] { t.add(a, 2, 3); }, "more queries than rows");
    const std::array<cc::PageId, 5> wide{};
    rejects_as<std::invalid_argument>([&] { t.add(wide, 1, 1); }, "a table wider than the stride");
    t.add({}, 0, 0);
    rejects_as<std::length_error>([&] { t.add_idle(); }, "the sequence bound");
    const std::array<cc::RowSlot, 4> more{};
    rejects_as<std::length_error>([&] { t.add_slots(more, 16); }, "the row bound");
    t.clear();
    require(!t.seqs() && !t.queries() && !t.rows() && !t.max_queries() && t.words()[t.starts_at()] == 0, "clear");
    rejects_as<std::invalid_argument>([] { pa::Tables(0, 1, 0); }, "no sequences");
}

using Bits = std::vector<std::uint16_t>;
// A sequence's logical rows: BF16 key and value bits per row ([rows][kv_heads * head_dim]) and their tokens.
struct Logical { Bits k, v; std::vector<cc::Token> tokens; };

// Keys and values as the kernels read them, decoded from the logical rows with the codecs alone.
std::vector<float> decoded(const Bits& bits, int width, int head_dim, pa::Format format) {
    std::vector<float> out(bits.size());
    for(std::size_t r = 0; r < bits.size() / std::size_t(width); ++r)
        for(int g = 0; g < width / head_dim; ++g) {
            std::vector<float> head(static_cast<std::size_t>(head_dim), 0.f);
            for(int d = 0; d < head_dim; ++d) head[std::size_t(d)] = pa::from_bf16(bits[r * width + std::size_t(g) * head_dim + std::size_t(d)]);
            const float scale = pa::e4m3_scale(head);
            for(int d = 0; d < head_dim; ++d) {
                const float x = head[std::size_t(d)];
                out[r * width + std::size_t(g) * head_dim + std::size_t(d)] =
                    format == pa::Format::bf16 ? x : pa::from_bf16(pa::to_bf16(pa::from_e4m3(pa::to_e4m3(x / scale)) * scale));
            }
        }
    return out;
}

struct Variant { int window = 0; bool sinks = false; float softcap = 0.f; bool tree = false; };

// Dense oracle for one sequence's n newest rows: written from the visibility rules, not from the reference's code.
void dense(const Logical& seq, const std::vector<float>& q, std::size_t q_row, int n, int heads, int kv_heads, int hd,
           pa::Format format, float scale, const Variant& variant, const std::vector<float>& sinks,
           const std::vector<std::uint64_t>& trees, std::vector<float>& out) {
    const int width = kv_heads * hd, kv = int(seq.tokens.size()), base = kv - n, group = heads / kv_heads;
    const auto keys = decoded(seq.k, width, hd, format), values = decoded(seq.v, width, hd, format);
    for(int i = 0; i < n; ++i) {
        const std::uint64_t tree = variant.tree ? trees[q_row + std::size_t(i)] : 0;
        const int depth = variant.tree ? std::popcount(tree) - 1 : i;
        std::vector<int> seen;
        for(int key = 0; key < kv; ++key) {
            bool visible;
            if(key >= base) visible = variant.tree ? ((tree >> (key - base)) & 1) != 0 : key - base <= i;
            else visible = true;
            if(variant.window > 0 && key < base + depth - variant.window + 1) visible = false;
            if(visible) seen.push_back(key);
        }
        for(int h = 0; h < heads; ++h) {
            const int g = h / group;
            std::vector<double> e;
            double top = variant.sinks ? double(sinks[std::size_t(h)]) : -1e300;
            for(const int key:seen) {
                double dot = 0;
                for(int d = 0; d < hd; ++d)
                    dot += double(q[(q_row + std::size_t(i)) * std::size_t(heads * hd) + std::size_t(h * hd + d)]) *
                           keys[std::size_t(key) * width + std::size_t(g * hd + d)];
                double score = dot * scale;
                if(variant.softcap > 0) score = variant.softcap * std::tanh(score / variant.softcap);
                e.push_back(score);
                top = std::max(top, score);
            }
            double total = variant.sinks ? std::exp(double(sinks[std::size_t(h)]) - top) : 0.0;
            for(auto& x:e) { x = std::exp(x - top); total += x; }
            for(int d = 0; d < hd; ++d) {
                double sum = 0;
                for(std::size_t j = 0; j < seen.size(); ++j) sum += e[j] * values[std::size_t(seen[j]) * width + std::size_t(g * hd + d)];
                out[(q_row + std::size_t(i)) * std::size_t(heads * hd) + std::size_t(h * hd + d)] = seen.empty() ? 0.f : float(sum / total);
            }
        }
    }
}

struct Fixture {
    int hd, heads, kv_heads, width;
    pa::Format format;
    std::mt19937 rng{2024};
    cc::Resources resources{std::size_t(1) << 24, 0};
    pa::HostPages pages;
    cc::StateCache cache;
    pa::Tables tables{8, 64, 512};
    std::map<std::uint32_t, Logical> logical;   // by sequence index
    std::vector<cc::Round> rounds;
    std::vector<float> q;
    double worst = 0;
    static cc::StateCacheConfig config() {
        cc::StateCacheConfig c;
        c.pages = 96; c.page_tokens = 8; c.sequences = 8; c.max_tokens = 512; c.max_round_rows = 160;
        c.prefix_cache = true; c.identity = 0x9a9e;
        return c;
    }
    Fixture(int head_dim, int h, int g, pa::Format f)
        : hd(head_dim), heads(h), kv_heads(g), width(g * head_dim), format(f), pages(1, 96, 8, g, head_dim, f),
          cache(config(), pages, resources) {}
    std::uint16_t draw(float sigma) { std::normal_distribution<float> normal(0.f, sigma); return pa::to_bf16(normal(rng)); }

    // Stages rows[i] new rows on seqs[i], writes their keys and values, draws BF16-rounded queries and packs the tables.
    void stage(const std::vector<cc::SequenceId>& seqs, const std::vector<int>& rows) {
        tables.clear();
        rounds.clear();
        q.clear();
        for(std::size_t i = 0; i < seqs.size(); ++i) {
            if(!rows[i]) { tables.add_idle(); continue; }
            const auto round = cache.begin(seqs[i], std::uint32_t(rows[i]));
            rounds.push_back(round);
            Bits k(std::size_t(rows[i]) * width), v(k.size());
            for(auto& x:k) x = draw(1.f);
            for(auto& x:v) x = draw(2.f);
            std::vector<std::int32_t> slots;
            for(const auto slot:round.kv) slots.push_back(std::int32_t(slot.page * 8 + slot.offset));
            pages.write(0, k, v, width, rows[i], slots);
            auto& seq = logical[seqs[i].index];
            seq.k.insert(seq.k.end(), k.begin(), k.end());
            seq.v.insert(seq.v.end(), v.begin(), v.end());
            for(int r = 0; r < rows[i]; ++r) seq.tokens.push_back(cc::Token(rng() % 30000));
            tables.add_slots(round.kv, 8);
            tables.add(cache, seqs[i]);
            for(int r = 0; r < rows[i] * heads * hd; ++r) q.push_back(pa::from_bf16(draw(1.f)));
        }
    }
    void commit(const std::vector<cc::SequenceId>& seqs) {
        std::size_t r = 0;
        for(const auto& seq:seqs) {
            if(!cache.staged(seq)) continue;
            const auto& tokens = logical[seq.index].tokens;
            cache.commit(seq, rounds[r].id, std::span<const cc::Token>(tokens).subspan(tokens.size() - rounds[r].rows));
            ++r;
        }
        cache.retire(cache.epoch());
    }
    // The reference against the dense oracle for every staged row.
    std::vector<float> check(const std::vector<cc::SequenceId>& seqs, const Variant& variant) {
        std::vector<float> sinks(static_cast<std::size_t>(heads), 0.f);
        for(auto& x:sinks) x = std::normal_distribution<float>(0.f, 2.f)(rng);
        std::vector<std::uint64_t> trees;
        if(variant.tree)
            for(const auto& seq:seqs) {
                // Column i's parent is a random earlier column: it sees its ancestors and itself.
                std::vector<std::uint64_t> masks;
                for(std::uint32_t i = 0; i < cache.staged(seq); ++i)
                    masks.push_back((i ? masks[rng() % i] : 0) | (1ull << i));
                trees.insert(trees.end(), masks.begin(), masks.end());
            }
        pa::Problem problem;
        problem.q = q; problem.q_stride = std::size_t(heads) * hd; problem.heads = heads;
        problem.scale = 1.f / std::sqrt(float(hd)); problem.window = variant.window; problem.softcap = variant.softcap;
        if(variant.sinks) problem.sinks = sinks;
        if(variant.tree) problem.tree = trees;
        const auto got = pa::attend(pages, tables, problem);
        std::vector<float> want(got.size());
        std::size_t row = 0;
        for(const auto& seq:seqs) {
            const int n = int(cache.staged(seq));
            if(!n) continue;
            dense(logical[seq.index], q, row, n, heads, kv_heads, hd, format, problem.scale, variant, sinks, trees, want);
            row += std::size_t(n);
        }
        double magnitude = 0, error = 0;
        for(std::size_t i = 0; i < got.size(); ++i) {
            magnitude = std::max(magnitude, std::fabs(double(want[i])));
            error = std::max(error, std::fabs(double(got[i]) - want[i]));
        }
        worst = std::max(worst, error / std::max(magnitude, 1e-30));
        require(error <= 1e-6 * magnitude, "the reference differs from the dense oracle: " + std::to_string(error / magnitude));
        return got;
    }
};

void reference() {
    for(const auto format:{pa::Format::bf16, pa::Format::e4m3})
        for(const auto [hd, heads, kv_heads]:{std::array<int, 3>{64, 8, 2}, std::array<int, 3>{128, 4, 4}}) {
            Fixture f(hd, heads, kv_heads, format);
            for(const Variant variant:{Variant{}, Variant{24, false, 0.f, false}, Variant{0, true, 0.f, false},
                                       Variant{0, false, 5.f, false}, Variant{0, false, 0.f, true}, Variant{80, true, 5.f, true}}) {
                f.logical.clear();
                const auto donor = f.cache.open(), a = f.cache.open();
                // Ragged prefill from empty sequences, rows crossing pages.
                f.stage({donor, a}, {32, 45});
                if(!variant.tree) f.check({donor, a}, variant);
                f.commit({donor, a});
                // A prompt reusing the donor's committed pages, and a fork of a whose partial tail is copied on write.
                const auto reused = f.cache.open();
                auto prompt = f.logical[donor.index].tokens;
                prompt.push_back(1);
                require(f.cache.attach(reused, prompt, std::uint32_t(prompt.size() - 1)) == 32, "the donor's pages are reused");
                f.logical[reused.index] = f.logical[donor.index];
                const auto fork = f.cache.fork(a);
                f.logical[fork.index] = f.logical[a.index];
                f.cache.close(donor);
                f.cache.retire(f.cache.epoch());
                // Decode, a verify of five columns on the fork, an idle slot and a chunk on the reused prefix.
                for(int step = 0; step < 3; ++step) {
                    f.stage({a, fork, reused}, {1, 5, variant.tree ? 3 : 9});
                    f.check({a, fork, reused}, variant);
                    f.commit({a, fork, reused});
                }
                f.cache.close(a); f.cache.close(fork); f.cache.close(reused);
                f.cache.retire(f.cache.epoch());
            }
            std::printf("paged reference head_dim=%d heads=%d/%d %s worst=%.3g\n", hd, heads, kv_heads,
                        format == pa::Format::bf16 ? "bf16" : "e4m3", f.worst);
        }
}

void invariance() {
    Fixture f(64, 4, 2, pa::Format::bf16);
    const auto a = f.cache.open(), b = f.cache.open();
    f.stage({a, b}, {40, 17});
    f.commit({a, b});
    // A four-column verify on a with a one-row decode on b.
    f.stage({a, b}, {4, 1});
    const auto batch = f.check({a, b}, {});
    const std::size_t stride = std::size_t(f.heads) * f.hd;
    const auto q = f.q;
    const auto table = f.cache.block_table(a);
    const auto b_table = f.cache.block_table(b);
    const auto committed = f.cache.committed(a), b_len = f.cache.committed(b) + 1;
    pa::Problem problem;
    problem.q_stride = stride; problem.heads = f.heads; problem.scale = 0.125f;
    for(std::uint32_t col = 0; col < 4; ++col) {
        // Column col alone, as a one-query sequence of its own length, after an idle slot.
        pa::Tables alone(2, 64, 0);
        alone.add_idle();
        alone.add(table, committed + col + 1, 1, 8);
        problem.q = std::span<const float>(q).subspan(col * stride, stride);
        const auto one = pa::attend(f.pages, alone, problem);
        require(std::memcmp(one.data(), batch.data() + col * stride, stride * sizeof(float)) == 0, "a verify column equals itself alone");
    }
    pa::Tables alone(1, 64, 0);
    alone.add(b_table, b_len, 1, 8);
    problem.q = std::span<const float>(q).subspan(4 * stride, stride);
    const auto one = pa::attend(f.pages, alone, problem);
    require(std::memcmp(one.data(), batch.data() + 4 * stride, stride * sizeof(float)) == 0, "a sequence's result does not depend on its batch");
    f.commit({a, b});
}

void rejects() {
    pa::HostPages pages(1, 4, 8, 2, 64, pa::Format::bf16);
    pa::Tables t(2, 4, 4);
    const std::array<cc::PageId, 1> table{0};
    t.add(table, 3, 2, 8);
    std::vector<float> q(2 * 4 * 64);
    pa::Problem p;
    p.q = q; p.q_stride = 4 * 64; p.heads = 4; p.scale = 1;
    require(pa::attend(pages, t, p).size() == q.size(), "a valid problem");
    const auto refused = [&](pa::Problem bad, const char* what) { rejects_as<std::invalid_argument>([&] { pa::attend(pages, t, bad); }, what); };
    auto bad = p; bad.heads = 3; refused(bad, "heads not a multiple of kv_heads");
    bad = p; bad.q_stride = 64; refused(bad, "a short query stride");
    bad = p; bad.window = -1; refused(bad, "a negative window");
    bad = p; bad.softcap = std::numeric_limits<float>::quiet_NaN(); refused(bad, "a NaN soft-cap");
    const std::vector<float> sinks(3);
    bad = p; bad.sinks = sinks; refused(bad, "sinks of the wrong size");
    const std::vector<std::uint64_t> trees{1, 3};
    bad = p; bad.tree = trees; bad.window = 64; refused(bad, "tree masks with a window of 64 keys");
    bad = p; bad.q = std::span<const float>(q).first(10); refused(bad, "queries shorter than the rows");
    pa::Tables far(1, 4, 0);
    const std::array<cc::PageId, 1> outside{9};
    far.add(outside, 3, 1, 8);
    rejects_as<std::invalid_argument>([&] { pa::attend(pages, far, p); }, "a page outside the pages");
    rejects_as<std::invalid_argument>([] { pa::HostPages(1, 4, 8, 2, 96, pa::Format::bf16); }, "head dimension 96");
    const std::vector<std::uint16_t> k(128);
    const std::array<std::int32_t, 1> slot{32};
    rejects_as<std::out_of_range>([&] { pages.write(0, k, k, 128, 1, slot); }, "a write slot outside the pages");
    rejects_as<std::out_of_range>([&] { pages.copy_rows(0, 4, 1); }, "a copy outside the pages");
    rejects_as<std::logic_error>([&] { pages.copy_slot(0, 0, 1); }, "recurrent slots");
}
}

int main(int argc, char** argv) {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"codecs", codecs}, {"tables", tables}, {"reference", reference}, {"invariance", invariance}, {"rejects", rejects}}};
    try {
        const std::string_view name = argc > 1 ? argv[1] : "";
        bool found = false;
        for(const auto& [label, run]:cases) if(name == "all" || name == label) {
            run(); found = true;
            std::printf("paged reference passed: %.*s\n", int(label.size()), label.data());
        }
        require(found, "unknown paged attention test case");
    } catch(const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
