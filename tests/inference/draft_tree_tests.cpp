// Token trees on the device against a host mirror of tree.py: EAGLE-2-style expansion, beam and final selection and
// the tree they build (tokens, parents, masks, depths, padding nodes); the tree walk's emitted tokens, path and
// control fields for random trees, budgets and stops, and equality with the greedy accept kernel on chains; and
// compaction of kept rows (contiguous and slot-mapped) that a rerun after a smaller count leaves unchanged.
#include "draft_tree.h"
#include "greedy_accept.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dt = crit::techniques::draft_trees;
namespace dr = crit::techniques::device_round;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

template<class T> struct Device {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t count) : n(count) { ok(cudaMalloc(&p, sizeof(T) * std::max<std::size_t>(count, 1)), "malloc"); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { put(v); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& v) { ok(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> get() const {
        std::vector<T> v(n);
        ok(cudaMemcpy(v.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost), "download");
        return v;
    }
};

struct Cand { int token, parent; double score; int depth; };

// A deterministic drafter: the logit row of a path, with values far enough apart that FP32 and FP64 rank alike.
std::vector<float> draft_row(const std::vector<int>& path, int vocab) {
    std::uint64_t h = 1469598103934665603ull;
    for(int t : path) h = (h ^ std::uint64_t(t + 1)) * 1099511628211ull;
    std::mt19937_64 rng(h);
    std::vector<float> row(std::size_t(vocab));
    std::vector<int> perm(std::size_t(vocab));
    for(int i = 0; i < vocab; ++i) perm[std::size_t(i)] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::uniform_real_distribution<float> jitter(0.f, 0.3f);   // keeps paths' scores apart, ranks within a row fixed
    for(int i = 0; i < vocab; ++i) row[std::size_t(perm[std::size_t(i)])] = 8.f - 0.37f * float(i) + jitter(rng);
    return row;
}

std::vector<int> path_of(const std::vector<Cand>& c, int i) {
    std::vector<int> out;
    for(; i >= 0; i = c[std::size_t(i)].parent) out.push_back(c[std::size_t(i)].token);
    std::reverse(out.begin(), out.end());
    return out;
}

std::vector<int> rank(const std::vector<Cand>& c, std::vector<int> ids, int m) {
    std::sort(ids.begin(), ids.end(), [&](int a, int b) {
        return c[std::size_t(a)].score != c[std::size_t(b)].score ? c[std::size_t(a)].score > c[std::size_t(b)].score : a < b;
    });
    if(int(ids.size()) > m) ids.resize(std::size_t(m));
    return ids;
}

void case_build() {
    constexpr int V = 300, depth = 4, branch = 3, beam = 3, nodes = 16, rows = 2, capacity = branch * (1 + (depth - 1) * beam);
    Device<std::int32_t> token(std::size_t(rows) * capacity), parent(token.n), dep(token.n), count(std::vector<std::int32_t>(rows, 0));
    Device<float> score(token.n);
    dt::Candidates c{token.p, parent.p, score.p, dep.p, count.p, capacity};
    std::vector<std::vector<Cand>> host(rows);
    std::vector<std::vector<int>> frontier(rows, std::vector<int>{-1});
    for(int d = 0; d < depth; ++d) {
        const int width = d == 0 ? 1 : beam;
        std::vector<float> logits(std::size_t(rows) * width * V);
        std::vector<std::int32_t> front(std::size_t(rows) * width, -2);
        for(int r = 0; r < rows; ++r) {
            std::vector<int> newest;
            for(int f = 0; f < int(frontier[std::size_t(r)].size()); ++f) {
                const int fid = frontier[std::size_t(r)][std::size_t(f)];
                front[std::size_t(r) * width + std::size_t(f)] = fid;
                const auto row = draft_row(fid >= 0 ? path_of(host[std::size_t(r)], fid) : std::vector<int>{r}, V);
                std::copy(row.begin(), row.end(), logits.begin() + std::ptrdiff_t((std::size_t(r) * width + std::size_t(f)) * V));
                // host expansion
                const float top = *std::max_element(row.begin(), row.end());
                double sum = 0.;
                for(float x : row) sum += std::exp(double(x - top));
                const double lse = double(top) + std::log(sum);
                std::vector<int> order(std::size_t(V));
                for(int t = 0; t < V; ++t) order[std::size_t(t)] = t;
                std::sort(order.begin(), order.end(), [&](int a, int b) { return row[std::size_t(a)] != row[std::size_t(b)] ? row[std::size_t(a)] > row[std::size_t(b)] : a < b; });
                const double base = fid >= 0 ? host[std::size_t(r)][std::size_t(fid)].score : 0.;
                const int hd = fid >= 0 ? host[std::size_t(r)][std::size_t(fid)].depth + 1 : 1;
                // indices follow the device layout: count + f * branch + k (holes for unused entries)
                const int at = d == 0 ? 0 : branch + (d - 1) * beam * branch + f * branch;
                if(int(host[std::size_t(r)].size()) < at + branch) host[std::size_t(r)].resize(std::size_t(at + branch), Cand{-1, -1, -1e300, -1});
                for(int k = 0; k < branch; ++k) {
                    host[std::size_t(r)][std::size_t(at + k)] = Cand{order[std::size_t(k)], fid, base + double(row[std::size_t(order[std::size_t(k)])]) - lse, hd};
                    newest.push_back(at + k);
                }
            }
            frontier[std::size_t(r)] = rank(host[std::size_t(r)], newest, beam);
            if(int(host[std::size_t(r)].size()) < branch + d * beam * branch)
                host[std::size_t(r)].resize(std::size_t(branch + d * beam * branch), Cand{-1, -1, -1e300, -1});
        }
        Device<float> dlogits(logits);
        Device<std::int32_t> dfront(front);
        ok(dt::enqueue_expand(dlogits.p, V, V, dfront.p, width, branch, c, rows, nullptr), "expand");
        Device<std::int32_t> beam_out(std::size_t(rows) * beam);
        ok(dt::enqueue_select(c, d + 1, beam, beam_out.p, rows, nullptr), "select beam");
        const auto got = beam_out.get();
        for(int r = 0; r < rows; ++r)
            for(int b = 0; b < beam; ++b)
                require(got[std::size_t(r) * beam + std::size_t(b)] == (b < int(frontier[std::size_t(r)].size()) ? frontier[std::size_t(r)][std::size_t(b)] : -1),
                        "device beam differs from the host's");
    }
    Device<std::int32_t> kept(std::size_t(rows) * (nodes - 1));
    ok(dt::enqueue_select(c, 0, nodes - 1, kept.p, rows, nullptr), "select nodes");
    Device<std::int32_t> drafts(std::size_t(rows) * (nodes - 1)), parents(drafts.n), depths(std::size_t(rows) * nodes), tokens(depths.n),
        roots(std::vector<std::int32_t>{7, 9});
    Device<std::uint64_t> masks(std::size_t(rows) * nodes);
    dt::Build b{kept.p, roots.p, 1, drafts.p, parents.p, masks.p, depths.p, tokens.p, nodes, rows};
    ok(dt::enqueue_build(c, b, nullptr), "build");
    const auto dk = kept.get(), dd = drafts.get(), dp = parents.get(), dm = masks.get(), ddep = depths.get(), dtok = tokens.get();
    for(int r = 0; r < rows; ++r) {
        std::vector<int> all;
        for(int i = 0; i < int(host[std::size_t(r)].size()); ++i) if(host[std::size_t(r)][std::size_t(i)].depth >= 1) all.push_back(i);
        const auto want = rank(host[std::size_t(r)], all, nodes - 1);
        for(int i = 0; i < nodes - 1; ++i) require(dk[std::size_t(r) * (nodes - 1) + std::size_t(i)] == want[std::size_t(i)], "kept nodes differ");
        // host build: by depth then rank
        std::vector<int> order(want.size());
        for(std::size_t i = 0; i < order.size(); ++i) order[i] = int(i);
        std::stable_sort(order.begin(), order.end(), [&](int a, int x) { return host[std::size_t(r)][std::size_t(want[std::size_t(a)])].depth < host[std::size_t(r)][std::size_t(want[std::size_t(x)])].depth; });
        std::map<int, int> node_of;
        std::vector<std::uint64_t> hm{1ull};
        std::vector<int> hdep{0};
        require(dtok[std::size_t(r) * nodes] == (r ? 9 : 7), "root token");
        for(std::size_t i = 0; i < order.size(); ++i) {
            const auto& cand = host[std::size_t(r)][std::size_t(want[std::size_t(order[i])])];
            const int par = cand.parent < 0 ? 0 : node_of.at(cand.parent);
            node_of[want[std::size_t(order[i])]] = int(i) + 1;
            hm.push_back(hm[std::size_t(par)] | (1ull << (i + 1)));
            hdep.push_back(hdep[std::size_t(par)] + 1);
            require(dd[std::size_t(r) * (nodes - 1) + i] == cand.token, "node token differs");
            require(dp[std::size_t(r) * (nodes - 1) + i] == par, "node parent differs");
            require(par < int(i) + 1, "parents come first");
        }
        for(int i = 0; i < nodes; ++i) {
            require(dm[std::size_t(r) * nodes + std::size_t(i)] == hm[std::size_t(i)], "node mask differs");
            require(ddep[std::size_t(r) * nodes + std::size_t(i)] == hdep[std::size_t(i)], "node depth differs");
        }
    }
    // Padding: a single expansion of 3 children cannot fill 8 nodes.
    {
        Device<std::int32_t> cnt(std::vector<std::int32_t>{0}), t1(16), p1(16), d1(16);
        Device<float> s1(16);
        dt::Candidates small{t1.p, p1.p, s1.p, d1.p, cnt.p, 16};
        const auto row = draft_row({1}, V);
        Device<float> lg(row);
        Device<std::int32_t> root(std::vector<std::int32_t>{-1});
        ok(dt::enqueue_expand(lg.p, V, V, root.p, 1, 3, small, 1, nullptr), "small expand");
        Device<std::int32_t> k8(7), dr8(7), pr8(7), dp8(8);
        Device<std::uint64_t> m8(8);
        ok(dt::enqueue_select(small, 0, 7, k8.p, 1, nullptr), "small select");
        dt::Build sb{k8.p, nullptr, 1, dr8.p, pr8.p, m8.p, dp8.p, nullptr, 8, 1};
        ok(dt::enqueue_build(small, sb, nullptr), "small build");
        const auto kk = k8.get(), pp = pr8.get(), dd8 = dr8.get();
        require(kk[3] == -1 && pp[3] == -1 && dd8[3] == 0 && pp[0] == 0, "padding nodes hang off nothing and carry token 0");
    }
    std::cout << "build: beams, kept nodes and trees equal the host's, padding nodes are unmatchable\n";
}

struct Walk { std::vector<int> out, path; };
Walk host_walk(const std::vector<int>& tokens, const std::vector<int>& parents, const std::vector<int>& predictions, int remaining,
               const std::vector<int>& stops) {
    Walk w;
    w.path.push_back(0);
    int v = 0;
    while(int(w.out.size()) < remaining) {
        const int t = predictions[std::size_t(v)];
        int child = 0;
        for(std::size_t i = 1; i < tokens.size() + 1; ++i)
            if(parents[i - 1] == v && tokens[i - 1] == t) { child = int(i); break; }
        w.out.push_back(t);
        if(!child) break;
        w.path.push_back(child);
        v = child;
        if(std::find(stops.begin(), stops.end(), t) != stops.end()) break;
    }
    return w;
}

void case_accept() {
    std::mt19937 rng(9);
    constexpr int R = 200, N = 12, K = N - 1;
    std::vector<std::int32_t> tokens(std::size_t(R) * K), parents(tokens.size()), predictions(std::size_t(R) * N), control(std::size_t(R) * dr::control::fields, 0);
    std::vector<int> budgets(R);
    const std::vector<std::int32_t> stops = {5};
    for(int r = 0; r < R; ++r) {
        for(int i = 1; i < N; ++i) {
            parents[std::size_t(r) * K + std::size_t(i - 1)] = std::uniform_int_distribution<int>(0, i - 1)(rng);
            tokens[std::size_t(r) * K + std::size_t(i - 1)] = std::uniform_int_distribution<int>(0, 9)(rng);
        }
        for(int v = 0; v < N; ++v) predictions[std::size_t(r) * N + std::size_t(v)] = std::uniform_int_distribution<int>(0, 9)(rng);
        budgets[std::size_t(r)] = 1 + r % N;
        control[std::size_t(r) * dr::control::fields + dr::control::frontier] = 50 + r;
        control[std::size_t(r) * dr::control::fields + dr::control::last_token] = 3;
        control[std::size_t(r) * dr::control::fields + dr::control::output_budget] = budgets[std::size_t(r)];
    }
    Device<std::int32_t> dtok(tokens), dpar(parents), dpred(predictions), dctl(control), dres(std::size_t(R) * dr::result_width(K)),
        dpath(std::size_t(R) * N), dstops(stops);
    dr::AcceptRows rows{dtok.p, dpred.p, dstops.p, 1, dctl.p, dres.p, R, K, nullptr};
    ok(dt::enqueue_tree_accept(rows, dpar.p, dpath.p, nullptr), "tree accept");
    const auto res = dres.get(), path = dpath.get(), ctl = dctl.get();
    int accepted = 0;
    for(int r = 0; r < R; ++r) {
        const std::vector<int> t(tokens.begin() + std::ptrdiff_t(std::size_t(r) * K), tokens.begin() + std::ptrdiff_t(std::size_t(r + 1) * K));
        const std::vector<int> p(parents.begin() + std::ptrdiff_t(std::size_t(r) * K), parents.begin() + std::ptrdiff_t(std::size_t(r + 1) * K));
        const std::vector<int> pr(predictions.begin() + std::ptrdiff_t(std::size_t(r) * N), predictions.begin() + std::ptrdiff_t(std::size_t(r + 1) * N));
        const auto w = host_walk(t, p, pr, budgets[std::size_t(r)], {5});
        const auto* got = res.data() + std::size_t(r) * dr::result_width(K);
        require(got[0] == int(w.out.size()), "emitted count differs");
        for(std::size_t i = 0; i < w.out.size(); ++i) require(got[1 + i] == w.out[i], "emitted token differs");
        for(std::size_t i = 0; i < w.path.size(); ++i) require(path[std::size_t(r) * N + i] == w.path[i], "path differs");
        const auto* c = ctl.data() + std::size_t(r) * dr::control::fields;
        require(c[dr::control::n_emitted] == int(w.out.size()) && c[dr::control::n_matched] == int(w.path.size()) - 1, "control counts");
        require(c[dr::control::frontier] == 50 + r + int(w.out.size()) && c[dr::control::base_frontier] == 50 + r, "frontier");
        require(c[dr::control::last_token] == w.out.back() && c[dr::control::base_last] == 3, "pending token");
        accepted += int(w.path.size()) - 1;
    }
    // Chains: the tree walk equals the greedy accept kernel.
    std::vector<std::int32_t> chain(parents.size());
    for(int r = 0; r < R; ++r)
        for(int i = 1; i < N; ++i) chain[std::size_t(r) * K + std::size_t(i - 1)] = i - 1;
    std::vector<std::int32_t> follow = predictions;
    for(int r = 0; r < R; ++r)   // predictions that match the chain for a random prefix
        for(int i = 0; i < N - 1 && i < r % N; ++i) follow[std::size_t(r) * N + std::size_t(i)] = tokens[std::size_t(r) * K + std::size_t(i)];
    Device<std::int32_t> dchain(chain), dfollow(follow), c1(control), c2(control), r1(dres.n), r2(dres.n);
    dr::AcceptRows a1{dtok.p, dfollow.p, dstops.p, 1, c1.p, r1.p, R, K, nullptr}, a2 = a1;
    a2.control = c2.p;
    a2.result = r2.p;
    ok(dt::enqueue_tree_accept(a1, dchain.p, dpath.p, nullptr), "chain tree accept");
    ok(dr::enqueue_greedy_accept(a2, nullptr), "greedy accept");
    require(r1.get() == r2.get() && c1.get() == c2.get(), "the tree walk on a chain differs from greedy accept");
    std::cout << "accept: " << R << " random trees equal the host walk (" << accepted << " accepted nodes), chains equal greedy accept\n";
}

void case_compact() {
    constexpr int R = 3, N = 8, row_bytes = 48, rows_per_request = 64;
    // Each row holds its own slot index; the path keeps nodes 0, 2, 5, 6 and the round emitted 4 tokens.
    std::vector<std::int32_t> control(std::size_t(R) * dr::control::fields, 0), path(std::size_t(R) * N, -1);
    const std::vector<int> kept = {0, 2, 5, 6};
    for(int r = 0; r < R; ++r) {
        control[std::size_t(r) * dr::control::fields + dr::control::base_frontier] = 10 + r;
        control[std::size_t(r) * dr::control::fields + dr::control::n_emitted] = 4;
        for(std::size_t i = 0; i < kept.size(); ++i) path[std::size_t(r) * N + i] = kept[i];
    }
    std::vector<unsigned char> bytes(std::size_t(rows_per_request) * row_bytes);
    for(int s = 0; s < rows_per_request; ++s) std::fill_n(bytes.begin() + std::ptrdiff_t(std::size_t(s) * row_bytes), row_bytes, static_cast<unsigned char>(s));
    // Request r's slots: contiguous from base_frontier, or (slot-mapped array) node i at 40 - i.
    std::vector<std::int32_t> slots(std::size_t(R) * N);
    for(int r = 0; r < R; ++r) for(int i = 0; i < N; ++i) slots[std::size_t(r) * N + std::size_t(i)] = 40 - i;
    Device<unsigned char> contiguous(bytes), mapped(bytes);
    Device<std::int32_t> dctl(control), dpath(path), dslots(slots);
    const std::vector<dt::RowArray> arrays = {{contiguous.p, row_bytes, row_bytes, nullptr}, {mapped.p, row_bytes, row_bytes, dslots.p}};
    Device<dt::RowArray> darrays(arrays);
    // One request only, so the contiguous rows of different requests do not overlap.
    ok(dt::enqueue_compact(darrays.p, 2, dctl.p, dpath.p, N, 1, nullptr), "compact");
    ok(dt::enqueue_settle(dctl.p, dpath.p, N, 1, nullptr), "settle");
    auto check = [&](const char* when) {
        const auto a = contiguous.get(), b = mapped.get();
        for(std::size_t i = 0; i < kept.size(); ++i) {
            require(a[std::size_t(10 + int(i)) * row_bytes] == static_cast<unsigned char>(10 + kept[i]), std::string("contiguous compaction ") + when);
            require(b[std::size_t(40 - int(i)) * row_bytes] == static_cast<unsigned char>(40 - kept[i]), std::string("slot-mapped compaction ") + when);
        }
    };
    check("places the kept rows");
    require(dpath.get()[3] == 3, "settle sets the kept path to identity");
    // Core trims to 2 tokens: rerunning commit_state moves nothing.
    std::vector<std::int32_t> trimmed = dctl.get();
    trimmed[dr::control::n_emitted] = 2;
    dctl.put(trimmed);
    ok(dt::enqueue_compact(darrays.p, 2, dctl.p, dpath.p, N, 1, nullptr), "compact again");
    ok(dt::enqueue_settle(dctl.p, dpath.p, N, 1, nullptr), "settle again");
    check("after a rerun");
    require(dt::enqueue_compact(darrays.p, 2, dctl.p, dpath.p, 65, 1, nullptr) == cudaErrorInvalidValue, "too many nodes");
    std::cout << "compact: kept rows placed in order, contiguous and slot-mapped, idempotent after a trim\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string which = argc > 1 ? argv[1] : "all";
        if(which == "build" || which == "all") case_build();
        if(which == "accept" || which == "all") case_accept();
        if(which == "compact" || which == "all") case_compact();
        ok(cudaDeviceSynchronize(), "sync");
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
