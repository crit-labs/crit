// Device expert cache on CUDA against a host replica of the same clock directory:
// identical slots, hits/misses and pool use over a long random routing sequence,
// complete bundle contents after the fills, and rejected routes and launches; then the ExpertCache runtime with a
// residency-profile preload and admission limits (a quota table by miss count and second-touch) against the replica.
#include "device_cache.h"
#include "runtime.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace dc = crit::techniques::device_cache;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

// Host replica (see device_cache.py).
struct Replica {
    int slots, experts, pool_slots, hand = 0, stamp = 0;
    std::map<int, int> key_slot;
    std::vector<int> slot_key, slot_ref, slot_pin;
    std::vector<int> tags[2];
    // admission limits: quota by miss count (empty: every miss admitted) and second-touch
    std::vector<int> table;
    bool second_touch = false;
    std::set<int> seen;
    static constexpr int kBypass = -2;  // a bypassed miss's slot
    void preload(const std::vector<int>& keys) {
        int n = 0;
        for(int k : keys) {
            if(n == slots) break;
            if(key_slot.count(k)) continue;
            if(slot_key[n] >= 0) key_slot.erase(slot_key[n]);
            slot_key[n] = k;
            key_slot[k] = n;
            slot_ref[n] = 0;
            ++n;
        }
        hand = n % slots;
    }
    Replica(int s, int e, int p) : slots(s), experts(e), pool_slots(p), slot_key(s, -1), slot_ref(s, 0), slot_pin(s, -1) {
        tags[0].assign(p, -1);
        tags[1].assign(p, -1);
    }
    int victim() {
        for(int i = 0; i < 4 * slots; ++i) {
            const int c = hand;
            hand = (hand + 1) % slots;
            if(slot_pin[c] == stamp) continue;
            if(slot_ref[c]) { slot_ref[c] = 0; continue; }
            return c;
        }
        throw std::runtime_error("replica: no victim");
    }
    // returns slots per route; hits, misses, pooled
    std::vector<int> plan(int group, const std::vector<int>& ex, int& hits, int& misses, int& pooled) {
        int bypassed = 0;
        return plan(group, ex, hits, misses, pooled, bypassed);
    }
    std::vector<int> plan(int group, const std::vector<int>& ex, int& hits, int& misses, int& pooled, int& bypassed) {
        ++stamp;
        std::vector<int> out(ex.size(), -1);
        hits = misses = pooled = bypassed = 0;
        for(std::size_t i = 0; i < ex.size(); ++i) {
            auto it = key_slot.find(group * experts + ex[i]);
            if(it != key_slot.end()) { slot_ref[it->second] = 1; slot_pin[it->second] = stamp; out[i] = it->second; ++hits; }
        }
        const int quota = table.empty() ? -1 : table[std::count(out.begin(), out.end(), -1)];
        int admitted = 0;
        for(std::size_t i = 0; i < ex.size(); ++i) {
            if(out[i] >= 0) continue;
            ++misses;
            const int k = group * experts + ex[i];
            bool admit = quota < 0 || admitted < quota;
            if(second_touch && !seen.count(k)) { seen.insert(k); admit = false; }
            if(!admit) { out[i] = kBypass; ++bypassed; continue; }
            ++admitted;
            const int s = victim();
            if(slot_key[s] >= 0) key_slot.erase(slot_key[s]);
            slot_key[s] = k;
            key_slot[k] = s;
            slot_ref[s] = 1;
            slot_pin[s] = stamp;
            for(int t : tags[group & 1]) if(t == k) { ++pooled; break; }
            out[i] = s;
        }
        return out;
    }
    // pool slots go to nonresident, distinct predictions in order
    int prefetch(int group, const std::vector<int>& pred) {
        auto& t = tags[group & 1];
        t.assign(pool_slots, -1);
        int n = 0;
        for(std::size_t i = 0; i < pred.size() && n < pool_slots; ++i) {
            const int k = group * experts + pred[i];
            if(key_slot.count(k) || std::count(t.begin(), t.begin() + n, k)) continue;
            t[n++] = k;
        }
        return n;
    }
};

template<class T> T* dev(std::size_t n) { void* p = nullptr; ok(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc"); return static_cast<T*>(p); }
template<class T> std::vector<T> get(const T* p, std::size_t n) {
    std::vector<T> v(n);
    ok(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
template<class T> void put(T* p, const std::vector<T>& v) { ok(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
std::uint8_t pattern(int key, std::size_t i) { return static_cast<std::uint8_t>((key * 131 + i * 7 + (i >> 9)) & 255); }
} // namespace

// One random routing sequence on a directory of S slots (S < 32 exercises the narrow clock window).
void run(const int S) {
    constexpr int G = 3, E = 40, P = 4;
    constexpr std::size_t B = 4096;
    std::uint8_t* host = nullptr;
    ok(cudaHostAlloc(reinterpret_cast<void**>(&host), std::size_t(G) * E * B, cudaHostAllocMapped), "host store");
    for(int k = 0; k < G * E; ++k)
        for(std::size_t i = 0; i < B; ++i) host[std::size_t(k) * B + i] = pattern(k, i);
    void* store = nullptr;
    ok(cudaHostGetDevicePointer(&store, host, 0), "mapped pointer");

    dc::Directory d;
    d.key_slot = dev<std::int32_t>(G * E);
    d.slot_key = dev<std::int32_t>(S);
    d.slot_ref = dev<std::uint8_t>(S);
    d.slot_pin = dev<std::int32_t>(S);
    d.state = dev<std::int32_t>(4);
    d.counters = dev<unsigned long long>(4);
    d.slots_base = dev<std::uint8_t>(S * B);
    d.store = static_cast<const std::uint8_t*>(store);
    d.bundle_bytes = B;
    d.slots = S;
    d.groups = G;
    d.experts = E;
    d.pool_slots = P;
    d.pool_base = dev<std::uint8_t>(2 * P * B);
    d.pool_tags = dev<std::int32_t>(2 * P);
    put(d.key_slot, std::vector<std::int32_t>(G * E, -1));
    put(d.slot_key, std::vector<std::int32_t>(S, -1));
    put(d.slot_ref, std::vector<std::uint8_t>(S, 0));
    put(d.slot_pin, std::vector<std::int32_t>(S, -1));
    put(d.state, std::vector<std::int32_t>(4, 0));
    put(d.counters, std::vector<unsigned long long>(4, 0));
    put(d.pool_tags, std::vector<std::int32_t>(2 * P, -1));

    auto* ids = dev<std::int32_t>(dc::kMaxRoutes);
    auto* w = dev<float>(dc::kMaxRoutes);
    auto* uses = dev<dc::Use>(dc::kMaxRoutes);
    auto* fills = dev<dc::Fill>(dc::kMaxRoutes);
    auto* pf_fills = dev<dc::Fill>(P);
    auto* count = dev<int>(1);
    auto* pf_count = dev<int>(1);

    Replica ref(S, E, P);
    std::mt19937 rng(7);
    std::vector<double> popularity;
    for(int e = 0; e < E; ++e) popularity.push_back(1.0 / (1 + e));
    std::discrete_distribution<int> popular(popularity.begin(), popularity.end());
    unsigned long long hits = 0, misses = 0, pooled = 0;
    for(int step = 0; step < 600; ++step) {
        const int group = step % G;
        std::set<int> chosen;
        const int k = 3 + static_cast<int>(rng() % 6);
        while(static_cast<int>(chosen.size()) < k) chosen.insert(popular(rng));
        std::vector<int> ex(chosen.begin(), chosen.end());
        std::shuffle(ex.begin(), ex.end(), rng);
        if(step % 3 == 0) {  // lookahead for this group: part right, part guessed, more than the pool
            std::vector<int> pred(ex.begin(), ex.begin() + std::min<int>(2, k));
            while(static_cast<int>(pred.size()) < P + 2) pred.push_back(static_cast<int>(rng() % E));
            put(ids, std::vector<std::int32_t>(pred.begin(), pred.end()));
            ok(dc::enqueue_prefetch_plan(d, group, ids, static_cast<int>(pred.size()), pf_fills, pf_count, nullptr), "prefetch plan");
            ok(dc::enqueue_fill(pf_fills, pf_count, P, B, 4, nullptr), "prefetch fill");
            require(get(pf_count, 1)[0] == ref.prefetch(group, pred), "prefetch count");
            const auto t = get(d.pool_tags + (group & 1) * P, P);
            require(std::equal(t.begin(), t.end(), ref.tags[group & 1].begin()), "pool tags differ from the replica");
        }
        put(ids, std::vector<std::int32_t>(ex.begin(), ex.end()));
        put(w, std::vector<float>(k, 0.5f));
        ok(dc::enqueue_plan(d, group, ids, w, k, uses, fills, count, nullptr), "plan");
        ok(dc::enqueue_fill(fills, count, k, B, 8, nullptr), "fill");
        ok(cudaDeviceSynchronize(), "sync");
        int h = 0, m = 0, pl = 0;
        const auto want = ref.plan(group, ex, h, m, pl);
        hits += h;
        misses += m;
        pooled += pl;
        const auto got = get(uses, k);
        require(get(count, 1)[0] == m, "fill count");
        const auto slots = get(d.slots_base, S * B);
        for(int i = 0; i < k; ++i) {
            require(got[i].key == group * E + ex[i], "use key");
            require(got[i].bundle == d.slots_base + std::size_t(want[i]) * B, "slot differs from the replica");
            require(got[i].weight == 0.5f, "weight");
            for(std::size_t b = 0; b < B; b += 509)
                require(slots[std::size_t(want[i]) * B + b] == pattern(got[i].key, b), "bundle bytes");
        }
    }
    const auto c = get(d.counters, 4);
    require(c[0] == hits && c[1] == misses && c[2] == pooled, "counters differ from the replica");
    require(pooled > 0 && hits > 0 && misses > 0, "sequence exercised hits, misses and pool use");
    require(get(d.state, 4)[2] == 0, "no failure on valid routes");

    // a resident prediction takes no pool slot: [resident, P nonresident] fills the whole pool
    {
        const int group = 1;
        std::vector<int> pred;
        for(int e = 0; e < E && pred.empty(); ++e) if(ref.key_slot.count(group * E + e)) pred.push_back(e);
        for(int e = 0; e < E && static_cast<int>(pred.size()) < P + 1; ++e) if(!ref.key_slot.count(group * E + e)) pred.push_back(e);
        require(static_cast<int>(pred.size()) == P + 1, "resident and nonresident experts available");
        put(ids, std::vector<std::int32_t>(pred.begin(), pred.end()));
        ok(dc::enqueue_prefetch_plan(d, group, ids, P + 1, pf_fills, pf_count, nullptr), "resident-first prefetch plan");
        ok(dc::enqueue_fill(pf_fills, pf_count, P, B, 4, nullptr), "resident-first prefetch fill");
        ok(cudaDeviceSynchronize(), "sync");
        require(get(pf_count, 1)[0] == P && ref.prefetch(group, pred) == P, "every pool slot copied");
        const auto t = get(d.pool_tags + (group & 1) * P, P);
        const auto pool = get(d.pool_base + std::size_t(group & 1) * P * B, std::size_t(P) * B);
        for(int p = 0; p < P; ++p) {
            require(t[p] == group * E + pred[p + 1], "pool slot holds the next nonresident prediction");
            for(std::size_t b = 0; b < B; b += 509) require(pool[std::size_t(p) * B + b] == pattern(t[p], b), "pool bytes");
        }
    }

    // a duplicate route is rejected, not admitted twice
    put(ids, std::vector<std::int32_t>{1, 2, 1});
    ok(dc::enqueue_plan(d, 0, ids, nullptr, 3, uses, fills, count, nullptr), "duplicate plan");
    ok(cudaDeviceSynchronize(), "sync");
    require(get(d.state, 4)[2] == 1, "duplicate route flagged");
    require(get(uses, 3)[2].key == -1, "duplicate use rejected");
    require(dc::enqueue_plan(d, 0, ids, nullptr, dc::kMaxRoutes + 1, uses, fills, count, nullptr) == cudaErrorInvalidValue,
            "too many routes rejected");
    dc::Directory bad = d;
    bad.bundle_bytes = 100;
    require(dc::enqueue_plan(bad, 0, ids, nullptr, 1, uses, fills, count, nullptr) == cudaErrorInvalidValue,
            "unaligned bundle rejected");
    std::cout << "device expert cache, " << S << " slots: " << hits << " hits, " << misses << " misses, " << pooled
              << " from the pool; matches replica\n";
    cudaFreeHost(host);
}

// The ExpertCache runtime: preload from a profile, then plans under a quota table and second-touch, against the
// replica: slots, bypassed misses naming the store bundle, copied bytes and the telemetry counters.
void run_runtime(const int S) {
    constexpr int G = 2, E = 48;
    constexpr std::size_t B = 2048;
    std::uint8_t* host = nullptr;
    ok(cudaHostAlloc(reinterpret_cast<void**>(&host), std::size_t(G) * E * B, cudaHostAllocMapped), "host store");
    for(int k = 0; k < G * E; ++k)
        for(std::size_t i = 0; i < B; ++i) host[std::size_t(k) * B + i] = pattern(k, i);
    void* store = nullptr;
    ok(cudaHostGetDevicePointer(&store, host, 0), "mapped pointer");
    auto* slots_base = dev<std::uint8_t>(S * B);
    dc::ExpertCache cache;
    ok(cache.init(dc::Geometry{S, G, E, 0, B}, slots_base, static_cast<const std::uint8_t*>(store), nullptr, nullptr), "init");

    Replica ref(S, E, 0);
    std::vector<std::int32_t> profile;  // every third key, then a duplicate and an invalid key the preload skips
    for(int k = 0; k < G * E && static_cast<int>(profile.size()) < S - 2; k += 3) profile.push_back(k);
    profile.push_back(profile[0]);
    profile.push_back(G * E + 5);
    ok(cache.preload(profile.data(), static_cast<int>(profile.size()), 4, nullptr), "preload");
    ref.preload(std::vector<int>(profile.begin(), profile.end() - 1));
    {
        const auto keys = get(cache.directory().slot_key, S);
        const auto bytes = get(slots_base, S * B);
        for(int s = 0; s < S; ++s) {
            require(keys[s] == ref.slot_key[s], "preloaded slot differs from the replica");
            if(keys[s] >= 0)
                for(std::size_t b = 0; b < B; b += 251) require(bytes[std::size_t(s) * B + b] == pattern(keys[s], b), "preloaded bytes");
        }
    }

    std::vector<std::int32_t> table(dc::kMaxRoutes + 1);
    for(int m = 0; m <= dc::kMaxRoutes; ++m) table[m] = m < 2 ? -1 : m / 2;  // half the misses take the link
    ok(cache.set_admission(-1, table.data(), true, nullptr), "admission");
    ref.table.assign(table.begin(), table.end());
    ref.second_touch = true;

    auto* ids = dev<std::int32_t>(dc::kMaxRoutes);
    std::mt19937 rng(11);
    unsigned long long hits = 0, misses = 0, bypassed = 0;
    for(int step = 0; step < 400; ++step) {
        const int group = step % G;
        std::set<int> chosen;
        const int k = 2 + static_cast<int>(rng() % 7);
        while(static_cast<int>(chosen.size()) < k) chosen.insert(static_cast<int>(rng() % 20) * (step % 5 == 0 ? 2 : 1));
        std::vector<int> ex(chosen.begin(), chosen.end());
        std::shuffle(ex.begin(), ex.end(), rng);
        put(ids, std::vector<std::int32_t>(ex.begin(), ex.end()));
        const dc::Use* uses = nullptr;
        ok(cache.plan_and_fetch(group, ids, nullptr, k, 4, nullptr, &uses), "plan and fetch");
        ok(cudaDeviceSynchronize(), "sync");
        int h = 0, m = 0, pl = 0, by = 0;
        const auto want = ref.plan(group, ex, h, m, pl, by);
        hits += h;
        misses += m;
        bypassed += by;
        const auto got = get(uses, k);
        const auto bytes = get(slots_base, S * B);
        for(int i = 0; i < k; ++i) {
            const int key = group * E + ex[i];
            require(got[i].key == key, "use key");
            if(want[i] == Replica::kBypass) {
                require(got[i].source == dc::Source::store, "bypassed miss names the store");
                require(got[i].bundle == static_cast<const std::uint8_t*>(store) + std::size_t(key) * B, "store bundle");
            } else {
                require(got[i].source == dc::Source::slot && got[i].bundle == slots_base + std::size_t(want[i]) * B,
                        "slot differs from the replica");
                for(std::size_t b = 0; b < B; b += 251) require(bytes[std::size_t(want[i]) * B + b] == pattern(key, b), "bundle bytes");
            }
        }
    }
    dc::Telemetry t;
    ok(cache.telemetry(t, nullptr), "telemetry");
    require(t.hits == hits && t.misses == misses && t.bypassed == bypassed && t.failure == 0, "telemetry differs from the replica");
    require(t.expert_reads() == hits + misses && t.expert_hits() == hits, "telemetry names");
    require(hits > 0 && bypassed > 0 && bypassed < misses, "sequence exercised hits, admissions and bypasses");
    require(cache.set_admission(-2, nullptr, false, nullptr) == cudaErrorInvalidValue, "invalid quota rejected");
    {   // a provider grouped GEMM's operands from one plan's Uses; bypassed routes left to the CPU
        put(ids, std::vector<std::int32_t>{30, 31, 32});
        const dc::Use* uses = nullptr;
        ok(cache.plan_and_fetch(1, ids, nullptr, 3, 4, nullptr, &uses), "operand plan");
        auto* x = dev<std::uint8_t>(3 * 64);
        auto* y = dev<std::uint8_t>(3 * 32);
        auto* token = dev<std::int32_t>(3);
        put(token, std::vector<std::int32_t>{2, 0, 1});
        dc::Operands out;
        out.matrix[0] = dev<const void*>(dc::kMaxRoutes);
        out.matrix[1] = dev<const void*>(dc::kMaxRoutes);
        out.x = dev<const void*>(dc::kMaxRoutes);
        out.y = dev<void*>(dc::kMaxRoutes);
        out.weight = dev<float>(dc::kMaxRoutes);
        dc::OperandSpec spec;
        spec.components = 2;
        spec.offsets[1] = B / 2;
        spec.x = x;
        spec.x_row_bytes = 64;
        spec.token = token;
        spec.y = y;
        spec.y_row_bytes = 32;
        spec.skip_bypassed = true;
        ok(dc::enqueue_operands(uses, 3, spec, out, nullptr), "operands");
        ok(cudaDeviceSynchronize(), "sync");
        const auto got = get(uses, 3);
        const auto m0 = get(out.matrix[0], 3), m1 = get(out.matrix[1], 3), xs = get(out.x, 3);
        const auto ys = get(out.y, 3);
        const auto ws = get(out.weight, 3);
        const int rows[3] = {2, 0, 1};
        for(int r = 0; r < 3; ++r) {
            if(got[r].source == dc::Source::store) {
                require(!m0[r] && !m1[r] && !xs[r] && !ys[r] && ws[r] == 0.f, "a bypassed route is left to the CPU");
                continue;
            }
            require(m0[r] == got[r].bundle && m1[r] == got[r].bundle + B / 2, "component pointers inside the bundle");
            require(xs[r] == x + rows[r] * 64 && ys[r] == y + r * 32 && ws[r] == 1.f, "activation, output and weight");
        }
        require(dc::enqueue_operands(uses, 3, dc::OperandSpec{}, out, nullptr) == cudaErrorInvalidValue, "no components rejected");
        for(void* p : {(void*)x, (void*)y, (void*)token, (void*)out.matrix[0], (void*)out.matrix[1], (void*)out.x, (void*)out.y,
                       (void*)out.weight})
            cudaFree(p);
    }
    std::cout << "expert cache runtime, " << S << " slots: " << hits << " hits, " << misses << " misses, " << bypassed
              << " bypassed; matches replica\n";
    cache.release();
    cudaFree(slots_base);
    cudaFree(ids);
    cudaFreeHost(host);
}

int main() {
    try {
        run(24);
        run(72);
        run_runtime(20);
        run_runtime(40);
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
