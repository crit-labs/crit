// Device expert cache on CUDA against a host replica of the same clock directory:
// identical slots, hits/misses and pool use over a long random routing sequence,
// complete bundle contents after the fills, and rejected routes and launches.
#include "device_cache.h"

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
        ++stamp;
        std::vector<int> out(ex.size(), -1);
        hits = misses = pooled = 0;
        for(std::size_t i = 0; i < ex.size(); ++i) {
            auto it = key_slot.find(group * experts + ex[i]);
            if(it != key_slot.end()) { slot_ref[it->second] = 1; slot_pin[it->second] = stamp; out[i] = it->second; ++hits; }
        }
        for(std::size_t i = 0; i < ex.size(); ++i) {
            if(out[i] >= 0) continue;
            ++misses;
            const int k = group * experts + ex[i];
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
    int prefetch(int group, const std::vector<int>& pred) {
        auto& t = tags[group & 1];
        int n = 0;
        for(int p = 0; p < pool_slots; ++p) {
            t[p] = -1;
            if(p >= static_cast<int>(pred.size())) continue;
            const int k = group * experts + pred[p];
            if(key_slot.count(k)) continue;
            bool repeated = false;
            for(int q = 0; q < p; ++q) repeated |= t[q] == k;
            if(repeated) continue;
            t[p] = k;
            ++n;
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
        if(step % 3 == 0) {  // lookahead for this group: part right, part guessed
            std::vector<int> pred(ex.begin(), ex.begin() + std::min<int>(2, k));
            while(static_cast<int>(pred.size()) < P) pred.push_back(static_cast<int>(rng() % E));
            put(ids, std::vector<std::int32_t>(pred.begin(), pred.end()));
            ok(dc::enqueue_prefetch_plan(d, group, ids, P, pf_fills, pf_count, nullptr), "prefetch plan");
            ok(dc::enqueue_fill(pf_fills, pf_count, P, B, 4, nullptr), "prefetch fill");
            require(get(pf_count, 1)[0] == ref.prefetch(group, pred), "prefetch count");
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

int main() {
    try {
        run(24);
        run(72);
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
