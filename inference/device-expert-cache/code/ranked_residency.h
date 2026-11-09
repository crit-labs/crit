// Background ranked residency, the host policy (ranked_residency.py is its twin): the cache adapts off the decode
// step's critical path. Plans count every route (Directory::route_counts) and admit no misses (admission quota 0:
// misses are computed on the CPU tier or read from the store). Every `period` windows, at a fixed window boundary,
// the engine reads the counts, and the policy decays each key's score and adds the routes since the last boundary,
// ranks every key of every layer together and swaps the top keys that are not resident in for the lowest-ranked
// resident ones (ExpertCache::begin_swaps on a side stream, commit_swaps at the next boundary). The placement at every
// window is a function of the route history alone, never of timing, so one request history always gives the same
// placement; with exact tiers (host-expert-arithmetic's integer rows) it changes no token either.
#pragma once
#include "device_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace crit::techniques::device_cache {

struct RankedOptions {
    double decay = 0.85;          // a score's weight per boundary: recent routes outweigh old ones
    int period = 3;               // windows between boundaries
    int max_swaps = 96;           // swaps per boundary
    int burst_swaps = 384;        // swaps per boundary when the ranking has moved (a topic change)
    double burst_fraction = 0.25; // ... that is, when this fraction of the top `slots` keys is not resident
};

class RankedPolicy {
public:
    RankedPolicy(int keys, int slots, RankedOptions o = {}) : o_(o), slots_(slots), scores_(std::size_t(keys), 0.0),
                                                            last_(std::size_t(keys), 0u) {
        if(keys < 1 || slots < 1 || !(o.decay > 0.0 && o.decay <= 1.0) || o.period < 1 || o.max_swaps < 0 ||
           o.burst_swaps < o.max_swaps || !(o.burst_fraction >= 0.0 && o.burst_fraction <= 1.0))
            throw std::invalid_argument("ranked residency: bad geometry or options");
    }
    // Is window w (counted from 0) a boundary? The first boundary is at window `period`.
    bool boundary(long long window) const { return window > 0 && window % o_.period == 0; }

    // At a boundary: the cumulative counts read from the device (unsigned, wrapping), then the swaps to begin, given
    // the slots' keys now (-1 empty, kReserved filling). Ties rank by key, so the result depends on the counts alone.
    std::vector<Swap> decide(const std::uint32_t* counts, const std::int32_t* slot_key) {
        const std::size_t keys = scores_.size();
        for(std::size_t k = 0; k < keys; ++k) {
            const std::uint32_t delta = counts[k] - last_[k];
            last_[k] = counts[k];
            scores_[k] = scores_[k] * o_.decay + double(delta);
        }
        std::vector<std::int32_t> order(keys);
        std::iota(order.begin(), order.end(), 0);
        const std::size_t top = std::min<std::size_t>(std::size_t(slots_), keys);
        std::partial_sort(order.begin(), order.begin() + std::ptrdiff_t(top), order.end(), [&](std::int32_t a, std::int32_t b) {
            return scores_[std::size_t(a)] != scores_[std::size_t(b)] ? scores_[std::size_t(a)] > scores_[std::size_t(b)] : a < b;
        });
        std::vector<char> resident(keys, 0), wanted(keys, 0);
        for(int s = 0; s < slots_; ++s)
            if(slot_key[s] >= 0) resident[std::size_t(slot_key[s])] = 1;
        std::vector<std::int32_t> admits;
        for(std::size_t i = 0; i < top; ++i) {
            const auto k = order[i];
            if(scores_[std::size_t(k)] <= 0.0) break;
            wanted[std::size_t(k)] = 1;
            if(!resident[std::size_t(k)]) admits.push_back(k);
        }
        // Victims: empty slots first (in slot order), then resident keys outside the top by score, lowest first.
        std::vector<std::int32_t> empty, losing;
        for(int s = 0; s < slots_; ++s) {
            if(slot_key[s] == -1) empty.push_back(s);
            else if(slot_key[s] >= 0 && !wanted[std::size_t(slot_key[s])]) losing.push_back(s);
        }
        std::sort(losing.begin(), losing.end(), [&](std::int32_t a, std::int32_t b) {
            const double sa = scores_[std::size_t(slot_key[a])], sb = scores_[std::size_t(slot_key[b])];
            return sa != sb ? sa < sb : slot_key[a] < slot_key[b];
        });
        const bool burst = double(admits.size()) > o_.burst_fraction * double(slots_);
        const std::size_t limit = std::size_t(burst ? o_.burst_swaps : o_.max_swaps);
        std::vector<Swap> out;
        std::size_t e = 0, l = 0;
        for(const auto k : admits) {
            if(out.size() == limit) break;
            if(e < empty.size()) out.push_back({empty[e++], k});
            else if(l < losing.size() && scores_[std::size_t(slot_key[losing[l]])] < scores_[std::size_t(k)])
                out.push_back({losing[l++], k});
            else break;
        }
        bursts_ += burst ? 1 : 0;
        return out;
    }
    double score(std::int32_t key) const { return scores_.at(std::size_t(key)); }
    long long bursts() const { return bursts_; }
    const RankedOptions& options() const { return o_; }

private:
    RankedOptions o_;
    int slots_;
    std::vector<double> scores_;
    std::vector<std::uint32_t> last_;
    long long bursts_ = 0;
};

} // namespace crit::techniques::device_cache
