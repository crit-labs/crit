// The native host tier and read queue (ssd-expert-materialization): hits, least-recently-used eviction, protected
// experts and refusal, fills that are not served until committed or are abandoned; then the read queue filling tier
// slots from a file with several readers, demand before read-ahead, one read per pending key, and a read past the end.
#include "host_tier.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ht = crit::techniques::host_tier;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
std::byte value(std::uint64_t key, std::size_t i) { return std::byte((key * 13 + i * 5 + 1) & 255); }
}

int main() {
    try {
        constexpr std::size_t bundle = 256, slots = 3;
        std::vector<std::byte> pinned(bundle * slots);
        ht::HostTier tier(pinned.data(), bundle, slots);
        require(!tier.find(7), "an empty tier misses");
        auto a = tier.admit(7);
        require(a && a->second.data() == pinned.data(), "the first free slot");
        require(!tier.find(7), "a filling slot is not served");
        require(!tier.admit(7), "a key filling is not admitted twice");
        tier.commit(a->first);
        require(tier.find(7) && tier.find(7)->data() == pinned.data(), "a committed slot is served");
        tier.commit(tier.admit(8)->first);
        tier.commit(tier.admit(9)->first);
        tier.find(7);                                   // 8 is now the least recently used
        auto d = tier.admit(10);
        require(d && !tier.holds(8) && tier.holds(7), "the least recently used expert leaves");
        tier.abandon(d->first);
        require(!tier.holds(10), "an abandoned fill frees its slot");
        auto e = tier.admit(11, {7, 9});
        require(e && e->first == d->first, "the freed slot is reused before an eviction");
        tier.commit(e->first);
        require(!tier.admit(12, {7, 9, 11}), "every slot protected: refused");
        const auto c = tier.counters();
        require(c.evictions == 1 && c.refusals == 1 && c.hits >= 3, "counters");

        const auto path = std::filesystem::temp_directory_path() / "crit_host_tier_test.bin";
        constexpr std::uint64_t keys = 64;
        {
            std::ofstream out(path, std::ios::binary);
            for(std::uint64_t k = 0; k < keys; ++k)
                for(std::size_t i = 0; i < bundle; ++i) out.put(char(value(k, i)));
        }
        {
            auto file = std::make_shared<crit::core::ReadOnlyFile>(crit::core::ReadOnlyFile::open(path));
            std::vector<std::byte> staging(bundle * keys);
            ht::ReadQueue queue(file, 4);
            std::vector<ht::Ticket> tickets;
            for(std::uint64_t k = 0; k < keys; ++k)
                tickets.push_back(queue.submit({k, k * bundle, {staging.data() + k * bundle, bundle}}, k % 4 == 0));
            const auto again = queue.submit({5, 5 * bundle, {staging.data() + 5 * bundle, bundle}}, true);
            again.wait();
            for(auto& t : tickets) t.wait();
            for(std::uint64_t k = 0; k < keys; ++k)
                for(std::size_t i = 0; i < bundle; i += 17)
                    require(staging[k * bundle + i] == value(k, i), "read bytes");
            require(queue.queued() == 0, "every read served");
            const auto past = queue.submit({99, keys * bundle, {staging.data(), bundle}}, true);
            bool failed = false;
            try { past.wait(); } catch(const std::exception&) { failed = true; }
            require(failed, "a read past the end reports its error");
        }
        std::filesystem::remove(path);
        std::cout << "host tier: " << c.hits << " hits, " << c.evictions << " eviction, " << c.refusals
                  << " refusal; read queue served " << keys << " reads with 4 readers\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
