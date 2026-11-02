// The native metadata table and its device replica (concurrent-request-execution): stable slots with the lowest free
// slot reused, unchanged payloads changing nothing, deltas of changed rows only, the device table equal to the host
// rows after every step over a random mutation sequence, the gather order uploaded only when it changed, staging
// halves reused only after their copies finished, and refused patches.
#include "device_metadata.h"

#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace md = crit::techniques::metadata;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
std::vector<std::byte> payload(std::uint64_t seed, std::size_t n) {
    std::vector<std::byte> out(n);
    for(std::size_t i = 0; i < n; ++i) out[i] = std::byte((seed * 29 + i * 3) & 255);
    return out;
}
}

int main() {
    try {
        constexpr std::uint32_t capacity = 16;
        constexpr std::size_t row_bytes = 32;
        md::MetadataTable table(capacity, row_bytes);
        require(table.set({1, 1}, payload(1, 8)) == 0 && table.set({2, 1}, payload(2, 8)) == 1, "slots in order");
        table.release({1, 1});
        require(table.set({3, 1}, payload(3, 8)) == 0, "the lowest free slot is reused");
        table.release({2, 1});
        table.release({3, 1});

        std::vector<std::byte> device_rows(capacity * row_bytes), pinned(2 * 1024);
        std::vector<std::uint32_t> device_order(capacity, 0xffffffffu);
        std::size_t copies = 0;
        int records = 0, finished_calls = 0;
        md::DeviceReplica replica(device_rows.data(), device_order.data(), capacity, row_bytes, pinned.data(), 1024,
                                  [&](void* dst, const void* src, std::size_t bytes) { std::memcpy(dst, src, bytes); ++copies; },
                                  [&](int) { ++finished_calls; return true; }, [&](int) { ++records; });

        std::mt19937 rng(3);
        md::Snapshot before = table.snapshot();
        std::vector<md::Key> live;
        std::size_t changed_rows = 0, copied_bytes = 0;
        for(int step = 0; step < 300; ++step) {
            const int action = int(rng() % 4);
            if(action == 0 && live.size() < capacity) {
                const md::Key k{rng() % 1000 + 1, 1};
                if(std::find(live.begin(), live.end(), k) == live.end()) {
                    live.push_back(k);
                    table.set(k, payload(rng(), 1 + rng() % row_bytes));
                }
            } else if(action == 1 && !live.empty()) {
                const auto at = rng() % live.size();
                table.release(live[at]);
                live.erase(live.begin() + long(at));
            } else if(action == 2 && !live.empty()) {
                table.set(live[rng() % live.size()], payload(rng(), 1 + rng() % row_bytes));
            }
            std::sort(live.begin(), live.end());
            const md::Snapshot now = table.snapshot(live);
            const md::Delta delta = md::diff(before, now);
            changed_rows += delta.changed.size();
            copied_bytes += replica.apply(delta);
            for(std::size_t i = 0; i < now.rows.size(); ++i) {
                const auto& r = now.rows[i];
                require(device_order[i] == r.slot, "device gather order");
                require(std::memcmp(device_rows.data() + r.slot * row_bytes, r.payload.data(), r.payload.size()) == 0,
                        "device row bytes");
            }
            before = now;
        }
        require(replica.order_uploads_skipped() > 0, "an unchanged order is not uploaded again");
        require(copied_bytes < changed_rows * row_bytes + 300 * capacity * sizeof(std::uint32_t), "only changed rows and orders copied");
        require(records == 300 && finished_calls > 0, "each step records its half and reused halves are checked");

        md::Delta bad;
        bad.changed.push_back({capacity, {9, 9}, 1, payload(1, 4)});
        bool refused = false;
        try { replica.apply(bad); } catch(const std::invalid_argument&) { refused = true; }
        require(refused, "a patch outside the table refused");
        std::cout << "device metadata: " << changed_rows << " changed rows over 300 steps, " << copies << " copies, "
                  << replica.order_uploads_skipped() << " order uploads skipped\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
