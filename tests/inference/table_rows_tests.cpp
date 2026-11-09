// Native lookup-table rows: request order, rows asked for twice, the hot-row cache and its counters, workers against a
// single reader, a table past the file's end, rows outside the table, the staging ring waiting on its copies, and the
// prefetcher: requests started one after another (as draft tokens appear) arrive in order with their own rows, queued
// ones share a batch, and a wait blocks only until its own rows are written.
#include "table_rows.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tr = crit::techniques::table_rows;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
std::byte value(std::uint64_t row, std::size_t i) { return std::byte((row * 31 + i * 7 + 3) & 255); }
}

int main() {
    try {
        const auto path = std::filesystem::temp_directory_path() / "crit_table_rows_test.bin";
        constexpr std::uint64_t rows = 500, offset = 24;
        constexpr std::size_t row_bytes = 40;
        {
            std::ofstream out(path, std::ios::binary);
            std::vector<char> head(offset, 'H');
            out.write(head.data(), std::streamsize(head.size()));
            for(std::uint64_t r = 0; r < rows; ++r)
                for(std::size_t i = 0; i < row_bytes; ++i) out.put(char(value(r, i)));
        }
        const tr::Region region{path.string(), offset, rows, row_bytes};
        std::mt19937 rng(5);
        for(const int workers : {1, 4}) {
            tr::RowGather table(region, workers, 16);
            for(int round = 0; round < 50; ++round) {
                std::vector<std::uint64_t> ids;
                for(int i = 0; i < 12; ++i) ids.push_back(rng() % 40);   // repeats within and across rounds
                std::vector<std::byte> got(ids.size() * row_bytes);
                table.gather(ids, got.data());
                for(std::size_t i = 0; i < ids.size(); ++i)
                    for(std::size_t b = 0; b < row_bytes; ++b)
                        require(got[i * row_bytes + b] == value(ids[i], b), "row bytes in request order");
            }
            const auto c = table.counters();
            require(c.hits > 0 && c.reads > 0 && c.bytes_read == c.reads * row_bytes, "cache hits and reads counted");
            std::vector<std::byte> one(row_bytes);
            const std::uint64_t outside[] = {rows};
            bool refused = false;
            try { table.gather(outside, one.data()); } catch(const std::out_of_range&) { refused = true; }
            require(refused, "row outside the table refused");
            std::cout << "table rows, " << workers << " worker(s): " << c.reads << " reads, " << c.hits << " hits, advice "
                      << table.advice() << "\n";
        }
        bool short_file = false;
        try { tr::RowGather(tr::Region{path.string(), offset, rows + 1, row_bytes}); } catch(const std::invalid_argument&) {
            short_file = true;
        }
        require(short_file, "a table past the file's end refused");

        std::vector<std::byte> pinned(3 * 64);
        std::vector<int> polls(3, 0);
        tr::StagingRing ring(pinned.data(), 64, 3, [&](int slot) { return ++polls[std::size_t(slot)] > 2; });
        for(int i = 0; i < 7; ++i) {
            const auto [slot, span] = ring.acquire();
            require(slot == i % 3 && span.data() == pinned.data() + slot * 64 && span.size() == 64, "ring order");
        }
        require(polls[0] == 4 && polls[1] == 3 && polls[2] == 3, "a reused slot waits for its copy, a fresh one does not");
        {
            tr::RowGather table(region, 4, 0);
            tr::RowPrefetch prefetch(table);
            std::vector<std::vector<std::uint64_t>> asked;
            std::vector<std::vector<std::byte>> dst;
            std::vector<tr::RowPrefetch::Ticket> tickets;
            for(int token = 0; token < 40; ++token) {   // 16 rows a draft token, some shared with earlier tokens
                std::vector<std::uint64_t> ids;
                for(int i = 0; i < 16; ++i) ids.push_back(rng() % rows);
                asked.push_back(ids);
                dst.emplace_back(ids.size() * row_bytes);
                tickets.push_back(prefetch.start(ids, dst.back().data()));
            }
            for(std::size_t k = 0; k < tickets.size(); ++k) {
                prefetch.wait(tickets[k]);
                require(prefetch.ready(tickets[k]), "a waited ticket is ready");
                for(std::size_t i = 0; i < asked[k].size(); ++i)
                    for(std::size_t b = 0; b < row_bytes; ++b)
                        require(dst[k][i * row_bytes + b] == value(asked[k][i], b), "prefetched rows in request order");
            }
            prefetch.wait_all();
            const auto s = prefetch.stats();
            require(s.requests == 40 && s.rows == 640 && s.batches >= 1 && s.batches <= 40, "prefetch counters");
            const std::uint64_t outside[] = {rows};
            bool refused = false;
            try { (void)prefetch.start(outside, dst[0].data()); } catch(const std::out_of_range&) { refused = true; }
            require(refused, "a prefetch outside the table refused at start");
            std::cout << "prefetch: 40 draft tokens of 16 rows in " << s.batches << " batch(es), " << s.ready_when_waited
                      << " ready when waited\n";
        }
        std::filesystem::remove(path);
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
