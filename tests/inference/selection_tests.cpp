// Recorded selection: buckets, recorded replay, tuning ties and failures, and the
// table round trip through to_json/parse.
#include "selection.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sel = crit::techniques::selection;

#define REQUIRE(x) do { if(!(x)) { std::fprintf(stderr, "FAILED %s:%d %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while(0)

template <class F>
bool throws(F&& f) {
    try { f(); } catch(const std::exception&) { return true; }
    return false;
}

int main() {
    REQUIRE(sel::bucket(1) == 1 && sel::bucket(3) == 4 && sel::bucket(4) == 4 && sel::bucket(700, 16) == 1024);
    REQUIRE(sel::bucket(5, 16) == 16 && sel::bucket(1000, 16, 1000) == 1000 && sel::bucket(600, 16, 1000) == 1000);
    REQUIRE(throws([] { sel::bucket(0); }) && throws([] { sel::bucket(5, 3); }) && throws([] { sel::bucket(1001, 16, 1000); }));

    // Tuning: fastest wins, ties keep the lower index, failed candidates are skipped,
    // and a key is timed once.
    sel::Table tune({}, true);
    int calls = 0;
    const double times[4] = {3.0, 1.0, 1.0, -1.0};
    auto c = tune.choose("gemm:16x8:64", 4, [&](std::uint32_t i) { ++calls; return times[i]; });
    REQUIRE(c.index == 1 && c.source == sel::Source::tuned && calls == 4);
    c = tune.choose("gemm:16x8:64", 4, [&](std::uint32_t) { ++calls; return 0.0; });
    REQUIRE(c.index == 1 && calls == 4);
    c = tune.choose("gemm:16x8:128", 3, [](std::uint32_t) { return -1.0; });  // nothing ran: fallback
    REQUIRE(c.index == 0 && c.source == sel::Source::fallback);
    const std::string json = tune.to_json();
    REQUIRE(json == "{\"gemm:16x8:128\":0,\"gemm:16x8:64\":1}");

    // Replay: the recorded index is used and no timing runs; unknown keys fall back.
    sel::Table replay(sel::Table::parse(json), true);
    c = replay.choose("gemm:16x8:64", 4, [&](std::uint32_t) { ++calls; return 0.0; });
    REQUIRE(c.index == 1 && c.source == sel::Source::recorded && calls == 4);
    sel::Table fixed(sel::Table::parse(" { } "));
    c = fixed.choose("other:1:1", 5, [&](std::uint32_t) { ++calls; return 0.0; });
    REQUIRE(c.index == 0 && c.source == sel::Source::fallback && calls == 4);  // not tuning: never timed
    REQUIRE(throws([&] { sel::Table t(sel::Table::parse("{\"k\":7}")); t.choose("k", 3); }));

    // Malformed tables are rejected.
    for(const char* bad : {"", "{", "{\"a\":-1}", "{\"a\":1,}", "{\"a\":1}x", "{\"a\":1,\"a\":2}", "{\"\":1}", "{a:1}", "{\"a\":99999999999}"})
        REQUIRE(throws([&] { sel::Table::parse(bad); }));
    REQUIRE(throws([] { sel::Table t; t.choose("k", 0); }));
    std::puts("selection ok");
    return 0;
}
