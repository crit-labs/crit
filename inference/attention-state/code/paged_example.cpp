#include "attention.h"
#include <crit/state.h>
#include <array>
#include <iostream>

// Core's StateCache keeps the bookkeeping (prefix reuse, rounds, rollback,
// retirement); StatePages keeps the FP32 rows; paged_attention reads them.
int main() {
  namespace cc = crit::core;
  using namespace crit::techniques;
  try {
    cc::Resources resources(std::size_t{1} << 20, 0);
    StatePages pages(4, 2, 2, 1);
    cc::StateCacheConfig config;
    config.pages = 4; config.page_tokens = 2; config.sequences = 2; config.max_tokens = 8; config.max_round_rows = 2;
    config.prefix_cache = true;
    config.identity = 0x5eed; // Names weights, layer, adapter, position profile and the FP32 KV representation.
    cc::StateCache cache(config, pages, resources);
    const auto prompt = cache.open();
    auto round = pages.stage(cache, prompt, std::array{1.f, 0.f, 0.f, 1.f}, std::array{2.f, 4.f});
    cache.commit(prompt, round.id, std::array<cc::Token, 2>{42, 7}); // A full page: indexed for reuse.
    cache.close(prompt);
    cache.retire(cache.epoch());
    const auto reply = cache.open();
    const std::array<cc::Token, 3> request{42, 7, 9};
    const auto reused = cache.attach(reply, request, 2);
    round = pages.stage(cache, reply, std::array{0.f, 1.f}, std::array{12.f});
    const auto output = paged_attention(pages.view(cache, reply, true), std::array{0.f, 0.f}, 1);
    cache.discard(reply, round.id); // The staged row is rejected; the shared prefix is untouched.
    const auto kept = pages.view(cache, reply);
    if (reused != 2 || output != std::vector<float>{6.f} || kept.rows() != 2 || kept.value(1)[0] != 4.f ||
        cache.stats().prefix_hits != 1) return 1;
    std::cout << "attention=" << output[0] << " over a reused prefix page; rejected row discarded\n";
    cache.close(reply);
    cache.retire(cache.epoch());
    if (cache.stats().live_pages != 0 || cache.stats().retiring_pages != 0) return 1;
    std::cout << "every page returned to the cache after retire\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
