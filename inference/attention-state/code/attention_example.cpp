#include "attention.h"
#include <crit/state.h>
#include <array>
#include <iostream>

int main() {
  using namespace crit::techniques;
  try {
    // Toy projections, not a model. Four query heads share two KV heads.
    const AttentionGeometry geometry{4, 2, 4, 2, .5};
    std::vector<float> queries(3 * 4 * 4), keys(3 * 2 * 4), values(3 * 2 * 2);
    for (std::size_t i = 0; i < queries.size(); ++i) queries[i] = static_cast<float>(i % 7) / 7;
    for (std::size_t i = 0; i < keys.size(); ++i) keys[i] = static_cast<float>(i % 5) / 5;
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i) / 3;
    // RoPE position IDs need not be the logical KV row indices used by the mask.
    const std::array<std::uint64_t, 3> positions{10, 11, 12};
    const auto q = rotary({queries, 3, 4, 4, 16, 4}, positions, std::array{1., .01}, RotaryLayout::split_half);
    const auto k = rotary({keys, 3, 2, 4, 8, 4}, positions, std::array{1., .01}, RotaryLayout::split_half);
    // Core's StateCache keeps sequences, rounds and prefix reuse; StatePages the FP32 rows.
    crit::core::Resources resources(std::size_t{1} << 20, 0);
    StatePages pages(8, 2, 8, 4);
    crit::core::StateCacheConfig config;
    config.pages = 8; config.page_tokens = 2; config.sequences = 2; config.max_tokens = 3; config.max_round_rows = 2;
    config.prefix_cache = true;
    config.identity = 0x10ad; // Names toy weights, layer, adapter, RoPE offset 10, split-half layout and FP32 KV.
    crit::core::StateCache cache(config, pages, resources);
    const auto prompt = cache.open();
    AttentionResult prefill;
    {
      const auto round = pages.stage(cache, prompt, std::span(k).first(16), std::span(values).first(8));
      PagedAttentionPlan plan(pages.view(cache, prompt, true), geometry, std::array<std::size_t, 2>{0, 1}, 2, 2, 4096);
      // In a Core session reserve arena, scratch, result and metadata separately;
      // these payload counters do not include allocator or GEMM-provider overhead.
      std::cout << "arena=" << pages.reserved_bytes() << ", cache metadata=" << cache.metadata_bytes()
        << ", scratch=" << plan.workspace_bytes() << ", result=" << plan.output_bytes()
        << ", metadata=" << plan.metadata_bytes() << '\n';
      prefill = plan.run({std::span(q).first(32), 2, 4, 4, 16, 4});
      // Publish only after the synchronous arithmetic succeeds; the full page is indexed.
      cache.commit(prompt, round.id, std::array<crit::core::Token, 2>{42, 7});
    }
    const auto reply = cache.open();
    const std::array<crit::core::Token, 3> request{42, 7, 9};
    if (cache.attach(reply, request, 2) != 2) throw std::runtime_error("Expected prefix hit");
    const auto round = pages.stage(cache, reply, std::span(k).subspan(16), std::span(values).subspan(8));
    PagedAttentionPlan decode(pages.view(cache, reply, true), geometry, std::array<std::size_t, 1>{2}, 1, 2, 4096);
    const auto output = decode.run({std::span(q).subspan(32), 1, 4, 4, 16, 4});
    cache.commit(reply, round.id, std::array<crit::core::Token, 1>{9});
    if (cache.committed(prompt) != 2 || cache.committed(reply) != 3 || prefill.values.size() != 16 || output.values.size() != 8)
      throw std::runtime_error("Attention/prefix geometry changed");
    std::cout << "prefill first=" << prefill.values.front() << ", decode first=" << output.values.front()
      << ", cached prefix remains two rows\n";
    // Each plan reads a copy of its block table. Pages a sequence releases stay
    // readable until cache.retire() covers their epoch; synchronous GEMM return
    // is the completion point, so a session retires after each step.
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
