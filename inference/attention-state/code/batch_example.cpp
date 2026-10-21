#include "batch_engine.h"
using namespace crit::examples::attention;
int main() {
  try {
    check_copy_on_write();
    auto stats = std::make_shared<Stats>();
    cc::Profile p{64, 2, 8 * 1024 * 1024, 1024, 16, 16};
    p.max_batch_tokens = 32; p.batch_policy = cc::BatchPolicy::decode_first;
    cc::Engine engine(p, std::make_shared<Backend>(), [stats](const auto&, cc::Services services) {
      return std::make_unique<Model>(stats, services.resources);
    });
    std::vector<cc::Handle> handles; std::vector<std::vector<cc::Token>> expected;
    for (int i = 0; i < 24; ++i) {
      std::vector<cc::Token> prompt(static_cast<std::size_t>(i % 3 == 0 ? 1 : 15), i % 16);
      const auto output = static_cast<std::uint32_t>(i % 3 == 0 ? 1 : 10);
      expected.push_back(dense(prompt, output)); handles.push_back(engine.submit(prompt, output));
    }
    stats->start = true;
    for (std::size_t i = 0; i < handles.size(); ++i) require(handles[i].wait().tokens == expected[i], "batched output differs from dense oracle");
    require(stats->largest_batch == 16 && stats->mixed && stats->refilled, "continuous batch coverage incomplete");
    std::cout << "24 ragged requests match dense attention; 16 active lanes, mixed phases and lane refill verified\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
