#include "recurrent_recovery.h"
#include <array>
#include <iostream>

int main() {
  using namespace crit::techniques::recurrent;
  Recovery recovery({2, 1}, Snapshot{63, {0.0, 0.0}, {{7}}});
  Step first{{0.3, 0.7}, {0.4, -0.2}, {1.0}, -0.2, 0.6};
  Step second{{0.5, -0.1}, {0.1, 0.8}, {-0.5}, -0.1, 0.4};
  // A physical sibling at node 1 is skipped by the verified/accepted path.
  auto prepared = recovery.prepare({{no_parent, first}, {0, first}, {0, second}}, {0, 2});
  if (checkpoint_node(63, std::array<std::size_t, 2>{0, 2}, 64) != 0)
    throw std::runtime_error("After-token checkpoint mapping differs");
  const auto result = recovery.commit(prepared.id, 1, 64,
      [](auto path, Siblings& siblings) { siblings.front().push_back(static_cast<std::uint8_t>(path.back())); });
  if (result.state.after_tokens != 64 || result.checkpoint->matrix != result.state.matrix)
    throw std::runtime_error("Checkpoint and accepted state differ");
  std::cout << "Recovered one accepted step; rejected state stayed private\n";
}
