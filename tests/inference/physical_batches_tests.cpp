#include "physical_batches.h"
#include <future>
#include <iostream>
#include <random>

using namespace crit::techniques::physical;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class Error, class F> void rejects(F f) {
  try { f(); } catch (const Error&) { return; }
  throw std::runtime_error("Expected rejection");
}

void shapes() {
  const LinearWorkspace bytes{32, 16};
  const BatchPlan target(17, {12}, {true}, bytes);
  const BatchPlan inherited(17, {12, 0}, {true}, bytes);
  const BatchPlan draft(17, {12, 4}, {true}, bytes);
  require(target.columns() == draft.columns() && inherited.peak_workspace_bytes() == target.peak_workspace_bytes(), "Inheritance/logical width");
  require(target.batches().size() == 2 && draft.batches().size() == 5, "Independent physical caps");
  require(target.peak_workspace_bytes() == 224 && draft.peak_workspace_bytes() == 96, "Workspace does not shrink");
  const BatchPlan blocks(15, {8}, {true, 3, 4}, bytes);
  require(blocks.batches().size() == 3 && blocks.physical_columns() == 20, "Atomic/padded shapes");
  require(blocks.batches().back().shape.columns == 3 && blocks.batches().back().shape.physical_columns == 4, "Padded tail");
  // The tail, not the widest call, is the expensive algorithm in this layout.
  const BatchPlan nonlinear(9, {4}, {true}, [](const BatchShape& s) { return s.columns == 1 ? 99u : 10u; });
  require(nonlinear.peak_workspace_bytes() == 99, "Nonmonotonic sizing lost the tail");
  rejects<std::length_error>([&] { BatchPlan p(9, {4}, {true}, [](const BatchShape& s) { return s.columns == 1 ? 99u : 10u; }, 98); });
  rejects<std::length_error>([&] { BatchPlan p(9, {8}, {}, bytes); });
  const BatchPlan whole(9, {12}, {false, 9, 4}, bytes);
  require(whole.batches().size() == 1 && whole.physical_columns() == 12, "Whole block split");
  rejects<std::length_error>([&] { BatchPlan p(9, {8}, {true, 9}, bytes); });
  rejects<std::invalid_argument>([&] { BatchPlan p(7, {8}, {true, 3}, bytes); });
  rejects<std::invalid_argument>([&] { BatchPlan p(1, {0}, {true}, bytes); });
  rejects<std::invalid_argument>([&] { BatchPlan p(1, {8}, {true, 0}, bytes); });
  rejects<std::invalid_argument>([&] { BatchPlan p(1, {8}, {true, 1, 0}, bytes); });
  rejects<std::length_error>([&] { BatchPlan p(1, {3}, {true, 1, 4}, bytes); });
  const BatchPlan empty(0, {8}, {}, [](const BatchShape&) -> std::size_t { throw std::runtime_error("Empty sized"); });
  require(empty.batches().empty() && empty.peak_workspace_bytes() == 0, "Empty allocated");
  const auto max = std::numeric_limits<std::size_t>::max();
  rejects<std::overflow_error>([&] { BatchPlan p(max, {max}, {}, LinearWorkspace{1, 1}); });
  rejects<std::overflow_error>([&] { BatchPlan p(max, {max}, {}, LinearWorkspace{0, 2}); });
  rejects<std::overflow_error>([&] { BatchPlan p(max, {max - 1}, {true, 1, 2}, LinearWorkspace{0, 0}); });
  const BatchPlan huge(max, {max}, {}, LinearWorkspace{0, 0});
  require(huge.batches().size() == 1 && huge.physical_columns() == max, "Large legal extent");

  // Independent small-shape oracle enumerates all possible valid chunk sizes.
  std::mt19937 rng(30459);
  for (int trial = 0; trial < 5000; ++trial) {
    const std::size_t atomic = 1 + rng() % 7, padding = 1 + rng() % 8;
    const std::size_t cap = 1 + rng() % 40, total = atomic * (1 + rng() % 30);
    std::size_t largest = 0;
    for (std::size_t n = atomic; n <= cap; n += atomic)
      if (((n + padding - 1) / padding) * padding <= cap) largest = n;
    if (!largest) {
      rejects<std::length_error>([&] { BatchPlan p(total, {cap}, {true, atomic, padding}, bytes); });
      continue;
    }
    const BatchPlan plan(total, {cap}, {true, atomic, padding}, bytes);
    std::size_t covered = 0, executed = 0, peak = 0;
    for (const auto& b : plan.batches()) {
      require(b.shape.offset == covered && b.shape.columns == std::min(largest, total - covered), "Coverage/order/oracle");
      require(b.shape.columns % atomic == 0 && b.shape.physical_columns % padding == 0 && b.shape.physical_columns <= cap, "Illegal physical shape");
      require(b.shape.physical_columns >= b.shape.columns && b.shape.physical_columns - b.shape.columns < padding, "Incorrect padding");
      covered += b.shape.columns; executed += b.shape.physical_columns; peak = std::max(peak, bytes(b.shape));
    }
    require(covered == total && plan.physical_columns() == executed && plan.peak_workspace_bytes() == peak, "Plan totals");
  }
}

void execution() {
  const BatchPlan plan(10, {4}, {true}, LinearWorkspace{0, 4});
  HostWorkspace workspace(plan.peak_workspace_bytes()), independent(plan.peak_workspace_bytes());
  require(workspace.allocated_bytes() == 16, "Payload allocation size");
  std::size_t calls = 0;
  std::byte* address = nullptr;
  workspace.run(plan, [&](const BatchShape& s, std::span<std::byte> scratch) {
    require(scratch.size() == s.physical_columns * 4, "Callback extent");
    if (address) require(scratch.data() == address, "Workspace resized");
    address = scratch.data();
    std::fill(scratch.begin(), scratch.end(), std::byte{0x5a});
    rejects<std::logic_error>([&] { workspace.run(plan, [](auto, auto) {}); });
    independent.run(plan, [](auto, auto) {});
    ++calls;
  });
  require(calls == 3, "Missing execution calls");
  HostWorkspace small(15);
  rejects<std::length_error>([&] { small.run(plan, [&](auto, auto) { ++calls; }); });
  require(calls == 3, "Undersized execution mutated state");
  std::atomic_bool cancelled = true;
  rejects<BatchCancelled>([&] { workspace.run(plan, [&](auto, auto) { ++calls; }, &cancelled); });
  require(calls == 3, "Cancelled before start");
  cancelled = false;
  calls = 0;
  rejects<BatchCancelled>([&] { workspace.run(plan, [&](auto, auto) { ++calls; cancelled = true; }, &cancelled); });
  require(calls == 1, "Cancellation did not stop the next batch");
  calls = 0;
  rejects<std::runtime_error>([&] { workspace.run(plan, [&](auto, auto) { if (++calls == 2) throw std::runtime_error("Injected"); }); });
  require(calls == 2, "Failure did not stop the next batch");
  workspace.run(plan, [](auto, auto) {}); // Both exceptional exits released the guard.

  std::promise<void> entered, release;
  auto go = release.get_future().share();
  auto pending = std::async(std::launch::async, [&] {
    bool first = true;
    workspace.run(plan, [&](auto, auto) { if (first) { first = false; entered.set_value(); go.wait(); } });
  });
  entered.get_future().wait();
  bool rejected = false;
  try { workspace.run(plan, [](auto, auto) {}); } catch (const std::logic_error&) { rejected = true; }
  release.set_value(); pending.get();
  require(rejected, "Overlapping workspace use was accepted");
  HostWorkspace zero(0);
  const BatchPlan no_work(0, {1}, {}, LinearWorkspace{16, 16});
  zero.run(no_work, [](auto, auto) { throw std::runtime_error("Empty executed"); });
  const BatchPlan no_scratch(2, {1}, {true}, LinearWorkspace{0, 0});
  zero.run(no_scratch, [](auto, auto scratch) { require(scratch.empty(), "Zero scratch extent"); });
}
int main() { try {
  shapes(); execution();
  std::cout << "Physical batch shape oracle, scratch ownership and failure checks passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; } }
