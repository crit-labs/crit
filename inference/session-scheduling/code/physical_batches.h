#pragma once
// Portable physical execution shapes and synchronous host scratch. Logical
// scheduling, model state and backend allocation remain engine responsibilities.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace crit::techniques::physical {

struct BatchLimit {
  std::size_t inherited_columns;
  std::size_t override_columns = 0; // Zero preserves the inherited physical cap.
  std::size_t resolve() const {
    const auto value = override_columns ? override_columns : inherited_columns;
    if (!value) throw std::invalid_argument("Positive physical column limit required");
    return value;
  }
};

struct SplitRules {
  bool splittable = false; // Opt in only when state can carry across ordered calls.
  std::size_t atomic_columns = 1; // Every boundary, including the end, is a group boundary.
  std::size_t padding_multiple = 1; // >1 explicitly promises safe masked padding.
};

struct BatchShape {
  std::size_t offset, columns, physical_columns;
};
struct PhysicalBatch {
  BatchShape shape;
  std::size_t workspace_bytes;
};

inline std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("Physical batch byte/column sum overflow");
  return a + b;
}
inline std::size_t checked_multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error("Physical batch byte product overflow");
  return a * b;
}

// A useful affine layout, not an assumption made by the planner. Engines can
// supply any exact shape-dependent sizing function, including nonmonotonic ones.
struct LinearWorkspace {
  std::size_t fixed_bytes, bytes_per_column;
  std::size_t operator()(const BatchShape& shape) const {
    return checked_add(fixed_bytes, checked_multiply(shape.physical_columns, bytes_per_column));
  }
};

class BatchPlan {
  std::vector<PhysicalBatch> batches_;
  std::size_t columns_, physical_columns_ = 0, peak_bytes_ = 0, limit_;
public:
  // One ordered logical stream per plan. No sequence packing or implicit state
  // reset. A failed plan publishes nothing and never allocates backend scratch.
  template<class WorkspaceSize>
  BatchPlan(std::size_t columns, BatchLimit limit, SplitRules rules, WorkspaceSize size,
            std::size_t workspace_budget = std::numeric_limits<std::size_t>::max())
      : columns_(columns), limit_(limit.resolve()) {
    if (!rules.atomic_columns || !rules.padding_multiple)
      throw std::invalid_argument("Positive atomic group and padding multiple required");
    if (columns % rules.atomic_columns)
      throw std::invalid_argument("Logical stream ends inside an atomic group");
    if (!columns) return;
    const auto usable = limit_ - limit_ % rules.padding_multiple;
    const auto chunk = usable - usable % rules.atomic_columns;
    if (!chunk || (!rules.splittable && columns > chunk))
      throw std::length_error("Physical limit cannot execute the declared atomic work");
    for (std::size_t offset = 0; offset < columns;) {
      const auto count = std::min(chunk, columns - offset);
      const auto padded = count % rules.padding_multiple
          ? checked_add(count, rules.padding_multiple - count % rules.padding_multiple) : count;
      const BatchShape shape{offset, count, padded};
      const std::size_t bytes = size(shape);
      if (bytes > workspace_budget) throw std::length_error("Physical workspace exceeds budget");
      batches_.push_back({shape, bytes});
      peak_bytes_ = std::max(peak_bytes_, bytes);
      physical_columns_ = checked_add(physical_columns_, padded);
      offset += count; // count <= columns-offset, so no wraparound.
    }
  }
  std::span<const PhysicalBatch> batches() const noexcept { return batches_; }
  std::size_t columns() const noexcept { return columns_; }
  std::size_t physical_columns() const noexcept { return physical_columns_; }
  std::size_t peak_workspace_bytes() const noexcept { return peak_bytes_; }
  std::size_t limit() const noexcept { return limit_; }
};

class BatchCancelled final : public std::runtime_error {
public:
  BatchCancelled() : std::runtime_error("Physical batch execution cancelled") {}
};

// Owns exactly `bytes` payload bytes (allocator overhead is separate). Allocates
// once, with no resizing inside execution. It is a host reference, not a device
// allocator; native engines reserve/allocate their actual layout through Core.
class HostWorkspace {
  const std::size_t bytes_;
  std::unique_ptr<std::byte[]> storage_;
  std::atomic_flag running_ = ATOMIC_FLAG_INIT;
  struct Guard {
    std::atomic_flag& flag;
    explicit Guard(std::atomic_flag& value) : flag(value) {
      if (flag.test_and_set(std::memory_order_acquire))
        throw std::logic_error("Workspace already in use");
    }
    ~Guard() { flag.clear(std::memory_order_release); }
  };
  static void check_cancelled(const std::atomic_bool* cancelled) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) throw BatchCancelled();
  }
public:
  explicit HostWorkspace(std::size_t bytes)
      : bytes_(bytes), storage_(bytes ? new std::byte[bytes] : nullptr) {}
  HostWorkspace(const HostWorkspace&) = delete;
  HostWorkspace& operator=(const HostWorkspace&) = delete;
  std::size_t allocated_bytes() const noexcept { return bytes_; }

  // Callback work and borrowed scratch must settle before return OR throw.
  // Failure/cancellation stops later calls; the engine discards private state.
  // This helper neither commits nor rolls back model state. Only cancellation
  // may change concurrently; independent in-flight work needs separate owners.
  template<class Execute>
  void run(const BatchPlan& plan, Execute execute, const std::atomic_bool* cancelled = nullptr) {
    Guard guard(running_);
    if (plan.peak_workspace_bytes() > bytes_)
      throw std::length_error("Workspace is smaller than the physical plan");
    check_cancelled(cancelled);
    for (const auto& batch : plan.batches()) {
      check_cancelled(cancelled);
      execute(batch.shape, std::span<std::byte>(storage_.get(), batch.workspace_bytes));
      check_cancelled(cancelled);
    }
  }
};
} // namespace crit::techniques::physical
