#pragma once

#include <crit/state.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace crit::techniques::recurrent {

inline constexpr std::size_t no_parent = std::numeric_limits<std::size_t>::max();
using Siblings = std::vector<std::vector<std::uint8_t>>;

struct Geometry {
  std::size_t keys = 0, values = 0;
  double epsilon = 1e-6, query_scale = 1.0;
  bool normalize_keys = true, normalize_queries = true;
};
struct Limits {
  std::size_t state_elements = 65536, nodes = 1024, path_steps = 64,
              output_elements = 65536, sibling_bytes = 65536, sibling_count = 16;
  std::uint64_t positions = std::uint64_t{1} << 32;
  double magnitude = 1e12;
};
struct Step {
  std::vector<double> query, key, value;
  double log_decay = 0, beta = 0;
};
struct Node {
  std::size_t parent = no_parent;
  Step step;
};
struct Snapshot {
  // matrix[k * geometry.values + v]; state after exactly after_tokens tokens.
  std::uint64_t after_tokens = 0;
  std::vector<double> matrix;
  Siblings siblings;
};
struct Prepared {
  std::uint64_t id = 0;
  // outputs[accepted ordinal * values + v], never one matrix per token.
  std::vector<double> outputs;
  std::size_t verified_steps = 0;
};
struct Committed {
  Snapshot state;
  std::optional<Snapshot> checkpoint;
};

// Core's after-token checkpoint selection (crit/state.h): the boundary is an absolute
// AFTER-token count, not a token index or a physical verify-node ID; nullopt names the
// initial state, which has no verify node; std::out_of_range outside the accepted prefix.
using crit::core::checkpoint_node;

class Recovery {
public:
  // Synchronous hook: mutate only the supplied PRIVATE sibling bytes. It must
  // settle all physical work on success and failure; native async use needs a
  // separately retained transaction and completion proof before this hook returns.
  using PrepareSiblings = std::function<void(std::span<const std::size_t>, Siblings&)>;

private:
  struct Round {
    std::uint64_t id;
    Snapshot initial;
    std::vector<Node> nodes;
    std::vector<std::size_t> path;
  };
  Geometry geometry_;
  Limits limits_;
  Snapshot state_;
  std::unique_ptr<Round> pending_;
  std::uint64_t next_id_ = 1;
  bool mutating_ = false;

  class Guard {
    bool& active_;
  public:
    explicit Guard(bool& active) : active_(active) {
      if (active_) throw std::logic_error("Reentrant recurrent-state mutation");
      active_ = true;
    }
    ~Guard() { active_ = false; }
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
  };

  static void cancelled(const std::atomic_bool* cancellation) {
    if (cancellation && cancellation->load(std::memory_order_relaxed))
      throw std::runtime_error("Recurrent recovery cancelled");
  }
  void finite(double value) const {
    if (!std::isfinite(value) || std::abs(value) > limits_.magnitude)
      throw std::domain_error("Nonfinite or out-of-bound recurrent arithmetic");
  }
  void siblings_valid(const Siblings& siblings) const {
    if (siblings.size() > limits_.sibling_count)
      throw std::length_error("Too many recurrent-state siblings");
    std::size_t used = 0;
    for (const auto& sibling : siblings) {
      if (sibling.size() > limits_.sibling_bytes - used)
        throw std::length_error("Recurrent-state sibling byte budget exceeded");
      used += sibling.size();
    }
  }
  void snapshot_valid(const Snapshot& state) const {
    if (state.after_tokens > limits_.positions ||
        state.matrix.size() != geometry_.keys * geometry_.values)
      throw std::invalid_argument("Recurrent state shape or absolute position differs");
    for (double value : state.matrix) finite(value);
    siblings_valid(state.siblings);
  }
  void step_valid(const Step& step) const {
    if (step.query.size() != geometry_.keys || step.key.size() != geometry_.keys ||
        step.value.size() != geometry_.values)
      throw std::invalid_argument("Gated-delta step geometry differs");
    for (const auto* values : {&step.query, &step.key, &step.value})
      for (double value : *values) finite(value);
    finite(step.log_decay);
    if (step.log_decay > 0 || !std::isfinite(step.beta) || step.beta < 0 || step.beta > 1)
      throw std::invalid_argument("Gated-delta decay or beta differs");
  }
  std::vector<double> normalized(const std::vector<double>& input, bool normalize) const {
    if (!normalize) return input;
    double norm = std::sqrt(geometry_.epsilon);
    for (double value : input) norm = std::hypot(norm, value);
    if (!std::isfinite(norm) || norm <= 0)
      throw std::domain_error("Invalid recurrent normalization");
    auto result = input;
    for (double& value : result) value /= norm;
    return result;
  }
  // Only one running state and one replacement state exist. No full per-token
  // recurrent-state cache is allocated. Projection/gates are caller-supplied.
  void advance(std::vector<double>& state, const Step& step,
               std::vector<double>* outputs) const {
    const auto key = normalized(step.key, geometry_.normalize_keys);
    const double decay = std::exp(step.log_decay);
    std::vector<double> residual = step.value;
    for (std::size_t v = 0; v < geometry_.values; ++v) {
      for (std::size_t k = 0; k < geometry_.keys; ++k)
        residual[v] -= key[k] * (state[k * geometry_.values + v] * decay);
      finite(residual[v]);
      residual[v] *= step.beta;
    }
    auto next = state;
    for (std::size_t k = 0; k < geometry_.keys; ++k)
      for (std::size_t v = 0; v < geometry_.values; ++v) {
        const auto index = k * geometry_.values + v;
        next[index] = state[index] * decay + key[k] * residual[v];
        finite(next[index]);
      }
    if (outputs) {
      const auto query = normalized(step.query, geometry_.normalize_queries);
      for (std::size_t v = 0; v < geometry_.values; ++v) {
        double output = 0;
        for (std::size_t k = 0; k < geometry_.keys; ++k)
          output += query[k] * next[k * geometry_.values + v];
        output *= geometry_.query_scale;
        finite(output);
        outputs->push_back(output);
      }
    }
    state.swap(next);
  }
  void path_valid(const std::vector<Node>& nodes, std::span<const std::size_t> path) const {
    if (nodes.size() > limits_.nodes || path.size() > limits_.path_steps ||
        path.size() > limits_.positions - state_.after_tokens ||
        path.size() > limits_.output_elements / geometry_.values)
      throw std::length_error("Recurrent verify bounds exceeded");
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      step_valid(nodes[i].step);
      // Physical nodes are topologically numbered, but a selected path can skip
      // siblings and is never indexed by its accepted ordinal.
      if (nodes[i].parent != no_parent && nodes[i].parent >= i)
        throw std::invalid_argument("Verify nodes require earlier physical parents");
    }
    std::size_t parent = no_parent;
    for (auto physical : path) {
      if (physical >= nodes.size() || nodes[physical].parent != parent)
        throw std::invalid_argument("Accepted node map is not one rooted chain");
      parent = physical;
    }
  }
  Snapshot recover(const Round& round, std::size_t count,
                   const PrepareSiblings& prepare_siblings,
                   const std::atomic_bool* cancellation) const {
    Snapshot result = round.initial;
    for (std::size_t i = 0; i < count; ++i) {
      cancelled(cancellation);
      advance(result.matrix, round.nodes[round.path[i]].step, nullptr);
    }
    result.after_tokens += static_cast<std::uint64_t>(count);
    if (count && prepare_siblings)
      prepare_siblings(std::span<const std::size_t>(round.path).first(count), result.siblings);
    cancelled(cancellation);
    snapshot_valid(result);
    return result;
  }

public:
  Recovery(Geometry geometry, Snapshot initial, Limits limits = {})
      : geometry_(geometry), limits_(limits), state_(std::move(initial)) {
    if (!limits_.state_elements || limits_.state_elements > (std::size_t{1} << 20) ||
        !limits_.nodes || limits_.nodes > 4096 || !limits_.path_steps || limits_.path_steps > 4096 ||
        !limits_.output_elements || limits_.output_elements > (std::size_t{1} << 20) ||
        limits_.sibling_bytes > (std::size_t{1} << 24) || limits_.sibling_count > 256 ||
        !limits_.positions || !std::isfinite(limits_.magnitude) || limits_.magnitude <= 0 ||
        !geometry_.keys || !geometry_.values || geometry_.keys > limits_.state_elements / geometry_.values ||
        !std::isfinite(geometry_.epsilon) || geometry_.epsilon <= 0 ||
        !std::isfinite(geometry_.query_scale) || geometry_.query_scale <= 0)
      throw std::invalid_argument("Explicit bounded gated-delta geometry required");
    snapshot_valid(state_);
  }
  Recovery(const Recovery&) = delete;
  Recovery& operator=(const Recovery&) = delete;

  const Snapshot& state() const noexcept { return state_; }
  bool pending() const noexcept { return static_cast<bool>(pending_); }

  Prepared prepare(std::vector<Node> nodes, std::vector<std::size_t> path,
                   const std::atomic_bool* cancellation = nullptr) {
    Guard guard(mutating_);
    cancelled(cancellation);
    if (pending_) throw std::logic_error("Overlapping recurrent preparation");
    if (next_id_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Recurrent round identity exhausted");
    path_valid(nodes, path);
    auto working = state_.matrix;
    Prepared result{next_id_, {}, path.size()};
    result.outputs.reserve(path.size() * geometry_.values);
    for (auto physical : path) {
      cancelled(cancellation);
      advance(working, nodes[physical].step, &result.outputs);
    }
    cancelled(cancellation);
    auto round = std::make_unique<Round>(Round{next_id_, state_, std::move(nodes), std::move(path)});
    pending_ = std::move(round);
    ++next_id_;
    return result;
  }

  Committed commit(std::uint64_t id, std::size_t accepted,
                   std::optional<std::uint64_t> checkpoint_after = std::nullopt,
                   const PrepareSiblings& prepare_siblings = {},
                   const std::atomic_bool* cancellation = nullptr) {
    Guard guard(mutating_);
    if (!pending_ || pending_->id != id) throw std::logic_error("Stale recurrent round");
    // Consume even a failed commit; all work remains private, while state_ stays
    // unchanged until every recurrence and sibling operation has succeeded.
    auto round = std::move(pending_);
    cancelled(cancellation);
    if (accepted > round->path.size()) throw std::out_of_range("Too many accepted steps");
    std::optional<Snapshot> checkpoint;
    if (checkpoint_after) {
      const auto accepted_path = std::span<const std::size_t>(round->path).first(accepted);
      static_cast<void>(checkpoint_node(round->initial.after_tokens, accepted_path, *checkpoint_after));
      checkpoint = recover(*round,
          static_cast<std::size_t>(*checkpoint_after - round->initial.after_tokens),
          prepare_siblings, cancellation);
    }
    auto private_state = recover(*round, accepted, prepare_siblings, cancellation);
    Committed result{private_state, std::move(checkpoint)};
    cancelled(cancellation);
    state_ = std::move(private_state);
    return result;
  }

  void discard(std::uint64_t id) {
    Guard guard(mutating_);
    if (!pending_ || pending_->id != id) throw std::logic_error("Stale recurrent round");
    pending_.reset();
  }
};

} // namespace crit::techniques::recurrent
