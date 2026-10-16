#pragma once
#include "attention.h"
#include <algorithm>
#include <compare>
#include <memory>

namespace crit::techniques {
// Engine-defined identities, independent of Core lane numbering or model type.
struct AttentionRequestKey {
  std::uint64_t request = 0, generation = 0;
  auto operator<=>(const AttentionRequestKey&) const = default;
};
template<class KVView> struct AttentionBatchSpec {
  AttentionRequestKey key;
  KVView kv;
  AttentionGeometry geometry;
  std::span<const std::size_t> query_rows;
};
struct AttentionBatchInput {
  AttentionRequestKey key;
  HeadTensorView queries;
  AttentionMask mask = {};
};
struct AttentionBatchResult {
  AttentionRequestKey key;
  AttentionResult attention;
};
struct AttentionBatchLimits {
  std::size_t requests = 0, query_rows = 0;
  std::size_t workspace_bytes = 0, output_bytes = 0, metadata_bytes = 0;
};

// Ragged CPU reference: private retained views/workspaces, one synchronous plan
// per request, no context padding or dense KV materialization. This is an
// execution/ownership reference for a fused device adapter, not a GPU kernel.
template<class KVView> class BasicAttentionBatch {
  using Plan = BasicAttentionPlan<KVView>;
  struct Entry {
    AttentionRequestKey key;
    std::size_t source;
    std::unique_ptr<Plan> plan;
  };
  std::vector<Entry> entries_;
  std::vector<const AttentionBatchInput*> inputs_;
  std::size_t workspace_bytes_ = 0, output_bytes_ = 0, metadata_bytes_ = 0;
  std::atomic_flag running_ = ATOMIC_FLAG_INIT;
  struct Guard {
    std::atomic_flag& flag;
    explicit Guard(std::atomic_flag& value) : flag(value) {
      if (flag.test_and_set(std::memory_order_acquire)) throw std::logic_error("Attention batch is already in use");
    }
    ~Guard() { flag.clear(std::memory_order_release); }
  };
  static void stop(const std::atomic_bool* cancelled) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) throw std::runtime_error("Attention batch cancelled");
  }
public:
  BasicAttentionBatch(std::span<const AttentionBatchSpec<KVView>> specs,
                      std::size_t query_tile, std::size_t key_tile, AttentionBatchLimits limits) {
    using namespace attention_detail;
    if (specs.empty() || specs.size() > limits.requests)
      throw std::length_error("Attention batch request limit exceeded or empty batch");
    // Includes result-row and membership-vector payloads used while running.
    metadata_bytes_ = add(sizeof(*this), product(specs.size(),
        sizeof(Entry) + sizeof(const AttentionBatchInput*) + sizeof(AttentionBatchResult)));
    if (metadata_bytes_ > limits.metadata_bytes)
      throw std::length_error("Attention batch metadata byte limit exceeded");
    std::size_t queries = 0;
    for (const auto& spec : specs) queries = add(queries, spec.query_rows.size());
    if (queries > limits.query_rows) throw std::length_error("Attention batch query row limit exceeded");
    entries_.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) entries_.push_back({specs[i].key, i, {}});
    std::sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.key < b.key; });
    for (std::size_t i = 1; i < entries_.size(); ++i)
      if (entries_[i - 1].key == entries_[i].key) throw std::invalid_argument("Duplicate attention batch identity");
    for (auto& entry : entries_) {
      const auto& spec = specs[entry.source];
      entry.plan = std::make_unique<Plan>(spec.kv, spec.geometry, spec.query_rows, query_tile, key_tile,
          limits.workspace_bytes - workspace_bytes_, limits.output_bytes - output_bytes_, limits.metadata_bytes - metadata_bytes_);
      workspace_bytes_ = add(workspace_bytes_, entry.plan->workspace_bytes());
      output_bytes_ = add(output_bytes_, entry.plan->output_bytes());
      metadata_bytes_ = add(metadata_bytes_, entry.plan->metadata_bytes());
    }
    inputs_.resize(specs.size());
  }
  BasicAttentionBatch(const BasicAttentionBatch&) = delete;
  BasicAttentionBatch& operator=(const BasicAttentionBatch&) = delete;
  std::size_t workspace_bytes() const { return workspace_bytes_; }
  std::size_t output_bytes() const { return output_bytes_; }
  std::size_t metadata_bytes() const { return metadata_bytes_; }

  template<class Gemm> requires std::invocable<Gemm&, bool, std::size_t, std::size_t, std::size_t,
                                              std::span<const float>, std::span<const float>, std::span<float>>
  std::vector<AttentionBatchResult> run(std::span<const AttentionBatchInput> inputs, Gemm& gemm,
                                        const std::atomic_bool* cancelled = nullptr) {
    Guard guard(running_);
    stop(cancelled);
    if (inputs.size() != entries_.size()) throw std::invalid_argument("Attention batch membership count mismatch");
    std::fill(inputs_.begin(), inputs_.end(), nullptr);
    for (const auto& input : inputs) {
      const auto it = std::lower_bound(entries_.begin(), entries_.end(), input.key,
          [](const Entry& entry, AttentionRequestKey key) { return entry.key < key; });
      if (it == entries_.end() || it->key != input.key) throw std::invalid_argument("Unknown attention batch identity");
      auto& slot = inputs_[static_cast<std::size_t>(it - entries_.begin())];
      if (slot) throw std::invalid_argument("Duplicate attention batch identity");
      slot = &input;
    }
    // Returned order is key order, never the input's row order. No partial batch
    // escapes a cancellation or kernel failure, even after earlier rows finished.
    std::vector<AttentionBatchResult> results;
    results.reserve(entries_.size());
    for (std::size_t i = 0; i < entries_.size(); ++i)
      results.push_back({entries_[i].key, entries_[i].plan->run(inputs_[i]->queries, inputs_[i]->mask, gemm, cancelled)});
    stop(cancelled);
    return results;
  }
  std::vector<AttentionBatchResult> run(std::span<const AttentionBatchInput> inputs,
                                        const std::atomic_bool* cancelled = nullptr) {
    ScalarAttentionGemm gemm;
    return run(inputs, gemm, cancelled);
  }
};
using PagedAttentionBatch = BasicAttentionBatch<StatePagesView>;
}
