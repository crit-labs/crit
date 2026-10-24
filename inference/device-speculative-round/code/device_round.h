#pragma once
// GPU-resident greedy speculative round: one Session::advance() per round, at
// most two CUDA graphs (Forward, Finish) and exactly one readback_async.
#include "greedy_accept.h"
#include <crit/cuda.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace crit::techniques::device_round {
// Device pointers for engine steps; stable for the controller's lifetime, so
// steps may be captured into graphs. One row.
struct RoundView {
    std::int32_t* control = nullptr;     // control::fields int32
    std::int32_t* drafts = nullptr;      // [K], written by draft
    std::int32_t* predictions = nullptr; // [K+1], written by verify
    const std::int32_t* result = nullptr; // [K+2] packed {n_emitted, tokens}
    std::uint32_t draft_tokens = 0;
};
// Submit only to the supplied stream; no host synchronization, no readback and no
// allocation inside a step. With graphs enabled the controller records draft and
// verify into Forward, and commit_state into Finish, once.
using DeviceStep = std::function<void(cudaStream_t, const RoundView&)>;
struct RoundSteps {
    // Write drafts[0..K) from control[last_token] at control[frontier].
    DeviceStep draft;
    // Feed control[last_token] then drafts at positions frontier..frontier+K and
    // write predictions[i] = target argmax after (last, d1..di). Keep enough
    // provisional state (KV columns, recurrent checkpoints or inputs) to select
    // any accepted prefix later.
    DeviceStep verify;
    // Apply control[n_emitted] (and base_frontier) to model state on the device.
    // Pure function of round-start state and verify outputs: it reruns eagerly
    // after a Core truncation with a smaller count.
    DeviceStep commit_state;
    // Optional replacement for the greedy accept kernel in Finish, for example draft-trees'
    // tree accept. It receives the AcceptRows the greedy kernel would get (drafts,
    // predictions, device stops, control, writable result) and must write the packed result
    // and the control fields (n_emitted, n_matched, base_frontier, base_last, frontier,
    // last_token) exactly as enqueue_greedy_accept defines them, so enqueue_truncate and
    // commit work unchanged. Unset: the greedy accept kernel.
    std::function<void(cudaStream_t, const AcceptRows&)> accept = {};
};
struct RoundConfig {
    std::uint32_t draft_tokens = 0;   // K; zero is ordinary one-token decoding
    std::uint32_t vocabulary = 0;
    std::uint32_t context = 0;        // maximum materialized positions (KV slots)
    std::vector<std::int32_t> stop_tokens;
    bool graphs = true;               // false runs the same steps eagerly
    std::uint32_t max_stops = 0;      // device stop capacity for set_stops; at least stop_tokens.size()
};

class DeviceSpeculativeRound {
public:
    DeviceSpeculativeRound(std::shared_ptr<core::CudaBackend>, RoundConfig, RoundSteps);
    ~DeviceSpeculativeRound();
    DeviceSpeculativeRound(const DeviceSpeculativeRound&) = delete;
    DeviceSpeculativeRound& operator=(const DeviceSpeculativeRound&) = delete;

    // Setup, not per round: set the device frontier and pending token after
    // prefill. Stream-ordered after earlier work; waits for its own small copy.
    void start(std::uint32_t frontier, std::int32_t last_token);
    // Setup, not per round: replace the device stop list (at most the configured
    // capacity) without recapturing; Core request options can change it per request.
    void set_stops(std::span<const std::int32_t> stops);
    // Drop a launched round that will never be committed, e.g. a request Core
    // cancelled after advance(). Waits for its readback; call start() before the
    // next round. Session::close should call it when a round is still pending.
    void abandon();
    // Capture Forward and Finish now instead of on the first launch.
    void prepare();
    // Enqueue one whole round: Forward graph, Finish graph, then one readback of
    // the packed result. Return value is Progress::readiness. No host fence.
    core::Readiness* launch(std::uint32_t output_tokens);
    // After readiness: the device-chosen tokens. Borrowed until commit/next launch.
    std::span<const std::int32_t> tokens();
    // Publish Core's accepted prefix. accepted < n_emitted enqueues the device
    // truncation and reruns commit_state; the host never re-verifies tokens.
    void commit(std::uint32_t accepted);
    // After readiness: Forward (DeviceStage::forward) and Finish (accept) timings.
    std::span<const core::DeviceTiming> timings();

    RoundView view() const noexcept;
    std::uint32_t frontier() const noexcept; // host mirror of committed positions
    std::uint64_t rounds() const noexcept;
    std::uint64_t readbacks() const noexcept;
    bool captured() const noexcept;
    bool pending() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
