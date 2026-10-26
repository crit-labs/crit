#pragma once
// GPU-resident greedy speculative round: one Session::advance() per round, two
// CUDA graphs (Forward, Finish) per draft width and exactly one readback of the
// round's readback block (tokens, counts, frontier, optional log-probabilities):
// a readback_async, or with RoundConfig::zero_copy the block written in place.
#include "greedy_accept.h"
#include <crit/cuda.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace crit::techniques::device_round {
// Device pointers for engine steps; stable for the controller's lifetime, so
// steps may be captured into graphs. One row. The buffers are sized for the widest
// width, RoundConfig::draft_tokens (max_draft_tokens); a round of K drafts uses
// their first K, K + 1 and K + 2 entries.
struct RoundView {
    std::int32_t* control = nullptr;     // control::fields int32
    std::int32_t* drafts = nullptr;      // [K], written by draft
    std::int32_t* predictions = nullptr; // [K+1], written by verify
    const std::int32_t* result = nullptr; // [K+2] packed {n_emitted, tokens}
    std::uint32_t draft_tokens = 0;      // K of this round
    // The widest K. The previous round may have emitted up to max_draft_tokens + 1
    // tokens, so a stateful drafter's catch-up covers that many columns whatever K is.
    std::uint32_t max_draft_tokens = 0;
    // With RoundConfig::logprobs, written by verify for columns 0..K: logprob[c] is the
    // log-probability of predictions[c] under column c's raw logits (core::TokenLogprob's
    // definition) and top_ids/top_logprobs[c * top + i] its `top` ranked alternatives
    // (log-probability descending, then id ascending). Null when logprobs are off.
    float* logprob = nullptr;            // [K+1]
    std::int32_t* top_ids = nullptr;     // [K+1][top]
    float* top_logprobs = nullptr;       // [K+1][top]
    std::uint32_t top = 0;
};
// Submit only to the supplied stream; no host synchronization, no readback and no
// allocation inside a step. With graphs enabled the controller records draft and
// verify into Forward, and commit_state into Finish, once.
using DeviceStep = std::function<void(cudaStream_t, const RoundView&)>;
struct RoundSteps {
    // Write drafts[0..K) from control[last_token] at control[frontier]. Each step is
    // recorded once per draft width it runs at (RoundView::draft_tokens).
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
    // last_token, commit_tokens) exactly as enqueue_greedy_accept defines them, so
    // enqueue_truncate and commit work unchanged. Unset: the greedy accept kernel.
    std::function<void(cudaStream_t, const AcceptRows&)> accept = {};
};
struct RoundConfig {
    std::uint32_t draft_tokens = 0;   // K, the widest width launch() may choose; zero is ordinary one-token decoding
    std::uint32_t vocabulary = 0;
    std::uint32_t context = 0;        // maximum materialized positions (KV slots)
    std::vector<std::int32_t> stop_tokens;
    bool graphs = true;               // false runs the same steps eagerly
    std::uint32_t max_stops = 0;      // device stop capacity for set_stops; at least stop_tokens.size()
    // Carry each emitted token's log-probability, and top_logprobs (<= core::top_logprobs_limit)
    // ranked alternatives, in the round's one readback (RoundView::logprob rows, then
    // logprobs() / top_logprobs() for Core's Progress spans).
    bool logprobs = false;
    std::uint32_t top_logprobs = 0;
    // Read the round's allowance from, and write its readback block to, mapped pinned host
    // memory in place (a kernel instead of the allowance copy, no readback copy); the
    // completion event follows Finish. Same tokens and state; choose it by measuring rounds
    // both ways (where copies next to kernels cost more than a host-memory access).
    bool zero_copy = false;
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
    // Setup, not per round, in place of start() when the prefill materialized all `frontier`
    // (>= 1) prompt positions and produced the first output `last_token` itself (`previous` is the
    // token at frontier - 1): enqueue_seed's one-column round, so a stateful drafter's catch-up
    // ingests (feature at frontier - 1, last_token) in the first round. The engine has put
    // last_token in its sampling history: control[commit_tokens] is 0. Waits for its kernel.
    void seed(std::uint32_t frontier, std::int32_t last_token, std::int32_t previous);
    // Setup, not per round: replace the device stop list (at most the configured
    // capacity) without recapturing; Core request options can change it per request.
    void set_stops(std::span<const std::int32_t> stops);
    // Setup, not per round: alternatives per token that top_logprobs() returns, at most
    // RoundConfig::top_logprobs (Core's RequestOptions::logprobs); verify keeps writing the
    // configured count and the first `top` of each ranked row are returned.
    void set_top_logprobs(std::uint32_t top);
    // Drop a launched round that will never be committed, e.g. a request Core
    // cancelled after advance(). Waits for its readback; call start() before the
    // next round. Session::close should call it when a round is still pending.
    void abandon();
    // Capture Forward and Finish for RoundConfig::draft_tokens, or for draft_tokens drafts,
    // now instead of on the first launch at that width.
    void prepare();
    void prepare(std::uint32_t draft_tokens);
    // Enqueue one whole round: Forward graph, Finish graph (accept, readback block,
    // commit_state), then one readback of the block. Return value is
    // Progress::readiness. No host fence.
    core::Readiness* launch(std::uint32_t output_tokens);
    // The same round with draft_tokens <= RoundConfig::draft_tokens drafts (for example a
    // width chosen from recent acceptance), over the same buffers: one Forward/Finish pair
    // per width, captured on its first launch.
    core::Readiness* launch(std::uint32_t output_tokens, std::uint32_t draft_tokens);
    // After readiness: the device-chosen tokens. Borrowed until commit/next launch.
    // Rejects a block whose count, tokens, matched count or frontier (the host
    // mirror plus the count) is inconsistent.
    std::span<const std::int32_t> tokens();
    // After readiness: the drafts this round accepted (n_matched; before a Core trim).
    std::uint32_t matched();
    // With RoundConfig::logprobs: storage for Progress::logprobs ([K+1]) and
    // Progress::top_logprobs ([K+1][top], row-major) with deferred tokens. Stable for
    // the controller's lifetime; tokens() fills the first n_emitted rows, which Core
    // reads after ready_tokens(). Empty without logprobs.
    std::span<const float> logprobs() const noexcept;
    std::span<const core::TokenLogprob> top_logprobs() const noexcept;
    // Publish Core's accepted prefix. accepted < n_emitted enqueues the device
    // truncation and reruns commit_state; the host never re-verifies tokens.
    void commit(std::uint32_t accepted);
    // After readiness: Forward (DeviceStage::forward) and Finish (accept) timings.
    std::span<const core::DeviceTiming> timings();

    RoundView view() const noexcept;
    std::uint32_t frontier() const noexcept; // host mirror of committed positions
    std::uint64_t rounds() const noexcept;
    std::uint64_t readbacks() const noexcept;
    bool captured() const noexcept; // at RoundConfig::draft_tokens
    bool captured(std::uint32_t draft_tokens) const noexcept;
    bool pending() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
