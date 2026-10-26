#pragma once
// Generic device greedy acceptance for speculative rounds.
// Reference kernels, not for performance: one thread per row.
#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::techniques::device_round {
// Device-resident int32 control block, one per row. The device advances it;
// the host only mirrors what it reads back from the packed result.
namespace control {
inline constexpr std::uint32_t frontier = 0;      // materialized positions
inline constexpr std::uint32_t last_token = 1;    // emitted, not yet materialized
inline constexpr std::uint32_t base_frontier = 2; // frontier at round start
inline constexpr std::uint32_t base_last = 3;     // pending token at round start
inline constexpr std::uint32_t n_emitted = 4;     // tokens emitted by this round
inline constexpr std::uint32_t n_matched = 5;     // accepted draft tokens
inline constexpr std::uint32_t output_budget = 6; // allowance for this round
// Result tokens the sampling history has still to take in (token_sampling::enqueue_commit's count): n_emitted after
// an accept or truncation, 0 after start() or a seed, whose token the engine put in the history when it published it.
inline constexpr std::uint32_t commit_tokens = 7;
inline constexpr std::uint32_t fields = 8;
}
#ifdef __CUDACC__
#define CRIT_DEVICE_ROUND_HD __host__ __device__
#else
#define CRIT_DEVICE_ROUND_HD
#endif
// Packed result per row: {n_emitted, tokens[K+1]}; unused slots hold -1.
CRIT_DEVICE_ROUND_HD constexpr std::uint32_t result_width(std::uint32_t draft_tokens) noexcept { return draft_tokens + 2; }

// Readback block per row (enqueue_pack), int32 words: the packed result {n_emitted, tokens[K+1]}, then n_matched and
// the frontier after the round, then with log-probabilities float logprob[K+1], int32 top_ids[K+1][top] and float
// top_logprobs[K+1][top]. Column c's rows belong to predictions[c], the c-th emitted token when c < n_emitted; columns
// c >= n_emitted hold -1 ids and -inf values. Copying it (or writing it to mapped host memory) is a round's readback.
struct PackedLayout {
    std::uint32_t draft_tokens = 0, top = 0;
    bool logprobs = false;
    CRIT_DEVICE_ROUND_HD constexpr std::uint32_t matched() const noexcept { return draft_tokens + 2; }
    CRIT_DEVICE_ROUND_HD constexpr std::uint32_t frontier() const noexcept { return draft_tokens + 3; }
    CRIT_DEVICE_ROUND_HD constexpr std::uint32_t logprob() const noexcept { return draft_tokens + 4; }
    CRIT_DEVICE_ROUND_HD constexpr std::uint32_t top_ids() const noexcept { return logprob() + draft_tokens + 1; }
    CRIT_DEVICE_ROUND_HD constexpr std::uint32_t top_logprobs() const noexcept { return top_ids() + (draft_tokens + 1) * top; }
    CRIT_DEVICE_ROUND_HD constexpr std::uint32_t words() const noexcept {
        return logprobs ? top_logprobs() + (draft_tokens + 1) * top : logprob();
    }
};

struct AcceptRows {
    const std::int32_t* drafts = nullptr;      // [rows][K]
    const std::int32_t* predictions = nullptr; // [rows][K+1]: argmax after last, d1..di
    const std::int32_t* stops = nullptr;       // shared stop IDs; may be null when count is 0
    std::uint32_t stop_count = 0;
    std::int32_t* control = nullptr;           // [rows][control::fields]
    std::int32_t* result = nullptr;            // [rows][K+2]
    std::uint32_t rows = 0, draft_tokens = 0;
    // Optional device-resident stop count that replaces stop_count, so one captured
    // round serves requests with different stop lists (stops then has capacity).
    const std::int32_t* device_stop_count = nullptr;
};

// Accept the matching draft prefix, then the target correction or bonus, bounded
// by control[output_budget] and ending after an accepted stop. Writes the packed
// result, then advances the device frontier by n_emitted and sets last_token
// (commit_tokens = n_emitted).
cudaError_t enqueue_greedy_accept(const AcceptRows&, cudaStream_t);

// Apply a smaller host-accepted count (Core truncation at a stop) to one row:
// frontier = base_frontier + accepted, last_token, the packed count and commit_tokens follow.
cudaError_t enqueue_truncate(std::int32_t* control, std::int32_t* result, std::uint32_t draft_tokens,
                             std::uint32_t row, std::uint32_t accepted, cudaStream_t);

// The readback block of `rows` rows from the control block and packed result an accept (or truncation) wrote, and
// optionally each column's log-probability rows (for example token-sampling's, written by the verify step).
struct PackRows {
    const std::int32_t* control = nullptr;     // [rows][control::fields]
    const std::int32_t* result = nullptr;      // [rows][K+2]
    std::int32_t* packed = nullptr;            // [rows][PackedLayout{K, top, logprob != nullptr}.words()]; device or mapped host memory
    const float* logprob = nullptr;            // optional [rows][K+1]: log-probability of predictions[c]
    const std::int32_t* top_ids = nullptr;     // [rows][K+1][top] ranked alternatives (with top > 0)
    const float* top_logprobs = nullptr;       // [rows][K+1][top]
    std::uint32_t rows = 0, draft_tokens = 0, top = 0;
};
cudaError_t enqueue_pack(const PackRows&, cudaStream_t);

// control[r][output_budget] = allowance[r] for `rows` rows, read uncached: `allowance` may be mapped pinned host
// memory the host rewrites before each round, so a round needs no host-to-device copy.
cudaError_t enqueue_set_budget(std::int32_t* control, const std::int32_t* allowance, std::uint32_t rows, cudaStream_t);

// Seed one row after a prefill that materialized `frontier` (>= 1) positions and produced the pending first output
// `token` itself: the state a one-column round at frontier - 1 with pending `previous` leaves after emitting `token`
// (base_frontier = frontier - 1, base_last = previous, n_emitted = 1, n_matched = 0, result {1, token, -1...}), so a
// stateful drafter's catch-up ingests (feature at frontier - 1, token) in the first round. commit_tokens = 0: the
// engine put `token` in its sampling history when it published it. A seed is not a round: never truncate it.
cudaError_t enqueue_seed(std::int32_t* control, std::int32_t* result, std::uint32_t draft_tokens, std::uint32_t row,
                         std::uint32_t frontier, std::int32_t token, std::int32_t previous, cudaStream_t);
}
