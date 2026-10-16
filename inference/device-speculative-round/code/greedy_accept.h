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
inline constexpr std::uint32_t fields = 8;
}
// Packed result per row: {n_emitted, tokens[K+1]}; unused slots hold -1.
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr std::uint32_t result_width(std::uint32_t draft_tokens) noexcept { return draft_tokens + 2; }

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
// result, then advances the device frontier by n_emitted and sets last_token.
cudaError_t enqueue_greedy_accept(const AcceptRows&, cudaStream_t);

// Apply a smaller host-accepted count (Core truncation at a stop) to one row:
// frontier = base_frontier + accepted, last_token and the packed count follow.
cudaError_t enqueue_truncate(std::int32_t* control, std::int32_t* result, std::uint32_t draft_tokens,
                             std::uint32_t row, std::uint32_t accepted, cudaStream_t);
}
