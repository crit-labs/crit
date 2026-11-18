#pragma once
#include <cstddef>
#include <cstdint>

namespace crit::core {
// Bounds host scheduler metadata; actual capacity also needs model admission.
inline constexpr std::uint32_t max_request_concurrency = 4096;
enum class BatchPolicy { fair, decode_first };
struct Profile {
    std::uint32_t context = 32768, prefill_chunk = 1024;
    std::size_t host_limit = 0, device_limit = 0;
    // Generation concurrency is opt-in. Scores join its FIFO as exclusive
    // barriers after generation retirement. Completed, unconsumed handles also
    // occupy the outstanding allowance (max_concurrency + max_pending).
    std::uint32_t max_concurrency = 1, max_pending = 0;
    std::uint32_t pending_timeout_ms = 30000;
    // Zero preserves separate prefill/decode callbacks. Otherwise require a
    // mixed_batches model and bound total token work per execute() group.
    std::uint32_t max_batch_tokens = 0;
    BatchPolicy batch_policy = BatchPolicy::fair;
    // With decode_first, a prefill waiting this many execute groups is eligible
    // for the oldest-waiter's reserved unit. Other aged prefills take turns.
    std::uint32_t prefill_starvation_steps = 8;
};
}
