#pragma once
#include <cstddef>
#include <cstdint>

namespace crit::core {
struct TraceConfig {
    // Zero disables tracing. Capacity is bounded to 65536 fixed-size events.
    std::size_t capacity = 0;
    // Zero disables slow-step classification; timestamps remain host timings.
    std::uint64_t slow_step_us = 0;
    // Zero disables operator ranges. Otherwise the engine owns a crit::trace::Recorder
    // of this many completed ranges (at most 65536), independent of capacity, bound as
    // the operator sink of the threads that run model callbacks (see crit/trace.h).
    std::size_t operators = 0;
};
// Bounds host scheduler metadata; actual capacity also needs model admission.
inline constexpr std::uint32_t max_request_concurrency = 4096;
enum class BatchPolicy { fair, decode_first };
enum class AdmissionPolicy { fifo, reuse_aware };
struct Profile {
    std::uint32_t context = 32768, prefill_chunk = 1024;
    std::size_t host_limit = 0, device_limit = 0;
    // Generation concurrency is opt-in. Scores join its FIFO as exclusive
    // barriers after generation retirement. Completed, unconsumed handles also
    // occupy the outstanding allowance (max_concurrency + max_pending).
    std::uint32_t max_concurrency = 1, max_pending = 0;
    std::uint32_t pending_timeout_ms = 30000, retirement_timeout_ms = 30000;
    TraceConfig trace = {};
    // Zero preserves separate prefill/decode callbacks. Otherwise require a
    // mixed_batches model and bound total token work per execute() group.
    std::uint32_t max_batch_tokens = 0;
    BatchPolicy batch_policy = BatchPolicy::fair;
    // With decode_first, a prefill waiting this many execute groups is eligible
    // for the oldest-waiter's reserved unit. Other aged prefills take turns.
    std::uint32_t prefill_starvation_steps = 8;
    // Zero preserves FIFO. Otherwise inspect at most this many followers of a
    // deferred head, never crossing a score barrier or a capped skipped request.
    std::uint32_t admission_lookahead = 0, max_admission_bypasses = 8;
    AdmissionPolicy admission_policy = AdmissionPolicy::fifo;
    std::uint32_t admission_starvation_ms = 1000;
    // Zero keeps fixed chunks. Advice may shrink grants, never enlarge them.
    std::uint64_t prefill_target_ns = 0;
    // Opt-in mixed execution pipeline. Depth one per lane unless configured below.
    std::uint32_t max_inflight_batches = 0, batch_timeout_ms = 30000;
    // Optional incremental allocation and recoverable suspension (async mixed batches).
    bool memory_preemption = false;
    // Dependent execution per request, opt-in. Output ceilings reserve unresolved
    // steps without confusing a step's grant with the request's remaining length.
    std::uint32_t max_request_steps = 1, max_step_output_tokens = 1;
};
}
