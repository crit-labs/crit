// Reference kernels, not for performance. One thread owns one row; the loop is
// the same greedy rule as the host GreedySpeculator.
#include "greedy_accept.h"

namespace crit::techniques::device_round {
namespace {
__device__ bool is_stop(const std::int32_t* stops, std::uint32_t count, std::int32_t token) {
    for(std::uint32_t i = 0; i < count; ++i) if(stops[i] == token) return true;
    return false;
}

__global__ void greedy_accept(AcceptRows rows) {
    const auto row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows.rows) return;
    const auto k = rows.draft_tokens;
    const auto* drafts = rows.drafts + std::size_t(row) * k;
    const auto* predictions = rows.predictions + std::size_t(row) * (k + 1);
    auto* control = rows.control + std::size_t(row) * control::fields;
    auto* result = rows.result + std::size_t(row) * result_width(k);
    auto* tokens = result + 1;
    const std::uint32_t stop_count = rows.device_stop_count ? std::uint32_t(*rows.device_stop_count) : rows.stop_count;
    const auto budget = control[control::output_budget];
    const std::uint32_t remaining = budget > 0 ? std::uint32_t(budget) : 0u;
    std::uint32_t out = 0, matched = 0;
    bool stopped = false;
    while(matched < k && out < remaining && drafts[matched] == predictions[matched]) {
        tokens[out++] = drafts[matched++];
        if(is_stop(rows.stops, stop_count, tokens[out - 1])) { stopped = true; break; }
    }
    if(out < remaining && !stopped) tokens[out++] = predictions[matched];
    for(auto i = out; i < k + 1; ++i) tokens[i] = -1;
    result[0] = std::int32_t(out);
    control[control::n_emitted] = std::int32_t(out);
    control[control::n_matched] = std::int32_t(matched);
    control[control::base_frontier] = control[control::frontier];
    control[control::base_last] = control[control::last_token];
    // Emitting m tokens materializes m inputs: the old pending token and the
    // first m-1 emitted tokens. The last emitted token becomes the next input.
    control[control::frontier] += std::int32_t(out);
    if(out) control[control::last_token] = tokens[out - 1];
}

__global__ void truncate(std::int32_t* control, std::int32_t* result, std::uint32_t k, std::uint32_t accepted) {
    const auto emitted = std::uint32_t(control[control::n_emitted]);
    const auto m = accepted < emitted ? accepted : emitted;
    auto* tokens = result + 1;
    for(auto i = m; i < k + 1; ++i) tokens[i] = -1;
    result[0] = std::int32_t(m);
    control[control::n_emitted] = std::int32_t(m);
    if(control[control::n_matched] > std::int32_t(m)) control[control::n_matched] = std::int32_t(m);
    control[control::frontier] = control[control::base_frontier] + std::int32_t(m);
    control[control::last_token] = m ? tokens[m - 1] : control[control::base_last];
}
}

cudaError_t enqueue_greedy_accept(const AcceptRows& rows, cudaStream_t stream) {
    if(!rows.rows) return cudaSuccess;
    if(!rows.drafts && rows.draft_tokens) return cudaErrorInvalidValue;
    if(!rows.predictions || !rows.control || !rows.result || ((rows.stop_count || rows.device_stop_count) && !rows.stops)) return cudaErrorInvalidValue;
    constexpr unsigned threads = 128;
    greedy_accept<<<(rows.rows + threads - 1) / threads, threads, 0, stream>>>(rows);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_truncate(std::int32_t* control, std::int32_t* result, std::uint32_t k,
                             std::uint32_t row, std::uint32_t accepted, cudaStream_t stream) {
    if(!control || !result) return cudaErrorInvalidValue;
    truncate<<<1, 1, 0, stream>>>(control + std::size_t(row) * control::fields,
                                  result + std::size_t(row) * result_width(k), k, accepted);
    return cudaPeekAtLastError();
}
}
