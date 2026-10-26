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
    control[control::commit_tokens] = std::int32_t(out);
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
    control[control::commit_tokens] = std::int32_t(m);
}

// One block per row; word e of the block from the result, control or log-probability rows, masked by the count.
__global__ void pack(PackRows rows, PackedLayout layout) {
    const auto row = blockIdx.x, k = rows.draft_tokens, k1 = k + 1, top = rows.top;
    const auto* control = rows.control + std::size_t(row) * control::fields;
    const auto* result = rows.result + std::size_t(row) * result_width(k);
    auto* packed = rows.packed + std::size_t(row) * layout.words();
    const auto count = result[0];
    const auto n = std::uint32_t(count < 0 ? 0 : (count > std::int32_t(k1) ? std::int32_t(k1) : count));
    constexpr auto minus_inf = std::int32_t(0xff800000u); // bits of -inf
    for(std::uint32_t e = threadIdx.x; e < layout.words(); e += blockDim.x) {
        std::int32_t word;
        if(e < result_width(k)) word = result[e];
        else if(e == layout.matched()) word = control[control::n_matched];
        else if(e == layout.frontier()) word = control[control::frontier];
        else if(e < layout.top_ids()) {
            const auto c = e - layout.logprob();
            word = c < n ? __float_as_int(rows.logprob[std::size_t(row) * k1 + c]) : minus_inf;
        } else if(e < layout.top_logprobs()) {
            const auto i = e - layout.top_ids();
            word = i / top < n ? rows.top_ids[std::size_t(row) * k1 * top + i] : -1;
        } else {
            const auto i = e - layout.top_logprobs();
            word = i / top < n ? __float_as_int(rows.top_logprobs[std::size_t(row) * k1 * top + i]) : minus_inf;
        }
        packed[e] = word;
    }
}

__global__ void set_budget(std::int32_t* control, const std::int32_t* allowance, std::uint32_t rows) {
    const auto row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < rows) control[std::size_t(row) * control::fields + control::output_budget] = __ldcv(allowance + row);
}

__global__ void seed(std::int32_t* control, std::int32_t* result, std::uint32_t k, std::int32_t frontier,
                     std::int32_t token, std::int32_t previous) {
    for(std::uint32_t i = 0; i < control::fields; ++i) control[i] = 0;
    control[control::frontier] = frontier;
    control[control::last_token] = token;
    control[control::base_frontier] = frontier - 1;
    control[control::base_last] = previous;
    control[control::n_emitted] = 1;
    result[0] = 1;
    result[1] = token;
    for(std::uint32_t i = 1; i < k + 1; ++i) result[1 + i] = -1;
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

cudaError_t enqueue_pack(const PackRows& rows, cudaStream_t stream) {
    if(!rows.rows) return cudaSuccess;
    if(!rows.control || !rows.result || !rows.packed || rows.rows > 0x7fffffffu) return cudaErrorInvalidValue;
    if(rows.top && (!rows.logprob || !rows.top_ids || !rows.top_logprobs)) return cudaErrorInvalidValue;
    const PackedLayout layout{rows.draft_tokens, rows.logprob ? rows.top : 0u, rows.logprob != nullptr};
    pack<<<rows.rows, 128, 0, stream>>>(rows, layout);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_set_budget(std::int32_t* control, const std::int32_t* allowance, std::uint32_t rows, cudaStream_t stream) {
    if(!rows) return cudaSuccess;
    if(!control || !allowance) return cudaErrorInvalidValue;
    constexpr unsigned threads = 128;
    set_budget<<<(rows + threads - 1) / threads, threads, 0, stream>>>(control, allowance, rows);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_seed(std::int32_t* control, std::int32_t* result, std::uint32_t k, std::uint32_t row,
                         std::uint32_t frontier, std::int32_t token, std::int32_t previous, cudaStream_t stream) {
    if(!control || !result || !frontier || frontier > std::uint32_t(INT32_MAX)) return cudaErrorInvalidValue;
    seed<<<1, 1, 0, stream>>>(control + std::size_t(row) * control::fields, result + std::size_t(row) * result_width(k), k,
                              std::int32_t(frontier), token, previous);
    return cudaPeekAtLastError();
}
}
