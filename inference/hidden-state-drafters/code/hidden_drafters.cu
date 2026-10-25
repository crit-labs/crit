// Reference kernels for hidden-state drafters; see hidden_drafters.h and drafting.py.
#include "hidden_drafters.h"

namespace crit::techniques::hidden_drafters {
namespace {

namespace control = device_round::control;

__global__ void eagle_ingest_kernel(const std::int32_t* control_all, const std::int32_t* result_all, std::uint32_t stride,
                                    std::uint32_t columns, std::int32_t* tokens, std::int32_t* positions, std::uint32_t rows) {
    const std::uint32_t r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= rows || c >= columns) return;
    const std::int32_t* ctl = control_all + std::size_t(r) * control::fields;
    const std::int32_t* result = result_all + std::size_t(r) * stride;
    const std::int32_t n = ctl[control::n_emitted];
    const std::size_t at = std::size_t(r) * columns + c;
    const bool live = std::int32_t(c) < n;
    tokens[at] = live ? result[1 + c] : -1;
    positions[at] = live ? ctl[control::base_frontier] + std::int32_t(c) : -1;
}

__global__ void block_input_kernel(const std::int32_t* control_all, std::int32_t mask, std::uint32_t block,
                                   std::int32_t* tokens, std::int32_t* positions, std::uint32_t rows) {
    const std::uint32_t r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= rows || i >= block) return;
    const std::int32_t* ctl = control_all + std::size_t(r) * control::fields;
    tokens[std::size_t(r) * block + i] = i ? mask : ctl[control::last_token];
    positions[std::size_t(r) * block + i] = ctl[control::frontier] + std::int32_t(i);
}

__global__ void map_kernel(std::int32_t* ids, std::uint32_t count, const std::int64_t* d2t, std::uint32_t vocab) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    const std::int32_t d = ids[i];
    if(d >= 0 && std::uint32_t(d) < vocab) ids[i] = std::int32_t(d + d2t[d]);
}

__global__ void scatter_kernel(const float* q, std::uint32_t draft_vocab, const std::int64_t* d2t, float* out,
                               std::uint32_t target_vocab) {
    const std::uint32_t r = blockIdx.x;
    float* row = out + std::size_t(r) * target_vocab;
    // Zero the row, then place: the offset table maps distinct draft ids to distinct targets.
    for(std::uint32_t t = threadIdx.x; t < target_vocab; t += blockDim.x) row[t] = 0.f;
    __syncthreads();
    for(std::uint32_t d = threadIdx.x; d < draft_vocab; d += blockDim.x) {
        const std::int64_t t = std::int64_t(d) + d2t[d];
        if(t >= 0 && std::uint64_t(t) < target_vocab) row[t] = q[std::size_t(r) * draft_vocab + d];
    }
}

__global__ void truncate_kernel(std::int32_t* drafts, const float* confidence, std::uint32_t k, float threshold, std::uint32_t rows) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= rows) return;
    bool cut = false;
    for(std::uint32_t i = 0; i < k; ++i) {
        cut = cut || !(confidence[std::size_t(r) * k + i] >= threshold);
        if(cut) drafts[std::size_t(r) * k + i] = -1;
    }
}

}  // namespace

cudaError_t enqueue_eagle_ingest(const std::int32_t* control, const std::int32_t* result, std::uint32_t k, std::int32_t* tokens,
                                 std::int32_t* positions, std::uint32_t rows, cudaStream_t stream) {
    return enqueue_eagle_ingest(control, result, device_round::result_width(k), k + 1, tokens, positions, rows, stream);
}

cudaError_t enqueue_eagle_ingest(const std::int32_t* control, const std::int32_t* result, std::uint32_t stride,
                                 std::uint32_t columns, std::int32_t* tokens, std::int32_t* positions, std::uint32_t rows,
                                 cudaStream_t stream) {
    if(!rows) return cudaSuccess;
    if(!control || !result || !tokens || !positions || rows > 65535 || !columns || stride < columns + 1) return cudaErrorInvalidValue;
    constexpr unsigned threads = 64;
    eagle_ingest_kernel<<<dim3((columns + threads - 1) / threads, rows), threads, 0, stream>>>(control, result, stride, columns,
                                                                                               tokens, positions, rows);
    return cudaGetLastError();
}

cudaError_t enqueue_block_input(const std::int32_t* control, std::int32_t mask, std::uint32_t block, std::int32_t* tokens,
                                std::int32_t* positions, std::uint32_t rows, cudaStream_t stream) {
    if(!rows) return cudaSuccess;
    if(!control || !tokens || !positions || block < 2 || mask < 0 || rows > 65535) return cudaErrorInvalidValue;
    constexpr unsigned threads = 64;
    block_input_kernel<<<dim3((block + threads - 1) / threads, rows), threads, 0, stream>>>(control, mask, block, tokens, positions, rows);
    return cudaGetLastError();
}

cudaError_t enqueue_map_tokens(std::int32_t* ids, std::uint32_t count, const std::int64_t* d2t, std::uint32_t vocab, cudaStream_t stream) {
    if(!count) return cudaSuccess;
    if(!ids || !d2t || !vocab) return cudaErrorInvalidValue;
    map_kernel<<<(count + 255) / 256, 256, 0, stream>>>(ids, count, d2t, vocab);
    return cudaGetLastError();
}

cudaError_t enqueue_scatter_law(const float* q, std::uint32_t draft_vocab, const std::int64_t* d2t, float* out,
                                std::uint32_t target_vocab, std::uint32_t rows, cudaStream_t stream) {
    if(!rows) return cudaSuccess;
    if(!q || !d2t || !out || !draft_vocab || target_vocab < draft_vocab) return cudaErrorInvalidValue;
    scatter_kernel<<<rows, 1024, 0, stream>>>(q, draft_vocab, d2t, out, target_vocab);
    return cudaGetLastError();
}

cudaError_t enqueue_truncate_drafts(std::int32_t* drafts, const float* confidence, std::uint32_t k, float threshold,
                                    std::uint32_t rows, cudaStream_t stream) {
    if(!rows || !k) return cudaSuccess;
    if(!drafts || !confidence) return cudaErrorInvalidValue;
    truncate_kernel<<<(rows + 127) / 128, 128, 0, stream>>>(drafts, confidence, k, threshold, rows);
    return cudaGetLastError();
}

}  // namespace crit::techniques::hidden_drafters
