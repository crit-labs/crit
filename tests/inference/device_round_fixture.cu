// Synthetic device target for device-round fixtures; reference kernels only.
// The target state S is a hash chain over materialized tokens and its greedy
// prediction is mix(S) % V. The drafter computes the same chain but corrupts
// positions selected by `corrupt`, so rounds see matches and mismatches.
#include "device_round_fixture.h"
#include "greedy_accept.h"

namespace crit::tests::device_round {
using namespace crit::techniques::device_round;
namespace {
__device__ std::uint32_t step_state(std::uint32_t s, std::int32_t token) { return s * 1000003u + std::uint32_t(token) + 7u; }

__global__ void draft(HashTarget t, const std::int32_t* control, std::int32_t* drafts, std::uint32_t k) {
    auto s = *t.state;
    auto token = control[control::last_token];
    const auto frontier = control[control::frontier];
    for(std::uint32_t i = 0; i < k; ++i) {
        s = step_state(s, token);
        auto prediction = std::int32_t(mix(s) % t.vocabulary);
        if(mix(std::uint32_t(frontier) + i + t.corrupt_seed) % t.corrupt_period == 0)
            prediction = (prediction + 1) % std::int32_t(t.vocabulary);
        drafts[i] = prediction;
        token = prediction;
    }
}

__global__ void verify(HashTarget t, const std::int32_t* control, const std::int32_t* drafts,
                       std::int32_t* predictions, std::uint32_t k) {
    auto s = *t.state;
    *t.start = s;
    const auto frontier = control[control::frontier];
    for(std::uint32_t i = 0; i <= k; ++i) {
        const auto token = i ? drafts[i - 1] : control[control::last_token];
        t.history[frontier + std::int32_t(i)] = token; // provisional "KV" column
        s = step_state(s, token);
        t.checkpoints[i] = s;
        predictions[i] = std::int32_t(mix(s) % t.vocabulary);
    }
}

__global__ void commit_state(HashTarget t, const std::int32_t* control) {
    const auto n = control[control::n_emitted];
    *t.state = n ? t.checkpoints[n - 1] : *t.start;
}

__global__ void prefill(HashTarget t, const std::int32_t* tokens, std::uint32_t count) {
    auto s = *t.state;
    for(std::uint32_t i = 0; i < count; ++i) { t.history[i] = tokens[i]; s = step_state(s, tokens[i]); }
    *t.state = s;
}
}

cudaError_t enqueue_draft(const HashTarget& t, const std::int32_t* control, std::int32_t* drafts, std::uint32_t k, cudaStream_t s) {
    draft<<<1, 1, 0, s>>>(t, control, drafts, k);
    return cudaPeekAtLastError();
}
cudaError_t enqueue_verify(const HashTarget& t, const std::int32_t* control, const std::int32_t* drafts,
                           std::int32_t* predictions, std::uint32_t k, cudaStream_t s) {
    verify<<<1, 1, 0, s>>>(t, control, drafts, predictions, k);
    return cudaPeekAtLastError();
}
cudaError_t enqueue_commit_state(const HashTarget& t, const std::int32_t* control, cudaStream_t s) {
    commit_state<<<1, 1, 0, s>>>(t, control);
    return cudaPeekAtLastError();
}
cudaError_t enqueue_prefill(const HashTarget& t, const std::int32_t* tokens, std::uint32_t count, cudaStream_t s) {
    prefill<<<1, 1, 0, s>>>(t, tokens, count);
    return cudaPeekAtLastError();
}
}
