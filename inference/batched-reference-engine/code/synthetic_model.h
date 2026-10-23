// Kernels of a synthetic decoder-only transformer whose rows are batch invariant: every kernel computes a row from
// that row alone in a fixed order, and a projection's K reduction is split the same way for every row count, so a
// request's bits do not depend on which other requests share its step.
//
// Layer: x += O(attention(rope(Q), rope(K), V)) with Q, K, V = QKV(rmsnorm(x)); x += Down(silu(Gate) * Up) with
// Gate, Up = GateUp(rmsnorm(x)). Logits = Head(rmsnorm(x)). The residual stream x is FP32; projections read BF16
// activations and BF16 weights [out][in] and accumulate in FP32 on tensor cores.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace crit::batched_engine {

struct Shape {
    int vocab = 0, hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, ffn = 0;
    float rope_theta = 10000.f;
    std::uint64_t seed = 0;
    int qkv() const { return (heads + 2 * kv_heads) * head_dim; }
    // Weights per layer in elements (QKV, O, GateUp, Down) and for the embedding plus the head.
    std::size_t layer_elements() const {
        return static_cast<std::size_t>(hidden) * (qkv() + static_cast<std::size_t>(heads) * head_dim + 3 * static_cast<std::size_t>(ffn));
    }
    std::size_t elements() const { return layer_elements() * layers + 2 * static_cast<std::size_t>(vocab) * hidden; }
};
// Hidden, ffn, the head width (heads * head_dim) and vocab must be multiples of 64; head_dim 64, 128 or 256.
bool valid(const Shape&);

// Uniform values in [-bound, bound] from (seed, index): the same weights on every device.
cudaError_t fill_uniform(__nv_bfloat16* out, std::size_t count, float bound, std::uint64_t seed, cudaStream_t stream);

// y = x * w, x [rows][k] BF16, w [n][k] BF16 (one output row per weight row), c [rows][n] FP32 stored or added to.
// Each output's K reduction is split into `splits` contiguous ranges reduced in range order; the plan depends on n, k
// and the device, never on rows, so results are bitwise independent of the row count.
struct GemmPlan {
    int n = 0, k = 0, splits = 1, tiles_per_split = 0;
};
GemmPlan plan_gemm(int n, int k, int multiprocessors);
std::size_t gemm_scratch_floats(const GemmPlan& plan, int max_rows);
cudaError_t gemm(const GemmPlan& plan, const __nv_bfloat16* x, int rows, const __nv_bfloat16* w, float* c, bool accumulate,
                 float* scratch, cudaStream_t stream);

// x[r] = embedding[tokens[r]] in FP32.
cudaError_t embed(const __nv_bfloat16* table, int hidden, const std::int32_t* tokens, int rows, float* x, cudaStream_t stream);
// y[i] = x[r] / rms(x[r]) in BF16 for r = index ? index[i] : i.
cudaError_t rmsnorm(const float* x, const std::int32_t* index, int rows, int hidden, __nv_bfloat16* y, cudaStream_t stream);
// Splits QKV rows into BF16 Q, K, V with split-half rotary embedding at positions[r] on Q and K.
cudaError_t qkv_rope(const float* qkv, const std::int32_t* positions, const float* inv_freq, int rows, const Shape& shape,
                     __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v, cudaStream_t stream);
// out[r][j] = silu(gu[r][j]) * gu[r][ffn + j] in BF16.
cudaError_t silu_mul(const float* gu, int rows, int ffn, __nv_bfloat16* out, cudaStream_t stream);
cudaError_t to_bf16(const float* x, std::size_t count, __nv_bfloat16* out, cudaStream_t stream);
// out[r] = the index of row r's largest logit, the lowest index among equal maxima (greedy selection as Core's Sampler
// makes it without penalties or bias).
cudaError_t argmax(const float* logits, int rows, int vocab, std::int32_t* out, cudaStream_t stream);

}  // namespace crit::batched_engine
