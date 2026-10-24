// CUDA kernels for the recurrent layers of hybrid models: a short causal depthwise convolution with its carried
// window, gated-delta attention (Qwen3-Next / Qwen3.5-style gated DeltaNet, KDA-style per-head gates) and Mamba-2's
// state-space recurrence, each as a chunked two-pass prefill for prompts and a column scan for decode and speculative
// verify. All values are FP32; projections, gates' activations, normalization and the output gate are the engine's.
//
// A column scan evaluates one token after another with the state in registers and the same expression for every
// column, so a column decoded alone equals the same column inside a verify (column invariance), and it can write the
// state after every column (Core's RecurrentCommit::column_checkpoints). The chunked prefills equal the token-by-token
// recurrence up to rounding (they are a different summation), which prompts tolerate; decode and verify use the scan.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace crit::recurrent_kernels {

// ---------------------------------------------------------------------------------------------- causal convolution
// out[t][c] = act(bias[c] + sum_i weight[c][i] * x_{t - (width - 1) + i}[c]), where inputs before row 0 come from
// state[c][0 .. width - 2] (oldest first); afterwards state holds the last width - 1 inputs. act is SiLU when silu.
// With checkpoints, checkpoints[t][c][..] is the state after token t (verify columns). width 2 .. 8.
struct Conv {
    const float* x = nullptr;
    std::size_t x_stride = 0;          // floats between rows
    int channels = 0, width = 0;
    const float* weight = nullptr;     // [channels][width]
    const float* bias = nullptr;       // [channels] or null
    bool silu = true;
    float* state = nullptr;            // [channels][width - 1], updated in place
    float* out = nullptr;
    std::size_t out_stride = 0;
    float* checkpoints = nullptr;      // optional [rows][channels][width - 1]
};
cudaError_t enqueue_conv(const Conv&, int rows, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------- gated delta
// Per value head h with key head kh = h / (v_heads / k_heads), state S[dk][dv] (S[key][value]), and per token the
// normalized key k and query q (dk), value v (dv), log decay l <= 0 and beta in [0, 1]:
//   delta = beta (v - e^l S^T k);  S' = e^l S + k delta^T;  out = scale (e^l S^T q + (q . k) delta)
// which is scale S'^T q in real arithmetic. dk is 64 or 128; dv a multiple of 64.
struct DeltaShape { int k_heads = 0, v_heads = 0, dk = 0, dv = 0; };
struct DeltaInputs {
    const float* q = nullptr;          // row t, key head kh at q + t * qk_stride + kh * dk
    const float* k = nullptr;
    std::size_t qk_stride = 0;
    const float* v = nullptr;          // row t, head h at v + t * v_stride + h * dv
    std::size_t v_stride = 0;
    const float* log_decay = nullptr;  // [t * gate_stride + h]
    const float* beta = nullptr;
    std::size_t gate_stride = 0;
    float scale = 1.f;                 // query scale (1 when q is pre-scaled)
};
// Chunked two-pass prefill over `rows` tokens (64-token chunks): pass 1 builds every chunk's state-independent
// matrices in parallel (T = (I + L)^-1 by forward substitution, W, T diag(beta), the decayed q . k), pass 2 walks the
// chunks in order per (head, 64 value columns) with the state slice in shared memory. state [v_heads][dk][dv] in/out.
std::size_t delta_scratch_floats(const DeltaShape&, int rows);
cudaError_t enqueue_delta_prefill(const DeltaShape&, const DeltaInputs&, int rows, float* state, float* scratch, float* out,
                                  std::size_t out_stride, cudaStream_t stream);
// Column scan: `rows` tokens one after another from state_in (which may alias state_out), writing outputs and, with
// checkpoints, the state after each token ([rows][v_heads][dk][dv]).
cudaError_t enqueue_delta_columns(const DeltaShape&, const DeltaInputs&, int rows, const float* state_in, float* state_out,
                                  float* checkpoints, float* out, std::size_t out_stride, cudaStream_t stream);

// ---------------------------------------------------------------------------------------------- Mamba-2
// Per head h in group g = h / (heads / groups), state H[p][n] (head_dim x d_state), and per token the input x (p),
// dt > 0 (after the engine's softplus and bias), B and C (n, shared by the group's heads), with A[h] < 0 and D[h]:
//   H' = e^{dt A} H + dt x B^T;  y = H' C + D x
// d_state is 64 or 128; head_dim a multiple of 32.
struct SsdShape { int heads = 0, groups = 0, head_dim = 0, d_state = 0; };
struct SsdInputs {
    const float* x = nullptr;          // row t, head h at x + t * x_stride + h * head_dim
    std::size_t x_stride = 0;
    const float* dt = nullptr;         // [t * dt_stride + h]
    std::size_t dt_stride = 0;
    const float* B = nullptr;          // row t, group g at B + t * bc_stride + g * d_state (C likewise)
    const float* C = nullptr;
    std::size_t bc_stride = 0;
    const float* A = nullptr;          // [heads]
    const float* D = nullptr;          // [heads] or null
};
// Chunked two-pass prefill (64-token chunks): pass 1 the decayed C . B matrix and cumulative decays of every chunk in
// parallel, pass 2 the chunks in order per (head, 32 rows of head_dim). state [heads][head_dim][d_state] in/out.
std::size_t ssd_scratch_floats(const SsdShape&, int rows);
cudaError_t enqueue_ssd_prefill(const SsdShape&, const SsdInputs&, int rows, float* state, float* scratch, float* out,
                                std::size_t out_stride, cudaStream_t stream);
cudaError_t enqueue_ssd_columns(const SsdShape&, const SsdInputs&, int rows, const float* state_in, float* state_out,
                                float* checkpoints, float* out, std::size_t out_stride, cudaStream_t stream);

}  // namespace crit::recurrent_kernels
