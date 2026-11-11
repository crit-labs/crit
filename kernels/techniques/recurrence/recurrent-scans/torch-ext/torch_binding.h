#pragma once

#include <optional>

#include <torch/torch.h>

// x [T, channels], weight [channels, width], bias [channels] or none; state [channels, width - 1] in/out (the window
// before row 0 in, the window after the last row out); out [T, channels].
void causal_conv1d(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& x, const torch::Tensor& weight,
                   const std::optional<torch::Tensor>& bias, bool silu);

// q, k [T, k_heads, dk]; v [T, v_heads, dv]; log_decay, beta [T, v_heads]; state [v_heads, dk, dv] in/out;
// out [T, v_heads, dv]. dk 64 or 128, dv a multiple of 64. Prefill: chunked two-pass. Scan: token by token, column
// invariant, optionally writing the state after every token to checkpoints [T, v_heads, dk, dv].
void gated_delta_prefill(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& q, const torch::Tensor& k,
                         const torch::Tensor& v, const torch::Tensor& log_decay, const torch::Tensor& beta, double scale);
// The prefill on tensor cores (FP16 operands, FP32 accumulation and state): dk 64 or 128, dv a multiple of 16.
void gated_delta_prefill_tc(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& q, const torch::Tensor& k,
                            const torch::Tensor& v, const torch::Tensor& log_decay, const torch::Tensor& beta, double scale);
void gated_delta_scan(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& q, const torch::Tensor& k,
                      const torch::Tensor& v, const torch::Tensor& log_decay, const torch::Tensor& beta, double scale,
                      const std::optional<torch::Tensor>& checkpoints);

// x [T, heads, head_dim]; dt [T, heads]; B, C [T, groups, d_state]; A, D [heads]; state [heads, head_dim, d_state]
// in/out; out [T, heads, head_dim]. d_state 64 or 128, head_dim a multiple of 32.
void ssd_prefill(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& x, const torch::Tensor& dt,
                 const torch::Tensor& B, const torch::Tensor& C, const torch::Tensor& A, const std::optional<torch::Tensor>& D);
void ssd_scan(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& x, const torch::Tensor& dt,
              const torch::Tensor& B, const torch::Tensor& C, const torch::Tensor& A, const std::optional<torch::Tensor>& D,
              const std::optional<torch::Tensor>& checkpoints);

// Pools of sequences (recurrent_rows.h): states [layers][sequences][v_heads][dk][dv], stash
// [layers][sequences][2 * half], control [layers][sequences][2 * blocks] (pool_delta), and windows
// [layers][sequences][channels][width - 1], stash [layers][sequences][capacity][channels], pending
// [layers][sequences][blocks] (pool_conv). Rows from meta [rows][4] (first, tokens, sequence, kind) or a round's
// starts / slots and the pool's slot table.
void pool_delta(torch::Tensor& out, torch::Tensor& states, torch::Tensor& stash, torch::Tensor& control,
                const std::optional<torch::Tensor>& scratch, const torch::Tensor& q, const torch::Tensor& k,
                const torch::Tensor& v, const torch::Tensor& log_decay, const torch::Tensor& beta,
                const std::optional<torch::Tensor>& meta, const std::optional<torch::Tensor>& starts,
                const std::optional<torch::Tensor>& slots, const std::optional<torch::Tensor>& table, int64_t rows,
                int64_t kind, int64_t layer, int64_t capacity, double epsilon, double scale, bool normalize_keys,
                bool normalize_queries, bool pipelined);
void pool_delta_replay(torch::Tensor& states, torch::Tensor& stash, torch::Tensor& control,
                       const std::optional<torch::Tensor>& scratch, const std::optional<torch::Tensor>& accepted,
                       const std::optional<torch::Tensor>& meta, const std::optional<torch::Tensor>& slots,
                       const std::optional<torch::Tensor>& table, int64_t rows, int64_t columns, int64_t capacity,
                       int64_t key_heads, int64_t mode);
void pool_conv(torch::Tensor& out, torch::Tensor& window, torch::Tensor& stash, torch::Tensor& pending,
               const std::optional<torch::Tensor>& scratch, const torch::Tensor& x, const torch::Tensor& weight,
               const std::optional<torch::Tensor>& bias, const std::optional<torch::Tensor>& meta,
               const std::optional<torch::Tensor>& starts, const std::optional<torch::Tensor>& slots,
               const std::optional<torch::Tensor>& table, int64_t rows, int64_t kind, int64_t layer, int64_t layers,
               int64_t capacity, bool silu);
void conv_segments(torch::Tensor& out, const torch::Tensor& window_in, torch::Tensor& window_out, const torch::Tensor& x,
                   const torch::Tensor& weight, const std::optional<torch::Tensor>& bias, bool silu);
// A captured prompt chunk's convolution: its count and sequence from meta [5] on the device (recurrent_rows.h).
void pool_conv_prompt(torch::Tensor& out, torch::Tensor& window, const torch::Tensor& stash, torch::Tensor& pending,
                      const torch::Tensor& x, const torch::Tensor& weight, const std::optional<torch::Tensor>& bias,
                      const torch::Tensor& meta, int64_t layer, int64_t capacity, bool silu);
