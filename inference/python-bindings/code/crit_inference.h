// C ABI over Inference's CUDA entries, for engines written in Python (crit_inference.py loads it with ctypes).
// Every buffer is a device address (torch's data_ptr(), cupy's data.ptr) unless a comment says host; every
// stream is a cudaStream_t passed as a pointer (torch's cuda_stream), 0 for the legacy default stream. Launch
// functions return the entry's cudaError_t (0 on success) and only enqueue; flags are ints (0 or 1). Plans are
// opaque CRIT_PLAN_BYTES-byte host buffers, 8-byte aligned, filled by the crit_st_plan* functions.
#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define CRIT_API __declspec(dllexport)
#else
#define CRIT_API __attribute__((visibility("default")))
#endif

#define CRIT_INFERENCE_ABI 1
#define CRIT_PLAN_BYTES 64

#ifdef __cplusplus
extern "C" {
#endif

typedef int crit_status;  // a cudaError_t

CRIT_API int crit_abi_version(void);
CRIT_API const char* crit_error_string(int error);

// Pinned host memory mapped into the device address space; *device is the address kernels read.
CRIT_API crit_status crit_host_alloc_mapped(size_t bytes, void** host, void** device);
CRIT_API crit_status crit_host_free(void* host);
CRIT_API crit_status crit_stream_synchronize(void* stream);

// stored-tile-gemm (crit::stored_tile). activation: 0 SiLU, 1 tanh-GELU, 2 erf-GELU.
CRIT_API int crit_st_fp4_mma_compiled(void);
CRIT_API crit_status crit_st_plan(void* plan, int fp4, int n, int k, int tokens, int wide_tokens, int min_blocks,
                                  int two_row_tiles, size_t* activation_bytes, size_t* token_scale_count,
                                  size_t* partial_floats);
CRIT_API crit_status crit_st_plan_gemv(void* plan, int fp4, int n, int k, int groups, int draft);
CRIT_API crit_status crit_st_plan_a16(void* plan, int fp4, int n, int k, int tokens, int min_blocks, size_t* partial_floats);
CRIT_API crit_status crit_st_repack_fp8(const void* codes, void* tiles, int n, int k, void* stream);
CRIT_API crit_status crit_st_repack_fp4(const void* codes, const void* scales, void* tiles, void* scale_tiles, int n, int k,
                                        void* stream);
CRIT_API crit_status crit_st_repack_fp8_rows(const void* codes, void* tiles, int n, int k, int first, int step, void* stream);
CRIT_API crit_status crit_st_repack_fp4_rows(const void* codes, const void* scales, void* tiles, void* scale_tiles, int n,
                                             int k, int first, int step, void* stream);
CRIT_API crit_status crit_st_place_row_scales(const void* src, void* dst, int n, int first, int step, void* stream);
CRIT_API crit_status crit_st_gemm_fp8(const void* plan, const void* tiles, const void* row_scale, int n, int k,
                                      const void* x, int tokens, float* y, int ldo, int add, void* activations,
                                      float* token_scale, float* partials, void* stream);
CRIT_API crit_status crit_st_gemm_fp4(const void* plan, const void* tiles, const void* scale_tiles,
                                      float weight_global_scale, float input_global_scale, int n, int k, const void* x,
                                      int tokens, float* y, int ldo, int add, void* activations, float* token_scale,
                                      float* partials, void* stream);
CRIT_API crit_status crit_st_gemm_fp8_gated(const void* plan, const void* tiles, const void* row_scale, int n, int k,
                                            const void* x, int tokens, int activation, void* out, int ldo,
                                            void* activations, float* token_scale, float* partials, void* stream);
CRIT_API crit_status crit_st_gemm_fp4_gated(const void* plan, const void* tiles, const void* scale_tiles,
                                            float weight_global_scale, float input_global_scale,
                                            float up_weight_global_scale, int n, int k, const void* x, int tokens,
                                            int activation, void* out, int ldo, void* activations, float* token_scale,
                                            float* partials, void* stream);
CRIT_API crit_status crit_st_gemv_fp8(const void* plan, const void* tiles, const void* row_scale, int n, int k,
                                      const void* x, int ldx, int m, float* y, int ldy, int add, void* stream);
CRIT_API crit_status crit_st_gemv_fp4(const void* plan, const void* tiles, const void* scale_tiles,
                                      float weight_global_scale, float input_global_scale, int n, int k, const void* x,
                                      int ldx, int m, float* y, int ldy, int add, void* stream);
CRIT_API crit_status crit_st_gemv_fp8_gated(const void* plan, const void* tiles, const void* row_scale, int n, int k,
                                            const void* x, int ldx, int m, int activation, void* out, int ldo,
                                            void* stream);
CRIT_API crit_status crit_st_gemv_fp4_gated(const void* plan, const void* tiles, const void* scale_tiles,
                                            float weight_global_scale, float input_global_scale,
                                            float up_weight_global_scale, int n, int k, const void* x, int ldx, int m,
                                            int activation, void* out, int ldo, void* stream);
CRIT_API crit_status crit_st_a16_fp8(const void* plan, const void* tiles, const void* row_scale, int n, int k,
                                     const void* x, int tokens, float* y, int ldo, int add, float* partials, void* stream);
CRIT_API crit_status crit_st_a16_fp4(const void* plan, const void* tiles, const void* scale_tiles,
                                     float weight_global_scale, float input_global_scale, int n, int k, const void* x,
                                     int tokens, float* y, int ldo, int add, float* partials, void* stream);
CRIT_API crit_status crit_st_a16_fp8_gated(const void* plan, const void* tiles, const void* row_scale, int n, int k,
                                           const void* x, int tokens, int activation, void* out, int ldo,
                                           float* partials, void* stream);
CRIT_API crit_status crit_st_a16_fp4_gated(const void* plan, const void* tiles, const void* scale_tiles,
                                           float weight_global_scale, float input_global_scale,
                                           float up_weight_global_scale, int n, int k, const void* x, int tokens,
                                           int activation, void* out, int ldo, float* partials, void* stream);

// recurrent-layer-kernels (crit::recurrent_kernels). Strides count floats.
CRIT_API crit_status crit_rk_conv(const float* x, size_t x_stride, int channels, int width, const float* weight,
                                  const float* bias, int silu, const float* state_in, float* state_out, float* out,
                                  size_t out_stride, float* checkpoints, int rows, void* stream);
CRIT_API int crit_rk_conv_deferred_slots(int channels);
CRIT_API crit_status crit_rk_conv_deferred(const float* x, size_t x_stride, int channels, int width,
                                           const float* weight, const float* bias, int silu, float* out,
                                           size_t out_stride, float* window, float* rows_stash, int capacity,
                                           int32_t* pending, int rows, void* stream);
CRIT_API size_t crit_rk_delta_scratch_floats(int k_heads, int v_heads, int dk, int dv, int rows);
CRIT_API crit_status crit_rk_delta_prefill(int k_heads, int v_heads, int dk, int dv, const float* q, const float* k,
                                           size_t qk_stride, const float* v, size_t v_stride, const float* log_decay,
                                           const float* beta, size_t gate_stride, float scale, int rows, float* state,
                                           float* scratch, float* out, size_t out_stride, void* stream);
CRIT_API crit_status crit_rk_delta_columns(int k_heads, int v_heads, int dk, int dv, const float* q, const float* k,
                                           size_t qk_stride, const float* v, size_t v_stride, const float* log_decay,
                                           const float* beta, size_t gate_stride, float scale, int rows,
                                           const float* state_in, float* state_out, float* checkpoints, float* out,
                                           size_t out_stride, void* stream);
CRIT_API int crit_rk_delta_deferred_slots(int k_heads, int v_heads, int dk, int dv);
CRIT_API size_t crit_rk_delta_stash_floats(int k_heads, int v_heads, int dk, int dv, int capacity);
CRIT_API crit_status crit_rk_delta_columns_deferred(int k_heads, int v_heads, int dk, int dv, const float* q,
                                                    const float* k, size_t qk_stride, const float* v, size_t v_stride,
                                                    const float* log_decay, const float* beta, size_t gate_stride,
                                                    float scale, int rows, float* state, float* stash, int capacity,
                                                    int32_t* control, float* out, size_t out_stride, void* stream);
CRIT_API size_t crit_rk_ssd_scratch_floats(int heads, int groups, int head_dim, int d_state, int rows);
CRIT_API crit_status crit_rk_ssd_prefill(int heads, int groups, int head_dim, int d_state, const float* x,
                                         size_t x_stride, const float* dt, size_t dt_stride, const float* b,
                                         const float* c, size_t bc_stride, const float* a, const float* d, int rows,
                                         float* state, float* scratch, float* out, size_t out_stride, void* stream);
CRIT_API crit_status crit_rk_ssd_columns(int heads, int groups, int head_dim, int d_state, const float* x,
                                         size_t x_stride, const float* dt, size_t dt_stride, const float* b,
                                         const float* c, size_t bc_stride, const float* a, const float* d, int rows,
                                         const float* state_in, float* state_out, float* checkpoints, float* out,
                                         size_t out_stride, void* stream);

// recurrent-state-recovery (crit::techniques::recurrent::cuda). Geometry: heads, keys, values, epsilon,
// query_scale, normalize_keys, normalize_queries; steps: query, key, value, log_decay, beta.
CRIT_API crit_status crit_rr_verify(uint32_t heads, uint32_t keys, uint32_t values, float epsilon, float query_scale,
                                    int normalize_keys, int normalize_queries, const float* start, const float* query,
                                    const float* key, const float* value, const float* log_decay, const float* beta,
                                    uint32_t columns, float* outputs, float* checkpoints, float* work, void* stream);
CRIT_API crit_status crit_rr_select_accepted(uint32_t heads, uint32_t keys, uint32_t values, const float* start,
                                             const float* checkpoints, const int32_t* accepted, uint32_t columns,
                                             float* state, void* stream);
CRIT_API crit_status crit_rr_recompute_accepted(uint32_t heads, uint32_t keys, uint32_t values, float epsilon,
                                                float query_scale, int normalize_keys, int normalize_queries,
                                                const float* start, const float* query, const float* key,
                                                const float* value, const float* log_decay, const float* beta,
                                                const int32_t* accepted, uint32_t columns, float* state, void* stream);
CRIT_API crit_status crit_rr_verify_deferred(uint32_t heads, uint32_t keys, uint32_t values, float epsilon,
                                             float query_scale, int normalize_keys, int normalize_queries, float* state,
                                             const float* query, const float* key, const float* value,
                                             const float* log_decay, const float* beta, uint32_t columns,
                                             int32_t* pending, float* stash_key, float* stash_decay, float* stash_delta,
                                             uint32_t capacity, float* outputs, float* work, void* stream);
// count 0: the single-layer form; otherwise count pending words in each of groups groups, stride ints apart.
CRIT_API crit_status crit_rr_record_accepted(const int32_t* accepted, uint32_t columns, int32_t* pending,
                                             uint32_t count, uint32_t groups, size_t stride, void* stream);

// fp8-kv-attention (crit::fp8_kv). Cache: k, v, k_scale, v_scale, kv_heads, head_dim, ring, capacity.
// pos0_device non-null reads the position from device memory instead of pos0.
CRIT_API crit_status crit_kv_write_bf16(void* k, void* v, float* k_scale, float* v_scale, int kv_heads, int head_dim,
                                        int ring, int capacity, const void* k_rows, const void* v_rows, int stride,
                                        int tokens, int pos0, const int* pos0_device, void* stream);
CRIT_API crit_status crit_kv_write_f32(void* k, void* v, float* k_scale, float* v_scale, int kv_heads, int head_dim,
                                       int ring, int capacity, const float* k_rows, const float* v_rows, int stride,
                                       int tokens, int pos0, const int* pos0_device, void* stream);
CRIT_API size_t crit_kv_prefill_shared_bytes(int head_dim, int fp8_queries);
// fp8_queries 1: prefill_fp8 (E4M3 queries and probabilities).
CRIT_API crit_status crit_kv_prefill(const void* q, int q_stride, void* k, void* v, float* k_scale, float* v_scale,
                                     int kv_heads, int head_dim, int ring, int capacity, float* out, int out_stride,
                                     int tokens, int pos0, int heads, float scale, int window, const float* sinks,
                                     float softcap, int fp8_queries, void* stream);
CRIT_API size_t crit_kv_decode_shared_bytes(int head_dim, int streams);
CRIT_API size_t crit_kv_decode_partial_floats(int kv_heads, int head_dim, int blocks);
CRIT_API crit_status crit_kv_decode(const void* q, int q_stride, void* k, void* v, float* k_scale, float* v_scale,
                                    int kv_heads, int head_dim, int ring, int capacity, float* out, int out_stride,
                                    int tokens, int pos0, const int* pos0_device, int heads, float scale, int blocks,
                                    float* partial, int window, const float* sinks, float softcap,
                                    const uint64_t* tree, int streams, void* stream);

// token-sampling (crit::techniques::token_sampling). Request and Params rows are packed on the host here and
// copied to device memory by the caller. Rows: logits, stride, vocab, requests, columns, request, staged,
// staged_stride, staged_base, position_base, scratch, tree, tree_tokens.
CRIT_API int crit_ts_max_top_k(void);
CRIT_API size_t crit_ts_params_bytes(void);
CRIT_API size_t crit_ts_params_position_offset(void);
CRIT_API crit_status crit_ts_params_pack(void* dst, double temperature, double top_p, int top_k, uint32_t stream_id,
                                         uint64_t seed, uint64_t position);
CRIT_API size_t crit_ts_request_bytes(void);
CRIT_API size_t crit_ts_request_position_offset(void);
CRIT_API crit_status crit_ts_request_pack(void* dst, double temperature, double top_p, double min_p, int32_t top_k,
                                          double repetition, double frequency, double presence, uint32_t min_tokens,
                                          uint32_t stream_id, uint64_t seed, uint64_t position, const int32_t* bias_ids,
                                          const float* bias_values, uint32_t bias_count, const int32_t* suppress_ids,
                                          uint32_t suppress_count, int32_t* counts, const uint8_t* prompt_seen);
CRIT_API size_t crit_ts_sparse_law_bytes(void);
CRIT_API size_t crit_ts_select_scratch_bytes(int vocab, int rows);
CRIT_API crit_status crit_ts_select(const float* logits, int ld, int rows, int vocab, void* params, void* scratch,
                                    int32_t* out, void* stream);
CRIT_API crit_status crit_ts_mask(float* logits, int vocab, int stride, const uint32_t* mask, int mask_stride, int rows,
                                  void* stream);
CRIT_API size_t crit_ts_scratch_floats(uint32_t vocab, uint32_t rows);
// purpose: crit::core::DrawPurpose; sparse may be null; n 0 computes no log-probabilities.
CRIT_API crit_status crit_ts_sample(const float* logits, uint32_t stride, uint32_t vocab, uint32_t requests,
                                    uint32_t columns, const void* request, const int32_t* staged,
                                    uint32_t staged_stride, uint32_t staged_base, uint32_t position_base,
                                    float* scratch, const uint64_t* tree, const int32_t* tree_tokens, uint32_t purpose,
                                    int32_t* out, float* probabilities, void* sparse, uint32_t n, float* chosen,
                                    int32_t* top_ids, float* top_logprobs, void* stream);
CRIT_API crit_status crit_ts_verify(const float* logits, uint32_t stride, uint32_t vocab, uint32_t requests,
                                    uint32_t columns, const void* request, const int32_t* staged,
                                    uint32_t staged_stride, uint32_t staged_base, uint32_t position_base,
                                    float* scratch, const uint64_t* tree, const int32_t* tree_tokens,
                                    const int32_t* drafts, const float* draft_probabilities, const void* draft_laws,
                                    int32_t* predictions, uint32_t n, float* chosen, int32_t* top_ids,
                                    float* top_logprobs, void* stream);
CRIT_API size_t crit_ts_logprob_scratch_bytes(int vocab, int rows, int n);
CRIT_API crit_status crit_ts_logprobs(const float* logits, int ld, int rows, int vocab, const int32_t* chosen, int n,
                                      float* chosen_logprob, int32_t* top_ids, float* top_logprobs, void* scratch,
                                      void* stream);
CRIT_API crit_status crit_ts_commit(void* requests, uint32_t count, uint32_t vocab, const int32_t* tokens,
                                    uint32_t token_stride, const int32_t* emitted, uint32_t emitted_stride,
                                    void* stream);

// device-speculative-round's greedy accept (crit::techniques::device_round).
CRIT_API uint32_t crit_dr_control_fields(void);
// offsets: matched, frontier, logprob, top_ids, top_logprobs, words (6 host uint32 values).
CRIT_API crit_status crit_dr_packed_layout(uint32_t draft_tokens, uint32_t top, int logprobs, uint32_t* offsets);
CRIT_API crit_status crit_dr_greedy_accept(const int32_t* drafts, const int32_t* predictions, const int32_t* stops,
                                           uint32_t stop_count, const int32_t* device_stop_count, int32_t* control,
                                           int32_t* result, uint32_t rows, uint32_t draft_tokens, void* stream);
CRIT_API crit_status crit_dr_truncate(int32_t* control, int32_t* result, uint32_t draft_tokens, uint32_t row,
                                      uint32_t accepted, void* stream);
CRIT_API crit_status crit_dr_pack(const int32_t* control, const int32_t* result, int32_t* packed, const float* logprob,
                                  const int32_t* top_ids, const float* top_logprobs, uint32_t rows,
                                  uint32_t draft_tokens, uint32_t top, void* stream);
CRIT_API crit_status crit_dr_set_budget(int32_t* control, const int32_t* allowance, uint32_t rows, void* stream);
CRIT_API crit_status crit_dr_seed(int32_t* control, int32_t* result, uint32_t draft_tokens, uint32_t row,
                                  uint32_t frontier, int32_t token, int32_t previous, void* stream);

#ifdef __cplusplus
}
#endif
