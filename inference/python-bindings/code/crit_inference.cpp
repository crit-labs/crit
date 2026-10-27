// The C ABI of crit_inference.h: each function rebuilds the entry's C++ argument structs and forwards the call.
#include "crit_inference.h"

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "fp8_kv_attention.h"
#include "greedy_accept.h"
#include "recurrent_cuda.h"
#include "recurrent_kernels.h"
#include "sampling.h"
#include "stored_tile_gemm.h"

namespace st = crit::stored_tile;
namespace rk = crit::recurrent_kernels;
namespace rr = crit::techniques::recurrent::cuda;
namespace fk = crit::fp8_kv;
namespace ts = crit::techniques::token_sampling;
namespace dr = crit::techniques::device_round;

namespace {
template <typename Plan> constexpr bool fits_plan_buffer = sizeof(Plan) <= CRIT_PLAN_BYTES && alignof(Plan) <= 8 &&
                                                           std::is_trivially_copyable_v<Plan>;
static_assert(fits_plan_buffer<st::Plan> && fits_plan_buffer<st::GemvPlan> && fits_plan_buffer<st::A16Plan>);

cudaStream_t S(void* stream) { return static_cast<cudaStream_t>(stream); }
const std::uint8_t* B(const void* p) { return static_cast<const std::uint8_t*>(p); }
std::uint8_t* B(void* p) { return static_cast<std::uint8_t*>(p); }
const __nv_bfloat16* H(const void* p) { return static_cast<const __nv_bfloat16*>(p); }
__nv_bfloat16* H(void* p) { return static_cast<__nv_bfloat16*>(p); }

// A plan buffer holds a copy of the entry's plan; a launch reads it back by value.
template <typename Plan> Plan plan_of(const void* buffer) {
    Plan p;
    std::memcpy(&p, buffer, sizeof p);
    return p;
}
template <typename Plan> void store(void* buffer, const Plan& p) {
    std::memset(buffer, 0, CRIT_PLAN_BYTES);
    std::memcpy(buffer, &p, sizeof p);
}
bool activation_ok(int activation) { return activation >= 0 && activation <= 2; }
st::Activation A(int activation) { return st::Activation(activation); }
st::Fp8Weights fp8(const void* tiles, const void* row_scale, int n, int k) { return {B(tiles), H(row_scale), n, k}; }
st::Fp4Weights fp4(const void* tiles, const void* scale_tiles, float wgs, float igs, int n, int k) {
    return {B(tiles), B(scale_tiles), wgs, igs, n, k};
}
st::Scratch scratch_of(void* activations, float* token_scale, float* partials) { return {activations, token_scale, partials}; }

rk::Conv conv(const float* x, size_t x_stride, int channels, int width, const float* weight, const float* bias, int silu,
              float* out, size_t out_stride) {
    rk::Conv c;
    c.x = x; c.x_stride = x_stride; c.channels = channels; c.width = width; c.weight = weight; c.bias = bias;
    c.silu = silu != 0; c.out = out; c.out_stride = out_stride;
    return c;
}
rk::DeltaInputs delta(const float* q, const float* k, size_t qk_stride, const float* v, size_t v_stride,
                      const float* log_decay, const float* beta, size_t gate_stride, float scale) {
    return {q, k, qk_stride, v, v_stride, log_decay, beta, gate_stride, scale};
}
rk::SsdInputs ssd(const float* x, size_t x_stride, const float* dt, size_t dt_stride, const float* b, const float* c,
                  size_t bc_stride, const float* a, const float* d) {
    return {x, x_stride, dt, dt_stride, b, c, bc_stride, a, d};
}

rr::DeviceGeometry geometry(uint32_t heads, uint32_t keys, uint32_t values, float epsilon, float query_scale,
                            int normalize_keys, int normalize_queries) {
    return {heads, keys, values, epsilon, query_scale, normalize_keys != 0, normalize_queries != 0};
}

fk::Cache cache(void* k, void* v, float* k_scale, float* v_scale, int kv_heads, int head_dim, int ring, int capacity) {
    fk::Cache c;
    c.k = B(k); c.v = B(v); c.k_scale = k_scale; c.v_scale = v_scale;
    c.kv_heads = kv_heads; c.head_dim = head_dim; c.ring = ring; c.capacity = capacity;
    return c;
}

ts::Rows rows_of(const float* logits, uint32_t stride, uint32_t vocab, uint32_t requests, uint32_t columns,
              const void* request, const int32_t* staged, uint32_t staged_stride, uint32_t staged_base,
              uint32_t position_base, float* scratch, const uint64_t* tree, const int32_t* tree_tokens) {
    ts::Rows r;
    r.logits = logits; r.stride = stride; r.vocab = vocab; r.requests = requests; r.columns = columns;
    r.request = static_cast<const ts::Request*>(request); r.staged = staged; r.staged_stride = staged_stride;
    r.staged_base = staged_base; r.position_base = position_base; r.scratch = scratch; r.tree = tree;
    r.tree_tokens = tree_tokens;
    return r;
}
}  // namespace

extern "C" {

int crit_abi_version(void) { return CRIT_INFERENCE_ABI; }
const char* crit_error_string(int error) { return cudaGetErrorString(cudaError_t(error)); }

crit_status crit_host_alloc_mapped(size_t bytes, void** host, void** device) {
    if(!host || !device) return cudaErrorInvalidValue;
    *host = *device = nullptr;
    void* p = nullptr;
    if(const cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable); e != cudaSuccess) return e;
    if(const cudaError_t e = cudaHostGetDevicePointer(device, p, 0); e != cudaSuccess) {
        cudaFreeHost(p);
        return e;
    }
    *host = p;
    return cudaSuccess;
}
crit_status crit_host_free(void* host) { return cudaFreeHost(host); }
crit_status crit_stream_synchronize(void* stream) { return cudaStreamSynchronize(S(stream)); }

// ---------------------------------------------------------------- stored-tile-gemm
int crit_st_fp4_mma_compiled(void) { return st::fp4_mma_compiled() ? 1 : 0; }
crit_status crit_st_plan(void* plan, int fp4, int n, int k, int tokens, int wide_tokens, int min_blocks, int two_row_tiles,
                         size_t* activation_bytes, size_t* token_scale_count, size_t* partial_floats) {
    if(!plan) return cudaErrorInvalidValue;
    const st::Plan p = st::plan(fp4 != 0, n, k, tokens, wide_tokens, min_blocks, two_row_tiles != 0);
    store(plan, p);
    if(activation_bytes) *activation_bytes = p.activation_bytes;
    if(token_scale_count) *token_scale_count = p.token_scale_count;
    if(partial_floats) *partial_floats = p.partial_floats;
    return cudaSuccess;
}
crit_status crit_st_plan_gemv(void* plan, int fp4, int n, int k, int groups, int draft) {
    if(!plan) return cudaErrorInvalidValue;
    st::GemvPlan p = st::plan_gemv(fp4 != 0, n, k, groups);
    p.role = draft ? st::KernelRole::draft : st::KernelRole::target;
    store(plan, p);
    return cudaSuccess;
}
crit_status crit_st_plan_a16(void* plan, int fp4, int n, int k, int tokens, int min_blocks, size_t* partial_floats) {
    if(!plan) return cudaErrorInvalidValue;
    const st::A16Plan p = st::plan_a16(fp4 != 0, n, k, tokens, min_blocks);
    store(plan, p);
    if(partial_floats) *partial_floats = p.partial_floats;
    return cudaSuccess;
}
crit_status crit_st_repack_fp8(const void* codes, void* tiles, int n, int k, void* stream) {
    return st::enqueue_repack_fp8(B(codes), B(tiles), n, k, S(stream));
}
crit_status crit_st_repack_fp4(const void* codes, const void* scales, void* tiles, void* scale_tiles, int n, int k,
                               void* stream) {
    return st::enqueue_repack_fp4(B(codes), B(scales), B(tiles), B(scale_tiles), n, k, S(stream));
}
crit_status crit_st_repack_fp8_rows(const void* codes, void* tiles, int n, int k, int first, int step, void* stream) {
    return st::enqueue_repack_fp8_rows(B(codes), B(tiles), n, k, first, step, S(stream));
}
crit_status crit_st_repack_fp4_rows(const void* codes, const void* scales, void* tiles, void* scale_tiles, int n, int k,
                                    int first, int step, void* stream) {
    return st::enqueue_repack_fp4_rows(B(codes), B(scales), B(tiles), B(scale_tiles), n, k, first, step, S(stream));
}
crit_status crit_st_place_row_scales(const void* src, void* dst, int n, int first, int step, void* stream) {
    return st::enqueue_place_row_scales(H(src), H(dst), n, first, step, S(stream));
}
crit_status crit_st_gemm_fp8(const void* plan, const void* tiles, const void* row_scale, int n, int k, const void* x,
                             int tokens, float* y, int ldo, int add, void* activations, float* token_scale,
                             float* partials, void* stream) {
    if(!plan) return cudaErrorInvalidValue;
    return st::enqueue_fp8(fp8(tiles, row_scale, n, k), H(x), tokens, y, ldo, add != 0, plan_of<st::Plan>(plan),
                           scratch_of(activations, token_scale, partials), S(stream));
}
crit_status crit_st_gemm_fp4(const void* plan, const void* tiles, const void* scale_tiles, float weight_global_scale,
                             float input_global_scale, int n, int k, const void* x, int tokens, float* y, int ldo, int add,
                             void* activations, float* token_scale, float* partials, void* stream) {
    if(!plan) return cudaErrorInvalidValue;
    return st::enqueue_fp4(fp4(tiles, scale_tiles, weight_global_scale, input_global_scale, n, k), H(x), tokens, y, ldo,
                           add != 0, plan_of<st::Plan>(plan), scratch_of(activations, token_scale, partials), S(stream));
}
crit_status crit_st_gemm_fp8_gated(const void* plan, const void* tiles, const void* row_scale, int n, int k, const void* x,
                                   int tokens, int activation, void* out, int ldo, void* activations, float* token_scale,
                                   float* partials, void* stream) {
    if(!plan || !activation_ok(activation)) return cudaErrorInvalidValue;
    return st::enqueue_fp8_gated(fp8(tiles, row_scale, n, k), H(x), tokens, A(activation), H(out), ldo,
                                 plan_of<st::Plan>(plan), scratch_of(activations, token_scale, partials), S(stream));
}
crit_status crit_st_gemm_fp4_gated(const void* plan, const void* tiles, const void* scale_tiles, float weight_global_scale,
                                   float input_global_scale, float up_weight_global_scale, int n, int k, const void* x,
                                   int tokens, int activation, void* out, int ldo, void* activations, float* token_scale,
                                   float* partials, void* stream) {
    if(!plan || !activation_ok(activation)) return cudaErrorInvalidValue;
    return st::enqueue_fp4_gated(fp4(tiles, scale_tiles, weight_global_scale, input_global_scale, n, k),
                                 up_weight_global_scale, H(x), tokens, A(activation), H(out), ldo, plan_of<st::Plan>(plan),
                                 scratch_of(activations, token_scale, partials), S(stream));
}
crit_status crit_st_gemv_fp8(const void* plan, const void* tiles, const void* row_scale, int n, int k, const void* x,
                             int ldx, int m, float* y, int ldy, int add, void* stream) {
    if(!plan) return cudaErrorInvalidValue;
    return st::enqueue_gemv_fp8(fp8(tiles, row_scale, n, k), H(x), ldx, m, y, ldy, add != 0, plan_of<st::GemvPlan>(plan),
                                S(stream));
}
crit_status crit_st_gemv_fp4(const void* plan, const void* tiles, const void* scale_tiles, float weight_global_scale,
                             float input_global_scale, int n, int k, const void* x, int ldx, int m, float* y, int ldy,
                             int add, void* stream) {
    if(!plan) return cudaErrorInvalidValue;
    return st::enqueue_gemv_fp4(fp4(tiles, scale_tiles, weight_global_scale, input_global_scale, n, k), H(x), ldx, m, y,
                                ldy, add != 0, plan_of<st::GemvPlan>(plan), S(stream));
}
crit_status crit_st_gemv_fp8_gated(const void* plan, const void* tiles, const void* row_scale, int n, int k, const void* x,
                                   int ldx, int m, int activation, void* out, int ldo, void* stream) {
    if(!plan || !activation_ok(activation)) return cudaErrorInvalidValue;
    return st::enqueue_gemv_fp8_gated(fp8(tiles, row_scale, n, k), H(x), ldx, m, A(activation), H(out), ldo,
                                      plan_of<st::GemvPlan>(plan), S(stream));
}
crit_status crit_st_gemv_fp4_gated(const void* plan, const void* tiles, const void* scale_tiles, float weight_global_scale,
                                   float input_global_scale, float up_weight_global_scale, int n, int k, const void* x,
                                   int ldx, int m, int activation, void* out, int ldo, void* stream) {
    if(!plan || !activation_ok(activation)) return cudaErrorInvalidValue;
    return st::enqueue_gemv_fp4_gated(fp4(tiles, scale_tiles, weight_global_scale, input_global_scale, n, k),
                                      up_weight_global_scale, H(x), ldx, m, A(activation), H(out), ldo,
                                      plan_of<st::GemvPlan>(plan), S(stream));
}
crit_status crit_st_a16_fp8(const void* plan, const void* tiles, const void* row_scale, int n, int k, const void* x,
                            int tokens, float* y, int ldo, int add, float* partials, void* stream) {
    if(!plan) return cudaErrorInvalidValue;
    return st::enqueue_fp8_a16(fp8(tiles, row_scale, n, k), H(x), tokens, y, ldo, add != 0, plan_of<st::A16Plan>(plan),
                               partials, S(stream));
}
crit_status crit_st_a16_fp4(const void* plan, const void* tiles, const void* scale_tiles, float weight_global_scale,
                            float input_global_scale, int n, int k, const void* x, int tokens, float* y, int ldo, int add,
                            float* partials, void* stream) {
    if(!plan) return cudaErrorInvalidValue;
    return st::enqueue_fp4_a16(fp4(tiles, scale_tiles, weight_global_scale, input_global_scale, n, k), H(x), tokens, y,
                               ldo, add != 0, plan_of<st::A16Plan>(plan), partials, S(stream));
}
crit_status crit_st_a16_fp8_gated(const void* plan, const void* tiles, const void* row_scale, int n, int k, const void* x,
                                  int tokens, int activation, void* out, int ldo, float* partials, void* stream) {
    if(!plan || !activation_ok(activation)) return cudaErrorInvalidValue;
    return st::enqueue_fp8_a16_gated(fp8(tiles, row_scale, n, k), H(x), tokens, A(activation), H(out), ldo,
                                     plan_of<st::A16Plan>(plan), partials, S(stream));
}
crit_status crit_st_a16_fp4_gated(const void* plan, const void* tiles, const void* scale_tiles, float weight_global_scale,
                                  float input_global_scale, float up_weight_global_scale, int n, int k, const void* x,
                                  int tokens, int activation, void* out, int ldo, float* partials, void* stream) {
    if(!plan || !activation_ok(activation)) return cudaErrorInvalidValue;
    return st::enqueue_fp4_a16_gated(fp4(tiles, scale_tiles, weight_global_scale, input_global_scale, n, k),
                                     up_weight_global_scale, H(x), tokens, A(activation), H(out), ldo,
                                     plan_of<st::A16Plan>(plan), partials, S(stream));
}

// ---------------------------------------------------------------- recurrent-layer-kernels
crit_status crit_rk_conv(const float* x, size_t x_stride, int channels, int width, const float* weight, const float* bias,
                         int silu, const float* state_in, float* state_out, float* out, size_t out_stride,
                         float* checkpoints, int rows, void* stream) {
    rk::Conv c = conv(x, x_stride, channels, width, weight, bias, silu, out, out_stride);
    c.state_in = state_in; c.state_out = state_out; c.checkpoints = checkpoints;
    return rk::enqueue_conv(c, rows, S(stream));
}
int crit_rk_conv_deferred_slots(int channels) { return rk::conv_deferred_slots(channels); }
crit_status crit_rk_conv_deferred(const float* x, size_t x_stride, int channels, int width, const float* weight,
                                  const float* bias, int silu, float* out, size_t out_stride, float* window,
                                  float* rows_stash, int capacity, int32_t* pending, int rows, void* stream) {
    const rk::ConvDeferred d{window, rows_stash, capacity, pending};
    return rk::enqueue_conv_deferred(conv(x, x_stride, channels, width, weight, bias, silu, out, out_stride), d, rows,
                                     S(stream));
}
size_t crit_rk_delta_scratch_floats(int k_heads, int v_heads, int dk, int dv, int rows) {
    return rk::delta_scratch_floats({k_heads, v_heads, dk, dv}, rows);
}
crit_status crit_rk_delta_prefill(int k_heads, int v_heads, int dk, int dv, const float* q, const float* k, size_t qk_stride,
                                  const float* v, size_t v_stride, const float* log_decay, const float* beta,
                                  size_t gate_stride, float scale, int rows, float* state, float* scratch, float* out,
                                  size_t out_stride, void* stream) {
    return rk::enqueue_delta_prefill({k_heads, v_heads, dk, dv},
                                     delta(q, k, qk_stride, v, v_stride, log_decay, beta, gate_stride, scale), rows, state,
                                     scratch, out, out_stride, S(stream));
}
crit_status crit_rk_delta_columns(int k_heads, int v_heads, int dk, int dv, const float* q, const float* k, size_t qk_stride,
                                  const float* v, size_t v_stride, const float* log_decay, const float* beta,
                                  size_t gate_stride, float scale, int rows, const float* state_in, float* state_out,
                                  float* checkpoints, float* out, size_t out_stride, void* stream) {
    return rk::enqueue_delta_columns({k_heads, v_heads, dk, dv},
                                     delta(q, k, qk_stride, v, v_stride, log_decay, beta, gate_stride, scale), rows,
                                     state_in, state_out, checkpoints, out, out_stride, S(stream));
}
int crit_rk_delta_deferred_slots(int k_heads, int v_heads, int dk, int dv) {
    return rk::delta_deferred_slots({k_heads, v_heads, dk, dv});
}
size_t crit_rk_delta_stash_floats(int k_heads, int v_heads, int dk, int dv, int capacity) {
    return rk::delta_stash_floats({k_heads, v_heads, dk, dv}, capacity);
}
crit_status crit_rk_delta_columns_deferred(int k_heads, int v_heads, int dk, int dv, const float* q, const float* k,
                                           size_t qk_stride, const float* v, size_t v_stride, const float* log_decay,
                                           const float* beta, size_t gate_stride, float scale, int rows, float* state,
                                           float* stash, int capacity, int32_t* control, float* out, size_t out_stride,
                                           void* stream) {
    const rk::DeltaDeferred d{stash, capacity, control};
    return rk::enqueue_delta_columns_deferred({k_heads, v_heads, dk, dv},
                                              delta(q, k, qk_stride, v, v_stride, log_decay, beta, gate_stride, scale),
                                              rows, state, d, out, out_stride, S(stream));
}
size_t crit_rk_ssd_scratch_floats(int heads, int groups, int head_dim, int d_state, int rows) {
    return rk::ssd_scratch_floats({heads, groups, head_dim, d_state}, rows);
}
crit_status crit_rk_ssd_prefill(int heads, int groups, int head_dim, int d_state, const float* x, size_t x_stride,
                                const float* dt, size_t dt_stride, const float* b, const float* c, size_t bc_stride,
                                const float* a, const float* d, int rows, float* state, float* scratch, float* out,
                                size_t out_stride, void* stream) {
    return rk::enqueue_ssd_prefill({heads, groups, head_dim, d_state}, ssd(x, x_stride, dt, dt_stride, b, c, bc_stride, a, d),
                                   rows, state, scratch, out, out_stride, S(stream));
}
crit_status crit_rk_ssd_columns(int heads, int groups, int head_dim, int d_state, const float* x, size_t x_stride,
                                const float* dt, size_t dt_stride, const float* b, const float* c, size_t bc_stride,
                                const float* a, const float* d, int rows, const float* state_in, float* state_out,
                                float* checkpoints, float* out, size_t out_stride, void* stream) {
    return rk::enqueue_ssd_columns({heads, groups, head_dim, d_state}, ssd(x, x_stride, dt, dt_stride, b, c, bc_stride, a, d),
                                   rows, state_in, state_out, checkpoints, out, out_stride, S(stream));
}

// ---------------------------------------------------------------- recurrent-state-recovery
crit_status crit_rr_verify(uint32_t heads, uint32_t keys, uint32_t values, float epsilon, float query_scale,
                           int normalize_keys, int normalize_queries, const float* start, const float* query,
                           const float* key, const float* value, const float* log_decay, const float* beta,
                           uint32_t columns, float* outputs, float* checkpoints, float* work, void* stream) {
    return rr::enqueue_verify(geometry(heads, keys, values, epsilon, query_scale, normalize_keys, normalize_queries), start,
                              {query, key, value, log_decay, beta}, columns, outputs, checkpoints, work, S(stream));
}
crit_status crit_rr_select_accepted(uint32_t heads, uint32_t keys, uint32_t values, const float* start,
                                    const float* checkpoints, const int32_t* accepted, uint32_t columns, float* state,
                                    void* stream) {
    rr::DeviceGeometry g;
    g.heads = heads; g.keys = keys; g.values = values;
    return rr::enqueue_select_accepted(g, start, checkpoints, accepted, columns, state, S(stream));
}
crit_status crit_rr_recompute_accepted(uint32_t heads, uint32_t keys, uint32_t values, float epsilon, float query_scale,
                                       int normalize_keys, int normalize_queries, const float* start, const float* query,
                                       const float* key, const float* value, const float* log_decay, const float* beta,
                                       const int32_t* accepted, uint32_t columns, float* state, void* stream) {
    return rr::enqueue_recompute_accepted(geometry(heads, keys, values, epsilon, query_scale, normalize_keys,
                                                   normalize_queries),
                                          start, {query, key, value, log_decay, beta}, accepted, columns, state, S(stream));
}
crit_status crit_rr_verify_deferred(uint32_t heads, uint32_t keys, uint32_t values, float epsilon, float query_scale,
                                    int normalize_keys, int normalize_queries, float* state, const float* query,
                                    const float* key, const float* value, const float* log_decay, const float* beta,
                                    uint32_t columns, int32_t* pending, float* stash_key, float* stash_decay,
                                    float* stash_delta, uint32_t capacity, float* outputs, float* work, void* stream) {
    return rr::enqueue_verify_deferred(geometry(heads, keys, values, epsilon, query_scale, normalize_keys, normalize_queries),
                                       state, {query, key, value, log_decay, beta}, columns, pending,
                                       {stash_key, stash_decay, stash_delta, capacity}, outputs, work, S(stream));
}
crit_status crit_rr_record_accepted(const int32_t* accepted, uint32_t columns, int32_t* pending, uint32_t count,
                                    uint32_t groups, size_t stride, void* stream) {
    if(count == 0) return rr::enqueue_record_accepted(accepted, columns, pending, S(stream));
    return rr::enqueue_record_accepted(accepted, columns, pending, count, groups, stride, S(stream));
}

// ---------------------------------------------------------------- fp8-kv-attention
crit_status crit_kv_write_bf16(void* k, void* v, float* k_scale, float* v_scale, int kv_heads, int head_dim, int ring,
                               int capacity, const void* k_rows, const void* v_rows, int stride, int tokens, int pos0,
                               const int* pos0_device, void* stream) {
    const fk::Cache c = cache(k, v, k_scale, v_scale, kv_heads, head_dim, ring, capacity);
    if(pos0_device) return fk::write(c, H(k_rows), H(v_rows), stride, tokens, pos0_device, S(stream));
    return fk::write(c, H(k_rows), H(v_rows), stride, tokens, pos0, S(stream));
}
crit_status crit_kv_write_f32(void* k, void* v, float* k_scale, float* v_scale, int kv_heads, int head_dim, int ring,
                              int capacity, const float* k_rows, const float* v_rows, int stride, int tokens, int pos0,
                              const int* pos0_device, void* stream) {
    const fk::Cache c = cache(k, v, k_scale, v_scale, kv_heads, head_dim, ring, capacity);
    if(pos0_device) return fk::write(c, k_rows, v_rows, stride, tokens, pos0_device, S(stream));
    return fk::write(c, k_rows, v_rows, stride, tokens, pos0, S(stream));
}
size_t crit_kv_prefill_shared_bytes(int head_dim, int fp8_queries) {
    return fp8_queries ? fk::prefill_fp8_shared_bytes(head_dim) : fk::prefill_shared_bytes(head_dim);
}
crit_status crit_kv_prefill(const void* q, int q_stride, void* k, void* v, float* k_scale, float* v_scale, int kv_heads,
                            int head_dim, int ring, int capacity, float* out, int out_stride, int tokens, int pos0,
                            int heads, float scale, int window, const float* sinks, float softcap, int fp8_queries,
                            void* stream) {
    fk::Prefill p;
    p.q = H(q); p.q_stride = q_stride; p.cache = cache(k, v, k_scale, v_scale, kv_heads, head_dim, ring, capacity);
    p.out = out; p.out_stride = out_stride; p.T = tokens; p.pos0 = pos0; p.heads = heads; p.scale = scale;
    p.window = window; p.sinks = sinks; p.softcap = softcap;
    return fp8_queries ? fk::prefill_fp8(p, S(stream)) : fk::prefill(p, S(stream));
}
size_t crit_kv_decode_shared_bytes(int head_dim, int streams) { return fk::decode_shared_bytes(head_dim, streams); }
size_t crit_kv_decode_partial_floats(int kv_heads, int head_dim, int blocks) {
    return fk::decode_partial_floats(kv_heads, head_dim, blocks);
}
crit_status crit_kv_decode(const void* q, int q_stride, void* k, void* v, float* k_scale, float* v_scale, int kv_heads,
                           int head_dim, int ring, int capacity, float* out, int out_stride, int tokens, int pos0,
                           const int* pos0_device, int heads, float scale, int blocks, float* partial, int window,
                           const float* sinks, float softcap, const uint64_t* tree, int streams, void* stream) {
    fk::Decode d;
    d.q = H(q); d.q_stride = q_stride; d.cache = cache(k, v, k_scale, v_scale, kv_heads, head_dim, ring, capacity);
    d.out = out; d.out_stride = out_stride; d.T = tokens; d.pos0 = pos0; d.pos0_device = pos0_device; d.heads = heads;
    d.scale = scale; d.blocks = blocks; d.partial = partial; d.window = window; d.sinks = sinks; d.softcap = softcap;
    d.tree = tree; d.streams = streams;
    return fk::decode(d, S(stream));
}

// ---------------------------------------------------------------- token-sampling
int crit_ts_max_top_k(void) { return ts::kMaxTopK; }
size_t crit_ts_params_bytes(void) { return sizeof(ts::Params); }
size_t crit_ts_params_position_offset(void) { return offsetof(ts::Params, position); }
crit_status crit_ts_params_pack(void* dst, double temperature, double top_p, int top_k, uint32_t stream_id, uint64_t seed,
                                uint64_t position) {
    if(!dst) return cudaErrorInvalidValue;
    ts::Params p;
    p.temperature = temperature; p.top_p = top_p; p.top_k = top_k; p.stream = stream_id; p.seed = seed;
    p.position = position;
    std::memcpy(dst, &p, sizeof p);
    return cudaSuccess;
}
size_t crit_ts_request_bytes(void) { return sizeof(ts::Request); }
size_t crit_ts_request_position_offset(void) { return offsetof(ts::Request, position); }
crit_status crit_ts_request_pack(void* dst, double temperature, double top_p, double min_p, int32_t top_k, double repetition,
                                 double frequency, double presence, uint32_t min_tokens, uint32_t stream_id, uint64_t seed,
                                 uint64_t position, const int32_t* bias_ids, const float* bias_values, uint32_t bias_count,
                                 const int32_t* suppress_ids, uint32_t suppress_count, int32_t* counts,
                                 const uint8_t* prompt_seen) {
    if(!dst) return cudaErrorInvalidValue;
    ts::Request r;
    r.temperature = temperature; r.top_p = top_p; r.min_p = min_p; r.top_k = top_k; r.repetition = repetition;
    r.frequency = frequency; r.presence = presence; r.min_tokens = min_tokens; r.stream = stream_id; r.seed = seed;
    r.position = position; r.bias_ids = bias_ids; r.bias_values = bias_values; r.bias_count = bias_count;
    r.suppress_ids = suppress_ids; r.suppress_count = suppress_count; r.counts = counts; r.prompt_seen = prompt_seen;
    std::memcpy(dst, &r, sizeof r);
    return cudaSuccess;
}
size_t crit_ts_sparse_law_bytes(void) { return sizeof(ts::SparseLaw); }
size_t crit_ts_select_scratch_bytes(int vocab, int rows) { return ts::select_scratch_bytes(vocab, rows); }
crit_status crit_ts_select(const float* logits, int ld, int rows, int vocab, void* params, void* scratch, int32_t* out,
                           void* stream) {
    return ts::enqueue_select(logits, ld, rows, vocab, static_cast<ts::Params*>(params), scratch, out, S(stream));
}
crit_status crit_ts_mask(float* logits, int vocab, int stride, const uint32_t* mask, int mask_stride, int rows,
                         void* stream) {
    return ts::enqueue_mask(logits, vocab, stride, mask, mask_stride, rows, S(stream));
}
size_t crit_ts_scratch_floats(uint32_t vocab, uint32_t rows) { return ts::scratch_floats(vocab, rows); }
crit_status crit_ts_sample(const float* logits, uint32_t stride, uint32_t vocab, uint32_t requests, uint32_t columns,
                           const void* request, const int32_t* staged, uint32_t staged_stride, uint32_t staged_base,
                           uint32_t position_base, float* scratch, const uint64_t* tree, const int32_t* tree_tokens,
                           uint32_t purpose, int32_t* out, float* probabilities, void* sparse, uint32_t n, float* chosen,
                           int32_t* top_ids, float* top_logprobs, void* stream) {
    const ts::Logprobs logprobs{n, chosen, top_ids, top_logprobs};
    return ts::enqueue_sample(rows_of(logits, stride, vocab, requests, columns, request, staged, staged_stride, staged_base,
                                   position_base, scratch, tree, tree_tokens),
                              crit::core::DrawPurpose(purpose), out, probabilities, static_cast<ts::SparseLaw*>(sparse),
                              logprobs, S(stream));
}
crit_status crit_ts_verify(const float* logits, uint32_t stride, uint32_t vocab, uint32_t requests, uint32_t columns,
                           const void* request, const int32_t* staged, uint32_t staged_stride, uint32_t staged_base,
                           uint32_t position_base, float* scratch, const uint64_t* tree, const int32_t* tree_tokens,
                           const int32_t* drafts, const float* draft_probabilities, const void* draft_laws,
                           int32_t* predictions, uint32_t n, float* chosen, int32_t* top_ids, float* top_logprobs,
                           void* stream) {
    ts::Verify v;
    v.target = rows_of(logits, stride, vocab, requests, columns, request, staged, staged_stride, staged_base, position_base,
                    scratch, tree, tree_tokens);
    v.drafts = drafts; v.draft_probabilities = draft_probabilities;
    v.draft_laws = static_cast<const ts::SparseLaw*>(draft_laws); v.predictions = predictions;
    v.logprobs = ts::Logprobs{n, chosen, top_ids, top_logprobs};
    return ts::enqueue_verify(v, S(stream));
}
size_t crit_ts_logprob_scratch_bytes(int vocab, int rows, int n) { return ts::logprob_scratch_bytes(vocab, rows, n); }
crit_status crit_ts_logprobs(const float* logits, int ld, int rows, int vocab, const int32_t* chosen, int n,
                             float* chosen_logprob, int32_t* top_ids, float* top_logprobs, void* scratch, void* stream) {
    return ts::enqueue_logprobs(logits, ld, rows, vocab, chosen, n, chosen_logprob, top_ids, top_logprobs, scratch, S(stream));
}
crit_status crit_ts_commit(void* requests, uint32_t count, uint32_t vocab, const int32_t* tokens, uint32_t token_stride,
                           const int32_t* emitted, uint32_t emitted_stride, void* stream) {
    return ts::enqueue_commit(static_cast<ts::Request*>(requests), count, vocab, tokens, token_stride, emitted,
                              emitted_stride, S(stream));
}

// ---------------------------------------------------------------- device-speculative-round
uint32_t crit_dr_control_fields(void) { return dr::control::fields; }
crit_status crit_dr_packed_layout(uint32_t draft_tokens, uint32_t top, int logprobs, uint32_t* offsets) {
    if(!offsets) return cudaErrorInvalidValue;
    const dr::PackedLayout l{draft_tokens, top, logprobs != 0};
    const uint32_t values[] = {l.matched(), l.frontier(), l.logprob(), l.top_ids(), l.top_logprobs(), l.words()};
    std::memcpy(offsets, values, sizeof values);
    return cudaSuccess;
}
crit_status crit_dr_greedy_accept(const int32_t* drafts, const int32_t* predictions, const int32_t* stops,
                                  uint32_t stop_count, const int32_t* device_stop_count, int32_t* control, int32_t* result,
                                  uint32_t rows, uint32_t draft_tokens, void* stream) {
    dr::AcceptRows a;
    a.drafts = drafts; a.predictions = predictions; a.stops = stops; a.stop_count = stop_count; a.control = control;
    a.result = result; a.rows = rows; a.draft_tokens = draft_tokens; a.device_stop_count = device_stop_count;
    return dr::enqueue_greedy_accept(a, S(stream));
}
crit_status crit_dr_truncate(int32_t* control, int32_t* result, uint32_t draft_tokens, uint32_t row, uint32_t accepted,
                             void* stream) {
    return dr::enqueue_truncate(control, result, draft_tokens, row, accepted, S(stream));
}
crit_status crit_dr_pack(const int32_t* control, const int32_t* result, int32_t* packed, const float* logprob,
                         const int32_t* top_ids, const float* top_logprobs, uint32_t rows, uint32_t draft_tokens,
                         uint32_t top, void* stream) {
    dr::PackRows p;
    p.control = control; p.result = result; p.packed = packed; p.logprob = logprob; p.top_ids = top_ids;
    p.top_logprobs = top_logprobs; p.rows = rows; p.draft_tokens = draft_tokens; p.top = top;
    return dr::enqueue_pack(p, S(stream));
}
crit_status crit_dr_set_budget(int32_t* control, const int32_t* allowance, uint32_t rows, void* stream) {
    return dr::enqueue_set_budget(control, allowance, rows, S(stream));
}
crit_status crit_dr_seed(int32_t* control, int32_t* result, uint32_t draft_tokens, uint32_t row, uint32_t frontier,
                         int32_t token, int32_t previous, void* stream) {
    return dr::enqueue_seed(control, result, draft_tokens, row, frontier, token, previous, S(stream));
}

}  // extern "C"
