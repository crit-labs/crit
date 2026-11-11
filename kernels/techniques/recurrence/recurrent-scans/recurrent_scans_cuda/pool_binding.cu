// Torch entry points over the pool kernels (recurrent_rows.h): shape checks, strides from the tensors, the current
// stream. Nothing here allocates or synchronizes, so every op runs inside a captured CUDA graph.
#include <optional>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "recurrent_rows.h"

namespace rk = crit::recurrent_kernels;

namespace {
void check_ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }

void check_on(const torch::Tensor& t, const char* name, torch::ScalarType type, int dims) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == type, name, " must be ", type);
    TORCH_CHECK(t.dim() == dims, name, " must have ", dims, " dimensions");
}
void check_contiguous(const torch::Tensor& t, const char* name, torch::ScalarType type, int dims) {
    check_on(t, name, type, dims);
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

rk::Scalar scalar_of(const torch::Tensor& t) {
    switch(t.scalar_type()) {
    case torch::kFloat32: return rk::Scalar::f32;
    case torch::kBFloat16: return rk::Scalar::bf16;
    case torch::kFloat16: return rk::Scalar::f16;
    default: TORCH_CHECK(false, "inputs must be float32, bfloat16 or float16"); return rk::Scalar::f32;
    }
}

// A step's rows (meta [rows][5] int32) or a round's (starts [rows + 1] int32, slots [rows] int64, table int32 of at
// least `rows` slots: a round's rows are slots 0 .. rows - 1).
rk::RowSource source_of(const std::optional<torch::Tensor>& meta, const std::optional<torch::Tensor>& starts,
                        const std::optional<torch::Tensor>& slots, const std::optional<torch::Tensor>& table, int64_t rows,
                        int64_t kind) {
    rk::RowSource s;
    s.rows = static_cast<int>(rows);
    s.kind = static_cast<int>(kind);
    TORCH_CHECK(rows >= 0 && rows <= 65535, "rows must be 0 .. 65535");
    if(meta) {
        check_contiguous(*meta, "meta", torch::kInt32, 2);
        TORCH_CHECK(meta->size(0) >= rows && meta->size(1) == 5, "meta must be [rows][5]");
        s.meta = meta->data_ptr<std::int32_t>();
        return s;
    }
    TORCH_CHECK(slots && table, "a round's rows need slots and the pool's slot table");
    check_on(*slots, "slots", torch::kInt64, 1);
    check_contiguous(*table, "table", torch::kInt32, 1);
    TORCH_CHECK(slots->stride(0) == 1 && slots->size(0) >= rows, "slots must be [rows], contiguous");
    TORCH_CHECK(table->size(0) >= rows, "the slot table must hold the round's ", rows, " rows' slots");
    s.slots = slots->data_ptr<std::int64_t>();
    s.table = table->data_ptr<std::int32_t>();
    if(starts) {
        check_on(*starts, "starts", torch::kInt32, 1);
        TORCH_CHECK(starts->stride(0) == 1 && starts->size(0) >= rows + 1, "starts must be [rows + 1], contiguous");
        s.starts = starts->data_ptr<std::int32_t>();
    }
    return s;
}

// states [layers][sequences][v_heads][dk][dv], stash [layers][sequences][2 * half], control [layers][sequences][2 * blocks].
rk::PoolDelta pool_of(const rk::DeltaShape& d, torch::Tensor& states, torch::Tensor& stash, torch::Tensor& control, int64_t capacity) {
    check_contiguous(states, "states", torch::kFloat32, 5);
    check_contiguous(stash, "stash", torch::kFloat32, 3);
    check_contiguous(control, "control", torch::kInt32, 3);
    TORCH_CHECK(capacity >= 1, "the stash holds one column or more");
    // Every tensor has the layers; the sequence counts may differ (a pool keeps a spare sequence in some of them):
    // every row's sequence index, read on the device, must lie in each.
    TORCH_CHECK(stash.size(0) == states.size(0) &&
                stash.size(2) == 2 * static_cast<int64_t>(rk::pool_delta_half_floats(d, static_cast<int>(capacity))),
                "stash must be [layers][sequences][2 * pool_delta_half_floats]");
    TORCH_CHECK(control.size(0) == states.size(0) && control.size(2) == 2 * rk::pool_delta_blocks(d),
                "control must be [layers][sequences][2 * pool_delta_blocks]");
    rk::PoolDelta p;
    p.states = states.data_ptr<float>();
    p.state_sequence = states.stride(1);
    p.state_layer = states.stride(0);
    p.stash = stash.data_ptr<float>();
    p.stash_sequence = stash.stride(1);
    p.stash_layer = stash.stride(0);
    p.capacity = static_cast<int>(capacity);
    p.control = control.data_ptr<std::int32_t>();
    p.control_sequence = control.stride(1);
    p.control_layer = control.stride(0);
    return p;
}

rk::DeltaShape shape_of(const torch::Tensor& states, int64_t key_heads) {
    rk::DeltaShape d;
    d.v_heads = static_cast<int>(states.size(2));
    d.dk = static_cast<int>(states.size(3));
    d.dv = static_cast<int>(states.size(4));
    d.k_heads = static_cast<int>(key_heads);
    TORCH_CHECK(rk::pool_delta_supported(d), "the pool's shape: dk 1 .. 256, value heads a multiple of key heads");
    return d;
}
}  // namespace

void pool_delta(torch::Tensor& out, torch::Tensor& states, torch::Tensor& stash, torch::Tensor& control,
                const std::optional<torch::Tensor>& scratch, const torch::Tensor& q, const torch::Tensor& k,
                const torch::Tensor& v, const torch::Tensor& log_decay, const torch::Tensor& beta,
                const std::optional<torch::Tensor>& meta, const std::optional<torch::Tensor>& starts,
                const std::optional<torch::Tensor>& slots, const std::optional<torch::Tensor>& table, int64_t rows,
                int64_t kind, int64_t layer, int64_t capacity, double epsilon, double scale, bool normalize_keys,
                bool normalize_queries, bool pipelined) {
    const c10::cuda::CUDAGuard guard(states.device());
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k [T][key heads][dk], v [T][value heads][dv]");
    const rk::DeltaShape d = shape_of(states, q.size(1));
    rk::PoolDelta p = pool_of(d, states, stash, control, capacity);
    TORCH_CHECK(layer >= 0 && layer < states.size(0), "layer out of range");
    p.states += layer * p.state_layer;
    p.stash += layer * p.stash_layer;
    p.control += layer * p.control_layer;
    const int64_t S = states.size(2) * states.size(3) * states.size(4);
    if(scratch) {   // a slot: (capacity + 1) states for a round's checkpoints, else one
        TORCH_CHECK(scratch->is_cuda() && scratch->scalar_type() == torch::kFloat32 && scratch->dim() >= 2 &&
                    scratch->is_contiguous(), "scratch must be a contiguous float32 [slots][...] tensor");
        const int64_t per = (!meta && kind == rk::kCheckpoints) ? (capacity + 1) * S : S;
        TORCH_CHECK(scratch->stride(0) >= per, "scratch must hold ", per, " floats per slot");
        TORCH_CHECK(meta || scratch->size(0) >= rows, "scratch must hold a slot per round row");
        p.scratch = scratch->data_ptr<float>();
        p.scratch_slot = scratch->stride(0);
    }
    const rk::Scalar dtype = scalar_of(q);
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(), "q, k and v share a dtype");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k and v must be CUDA tensors");
    TORCH_CHECK(q.sizes() == k.sizes() && q.size(2) == d.dk && v.size(1) == d.v_heads && v.size(2) == d.dv &&
                v.size(0) == q.size(0), "q, k [T][key heads][dk] and v [T][value heads][dv] of the pool's shape");
    TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1, "q, k and v rows must be contiguous");
    check_on(log_decay, "log_decay", torch::kFloat32, 2);
    check_on(beta, "beta", torch::kFloat32, 2);
    TORCH_CHECK(log_decay.size(1) == d.v_heads && beta.size(1) == d.v_heads && log_decay.stride(1) == 1 && beta.stride(1) == 1,
                "log_decay and beta must be [T][value heads]");
    check_on(out, "out", torch::kFloat32, 3);
    TORCH_CHECK(out.size(1) == d.v_heads && out.size(2) == d.dv && out.stride(2) == 1 && out.stride(1) == d.dv,
                "out must be [T][value heads][dv], heads contiguous within a row");
    TORCH_CHECK(out.size(0) == q.size(0) && log_decay.size(0) == q.size(0) && beta.size(0) == q.size(0),
                "out, log_decay and beta must have q's ", q.size(0), " rows");
    rk::RowInputs in;
    in.q = q.data_ptr();
    in.k = k.data_ptr();
    in.v = v.data_ptr();
    in.dtype = dtype;
    in.q_row = q.stride(0);
    in.q_head = q.stride(1);
    in.k_row = k.stride(0);
    in.k_head = k.stride(1);
    in.v_row = v.stride(0);
    in.v_head = v.stride(1);
    in.log_decay = log_decay.data_ptr<float>();
    in.beta = beta.data_ptr<float>();
    in.gate_row = log_decay.stride(0);
    in.beta_row = beta.stride(0);
    in.epsilon = static_cast<float>(epsilon);
    in.scale = static_cast<float>(scale);
    in.normalize_keys = normalize_keys;
    in.normalize_queries = normalize_queries;
    in.out = out.data_ptr<float>();
    in.out_row = out.stride(0);
    const rk::RowSource s = source_of(meta, starts, slots, table, rows, kind);
    TORCH_CHECK(s.meta || s.starts, "a round's rows need starts");
    check_ok(rk::enqueue_pool_delta(d, in, p, s, at::cuda::getCurrentCUDAStream(), pipelined), "pool_delta");
}

void pool_delta_replay(torch::Tensor& states, torch::Tensor& stash, torch::Tensor& control,
                       const std::optional<torch::Tensor>& scratch, const std::optional<torch::Tensor>& accepted,
                       const std::optional<torch::Tensor>& meta, const std::optional<torch::Tensor>& slots,
                       const std::optional<torch::Tensor>& table, int64_t rows, int64_t columns, int64_t capacity,
                       int64_t key_heads, int64_t mode) {
    const c10::cuda::CUDAGuard guard(states.device());
    const rk::DeltaShape d = shape_of(states, key_heads);
    rk::PoolDelta p = pool_of(d, states, stash, control, capacity);
    if(scratch) {   // [layers][slots][...]: a start state per slot
        TORCH_CHECK(scratch->is_cuda() && scratch->scalar_type() == torch::kFloat32 && scratch->dim() >= 3 &&
                    scratch->is_contiguous() && scratch->size(0) == states.size(0),
                    "scratch must be a contiguous float32 [layers][slots][...] tensor");
        TORCH_CHECK(scratch->stride(1) >= states.size(2) * states.size(3) * states.size(4) && (meta || scratch->size(1) >= rows),
                    "scratch must hold a start state per round row");
        p.scratch = scratch->data_ptr<float>();
        p.scratch_layer = scratch->stride(0);
        p.scratch_slot = scratch->stride(1);
    }
    const std::int64_t* counts = nullptr;
    if(accepted) {
        check_on(*accepted, "accepted", torch::kInt64, 1);
        TORCH_CHECK(accepted->stride(0) == 1 && accepted->size(0) >= rows, "accepted must be [rows], contiguous");
        counts = accepted->data_ptr<std::int64_t>();
    }
    const rk::RowSource s = source_of(meta, std::nullopt, slots, table, rows, 0);
    check_ok(rk::enqueue_pool_delta_replay(d, p, s, static_cast<int>(states.size(0)), counts, static_cast<int>(columns),
                                           static_cast<rk::CommitMode>(mode), at::cuda::getCurrentCUDAStream()),
             "pool_delta_replay");
}

namespace {
rk::ConvInputs conv_inputs(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& weight,
                           const std::optional<torch::Tensor>& bias, bool silu) {
    TORCH_CHECK(x.is_cuda() && out.is_cuda() && x.dim() == 2 && out.dim() == 2, "x and out must be [T][channels] CUDA tensors");
    TORCH_CHECK(out.scalar_type() == x.scalar_type() && out.size(0) == x.size(0) && out.size(1) == x.size(1),
                "out must have x's shape and dtype");
    TORCH_CHECK(x.numel() == 0 || out.data_ptr() != x.data_ptr(), "out must not be x");
    TORCH_CHECK(x.stride(1) == 1 && out.stride(1) == 1, "x and out rows must be contiguous");
    check_contiguous(weight, "weight", torch::kFloat32, 2);
    TORCH_CHECK(weight.size(0) == x.size(1) && weight.size(1) >= 2 && weight.size(1) <= 8, "weight must be [channels][2 .. 8]");
    rk::ConvInputs c;
    c.x = x.data_ptr();
    c.out = out.data_ptr();
    c.dtype = scalar_of(x);
    c.x_row = x.stride(0);
    c.out_row = out.stride(0);
    c.weight = weight.data_ptr<float>();
    if(bias) {
        check_contiguous(*bias, "bias", torch::kFloat32, 1);
        TORCH_CHECK(bias->size(0) == x.size(1), "bias must be [channels]");
        c.bias = bias->data_ptr<float>();
    }
    c.channels = static_cast<int>(x.size(1));
    c.width = static_cast<int>(weight.size(1));
    c.silu = silu;
    return c;
}
}  // namespace

// The dilation a window of `span` inputs per channel gives a convolution of c.width taps: span = (width - 1) * dilation.
void set_dilation(rk::ConvInputs& c, int64_t span) {
    TORCH_CHECK(c.width >= 2 && span >= c.width - 1 && span % (c.width - 1) == 0,
                "a window holds (width - 1) * dilation inputs per channel, not ", span, " for ", c.width, " taps");
    c.dilation = static_cast<int>(span / (c.width - 1));
    TORCH_CHECK(rk::pool_conv_supported(c.width, c.dilation), "a convolution of ", c.width, " taps at dilation ",
                c.dilation, " is none the kernels run (2 .. 8 taps undilated, 2 .. 4 taps at dilation 2 .. 4)");
}

void pool_conv(torch::Tensor& out, torch::Tensor& window, torch::Tensor& stash, torch::Tensor& pending,
               const std::optional<torch::Tensor>& scratch, const torch::Tensor& x, const torch::Tensor& weight,
               const std::optional<torch::Tensor>& bias, const std::optional<torch::Tensor>& meta,
               const std::optional<torch::Tensor>& starts, const std::optional<torch::Tensor>& slots,
               const std::optional<torch::Tensor>& table, int64_t rows, int64_t kind, int64_t layer, int64_t layers,
               int64_t capacity, bool silu) {
    const c10::cuda::CUDAGuard guard(window.device());
    rk::ConvInputs c = conv_inputs(out, x, weight, bias, silu);
    check_contiguous(window, "window", torch::kFloat32, 4);
    check_contiguous(stash, "stash", torch::kFloat32, 4);
    check_contiguous(pending, "pending", torch::kInt32, 3);
    TORCH_CHECK(window.size(2) == c.channels, "window must be [layers][sequences][channels][(width - 1) * dilation]");
    set_dilation(c, window.size(3));
    TORCH_CHECK(stash.size(0) == window.size(0) && stash.size(2) == capacity && stash.size(3) == c.channels,
                "stash must be [layers][sequences][capacity][channels]");
    TORCH_CHECK(pending.size(0) == window.size(0) && pending.size(2) == rk::pool_conv_blocks(c.channels),
                "pending must be [layers][sequences][pool_conv_blocks]");
    TORCH_CHECK(layer >= 0 && layers >= 1 && layer + layers <= window.size(0), "layers out of range");
    rk::PoolConv p;
    p.window = window.data_ptr<float>() + layer * window.stride(0);
    p.window_sequence = window.stride(1);
    p.window_layer = window.stride(0);
    p.stash = stash.data_ptr<float>() + layer * stash.stride(0);
    p.stash_sequence = stash.stride(1);
    p.stash_layer = stash.stride(0);
    p.capacity = static_cast<int>(capacity);
    p.pending = pending.data_ptr<std::int32_t>() + layer * pending.stride(0);
    p.pending_sequence = pending.stride(1);
    p.pending_layer = pending.stride(0);
    if(scratch) {   // a final window per slot
        TORCH_CHECK(scratch->is_cuda() && scratch->scalar_type() == torch::kFloat32 && scratch->is_contiguous() && scratch->dim() >= 2,
                    "scratch must be a contiguous float32 [slots][...] tensor");
        TORCH_CHECK(scratch->stride(0) >= int64_t(c.channels) * window.size(3), "scratch must hold a window per slot");
        p.scratch = scratch->data_ptr<float>();
        p.scratch_slot = scratch->stride(0);
    }
    const rk::RowSource s = source_of(meta, starts, slots, table, rows, kind);
    TORCH_CHECK(s.meta || s.starts, "a round's rows need starts");
    check_ok(rk::enqueue_pool_conv(c, p, s, static_cast<int>(layers), at::cuda::getCurrentCUDAStream()), "pool_conv");
}

void pool_conv_prompt(torch::Tensor& out, torch::Tensor& window, const torch::Tensor& stash, torch::Tensor& pending,
                      const torch::Tensor& x, const torch::Tensor& weight, const std::optional<torch::Tensor>& bias,
                      const torch::Tensor& meta, int64_t layer, int64_t capacity, bool silu) {
    const c10::cuda::CUDAGuard guard(window.device());
    rk::ConvInputs c = conv_inputs(out, x, weight, bias, silu);
    check_contiguous(window, "window", torch::kFloat32, 4);
    check_contiguous(stash, "stash", torch::kFloat32, 4);
    check_contiguous(pending, "pending", torch::kInt32, 3);
    TORCH_CHECK(window.size(2) == c.channels, "window must be [layers][sequences][channels][(width - 1) * dilation]");
    set_dilation(c, window.size(3));
    TORCH_CHECK(stash.size(0) == window.size(0) && stash.size(2) == capacity && stash.size(3) == c.channels,
                "stash must be [layers][sequences][capacity][channels]");
    TORCH_CHECK(pending.size(0) == window.size(0) && pending.size(2) == rk::pool_conv_blocks(c.channels),
                "pending must be [layers][sequences][pool_conv_blocks]");
    TORCH_CHECK(layer >= 0 && layer < window.size(0), "layer out of range");
    TORCH_CHECK(meta.is_cuda() && meta.scalar_type() == torch::kInt32 && meta.is_contiguous() && meta.numel() >= 5,
                "meta must be a contiguous int32 row [5] on the device");
    TORCH_CHECK(out.size(0) == x.size(0), "out must have x's rows");
    rk::PoolConv p;
    p.window = window.data_ptr<float>() + layer * window.stride(0);
    p.window_sequence = window.stride(1);
    p.window_layer = window.stride(0);
    p.stash = stash.data_ptr<float>() + layer * stash.stride(0);
    p.stash_sequence = stash.stride(1);
    p.stash_layer = stash.stride(0);
    p.capacity = static_cast<int>(capacity);
    p.pending = pending.data_ptr<std::int32_t>() + layer * pending.stride(0);
    p.pending_sequence = pending.stride(1);
    p.pending_layer = pending.stride(0);
    check_ok(rk::enqueue_pool_conv_prompt(c, p, meta.data_ptr<std::int32_t>(), static_cast<int>(x.size(0)),
                                          at::cuda::getCurrentCUDAStream()),
             "pool_conv_prompt");
}

void conv_segments(torch::Tensor& out, const torch::Tensor& window_in, torch::Tensor& window_out, const torch::Tensor& x,
                   const torch::Tensor& weight, const std::optional<torch::Tensor>& bias, bool silu) {
    const c10::cuda::CUDAGuard guard(x.device());
    rk::ConvInputs c = conv_inputs(out, x, weight, bias, silu);
    check_contiguous(window_in, "window_in", torch::kFloat32, 2);
    check_contiguous(window_out, "window_out", torch::kFloat32, 2);
    TORCH_CHECK(window_in.size(0) == c.channels && window_out.sizes() == window_in.sizes(),
                "windows must be [channels][(width - 1) * dilation]");
    set_dilation(c, window_in.size(1));
    check_ok(rk::enqueue_conv_segments(c, window_in.data_ptr<float>(), window_out.data_ptr<float>(), static_cast<int>(x.size(0)),
                                       at::cuda::getCurrentCUDAStream()),
             "conv_segments");
}
