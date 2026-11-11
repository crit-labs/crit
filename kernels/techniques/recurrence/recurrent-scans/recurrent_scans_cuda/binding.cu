// Torch entry points over the recurrent kernels: shape checks, strides from the tensors, scratch from the caching
// allocator, the current stream.
#include <optional>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "recurrent_kernels.h"

namespace rk = crit::recurrent_kernels;

namespace {
void check_f32(const torch::Tensor& t, const char* name, int dims) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.dim() == dims, name, " must have ", dims, " dimensions");
    TORCH_CHECK(t.stride(-1) == 1, name, "'s last dimension must be contiguous");
}
void check_ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }

rk::DeltaShape delta_shape(const torch::Tensor& q, const torch::Tensor& v) {
    rk::DeltaShape d;
    d.k_heads = static_cast<int>(q.size(1));
    d.dk = static_cast<int>(q.size(2));
    d.v_heads = static_cast<int>(v.size(1));
    d.dv = static_cast<int>(v.size(2));
    return d;
}
rk::DeltaInputs delta_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                             const torch::Tensor& log_decay, const torch::Tensor& beta, double scale) {
    check_f32(q, "q", 3); check_f32(k, "k", 3); check_f32(v, "v", 3); check_f32(log_decay, "log_decay", 2);
    check_f32(beta, "beta", 2);
    TORCH_CHECK(q.sizes() == k.sizes() && q.stride(0) == k.stride(0), "q and k must have the same shape and row stride");
    TORCH_CHECK(q.stride(1) == q.size(2) && v.stride(1) == v.size(2), "heads must be contiguous within a row");
    TORCH_CHECK(log_decay.sizes() == beta.sizes() && log_decay.stride(0) == beta.stride(0) && log_decay.size(1) == v.size(1),
                "log_decay and beta must be [T, v_heads] with the same row stride");
    TORCH_CHECK(v.size(1) % q.size(1) == 0, "v_heads must be a multiple of k_heads");
    rk::DeltaInputs in;
    in.q = q.data_ptr<float>(); in.k = k.data_ptr<float>(); in.qk_stride = q.stride(0);
    in.v = v.data_ptr<float>(); in.v_stride = v.stride(0);
    in.log_decay = log_decay.data_ptr<float>(); in.beta = beta.data_ptr<float>(); in.gate_stride = log_decay.stride(0);
    in.scale = static_cast<float>(scale);
    return in;
}
// out [rows, heads, width] (heads contiguous within a row), state contiguous [heads, s1, s2].
void check_out_state(const torch::Tensor& out, const torch::Tensor& state, int64_t rows, int64_t heads, int64_t width,
                     int64_t s1, int64_t s2) {
    check_f32(out, "out", 3); check_f32(state, "state", 3);
    TORCH_CHECK(out.size(0) == rows && out.size(1) == heads && out.size(2) == width && out.stride(1) == width,
                "out must be [T, heads, ", width, "]");
    TORCH_CHECK(state.is_contiguous() && state.size(0) == heads && state.size(1) == s1 && state.size(2) == s2,
                "state must be contiguous [", heads, ", ", s1, ", ", s2, "]");
}

rk::SsdInputs ssd_inputs(const torch::Tensor& x, const torch::Tensor& dt, const torch::Tensor& B, const torch::Tensor& C,
                         const torch::Tensor& A, const std::optional<torch::Tensor>& D) {
    check_f32(x, "x", 3); check_f32(dt, "dt", 2); check_f32(B, "B", 3); check_f32(C, "C", 3); check_f32(A, "A", 1);
    TORCH_CHECK(B.sizes() == C.sizes() && B.stride(0) == C.stride(0), "B and C must have the same shape and row stride");
    TORCH_CHECK(x.stride(1) == x.size(2) && B.stride(1) == B.size(2), "heads and groups must be contiguous within a row");
    TORCH_CHECK(A.is_contiguous() && A.size(0) == x.size(1), "A must be [heads]");
    rk::SsdInputs in;
    in.x = x.data_ptr<float>(); in.x_stride = x.stride(0);
    in.dt = dt.data_ptr<float>(); in.dt_stride = dt.stride(0);
    in.B = B.data_ptr<float>(); in.C = C.data_ptr<float>(); in.bc_stride = B.stride(0);
    in.A = A.data_ptr<float>();
    if(D) {
        check_f32(*D, "D", 1);
        TORCH_CHECK(D->is_contiguous() && D->size(0) == x.size(1), "D must be [heads]");
        in.D = D->data_ptr<float>();
    }
    return in;
}
rk::SsdShape ssd_shape(const torch::Tensor& x, const torch::Tensor& B) {
    rk::SsdShape s;
    s.heads = static_cast<int>(x.size(1)); s.head_dim = static_cast<int>(x.size(2));
    s.groups = static_cast<int>(B.size(1)); s.d_state = static_cast<int>(B.size(2));
    return s;
}
}  // namespace

void causal_conv1d(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& x, const torch::Tensor& weight,
                   const std::optional<torch::Tensor>& bias, bool silu) {
    check_f32(x, "x", 2); check_f32(weight, "weight", 2); check_f32(out, "out", 2); check_f32(state, "state", 2);
    const c10::cuda::CUDAGuard guard(x.device());
    rk::Conv c;
    c.x = x.data_ptr<float>(); c.x_stride = x.stride(0);
    c.channels = static_cast<int>(x.size(1)); c.width = static_cast<int>(weight.size(1));
    TORCH_CHECK(weight.is_contiguous() && weight.size(0) == x.size(1), "weight must be contiguous [channels, width]");
    TORCH_CHECK(state.is_contiguous() && state.size(0) == x.size(1) && state.size(1) == weight.size(1) - 1,
                "state must be contiguous [channels, width - 1]");
    TORCH_CHECK(out.sizes() == x.sizes(), "out must have x's shape");
    c.weight = weight.data_ptr<float>();
    if(bias) {
        check_f32(*bias, "bias", 1);
        c.bias = bias->data_ptr<float>();
    }
    c.silu = silu;
    c.state_in = state.data_ptr<float>(); c.state_out = state.data_ptr<float>();
    c.out = out.data_ptr<float>(); c.out_stride = out.stride(0);
    check_ok(rk::enqueue_conv(c, static_cast<int>(x.size(0)), at::cuda::getCurrentCUDAStream()), "causal_conv1d");
}

void gated_delta_prefill(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& q, const torch::Tensor& k,
                         const torch::Tensor& v, const torch::Tensor& log_decay, const torch::Tensor& beta, double scale) {
    const c10::cuda::CUDAGuard guard(q.device());
    const rk::DeltaInputs in = delta_inputs(q, k, v, log_decay, beta, scale);
    const rk::DeltaShape d = delta_shape(q, v);
    const int rows = static_cast<int>(q.size(0));
    check_out_state(out, state, rows, d.v_heads, d.dv, d.dk, d.dv);
    auto scratch = torch::empty({static_cast<int64_t>(rk::delta_scratch_floats(d, rows))}, q.options());
    check_ok(rk::enqueue_delta_prefill(d, in, rows, state.data_ptr<float>(), scratch.data_ptr<float>(), out.data_ptr<float>(),
                                       out.stride(0), at::cuda::getCurrentCUDAStream()), "gated_delta_prefill");
}

void gated_delta_prefill_tc(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& q, const torch::Tensor& k,
                            const torch::Tensor& v, const torch::Tensor& log_decay, const torch::Tensor& beta, double scale) {
    const c10::cuda::CUDAGuard guard(q.device());
    const rk::DeltaInputs in = delta_inputs(q, k, v, log_decay, beta, scale);
    const rk::DeltaShape d = delta_shape(q, v);
    const int rows = static_cast<int>(q.size(0));
    check_out_state(out, state, rows, d.v_heads, d.dv, d.dk, d.dv);
    TORCH_CHECK(rk::delta_tc_supported(d), "gated_delta_prefill on tensor cores takes dk 64 or 128 and dv a multiple of 16");
    if(!rows) return;
    auto scratch = torch::empty({static_cast<int64_t>(rk::delta_tc_scratch_bytes(d, rows))}, q.options().dtype(torch::kUInt8));
    check_ok(rk::enqueue_delta_prefill_tc(d, in, rows, state.data_ptr<float>(), scratch.data_ptr(), out.data_ptr<float>(),
                                          out.stride(0), at::cuda::getCurrentCUDAStream()), "gated_delta_prefill (tensor cores)");
}

void gated_delta_scan(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& q, const torch::Tensor& k,
                      const torch::Tensor& v, const torch::Tensor& log_decay, const torch::Tensor& beta, double scale,
                      const std::optional<torch::Tensor>& checkpoints) {
    const c10::cuda::CUDAGuard guard(q.device());
    const rk::DeltaInputs in = delta_inputs(q, k, v, log_decay, beta, scale);
    const rk::DeltaShape d = delta_shape(q, v);
    const int rows = static_cast<int>(q.size(0));
    check_out_state(out, state, rows, d.v_heads, d.dv, d.dk, d.dv);
    float* cp = nullptr;
    if(checkpoints) {
        TORCH_CHECK(checkpoints->is_contiguous() && checkpoints->numel() == state.numel() * rows,
                    "checkpoints must be contiguous [T, v_heads, dk, dv]");
        cp = checkpoints->data_ptr<float>();
    }
    check_ok(rk::enqueue_delta_columns(d, in, rows, state.data_ptr<float>(), state.data_ptr<float>(), cp,
                                       out.data_ptr<float>(), out.stride(0), at::cuda::getCurrentCUDAStream()),
             "gated_delta_scan");
}

void ssd_prefill(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& x, const torch::Tensor& dt,
                 const torch::Tensor& B, const torch::Tensor& C, const torch::Tensor& A, const std::optional<torch::Tensor>& D) {
    const c10::cuda::CUDAGuard guard(x.device());
    const rk::SsdInputs in = ssd_inputs(x, dt, B, C, A, D);
    const rk::SsdShape s = ssd_shape(x, B);
    const int rows = static_cast<int>(x.size(0));
    check_out_state(out, state, rows, s.heads, s.head_dim, s.head_dim, s.d_state);
    auto scratch = torch::empty({static_cast<int64_t>(rk::ssd_scratch_floats(s, rows))}, x.options());
    check_ok(rk::enqueue_ssd_prefill(s, in, rows, state.data_ptr<float>(), scratch.data_ptr<float>(), out.data_ptr<float>(),
                                     out.stride(0), at::cuda::getCurrentCUDAStream()), "ssd_prefill");
}

void ssd_scan(torch::Tensor& out, torch::Tensor& state, const torch::Tensor& x, const torch::Tensor& dt,
              const torch::Tensor& B, const torch::Tensor& C, const torch::Tensor& A, const std::optional<torch::Tensor>& D,
              const std::optional<torch::Tensor>& checkpoints) {
    const c10::cuda::CUDAGuard guard(x.device());
    const rk::SsdInputs in = ssd_inputs(x, dt, B, C, A, D);
    const rk::SsdShape s = ssd_shape(x, B);
    const int rows = static_cast<int>(x.size(0));
    check_out_state(out, state, rows, s.heads, s.head_dim, s.head_dim, s.d_state);
    float* cp = nullptr;
    if(checkpoints) {
        TORCH_CHECK(checkpoints->is_contiguous() && checkpoints->numel() == state.numel() * rows,
                    "checkpoints must be contiguous [T, heads, head_dim, d_state]");
        cp = checkpoints->data_ptr<float>();
    }
    check_ok(rk::enqueue_ssd_columns(s, in, rows, state.data_ptr<float>(), state.data_ptr<float>(), cp, out.data_ptr<float>(),
                                     out.stride(0), at::cuda::getCurrentCUDAStream()), "ssd_scan");
}
