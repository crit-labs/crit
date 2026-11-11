// Torch entry points over grouped_experts.h: shape and dtype checks, the current stream. A layout is an int list
// [format, bytes, gate_up, gate_up_scale, gate_up_inverse, down, down_scale, down_inverse, input, hidden, inter] (byte
// offsets, -1 for a component the format lacks), as crit's grouped-experts plugin computes it.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <optional>

#include "grouped_experts.h"

namespace gx = crit::techniques::grouped_experts;

namespace {
void ok(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }
void cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
}
void typed(const torch::Tensor& t, const char* name, torch::ScalarType type) {
    cuda(t, name);
    TORCH_CHECK(t.scalar_type() == type, name, " has dtype ", t.scalar_type(), ", not ", type);
}
gx::Layout layout_of(c10::ArrayRef<int64_t> v) {
    TORCH_CHECK(v.size() == 11 || v.size() == 12, "a layout is 11 integers (12: an int4_zp group last)");
    gx::Layout L;
    L.format = static_cast<gx::Format>(v[0]);
    L.bytes = v[1];
    L.gate_up = v[2], L.gate_up_scale = v[3], L.gate_up_inverse = v[4];
    L.down = v[5], L.down_scale = v[6], L.down_inverse = v[7], L.input = v[8];
    L.hidden = static_cast<int>(v[9]), L.inter = static_cast<int>(v[10]);
    L.group = v.size() == 12 ? static_cast<int>(v[11]) : 0;
    TORCH_CHECK(v[0] >= 0 && v[0] <= 6, "unknown format ", v[0]);
    return L;
}
gx::Items items_of(const torch::Tensor& slot, const torch::Tensor& routes, const torch::Tensor& count) {
    typed(slot, "item_slot", torch::kInt32);
    typed(routes, "item_routes", torch::kInt32);
    typed(count, "item_count", torch::kInt32);
    TORCH_CHECK(routes.dim() == 2 && routes.size(0) == slot.size(0), "item_routes is [capacity][width]");
    gx::Items it;
    it.slot = slot.data_ptr<int>();
    it.routes = routes.data_ptr<int>();
    it.count = count.data_ptr<int>();
    it.width = static_cast<int>(routes.size(1));
    it.capacity = static_cast<int>(slot.size(0));
    return it;
}
const int* slots_of(const std::optional<torch::Tensor>& slots, int64_t experts) {
    if(!slots || !slots->defined() || slots->numel() == 0) return nullptr;
    typed(*slots, "slots", torch::kInt32);
    TORCH_CHECK(slots->numel() >= experts, "slots has one entry per expert");
    return slots->data_ptr<int>();
}
void ids_of(const torch::Tensor& ids, const long long*& ids64, const int*& ids32) {
    cuda(ids, "ids");
    ids64 = nullptr;
    ids32 = nullptr;
    if(ids.scalar_type() == torch::kInt64) ids64 = reinterpret_cast<const long long*>(ids.data_ptr<int64_t>());
    else if(ids.scalar_type() == torch::kInt32) ids32 = ids.data_ptr<int>();
    else TORCH_CHECK(false, "ids are int64 or int32");
}
void store_of(const torch::Tensor& store, const gx::Layout& L) {
    typed(store, "store", torch::kUInt8);
    TORCH_CHECK(store.dim() == 2 && store.size(1) == L.bytes, "store is [slots][bundle bytes]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(store.data_ptr()) % 16 == 0, "store must be 16-byte aligned");
}
}  // namespace

void plan(const torch::Tensor& ids, const std::optional<torch::Tensor>& slots, int64_t experts, torch::Tensor& item_slot,
          torch::Tensor& item_routes, torch::Tensor& item_count) {
    const c10::cuda::CUDAGuard guard(ids.device());
    const long long* ids64;
    const int* ids32;
    ids_of(ids, ids64, ids32);
    ok(gx::enqueue_plan(ids64, ids32, static_cast<int>(ids.numel()), static_cast<int>(experts), slots_of(slots, experts),
                        items_of(item_slot, item_routes, item_count), at::cuda::getCurrentCUDAStream()),
       "plan");
}

void items(const torch::Tensor& order, const torch::Tensor& counts, const torch::Tensor& starts,
           const torch::Tensor& item_end, const std::optional<torch::Tensor>& slots, int64_t experts,
           torch::Tensor& item_slot, torch::Tensor& item_routes, torch::Tensor& item_count) {
    const c10::cuda::CUDAGuard guard(order.device());
    typed(order, "order", torch::kInt64);
    typed(counts, "counts", torch::kInt32);
    typed(starts, "starts", torch::kInt32);
    typed(item_end, "item_end", torch::kInt32);
    TORCH_CHECK(counts.numel() >= experts && starts.numel() >= experts && item_end.numel() >= experts,
                "counts, starts and item_end have one entry per expert");
    ok(gx::enqueue_items(reinterpret_cast<const long long*>(order.data_ptr<int64_t>()), counts.data_ptr<int>(),
                         starts.data_ptr<int>(), item_end.data_ptr<int>(), static_cast<int>(experts),
                         slots_of(slots, experts), items_of(item_slot, item_routes, item_count),
                         at::cuda::getCurrentCUDAStream()),
       "items");
}

void gate_up(const torch::Tensor& x, int64_t top, const torch::Tensor& store, c10::ArrayRef<int64_t> layout,
             bool a4, const torch::Tensor& item_slot, const torch::Tensor& item_routes, const torch::Tensor& item_count,
             torch::Tensor& h, int64_t tokens) {
    const c10::cuda::CUDAGuard guard(x.device());
    const gx::Layout L = layout_of(layout);
    typed(x, "x", torch::kBFloat16);
    typed(h, "h", torch::kBFloat16);
    store_of(store, L);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == L.hidden, "x is [tokens][hidden]");
    TORCH_CHECK(h.dim() == 2 && h.size(1) == L.inter && h.size(0) >= x.size(0) * top, "h is [routes][inter]");
    ok(gx::enqueue_gate_up(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(top), store.data_ptr<uint8_t>(),
                           L, a4, items_of(item_slot, item_routes, item_count), reinterpret_cast<__nv_bfloat16*>(h.data_ptr()),
                           at::cuda::getCurrentCUDAStream(), static_cast<int>(tokens)),
       "gate_up");
}

void down(const torch::Tensor& h, const torch::Tensor& store, c10::ArrayRef<int64_t> layout, bool a4,
          const torch::Tensor& item_slot, const torch::Tensor& item_routes, const torch::Tensor& item_count,
          torch::Tensor& y, int64_t tokens) {
    const c10::cuda::CUDAGuard guard(h.device());
    const gx::Layout L = layout_of(layout);
    typed(h, "h", torch::kBFloat16);
    typed(y, "y", torch::kFloat32);
    store_of(store, L);
    TORCH_CHECK(h.dim() == 2 && h.size(1) == L.inter, "h is [routes][inter]");
    TORCH_CHECK(y.dim() == 2 && y.size(1) == L.hidden && y.size(0) >= h.size(0), "y is [routes][hidden]");
    ok(gx::enqueue_down(reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()), store.data_ptr<uint8_t>(), L, a4,
                        items_of(item_slot, item_routes, item_count), y.data_ptr<float>(), at::cuda::getCurrentCUDAStream(),
                        static_cast<int>(tokens)),
       "down");
}

void combine(const torch::Tensor& y, const torch::Tensor& ids, const torch::Tensor& weights,
             const std::optional<torch::Tensor>& slots, int64_t experts, bool skip_negative, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(y.device());
    typed(y, "y", torch::kFloat32);
    typed(weights, "weights", torch::kFloat32);
    typed(out, "out", torch::kFloat32);
    const long long* ids64;
    const int* ids32;
    ids_of(ids, ids64, ids32);
    TORCH_CHECK(ids.dim() == 2 && weights.sizes() == ids.sizes(), "ids and weights are [tokens][top]");
    const int64_t tokens = ids.size(0), top = ids.size(1), hidden = out.size(1);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == tokens && y.size(0) >= tokens * top && y.size(1) == hidden,
                "y is [routes][hidden] and out [tokens][hidden]");
    ok(gx::enqueue_combine(y.data_ptr<float>(), ids64, ids32, weights.data_ptr<float>(), slots_of(slots, experts),
                           static_cast<int>(experts), static_cast<int>(tokens), static_cast<int>(top),
                           static_cast<int>(hidden), skip_negative, out.data_ptr<float>(), at::cuda::getCurrentCUDAStream()),
       "combine");
}

void quantize_e2m1x2(const torch::Tensor& x, double global, torch::Tensor& values, torch::Tensor& scales) {
    const c10::cuda::CUDAGuard guard(x.device());
    typed(x, "x", torch::kFloat32);
    typed(values, "values", torch::kInt8);
    typed(scales, "scales", torch::kFloat32);
    TORCH_CHECK(x.dim() == 2 && values.sizes() == x.sizes() && scales.size(0) == x.size(0) &&
                    scales.size(1) * 16 == x.size(1),
                "x [rows][cols], values the same, scales [rows][cols / 16]");
    ok(gx::enqueue_quantize_e2m1x2(x.data_ptr<float>(), static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
                                   static_cast<float>(global), values.data_ptr<int8_t>(), scales.data_ptr<float>(),
                                   at::cuda::getCurrentCUDAStream()),
       "quantize_e2m1x2");
}

void a4_inputs(const torch::Tensor& x, int64_t top, const torch::Tensor& store, c10::ArrayRef<int64_t> layout,
               const torch::Tensor& item_slot, const torch::Tensor& item_routes, const torch::Tensor& item_count,
               torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(x.device());
    const gx::Layout L = layout_of(layout);
    typed(x, "x", torch::kBFloat16);
    typed(out, "out", torch::kBFloat16);
    store_of(store, L);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == L.hidden && out.dim() == 2 && out.size(1) == L.hidden &&
                    out.size(0) >= x.size(0) * top,
                "x [tokens][hidden], out [routes][hidden]");
    ok(gx::enqueue_a4_inputs(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(top),
                             store.data_ptr<uint8_t>(), L, items_of(item_slot, item_routes, item_count),
                             reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), at::cuda::getCurrentCUDAStream()),
       "a4_inputs");
}

namespace {
int block_of(const gx::Layout& L) { return L.format == gx::Format::int4 ? 32 : 16; }
}  // namespace

void declared_inputs(const torch::Tensor& x, int64_t top, const torch::Tensor& store, c10::ArrayRef<int64_t> layout,
                     const torch::Tensor& item_slot, const torch::Tensor& item_routes, const torch::Tensor& item_count,
                     torch::Tensor& values, torch::Tensor& scales) {
    const c10::cuda::CUDAGuard guard(x.device());
    const gx::Layout L = layout_of(layout);
    typed(x, "x", torch::kBFloat16);
    typed(values, "values", torch::kInt8);
    typed(scales, "scales", torch::kFloat32);
    store_of(store, L);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == L.hidden, "x is [tokens][hidden]");
    TORCH_CHECK(values.dim() == 2 && values.size(1) == L.hidden && values.size(0) >= x.size(0) * top &&
                    scales.size(0) == values.size(0) && scales.size(1) * block_of(L) == L.hidden,
                "values [routes][hidden], scales [routes][hidden / block]");
    ok(gx::enqueue_declared_inputs(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(top),
                                   store.data_ptr<uint8_t>(), L, items_of(item_slot, item_routes, item_count),
                                   values.data_ptr<int8_t>(), scales.data_ptr<float>(), at::cuda::getCurrentCUDAStream()),
       "declared_inputs");
}

void declared_rows(const torch::Tensor& values, const torch::Tensor& scales, const torch::Tensor& store,
                   c10::ArrayRef<int64_t> layout, bool gate_up, const torch::Tensor& item_slot,
                   const torch::Tensor& item_routes, const torch::Tensor& item_count, torch::Tensor& out) {
    const c10::cuda::CUDAGuard guard(values.device());
    const gx::Layout L = layout_of(layout);
    typed(values, "values", torch::kInt8);
    typed(scales, "scales", torch::kFloat32);
    typed(out, "out", torch::kFloat32);
    store_of(store, L);
    const int64_t K = gate_up ? L.hidden : L.inter, M = gate_up ? 2 * L.inter : L.hidden;
    TORCH_CHECK(values.dim() == 2 && values.size(1) == K && scales.size(1) * block_of(L) == K &&
                    out.dim() == 2 && out.size(1) == M && out.size(0) >= values.size(0),
                "values [routes][k], scales [routes][k / block], out [routes][rows]");
    ok(gx::enqueue_declared_rows(values.data_ptr<int8_t>(), scales.data_ptr<float>(), store.data_ptr<uint8_t>(), L,
                                 gate_up, items_of(item_slot, item_routes, item_count), out.data_ptr<float>(),
                                 at::cuda::getCurrentCUDAStream()),
       "declared_rows");
}

void declared_intermediate(const torch::Tensor& gate_up, const torch::Tensor& store, c10::ArrayRef<int64_t> layout,
                           const torch::Tensor& item_slot, const torch::Tensor& item_routes,
                           const torch::Tensor& item_count, torch::Tensor& values, torch::Tensor& scales) {
    const c10::cuda::CUDAGuard guard(gate_up.device());
    const gx::Layout L = layout_of(layout);
    typed(gate_up, "gate_up", torch::kFloat32);
    typed(values, "values", torch::kInt8);
    typed(scales, "scales", torch::kFloat32);
    store_of(store, L);
    TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(1) == 2 * L.inter && values.size(1) == L.inter &&
                    values.size(0) >= gate_up.size(0) && scales.size(1) * block_of(L) == L.inter,
                "gate_up [routes][2 * inter], values [routes][inter], scales [routes][inter / block]");
    ok(gx::enqueue_declared_intermediate(gate_up.data_ptr<float>(), store.data_ptr<uint8_t>(), L,
                                         items_of(item_slot, item_routes, item_count), values.data_ptr<int8_t>(),
                                         scales.data_ptr<float>(), at::cuda::getCurrentCUDAStream()),
       "declared_intermediate");
}
