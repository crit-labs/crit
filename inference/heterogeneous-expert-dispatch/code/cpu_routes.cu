// A routed layer's CPU share joined inside a CUDA stream (captured graphs too), over one of the host's handoffs
// (Core's, crit_plugins.host): one kernel finds the routes whose expert the slot table sends to the CPU tier (slot
// -2), writes them into a handoff slot's payload (each route's expert key and token, the tokens' rows in FP32) and
// posts the slot, or answers it itself when there is nothing for the CPU; after the stream's join (the handoff's) a
// second kernel copies the CPU's rows into the device part's route rows, where the device part's combine sums every
// route in route order. One post and one join per layer: the next layer's input needs this layer's sum, so a decode
// step cannot join less often; a layer with no CPU routes costs its post kernel and a join that passes at once.
//
// The payload is host-expert-tier's InputLayout: header {count, tokens, route capacity, token capacity} (uint32), experts
// int32 [routes], token int32 [routes], x FP32 [tokens][hidden], out FP32 [routes][hidden], each part 64-byte aligned;
// its handler is host-expert-tier's crit_tier_serve, given to the handoff as a C function and its context.
//
// Mechanism code, built at load through Core's native build (crit_plugins.native.extension; the Python layer:
// device_dispatch.py) with Core's handoff_device.cuh. Bindings: pybind11 over torch tensors on the current stream.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "handoff_device.cuh"

namespace hm = crit::host;

namespace {

struct Layout {
    std::int64_t experts, token, x, out, bytes;
};

Layout layout_of(std::int64_t hidden, std::int64_t routes, std::int64_t tokens) {
    auto up = [](std::int64_t n) { return (n + 63) / 64 * 64; };
    Layout p{};
    p.experts = 64;
    p.token = up(p.experts + routes * 4);
    p.x = up(p.token + routes * 4);
    p.out = up(p.x + tokens * hidden * 4);
    p.bytes = p.out + routes * hidden * 4;
    return p;
}

// The declared arithmetic quantizes bf16(x) as FP32, as the device experts read x: an FP32 or FP16 row is rounded to
// BF16 first (to nearest, ties to even), so a token's codes are the same whatever dtype carries it.
__device__ inline float as_float(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ inline float as_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ inline float as_float(__half v) { return __bfloat162float(__float2bfloat16_rn(__half2float(v))); }

// One block: every thread reads its routes' ids and slots into shared memory (the reads in parallel), thread 0 lists
// the CPU routes in route order from there (and the tokens they read, in first-use order), every thread copies those
// tokens' rows, all fence to system scope, then thread 0 posts the slot (or skips it).
template <class Id, class X>
__global__ void post_kernel(const Id* ids, int tokens, int top, const std::int32_t* slots, int experts, const X* x,
                            int hidden, int key_base, hm::DeviceSlot slot, Layout layout, int cap_routes, int cap_tokens,
                            std::int32_t* cpu_routes, std::int32_t* cpu_count, unsigned long long* counters) {
    extern __shared__ int shared[];
    int* token_map = shared;              // [tokens]
    int* cpu_expert = shared + tokens;    // [tokens * top]: the route's expert when the CPU's, else -1
    __shared__ int count;
    for(int r = int(threadIdx.x); r < tokens * top; r += int(blockDim.x)) {
        const long long id = static_cast<long long>(ids[r]);
        cpu_expert[r] = id >= 0 && id < experts && slots[id] == -2 ? int(id) : -1;
    }
    __syncthreads();
    std::byte* payload = slot.payload;
    auto* head = reinterpret_cast<std::uint32_t*>(payload);
    auto* expert_out = reinterpret_cast<std::int32_t*>(payload + layout.experts);
    auto* token_out = reinterpret_cast<std::int32_t*>(payload + layout.token);
    auto* x_out = reinterpret_cast<float*>(payload + layout.x);
    if(threadIdx.x == 0) {
        int routes = 0, used = 0, overflow = 0;
        for(int t = 0; t < tokens; ++t) token_map[t] = -1;
        for(int r = 0; r < tokens * top; ++r) {
            const int id = cpu_expert[r];
            if(id < 0) continue;
            const int t = r / top;
            if(token_map[t] < 0) {
                if(used >= cap_tokens) { overflow = 1; continue; }
                token_map[t] = used++;
            }
            if(routes >= cap_routes) { overflow = 1; continue; }
            expert_out[routes] = key_base + id;
            token_out[routes] = token_map[t];
            cpu_routes[routes] = r;
            ++routes;
        }
        head[0] = std::uint32_t(routes);
        head[1] = std::uint32_t(used);
        head[2] = std::uint32_t(cap_routes);
        head[3] = std::uint32_t(cap_tokens);
        cpu_count[0] = routes;
        cpu_count[1] = overflow;
        count = routes;
        if(routes) {   // the device's own count of the jobs it posted and their routes, replays included
            atomicAdd(&counters[0], 1ull);
            atomicAdd(&counters[1], static_cast<unsigned long long>(routes));
        }
    }
    __syncthreads();
    for(int t = 0; t < tokens; ++t) {
        const int to = token_map[t];
        if(to < 0) continue;
        for(int h = int(threadIdx.x); h < hidden; h += int(blockDim.x))
            x_out[std::size_t(to) * std::size_t(hidden) + std::size_t(h)] = as_float(x[std::size_t(t) * std::size_t(hidden) + std::size_t(h)]);
    }
    if(count > 0) __threadfence_system();   // the rows each thread wrote, before the doorbell
    __syncthreads();
    if(threadIdx.x == 0) {
        if(count > 0) hm::post_job(slot);
        else hm::skip_job(slot);
    }
}

// After the join: route i's row of the payload into y at its route position. A slot the host failed or released leaves
// NaN rows (never a silent partial sum) and sets the status word the caller checks. Reads of mapped host memory cross
// the link and serialize, so only the blocks of real routes read the status, once each, after the route count (device
// memory) says whether they have work; a slot the post answered itself (no CPU routes) is read from nowhere.
__global__ void scatter_kernel(float* y, int hidden, hm::DeviceSlot slot, Layout layout, const std::int32_t* cpu_routes,
                               const std::int32_t* cpu_count, std::int32_t* status_out) {
    const int i = int(blockIdx.x);
    const int count = cpu_count[0];
    if(count == 0) {
        if(i == 0 && threadIdx.x == 0) status_out[0] = int(hm::skipped);
        return;
    }
    if(i >= count) return;
    __shared__ std::uint32_t status;
    if(threadIdx.x == 0) {
        status = hm::read_status(slot);
        if(i == 0) status_out[0] = int(status);
    }
    __syncthreads();
    const bool good = status == hm::served;
    const auto* out = reinterpret_cast<const float*>(slot.payload + layout.out);
    const int r = cpu_routes[i];
    for(int h = int(threadIdx.x); h < hidden; h += int(blockDim.x))
        y[std::size_t(r) * std::size_t(hidden) + std::size_t(h)] =
            good ? hm::read_result(out + std::size_t(i) * std::size_t(hidden) + std::size_t(h)) : __int_as_float(0x7fc00000);
}

// The post and scatter kernels over one handoff's slots: words holds each slot's device addresses (ready, done,
// status, payload), as crit_plugins.host's Handoff.words(slot) gives them. The handoff and its join are the host's.
class Routes {
public:
    Routes(std::int64_t hidden, std::int64_t routes, std::int64_t tokens,
           const std::vector<std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t>>& words)
        : hidden_(hidden), routes_(routes), tokens_(tokens), layout_(layout_of(hidden, routes, tokens)) {
        for(const auto& [ready, done, status, payload] : words)
            slots_.push_back(hm::DeviceSlot{reinterpret_cast<std::uint32_t*>(ready), reinterpret_cast<std::uint32_t*>(done),
                                            reinterpret_cast<std::uint32_t*>(status), reinterpret_cast<std::byte*>(payload)});
        if(slots_.empty()) throw std::invalid_argument("cpu routes: a handoff's slots");
    }

    void post(std::int64_t slot, const torch::Tensor& ids, const torch::Tensor& slots, const torch::Tensor& x,
              std::int64_t key_base, torch::Tensor& cpu_routes, torch::Tensor& cpu_count, torch::Tensor& counters) {
        TORCH_CHECK(ids.is_cuda() && slots.is_cuda() && x.is_cuda() && cpu_routes.is_cuda() && cpu_count.is_cuda() &&
                    counters.is_cuda() && counters.scalar_type() == torch::kInt64 && counters.numel() >= 2,
                    "cpu routes: CUDA tensors, counters int64 [2]");
        TORCH_CHECK(ids.dim() == 2 && ids.is_contiguous() && slots.scalar_type() == torch::kInt32 && slots.is_contiguous() &&
                    x.dim() == 2 && x.size(0) == ids.size(0) && x.size(1) == hidden_ && x.is_contiguous() &&
                    cpu_routes.scalar_type() == torch::kInt32 && cpu_routes.numel() >= routes_ &&
                    cpu_count.scalar_type() == torch::kInt32 && cpu_count.numel() >= 2,
                    "cpu routes: ids [tokens][top], slots int32 [experts], x [tokens][hidden], cpu_routes int32 [routes], "
                    "cpu_count int32 [2]");
        const c10::cuda::CUDAGuard guard(x.device());
        const int tokens = int(ids.size(0)), top = int(ids.size(1));
        const std::size_t shared = std::size_t(tokens) * std::size_t(top + 1) * sizeof(int);
        TORCH_CHECK(tokens <= tokens_ && shared <= 48 * 1024, "cpu routes: at most ", tokens_, " tokens a call (the "
                    "handoff's capacity), and tokens x (top + 1) x 4 bytes within 48 KiB");
        const hm::DeviceSlot v = slots_.at(std::size_t(slot));
        auto stream = at::cuda::getCurrentCUDAStream();
        auto launch = [&](auto* id_ptr, auto* x_ptr) {
            post_kernel<<<1, 256, shared, stream>>>(id_ptr, tokens, top, slots.data_ptr<std::int32_t>(), int(slots.numel()),
                                                    x_ptr, int(hidden_), int(key_base), v, layout_, int(routes_),
                                                    int(tokens_), cpu_routes.data_ptr<std::int32_t>(),
                                                    cpu_count.data_ptr<std::int32_t>(),
                                                    reinterpret_cast<unsigned long long*>(counters.data_ptr<std::int64_t>()));
        };
        auto with_x = [&](auto* id_ptr) {
            if(x.scalar_type() == torch::kBFloat16) launch(id_ptr, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()));
            else if(x.scalar_type() == torch::kHalf) launch(id_ptr, reinterpret_cast<const __half*>(x.data_ptr()));
            else if(x.scalar_type() == torch::kFloat32) launch(id_ptr, x.data_ptr<float>());
            else TORCH_CHECK(false, "cpu routes: x is bf16, fp16 or fp32");
        };
        if(ids.scalar_type() == torch::kInt64) with_x(ids.data_ptr<std::int64_t>());
        else if(ids.scalar_type() == torch::kInt32) with_x(ids.data_ptr<std::int32_t>());
        else TORCH_CHECK(false, "cpu routes: ids are int64 or int32");
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // routes: the call's route count (tokens x top), the most its post can list.
    void scatter(std::int64_t slot, torch::Tensor& y, const torch::Tensor& cpu_routes, const torch::Tensor& cpu_count,
                 torch::Tensor& status, std::int64_t routes) {
        TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32 && y.is_contiguous() && y.size(1) == hidden_,
                    "cpu routes: y FP32 [routes][hidden] on the device");
        TORCH_CHECK(routes >= 1 && routes <= routes_ && routes <= y.size(0), "cpu routes: a call's routes within capacity");
        const c10::cuda::CUDAGuard guard(y.device());
        const hm::DeviceSlot v = slots_.at(std::size_t(slot));
        scatter_kernel<<<unsigned(routes), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            y.data_ptr<float>(), int(hidden_), v, layout_, cpu_routes.data_ptr<std::int32_t>(),
            cpu_count.data_ptr<std::int32_t>(), status.data_ptr<std::int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    std::int64_t payload_bytes() const { return layout_.bytes; }
    // The payload's offsets (experts, token, x, out, bytes), to check against the tier's input_layout.
    std::vector<std::int64_t> layout() const { return {layout_.experts, layout_.token, layout_.x, layout_.out, layout_.bytes}; }

private:
    std::int64_t hidden_, routes_, tokens_;
    Layout layout_;
    std::vector<hm::DeviceSlot> slots_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Routes>(m, "Routes")
        .def(pybind11::init<std::int64_t, std::int64_t, std::int64_t,
                            const std::vector<std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t>>&>())
        .def("post", &Routes::post)
        .def("scatter", &Routes::scatter)
        .def("payload_bytes", &Routes::payload_bytes)
        .def("layout", &Routes::layout);
}
