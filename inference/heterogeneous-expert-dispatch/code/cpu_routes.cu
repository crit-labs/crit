// A routed layer's CPU share joined inside a CUDA stream (captured graphs too), over host-work-mailbox's Mailbox: one
// kernel finds the routes whose expert the slot table sends to the CPU tier (slot -2), writes them into a mailbox slot's
// payload (each route's expert key and token, the tokens' rows in FP32) and posts the slot, or answers it itself when
// there is nothing for the CPU; after the stream's join a second kernel copies the CPU's rows into the device part's
// route rows, where the device part's combine sums every route in route order. One post and one join per layer: the
// next layer's input needs this layer's sum, so a decode step cannot join less often; a layer with no CPU routes costs
// its post kernel and a join that passes at once.
//
// The payload is host-expert-tier's InputLayout: header {count, tokens, route capacity, token capacity} (uint32), experts
// int32 [routes], token int32 [routes], x FP32 [tokens][hidden], out FP32 [routes][hidden], each part 64-byte aligned;
// its handler is host-expert-tier's crit_tier_serve, given to the mailbox as a C function and its context.
//
// Mechanism code, built at load through torch's extension builder with host-work-mailbox's mailbox.cu (the Python layer:
// device_dispatch.py). Bindings: pybind11 over torch tensors on the current stream.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mailbox.h"
#include "mailbox_device.cuh"

namespace hm = crit::techniques::host_mailbox;

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
                            int hidden, int key_base, hm::SlotView slot, Layout layout, int cap_routes, int cap_tokens,
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
    std::byte* payload = slot.device_payload;
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
        if(count > 0) hm::post_job(slot.device_ready, slot.device_done, slot.device_status);
        else hm::skip_job(slot.device_done, slot.device_status);
    }
}

// After the join: route i's row of the payload into y at its route position. A slot the host failed or released leaves
// NaN rows (never a silent partial sum) and sets the status word the caller checks. Reads of mapped host memory cross
// the link and serialize, so only the blocks of real routes read the status, once each, after the route count (device
// memory) says whether they have work; a slot the post answered itself (no CPU routes) is read from nowhere.
__global__ void scatter_kernel(float* y, int hidden, hm::SlotView slot, Layout layout, const std::int32_t* cpu_routes,
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
        status = hm::read_status(slot.device_status);
        if(i == 0) status_out[0] = int(status);
    }
    __syncthreads();
    const bool good = status == hm::served;
    const auto* out = reinterpret_cast<const float*>(slot.device_payload + layout.out);
    const int r = cpu_routes[i];
    for(int h = int(threadIdx.x); h < hidden; h += int(blockDim.x))
        y[std::size_t(r) * std::size_t(hidden) + std::size_t(h)] =
            good ? hm::read_result(out + std::size_t(i) * std::size_t(hidden) + std::size_t(h)) : __int_as_float(0x7fc00000);
}

using Serve = int (*)(void*, std::uint64_t, void*, std::uint64_t, char*, std::uint64_t);

// A mailbox over mapped pinned memory this object owns, whose handler is a C function of a host library (the tier).
class Box {
public:
    Box(std::int64_t slots, std::int64_t hidden, std::int64_t routes, std::int64_t tokens, const std::string& join,
        std::int64_t serve, std::int64_t context, std::int64_t idle_after_us, std::int64_t idle_sleep_us)
        : hidden_(hidden), routes_(routes), tokens_(tokens), layout_(layout_of(hidden, routes, tokens)) {
        const hm::Join kind = join == "stream_wait" ? hm::Join::stream_wait
                              : join == "device_poll" ? hm::Join::device_poll : hm::Join::host_eager;
        const std::size_t bytes = hm::region_bytes(std::size_t(slots), std::size_t(layout_.bytes));
        void* host = nullptr;
        void* device = nullptr;
        if(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
            throw std::runtime_error("cpu routes: mapped pinned memory for the mailbox");
        std::shared_ptr<void> owner(host, [](void* p) { cudaFreeHost(p); });
        if(cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess)
            throw std::runtime_error("cpu routes: no device address for the mailbox's mapped memory");
        const Serve fn = reinterpret_cast<Serve>(serve);
        void* ctx = reinterpret_cast<void*>(context);
        hm::Handler handler = [fn, ctx](std::size_t slot, std::span<std::byte> payload) {
            char error[1024] = {};
            if(fn(ctx, slot, payload.data(), payload.size(), error, sizeof(error)))
                throw std::runtime_error(std::string("CPU tier: ") + error);
        };
        hm::Options options;
        options.idle_after = std::chrono::microseconds(idle_after_us);
        options.idle_sleep = std::chrono::microseconds(idle_sleep_us);
        box_ = std::make_unique<hm::Mailbox>(hm::MappedRegion{host, device, bytes, owner}, std::size_t(slots),
                                             std::size_t(layout_.bytes), kind, std::move(handler), options);
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
                    "mailbox's capacity), and tokens x (top + 1) x 4 bytes within 48 KiB");
        const hm::SlotView& v = box_->slot(std::size_t(slot));
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

    void join(std::int64_t slot) {
        const cudaError_t e = box_->join(std::size_t(slot), at::cuda::getCurrentCUDAStream());
        TORCH_CHECK(e == cudaSuccess, "cpu routes: join: ", cudaGetErrorString(e));
    }

    // routes: the call's route count (tokens x top), the most its post can list.
    void scatter(std::int64_t slot, torch::Tensor& y, const torch::Tensor& cpu_routes, const torch::Tensor& cpu_count,
                 torch::Tensor& status, std::int64_t routes) {
        TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat32 && y.is_contiguous() && y.size(1) == hidden_,
                    "cpu routes: y FP32 [routes][hidden] on the device");
        TORCH_CHECK(routes >= 1 && routes <= routes_ && routes <= y.size(0), "cpu routes: a call's routes within capacity");
        const c10::cuda::CUDAGuard guard(y.device());
        const hm::SlotView& v = box_->slot(std::size_t(slot));
        scatter_kernel<<<unsigned(routes), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            y.data_ptr<float>(), int(hidden_), v, layout_, cpu_routes.data_ptr<std::int32_t>(),
            cpu_count.data_ptr<std::int32_t>(), status.data_ptr<std::int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    void release() { box_->release(); }
    bool rearm(std::int64_t timeout_ms) { return box_->rearm(std::chrono::milliseconds(timeout_ms)); }
    bool failed() const { return box_->failed(); }
    std::string error() const { return box_->error(); }
    std::int64_t served() const { return std::int64_t(box_->served_jobs()); }
    std::string join_kind() const { return hm::join_name(box_->join_kind()); }
    std::int64_t payload_bytes() const { return layout_.bytes; }
    // The payload's offsets (experts, token, x, out, bytes), to check against the tier's input_layout.
    std::vector<std::int64_t> layout() const { return {layout_.experts, layout_.token, layout_.x, layout_.out, layout_.bytes}; }

private:
    std::int64_t hidden_, routes_, tokens_;
    Layout layout_;
    std::unique_ptr<hm::Mailbox> box_;
};

pybind11::dict measure(std::int64_t device, std::int64_t joins, std::int64_t replays) {
    const hm::JoinCosts c = hm::measure_joins(int(device), int(joins), int(replays));
    pybind11::dict out;
    out["mapped"] = c.support.mapped;
    out["stream_wait"] = c.support.stream_wait_captured;
    out["detail"] = c.support.detail;
    out["stream_wait_us"] = c.stream_wait_us ? pybind11::cast(*c.stream_wait_us) : pybind11::none();
    out["device_poll_us"] = c.device_poll_us ? pybind11::cast(*c.device_poll_us) : pybind11::none();
    out["choice"] = std::string(hm::join_name(c.choice()));
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Box>(m, "Box")
        .def(pybind11::init<std::int64_t, std::int64_t, std::int64_t, std::int64_t, const std::string&, std::int64_t,
                            std::int64_t, std::int64_t, std::int64_t>())
        .def("post", &Box::post)
        .def("join", &Box::join)
        .def("scatter", &Box::scatter)
        .def("release", &Box::release)
        .def("rearm", &Box::rearm)
        .def("failed", &Box::failed)
        .def("error", &Box::error)
        .def("served", &Box::served)
        .def("join_kind", &Box::join_kind)
        .def("payload_bytes", &Box::payload_bytes)
        .def("layout", &Box::layout);
    m.def("measure_joins", &measure);
}
