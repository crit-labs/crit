// crit_plugins.host's handoffs from Python: a Handoff (handoff.h) over mapped pinned memory this object owns, whose
// handler is a C function of a host library, joined on torch's current stream; the measured cost of the capturable
// joins; and the service thread's settings. Built at first use through crit_plugins.native.extension.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include "handoff.h"

namespace ch = crit::host;

namespace {

// A job's handler in a host library: 0 when it served the slot's payload, else 1 with the error's text.
using Serve = int (*)(void* context, std::uint64_t slot, void* payload, std::uint64_t bytes, char* error,
                      std::uint64_t error_bytes);

class Handoff {
public:
    Handoff(std::int64_t device, std::int64_t slots, std::int64_t payload_bytes, const std::string& join,
            std::int64_t serve, std::int64_t context)
        : device_(int(device)), payload_bytes_(payload_bytes) {
        const ch::Join kind = join == "stream_wait" ? ch::Join::stream_wait
                              : join == "device_poll" ? ch::Join::device_poll : ch::Join::host_eager;
        const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
        bytes_ = ch::region_bytes(std::size_t(slots), std::size_t(payload_bytes));
        void* host = nullptr;
        void* mapped = nullptr;
        if(cudaHostAlloc(&host, bytes_, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
            throw std::runtime_error("handoff: no mapped pinned memory for its slots");
        std::shared_ptr<void> owner(host, [](void* p) { cudaFreeHost(p); });
        if(cudaHostGetDevicePointer(&mapped, host, 0) != cudaSuccess)
            throw std::runtime_error("handoff: no device address for its mapped memory");
        const Serve fn = reinterpret_cast<Serve>(serve);
        void* ctx = reinterpret_cast<void*>(context);
        if(!fn) throw std::invalid_argument("handoff: a handler (the address of a C function)");
        ch::Handler handler = [fn, ctx](std::size_t slot, std::span<std::byte> payload) {
            char error[1024] = {};
            if(fn(ctx, slot, payload.data(), payload.size(), error, sizeof(error))) throw std::runtime_error(error);
        };
        box_ = std::make_unique<ch::Handoff>(ch::MappedRegion{host, mapped, bytes_, owner}, std::size_t(slots),
                                             std::size_t(payload_bytes), kind, std::move(handler));
    }

    // The slot's device addresses: ready, done, status, payload (handoff_device.cuh's DeviceSlot).
    std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t> words(std::int64_t slot) const {
        const ch::DeviceSlot& d = box_->slot(std::size_t(slot)).device;
        return {reinterpret_cast<std::int64_t>(d.ready), reinterpret_cast<std::int64_t>(d.done),
                reinterpret_cast<std::int64_t>(d.status), reinterpret_cast<std::int64_t>(d.payload)};
    }

    void post(std::int64_t slot) { check(box_->post(std::size_t(slot), stream()), "post"); }
    void skip(std::int64_t slot) { check(box_->skip(std::size_t(slot), stream()), "skip"); }
    void join(std::int64_t slot) { check(box_->join(std::size_t(slot), stream()), "join"); }
    void release() { box_->release(); }
    bool rearm(std::int64_t timeout_ms) { return box_->rearm(std::chrono::milliseconds(timeout_ms)); }
    bool failed() const { return box_->failed(); }
    std::string error() const { return box_->error(); }
    std::int64_t served() const { return std::int64_t(box_->served_jobs()); }
    std::string join_kind() const { return ch::join_name(box_->join_kind()); }
    std::int64_t slots() const { return std::int64_t(box_->slots()); }
    std::int64_t payload_bytes() const { return payload_bytes_; }
    std::int64_t bytes() const { return std::int64_t(bytes_); }

private:
    cudaStream_t stream() const { return at::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_)); }
    static void check(cudaError_t e, const char* what) {
        TORCH_CHECK(e == cudaSuccess, "handoff: ", what, ": ", cudaGetErrorString(e));
    }
    int device_;
    std::int64_t payload_bytes_;
    std::size_t bytes_ = 0;
    std::unique_ptr<ch::Handoff> box_;
};

pybind11::dict measure(std::int64_t device, std::int64_t joins, std::int64_t replays) {
    const ch::JoinCosts c = ch::measure_joins(int(device), int(joins), int(replays));
    pybind11::dict out;
    out["mapped"] = c.support.mapped;
    out["stream_wait"] = c.support.stream_wait_captured;
    out["detail"] = c.support.detail;
    out["stream_wait_us"] = c.stream_wait_us ? pybind11::cast(*c.stream_wait_us) : pybind11::none();
    out["device_poll_us"] = c.device_poll_us ? pybind11::cast(*c.device_poll_us) : pybind11::none();
    out["choice"] = std::string(ch::join_name(c.choice()));
    return out;
}

// cpu: the logical CPU the service thread is pinned to (< 0: none) by `pin`, the address of a C function
// int(int cpu) that pins the calling thread (the pool's crit_host_pin_thread; 0: no pinning).
// job_budget_ms: the longest a handoff's job may run before the service's watch fails the handoff and releases the
// device's waits (0: no watch).
void service(std::int64_t cpu, std::int64_t pin, std::int64_t idle_after_us, std::int64_t idle_sleep_us,
             std::int64_t job_budget_ms) {
    ch::Service::instance().configure(int(cpu), reinterpret_cast<ch::Service::Pin>(pin),
                                      std::chrono::microseconds(idle_after_us), std::chrono::microseconds(idle_sleep_us),
                                      std::chrono::milliseconds(job_budget_ms));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Handoff>(m, "Handoff")
        .def(pybind11::init<std::int64_t, std::int64_t, std::int64_t, const std::string&, std::int64_t, std::int64_t>())
        .def("words", &Handoff::words)
        .def("post", &Handoff::post)
        .def("skip", &Handoff::skip)
        .def("join", &Handoff::join)
        .def("release", &Handoff::release)
        .def("rearm", &Handoff::rearm)
        .def("failed", &Handoff::failed)
        .def("error", &Handoff::error)
        .def("served", &Handoff::served)
        .def("join_kind", &Handoff::join_kind)
        .def("slots", &Handoff::slots)
        .def("payload_bytes", &Handoff::payload_bytes)
        .def("bytes", &Handoff::bytes);
    m.def("measure_joins", &measure);
    m.def("service", &service);
}
