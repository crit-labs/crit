#include "mailbox.h"
#include "mailbox_device.cuh"

#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CRIT_MAILBOX_PAUSE() _mm_pause()
#else
#define CRIT_MAILBOX_PAUSE() std::this_thread::yield()
#endif

namespace crit::techniques::host_mailbox {
namespace {

using WaitValue32 = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);

// Resolved through the runtime, so nothing links the driver library directly. Null when the driver lacks it.
WaitValue32 wait_value32() {
    static const WaitValue32 fn = [] {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult found{};
        if(cudaGetDriverEntryPointByVersion("cuStreamWaitValue32", &p, 12000, cudaEnableDefault, &found) != cudaSuccess ||
           found != cudaDriverEntryPointSuccess)
            p = nullptr;
        return reinterpret_cast<WaitValue32>(p);
    }();
    return fn;
}

std::uint32_t load(const std::uint32_t* p) noexcept {
    return std::atomic_ref<std::uint32_t>(*const_cast<std::uint32_t*>(p)).load(std::memory_order_acquire);
}
void store(std::uint32_t* p, std::uint32_t v) noexcept {
    std::atomic_ref<std::uint32_t>(*p).store(v, std::memory_order_release);
}
std::size_t round_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

__global__ void post_kernel(std::uint32_t* ready, std::uint32_t* done, std::uint32_t* status) {
    post_job(ready, done, status);
}
__global__ void skip_kernel(std::uint32_t* done, std::uint32_t* status) { skip_job(done, status); }
// One thread waits on the done word; release() and failure answering guarantee it ends.
__global__ void poll_kernel(const std::uint32_t* done) {
    const volatile std::uint32_t* p = done;
    while(*p == 0u) __nanosleep(64);
    __threadfence_system();
}

cudaError_t stream_wait(cudaStream_t stream, const std::uint32_t* device_word) {
    const auto fn = wait_value32();
    if(!fn) return cudaErrorNotSupported;
    const CUresult r = fn(reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(device_word), 1u,
                          CU_STREAM_WAIT_VALUE_GEQ);
    return r == CUDA_SUCCESS ? cudaSuccess : cudaErrorNotSupported;
}

}  // namespace

const char* join_name(Join j) noexcept {
    switch(j) {
        case Join::stream_wait: return "stream_wait";
        case Join::device_poll: return "device_poll";
        default: return "host_eager";
    }
}

std::size_t region_bytes(std::size_t slots, std::size_t payload_bytes) {
    return slots * (kLine + round_up(payload_bytes, kLine));
}

Mailbox::Mailbox(MappedRegion region, std::size_t slots, std::size_t payload_bytes, Join join, Handler handler,
                 Options options)
    : region_(std::move(region)), join_(join), handler_(std::move(handler)), options_(options) {
    if(!slots || !region_.host || !region_.device || !handler_ || region_.bytes < region_bytes(slots, payload_bytes) ||
       reinterpret_cast<std::uintptr_t>(region_.host) % 16 || reinterpret_cast<std::uintptr_t>(region_.device) % 16)
        throw std::invalid_argument("mailbox: slots, a handler and a 16-byte aligned mapped region of region_bytes()");
    std::memset(region_.host, 0, region_bytes(slots, payload_bytes));
    const std::size_t stride = kLine + round_up(payload_bytes, kLine);
    auto* host = static_cast<std::byte*>(region_.host);
    auto* device = static_cast<std::byte*>(region_.device);
    for(std::size_t i = 0; i < slots; ++i) {
        std::byte* h = host + i * stride;
        std::byte* d = device + i * stride;
        // ready on its own half line from done/status: the device writes ready, the host writes done.
        slots_.push_back(SlotView{reinterpret_cast<std::uint32_t*>(h), reinterpret_cast<std::uint32_t*>(h + 64),
                                  reinterpret_cast<std::uint32_t*>(h + 68), h + kLine,
                                  reinterpret_cast<std::uint32_t*>(d), reinterpret_cast<std::uint32_t*>(d + 64),
                                  reinterpret_cast<std::uint32_t*>(d + 68), d + kLine, payload_bytes});
    }
    if(join_ != Join::host_eager) service_ = std::thread([this] { service(); });
}

Mailbox::~Mailbox() {
    released_.store(true, std::memory_order_release);
    for(std::size_t i = 0; i < slots_.size(); ++i) answer(i, released);
    stop_.store(true, std::memory_order_release);
    if(service_.joinable()) service_.join();  // a job that never returns blocks here: handlers must be stoppable
    if(sweeper_.joinable()) sweeper_.join();
}

const SlotView& Mailbox::slot(std::size_t i) const {
    if(i >= slots_.size()) throw std::out_of_range("mailbox slot");
    return slots_[i];
}

std::string Mailbox::error() const {
    const std::lock_guard lock(error_mutex_);
    return error_;
}

void Mailbox::answer(std::size_t i, Status s) noexcept {
    const SlotView& v = slots_[i];
    store(v.status, s);
    store(v.done, s == released ? kReleasedDone : 1u);
}

void Mailbox::release() {
    released_.store(true, std::memory_order_release);
    for(std::size_t i = 0; i < slots_.size(); ++i) answer(i, released);
    const std::lock_guard lock(sweeper_mutex_);
    if(!sweeper_.joinable()) sweeper_ = std::thread([this] { sweep(); });
}

// While released, answers raised doorbells without the serve lock: the service thread may be inside a job that
// does not return, and the window's later slots must not wait for it.
void Mailbox::sweep() {
    while(released_.load(std::memory_order_acquire) && !stop_.load(std::memory_order_acquire)) {
        for(std::size_t i = 0; i < slots_.size(); ++i)
            if(load(slots_[i].ready) != 0) {
                store(slots_[i].ready, 0);
                answer(i, released);
            }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

bool Mailbox::rearm(std::chrono::milliseconds timeout) {
    if(failed()) return false;
    std::unique_lock lock(serve_mutex_, std::defer_lock);
    if(!lock.try_lock_for(timeout)) return false;
    released_.store(false, std::memory_order_release);
    {
        const std::lock_guard sweeper(sweeper_mutex_);
        if(sweeper_.joinable()) sweeper_.join();
    }
    for(const SlotView& v : slots_) {  // the streams are drained: nothing posts while the words are cleared
        store(v.ready, 0);
        store(v.status, pending);
        store(v.done, 0);
    }
    return !failed();
}

// Claims a raised doorbell and answers it. Returns false when the slot was not raised.
bool Mailbox::serve_ready(std::size_t i) {
    const SlotView& v = slots_[i];
    if(load(v.ready) == 0) return false;
    const std::lock_guard lock(serve_mutex_);
    if(load(v.ready) == 0) return false;
    store(v.ready, 0);
    if(failed()) return answer(i, Status::failed), true;
    if(released_.load(std::memory_order_acquire)) return answer(i, released), true;
    try {
        handler_(i, std::span<std::byte>(v.payload, v.payload_bytes));
    } catch(const std::exception& e) {
        {
            const std::lock_guard elock(error_mutex_);
            error_ = e.what();
        }
        failed_.store(true, std::memory_order_release);
        answer(i, Status::failed);
        return true;
    } catch(...) {
        {
            const std::lock_guard elock(error_mutex_);
            error_ = "mailbox handler threw a non-standard exception";
        }
        failed_.store(true, std::memory_order_release);
        answer(i, Status::failed);
        return true;
    }
    served_.fetch_add(1, std::memory_order_relaxed);
    answer(i, served);
    return true;
}

// Polls the slot after the last one served first (layers usually post in order), and every slot now and then,
// so slices of one layer, out-of-order posts and windows aborted between slots are all answered.
void Mailbox::service() {
    using clock = std::chrono::steady_clock;
    std::size_t next = 0;
    auto last = clock::now();
    for(std::uint32_t polls = 1; !stop_.load(std::memory_order_acquire); ++polls) {
        bool found = false;
        if(serve_ready(next)) {
            next = (next + 1) % slots_.size();
            found = true;
        } else if((polls & 63u) == 0) {
            for(std::size_t i = 0; i < slots_.size() && !found; ++i)
                if(serve_ready(i)) {
                    next = (i + 1) % slots_.size();
                    found = true;
                }
        }
        if(found) {
            last = clock::now();
        } else if(clock::now() - last > options_.idle_after) {
            std::this_thread::sleep_for(options_.idle_sleep);
        } else {
            CRIT_MAILBOX_PAUSE();
        }
    }
}

cudaError_t Mailbox::post(std::size_t i, cudaStream_t stream) const {
    const SlotView& v = slot(i);
    post_kernel<<<1, 1, 0, stream>>>(v.device_ready, v.device_done, v.device_status);
    return cudaGetLastError();
}

cudaError_t Mailbox::skip(std::size_t i, cudaStream_t stream) const {
    const SlotView& v = slot(i);
    skip_kernel<<<1, 1, 0, stream>>>(v.device_done, v.device_status);
    return cudaGetLastError();
}

cudaError_t Mailbox::join(std::size_t i, cudaStream_t stream) {
    const SlotView& v = slot(i);
    switch(join_) {
        case Join::stream_wait: return stream_wait(stream, v.device_done);
        case Join::device_poll:
            poll_kernel<<<1, 1, 0, stream>>>(v.device_done);
            return cudaGetLastError();
        default: break;
    }
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    if(const cudaError_t e = cudaStreamIsCapturing(stream, &capturing); e != cudaSuccess) return e;
    if(capturing != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    if(const cudaError_t e = cudaStreamSynchronize(stream); e != cudaSuccess) return e;
    serve_ready(i);
    return load(v.done) != 0 ? cudaSuccess : cudaErrorNotReady;  // NotReady: joined a slot nothing posted
}

namespace {

struct HostWord {
    std::uint32_t* host = nullptr;
    std::uint32_t* device = nullptr;
    HostWord() {
        if(cudaHostAlloc(reinterpret_cast<void**>(&host), 4096, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            host = nullptr;
            return;
        }
        std::memset(host, 0, 4096);
        if(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0) != cudaSuccess) device = nullptr;
    }
    ~HostWord() {
        if(host) cudaFreeHost(host);
    }
};

// Raises the word after the wait is enqueued and reports whether the stream completed within the deadline.
bool completes(cudaStream_t s, std::uint32_t* word) {
    store(word, 1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while(std::chrono::steady_clock::now() < deadline) {
        const cudaError_t q = cudaStreamQuery(s);
        if(q == cudaSuccess) return true;
        if(q != cudaErrorNotReady) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return false;
}

}  // namespace

JoinSupport probe_joins(int device) {
    JoinSupport out;
    if(cudaSetDevice(device) != cudaSuccess) return out.detail = "cudaSetDevice failed", out;
    int map = 0;
    cudaDeviceGetAttribute(&map, cudaDevAttrCanMapHostMemory, device);
    HostWord word;
    out.mapped = map && word.host && word.device;
    if(!out.mapped) return out.detail = "no mapped pinned memory", out;
    if(!wait_value32()) return out.detail = "driver has no cuStreamWaitValue32", out;
    cudaStream_t s = nullptr;
    if(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return out.detail = "stream", out;
    // Eager: the wait is enqueued before the host raises the word.
    store(word.host, 0);
    out.stream_wait = stream_wait(s, word.device) == cudaSuccess && completes(s, word.host);
    if(!out.stream_wait) {
        store(word.host, kReleasedDone);
        out.detail = "eager stream wait on mapped memory failed";
        // A stream still waiting cannot be destroyed safely; leave it to process teardown.
        if(cudaStreamQuery(s) == cudaSuccess) cudaStreamDestroy(s);
        return out;
    }
    // Captured: the same wait recorded into a graph, instantiated and replayed.
    store(word.host, 0);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool captured = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    const bool enqueued = captured && stream_wait(s, word.device) == cudaSuccess;
    if(captured) captured = cudaStreamEndCapture(s, &graph) == cudaSuccess && enqueued;
    if(captured) captured = cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
    if(captured) captured = cudaGraphLaunch(exec, s) == cudaSuccess && completes(s, word.host);
    out.stream_wait_captured = captured;
    if(!captured) {
        store(word.host, kReleasedDone);
        out.detail = "captured stream wait on mapped memory failed";
    }
    if(cudaStreamQuery(s) == cudaSuccess) {
        if(exec) cudaGraphExecDestroy(exec);
        if(graph) cudaGraphDestroy(graph);
        cudaStreamDestroy(s);
    }
    cudaGetLastError();
    return out;
}

Join JoinCosts::choice() const noexcept {
    if(stream_wait_us && (!device_poll_us || *stream_wait_us <= *device_poll_us)) return Join::stream_wait;
    if(device_poll_us) return Join::device_poll;
    return Join::host_eager;
}

namespace {

std::optional<double> measure(Join join, int joins, int replays) {
    void* host = nullptr;
    void* dev = nullptr;
    const std::size_t bytes = region_bytes(1, 64);
    if(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return {};
    std::shared_ptr<void> owner(host, [](void* p) { cudaFreeHost(p); });
    if(cudaHostGetDevicePointer(&dev, host, 0) != cudaSuccess) return {};
    std::optional<double> result;
    cudaStream_t s = nullptr;
    if(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return {};
    {
        Mailbox box(MappedRegion{host, dev, bytes, owner}, 1, 64, join, [](std::size_t, std::span<std::byte>) {});
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaEvent_t a = nullptr, b = nullptr;
        bool ok = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        for(int i = 0; ok && i < joins; ++i) ok = box.post(0, s) == cudaSuccess && box.join(0, s) == cudaSuccess;
        const cudaError_t ended = cudaStreamEndCapture(s, &graph);
        ok = ok && ended == cudaSuccess && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess &&
             cudaEventCreate(&a) == cudaSuccess && cudaEventCreate(&b) == cudaSuccess;
        std::vector<double> per;
        for(int r = 0; ok && r < replays + 5; ++r) {
            ok = cudaEventRecord(a, s) == cudaSuccess && cudaGraphLaunch(exec, s) == cudaSuccess &&
                 cudaEventRecord(b, s) == cudaSuccess && cudaEventSynchronize(b) == cudaSuccess;
            float ms = 0.f;
            if(ok && r >= 5 && cudaEventElapsedTime(&ms, a, b) == cudaSuccess) per.push_back(ms * 1000.0 / joins);
        }
        cudaStreamSynchronize(s);
        if(ok && !per.empty()) {
            std::nth_element(per.begin(), per.begin() + per.size() / 2, per.end());
            result = per[per.size() / 2];
        }
        if(a) cudaEventDestroy(a);
        if(b) cudaEventDestroy(b);
        if(exec) cudaGraphExecDestroy(exec);
        if(graph) cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(s);
    cudaGetLastError();
    return result;
}

}  // namespace

JoinCosts measure_joins(int device, int joins_per_graph, int replays) {
    JoinCosts out;
    out.support = probe_joins(device);
    if(!out.support.mapped) return out;
    out.device_poll_us = measure(Join::device_poll, joins_per_graph, replays);
    if(out.support.stream_wait_captured) out.stream_wait_us = measure(Join::stream_wait, joins_per_graph, replays);
    return out;
}

}  // namespace crit::techniques::host_mailbox
