// A handoff's host side, its joins and the service thread; see handoff.h.
#include "handoff.h"

#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CRIT_HANDOFF_PAUSE() _mm_pause()
#else
#define CRIT_HANDOFF_PAUSE() std::this_thread::yield()
#endif

namespace crit::host {
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

__global__ void post_kernel(DeviceSlot slot) { post_job(slot); }
__global__ void skip_kernel(DeviceSlot slot) { skip_job(slot); }
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

Handoff::Handoff(MappedRegion region, std::size_t slots, std::size_t payload_bytes, Join join, Handler handler)
    : region_(std::move(region)), join_(join), handler_(std::move(handler)) {
    if(!slots || !region_.host || !region_.device || !handler_ || region_.bytes < region_bytes(slots, payload_bytes) ||
       reinterpret_cast<std::uintptr_t>(region_.host) % 16 || reinterpret_cast<std::uintptr_t>(region_.device) % 16)
        throw std::invalid_argument("handoff: slots, a handler and a 16-byte aligned mapped region of region_bytes()");
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
                                  DeviceSlot{reinterpret_cast<std::uint32_t*>(d), reinterpret_cast<std::uint32_t*>(d + 64),
                                             reinterpret_cast<std::uint32_t*>(d + 68), d + kLine},
                                  payload_bytes});
    }
    if(join_ != Join::host_eager) Service::instance().add(this);
}

Handoff::~Handoff() {
    released_.store(true, std::memory_order_release);
    for(std::size_t i = 0; i < slots_.size(); ++i) answer(i, released);
    stop_.store(true, std::memory_order_release);
    if(join_ != Join::host_eager) Service::instance().remove(this);  // a job that never returns blocks here
    if(sweeper_.joinable()) sweeper_.join();
}

const SlotView& Handoff::slot(std::size_t i) const {
    if(i >= slots_.size()) throw std::out_of_range("handoff slot");
    return slots_[i];
}

std::string Handoff::error() const {
    const std::lock_guard lock(error_mutex_);
    return error_;
}

void Handoff::answer(std::size_t i, Status s) noexcept {
    const SlotView& v = slots_[i];
    store(v.status, s);
    store(v.done, s == released ? kReleasedDone : 1u);
}

void Handoff::release() {
    released_.store(true, std::memory_order_release);
    for(std::size_t i = 0; i < slots_.size(); ++i) answer(i, released);
    const std::lock_guard lock(sweeper_mutex_);
    if(!sweeper_.joinable()) sweeper_ = std::thread([this] { sweep(); });
}

// While released, answers raised doorbells without the serve lock: the service thread may be inside a job that
// does not return, and the window's later slots must not wait for it.
void Handoff::sweep() {
    while(released_.load(std::memory_order_acquire) && !stop_.load(std::memory_order_acquire)) {
        for(std::size_t i = 0; i < slots_.size(); ++i)
            if(load(slots_[i].ready) != 0) {
                store(slots_[i].ready, 0);
                answer(i, released);
            }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

bool Handoff::rearm(std::chrono::milliseconds timeout) {
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
bool Handoff::serve_ready(std::size_t i) {
    const SlotView& v = slots_[i];
    if(load(v.ready) == 0) return false;
    const std::lock_guard lock(serve_mutex_);
    if(load(v.ready) == 0) return false;
    store(v.ready, 0);
    if(failed()) return answer(i, Status::failed), true;
    if(released_.load(std::memory_order_acquire)) return answer(i, released), true;
    struct Busy {   // the watch reads how long this job has run
        std::atomic<std::int64_t>& since;
        explicit Busy(std::atomic<std::int64_t>& s) : since(s) {
            since.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_release);
        }
        ~Busy() { since.store(0, std::memory_order_release); }
    } busy(busy_since_ns_);
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
            error_ = "a handoff's handler threw a non-standard exception";
        }
        failed_.store(true, std::memory_order_release);
        answer(i, Status::failed);
        return true;
    }
    if(failed()) return answer(i, Status::failed), true;   // it ran past the job budget: the watch answered for it
    served_.fetch_add(1, std::memory_order_relaxed);
    answer(i, served);
    return true;
}

std::chrono::milliseconds Handoff::busy() const noexcept {
    const auto since = busy_since_ns_.load(std::memory_order_acquire);
    if(since == 0) return std::chrono::milliseconds(0);
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::duration(now - since));
}

void Handoff::expire(std::chrono::milliseconds ran) {
    {
        const std::lock_guard lock(error_mutex_);
        if(error_.empty())
            error_ = "a host job of this handoff has run for " + std::to_string(ran.count()) +
                     " ms, past the service's job budget: the device's waits were released and what its window "
                     "computed is invalid";
    }
    failed_.store(true, std::memory_order_release);
    release();
}

// Polls the slot after the last one served first (layers usually post in order), and every slot now and then,
// so slices of one layer, out-of-order posts and windows aborted between slots are all answered.
bool Handoff::poll(std::uint32_t pass) {
    if(serve_ready(next_)) {
        next_ = (next_ + 1) % slots_.size();
        return true;
    }
    if((pass & 63u) == 0)
        for(std::size_t i = 0; i < slots_.size(); ++i)
            if(serve_ready(i)) {
                next_ = (i + 1) % slots_.size();
                return true;
            }
    return false;
}

Service& Service::instance() {
    static Service service;
    return service;
}

void Service::configure(int cpu, Pin pin, std::chrono::microseconds idle_after, std::chrono::microseconds idle_sleep,
                        std::chrono::milliseconds job_budget) {
    job_budget_ms_.store(job_budget.count(), std::memory_order_release);
    pin_.store(pin, std::memory_order_release);
    cpu_.store(cpu, std::memory_order_release);
    idle_after_us_.store(idle_after.count(), std::memory_order_release);
    idle_sleep_us_.store(idle_sleep.count(), std::memory_order_release);
}

// The watch: every tenth of a second, a job that has run past the budget fails its handoff. It takes no lock a
// job's thread holds, so a job that never returns cannot stop it.
void Service::watch() {
    std::unique_lock lock(watch_mutex_);
    while(!watch_stop_) {
        watch_wake_.wait_for(lock, std::chrono::milliseconds(100));
        const auto budget = job_budget_ms_.load(std::memory_order_acquire);
        if(budget <= 0) continue;
        for(Handoff* box : watched_) {
            const auto ran = box->busy();
            if(ran.count() > budget && !box->failed()) box->expire(ran);
        }
    }
}

void Service::add(Handoff* box) {
    {
        const std::lock_guard lock(watch_mutex_);
        watched_.push_back(box);
        if(!watch_thread_.joinable()) watch_thread_ = std::thread([this] { watch(); });
    }
    waiting_.fetch_add(1, std::memory_order_acq_rel);
    {
        const std::lock_guard lock(mutex_);
        boxes_.push_back(box);
        if(!thread_.joinable()) {
            stop_ = std::make_shared<std::atomic<bool>>(false);
            thread_ = std::thread([this, stop = stop_] { loop(stop); });
        }
    }
    waiting_.fetch_sub(1, std::memory_order_acq_rel);
}

void Service::remove(Handoff* box) {
    {   // first: the watch reads the handoff no more
        const std::lock_guard lock(watch_mutex_);
        watched_.erase(std::remove(watched_.begin(), watched_.end(), box), watched_.end());
    }
    std::thread ended;
    waiting_.fetch_add(1, std::memory_order_acq_rel);
    {
        const std::lock_guard lock(mutex_);
        boxes_.erase(std::remove(boxes_.begin(), boxes_.end(), box), boxes_.end());
        if(boxes_.empty() && thread_.joinable()) {   // the last handoff left: the thread ends with it
            stop_->store(true, std::memory_order_release);
            ended = std::move(thread_);
        }
    }
    waiting_.fetch_sub(1, std::memory_order_acq_rel);
    if(ended.joinable()) ended.join();
}

Service::~Service() {
    {
        const std::lock_guard lock(watch_mutex_);
        watch_stop_ = true;
    }
    watch_wake_.notify_all();
    if(watch_thread_.joinable()) watch_thread_.join();
    std::thread ended;
    {
        const std::lock_guard lock(mutex_);
        if(stop_) stop_->store(true, std::memory_order_release);
        ended = std::move(thread_);
    }
    if(ended.joinable()) ended.join();
}

void Service::loop(std::shared_ptr<std::atomic<bool>> stop) {
    using clock = std::chrono::steady_clock;
    auto last = clock::now();
    int pinned = -1;
    for(std::uint32_t pass = 1; !stop->load(std::memory_order_acquire); ++pass) {
        const int cpu = cpu_.load(std::memory_order_acquire);
        if(cpu != pinned) {   // the plan's CPU for the service, when it is given or changes
            const Pin pin = pin_.load(std::memory_order_acquire);
            if(cpu >= 0 && pin) pin(cpu);
            pinned = cpu;
        }
        bool found = false;
        if(waiting_.load(std::memory_order_acquire) == 0) {   // a handoff arriving or leaving goes first
            const std::lock_guard lock(mutex_);
            for(Handoff* box : boxes_) found = box->poll(pass) || found;
        } else {
            std::this_thread::yield();
        }
        if(found) {
            last = clock::now();
        } else if(clock::now() - last > std::chrono::microseconds(idle_after_us_.load(std::memory_order_relaxed))) {
            std::this_thread::sleep_for(std::chrono::microseconds(idle_sleep_us_.load(std::memory_order_relaxed)));
        } else {
            CRIT_HANDOFF_PAUSE();
        }
    }
}

cudaError_t Handoff::post(std::size_t i, cudaStream_t stream) const {
    post_kernel<<<1, 1, 0, stream>>>(slot(i).device);
    return cudaGetLastError();
}

cudaError_t Handoff::skip(std::size_t i, cudaStream_t stream) const {
    skip_kernel<<<1, 1, 0, stream>>>(slot(i).device);
    return cudaGetLastError();
}

cudaError_t Handoff::join(std::size_t i, cudaStream_t stream) {
    const SlotView& v = slot(i);
    switch(join_) {
        case Join::stream_wait: return stream_wait(stream, v.device.done);
        case Join::device_poll:
            poll_kernel<<<1, 1, 0, stream>>>(v.device.done);
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

namespace {
// Makes `device` current for a scope and gives the caller's back.
struct DeviceScope {
    int previous = -1;
    explicit DeviceScope(int device) {
        if(cudaGetDevice(&previous) != cudaSuccess) previous = -1;
        cudaSetDevice(device);
    }
    ~DeviceScope() {
        if(previous >= 0) cudaSetDevice(previous);
    }
};
}  // namespace

JoinSupport probe_joins(int device) {
    JoinSupport out;
    const DeviceScope scope(device);
    int current = -1;
    if(cudaGetDevice(&current) != cudaSuccess || current != device) return out.detail = "cudaSetDevice failed", out;
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
        Handoff box(MappedRegion{host, dev, bytes, owner}, 1, 64, join, [](std::size_t, std::span<std::byte>) {});
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
    const DeviceScope scope(device);
    out.support = probe_joins(device);
    if(!out.support.mapped) return out;
    out.device_poll_us = measure(Join::device_poll, joins_per_graph, replays);
    if(out.support.stream_wait_captured) out.stream_wait_us = measure(Join::stream_wait, joins_per_graph, replays);
    return out;
}

}  // namespace crit::host
