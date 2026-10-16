#include "crit/cuda.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace crit::core {
namespace {
void check(cudaError_t status, const char* operation) {
    if(status != cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorName(status)+": "+cudaGetErrorString(status));
}
void log(cudaError_t status, const char* operation) noexcept {
    if(status != cudaSuccess) std::fprintf(stderr,"crit CUDA cleanup %s: %s\n",operation,cudaGetErrorName(status));
}
unsigned flags(SyncSchedule schedule) {
    switch(schedule) {
    case SyncSchedule::spin: return cudaDeviceScheduleSpin;
    case SyncSchedule::blocking: return cudaDeviceScheduleBlockingSync;
    case SyncSchedule::yield: return cudaDeviceScheduleYield;
    case SyncSchedule::automatic: return cudaDeviceScheduleAuto;
    }
    throw std::invalid_argument("invalid CUDA synchronization schedule");
}
std::atomic<std::size_t> allocation_quarantine{0}, backend_quarantine{0};
}
struct CudaState {
    int device;
    cudaDeviceProp properties{};
    cudaStream_t compute = nullptr, transfer = nullptr;
    std::mutex mutex;
    std::optional<Resources> resources;
    CudaState* retained_next = nullptr;
    explicit CudaState(int d):device(d) {}
    void bind() const { check(cudaSetDevice(device),"cudaSetDevice"); }
    cudaError_t settle() const noexcept {
        auto status = cudaSetDevice(device);
        if(status != cudaSuccess) return status;
        // Attempt both streams even if the first reports an error.
        const auto first = compute ? cudaStreamSynchronize(compute) : cudaSuccess;
        const auto second = transfer ? cudaStreamSynchronize(transfer) : cudaSuccess;
        return first != cudaSuccess ? first : second;
    }
    cudaStream_t stream(Queue queue) const {
        if(queue == Queue::compute) return compute;
        if(queue == Queue::transfer) return transfer;
        throw std::invalid_argument("invalid CUDA queue");
    }
    void require_stream(cudaStream_t s) const {
        if(!s || (s != compute && s != transfer)) throw std::invalid_argument("foreign/null CUDA stream");
    }
    ~CudaState() {
        log(cudaSetDevice(device),"bind backend destruction");
        if(transfer) log(cudaStreamDestroy(transfer),"destroy transfer stream");
        if(compute) log(cudaStreamDestroy(compute),"destroy compute stream");
    }
};
namespace {
void release_state(CudaState* state) noexcept {
    const auto status = state->settle();
    if(status == cudaSuccess) { delete state; return; }
    log(status,"drain backend; retaining owner");
    // Intrusive quarantine allocates no metadata during failed destruction.
    static CudaState* retained = nullptr;
    static std::mutex mutex;
    try { std::lock_guard lock(mutex); state->retained_next = retained; retained = state; }
    catch(...) { /* keep the pointed-to state even if diagnostics cannot grow */ }
    ++backend_quarantine;
}
struct Storage {
    std::shared_ptr<CudaState> state;
    Reservation reservation;
    Memory memory;
    void* pointer = nullptr;
    std::size_t bytes;
    Storage* retained_next = nullptr;
    Storage(std::shared_ptr<CudaState> s,Reservation r,Memory m,std::size_t b)
        :state(std::move(s)),reservation(std::move(r)),memory(m),bytes(b) {}
};
void release_storage(Storage* storage) noexcept {
    if(!storage->pointer) {delete storage; return;}
    auto status = storage->state->settle();
    if(status == cudaSuccess)
        status = storage->memory == Memory::device ? cudaFree(storage->pointer) : cudaFreeHost(storage->pointer);
    if(status == cudaSuccess) {delete storage; return;}
    log(status,"release allocation; retaining storage/reservation/backend");
    static Storage* retained = nullptr;
    static std::mutex mutex;
    try { std::lock_guard lock(mutex); storage->retained_next = retained; retained = storage; }
    catch(...) { /* never free unproven live storage */ }
    ++allocation_quarantine;
}
void range(std::size_t size,std::size_t offset,std::size_t count) {
    if(offset > size || count > size-offset) throw std::out_of_range("CUDA buffer range exceeded");
}
}
struct CudaBuffer::Impl {
    std::unique_ptr<Storage,decltype(&release_storage)> storage;
    explicit Impl(decltype(storage) s):storage(std::move(s)) {}
};
CudaBuffer::CudaBuffer(std::shared_ptr<Impl> impl):impl_(std::move(impl)) {}
void* CudaBuffer::data() const noexcept { return impl_ ? impl_->storage->pointer : nullptr; }
std::size_t CudaBuffer::size() const noexcept { return impl_ ? impl_->storage->bytes : 0; }
Memory CudaBuffer::memory() const {
    if(!impl_) throw std::logic_error("empty CUDA buffer");
    return impl_->storage->memory;
}
void* CudaBuffer::device_address() const {
    if(!impl_) throw std::logic_error("empty CUDA buffer");
    const auto& storage = *impl_->storage;
    if(storage.memory == Memory::device || !storage.pointer) return storage.pointer;
    storage.state->bind();
    void* device = nullptr;
    check(cudaHostGetDevicePointer(&device,storage.pointer,0),"cudaHostGetDevicePointer");
    return device;
}
void CudaBuffer::upload(std::span<const std::byte> source,std::size_t offset,Queue queue) {
    range(size(),offset,source.size());
    if(source.empty()) return;
    if(memory() != Memory::device) throw std::invalid_argument("upload requires device buffer");
    if(!source.data()) throw std::invalid_argument("null upload source");
    auto& state = *impl_->storage->state;
    auto staging = create(impl_->storage->state,Memory::host,source.size());
    std::memcpy(staging.data(),source.data(),source.size());
    state.bind();
    const auto stream = state.stream(queue);
    check(cudaMemcpyAsync(static_cast<std::byte*>(data())+offset,staging.data(),source.size(),cudaMemcpyHostToDevice,stream),"cudaMemcpyAsync upload");
    check(cudaStreamSynchronize(stream),"upload synchronization");
}
void CudaBuffer::download(std::span<std::byte> destination,std::size_t offset,Queue queue) const {
    range(size(),offset,destination.size());
    if(destination.empty()) return;
    if(memory() != Memory::device) throw std::invalid_argument("download requires device buffer");
    if(!destination.data()) throw std::invalid_argument("null download destination");
    auto& state = *impl_->storage->state;
    auto staging = create(impl_->storage->state,Memory::host,destination.size());
    state.bind();
    const auto stream = state.stream(queue);
    check(cudaMemcpyAsync(staging.data(),static_cast<std::byte*>(data())+offset,destination.size(),cudaMemcpyDeviceToHost,stream),"cudaMemcpyAsync download");
    check(cudaStreamSynchronize(stream),"download synchronization");
    std::memcpy(destination.data(),staging.data(),destination.size());
}
struct CudaEvent::Impl {
    std::shared_ptr<CudaState> state;
    cudaEvent_t event = nullptr;
    bool recorded = false, timing;
    Impl(std::shared_ptr<CudaState> s,bool time):state(std::move(s)),timing(time) {
        state->bind();
        check(cudaEventCreateWithFlags(&event,time ? cudaEventDefault : cudaEventDisableTiming),"cudaEventCreateWithFlags");
    }
    ~Impl() {
        log(cudaSetDevice(state->device),"bind event destruction");
        // CUDA defers event resource release while its recorded work is pending.
        if(event) log(cudaEventDestroy(event),"cudaEventDestroy");
    }
    void require_recorded() const {
        if(!recorded) throw std::logic_error("CUDA event has not been recorded");
    }
};
CudaEvent::CudaEvent() noexcept = default;
CudaEvent::CudaEvent(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
CudaEvent::~CudaEvent() = default;
CudaEvent::CudaEvent(CudaEvent&&) noexcept = default;
CudaEvent& CudaEvent::operator=(CudaEvent&&) noexcept = default;
void CudaEvent::record(cudaStream_t stream) {
    if(!impl_) throw std::logic_error("empty CUDA event");
    impl_->state->require_stream(stream);
    impl_->state->bind();
    check(cudaEventRecord(impl_->event,stream),"cudaEventRecord");
    impl_->recorded = true;
}
void CudaEvent::order(cudaStream_t stream) const {
    if(!impl_) throw std::logic_error("empty CUDA event");
    impl_->require_recorded();
    impl_->state->require_stream(stream);
    impl_->state->bind();
    check(cudaStreamWaitEvent(stream,impl_->event,0),"cudaStreamWaitEvent");
}
bool CudaEvent::ready() const {
    if(!impl_) throw std::logic_error("empty CUDA event");
    impl_->require_recorded();
    impl_->state->bind();
    const auto status = cudaEventQuery(impl_->event);
    if(status == cudaErrorNotReady) return false;
    check(status,"cudaEventQuery");
    return true;
}
void CudaEvent::wait() {
    if(!impl_) throw std::logic_error("empty CUDA event");
    impl_->require_recorded();
    impl_->state->bind();
    check(cudaEventSynchronize(impl_->event),"cudaEventSynchronize");
}
CudaTimer::CudaTimer(CudaEvent start,CudaEvent stop,cudaStream_t stream)
    :start_(std::move(start)),stop_(std::move(stop)),stream_(stream) {}
void CudaTimer::start() {start_.record(stream_); stop_.impl_->recorded = false;}
void CudaTimer::record_stop() {
    if(!start_.impl_) throw std::logic_error("empty CUDA timer");
    start_.impl_->require_recorded(); stop_.record(stream_);
}
float CudaTimer::elapsed_ms() const {
    if(!start_.impl_ || !stop_.impl_) throw std::logic_error("empty CUDA timer");
    start_.impl_->require_recorded(); stop_.impl_->require_recorded();
    start_.impl_->state->bind();
    float value = 0;
    check(cudaEventElapsedTime(&value,start_.impl_->event,stop_.impl_->event),"cudaEventElapsedTime");
    return value;
}
float CudaTimer::stop_ms() {record_stop(); stop_.wait(); return elapsed_ms();}
struct CudaStageTimer::Impl {
    std::shared_ptr<CudaState> state;
    std::vector<cudaEvent_t> starts, stops;
    std::vector<DeviceTiming> timings;
    std::size_t count = 0;
    bool open = false;
    Impl(std::shared_ptr<CudaState> s,std::size_t capacity)
        :state(std::move(s)),starts(capacity,nullptr),stops(capacity,nullptr),timings(capacity,DeviceTiming{DeviceStage::other}) {}
    ~Impl() {
        log(cudaSetDevice(state->device),"bind stage timer destruction");
        for(auto event:starts) if(event) log(cudaEventDestroy(event),"cudaEventDestroy");
        for(auto event:stops) if(event) log(cudaEventDestroy(event),"cudaEventDestroy");
    }
    void record(cudaEvent_t event) {
        state->bind();
        auto capture = cudaStreamCaptureStatusNone;
        check(cudaStreamIsCapturing(state->compute,&capture),"cudaStreamIsCapturing");
        if(capture != cudaStreamCaptureStatusNone)
            throw std::logic_error("CUDA stage timer cannot record inside stream capture; time around graph launches");
        check(cudaEventRecord(event,state->compute),"cudaEventRecord");
    }
};
CudaStageTimer::CudaStageTimer() noexcept = default;
CudaStageTimer::CudaStageTimer(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
CudaStageTimer::~CudaStageTimer() = default;
CudaStageTimer::CudaStageTimer(CudaStageTimer&&) noexcept = default;
CudaStageTimer& CudaStageTimer::operator=(CudaStageTimer&&) noexcept = default;
void CudaStageTimer::begin(DeviceStage stage,std::uint32_t index) {
    if(!impl_) throw std::logic_error("empty CUDA stage timer");
    if(impl_->open) throw std::logic_error("CUDA stage already open");
    if(impl_->count == impl_->starts.size()) throw std::out_of_range("CUDA stage timer capacity exceeded");
    impl_->record(impl_->starts[impl_->count]);
    impl_->timings[impl_->count] = {stage,index,0};
    impl_->open = true;
}
void CudaStageTimer::end() {
    if(!impl_) throw std::logic_error("empty CUDA stage timer");
    if(!impl_->open) throw std::logic_error("CUDA stage is not open");
    impl_->record(impl_->stops[impl_->count]);
    ++impl_->count; impl_->open = false;
}
void CudaStageTimer::reset() {
    if(!impl_) throw std::logic_error("empty CUDA stage timer");
    impl_->count = 0; impl_->open = false;
}
std::span<const DeviceTiming> CudaStageTimer::collect() {
    if(!impl_) throw std::logic_error("empty CUDA stage timer");
    if(impl_->open) throw std::logic_error("CUDA stage is still open");
    impl_->state->bind();
    for(std::size_t i=0;i<impl_->count;++i) {
        const auto status = cudaEventQuery(impl_->stops[i]);
        if(status == cudaErrorNotReady) throw std::runtime_error("CUDA stage timings are not ready");
        check(status,"cudaEventQuery");
        float ms = 0;
        check(cudaEventElapsedTime(&ms,impl_->starts[i],impl_->stops[i]),"cudaEventElapsedTime");
        impl_->timings[i].duration_ns = std::uint64_t(double(ms)*1e6);
    }
    return {impl_->timings.data(),impl_->count};
}
std::size_t CudaStageTimer::capacity() const noexcept {return impl_ ? impl_->starts.size() : 0;}
CudaBackend::CudaBackend(int device,SyncSchedule schedule) {
    const auto requested = flags(schedule);
    int count = 0;
    check(cudaGetDeviceCount(&count),"cudaGetDeviceCount");
    if(device < 0 || device >= count) throw std::invalid_argument("invalid CUDA device");
    state_ = std::shared_ptr<CudaState>(new CudaState(device),release_state);
    state_->bind();
    check(cudaSetDeviceFlags(requested),"cudaSetDeviceFlags");
    check(cudaGetDeviceProperties(&state_->properties,device),"cudaGetDeviceProperties");
    check(cudaStreamCreateWithFlags(&state_->compute,cudaStreamNonBlocking),"create compute stream");
    check(cudaStreamCreateWithFlags(&state_->transfer,cudaStreamNonBlocking),"create transfer stream");
}
CudaBackend::~CudaBackend() = default;
void CudaBackend::initialize(Resources& resources) {
    std::lock_guard lock(state_->mutex);
    if(state_->resources) throw std::logic_error("CUDA backend already initialized for an engine");
    state_->resources = resources;
}
void CudaBackend::bind_thread() {state_->bind();}
void CudaBackend::drain() {check(state_->settle(),"CUDA backend drain");}
int CudaBackend::device() const noexcept {return state_->device;}
cudaDeviceProp CudaBackend::properties() const {return state_->properties;}
cudaStream_t CudaBackend::stream(Queue queue) const {return state_->stream(queue);}
SyncSchedule CudaBackend::sync_schedule() const {
    state_->bind();
    unsigned value = 0;
    check(cudaGetDeviceFlags(&value),"cudaGetDeviceFlags");
    switch(value & cudaDeviceScheduleMask) {
    case cudaDeviceScheduleSpin: return SyncSchedule::spin;
    case cudaDeviceScheduleBlockingSync: return SyncSchedule::blocking;
    case cudaDeviceScheduleYield: return SyncSchedule::yield;
    case cudaDeviceScheduleAuto: return SyncSchedule::automatic;
    }
    throw std::runtime_error("unknown applied CUDA synchronization schedule");
}
CudaBuffer CudaBackend::allocate(Memory memory,std::size_t bytes) {
    return CudaBuffer::create(state_,memory,bytes);
}
std::size_t CudaBackend::spare_device_bytes(std::size_t margin) const {
    Usage usage;
    {
        std::lock_guard lock(state_->mutex);
        if(!state_->resources) throw std::logic_error("CUDA backend has not been initialized");
        usage = state_->resources->usage(Memory::device);
    }
    state_->bind();
    std::size_t free = 0,total = 0;
    check(cudaMemGetInfo(&free,&total),"cudaMemGetInfo");
    const std::size_t available = std::min(free,usage.limit-usage.live);
    return available > margin ? available-margin : 0;
}
std::optional<CudaBuffer> CudaBackend::allocate_spare(std::size_t bytes,std::size_t margin) {
    if(spare_device_bytes(margin) < bytes) return std::nullopt;
    return allocate(Memory::device,bytes);
}
CudaBuffer CudaBuffer::create(std::shared_ptr<CudaState> state,Memory memory,std::size_t bytes) {
    if(memory != Memory::host && memory != Memory::device) throw std::invalid_argument("invalid CUDA memory domain");
    Reservation reservation;
    {
        std::lock_guard lock(state->mutex);
        if(!state->resources) throw std::logic_error("CUDA backend has not been initialized");
        reservation = state->resources->reserve(memory,bytes);
    }
    std::unique_ptr<Storage,decltype(&release_storage)> storage(new Storage(state,std::move(reservation),memory,bytes),release_storage);
    auto impl = std::make_shared<CudaBuffer::Impl>(std::move(storage));
    if(bytes) {
        state->bind();
        auto& pointer = impl->storage->pointer;
        check(memory == Memory::device ? cudaMalloc(&pointer,bytes) : cudaMallocHost(&pointer,bytes),"CUDA allocation");
    }
    return CudaBuffer(std::move(impl));
}
CudaEvent CudaBackend::event() {return CudaEvent(std::make_unique<CudaEvent::Impl>(state_,false));}
CudaTimer CudaBackend::timer(Queue queue) {
    return CudaTimer(CudaEvent(std::make_unique<CudaEvent::Impl>(state_,true)),
                     CudaEvent(std::make_unique<CudaEvent::Impl>(state_,true)),stream(queue));
}
void CudaBackend::copy_async(const CudaBuffer& dst,std::size_t dst_offset,const CudaBuffer& src,std::size_t src_offset,std::size_t bytes,Queue queue) {
    range(dst.size(),dst_offset,bytes); range(src.size(),src_offset,bytes);
    if(!dst.impl_ || !src.impl_ || dst.impl_->storage->state != state_ || src.impl_->storage->state != state_)
        throw std::invalid_argument("foreign/empty CUDA buffer");
    const auto s = src.memory(), d = dst.memory();
    if(s == Memory::host && d == Memory::host) throw std::invalid_argument("host-to-host async copy is unsupported");
    const auto stream = state_->stream(queue);
    if(!bytes) return;
    // Overlap within an aliased allocation is unsupported by CUDA memcpy.
    if(dst.impl_ == src.impl_ && dst_offset < src_offset+bytes && src_offset < dst_offset+bytes)
        throw std::invalid_argument("overlapping CUDA copy");
    state_->bind();
    const auto kind = s == Memory::host ? cudaMemcpyHostToDevice : d == Memory::host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
    check(cudaMemcpyAsync(static_cast<std::byte*>(dst.data())+dst_offset,static_cast<std::byte*>(src.data())+src_offset,bytes,kind,stream),"cudaMemcpyAsync");
}
void CudaBackend::fill_async(const CudaBuffer& buffer,int value,Queue queue) {
    if(!buffer.impl_ || buffer.impl_->storage->state != state_) throw std::invalid_argument("foreign/empty CUDA buffer");
    if(buffer.memory() != Memory::device) throw std::invalid_argument("fill requires device buffer");
    const auto stream = state_->stream(queue);
    if(!buffer.size()) return;
    state_->bind();
    check(cudaMemsetAsync(buffer.data(),value,buffer.size(),stream),"cudaMemsetAsync");
}
void CudaBackend::readback_async(const CudaBuffer& host_dst,std::size_t dst_offset,const CudaBuffer& device_src,std::size_t src_offset,std::size_t bytes,CudaEvent& completion) {
    if(!host_dst.impl_ || !device_src.impl_ || host_dst.impl_->storage->state != state_ || device_src.impl_->storage->state != state_)
        throw std::invalid_argument("foreign/empty CUDA buffer");
    if(!completion.impl_ || completion.impl_->state != state_) throw std::invalid_argument("foreign/empty CUDA event");
    if(host_dst.memory() != Memory::host) throw std::invalid_argument("readback requires pinned host destination");
    if(device_src.memory() != Memory::device) throw std::invalid_argument("readback requires device source");
    range(host_dst.size(),dst_offset,bytes); range(device_src.size(),src_offset,bytes);
    state_->bind();
    if(bytes) check(cudaMemcpyAsync(static_cast<std::byte*>(host_dst.data())+dst_offset,static_cast<std::byte*>(device_src.data())+src_offset,
                                    bytes,cudaMemcpyDeviceToHost,state_->compute),"cudaMemcpyAsync readback");
    completion.record(state_->compute);
}
CudaStageTimer CudaBackend::stage_timer(std::size_t capacity) {
    auto impl = std::make_unique<CudaStageTimer::Impl>(state_,capacity);
    state_->bind();
    for(std::size_t i=0;i<capacity;++i) {
        check(cudaEventCreateWithFlags(&impl->starts[i],cudaEventDefault),"cudaEventCreateWithFlags");
        check(cudaEventCreateWithFlags(&impl->stops[i],cudaEventDefault),"cudaEventCreateWithFlags");
    }
    return CudaStageTimer(std::move(impl));
}
std::size_t quarantined_cuda_allocations() noexcept {return allocation_quarantine.load();}
std::size_t quarantined_cuda_backends() noexcept {return backend_quarantine.load();}
}
