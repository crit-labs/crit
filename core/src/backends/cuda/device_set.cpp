#include "crit/cuda_devices.h"
#include "device_scope.h"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace crit::core {
namespace {
void check(cudaError_t status, const char* operation) {
    if(status != cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorName(status)+": "+cudaGetErrorString(status));
}
void log(cudaError_t status, const char* operation) noexcept {
    if(status != cudaSuccess) std::fprintf(stderr,"crit CUDA device set cleanup %s: %s\n",operation,cudaGetErrorName(status));
}
std::size_t queue_index(Queue queue) {
    if(queue == Queue::compute) return 0;
    if(queue == Queue::transfer) return 1;
    throw std::invalid_argument("invalid CUDA queue");
}
}
struct CudaDeviceSet::Impl {
    std::vector<std::unique_ptr<CudaBackend>> members;
    std::vector<std::vector<bool>> peers;
    // One event per member and queue, recorded where another member's stream waits; reused, since a wait takes the
    // event's state when it is enqueued.
    std::vector<std::array<cudaEvent_t,2>> events;
    ~Impl() {
        for(std::size_t i=0;i<events.size();++i) {
            const detail::DeviceScope scope(members[i]->device(),std::nothrow);
            log(scope.status(),"bind event destruction");
            for(auto event:events[i]) if(event) log(cudaEventDestroy(event),"cudaEventDestroy");
        }
    }
    CudaBackend& at(std::size_t index) const {
        if(index >= members.size()) throw std::out_of_range("no such device set member");
        return *members[index];
    }
};

CudaDeviceSet::CudaDeviceSet(std::vector<int> devices,SyncSchedule schedule):impl_(std::make_unique<Impl>()) {
    if(devices.empty()) throw std::invalid_argument("a device set needs at least one device");
    for(int device:devices) impl_->members.push_back(std::make_unique<CudaBackend>(device,schedule));
    const auto n = devices.size();
    impl_->peers.assign(n,std::vector<bool>(n,false));
    detail::DeviceScope scope(devices.front());
    for(std::size_t i=0;i<n;++i) {
        for(std::size_t j=0;j<n;++j) {
            if(devices[i] == devices[j]) { impl_->peers[i][j] = true; continue; }
            int can = 0;
            check(cudaDeviceCanAccessPeer(&can,devices[i],devices[j]),"cudaDeviceCanAccessPeer");
            if(!can) continue;
            scope.bind(devices[i]);
            auto status = cudaDeviceEnablePeerAccess(devices[j],0);
            if(status == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();   // enabled by an earlier set in this process: clear the sticky error
                status = cudaSuccess;
            }
            check(status,"cudaDeviceEnablePeerAccess");
            impl_->peers[i][j] = true;
        }
    }
    impl_->events.assign(n,{nullptr,nullptr});
    for(std::size_t i=0;i<n;++i) {
        scope.bind(devices[i]);
        for(auto& event:impl_->events[i])
            check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"cudaEventCreateWithFlags");
    }
}
CudaDeviceSet::~CudaDeviceSet() = default;
std::size_t CudaDeviceSet::size() const noexcept {return impl_->members.size();}
CudaBackend& CudaDeviceSet::member(std::size_t index) {return impl_->at(index);}
int CudaDeviceSet::device(std::size_t index) const {return impl_->at(index).device();}
bool CudaDeviceSet::peer(std::size_t reader,std::size_t owner) const {
    impl_->at(reader); impl_->at(owner);
    return impl_->peers[reader][owner];
}
void CudaDeviceSet::order(std::size_t waiter,Queue waiter_queue,std::size_t source,Queue source_queue) {
    auto& from = impl_->at(source);
    auto& to = impl_->at(waiter);
    if(waiter == source && waiter_queue == source_queue) return;
    const auto event = impl_->events[source][queue_index(source_queue)];
    detail::DeviceScope scope(from.device());
    check(cudaEventRecord(event,from.stream(source_queue)),"cudaEventRecord");
    scope.bind(to.device());
    check(cudaStreamWaitEvent(to.stream(waiter_queue),event,0),"cudaStreamWaitEvent");
}
void CudaDeviceSet::copy_async(std::size_t dst_member,const CudaBuffer& dst,std::size_t dst_offset,
                               std::size_t src_member,const CudaBuffer& src,std::size_t src_offset,std::size_t bytes,
                               Queue queue) {
    auto& to = impl_->at(dst_member);
    auto& from = impl_->at(src_member);
    if(!to.owns(dst) || !from.owns(src)) throw std::invalid_argument("each buffer must belong to its member");
    if(dst.memory() != Memory::device || src.memory() != Memory::device)
        throw std::invalid_argument("copies between members are of device buffers");
    if(dst_offset > dst.size() || bytes > dst.size()-dst_offset || src_offset > src.size() || bytes > src.size()-src_offset)
        throw std::out_of_range("CUDA buffer range exceeded");
    if(!bytes) return;
    order(dst_member,queue,src_member,Queue::compute);
    const detail::DeviceScope scope(to.device());
    check(cudaMemcpyPeerAsync(static_cast<std::byte*>(dst.data())+dst_offset,to.device(),
                              static_cast<const std::byte*>(src.data())+src_offset,from.device(),bytes,to.stream(queue)),
          "cudaMemcpyPeerAsync");
    // The source member's transfer stream waits for the copy, so releasing `src` (which settles its own member's
    // streams) never frees memory the copy still reads.
    order(src_member,Queue::transfer,dst_member,queue);
}
void CudaDeviceSet::barrier(Queue queue) {
    const auto q = queue_index(queue);
    const auto n = size();
    detail::DeviceScope scope(impl_->members.front()->device());
    for(std::size_t i=0;i<n;++i) {
        scope.bind(impl_->members[i]->device());
        check(cudaEventRecord(impl_->events[i][q],impl_->members[i]->stream(queue)),"cudaEventRecord");
    }
    for(std::size_t j=0;j<n;++j) {
        scope.bind(impl_->members[j]->device());
        for(std::size_t i=0;i<n;++i)
            if(i != j) check(cudaStreamWaitEvent(impl_->members[j]->stream(queue),impl_->events[i][q],0),"cudaStreamWaitEvent");
    }
}
CudaCategoryTimer CudaDeviceSet::category_timer(std::size_t index,std::size_t capacity) const {
    return CudaCategoryTimer(capacity,device(index));
}
}
