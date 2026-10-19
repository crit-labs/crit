#include "crit/cuda_memory.h"
#include <atomic>
#include <stdexcept>
#include <string>
#include <utility>

namespace crit::core {
namespace {
std::atomic<std::size_t> mapping_quarantine{0};
void check(cudaError_t status, const char* operation) {
    if(status != cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorName(status)+": "+cudaGetErrorString(status));
}
class CudaLeases final : public LeaseBackend {
public:
    CudaLeases(std::shared_ptr<CudaBackend> backend, Memory memory):backend_(std::move(backend)), memory_(memory) {}
    void* allocate(std::size_t bytes) override {
        if(!bytes) return nullptr;
        backend_->bind_thread();
        void* pointer = nullptr;
        check(memory_ == Memory::device ? cudaMalloc(&pointer, bytes) : cudaMallocHost(&pointer, bytes), "CUDA lease allocation");
        return pointer;
    }
    void settle() override { backend_->drain(); }
    void release(void* physical, std::size_t) override {
        if(!physical) return;
        backend_->bind_thread();
        check(memory_ == Memory::device ? cudaFree(physical) : cudaFreeHost(physical), "CUDA lease release");
    }
private:
    std::shared_ptr<CudaBackend> backend_;
    Memory memory_;
};
struct Registration {
    std::shared_ptr<CudaBackend> backend;
    void* host;
    Registration* retained_next = nullptr;
};
void unregister(Registration* registration) noexcept {
    try {
        registration->backend->drain();
        registration->backend->bind_thread();
        if(cudaHostUnregister(registration->host) == cudaSuccess) { delete registration; return; }
    } catch(...) { /* unproven settlement: retain below */ }
    // Intrusive retention allocates nothing and keeps the registration and backend.
    static std::atomic<Registration*> head{nullptr};
    auto previous = head.load(std::memory_order_relaxed);
    do { registration->retained_next = previous; }
    while(!head.compare_exchange_weak(previous, registration, std::memory_order_release, std::memory_order_relaxed));
    mapping_quarantine.fetch_add(1, std::memory_order_relaxed);
}
}
std::shared_ptr<LeaseBackend> cuda_lease_backend(std::shared_ptr<CudaBackend> backend, Memory memory) {
    if(!backend) throw std::invalid_argument("CUDA lease hooks require a backend");
    if(memory != Memory::host && memory != Memory::device) throw std::invalid_argument("invalid CUDA memory domain");
    return std::make_shared<CudaLeases>(std::move(backend), memory);
}
MappedSpan cuda_map_host(std::shared_ptr<CudaBackend> backend, void* host, std::size_t bytes) {
    if(!backend || !host || !bytes) throw std::invalid_argument("CUDA host mapping requires a backend and host bytes");
    auto registration = std::make_unique<Registration>(Registration{backend, host});
    backend->bind_thread();
    check(cudaHostRegister(host, bytes, cudaHostRegisterMapped), "cudaHostRegister");
    // From here the owner unregisters, including when the alias query below fails.
    const std::shared_ptr<Registration> owner(registration.release(), unregister);
    void* device = nullptr;
    check(cudaHostGetDevicePointer(&device, host, 0), "cudaHostGetDevicePointer");
    return {reinterpret_cast<std::uintptr_t>(host), reinterpret_cast<std::uintptr_t>(device), bytes, owner};
}
std::size_t quarantined_cuda_mappings() noexcept { return mapping_quarantine.load(std::memory_order_relaxed); }
MappedSpan cuda_pin_host_block(std::shared_ptr<CudaBackend> backend, HostBlock& block, int touch_threads) {
    if(block.attached()) throw std::invalid_argument("cuda_pin_host_block: block already pinned");
    MappedSpan span = cuda_map_host(std::move(backend), block.data(), block.size());
    block.attach(span.owner);
    if(touch_threads > 0) block.touch(touch_threads);
    return span;
}
}
