// Core CUDA hooks for shared allocations and mapped sources: device leases charged once and released after both
// backend streams settle, and host registration reporting the actual device alias. Requires a CUDA device.
#include <crit/cuda_memory.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace crit::core;
namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
void check(cudaError_t status) { if(status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
struct Fixture {
    Resources resources{1 << 20, 1 << 20};
    std::shared_ptr<CudaBackend> backend = std::make_shared<CudaBackend>();
    Fixture() { backend->initialize(resources); }
};
void leases() {
    Fixture f;
    BufferService service(cuda_lease_backend(f.backend, Memory::device), f.resources, Memory::device);
    auto lease = service.allocate(64);
    auto alias = lease.share();
    require(f.resources.usage(Memory::device).live == 64, "one charge per allocation");
    check(cudaMemsetAsync(lease.data(), 7, 64, f.backend->stream()));
    lease.close();
    std::array<unsigned char, 64> bytes{};
    check(cudaMemcpyAsync(bytes.data(), alias.data(), 64, cudaMemcpyDeviceToHost, f.backend->stream()));
    f.backend->drain();
    require(bytes[0] == 7 && bytes[63] == 7, "a shared lease lost the allocation");
    alias.close();
    require(f.resources.usage(Memory::device).live == 0 && !service.retained(), "final lease release");
    BufferService pinned(cuda_lease_backend(f.backend, Memory::host), f.resources, Memory::host);
    auto host = pinned.allocate(16);
    std::memset(host.data(), 1, 16);
    host.close();
    require(f.resources.usage(Memory::host).live == 0, "pinned lease release");
}
void mapping() {
    Fixture f;
    std::vector<unsigned char> host(4096, 0);
    {
        auto span = cuda_map_host(f.backend, host.data(), host.size());
        const MappedSource source(1, {span});
        const auto resolved = source.resolve(span.host_address + 128, 256);
        require(resolved.device_address == span.device_address + 128 && resolved.owners.size() == 1, "mapped alias");
        check(cudaMemsetAsync(reinterpret_cast<void*>(resolved.device_address), 9, 256, f.backend->stream()));
        f.backend->drain();
    }
    require(host[127] == 0 && host[128] == 9 && host[383] == 9 && host[384] == 0, "device writes through the alias");
    require(quarantined_cuda_mappings() == 0, "settled registration was retained");
}
}
int main(int argc, char** argv) {
    try {
        const std::string test = argc > 1 ? argv[1] : "leases";
        if(test == "leases") leases();
        else if(test == "mapping") mapping();
        else throw std::runtime_error("unknown test");
        std::cout << "crit core CUDA memory passed: " << test << '\n';
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
