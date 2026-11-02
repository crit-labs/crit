// The host store of a device expert cache (expert_store.py writes it) pinned for the device: on CUDA devices that
// register read-only memory, Core's read-only file mapping registered as mapped pinned memory in one call (registering
// a mapped file takes seconds where allocating and filling tens of GB of pinned memory took minutes on one machine);
// elsewhere (HIP, or cudaDevAttrHostRegisterReadOnlySupported 0) the file read by several threads into a Core
// HostBlock, then registered. Directory::store is PinnedStore::device.
#pragma once
#include <crit/file.h>
#include <crit/host_block.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__HIPCC__) && !defined(CRIT_CORE_HIP)
#define CRIT_CORE_HIP 1
#endif
#if defined(CRIT_CORE_HIP)
#include "crit/gpu_runtime.h"
#else
#include <cuda_runtime.h>
#endif

namespace crit::techniques::device_cache {

struct PinnedStore {
    std::span<const std::byte> host;        // the store's bytes, keys * bundle_bytes
    const std::uint8_t* device = nullptr;   // the device address of the same bytes: Directory::store
    bool mapped_file = false;               // false: read into a pinned HostBlock
    std::shared_ptr<void> owner;            // keeps the mapping or block and the registration alive
};

namespace store_detail {
inline void check(cudaError_t e, const char* what) {
    if(e != cudaSuccess) throw std::runtime_error(std::string("expert store: ") + what + ": " + cudaGetErrorString(e));
}
struct Registration {
    void* host = nullptr;
    std::shared_ptr<const void> mapping;
    std::unique_ptr<core::HostBlock> block;
    ~Registration() {
        if(host) cudaHostUnregister(host);
    }
};
inline bool read_only_registration() {
#if defined(CRIT_CORE_HIP)
    return false;
#else
    int device = 0, supported = 0;
    if(cudaGetDevice(&device) != cudaSuccess) return false;
    return cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterReadOnlySupported, device) == cudaSuccess && supported;
#endif
}
// The file into `dst` with `threads` readers, 64 MiB a piece.
inline void read_parallel(core::ReadOnlyFile& file, std::byte* dst, std::size_t bytes, int threads) {
    constexpr std::size_t piece = std::size_t(64) << 20;
    const std::size_t pieces = (bytes + piece - 1) / piece;
    const int workers = std::max(1, std::min<int>(threads, int(pieces)));
    file.set_readers(std::size_t(workers));
    std::exception_ptr failure;
    std::mutex lock;
    std::vector<std::thread> pool;
    for(int w = 0; w < workers; ++w)
        pool.emplace_back([&, w] {
            try {
                for(std::size_t p = std::size_t(w); p < pieces; p += std::size_t(workers)) {
                    const std::size_t offset = p * piece, n = std::min(piece, bytes - offset);
                    file.read_exact(offset, std::span<std::byte>(dst + offset, n), std::size_t(w));
                }
            } catch(...) {
                const std::lock_guard<std::mutex> guard(lock);
                if(!failure) failure = std::current_exception();
            }
        });
    for(auto& t : pool) t.join();
    if(failure) std::rethrow_exception(failure);
}
} // namespace store_detail

// Pins `path` (expert_store.py's output) for the current device. bytes: the store's size from its index (groups *
// experts * bundle_bytes); a file of another size is refused. threads: readers for the HostBlock path.
inline PinnedStore pin_store(const std::filesystem::path& path, std::size_t bytes, int threads = 8) {
    auto file = core::ReadOnlyFile::open(path);
    if(file.identity().size != bytes)
        throw std::runtime_error("expert store: " + core::display_path(path) + " is not " + std::to_string(bytes) + " bytes");
    auto registration = std::make_shared<store_detail::Registration>();
    PinnedStore out;
    if(store_detail::read_only_registration()) {
        const auto view = file.map();
        registration->mapping = file.mapping_owner();
        void* host = const_cast<std::byte*>(view.data());
#if !defined(CRIT_CORE_HIP)
        store_detail::check(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterReadOnly),
                            "cudaHostRegister (read-only file mapping)");
#endif
        registration->host = host;
        out.host = view;
        out.mapped_file = true;
    } else {
        registration->block = std::make_unique<core::HostBlock>(bytes);
        std::byte* data = registration->block->data();
        store_detail::read_parallel(file, data, bytes, threads);
        store_detail::check(cudaHostRegister(data, bytes, cudaHostRegisterMapped), "cudaHostRegister (host block)");
        registration->host = data;
        out.host = std::span<const std::byte>(data, bytes);
    }
    void* mapped = nullptr;
    store_detail::check(cudaHostGetDevicePointer(&mapped, registration->host, 0), "cudaHostGetDevicePointer");
    out.device = static_cast<const std::uint8_t*>(mapped);
    out.owner = std::move(registration);
    return out;
}

} // namespace crit::techniques::device_cache
