#pragma once
#include "crit/cuda.h"
#include "crit/device_memory.h"
#include "crit/host_block.h"
#include <memory>

// Optional CUDA hooks for the shared allocation and mapped-source protocol in crit/device_memory.h.
namespace crit::core {
// BufferService hooks on one backend: cudaMalloc (device) or cudaMallocHost (pinned host) on its device,
// settlement of both backend streams, and cudaFree/cudaFreeHost. Work on other streams needs its own settlement
// before the last lease closes. Charge through the engine's Resources; never charge one allocation twice.
std::shared_ptr<LeaseBackend> cuda_lease_backend(std::shared_ptr<CudaBackend>, Memory);
// Registers caller-owned host bytes as mapped pinned memory and reports the actual device alias. The span's
// owner settles both backend streams and unregisters at its last release; a failed settle or unregister retains
// the registration (quarantined_cuda_mappings()). The caller keeps the host bytes alive past that release.
MappedSpan cuda_map_host(std::shared_ptr<CudaBackend>, void* host, std::size_t bytes);
std::size_t quarantined_cuda_mappings() noexcept;
// Registers a filled HostBlock as mapped pinned memory (cuda_map_host), attaches the registration to the block so it
// ends before the pages are freed, then touches every page with touch_threads threads (0: no touch). Returns the span
// with the device alias. A block that is already attached is not registered again: pin a block once.
MappedSpan cuda_pin_host_block(std::shared_ptr<CudaBackend>, HostBlock&, int touch_threads);
}
