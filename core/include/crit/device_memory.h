#pragma once
#include "crit/core.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

// Host protocol of shared device allocations and mapped sources: leases over one physical allocation charged once
// through Resources, settled release with failure retention, and exact resolution of host extents through
// backend-reported device aliases. It makes no device call; crit/cuda_memory.h supplies the CUDA hooks.
namespace crit::core {

// Backend hooks. allocate owns its own partial failure before returning. settle proves that every reader and
// writer of the service's allocations has completed, and throws when it cannot. release frees one physical
// allocation and throws when the outcome is unknown.
class LeaseBackend {
public:
    virtual ~LeaseBackend() = default;
    virtual void* allocate(std::size_t bytes) = 0;
    virtual void settle() = 0;
    virtual void release(void* physical, std::size_t bytes) = 0;
};

class BufferLease;
// Each allocation reserves its bytes once in Resources. Sharing adds a lease without allocating or charging again.
// The last lease settles the backend, releases the allocation and drops the charge. A failed settle retains the
// allocation and charge for retry_retained(); a failed release is never retried, since double release is unsafe,
// and keeps its charge. Operations are serialized internally; backend hooks run under that lock.
class BufferService {
public:
    BufferService(std::shared_ptr<LeaseBackend>, Resources, Memory);
    BufferLease allocate(std::size_t bytes);
    // Retries the final release of allocations whose settlement failed. Throws on the first failure, including an
    // allocation whose release is uncertain and needs backend recovery.
    void retry_retained();
    std::size_t retained() const;  // Allocations awaiting retry or recovery.
    std::size_t uncertain() const; // Allocations whose physical release failed.
    struct State;
    struct Shared;
private:
    std::shared_ptr<Shared> shared_;
};
// One lease on a shared allocation. close() (and destruction) gives the lease up; the final close releases. A
// destructor swallows a failed release; the service retains that allocation.
class BufferLease {
public:
    BufferLease() noexcept = default;
    ~BufferLease();
    BufferLease(BufferLease&&) noexcept;
    BufferLease& operator=(BufferLease&&) noexcept;
    BufferLease(const BufferLease&) = delete;
    BufferLease& operator=(const BufferLease&) = delete;
    void* data() const;
    std::size_t size() const;
    BufferLease share() const;
    void close();
    bool closed() const noexcept { return !state_; }
private:
    std::shared_ptr<BufferService::Shared> service_;
    std::shared_ptr<BufferService::State> state_;
    BufferLease(std::shared_ptr<BufferService::Shared>, std::shared_ptr<BufferService::State>) noexcept;
    friend class BufferService;
};
// Services destroyed while holding allocations whose release failed; those allocations and charges stay retained.
std::size_t quarantined_buffer_services() noexcept;

// One backend-reported host range and its actual device alias. The owner keeps the registration alive.
struct MappedSpan {
    std::uintptr_t host_address = 0, device_address = 0;
    std::size_t bytes = 0;
    std::shared_ptr<const void> owner;
};
struct ResolvedSource {
    std::uintptr_t device_address = 0;
    std::size_t bytes = 0;
    std::uint64_t generation = 0;
    std::vector<std::shared_ptr<const void>> owners; // Every contributing registration, in host order.
};
// An immutable address map of one generation; replacing bindings needs a new generation. Host-contiguous spans
// may have discontinuous device aliases: one device pointer is returned only when every covered alias continues
// the previous one. Positive addresses do not prove device accessibility.
class MappedSource {
public:
    // Throws std::invalid_argument for generation 0, no spans, zero addresses/extents, missing owners, address
    // overflow or overlapping host ranges.
    MappedSource(std::uint64_t generation, std::vector<MappedSpan> spans);
    // Throws std::invalid_argument for an invalid extent, a coverage gap or a device alias discontinuity.
    ResolvedSource resolve(std::uintptr_t host_address, std::size_t bytes) const;
    std::uint64_t generation() const noexcept { return generation_; }
    std::span<const MappedSpan> spans() const noexcept { return spans_; }
private:
    std::uint64_t generation_;
    std::vector<MappedSpan> spans_;
};
}
