#pragma once
#include "crit/core.h"
#include <algorithm>
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
// Retention bookkeeping is reserved before allocation, so final close needs no Core heap allocation.
// Destroying an unsettled service quarantines its backend as well as the allocations and charges.
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

// Skips host-to-device uploads of small per-step metadata (slot indices, lengths, page tables between page boundaries)
// whose bytes the destination already holds: one host shadow per destination, the bytes last written through it.
// upload(dst, src, copy) calls copy(dst, src) and remembers src unless dst received the same bytes last time. Only for
// destinations nothing else writes (no kernel, no other copy): invalidate() after any other write, and forget() when
// a destination is freed. Copies a graph records always run when it replays, so it serves eager launches between
// graphs. Not thread-safe; allocates when it first sees a destination or a larger upload.
class UploadShadow {
public:
    template<class Copy> bool upload(void* dst, std::span<const std::byte> src, Copy&& copy) {
        auto& shadow = entry(dst);
        if(shadow.valid && shadow.bytes.size() == src.size() &&
           std::equal(src.begin(), src.end(), shadow.bytes.begin())) { ++skipped_; return false; }
        copy(dst, src);
        shadow.bytes.assign(src.begin(), src.end());
        shadow.valid = true;
        ++copied_;
        return true;
    }
    void invalidate(const void* dst) noexcept {
        for(auto& e:entries_) if(e.dst == dst) e.valid = false;
    }
    void forget(const void* dst) noexcept {
        std::erase_if(entries_, [&](const Entry& e) { return e.dst == dst; });
    }
    std::uint64_t copied() const noexcept { return copied_; }
    std::uint64_t skipped() const noexcept { return skipped_; }
private:
    struct Entry { const void* dst = nullptr; bool valid = false; std::vector<std::byte> bytes; };
    Entry& entry(const void* dst) {
        for(auto& e:entries_) if(e.dst == dst) return e;
        entries_.push_back({dst, false, {}});
        return entries_.back();
    }
    std::vector<Entry> entries_;
    std::uint64_t copied_ = 0, skipped_ = 0;
};
}
