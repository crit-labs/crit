#include "crit/device_memory.h"
#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace crit::core {
struct BufferService::State {
    void* physical = nullptr;
    std::size_t bytes = 0;
    Reservation reservation;
    std::size_t leases = 1;
    bool freed = false, uncertain = false;
};
namespace {
// Retained allocations of a destroyed service, reachable from a static list for the life of the process.
struct Quarantine {
    std::shared_ptr<LeaseBackend> backend;
    std::vector<std::shared_ptr<BufferService::State>> states;
    Quarantine* next = nullptr;
};
std::atomic<Quarantine*> quarantine_head{nullptr};
std::atomic<std::size_t> quarantine_count{0};
}
struct BufferService::Shared {
    std::shared_ptr<LeaseBackend> backend;
    Resources resources;
    Memory memory;
    mutable std::mutex mutex;
    std::unique_ptr<Quarantine> quarantine = std::make_unique<Quarantine>();
    std::vector<std::shared_ptr<State>>* retained = &quarantine->states;
    std::size_t allocations = 0;
    Shared(std::shared_ptr<LeaseBackend> b, Resources r, Memory m):backend(std::move(b)), resources(std::move(r)), memory(m) {}
    ~Shared() {
        // Never release unproven allocations or their charges.
        if(retained->empty()) return;
        auto* kept = quarantine.release();
        kept->backend = std::move(backend);
        auto previous = quarantine_head.load(std::memory_order_relaxed);
        do { kept->next = previous; }
        while(!quarantine_head.compare_exchange_weak(previous, kept, std::memory_order_release, std::memory_order_relaxed));
        quarantine_count.fetch_add(1, std::memory_order_relaxed);
    }
    void keep(const std::shared_ptr<State>& state) noexcept {
        // Capacity was reserved before the backend call, so this cannot allocate.
        if(std::find(retained->begin(), retained->end(), state) == retained->end()) retained->push_back(state);
    }
    void retire(const std::shared_ptr<State>& state) {
        if(state->leases || state->freed) return;
        if(state->uncertain) throw std::logic_error("failed physical release requires recovery; cannot retry blindly");
        // allocate() reserved a retention slot for every outstanding physical allocation.
        // Final close must not lose its last owner to a bookkeeping allocation failure.
        try { backend->settle(); }
        catch(...) { keep(state); throw; }
        try { backend->release(state->physical, state->bytes); }
        catch(...) { state->uncertain = true; keep(state); throw; }
        state->freed = true;
        state->reservation = {};
        std::erase(*retained, state);
        --allocations;
    }
};
BufferService::BufferService(std::shared_ptr<LeaseBackend> backend, Resources resources, Memory memory) {
    if(!backend) throw std::invalid_argument("buffer service requires backend hooks");
    if(memory != Memory::host && memory != Memory::device) throw std::invalid_argument("invalid memory domain");
    shared_ = std::make_shared<Shared>(std::move(backend), std::move(resources), memory);
}
BufferLease BufferService::allocate(std::size_t bytes) {
    std::lock_guard lock(shared_->mutex);
    auto state = std::make_shared<State>();
    state->bytes = bytes;
    state->reservation = shared_->resources.reserve(shared_->memory, bytes);
    if(shared_->allocations == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("buffer allocation count exhausted");
    shared_->retained->reserve(shared_->allocations + 1);
    state->physical = shared_->backend->allocate(bytes); // A failure drops the reservation with the state.
    ++shared_->allocations;
    return BufferLease(shared_, std::move(state));
}
void BufferService::retry_retained() {
    std::lock_guard lock(shared_->mutex);
    const auto pending = *shared_->retained;
    for(const auto& state : pending) shared_->retire(state);
}
std::size_t BufferService::retained() const {
    std::lock_guard lock(shared_->mutex);
    return shared_->retained->size();
}
std::size_t BufferService::uncertain() const {
    std::lock_guard lock(shared_->mutex);
    return static_cast<std::size_t>(std::count_if(shared_->retained->begin(), shared_->retained->end(),
                                                  [](const auto& state) { return state->uncertain; }));
}
BufferLease::BufferLease(std::shared_ptr<BufferService::Shared> service, std::shared_ptr<BufferService::State> state) noexcept
    :service_(std::move(service)), state_(std::move(state)) {}
BufferLease::~BufferLease() {
    try { close(); } catch(...) { /* the service retains the allocation and its charge */ }
}
BufferLease::BufferLease(BufferLease&& other) noexcept
    :service_(std::move(other.service_)), state_(std::move(other.state_)) {}
BufferLease& BufferLease::operator=(BufferLease&& other) noexcept {
    if(this != &other) {
        try { close(); } catch(...) { /* retained by the service */ }
        service_ = std::move(other.service_);
        state_ = std::move(other.state_);
    }
    return *this;
}
void* BufferLease::data() const {
    if(!state_) throw std::logic_error("closed buffer lease");
    return state_->physical;
}
std::size_t BufferLease::size() const {
    if(!state_) throw std::logic_error("closed buffer lease");
    return state_->bytes;
}
BufferLease BufferLease::share() const {
    if(!state_) throw std::logic_error("closed buffer lease");
    std::lock_guard lock(service_->mutex);
    ++state_->leases;
    return BufferLease(service_, state_);
}
void BufferLease::close() {
    if(!state_) return;
    const auto service = std::move(service_);
    const auto state = std::move(state_);
    std::lock_guard lock(service->mutex);
    --state->leases;
    service->retire(state);
}

std::size_t quarantined_buffer_services() noexcept { return quarantine_count.load(std::memory_order_relaxed); }

namespace {
constexpr auto address_max = std::numeric_limits<std::uintptr_t>::max();
}
MappedSource::MappedSource(std::uint64_t generation, std::vector<MappedSpan> spans)
    :generation_(generation), spans_(std::move(spans)) {
    if(!generation_ || spans_.empty())
        throw std::invalid_argument("positive immutable source generation and mapped spans required");
    for(const auto& span : spans_) {
        if(!span.host_address || !span.device_address || !span.bytes || !span.owner)
            throw std::invalid_argument("mapped spans require explicit positive addresses, extents and owners");
        if(span.bytes > address_max - span.host_address || span.bytes > address_max - span.device_address)
            throw std::invalid_argument("mapped span exceeds the address space");
    }
    std::sort(spans_.begin(), spans_.end(),
              [](const MappedSpan& a, const MappedSpan& b) { return a.host_address < b.host_address; });
    for(std::size_t i = 1; i < spans_.size(); ++i)
        if(spans_[i - 1].host_address + spans_[i - 1].bytes > spans_[i].host_address)
            throw std::invalid_argument("overlapping host mappings require a separate alias contract");
}
ResolvedSource MappedSource::resolve(std::uintptr_t host_address, std::size_t bytes) const {
    if(!host_address || !bytes) throw std::invalid_argument("positive source address and extent required");
    if(bytes > address_max - host_address) throw std::invalid_argument("source extent exceeds the address space");
    const auto end = host_address + bytes;
    auto cursor = host_address;
    ResolvedSource result{0, bytes, generation_, {}};
    std::uintptr_t expected = 0;
    auto span = std::partition_point(spans_.begin(), spans_.end(),
                                     [&](const MappedSpan& s) { return s.host_address + s.bytes <= cursor; });
    for(; span != spans_.end() && span->host_address <= cursor; ++span) {
        const auto device = span->device_address + (cursor - span->host_address);
        if(cursor == host_address) result.device_address = device;
        else if(device != expected) throw std::invalid_argument("host-contiguous source has noncontiguous device aliases");
        const auto consumed = std::min(end, span->host_address + span->bytes) - cursor;
        expected = device + consumed;
        cursor += consumed;
        if(std::find(result.owners.begin(), result.owners.end(), span->owner) == result.owners.end())
            result.owners.push_back(span->owner);
        if(cursor == end) return result;
    }
    throw std::invalid_argument("source extent has a mapped coverage gap");
}
}
