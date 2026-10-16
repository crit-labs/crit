#include "crit/core.h"
#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace crit::core {
struct Resources::Impl {
    std::mutex mutex;
    std::array<Usage,2> values;
    Impl(std::size_t host, std::size_t device): values{{{0,0,host},{0,0,device}}} {}
};
struct Reservation::Impl {
    std::size_t size;
    std::function<void()> release;
    ~Impl() { release(); }
};
Reservation::Reservation(std::shared_ptr<Impl> value): impl_(std::move(value)) {}
std::size_t Reservation::bytes() const noexcept { return impl_ ? impl_->size : 0; }
Resources::Resources(std::size_t host, std::size_t device): impl_(std::make_shared<Impl>(host,device)) {}
static std::size_t index(Memory memory) {
    if(memory == Memory::host) return 0;
    if(memory == Memory::device) return 1;
    throw std::invalid_argument("invalid memory domain");
}
Reservation Resources::reserve(Memory memory, std::size_t bytes) {
    const auto i = index(memory);
    auto owner = impl_;
    auto reservation = std::make_shared<Reservation::Impl>();
    reservation->size = bytes;
    reservation->release = []{};
    std::function<void()> release = [owner,i,bytes] {
        std::lock_guard lock(owner->mutex);
        owner->values[i].live -= bytes;
    };
    std::lock_guard lock(owner->mutex);
    auto& value = owner->values[i];
    if(bytes > value.limit-value.live) throw std::runtime_error("resource reservation exceeds ceiling");
    value.live += bytes;
    value.peak = std::max(value.peak,value.live);
    reservation->release = std::move(release);
    return Reservation(std::move(reservation));
}
Usage Resources::usage(Memory memory) const {
    const auto i = index(memory);
    std::lock_guard lock(impl_->mutex);
    return impl_->values[i];
}
Arena::Arena(void* storage, std::size_t capacity):storage_(static_cast<std::byte*>(storage)),capacity_(capacity) {
    if(!storage || !capacity) throw std::invalid_argument("arena requires nonempty borrowed storage");
    const auto base = reinterpret_cast<std::uintptr_t>(storage);
    if(capacity > std::numeric_limits<std::uintptr_t>::max()-base) throw std::overflow_error("arena address overflow");
}
std::span<std::byte> Arena::allocate(std::size_t bytes, std::size_t alignment) {
    if(!alignment || (alignment & (alignment-1))) throw std::invalid_argument("arena alignment must be a power of two");
    const auto address = reinterpret_cast<std::uintptr_t>(storage_)+offset_;
    const auto mask = alignment-1;
    if(mask > std::numeric_limits<std::uintptr_t>::max()-address) throw std::overflow_error("arena alignment overflow");
    const auto aligned = (address+mask)&~mask;
    const auto start = aligned-reinterpret_cast<std::uintptr_t>(storage_);
    if(start > capacity_ || bytes > capacity_-start) throw std::runtime_error("arena capacity exceeded");
    offset_ = start+bytes;
    peak_ = std::max(peak_,offset_);
    return {storage_+start,bytes};
}
std::size_t Arena::mark() const noexcept { return offset_; }
void Arena::rewind(std::size_t mark) {
    if(mark > offset_) throw std::invalid_argument("arena cannot rewind to a future mark");
    offset_ = mark;
}
std::size_t Arena::peak() const noexcept { return peak_; }
}
