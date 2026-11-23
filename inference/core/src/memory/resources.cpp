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
}
