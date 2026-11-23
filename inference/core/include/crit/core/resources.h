#pragma once
#include <cstddef>
#include <memory>
#include <span>

namespace crit::core {
enum class Memory { host, device };
struct Usage { std::size_t live = 0, peak = 0, limit = 0; };

// A reservation is shared by aliases and releases exactly once at the last owner.
class Reservation {
public:
    Reservation() = default;
    std::size_t bytes() const noexcept;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit Reservation(std::shared_ptr<Impl>);
    friend class Resources;
};
class Resources {
public:
    Resources(std::size_t host_limit, std::size_t device_limit);
    Reservation reserve(Memory, std::size_t bytes);
    Usage usage(Memory) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
}
