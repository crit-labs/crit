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

// Borrowed storage. Marks/rewind are arithmetic operations, never device fences.
class Arena {
public:
    Arena(void* storage, std::size_t capacity);
    std::span<std::byte> allocate(std::size_t bytes, std::size_t alignment = 256);
    std::size_t mark() const noexcept;
    void rewind(std::size_t mark);
    std::size_t peak() const noexcept;
private:
    std::byte* storage_;
    std::size_t capacity_, offset_ = 0, peak_ = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    // Called once before model construction. Owners may retain a copy of the ledger.
    virtual void initialize(Resources&) {}
    virtual void bind_thread() = 0;
    // Settles all backend work submitted by model/session calls, including failures.
    virtual void drain() = 0;
};
struct Services { Resources& resources; Backend& backend; };
// Failed backend drains retain the complete owner rather than freeing live storage.
// A permanently failed backend requires process recovery; this counter is diagnostic.
std::size_t quarantined_engines() noexcept;
}
