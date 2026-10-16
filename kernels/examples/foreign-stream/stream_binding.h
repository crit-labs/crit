#pragma once
#include <exception>
#include <thread>

namespace crit::kernels {
// Adapter current()/select(Stream) must access the foreign launcher's actual
// per-thread slot, without throwing. Validate device/ABI before constructing.
// This binds submission only: buffers and completion receipts remain caller-owned.
template<class Adapter> class ScopedStreamBinding {
public:
    using Stream = decltype(Adapter::current());
    static_assert(noexcept(Adapter::current()), "Stream lookup must be noexcept");
    static_assert(noexcept(Adapter::select(Stream{})), "Stream restoration must be noexcept");
    explicit ScopedStreamBinding(Stream stream) noexcept
        : previous_(Adapter::current()), thread_(std::this_thread::get_id()) {
        Adapter::select(stream);
    }
    ~ScopedStreamBinding() noexcept {
        if (thread_ != std::this_thread::get_id()) std::terminate();
        Adapter::select(previous_);
    }
    ScopedStreamBinding(const ScopedStreamBinding&) = delete;
    ScopedStreamBinding& operator=(const ScopedStreamBinding&) = delete;
    ScopedStreamBinding(ScopedStreamBinding&&) = delete;
    ScopedStreamBinding& operator=(ScopedStreamBinding&&) = delete;
private:
    Stream previous_;
    std::thread::id thread_;
};
}
