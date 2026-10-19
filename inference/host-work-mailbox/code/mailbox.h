// Host work joined inside a CUDA stream, including captured graphs: the device posts a job into mapped pinned
// memory, a host service thread runs it, and the stream waits for the answer before it reads the result.
//
// Per slot the device resets done = 0, then raises ready = 1 (system-fenced, after its payload writes); the host
// clears ready, runs the job, writes status and then done = 1; the stream's join waits for done >= 1. The device
// resets done because the values of captured stream memory operations are fixed at capture: a wait for a
// per-replay sequence number cannot be captured. A slot the device leaves without host work it answers itself
// (skip: status = skipped, done = 1), so every captured join always completes.
//
// Two capturable joins wait on the same word: the stream front end (cuStreamWaitValue32; no SM is held) or a
// one-thread poll kernel. Their cost depends on the driver and platform, so choose by measure_joins(). The eager
// join synchronizes the stream on the host and works wherever mapped memory does, outside capture only.
#pragma once
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace crit::techniques::host_mailbox {

// Status words. The device writes skipped; the host writes the others before done.
enum Status : std::uint32_t { pending = 0, served = 1, skipped = 2, failed = 3, released = 4 };
constexpr std::uint32_t kReleasedDone = 0x7fffffffu;  // satisfies every done >= 1 wait
constexpr std::size_t kLine = 128;                    // each slot's words on their own line

// One slot's words and payload; host and device addresses of the same mapped bytes.
struct SlotView {
    std::uint32_t* ready = nullptr;  // host address
    std::uint32_t* done = nullptr;
    std::uint32_t* status = nullptr;
    std::byte* payload = nullptr;
    std::uint32_t* device_ready = nullptr;  // device addresses (kernels and stream operations)
    std::uint32_t* device_done = nullptr;
    std::uint32_t* device_status = nullptr;
    std::byte* device_payload = nullptr;
    std::size_t payload_bytes = 0;
};

enum class Join { stream_wait, device_poll, host_eager };
const char* join_name(Join) noexcept;

// Mapped bytes the mailbox lays its slots over: from Core (CudaBackend::allocate(Memory::host) with
// device_address(), or cuda_map_host) or cudaHostAlloc(cudaHostAllocMapped). The owner keeps them alive and
// accounted for until the mailbox is destroyed.
struct MappedRegion {
    void* host = nullptr;
    void* device = nullptr;
    std::size_t bytes = 0;
    std::shared_ptr<const void> owner;
};
std::size_t region_bytes(std::size_t slots, std::size_t payload_bytes);

// Runs one job; throwing fails the mailbox. Called on the service thread with the slot index and its payload
// (inputs the device wrote, outputs the device reads after the join).
using Handler = std::function<void(std::size_t slot, std::span<std::byte> payload)>;

struct Options {
    std::chrono::microseconds idle_after{std::chrono::milliseconds(200)};  // spin this long, then back off
    std::chrono::microseconds idle_sleep{50};
};

class Mailbox {
public:
    // The join kind is fixed per mailbox: graphs capture it. host_eager starts no service thread.
    Mailbox(MappedRegion, std::size_t slots, std::size_t payload_bytes, Join, Handler, Options = {});
    ~Mailbox();  // release(), then stops the service; the owner has drained every stream that joins
    Mailbox(const Mailbox&) = delete;
    Mailbox& operator=(const Mailbox&) = delete;

    std::size_t slots() const noexcept { return slots_.size(); }
    Join join_kind() const noexcept { return join_; }
    const SlotView& slot(std::size_t) const;

    // Stream-level doorbell (capturable): a one-thread kernel resets done, fences and raises ready. Engines
    // usually post from their own kernel with mailbox_device.cuh instead; every payload write must be
    // system-fenced first (in the writing thread, or by an earlier kernel in stream order).
    cudaError_t post(std::size_t slot, cudaStream_t) const;
    // The device answers the slot itself (no host work this time); capturable.
    cudaError_t skip(std::size_t slot, cudaStream_t) const;
    // Waits in the stream until the slot is answered. Capturable for stream_wait and device_poll; host_eager
    // synchronizes the stream, serves the job on the calling thread and rejects capture.
    cudaError_t join(std::size_t slot, cudaStream_t);

    // Answers every pending and later doorbell at once with status released (shutdown, or an owner's watchdog
    // for a stuck window), from a sweeper thread so it works while a job is still running. Outputs of windows
    // that saw it are invalid. rearm() starts the next window.
    void release();
    // Needs every joining stream drained. Waits for a running job to return (up to `timeout`), clears all words
    // and accepts work again. Returns false when a job is still running or the mailbox has failed.
    bool rearm(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

    // A handler that threw fails the mailbox for good: every later doorbell is answered at once with status
    // failed and the handler is never called again. The owner discards outputs of windows that saw it.
    bool failed() const noexcept { return failed_.load(std::memory_order_acquire); }
    std::string error() const;
    std::uint64_t served_jobs() const noexcept { return served_.load(std::memory_order_relaxed); }

private:
    void service();
    void sweep();
    bool serve_ready(std::size_t slot);
    void answer(std::size_t slot, Status) noexcept;
    std::vector<SlotView> slots_;
    MappedRegion region_;
    Join join_;
    Handler handler_;
    Options options_;
    std::thread service_, sweeper_;
    std::atomic<bool> stop_{false}, failed_{false}, released_{false};
    std::atomic<std::uint64_t> served_{0};
    mutable std::mutex error_mutex_;
    std::mutex sweeper_mutex_;
    std::timed_mutex serve_mutex_;  // held while a job runs; rearm() waits on it
    std::string error_;
};

// What the device and driver support, from functional checks rather than attributes: stream_wait needs the
// driver's 32-bit stream memory operations on mapped memory, eagerly and inside a captured graph.
struct JoinSupport {
    bool mapped = false, stream_wait = false, stream_wait_captured = false;
    std::string detail;
};
JoinSupport probe_joins(int device);

// Cost of one post plus join with an empty handler, inside a captured graph replayed `replays` times (median of
// per-replay means), for every supported capturable join. Microseconds; empty when unsupported.
struct JoinCosts {
    JoinSupport support;
    std::optional<double> stream_wait_us, device_poll_us;
    // The cheaper measured capturable join, or host_eager when none is supported.
    Join choice() const noexcept;
};
JoinCosts measure_joins(int device, int joins_per_graph = 32, int replays = 50);

}  // namespace crit::techniques::host_mailbox
