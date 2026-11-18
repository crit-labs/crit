// Host work joined inside a CUDA stream, captured graphs included (crit_plugins.host's handoff): the device posts a
// job into mapped pinned memory, the host's service thread runs it, and the stream waits for the answer before it
// reads the result.
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
//
// One service thread a process answers the doorbells of every handoff (Service): jobs of different handoffs run one
// after the other, as batches of the host's pool do, and the thread that runs a job is the caller of the batches
// the job hands to the pool.
//
// A job that does not return would leave the device waiting for good: a stream's join has no clock. So the service
// keeps a second thread that only watches. A job that runs past the service's job budget fails its handoff
// (expire): every pending and later doorbell is answered at once, the device's streams drain, failed() is true with
// the reason, and the owner stops serving. The job's thread is not stopped; nothing of it is read again.
#pragma once
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
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

#include "handoff_device.cuh"

namespace crit::host {

constexpr std::uint32_t kReleasedDone = 0x7fffffffu;  // satisfies every done >= 1 wait
constexpr std::size_t kLine = 128;                    // each slot's words on their own line

// One slot's words and payload; host and device addresses of the same mapped bytes.
struct SlotView {
    std::uint32_t* ready = nullptr;  // host addresses
    std::uint32_t* done = nullptr;
    std::uint32_t* status = nullptr;
    std::byte* payload = nullptr;
    DeviceSlot device;               // device addresses (kernels and stream operations)
    std::size_t payload_bytes = 0;
};

enum class Join { stream_wait, device_poll, host_eager };
const char* join_name(Join) noexcept;

// Mapped bytes a handoff lays its slots over: cudaHostAlloc(cudaHostAllocMapped) or a mapped host allocation of the
// owner. The owner keeps them alive and accounted for until the handoff is destroyed.
struct MappedRegion {
    void* host = nullptr;
    void* device = nullptr;
    std::size_t bytes = 0;
    std::shared_ptr<const void> owner;
};
std::size_t region_bytes(std::size_t slots, std::size_t payload_bytes);

// Runs one job; throwing fails the handoff. Called on the service thread with the slot index and its payload
// (inputs the device wrote, outputs the device reads after the join).
using Handler = std::function<void(std::size_t slot, std::span<std::byte> payload)>;

class Handoff {
public:
    // The join kind is fixed per handoff: graphs capture it. host_eager is served by the joining thread.
    Handoff(MappedRegion, std::size_t slots, std::size_t payload_bytes, Join, Handler);
    ~Handoff();  // release(), then leaves the service; the owner has drained every stream that joins
    Handoff(const Handoff&) = delete;
    Handoff& operator=(const Handoff&) = delete;

    std::size_t slots() const noexcept { return slots_.size(); }
    Join join_kind() const noexcept { return join_; }
    const SlotView& slot(std::size_t) const;

    // Stream-level doorbell (capturable): a one-thread kernel resets done, fences and raises ready. Engines
    // usually post from their own kernel with handoff_device.cuh instead; every payload write must be
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
    // and accepts work again. Returns false when a job is still running or the handoff has failed.
    bool rearm(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

    // A handler that threw fails the handoff for good: every later doorbell is answered at once with status
    // failed and the handler is never called again. The owner discards outputs of windows that saw it.
    bool failed() const noexcept { return failed_.load(std::memory_order_acquire); }
    // How long the job now running has run (0: none runs), and the watch's verdict on one that ran `ran`: the
    // handoff fails with that reason and release()s. Called from the service's watch thread.
    std::chrono::milliseconds busy() const noexcept;
    void expire(std::chrono::milliseconds ran);
    std::string error() const;
    std::uint64_t served_jobs() const noexcept { return served_.load(std::memory_order_relaxed); }

    // One pass of the service over this handoff's doorbells (the slot after the last one served first, every
    // slot each 64th pass): whether a doorbell was answered.
    bool poll(std::uint32_t pass);

private:
    void sweep();
    bool serve_ready(std::size_t slot);
    void answer(std::size_t slot, Status) noexcept;
    std::vector<SlotView> slots_;
    MappedRegion region_;
    Join join_;
    Handler handler_;
    std::thread sweeper_;
    std::size_t next_ = 0;
    std::atomic<bool> stop_{false}, failed_{false}, released_{false};
    std::atomic<std::uint64_t> served_{0};
    std::atomic<std::int64_t> busy_since_ns_{0};   // the running job's start on the steady clock, 0 when none runs
    mutable std::mutex error_mutex_;
    std::mutex sweeper_mutex_;
    std::timed_mutex serve_mutex_;  // held while a job runs; rearm() waits on it
    std::string error_;
};

// The process's service thread: it polls the doorbells of every handoff with a capturable join, runs their jobs
// and answers them. It runs while a handoff is registered; idle, it spins `idle_after` and then sleeps
// `idle_sleep` between passes. cpu: the logical CPU it is pinned to (Linux; < 0: not pinned).
class Service {
public:
    using Pin = int (*)(int cpu);   // pins the calling thread (the pool's crit_host_pin_thread)
    static Service& instance();
    // job_budget: the longest a job may run before its handoff fails (zero or less: no watch).
    void configure(int cpu, Pin pin, std::chrono::microseconds idle_after, std::chrono::microseconds idle_sleep,
                   std::chrono::milliseconds job_budget = std::chrono::milliseconds(20000));
    void add(Handoff*);
    void remove(Handoff*);   // returns once no pass reads the handoff: a job that never returns blocks here
    ~Service();

private:
    Service() = default;
    void loop(std::shared_ptr<std::atomic<bool>> stop);
    void watch();
    std::mutex mutex_;            // boxes_ and the thread, and held through a pass
    std::vector<Handoff*> boxes_;
    std::thread thread_;
    std::shared_ptr<std::atomic<bool>> stop_;   // the running thread's own: a thread that follows it has another
    std::atomic<int> waiting_{0}, cpu_{-1};
    std::atomic<Pin> pin_{nullptr};
    std::atomic<std::int64_t> idle_after_us_{2000}, idle_sleep_us_{50}, job_budget_ms_{20000};
    std::mutex watch_mutex_;      // watched_ and watch_stop_; never held with mutex_, which a running job's pass holds
    std::condition_variable watch_wake_;
    std::vector<Handoff*> watched_;
    std::thread watch_thread_;
    bool watch_stop_ = false;
};

// What the device and driver support, from functional checks rather than attributes: stream_wait needs the
// driver's 32-bit stream memory operations on mapped memory, eagerly and inside a captured graph.
struct JoinSupport {
    bool mapped = false, stream_wait = false, stream_wait_captured = false;
    std::string detail;
};
JoinSupport probe_joins(int device);   // on `device`; the caller's current device is left as it was

// Cost of one post plus join with an empty handler, inside a captured graph replayed `replays` times (median of
// per-replay means), for every supported capturable join. Microseconds; empty when unsupported.
struct JoinCosts {
    JoinSupport support;
    std::optional<double> stream_wait_us, device_poll_us;
    // The cheaper measured capturable join, or host_eager when none is supported.
    Join choice() const noexcept;
};
JoinCosts measure_joins(int device, int joins_per_graph = 32, int replays = 50);

}  // namespace crit::host
