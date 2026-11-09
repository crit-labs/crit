// A worker pool for a CPU expert tier: a routed layer's few CPU experts are many rows, and the rows are DRAM-bound, so
// the pool's job is to keep one thread per physical core streaming expert bytes for the whole layer at a cost far
// below one expert's rows.
//
// One flat batch at a time: a layer's work must finish before the device reads it, so a layer is a barrier by
// construction and the queue never holds more than one batch. A batch is `tasks` task indices that the workers and
// the calling thread claim with one atomic increment each (no lock); run() returns when every task has finished.
// Each task is a row range of one expert, so a layer with fewer experts than threads still streams every expert at
// every core's rate. Idle workers spin (with a pause hint), then yield until `sleep_after`, then sleep on a condition
// variable, so a pool between batches costs no CPU once the gap passes; a batch published to sleeping workers pays one
// wake-up (tens to hundreds of microseconds under a loaded or virtualized OS). With `spin_max`, a worker spins as long
// as the longest of the last 16 gaps between batches (1.5 times it, at least `spin`, at most `spin_max`): through a
// decode step's layers, a millisecond apart, workers stay awake, and after the step they sleep.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace crit::techniques::host_tier {

struct PoolOptions {
    int workers = -1;                                         // -1: physical cores minus one (the caller's)
    bool pin = true;   // pin each worker to a physical core of its own (Linux): of the process's CPU set, after its
                       // first core, none another pool of the process holds
    std::chrono::microseconds spin{100};
    std::chrono::microseconds sleep_after{100};
    std::chrono::microseconds spin_max{0};   // > spin: spin as long as recent gaps (adaptive), at most this
};

// Physical cores: the hardware threads halved where the CPU reports two per core (an estimate where it cannot).
int physical_cores() noexcept;

class ExpertPool {
public:
    explicit ExpertPool(PoolOptions = {});
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    int workers() const noexcept { return int(threads_.size()); }
    const std::vector<int>& pinned() const noexcept { return pinned_; }   // the cores its workers were pinned to
    // Runs task(i) for i in [0, tasks) on the workers and the calling thread; returns when all have finished. A task
    // that throws makes run() rethrow the first exception after the batch has drained. One batch at a time.
    void run(std::size_t tasks, const std::function<void(std::size_t)>& task);
    std::uint64_t batches() const noexcept { return batches_.load(std::memory_order_relaxed); }
    // Batches whose publisher found sleeping workers and woke them.
    std::uint64_t wakes() const noexcept { return wakes_.load(std::memory_order_relaxed); }

private:
    void worker(int index);
    void drain();
    PoolOptions options_;
    std::vector<std::thread> threads_;
    std::vector<int> pinned_;
    std::atomic<std::uint64_t> epoch_{0};
    std::atomic<std::uint64_t> head_{0};   // epoch << 32 | next task
    std::atomic<std::size_t> done_{0};
    std::atomic<std::size_t> tasks_{0};
    std::atomic<const std::function<void(std::size_t)>*> task_{nullptr};
    std::atomic<bool> stop_{false};
    std::atomic<std::uint64_t> batches_{0}, wakes_{0};
    // The adaptive spin: the last 16 gaps between one batch's end and the next one's start (us, capped at spin_max).
    std::atomic<std::uint32_t> gaps_[16] = {};
    std::atomic<std::uint64_t> last_end_us_{0};
    std::chrono::microseconds spin_budget() const noexcept;
    std::mutex sleep_mutex_, error_mutex_;
    std::condition_variable sleep_cv_;
    std::atomic<int> sleepers_{0};
    std::exception_ptr error_;
};

}  // namespace crit::techniques::host_tier
