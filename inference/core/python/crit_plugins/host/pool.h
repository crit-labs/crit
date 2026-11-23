// Core's worker pool: the host's cores as one pool a process (crit_plugins.host), which Core's and every plugin's
// native code runs its batches on, so the threads of an engine never outnumber its cores.
//
// One flat batch at a time: host work inside a device's step must finish before the device reads it, so a batch is a
// barrier by construction and the queue never holds more than one. A batch is `tasks` task indices that the workers
// and the calling thread claim with one atomic increment each (no lock); run() returns when every task has finished.
// Batches of different callers run one after the other. Idle workers spin (with a pause hint), then yield until
// `sleep_after`, then sleep on a condition variable, so a pool between batches costs no CPU once the gap passes; a
// batch published to sleeping workers pays one wake-up (tens to hundreds of microseconds under a loaded or
// virtualized OS). With `spin_max`, a worker spins as long as the longest of the last 16 gaps between batches (1.5
// times it, at least `spin`, at most `spin_max`): through a decode step's layers, a millisecond apart, workers stay
// awake, and after the step they sleep.
//
// Native code that is built apart from the pool (a plugin's library) runs on it through PoolRef: the pool's C entry
// and the pool, which crit_plugins.host hands out as two addresses.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace crit::host {

// A task of a batch: called with the batch's context and the task's index. It does not throw (PoolRef::run wraps
// tasks that may) and runs no batch of its own.
using PoolTask = void (*)(void* context, std::size_t index);

struct PoolOptions {
    int workers = -1;   // without `cpus`: -1 is one per physical core but the first (the caller's)
    bool pin = true;    // without `cpus`: pin each worker to a physical core of its own (Linux): of the process's
                        // CPU set, after its first core, none another pool of the process holds
    std::chrono::microseconds spin{100};
    std::chrono::microseconds sleep_after{100};
    std::chrono::microseconds spin_max{0};   // > spin: spin as long as recent gaps (adaptive), at most this
    std::vector<int> cpus = {};   // a plan (crit_plugins.host's): one worker per entry, pinned to that CPU (< 0: not)
};

// The first logical CPU of each physical core this process may run on (its affinity mask), in core order, where the
// OS says (Linux sysfs); empty elsewhere.
std::vector<int> core_leaders();
// Physical cores: core_leaders()' count, else the hardware threads halved (an estimate).
int physical_cores() noexcept;
// Pins the calling thread to one logical CPU; false where the OS does not (or it is not in the process's set).
bool pin_thread(int cpu) noexcept;

class Pool {
public:
    explicit Pool(PoolOptions = {});
    ~Pool();
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    int workers() const noexcept { return int(threads_.size()); }
    const std::vector<int>& pinned() const noexcept { return pinned_; }   // the CPUs its workers were pinned to
    // Runs task(context, i) for i in [0, tasks) on the workers and the calling thread; returns when all have
    // finished. A call from inside a task runs its tasks on the calling thread.
    void run(std::size_t tasks, PoolTask task, void* context);
    // The same for any callable task(i): the first exception a task threw is rethrown after the batch has drained.
    template <class Fn> void run(std::size_t tasks, Fn&& task);
    std::uint64_t batches() const noexcept { return batches_.load(std::memory_order_relaxed); }
    // Batches whose publisher found sleeping workers and woke them.
    std::uint64_t wakes() const noexcept { return wakes_.load(std::memory_order_relaxed); }

private:
    void worker(int index);
    void drain();
    PoolOptions options_;
    std::vector<std::thread> threads_;
    std::vector<int> pinned_, claimed_;
    std::atomic<std::uint64_t> epoch_{0};
    std::atomic<std::uint64_t> head_{0};   // epoch << 32 | next task
    std::atomic<std::size_t> done_{0};
    std::atomic<std::size_t> tasks_{0};
    std::atomic<PoolTask> task_{nullptr};
    std::atomic<void*> context_{nullptr};
    std::atomic<bool> stop_{false};
    std::atomic<std::uint64_t> batches_{0}, wakes_{0};
    // The adaptive spin: the last 16 gaps between one batch's end and the next one's start (us, capped at spin_max).
    std::atomic<std::uint32_t> gaps_[16] = {};
    std::atomic<std::uint64_t> last_end_us_{0};
    std::chrono::microseconds spin_budget() const noexcept;
    std::mutex sleep_mutex_, run_mutex_;
    std::condition_variable sleep_cv_;
    std::atomic<int> sleepers_{0};
};

}  // namespace crit::host

extern "C" {
// The pool's C interface (ctypes, and PoolRef's entry). create: `workers` threads, worker i pinned to cpus[i] where
// cpus is given and cpus[i] >= 0; null when a thread could not start.
void* crit_pool_create(const int* cpus, int workers, int spin_us, int sleep_after_us, int spin_max_us);
void crit_pool_destroy(void* pool);
// Runs the batch; 0. `task` does not throw.
int crit_pool_run(void* pool, std::size_t tasks, crit::host::PoolTask task, void* context);
// batches, wakes, workers.
void crit_pool_stats(void* pool, std::uint64_t* out);
// The first logical CPU of each physical core of the process's CPU set into out[capacity]; their count.
int crit_host_core_leaders(int* out, int capacity);
// Pins the calling thread to `cpu`; 1 when it was pinned.
int crit_host_pin_thread(int cpu);
}

namespace crit::host {

// A pool as native code built apart from it holds it: the pool's C entry (crit_pool_run) and the pool. Without an
// entry the batch runs on the calling thread.
struct PoolRef {
    using Entry = int (*)(void* pool, std::size_t tasks, PoolTask task, void* context);
    Entry entry = nullptr;
    void* pool = nullptr;
    PoolRef() = default;
    PoolRef(Entry entry_, void* pool_) : entry(entry_), pool(pool_) {}
    PoolRef(Pool& linked) : entry(&crit_pool_run), pool(&linked) {}   // for code that links the pool

    // Runs task(i) for i in [0, tasks); returns when all have finished and rethrows the first exception a task threw,
    // after the batch has drained.
    template <class Fn> void run(std::size_t tasks, Fn&& task) const {
        if(tasks == 0) return;
        struct Call {
            std::remove_reference_t<Fn>* task;
            std::mutex mutex;
            std::exception_ptr error;
        } call{&task, {}, nullptr};
        const PoolTask one = [](void* context, std::size_t index) {
            auto* c = static_cast<Call*>(context);
            try {
                (*c->task)(index);
            } catch(...) {
                const std::lock_guard<std::mutex> lock(c->mutex);
                if(!c->error) c->error = std::current_exception();
            }
        };
        if(entry) entry(pool, tasks, one, &call);
        else for(std::size_t i = 0; i < tasks; ++i) one(&call, i);
        if(call.error) std::rethrow_exception(call.error);
    }
};

template <class Fn> void Pool::run(std::size_t tasks, Fn&& task) { PoolRef(*this).run(tasks, std::forward<Fn>(task)); }

}  // namespace crit::host
