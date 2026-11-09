// The CPU expert tier's worker pool; see expert_pool.h.
//
// Claims carry their batch: `head` is one 64-bit word, epoch << 32 | next task, and a claim is a compare-and-swap that
// only succeeds for the epoch the worker woke for. run() first closes the finished batch's head (its epoch with an
// index no batch reaches), so a worker still in the earlier batch that reads the new task count cannot claim a task
// with it; then it publishes the batch's task count and function before the new head, so a worker that sees the epoch
// sees the batch. done counts finished tasks of the current batch only, so run() returns once its own tasks are done.
#include "expert_pool.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <set>
#include <string>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CRIT_POOL_PAUSE() _mm_pause()
#else
#define CRIT_POOL_PAUSE() ((void)0)
#endif

namespace crit::techniques::host_tier {
namespace {

// The first logical CPU of each physical core this process may run on (its affinity mask), in core order, where the
// OS says (Linux sysfs); empty elsewhere. Give each engine process its own cores (taskset, cgroups): pools of different
// processes then never pin onto the same core.
std::vector<int> core_leaders() {
    std::vector<int> leaders;
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    const bool masked = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
    std::set<std::pair<int, int>> seen;
    const unsigned n = std::thread::hardware_concurrency();
    for(unsigned cpu = 0; cpu < n; ++cpu) {
        if(masked && !CPU_ISSET(cpu, &allowed)) continue;
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::ifstream core(base + "core_id"), package(base + "physical_package_id");
        int c = -1, p = -1;
        if(!(core >> c) || !(package >> p)) return {};
        if(seen.insert({p, c}).second) leaders.push_back(int(cpu));
    }
#endif
    return leaders;
}

// The cores pools of this process have pinned workers to: a second pool takes others, or does not pin.
std::mutex claimed_mutex;
std::set<int> claimed;

}  // namespace

int physical_cores() noexcept {
    try {
        const auto leaders = core_leaders();
        if(!leaders.empty()) return int(leaders.size());
    } catch(...) {
    }
    const unsigned n = std::thread::hardware_concurrency();
    return n > 1 ? int(n / 2) : 1;
}

ExpertPool::ExpertPool(PoolOptions options) : options_(options) {
    int count = options_.workers >= 0 ? options_.workers : std::max(0, physical_cores() - 1);
    std::vector<int> leaders;
    if(options_.pin) {
        try {
            leaders = core_leaders();
        } catch(...) {
        }
    }
    // Cores for the workers: the process's first core stays the caller's (the host loop), then cores no other pool of
    // the process holds; workers past the free cores run unpinned.
    std::vector<int> mine;
    if(!leaders.empty()) {
        const std::lock_guard lock(claimed_mutex);
        for(std::size_t c = 1; c < leaders.size() && int(mine.size()) < count; ++c)
            if(claimed.insert(leaders[c]).second) mine.push_back(leaders[c]);
    }
    pinned_ = mine;
    for(int i = 0; i < count; ++i) {
        threads_.emplace_back([this, i] { worker(i); });
#if defined(__linux__)
        if(std::size_t(i) < mine.size()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(mine[std::size_t(i)], &set);
            pthread_setaffinity_np(threads_.back().native_handle(), sizeof(set), &set);
        }
#endif
    }
}

ExpertPool::~ExpertPool() {
    stop_.store(true, std::memory_order_seq_cst);
    {
        std::lock_guard<std::mutex> lock(sleep_mutex_);
    }
    sleep_cv_.notify_all();
    for(auto& t : threads_) t.join();
    const std::lock_guard lock(claimed_mutex);
    for(int core : pinned_) claimed.erase(core);
}

void ExpertPool::drain() {
    const std::uint64_t epoch = epoch_.load(std::memory_order_acquire);
    for(;;) {
        std::uint64_t head = head_.load(std::memory_order_acquire);
        if((head >> 32) != (epoch & 0xffffffffu)) return;
        const std::size_t index = std::size_t(head & 0xffffffffu);
        if(index >= tasks_.load(std::memory_order_acquire)) return;
        if(!head_.compare_exchange_weak(head, head + 1, std::memory_order_acq_rel)) continue;
        try {
            (*task_.load(std::memory_order_acquire))(index);
        } catch(...) {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if(!error_) error_ = std::current_exception();
        }
        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

namespace {
std::uint64_t now_us() {
    return std::uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now().time_since_epoch()).count());
}
}  // namespace

std::chrono::microseconds ExpertPool::spin_budget() const noexcept {
    if(options_.spin_max <= options_.spin) return options_.spin;
    std::uint32_t longest = 0;
    for(const auto& gap : gaps_) longest = std::max(longest, gap.load(std::memory_order_relaxed));
    const auto wanted = std::chrono::microseconds(std::uint64_t(longest) * 3 / 2);
    return std::clamp<std::chrono::microseconds>(wanted, options_.spin, options_.spin_max);
}

void ExpertPool::worker(int) {
    std::uint64_t seen = epoch_.load(std::memory_order_acquire);
    while(!stop_.load(std::memory_order_acquire)) {
        const auto start = std::chrono::steady_clock::now();
        const auto spin = spin_budget();
        const auto sleep_after = std::max<std::chrono::microseconds>(spin, options_.sleep_after);
        while(epoch_.load(std::memory_order_acquire) == seen && !stop_.load(std::memory_order_acquire)) {
            const auto waited = std::chrono::steady_clock::now() - start;
            if(waited < spin) {
                CRIT_POOL_PAUSE();
                continue;
            }
            if(waited < sleep_after) {
                std::this_thread::yield();
                continue;
            }
            // Sleep until the next batch: the epoch is read under the lock the publisher takes to notify.
            std::unique_lock<std::mutex> lock(sleep_mutex_);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lock, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_seq_cst);
            });
            sleepers_.fetch_sub(1, std::memory_order_seq_cst);
        }
        if(stop_.load(std::memory_order_acquire)) return;
        seen = epoch_.load(std::memory_order_acquire);
        drain();
    }
}

void ExpertPool::run(std::size_t tasks, const std::function<void(std::size_t)>& task) {
    if(tasks == 0) return;
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_ = nullptr;
    }
    const std::uint64_t previous = epoch_.load(std::memory_order_relaxed);
    if(options_.spin_max > options_.spin) {   // the gap since the last batch, for the adaptive spin
        const std::uint64_t last = last_end_us_.load(std::memory_order_relaxed);
        const std::uint64_t gap = last ? now_us() - last : 0;
        const std::uint64_t cap = std::uint64_t(options_.spin_max.count());
        gaps_[previous % 16].store(std::uint32_t(std::min(gap, cap)), std::memory_order_relaxed);
    }
    head_.store(((previous & 0xffffffffu) << 32) | 0xffffffffu, std::memory_order_seq_cst);   // the last batch closed
    tasks_.store(tasks, std::memory_order_release);
    task_.store(&task, std::memory_order_release);
    done_.store(0, std::memory_order_release);
    const std::uint64_t epoch = previous + 1;
    head_.store((epoch & 0xffffffffu) << 32, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(sleep_mutex_);
        epoch_.store(epoch, std::memory_order_seq_cst);
    }
    if(sleepers_.load(std::memory_order_seq_cst) > 0) {
        sleep_cv_.notify_all();
        wakes_.fetch_add(1, std::memory_order_relaxed);
    }
    drain();   // the caller claims tasks too
    while(done_.load(std::memory_order_acquire) < tasks) std::this_thread::yield();
    batches_.fetch_add(1, std::memory_order_relaxed);
    if(options_.spin_max > options_.spin) last_end_us_.store(now_us(), std::memory_order_relaxed);
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error = error_;
    }
    if(error) std::rethrow_exception(error);
}

}  // namespace crit::techniques::host_tier
