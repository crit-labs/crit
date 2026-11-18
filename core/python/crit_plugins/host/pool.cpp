// Core's worker pool; see pool.h.
//
// Claims carry their batch: `head` is one 64-bit word, epoch << 32 | next task, and a claim is a compare-and-swap that
// only succeeds for the epoch the worker woke for. run() first closes the finished batch's head (its epoch with an
// index no batch reaches), so a worker still in the earlier batch that reads the new task count cannot claim a task
// with it; then it publishes the batch's task count and function before the new head, so a worker that sees the epoch
// sees the batch. done counts finished tasks of the current batch only, so run() returns once its own tasks are done.
#include "pool.h"

#include <algorithm>
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

namespace crit::host {
namespace {

// The cores pools of this process pinned workers to by themselves (no plan): a second such pool takes others, or
// does not pin.
std::mutex claimed_mutex;
std::set<int> claimed;

thread_local bool in_task = false;   // the calling thread runs a task of a batch: a batch from it runs inline

std::uint64_t now_us() {
    return std::uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now().time_since_epoch()).count());
}

}  // namespace

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

int physical_cores() noexcept {
    try {
        const auto leaders = core_leaders();
        if(!leaders.empty()) return int(leaders.size());
    } catch(...) {
    }
    const unsigned n = std::thread::hardware_concurrency();
    return n > 1 ? int(n / 2) : 1;
}

bool pin_thread(int cpu) noexcept {
#if defined(__linux__)
    if(cpu < 0 || cpu >= CPU_SETSIZE) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

Pool::Pool(PoolOptions options) : options_(std::move(options)) {
    std::vector<int> mine;   // worker i's CPU, where it has one
    int count = 0;
    if(!options_.cpus.empty()) {   // a plan: one worker per entry
        mine = options_.cpus;
        count = int(mine.size());
    } else {
        count = options_.workers >= 0 ? options_.workers : std::max(0, physical_cores() - 1);
        std::vector<int> leaders;
        if(options_.pin) {
            try {
                leaders = core_leaders();
            } catch(...) {
            }
        }
        // Cores for the workers: the process's first core stays the caller's, then cores no other pool of the
        // process took by itself; workers past the free cores run unpinned.
        if(!leaders.empty()) {
            const std::lock_guard<std::mutex> lock(claimed_mutex);
            for(std::size_t c = 1; c < leaders.size() && int(mine.size()) < count; ++c)
                if(claimed.insert(leaders[c]).second) mine.push_back(leaders[c]);
        }
        claimed_ = mine;
    }
    for(int i = 0; i < count; ++i) {
        const int cpu = std::size_t(i) < mine.size() ? mine[std::size_t(i)] : -1;
        threads_.emplace_back([this, i, cpu] {
            if(cpu >= 0) pin_thread(cpu);
            worker(i);
        });
        if(cpu >= 0) pinned_.push_back(cpu);
    }
}

Pool::~Pool() {
    stop_.store(true, std::memory_order_seq_cst);
    {
        std::lock_guard<std::mutex> lock(sleep_mutex_);
    }
    sleep_cv_.notify_all();
    for(auto& t : threads_) t.join();
    const std::lock_guard<std::mutex> lock(claimed_mutex);
    for(int core : claimed_) claimed.erase(core);
}

void Pool::drain() {
    const std::uint64_t epoch = epoch_.load(std::memory_order_acquire);
    for(;;) {
        std::uint64_t head = head_.load(std::memory_order_acquire);
        if((head >> 32) != (epoch & 0xffffffffu)) return;
        const std::size_t index = std::size_t(head & 0xffffffffu);
        if(index >= tasks_.load(std::memory_order_acquire)) return;
        if(!head_.compare_exchange_weak(head, head + 1, std::memory_order_acq_rel)) continue;
        in_task = true;
        task_.load(std::memory_order_acquire)(context_.load(std::memory_order_acquire), index);
        in_task = false;
        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

std::chrono::microseconds Pool::spin_budget() const noexcept {
    if(options_.spin_max <= options_.spin) return options_.spin;
    std::uint32_t longest = 0;
    for(const auto& gap : gaps_) longest = std::max(longest, gap.load(std::memory_order_relaxed));
    const auto wanted = std::chrono::microseconds(std::uint64_t(longest) * 3 / 2);
    return std::clamp<std::chrono::microseconds>(wanted, options_.spin, options_.spin_max);
}

void Pool::worker(int) {
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

void Pool::run(std::size_t tasks, PoolTask task, void* context) {
    if(tasks == 0) return;
    if(in_task) {   // a task's own batch: on its thread, as the pool is in a batch
        for(std::size_t i = 0; i < tasks; ++i) task(context, i);
        return;
    }
    const std::lock_guard<std::mutex> one(run_mutex_);   // callers' batches run one after the other
    const std::uint64_t previous = epoch_.load(std::memory_order_relaxed);
    if(options_.spin_max > options_.spin) {   // the gap since the last batch, for the adaptive spin
        const std::uint64_t last = last_end_us_.load(std::memory_order_relaxed);
        const std::uint64_t gap = last ? now_us() - last : 0;
        const std::uint64_t cap = std::uint64_t(options_.spin_max.count());
        gaps_[previous % 16].store(std::uint32_t(std::min(gap, cap)), std::memory_order_relaxed);
    }
    head_.store(((previous & 0xffffffffu) << 32) | 0xffffffffu, std::memory_order_seq_cst);   // the last batch closed
    tasks_.store(tasks, std::memory_order_release);
    task_.store(task, std::memory_order_release);
    context_.store(context, std::memory_order_release);
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
}

}  // namespace crit::host

extern "C" {

void* crit_pool_create(const int* cpus, int workers, int spin_us, int sleep_after_us, int spin_max_us) {
    try {
        crit::host::PoolOptions options;
        options.workers = std::max(0, workers);
        options.pin = false;
        if(cpus && workers > 0) options.cpus.assign(cpus, cpus + workers);
        options.spin = std::chrono::microseconds(std::max(0, spin_us));
        options.sleep_after = std::chrono::microseconds(std::max(spin_us, sleep_after_us));
        options.spin_max = std::chrono::microseconds(std::max(0, spin_max_us));
        return new crit::host::Pool(std::move(options));
    } catch(...) {
        return nullptr;
    }
}

void crit_pool_destroy(void* pool) { delete static_cast<crit::host::Pool*>(pool); }

int crit_pool_run(void* pool, std::size_t tasks, crit::host::PoolTask task, void* context) {
    static_cast<crit::host::Pool*>(pool)->run(tasks, task, context);
    return 0;
}

void crit_pool_stats(void* pool, std::uint64_t* out) {
    const auto* p = static_cast<const crit::host::Pool*>(pool);
    out[0] = p->batches();
    out[1] = p->wakes();
    out[2] = std::uint64_t(p->workers());
}

int crit_host_core_leaders(int* out, int capacity) {
    try {
        const auto leaders = crit::host::core_leaders();
        for(std::size_t i = 0; i < leaders.size() && int(i) < capacity; ++i) out[i] = leaders[i];
        return int(leaders.size());
    } catch(...) {
        return 0;
    }
}

int crit_host_pin_thread(int cpu) { return crit::host::pin_thread(cpu) ? 1 : 0; }

}  // extern "C"
