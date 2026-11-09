// Native gather of file-backed lookup-table rows (token or hashed n-gram embeddings), the C++ form of table_rows.py:
// the table opened for random access (POSIX_FADV_RANDOM; FILE_FLAG_RANDOM_ACCESS on Windows), exactly the rows a step
// needs read with positioned reads by a persistent worker pool, rows asked for twice read once, an optional
// least-recently-used cache of hot rows, and the rows written in request order straight into the caller's buffer: a
// slot of a StagingRing over pinned host memory, which the engine then copies to the device. RowPrefetch starts a
// token's rows as soon as the token exists and lets the step that reads them wait only for rows still in flight.
#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <list>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace crit::techniques::table_rows {

struct Region {
    std::string path;
    std::uint64_t offset = 0;      // the first row's byte offset in the file
    std::uint64_t rows = 0;
    std::size_t row_bytes = 0;
};

struct Counters {
    std::uint64_t reads = 0, bytes_read = 0, hits = 0;
};

namespace detail {
// One file opened read-only for random access, read with positioned reads from any thread.
class RandomFile {
public:
    explicit RandomFile(const std::string& path) {
#if defined(_WIN32)
        const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring name(static_cast<std::size_t>(wide > 0 ? wide : 1), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, name.data(), wide);
        handle_ = CreateFileW(name.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if(handle_ == INVALID_HANDLE_VALUE) fail(path + ": open");
        LARGE_INTEGER size;
        if(!GetFileSizeEx(handle_, &size)) fail(path + ": size");
        size_ = static_cast<std::uint64_t>(size.QuadPart);
#else
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if(fd_ < 0) fail(path + ": open");
        struct stat st;
        if(::fstat(fd_, &st) != 0) fail(path + ": stat");
        size_ = static_cast<std::uint64_t>(st.st_size);
#endif
    }
    RandomFile(const RandomFile&) = delete;
    RandomFile& operator=(const RandomFile&) = delete;
    ~RandomFile() {
#if defined(_WIN32)
        if(handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
        if(fd_ >= 0) ::close(fd_);
#endif
    }
    std::uint64_t size() const noexcept { return size_; }
    // No read-ahead over [offset, offset + length): returns the advice applied ("fadvise_random", or "open_random"
    // on Windows, where FILE_FLAG_RANDOM_ACCESS at open is the advice).
    const char* advise_random(std::uint64_t offset, std::uint64_t length) const {
#if defined(_WIN32)
        (void)offset;
        (void)length;
        return "open_random";
#elif defined(POSIX_FADV_RANDOM)
        return ::posix_fadvise(fd_, static_cast<off_t>(offset), static_cast<off_t>(length), POSIX_FADV_RANDOM) == 0
                   ? "fadvise_random" : "none";
#else
        (void)offset;
        (void)length;
        return "none";
#endif
    }
    void read_exact(std::uint64_t at, std::byte* dst, std::size_t bytes) const {
        std::size_t done = 0;
        while(done < bytes) {
#if defined(_WIN32)
            OVERLAPPED where{};
            const std::uint64_t pos = at + done;
            where.Offset = static_cast<DWORD>(pos & 0xffffffffu);
            where.OffsetHigh = static_cast<DWORD>(pos >> 32);
            DWORD got = 0;
            const DWORD want = static_cast<DWORD>(std::min<std::size_t>(bytes - done, 1u << 30));
            if(!ReadFile(handle_, dst + done, want, &got, &where)) fail("read");
#else
            const ssize_t got = ::pread(fd_, dst + done, bytes - done, static_cast<off_t>(at + done));
            if(got < 0) {
                if(errno == EINTR) continue;
                fail("read");
            }
#endif
            if(got == 0) throw std::runtime_error("table_rows: short read at byte " + std::to_string(at + done));
            done += static_cast<std::size_t>(got);
        }
    }

private:
    [[noreturn]] static void fail(const std::string& what) {
#if defined(_WIN32)
        throw std::system_error(int(GetLastError()), std::system_category(), "table_rows: " + what);
#else
        throw std::system_error(errno, std::generic_category(), "table_rows: " + what);
#endif
    }
#if defined(_WIN32)
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
    std::uint64_t size_ = 0;
};
} // namespace detail

// Rows of one Region. workers: parallel reads for the rows a gather misses; cache_rows: hot rows kept in host
// memory, least recently used out (charge them to host memory admission). One gather at a time.
class RowGather {
public:
    RowGather(Region region, int workers = 8, std::size_t cache_rows = 0)
        : region_(std::move(region)), file_(region_.path), cache_rows_(cache_rows) {
        if(!region_.rows || !region_.row_bytes || workers < 1)
            throw std::invalid_argument("table_rows: a table needs rows, a row size and at least one worker");
        const std::uint64_t end = region_.offset + region_.rows * region_.row_bytes;
        if(file_.size() < end) throw std::invalid_argument("table_rows: " + region_.path + " is shorter than the table");
        advice_ = file_.advise_random(region_.offset, end - region_.offset);
        for(int w = 1; w < workers; ++w) pool_.emplace_back([this] { serve(); });
    }
    RowGather(const RowGather&) = delete;
    RowGather& operator=(const RowGather&) = delete;
    ~RowGather() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for(auto& t : pool_) t.join();
    }

    const char* advice() const noexcept { return advice_; }
    const Counters& counters() const noexcept { return counters_; }
    std::size_t row_bytes() const noexcept { return region_.row_bytes; }
    std::uint64_t rows() const noexcept { return region_.rows; }

    // Writes row ids[i] at dst + i * row_bytes, for every i; dst holds ids.size() rows.
    void gather(std::span<const std::uint64_t> ids, std::byte* dst) {
        const std::size_t n = region_.row_bytes;
        for(const auto r : ids)
            if(r >= region_.rows) throw std::out_of_range("table_rows: row outside the table");
        // First occurrence of each row: cached rows are copied now, the rest read into their first slot.
        std::unordered_map<std::uint64_t, std::size_t> first;
        std::vector<std::pair<std::uint64_t, std::byte*>> misses;
        for(std::size_t i = 0; i < ids.size(); ++i) {
            if(!first.emplace(ids[i], i).second) continue;
            auto hit = cache_index_.find(ids[i]);
            if(hit != cache_index_.end()) {
                cache_.splice(cache_.begin(), cache_, hit->second);
                std::memcpy(dst + i * n, hit->second->second.data(), n);
                ++counters_.hits;
            } else {
                misses.emplace_back(ids[i], dst + i * n);
            }
        }
        read_all(misses);
        for(const auto& [row, at] : misses) {
            ++counters_.reads;
            counters_.bytes_read += n;
            remember(row, at);
        }
        for(std::size_t i = 0; i < ids.size(); ++i) {
            const std::size_t j = first[ids[i]];
            if(j != i) std::memcpy(dst + i * n, dst + j * n, n);
        }
    }

private:
    void read_one(std::uint64_t row, std::byte* at) const {
        file_.read_exact(region_.offset + row * region_.row_bytes, at, region_.row_bytes);
    }
    void read_all(const std::vector<std::pair<std::uint64_t, std::byte*>>& misses) {
        if(pool_.empty() || misses.size() < 2) {
            for(const auto& [row, at] : misses) read_one(row, at);
            return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        jobs_ = &misses;
        next_ = 0;
        left_ = misses.size();
        failure_ = nullptr;
        lock.unlock();
        wake_.notify_all();
        work();   // the calling thread reads too
        lock.lock();
        done_.wait(lock, [this] { return left_ == 0; });
        jobs_ = nullptr;
        if(failure_) std::rethrow_exception(failure_);
    }
    // Takes rows until the batch is out; returns once it has no more to take.
    void work() {
        for(;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            if(!jobs_ || next_ >= jobs_->size()) return;
            const auto job = (*jobs_)[next_++];
            lock.unlock();
            std::exception_ptr error;
            try {
                read_one(job.first, job.second);
            } catch(...) {
                error = std::current_exception();
            }
            lock.lock();
            if(error && !failure_) failure_ = error;
            if(--left_ == 0) done_.notify_all();
        }
    }
    void serve() {
        for(;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] { return stop_ || (jobs_ && next_ < jobs_->size()); });
                if(stop_) return;
            }
            work();
        }
    }
    void remember(std::uint64_t row, const std::byte* at) {
        if(!cache_rows_) return;
        cache_.emplace_front(row, std::vector<std::byte>(at, at + region_.row_bytes));
        cache_index_[row] = cache_.begin();
        if(cache_.size() > cache_rows_) {
            cache_index_.erase(cache_.back().first);
            cache_.pop_back();
        }
    }

    Region region_;
    detail::RandomFile file_;
    const char* advice_ = "none";
    std::size_t cache_rows_ = 0;
    std::list<std::pair<std::uint64_t, std::vector<std::byte>>> cache_;
    std::unordered_map<std::uint64_t, std::list<std::pair<std::uint64_t, std::vector<std::byte>>>::iterator> cache_index_;
    Counters counters_;
    std::vector<std::thread> pool_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::vector<std::pair<std::uint64_t, std::byte*>>* jobs_ = nullptr;
    std::size_t next_ = 0, left_ = 0;
    std::exception_ptr failure_;
    bool stop_ = false;
};

// Slots of pinned host memory a gather writes and the engine copies to the device, in turn. acquire() returns the next
// slot once the copy that last read it has finished (`finished(slot)`, for example cudaEventQuery on the event the
// engine recorded after that copy, cudaSuccess meaning done); release() is implicit in that predicate.
class StagingRing {
public:
    StagingRing(std::byte* base, std::size_t slot_bytes, int slots, std::function<bool(int)> finished)
        : base_(base), slot_bytes_(slot_bytes), slots_(slots), finished_(std::move(finished)), used_(std::size_t(slots), false) {
        if(!base_ || !slot_bytes_ || slots_ < 1 || !finished_)
            throw std::invalid_argument("table_rows: a staging ring needs memory, a slot size, slots and a finished test");
    }
    // The next slot and its memory; waits (yielding) until the copy that read it last has finished.
    std::pair<int, std::span<std::byte>> acquire() {
        const int slot = next_;
        next_ = (next_ + 1) % slots_;
        if(used_[std::size_t(slot)])
            while(!finished_(slot)) std::this_thread::yield();
        used_[std::size_t(slot)] = true;
        return {slot, std::span<std::byte>(base_ + std::size_t(slot) * slot_bytes_, slot_bytes_)};
    }
    std::size_t slot_bytes() const noexcept { return slot_bytes_; }

private:
    std::byte* base_;
    std::size_t slot_bytes_;
    int slots_, next_ = 0;
    std::function<bool(int)> finished_;
    std::vector<bool> used_;
};

// Draft-time prefetch: start a token's rows the moment the token exists (a draft step's output, a sampled token) and let
// the step that reads them (a verify pass) wait only for rows still in flight. A service thread takes every queued
// request at once and gathers their rows in one batch, so the rows of consecutive draft tokens share the disk's queue
// depth instead of queueing behind each other. Requests finish in order. The RowGather is this prefetcher's alone
// while it lives (one gather at a time).
class RowPrefetch {
public:
    using Ticket = std::uint64_t;
    struct Stats {
        std::uint64_t requests = 0, batches = 0, rows = 0;
        std::uint64_t ready_when_waited = 0;   // waits that found their rows already there
        double waited_ms = 0.0;                // time callers spent blocked in wait()
    };

    explicit RowPrefetch(RowGather& gather) : gather_(gather), thread_([this] { serve(); }) {}
    RowPrefetch(const RowPrefetch&) = delete;
    RowPrefetch& operator=(const RowPrefetch&) = delete;
    ~RowPrefetch() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }

    // Rows ids[i] are written at dst + i * row_bytes, after this returns; dst must stay valid until wait(ticket)
    // returns. Rows outside the table throw here, not later.
    Ticket start(std::span<const std::uint64_t> ids, std::byte* dst) {
        for(const auto r : ids)
            if(r >= gather_.rows()) throw std::out_of_range("table_rows: row outside the table");
        const std::lock_guard<std::mutex> lock(mutex_);
        const Ticket t = ++issued_;
        queue_.push_back({t, std::vector<std::uint64_t>(ids.begin(), ids.end()), dst});
        stats_.requests += 1;
        stats_.rows += ids.size();
        wake_.notify_all();
        return t;
    }
    bool ready(Ticket t) const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return t <= finished_;
    }
    // Blocks until ticket t's rows are written; rethrows its batch's read failure.
    void wait(Ticket t) {
        const auto start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        if(t > issued_) throw std::invalid_argument("table_rows: no such prefetch");
        if(t <= finished_) stats_.ready_when_waited += 1;
        done_.wait(lock, [&] { return t <= finished_; });
        stats_.waited_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        auto failed = failures_.find(t);
        if(failed != failures_.end()) {
            const auto error = failed->second;
            failures_.erase(failed);
            std::rethrow_exception(error);
        }
    }
    void wait_all() { wait(last_issued()); }
    Ticket last_issued() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return issued_;
    }
    Stats stats() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    struct Request {
        Ticket ticket;
        std::vector<std::uint64_t> ids;
        std::byte* dst;
    };
    void serve() {
        std::vector<Request> batch;
        std::vector<std::uint64_t> ids;
        std::vector<std::byte> rows;
        for(;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if(queue_.empty()) return;   // stopping with nothing left
                batch.swap(queue_);
                stats_.batches += 1;
            }
            // One gather for every queued request: row ids concatenated, then each request's rows copied out.
            ids.clear();
            for(const auto& r : batch) ids.insert(ids.end(), r.ids.begin(), r.ids.end());
            const std::size_t n = gather_.row_bytes();
            rows.resize(ids.size() * n);
            std::exception_ptr error;
            try {
                gather_.gather(ids, rows.data());
                std::size_t at = 0;
                for(const auto& r : batch) {
                    if(!r.ids.empty()) std::memcpy(r.dst, rows.data() + at * n, r.ids.size() * n);
                    at += r.ids.size();
                }
            } catch(...) {
                error = std::current_exception();
            }
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                for(const auto& r : batch) {
                    if(error) failures_[r.ticket] = error;
                    finished_ = std::max(finished_, r.ticket);
                }
            }
            done_.notify_all();
            batch.clear();
        }
    }

    RowGather& gather_;
    mutable std::mutex mutex_;
    std::condition_variable wake_, done_;
    std::vector<Request> queue_;
    std::unordered_map<Ticket, std::exception_ptr> failures_;
    Ticket issued_ = 0, finished_ = 0;
    Stats stats_;
    bool stop_ = false;
    std::thread thread_;
};

} // namespace crit::techniques::table_rows
