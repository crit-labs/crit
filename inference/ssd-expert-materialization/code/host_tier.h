// The native host tier between an expert file and a device expert cache, and the read queue that fills it.
//
// HostTier is a ledger over fixed slots of pinned host memory the engine owns (a Core HostBlock registered with
// cuda_pin_host_block), one bundle per slot: find() serves a hit, admit() hands out a slot for an expert about to be
// filled (from the file, or from a device slot the cache is evicting), commit() publishes it once its bytes have
// landed, abandon() frees a slot whose fill failed. The least recently used committed slot goes first; an expert the
// current route selects (`protected`) is never evicted to make room, and when every slot is protected or still filling
// the admission is refused rather than half held. The Python HostExpertTier states the same policy.
//
// ReadQueue reads expert regions from the file into such slots with a pool of positioned readers (Core's ReadOnlyFile
// read slots): demand reads before read-ahead, a key already queued or reading is not queued twice, and each read
// reports through a Ticket the caller waits on before it commits the slot and uploads it.
#pragma once
#include <crit/file.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace crit::techniques::host_tier {

struct TierCounters {
    std::uint64_t hits = 0, misses = 0, admitted = 0, evictions = 0, refusals = 0;
};

class HostTier {
public:
    // slots of slot_bytes each at base (pinned host memory the engine owns and keeps alive).
    HostTier(std::byte* base, std::size_t slot_bytes, std::size_t slots)
        : base_(base), slot_bytes_(slot_bytes), state_(slots, State::free), key_(slots, 0) {
        if(!base_ || !slot_bytes_ || !slots) throw std::invalid_argument("host tier: memory, a slot size and slots");
        for(std::size_t s = 0; s < slots; ++s) free_.push_back(s);
    }

    // The committed slot holding `key`, most recently used now; nothing on a miss (or while it is still filling).
    std::optional<std::span<std::byte>> find(std::uint64_t key) {
        const auto it = where_.find(key);
        if(it == where_.end() || state_[it->second] != State::committed) {
            ++counters_.misses;
            return std::nullopt;
        }
        ++counters_.hits;
        touch(it->second);
        return slot(it->second);
    }

    // A slot to fill with `key`: a free one, else the least recently used committed one not in `protect`. Nothing
    // when the key is already held or filling, or when no slot can be taken (refused).
    std::optional<std::pair<std::size_t, std::span<std::byte>>> admit(std::uint64_t key,
                                                                      const std::unordered_set<std::uint64_t>& protect = {}) {
        if(where_.count(key)) return std::nullopt;
        std::size_t s;
        if(!free_.empty()) {
            s = free_.front();
            free_.pop_front();
        } else {
            auto victim = lru_.end();
            for(auto it = lru_.begin(); it != lru_.end(); ++it)   // front = least recently used
                if(!protect.count(key_[*it])) { victim = it; break; }
            if(victim == lru_.end()) {
                ++counters_.refusals;
                return std::nullopt;
            }
            s = *victim;
            lru_.erase(victim);
            in_lru_.erase(s);
            where_.erase(key_[s]);
            ++counters_.evictions;
        }
        state_[s] = State::filling;
        key_[s] = key;
        where_[key] = s;
        ++counters_.admitted;
        return std::make_pair(s, slot(s));
    }

    // The fill of slot `s` has landed: find() serves it from now on.
    void commit(std::size_t s) {
        check_filling(s);
        state_[s] = State::committed;
        touch(s);
    }
    // The fill of slot `s` failed: the slot is free again and the key not held.
    void abandon(std::size_t s) {
        check_filling(s);
        where_.erase(key_[s]);
        state_[s] = State::free;
        free_.push_back(s);
    }

    const TierCounters& counters() const noexcept { return counters_; }
    std::size_t slot_bytes() const noexcept { return slot_bytes_; }
    bool holds(std::uint64_t key) const { return where_.count(key) != 0; }

private:
    enum class State : std::uint8_t { free, filling, committed };
    std::span<std::byte> slot(std::size_t s) const { return {base_ + s * slot_bytes_, slot_bytes_}; }
    void check_filling(std::size_t s) const {
        if(s >= state_.size() || state_[s] != State::filling) throw std::logic_error("host tier: slot is not filling");
    }
    void touch(std::size_t s) {
        const auto it = in_lru_.find(s);
        if(it != in_lru_.end()) lru_.erase(it->second);
        lru_.push_back(s);
        in_lru_[s] = std::prev(lru_.end());
    }

    std::byte* base_;
    std::size_t slot_bytes_;
    std::vector<State> state_;
    std::vector<std::uint64_t> key_;
    std::deque<std::size_t> free_;
    std::list<std::size_t> lru_;
    std::unordered_map<std::size_t, std::list<std::size_t>::iterator> in_lru_;
    std::unordered_map<std::uint64_t, std::size_t> where_;
    TierCounters counters_;
};

// A queued read's completion: wait() returns once its bytes landed and rethrows a read error.
class Ticket {
public:
    void wait() const {
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->ready.wait(lock, [this] { return state_->done; });
        if(state_->error) std::rethrow_exception(state_->error);
    }
    bool done() const {
        const std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->done;
    }

private:
    friend class ReadQueue;
    struct State {
        std::mutex mutex;
        std::condition_variable ready;
        bool done = false;
        std::exception_ptr error;
    };
    std::shared_ptr<State> state_ = std::make_shared<State>();
};

struct Read {
    std::uint64_t key = 0, offset = 0;   // the expert's bytes at [offset, offset + dst.size()) of the file
    std::span<std::byte> dst;            // usually a HostTier slot
};

class ReadQueue {
public:
    // Sets the file's reader slots to `workers` (Core's ReadOnlyFile::set_readers) before its readers start.
    ReadQueue(std::shared_ptr<core::ReadOnlyFile> file, int workers) : file_(std::move(file)) {
        if(!file_ || workers < 1) throw std::invalid_argument("read queue: a file and at least one worker");
        file_->set_readers(std::size_t(workers));
        for(int w = 0; w < workers; ++w) pool_.emplace_back([this, w] { serve(std::size_t(w)); });
    }
    ReadQueue(const ReadQueue&) = delete;
    ReadQueue& operator=(const ReadQueue&) = delete;
    ~ReadQueue() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for(auto& t : pool_) t.join();
    }

    // Queues a read; demand reads go before every queued read-ahead. A key already queued or reading returns its
    // ticket (a demand for a queued read-ahead moves it to the front).
    Ticket submit(const Read& read, bool demand) {
        const std::lock_guard<std::mutex> lock(mutex_);
        const auto found = pending_.find(read.key);
        if(found != pending_.end()) {
            if(demand)
                for(auto it = ahead_.begin(); it != ahead_.end(); ++it)
                    if(it->first.key == read.key) {
                        demand_.push_back(*it);
                        ahead_.erase(it);
                        break;
                    }
            return found->second;
        }
        Ticket ticket;
        pending_[read.key] = ticket;
        (demand ? demand_ : ahead_).emplace_back(read, ticket);
        wake_.notify_one();
        return ticket;
    }

    std::size_t queued() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return demand_.size() + ahead_.size();
    }

private:
    void serve(std::size_t worker) {
        for(;;) {
            std::pair<Read, Ticket> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] { return stop_ || !demand_.empty() || !ahead_.empty(); });
                if(stop_ && demand_.empty() && ahead_.empty()) return;
                auto& from = demand_.empty() ? ahead_ : demand_;
                job = from.front();
                from.pop_front();
            }
            std::exception_ptr error;
            try {
                file_->read_exact(job.first.offset, job.first.dst, worker);
            } catch(...) {
                error = std::current_exception();
            }
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                pending_.erase(job.first.key);
            }
            {
                const std::lock_guard<std::mutex> lock(job.second.state_->mutex);
                job.second.state_->done = true;
                job.second.state_->error = error;
            }
            job.second.state_->ready.notify_all();
        }
    }

    std::shared_ptr<core::ReadOnlyFile> file_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::pair<Read, Ticket>> demand_, ahead_;
    std::unordered_map<std::uint64_t, Ticket> pending_;
    std::vector<std::thread> pool_;
    bool stop_ = false;
};

} // namespace crit::techniques::host_tier
