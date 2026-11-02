// The native, device-side form of metadata.py: per-request metadata rows (slot indices, lengths, block-table heads)
// kept in stable slots of a device table, updated by byte patches of the rows that changed. MetadataTable holds the
// rows on the host (stable slots, versions, lowest free slot reused first); snapshot() and diff() give what changed
// since the last upload; DeviceReplica stages the changed rows in pinned host memory and has the engine copy them
// into the device table, and uploads the gather order (the slots of this step's rows) only when it changed (Core's
// UploadShadow). The staging alternates between two halves, and a half is written again only after the copies that
// read it have finished (`finished(half)`, for example cudaEventQuery on an event recorded after them), so a step's
// patches never overwrite bytes an earlier copy is still reading. Backend-agnostic: the engine passes its copy.
#pragma once
#include <crit/device_memory.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace crit::techniques::metadata {

struct Key {
    std::uint64_t request = 0, generation = 0;
    auto operator<=>(const Key&) const = default;
};

struct Row {
    std::uint32_t slot = 0;
    Key key;
    std::uint64_t version = 0;
    std::vector<std::byte> payload;   // at most row_bytes
    bool operator==(const Row&) const = default;
};

struct Snapshot {
    std::uint64_t sequence = 0;
    std::vector<Row> rows;            // in gather order
};

struct Delta {
    std::vector<std::uint32_t> removed;
    std::vector<Row> changed;
    std::vector<std::uint32_t> order;   // the slots in gather order
};

class MetadataTable {
public:
    MetadataTable(std::uint32_t capacity, std::size_t row_bytes) : capacity_(capacity), row_bytes_(row_bytes) {
        if(!capacity || !row_bytes) throw std::invalid_argument("metadata: positive row and byte bounds required");
    }
    // The key's slot; an unchanged payload changes nothing, a new key takes the lowest free slot.
    std::uint32_t set(Key key, std::span<const std::byte> payload) {
        if(payload.size() > row_bytes_) throw std::invalid_argument("metadata: payload exceeds row bytes");
        auto it = rows_.find(key);
        if(it != rows_.end() && std::equal(payload.begin(), payload.end(), it->second.payload.begin(), it->second.payload.end()))
            return it->second.slot;
        std::uint32_t slot;
        if(it != rows_.end()) slot = it->second.slot;
        else if(!free_.empty()) slot = *free_.begin(), free_.erase(free_.begin());
        else if(next_ < capacity_) slot = next_++;
        else throw std::length_error("metadata: capacity exhausted");
        rows_[key] = Row{slot, key, ++version_, {payload.begin(), payload.end()}};
        return slot;
    }
    void release(Key key) {
        const auto it = rows_.find(key);
        if(it == rows_.end()) throw std::invalid_argument("metadata: unknown key");
        free_.insert(it->second.slot);
        rows_.erase(it);
    }
    // The rows of `keys` in that order (all rows by key when empty).
    Snapshot snapshot(std::span<const Key> keys = {}) {
        Snapshot out{++sequence_, {}};
        if(keys.empty())
            for(const auto& [k, r] : rows_) out.rows.push_back(r);
        else
            for(const auto& k : keys) {
                const auto it = rows_.find(k);
                if(it == rows_.end()) throw std::invalid_argument("metadata: unknown key in a snapshot");
                out.rows.push_back(it->second);
            }
        return out;
    }
    std::uint32_t capacity() const noexcept { return capacity_; }
    std::size_t row_bytes() const noexcept { return row_bytes_; }

private:
    std::uint32_t capacity_, next_ = 0;
    std::size_t row_bytes_;
    std::uint64_t version_ = 0, sequence_ = 0;
    std::map<Key, Row> rows_;
    std::set<std::uint32_t> free_;
};

// What changed from `before` to `after`: rows gone, rows new or with a new version, and the new gather order.
inline Delta diff(const Snapshot& before, const Snapshot& after) {
    if(after.sequence <= before.sequence) throw std::invalid_argument("metadata: snapshots out of order");
    std::map<std::uint32_t, const Row*> old, now;
    for(const auto& r : before.rows) old[r.slot] = &r;
    for(const auto& r : after.rows) now[r.slot] = &r;
    Delta d;
    for(const auto& [slot, r] : old)
        if(!now.count(slot)) d.removed.push_back(slot);
    for(const auto& r : after.rows) {
        const auto it = old.find(r.slot);
        if(it == old.end() || !(*it->second == r)) d.changed.push_back(r);
        d.order.push_back(r.slot);
    }
    return d;
}

// The device table: `rows` holds capacity * row_bytes bytes (row s at rows + s * row_bytes), `order` capacity uint32
// slots; both device memory the engine owns. pinned: 2 * half_bytes of pinned host memory for the staged patches.
// copy(dst, src, bytes) enqueues a host-to-device copy on the engine's stream; finished(half) says whether the copies
// that read staging half `half` (0 or 1) have completed; record(half) is called after a step's copies from `half`
// were enqueued (the engine records its event there).
class DeviceReplica {
public:
    using Copy = std::function<void(void* dst, const void* src, std::size_t bytes)>;
    DeviceReplica(void* rows, std::uint32_t* order, std::uint32_t capacity, std::size_t row_bytes, std::byte* pinned,
                  std::size_t half_bytes, Copy copy, std::function<bool(int)> finished, std::function<void(int)> record)
        : rows_(static_cast<std::byte*>(rows)), order_(order), capacity_(capacity), row_bytes_(row_bytes), pinned_(pinned),
          half_bytes_(half_bytes), copy_(std::move(copy)), finished_(std::move(finished)), record_(std::move(record)) {
        if(!rows_ || !order_ || !capacity_ || !row_bytes_ || !pinned_ || half_bytes_ < row_bytes_ + capacity_ * sizeof(std::uint32_t) ||
           !copy_ || !finished_ || !record_)
            throw std::invalid_argument("metadata: device table, staging of at least one row plus the order, and hooks");
    }

    // Uploads one delta: the changed rows (each one copy, from staging) and the order when it changed. Returns the
    // bytes copied.
    std::size_t apply(const Delta& d) {
        if(d.order.size() > capacity_) throw std::invalid_argument("metadata: more rows than the table holds");
        const int half = half_;
        half_ ^= 1;
        if(used_[half])
            while(!finished_(half)) std::this_thread::yield();
        std::byte* stage = pinned_ + std::size_t(half) * half_bytes_;
        std::size_t used = 0, copied = 0;
        auto flush_if_full = [&](std::size_t need) {
            if(used + need <= half_bytes_) return;
            record_(half);   // a patch larger than one half: wait for what this half already sent
            while(!finished_(half)) std::this_thread::yield();
            used = 0;
        };
        for(const auto& r : d.changed) {
            if(r.slot >= capacity_ || r.payload.size() > row_bytes_) throw std::invalid_argument("metadata: invalid patch");
            flush_if_full(row_bytes_);
            std::memset(stage + used, 0, row_bytes_);
            std::memcpy(stage + used, r.payload.data(), r.payload.size());
            copy_(rows_ + std::size_t(r.slot) * row_bytes_, stage + used, row_bytes_);
            used += row_bytes_;
            copied += row_bytes_;
        }
        const std::size_t order_bytes = d.order.size() * sizeof(std::uint32_t);
        flush_if_full(order_bytes);
        std::memcpy(stage + used, d.order.data(), order_bytes);
        const auto order_span = std::span<const std::byte>(stage + used, order_bytes);
        if(shadow_.upload(order_, order_span, [&](void* dst, std::span<const std::byte> src) { copy_(dst, src.data(), src.size()); }))
            copied += order_bytes;
        record_(half);
        used_[half] = true;
        return copied;
    }
    std::uint64_t order_uploads_skipped() const noexcept { return shadow_.skipped(); }

private:
    std::byte* rows_;
    std::uint32_t* order_;
    std::uint32_t capacity_;
    std::size_t row_bytes_;
    std::byte* pinned_;
    std::size_t half_bytes_;
    Copy copy_;
    std::function<bool(int)> finished_;
    std::function<void(int)> record_;
    core::UploadShadow shadow_;
    int half_ = 0;
    bool used_[2] = {false, false};
};

} // namespace crit::techniques::metadata
