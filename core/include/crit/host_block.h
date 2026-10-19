#pragma once
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace crit::core {
// A large page-aligned host block for weights the device reads through a mapping (host-resident experts,
// embeddings). Fill it with the loader's threads first, then register it with the device (cuda_pin_host_block in
// crit/cuda_memory.h): first-touch page faults run in parallel and the registration is one call. Under WSL2,
// cudaMallocHost of 60 GB took about 90 s; anonymous pages filled by 12 threads and then registered took 35 s.
class HostBlock {
public:
    explicit HostBlock(std::size_t bytes);   // throws std::runtime_error when the pages cannot be reserved
    ~HostBlock();
    HostBlock(const HostBlock&) = delete;
    HostBlock& operator=(const HostBlock&) = delete;
    std::byte* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return bytes_; }
    // Reads one word of every page with `threads` threads. After a device registration under WSL2 the first CPU
    // read of each page was slow (about 10 s over 60 GB on the first request); touching once at load moves that cost.
    void touch(int threads) const;
    // Owner released before the pages (a device registration). One attachment; attaching again replaces it.
    void attach(std::shared_ptr<const void> owner) { attachment_ = std::move(owner); }
    bool attached() const noexcept { return attachment_ != nullptr; }

private:
    std::byte* data_ = nullptr;
    std::size_t bytes_ = 0;
    std::shared_ptr<const void> attachment_;
};

// Process-wide blocks kept between engines opened one after another in one process (crit Eval, crit run measure
// and crit serve switches), so the next engine on the same weights reuses them instead of reading and pinning
// tens of GB again. acquire() returns the cached block when the key, the size and the identity match; otherwise it
// drops the cached block for that key BEFORE the caller allocates, so a mismatched layout (an engine opened without a
// draft head, then one with it) never holds two copies. `identity` names what the bytes were filled from (for example
// the weight files' hashes or sizes and modification times): a different checkpoint of the same size under the same
// key never reuses stale bytes. A block still held by a live engine stays until that engine releases it; the pool
// never frees bytes someone else holds.
class HostBlockPool {
public:
    static HostBlockPool& process();
    // reused = true: the block holds the bytes the previous holder left (filled and pinned). Otherwise a new, unfilled
    // block, cached under `key` when keep is true.
    std::shared_ptr<HostBlock> acquire(const std::string& key, std::size_t bytes, bool keep, bool* reused = nullptr,
                                       const std::string& identity = {});
    void release(const std::string& key);
    void clear();
    std::size_t cached_bytes() const;

private:
    struct Entry { std::string identity; std::shared_ptr<HostBlock> block; };
    mutable std::mutex mutex_;
    std::map<std::string, Entry> blocks_;
};
}
