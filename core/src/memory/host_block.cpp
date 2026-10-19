#include "crit/host_block.h"

#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace crit::core {
namespace {
constexpr std::size_t kPage = 4096;
}

HostBlock::HostBlock(std::size_t bytes) : bytes_(bytes) {
    if(!bytes) throw std::invalid_argument("HostBlock needs a size");
#ifdef _WIN32
    void* p = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if(!p) throw std::runtime_error("HostBlock: reserving " + std::to_string(bytes) + " bytes failed");
#else
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if(p == MAP_FAILED) throw std::runtime_error("HostBlock: reserving " + std::to_string(bytes) + " bytes failed");
#endif
    data_ = static_cast<std::byte*>(p);
}

HostBlock::~HostBlock() {
    attachment_.reset();   // a device registration ends before the pages go away
#ifdef _WIN32
    if(data_) VirtualFree(data_, 0, MEM_RELEASE);
#else
    if(data_) munmap(data_, bytes_);
#endif
}

void HostBlock::touch(int threads) const {
    const std::size_t pages = (bytes_ + kPage - 1) / kPage;
    if(threads < 1) threads = 1;
    std::atomic<unsigned long long> sink{0};
    auto work = [&](int i) {
        unsigned long long acc = 0;
        const auto* p = reinterpret_cast<const volatile unsigned char*>(data_);
        for(std::size_t k = pages * i / threads; k < pages * (i + 1) / threads; ++k) acc += p[k * kPage];
        sink.fetch_add(acc, std::memory_order_relaxed);
    };
    std::vector<std::thread> pool;
    for(int i = 1; i < threads; ++i) pool.emplace_back(work, i);
    work(0);
    for(auto& t : pool) t.join();
}

HostBlockPool& HostBlockPool::process() {
    static HostBlockPool pool;
    return pool;
}

std::shared_ptr<HostBlock> HostBlockPool::acquire(const std::string& key, std::size_t bytes, bool keep, bool* reused,
                                                  const std::string& identity) {
    std::shared_ptr<HostBlock> old;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = blocks_.find(key);
        if(it != blocks_.end()) {
            if(it->second.block->size() == bytes && it->second.identity == identity) {
                if(reused) *reused = true;
                return it->second.block;
            }
            old = std::move(it->second.block);
            blocks_.erase(it);
        }
    }
    old.reset();   // freed here unless a live engine still holds it
    if(reused) *reused = false;
    auto block = std::make_shared<HostBlock>(bytes);
    if(keep) {
        std::lock_guard<std::mutex> lock(mutex_);
        blocks_[key] = Entry{identity, block};
    }
    return block;
}

void HostBlockPool::release(const std::string& key) {
    std::shared_ptr<HostBlock> old;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = blocks_.find(key);
        if(it == blocks_.end()) return;
        old = std::move(it->second.block);
        blocks_.erase(it);
    }
}

void HostBlockPool::clear() {
    std::map<std::string, Entry> old;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        old.swap(blocks_);
    }
}

std::size_t HostBlockPool::cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t total = 0;
    for(const auto& [key, entry] : blocks_) total += entry.block->size();
    return total;
}
}
