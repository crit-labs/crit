// Core shared device allocation protocol on the host: leases charged once through Resources, settled final
// release with failure retention, and mapped-source resolution. Backend hooks are host fakes; no device call.
#include <crit/device_memory.h>
#include <crit/host_block.h>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace crit::core;
namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
template<class Error = std::exception, class F> void rejects(F action, const char* message) {
    try { action(); }
    catch(const Error& e) {
        require(std::string(e.what()).find(message) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error(std::string("expected rejection missing: ") + message);
}
struct FakeBackend final : LeaseBackend {
    std::vector<std::string> events;
    std::vector<std::unique_ptr<std::byte[]>> storage;
    bool fail_allocate = false, fail_settle = false, fail_release = false;
    void* allocate(std::size_t bytes) override {
        events.push_back("allocate " + std::to_string(bytes));
        if(fail_allocate) throw std::runtime_error("allocation failed");
        storage.push_back(std::make_unique<std::byte[]>(bytes + 1));
        return storage.back().get();
    }
    void settle() override {
        events.push_back("settle");
        if(fail_settle) throw std::runtime_error("unsettled");
    }
    void release(void*, std::size_t bytes) override {
        events.push_back("release " + std::to_string(bytes));
        if(fail_release) throw std::runtime_error("ambiguous release");
    }
};
struct Fixture {
    std::shared_ptr<FakeBackend> backend = std::make_shared<FakeBackend>();
    Resources resources{0, 8};
    BufferService service{backend, resources, Memory::device};
    std::size_t live() const { return resources.usage(Memory::device).live; }
};
using Events = std::vector<std::string>;
std::shared_ptr<const void> owner() { return std::make_shared<int>(0); }

void memory_leases() {
    Fixture f;
    auto first = f.service.allocate(8);
    auto alias = first.share();
    require(alias.data() == first.data() && alias.size() == 8, "a shared lease names the same allocation");
    first.close();
    require(f.live() == 8 && f.backend->events == Events{"allocate 8"}, "an earlier lease released the allocation");
    rejects<std::runtime_error>([&] { f.service.allocate(1); }, "exceeds");
    require(f.backend->events.size() == 1, "an over-budget allocation reached the backend");
    rejects<std::logic_error>([&] { first.share(); }, "closed");
    rejects<std::logic_error>([&] { static_cast<void>(first.data()); }, "closed");
    alias.close();
    alias.close();
    require(f.live() == 0 && f.backend->events == Events{"allocate 8", "settle", "release 8"},
            "the final lease must settle, release and drop the charge once");
    {
        auto zero = f.service.allocate(0);
        require(zero.size() == 0 && f.live() == 0, "zero-byte allocation");
    }
    require(f.backend->events.back() == "release 0", "destruction closes the lease");
    rejects<std::invalid_argument>([] { BufferService bad(nullptr, Resources(0, 1), Memory::device); }, "hooks");
}
void memory_lifetime() {
    Fixture f;
    BufferLease moved;
    {
        BufferService scoped(f.backend, f.resources, Memory::device);
        auto lease = scoped.allocate(4);
        moved = std::move(lease);
        require(lease.closed() && !moved.closed(), "move transfers the lease");
    }
    require(f.live() == 4, "a lease keeps its allocation after the service handle dies");
    moved = BufferLease{};
    require(f.live() == 0 && f.backend->events.back() == "release 4", "assignment closes the previous lease");
    f.backend->fail_allocate = true;
    rejects<std::runtime_error>([&] { f.service.allocate(4); }, "allocation failed");
    require(f.live() == 0, "a failed allocation kept its charge");
}
void memory_settle_failure() {
    Fixture f;
    auto lease = f.service.allocate(8);
    f.backend->fail_settle = true;
    rejects<std::runtime_error>([&] { lease.close(); }, "unsettled");
    require(lease.closed() && f.live() == 8 && f.service.retained() == 1 && !f.service.uncertain(),
            "a failed settle must keep the allocation and its charge");
    rejects<std::runtime_error>([&] { f.service.retry_retained(); }, "unsettled");
    f.backend->fail_settle = false;
    f.service.retry_retained();
    require(f.live() == 0 && !f.service.retained() && f.backend->events.back() == "release 8", "retry after settlement");
    {
        auto dropped = f.service.allocate(2);
        f.backend->fail_settle = true;
    }
    require(f.service.retained() == 1 && f.live() == 2, "a destructor kept a failed release retained");
    f.backend->fail_settle = false;
    f.service.retry_retained();
    require(f.live() == 0, "retained destructor release");
}
void memory_release_failure() {
    const auto before = quarantined_buffer_services();
    Resources resources(0, 8);
    {
        const auto backend = std::make_shared<FakeBackend>();
        BufferService service(backend, resources, Memory::device);
        auto lease = service.allocate(4);
        backend->fail_release = true;
        rejects<std::runtime_error>([&] { lease.close(); }, "ambiguous release");
        backend->fail_release = false;
        rejects<std::logic_error>([&] { service.retry_retained(); }, "cannot retry blindly");
        std::size_t releases = 0;
        for(const auto& event : backend->events) releases += event.rfind("release", 0) == 0;
        require(releases == 1 && resources.usage(Memory::device).live == 4 && service.uncertain() == 1,
                "an ambiguous release was retried");
    }
    require(quarantined_buffer_services() == before + 1 && resources.usage(Memory::device).live == 4,
            "a destroyed service released an uncertain allocation or its charge");
}
void memory_threads() {
    Fixture f;
    auto root = f.service.allocate(8);
    std::vector<std::thread> threads;
    std::atomic<int> failures{0};
    for(int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            try {
                for(int i = 0; i < 2000; ++i) {
                    auto lease = root.share();
                    if(lease.data() != root.data()) ++failures;
                }
            } catch(...) { ++failures; }
        });
    for(auto& thread : threads) thread.join();
    require(!failures && f.live() == 8 && f.backend->events.size() == 1, "concurrent leases released early");
    root.close();
    require(f.live() == 0 && f.backend->events.size() == 3, "concurrent leases final release");
}
void mapped_contiguous() {
    const auto first = owner(), second = owner();
    MappedSource source(3, {{1100, 8100, 100, second}, {1000, 8000, 100, first}});
    const auto resolved = source.resolve(1050, 100);
    require(resolved.device_address == 8050 && resolved.bytes == 100 && resolved.generation == 3, "contiguous alias");
    require(resolved.owners == std::vector<std::shared_ptr<const void>>{first, second}, "all contributing owners");
    require(source.spans()[0].host_address == 1000, "spans sorted by host address");
    MappedSource same(4, {{1000, 8000, 100, first}, {1100, 8100, 100, first}});
    require(same.resolve(1050, 100).owners.size() == 1, "owners are not repeated");
    require(same.resolve(1000, 200).device_address == 8000 && same.resolve(1199, 1).device_address == 8199, "edges");
}
void mapped_discontinuous() {
    MappedSource source(1, {{1000, 8000, 100, owner()}, {1100, 16000, 100, owner()}});
    rejects<std::invalid_argument>([&] { source.resolve(1050, 100); }, "noncontiguous device aliases");
    require(source.resolve(1100, 50).device_address == 16000 && source.generation() == 1,
            "a rejected extent changes no source state");
}
void mapped_rejects() {
    const auto kept = owner();
    MappedSource source(1, {{1000, 8000, 100, kept}, {1200, 8200, 100, kept}});
    for(const auto& [start, extent] : std::vector<std::pair<std::uintptr_t, std::size_t>>{
            {1050, 200}, {950, 100}, {1250, 100}, {1000, 0}, {0, 1}, {UINTPTR_MAX - 1, 4}})
        rejects<std::invalid_argument>([&] { source.resolve(start, extent); }, "");
    rejects<std::invalid_argument>([&] { MappedSource bad(1, {{1000, 8000, 100, kept}, {1050, 9000, 100, kept}}); },
                                   "overlapping");
    rejects<std::invalid_argument>([&] { MappedSource bad(0, {{1000, 8000, 100, kept}}); }, "generation");
    rejects<std::invalid_argument>([&] { MappedSource bad(1, {}); }, "generation");
    rejects<std::invalid_argument>([&] { MappedSource bad(1, {{1000, 8000, 100, nullptr}}); }, "owners");
    rejects<std::invalid_argument>([&] { MappedSource bad(1, {{1000, 0, 100, kept}}); }, "positive");
    rejects<std::invalid_argument>([&] { MappedSource bad(1, {{UINTPTR_MAX - 10, 8000, 100, kept}}); }, "address space");
}
void mapped_generations() {
    const auto old_owner = owner(), new_owner = owner();
    const auto old = MappedSource(1, {{1000, 8000, 100, old_owner}}).resolve(1010, 20);
    const auto fresh = MappedSource(2, {{1000, 16000, 100, new_owner}}).resolve(1010, 20);
    require(old.device_address == 8010 && old.generation == 1 && old.owners[0] == old_owner, "retained old source");
    require(fresh.device_address == 16010 && fresh.generation == 2 && fresh.owners[0] == new_owner, "new generation");
}
}

// HostBlock: page-aligned, writable, touched; the pool reuses a matching block and drops a mismatched one before
// the caller allocates, never freeing a block a live holder still uses.
void host_block_pool() {
    HostBlock block(3 * 4096 + 17);
    require(block.size() == 3 * 4096 + 17 && reinterpret_cast<std::uintptr_t>(block.data()) % 4096 == 0, "page aligned");
    block.data()[block.size() - 1] = std::byte{7};
    block.touch(3);
    require(!block.attached(), "fresh block attached");
    bool destroyed = false;
    block.attach(std::shared_ptr<const void>(&destroyed, [](const void* p) { *static_cast<bool*>(const_cast<void*>(p)) = true; }));
    require(block.attached(), "attachment");
    {
        HostBlock scoped(4096);
        scoped.attach(std::shared_ptr<const void>(&destroyed, [](const void* p) { *static_cast<bool*>(const_cast<void*>(p)) = true; }));
    }
    require(destroyed, "attachment released with the block");
    rejects<std::invalid_argument>([] { HostBlock empty(0); }, "needs a size");

    HostBlockPool pool;
    bool reused = true;
    auto a = pool.acquire("model/experts", 8192, true, &reused);
    require(!reused && pool.cached_bytes() == 8192, "first acquire allocates and caches");
    a->data()[0] = std::byte{42};
    auto b = pool.acquire("model/experts", 8192, true, &reused);
    require(reused && b == a && b->data()[0] == std::byte{42}, "matching size reuses the block");
    std::weak_ptr<HostBlock> old = a;
    a.reset();
    b.reset();
    auto c = pool.acquire("model/experts", 12288, true, &reused);
    require(!reused && old.expired() && c->size() == 12288 && pool.cached_bytes() == 12288,
            "a size mismatch frees the cached block before the new one");
    auto same = pool.acquire("model/experts", 12288, true, &reused, "");
    require(reused && same == c, "the default identity matches itself");
    same.reset();
    std::weak_ptr<HostBlock> stale = c;
    c.reset();
    c = pool.acquire("model/experts", 12288, true, &reused, "other-checkpoint");
    require(!reused && stale.expired() && pool.cached_bytes() == 12288,
            "another identity of the same size never reuses the cached bytes");
    auto held = pool.acquire("model/embed", 4096, false, &reused);
    require(!reused && pool.cached_bytes() == 12288, "keep=false does not cache");
    pool.release("model/experts");
    require(pool.cached_bytes() == 0 && c->size() == 12288, "release keeps bytes a holder still uses");
    auto d = pool.acquire("k", 4096, true);
    pool.clear();
    require(pool.cached_bytes() == 0 && d->size() == 4096, "clear");
}

int main(int argc, char** argv) {
    try {
        const std::string test = argc > 1 ? argv[1] : "all";
        const std::vector<std::pair<std::string, std::function<void()>>> tests{
            {"memory_leases", memory_leases}, {"memory_lifetime", memory_lifetime},
            {"memory_settle_failure", memory_settle_failure}, {"memory_release_failure", memory_release_failure},
            {"memory_threads", memory_threads}, {"mapped_contiguous", mapped_contiguous},
            {"mapped_discontinuous", mapped_discontinuous}, {"mapped_rejects", mapped_rejects},
            {"mapped_generations", mapped_generations}, {"host_block_pool", host_block_pool}};
        bool found = false;
        for(const auto& [name, run] : tests) if(test == "all" || test == name) {
            run();
            found = true;
            std::cout << "crit core device memory passed: " << name << '\n';
        }
        require(found, "unknown test");
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
