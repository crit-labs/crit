#include <crit/cuda.h>
#include "stream_binding.h"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

using namespace crit::core;
namespace {
void check(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct ForeignAdapter {
    static inline thread_local cudaStream_t slot = nullptr;
    static cudaStream_t current() noexcept { return slot; }
    static void select(cudaStream_t value) noexcept { slot = value; }
};
void foreign_copy(void* destination, const void* source, std::size_t bytes) {
    check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDefault, ForeignAdapter::current()));
}
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false;
    void hold() {
        std::unique_lock lock(mutex);
        entered = true; cv.notify_all();
        cv.wait_for(lock, std::chrono::seconds(10), [&] { return released; });
    }
    void observed() {
        std::unique_lock lock(mutex);
        require(cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "CUDA gate did not enter");
    }
    void release() { std::lock_guard lock(mutex); released = true; cv.notify_all(); }
};
void CUDART_CB gated(void* data) { static_cast<Gate*>(data)->hold(); }
struct Release {
    Gate& gate;
    cudaStream_t stream;
    ~Release() { gate.release(); cudaStreamSynchronize(stream); }
};
struct Fixture {
    Resources resources{1 << 20, 1 << 20};
    CudaBackend backend;
    Fixture() { backend.initialize(resources); }
};
void binding() {
    Fixture f;
    auto source = f.backend.allocate(Memory::host, 16);
    auto destination = f.backend.allocate(Memory::device, 16);
    auto result = f.backend.allocate(Memory::host, 16);
    const std::array<int, 4> expected{11, 23, 37, 41};
    std::memcpy(source.data(), expected.data(), 16);
    f.backend.fill_async(destination, 0);
    check(cudaStreamSynchronize(f.backend.stream()));
    Gate gate; Release release{gate, f.backend.stream(Queue::transfer)};
    ForeignAdapter::select(f.backend.stream(Queue::transfer));
    check(cudaLaunchHostFunc(f.backend.stream(Queue::transfer), gated, &gate));
    gate.observed();
    {
        crit::kernels::ScopedStreamBinding<ForeignAdapter> bound(f.backend.stream());
        require(ForeignAdapter::current() == f.backend.stream(), "Foreign slot was not bound");
        try {
            crit::kernels::ScopedStreamBinding<ForeignAdapter> nested(f.backend.stream(Queue::transfer));
            throw std::runtime_error("nested unwind");
        } catch (const std::runtime_error&) {}
        require(ForeignAdapter::current() == f.backend.stream(), "Nested unwind lost stream slot");
        foreign_copy(destination.data(), source.data(), 16);
        f.backend.copy_async(result, 0, destination, 0, 16, Queue::compute);
    }
    require(ForeignAdapter::current() == f.backend.stream(Queue::transfer), "Outer slot was not restored");
    auto receipt = f.backend.event(); receipt.record(f.backend.stream()); receipt.wait();
    // The other stream is still blocked. A launcher ignoring our binding would
    // leave zeros here while the compute receipt falsely appeared complete.
    require(std::memcmp(result.data(), expected.data(), 16) == 0, "Completion event did not cover foreign work");
    bool separate = false;
    std::thread other([&] { separate = ForeignAdapter::current() == nullptr; });
    other.join();
    require(separate, "Foreign slot was not per-thread");
    gate.release(); f.backend.drain();
    ForeignAdapter::select(nullptr);
}
void restore_ordering() {
    Fixture f;
    auto source = f.backend.allocate(Memory::host, 16);
    auto destination = f.backend.allocate(Memory::device, 16);
    auto result = f.backend.allocate(Memory::host, 16);
    const std::array<int, 4> expected{101, 103, 107, 109};
    std::memcpy(source.data(), expected.data(), 16);
    auto producer = f.backend.event(), receipt = f.backend.event();
    // Private pending transaction owns every borrowed source/destination.
    std::array retained{source, destination, result};
    Gate gate; Release release{gate, f.backend.stream(Queue::transfer)};
    check(cudaLaunchHostFunc(f.backend.stream(Queue::transfer), gated, &gate)); gate.observed();
    f.backend.fill_async(destination, 9, Queue::transfer);
    producer.record(f.backend.stream(Queue::transfer));
    producer.order(f.backend.stream());
    {
        crit::kernels::ScopedStreamBinding<ForeignAdapter> bound(f.backend.stream());
        foreign_copy(destination.data(), source.data(), 16);
        f.backend.copy_async(result, 0, destination, 0, 16, Queue::compute);
        receipt.record(f.backend.stream());
    }
    source = {}; destination = {}; result = {};
    require(f.resources.usage(Memory::device).live == 16 && f.resources.usage(Memory::host).live == 32,
            "Pending restore lost allocation owners");
    require(!receipt.ready(), "Restore ignored its destination producer");
    gate.release(); receipt.wait();
    require(std::memcmp(retained[2].data(), expected.data(), 16) == 0, "Prior producer overwrote restored destination");
    retained = {};
    require(f.resources.usage(Memory::device).live == 0 && f.resources.usage(Memory::host).live == 0,
            "Settled restore retained buffers");
}
}
int main() {
    try { binding(); restore_ordering(); std::cout << "Foreign CUDA slot, receipt and restore ordering passed\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
