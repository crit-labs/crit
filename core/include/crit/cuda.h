#pragma once
#include "crit/core.h"
#include <cuda_runtime_api.h>
#include <optional>

namespace crit::core {
enum class Queue { compute, transfer };
enum class SyncSchedule { spin, blocking, yield, automatic };
struct CudaState;

// Shared allocation aliases charge once. The last owner settles both backend
// streams before freeing, including during failed partial model construction.
// Raw views may be used only on those streams and must not outlive their owner.
class CudaBuffer {
public:
    CudaBuffer() = default;
    void* data() const noexcept;
    std::size_t size() const noexcept;
    Memory memory() const;
    // The address kernels use: data() for device memory, and the device mapping of pinned host memory, which
    // kernels can read in place (for example a gathered table that does not have to occupy device memory).
    void* device_address() const;
    // Loading only, never per-step work: both synchronize the stream. Pageable
    // transfers use owned pinned staging, so a failed fence never leaves GPU work
    // using the borrowed host span. In work loops use copy_async/readback_async
    // with owned pinned buffers. The caller orders earlier uses on the other stream.
    void upload(std::span<const std::byte>, std::size_t offset = 0, Queue = Queue::transfer);
    void download(std::span<std::byte>, std::size_t offset = 0, Queue = Queue::transfer) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit CudaBuffer(std::shared_ptr<Impl>);
    static CudaBuffer create(std::shared_ptr<CudaState>, Memory, std::size_t);
    friend class CudaBackend;
};

// Events order work without a host fence. wait() is the explicit result fence.
// Recording/querying before record and ordering foreign streams are rejected.
// Event operations require exclusive object access; backend owners may be shared.
class CudaEvent final : public Readiness {
public:
    CudaEvent() noexcept;
    ~CudaEvent();
    CudaEvent(CudaEvent&&) noexcept;
    CudaEvent& operator=(CudaEvent&&) noexcept;
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    void record(cudaStream_t);
    void order(cudaStream_t) const;
    bool ready() const;
    void wait() override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit CudaEvent(std::unique_ptr<Impl>);
    friend class CudaBackend;
    friend class CudaTimer;
};
// Fixed-capacity stage timings on the compute stream, with preallocated events
// and no per-step allocation. Time around graph launches: recording inside a
// stream capture is rejected. collect() requires completed work and never waits.
class CudaStageTimer {
public:
    CudaStageTimer() noexcept;
    ~CudaStageTimer();
    CudaStageTimer(CudaStageTimer&&) noexcept;
    CudaStageTimer& operator=(CudaStageTimer&&) noexcept;
    void begin(DeviceStage, std::uint32_t index = 0);
    void end();
    void reset();
    std::span<const DeviceTiming> collect();
    std::size_t capacity() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit CudaStageTimer(std::unique_ptr<Impl>);
    friend class CudaBackend;
};
class CudaTimer {
public:
    void start();
    void record_stop();
    float elapsed_ms() const; // Explicitly wait on stop before asking for elapsed.
    float stop_ms();
private:
    CudaEvent start_, stop_;
    cudaStream_t stream_;
    CudaTimer(CudaEvent, CudaEvent, cudaStream_t);
    friend class CudaBackend;
};

// One backend belongs to one engine/ledger. Buffers and events retain the CUDA
// state even after this facade dies. Extensions must submit all work to its two
// streams; no device-global synchronization is imposed in the work loop.
class CudaBackend final : public Backend {
public:
    explicit CudaBackend(int device = 0, SyncSchedule = SyncSchedule::spin);
    ~CudaBackend() override;
    void initialize(Resources&) override;
    void bind_thread() override;
    void drain() override;
    int device() const noexcept;
    cudaDeviceProp properties() const;
    cudaStream_t stream(Queue = Queue::compute) const;
    SyncSchedule sync_schedule() const;
    CudaBuffer allocate(Memory, std::size_t bytes);
    // Device memory still usable for optional work: the smaller of the device's free memory and this engine's
    // remaining device budget, less margin (zero below it). An engine built for a required context uses what a
    // smaller context leaves for speedups such as faster weight layouts, draft copies or device-resident tables.
    std::size_t spare_device_bytes(std::size_t margin = 0) const;
    // A device allocation only when spare_device_bytes(margin) covers it; none otherwise. For optional speedups;
    // required state uses allocate().
    std::optional<CudaBuffer> allocate_spare(std::size_t bytes, std::size_t margin);
    CudaEvent event();
    CudaTimer timer(Queue = Queue::compute);
    // Owners must belong to this backend. Async calls do not add host fences;
    // last-owner release protects pending accesses. Zero-byte calls are valid.
    void copy_async(const CudaBuffer& dst, std::size_t dst_offset,
                    const CudaBuffer& src, std::size_t src_offset, std::size_t bytes,
                    Queue = Queue::transfer);
    void fill_async(const CudaBuffer&, int value = 0, Queue = Queue::compute);
    // Pinned-host destination, device source. Copies on the compute queue after
    // the kernels that produced the data, then records completion; no host fence.
    // Return &completion as Progress::readiness.
    void readback_async(const CudaBuffer& host_dst, std::size_t dst_offset,
                        const CudaBuffer& device_src, std::size_t src_offset, std::size_t bytes,
                        CudaEvent& completion);
    CudaStageTimer stage_timer(std::size_t capacity);
private:
    std::shared_ptr<CudaState> state_;
};
// Failed CUDA drains/free calls retain storage plus its reservation and backend.
// Diagnostic exceptional retention requires process recovery, not normal reuse.
std::size_t quarantined_cuda_allocations() noexcept;
std::size_t quarantined_cuda_backends() noexcept;
}
namespace crit::core {
// Device time by category without intermediate host synchronization: event pairs
// around categories of work on one stream, summed after the single wait in
// finish(), plus the host's issue time against the GPU span of the same work.
// Diagnostic only; each pair adds stream nodes, so keep it out of replayed graphs.
struct CudaCategoryTimes {
    std::vector<double> category_ms;  // indexed by category
    double gpu_span_ms = 0;           // device time from start() to finish()
    double host_issue_ms = 0;         // host time from start() until finish() was called
    double host_total_ms = 0;         // host time from start() until the device finished
    std::size_t dropped = 0;          // pairs not counted: capacity, nesting, unclosed, unknown category
    bool complete = false;            // no pair was dropped
};
class CudaCategoryTimer {
public:
    // Preallocates `capacity` event pairs on `device`; start() takes a stream of that device.
    CudaCategoryTimer(std::size_t capacity, int device);
    ~CudaCategoryTimer();
    CudaCategoryTimer(CudaCategoryTimer&&) noexcept;
    CudaCategoryTimer& operator=(CudaCategoryTimer&&) noexcept;
    void start(cudaStream_t);
    // Categories do not nest: begin() inside an open category drops both pairs.
    // A pair beyond capacity is dropped. Dropped pairs are counted, never thrown.
    void begin(std::uint32_t category);
    void end();
    // Records the span's end, waits for it once and sums every pair below `categories`.
    CudaCategoryTimes finish(std::uint32_t categories);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
