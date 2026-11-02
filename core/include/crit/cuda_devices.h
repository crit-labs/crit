#pragma once
#include "crit/cuda.h"
#include <memory>
#include <vector>

namespace crit::core {
// Several GPUs for one engine: one CudaBackend per member, each with its own two streams, allocations and resource
// ledger (initialize every member with the engine's Resources for that device), direct peer access between members
// where the hardware allows it, copies between members' buffers, events that order one member's stream after
// another's work, a barrier over all of them, and device-time categories per member. A device listed twice is two
// members on one device (its own streams and ledger), which splits one device's work the same way and lets the
// set's ordering be checked on a machine with one GPU.
//
// Nothing here waits on the host: copies, orders and barriers are stream work. Members' buffers stay usable only on
// their own member's streams, except as the source or destination of copy_async. Every call leaves the calling
// thread's current device as it found it: bind a member (member(i).bind_thread()) before launching into its streams.
class CudaDeviceSet {
public:
    // Enables peer access between every pair of distinct devices that supports it; throws on an invalid device.
    explicit CudaDeviceSet(std::vector<int> devices, SyncSchedule = SyncSchedule::spin);
    ~CudaDeviceSet();
    CudaDeviceSet(const CudaDeviceSet&) = delete;
    CudaDeviceSet& operator=(const CudaDeviceSet&) = delete;

    std::size_t size() const noexcept;
    CudaBackend& member(std::size_t index);
    int device(std::size_t index) const;
    // Whether member `reader`'s kernels can read member `owner`'s device memory in place: the same device, or peer
    // access enabled between their devices. A copy between members works either way.
    bool peer(std::size_t reader, std::size_t owner) const;

    // Member `waiter`'s `waiter_queue` waits for the work submitted to member `source`'s `source_queue` so far.
    void order(std::size_t waiter, Queue waiter_queue, std::size_t source, Queue source_queue = Queue::compute);
    // Copies `bytes` of device buffer `src` (member src_member's) into device buffer `dst` (member dst_member's) on
    // dst_member's `queue`, after the source member's compute work so far (the kernels that produced `src`). A peer
    // copy where the devices reach each other, else staged by the driver; no host fence. Zero bytes are valid.
    void copy_async(std::size_t dst_member, const CudaBuffer& dst, std::size_t dst_offset,
                    std::size_t src_member, const CudaBuffer& src, std::size_t src_offset, std::size_t bytes,
                    Queue queue = Queue::transfer);
    // Every member's `queue` waits for every member's `queue` work so far: the step between phases that each read
    // what the others wrote (an all-reduce's halves, a pipeline stage's hand-over).
    void barrier(Queue queue = Queue::compute);
    // Device time by category on member `index`'s device (CudaCategoryTimer, one per member).
    CudaCategoryTimer category_timer(std::size_t index, std::size_t capacity) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
