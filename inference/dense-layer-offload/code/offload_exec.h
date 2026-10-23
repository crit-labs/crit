// Running a layer placement (offload_plan.h): a two-slot weight streamer that uploads streamed layers on a copy stream
// ahead of their use, an executor that walks the layers of a step in order (device layers on the compute stream,
// streamed layers from a staging slot, runs of host layers after copying the activations down), and measurements of
// the rates the planner needs on this machine.
#pragma once
#include "host_dense.h"
#include "offload_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace crit::dense_offload {

// Two device slots of slot_bytes. upload() copies a layer's pinned host weights into the next slot on `copy` once
// the slot's previous reader has released it; acquire() makes the compute stream wait for that upload; release()
// marks the slot free when the compute stream reaches it. Events order everything; nothing blocks the host.
class WeightStream {
public:
    WeightStream(std::size_t slot_bytes, cudaStream_t copy);
    ~WeightStream();
    WeightStream(const WeightStream&) = delete;
    WeightStream& operator=(const WeightStream&) = delete;
    int upload(const void* host, std::size_t bytes);
    void* acquire(int slot, cudaStream_t compute);
    void release(int slot, cudaStream_t compute);
    std::size_t slot_bytes() const noexcept { return slot_bytes_; }
private:
    std::size_t slot_bytes_;
    cudaStream_t copy_;
    void* slots_[2] = {nullptr, nullptr};
    cudaEvent_t ready_[2] = {}, free_[2] = {};
    int next_ = 0;
};

// The engine's layer implementations over its own activations and KV.
class LayerRunner {
public:
    virtual ~LayerRunner() = default;
    virtual void device_layer(int layer, const void* weights, cudaStream_t compute) = 0;  // weights: the layer's device bytes
    virtual void host_layer(int layer) = 0;                                              // on the host activations
    virtual void to_host(cudaStream_t compute) = 0;                                      // enqueue activations device -> host
    virtual void to_device(cudaStream_t compute) = 0;                                    // enqueue activations host -> device
};

// Where each layer's weights are: device[i] for resident layers, host[i] (pinned) for streamed and host layers.
struct LayerStore {
    std::vector<const void*> device, host;
    std::vector<std::size_t> bytes;
};

// One step's layers under `plan`. Streamed layers are uploaded two ahead: the upload of streamed layer k + 2 waits for
// layer k to release its slot and overlaps the layers in between. Before a run of host layers the activations are
// copied down and the compute stream synchronized; after it they are copied back.
void run_layers(const Plan& plan, const LayerStore& store, LayerRunner& runner, WeightStream* stream, cudaStream_t compute);

// Measured rates for Measurements (GB/s = bytes per ns), each the median of `repeats` separately timed runs: pinned
// host-to-device copies of `bytes`; the host GEMV on a rows x cols BF16 matrix for `batch` activation rows; the mean
// of activation round trips of `bytes` with a synchronization.
double measure_upload_gbs(std::size_t bytes, int repeats);
double measure_host_gemv_gbs(ThreadPool&, Isa, std::size_t rows, std::size_t cols, int batch, int repeats);
double measure_round_trip_ns(std::size_t bytes, int repeats);

}  // namespace crit::dense_offload
