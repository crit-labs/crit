// A synthetic dense model run under a layer placement: the decoder layers of the batched reference engine's synthetic
// transformer, each executable on the device (resident or streamed weights) or on the host (host_block), for one
// sequence. Embedding, final norm and head stay on the device. It is the composition the tests and the measurement
// tool use; an engine supplies its own model through LayerRunner.
#pragma once
#include "host_dense.h"
#include "offload_exec.h"
#include "offload_plan.h"
#include "paged_attention.h"
#include "synthetic_model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace crit::dense_offload {

class SyntheticOffloadModel final : public LayerRunner {
public:
    // BF16 weights from the shape's seed; context: most positions of the sequence. threads: host workers (0 = all).
    SyntheticOffloadModel(const batched_engine::Shape& shape, int context, cudaStream_t compute, unsigned threads = 0);
    ~SyntheticOffloadModel() override;
    SyntheticOffloadModel(const SyntheticOffloadModel&) = delete;
    SyntheticOffloadModel& operator=(const SyntheticOffloadModel&) = delete;

    std::size_t layer_bytes() const noexcept { return layer_bytes_; }
    int layers() const noexcept { return shape_.layers; }
    // Device copies of the plan's resident layers (others freed); pinned host copies serve streamed and host layers.
    const LayerStore& place(const Plan& plan);
    void reset() noexcept { pos_ = 0; }              // a new sequence
    // Runs `tokens` (a prompt chunk or one decode token, at most kMaxRows) and returns the last row's logits.
    std::vector<float> step(const Plan& plan, WeightStream* stream, std::span<const std::int32_t> tokens);
    static constexpr int kMaxRows = 64;

    void device_layer(int layer, const void* weights, cudaStream_t compute) override;
    void host_layer(int layer) override;
    void to_host(cudaStream_t compute) override;
    void to_device(cudaStream_t compute) override;

private:
    batched_engine::Shape shape_;
    int context_;
    cudaStream_t compute_;
    ThreadPool pool_;
    Isa isa_;
    std::size_t layer_bytes_ = 0, qkv_elems_ = 0, o_elems_ = 0, gate_up_elems_ = 0;
    std::vector<std::uint16_t*> host_weights_;       // pinned, one packed layer each: QKV | O | GateUp | Down
    std::vector<void*> resident_;                    // device copies of resident layers
    LayerStore store_;
    std::vector<std::vector<float>> host_k_, host_v_;
    std::vector<float> inv_freq_;
    float* host_x_ = nullptr;                        // pinned activations [kMaxRows][hidden]
    int pos_ = 0, rows_ = 0;
    struct Device;
    std::unique_ptr<Device> d_;
};

}  // namespace crit::dense_offload
