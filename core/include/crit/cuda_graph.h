#pragma once
#include "crit/cuda.h"
#include "crit/device_graph.h"
#include <functional>
#include <memory>
#include <span>
#include <vector>

// Optional CUDA graph capture harness: owned input/output staging, captured phases with eager bridges between
// them, ordered prerequisites and complete-owner retention on failure. The host protocol is crit/device_graph.h.
namespace crit::core {
struct MatrixView {
    CudaBuffer owner;
    MatrixLayout layout;
};
struct StagedBuffers {
    CudaBuffer input, output;
    MatrixLayout input_layout, output_layout; // Contiguous, padded rows.
    std::size_t active_rows;
};
// Each callback submits only to the supplied compute stream. Capture callbacks must be capture-safe; initialize
// libraries beforehand. Own every additional weight/workspace/library handle in the closure or retained_buffers.
using GraphStage = std::function<void(cudaStream_t, const StagedBuffers&)>;

class CudaStagedGraph {
public:
    CudaStagedGraph(std::shared_ptr<CudaBackend>, MatrixLayout input, MatrixLayout output, std::size_t padded_rows,
                    std::vector<GraphStage> phases, std::vector<GraphStage> bridges = {},
                    std::vector<CudaBuffer> retained_buffers = {});
    ~CudaStagedGraph();
    CudaStagedGraph(const CudaStagedGraph&) = delete;
    CudaStagedGraph& operator=(const CudaStagedGraph&) = delete;
    CudaStagedGraph(CudaStagedGraph&&) noexcept;
    CudaStagedGraph& operator=(CudaStagedGraph&&) noexcept;
    // Pure layout query, independent of health/readiness. Different layouts use another slot or an eager path.
    // A failed replay still throws on a matching layout.
    bool compatible(MatrixLayout input, MatrixLayout output) const noexcept;
    // One replay in flight. Retains external buffers and prerequisite events. Input producers on other streams
    // must supply recorded dependencies.
    void replay(const MatrixView& input, const MatrixView& output,
                std::span<const std::shared_ptr<CudaEvent>> prerequisites = {});
    bool ready();
    void wait();
    // Drain, destroy captures, release owners. A failed drain retains everything.
    void close();
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
// A failed destructor cannot prove settlement and retains the complete owner.
std::size_t quarantined_staged_graphs() noexcept;
}
