#include "crit/cuda_graph.h"
#include <atomic>
#include <stdexcept>
#include <string>
#include <utility>

namespace crit::core {
namespace {
std::atomic<std::size_t> staged_quarantine{0};
void check(cudaError_t status, const char* operation) {
    if(status != cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
void validate(const std::shared_ptr<CudaBackend>& backend, const std::vector<GraphStage>& phases,
              const std::vector<GraphStage>& bridges) {
    if(!backend) throw UnsupportedLayout("staged graph requires a CUDA backend");
    for(const auto& stage : phases) if(!stage) throw UnsupportedLayout("empty graph phase");
    for(const auto& bridge : bridges) if(!bridge) throw UnsupportedLayout("empty graph bridge");
}
}
struct CudaStagedGraph::Impl {
    std::shared_ptr<CudaBackend> backend;
    StagedGraphPlan plan;
    StagedGraphState state;
    StagedBuffers buffers;
    std::vector<GraphStage> phases, bridges;
    std::vector<CudaBuffer> retained, active;
    std::vector<std::shared_ptr<CudaEvent>> dependencies;
    std::unique_ptr<CudaEvent> completion;
    struct Capture { cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr; };
    std::vector<Capture> captures;
    Impl* retained_next = nullptr;

    Impl(std::shared_ptr<CudaBackend> b, StagedGraphPlan p, std::vector<GraphStage> stages,
         std::vector<GraphStage> eager, std::vector<CudaBuffer> owners)
        :backend(std::move(b)), plan(std::move(p)), phases(std::move(stages)), bridges(std::move(eager)),
         retained(std::move(owners)), captures(phases.size()) {}

    void initialize() {
        backend->bind_thread();
        buffers.input_layout = plan.staged_input();
        buffers.output_layout = plan.staged_output();
        buffers.active_rows = plan.input().rows;
        buffers.input = backend->allocate(Memory::device, matrix_bytes(buffers.input_layout));
        buffers.output = backend->allocate(Memory::device, matrix_bytes(buffers.output_layout));
        const auto stream = backend->stream();
        state.capture(phases.size(), [&](std::size_t i) {
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
            try { phases[i](stream, buffers); }
            catch(...) {
                // End invalidated captures as well: an exception must not leave the shared stream capturing.
                // Keep any returned graph for destruction after the drain.
                cudaStreamEndCapture(stream, &captures[i].graph);
                throw;
            }
            check(cudaStreamEndCapture(stream, &captures[i].graph), "end graph capture");
            check(cudaGraphInstantiate(&captures[i].executable, captures[i].graph, 0), "instantiate graph");
        });
    }
    void destroy_captures() {
        backend->bind_thread();
        backend->drain();
        for(auto& capture : captures) {
            if(capture.executable) {
                check(cudaGraphExecDestroy(capture.executable), "destroy graph executable");
                capture.executable = nullptr;
            }
            if(capture.graph) {
                check(cudaGraphDestroy(capture.graph), "destroy graph");
                capture.graph = nullptr;
            }
        }
    }
    bool settle_destroy() noexcept { return state.close([&] { destroy_captures(); }); }
    static void release(Impl* owner) noexcept {
        if(owner->settle_destroy()) { delete owner; return; }
        // Intrusive retention needs no allocation and keeps library closures, external replay buffers,
        // captures, events and the backend facade.
        static std::atomic<Impl*> head{nullptr};
        auto previous = head.load(std::memory_order_relaxed);
        do { owner->retained_next = previous; }
        while(!head.compare_exchange_weak(previous, owner, std::memory_order_release, std::memory_order_relaxed));
        staged_quarantine.fetch_add(1, std::memory_order_relaxed);
    }
    void settled() { completion.reset(); dependencies.clear(); active.clear(); }
    bool ready() { return state.ready([&] { return completion->ready(); }, [&] { settled(); }); }
};
CudaStagedGraph::CudaStagedGraph(std::shared_ptr<CudaBackend> backend, MatrixLayout input, MatrixLayout output,
                                 std::size_t padded_rows, std::vector<GraphStage> phases,
                                 std::vector<GraphStage> bridges, std::vector<CudaBuffer> retained_buffers) {
    StagedGraphPlan plan(input, output, padded_rows, phases.size(), bridges.size());
    validate(backend, phases, bridges);
    impl_ = std::shared_ptr<Impl>(new Impl(std::move(backend), std::move(plan), std::move(phases), std::move(bridges),
                                           std::move(retained_buffers)), Impl::release);
    impl_->initialize();
}
CudaStagedGraph::~CudaStagedGraph() = default;
CudaStagedGraph::CudaStagedGraph(CudaStagedGraph&&) noexcept = default;
CudaStagedGraph& CudaStagedGraph::operator=(CudaStagedGraph&&) noexcept = default;
bool CudaStagedGraph::compatible(MatrixLayout input, MatrixLayout output) const noexcept {
    return impl_ && !impl_->state.closed() && impl_->plan.compatible(input, output);
}
void CudaStagedGraph::replay(const MatrixView& input, const MatrixView& output,
                             std::span<const std::shared_ptr<CudaEvent>> prerequisites) {
    if(!impl_) throw std::logic_error("staged graph is empty");
    auto& p = *impl_;
    p.state.require_usable();
    p.plan.check(input.layout, input.owner.size(), output.layout, output.owner.size());
    if(p.state.pending() && !p.ready()) throw std::logic_error("staging still has in-flight readers");
    // Validate ownership before any work. Zero-length Core copies do not enqueue.
    p.backend->copy_async(p.buffers.input, 0, input.owner, 0, 0, Queue::compute);
    p.backend->copy_async(output.owner, 0, p.buffers.output, 0, 0, Queue::compute);
    for(const auto& event : prerequisites) if(!event) throw std::invalid_argument("null prerequisite event");
    p.active = {input.owner, output.owner};
    p.dependencies.assign(prerequisites.begin(), prerequisites.end());
    p.completion = std::make_unique<CudaEvent>(p.backend->event());
    const auto stream = p.backend->stream();
    p.state.replay(p.captures.size(),
        [&] {
            p.backend->bind_thread();
            for(const auto& event : p.dependencies) event->order(stream);
            p.backend->fill_async(p.buffers.input, 0);
            p.backend->fill_async(p.buffers.output, 0);
            for(const auto& row : p.plan.input_rows())
                p.backend->copy_async(p.buffers.input, row.destination, input.owner, row.source, row.bytes, Queue::compute);
        },
        [&](std::size_t i) { check(cudaGraphLaunch(p.captures[i].executable, stream), "launch graph phase"); },
        [&](std::size_t i) { p.bridges[i](stream, p.buffers); },
        [&] {
            for(const auto& row : p.plan.output_rows())
                p.backend->copy_async(output.owner, row.destination, p.buffers.output, row.source, row.bytes, Queue::compute);
            p.completion->record(stream);
        });
}
bool CudaStagedGraph::ready() {
    if(!impl_) throw std::logic_error("staged graph is empty");
    return impl_->ready();
}
void CudaStagedGraph::wait() {
    if(!impl_) throw std::logic_error("staged graph is empty");
    auto& p = *impl_;
    p.state.wait([&] { p.completion->wait(); }, [&] { p.settled(); });
}
void CudaStagedGraph::close() {
    if(!impl_) return;
    if(!impl_->settle_destroy()) throw std::runtime_error("staged graph cleanup failed; owners retained");
    impl_.reset();
}
std::size_t quarantined_staged_graphs() noexcept { return staged_quarantine.load(std::memory_order_relaxed); }
}
