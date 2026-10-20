#pragma once
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Host protocol of captured device graphs: graph identity and settled lifetime, staged matrix geometry and the
// replay/failure state machine. It makes no device call; crit/cuda_graph.h supplies the CUDA capture harness.
namespace crit::core {

// Every assumption a captured graph depends on. Extend phase/representation with any other captured assumption.
struct GraphKey {
    std::string phase;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> rows; // Ordered request/lane generation pairs.
    std::vector<std::uintptr_t> addresses;                     // Physical bindings read or written by the graph.
    std::vector<std::vector<std::size_t>> shapes;
    std::string representation;
    std::uint64_t allocation_generation = 1;
    std::uint32_t device = 0;
    std::uint64_t stream_identity = 0;
    std::vector<std::uint64_t> binding_generations; // Empty, or one per address.
    // Throws std::invalid_argument for incomplete identity, nonpositive values or duplicate rows.
    void validate() const;
    auto operator<=>(const GraphKey&) const = default;
};

// A settlement receipt. done() throws when completion cannot be proved; result() throws when settled work failed.
class GraphReceipt {
public:
    virtual ~GraphReceipt() = default;
    virtual bool done() = 0;
    virtual void result() {}
};
using GraphHandle = std::shared_ptr<void>;
using GraphOwner = std::shared_ptr<const void>;

// Bounded cache of captured graphs. An identical key must name the identical physical owners. Replay requires
// settled successful prerequisites and keeps only in-flight receipts plus the latest. A failed capture quarantines
// its owners; a failed/receiptless launch or ambiguous destruction blocks replay and eviction. Not thread-safe.
// Destruction releases settled entries; unproved work retains the cache's owners, executables and receipts
// for process recovery. Capture reserves ownership bookkeeping before invoking the backend hook.
class GraphCache {
public:
    using Capture = std::function<GraphHandle(const GraphKey&)>;
    using Launch = std::function<std::shared_ptr<GraphReceipt>(const GraphHandle&)>;
    using Destroy = std::function<void(const GraphHandle&)>;
    explicit GraphCache(std::size_t capacity);
    ~GraphCache();
    GraphCache(const GraphCache&);
    GraphCache& operator=(const GraphCache&);
    GraphCache(GraphCache&&) noexcept;
    GraphCache& operator=(GraphCache&&) noexcept;
    GraphHandle capture(const GraphKey&, std::vector<GraphOwner> owners, const Capture&);
    std::shared_ptr<GraphReceipt> replay(const GraphKey&, const Launch&,
                                         std::span<const std::shared_ptr<GraphReceipt>> prerequisites = {});
    // Destroys only after every retained launch has settled.
    void evict(const GraphKey&, const Destroy&);
    std::size_t size() const noexcept;
    bool contains(const GraphKey&) const;
    std::span<const GraphOwner> owners(const GraphKey&) const;
    std::size_t in_flight(const GraphKey&) const; // Retained receipts, without probing them.
    bool failed(const GraphKey&) const;
    std::span<const std::vector<GraphOwner>> quarantined() const noexcept;
private:
    struct Entry {
        GraphHandle graph;
        std::vector<GraphOwner> owners;
        std::vector<std::shared_ptr<GraphReceipt>> completions;
        bool failed_operation = false;
        bool settled();
    };
    const Entry& at(const GraphKey&) const;
    std::size_t capacity_;
    std::map<GraphKey, Entry> entries_;
    std::vector<std::vector<GraphOwner>> quarantined_;
    struct Retention;
    std::unique_ptr<Retention> retention_;
    void swap(GraphCache&) noexcept;
};

// Staged matrix geometry. Stride is in elements, including row padding.
enum class MatrixElement { f16, bf16, f32 };
struct MatrixLayout {
    std::size_t rows, columns, stride;
    MatrixElement element;
    bool operator==(const MatrixLayout&) const = default;
};
// A pre-submission geometry rejection; callers may pick another graph or an eager path. Never a runtime failure.
class UnsupportedLayout : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};
std::size_t element_bytes(MatrixElement);
// rows * stride * element bytes, charging the padded final row. Throws UnsupportedLayout.
std::size_t matrix_bytes(const MatrixLayout&);
struct RowCopy {
    std::size_t destination, source, bytes;
};

// Validated staging geometry: one phase or more (at most 64), one bridge between adjacent phases, contiguous
// staging of padded_rows rows and the per-row copies between strided owners and staging.
class StagedGraphPlan {
public:
    StagedGraphPlan(MatrixLayout input, MatrixLayout output, std::size_t padded_rows, std::size_t phases,
                    std::size_t bridges);
    const MatrixLayout& input() const noexcept { return input_; }
    const MatrixLayout& output() const noexcept { return output_; }
    const MatrixLayout& staged_input() const noexcept { return staged_input_; }
    const MatrixLayout& staged_output() const noexcept { return staged_output_; }
    std::size_t phases() const noexcept { return phases_; }
    bool compatible(const MatrixLayout& input, const MatrixLayout& output) const noexcept;
    // Throws UnsupportedLayout when the layouts differ or an owner does not cover its layout.
    void check(const MatrixLayout& input, std::size_t input_bytes, const MatrixLayout& output,
               std::size_t output_bytes) const;
    std::span<const RowCopy> input_rows() const noexcept { return input_rows_; }   // Staging <- input owner.
    std::span<const RowCopy> output_rows() const noexcept { return output_rows_; } // Output owner <- staging.
private:
    MatrixLayout input_, output_, staged_input_, staged_output_;
    std::size_t phases_;
    std::vector<RowCopy> input_rows_, output_rows_;
};

// Replay lifetime of one staged graph: idle, one replay pending, failed or closed. Any failure after submission
// starts poisons replay (it is never a layout fallback); only a successful drain closes and releases owners.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4702) // callbacks that always throw make the inlined bookkeeping unreachable
#endif
class StagedGraphState {
public:
    bool pending() const noexcept { return pending_; }
    bool failed() const noexcept { return failed_; }
    bool closed() const noexcept { return closed_; }
    void require_usable() const {
        if(closed_ || failed_) throw std::logic_error("staged graph closed or failed; drain/close required");
    }
    // Captures each phase in order; a failure poisons the state.
    template<class CapturePhase> void capture(std::size_t phases, CapturePhase&& capture_phase) {
        guard([&] { for(std::size_t i = 0; i < phases; ++i) capture_phase(i); });
    }
    // prepare (dependencies, staging, input rows), then launch(0), bridge(0), launch(1) ... launch(n-1), then
    // finish (output rows, completion). Owners stay retained from the first enqueue even if it fails ambiguously.
    template<class Prepare, class LaunchPhase, class Bridge, class Finish>
    void replay(std::size_t phases, Prepare&& prepare, LaunchPhase&& launch, Bridge&& bridge, Finish&& finish) {
        require_usable();
        if(pending_) throw std::logic_error("staging still has in-flight readers");
        pending_ = true;
        guard([&] {
            prepare();
            for(std::size_t i = 0; i < phases; ++i) {
                launch(i);
                if(i + 1 < phases) bridge(i);
            }
            finish();
        });
    }
    // Probes the pending replay; settle releases its retained owners once complete. A failed probe poisons.
    template<class Probe, class Settle> bool ready(Probe&& probe, Settle&& settle) {
        require_usable();
        if(!pending_) return true;
        bool complete = false;
        guard([&] {
            complete = probe();
            if(complete) settle();
        });
        if(complete) pending_ = false;
        return complete;
    }
    template<class Fence, class Settle> void wait(Fence&& fence, Settle&& settle) {
        require_usable();
        if(!pending_) return;
        guard([&] { fence(); settle(); });
        pending_ = false;
    }
    // drain must settle every use and destroy captures. False (and failed) when it throws: retain everything.
    template<class Drain> bool close(Drain&& drain) noexcept {
        if(closed_) return true;
        try { drain(); }
        catch(...) { failed_ = true; return false; }
        pending_ = false;
        closed_ = true;
        return true;
    }
private:
    template<class Work> void guard(Work&& work) {
        try { work(); }
        catch(...) { failed_ = true; throw; }
    }
    bool pending_ = false, failed_ = false, closed_ = false;
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}
