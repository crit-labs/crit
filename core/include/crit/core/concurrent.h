#pragma once
#include "crit/core/model.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace crit::core {
// These identities are logical, never physical allocation or graph-cache keys.
// All callback methods run on the one engine worker. They must return; deadlines
// cannot preempt a stuck extension callback. No callback may call this Engine.
struct RequestKey {
    std::uint64_t request = 0, generation = 0;
    std::uint32_t lane = 0;
    bool operator==(const RequestKey&) const = default;
};
class ConcurrentSession {
public:
    virtual ~ConcurrentSession() = default;
    // Only after a successful retirement receipt or quiesce + backend drain.
    // Must not enqueue, throw, or retain borrowed prompt views.
    virtual void close(Finish) noexcept = 0;
};
struct RequestView {
    RequestKey key;
    std::span<const Token> prompt;
    std::uint32_t output_tokens;
    // Always non-null when supplied by Core; same lifetime as the prompt view.
    const RequestOptions* options = nullptr;
};
enum class Admission { ready, deferred, infeasible };
struct SchedulingHint {
    // Zero cost is unknown. Include restoration and remaining computation.
    std::uint64_t remaining_ns = 0;
    std::uint32_t reusable_tokens = 0;
    // Nonzero equal keys suggest a shared pending prefix. Advisory only: they
    // never authorize state reuse. The engine defines compatible namespaces.
    std::uint64_t prefix_affinity = 0;
    bool prefix_pending = false;
};
struct StepKey {
    RequestKey request;
    std::uint64_t sequence = 0;
    bool operator==(const StepKey&) const = default;
};
// An owned, already-established dependency and private engine-accepted carry.
// It is not host-readable progress. A successor uses actual accepted tokens and
// positions, never guessed tokens or unverified speculative candidates.
class StepContinuation { public: virtual ~StepContinuation() = default; };
struct WorkItem {
    RequestKey key;
    ConcurrentSession* session;
    WorkBudget budget;
    std::uint64_t step = 0;
    std::shared_ptr<StepContinuation> predecessor = {};
};
struct RowProgress { RequestKey key; Progress progress; std::uint64_t step = 0; };
struct CommitDecision {
    RequestKey key;
    ConcurrentSession* session;
    std::uint32_t accepted;
    bool terminal, cancelled;
    std::uint64_t step = 0;
    // A terminal predecessor makes this descendant private suffix discard-only.
    bool discard = false;
};
class Retirement {
public:
    virtual ~Retirement() = default;
    // Nonblocking certificate: every session reader AND host producer has settled.
    // False preserves the occupied lane. An exception poisons the whole engine.
    virtual bool ready() = 0;
};
class BatchExecution {
public:
    using Notify = std::function<void()>;
    virtual ~BatchExecution() = default;
    // Core retains the operation and its immutable WorkItems before submit.
    // A throwing/ambiguous submission stays retained through quiesce/drain.
    // Call notify after ready() can observe settlement, including failed work.
    // It is safe on any thread, including before submit returns. It only wakes
    // Core; it neither publishes output nor certifies readiness. Quiesce and
    // destruction must join producers, including any outstanding notification.
    virtual void submit(Notify notify) = 0;
    // Called after successful submit, before host readiness. Required for opted-in
    // dependent steps. The dependency and storage outlive this operation as needed.
    virtual std::shared_ptr<StepContinuation> continuation(StepKey) { return {}; }
    // Nonblocking; true proves all submitted operation readers/producers settled.
    virtual bool ready() = 0;
    virtual std::vector<RowProgress> progress() = 0;
    virtual std::span<const Token> ready_tokens(const RequestKey&) { return {}; }
    virtual std::span<const DeviceTiming> device_timings() { return {}; }
    // No output is published if commit throws. Transfer any owners needed by
    // newly enqueued state work into the sessions; operation storage may retire.
    virtual void commit(std::span<const CommitDecision>) = 0;
    virtual void quiesce() = 0; // Idempotently stop/join its host producers.
};
// Engine-defined pool pressure. A blocked preparation has launched nothing and
// released every provisional reservation. Allocation/launch exceptions are failures.
class MemoryPressure { public: virtual ~MemoryPressure() = default; };
struct BatchPreparation {
    Admission status = Admission::ready;
    std::unique_ptr<BatchExecution> execution;
    RequestKey blocked;
    std::shared_ptr<const MemoryPressure> pressure;
};
struct ContinuationView {
    RequestView request;
    std::span<const Token> accepted;
    std::uint32_t prompt_tokens;
    bool prefill_complete;
};
// Complete engine-owned conditioning/state or a lossless replay recipe. The
// engine charges retained storage to Resources before releasing source state.
class ResumeImage { public: virtual ~ResumeImage() = default; };
class Suspension {
public:
    virtual ~Suspension() = default;
    virtual void submit(BatchExecution::Notify) = 0;
    // True certifies a recoverable image AND release of all source readers/state.
    virtual bool ready() = 0;
    virtual std::unique_ptr<ResumeImage> take() = 0;
    virtual void quiesce() = 0;
};
class Restoration {
public:
    virtual ~Restoration() = default;
    virtual void submit(BatchExecution::Notify) = 0;
    virtual bool ready() = 0;
    virtual std::unique_ptr<ConcurrentSession> take() = 0;
    virtual std::uint64_t recomputed_tokens() const noexcept = 0;
    virtual void quiesce() = 0;
};
class ConcurrentModel : public Model {
public:
    std::unique_ptr<Session> open(std::span<const Token>) final;
    // Inspection only; no submissions or provisional allocations. Request memory
    // entitlement is extension policy: reserve in open, or opt into the incremental
    // preparation/recovery contract below before allocating per-step state.
    virtual Admission admission(const RequestView&) { return Admission::ready; }
    // Bounded, host-only inspection; no work submission, waits or retained views.
    virtual SchedulingHint scheduling_hint(const RequestView&) { return {}; }
    virtual std::uint32_t prefill_limit(const WorkItem&, const PrefillWindow&) { return 0; }
    virtual std::unique_ptr<ConcurrentSession> open(const RequestView&) = 0;
    // Request-specific eligibility: host-dependent sampling/grammar uses one.
    virtual std::uint32_t execution_depth(const RequestView&) { return 1; }
    virtual RowProgress prefill(const WorkItem&) = 0;
    // Exactly one row per compact WorkItem, in any order; variable candidates per
    // row allow ordinary, batched speculative, or solo-specific strategies.
    virtual std::vector<RowProgress> decode(std::span<const WorkItem>) = 0;
    // Optional mixed prefill/decode group. Exactly one progress row per item;
    // all grants are ceilings. One-token grants must make progress. Returned
    // storage/readiness/commit obey the same group contract as decode().
    virtual std::vector<RowProgress> execute(std::span<const WorkItem>);
    // Prepare ownership only, without launching work. Operations own private
    // metadata/output/scratch. Lanes are disjoint at depth one; larger depths
    // require the explicit predecessor/step protocol and private acceptance.
    virtual std::unique_ptr<BatchExecution> prepare_batch(std::span<const WorkItem>);
    // Optional incremental path. Infeasible identifies a request that cannot fit
    // alone. Deferred identifies the exact blocked row and noninterchangeable pools.
    virtual BatchPreparation prepare_incremental(std::span<const WorkItem>);
    // Pure bounded inspection of a settled victim: true only if its suspension
    // actually helps this pressure (shared pages must not be counted twice).
    virtual bool reclaimable(const MemoryPressure&, const ContinuationView&, const ConcurrentSession&) { return false; }
    // Preparation launches nothing. Core retains operation and session before submit.
    // take() is called once after ready; the old session is then destroyed, without
    // retire/close: suspension itself releases it, and never completes the request.
    virtual std::unique_ptr<Suspension> suspend(const ContinuationView&, ConcurrentSession&);
    // Core restores with other resident requests retired. Reserve through the old
    // frontier PLUS one useful new step before submission. Never resample/re-emit
    // accepted tokens. Preserve all state, including recurrent/draft/conditioning.
    virtual std::unique_ptr<Restoration> restore(const ContinuationView&, const ResumeImage&);
    // Timings for the last decode()/prefill()/execute() group, for every row key.
    // Same calling and lifetime rules as Session::device_timings().
    virtual std::span<const DeviceTiming> device_timings() { return {}; }
    // Final tokens of a row whose RowProgress set deferred_tokens; called once per
    // such row after every row's readiness, before commit.
    virtual std::span<const Token> ready_tokens(const RequestKey&);
    // All borrowed rows are copied before this call. No output is published until
    // it returns successfully. A throw need not roll back model state: it causes
    // whole-engine quiescence/failure, with no publication from this group.
    virtual void commit(std::span<const CommitDecision>) = 0;
    virtual std::unique_ptr<Retirement> retire(RequestKey, ConcurrentSession&, Finish) = 0;
    // Idempotently stop/join producers, preventing all subsequent submissions.
    // Called before final/failure backend drain; throwing retains complete owners.
    virtual void quiesce() = 0;
};
}
