#pragma once
#include "crit/core/model.h"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace crit::core {
// These identities are logical, never physical allocation or graph-cache keys.
// All callback methods run on the one engine worker and return with their work done; deadlines
// cannot preempt a stuck callback. No callback may call this Engine.
struct RequestKey {
    std::uint64_t request = 0, generation = 0;
    std::uint32_t lane = 0;
    bool operator==(const RequestKey&) const = default;
};
// A request's state in the model: Core destroys it after retire(), or without it when the engine fails or closes.
class Session {
public:
    virtual ~Session() = default;
};
struct RequestView {
    RequestKey key;
    std::span<const Token> prompt;
    std::uint32_t output_tokens;
    // Always non-null when supplied by Core; same lifetime as the prompt view.
    const RequestOptions* options = nullptr;
};
enum class Admission { ready, deferred, infeasible };
struct WorkItem {
    RequestKey key;
    Session* session;
    WorkBudget budget;
};
struct RowProgress { RequestKey key; Progress progress; };
struct CommitDecision {
    RequestKey key;
    Session* session;
    std::uint32_t accepted;
    bool terminal, cancelled;
};
// A model Core schedules: requests in generation lanes, stepped in groups on the one engine worker.
class Model {
public:
    virtual ~Model() = default;
    virtual ModelInfo info() const = 0;
    // Return one finite score per target, with the work done. Scoring runs only with all generation sessions
    // retired. Core separately reserves public results.
    virtual std::vector<float> score_request(std::span<const Token>, std::uint32_t first_target, const RequestOptions&);
    // Requires ModelInfo::query_scoring. `target` holds log-probabilities.
    virtual QueryScores score_queries(std::span<const Token>, std::uint32_t first_target, const ScoreQueries&,
                                      const RequestOptions&);
    // Inspection only; no submissions or provisional allocations. Request memory
    // entitlement is extension policy: reserve in open.
    virtual Admission admission(const RequestView&) { return Admission::ready; }
    virtual std::unique_ptr<Session> open(const RequestView&) = 0;
    virtual RowProgress prefill(const WorkItem&) = 0;
    // Exactly one row per compact WorkItem, in any order; variable candidates per
    // row allow ordinary, batched speculative, or solo-specific strategies.
    virtual std::vector<RowProgress> decode(std::span<const WorkItem>) = 0;
    // Optional mixed prefill/decode group. Exactly one progress row per item;
    // all grants are ceilings. One-token grants must make progress. Returned
    // storage and commit obey the same group contract as decode().
    virtual std::vector<RowProgress> execute(std::span<const WorkItem>);
    // All borrowed rows are copied before this call. No output is published until
    // it returns successfully. A throw need not roll back model state: it fails
    // the whole engine, with no publication from this group.
    virtual void commit(std::span<const CommitDecision>) = 0;
    // The request ended with `finish`; its session is destroyed after this returns.
    virtual void retire(RequestKey, Session&, Finish) = 0;
};
}
