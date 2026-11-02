#pragma once
#include <crit/core.h>
#include <crit/sampling.h>
#include <crit/state.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace crit::core {
// Batched steps: how one mixed step lays out the rows of many requests, which rows produce logits and the pages
// admission reserves per request. Host bookkeeping only: the engine's arithmetic, state bytes and logits stay its own.
enum class RowPhase : std::uint8_t { decode, prefill };
// One work item's rows in a step. A decode item stages its pending token (with its drafts in a verify), a prefill item
// its prompt chunk. The last `logits` rows produce logits: every row of a decode or verify item, the last row of a
// chunk that ends its prompt, none of another chunk.
struct StepItem {
    RequestKey key;
    RowPhase phase = RowPhase::decode;
    std::uint32_t rows = 1, logits = 1;
};
// Where arrange() put one item: step rows [first, first + rows) and gathered logits rows [first_logit, first_logit +
// logits). `item` is its index in the arrange() input.
struct StepPlacement {
    RequestKey key;
    RowPhase phase = RowPhase::decode;
    std::uint32_t item = 0, first = 0, rows = 0, first_logit = 0, logits = 0;
};
// One step's rows: decode items first, then prefill items, each phase in input order, so a step's decode rows are
// one contiguous block ([0, decode_rows())) and its prompt chunks another, which an engine can compute by phase. A
// row's work follows its phase, never the step's row counts. logit_rows() lists the step row of every logits row in
// ascending order: the gather index of one logits pass over just those rows. Reuse one layout per engine; arrange()
// keeps its storage.
class StepLayout {
public:
    // invalid_argument for an item without rows, more logits rows than rows, a decode item without logits, or more
    // than 2^31 - 1 rows in the step. Changes nothing when it throws.
    void arrange(std::span<const StepItem> items);
    std::span<const StepPlacement> placements() const noexcept; // row order
    const StepPlacement& placement(std::size_t item) const;    // by input index; out_of_range past the items
    std::span<const std::uint32_t> logit_rows() const noexcept;
    std::uint32_t rows() const noexcept;
    std::uint32_t decode_rows() const noexcept;
    std::uint32_t decode_items() const noexcept;
private:
    std::vector<StepPlacement> placements_;
    std::vector<std::uint32_t> logit_rows_, by_item_;
    std::uint32_t rows_ = 0, decode_rows_ = 0, decode_items_ = 0;
};
// The StepItem of a WorkItem. A decode item (prefill complete) stages decode_rows rows (1, or a verify's drafts + 1),
// at most its token work on mixed batches and at least one: a smaller grant falls back to fewer drafts, down to an
// ordinary step. A prefill item computes the prompt left after `frontier`, at most its prefill grant and its token
// work, and produces logits when those rows end the prompt. invalid_argument for a frontier at or past the prompt's
// end on a prefill item, or zero decode_rows.
StepItem step_item(const WorkItem&, std::uint32_t prompt_tokens, std::uint32_t frontier, std::uint32_t decode_rows = 1);

// Pages of page_tokens rows that hold `rows` rows; invalid_argument for zero page_tokens, overflow_error past 2^32 - 1.
std::uint32_t row_pages(std::uint64_t rows, std::uint32_t page_tokens);
// What admission reserves per request: pages for every row it can stage, its prompt, its output and a verify's extra
// columns (verify_rows - 1 past the last output), at most max_tokens rows, without credit for prefix pages it may
// share. Admitted requests then never run out of pages: a sequence's block table never holds more pages than its
// rows need, shared and copied-on-write pages included, as long as the engine retires released pages before it
// begins the next round (a synchronized step) or keeps a pool margin for pages still retiring.
class PageReservations {
public:
    struct Config {
        std::uint32_t pages = 0, page_tokens = 16;  // the pool (StateCacheConfig pages and page_tokens)
        std::uint32_t max_tokens = 0;               // rows per sequence (StateCacheConfig max_tokens)
        std::uint32_t verify_rows = 1;              // most rows a decode round stages: drafts + 1
    };
    // invalid_argument for zero pages, page_tokens or max_tokens, verify_rows outside [1, max_tokens], or a pool
    // with fewer rows than max_tokens.
    explicit PageReservations(Config);
    const Config& config() const noexcept;
    std::uint32_t need(const RequestView&) const noexcept;
    // infeasible: an empty prompt or prompt plus output beyond max_tokens (the pool holds any one sequence);
    // deferred while the reservations leave too few pages; otherwise ready. Inspection only.
    Admission admission(const RequestView&) const noexcept;
    // Reserves need(view) and returns it, the count release() takes back; logic_error unless admission() is ready.
    std::uint32_t reserve(const RequestView&);
    // logic_error above the reserved pages.
    void release(std::uint32_t pages);
    std::uint32_t reserved() const noexcept;
private:
    Config config_;
    std::uint32_t reserved_ = 0;
};

// A request's page demand for a projection: the rows it holds or has reserved (its prompt, computed or not, and the
// output so far) and the output rows it may still stage.
struct PageDemand {
    std::uint32_t held_rows = 0, remaining_rows = 0;
};
// An upper bound on the pages `demands` hold at once while each stages at most one row per step and frees its pages
// when its remaining rows run out: requests that finish early free their pages before the longer ones peak, so the
// bound is often far below the sum of each request's maximum, which it never exceeds. verify_rows - 1 rows past the
// last output count as for PageReservations. It holds only if no request advances faster than the others or starts
// late (speculative rounds, prompts still prefilling): admit by it only with a fallback when pages run out, such as
// Core's memory preemption (Profile.memory_preemption) or a pool margin. invalid_argument for zero page_tokens or
// verify_rows.
std::uint64_t peak_pages(std::span<const PageDemand> demands, std::uint32_t page_tokens, std::uint32_t verify_rows = 1);

// One request's speculative round over its StateCache sequence, for batched verification: every request of a step
// verifies its own drafts in its own decode item, accepts its own prefix and commits or rolls back its own rows.
// begin() stages the pending token and K drafts as one round of K + 1 rows at positions committed .. committed + K (a
// decode StepItem of K + 1 rows, all producing logits); the engine writes their state at Round::kv and computes the
// target's prediction after each row, predictions[i] after inputs[0..i]. accept() applies Core's greedy rule
// (accept_greedy): the matching draft prefix, then a correction or bonus. Emitting m tokens materializes the pending
// token and the first m - 1 emitted tokens, so commit(m) keeps the round's first m rows and makes the last emitted token
// the new pending one; the other rows roll back, their pages and copy-on-write copies returning at the cache's next
// epoch. Core may keep fewer tokens than the round emitted (stops, the output limit, cancellation): commit what Core
// accepted. Recurrent components follow StateCache's commit modes. Draws for sampled verification stay the engine's.
class SpeculativeRow {
public:
    // A sequence whose committed rows end just before `pending`, the token emitted last and not yet materialized
    // (after prefill: the first output). The cache must outlive the row.
    SpeculativeRow(StateCache&, SequenceId, Token pending);
    // Stages the pending token and the drafts, cut so the round fits max_round_rows and max_tokens (each draft takes a
    // row past the pending token's). logic_error while a round is staged; the cache's errors otherwise.
    const Round& begin(std::span<const Token> drafts);
    // The staged rows' tokens: the pending token, then the drafts kept.
    std::span<const Token> inputs() const noexcept;
    std::uint32_t drafts() const noexcept;
    // Greedy verification of the staged round: one prediction per staged row. `remaining` is the output allowance;
    // a stop in `stops` ends the round after it. invalid_argument for another count; logic_error without a round.
    Acceptance accept(std::span<const Token> predictions, std::uint32_t remaining, std::span<const Token> stops = {});
    // The round's emitted tokens (predictions[0, emitted)) after accept().
    std::span<const Token> emitted() const noexcept;
    // Keeps `accepted` rows (at most the emitted count): zero discards the round and keeps the pending token.
    // out_of_range above the emitted count; logic_error without an accepted round; the cache's errors otherwise, and
    // a round the cache keeps pending after one (a throwing snapshot copy) stays staged here for discard() or a retry.
    void commit(std::uint32_t accepted);
    void discard();
    Token pending() const noexcept;
    bool staged() const noexcept;
    SequenceId sequence() const noexcept;
private:
    StateCache* cache_;
    SequenceId sequence_;
    Token pending_;
    std::optional<Round> round_;
    std::optional<Acceptance> acceptance_;
    std::vector<Token> inputs_, emitted_;
};
}
