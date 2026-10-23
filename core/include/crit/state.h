#pragma once
#include <crit/core.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace crit::core {
// Storage-agnostic ledger of KV pages and recurrent state slots. Core never touches state bytes: the engine owns
// storage, writes where a Round points and copies through StateStorage. One PageId indexes every full-attention
// layer; each sliding-window group has a PageId space of its own; slot ids are per component. No internal locking:
// call every method from one thread (the engine worker).
using PageId = std::uint32_t;
using SlotId = std::uint32_t;
inline constexpr PageId no_page = 0xffffffffu;
inline constexpr SlotId no_slot = 0xffffffffu;
// How n accepted rows commit a component. column_checkpoints: the engine writes the state after column i into
// scratch[i] and live becomes scratch[n-1] by id swap. recompute: the engine writes the accepted state into
// scratch[0] (from start) and live becomes it when n > 0. in_place: the engine updates the live slot; counts only.
enum class RecurrentCommit : std::uint8_t { column_checkpoints, recompute, in_place };
// Layers sharing one slot geometry. slots >= sequences + one max_round_rows round's scratch (rows for
// column_checkpoints, 1 for recompute, 0 for in_place) + snapshots; sequences x (1 + scratch) lets every sequence run
// a round.
struct StateComponent {
    std::uint32_t slots = 0;
    RecurrentCommit commit = RecurrentCommit::column_checkpoints;
    // Prefix cache only: at most this many snapshots (the state after an indexed page's last row) kept for reuse,
    // oldest dropped first. A commit snapshots each page it fills whose after-page state exists: any boundary inside
    // a column_checkpoints round, only the round's last accepted row for recompute and in_place.
    std::uint32_t snapshots = 0;
};
// Attention layers that keep keys and values for a sliding window of `window` rows (a row's own included), in pages
// of their own: `pages` ids in a PageId space per group. A page leaves a sequence's window table once every row in it
// is older than the window of the sequence's next row; with the prefix cache, full pages stay reusable after that.
struct WindowGroup {
    std::uint32_t pages = 0, window = 0;
};
// Page hash chain h_i = hash(h_{i-1}, identity, tokens of page i) with h_{-1} = 0. Every hit is verified by exact
// tokens and parent link (page id and generation), so a collision costs reuse, never correctness. chain_hash is the
// default; not cryptographic.
using PageHash = std::uint64_t (*)(std::uint64_t previous, std::uint64_t identity, std::span<const Token> tokens) noexcept;
std::uint64_t chain_hash(std::uint64_t previous, std::uint64_t identity, std::span<const Token> tokens) noexcept;
struct StateCacheConfig {
    std::uint32_t pages = 0;           // physical KV pages; zero for attention-free models
    std::uint32_t page_tokens = 16;    // rows per page
    std::uint32_t sequences = 1;       // open sequences, forks included
    std::uint32_t max_tokens = 0;      // logical rows per sequence
    std::uint32_t max_round_rows = 1;  // most rows one round stages (drafts + 1 for speculation); <= max_tokens
    // Reuse committed full pages across sequences. Requires pages, a nonzero identity naming weights, adapter, KV and
    // recurrent representation and position policy, and snapshots for every recurrent component. Reuse ends at the
    // last matched page that has a snapshot of every component and the window pages each group needs.
    bool prefix_cache = false;
    std::uint64_t identity = 0;
    std::vector<StateComponent> components = {};
    // Sliding-window groups (window index = position here). They need pages: the full-attention page ids also name
    // the token rows that prefix reuse matches, even when no layer stores full-attention KV.
    std::vector<WindowGroup> windows = {};
    PageHash hash = nullptr;           // null selects chain_hash; a seam for collision tests
};
// Engine hooks, called synchronously from begin/try_begin (copy_rows, copy_window_rows), fork and attach (copy_slot)
// and commit (copy_slot, for prefix snapshots). Enqueue on the engine's ordered queue. A throw unwinds the call:
// nothing is published and destination ids are returned.
class StateStorage {
public:
    virtual ~StateStorage() = default;
    // Copy the first `rows` rows of every full-attention layer from page `from` to page `to` (copy-on-write).
    virtual void copy_rows(PageId from, PageId to, std::uint32_t rows) = 0;
    // Copy one component's state from slot `from` to slot `to` (fork, snapshot, restore).
    virtual void copy_slot(std::uint32_t component, SlotId from, SlotId to) = 0;
    // The same as copy_rows for the layers of window group `window`, with that group's page ids. Required with
    // windows; the default throws logic_error.
    virtual void copy_window_rows(std::uint32_t window, PageId from, PageId to, std::uint32_t rows);
};
// A sequence record and the generation that makes closed ids stale.
struct SequenceId {
    std::uint32_t index = 0;
    std::uint64_t generation = 0;
    bool operator==(const SequenceId&) const = default;
};
struct RowSlot { PageId page = no_page; std::uint32_t offset = 0; };
// The one pending round of a sequence; spans stay valid until that sequence's next mutating call.
struct Round {
    std::uint64_t id = 0;
    std::uint32_t first_row = 0, rows = 0;   // staged rows [first_row, first_row + rows); first_row = committed
    std::span<const RowSlot> kv = {};        // where row i's K/V goes; never a page another owner references
    std::span<const SlotId> start = {};      // per component: the live slot at round start, read-only in the round
    std::span<const SlotId> scratch = {};    // components x rows, row-major; unused entries are no_slot
    std::span<const RowSlot> window_kv = {};  // windows x rows, row-major: row i of window group w at [w * rows + i]
};
// No free id of the requested kind; retiring ids wait for retire(). try_begin reports this as nullopt.
class CapacityExhausted : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
struct StateCacheStats {
    // free + cached + retiring + live == pages. Cached pages are indexed, unreferenced and safe to reuse.
    std::uint32_t free_pages = 0, cached_pages = 0, retiring_pages = 0, live_pages = 0, indexed_pages = 0;
    std::uint64_t free_slots = 0, retiring_slots = 0, live_slots = 0;   // summed over components
    std::uint32_t sequences = 0;                                       // open
    // Full prompt pages attach() reused or could not reuse (within its cap); cached pages dropped from the cache
    // (allocation pressure, evict(), or an orphaned entry replaced by a linked copy).
    std::uint64_t prefix_hits = 0, prefix_misses = 0, evictions = 0;
    // Recurrent snapshots held for prefix reuse, summed over components (their slots count as live).
    std::uint64_t snapshots = 0;
};
// Errors: invalid_argument for bad input; logic_error for stale ids, overlapping rounds and misuse (including
// mutation from inside a storage hook); length_error past max_tokens; out_of_range for ranges; CapacityExhausted
// when ids run out; overflow_error for accounting. Failed calls change nothing unless stated.
class StateCache {
public:
    // Validates, reserves array metadata once as Memory::host, then preallocates every array; later calls never
    // allocate except to throw. Storage must outlive the cache; KV and slot bytes are the engine's to reserve.
    StateCache(StateCacheConfig, StateStorage&, Resources&);
    ~StateCache();
    StateCache(const StateCache&) = delete;
    StateCache& operator=(const StateCache&) = delete;
    const StateCacheConfig& config() const noexcept;
    // An empty sequence with a fresh live slot per component, which the engine initializes before its first round.
    SequenceId open();
    // Discards a pending round, then releases every page and slot at epoch(). False, with no change, for a stale
    // id or a call from inside a storage hook.
    bool close(SequenceId) noexcept;
    // Shares the parent's committed pages and copies each live slot through copy_slot. Requires no pending round.
    SequenceId fork(SequenceId parent);
    // Prompt tokens attach() would reuse now: verified full pages, at most min(max_reuse, max_tokens) tokens, ending
    // at a page with every component's snapshot and every window group's pages. max_reuse is normally
    // prompt.size() - 1, because reused rows carry no logits.
    std::uint32_t reusable(std::span<const Token> prompt, std::uint32_t max_reuse) const;
    // Empty sequences only: references the verified pages (and window pages) and commits their tokens; with
    // components, copies the last reused page's snapshots into the live slots (copy_slot), which the engine then must
    // not reinitialize. Returns reused tokens.
    std::uint32_t attach(SequenceId, std::span<const Token> prompt, std::uint32_t max_reuse);
    // Pages a round of `rows` rows would acquire now, including a copy-on-write tail; advisory. Rejects what begin
    // rejects for the request itself (stale id, pending round, rows over max_round_rows or past max_tokens).
    std::uint32_t pages_needed(SequenceId, std::uint32_t rows) const;
    // Stages rows after the committed frontier: validate, acquire every page and slot, copy a shared or indexed
    // partial tail through copy_rows, then publish. nullopt (try_begin) or CapacityExhausted means no change;
    // a copy_rows throw returns its page (a page evicted for it goes to the free list, not back to the cache).
    std::optional<Round> try_begin(SequenceId, std::uint32_t rows);
    Round begin(SequenceId, std::uint32_t rows);
    // Keeps the first inputs.size() staged rows; inputs are their materialized tokens. Zero inputs equals discard.
    // Consumes the round even when it throws (too many inputs, negative tokens): the round is then discarded. A
    // throwing snapshot copy_slot is the exception: the round stays pending and nothing is published.
    // Released pages and slots enter retirement at epoch(); full pages are hashed and indexed (first writer wins
    // while its parent link checks out; an entry whose parent was evicted yields to a linked page).
    void commit(SequenceId, std::uint64_t round, std::span<const Token> inputs);
    void discard(SequenceId, std::uint64_t round);
    // Shortens committed rows (attention only; a sequence with components cannot rewind, and none rewinds to where a
    // window group's dropped pages would be read again). No pending round.
    void rewind(SequenceId, std::uint32_t tokens);
    // Pages for committed + staged rows, in row order.
    std::span<const PageId> block_table(SequenceId) const;
    // Window group `window`'s pages for committed + staged rows, in row order; no_page where the window has passed.
    std::span<const PageId> window_table(SequenceId, std::uint32_t window) const;
    std::uint32_t committed(SequenceId) const;
    std::uint32_t staged(SequenceId) const;
    SlotId live_slot(SequenceId, std::uint32_t component) const;
    // Changes whenever the block table, live slots or round scratch take new ids; unique across sequences.
    std::uint64_t layout_generation(SequenceId) const;
    // Releases are stamped with epoch(). advance_epoch() closes it and returns the closed epoch (call after
    // submitting device work that may read released ids); retire(e) makes ids released at <= e allocatable once
    // all work submitted up to e has settled. e >= epoch() first closes the current epoch, so a single-stream
    // engine calls retire(epoch()) after each step's readiness. Inside a storage hook retire and evict do nothing.
    std::uint64_t epoch() const noexcept;
    std::uint64_t advance_epoch();
    void retire(std::uint64_t completed) noexcept;
    // Evicts up to `pages` cached pages, least recently released first; referenced pages are never evicted.
    // Allocation does the same automatically when no free page remains.
    std::uint32_t evict(std::uint32_t pages) noexcept;
    StateCacheStats stats() const noexcept;
    // The page fields (free, cached, retiring, live; indexed = pages linked for reuse) of window group `window`.
    StateCacheStats window_stats(std::uint32_t window) const;
    std::size_t metadata_bytes() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// After-token checkpoint selection: the state after `boundary_after` tokens of a round starting after
// `prefix_after` is accepted node boundary_after - prefix_after - 1; nullopt names the round-start state.
// out_of_range outside [prefix_after, prefix_after + accepted.size()].
std::optional<std::size_t> checkpoint_node(std::uint64_t prefix_after, std::span<const std::size_t> accepted,
                                           std::uint64_t boundary_after);
}
