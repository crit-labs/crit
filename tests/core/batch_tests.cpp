// Batched steps (crit/batch.h): the step layout, WorkItem rows, page reservations, and a toy paged
// engine through a Core Engine whose requests decode the same tokens batched as alone, with page-reserving admission;
// speculative rows (per-row acceptance, commit and rollback in StateCache pages) and a toy speculative engine whose
// requests verify their own drafts at their own widths in shared steps and decode exactly the plain greedy tokens.
#include "crit/batch.h"
#include "crit/state.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace crit::core;

namespace {
using Tokens = std::vector<Token>;
void require(bool ok, const std::string& what) { if(!ok) throw std::runtime_error(what); }
template<class Error, class F> void rejects_as(F&& f, const std::string& what) {
    try { f(); } catch(const Error&) { return; }
    throw std::runtime_error("expected rejection: " + what);
}
RequestKey key(std::uint64_t request) { return {request, 1, std::uint32_t(request)}; }

void layout_rules() {
    // A mixed step: a chunk that ends its prompt, a decode, a chunk that does not, a decode, a three-column verify.
    const std::array<StepItem, 5> items{{{key(1), RowPhase::prefill, 3, 1}, {key(2), RowPhase::decode, 1, 1},
                                         {key(3), RowPhase::prefill, 4, 0}, {key(4), RowPhase::decode, 1, 1},
                                         {key(5), RowPhase::decode, 3, 3}}};
    StepLayout step;
    step.arrange(items);
    require(step.rows() == 12 && step.decode_rows() == 5 && step.decode_items() == 3, "row counts");
    std::vector<std::uint64_t> order;
    for(const auto& p:step.placements()) order.push_back(p.key.request);
    require(order == std::vector<std::uint64_t>{2, 4, 5, 1, 3}, "decode items first, each phase in input order");
    const auto& verify = step.placement(4);
    require(verify.first == 2 && verify.rows == 3 && verify.first_logit == 2 && verify.logits == 3 && verify.item == 4,
            "a verify's columns all produce logits");
    const auto& ends = step.placement(0);
    require(ends.first == 5 && ends.first_logit == 5 && ends.logits == 1 && ends.phase == RowPhase::prefill, "prompt end");
    require(step.placement(2).logits == 0 && step.placement(2).first == 8, "a chunk inside its prompt");
    const auto rows = step.logit_rows();
    require(std::vector<std::uint32_t>(rows.begin(), rows.end()) == std::vector<std::uint32_t>{0, 1, 2, 3, 4, 7},
            "logits rows: both decodes, the verify's columns and the prompt's last row");
    // Rearranging reuses storage; a failed arrange changes nothing.
    const std::array<StepItem, 1> one{{{key(9), RowPhase::prefill, 2, 1}}};
    step.arrange(one);
    require(step.rows() == 2 && step.decode_rows() == 0 && step.logit_rows().size() == 1 && step.logit_rows()[0] == 1,
            "a prefill-only step");
    const std::array<StepItem, 2> bad{{{key(1), RowPhase::decode, 1, 1}, {key(2), RowPhase::prefill, 0, 0}}};
    rejects_as<std::invalid_argument>([&] { step.arrange(bad); }, "an item without rows");
    require(step.rows() == 2 && step.placements().size() == 1, "a failed arrange keeps the previous layout");
    const std::array<StepItem, 1> greedy_logits{{{key(1), RowPhase::prefill, 2, 3}}};
    rejects_as<std::invalid_argument>([&] { step.arrange(greedy_logits); }, "more logits than rows");
    const std::array<StepItem, 1> silent{{{key(1), RowPhase::decode, 1, 0}}};
    rejects_as<std::invalid_argument>([&] { step.arrange(silent); }, "a decode item without logits");
    const std::array<StepItem, 2> huge{{{key(1), RowPhase::prefill, 0x7fffffffu, 0}, {key(2), RowPhase::decode, 1, 1}}};
    rejects_as<std::invalid_argument>([&] { step.arrange(huge); }, "more than 2^31 - 1 rows");
    rejects_as<std::out_of_range>([&] { (void)step.placement(1); }, "a placement past the items");
    step.arrange({});
    require(!step.rows() && step.placements().empty() && step.logit_rows().empty(), "an empty step");
}

void item_rules() {
    WorkItem decode{key(1), nullptr, {0, 9, true, 0}};
    auto item = step_item(decode, 10, 10);
    require(item.phase == RowPhase::decode && item.rows == 1 && item.logits == 1, "one decode row");
    item = step_item(decode, 10, 10, 5);
    require(item.rows == 5 && item.logits == 5, "a verify of four drafts");
    decode.budget.token_work = 3;
    require(step_item(decode, 10, 10, 5).rows == 3, "a smaller grant drops drafts");
    decode.budget.token_work = 1;
    require(step_item(decode, 10, 10, 5).rows == 1, "a one-unit grant is an ordinary step");
    rejects_as<std::invalid_argument>([&] { step_item(decode, 10, 10, 0); }, "zero decode rows");

    WorkItem prefill{key(2), nullptr, {16, 9, false, 0}};
    item = step_item(prefill, 40, 0);
    require(item.phase == RowPhase::prefill && item.rows == 16 && item.logits == 0, "a chunk within its prompt");
    item = step_item(prefill, 40, 30);
    require(item.rows == 10 && item.logits == 1, "the last chunk produces logits");
    prefill.budget.token_work = 4;
    item = step_item(prefill, 40, 30);
    require(item.rows == 4 && item.logits == 0, "token work bounds the chunk");
    prefill.budget.token_work = 0;
    rejects_as<std::invalid_argument>([&] { step_item(prefill, 40, 40); }, "a prefill past its prompt");
    prefill.budget.prefill_tokens = 0;
    rejects_as<std::invalid_argument>([&] { step_item(prefill, 40, 0); }, "a prefill without a grant");
}

void page_rules() {
    require(row_pages(0, 16) == 0 && row_pages(1, 16) == 1 && row_pages(16, 16) == 1 && row_pages(17, 16) == 2, "row pages");
    rejects_as<std::invalid_argument>([] { row_pages(1, 0); }, "zero page tokens");
    rejects_as<std::overflow_error>([] { row_pages(std::uint64_t(1) << 40, 1); }, "too many pages");

    PageReservations pool({10, 16, 160, 1});
    const Tokens prompt(53, 1);
    const RequestView a{key(1), prompt, 24, nullptr};
    require(pool.need(a) == 5 && pool.admission(a) == Admission::ready, "prompt and output pages");
    require(pool.reserve(a) == 5 && pool.reserved() == 5, "a reservation");
    require(pool.reserve(a) == 5 && pool.reserved() == 10, "a second request fills the pool");
    require(pool.admission(a) == Admission::deferred, "a full pool defers");
    rejects_as<std::logic_error>([&] { pool.reserve(a); }, "reserving past admission");
    pool.release(5);
    require(pool.admission(a) == Admission::ready && pool.reserved() == 5, "released pages admit again");
    rejects_as<std::logic_error>([&] { pool.release(6); }, "releasing more than is reserved");
    const Tokens empty;
    require(pool.admission({key(2), empty, 4, nullptr}) == Admission::infeasible, "an empty prompt");
    require(pool.admission({key(2), prompt, 108, nullptr}) == Admission::infeasible, "beyond max_tokens");
    require(pool.need({key(2), prompt, 107, nullptr}) == 10, "a request filling a sequence");

    // A verify stages drafts past the last output: three more rows, capped at the sequence.
    PageReservations speculative({10, 16, 160, 4});
    require(speculative.need({key(1), Tokens(13, 1), 3, nullptr}) == 2, "16 + 3 rows take two pages");
    require(speculative.need({key(1), Tokens(150, 1), 10, nullptr}) == 10, "capped at max_tokens");
    rejects_as<std::invalid_argument>([] { PageReservations({0, 16, 16, 1}); }, "no pages");
    rejects_as<std::invalid_argument>([] { PageReservations({1, 16, 17, 1}); }, "a sequence larger than the pool");
    rejects_as<std::invalid_argument>([] { PageReservations({1, 16, 16, 17}); }, "verify rows past the sequence");
}

// The toy paged engine: a row's state is its token id, stored in StateCache pages; the next token is a hash of every
// row a sequence sees through its block table, so wrong page bookkeeping changes the tokens, while requests never read
// each other's rows: a request's tokens do not depend on its step-mates.
struct HostPages final : StateStorage {
    std::uint32_t page_tokens;
    std::vector<Token> rows;
    HostPages(std::uint32_t pages, std::uint32_t tokens) : page_tokens(tokens), rows(std::size_t(pages) * tokens, -1) {}
    void copy_rows(PageId from, PageId to, std::uint32_t count) override {
        std::copy_n(rows.begin() + std::ptrdiff_t(from) * page_tokens, count, rows.begin() + std::ptrdiff_t(to) * page_tokens);
    }
    void copy_slot(std::uint32_t, SlotId, SlotId) override { throw std::logic_error("no recurrent state"); }
    Token& at(RowSlot slot) { return rows[std::size_t(slot.page) * page_tokens + slot.offset]; }
};
constexpr std::uint32_t vocabulary = 1000;
// The toy target: the next token is a hash of every token before it.
Token rule(std::span<const Token> rows) {
    std::uint64_t h = 1469598103934665603ull;
    for(std::size_t p = 0; p < rows.size(); ++p) h = (h ^ std::uint64_t(rows[p] + 1) * (p + 1)) * 1099511628211ull;
    return Token(h % vocabulary);
}
// The first `keys` rows a sequence holds, read through its block table.
Tokens gather(const HostPages& pages, std::span<const PageId> table, std::uint32_t keys) {
    Tokens seen(keys);
    for(std::uint32_t p = 0; p < keys; ++p) seen[p] = pages.rows[std::size_t(table[p / pages.page_tokens]) * pages.page_tokens + p % pages.page_tokens];
    return seen;
}
Token attend(const HostPages& pages, std::span<const PageId> table, std::uint32_t keys) { return rule(gather(pages, table, keys)); }
struct Host final : Backend { void bind_thread() override {} void drain() override {} };
struct Stats { std::atomic<std::uint32_t> largest{0}, mixed{0}, steps{0}; };

class PagedModel final : public ConcurrentModel {
    struct Receipt final : Retirement { bool ready() override { return true; } };
    struct Session final : ConcurrentSession {
        RequestView request;
        StateCache& cache;
        SequenceId sequence;
        std::uint32_t reserved = 0;
        std::optional<Round> round;
        Tokens inputs, produced;
        Token last = 0;
        Session(const RequestView& r, StateCache& c) : request(r), cache(c), sequence(c.open()) {}
        ~Session() override { cache.close(sequence); }
        void close(Finish) noexcept override { cache.close(sequence); }
    };
    std::uint32_t concurrency_;
    HostPages pages_;
    StateCache cache_;
    PageReservations reservations_;
    StepLayout layout_;
    std::shared_ptr<Stats> stats_;
    static StateCacheConfig config(std::uint32_t pages, std::uint32_t concurrency) {
        StateCacheConfig c;
        c.pages = pages; c.page_tokens = 4; c.sequences = concurrency; c.max_tokens = 64; c.max_round_rows = 64;
        return c;
    }
public:
    PagedModel(std::uint32_t pages, std::uint32_t concurrency, Services services, std::shared_ptr<Stats> stats)
        : concurrency_(concurrency), pages_(pages, 4), cache_(config(pages, concurrency), pages_, services.resources),
          reservations_({pages, 4, 64, 1}), stats_(std::move(stats)) {}
    ModelInfo info() const override {
        ModelInfo i;
        i.implementation = "toy paged";
        i.vocabulary = vocabulary; i.context = 64; i.prefill_chunk = 8;
        i.max_concurrency = concurrency_;
        i.mixed_batches = true;
        return i;
    }
    Admission admission(const RequestView& r) override { return reservations_.admission(r); }
    std::unique_ptr<ConcurrentSession> open(const RequestView& r) override {
        auto session = std::make_unique<Session>(r, cache_);
        session->reserved = reservations_.reserve(r);
        return session;
    }
    std::vector<RowProgress> execute(std::span<const WorkItem> work) override {
        std::vector<StepItem> items;
        for(const auto& item:work) {
            const auto& s = static_cast<const Session&>(*item.session);
            items.push_back(step_item(item, std::uint32_t(s.request.prompt.size()), cache_.committed(s.sequence)));
        }
        layout_.arrange(items);
        // Stage every row first (each item's rounds take its pages), then write its tokens at Round::kv.
        for(const auto& placement:layout_.placements()) {
            auto& s = static_cast<Session&>(*work[placement.item].session);
            const auto start = cache_.committed(s.sequence);
            s.round = cache_.begin(s.sequence, placement.rows);
            s.inputs.clear();
            for(std::uint32_t r = 0; r < placement.rows; ++r)
                s.inputs.push_back(placement.phase == RowPhase::decode ? s.last : s.request.prompt[start + r]);
            for(std::uint32_t r = 0; r < placement.rows; ++r) pages_.at(s.round->kv[r]) = s.inputs[r];
        }
        std::vector<RowProgress> progress;
        for(const auto& placement:layout_.placements()) {
            const auto& item = work[placement.item];
            auto& s = static_cast<Session&>(*item.session);
            s.produced.clear();
            const auto keys = cache_.committed(s.sequence) + placement.rows;
            if(placement.logits) s.produced.push_back(attend(pages_, cache_.block_table(s.sequence), keys));
            Progress p;
            p.prompt_tokens = placement.phase == RowPhase::prefill ? placement.rows : 0;
            p.prefill_complete = placement.logits > 0;
            p.tokens = s.produced;
            p.token_work = placement.rows;
            progress.push_back({item.key, p});
        }
        stats_->steps++;
        stats_->largest = std::max<std::uint32_t>(stats_->largest, std::uint32_t(work.size()));
        if(layout_.decode_items() && layout_.decode_items() < work.size()) stats_->mixed++;
        return progress;
    }
    RowProgress prefill(const WorkItem&) override { throw std::logic_error("mixed batches only"); }
    std::vector<RowProgress> decode(std::span<const WorkItem>) override { throw std::logic_error("mixed batches only"); }
    void commit(std::span<const CommitDecision> decisions) override {
        for(const auto& d:decisions) {
            auto& s = static_cast<Session&>(*d.session);
            if(!s.round) continue;
            const auto round = std::exchange(s.round, std::nullopt)->id;
            if(d.cancelled || d.discard) { cache_.discard(s.sequence, round); continue; }
            cache_.commit(s.sequence, round, s.inputs);
            if(d.accepted) s.last = s.produced.front();
        }
        cache_.retire(cache_.epoch());
    }
    std::unique_ptr<Retirement> retire(RequestKey, ConcurrentSession& session, Finish) override {
        auto& s = static_cast<Session&>(session);
        if(s.round) { cache_.discard(s.sequence, s.round->id); s.round.reset(); }
        cache_.close(s.sequence);
        cache_.retire(cache_.epoch());
        reservations_.release(std::exchange(s.reserved, 0u));
        return std::make_unique<Receipt>();
    }
    void quiesce() override {}
};

std::unique_ptr<Engine> paged_engine(std::uint32_t pages, std::uint32_t concurrency, std::shared_ptr<Stats> stats) {
    Profile p;
    p.context = 64; p.prefill_chunk = 8;
    p.host_limit = std::size_t(64) << 20; p.device_limit = std::size_t(1) << 20;
    p.max_concurrency = concurrency; p.max_pending = 16;
    p.max_batch_tokens = 24;
    return std::make_unique<Engine>(p, std::make_shared<Host>(), [=](const Profile&, Services services) -> std::unique_ptr<Model> {
        return std::make_unique<PagedModel>(pages, concurrency, services, stats);
    });
}

std::vector<Tokens> prompts() {
    std::vector<Tokens> out;
    for(int r = 0; r < 9; ++r) {
        Tokens prompt;
        for(int i = 0; i < 1 + r * 5; ++i) prompt.push_back(Token((r * 37 + i * 11) % int(vocabulary)));
        out.push_back(std::move(prompt));
    }
    return out;
}

// Plain greedy decoding of the toy model on flat rows: the oracle for every run.
Tokens oracle(const Tokens& prompt, std::uint32_t output) {
    Tokens rows(prompt), out;
    for(std::uint32_t i = 0; i < output; ++i) {
        out.push_back(rule(rows));
        rows.push_back(out.back());
    }
    return out;
}

void engine_requests() {
    const auto all = prompts();
    constexpr std::uint32_t output = 12;
    // One at a time, then all at once on the same pool: every request decodes the oracle's tokens.
    for(const std::uint32_t concurrency:{1u, 6u}) {
        auto stats = std::make_shared<Stats>();
        auto engine = paged_engine(160, concurrency, stats);
        std::vector<Handle> handles;
        for(const auto& prompt:all) handles.push_back(engine->submit(prompt, output));
        for(std::size_t r = 0; r < all.size(); ++r)
            require(handles[r].wait().tokens == oracle(all[r], output), "request " + std::to_string(r) + " differs from the oracle");
        if(concurrency > 1) require(stats->largest > 1 && stats->mixed > 0, "concurrent steps hold several requests and mix phases");
        else require(stats->largest == 1, "a serial engine runs one request per step");
    }
    // A pool of 20 pages admits two of these requests at a time: page-reserving admission defers the rest, which
    // finish once earlier requests release their pages.
    auto stats = std::make_shared<Stats>();
    auto small = paged_engine(20, 6, stats);
    std::vector<Handle> handles;
    for(std::size_t r = 4; r < all.size(); ++r) handles.push_back(small->submit(all[r], output));
    for(std::size_t r = 4; r < all.size(); ++r) require(handles[r - 4].wait().tokens == oracle(all[r], output), "deferred requests");
    require(stats->largest <= 2, "reservations bound the requests a step holds");
}

// A commit whose prefix snapshot copy throws leaves the round pending in the cache (crit/state.h): the row keeps it, so
// discarding or committing it again ends it and the sequence stages its next round.
struct FailingSnapshots final : StateStorage {
    bool fail = false;
    void copy_rows(PageId, PageId, std::uint32_t) override {}
    void copy_slot(std::uint32_t, SlotId, SlotId) override { if(fail) throw std::runtime_error("injected snapshot failure"); }
};
void snapshot_failure() {
    Resources resources(std::size_t(1) << 20, 0);
    FailingSnapshots storage;
    StateCacheConfig c;
    c.pages = 8; c.page_tokens = 4; c.sequences = 1; c.max_tokens = 16; c.max_round_rows = 4;
    c.prefix_cache = true; c.identity = 7;
    c.components = {{1 + 4 + 2, RecurrentCommit::column_checkpoints, 2}};
    StateCache cache(c, storage, resources);
    const auto seq = cache.open();
    const auto prompt = cache.begin(seq, 3);
    cache.commit(seq, prompt.id, Tokens{10, 11, 12});
    cache.retire(cache.epoch());
    SpeculativeRow row(cache, seq, 13);
    row.begin(Tokens{14, 15});
    require(row.accept(Tokens{14, 15, 16}, 9).emitted == 3, "every draft accepted");
    storage.fail = true;   // the round fills the first page, whose snapshot copy throws
    rejects_as<std::runtime_error>([&] { row.commit(3); }, "a throwing snapshot copy");
    require(cache.staged(seq) == 3 && row.staged(), "the round the cache keeps pending stays staged in the row");
    storage.fail = false;
    row.commit(3);
    require(cache.committed(seq) == 6 && row.pending() == 16 && !row.staged(), "a retried commit keeps the round");
    cache.retire(cache.epoch());
    row.begin(Tokens{17});
    require(row.accept(Tokens{17, 18}, 9).emitted == 2, "the draft accepted");
    storage.fail = true;   // this round fills the second page
    rejects_as<std::runtime_error>([&] { row.commit(2); }, "a throwing snapshot copy");
    storage.fail = false;
    row.discard();
    require(cache.staged(seq) == 0 && !row.staged() && cache.committed(seq) == 6 && row.pending() == 16,
            "discard ends the pending round");
    require(row.begin(Tokens{}).rows == 1, "the sequence stages again");
}

void speculative_rules() {
    Resources resources(std::size_t(1) << 20, 0);
    HostPages pages(8, 4);
    StateCacheConfig c;
    c.pages = 8; c.page_tokens = 4; c.sequences = 2; c.max_tokens = 16; c.max_round_rows = 6;
    StateCache cache(c, pages, resources);
    const auto seq = cache.open();
    const auto prompt = cache.begin(seq, 5);
    for(std::uint32_t r = 0; r < 5; ++r) pages.at(prompt.kv[r]) = Token(r + 1);
    cache.commit(seq, prompt.id, Tokens{1, 2, 3, 4, 5});
    SpeculativeRow row(cache, seq, 9);
    const auto& round = row.begin(Tokens{10, 11, 12, 13, 14, 15});
    for(std::size_t r = 0; r < row.inputs().size(); ++r) pages.at(round.kv[r]) = row.inputs()[r];
    require(round.rows == 6 && round.first_row == 5 && row.drafts() == 5 && row.inputs()[0] == 9 && row.staged(),
            "the pending token and the drafts a round holds");
    rejects_as<std::logic_error>([&] { row.begin(Tokens{1}); }, "one round at a time");
    rejects_as<std::invalid_argument>([&] { row.accept(Tokens{1, 2}, 9); }, "one prediction per staged row");
    // Drafts 10 and 11 match, 99 corrects the third: three tokens, three rows kept, the rest rolled back.
    const auto a = row.accept(Tokens{10, 11, 99, 13, 14, 15}, 9);
    require(a.emitted == 3 && a.matched == 2 && !a.stopped && row.emitted().size() == 3 && row.emitted()[2] == 99, "greedy acceptance");
    row.commit(3);
    cache.retire(cache.epoch());
    require(cache.committed(seq) == 8 && row.pending() == 99 && !row.staged(), "the pending token and two drafts are materialized");
    require(gather(pages, cache.block_table(seq), 8) == Tokens({1, 2, 3, 4, 5, 9, 10, 11}), "the kept rows hold their tokens");
    require(cache.stats().live_pages == 2, "the rejected rows' page returned");
    // All drafts match and the bonus follows: four tokens, of which Core keeps two (a stop, the output limit).
    row.begin(Tokens{20, 21, 22});
    require(row.accept(Tokens{20, 21, 22, 23}, 9).emitted == 4, "a bonus token");
    row.commit(2);
    require(cache.committed(seq) == 10 && row.pending() == 21, "a trimmed round keeps Core's count, the last kept token pending");
    // The output allowance bounds the round; zero kept discards it and keeps the pending token.
    row.begin(Tokens{30, 31});
    require(row.accept(Tokens{30, 31, 32}, 1).emitted == 1, "the allowance bounds the round");
    rejects_as<std::out_of_range>([&] { row.commit(2); }, "more rows than tokens emitted");
    row.commit(0);
    require(cache.committed(seq) == 10 && row.pending() == 21 && !row.staged(), "a discarded round");
    // An accepted stop ends the round.
    row.begin(Tokens{40, 41});
    const auto stopped = row.accept(Tokens{40, 41, 42}, 9, Tokens{40});
    require(stopped.emitted == 1 && stopped.stopped, "a stop ends the round");
    row.discard();
    // Near max_tokens drafts are cut: 10 rows committed of 16 leave five drafts after the pending token.
    require(row.begin(Tokens{1, 2, 3, 4, 5, 6, 7}).rows == 6 && row.drafts() == 5, "drafts cut to the round and the sequence");
    row.discard();
    rejects_as<std::logic_error>([&] { row.commit(1); }, "no round to commit");
    snapshot_failure();
}

// The toy speculative engine: request r drafts 1 + r % 4 tokens from the target's rule with deliberate mistakes,
// verifies them in one decode item of drafts + 1 rows over its pages, accepts its own prefix and commits or rolls back
// its own rows; one step holds several requests' verifies and prompt chunks.
struct SpeculativeStats { std::atomic<std::uint32_t> largest{0}, rounds{0}, rejected{0}, extra{0}, trimmed{0}; };
class SpeculativeModel final : public ConcurrentModel {
    struct Receipt final : Retirement { bool ready() override { return true; } };
    struct Session final : ConcurrentSession {
        RequestView request;
        StateCache& cache;
        SequenceId sequence;
        std::uint32_t width, reserved = 0;
        bool stops_aware;
        Tokens stops, chunk, produced;
        std::optional<Round> prefill;
        std::optional<SpeculativeRow> row;
        Session(const RequestView& r, StateCache& c, Tokens s)
            : request(r), cache(c), sequence(c.open()), width(1 + std::uint32_t(r.key.request % 4)), stops_aware(r.key.request % 2 == 0),
              stops(std::move(s)) {}
        ~Session() override { cache.close(sequence); }
        void close(Finish) noexcept override { cache.close(sequence); }
    };
    HostPages pages_;
    StateCache cache_;
    PageReservations reservations_;
    StepLayout layout_;
    std::vector<Token> model_stops_;
    std::shared_ptr<SpeculativeStats> stats_;
    std::uint32_t concurrency_;
    static StateCacheConfig config(std::uint32_t concurrency) {
        StateCacheConfig c;
        c.pages = 160; c.page_tokens = 4; c.sequences = concurrency; c.max_tokens = 64; c.max_round_rows = 64;
        return c;
    }
    // Drafts from the target's rule over the request's own rows, wrong whenever the position and request say so.
    Tokens draft(const Session& s, std::uint32_t count) const {
        auto history = gather(pages_, cache_.block_table(s.sequence), cache_.committed(s.sequence));
        history.push_back(s.row->pending());
        Tokens out;
        for(std::uint32_t j = 0; j < count; ++j) {
            auto token = rule(history);
            if((history.size() + s.request.key.request) % 3 == 0) token = Token((token + 1) % vocabulary);
            out.push_back(token);
            history.push_back(token);
        }
        return out;
    }
public:
    SpeculativeModel(std::uint32_t concurrency, Services services, std::shared_ptr<SpeculativeStats> stats)
        : pages_(160, 4), cache_(config(concurrency), pages_, services.resources), reservations_({160, 4, 64, 5}),
          stats_(std::move(stats)), concurrency_(concurrency) {}
    ModelInfo info() const override {
        ModelInfo i;
        i.implementation = "toy speculative";
        i.vocabulary = vocabulary; i.context = 64; i.prefill_chunk = 8;
        i.max_concurrency = concurrency_;
        i.mixed_batches = true;
        i.decode_token_work = 5;
        return i;
    }
    Admission admission(const RequestView& r) override { return reservations_.admission(r); }
    std::unique_ptr<ConcurrentSession> open(const RequestView& r) override {
        auto session = std::make_unique<Session>(r, cache_, r.options->stop_tokens ? *r.options->stop_tokens : Tokens{});
        session->reserved = reservations_.reserve(r);
        return session;
    }
    std::vector<RowProgress> execute(std::span<const WorkItem> work) override {
        std::vector<StepItem> items;
        for(const auto& item:work) {
            const auto& s = static_cast<const Session&>(*item.session);
            items.push_back(step_item(item, std::uint32_t(s.request.prompt.size()), cache_.committed(s.sequence), s.width + 1));
        }
        layout_.arrange(items);
        std::uint32_t verifies = 0;
        for(const auto& placement:layout_.placements()) {
            auto& s = static_cast<Session&>(*work[placement.item].session);
            if(placement.phase == RowPhase::prefill) {
                const auto start = cache_.committed(s.sequence);
                s.prefill = cache_.begin(s.sequence, placement.rows);
                s.chunk.assign(s.request.prompt.begin() + start, s.request.prompt.begin() + start + placement.rows);
                for(std::uint32_t r = 0; r < placement.rows; ++r) pages_.at(s.prefill->kv[r]) = s.chunk[r];
                continue;
            }
            const auto& round = s.row->begin(draft(s, placement.rows - 1));
            const auto inputs = s.row->inputs();
            for(std::size_t r = 0; r < inputs.size(); ++r) pages_.at(round.kv[r]) = inputs[r];
            ++verifies;
        }
        std::vector<RowProgress> progress;
        for(const auto& placement:layout_.placements()) {
            const auto& item = work[placement.item];
            auto& s = static_cast<Session&>(*item.session);
            const auto table = cache_.block_table(s.sequence);
            const auto committed = cache_.committed(s.sequence);
            s.produced.clear();
            Progress p;
            p.prefill_complete = true;
            if(placement.phase == RowPhase::prefill) {
                if(placement.logits) s.produced.push_back(attend(pages_, table, committed + placement.rows));
                p.prompt_tokens = placement.rows;
                p.prefill_complete = placement.logits > 0;
                p.token_work = placement.rows;
            } else {
                // The target's prediction after each staged row, read through the pages: one verify per request.
                const auto staged = std::uint32_t(s.row->inputs().size());
                Tokens predictions;
                for(std::uint32_t i = 0; i < staged; ++i) predictions.push_back(attend(pages_, table, committed + i + 1));
                const auto acceptance = s.row->accept(predictions, item.budget.output_tokens, s.stops_aware ? std::span<const Token>(s.stops)
                                                                                                             : std::span<const Token>());
                stats_->rounds++;
                if(acceptance.matched < s.row->drafts() && acceptance.emitted < item.budget.output_tokens) stats_->rejected++;
                if(acceptance.emitted > 1) stats_->extra++;
                s.produced.assign(s.row->emitted().begin(), s.row->emitted().end());
                p.token_work = staged;
            }
            p.tokens = s.produced;
            progress.push_back({item.key, p});
        }
        stats_->largest = std::max<std::uint32_t>(stats_->largest, verifies);
        return progress;
    }
    RowProgress prefill(const WorkItem&) override { throw std::logic_error("mixed batches only"); }
    std::vector<RowProgress> decode(std::span<const WorkItem>) override { throw std::logic_error("mixed batches only"); }
    void commit(std::span<const CommitDecision> decisions) override {
        for(const auto& d:decisions) {
            auto& s = static_cast<Session&>(*d.session);
            if(s.prefill) {
                const auto round = std::exchange(s.prefill, std::nullopt)->id;
                if(d.cancelled || d.discard) { cache_.discard(s.sequence, round); continue; }
                cache_.commit(s.sequence, round, s.chunk);
                if(d.accepted) s.row.emplace(cache_, s.sequence, s.produced.front());
                continue;
            }
            if(!s.row || !s.row->staged()) continue;
            if(d.cancelled || d.discard) { s.row->discard(); continue; }
            if(d.accepted < s.row->emitted().size()) stats_->trimmed++;
            s.row->commit(d.accepted);
        }
        cache_.retire(cache_.epoch());
    }
    std::unique_ptr<Retirement> retire(RequestKey, ConcurrentSession& session, Finish) override {
        auto& s = static_cast<Session&>(session);
        if(s.prefill) { cache_.discard(s.sequence, s.prefill->id); s.prefill.reset(); }
        if(s.row) s.row->discard();
        cache_.close(s.sequence);
        cache_.retire(cache_.epoch());
        reservations_.release(std::exchange(s.reserved, 0u));
        return std::make_unique<Receipt>();
    }
    void quiesce() override {}
};

void speculative_requests() {
    const auto all = prompts();
    constexpr std::uint32_t output = 20;
    // Two of every three requests stop at their oracle's sixth token (or its first occurrence): requests with an even
    // id end their round at it themselves, the others leave the trim to Core.
    std::vector<Tokens> expected;
    std::vector<RequestOptions> options(all.size());
    for(std::size_t r = 0; r < all.size(); ++r) {
        auto want = oracle(all[r], output);
        options[r].stop_tokens = Tokens{};
        if(r % 3 != 2) {
            const auto stop = want[5];
            options[r].stop_tokens = Tokens{stop};
            want.resize(std::size_t(std::find(want.begin(), want.end(), stop) - want.begin()) + 1);
        }
        expected.push_back(std::move(want));
    }
    for(const std::uint32_t concurrency:{1u, 6u}) {
        auto stats = std::make_shared<SpeculativeStats>();
        Profile p;
        p.context = 64; p.prefill_chunk = 8;
        p.host_limit = std::size_t(64) << 20; p.device_limit = std::size_t(1) << 20;
        p.max_concurrency = concurrency; p.max_pending = 16;
        p.max_batch_tokens = 32;
        Engine engine(p, std::make_shared<Host>(), [=](const Profile&, Services services) -> std::unique_ptr<Model> {
            return std::make_unique<SpeculativeModel>(concurrency, services, stats);
        });
        std::vector<Handle> handles;
        for(std::size_t r = 0; r < all.size(); ++r) handles.push_back(engine.submit(all[r], output, options[r]));
        for(std::size_t r = 0; r < all.size(); ++r) {
            const auto result = handles[r].wait();
            require(result.tokens == expected[r], "speculative request " + std::to_string(r) + " differs from plain greedy");
            require(result.finish == (r % 3 != 2 ? Finish::stop : Finish::length), "finish " + std::to_string(r));
        }
        require(stats->rounds > 0 && stats->rejected > 0 && stats->extra > 0, "rounds accept several tokens and reject drafts");
        require(stats->trimmed > 0, "Core trims a round at a stop the request did not know");
        if(concurrency > 1) require(stats->largest > 1, "a step verifies several requests' drafts");
        else require(stats->largest == 1, "a serial engine verifies one request per step");
    }
}
}

int main(int argc, char** argv) {
    const std::array<std::pair<std::string_view, void (*)()>, 6> cases{{
        {"batch_layout", layout_rules}, {"batch_items", item_rules}, {"batch_pages", page_rules},
        {"batch_engine", engine_requests}, {"batch_speculative_rows", speculative_rules},
        {"batch_speculative_engine", speculative_requests}}};
    try {
        const std::string_view name = argc > 1 ? argv[1] : "";
        bool found = false;
        for(const auto& [label, run]:cases) if(name == "all" || name == label) {
            run(); found = true;
            std::printf("crit core batch passed: %.*s\n", int(label.size()), label.data());
        }
        require(found, "unknown batch test case");
    } catch(const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
