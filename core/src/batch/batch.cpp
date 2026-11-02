#include "crit/batch.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace crit::core {
namespace {
constexpr std::uint64_t max_step_rows = std::numeric_limits<std::int32_t>::max();
}

void StepLayout::arrange(std::span<const StepItem> items) {
    std::uint64_t total = 0;
    for(const auto& item:items) {
        if(!item.rows) throw std::invalid_argument("a step item needs rows");
        if(item.logits > item.rows) throw std::invalid_argument("a step item has more logits rows than rows");
        if(item.phase == RowPhase::decode && !item.logits) throw std::invalid_argument("a decode item needs logits");
        if(item.phase != RowPhase::decode && item.phase != RowPhase::prefill) throw std::invalid_argument("unknown row phase");
        total += item.rows;
        if(total > max_step_rows) throw std::invalid_argument("a step holds at most 2^31 - 1 rows");
    }
    placements_.clear();
    logit_rows_.clear();
    by_item_.assign(items.size(), 0);
    rows_ = decode_rows_ = decode_items_ = 0;
    std::uint32_t logits = 0;
    for(const auto phase:{RowPhase::decode, RowPhase::prefill}) {
        for(std::size_t i = 0; i < items.size(); ++i) {
            const auto& item = items[i];
            if(item.phase != phase) continue;
            by_item_[i] = std::uint32_t(placements_.size());
            placements_.push_back({item.key, phase, std::uint32_t(i), rows_, item.rows, logits, item.logits});
            for(std::uint32_t r = item.rows - item.logits; r < item.rows; ++r) logit_rows_.push_back(rows_ + r);
            rows_ += item.rows;
            logits += item.logits;
            if(phase == RowPhase::decode) { decode_rows_ += item.rows; ++decode_items_; }
        }
    }
}
std::span<const StepPlacement> StepLayout::placements() const noexcept { return placements_; }
const StepPlacement& StepLayout::placement(std::size_t item) const {
    if(item >= by_item_.size()) throw std::out_of_range("step item outside the layout");
    return placements_[by_item_[item]];
}
std::span<const std::uint32_t> StepLayout::logit_rows() const noexcept { return logit_rows_; }
std::uint32_t StepLayout::rows() const noexcept { return rows_; }
std::uint32_t StepLayout::decode_rows() const noexcept { return decode_rows_; }
std::uint32_t StepLayout::decode_items() const noexcept { return decode_items_; }

StepItem step_item(const WorkItem& work, std::uint32_t prompt_tokens, std::uint32_t frontier, std::uint32_t decode_rows) {
    const auto& budget = work.budget;
    if(budget.prefill_complete) {
        if(!decode_rows) throw std::invalid_argument("a decode item stages at least one row");
        const auto rows = std::max(1u, budget.token_work ? std::min(decode_rows, budget.token_work) : decode_rows);
        return {work.key, RowPhase::decode, rows, rows};
    }
    if(frontier >= prompt_tokens) throw std::invalid_argument("a prefill item past the end of its prompt");
    auto rows = std::min(budget.prefill_tokens, prompt_tokens - frontier);
    if(budget.token_work) rows = std::min(rows, budget.token_work);
    if(!rows) throw std::invalid_argument("a prefill item without prompt rows");
    return {work.key, RowPhase::prefill, rows, frontier + rows == prompt_tokens ? 1u : 0u};
}

std::uint32_t row_pages(std::uint64_t rows, std::uint32_t page_tokens) {
    if(!page_tokens) throw std::invalid_argument("page_tokens must be positive");
    const auto pages = rows / page_tokens + (rows % page_tokens ? 1 : 0);
    if(pages > std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("page count exceeds 2^32 - 1");
    return std::uint32_t(pages);
}

PageReservations::PageReservations(Config config) : config_(config) {
    if(!config.pages || !config.page_tokens || !config.max_tokens)
        throw std::invalid_argument("page reservations need pages, page_tokens and max_tokens");
    if(!config.verify_rows || config.verify_rows > config.max_tokens)
        throw std::invalid_argument("verify_rows must be in [1, max_tokens]");
    if(std::uint64_t(config.pages) * config.page_tokens < config.max_tokens)
        throw std::invalid_argument("max_tokens exceeds pages x page_tokens");
}
const PageReservations::Config& PageReservations::config() const noexcept { return config_; }
std::uint32_t PageReservations::need(const RequestView& view) const noexcept {
    const auto rows = std::min<std::uint64_t>(std::uint64_t(view.prompt.size()) + view.output_tokens + config_.verify_rows - 1,
                                              config_.max_tokens);
    return std::uint32_t((rows + config_.page_tokens - 1) / config_.page_tokens);
}
Admission PageReservations::admission(const RequestView& view) const noexcept {
    if(view.prompt.empty() || std::uint64_t(view.prompt.size()) + view.output_tokens > config_.max_tokens)
        return Admission::infeasible;
    return need(view) <= config_.pages - reserved_ ? Admission::ready : Admission::deferred;
}
std::uint32_t PageReservations::reserve(const RequestView& view) {
    if(admission(view) != Admission::ready) throw std::logic_error("page reservation for a request admission does not admit");
    const auto pages = need(view);
    reserved_ += pages;
    return pages;
}
void PageReservations::release(std::uint32_t pages) {
    if(pages > reserved_) throw std::logic_error("releasing more pages than are reserved");
    reserved_ -= pages;
}
std::uint32_t PageReservations::reserved() const noexcept { return reserved_; }

SpeculativeRow::SpeculativeRow(StateCache& cache, SequenceId sequence, Token pending)
    : cache_(&cache), sequence_(sequence), pending_(pending) {}
const Round& SpeculativeRow::begin(std::span<const Token> drafts) {
    if(round_) throw std::logic_error("a speculative row already has a staged round");
    const auto& config = cache_->config();
    const auto committed = cache_->committed(sequence_);
    if(committed >= config.max_tokens) throw std::length_error("a speculative row has no row left for its pending token");
    const auto room = std::min<std::uint64_t>(config.max_round_rows - 1, config.max_tokens - committed - 1);
    const auto kept = std::size_t(std::min<std::uint64_t>(drafts.size(), room));
    inputs_.assign(1, pending_);
    inputs_.insert(inputs_.end(), drafts.begin(), drafts.begin() + std::ptrdiff_t(kept));
    acceptance_.reset();
    emitted_.clear();
    round_ = cache_->begin(sequence_, std::uint32_t(inputs_.size()));
    return *round_;
}
std::span<const Token> SpeculativeRow::inputs() const noexcept { return inputs_; }
std::uint32_t SpeculativeRow::drafts() const noexcept { return inputs_.empty() ? 0 : std::uint32_t(inputs_.size() - 1); }
Acceptance SpeculativeRow::accept(std::span<const Token> predictions, std::uint32_t remaining, std::span<const Token> stops) {
    if(!round_) throw std::logic_error("a speculative row accepts only a staged round");
    if(predictions.size() != inputs_.size()) throw std::invalid_argument("a speculative round needs one prediction per staged row");
    const auto acceptance = accept_greedy(std::span<const Token>(inputs_).subspan(1), predictions, remaining, stops);
    emitted_.assign(predictions.begin(), predictions.begin() + std::ptrdiff_t(acceptance.emitted));
    acceptance_ = acceptance;
    return acceptance;
}
std::span<const Token> SpeculativeRow::emitted() const noexcept { return emitted_; }
void SpeculativeRow::commit(std::uint32_t accepted) {
    if(!round_ || !acceptance_) throw std::logic_error("a speculative row commits only an accepted round");
    if(accepted > emitted_.size()) throw std::out_of_range("a speculative row keeps at most the tokens it emitted");
    if(!accepted) { discard(); return; }
    // Rows 0 .. accepted - 1 hold the pending token and the first accepted - 1 emitted tokens.
    try { cache_->commit(sequence_, round_->id, std::span<const Token>(inputs_).first(accepted)); }
    catch(...) {
        // A rejected commit consumes the round; a throwing snapshot copy leaves it pending in the cache, so the row
        // keeps it too and discard() or another commit can still end it.
        bool pending = false;
        try { pending = cache_->staged(sequence_) != 0; } catch(...) {}
        if(!pending) { round_.reset(); acceptance_.reset(); }
        throw;
    }
    round_.reset();
    acceptance_.reset();
    pending_ = emitted_[accepted - 1];
}
void SpeculativeRow::discard() {
    if(round_) cache_->discard(sequence_, round_->id);
    round_.reset();
    acceptance_.reset();
    emitted_.clear();
}
Token SpeculativeRow::pending() const noexcept { return pending_; }
bool SpeculativeRow::staged() const noexcept { return round_.has_value(); }
SequenceId SpeculativeRow::sequence() const noexcept { return sequence_; }
}
