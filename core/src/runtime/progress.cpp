// Checks and output staging that serial sessions and concurrent rows share for each step of a request.
#include "request.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace crit::core::detail {
namespace {
// Log-probabilities may exceed zero only by accumulated rounding.
constexpr float logprob_tolerance = 1e-3F;
// Grows result.text's charged capacity to `needed` bytes in doubling chunks of at least 256 bytes, or by the exact
// need near the ceiling. Call with the request lock held.
void reserve_text(Request& r, Resources& resources, std::size_t needed) {
    if(needed > std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("text offsets exceed 32 bits");
    if(needed <= r.text_charged) return;
    const auto need = needed-r.text_charged, chunk = std::max({need,r.text_charged,std::size_t(256)});
    Reservation grant;
    try { grant = resources.reserve(Memory::host,chunk); }
    catch(const std::runtime_error&) { if(chunk == need) throw; grant = resources.reserve(Memory::host,need); }
    r.text_storage.push_back(std::move(grant));
    r.text_charged += r.text_storage.back().bytes();
    r.result.text.reserve(r.text_charged);
}
}
void validate_tokens(std::span<const Token> tokens, std::uint32_t vocabulary,
                     const char* message) {
    for(auto token:tokens) if(token < 0 || std::uint32_t(token) >= vocabulary)
        throw std::invalid_argument(message);
}
void validate_deferred_progress(const Progress& p, const char* message) {
    if(p.deferred_tokens && (!p.readiness || !p.tokens.empty() || !p.prefill_complete))
        throw std::runtime_error(message);
}
void validate_progress(const Progress& p, const WorkBudget& budget, std::uint32_t frontier,
                       const Request& request, const char* message) {
    const auto next_frontier = frontier + std::uint64_t(p.prompt_tokens);
    if(p.reused_prompt_tokens > p.prompt_tokens || p.prompt_tokens - p.reused_prompt_tokens > budget.prefill_tokens ||
       (p.reused_prompt_tokens && !request.phases.reuse_allowed()) ||
       next_frontier > request.prompt.size() || (budget.prefill_complete && p.prompt_tokens) ||
       (budget.prefill_complete && !p.prefill_complete) ||
       p.prefill_complete != (next_frontier == request.prompt.size()) ||
       (!p.prefill_complete && !p.tokens.empty()) || p.tokens.size() > budget.output_tokens ||
       (!p.prompt_tokens && p.tokens.empty()))
        throw std::runtime_error(message);
}
bool logprob_value(float x) { return !std::isnan(x) && x <= logprob_tolerance; }
void validate_ranked(std::span<const TokenLogprob> rows, std::size_t width, std::uint32_t valid) {
    std::array<Token,top_logprobs_limit> ids{};
    for(std::size_t start = 0; width && start < rows.size(); start += width) {
        const auto row = rows.subspan(start,width);
        for(std::size_t k = 0; k < width; ++k) {
            if(!logprob_value(row[k].logprob)) throw std::runtime_error("logprob value is nan, infinite or positive");
            if(row[k].token < 0 || std::uint32_t(row[k].token) >= valid)
                throw std::runtime_error("logprob row token outside the valid vocabulary");
            if(k && row[k].logprob > row[k-1].logprob) throw std::runtime_error("top logprob row is not ranked");
            ids[k] = row[k].token;
        }
        std::sort(ids.begin(),ids.begin()+std::ptrdiff_t(width));
        if(std::adjacent_find(ids.begin(),ids.begin()+std::ptrdiff_t(width)) != ids.begin()+std::ptrdiff_t(width))
            throw std::runtime_error("top logprob row ids are not distinct");
    }
}
LogprobRows logprob_rows(const RequestOptions& options, const Progress& progress, std::uint32_t valid) {
    if(!options.logprobs) return {};
    const std::size_t width = *options.logprobs, count = progress.tokens.size();
    const bool counts = progress.deferred_tokens
        ? progress.logprobs.size() >= count && progress.top_logprobs.size() >= count*width
        : progress.logprobs.size() == count && progress.top_logprobs.size() == count*width;
    if(!counts) throw std::runtime_error("logprob rows count mismatch");
    LogprobRows rows{progress.logprobs.first(count),progress.top_logprobs.first(count*width),width};
    if(!std::all_of(rows.chosen.begin(),rows.chosen.end(),logprob_value))
        throw std::runtime_error("logprob value is nan, infinite or positive");
    validate_ranked(rows.top,width,valid);
    return rows;
}
Decision decide(std::span<const Token> tokens, std::span<const Token> stops, std::uint32_t remaining,
                TextStream* text, StepText& out) {
    Decision d{std::uint32_t(tokens.size())};
    for(std::uint32_t i = 0; i < d.accepted; ++i) if(std::find(stops.begin(),stops.end(),tokens[i]) != stops.end()) {
        d.accepted = i+1; d.finish = Finish::stop; break;
    }
    d.terminal = d.finish == Finish::stop || d.accepted == remaining;
    if(!text || !d.accepted) return d;
    const auto decoded = d.finish == Finish::stop ? d.accepted-1 : d.accepted;
    const auto offset = [&]{
        if(out.bytes.size() > std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("text offsets exceed 32 bits");
        return std::uint32_t(out.bytes.size());
    };
    for(std::uint32_t i = 0; i < decoded; ++i) {
        const auto piece = text->push(tokens.subspan(i,1));
        out.bytes += piece.text;
        out.ends.push_back(offset());
        if(piece.stopped) return {i+1,Finish::stop,true};
    }
    if(d.terminal) {
        const auto tail = text->finish();
        out.bytes += tail.text;
        if(tail.stopped) d.finish = Finish::stop;
        if(decoded < d.accepted) out.ends.push_back(offset()); else out.ends.back() = offset();
    }
    return d;
}
Extents extents(const Result& r) {
    return {r.tokens.size(),r.logprobs.size(),r.top_logprobs.size(),r.text.size(),r.text_ends.size()};
}
void restore(Result& r, const Extents& e) {
    r.tokens.resize(e.tokens); r.logprobs.resize(e.logprobs); r.top_logprobs.resize(e.top);
    r.text.resize(e.text); r.text_ends.resize(e.text_ends);
}
void stage_output(Request& r, Resources& resources, const LogprobRows& rows, std::uint32_t accepted, const StepText& text) {
    auto& result = r.result;
    if(r.text) {
        reserve_text(r,resources,result.text.size()+text.bytes.size());
        const auto base = std::uint32_t(result.text.size());
        result.text += text.bytes;
        for(const auto end:text.ends) result.text_ends.push_back(base+end);
    }
    if(r.options.logprobs) {
        result.logprobs.insert(result.logprobs.end(),rows.chosen.begin(),rows.chosen.begin()+accepted);
        result.top_logprobs.insert(result.top_logprobs.end(),rows.top.begin(),rows.top.begin()+std::ptrdiff_t(accepted*rows.width));
    }
}
}
