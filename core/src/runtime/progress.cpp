// Checks and output staging for each step of a request.
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
}
void validate_tokens(std::span<const Token> tokens, std::uint32_t vocabulary,
                     const char* message) {
    for(auto token:tokens) if(token < 0 || std::uint32_t(token) >= vocabulary)
        throw std::invalid_argument(message);
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
    if(progress.logprobs.size() != count || progress.top_logprobs.size() != count*width) throw std::runtime_error("logprob rows count mismatch");
    LogprobRows rows{progress.logprobs.first(count),progress.top_logprobs.first(count*width),width};
    if(!std::all_of(rows.chosen.begin(),rows.chosen.end(),logprob_value))
        throw std::runtime_error("logprob value is nan, infinite or positive");
    validate_ranked(rows.top,width,valid);
    return rows;
}
Decision decide(std::span<const Token> tokens, std::span<const Token> stops, std::uint32_t remaining) {
    Decision d{std::uint32_t(tokens.size())};
    for(std::uint32_t i = 0; i < d.accepted; ++i) if(std::find(stops.begin(),stops.end(),tokens[i]) != stops.end()) {
        d.accepted = i+1; d.finish = Finish::stop; break;
    }
    d.terminal = d.finish == Finish::stop || d.accepted == remaining;
    return d;
}
Extents extents(const Result& r) {
    return {r.tokens.size(),r.logprobs.size(),r.top_logprobs.size()};
}
void restore(Result& r, const Extents& e) {
    r.tokens.resize(e.tokens); r.logprobs.resize(e.logprobs); r.top_logprobs.resize(e.top);
}
void stage_output(Request& r, const LogprobRows& rows, std::uint32_t accepted) {
    auto& result = r.result;
    if(r.options.logprobs) {
        result.logprobs.insert(result.logprobs.end(),rows.chosen.begin(),rows.chosen.begin()+accepted);
        result.top_logprobs.insert(result.top_logprobs.end(),rows.top.begin(),rows.top.begin()+std::ptrdiff_t(accepted*rows.width));
    }
}
}
