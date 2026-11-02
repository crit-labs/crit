#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace crit::core {
using Token = std::int32_t;
// Most alternatives any log-probability row may carry.
inline constexpr std::uint32_t top_logprobs_limit = 64;
// Additive logit bias: finite, or negative infinity to ban the token.
struct LogitBias { Token token = 0; float bias = 0; };
struct SamplingOptions {
    // Unset uses the adapter default. Temperature zero requests greedy output;
    // top_k zero disables that filter. crit/sampling.h implements these rules.
    std::optional<double> temperature, top_p;
    std::optional<std::uint32_t> top_k;
    std::optional<std::uint64_t> seed;
    // In [0, 1]: drop tokens whose probability is below min_p times the largest.
    std::optional<double> min_p = {};
    // Repetition (finite, > 0, 1 is off) covers prompt and output tokens; frequency
    // and presence (finite, in [-2, 2]) cover output tokens only.
    std::optional<double> repetition_penalty = {}, frequency_penalty = {}, presence_penalty = {};
    // Unique in-vocabulary tokens, at most 1024 entries.
    std::vector<LogitBias> logit_bias = {};
    // Stop tokens are suppressed until this many tokens were generated.
    std::optional<std::uint32_t> min_tokens = {};
};
struct SamplingSupport {
    bool temperature = false, top_p = false, top_k = false, seed = false;
    bool min_p = false, penalties = false, logit_bias = false, min_tokens = false;
};
class RequestContext {
public:
    virtual ~RequestContext() = default;
    // Immutable host-only extension parameters. Report their retained payload
    // extent; Core charges it conservatively for each request, even when shared.
    // This metadata call must not block, enqueue work, or change the extent.
    virtual std::size_t host_bytes() const noexcept = 0;
};
struct RequestOptions {
    SamplingOptions sampling;
    // Unset inherits ModelInfo.stop_tokens; an empty vector disables token stops.
    std::optional<std::vector<Token>> stop_tokens;
    std::shared_ptr<const RequestContext> context;
    // Generation log-probabilities. Unset requests none; n returns each token's own
    // log-probability plus its n most likely alternatives. Requires
    // ModelInfo::token_logprobs and n <= ModelInfo::max_top_logprobs.
    std::optional<std::uint32_t> logprobs = {};
    // Text stops matched on the decoded output (at most 16 non-empty strings of at
    // most 256 bytes). Requires an independent text frontend. A match ends the
    // request with Finish::stop after the token that completed it; Result::text
    // excludes the stop text.
    std::vector<std::string> stop_strings = {};
    // Decode accepted tokens into Result::text and OutputEvent::text. Implied by
    // non-empty stop_strings; requires an independent text frontend.
    bool text = false;
};
// A token and its natural-log probability under the raw logits of the pass that
// produced or scored the position: before temperature, penalties, bias and filters,
// normalized over the valid vocabulary. Rows of alternatives are ordered by
// log-probability descending, then token id ascending.
struct TokenLogprob { Token token = 0; float logprob = 0; };
// Query scoring (for example against a reference's top-k tokens): `ids` holds `per_target` token ids for each
// scored target, row-major in target order; per_target zero with no ids requests target log-probabilities only.
struct ScoreQueries {
    std::vector<Token> ids;
    std::uint32_t per_target = 0;
    // The model's own top_k alternatives per target; requires ModelInfo::max_top_logprobs >= top_k.
    std::uint32_t top_k = 0;
    // One of ModelInfo::score_paths; empty selects the default path.
    std::string path = {};
};
// One natural-log probability per target, targets x per_target query log-probabilities (finite or negative
// infinity) and, optionally, the model's argmax per target (empty when not provided).
struct QueryScores {
    std::vector<float> target;
    std::vector<float> queries;
    std::vector<Token> argmax;
    // targets x top_k ranked alternatives when ScoreQueries::top_k is nonzero.
    std::vector<TokenLogprob> top = {};
};
}
