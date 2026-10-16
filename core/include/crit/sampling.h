#pragma once
#include <crit/core.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Host and device annotation for the header-only generator, so device kernels can reproduce draws bit for bit.
#ifndef CRIT_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CRIT_HOST_DEVICE __host__ __device__
#else
#define CRIT_HOST_DEVICE
#endif
#endif

namespace crit::core {
// Every draw is a pure function of (seed, position, stream, purpose): rollback, stop/budget trimming,
// cancellation and speculative redo need no generator state. Position = committed outputs before the
// draw plus the verify column.
enum class DrawPurpose : std::uint32_t { sample = 0, draft = 1, accept = 2, residual = 3, user = 16 };
// The four 32-bit output words of one Philox4x32 block, in Random123 order (a plain array is device-safe).
struct PhiloxBlock { std::uint32_t word[4] = {}; };
// Philox4x32-10 with the Random123 constants. Key = {seed low, seed high};
// counter = {position low, position high, stream, purpose}.
CRIT_HOST_DEVICE constexpr PhiloxBlock philox4x32(std::uint64_t seed, std::uint64_t position, std::uint32_t stream,
                                                  std::uint32_t purpose) noexcept {
    std::uint32_t c0 = static_cast<std::uint32_t>(position), c1 = static_cast<std::uint32_t>(position >> 32);
    std::uint32_t c2 = stream, c3 = purpose;
    std::uint32_t k0 = static_cast<std::uint32_t>(seed), k1 = static_cast<std::uint32_t>(seed >> 32);
    for(int round = 0; round < 10; ++round) {
        if(round) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        const std::uint64_t p0 = std::uint64_t{0xD2511F53u} * c0, p1 = std::uint64_t{0xCD9E8D57u} * c2;
        const std::uint32_t hi0 = static_cast<std::uint32_t>(p0 >> 32), lo0 = static_cast<std::uint32_t>(p0);
        const std::uint32_t hi1 = static_cast<std::uint32_t>(p1 >> 32), lo1 = static_cast<std::uint32_t>(p1);
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
    }
    return {{c0, c1, c2, c3}};
}
// [0, 1) with 53 random bits: ((uint64(word[0]) << 32 | word[1]) >> 11) * 2^-53.
CRIT_HOST_DEVICE constexpr double uniform(std::uint64_t seed, std::uint64_t position, std::uint32_t stream = 0,
                                          DrawPurpose purpose = DrawPurpose::sample) noexcept {
    const PhiloxBlock block = philox4x32(seed, position, stream, static_cast<std::uint32_t>(purpose));
    const std::uint64_t bits = (std::uint64_t{block.word[0]} << 32 | block.word[1]) >> 11;
    return static_cast<double>(bits) * 0x1.0p-53;
}

// Most logit_bias entries one request may carry.
inline constexpr std::size_t logit_bias_limit = 1024;
// Resolved sampling rules for one request; validate() states the ranges. core/src/sampling/README.md gives the order.
struct SamplingParams {
    double temperature = 1.0, top_p = 1.0, min_p = 0.0;                // temperature 0 = greedy
    std::uint32_t top_k = 0;                                            // 0 = no filter
    double repetition_penalty = 1.0, frequency_penalty = 0.0, presence_penalty = 0.0;
    std::vector<LogitBias> logit_bias = {};                             // unique in-vocabulary ids; -inf bans
    std::uint32_t min_tokens = 0;                                       // suppress the list below until reached
    std::vector<Token> suppress_until_min = {};                         // usually the stop tokens; see validate()
    std::uint64_t seed = 0;                                             // an unset request seed takes this one
    std::uint32_t stream = 0;                                           // e.g. the choice index when n > 1
};
// Set request fields override `defaults` (a non-empty logit_bias replaces the default list); suppress_until_min
// and stream come from `defaults`. An unset request seed takes defaults.seed, so unseeded requests sharing one
// defaults object draw identical u per position: put a fresh random seed in `defaults` per request (and record it
// for replay) unless shared draws are intended. `vocabulary` is the valid vocabulary the Sampler rows hold; padding
// stop tokens in suppress_until_min are accepted and ignored. The result is validated; invalid_argument names the
// offending field.
SamplingParams resolve_sampling(const SamplingOptions&, const SamplingParams& defaults, std::uint32_t vocabulary);
// invalid_argument unless: vocabulary > 0; temperature finite >= 0; top_p finite in (0, 1]; min_p finite in [0, 1];
// repetition_penalty finite > 0; frequency/presence finite in [-2, 2]; logit_bias at most logit_bias_limit unique
// ids < vocabulary with finite or -inf bias; suppress_until_min ids nonnegative when min_tokens > 0 (the list is
// unused otherwise). suppress_until_min ids >= vocabulary are ignored, like padding stop tokens, which can never be
// drawn. top_k is unbounded: at or above the selectable count it keeps every token (the runtime bounds it by the
// declared, possibly padded vocabulary).
void validate(const SamplingParams&, std::uint32_t vocabulary);

// Penalty context on the same frontier as KV: the prompt, committed outputs and staged (provisional) outputs as
// sparse per-token counts, never O(vocabulary). A draw at verify column c sees committed outputs plus staged[0, c).
// Memory grows with the distinct tracked tokens and the largest staged round, not the prompt length.
// Limits: 2^31 - 1 prompt tokens and 2^31 - 1 outputs. Not thread-safe; one owner per sequence.
class TokenHistory {
public:
    // With count_prompt_for_repetition false the repetition penalty covers outputs only. invalid_argument on a
    // negative token; runtime_error above the prompt limit.
    explicit TokenHistory(std::span<const Token> prompt, bool count_prompt_for_repetition = true);
    // A copy (e.g. a fork per choice, even between stage and commit) keeps the source's reserved room, so its
    // commit() never allocates either; it is a separate history for host_bytes. A move leaves the source empty.
    TokenHistory(const TokenHistory&);
    TokenHistory& operator=(const TokenHistory&);         // strong guarantee: builds the copy, then swaps
    TokenHistory(TokenHistory&&) noexcept;
    TokenHistory& operator=(TokenHistory&&) noexcept;
    // Provisional outputs (the tokens handed to Core, or verify drafts). logic_error while tokens are staged;
    // invalid_argument on a negative token; overflow_error past the output limit. Allocates everything commit needs.
    void stage(std::span<const Token> provisional);
    // Keeps the first `accepted` staged tokens as outputs and drops the rest; out_of_range above staged().
    // Never allocates, and changes nothing when it throws.
    void commit(std::size_t accepted);
    void rollback() noexcept;                              // drops staged tokens
    std::uint64_t outputs() const noexcept;                // committed outputs = next draw position
    std::size_t staged() const noexcept;
    std::span<const Token> staged_tokens() const noexcept;
    std::uint32_t output_count(Token) const noexcept;      // committed occurrences of one token
    std::size_t distinct() const noexcept;                 // tracked tokens: counted prompt plus committed outputs
    // Upper bound on the heap bytes one history holds, transient growth included, with at most `prompt` distinct
    // prompt tokens (the prompt length bounds them) and `outputs` committed plus staged outputs; reserve it through
    // Resources. Saturates at SIZE_MAX.
    static std::size_t host_bytes(std::size_t prompt, std::size_t outputs) noexcept;
private:
    struct Entry { Token token = 0; std::uint32_t outputs = 0; bool prompt = false; };
    std::vector<Entry> entries_;
    std::vector<std::uint32_t> table_;                     // open addressing: entry index + 1, zero is empty
    std::vector<Token> staged_;
    std::uint64_t outputs_ = 0;
    void reserve(std::size_t entries);                     // room for `entries` tracked tokens; strong guarantee
    friend class Sampler;
};

// The chosen token and its raw log-probability (the TokenLogprob definition, over the given row).
struct Choice { Token token = 0; float logprob = 0; };
// The sorted unique token ids a constraint permits at one position (TokenMatcher::allowed() in crit/constraints.h).
struct AllowedTokens { std::span<const Token> ids = {}; };
// One per worker thread. Owns O(vocabulary) scratch allocated once; sample() never allocates and distribution()
// allocates only to grow `out`. Logits rows hold exactly vocabulary() entries: pass only the valid vocabulary.
// Errors: invalid_argument("logits must be finite or -inf") for NaN or +inf, invalid_argument("no selectable
// token") when every token is masked after processing, validate()'s errors, out_of_range past the staged column.
class Sampler {
public:
    explicit Sampler(std::uint32_t vocabulary);           // invalid_argument for 0 or more than 2^31
    Sampler(Sampler&&) noexcept;                          // the source keeps vocabulary 0
    Sampler& operator=(Sampler&&) noexcept;
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    std::uint32_t vocabulary() const noexcept;
    std::size_t scratch_bytes() const noexcept;
    static std::size_t scratch_bytes(std::uint32_t vocabulary) noexcept;   // what the constructor allocates
    // Order: bias/bans/min_tokens suppression -> repetition -> frequency/presence -> greedy if T == 0
    // (lowest id on ties) -> /T -> top_k -> softmax -> top_p (smallest rank prefix reaching p, keep >= 1)
    // -> min_p (keep p >= min_p * p_max) -> u = uniform(seed, history.outputs()+column, stream) over the
    // CDF walked in rank order. `top` gets top.size() raw alternatives, ranked (logprob desc, id asc).
    Choice sample(std::span<const float> logits, const SamplingParams&, const TokenHistory&,
                  std::size_t column = 0, std::span<TokenLogprob> top = {});
    // The exact law sample() draws from: the kept tokens in rank order with natural-log probabilities.
    void distribution(std::span<const float> logits, const SamplingParams&, const TokenHistory&,
                      std::size_t column, std::vector<TokenLogprob>& out);
    // Constrained forms (structured output, crit/constraints.h): every id outside `allowed` is masked to -inf
    // before step 1's bias, so the filters see only the allowed tokens and a filter cannot empty the set. The
    // returned logprob and `top` stay raw over the whole row. invalid_argument("allowed tokens must be nonempty,
    // sorted, unique and in the vocabulary") otherwise; "no selectable token" when bans or min_tokens suppression
    // remove every allowed token. Without these forms behavior is unchanged.
    Choice sample(std::span<const float> logits, AllowedTokens allowed, const SamplingParams&, const TokenHistory&,
                  std::size_t column = 0, std::span<TokenLogprob> top = {});
    void distribution(std::span<const float> logits, AllowedTokens allowed, const SamplingParams&,
                      const TokenHistory&, std::size_t column, std::vector<TokenLogprob>& out);
private:
    struct Candidate { double value = 0, weight = 0; std::uint32_t id = 0, bin = 0; };
    struct Bin { double weight = 0; std::size_t count = 0; };
    struct Law;
    std::uint32_t vocabulary_ = 0;
    std::vector<Candidate> candidates_;
    std::vector<std::uint32_t> counts_;                   // staged occurrences; all zero between calls
    std::vector<Bin> bins_;                               // log-weight histogram of the kept tokens
    std::vector<Token> bias_ids_;                         // validation scratch, logit_bias_limit capacity
    Law prepare(std::span<const float> logits, const SamplingParams&, const TokenHistory&, std::size_t column,
                const AllowedTokens* allowed);
    Choice sample_law(std::span<const float> logits, const SamplingParams&, const TokenHistory&, std::size_t column,
                      std::span<TokenLogprob> top, const AllowedTokens* allowed);
    void distribution_law(std::span<const float> logits, const SamplingParams&, const TokenHistory&, std::size_t column,
                          std::vector<TokenLogprob>& out, const AllowedTokens* allowed);
    void penalize(const SamplingParams&, const TokenHistory&, std::size_t column) noexcept;
    void sort_prefix(std::size_t need, std::size_t count, std::size_t& sorted);
    void sort_to_target(double target, std::size_t count, std::size_t& sorted);
};

// Raw-row helpers: natural-log softmax with double accumulation over exactly the given row. Logits are finite or
// -inf (invalid_argument "logits must be finite or -inf"); a row without a finite entry is invalid_argument
// "no selectable token". Masked entries get -inf. Outputs that overlap the row are invalid_argument, except an exact
// alias in log_softmax.
void log_softmax(std::span<const float> logits, std::span<float> out);              // same size; may alias exactly
void top_logprobs(std::span<const float> logits, std::span<TokenLogprob> out);      // out.size() best, ranked
void query_logprobs(std::span<const float> logits, std::span<const Token> ids, std::span<float> out);
// `rows` logit rows; row r starts at logits[r * stride] and its first `valid` entries are the valid vocabulary.
struct LogitRows { std::span<const float> logits = {}; std::size_t stride = 0; std::uint32_t rows = 0, valid = 0; };
// Fills a QueryScores for score_queries(): one target log-probability per row, rows x per_target query
// log-probabilities, the argmax per row (lowest id on ties) when requested, and rows x top_k ranked alternatives.
// Ids must be below `valid`; a masked target yields -inf, which the runtime rejects as non-finite.
QueryScores score_logit_rows(const LogitRows&, std::span<const Token> targets, std::span<const Token> queries,
                             std::uint32_t per_target, std::uint32_t top_k, bool argmax);
// Greedy verification of one round: drafts[i] is checked against predictions[i] (predictions has one more entry).
// Emitted tokens are always predictions[0, emitted): the matched drafts, then a correction or bonus unless a stop
// was emitted or `remaining` (the output budget) is spent. stopped: the last emitted token is a stop.
struct Acceptance { std::uint32_t emitted = 0, matched = 0; bool stopped = false; };
Acceptance accept_greedy(std::span<const Token> drafts, std::span<const Token> predictions,
                         std::uint32_t remaining, std::span<const Token> stops);
}
