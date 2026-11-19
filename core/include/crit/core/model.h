#pragma once
#include "crit/core/request.h"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace crit::core {
struct ModelInfo {
    std::string implementation;
    std::uint32_t vocabulary = 0, context = 0, prefill_chunk = 0;
    std::vector<Token> stop_tokens;
    bool generation = true, scoring = false;
    std::uint32_t max_concurrency = 1;
    SamplingSupport sampling = {};
    bool request_context = false;
    // Model::score_queries: per target, log-probabilities of caller-chosen token ids and the argmax.
    bool query_scoring = false;
    bool mixed_batches = false;
    // Preferred decode work per row; execute() must also support smaller grants.
    std::uint32_t decode_token_work = 1;
    // Ids below this are real tokens; zero means vocabulary. Padding ids at or above
    // it never appear in log-probability rows, top-k rows or argmax.
    std::uint32_t valid_vocabulary = 0;
    // Progress::logprobs for generated tokens (see TokenLogprob for the definition).
    bool token_logprobs = false;
    // Most alternatives per generated token or scored target; at most top_logprobs_limit.
    std::uint32_t max_top_logprobs = 0;
    // Named scoring paths (for example prefill and decode); empty means one default path.
    std::vector<std::string> score_paths = {};
    // Mixed prefill work is granted in multiples of this quantum, including
    // padded columns. Prompt consumption remains bounded by actual input.
    // One preserves single-token grants; decode always supports them.
    std::uint32_t prefill_work_quantum = 1;
    // Prefill passes. A model may work on a span of the prompt over several steps before its tokens count: a span
    // run one layer after another, so whatever a layer needs (weights brought to the device) is paid once for the
    // whole span, in steps as short as any other. prefill_pass_tokens is the most prompt tokens a request holds in
    // such a pass (zero: no passes, every prefill step consumes prompt tokens), prefill_pass_steps the most steps
    // in a row its prefill may take without consuming any.
    std::uint32_t prefill_pass_tokens = 0, prefill_pass_steps = 0;
};
struct WorkBudget {
    std::uint32_t prefill_tokens, output_tokens;
    bool prefill_complete;
    // Mixed execution only: prompt computation, draft/verification and padding
    // must fit this grant. Token work is a quota, not a latency estimate.
    std::uint32_t token_work = 0;
};
struct Progress {
    std::uint32_t prompt_tokens = 0;
    bool prefill_complete = false;
    std::span<const Token> tokens;
    // Of prompt_tokens, those restored from state the model retained from an
    // earlier request instead of being computed. Reuse must cover a prompt
    // prefix (before any computed prompt token) and does not count against the
    // step's prefill budget; reported in RequestTiming::reused_prompt_tokens.
    std::uint32_t reused_prompt_tokens = 0;
    // Actual token work used by execute(), including hidden/padded columns.
    std::uint32_t token_work = 0;
    // The prompt tokens past those consumed that the model holds in a pass after this step (ModelInfo's
    // prefill_pass_tokens): worked on, not yet counted. A step that consumes no prompt token and produces no output
    // is progress while it reports a pass, for at most prefill_pass_steps steps in a row. The step that ends a pass
    // reports its tokens in prompt_tokens: they may exceed that step's prefill grant by the pass the row held.
    std::uint32_t pass_tokens = 0;
    // With RequestOptions::logprobs: one log-probability per token and n ranked
    // alternatives per token (row-major), borrowed like tokens.
    std::span<const float> logprobs = {};
    std::span<const TokenLogprob> top_logprobs = {};
};
enum class Finish { length, stop, cancelled, failed };
}
