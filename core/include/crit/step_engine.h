#pragma once
// Step engines: a ConcurrentModel whose model code lives behind C callbacks, for engines written in another language
// (crit_core.native.StepEngine drives one from Python through ctypes). Core's runtime keeps everything else: the
// scheduler with mixed prefill/decode batches, admission, request options, stop tokens, cancellation, retirement and
// the scoring barrier. The engine implements a step: given one row per scheduled request, it computes them together
// and reports each row's progress and tokens. The callbacks run on the engine's one worker thread, never at once.
//
// Shared library: Core builds crit_core_steps, which exports the native engine ABI (crit/native_engine.h) plus
//   void* crit_engine_open_steps(const char* configuration_json, const crit::core::StepCallbacks* callbacks)
// returning an engine handle the other crit_engine_* functions take. The callbacks and their user pointer must stay
// valid until crit_engine_close returns.
#include <crit/native_engine.h>
#include <cstdint>
#include <string_view>

namespace crit::core {
// The structures are standard layout with fixed-width fields, so other languages can mirror them.
// One request: its key, prompt and output limit, and its options as a JSON object: the sampling fields it set
// ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty", "frequency_penalty", "presence_penalty",
// "logit_bias" as {"id": bias}, a banned token's bias as -Infinity (Python's json spelling), "min_tokens"),
// "stop_tokens" (the effective list), "logprobs" (alternatives per token),
// "replay" (a forced continuation, ReplayContinuation) and "plain" (decode without drafting, PlainDecoding).
struct StepRequest {
    std::uint64_t request = 0, generation = 0;
    std::uint32_t lane = 0, prompt_tokens = 0;
    const std::int32_t* prompt = nullptr;
    std::uint32_t output_tokens = 0, reserved = 0;
    const char* options_json = nullptr;
};
// One scheduled row: the session the engine's open returned and the row's WorkBudget.
struct StepRow {
    std::uint64_t session = 0;
    std::uint32_t prefill_tokens = 0, output_tokens = 0, token_work = 0;
    std::int32_t prefill_complete = 0;
};
// What the engine reports per row (Core's Progress): prompt tokens consumed this step (reused ones included), the
// reused prefix, the token work actually computed (mixed batches; zero takes the prompt tokens computed, or one per
// token), the tokens produced and whether prefill is complete. The tokens go to tokens[row * max_row_tokens ...]; with
// token log-probabilities, each token's into logprobs[row * max_row_tokens + i] and its alternatives into
// top_ids/top_logprobs[(row * max_row_tokens + i) * max_top_logprobs ...], ranked as Core's TokenLogprob rows are.
struct StepProgress {
    std::uint32_t prompt_tokens = 0, reused_prompt_tokens = 0, token_work = 0, tokens = 0;
    std::int32_t prefill_complete = 0, reserved = 0;
};
// Core's CommitDecision for one row of the last step: the tokens Core kept (stops and the output limit trim them),
// whether the request ends, was cancelled, or this row is to be discarded.
struct StepDecision {
    std::uint64_t session = 0;
    std::uint32_t accepted = 0;
    std::int32_t terminal = 0, cancelled = 0, discard = 0;
};
// Every callback but release returns 0 on success; any other value fails the engine, with error()'s message. Optional
// callbacks may be null.
struct StepCallbacks {
    void* user = nullptr;
    // Optional (null: always ready): 0 ready, 1 deferred, 2 infeasible (the request fails alone). Inspection only.
    int (*admission)(void* user, const StepRequest*) = nullptr;
    // A new session for the request; *session is the engine's handle for it in every later call.
    int (*open)(void* user, const StepRequest*, std::uint64_t* session) = nullptr;
    // One step over `count` rows (one per scheduled request, in any order). Buffers are sized from the configuration's
    // max_row_tokens and max_top_logprobs; the logprob buffers are null without token_logprobs.
    int (*step)(void* user, const StepRow* rows, std::uint32_t count, StepProgress* progress, std::int32_t* tokens,
                float* logprobs, std::int32_t* top_ids, float* top_logprobs) = nullptr;
    // Optional: Core's decisions for every row of the last step, before it publishes their tokens.
    int (*commit)(void* user, const StepDecision* decisions, std::uint32_t count) = nullptr;
    // Optional: the request ended (Finish as int: 0 length, 1 stop, 2 cancelled, 3 failed) and its session retires.
    int (*retire)(void* user, std::uint64_t session, int finish) = nullptr;
    // Optional: the session is destroyed; the engine drops it. Must not fail.
    void (*release)(void* user, std::uint64_t session) = nullptr;
    // Optional scoring (configuration "scoring": true): native_api::score's arguments and buffers for one sequence.
    int (*score)(void* user, const std::int32_t* ids, std::uint32_t count, std::uint32_t first_target, const char* path,
                 const std::int32_t* queries, std::uint32_t per_target, std::uint32_t top_k, float* target, float* query,
                 std::int32_t* argmax, std::int32_t* has_argmax, std::int32_t* top_ids, float* top_logprobs) = nullptr;
    // The last failure's message into out (at most capacity bytes with the terminating zero); returns its length.
    int (*error)(void* user, char* out, int capacity) = nullptr;
};

// A NativeEngine (concurrent_requests set) around a step model. The configuration JSON holds "engine", Core's
// settings: "implementation", "vocabulary" and "context" (required), "valid_vocabulary", "prefill_chunk" (default
// min(512, context)), "stop_tokens", "max_concurrency" (default 8), "max_pending" (default max_concurrency),
// "max_batch_tokens" (mixed batches; default prefill_chunk + max_concurrency * max_row_tokens, 0 separates prefill and
// decode steps), "batch_policy" ("fair" or "decode_first"), "max_row_tokens" (most tokens one row reports, default 1),
// "decode_token_work" (default 1), "sampling" (the fields the engine applies, as ModelInfo::sampling), "token_logprobs"
// and "max_top_logprobs", "scoring" and "score_paths", "host_limit" (bytes, default 1 GiB); and "facts", a JSON object
// merged into crit_engine_info. invalid_argument for a missing or invalid setting, a missing open or step callback.
NativeEngine step_engine(std::string_view configuration_json, const StepCallbacks&);
}
