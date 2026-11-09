#pragma once
// Core's engine C ABI: the requests, scores and facts of an engine Core's step runtime runs (crit/step_engine.h),
// as crit_core_steps exports them for crit_core.native (Python, ctypes). The engine itself is always an assembly of
// plugins that Core's loader opened (crit_core.engine); no library hands Core an engine of its own.
#include "crit/core.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace crit::core {
struct NativeEngine;
// What builds a NativeEngine from its configuration and a context the caller passes through open_bound (Core's step
// engine and its step callbacks).
using BoundFactory = NativeEngine (*)(std::string_view configuration_json, const void* context);
// Declared here, ahead of NativeEngine, which names them its friends; the rest of the C ABI follows it.
namespace native_api {
// A handle for what `factory` builds, with `context` passed to it; the bridge retains its owner.
void* open_bound(BoundFactory, const void* context, const char* configuration_json) noexcept;
}
// What a BoundFactory builds from its configuration (Core's step engine: its "engine" settings and "facts").
struct NativeEngine {
    // The native bridge retains this with the runtime, including quarantine after a failed final drain.
    // For example, model files or allocations the engine borrows. Construction work must already be owned
    // by the backend/model: the bridge can attach this owner only after the factory returns successfully.
    std::shared_ptr<void> owner;
    std::unique_ptr<Engine> engine;
    // Optional: a JSON object merged into crit_engine_info, for engine facts such as "speculative",
    // "draft_width", "bytes_per_pass", "draft_bytes_per_step" or "replay".
    std::function<std::string()> info;
    // Optional: a JSON object merged into the result of each crit_engine_generate, called after the request
    // finished on the calling thread (for example "accepted" drafts per round, "drafted" per round when the draft
    // width varies, "verify_ms", "draft_ms", and
    // "rounds", the target passes after prefill, when it differs from the length of "accepted").
    std::function<std::string()> request;
    // Explicit opt-in: all factory callbacks and engine state permit concurrent callers.
    // The legacy request callback is incompatible with this flag. Use result facts
    // derived from the completed, request-owned Result instead of shared model counters.
    bool concurrent_requests = false;
    std::function<std::string(const Result&)> result;
private:
    // Called by the bridge once after a successful factory return, before exposing the handle.
    void retain_owner() const noexcept;
    friend void* native_api::open_bound(BoundFactory, const void*, const char*) noexcept;
};

// A generation whose drafts are verified against a fixed continuation (Eval's replay): the session accepts a
// draft while it equals the continuation's next token and takes each correction or bonus token from it, so
// every engine decodes the same text. Passed through RequestOptions::context by crit_engine_generate when the
// options carry "replay"; engines that implement it declare "replay": true in their info.
struct ReplayContinuation final : RequestContext {
    std::vector<Token> tokens;
    std::size_t host_bytes() const noexcept override { return tokens.size() * sizeof(Token); }
};

// A greedy generation decoded one token per target pass, without drafting, on an engine built for speculative
// decoding (Eval's exactness check on the loaded engine, so it opens no second engine). Passed through
// RequestOptions::context by crit_engine_generate when the options carry "plain": true; engines that implement it
// declare "plain": true in their info.
struct PlainDecoding final : RequestContext {
    std::size_t host_bytes() const noexcept override { return 0; }
};

// A generation the host side of Core's runtime keeps state for, named by `attachment` (crit_core.native: the matcher
// of a request's constraint, a prompt's prepared inputs), which the step model's request carries. `constrained`: Core's
// host side masks every token of it (structured output, forced calls), so it decodes one token per target pass, as
// PlainDecoding does, and never in native rounds. `plain` as PlainDecoding. Passed through RequestOptions::context
// when the options carry "attachment".
struct HostAttachment final : RequestContext {
    std::uint64_t attachment = 0;
    bool constrained = false, plain = false;
    std::size_t host_bytes() const noexcept override { return 0; }
};

namespace native_api {
inline constexpr int abi_version = 5;
// Called with the tokens published since its last call (a step's or a round's, in order); nonzero cancels the request.
// request_wait takes a null callback for a request nobody listens to: its waiter then wakes only when it ends.
using TokenCallback = int (*)(void* user, const std::int32_t* tokens, int count);
// Called with each published token of a request opened with "logprobs": n, with its log-probability and its n most
// likely alternatives (tokens and log-probabilities, ranked as TokenLogprob rows are); nonzero cancels the request.
using OutputCallback = int (*)(void* user, std::int32_t token, float logprob, const std::int32_t* top_tokens,
                               const float* top_logprobs, int alternatives);
void close(void* engine) noexcept;
int info(void* engine, char* out, int capacity) noexcept;
int generate(void* engine, const std::int32_t* prompt, int count, int output_tokens, const char* options_json,
             TokenCallback callback, void* user, std::int32_t* out, char* result, int capacity) noexcept;
// One waiter per request; cancel may run concurrently. Close the request after wait (or to cancel and settle it),
// before closing engine.
void* request_open(void* engine, const std::int32_t* prompt, int count, int output_tokens,
                   const char* options_json) noexcept;
void request_cancel(void* request) noexcept;
void request_close(void* request) noexcept;
int request_wait(void* request, TokenCallback, void* user, std::int32_t* out, char* result, int capacity) noexcept;
// request_wait for a request opened with "logprobs", delivering each token's log-probabilities.
int request_wait_output(void* request, OutputCallback, void* user, std::int32_t* out, char* result,
                        int capacity) noexcept;
// Scores every target ids[i], i >= first_target, into `target`. Query scoring (ScoreQueries, requires
// ModelInfo::query_scoring) adds per target the log-probabilities of `per_target` ids of `queries` (row-major) into
// `query`, the argmax into `argmax` (has_argmax tells whether the engine returned one) and, with top_k, the
// engine's own top_k ranked alternatives into top_tokens and top_logprobs (targets x top_k; requires
// max_top_logprobs >= top_k). `path` names one of ModelInfo::score_paths (null or empty: the default). Engines
// that declare query scoring always take that path; buffers of parts not asked for may be null.
int score(void* engine, const std::int32_t* ids, int count, int first_target, const char* path,
          const std::int32_t* queries, int per_target, int top_k, float* target, float* query, std::int32_t* argmax,
          int* has_argmax, std::int32_t* top_tokens, float* top_logprobs) noexcept;
const char* last_error() noexcept;
}  // namespace native_api
}  // namespace crit::core

#if defined(_WIN32)
#define CRIT_ENGINE_EXPORT extern "C" __declspec(dllexport)
#else
#define CRIT_ENGINE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// The C ABI crit_core_steps exports (src/runtime/step_library.cpp), which crit_core.native.StepEngine loads. An engine
// is never a library of its own: Core's loader (crit_core.engine) assembles a package's plugins and opens their model
// on Core's step engine with crit_engine_open_steps (crit/step_engine.h). Functions:
//   int   crit_engine_abi()                     ABI version (5)
//   void* crit_engine_open_steps(config_json, callbacks)
//   void* crit_engine_open_steps_rounds(config_json, callbacks, rounds)
//                                               engine handle, or null (crit_engine_last_error)
//   void  crit_engine_close(handle)
//   int   crit_engine_info(handle, out, cap)    JSON length: ModelInfo fields (token_logprobs, max_top_logprobs,
//                                               ...) plus the model's facts
//   void* crit_engine_request_open(handle, prompt, count, output_tokens, options_json)
//                                               a request, or null. options_json: optional "stop_tokens" (array;
//                                               [] disables), "temperature", "top_p", "top_k", "min_p", "seed",
//                                               "repetition_penalty", "frequency_penalty", "presence_penalty",
//                                               "logit_bias" ({"id": bias}), "min_tokens", "replay" (token array),
//                                               "plain" (true: no drafting), "logprobs" (alternatives per token;
//                                               requires token_logprobs, at most max_top_logprobs), "attachment"
//                                               (a positive id of host-side state, HostAttachment) with
//                                               "constrained" (true: every token masked; no replay)
//   int   crit_engine_request_wait(request, callback, user, out, result, cap)
//   int   crit_engine_request_wait_output(request, callback, user, out, result, cap)
//                                               tokens written; -1 error, -2 cancelled. result: JSON with
//                                               "finish", "timing" (ms, prefill_tokens, decode_steps) and the
//                                               model's request facts
//   void  crit_engine_request_cancel(request), crit_engine_request_close(request)
//   int   crit_engine_score(handle, ids, count, first_target, path, queries, per_target, top_k, target, query,
//                           argmax, has_argmax, top_tokens, top_logprobs)
//                                               0, or -1 error (native_api::score)
//   const char* crit_engine_last_error()        the calling thread's last error
// and Core's structured-output matchers (crit/constraints.h, src/runtime/constraint_library.cpp), with which
// crit_core.native masks a constrained request's tokens:
//   void* crit_vocabulary_open(bytes, ends, count, eos)
//                                               a ByteVocabulary of `count` pieces, piece i bytes[ends[i - 1] ..
//                                               ends[i]) (uint64 ends), or null
//   void* crit_language_open(vocabulary, kind, artifact, size)
//                                               "json_object" (artifact unused) or "grammar" (a crit-byte-grammar-v1
//                                               artifact) over the vocabulary, with the AllowedCache its matchers
//                                               share, or null
//   void* crit_matcher_open(language)           a fresh TokenMatcher, or null
//   int   crit_matcher_allowed(matcher, words, count)
//                                               allowed_bits into `count` uint32 words; 0, or -1 (dead end, limit)
//   int   crit_matcher_append(matcher, token)   0, or -1 for a token the language forbids
//   void  crit_vocabulary_close(v), crit_language_close(l), crit_matcher_close(m)
//   const char* crit_constraint_last_error()    the calling thread's last constraint error
