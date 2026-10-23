#pragma once
// A C ABI for native engines. An engine library builds its core::Engine in a factory and exports it with
// CRIT_EXPORT_ENGINE(factory) in one source file; crit_core.native (Python) loads the library with ctypes and
// supplies the stock Eval (eval-v1) and crit serve (chat-resident-v1) adapters on top of it.
#include "crit/core.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace crit::core {
struct NativeEngine;
using NativeFactory = NativeEngine (*)(std::string_view configuration_json);
namespace native_api {
void* open(NativeFactory, const char* configuration_json) noexcept;
}
// What an engine library builds from its configuration (a JSON object: the adapter's configuration plus "model",
// the model directory, and "context_tokens", the context the engine must serve).
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
    // finished on the calling thread (for example "accepted" drafts per round, "verify_ms", "draft_ms", and
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
    friend void* native_api::open(NativeFactory, const char*) noexcept;
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

namespace native_api {
inline constexpr int abi_version = 2;
// Called with each published token; nonzero cancels the request.
using TokenCallback = int (*)(void* user, const std::int32_t* tokens, int count);
// Called with each published token of a request opened with "logprobs": n, with its log-probability and its n most
// likely alternatives (tokens and log-probabilities, ranked as TokenLogprob rows are); nonzero cancels the request.
using OutputCallback = int (*)(void* user, std::int32_t token, float logprob, const std::int32_t* top_tokens,
                               const float* top_logprobs, int alternatives);
void* open(NativeFactory, const char* configuration_json) noexcept;
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

// Exports the C ABI around `factory` (a NativeFactory). Use once per shared library. Functions:
//   int   crit_engine_abi()                     ABI version (2)
//   void* crit_engine_open(config_json)         engine handle, or null (crit_engine_last_error)
//   void  crit_engine_close(handle)
//   int   crit_engine_info(handle, out, cap)    JSON length: ModelInfo fields (token_logprobs, max_top_logprobs,
//                                               ...) plus the factory's info
//   int   crit_engine_generate(handle, prompt, count, output_tokens, options_json, callback, user, out, result,
//                              cap)             tokens written; -1 error, -2 cancelled. options_json: optional
//                                               "stop_tokens" (array; [] disables), "temperature", "top_p",
//                                               "top_k", "min_p", "seed", "replay" (token array), "plain"
//                                               (true: no drafting), "logprobs" (alternatives per token;
//                                               requires token_logprobs, at most max_top_logprobs; wait with
//                                               crit_engine_request_wait_output). result: JSON
//                                               with "finish", "timing" (ms, prefill_tokens, decode_steps) and
//                                               the factory's request facts
//   int   crit_engine_score(handle, ids, count, first_target, path, queries, per_target, top_k, target, query,
//                           argmax, has_argmax, top_tokens, top_logprobs)
//                                               0, or -1 error (native_api::score)
//   const char* crit_engine_last_error()        the calling thread's last error
#define CRIT_EXPORT_ENGINE(factory)                                                                                \
    CRIT_ENGINE_EXPORT void* crit_engine_request_open(void* engine, const std::int32_t* prompt, int count, int output, const char* options) { \
        return ::crit::core::native_api::request_open(engine, prompt, count, output, options);                       \
    }                                                                                                              \
    CRIT_ENGINE_EXPORT void crit_engine_request_cancel(void* request) { ::crit::core::native_api::request_cancel(request); } \
    CRIT_ENGINE_EXPORT void crit_engine_request_close(void* request) { ::crit::core::native_api::request_close(request); } \
    CRIT_ENGINE_EXPORT int crit_engine_request_wait(void* request, ::crit::core::native_api::TokenCallback callback, void* user, std::int32_t* out, char* result, int capacity) { \
        return ::crit::core::native_api::request_wait(request, callback, user, out, result, capacity);                \
    }                                                                                                              \
    CRIT_ENGINE_EXPORT int crit_engine_request_wait_output(void* request, ::crit::core::native_api::OutputCallback callback, void* user, std::int32_t* out, char* result, int capacity) { \
        return ::crit::core::native_api::request_wait_output(request, callback, user, out, result, capacity);         \
    }                                                                                                              \
    CRIT_ENGINE_EXPORT int crit_engine_abi() { return ::crit::core::native_api::abi_version; }                     \
    CRIT_ENGINE_EXPORT void* crit_engine_open(const char* configuration) {                                         \
        return ::crit::core::native_api::open(&(factory), configuration);                                          \
    }                                                                                                              \
    CRIT_ENGINE_EXPORT void crit_engine_close(void* engine) { ::crit::core::native_api::close(engine); }           \
    CRIT_ENGINE_EXPORT int crit_engine_info(void* engine, char* out, int capacity) {                               \
        return ::crit::core::native_api::info(engine, out, capacity);                                              \
    }                                                                                                              \
    CRIT_ENGINE_EXPORT int crit_engine_generate(void* engine, const std::int32_t* prompt, int count,               \
                                                int output_tokens, const char* options,                            \
                                                ::crit::core::native_api::TokenCallback callback, void* user,      \
                                                std::int32_t* out, char* result, int capacity) {                   \
        return ::crit::core::native_api::generate(engine, prompt, count, output_tokens, options, callback, user,   \
                                                  out, result, capacity);                                          \
    }                                                                                                              \
    CRIT_ENGINE_EXPORT int crit_engine_score(void* engine, const std::int32_t* ids, int count, int first_target,   \
                                             const char* path, const std::int32_t* queries, int per_target,        \
                                             int top_k, float* target, float* query, std::int32_t* argmax,         \
                                             int* has_argmax, std::int32_t* top_tokens, float* top_logprobs) {     \
        return ::crit::core::native_api::score(engine, ids, count, first_target, path, queries, per_target, top_k, \
                                               target, query, argmax, has_argmax, top_tokens, top_logprobs);       \
    }                                                                                                              \
    CRIT_ENGINE_EXPORT const char* crit_engine_last_error() { return ::crit::core::native_api::last_error(); }
