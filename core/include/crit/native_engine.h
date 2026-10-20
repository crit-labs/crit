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
inline constexpr int abi_version = 1;
// Called with each published token; nonzero cancels the request.
using TokenCallback = int (*)(void* user, const std::int32_t* tokens, int count);
void* open(NativeFactory, const char* configuration_json) noexcept;
void close(void* engine) noexcept;
int info(void* engine, char* out, int capacity) noexcept;
int generate(void* engine, const std::int32_t* prompt, int count, int output_tokens, const char* options_json,
             TokenCallback callback, void* user, std::int32_t* out, char* result, int capacity) noexcept;
int score(void* engine, const std::int32_t* ids, int count, int first_target, const char* path,
          const std::int32_t* queries, int per_target, float* target, float* query, std::int32_t* argmax,
          int* has_argmax) noexcept;
const char* last_error() noexcept;
}  // namespace native_api
}  // namespace crit::core

#if defined(_WIN32)
#define CRIT_ENGINE_EXPORT extern "C" __declspec(dllexport)
#else
#define CRIT_ENGINE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Exports the C ABI around `factory` (a NativeFactory). Use once per shared library. Functions:
//   int   crit_engine_abi()                     ABI version (1)
//   void* crit_engine_open(config_json)         engine handle, or null (crit_engine_last_error)
//   void  crit_engine_close(handle)
//   int   crit_engine_info(handle, out, cap)    JSON length: ModelInfo fields plus the factory's info
//   int   crit_engine_generate(handle, prompt, count, output_tokens, options_json, callback, user, out, result,
//                              cap)             tokens written; -1 error, -2 cancelled. options_json: optional
//                                               "stop_tokens" (array; [] disables), "temperature", "top_p",
//                                               "top_k", "min_p", "seed", "replay" (token array), "plain"
//                                               (true: no drafting). result: JSON
//                                               with "finish", "timing" (ms, prefill_tokens, decode_steps) and
//                                               the factory's request facts
//   int   crit_engine_score(handle, ids, count, first_target, path, queries, per_target, target, query, argmax,
//                           has_argmax)         0, or -1 error
//   const char* crit_engine_last_error()        the calling thread's last error
#define CRIT_EXPORT_ENGINE(factory)                                                                                \
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
                                             float* target, float* query, std::int32_t* argmax, int* has_argmax) { \
        return ::crit::core::native_api::score(engine, ids, count, first_target, path, queries, per_target,        \
                                               target, query, argmax, has_argmax);                                 \
    }                                                                                                              \
    CRIT_ENGINE_EXPORT const char* crit_engine_last_error() { return ::crit::core::native_api::last_error(); }
