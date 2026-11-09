// crit_core_steps: Core's concurrent runtime for the model an engine's plugins assemble. Core's engine ABI
// (crit/native_engine.h) around its step engine (crit/step_engine.h): crit_core.native.StepEngine opens one with the
// model plugin's step callbacks, and every request, score and fact of the engine goes through these exports.
#include "crit/step_engine.h"
#include <stdexcept>
#include <string_view>

namespace {
crit::core::NativeEngine bound(std::string_view configuration, const void* callbacks) {
    if(!callbacks) throw std::invalid_argument("crit_engine_open_steps needs callbacks");
    return crit::core::step_engine(configuration, *static_cast<const crit::core::StepCallbacks*>(callbacks));
}
struct WithRounds {
    const crit::core::StepCallbacks* callbacks;
    const crit::core::StepRounds* rounds;
};
crit::core::NativeEngine bound_rounds(std::string_view configuration, const void* both) {
    const auto* b = static_cast<const WithRounds*>(both);
    if(!b->callbacks) throw std::invalid_argument("crit_engine_open_steps_rounds needs callbacks");
    return crit::core::step_engine(configuration, *b->callbacks, b->rounds);
}
}  // namespace

namespace api = crit::core::native_api;
CRIT_ENGINE_EXPORT int crit_engine_abi() { return api::abi_version; }
CRIT_ENGINE_EXPORT void crit_engine_close(void* engine) { api::close(engine); }
CRIT_ENGINE_EXPORT int crit_engine_info(void* engine, char* out, int capacity) { return api::info(engine, out, capacity); }
CRIT_ENGINE_EXPORT void* crit_engine_request_open(void* engine, const std::int32_t* prompt, int count, int output,
                                                  const char* options) {
    return api::request_open(engine, prompt, count, output, options);
}
CRIT_ENGINE_EXPORT void crit_engine_request_cancel(void* request) { api::request_cancel(request); }
CRIT_ENGINE_EXPORT void crit_engine_request_close(void* request) { api::request_close(request); }
CRIT_ENGINE_EXPORT int crit_engine_request_wait(void* request, api::TokenCallback callback, void* user, std::int32_t* out,
                                                char* result, int capacity) {
    return api::request_wait(request, callback, user, out, result, capacity);
}
CRIT_ENGINE_EXPORT int crit_engine_request_wait_output(void* request, api::OutputCallback callback, void* user,
                                                       std::int32_t* out, char* result, int capacity) {
    return api::request_wait_output(request, callback, user, out, result, capacity);
}
CRIT_ENGINE_EXPORT int crit_engine_score(void* engine, const std::int32_t* ids, int count, int first_target,
                                         const char* path, const std::int32_t* queries, int per_target, int top_k,
                                         float* target, float* query, std::int32_t* argmax, int* has_argmax,
                                         std::int32_t* top_tokens, float* top_logprobs) {
    return api::score(engine, ids, count, first_target, path, queries, per_target, top_k, target, query, argmax,
                      has_argmax, top_tokens, top_logprobs);
}
CRIT_ENGINE_EXPORT const char* crit_engine_last_error() { return api::last_error(); }
CRIT_ENGINE_EXPORT void* crit_engine_open_steps(const char* configuration, const crit::core::StepCallbacks* callbacks) {
    return crit::core::native_api::open_bound(&bound, callbacks, configuration);
}
CRIT_ENGINE_EXPORT void* crit_engine_open_steps_rounds(const char* configuration, const crit::core::StepCallbacks* callbacks,
                                                       const crit::core::StepRounds* rounds) {
    const WithRounds both{callbacks, rounds};
    return crit::core::native_api::open_bound(&bound_rounds, &both, configuration);
}
