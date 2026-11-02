// crit_core_steps: the native engine ABI (crit/native_engine.h) around step engines (crit/step_engine.h), so an engine
// written in another language implements step callbacks and gets Core's concurrent runtime. crit_core.native.StepEngine
// loads it from Python. crit_engine_open is refused: a step engine opens with its callbacks.
#include "crit/step_engine.h"
#include <stdexcept>
#include <string_view>

namespace {
crit::core::NativeEngine refuse(std::string_view) {
    throw std::logic_error("open a step engine with crit_engine_open_steps(configuration, callbacks)");
}
crit::core::NativeEngine bound(std::string_view configuration, const void* callbacks) {
    if(!callbacks) throw std::invalid_argument("crit_engine_open_steps needs callbacks");
    return crit::core::step_engine(configuration, *static_cast<const crit::core::StepCallbacks*>(callbacks));
}
}  // namespace

CRIT_EXPORT_ENGINE(refuse)
CRIT_ENGINE_EXPORT void* crit_engine_open_steps(const char* configuration, const crit::core::StepCallbacks* callbacks) {
    return crit::core::native_api::open_bound(&bound, callbacks, configuration);
}
