// crit::trace::NvtxSink, compiled only with CRIT_CORE_NVTX=ON. The NVTX 3 headers are
// included here alone: on Windows they include windows.h, which must not reach consumers.
#include "crit/trace.h"
#include <cstdio>
#include <nvtx3/nvToolsExt.h>

namespace crit::trace {
void NvtxSink::begin(const Work& work) noexcept {
    char text[384];
    const auto length = static_cast<int>(work.stage.size() < stage_bytes - 1 ? work.stage.size() : stage_bytes - 1);
    std::snprintf(text, sizeof(text),
                  "crit.work/1|%.*s|tokens=%llu|batch=%llu|context=%llu|useful_bytes=%llu..%llu|flops=%llu", length,
                  work.stage.empty() ? "" : work.stage.data(), static_cast<unsigned long long>(work.tokens),
                  static_cast<unsigned long long>(work.batch), static_cast<unsigned long long>(work.context),
                  static_cast<unsigned long long>(work.useful_bytes_min),
                  static_cast<unsigned long long>(work.useful_bytes_max),
                  static_cast<unsigned long long>(work.floating_ops));
    nvtxRangePushA(text);
}
void NvtxSink::end(bool) noexcept { nvtxRangePop(); }
}  // namespace crit::trace
