// StepRounds launchers over captured CUDA graphs (crit/step_engine.h): the CUDA driver API loaded at run time, so Core
// needs no CUDA toolkit to build and shares the process's one driver with whatever framework captured the graphs.
#include "crit/step_engine.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {
using CUresult = int;
using LaunchFn = CUresult (*)(void* exec, void* stream);
using SyncFn = CUresult (*)(void* stream);
using ErrorFn = CUresult (*)(CUresult, const char**);

struct Driver {
    LaunchFn launch = nullptr;
    SyncFn synchronize = nullptr;
    ErrorFn error = nullptr;
    std::string failure;
};
std::mutex lock;
std::string last_error;

Driver& driver() {
    static Driver d = [] {
        Driver out;
#if defined(_WIN32)
        HMODULE h = LoadLibraryA("nvcuda.dll");
        auto symbol = [&](const char* name) { return h ? reinterpret_cast<void*>(GetProcAddress(h, name)) : nullptr; };
#else
        void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        auto symbol = [&](const char* name) { return h ? dlsym(h, name) : nullptr; };
#endif
        if(!h) {
            out.failure = "the CUDA driver library is not loadable";
            return out;
        }
        out.launch = reinterpret_cast<LaunchFn>(symbol("cuGraphLaunch"));
        out.synchronize = reinterpret_cast<SyncFn>(symbol("cuStreamSynchronize"));
        out.error = reinterpret_cast<ErrorFn>(symbol("cuGetErrorString"));
        if(!out.launch || !out.synchronize) out.failure = "the CUDA driver lacks cuGraphLaunch or cuStreamSynchronize";
        return out;
    }();
    return d;
}

struct Rounds {
    void* stream = nullptr;
    std::vector<void*> graphs[2];   // the plain rounds, then the log-probability rounds (empty: none)
    std::uint32_t max_rows = 0, max_width = 0;
};

void fail(const std::string& message) {
    const std::lock_guard<std::mutex> guard(lock);
    last_error = message;
}
int check(CUresult r, const char* what) {
    if(r == 0) return 0;
    const char* text = nullptr;
    if(driver().error) driver().error(r, &text);
    fail(std::string(what) + " failed: " + (text ? text : std::to_string(r)));
    return 1;
}
}  // namespace

extern "C" {
CRIT_ENGINE_EXPORT void* crit_rounds_cuda_open(std::uint64_t stream, const std::uint64_t* graphs,
                                               const std::uint64_t* logprob_graphs, std::uint32_t max_rows,
                                               std::uint32_t max_width) {
    if(!driver().failure.empty()) {
        fail(driver().failure);
        return nullptr;
    }
    if(!graphs || !max_rows) {
        fail("crit_rounds_cuda_open needs graphs and rows");
        return nullptr;
    }
    auto* r = new Rounds;
    r->stream = reinterpret_cast<void*>(stream);
    r->max_rows = max_rows;
    r->max_width = max_width;
    const std::size_t n = std::size_t(max_rows) * (std::size_t(max_width) + 1);
    const std::uint64_t* tables[2] = {graphs, logprob_graphs};
    for(int form = 0; form < 2; ++form) {
        if(!tables[form]) continue;
        r->graphs[form].resize(n);
        for(std::size_t i = 0; i < n; ++i) r->graphs[form][i] = reinterpret_cast<void*>(tables[form][i]);
    }
    return r;
}
CRIT_ENGINE_EXPORT int crit_rounds_cuda_launch(void* user, std::uint32_t rows, std::uint32_t width, std::uint32_t logprobs) {
    auto* r = static_cast<Rounds*>(user);
    if(!r || !rows || rows > r->max_rows || width > r->max_width) {
        fail("native round outside the registered rows and widths");
        return 1;
    }
    const auto& graphs = r->graphs[logprobs ? 1 : 0];
    void* exec = graphs.empty() ? nullptr : graphs[std::size_t(rows - 1) * (std::size_t(r->max_width) + 1) + width];
    if(!exec) {
        fail(std::string("no ") + (logprobs ? "log-probability " : "plain ") + "graph registered for " +
             std::to_string(rows) + " rows at width " + std::to_string(width));
        return 1;
    }
    return check(driver().launch(exec, r->stream), "cuGraphLaunch");
}
CRIT_ENGINE_EXPORT int crit_rounds_cuda_wait(void* user) {
    auto* r = static_cast<Rounds*>(user);
    if(!r) {
        fail("no rounds");
        return 1;
    }
    return check(driver().synchronize(r->stream), "cuStreamSynchronize");
}
CRIT_ENGINE_EXPORT void crit_rounds_cuda_close(void* user) { delete static_cast<Rounds*>(user); }
CRIT_ENGINE_EXPORT int crit_rounds_cuda_error(char* out, int capacity) {
    const std::lock_guard<std::mutex> guard(lock);
    if(!out || capacity < 1) return int(last_error.size());
    const auto n = std::min<std::size_t>(last_error.size(), std::size_t(capacity) - 1);
    std::memcpy(out, last_error.data(), n);
    out[n] = 0;
    return int(n);
}
}
