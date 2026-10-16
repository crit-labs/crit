#include "crit/cuda.h"
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace crit::core {
namespace {
void check_timer(cudaError_t status, const char* operation) {
    if(status != cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorName(status)+": "+cudaGetErrorString(status));
}
void log_timer(cudaError_t status, const char* operation) noexcept {
    if(status != cudaSuccess) std::fprintf(stderr,"crit CUDA category timer %s: %s\n",operation,cudaGetErrorName(status));
}
using HostClock = std::chrono::steady_clock;
}
struct CudaCategoryTimer::Impl {
    struct Pair { cudaEvent_t start = nullptr, stop = nullptr; std::uint32_t category = 0; };
    enum class Open { none, recording, dropped };
    int device;
    std::vector<Pair> pairs;
    cudaEvent_t span_start = nullptr, span_stop = nullptr;
    cudaStream_t stream = nullptr;
    HostClock::time_point host_start{};
    std::size_t used = 0, dropped = 0;
    Open open = Open::none;
    bool started = false;
    Impl(std::size_t capacity,int d):device(d),pairs(capacity) {}
    ~Impl() {
        log_timer(cudaSetDevice(device),"bind for destruction");
        for(auto& pair:pairs) {
            if(pair.start) log_timer(cudaEventDestroy(pair.start),"cudaEventDestroy");
            if(pair.stop) log_timer(cudaEventDestroy(pair.stop),"cudaEventDestroy");
        }
        if(span_start) log_timer(cudaEventDestroy(span_start),"cudaEventDestroy");
        if(span_stop) log_timer(cudaEventDestroy(span_stop),"cudaEventDestroy");
    }
};
CudaCategoryTimer::CudaCategoryTimer(std::size_t capacity,int device):impl_(std::make_unique<Impl>(capacity,device)) {
    check_timer(cudaSetDevice(device),"cudaSetDevice");
    for(auto& pair:impl_->pairs) {
        check_timer(cudaEventCreateWithFlags(&pair.start,cudaEventDefault),"cudaEventCreateWithFlags");
        check_timer(cudaEventCreateWithFlags(&pair.stop,cudaEventDefault),"cudaEventCreateWithFlags");
    }
    check_timer(cudaEventCreateWithFlags(&impl_->span_start,cudaEventDefault),"cudaEventCreateWithFlags");
    check_timer(cudaEventCreateWithFlags(&impl_->span_stop,cudaEventDefault),"cudaEventCreateWithFlags");
}
CudaCategoryTimer::~CudaCategoryTimer() = default;
CudaCategoryTimer::CudaCategoryTimer(CudaCategoryTimer&&) noexcept = default;
CudaCategoryTimer& CudaCategoryTimer::operator=(CudaCategoryTimer&&) noexcept = default;
void CudaCategoryTimer::start(cudaStream_t stream) {
    if(!impl_) throw std::logic_error("empty CUDA category timer");
    if(!stream) throw std::invalid_argument("CUDA category timer needs an explicit stream");
    auto& s = *impl_;
    s.stream = stream; s.used = 0; s.dropped = 0; s.open = Impl::Open::none; s.started = false;
    check_timer(cudaSetDevice(s.device),"cudaSetDevice");
    s.host_start = HostClock::now();
    check_timer(cudaEventRecord(s.span_start,stream),"cudaEventRecord");
    s.started = true;
}
void CudaCategoryTimer::begin(std::uint32_t category) {
    if(!impl_ || !impl_->started) throw std::logic_error("CUDA category timer is not started");
    auto& s = *impl_;
    if(s.open == Impl::Open::recording) { s.dropped += 2; s.open = Impl::Open::dropped; return; }
    if(s.open == Impl::Open::dropped || s.used == s.pairs.size()) { ++s.dropped; s.open = Impl::Open::dropped; return; }
    s.pairs[s.used].category = category;
    check_timer(cudaEventRecord(s.pairs[s.used].start,s.stream),"cudaEventRecord");
    s.open = Impl::Open::recording;
}
void CudaCategoryTimer::end() {
    if(!impl_ || !impl_->started) throw std::logic_error("CUDA category timer is not started");
    auto& s = *impl_;
    if(s.open == Impl::Open::recording) {
        check_timer(cudaEventRecord(s.pairs[s.used].stop,s.stream),"cudaEventRecord");
        ++s.used;
    }
    s.open = Impl::Open::none;
}
CudaCategoryTimes CudaCategoryTimer::finish(std::uint32_t categories) {
    if(!impl_ || !impl_->started) throw std::logic_error("CUDA category timer is not started");
    auto& s = *impl_;
    CudaCategoryTimes result;
    result.category_ms.assign(categories,0.0);
    if(s.open == Impl::Open::recording) ++s.dropped;  // unclosed
    s.open = Impl::Open::none;
    s.started = false;
    const auto issued = HostClock::now();
    check_timer(cudaSetDevice(s.device),"cudaSetDevice");
    check_timer(cudaEventRecord(s.span_stop,s.stream),"cudaEventRecord");
    check_timer(cudaEventSynchronize(s.span_stop),"cudaEventSynchronize");
    const auto done = HostClock::now();
    for(std::size_t i = 0; i < s.used; ++i) {
        const auto& pair = s.pairs[i];
        if(pair.category >= categories) { ++s.dropped; continue; }
        float ms = 0;
        check_timer(cudaEventElapsedTime(&ms,pair.start,pair.stop),"cudaEventElapsedTime");
        result.category_ms[pair.category] += double(ms);
    }
    float span = 0;
    check_timer(cudaEventElapsedTime(&span,s.span_start,s.span_stop),"cudaEventElapsedTime");
    result.gpu_span_ms = double(span);
    result.host_issue_ms = std::chrono::duration<double,std::milli>(issued-s.host_start).count();
    result.host_total_ms = std::chrono::duration<double,std::milli>(done-s.host_start).count();
    result.dropped = s.dropped;
    result.complete = s.dropped == 0;
    return result;
}
}
