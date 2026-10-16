// CudaCategoryTimer: device time by category without intermediate host synchronization, the GPU span against
// host issue time, and drop accounting. Host callbacks that spin occupy the stream for known durations, so the
// fixture needs no kernels (and no CUDA compiler); it requires an accessible CUDA device.
#include "crit/cuda.h"
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
void check(cudaError_t status, const char* operation) {
    if(status != cudaSuccess) throw std::runtime_error(operation);
}
// Spins rather than sleeps: Windows rounds sleeps up to its timer tick (about 15.6 ms), which skews the durations.
void CUDART_CB sleep_on_stream(void* milliseconds) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(*static_cast<const int*>(milliseconds));
    while(std::chrono::steady_clock::now() < until) std::this_thread::yield();
}
}

int main() {
    try {
        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        require(count > 0, "no CUDA device");
        check(cudaSetDevice(0), "cudaSetDevice");
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        static const int short_ms = 4, long_ms = 16;
        crit::core::CudaCategoryTimer timer(8, 0);
        timer.start(stream);
        for(int i = 0; i < 3; ++i) {
            timer.begin(0);
            check(cudaLaunchHostFunc(stream, sleep_on_stream, const_cast<int*>(&short_ms)), "cudaLaunchHostFunc");
            timer.end();
            timer.begin(1);
            check(cudaLaunchHostFunc(stream, sleep_on_stream, const_cast<int*>(&long_ms)), "cudaLaunchHostFunc");
            timer.end();
        }
        auto times = timer.finish(2);
        std::printf("category 0 %.3f ms, category 1 %.3f ms, span %.3f ms, host issue %.3f ms, host total %.3f ms\n",
                    times.category_ms[0], times.category_ms[1], times.gpu_span_ms, times.host_issue_ms,
                    times.host_total_ms);
        require(times.complete && times.dropped == 0, "complete sequence reported loss");
        require(times.category_ms[0] >= 3 * short_ms * 0.9 && times.category_ms[1] > 3.0 * times.category_ms[0],
                "category sums differ from the stream occupancy");
        require(times.gpu_span_ms >= (times.category_ms[0] + times.category_ms[1]) * 0.99, "span shorter than its categories");
        require(times.host_issue_ms < times.host_total_ms && times.host_issue_ms < times.gpu_span_ms,
                "the host waited before finish()");

        timer.start(stream);
        timer.begin(0);
        timer.begin(1);  // nested: drops the open pair and itself
        timer.end();
        timer.begin(5);  // category outside finish(2)
        timer.end();
        timer.begin(0);  // left open
        times = timer.finish(2);
        require(times.dropped == 4 && !times.complete && times.category_ms[0] == 0 && times.category_ms[1] == 0,
                "nested, unknown and unclosed pairs not dropped and counted");

        timer.start(stream);
        for(int i = 0; i < 10; ++i) { timer.begin(0); timer.end(); }  // two beyond capacity
        times = timer.finish(1);
        require(times.dropped == 2 && !times.complete, "pairs beyond capacity not dropped and counted");
        check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        std::printf("crit core CUDA category timer passed\n");
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
