// Core's programmatic dependent launch: the support query follows the device's compute capability; a consumer launched
// programmatically after a slow producer (which lets dependents start early) still reads every value the producer
// wrote, eagerly and from a captured graph replayed twice; and an ordinary launch through the same helper agrees.
#include <crit/cuda_launch.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace cc = crit::core;

namespace {
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }

__global__ void producer(int* data, int n, long long spin, int base) {
    cc::launch_dependents();   // the consumer may start now; its wait still covers this grid's writes
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const long long start = clock64();
    while(clock64() - start < spin) {
    }
    if(i < n) data[i] = base + i;
}

__global__ void consumer(const int* data, int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // prologue: nothing read from the producer yet
    cc::wait_for_prior_grid();
    if(i < n) out[i] = 2 * data[i];
}

void check(const int* d_out, int n, int base, const char* what) {
    std::vector<int> out(static_cast<std::size_t>(n));
    ok(cudaMemcpy(out.data(), d_out, sizeof(int) * out.size(), cudaMemcpyDeviceToHost), "download");
    for(int i = 0; i < n; ++i) require(out[static_cast<std::size_t>(i)] == 2 * (base + i), what);
}
}  // namespace

int main() {
    int count = 0;
    if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::printf("no CUDA device; skipped\n"); return 0; }
    try {
        int major = 0;
        ok(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), "capability");
        require(cc::programmatic_launch_supported(0) == (major >= 9), "support follows the compute capability");
        require(!cc::programmatic_launch_supported(count + 7), "a missing device is unsupported");
        cudaStream_t stream = nullptr;
        ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        require(cc::programmatic_launch_supported(stream) == (major >= 9), "a stream's device answers the query");
        constexpr int n = 1 << 16;
        int *data = nullptr, *out = nullptr;
        ok(cudaMalloc(&data, sizeof(int) * n), "malloc");
        ok(cudaMalloc(&out, sizeof(int) * n), "malloc");
        const cc::LaunchShape shape{dim3((n + 255) / 256), dim3(256), 0};
        for(const bool programmatic : {false, true}) {
            ok(cudaMemsetAsync(data, 0, sizeof(int) * n, stream), "reset");
            ok(cc::launch(producer, shape, stream, programmatic, data, n, 200000LL, 11), "producer");
            ok(cc::launch(consumer, shape, stream, programmatic, static_cast<const int*>(data), out, n), "consumer");
            ok(cudaStreamSynchronize(stream), "sync");
            check(out, n, 11, programmatic ? "a programmatic consumer read stale producer data" : "an ordinary launch differs");
        }
        // Captured: the programmatic edge survives capture and replays.
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        ok(cudaMemsetAsync(data, 0, sizeof(int) * n, stream), "captured reset");
        ok(cc::launch(producer, shape, stream, true, data, n, 200000LL, 23), "captured producer");
        ok(cc::launch(consumer, shape, stream, true, static_cast<const int*>(data), out, n), "captured consumer");
        ok(cudaStreamEndCapture(stream, &graph), "end capture");
        ok(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for(int replay = 0; replay < 2; ++replay) {
            ok(cudaMemsetAsync(out, 0, sizeof(int) * n, stream), "clear output");
            ok(cudaGraphLaunch(exec, stream), "replay");
            ok(cudaStreamSynchronize(stream), "replay sync");
            check(out, n, 23, "a captured programmatic consumer read stale producer data");
        }
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaFree(data);
        cudaFree(out);
        cudaStreamDestroy(stream);
        std::printf("programmatic launch: %s, consumers read the producer's writes eagerly and from a graph\n",
                    major >= 9 ? "supported" : "falls back to ordinary launches");
        return 0;
    } catch(const std::exception& e) {
        std::printf("programmatic launch FAILED: %s\n", e.what());
        return 1;
    }
}
