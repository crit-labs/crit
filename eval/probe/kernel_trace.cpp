// Kernel tracer for `crit eval kernels`: a CUPTI injection library (CUDA_INJECTION64_PATH) that records every
// kernel the engine process runs, graph nodes included, without changes to the engine. Writes one file per process
// to $CRIT_KTRACE_DIR/trace-<pid>.txt at exit:
//   #clock <cupti_ns> <realtime_ns>   pairs CUPTI's clock with CLOCK_REALTIME (at start and at exit)
//   #device <sm_count> <name>
//   <start_ns> <end_ns> <grid_x*grid_y*grid_z> <block_threads> <name>
// Timestamps are CUPTI's; the clock lines convert them to the host's realtime clock, which the driver uses for
// its phase marks. Linux only.
#include <cuda.h>
#include <cupti.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>
#include <unistd.h>

namespace {

FILE* g_out = nullptr;
std::mutex g_mu;

void clock_line() {
    uint64_t cupti = 0;
    cuptiGetTimestamp(&cupti);
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    const unsigned long long real = static_cast<unsigned long long>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
    fprintf(g_out, "#clock %llu %llu\n", static_cast<unsigned long long>(cupti), real);
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_records) {
    *size = 8u << 20;
    *buffer = static_cast<uint8_t*>(aligned_alloc(8, *size));
    *max_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
    CUpti_Activity* record = nullptr;
    std::lock_guard<std::mutex> lock(g_mu);
    while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
        if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && record->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
        const auto* k = reinterpret_cast<const CUpti_ActivityKernel11*>(record);
        if (g_out)
            fprintf(g_out, "%llu %llu %llu %u %s\n", static_cast<unsigned long long>(k->start),
                    static_cast<unsigned long long>(k->end),
                    static_cast<unsigned long long>(k->gridX) * k->gridY * k->gridZ, k->blockX * k->blockY * k->blockZ,
                    k->name ? k->name : "?");
    }
    free(buffer);
}

void finish() {
    cuptiActivityFlushAll(1);
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_out) return;
    clock_line();
    int sms = 0;
    char name[256] = "?";
    CUdevice device;
    if (cuDeviceGet(&device, 0) == CUDA_SUCCESS) {
        cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        cuDeviceGetName(name, sizeof(name), device);
    }
    fprintf(g_out, "#device %d %s\n", sms, name);
    fclose(g_out);
    g_out = nullptr;
}

}  // namespace

extern "C" int InitializeInjection() {
    const char* dir = getenv("CRIT_KTRACE_DIR");
    if (!dir) return 1;
    const std::string path = std::string(dir) + "/trace-" + std::to_string(getpid()) + ".txt";
    g_out = fopen(path.c_str(), "w");
    if (!g_out) return 1;
    clock_line();
    cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    atexit(finish);
    return 1;
}
