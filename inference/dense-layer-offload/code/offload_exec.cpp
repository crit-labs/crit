// Layer placement execution and measurements; see offload_exec.h.
#include "offload_exec.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

namespace crit::dense_offload {
namespace {
void check(cudaError_t e, const char* what) {
    if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
}  // namespace

WeightStream::WeightStream(std::size_t slot_bytes, cudaStream_t copy) : slot_bytes_(slot_bytes), copy_(copy) {
    if(!slot_bytes) throw std::invalid_argument("weight stream slots need bytes");
    try {
        for(int s = 0; s < 2; ++s) {
            check(cudaMalloc(&slots_[s], slot_bytes), "weight slot");
            check(cudaEventCreateWithFlags(&ready_[s], cudaEventDisableTiming), "event");
            check(cudaEventCreateWithFlags(&free_[s], cudaEventDisableTiming), "event");
            check(cudaEventRecord(free_[s], copy_), "event");
        }
    } catch(...) {
        for(int s = 0; s < 2; ++s) { cudaFree(slots_[s]); if(ready_[s]) cudaEventDestroy(ready_[s]); if(free_[s]) cudaEventDestroy(free_[s]); }
        throw;
    }
}
WeightStream::~WeightStream() {
    cudaStreamSynchronize(copy_);
    for(int s = 0; s < 2; ++s) { cudaFree(slots_[s]); cudaEventDestroy(ready_[s]); cudaEventDestroy(free_[s]); }
}
int WeightStream::upload(const void* host, std::size_t bytes) {
    if(!host || bytes > slot_bytes_) throw std::invalid_argument("streamed layer larger than a slot");
    const int slot = next_;
    next_ ^= 1;
    check(cudaStreamWaitEvent(copy_, free_[slot], 0), "slot wait");
    check(cudaMemcpyAsync(slots_[slot], host, bytes, cudaMemcpyHostToDevice, copy_), "layer upload");
    check(cudaEventRecord(ready_[slot], copy_), "upload event");
    return slot;
}
void* WeightStream::acquire(int slot, cudaStream_t compute) {
    check(cudaStreamWaitEvent(compute, ready_[slot], 0), "upload wait");
    return slots_[slot];
}
void WeightStream::release(int slot, cudaStream_t compute) { check(cudaEventRecord(free_[slot], compute), "slot release"); }

void run_layers(const Plan& plan, const LayerStore& store, LayerRunner& runner, WeightStream* stream, cudaStream_t compute) {
    const std::size_t layers = plan.places.size();
    if(store.bytes.size() != layers || store.device.size() != layers || store.host.size() != layers)
        throw std::invalid_argument("layer store and plan disagree");
    std::vector<std::size_t> streamed;
    for(std::size_t i = 0; i < layers; ++i) if(plan.places[i] == Place::stream) streamed.push_back(i);
    if(!streamed.empty() && !stream) throw std::invalid_argument("a streaming plan needs a weight stream");
    std::vector<int> slot(layers, -1);
    std::size_t queued = 0;
    const auto prefetch = [&] {
        while(queued < streamed.size() && queued < 2 + std::size_t(std::count_if(slot.begin(), slot.end(), [](int s) { return s == -2; }))) {
            const auto i = streamed[queued++];
            slot[i] = stream->upload(store.host[i], store.bytes[i]);
        }
    };
    prefetch();
    for(std::size_t i = 0; i < layers; ++i) {
        switch(plan.places[i]) {
        case Place::device:
            runner.device_layer(int(i), store.device[i], compute);
            break;
        case Place::stream: {
            void* weights = stream->acquire(slot[i], compute);
            runner.device_layer(int(i), weights, compute);
            stream->release(slot[i], compute);
            slot[i] = -2;   // done: its slot may take the next streamed layer
            prefetch();
            break;
        }
        case Place::host:
            if(i == 0 || plan.places[i - 1] != Place::host) {
                runner.to_host(compute);
                check(cudaStreamSynchronize(compute), "activations to host");
            }
            runner.host_layer(int(i));
            if(i + 1 == layers || plan.places[i + 1] != Place::host) runner.to_device(compute);
            break;
        }
    }
}

double measure_upload_gbs(std::size_t bytes, int repeats) {
    void* host = nullptr;
    void* device = nullptr;
    cudaEvent_t a = nullptr, b = nullptr;
    double gbs = 0;
    try {
        check(cudaMallocHost(&host, bytes), "pinned buffer");
        check(cudaMalloc(&device, bytes), "device buffer");
        check(cudaEventCreate(&a), "event");
        check(cudaEventCreate(&b), "event");
        check(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "warm-up copy");
        // The median of separately timed copies, so a transient competitor on the link does not set the rate.
        std::vector<double> rates;
        for(int r = 0; r < repeats; ++r) {
            check(cudaEventRecord(a, nullptr), "event");
            check(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, nullptr), "copy");
            check(cudaEventRecord(b, nullptr), "event");
            check(cudaEventSynchronize(b), "sync");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, a, b), "elapsed");
            rates.push_back(static_cast<double>(bytes) / (ms * 1e6));
        }
        std::sort(rates.begin(), rates.end());
        gbs = rates[rates.size() / 2];
    } catch(...) {
        cudaFreeHost(host); cudaFree(device); if(a) cudaEventDestroy(a); if(b) cudaEventDestroy(b);
        throw;
    }
    cudaFreeHost(host); cudaFree(device); cudaEventDestroy(a); cudaEventDestroy(b);
    return gbs;
}

double measure_host_gemv_gbs(ThreadPool& pool, Isa isa, std::size_t rows, std::size_t cols, int batch, int repeats) {
    std::vector<std::uint16_t> w(rows * cols);
    for(std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<std::uint16_t>(0x3c00 + (i * 2654435761u >> 24) % 0x200);
    std::vector<float> x(static_cast<std::size_t>(batch) * cols, 0.5f), y(static_cast<std::size_t>(batch) * rows);
    gemv(pool, isa, w.data(), rows, cols, x.data(), cols, batch, y.data(), rows);
    std::vector<double> rates;
    for(int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        gemv(pool, isa, w.data(), rows, cols, x.data(), cols, batch, y.data(), rows);
        rates.push_back(static_cast<double>(w.size() * 2) / std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(rates.begin(), rates.end());
    return rates[rates.size() / 2];
}

double measure_round_trip_ns(std::size_t bytes, int repeats) {
    void* host = nullptr;
    void* device = nullptr;
    cudaStream_t s = nullptr;
    double ns = 0;
    try {
        check(cudaMallocHost(&host, bytes), "pinned buffer");
        check(cudaMalloc(&device, bytes), "device buffer");
        check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        const auto once = [&] {
            check(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, s), "down");
            check(cudaStreamSynchronize(s), "sync");
            check(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, s), "up");
        };
        once();
        check(cudaStreamSynchronize(s), "sync");
        const auto start = std::chrono::steady_clock::now();
        for(int r = 0; r < repeats; ++r) once();
        check(cudaStreamSynchronize(s), "sync");
        ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / repeats;
    } catch(...) {
        cudaFreeHost(host); cudaFree(device); if(s) cudaStreamDestroy(s);
        throw;
    }
    cudaFreeHost(host); cudaFree(device); cudaStreamDestroy(s);
    return ns;
}

}  // namespace crit::dense_offload
