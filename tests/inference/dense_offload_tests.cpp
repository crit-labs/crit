// Dense layer offload: the planner's candidates and predictions, the host GEMV paths against an FP64 product (and
// bitwise independent of the thread split), a host block (attention and MLP over a host KV cache) independent of the
// thread split and of how its tokens are batched, and on a CUDA device the executor: the two-slot weight stream and
// run_layers over device, streamed and host layers, each layer reading its own bytes in step order, and the measured
// rates the planner takes.
#include "offload_exec.h"
#include "offload_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace od = crit::dense_offload;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }

void case_plan() {
    const std::vector<std::uint64_t> bytes(8, 10);
    od::Measurements m;
    m.device_gbs = 1; m.upload_gbs = 0.5; m.host_gbs = 0.25; m.host_layer_ns = 5;
    const auto plans = od::candidates(bytes, 50, m);
    require(plans.size() == 2 && std::string(plans[0].name) == "stream" && std::string(plans[1].name) == "host", "candidate order");
    // stream: two 10-byte slots leave room for 3 resident layers; 5 streamed layers cost max(10, 20) each
    require(plans[0].predicted_ns == 3 * 10 + 5 * 20 && plans[0].staging_bytes == 20 && plans[0].device_bytes == 50, "streaming prediction");
    // host: 5 resident layers, one run of 3 host layers at 40 each plus one round trip
    require(plans[1].predicted_ns == 5 * 10 + 3 * 40 + 5 && plans[1].device_bytes == 50, "host prediction");
    require(od::candidates(bytes, 80, m).front().predicted_ns == 80 && std::string(od::candidates(bytes, 80, m).front().name) == "resident", "all resident");
    m.upload_gbs = 0.05;   // a slow link: host execution wins
    require(std::string(od::candidates(bytes, 50, m).front().name) == "host", "a slow upload favours host layers");
    int timed = 0;
    const auto chosen = od::select_measured(plans, 10.0, 3, [&](const od::Plan& p, int) { ++timed; return std::string(p.name) == "host" ? 1.0 : 2.0; });
    require(timed == 2 && std::string(chosen.name) == "host", "the measured fastest plan wins over the predicted one");
}

void case_host_gemv() {
    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.f, 1.f);
    const std::size_t rows = 77, cols = 256;
    std::vector<std::uint16_t> w(rows * cols);
    for(auto& v : w) { const float f = od::round_bf16(normal(rng) * 0.1f); std::uint32_t bits; std::memcpy(&bits, &f, 4); v = static_cast<std::uint16_t>(bits >> 16); }
    od::ThreadPool one(1), four(4);
    for(const auto isa : {od::Isa::scalar, od::Isa::avx2, od::Isa::avx512}) {
        if(!od::supported(isa)) { std::cout << od::isa_name(isa) << ": not supported here; skipped\n"; continue; }
        for(const int batch : {1, 5, 8}) {
            std::vector<float> x(static_cast<std::size_t>(batch) * cols), y1(static_cast<std::size_t>(batch) * rows), y4(y1.size());
            for(auto& v : x) v = normal(rng);
            od::gemv(one, isa, w.data(), rows, cols, x.data(), cols, batch, y1.data(), rows);
            od::gemv(four, isa, w.data(), rows, cols, x.data(), cols, batch, y4.data(), rows);
            require(std::memcmp(y1.data(), y4.data(), y1.size() * 4) == 0, "the thread split changed a host GEMV result");
            for(int b = 0; b < batch; ++b)
                for(std::size_t r = 0; r < rows; ++r) {
                    double ref = 0, mag = 0;
                    for(std::size_t c = 0; c < cols; ++c) {
                        const double t = double(od::bf16_to_float(w[r * cols + c])) * x[static_cast<std::size_t>(b) * cols + c];
                        ref += t; mag += std::fabs(t);
                    }
                    require(std::fabs(y1[static_cast<std::size_t>(b) * rows + r] - ref) <= 1e-5 * mag + 1e-7, "host GEMV differs from FP64");
                }
        }
        std::cout << od::isa_name(isa) << ": host GEMV within 1e-5 of FP64, independent of the thread split\n";
    }
    require(od::best_isa() != od::Isa::avx2 || od::supported(od::Isa::avx2), "best ISA is supported");
}

std::vector<std::uint16_t> bf16_matrix(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<std::uint16_t> w(n);
    for(auto& v : w) { const float f = od::round_bf16(normal(rng) * scale); std::uint32_t bits; std::memcpy(&bits, &f, 4); v = static_cast<std::uint16_t>(bits >> 16); }
    return w;
}

// A grouped-query block over a host KV cache: the result is bitwise the same whatever the thread split, three tokens
// in one call give what three single-token calls give (within float rounding of the batched GEMV), and the cache
// holds exactly the positions written.
void case_host_block() {
    std::mt19937 rng(11);
    const od::BlockShape shape{64, 4, 2, 16, 128};
    const std::size_t H = 64, qkv = (4 + 2 * 2) * 16, att = 4 * 16, ffn = 128;
    const auto wqkv = bf16_matrix(rng, qkv * H, 0.1f), wo = bf16_matrix(rng, H * att, 0.1f);
    const auto wgu = bf16_matrix(rng, 2 * ffn * H, 0.1f), wdown = bf16_matrix(rng, H * ffn, 0.1f);
    const od::BlockWeights w{wqkv.data(), wo.data(), wgu.data(), wdown.data()};
    std::vector<float> inv_freq(8);
    for(int i = 0; i < 8; ++i) inv_freq[i] = std::pow(10000.f, -2.f * static_cast<float>(i) / 16.f);
    const int capacity = 8, batch = 3;
    const std::size_t kv_floats = static_cast<std::size_t>(capacity) * 2 * 16;
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> x0(H * batch);
    for(auto& v : x0) v = normal(rng);
    const auto run = [&](od::ThreadPool& pool, int step, std::vector<float>& k, std::vector<float>& v) {
        std::vector<float> x = x0;
        for(int pos = 0; pos < batch; pos += step)
            od::host_block(pool, od::best_isa(), shape, w, od::HostKv{k.data(), v.data(), capacity}, inv_freq.data(), pos,
                           step, x.data() + static_cast<std::size_t>(pos) * H);
        return x;
    };
    od::ThreadPool one(1), four(4);
    std::vector<float> k1(kv_floats, 0.f), v1(kv_floats, 0.f), k4(kv_floats, 0.f), v4(kv_floats, 0.f);
    std::vector<float> ks(kv_floats, 0.f), vs(kv_floats, 0.f);
    const auto y1 = run(one, batch, k1, v1), y4 = run(four, batch, k4, v4), ys = run(four, 1, ks, vs);
    require(std::memcmp(y1.data(), y4.data(), y1.size() * 4) == 0 && k1 == k4 && v1 == v4,
            "the thread split changed a host block's output or cache");
    for(std::size_t i = 0; i < y1.size(); ++i) {
        require(std::isfinite(y1[i]), "a host block output is not finite");
        require(std::fabs(y1[i] - ys[i]) <= 1e-5f * (1.f + std::fabs(ys[i])), "three tokens in one call differ from three single-token calls");
    }
    const std::size_t row = 2 * 16;   // one position's kv_heads x head_dim floats
    for(int pos = 0; pos < capacity; ++pos) {
        const bool written = pos < batch;
        const bool any = std::any_of(k1.begin() + pos * row, k1.begin() + (pos + 1) * row, [](float f) { return f != 0.f; });
        require(any == written, "the cache holds other positions than the ones written");
    }
    std::cout << "host block: independent of the thread split, batched within 1e-5 of single tokens\n";
}

void check(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

// Records the calls run_layers makes, and copies the bytes each device or streamed layer is handed, in stream order,
// so a slot that held another layer's weights at that point would show.
struct Recorder final : od::LayerRunner {
    std::vector<std::string> calls;
    std::vector<unsigned char*> seen;
    std::size_t bytes = 0;
    void device_layer(int layer, const void* weights, cudaStream_t compute) override {
        calls.push_back("device " + std::to_string(layer));
        check(cudaMemcpyAsync(seen[layer], weights, bytes, cudaMemcpyDeviceToHost, compute), "copy what a layer read");
    }
    void host_layer(int layer) override { calls.push_back("host " + std::to_string(layer)); }
    void to_host(cudaStream_t) override { calls.push_back("to host"); }
    void to_device(cudaStream_t) override { calls.push_back("to device"); }
};

// On a CUDA device: six layers placed device, stream, stream, host, stream, device run in order, the third streamed
// layer reusing a slot after the first released it; each layer sees its own bytes; the host run is bracketed by the
// activations' copies; and the planner's measured rates are positive.
void case_stream() {
    int devices = 0;
    if(cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::cout << "stream: no CUDA device here; skipped\n"; return; }
    const std::size_t layers = 6, bytes = 1 << 16;
    const std::vector<od::Place> places{od::Place::device, od::Place::stream, od::Place::stream, od::Place::host,
                                        od::Place::stream, od::Place::device};
    od::Plan plan;
    plan.places = places;
    od::LayerStore store;
    store.device.assign(layers, nullptr);
    store.host.assign(layers, nullptr);
    store.bytes.assign(layers, bytes);
    std::vector<std::vector<unsigned char>> pattern(layers, std::vector<unsigned char>(bytes));
    std::vector<void*> owned_device, owned_host;
    Recorder runner;
    runner.bytes = bytes;
    runner.seen.assign(layers, nullptr);
    for(std::size_t i = 0; i < layers; ++i) {
        for(std::size_t j = 0; j < bytes; ++j) pattern[i][j] = static_cast<unsigned char>(i * 37 + j * 13 + (j >> 8));
        void* seen = nullptr;
        check(cudaMallocHost(&seen, bytes), "pinned capture");
        owned_host.push_back(seen);
        runner.seen[i] = static_cast<unsigned char*>(seen);
        std::memset(seen, 0, bytes);
        if(places[i] == od::Place::device) {
            void* d = nullptr;
            check(cudaMalloc(&d, bytes), "resident layer");
            check(cudaMemcpy(d, pattern[i].data(), bytes, cudaMemcpyHostToDevice), "resident layer upload");
            owned_device.push_back(d);
            store.device[i] = d;
        } else {
            void* h = nullptr;
            check(cudaMallocHost(&h, bytes), "pinned layer");
            std::memcpy(h, pattern[i].data(), bytes);
            owned_host.push_back(h);
            store.host[i] = h;
        }
    }
    cudaStream_t compute = nullptr, copy = nullptr;
    check(cudaStreamCreate(&compute), "compute stream");
    check(cudaStreamCreate(&copy), "copy stream");
    {
        od::WeightStream stream(bytes, copy);
        od::run_layers(plan, store, runner, &stream, compute);
        check(cudaStreamSynchronize(compute), "the step");
    }
    const std::vector<std::string> expected{"device 0", "device 1", "device 2", "to host", "host 3", "to device",
                                            "device 4", "device 5"};
    require(runner.calls == expected, "run_layers called the layers out of order or without the host run's copies");
    for(std::size_t i = 0; i < layers; ++i)
        if(places[i] != od::Place::host)
            require(std::memcmp(runner.seen[i], pattern[i].data(), bytes) == 0,
                    "layer " + std::to_string(i) + " read other bytes than its own");
    od::ThreadPool pool(2);
    const double upload = od::measure_upload_gbs(1 << 20, 3), trip = od::measure_round_trip_ns(4096, 3);
    const double gemv = od::measure_host_gemv_gbs(pool, od::best_isa(), 256, 256, 1, 3);
    require(std::isfinite(upload) && upload > 0 && std::isfinite(trip) && trip > 0 && std::isfinite(gemv) && gemv > 0,
            "a measured rate is not positive");
    for(void* d : owned_device) cudaFree(d);
    for(void* h : owned_host) cudaFreeHost(h);
    cudaStreamDestroy(copy);
    cudaStreamDestroy(compute);
    std::cout << "stream: every layer read its own bytes in order (a slot reused), rates " << upload << " GB/s up, "
              << gemv << " GB/s host GEMV, " << trip << " ns round trip\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string which = argc > 1 ? argv[1] : "all";
        if(which == "plan" || which == "all") case_plan();
        if(which == "host" || which == "all") case_host_gemv();
        if(which == "host_block" || which == "all") case_host_block();
        if(which == "stream" || which == "all") case_stream();
        std::cout << "dense offload " << which << " passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
