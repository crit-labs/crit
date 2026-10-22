// Plans a synthetic dense model under a device budget below its weights, measures the rates the planner needs on this
// machine, predicts each candidate plan's decode step and times it; prints one JSON object per line.
//
//   crit_dense_offload_bench [budget_fraction hidden layers heads kv_heads head_dim ffn vocab device_gbs]
//
// device_gbs: the probe's bandwidth (crit eval probe, device.json "bandwidth_gbs"); 0 measures it from an all-resident
// run of the same model.
#include "offload_exec.h"
#include "offload_plan.h"
#include "synthetic_offload.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace od = crit::dense_offload;
namespace be = crit::batched_engine;

int main(int argc, char** argv) {
    try {
        const auto num = [&](int i, double fallback) { return argc > i ? std::atof(argv[i]) : fallback; };
        const double fraction = num(1, 0.25);
        const be::Shape shape{int(num(8, 32000)), int(num(2, 4096)), int(num(3, 16)), int(num(4, 32)), int(num(5, 8)), int(num(6, 128)),
                              int(num(7, 14336)), 10000.f, 7};
        double device_gbs = num(9, 0);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        cudaStream_t compute, copy;
        cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking);
        od::SyntheticOffloadModel model(shape, 256, compute);
        od::WeightStream stream(model.layer_bytes(), copy);
        const std::vector<std::uint64_t> bytes(static_cast<std::size_t>(shape.layers), model.layer_bytes());
        const std::uint64_t total = model.layer_bytes() * static_cast<std::uint64_t>(shape.layers);
        const auto budget = static_cast<std::uint64_t>(static_cast<double>(total) * fraction);
        std::mt19937 rng(1);
        std::uniform_int_distribution<int> token(0, shape.vocab - 1);
        std::vector<std::int32_t> prompt(32);
        for(auto& t : prompt) t = token(rng);
        // Decode steps after a 32-token prompt, timed per step (ns).
        const auto time_plan = [&](const od::Plan& plan, int steps) {
            model.place(plan);
            model.reset();
            model.step(plan, &stream, prompt);
            std::int32_t t = prompt.back();
            model.step(plan, &stream, std::span<const std::int32_t>(&t, 1));
            const auto start = std::chrono::steady_clock::now();
            for(int i = 0; i < steps; ++i) model.step(plan, &stream, std::span<const std::int32_t>(&t, 1));
            return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        };
        od::ThreadPool pool(0);
        od::Measurements m;
        m.upload_gbs = od::measure_upload_gbs(std::size_t{256} << 20, 8);
        m.host_gbs = od::measure_host_gemv_gbs(pool, od::best_isa(), static_cast<std::size_t>(shape.hidden), static_cast<std::size_t>(shape.ffn), 1, 10);
        m.host_layer_ns = od::measure_round_trip_ns(sizeof(float) * shape.hidden, 50);
        double resident_ns = 0;
        if(device_gbs <= 0) {
            od::Plan resident;
            resident.places.assign(static_cast<std::size_t>(shape.layers), od::Place::device);
            resident_ns = time_plan(resident, 20) / 20;
            device_gbs = static_cast<double>(total) / resident_ns;
        }
        m.device_gbs = device_gbs;
        std::printf("{\"device\": \"%s\", \"host_isa\": \"%s\", \"host_threads\": %u, \"layer_bytes\": %zu, \"layers\": %d, \"budget_bytes\": %llu, "
                    "\"device_gbs\": %.0f, \"upload_gbs\": %.1f, \"host_gbs\": %.1f, \"round_trip_us\": %.1f, \"all_resident_step_ms\": %.3f}\n",
                    prop.name, od::isa_name(od::best_isa()), pool.size(), model.layer_bytes(), shape.layers, static_cast<unsigned long long>(budget),
                    m.device_gbs, m.upload_gbs, m.host_gbs, m.host_layer_ns / 1e3, resident_ns / 1e6);
        const auto plans = od::candidates(bytes, budget, m);
        for(const auto& plan : plans) {
            int resident = 0, streamed = 0, host = 0;
            for(const auto p : plan.places) (p == od::Place::device ? resident : p == od::Place::stream ? streamed : host)++;
            const double measured = time_plan(plan, 10) / 10;
            std::printf("{\"plan\": \"%s\", \"resident\": %d, \"streamed\": %d, \"host\": %d, \"device_bytes\": %llu, \"predicted_ms\": %.3f, \"measured_ms\": %.3f}\n",
                        plan.name, resident, streamed, host, static_cast<unsigned long long>(plan.device_bytes), plan.predicted_ns / 1e6, measured / 1e6);
        }
        const auto chosen = od::select_measured(plans, 4.0, 5, time_plan);
        std::printf("{\"selected\": \"%s\"}\n", chosen.name);
        cudaStreamDestroy(compute);
        cudaStreamDestroy(copy);
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
