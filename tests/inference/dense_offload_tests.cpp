// Dense layer offload: the planner's candidates and predictions, the host GEMV paths against an FP64 product (and
// bitwise independent of the thread split), and one synthetic model run under resident, streamed, host and mixed
// placements: streamed layers give the resident run's logits bit for bit (same kernels, weights from a staging slot,
// with slots recycled across layers), host layers agree within summation-order differences.
#include "offload_exec.h"
#include "offload_plan.h"
#include "synthetic_offload.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace od = crit::dense_offload;
namespace be = crit::batched_engine;

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

od::Plan plan_of(const char* places, const char* name) {
    od::Plan p;
    p.name = name;
    for(const char* c = places; *c; ++c) p.places.push_back(*c == 'd' ? od::Place::device : *c == 's' ? od::Place::stream : od::Place::host);
    return p;
}

void case_plans() {
    int devices = 0;
    if(cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::cout << "no CUDA device; skipped\n"; return; }
    cudaStream_t compute, copy;
    cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking);
    {
        const be::Shape shape{512, 256, 6, 4, 2, 64, 512, 10000.f, 21};
        od::SyntheticOffloadModel model(shape, 64, compute, 4);
        od::WeightStream stream(model.layer_bytes(), copy);
        std::mt19937 rng(9);
        std::uniform_int_distribution<int> token(0, 511);
        std::vector<std::int32_t> prompt(13), decode(6);
        for(auto& t : prompt) t = token(rng);
        for(auto& t : decode) t = token(rng);
        const auto run = [&](const od::Plan& plan) {
            model.place(plan);
            model.reset();
            std::vector<std::vector<float>> logits{model.step(plan, &stream, prompt)};
            for(const auto t : decode) logits.push_back(model.step(plan, &stream, std::span<const std::int32_t>(&t, 1)));
            return logits;
        };
        const auto resident = run(plan_of("dddddd", "resident"));
        const auto streamed = run(plan_of("dssdss", "stream"));   // four streamed layers recycle the two slots
        for(std::size_t i = 0; i < resident.size(); ++i)
            require(std::memcmp(resident[i].data(), streamed[i].data(), resident[i].size() * 4) == 0, "streamed layers changed the logits");
        double worst = 0;
        for(const char* places : {"ddhhdd", "dsddhd", "hhhhhh"}) {
            const auto other = run(plan_of(places, "mixed"));
            for(std::size_t i = 0; i < resident.size(); ++i) {
                double mag = 0, diff = 0;
                for(std::size_t v = 0; v < resident[i].size(); ++v) {
                    mag = std::max(mag, std::fabs(double(resident[i][v])));
                    diff = std::max(diff, std::fabs(double(resident[i][v]) - other[i][v]));
                }
                worst = std::max(worst, diff / mag);
                // Each host layer differs from its device twin by FP32 summation order and by exact softmax weights
                // (the device rounds attention probabilities to BF16); BF16-rounded intermediates then flip and the
                // difference grows with the host layers (at most 0.6% for two and 1.3% for all six on sm_120).
                require(diff <= 2e-2 * mag, std::string("host layers moved the logits too far: ") + places);
            }
        }
        std::cout << "streamed plan equals the resident one bit for bit; host and mixed plans within " << worst << " of the logits' magnitude\n";
        require(od::measure_upload_gbs(std::size_t{1} << 22, 3) > 0 && od::measure_round_trip_ns(4096, 3) > 0, "measurements");
    }
    cudaStreamDestroy(compute);
    cudaStreamDestroy(copy);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string which = argc > 1 ? argv[1] : "all";
        if(which == "plan" || which == "all") case_plan();
        if(which == "host" || which == "all") case_host_gemv();
        if(which == "plans" || which == "all") case_plans();
        std::cout << "dense offload " << which << " passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
