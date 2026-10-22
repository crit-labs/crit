// GGUF block projections: the host decode equals gguf-py's quants.dequantize (SHA-256 of the FP32 values of the
// deterministic blocks gguf_blocks.py's test_blocks also generates, digests recorded from gguf 0.19.0), the device
// decode equals the host bit for bit, and the GEMV matches an FP64 product of the decoded weights for 1 to 8 rows.
#include "gguf_cuda.h"

#include <crit/sha256.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gb = crit::gguf_blocks;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
void ok(cudaError_t e, const char* what) { if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }

struct Case { gb::Kind kind; const char* name; const char* digest; };
const Case cases[] = {
    {gb::Kind::q4_0, "Q4_0", "52a944d0347e56eeb04f2dbd96e806d71f7ab3cdf242c1a88f92369aee862dc0"},
    {gb::Kind::q8_0, "Q8_0", "d0490b67f7527a093422496ba0ee467127deaf0710b10355ddada41f076be3f0"},
    {gb::Kind::q4_k, "Q4_K", "f0c481aeb5d28f121f3c2a9debd69aed8a87478857898010cd185a0ce64e8fc0"},
    {gb::Kind::q5_k, "Q5_K", "8572635116b72c2c501bcc25e82c2273e5d54cfb76d91072a1b7bc1538e5feda"},
    {gb::Kind::q6_k, "Q6_K", "1517224025493bd0dee5259d7a4018aaad62949ea072dfc18cba6fe638824d9b"},
};

// gguf_blocks.py's test_blocks: LCG bytes, then each FP16 field finite with magnitude 2^-8 .. 2^-3 and random sign.
std::vector<std::uint8_t> test_blocks(gb::Kind kind, int count, std::uint64_t seed) {
    const auto f = gb::format(kind);
    std::vector<int> fields;
    if(kind == gb::Kind::q4_k || kind == gb::Kind::q5_k) fields = {0, 2};
    else if(kind == gb::Kind::q6_k) fields = {208};
    else fields = {0};
    std::uint64_t state = seed;
    const auto next = [&] { state = state * 6364136223846793005ull + 1442695040888963407ull; return state; };
    std::vector<std::uint8_t> out;
    for(int b = 0; b < count; ++b) {
        std::vector<std::uint8_t> block(static_cast<std::size_t>(f.bytes));
        for(auto& byte : block) byte = static_cast<std::uint8_t>(next() >> 56);
        for(const int at : fields) {
            const std::uint64_t s = next();
            const auto bits = static_cast<std::uint16_t>((0x1C00 + (s >> 40) % 0x1400) | ((s >> 63) << 15));
            block[static_cast<std::size_t>(at)] = static_cast<std::uint8_t>(bits & 0xff);
            block[static_cast<std::size_t>(at) + 1] = static_cast<std::uint8_t>(bits >> 8);
        }
        out.insert(out.end(), block.begin(), block.end());
    }
    return out;
}

std::vector<float> host_decode(gb::Kind kind, const std::vector<std::uint8_t>& data) {
    const auto f = gb::format(kind);
    std::vector<float> out(data.size() / static_cast<std::size_t>(f.bytes) * static_cast<std::size_t>(f.values));
    for(std::size_t block = 0; block * f.bytes < data.size(); ++block)
        for(int first = 0; first < f.values; first += 8) gb::decode8(kind, data.data() + block * f.bytes, first, out.data() + block * f.values + first);
    return out;
}

template <class T> struct Device {
    T* p = nullptr;
    explicit Device(std::size_t n) { ok(cudaMalloc(&p, sizeof(T) * n), "malloc"); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
}  // namespace

int main() {
    try {
        int devices = 0;
        if(cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::cout << "no CUDA device; skipped\n"; return 0; }
        std::mt19937 rng(3);
        for(const auto& c : cases) {
            const auto data = test_blocks(c.kind, 6, 0x5eed + std::strlen(c.name));
            const auto host = host_decode(c.kind, data);
            const auto digest = crit::core::sha256_hex(std::as_bytes(std::span<const float>(host)));
            require(digest == c.digest, std::string(c.name) + ": host decode differs from gguf-py");
            Device<std::uint8_t> blocks(data.size());
            Device<float> out(host.size());
            ok(cudaMemcpy(blocks.p, data.data(), data.size(), cudaMemcpyHostToDevice), "upload");
            ok(gb::dequantize(c.kind, blocks.p, data.size() / static_cast<std::size_t>(gb::format(c.kind).bytes), out.p, nullptr), "dequantize");
            std::vector<float> device(host.size());
            ok(cudaMemcpy(device.data(), out.p, sizeof(float) * device.size(), cudaMemcpyDeviceToHost), "download");
            require(std::memcmp(device.data(), host.data(), sizeof(float) * host.size()) == 0, std::string(c.name) + ": device decode differs from the host");

            // GEMV over a 37 x 1024 matrix for 1, 3 and 8 activation rows against an FP64 product of the decoded weights.
            const int rows = 37, cols = 1024;
            const auto f = gb::format(c.kind);
            const auto matrix = test_blocks(c.kind, rows * cols / f.values, 0xabc + static_cast<std::uint64_t>(c.kind));
            const auto w = host_decode(c.kind, matrix);
            Device<std::uint8_t> dm(matrix.size());
            ok(cudaMemcpy(dm.p, matrix.data(), matrix.size(), cudaMemcpyHostToDevice), "upload");
            std::normal_distribution<float> normal(0.f, 1.f);
            for(const int batch : {1, 3, 8}) {
                std::vector<float> x(static_cast<std::size_t>(batch) * cols);
                for(auto& v : x) v = normal(rng);
                Device<float> dx(x.size()), dy(static_cast<std::size_t>(batch) * rows);
                ok(cudaMemcpy(dx.p, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice), "upload");
                ok(gb::gemv(c.kind, dm.p, rows, cols, dx.p, batch, dy.p, nullptr), "gemv");
                std::vector<float> y(static_cast<std::size_t>(batch) * rows);
                ok(cudaMemcpy(y.data(), dy.p, sizeof(float) * y.size(), cudaMemcpyDeviceToHost), "download");
                for(int b = 0; b < batch; ++b)
                    for(int r = 0; r < rows; ++r) {
                        double ref = 0, mag = 0;
                        for(int k = 0; k < cols; ++k) {
                            const double t = double(w[static_cast<std::size_t>(r) * cols + k]) * x[static_cast<std::size_t>(b) * cols + k];
                            ref += t;
                            mag += std::fabs(t);
                        }
                        require(std::fabs(y[static_cast<std::size_t>(b) * rows + r] - ref) <= 1e-5 * mag + 1e-7, std::string(c.name) + ": GEMV differs from the FP64 product");
                    }
            }
            require(gb::gemv(c.kind, dm.p, rows, cols + 8, nullptr, 1, nullptr, nullptr) == cudaErrorInvalidValue, "invalid GEMV accepted");
            std::cout << c.name << ": host decode equals gguf-py, device decode equals host, GEMV within 1e-5 of FP64\n";
        }
        Device<float> x(8), y(8);
        require(gb::gemv(gb::Kind::q4_k, x.p, 1, 256, x.p, 9, y.p, nullptr) == cudaErrorInvalidValue, "batch above 8 accepted");
        require(gb::gemv(gb::Kind::q4_k, x.p, 1, 100, x.p, 1, y.p, nullptr) == cudaErrorInvalidValue, "partial block accepted");
        std::cout << "gguf projections passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
