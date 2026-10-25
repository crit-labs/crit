// NVFP4 codec: every E2M1 code, E4M3 decode/encode round trips and ties, the shared scale encoder vectors
// (argument 1: tests/inference/e4m3_scale_vectors.txt), every scale byte's decode, packing, quantization error bounds
// and batched projection against an FP64 reference.
#include "nvfp4.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <stdexcept>
#include <vector>

namespace fp4 = crit::nvfp4;

#define REQUIRE(x) do { if(!(x)) { std::fprintf(stderr, "FAILED %s:%d %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while(0)

template <class F>
bool throws(F&& f) {
    try { f(); } catch(const std::exception&) { return true; }
    return false;
}

int main(int argc, char** argv) {
    REQUIRE(argc > 1);
    std::ifstream vectors(argv[1]);
    REQUIRE(vectors.good());
    int checked = 0;
    for(std::string line; std::getline(vectors, line);) {
        std::istringstream fields(line.substr(0, line.find('#')));
        std::string bits_text, code_text;
        if(!(fields >> bits_text >> code_text)) continue;
        const auto bits = std::uint32_t(std::stoul(bits_text, nullptr, 16));
        float x;
        std::memcpy(&x, &bits, sizeof x);
        if(unsigned(fp4::e4m3_encode(x)) != unsigned(std::stoul(code_text, nullptr, 16))) {
            std::fprintf(stderr, "FAILED scale vector %s: got 0x%02X, want %s\n", bits_text.c_str(), unsigned(fp4::e4m3_encode(x)), code_text.c_str());
            return 1;
        }
        ++checked;
    }
    REQUIRE(checked > 0);
    // Every byte a checkpoint can hold decodes to its fields' value: bit 7 the sign, 0x7F and 0xFF NaN, magnitude
    // mantissa * 2^-9 at exponent 0, else (8 + mantissa) * 2^(exponent - 10).
    for(int b = 0; b < 0x80; ++b) {
        const float low = fp4::e4m3(std::uint8_t(b)), high = fp4::e4m3(std::uint8_t(b | 0x80));
        const int exponent = b >> 3, mantissa = b & 7;
        const double magnitude = exponent == 0 ? std::ldexp(double(mantissa), -9) : std::ldexp(double(8 + mantissa), exponent - 10);
        REQUIRE(b == 0x7F ? std::isnan(low) && std::isnan(high) : double(low) == magnitude && high == -low && std::signbit(high));
    }
    const float expect[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6};
    for(int c = 0; c < 16; ++c) REQUIRE(fp4::e2m1(std::uint8_t(c)) == expect[c]);
    REQUIRE(fp4::e4m3(0x7E) == 448.0f && fp4::e4m3(0x38) == 1.0f && fp4::e4m3(0x08) == std::ldexp(1.0f, -6));
    REQUIRE(fp4::e4m3(0x01) == std::ldexp(1.0f, -9) && std::isnan(fp4::e4m3(0x7F)) && std::isnan(fp4::e4m3(0xFF)) && fp4::e4m3(0xB8) == -1.0f);
    for(int b = 0; b < 0x7F; ++b) {
        REQUIRE(fp4::e4m3_encode(fp4::e4m3(std::uint8_t(b))) == b);
        if(b) REQUIRE(fp4::e4m3_encode(-fp4::e4m3(std::uint8_t(b))) == (b | 0x80));
    }
    REQUIRE(fp4::e4m3_encode(1000.0f) == 0x7E && fp4::e4m3_encode(1.0625f) == 0x38);  // saturate; tie 1 | 1.125 -> even
    REQUIRE(fp4::e4m3_encode(1.1875f) == 0x3A);                                        // tie 1.125 | 1.25 -> even (1.25)
    const std::pair<float, int> ties[] = {{0.25f, 0}, {0.75f, 2}, {1.25f, 2}, {1.75f, 4}, {2.5f, 4}, {3.5f, 6}, {5.0f, 6}, {7.0f, 7},
                                          {-0.25f, 0}, {-5.0f, 14}, {0.3f, 1}};
    for(auto [x, code] : ties) REQUIRE(fp4::e2m1_encode(x) == code);
    REQUIRE(throws([] { fp4::e2m1_encode(NAN); }) && throws([] { fp4::e4m3_encode(NAN); }));

    std::mt19937 random(5);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const std::size_t rows = 7, cols = 64, tokens = 3;
    std::vector<float> dense(rows * cols), x(tokens * cols);
    for(auto& v : dense) v = normal(random);
    dense[5] = 0;                                  // a zero inside a block
    for(std::size_t i = 0; i < 16; ++i) dense[cols + i] = 0;  // an all-zero block
    for(auto& v : x) v = normal(random);
    const auto m = fp4::Matrix::quantize(dense, rows, cols);
    m.validate();
    float decoded[16];
    double worst = 0;
    std::vector<float> w(rows * cols);
    for(std::size_t n = 0; n < rows; ++n)
        for(std::size_t b = 0; b < cols / 16; ++b) {
            m.decode_block(n, b, decoded);
            float bmax = 0;
            for(std::size_t i = 0; i < 16; ++i) bmax = std::fmax(bmax, std::fabs(dense[n * cols + b * 16 + i]));
            for(std::size_t i = 0; i < 16; ++i) {
                w[n * cols + b * 16 + i] = decoded[i];
                // Elementwise error is at most half an E2M1 step (1.0 at the top) of the block scale, plus scale rounding.
                worst = std::fmax(worst, std::fabs(decoded[i] - dense[n * cols + b * 16 + i]) / (bmax > 0 ? bmax : 1));
            }
        }
    REQUIRE(worst <= 0.25 && w[5] == 0 && w[cols] == 0 && w[cols + 15] == 0);
    std::vector<float> y(tokens * rows);
    fp4::project(m, x, cols, tokens, y, rows);
    for(std::size_t t = 0; t < tokens; ++t)
        for(std::size_t n = 0; n < rows; ++n) {
            double ref = 0;
            for(std::size_t k = 0; k < cols; ++k) ref += double(w[n * cols + k]) * x[t * cols + k];
            REQUIRE(std::fabs(y[t * rows + n] - ref) <= 1e-5 * (1 + std::fabs(ref)));
        }
    std::vector<float> one(rows);
    fp4::project(m, std::span<const float>(x).subspan(cols), cols, 1, one, rows);  // batch of one equals row 1 of the batch
    for(std::size_t n = 0; n < rows; ++n) REQUIRE(one[n] == y[rows + n]);

    REQUIRE(throws([&] { fp4::Matrix::quantize(std::vector<float>(rows * 20), rows, 20); }));
    auto broken = m;
    broken.scales.pop_back();
    REQUIRE(throws([&] { broken.validate(); }));
    broken = m;
    broken.global_scale = 0;
    REQUIRE(throws([&] { broken.validate(); }));
    REQUIRE(throws([&] { fp4::project(m, x, cols - 1, tokens, y, rows); }));
    std::puts("nvfp4 ok");
    return 0;
}
