// Integer block rows: every ISA this CPU runs gives the scalar oracle's bits for both weight value sets, 16- and
// 32-element activation blocks, 16 and 32 lanes, both reductions, token masks and both scale paths; any row split gives
// the same bits; the block sums are exact (values match a double-precision reference of the same integers within FP32
// rounding of the scale step); the declared quantizers round-trip their own scale and the declared exp stays within
// 2 ulp of std::exp. Prints the throughput of 1, 3 and 8 tokens per ISA and thread count (not asserted).
#include "host_int.h"
#include "nvfp4.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace hi = crit::techniques::host_int;
namespace hn = crit::techniques::host_nvfp4;

namespace {
void require(bool c, const std::string& m) {
    if(!c) throw std::runtime_error(m);
}
std::uint32_t bits(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

struct Case {
    std::size_t rows, cols;
    std::vector<std::uint8_t> codes, scales;
    std::vector<std::int8_t> x;
    std::vector<float> xs;
    int block;
    hi::Weights weights;
    std::vector<std::uint8_t> interleaved;
    hi::IntMatrix matrix() const { return {codes.data(), scales.data(), rows, cols, 0, 0, weights}; }
    hi::IntMatrix interleaved_matrix() const {
        return {interleaved.data(), scales.data(), rows, cols, 0, 0, weights, hi::Layout::interleaved};
    }
    hn::MatrixView view() const { return {codes.data(), scales.data(), rows, cols}; }
    hi::IntActs acts() const { return {x.data(), cols, xs.data(), cols / std::size_t(block), block}; }
};

Case make(std::size_t rows, std::size_t cols, int block, hi::Weights weights, bool e2m1_acts, std::mt19937& rng) {
    Case c{rows, cols, std::vector<std::uint8_t>(rows * cols / 2), std::vector<std::uint8_t>(rows * cols / 16),
           std::vector<std::int8_t>(hi::kMaxTokens * cols), std::vector<float>(hi::kMaxTokens * cols / std::size_t(block)),
           block, weights, {}};
    std::uniform_int_distribution<int> byte(0, 255), scale(0x20, 0x5f), act(-127, 127), e2m1(0, 15);
    std::uniform_real_distribution<float> s(0.001f, 0.1f);
    for(auto& b : c.codes) b = std::uint8_t(byte(rng));
    for(auto& v : c.scales) v = std::uint8_t(scale(rng) | (byte(rng) & 0x80));
    const std::int8_t x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for(auto& v : c.x) v = e2m1_acts ? x2[e2m1(rng)] : std::int8_t(act(rng));
    for(auto& v : c.xs) v = s(rng);
    const std::size_t bytes = hi::row_bytes(hi::Layout::interleaved, cols);
    c.interleaved.resize(rows * bytes);
    for(std::size_t r = 0; r < rows; ++r) hi::interleave_row(c.codes.data() + r * cols / 2, cols, c.interleaved.data() + r * bytes);
    return c;
}

std::vector<float> run(hi::Isa isa, const hi::IntOrder& order, const Case& c, const hn::ScaleDecoder& d, int tokens,
                       std::uint32_t mask, float post, bool interleaved = false) {
    std::vector<float> out(hi::kMaxTokens * c.rows, -1.0f);
    hi::dot_rows_int(isa, order, interleaved ? c.interleaved_matrix() : c.matrix(), d, c.acts(), tokens, mask, 0, c.rows,
                     post, out.data(), c.rows);
    return out;
}

void same(const std::vector<float>& a, const std::vector<float>& b, const std::string& what) {
    for(std::size_t i = 0; i < a.size(); ++i)
        if(bits(a[i]) != bits(b[i]))
            throw std::runtime_error(what + ": element " + std::to_string(i) + " " + std::to_string(a[i]) + " vs " +
                                     std::to_string(b[i]));
}

void check_paths() {
    std::mt19937 rng(11);
    std::vector<hi::Isa> isas{hi::Isa::scalar};
    for(auto isa : {hi::Isa::avx2, hi::Isa::avx512})
        if(hi::int_supported(isa)) isas.push_back(isa);
    int compared = 0;
    for(auto weights : {hi::Weights::e2m1x2, hi::Weights::int4})
        for(int block : {16, 32})
            for(bool e2m1_acts : {true, false})
                for(std::size_t cols : {64u, 192u, 256u, 704u, 2560u}) {
                    Case c = make(37, cols, block, weights, e2m1_acts, rng);
                    hn::MatrixView view = c.view();
                    const hn::ScaleDecoder decoded = hn::e4m3_scales(view);
                    const hn::ScaleDecoder hook{decoded.decode, decoded.context, nullptr};   // decoded rows, no table
                    for(int lanes : {16, 32})
                        for(auto reduce : {hn::Reduce::halving, hn::Reduce::sequential})
                            for(int tokens : {1, 3, 8})
                                for(std::uint32_t mask : {0xffu, 0x5u}) {
                                    const hi::IntOrder order{lanes, reduce};
                                    const auto oracle = run(hi::Isa::scalar, order, c, decoded, tokens, mask, 0.25f);
                                    for(auto isa : isas)
                                        for(const auto* d : {&decoded, &hook})
                                            for(bool interleaved : {false, true}) {
                                                same(run(isa, order, c, *d, tokens, mask, 0.25f, interleaved), oracle,
                                                     std::string(hn::isa_name(isa)) + (interleaved ? " interleaved" : "") +
                                                         " cols " + std::to_string(cols));
                                                ++compared;
                                            }
                                }
                    // Any row split gives the same bits.
                    const hi::IntOrder order{};
                    const auto whole = run(isas.back(), order, c, decoded, 8, 0xff, 1.0f);
                    std::vector<float> split(whole.size(), -1.0f);
                    for(std::size_t r0 = 0; r0 < c.rows; r0 += 7) {
                        const std::size_t r1 = std::min(c.rows, r0 + 7);
                        hi::dot_rows_int(isas.back(), order, c.matrix(), decoded, c.acts(), 8, 0xff, r0, r1, 1.0f,
                                         split.data(), c.rows);
                    }
                    same(split, whole, "row split");
                    // Values: a double-precision dot of the same integers and scales, within FP32 rounding.
                    const std::int8_t* w = hi::detail::weight_values(weights);
                    for(std::size_t r = 0; r < c.rows; ++r)
                        for(int t = 0; t < 8; ++t) {
                            double ref = 0, mag = 0;
                            for(std::size_t k = 0; k < cols; ++k) {
                                const std::uint8_t byte = c.codes[r * cols / 2 + k / 2];
                                const double v = double(w[k % 2 ? byte >> 4 : byte & 15]) *
                                                 crit::nvfp4::e4m3(c.scales[r * cols / 16 + k / 16]) *
                                                 c.x[std::size_t(t) * cols + k] * c.xs[(std::size_t(t) * cols + k) / std::size_t(block)];
                                ref += v;
                                mag += std::fabs(v);
                            }
                            const float got = whole[std::size_t(t) * c.rows + r];
                            require(std::fabs(double(got) - ref) <= 1e-5 * mag + 1e-30,
                                    "value off the double reference by " + std::to_string(double(got) - ref));
                        }
                }
    std::cout << "integer rows: " << compared << " runs bitwise equal to the oracle on";
    for(auto isa : isas) std::cout << " " << hn::isa_name(isa);
    std::cout << "\n";
}

void check_quantizers() {
    std::mt19937 rng(5);
    std::normal_distribution<float> n(0.0f, 3.0f);
    std::vector<float> x(4096);
    for(auto& v : x) v = n(rng);
    std::vector<std::int8_t> q(x.size());
    std::vector<float> s(x.size() / 16);
    std::vector<std::uint8_t> bytes(x.size() / 16);
    const float global = 64.0f;
    hi::quantize_e2m1x2(x.data(), x.size(), global, q.data(), s.data(), bytes.data());
    for(std::size_t i = 0; i < x.size(); ++i) {
        const float decoded = crit::nvfp4::e4m3(bytes[i / 16]);
        require(s[i / 16] == decoded / global, "e2m1 scale");
        const float expect = 2.0f * crit::nvfp4::e2m1(crit::nvfp4::e2m1_encode((x[i] * global) / decoded));
        require(float(q[i]) == expect, "e2m1 value");
    }
    // The block scale's code is crit::nvfp4::e4m3_encode's (a search for the nearest of 127 codes, ties to even) for
    // block maxima that land on every E4M3 midpoint and a few FP32 steps around it, and for random ones.
    std::size_t scale_checks = 0, value_checks = 0;
    for(int code = 0; code < 126; ++code) {
        const float a = crit::nvfp4::e4m3(std::uint8_t(code)), b = crit::nvfp4::e4m3(std::uint8_t(code + 1));
        const float mid = a + (b - a) / 2.0f;
        for(int step = -3; step <= 3; ++step) {
            float wanted = mid;
            for(int k = 0; k < (step < 0 ? -step : step); ++k) wanted = std::nextafter(wanted, step < 0 ? 0.0f : 1000.0f);
            std::vector<float> block(16, 0.0f);
            block[3] = wanted * 6.0f;
            std::vector<std::int8_t> bv(16);
            float bs = 0.0f;
            std::uint8_t byte = 0;
            hi::quantize_e2m1x2(block.data(), 16, 1.0f, bv.data(), &bs, &byte);
            require(byte == crit::nvfp4::e4m3_encode((block[3] * 1.0f) / 6.0f), "e4m3 scale code at a midpoint");
            ++scale_checks;
        }
    }
    for(std::size_t b = 0; b < x.size() / 16; ++b) {
        float amax = 0.0f;
        for(int i = 0; i < 16; ++i) amax = std::fmax(amax, std::fabs(x[16 * b + std::size_t(i)]));
        require(bytes[b] == crit::nvfp4::e4m3_encode((amax * global) / 6.0f), "e4m3 scale code");
        ++scale_checks;
    }
    // E2M1 codes at every midpoint and a few FP32 steps around it: a block whose maximum 6 gives the scale 1, so each
    // value's quotient is the value itself.
    for(float mid : {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f})
        for(int step = -3; step <= 3; ++step)
            for(float sign : {1.0f, -1.0f}) {
                float v = mid;
                for(int k = 0; k < (step < 0 ? -step : step); ++k) v = std::nextafter(v, step < 0 ? 0.0f : 10.0f);
                std::vector<float> block(16, 0.0f);
                block[0] = 6.0f;
                block[5] = sign * v;
                std::vector<std::int8_t> bv(16);
                float bs = 0.0f;
                hi::quantize_e2m1x2(block.data(), 16, 1.0f, bv.data(), &bs);
                require(bs == 1.0f, "a block of maximum 6 has the scale 1");
                require(float(bv[5]) == 2.0f * crit::nvfp4::e2m1(crit::nvfp4::e2m1_encode(sign * v)), "e2m1 code at a midpoint");
                ++value_checks;
            }
    std::vector<float> d(x.size() / 32);
    hi::quantize_int8(x.data(), x.size(), 32, q.data(), d.data());
    for(std::size_t i = 0; i < x.size(); ++i) require(std::fabs(float(q[i]) * d[i / 32] - x[i]) <= 0.5f * d[i / 32] * 1.0001f,
                                                      "int8 rounding");
    double worst = 0;
    for(float v = -87.0f; v <= 88.0f; v += 0.01953125f) {
        const double exact = std::exp(double(v)), got = hi::exp_declared(v);
        worst = std::max(worst, std::fabs(got - exact) / exact);
    }
    require(worst < 2.5e-7, "exp_declared relative error " + std::to_string(worst));
    require(hi::silu_declared(0.0f) == 0.0f && hi::silu_declared(20.0f) > 19.99f, "silu");
    std::cout << "quantizers and exp: " << scale_checks << " scale codes and " << value_checks << " midpoint values as crit::nvfp4's encoders give them, max relative exp error " << worst << "\n";
}

void throughput() {
    std::mt19937 rng(3);
    const std::size_t rows = 1920, cols = 2560, experts = 96;   // 2.46 MB of codes each, past the caches
    std::vector<Case> cases;
    for(std::size_t e = 0; e < experts; ++e) cases.push_back(make(rows, cols, 32, hi::Weights::e2m1x2, false, rng));
    std::vector<hi::Isa> isas{hi::Isa::scalar};
    for(auto isa : {hi::Isa::avx2, hi::Isa::avx512})
        if(hi::int_supported(isa)) isas.push_back(isa);
    const unsigned threads = std::max(1u, std::min(6u, std::thread::hardware_concurrency() / 2));
    for(auto isa : isas)
      for(bool interleaved : {false, true})
        for(int tokens : {1, 3, 8})
            for(unsigned workers : {1u, threads}) {
                if(isa == hi::Isa::scalar && (workers > 1 || interleaved)) continue;
                const auto t0 = std::chrono::steady_clock::now();
                std::vector<std::thread> pool;
                for(unsigned w = 0; w < workers; ++w)
                    pool.emplace_back([&, w] {
                        std::vector<float> out(hi::kMaxTokens * rows);
                        for(const auto& c : cases) {
                            hn::MatrixView view = c.view();
                            hi::dot_rows_int(isa, {}, interleaved ? c.interleaved_matrix() : c.matrix(),
                                             hn::e4m3_scales(view), c.acts(), tokens, 0xff,
                                             rows * w / workers, rows * (w + 1) / workers, 1.0f, out.data(), rows);
                        }
                    });
                for(auto& p : pool) p.join();
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const double bytes = double(experts) * rows * (cols / 2 + cols / 16);
                std::cout << "  " << hn::isa_name(isa) << (interleaved ? " interleaved" : " rows") << " tokens " << tokens << " threads " << workers << ": "
                          << bytes / s / 1e9 << " GB/s of weights\n";
            }
}
}  // namespace

int main() {
    try {
        check_paths();
        check_quantizers();
        throughput();
        std::cout << "host_int tests passed\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
