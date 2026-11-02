// Host block-scaled rows in a declared order: every ISA this CPU supports gives the scalar oracle's bits for every
// order, shape, token mask and scale hook, and the portable path does for every format (NVFP4, MXFP4, MXFP8 E4M3 and
// E5M2, MXINT8, INT4); any row split gives the same bits; values agree with crit::nvfp4's projection and with a
// double-precision reference for every format; a different reduction order changes bits. Prints single-thread
// throughput per ISA (not asserted).
#include "host_nvfp4.h"
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
    std::vector<float> x;
    hn::MatrixView view() const { return {codes.data(), scales.data(), rows, cols}; }
};

Case make(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    Case c{rows, cols, std::vector<std::uint8_t>(rows * cols / 2), std::vector<std::uint8_t>(rows * cols / 16),
           std::vector<float>(hn::kMaxTokens * cols)};
    std::uniform_int_distribution<int> byte(0, 255), scale(0x20, 0x5f);  // finite, moderate E4M3 scales
    std::normal_distribution<float> act(0.0f, 1.0f);
    for(auto& b : c.codes) b = std::uint8_t(byte(rng));
    for(auto& s : c.scales) s = std::uint8_t(scale(rng) | (byte(rng) & 0x80));
    for(auto& v : c.x) v = act(rng);
    return c;
}

std::vector<float> run(hn::Isa isa, const hn::Order& order, const Case& c, const hn::ScaleDecoder& d, int tokens,
                       std::uint32_t mask, float post) {
    std::vector<float> out(hn::kMaxTokens * c.rows, -1.0f);
    hn::dot_rows(isa, order, c.view(), d, c.x.data(), c.cols, tokens, mask, 0, c.rows, post, out.data(), c.rows);
    return out;
}

// A checkpoint-style hook: scale bytes through an engine table (here E4M3 halved, a different value per byte).
struct HalvedTable {
    hn::MatrixView m;
    float value[256];
};
void decode_halved(const void* ctx, std::size_t row, float* out) {
    const auto& h = *static_cast<const HalvedTable*>(ctx);
    for(std::size_t b = 0; b < h.m.cols / 16; ++b) out[b] = h.value[h.m.scales[row * (h.m.cols / 16) + b]];
}

void parity() {
    std::mt19937 rng(20261019);
    const std::vector<hn::Isa> isas{hn::Isa::portable, hn::Isa::avx2, hn::Isa::avx512};
    int compared = 0;
    for(auto [rows, cols] : {std::pair<std::size_t, std::size_t>{33, 64}, {40, 1024}, {24, 2688}, {8, 4096}, {18, 640},
                             {11, 1040}}) {
        const Case c = make(rows, cols, rng);
        const hn::MatrixView v = c.view();
        HalvedTable halved{v, {}};
        for(int b = 0; b < 256; ++b) halved.value[b] = 0.5f * crit::nvfp4::e4m3(std::uint8_t(b));
        const hn::ScaleDecoder decoders[2] = {hn::e4m3_scales(v), {decode_halved, &halved}};
        for(hn::Product product : {hn::Product::scaled_elements, hn::Product::block_dot})
        for(int lanes : {16, 32})
            for(hn::Reduce reduce : {hn::Reduce::halving, hn::Reduce::sequential})
                for(const auto& d : decoders)
                    for(auto [tokens, mask] : {std::pair<int, std::uint32_t>{1, 1u}, {3, 5u}, {4, 15u}, {8, 0xa7u}}) {
                        if(product == hn::Product::scaled_elements && cols % std::size_t(lanes)) continue;
                        const hn::Order order{lanes, reduce, product};
                        const auto oracle = run(hn::Isa::scalar, order, c, d, tokens, mask, 0.0625f);
                        for(hn::Isa isa : isas) {
                            if(!hn::supported(isa)) continue;
                            const auto got = run(isa, order, c, d, tokens, mask, 0.0625f);
                            for(std::size_t i = 0; i < got.size(); ++i)
                                require(bits(got[i]) == bits(oracle[i]), std::string(hn::isa_name(isa)) +
                                        " differs from the oracle at " + std::to_string(i) + " cols " +
                                        std::to_string(cols));
                            ++compared;
                        }
                    }
    }
    std::cout << std::unitbuf << "parity: " << compared << " ISA runs bitwise equal to the scalar oracle (best "
              << hn::isa_name(hn::best_isa()) << ")\n";
}

// Rows split over threads in different chunkings give the same bits.
void splits() {
    std::mt19937 rng(7);
    const Case c = make(97, 2688, rng);
    const hn::MatrixView c_view = c.view();
    const auto d = hn::e4m3_scales(c_view);
    const hn::Order order{};
    const auto whole = run(hn::best_isa(), order, c, d, 4, 15u, 1.0f);
    for(std::size_t parts : {2u, 3u, 7u}) {
        std::vector<float> out(hn::kMaxTokens * c.rows, -1.0f);
        std::vector<std::thread> pool;
        for(std::size_t p = 0; p < parts; ++p)
            pool.emplace_back([&, p] {
                const std::size_t r0 = c.rows * p / parts, r1 = c.rows * (p + 1) / parts;
                hn::dot_rows(hn::best_isa(), order, c.view(), d, c.x.data(), c.cols, 4, 15u, r0, r1, 1.0f, out.data(),
                             c.rows);
            });
        for(auto& t : pool) t.join();
        for(std::size_t i = 0; i < 4 * c.rows; ++i) require(bits(out[i]) == bits(whole[i]), "split changed bits");
    }
    std::cout << "splits: 2, 3 and 7 threads equal one call\n";
}

// The declared order computes the published format: close to crit::nvfp4::project (its own fixed order).
void semantics() {
    std::mt19937 rng(11);
    const Case c = make(16, 1024, rng);
    crit::nvfp4::Matrix m{c.rows, c.cols, c.codes, c.scales, 4.0f};
    for(auto& s : m.scales) s &= 0x7f;  // the codec rejects nothing, but keep scales positive like checkpoints
    Case positive = c;
    positive.scales = m.scales;
    const hn::MatrixView positive_view = positive.view();
    std::vector<float> ref(c.rows);
    crit::nvfp4::project(m, std::span<const float>(c.x.data(), c.cols), c.cols, 1, ref, c.rows);
    const auto got = run(hn::Isa::scalar, {}, positive, hn::e4m3_scales(positive_view), 1, 1u, 1.0f / 4.0f);
    for(std::size_t r = 0; r < c.rows; ++r) {
        double mag = 0;
        for(std::size_t k = 0; k < c.cols; ++k) mag += std::fabs(double(c.x[k]));
        require(std::fabs(double(got[r]) - double(ref[r])) <= 1e-5 * mag * 448.0 * 6.0 / 4.0,
                "declared order disagrees with crit::nvfp4::project beyond FP32 rounding");
    }
    std::cout << "semantics: agrees with crit::nvfp4::project\n";
}

// Two orders over the same bytes: some rows round differently, which is why the order is part of the contract.
void orders_differ() {
    std::mt19937 rng(3);
    const Case c = make(256, 2688, rng);
    const hn::MatrixView c_view = c.view();
    const auto d = hn::e4m3_scales(c_view);
    const auto a = run(hn::Isa::scalar, {16, hn::Reduce::halving}, c, d, 1, 1u, 1.0f);
    const auto b = run(hn::Isa::scalar, {16, hn::Reduce::sequential}, c, d, 1, 1u, 1.0f);
    const auto e = run(hn::Isa::scalar, {32, hn::Reduce::halving}, c, d, 1, 1u, 1.0f);
    int differ_reduce = 0, differ_lanes = 0;
    for(std::size_t r = 0; r < c.rows; ++r) {
        differ_reduce += bits(a[r]) != bits(b[r]);
        differ_lanes += bits(a[r]) != bits(e[r]);
    }
    require(differ_reduce > 0 && differ_lanes > 0, "orders should not all round alike");
    std::cout << "orders: " << differ_reduce << "/256 rows differ between reductions, " << differ_lanes
              << "/256 between 16 and 32 lanes\n";
}

// Known answers from the Python reference (implementation.py) for inputs drawn from a portable LCG: the oracle
// and every ISA give the same bits on every platform and compiler.
void golden() {
    std::uint32_t state = 12345;
    auto lcg = [&] { return state = state * 1664525u + 1013904223u; };
    const std::size_t rows = 4, cols = 256;
    Case c{rows, cols, std::vector<std::uint8_t>(rows * cols / 2), std::vector<std::uint8_t>(rows * cols / 16),
           std::vector<float>(cols)};
    for(auto& b : c.codes) b = std::uint8_t(lcg() >> 24);
    for(auto& s : c.scales) {
        const std::uint32_t magnitude = 0x20 + (lcg() >> 24) % 0x40;
        const std::uint32_t sign = (lcg() >> 31) ? 0x80 : 0;
        s = std::uint8_t(magnitude | sign);
    }
    for(auto& v : c.x) v = float(lcg() >> 8) / float(1 << 24) * 4.0f - 2.0f;
    const std::uint32_t expect[32] = {
        0x411e67a4u, 0x41eac158u, 0xc2df4c6bu, 0x42d91c6au, 0x411e67a4u, 0x41eac15au, 0xc2df4c6du, 0x42d91c6bu,
        0x411e67acu, 0x41eac156u, 0xc2df4c6bu, 0x42d91c69u, 0x411e67a9u, 0x41eac15bu, 0xc2df4c70u, 0x42d91c69u,
        0x411e67b2u, 0x41eac15au, 0xc2df4c6eu, 0x42d91c6au, 0x411e67b1u, 0x41eac15au, 0xc2df4c6fu, 0x42d91c68u,
        0x411e67b2u, 0x41eac15au, 0xc2df4c6eu, 0x42d91c6au, 0x411e67b1u, 0x41eac15au, 0xc2df4c6fu, 0x42d91c68u};
    const hn::MatrixView c_view = c.view();
    int i = 0;
    for(hn::Product product : {hn::Product::scaled_elements, hn::Product::block_dot})
    for(int lanes : {16, 32})
        for(hn::Reduce reduce : {hn::Reduce::halving, hn::Reduce::sequential}) {
            for(hn::Isa isa : {hn::Isa::scalar, hn::Isa::portable, hn::Isa::avx2, hn::Isa::avx512}) {
                if(!hn::supported(isa)) continue;
                std::vector<float> out(rows);
                hn::dot_rows(isa, {lanes, reduce, product}, c_view, hn::e4m3_scales(c_view), c.x.data(), cols, 1, 1u, 0, rows,
                             0.25f, out.data(), rows);
                for(std::size_t r = 0; r < rows; ++r)
                    require(bits(out[r]) == expect[i + int(r)], std::string("golden mismatch on ") + hn::isa_name(isa));
            }
            i += int(rows);
        }
    std::cout << "golden: the Python reference's bits on every supported ISA\n";
}

// Every format on every path: the portable path (and any ISA the format reaches) gives the oracle's bits, and the
// oracle's value is the format's value: within FP32 rounding of a double-precision sum of the decoded elements.
struct Named {
    const char* name;
    hn::Format format;
    bool e8m0;  // block scales as E8M0 bytes, else E4M3
};
void formats() {
    std::mt19937 rng(20261101);
    const Named all[] = {{"nvfp4", hn::nvfp4_format(), false},
                         {"mxfp4", hn::mxfp4_format(), true},
                         {"mxfp8-e4m3", hn::mxfp8_e4m3_format(), true},
                         {"mxfp8-e5m2", hn::mxfp8_e5m2_format(), true},
                         {"mxint8", hn::mxint8_format(), true},
                         {"int4-g32-e4m3", {4, 32, hn::int4_values()}, false},
                         {"int4-g16-e8m0", {4, 16, hn::int4_values()}, true}};
    int compared = 0;
    for(const Named& f : all) {
        for(auto [rows, cols] : {std::pair<std::size_t, std::size_t>{17, 512}, {9, 2688}, {5, 1056}}) {
            const std::size_t B = std::size_t(f.format.block);
            if(cols % B) continue;
            std::vector<std::uint8_t> codes(rows * cols * std::size_t(f.format.bits) / 8), scales(rows * cols / B);
            std::vector<float> x(hn::kMaxTokens * cols);
            std::uniform_int_distribution<int> byte(0, 255), e8m0(117, 137), e4m3(0x20, 0x5f);
            std::normal_distribution<float> act(0.0f, 1.0f);
            for(auto& b : codes) {
                b = std::uint8_t(byte(rng));
                // FP8 codes that decode to NaN or infinity are not weights
                while(f.format.bits == 8 && !std::isfinite(f.format.values[b])) b = std::uint8_t(byte(rng));
            }
            for(auto& b : scales) b = std::uint8_t(f.e8m0 ? e8m0(rng) : (e4m3(rng) | (byte(rng) & 0x80)));
            for(auto& v : x) v = act(rng);
            hn::MatrixView v{codes.data(), scales.data(), rows, cols};
            v.format = f.format;
            const hn::ScaleDecoder d = f.e8m0 ? hn::e8m0_scales(v) : hn::e4m3_scales(v);
            for(hn::Product product : {hn::Product::scaled_elements, hn::Product::block_dot})
                for(int lanes : {16, 32})
                    for(hn::Reduce reduce : {hn::Reduce::halving, hn::Reduce::sequential}) {
                        if(product == hn::Product::scaled_elements && cols % std::size_t(lanes)) continue;
                        const hn::Order order{lanes, reduce, product};
                        auto run_on = [&](hn::Isa isa) {
                            std::vector<float> out(hn::kMaxTokens * rows, -1.0f);
                            hn::dot_rows(isa, order, v, d, x.data(), cols, 3, 5u, 0, rows, 0.5f, out.data(), rows);
                            return out;
                        };
                        const auto oracle = run_on(hn::Isa::scalar);
                        for(hn::Isa isa : {hn::Isa::portable, hn::Isa::avx2, hn::Isa::avx512}) {
                            if(!hn::supported(isa)) continue;
                            const auto got = run_on(isa);
                            for(std::size_t i = 0; i < got.size(); ++i)
                                require(bits(got[i]) == bits(oracle[i]), std::string(f.name) + " on " +
                                        hn::isa_name(isa) + " differs from the oracle at " + std::to_string(i));
                            ++compared;
                        }
                        // the value: a double sum of element * scale * x, within FP32 accumulation error
                        std::vector<float> scale_row(cols / B);
                        for(std::size_t r = 0; r < rows; ++r) {
                            d.decode(d.context, r, scale_row.data());
                            for(int t : {0, 2}) {
                                double sum = 0, mag = 0;
                                for(std::size_t k = 0; k < cols; ++k) {
                                    const std::uint8_t* row = codes.data() + r * cols * std::size_t(f.format.bits) / 8;
                                    const int code = f.format.bits == 8 ? row[k] : (k % 2 ? row[k / 2] >> 4 : row[k / 2] & 15);
                                    const double term = double(f.format.values[code]) * scale_row[k / B] * x[std::size_t(t) * cols + k];
                                    sum += term;
                                    mag += std::fabs(term);
                                }
                                const double got = oracle[std::size_t(t) * rows + r];
                                require(std::fabs(got - 0.5 * sum) <= 1e-5 * mag, std::string(f.name) +
                                        ": the declared order does not compute the format's value");
                            }
                        }
                    }
        }
    }
    std::cout << "formats: " << compared << " runs of 7 formats bitwise equal to the oracle and to their values\n";
}

void throughput() {
    std::mt19937 rng(5);
    const Case c = make(1024, 4096, rng);  // 2 MiB of codes
    const hn::MatrixView c_view = c.view();
    const auto d = hn::e4m3_scales(c_view);
    std::vector<float> out(hn::kMaxTokens * c.rows);
    for(hn::Isa isa : {hn::Isa::scalar, hn::Isa::portable, hn::Isa::avx2, hn::Isa::avx512}) {
        if(!hn::supported(isa)) continue;
        for(int tokens : {1, 4}) {
            const int reps = isa == hn::Isa::scalar ? 2 : 20;
            const auto t0 = std::chrono::steady_clock::now();
            for(int i = 0; i < reps; ++i)
                hn::dot_rows(isa, {}, c.view(), d, c.x.data(), c.cols, tokens, (1u << tokens) - 1, 0, c.rows, 1.0f,
                             out.data(), c.rows);
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double bytes = double(c.codes.size() + c.scales.size()) * reps;
            std::cout << "throughput " << hn::isa_name(isa) << " tokens=" << tokens << ": " << bytes / s / 1e9
                      << " GB/s of weights, one thread\n";
        }
    }
    // A format the hand-written paths do not cover runs the portable rows, compiled for each ISA.
    std::vector<std::uint8_t> codes(1024 * 4096), scales(1024 * 4096 / 32);
    for(auto& b : codes) b = std::uint8_t(rng() % 0x7e);  // finite E4M3
    for(auto& b : scales) b = std::uint8_t(120 + rng() % 16);
    hn::MatrixView v{codes.data(), scales.data(), 1024, 4096};
    v.format = hn::mxfp8_e4m3_format();
    const auto e8 = hn::e8m0_scales(v);
    for(hn::Isa isa : {hn::Isa::portable, hn::Isa::avx2, hn::Isa::avx512}) {
        if(!hn::supported(isa)) continue;
        const auto t0 = std::chrono::steady_clock::now();
        for(int i = 0; i < 5; ++i) hn::dot_rows(isa, {}, v, e8, c.x.data(), 4096, 1, 1u, 0, 1024, 1.0f, out.data(), 1024);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "throughput mxfp8 " << hn::isa_name(isa) << " tokens=1: "
                  << double(codes.size() + scales.size()) * 5 / s / 1e9 << " GB/s of weights, one thread\n";
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        parity();
        splits();
        semantics();
        orders_differ();
        golden();
        formats();
        if(argc > 1 && std::string(argv[1]) == "--throughput") throughput();
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
