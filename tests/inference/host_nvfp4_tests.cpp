// Host NVFP4 rows in a declared order: every ISA this CPU supports gives the scalar oracle's bits for every
// order, shape, token mask and scale hook; any row split gives the same bits; values agree with crit::nvfp4's
// projection; a different reduction order changes bits. Prints single-thread throughput per ISA (not asserted).
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
    const std::vector<hn::Isa> isas{hn::Isa::avx2, hn::Isa::avx512};
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
            for(hn::Isa isa : {hn::Isa::scalar, hn::Isa::avx2, hn::Isa::avx512}) {
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

void throughput() {
    std::mt19937 rng(5);
    const Case c = make(1024, 4096, rng);  // 2 MiB of codes
    const hn::MatrixView c_view = c.view();
    const auto d = hn::e4m3_scales(c_view);
    std::vector<float> out(hn::kMaxTokens * c.rows);
    for(hn::Isa isa : {hn::Isa::scalar, hn::Isa::avx2, hn::Isa::avx512}) {
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
}
}  // namespace

int main(int argc, char** argv) {
    try {
        parity();
        splits();
        semantics();
        orders_differ();
        golden();
        if(argc > 1 && std::string(argv[1]) == "--throughput") throughput();
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
