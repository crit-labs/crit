// Core's draw (crit/sampling.h): Philox known answers and the uniform built on them.
#include <crit/sampling.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>
using namespace crit::core;

namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }

void philox() {
    static_assert(philox4x32(0, 0, 0, 0).word[0] == 0x6627e8d5u && philox4x32(0, 0, 0, 0).word[3] == 0x9b00dbd8u,
                  "philox is usable in constant expressions");
    const auto check = [](PhiloxBlock block, std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d) {
        require(block.word[0] == a && block.word[1] == b && block.word[2] == c && block.word[3] == d,
                "philox4x32-10 matches the Random123 known answer");
    };
    check(philox4x32(0, 0, 0, 0), 0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u);
    check(philox4x32(~0ull, ~0ull, ~0u, ~0u), 0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu);
    // counter {243f6a88, 85a308d3, 13198a2e, 03707344}, key {a4093822, 299f31d0}
    check(philox4x32(0x299f31d0a4093822ull, 0x85a308d3243f6a88ull, 0x13198a2eu, 0x03707344u),
          0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u);
    // Each counter and key word changes the block.
    const auto base = philox4x32(1, 2, 3, 4);
    for(const auto& other:{philox4x32(2, 2, 3, 4), philox4x32(1ull << 32 | 1, 2, 3, 4), philox4x32(1, 3, 3, 4),
                           philox4x32(1, 1ull << 32 | 2, 3, 4), philox4x32(1, 2, 4, 4), philox4x32(1, 2, 3, 5)})
        require(other.word[0] != base.word[0] || other.word[1] != base.word[1], "every input word reaches the output");
}
void uniform_draws() {
    static_assert(uniform(0, 0) == 0xcc4fd1abc2d38p-53 && uniform(~0ull, ~0ull, ~0u, DrawPurpose(~0u)) == 0x811e4eda83907p-53,
                  "uniform known answers from the philox vectors");
    constexpr std::size_t draws = 200000;
    double sum = 0, low = 1, high = 0;
    for(std::uint64_t position = 0; position < draws; ++position) {
        const double u = uniform(42, position, 7, DrawPurpose::draft);
        const auto block = philox4x32(42, position, 7, std::uint32_t(DrawPurpose::draft));
        const std::uint64_t bits = (std::uint64_t(block.word[0]) << 32 | block.word[1]) >> 11;
        require(u == double(bits)/9007199254740992.0, "uniform takes the top 53 bits of words 0 and 1");
        require(u >= 0 && u < 1, "uniform lies in [0, 1)");
        require(u == uniform(42, position, 7, DrawPurpose::draft), "uniform is deterministic");
        sum += u; low = std::min(low, u); high = std::max(high, u);
    }
    const double mean = sum/draws;
    require(std::fabs(mean - 0.5) < 4/std::sqrt(12.0*draws), "uniform mean is one half");
    require(low < 1e-4 && high > 1 - 1e-4, "uniform covers the interval");
    require(uniform(1, 5) == uniform(1, 5, 0, DrawPurpose::sample), "defaults are stream 0 and the sample purpose");
    std::size_t equal = 0;
    for(std::uint64_t position = 0; position < 1000; ++position) {
        const double u = uniform(9, position);
        equal += u == uniform(9, position, 1) || u == uniform(9, position, 0, DrawPurpose::accept) ||
                 u == uniform(10, position) || u == uniform(9, position + 1);
    }
    require(equal == 0, "seed, position, stream and purpose select independent draws");
}
}

int main(int argc, char** argv) {
    try {
        const std::string_view name = argc > 1 ? argv[1] : "all";
        const std::vector<std::pair<std::string_view, std::function<void()>>> cases{
            {"sampling_philox", philox}, {"sampling_uniform", uniform_draws}};
        bool found = false;
        for(const auto& [case_name, run]:cases) if(name == "all" || name == case_name) {
            run();
            found = true;
            std::cout << "crit core sampling passed: " << case_name << '\n';
        }
        require(found, "unknown test case");
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
