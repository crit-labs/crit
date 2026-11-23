#pragma once
#include <cstdint>

// The draw of Core's sampling rules (core/src/sampling/README.md), header-only. Host and device annotation, so device kernels can reproduce draws bit for bit.
#ifndef CRIT_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CRIT_HOST_DEVICE __host__ __device__
#else
#define CRIT_HOST_DEVICE
#endif
#endif

namespace crit::core {
// Every draw is a pure function of (seed, position, stream, purpose): rollback, stop/budget trimming,
// cancellation and speculative redo need no generator state. Position = committed outputs before the
// draw plus the verify column.
enum class DrawPurpose : std::uint32_t { sample = 0, draft = 1, accept = 2, residual = 3, block = 4, user = 16 };
// The four 32-bit output words of one Philox4x32 block, in Random123 order (a plain array is device-safe).
struct PhiloxBlock { std::uint32_t word[4] = {}; };
// Philox4x32-10 with the Random123 constants. Key = {seed low, seed high};
// counter = {position low, position high, stream, purpose}.
CRIT_HOST_DEVICE constexpr PhiloxBlock philox4x32(std::uint64_t seed, std::uint64_t position, std::uint32_t stream,
                                                  std::uint32_t purpose) noexcept {
    std::uint32_t c0 = static_cast<std::uint32_t>(position), c1 = static_cast<std::uint32_t>(position >> 32);
    std::uint32_t c2 = stream, c3 = purpose;
    std::uint32_t k0 = static_cast<std::uint32_t>(seed), k1 = static_cast<std::uint32_t>(seed >> 32);
    for(int round = 0; round < 10; ++round) {
        if(round) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        const std::uint64_t p0 = std::uint64_t{0xD2511F53u} * c0, p1 = std::uint64_t{0xCD9E8D57u} * c2;
        const std::uint32_t hi0 = static_cast<std::uint32_t>(p0 >> 32), lo0 = static_cast<std::uint32_t>(p0);
        const std::uint32_t hi1 = static_cast<std::uint32_t>(p1 >> 32), lo1 = static_cast<std::uint32_t>(p1);
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
    }
    return {{c0, c1, c2, c3}};
}
// [0, 1) with 53 random bits: ((uint64(word[0]) << 32 | word[1]) >> 11) * 2^-53.
CRIT_HOST_DEVICE constexpr double uniform(std::uint64_t seed, std::uint64_t position, std::uint32_t stream = 0,
                                          DrawPurpose purpose = DrawPurpose::sample) noexcept {
    const PhiloxBlock block = philox4x32(seed, position, stream, static_cast<std::uint32_t>(purpose));
    const std::uint64_t bits = (std::uint64_t{block.word[0]} << 32 | block.word[1]) >> 11;
    return static_cast<double>(bits) * 0x1.0p-53;
}
}
