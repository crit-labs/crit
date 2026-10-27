#pragma once
// Stochastic rounding of FP32 recurrent state to BF16 storage, for engines that keep state below FP32. Round to
// nearest biases a state that is rounded once per token: the error of every step points the same way for a slowly
// changing value, and it accumulates over a long sequence. Rounding up with probability equal to the discarded
// fraction makes each stored value an unbiased estimate of the FP32 one. The random bits are Core's counter-based
// Philox draw keyed by (seed, position, layer, element), so a replay or a recompute of the same step stores the same
// bits: recovery's bitwise checkpoint-recompute identity still holds.
#include <crit/sampling.h>

#include <cstdint>
#include <cstring>

namespace crit::techniques::recurrent {

// Philox purposes at or above this value are reserved for state rounding; the low bits carry the layer.
inline constexpr std::uint32_t kStateRoundingPurpose = 0x80000000u;

CRIT_HOST_DEVICE inline std::uint32_t float_bits(float x) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(x);
#else
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return bits;
#endif
}

// BF16 bits of x rounded up with probability (discarded low 16 bits) / 2^16 from the 16 random bits given. Exact BF16
// values and infinities are unchanged; NaN stays a quiet NaN. A finite value within one BF16 step of the largest
// finite BF16 may round to infinity, as round-to-nearest would at the halfway point.
CRIT_HOST_DEVICE inline std::uint16_t bf16_stochastic(float x, std::uint32_t random16) {
    const std::uint32_t bits = float_bits(x);
    if((bits & 0x7f800000u) == 0x7f800000u)   // infinity, or NaN made quiet
        return static_cast<std::uint16_t>((bits >> 16) | ((bits & 0x007fffffu) ? 0x0040u : 0u));
    return static_cast<std::uint16_t>((bits + (random16 & 0xffffu)) >> 16);
}

// The 16 random bits for element `element` of layer `layer`'s state written at token `position`: one Philox block
// serves eight consecutive elements.
CRIT_HOST_DEVICE inline std::uint32_t state_rounding_bits(std::uint64_t seed, std::uint64_t position, std::uint32_t layer,
                                                          std::uint64_t element) {
    const auto block = crit::core::philox4x32(seed, position, static_cast<std::uint32_t>(element >> 3),
                                              kStateRoundingPurpose | (layer & 0x7fffffffu));
    const std::uint32_t word = block.word[(element >> 1) & 3u];
    return (element & 1u) ? word >> 16 : word & 0xffffu;
}

}  // namespace crit::techniques::recurrent
