// Host side of the matched FP8 projections (the order the README declares; its device functions and kernels are
// crit's kernel fp8-block-rows): the declared activation quantizer's twin, E8M0 scale decoding, and an FP64
// projection to bound a kernel's error. The tensor core's block sum has no host twin: FP8
// outputs are compared kernel against kernel, on the target GPU.
#pragma once
#include "row_fp8.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace crit::fp8_block {

enum class ScaleKind : int { mx = 0, amax = 1 };

// An MX scale byte: 2^(e - 127); 255 is NaN and rejected.
inline float e8m0(std::uint8_t e) {
    if(e == 255) throw std::invalid_argument("E8M0 NaN scale");
    return std::ldexp(1.0f, int(e) - 127);
}

// The device quantizer's twin, operation for operation in FP32: codes for n values and the group's scale.
inline float quantize_group(const float* v, std::size_t n, ScaleKind kind, std::uint8_t* codes) {
    float amax = 0.0f;
    for(std::size_t k = 0; k < n; ++k) amax = std::fmax(amax, std::fabs(v[k]));
    float scale;
    if(kind == ScaleKind::mx) {
        int e = 0;
        (void)std::frexp(amax, &e);
        int shared = amax > 0.0f ? e - 1 - 8 : 0;
        shared = shared < -127 ? -127 : shared > 127 ? 127 : shared;
        scale = std::ldexp(1.0f, shared);
    } else {
        scale = amax / 448.0f;
    }
    for(std::size_t k = 0; k < n; ++k) codes[k] = scale > 0.0f ? fp8::encode(v[k] / scale) : 0;
    return scale;
}

// FP64 value of one output: sum over blocks of weight scale * activation scale * the block's products, for error
// bounds. weight_scales [rows / scale_rows][cols / scale_k], act_scales [cols / group] for this token.
inline double project_fp64(const std::uint8_t* wrow, const float* weight_scales, std::size_t scale_k,
                           const std::uint8_t* x, const float* act_scales, std::size_t group, std::size_t cols,
                           double* magnitude = nullptr) {
    double sum = 0.0, abs_sum = 0.0;
    for(std::size_t k = 0; k < cols; ++k) {
        const double p = double(fp8::decode(wrow[k])) * double(fp8::decode(x[k])) * double(weight_scales[k / scale_k]) *
                         double(act_scales[k / group]);
        sum += p;
        abs_sum += std::fabs(p);
    }
    if(magnitude) *magnitude = abs_sum;
    return sum;
}

}  // namespace crit::fp8_block
