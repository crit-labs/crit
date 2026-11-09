// Device side of host_int.h, for CUDA and HIP from one source: the integer block rows, the declared activation
// quantizers and the declared exp, SiLU and SwiGLU. Block sums are exact integers (dp4a on CUDA, plain int32 math
// elsewhere), and every FP32 step uses an explicit round-to-nearest intrinsic so nothing contracts, so a device row
// gives host_int's oracle bits. The crit-labs/int-block-experts Hub kernel carries these functions.
#pragma once
#include <cstdint>

#include "host_nvfp4_device.cuh"

namespace crit::techniques::host_int {

namespace device_detail {
// E2M1 x 2 and INT4 values of a 4-bit code.
__device__ inline int weight_value(unsigned code, bool int4) {
    if(int4) return code < 8 ? int(code) : int(code) - 16;
    const int magnitude[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    return (code & 8u) ? -magnitude[code & 7u] : magnitude[code & 7u];
}
// Four consecutive codes of a rows-layout row (from element k, k % 4 == 0) as four packed int8 values.
__device__ inline int four_weights(const std::uint8_t* codes, int k, bool int4) {
    const unsigned short pair = *reinterpret_cast<const unsigned short*>(codes + k / 2);
    unsigned packed = 0;
    for(int i = 0; i < 4; ++i)
        packed |= (unsigned(weight_value((pair >> (4 * i)) & 15u, int4)) & 0xffu) << (8 * i);
    return int(packed);
}
__device__ inline int dot4(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int sum = c;
    for(int i = 0; i < 4; ++i) sum += int(std::int8_t(a >> (8 * i))) * int(std::int8_t(b >> (8 * i)));
    return sum;
#endif
}
}  // namespace device_detail

// The integer block row: called by L converged lanes (16 or 32) of one group. codes: the row's rows-layout codes;
// wscales: its decoded block scales; values / ascales: one token's int8 activations and block scales (ablock 16 or
// 32). Lane j takes blocks j, j + L, ... in order: exact block sum, then lane = fma(float(sum), wscale * ascale, lane);
// the halving shuffle tree; times post. Returns the row's value on the group's first lane.
template <int L>
__device__ inline float dot_row_int(const std::uint8_t* codes, const float* wscales, const std::int8_t* values,
                                    const float* ascales, int ablock, int cols, bool int4, float post) {
    static_assert(L == 16 || L == 32, "16 or 32 lanes");
    const unsigned lane = threadIdx.x % 32u;
    const int ratio = ablock / 16;
    float acc = 0.0f;
    for(int b = int(lane % unsigned(L)); b < cols / 16; b += L) {
        int sum = 0;
        for(int q = 0; q < 4; ++q)
            sum = device_detail::dot4(device_detail::four_weights(codes, 16 * b + 4 * q, int4),
                                      *reinterpret_cast<const int*>(values + 16 * b + 4 * q), sum);
        acc = __fmaf_rn(float(sum), __fmul_rn(wscales[b], ascales[b / ratio]), acc);
    }
    return __fmul_rn(host_nvfp4::device_detail::halving<L>(acc), post);
}

// crit::nvfp4's encoders on the device: nearest value, ties to the even code, saturating (E2M1 at 6, E4M3 at 448).
__device__ inline unsigned e2m1_encode_device(float x) {
    const float m = fminf(fabsf(x), 6.0f);
    const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    unsigned code = 0;
    while(code < 7 && (m > mids[code] || (m == mids[code] && (code % 2) == 1))) ++code;
    return code | (x < 0.0f && code ? 8u : 0u);
}
__device__ inline float e4m3_device(unsigned byte) {
    const int exponent = int(byte >> 3) & 15, mantissa = int(byte) & 7;
    const float magnitude = exponent == 0 ? ldexpf(float(mantissa) / 8.0f, -6)
                                          : ldexpf(1.0f + float(mantissa) / 8.0f, exponent - 7);
    return (byte & 0x80u) ? -magnitude : magnitude;
}
__device__ inline unsigned e4m3_encode_device(float x) {
    const float m = fminf(fabsf(x), 448.0f);
    unsigned best = 0;
    for(unsigned i = 1; i < 0x7Fu; ++i) {
        const float d = fabsf(m - e4m3_device(i)), bd = fabsf(m - e4m3_device(best));
        if(d < bd || (d == bd && i % 2 == 0 && best % 2 == 1)) best = i;
    }
    return best | (x < 0.0f && best ? 0x80u : 0u);
}

// host_int's quantize_e2m1x2 for one 16-value block (one thread): values and the block's scale.
__device__ inline float quantize_e2m1x2_block(const float* x, float global, std::int8_t* values) {
    float amax = 0.0f;
    for(int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float wanted = __fdiv_rn(__fmul_rn(amax, global), 6.0f);
    const float decoded = e4m3_device(e4m3_encode_device(isfinite(wanted) ? wanted : 448.0f));
    for(int i = 0; i < 16; ++i) {
        const float q = __fdiv_rn(__fmul_rn(x[i], global), decoded);
        values[i] = decoded > 0.0f && q == q ? std::int8_t(2.0f * host_nvfp4::e2m1_device(e2m1_encode_device(q))) : 0;
    }
    return __fdiv_rn(decoded, global);
}
// host_int's quantize_int8 for one block of 16 or 32 values (one thread).
__device__ inline float quantize_int8_block(const float* x, int block, std::int8_t* values) {
    float amax = 0.0f;
    for(int i = 0; i < block; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float d = __fdiv_rn(amax, 127.0f);
    for(int i = 0; i < block; ++i) {
        float q = d > 0.0f ? rintf(__fdiv_rn(x[i], d)) : 0.0f;
        q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
        values[i] = std::int8_t(q);
    }
    return d;
}

// host_int's exp_declared, silu_declared and swiglu_declared, operation for operation (named _device beside them).
__device__ inline float exp_declared_device(float x) {
    x = x > 88.0f ? 88.0f : x < -87.0f ? -87.0f : x;
    const float n = rintf(__fmul_rn(x, 1.44269504f));
    float r = __fmaf_rn(n, -0.693145752f, x);
    r = __fmaf_rn(n, -1.42860677e-06f, r);
    float p = 1.98412698e-04f;
    p = __fmaf_rn(p, r, 1.38888889e-03f);
    p = __fmaf_rn(p, r, 8.33333333e-03f);
    p = __fmaf_rn(p, r, 4.16666667e-02f);
    p = __fmaf_rn(p, r, 1.66666667e-01f);
    p = __fmaf_rn(p, r, 0.5f);
    p = __fmaf_rn(p, r, 1.0f);
    p = __fmaf_rn(p, r, 1.0f);
    return __fmul_rn(p, __int_as_float((int(n) + 127) << 23));
}
__device__ inline float silu_declared_device(float x) {
    return __fdiv_rn(x, __fadd_rn(1.0f, exp_declared_device(-x)));
}
__device__ inline float swiglu_declared_device(float gate, float up) {
    return __fmul_rn(silu_declared_device(gate), up);
}

}  // namespace crit::techniques::host_int
