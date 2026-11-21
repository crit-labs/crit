// crit's kernel core: epilogues (kernels/core/README.md, "epilogue"). What a finished sum goes through before it is
// stored: crit's declared exp and SwiGLU (bit-exact functions every backend computes alike), the plain activations,
// and stores of a value or of an MMA accumulator fragment in the output's type.
#pragma once
#include "crit_kcore/arch.cuh"
#include "crit_kcore/formats.cuh"

#include <cstddef>
#include <cstdint>

namespace crit::kcore {

// ------------------------------------------------------------------------------------------- declared functions

// crit's declared exp: x clamped to [-87, 88], n = rint(x log2 e), a two-constant Cody-Waite reduction and a degree-8
// polynomial, every step rounded (__fmaf_rn / __fmul_rn: never contracted or approximated), times 2^n. The same bits on
// every device and on crit's CPU tiers.
__device__ __forceinline__ float exp_declared(float x) {
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

// crit's declared SwiGLU: (gate / (1 + exp_declared(-gate))) * up, each operation rounded.
__device__ __forceinline__ float swiglu_declared(float gate, float up) {
    return __fmul_rn(__fdiv_rn(gate, __fadd_rn(1.0f, exp_declared(-gate))), up);
}

// gelu(gate) * up with GELU's tanh form, 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))): the CUDA math library's
// tanhf (within its ulp bound: one function of its argument on a device, not declared bits), every other operation
// rounded, so a row's bits are its own whatever shares its launch.
__device__ __forceinline__ float gelu_tanh_mul(float gate, float up) {
    const float cube = __fmul_rn(__fmul_rn(gate, gate), gate);
    const float inner = __fmul_rn(0.79788456f, __fmaf_rn(0.044715f, cube, gate));
    return __fmul_rn(__fmul_rn(__fmul_rn(0.5f, gate), __fadd_rn(1.0f, tanhf(inner))), up);
}

// SwiGLU whose linear gate factor and up projection are each capped by a tanh: cap_g tanh(gate / cap_g) sigmoid(gate)
// times cap_u tanh(up / cap_u), the sigmoid of the raw gate. The CUDA math library's tanhf and expf (within their ulp
// bounds), every other operation rounded, so a row's bits are its own whatever shares its launch.
__device__ __forceinline__ float softcapped_swiglu(float gate, float up, float gate_cap, float up_cap) {
    const float capped = __fmul_rn(gate_cap, tanhf(__fdiv_rn(gate, gate_cap)));
    const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gate)));
    return __fmul_rn(__fmul_rn(capped, sig), __fmul_rn(up_cap, tanhf(__fdiv_rn(up, up_cap))));
}

// ------------------------------------------------------------------------------------------- activations
// The CUDA math library's functions (expf: within its ulp bound, not declared bits).

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// ------------------------------------------------------------------------------------------- stores

// The output types an epilogue stores in, by a runtime kind.
enum class Out : int { f32 = 0, bf16 = 1, f16 = 2 };

template <typename T>
__device__ __forceinline__ void store(T* out, size_t at, float v) { out[at] = from_float<T>(v); }

__device__ __forceinline__ void store(void* out, Out kind, size_t at, float v) {
    if(kind == Out::f32) static_cast<float*>(out)[at] = v;
    else if(kind == Out::bf16) static_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16_rn(v);
    else static_cast<__half*>(out)[at] = __float2half_rn(v);
}

// An m16 x n8 accumulator fragment (c[i] at row r0 + gid + 8 (i >> 1), column c0 + 2 tig + (i & 1), mma.cuh) into a
// row-major T matrix with rows `ld` elements apart, rows >= rows and columns >= cols skipped; adjacent column pairs of
// a 16-bit T go out as one 32-bit store when both are inside (c0 and ld even).
template <typename T>
__device__ __forceinline__ void store_c_fragment(T* out, size_t ld, int r0, int c0, int rows, int cols, const float* c,
                                                 int lane) {
    const int r = r0 + (lane >> 2), col = c0 + 2 * (lane & 3);
#pragma unroll
    for(int h = 0; h < 2; ++h) {
        const int row = r + 8 * h;
        if(row >= rows) continue;
        T* at = out + (size_t)row * ld + col;
        if constexpr(sizeof(T) == 2) {
            if(col + 1 < cols) {
                T pair[2] = {from_float<T>(c[2 * h]), from_float<T>(c[2 * h + 1])};
                *reinterpret_cast<uint32_t*>(at) = *reinterpret_cast<const uint32_t*>(pair);
                continue;
            }
        }
        if(col < cols) at[0] = from_float<T>(c[2 * h]);
        if(col + 1 < cols) at[1] = from_float<T>(c[2 * h + 1]);
    }
}

}  // namespace crit::kcore
