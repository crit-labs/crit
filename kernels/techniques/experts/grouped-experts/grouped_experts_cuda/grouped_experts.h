// Routed SwiGLU experts over a slot store of expert bundles, for decode rows and prompt chunks alike: crit's kernel
// grouped-experts, which crit's grouped-experts plugin (inference/grouped-experts) loads, built for the engine's device with
// the engine's tuning (crit_tuning.h: warps, row tiles, weight blocks in flight and staged chunk per item width,
// format, pass, shape and token count) or the default knobs.
//
// An expert is one contiguous bundle (crit's experts:bundles layout): its gate/up matrix [2 * inter][hidden] (gate rows
// first) and down matrix [hidden][inter] as the checkpoint stores them (row major), their scales and per-row factors.
// A call's routes (ids [tokens][top]) are grouped by expert on the device into items of `width` routes (8 for decode
// rows, 32 or 64 for prompt chunks), each item naming the slot of its expert's bundle in the store (slots [experts]: >= 0 a
// slot, -2 an expert another tier computes, -1 an expert not routed; a route to a negative slot is skipped).
//
// The arithmetic is one function of a route and its expert, whatever the item, the chunk or the rows beside it, so
// decode rows and prompt chunks give the same bits and a token's result never depends on its neighbours:
//   - weights decode exactly to BF16 (bf16; E4M3 codes; E2M1 x E4M3 block scale; INT4 x E4M3; E2M1 x 2^(e - 127),
//     the E8M0 byte 0 being 2^-127 and 255 NaN); fp8 block scales one grid for gate's rows, one for up's; int4_zp's
//     (code - zero) x FP16 scale, which BF16 does not hold, as two BF16 parts (hi + lo, exact) whose MMAs run in turn;
//   - activations in BF16, or (nvfp4 with `a4`) quantized per route by crit's declared NVFP4 quantizer at the expert's
//     input scale, the operand e2m1 x decoded scale (exact in BF16);
//   - mma.sync m16n8k16 BF16 -> FP32, k in 64-element blocks in ascending order (lane quad t4 holds the 16 values
//     k0 + 16 t4 .. + 15 of a block, the same permutation in both operands), one accumulator chain per output; fp8 128 x
//     128 block scales multiply each 128-k block's sum into the running one (fmaf);
//   - per-row factors (1 / global scale, channel scale, 1 / input scale) multiply the sums; crit's declared SwiGLU
//     (IEEE operations only); the intermediate in BF16 or, with a4, quantized as the inputs at the down's input scale;
//   - the down rows' FP32 outputs unweighted per route, then each token's routes summed in route order with
//     fmaf(weight, output, sum).
// No atomics, no split of a reduction by the row count. A baseline to replace: measured against providers in README.md.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace crit::techniques::grouped_experts {

enum class Format : int { bf16 = 0, fp8_channel = 1, fp8_block = 2, nvfp4 = 3, int4 = 4, mxfp4 = 5, int4_zp = 6 };

// Byte offsets of a bundle's components (crit's experts:bundles layout); -1 for a component the format lacks. int4_zp
// (INT4 with a zero point a group: AWQ, GPTQ in crit's rows, decode-gemv's repack_awq / repack_gptq): codes [rows][K /
// 2] unsigned (element 2i in the low nibble of byte i), *_scale FP16 [rows][K / group], *_inverse the zero points
// uint8 [rows][K / group] (0..16), group a multiple of 16 dividing hidden and inter.
struct Layout {
    long long bytes = 0;                                   // one bundle, a multiple of 16
    long long gate_up = -1, gate_up_scale = -1, gate_up_inverse = -1;
    long long down = -1, down_scale = -1, down_inverse = -1;
    long long input = -1;                                  // float[4]: gate/up and down input scales, their inverses
    int hidden = 0, inter = 0;
    Format format = Format::bf16;
    int group = 0;                                         // int4_zp: inputs a scale and zero point cover
};

// The items of a call: item u is the slot of one expert and `width` of its routes (route = token * top + j, in
// increasing route order, -1 padded); `count` (device) is how many items there are, at most `capacity`.
struct Items {
    int* slot = nullptr;      // [capacity]
    int* routes = nullptr;    // [capacity][width]
    int* count = nullptr;     // [1]
    int width = 8, capacity = 0;   // width 8, 32 or 64
};

// The largest call the one-block plan takes; larger calls group their routes from a stable sort (enqueue_items).
constexpr int kSmallPlanRoutes = 1024;
// The experts a call may name, in either plan: a call's items (at most 65535, the launches' second grid dimension)
// hold at least one per routed expert.
constexpr int kMaxExperts = 65535;

// Items from ids (int64 or int32, [routes]) for routes <= kSmallPlanRoutes: one block, no host synchronization.
cudaError_t enqueue_plan(const long long* ids64, const int* ids32, int routes, int experts, const int* slots,
                         const Items& items, cudaStream_t stream);
// Items from routes sorted by expert (order, stable), each expert's count and first position, and the inclusive
// running sum of its items (ceil(count / width)); invalid routes sort past every expert.
cudaError_t enqueue_items(const long long* order, const int* counts, const int* starts, const int* item_end,
                          int experts, const int* slots, const Items& items, cudaStream_t stream);
// h [routes][inter] BF16: SwiGLU(gate, up) of every planned route (x [tokens][hidden] BF16 rows, route r reads token
// r / top). a4: x is the routes' W4A4 operands enqueue_a4_inputs made ([routes][hidden], top 1), h quantized at the
// expert's down input scale. tokens: the call's token count, which picks the tuned configuration (0: the defaults).
cudaError_t enqueue_gate_up(const __nv_bfloat16* x, int top, const std::uint8_t* store, const Layout& layout, bool a4,
                            const Items& items, __nv_bfloat16* h, cudaStream_t stream, int tokens = 0);
// The W4A4 gate/up operands of every planned route, out [routes][hidden] BF16: x quantized once per route at its
// expert's input scale by crit's declared quantizer (operand e2m1 x decoded scale, exact in BF16).
cudaError_t enqueue_a4_inputs(const __nv_bfloat16* x, int top, const std::uint8_t* store, const Layout& layout,
                              const Items& items, __nv_bfloat16* out, cudaStream_t stream);
// y [routes][hidden] FP32: the down rows of every planned route, unweighted.
cudaError_t enqueue_down(const __nv_bfloat16* h, const std::uint8_t* store, const Layout& layout, bool a4,
                         const Items& items, float* y, cudaStream_t stream, int tokens = 0);
// out [tokens][hidden] FP32: sum = fmaf(weights[t][j], y[t * top + j], sum) over j ascending for the routes with
// 0 <= id < experts and, when skip_negative and slots are given, slots[id] >= 0.
cudaError_t enqueue_combine(const float* y, const long long* ids64, const int* ids32, const float* weights,
                            const int* slots, int experts, int tokens, int top, int hidden, bool skip_negative,
                            float* out, cudaStream_t stream);
// The declared arithmetic (crit's integer block order, host-expert-arithmetic's host_int.h, which crit's CPU expert
// tier computes, so a route's bits are the same on either processor), nvfp4 W4A4 and int4 W4A8 bundles:
// - inputs: each planned route's x row quantized at its expert's gate/up input scale (nvfp4: 2 x E2M1 per 16, scales
//   decoded / global) or as INT8 per 32 (int4): values int8 [routes][hidden], scales [routes][hidden / block];
// - rows: out [routes][rows] FP32 = the declared rows of the route's expert (gate_up: [2 * inter], else [hidden]);
// - intermediate: SwiGLU (declared) of the gate/up rows, quantized at the down input scale (nvfp4) or INT8 per 32.
cudaError_t enqueue_declared_inputs(const __nv_bfloat16* x, int top, const std::uint8_t* store, const Layout& layout,
                                    const Items& items, std::int8_t* values, float* scales, cudaStream_t stream);
cudaError_t enqueue_declared_rows(const std::int8_t* values, const float* scales, const std::uint8_t* store,
                                  const Layout& layout, bool gate_up, const Items& items, float* out,
                                  cudaStream_t stream);
cudaError_t enqueue_declared_intermediate(const float* gate_up, const std::uint8_t* store, const Layout& layout,
                                          const Items& items, std::int8_t* values, float* scales, cudaStream_t stream);
// The declared NVFP4 activation quantizer (crit's host_int quantize_e2m1x2): x [rows][cols] FP32 -> values int8
// (2 * e2m1) and scales FP32 [rows][cols / 16] (decoded / global).
cudaError_t enqueue_quantize_e2m1x2(const float* x, int rows, int cols, float global, std::int8_t* values,
                                    float* scales, cudaStream_t stream);

}  // namespace crit::techniques::grouped_experts
