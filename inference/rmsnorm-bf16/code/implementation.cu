// crit-native row normalization. Caller owns device storage and CUDA stream.
#include "crit/rmsnorm.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace crit::rmsnorm {
namespace {
constexpr int threads = 256;

__global__ void normalize(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                          const __nv_bfloat16* gate, __nv_bfloat16* out,
                          std::int64_t width, float epsilon, Mode mode) {
    __shared__ float reduction[threads];
    const auto offset = std::int64_t(blockIdx.x) * width;
    float squares = 0.0f;
    for (std::int64_t column = threadIdx.x; column < width; column += threads) {
        const float value = __bfloat162float(x[offset + column]);
        squares += value * value;
    }
    reduction[threadIdx.x] = squares;
    __syncthreads();
    for (int step = threads / 2; step; step /= 2) {
        if (threadIdx.x < step) reduction[threadIdx.x] += reduction[threadIdx.x + step];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduction[0] / static_cast<float>(width) + epsilon);
    for (std::int64_t column = threadIdx.x; column < width; column += threads) {
        float gain = __bfloat162float(weight[column]);
        if (mode == Mode::unit_offset) gain += 1.0f;
        float value = __bfloat162float(x[offset + column]) * inverse * gain;
        if (mode == Mode::gated) {
            const float g = __bfloat162float(gate[offset + column]);
            value *= g / (1.0f + expf(-g));
        }
        out[offset + column] = __float2bfloat16_rn(value);
    }
}
}

Dispatch enqueue(ReadView x, ReadView weight, const ReadView* gate, WriteView out,
                 Parameters parameters, ExecutionContext context, Workspace workspace) {
    validate(x, weight, gate, out, parameters, workspace);
    normalize<<<static_cast<unsigned>(x.rows), threads, 0, context.stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        gate ? static_cast<const __nv_bfloat16*>(gate->data) : nullptr,
        static_cast<__nv_bfloat16*>(out.data), x.width, parameters.epsilon, parameters.mode);
    const auto error = cudaPeekAtLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return {"crit-rmsnorm-fp32", 1};
}
}
