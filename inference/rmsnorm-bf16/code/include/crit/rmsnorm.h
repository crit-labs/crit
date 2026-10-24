#pragma once
// crit RMSNorm interface v2. CUDA is the only backend dependency in this ABI.
// Views borrow device storage; enqueue never implies completion.
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace crit::rmsnorm {
enum class Representation : std::uint32_t { bf16 = 1 };
enum class Mode : std::uint32_t { plain = 0, unit_offset = 1, gated = 2 };
struct ReadView {
    const void* data;
    std::int64_t width, rows, row_stride_bytes;
    Representation representation = Representation::bf16;
};
struct WriteView {
    void* data;
    std::int64_t width, rows, row_stride_bytes;
    Representation representation = Representation::bf16;
};
// single rounds x * r * gain (and gate) once from FP32 to BF16 (Gemma style).
// normalize_first rounds x * r to BF16, then the product with the gain to BF16
// and, in gated mode, the gated product again (HF Llama-family style).
enum class Rounding : std::uint32_t { single = 0, normalize_first = 1 };
struct Parameters { Mode mode; float epsilon; Rounding rounding = Rounding::single; };
struct ExecutionContext { cudaStream_t stream; };
struct Workspace { void* data = nullptr; std::size_t bytes = 0; };
struct Dispatch { const char* implementation; unsigned kernel_launches; };
inline constexpr unsigned interface_version = 2;
// The kernel strides a fixed 256-thread block over any row, so width needs no
// shared memory. Widths stay exact in the FP32 mean; one row per grid-x block.
inline constexpr std::int64_t max_width = std::int64_t(1) << 24, max_rows = 0x7fffffff;

// Throws before enqueue for an invalid view/domain/alias/resource declaration.
// This implementation requires exactly zero caller workspace, zero allocations.
void validate(ReadView x, ReadView weight, const ReadView* gate, WriteView out,
              Parameters parameters, Workspace workspace);
Dispatch enqueue(ReadView x, ReadView weight, const ReadView* gate, WriteView out,
                 Parameters parameters, ExecutionContext context, Workspace workspace = {});
}
