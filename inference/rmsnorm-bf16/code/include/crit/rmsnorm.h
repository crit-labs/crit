#pragma once
// crit RMSNorm interface v1. CUDA is the only backend dependency in this ABI.
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
struct Parameters { Mode mode; float epsilon; };
struct ExecutionContext { cudaStream_t stream; };
struct Workspace { void* data = nullptr; std::size_t bytes = 0; };
struct Dispatch { const char* implementation; unsigned kernel_launches; };
inline constexpr unsigned interface_version = 1;
inline constexpr std::int64_t max_width = 8192, max_rows = 49152;

// Throws before enqueue for an invalid view/domain/alias/resource declaration.
// This implementation requires exactly zero caller workspace, zero allocations.
void validate(ReadView x, ReadView weight, const ReadView* gate, WriteView out,
              Parameters parameters, Workspace workspace);
Dispatch enqueue(ReadView x, ReadView weight, const ReadView* gate, WriteView out,
                 Parameters parameters, ExecutionContext context, Workspace workspace = {});
}
