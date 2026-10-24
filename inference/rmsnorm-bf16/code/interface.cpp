#include "crit/rmsnorm.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace crit::rmsnorm {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
std::uintptr_t end(const void* data, std::uint64_t bytes) {
    const auto start = reinterpret_cast<std::uintptr_t>(data);
    require(bytes <= std::numeric_limits<std::uintptr_t>::max() - start, "crit rmsnorm: address range overflow");
    return start + bytes;
}
bool overlap(const void* a, std::uint64_t na, const void* b, std::uint64_t nb) {
    return reinterpret_cast<std::uintptr_t>(a) < end(b, nb) && reinterpret_cast<std::uintptr_t>(b) < end(a, na);
}
void view(ReadView v) {
    require(v.width >= 1 && v.width <= max_width && v.rows >= 1 && v.rows <= max_rows,
            "crit rmsnorm: width must be 1..2^24 and rows 1..2^31-1");
    require(v.representation == Representation::bf16, "crit rmsnorm: only BF16 views admitted");
    require(v.data && reinterpret_cast<std::uintptr_t>(v.data) % 2 == 0, "crit rmsnorm: non-null 2-byte aligned device view required");
    require(v.row_stride_bytes == 2 * v.width, "crit rmsnorm: contiguous rows required; materialization is caller policy");
    end(v.data, 2 * v.width * v.rows);
}
}
void validate(ReadView x, ReadView weight, const ReadView* gate, WriteView out,
              Parameters p, Workspace workspace) {
    view(x); view(weight); view({out.data,out.width,out.rows,out.row_stride_bytes,out.representation});
    require(out.width == x.width && out.rows == x.rows, "crit rmsnorm: output shape mismatch");
    require(weight.width == x.width && weight.rows == 1, "crit rmsnorm: gain must contain one feature row");
    require(std::isfinite(p.epsilon) && p.epsilon > 0, "crit rmsnorm: epsilon must be positive finite FP32");
    require(p.mode == Mode::plain || p.mode == Mode::unit_offset || p.mode == Mode::gated, "crit rmsnorm: unsupported mode");
    require(p.rounding == Rounding::single || p.rounding == Rounding::normalize_first, "crit rmsnorm: unsupported rounding");
    require((p.mode == Mode::gated) == (gate != nullptr), "crit rmsnorm: gate required exactly in gated mode");
    require(workspace.data == nullptr && workspace.bytes == 0, "crit rmsnorm: declared workspace must be zero");
    const auto bytes = std::uint64_t(2 * x.width * x.rows);
    require(!overlap(out.data,bytes,weight.data,2 * weight.width), "crit rmsnorm: output must not overlap gain");
    require(out.data == x.data || !overlap(out.data,bytes,x.data,bytes), "crit rmsnorm: partial input/output alias forbidden");
    if (gate) {
        view(*gate);
        require(gate->width == x.width && gate->rows == x.rows, "crit rmsnorm: gate shape mismatch");
        require(out.data == gate->data || !overlap(out.data,bytes,gate->data,bytes), "crit rmsnorm: partial gate/output alias forbidden");
    }
}
}
