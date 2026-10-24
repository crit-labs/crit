#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

// Programmatic dependent launch (compute capability 9.0 and later): a kernel launched with programmatic stream
// serialization may start while the previous kernel in its stream is still running. Its blocks can do everything that
// does not read the previous kernel's output (address math, loading weights and constants into registers or shared
// memory) and then call crit::core::wait_for_prior_grid() before the first such read; the previous kernel may call
// crit::core::launch_dependents() once its last block no longer needs the SMs to itself. The gap between two kernels
// of a decode step, their launch and ramp-up, then overlaps the previous kernel's tail. Stream capture records the
// launches as programmatic graph edges, so captured decode graphs keep the overlap.
//
// Whether a device supports it is a device query, never a constant: launch() falls back to an ordinary launch where it
// does not, and the device functions compile to nothing below sm_90.
namespace crit::core {

// Compute capability 9.0 or later on `device`; false when the query fails. Cached per device after the first call.
bool programmatic_launch_supported(int device) noexcept;
// The same for the device that `stream` belongs to (the current device for the legacy or per-thread default stream).
bool programmatic_launch_supported(cudaStream_t stream) noexcept;

struct LaunchShape {
    dim3 grid, block;
    std::size_t shared = 0;   // dynamic shared memory bytes
};

// kernel<<<shape.grid, shape.block, shape.shared, stream>>>(args...), with programmatic stream serialization when
// `programmatic` is set and the stream's device supports it. Returns the launch's error.
template <class... Params, class... Args>
cudaError_t launch(void (*kernel)(Params...), const LaunchShape& shape, cudaStream_t stream, bool programmatic, Args&&... args) {
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config{};
    config.gridDim = shape.grid;
    config.blockDim = shape.block;
    config.dynamicSmemBytes = shape.shared;
    config.stream = stream;
    if(programmatic && programmatic_launch_supported(stream)) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
    }
    return cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...);
}

#if defined(__CUDACC__)
// Waits until every grid this one depends on through programmatic serialization has completed and its memory
// operations are visible. A no-op for ordinary launches and below sm_90.
__device__ __forceinline__ void wait_for_prior_grid() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
// Lets the next programmatically serialized grid in the stream start before this one ends (its wait_for_prior_grid
// still waits for this grid's completion). Call it after the last work that needs this grid's SMs to itself.
__device__ __forceinline__ void launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
#endif

}  // namespace crit::core
