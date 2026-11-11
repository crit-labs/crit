// crit's kernel core: loads, stores and pipelines (kernels/core/README.md, "load"). Read-only and streaming global
// loads, cp.async staging (8.0 on) with a multi-stage ring, ldmatrix, shared-memory swizzles, mbarriers and the
// Tensor Memory Accelerator (cp.async.bulk, 1-D and tensor tiles, 9.0 on) with a host helper for tensor maps, and
// programmatic dependent launch. No block here changes a value: they move bytes.
//
// Mechanisms after CUTLASS (BSD-3-Clause: the swizzle functor's form, the mbarrier and TMA instruction sequences) and
// FlashInfer (Apache-2.0: its cp.async and PDL usage); the code is written from the PTX ISA.
#pragma once
#include "crit_kcore/arch.cuh"

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

namespace crit::kcore {

// ------------------------------------------------------------------------------------------- addresses

// A generic pointer into shared memory as a 32-bit shared address (what cp.async, ldmatrix and mbarriers take).
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// ------------------------------------------------------------------------------------------- global loads

// 16 bytes through the read-only path (p 16-byte aligned); zeros when !valid (nothing read).
__device__ __forceinline__ uint4 ldg16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ uint4 ldg16(const void* p, bool valid) {
    return valid ? __ldg(reinterpret_cast<const uint4*>(p)) : make_uint4(0u, 0u, 0u, 0u);
}

// 16 / 8 bytes of data read once (weights in a decode): L1::no_allocate keeps them from evicting what is reused.
__device__ __forceinline__ uint4 ld_stream16(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint2 ld_stream8(const void* p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
// A kernel's STREAM knob: the streaming load, or the read-only one.
template <int STREAM>
__device__ __forceinline__ uint4 ld16(const void* p) {
    if constexpr(STREAM != 0) return ld_stream16(p);
    else return ldg16(p);
}
template <int STREAM>
__device__ __forceinline__ uint2 ld8(const void* p) {
    if constexpr(STREAM != 0) return ld_stream8(p);
    else return *reinterpret_cast<const uint2*>(p);
}

// ------------------------------------------------------------------------------------------- cp.async (8.0 on)

// 16 bytes global -> shared, asynchronously, through L2 only (.cg); zeros when !valid (src not read).
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid = true) {
#if CRIT_KCORE_HAS_CP_ASYNC
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(valid ? 16 : 0) : "memory");
#else
    *reinterpret_cast<uint4*>(dst) = valid ? *reinterpret_cast<const uint4*>(src) : make_uint4(0u, 0u, 0u, 0u);
#endif
}
// BYTES (4, 8 or 16) global -> shared through L1 (.ca); zeros when !valid.
template <int BYTES>
__device__ __forceinline__ void cp_async_ca(void* dst, const void* src, bool valid = true) {
    static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16, "cp.async copies 4, 8 or 16 bytes");
#if CRIT_KCORE_HAS_CP_ASYNC
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_u32(dst)), "l"(src), "n"(BYTES),
                 "r"(valid ? BYTES : 0) : "memory");
#else
    unsigned char* d = static_cast<unsigned char*>(dst);
    const unsigned char* s = static_cast<const unsigned char*>(src);
    for(int i = 0; i < BYTES; ++i) d[i] = valid ? s[i] : 0;
#endif
}
// Close the group of copies issued since the last commit.
__device__ __forceinline__ void cp_async_commit() {
#if CRIT_KCORE_HAS_CP_ASYNC
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
// Wait until at most N committed groups are still in flight (N = 0: all have landed). The thread's own copies only:
// __syncthreads afterwards before reading what other threads copied.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if CRIT_KCORE_HAS_CP_ASYNC
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait_all() { cp_async_wait<0>(); }

// A STAGES-deep ring of shared-memory stages filled by cp.async: the producer issues stage i's copies, then commit();
// the consumer, before reading stage i, calls wait() (its copies and every thread's have landed: at most STAGES - 2
// newer groups still in flight, then a barrier). slot(i) is the buffer stage i lives in. The classic multistage main
// loop: prologue fills STAGES - 1 stages; each iteration waits for the oldest, issues the next, computes the oldest.
template <int STAGES>
struct CpAsyncRing {
    static_assert(STAGES >= 2, "a ring has at least two stages");
    __device__ __forceinline__ static int slot(int stage) { return stage % STAGES; }
    __device__ __forceinline__ static void commit() { cp_async_commit(); }
    __device__ __forceinline__ static void wait() {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
    }
};

// ------------------------------------------------------------------------------------------- ldmatrix (8.0 on)
// Each lane gives one 16-byte row address in shared memory: lanes 8i .. 8i + 7 the 8 rows of matrix i. Lane l then
// holds, of every matrix, the 32 bits at row l / 4, bytes 4 (l % 4) .. + 3 (the 16-bit A/B fragment layout); .trans
// loads each 8 x 8 matrix of 16-bit values transposed.

__device__ __forceinline__ void ldmatrix_x1(uint32_t& r0, const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n" : "=r"(r0) : "r"(smem_u32(row)));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(row)));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(row)));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(smem_u32(row)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                                  const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(row)));
}

// ------------------------------------------------------------------------------------------- swizzles
// Swizzle<B, M, S> (CUTLASS's form): an offset's bits [M + S, M + S + B) XORed into bits [M, M + B). With M = 4 (16-byte
// chunks) a row of 2^(M + S) bytes has its chunks permuted by the row index, so a column of chunks read by 8 lanes hits
// 8 different bank groups. swizzle_128b is TMA's CU_TENSOR_MAP_SWIZZLE_128B on byte offsets (Swizzle<3, 4, 3>),
// swizzle_64b and swizzle_32b its 64B and 32B modes.
template <int B, int M, int S>
__host__ __device__ __forceinline__ constexpr uint32_t swizzle(uint32_t offset) {
    constexpr uint32_t mask = ((1u << B) - 1u) << (M + S);
    return offset ^ ((offset & mask) >> S);
}
__host__ __device__ __forceinline__ constexpr uint32_t swizzle_128b(uint32_t byte_offset) { return swizzle<3, 4, 3>(byte_offset); }
__host__ __device__ __forceinline__ constexpr uint32_t swizzle_64b(uint32_t byte_offset) { return swizzle<2, 4, 3>(byte_offset); }
__host__ __device__ __forceinline__ constexpr uint32_t swizzle_32b(uint32_t byte_offset) { return swizzle<1, 4, 3>(byte_offset); }

// ------------------------------------------------------------------------------------------- mbarriers (9.0 on)
// A 64-bit barrier in shared memory: init(count) once by one thread (then fence_barrier_init and a __syncthreads),
// each phase completes when `count` arrivals and every expected transaction byte have arrived; waiters spin on the
// phase's parity (0 for the first phase, then 1, 0, ...).

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, int count) {
#if CRIT_KCORE_HAS_TMA
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
#endif
}
// Make barrier initializations visible to the async proxy (TMA) and the cluster.
__device__ __forceinline__ void fence_barrier_init() {
#if CRIT_KCORE_HAS_TMA
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
}
// Order this thread's generic-proxy shared-memory writes before later async-proxy (TMA store) reads.
__device__ __forceinline__ void fence_proxy_async() {
#if CRIT_KCORE_HAS_TMA
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
#if CRIT_KCORE_HAS_TMA
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
#endif
}
// Arrive and add `bytes` to the phase's expected transaction count (the TMA copies that complete it).
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
#if CRIT_KCORE_HAS_TMA
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
#endif
}
// Spin until the phase of parity `phase` has completed.
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
#if CRIT_KCORE_HAS_TMA
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                 "@p bra DONE;\n\t"
                 "bra LAB_WAIT;\n\t"
                 "DONE:\n\t}\n" ::"r"(smem_u32(bar)), "r"(phase) : "memory");
#endif
}

// ------------------------------------------------------------------------------------------- TMA (9.0 on)

// The loads' destination state space: .shared::cta (PTX ISA 8.6, CUDA 12.8 on) where the toolkit has it. The
// .shared::cluster form compiles on sm_120 to a syscall a load (__cuda_syscall_cp_async_bulk_*), which makes the
// kernel's stack size dynamic: the driver then reserves local memory for every resident thread at its first launch
// (14.5 KB a thread, 3.3 GB on an RTX 5090), held for the process. A CTA's own shared memory has the same address in
// both spaces, so the copies are the same.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
#define CRIT_KCORE_TMA_DST "shared::cta"
#else
#define CRIT_KCORE_TMA_DST "shared::cluster"
#endif

// `bytes` (a multiple of 16; both addresses 16-byte aligned) global -> shared by the copy engine, completing on `bar`
// (pair with mbarrier_arrive_expect_tx for the same bytes).
__device__ __forceinline__ void tma_load_1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
#if CRIT_KCORE_HAS_TMA
    asm volatile("cp.async.bulk." CRIT_KCORE_TMA_DST ".global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                     smem_u32(dst)),
                 "l"(src), "r"(bytes), "r"(smem_u32(bar))
                 : "memory");
#endif
}
// The box at element coordinates (x inner, y outer) of a 2-D tensor map -> shared (the map's box, swizzle and
// out-of-bounds fill), completing on `bar`. `map` is the CUtensorMap's address in kernel parameter, constant or global
// memory (pass it as a __grid_constant__ const CUtensorMap parameter).
__device__ __forceinline__ void tma_load_2d(void* dst, const void* map, int x, int y, uint64_t* bar) {
#if CRIT_KCORE_HAS_TMA
    asm volatile("cp.async.bulk.tensor.2d." CRIT_KCORE_TMA_DST ".global.tile.mbarrier::complete_tx::bytes [%0], "
                 "[%1, {%2, %3}], [%4];\n" ::"r"(smem_u32(dst)),
                 "l"(map), "r"(x), "r"(y), "r"(smem_u32(bar))
                 : "memory");
#endif
}
// Shared -> the box at (x, y) of a 2-D tensor map (fence_proxy_async first, then tma_store_commit / tma_store_wait).
__device__ __forceinline__ void tma_store_2d(const void* map, const void* src, int x, int y) {
#if CRIT_KCORE_HAS_TMA
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\n" ::"l"(map),
                 "r"(smem_u32(src)), "r"(x), "r"(y)
                 : "memory");
#endif
}
__device__ __forceinline__ void tma_store_commit() {
#if CRIT_KCORE_HAS_TMA
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
#endif
}
// Wait until the committed stores have read their shared memory (it may be reused).
__device__ __forceinline__ void tma_store_wait() {
#if CRIT_KCORE_HAS_TMA
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void tma_prefetch_map(const void* map) {
#if CRIT_KCORE_HAS_TMA
    asm volatile("prefetch.tensormap [%0];\n" ::"l"(map) : "memory");
#endif
}

// A 2-D tiled tensor map (host) over a row-major [rows][cols] matrix of `type` with rows `row_bytes` apart (a multiple
// of 16), boxes of box_rows x box_cols elements (box_cols * element size <= the swizzle span, 256 bytes at most
// unswizzled), out-of-bounds elements read as zero. The driver's cuTensorMapEncodeTiled through the runtime's entry
// point (no link against libcuda). Returns the driver's result.
inline CUresult make_tensor_map_2d(CUtensorMap* map, CUtensorMapDataType type, const void* base, uint64_t rows,
                                   uint64_t cols, uint64_t row_bytes, uint32_t box_rows, uint32_t box_cols,
                                   CUtensorMapSwizzle swizzle_mode = CU_TENSOR_MAP_SWIZZLE_NONE) {
    static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult status{};
#if CUDART_VERSION >= 12050
        if(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &status) != cudaSuccess)
            fn = nullptr;
#else
        if(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &status) != cudaSuccess) fn = nullptr;
#endif
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
    }();
    if(!encode) return CUDA_ERROR_NOT_FOUND;
    const cuuint64_t dims[2] = {cols, rows};
    const cuuint64_t strides[1] = {row_bytes};
    const cuuint32_t box[2] = {box_cols, box_rows};
    const cuuint32_t step[2] = {1, 1};
    return encode(map, type, 2, const_cast<void*>(base), dims, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  swizzle_mode, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// ------------------------------------------------------------------------------------------- programmatic launch

// Programmatic dependent launch (9.0 on; nothing below): wait for the previous kernel in the stream before reading
// what it wrote, and let the next one start its prologue.
__device__ __forceinline__ void griddep_wait() {
#if CRIT_KCORE_HAS_PDL
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void griddep_launch_dependents() {
#if CRIT_KCORE_HAS_PDL
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Whether the current device has programmatic dependent launch (compute capability 9.0 on), cached per device.
inline bool programmatic_device() {
    static int major[64] = {};
    int device = 0;
    if(cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return false;
    if(!major[device] && cudaDeviceGetAttribute(&major[device], cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
        return false;
    return major[device] >= 9;
}

// A launch on `stream`, as the programmatic dependent of the kernel before it where the device has it.
template <class... Params, class... Args>
cudaError_t launch_programmatic(void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                                Args... args) {
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    if(programmatic_device()) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
    }
    return cudaLaunchKernelEx(&config, kernel, args...);
}

}  // namespace crit::kcore
