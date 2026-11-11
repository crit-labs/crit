// crit's kernel core: which instructions the architecture being compiled has (kernels/core/README.md, "arch").
//
// Every macro is 0 on the host pass and on architectures without the instruction, 1 where it exists, so a block
// guards its instruction with `#if CRIT_KCORE_HAS_...` and a kernel picks a path with `if constexpr`. nvcc's names:
// plain targets (sm_80, sm_86, sm_89, sm_90, sm_100, sm_120) define __CUDA_ARCH__; the architecture-specific ones
// (sm_90a, sm_100a, sm_120a, ...) also define __CUDA_ARCH_SPECIFIC__ (and __CUDA_ARCH_FEAT_SM<N>_ALL), the family
// ones (sm_100f, sm_120f) __CUDA_ARCH_FAMILY_SPECIFIC__.
#pragma once

#if defined(__CUDA_ARCH__)
#define CRIT_KCORE_ARCH __CUDA_ARCH__
#else
#define CRIT_KCORE_ARCH 0
#endif

// cp.async (global to shared, 4/8/16 bytes, zero fill) and ldmatrix: compute capability 8.0 on.
#define CRIT_KCORE_HAS_CP_ASYNC (CRIT_KCORE_ARCH >= 800)

// mma.sync on E4M3/E5M2 (m16n8k32) and the cvt.e4m3x2 conversions: 8.9 on.
#define CRIT_KCORE_HAS_FP8_MMA (CRIT_KCORE_ARCH >= 890)

// Thread block clusters, mbarrier transaction counts, cp.async.bulk (TMA, bulk and tensor), programmatic dependent
// launch (griddepcontrol): 9.0 on.
#define CRIT_KCORE_HAS_TMA (CRIT_KCORE_ARCH >= 900)
#define CRIT_KCORE_HAS_PDL (CRIT_KCORE_ARCH >= 900)

// wgmma (warpgroup MMA from shared memory descriptors): sm_90a only (not 10.x or 12.x).
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define CRIT_KCORE_HAS_WGMMA 1
#else
#define CRIT_KCORE_HAS_WGMMA 0
#endif

// tcgen05 (tensor memory, 5th generation MMA): the sm_100 family's specific targets (100a/f, 103a/f, 110a/f).
#if (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) || defined(__CUDA_ARCH_FEAT_SM110_ALL)) || \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1200)
#define CRIT_KCORE_HAS_TCGEN05 1
#else
#define CRIT_KCORE_HAS_TCGEN05 0
#endif

// cvt.rn.f16x2.e2m1x2 (E2M1 pairs to FP16 in one instruction): the architecture-specific and family targets from
// sm_100 on (100a/f, 103a, 110a/f, 120a/f, 121a), not plain sm_100 or sm_120 nor before.
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
#define CRIT_KCORE_HAS_E2M1_CVT 1
#else
#define CRIT_KCORE_HAS_E2M1_CVT 0
#endif

// Block-scaled mma.sync (kind::mxf8f6f4, kind::mxf4, kind::mxf4nvf4 with UE8M0/UE4M3 scales): the sm_120 family's
// specific targets (120a, 121a).
#if defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200 || __CUDA_ARCH_SPECIFIC__ == 1210)
#define CRIT_KCORE_HAS_BLOCK_SCALED_MMA 1
#else
#define CRIT_KCORE_HAS_BLOCK_SCALED_MMA 0
#endif
