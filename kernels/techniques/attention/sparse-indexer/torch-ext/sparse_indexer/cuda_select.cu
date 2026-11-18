// sparse-indexer's select for a few tokens (kernel.json "cuda"): the radix select of _kernels.topk_kernel with a
// slice's scores held in shared memory. Compiled by the build with its crit_tuning.h and loaded by the package
// through ctypes (_crit_cuda.library); launched on the caller's stream, so CUDA graphs capture it.
//
// A program of topk_kernel reads its scores from global memory five times (four digit passes and the write), which
// for a decode call is most of the selection's time: one token's 50,000 scores are 13 programs, each a chain of loads.
// Here a CTA copies its slice once (cp.async, 128 bytes a warp's request), turns the scores into their
// order-preserving keys in place and runs the same five passes in shared memory: a 256-bin histogram a digit by
// shared-memory atomics, the digit's choice by a suffix sum over the bins (a lane a bin), then every thread's
// contiguous run of the slice counted and written at its offset. The picks are topk_kernel's by definition (the k
// largest valid scores, ties to the lower position, a NaN below every number, positions ascending, then -1): no
// arithmetic, so no order to keep.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "crit_tuning.h"
#include "crit_kcore/device.cuh"
#include "crit_kcore/load.cuh"

namespace {

namespace kc = crit::kcore;

constexpr int kThreads = 256;   // a lane a bin of a digit's histogram
constexpr int kWarps = kThreads / 32;
constexpr int kStatic = 8192;   // room left for the kernel's static shared memory (its histograms) beside a slice
constexpr int kBins = 264;      // a digit's 256 bins and one for the keys a pass does not count
constexpr int kRun = 128;       // the most elements of a slice a thread takes in the write (kThreads x kRun a slice)
constexpr int kPad = 4;         // elements before an array of the slice: its source's place within 16 bytes

struct Select {
    const float* s;       // scores [tokens][n] (rows stride_s apart), or the candidates' scores
    const int* pos;       // candidates: their positions [tokens][n], -1 for none (nullptr: the scores' own indices)
    const int* ends;      // each token's valid scores (nullptr: n)
    int* out;             // [tokens][parts][k] positions, ascending, then -1
    float* sout;          // the chosen scores beside them (nullptr: not written)
    int n, k, slice;
    long long stride_s, stride_o;
};

// FP32 scores as uint32 keys of the same order, a NaN as 0 (below every number, -inf included): _kernels._order_key.
__device__ __forceinline__ uint32_t order_key(uint32_t bits) {
    const uint32_t key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return (bits & 0x7fffffffu) > 0x7f800000u ? 0u : key;
}

// CTA (token t, slice p): the k best of the token's valid scores in positions p x slice .. (p + 1) x slice - 1, at
// out[t][p x k ..].
__global__ void __launch_bounds__(kThreads) select_kernel(Select a) {
    extern __shared__ __align__(16) uint32_t slice_smem[];   // the slice's keys, then (candidates) their positions
    __shared__ int hist[4][kBins];   // a digit pass's histogram each (bin 256: the keys that do not count)
    __shared__ int warp_sum[kWarps][2];
    const int t = blockIdx.x, p = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int end = a.n;
    if(a.ends) end = max(min(a.ends[t], a.n), 0);
    const int lo = p * a.slice, hi = min(end, lo + a.slice), count = max(hi - lo, 0);
    const float* row = a.s + static_cast<long long>(t) * a.stride_s + lo;
    const int* prow = a.pos ? a.pos + static_cast<long long>(t) * a.stride_s + lo : nullptr;
    int* orow = a.out + static_cast<long long>(t) * a.stride_o + static_cast<long long>(p) * a.k;
    float* srow = a.sout ? a.sout + static_cast<long long>(t) * a.stride_o + static_cast<long long>(p) * a.k : nullptr;
    // The slice into shared memory, once. An element sits at its source's place within 16 bytes (kPad elements of
    // room before each array), so where the device has bulk copies the 16-byte aligned middle of the slice is one
    // copy an array (a multiprocessor takes small copies at about 8 GB/s, a bulk copy several times faster) and only
    // the few elements before and after it go one by one; elsewhere every element does (a warp's request 128 bytes).
    auto phase = [](const void* at) { return static_cast<int>((reinterpret_cast<std::uintptr_t>(at) >> 2) & 3); };
    uint32_t* keys = slice_smem + phase(row);
    int* where = reinterpret_cast<int*>(slice_smem + a.slice + kPad) + (prow ? phase(prow) : 0);
    int first = 0, last = 0;   // the bulk copy's elements [first, last) of each array (the same phase: checked)
#if CRIT_KCORE_HAS_TMA
    __shared__ uint64_t landed;
    if(count >= 64 && (!prow || phase(prow) == phase(row))) {
        first = (4 - phase(row)) & 3;
        last = first + (count - first) / 4 * 4;
        if(tid == 0) {
            kc::mbarrier_init(&landed, 1);
            kc::fence_barrier_init();
        }
        __syncthreads();
        if(tid == 0) {
            const uint32_t bytes = static_cast<uint32_t>(last - first) * 4u;
            kc::mbarrier_arrive_expect_tx(&landed, prow ? 2u * bytes : bytes);
            kc::tma_load_1d(keys + first, row + first, bytes, &landed);
            if(prow) kc::tma_load_1d(where + first, prow + first, bytes, &landed);
        }
    }
#endif
    for(int j = tid; j < count; j += kThreads) {
        if(j >= first && j < last) continue;
        kc::cp_async_ca<4>(keys + j, row + j);
        if(prow) kc::cp_async_ca<4>(where + j, prow + j);
    }
    kc::cp_async_commit();
    kc::cp_async_wait_all();
#if CRIT_KCORE_HAS_TMA
    if(last > first) kc::mbarrier_wait(&landed, 0u);
#endif
    __syncthreads();
    for(int j = tid; j < count; j += kThreads) keys[j] = order_key(keys[j]);
#pragma unroll
    for(int d = 0; d < 4; ++d)
        for(int b = tid; b < kBins; b += kThreads) hist[d][b] = 0;
    __syncthreads();
    auto valid = [&](int j) { return !prow || where[j] >= 0; };

    // Four 8-bit passes, the highest digit first: the k-th largest key's digit among the keys matching the prefix.
    // A pass costs one barrier (after its histogram): every warp then reads the 256 bins itself (a lane 8 of them),
    // so each finds the digit without a word from the others. A lane's suffix sum over the warp's 8-bin groups
    // locates the group that holds the k-th key, that group's lane finds the bin, and a shuffle tells the warp.
    int need = a.k;
    uint32_t prefix = 0u;
#pragma unroll
    for(int d = 0; d < 4; ++d) {
        const int shift = 24 - 8 * d;
        int* bins = hist[d];
        for(int j = tid; j < count; j += kThreads) {   // (no branch on the data: a key that does not count adds to bin 256)
            const uint32_t key = keys[j];
            const bool counts = valid(j) && (d == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8)));
            atomicAdd(&bins[counts ? (key >> shift) & 0xFFu : 256u], 1);
        }
        __syncthreads();
        int group[8], ge = 0;   // the lane's bins 8 lane .. 8 lane + 7, and the keys of those digits or above
#pragma unroll
        for(int i = 0; i < 8; ++i) {
            group[i] = bins[8 * lane + i];
            ge += group[i];
        }
#pragma unroll
        for(int off = 1; off < 32; off <<= 1) {
            const int v = __shfl_down_sync(0xffffffffu, ge, off);
            if(lane + off < 32) ge += v;
        }
        if(d == 0) need = min(need, __shfl_sync(0xffffffffu, ge, 0));   // every valid key where there are k or fewer
        // The highest group with `need` keys at or above it (the last one where need is 0), then its highest such bin
        // and what is still needed among that bin's keys.
        const int holder = 31 - __clz(__ballot_sync(0xffffffffu, ge >= need) | 1u);
        int above = ge, digit = 8 * lane, left = need;
#pragma unroll
        for(int i = 0; i < 8; ++i) above -= group[i];   // the keys above the lane's group
#pragma unroll
        for(int i = 7; i >= 0; --i) {
            if(above + group[i] >= left && digit == 8 * lane) {   // (the first bin from the top that reaches it)
                digit = 8 * lane + i + 256;
                left -= above;
            }
            above += group[i];
        }
        digit = __shfl_sync(0xffffffffu, digit, holder) & 0xFF;
        need = __shfl_sync(0xffffffffu, left, holder);
        prefix |= static_cast<uint32_t>(digit) << shift;
    }

    // The threshold key `prefix`: every key above it and `need` of its ties (lowest positions first), ascending. A
    // thread takes a contiguous run of the slice, as masks of its keys above the threshold and of its ties (32
    // elements a word, no branch on the data); the runs before it give its offset and its ties' ranks, and it then
    // writes only its set bits.
    const int run = (count + kThreads - 1) / kThreads;
    const int j0 = min(count, tid * run), j1 = min(count, j0 + run);
    uint32_t over[kRun / 32], equal[kRun / 32];
    int above = 0, ties = 0;
#pragma unroll
    for(int c = 0; c < kRun / 32; ++c) {
        over[c] = equal[c] = 0u;
        if(32 * c < run) {
#pragma unroll 8
            for(int i = 0; i < 32; ++i) {
                const int j = j0 + 32 * c + i;
                const bool live = j < j1 && valid(min(j, count - 1));
                const uint32_t key = keys[min(j, count - 1)];
                over[c] |= static_cast<uint32_t>(live && key > prefix) << i;
                equal[c] |= static_cast<uint32_t>(live && key == prefix) << i;
            }
        }
        above += __popc(over[c]);
        ties += __popc(equal[c]);
    }
    int sum_above = above, sum_ties = ties;   // inclusive prefix sums over the threads: the warp's, then the warps below
#pragma unroll
    for(int off = 1; off < 32; off <<= 1) {
        const int va = __shfl_up_sync(0xffffffffu, sum_above, off), vt = __shfl_up_sync(0xffffffffu, sum_ties, off);
        if(lane >= off) {
            sum_above += va;
            sum_ties += vt;
        }
    }
    if(lane == 31) {
        warp_sum[warp][0] = sum_above;
        warp_sum[warp][1] = sum_ties;
    }
    __syncthreads();
    int all_above = 0, all_ties = 0;
#pragma unroll
    for(int w = 0; w < kWarps; ++w) {
        if(w < warp) {
            sum_above += warp_sum[w][0];
            sum_ties += warp_sum[w][1];
        }
        all_above += warp_sum[w][0];
        all_ties += warp_sum[w][1];
    }
    const int before = sum_ties - ties;                       // the ties before this thread's run
    int allowed = min(max(need - before, 0), ties);           // of its own ties, the first `allowed` are kept
    int at = sum_above - above + min(before, need);           // the keys written before it
#pragma unroll
    for(int c = 0; c < kRun / 32; ++c) {
        uint32_t bits = over[c] | equal[c];
        while(bits) {
            const int i = __ffs(bits) - 1;
            bits &= bits - 1u;
            const bool tie = (equal[c] >> i) & 1u;
            if(tie && allowed <= 0) continue;
            allowed -= tie;
            const int j = j0 + 32 * c + i;
            orow[at] = prow ? where[j] : lo + j;
            if(srow) {   // the score back from its key (a NaN stays a NaN: the candidates' pass keys it as 0 again)
                const uint32_t key = keys[j];
                reinterpret_cast<uint32_t*>(srow)[at] = (key & 0x80000000u) ? key & 0x7fffffffu : ~key;
            }
            ++at;
        }
    }
    const int written = all_above + min(all_ties, need);
    for(int j = written + tid; j < a.k; j += kThreads) orow[j] = -1;
}

}  // namespace

// The select of `tokens` tokens in `parts` slices of `slice` scores each (parts x slice >= n). 0, or 2 (operands), 3
// (a slice past the shared memory a block gets), 5 (the launch).
extern "C" int crit_si_select(const float* s, const int* pos, const int* ends, int* out, float* sout, int tokens,
                              int parts, int n, int k, int slice, long long stride_s, long long stride_o,
                              void* stream) {
    if(!s || !out || tokens < 1 || parts < 1 || n < 1 || k < 1 || slice < 1 ||
       static_cast<long long>(parts) * slice < n || parts > 65535)
        return 2;
    static kc::PerDevice<int> limit;
    const int most = limit.get([] {
        int bytes = 0;
        if(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, kc::current_device()) != cudaSuccess)
            (void)cudaGetLastError();
        return bytes;
    });
    // The slice beside the kernel's own shared memory (its histogram and sums: kStatic bytes at most).
    const long long bytes = static_cast<long long>(slice + kPad) * 4 * (pos ? 2 : 1);
    if(bytes > most - kStatic || slice > kThreads * kRun) return 3;
    static kc::PerDevice<cudaError_t> optin;
    if(kc::shared_memory_optin(optin, select_kernel, most - kStatic) != cudaSuccess) return 3;
    Select a{s, pos, ends, out, sout, n, k, slice, stride_s, stride_o};
    select_kernel<<<dim3(tokens, parts), kThreads, static_cast<size_t>(bytes), static_cast<cudaStream_t>(stream)>>>(a);
    return cudaGetLastError() == cudaSuccess ? 0 : 5;
}

// The most scores a slice holds on the current device (a candidate takes twice a score's room).
extern "C" int crit_si_slice_room() {
    int bytes = 0;
    if(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, kc::current_device()) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    const int room = bytes > kStatic ? (bytes - kStatic) / 4 - 2 * kPad : 0;
    return room < kThreads * kRun ? room : kThreads * kRun;
}
