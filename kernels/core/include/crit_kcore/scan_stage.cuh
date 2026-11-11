// crit's kernel core: staged state tiles for column scans (kernels/core/README.md, "scan"). A column scan's lane holds
// key rows lane * R .. + R - 1 of some value columns of a [dk][dv] state; loading those straight into registers reads
// 16 bytes a lane at a stride of dv, and the registers are held while the load is in flight. Staging copies the 32 R
// key rows x 32 value columns of a slot into shared memory by cp.async instead, each key row's 128 contiguous bytes by
// 8 neighbouring threads (whole lines), so a block can have several slots in flight ahead of the one it computes; each
// lane then reads its piece, the values a direct load gives (zeros past dk and dv). The 16-byte chunks of a row are
// swizzled by the reading lane, so the 8 lanes of a float4 phase hit 8 different bank groups. Moves bytes only: a
// scan's values and sums are its own (scan_columns.cuh). Mechanism after FlashInfer's gated-delta decode
// (Apache-2.0: cp.async-staged state tiles); written from the PTX ISA for crit's [dk][dv] layout.
#pragma once
#include <cstddef>
#include <cstdint>

#include "crit_kcore/load.cuh"

namespace crit::kcore::scan {

// Floats of a staged tile of 32 R key rows x 32 value columns.
template <int R>
__host__ __device__ constexpr int staged_tile_floats() { return 32 * R * 32; }

// Where 16-byte chunk c (columns 4 c .. 4 c + 3) of key row x lives in a staged tile: chunks permuted by the reading
// lane (x / R), so lanes l .. l + 7 reading the same chunk index hit different bank groups.
template <int R>
__device__ __forceinline__ int staged_offset(int x, int c) { return x * 32 + ((c ^ ((x / R) & 7)) << 2); }

// Stage columns col0 .. col0 + 31 of key rows 0 .. 32 R - 1 of a [dk][dv] matrix (head, row stride dv floats) by the
// NT threads of a block (32 R x 8 a multiple of NT), zeros past dk and dv: cp.async where every row is 16-byte
// aligned (dv a multiple of 4, head aligned), else plain loads. The caller commits the group and waits before reading.
template <int R, int NT>
__device__ __forceinline__ void stage_columns(float* tile, const float* head, int dk, int dv, int col0) {
    constexpr int kChunks = 32 * R * 8;
    static_assert(kChunks % NT == 0, "the block's threads divide the tile's chunks");
    const bool wide = (dv & 3) == 0 && (reinterpret_cast<std::uintptr_t>(head) & 15u) == 0;
#pragma unroll
    for(int i0 = 0; i0 < kChunks; i0 += NT) {
        const int i = i0 + int(threadIdx.x), x = i >> 3, c = i & 7, j = col0 + 4 * c;
        float* dst = tile + staged_offset<R>(x, c);
        if(wide) {
            const bool valid = x < dk && j < dv;
            cp_async16(dst, valid ? head + std::size_t(x) * dv + j : head, valid);
        } else {
#pragma unroll
            for(int jj = 0; jj < 4; ++jj) dst[jj] = x < dk && j + jj < dv ? head[std::size_t(x) * dv + j + jj] : 0.f;
        }
    }
}

// A lane's piece of a staged tile: S[r][j] = key row lane * R + r, column jl + j of the slot (jl a multiple of 4).
template <int R, int CW>
__device__ __forceinline__ void read_columns(float (&S)[R][CW], const float* tile, int lane, int jl) {
    static_assert(CW % 4 == 0, "whole 16-byte chunks");
#pragma unroll
    for(int c = 0; c < CW; c += 4) {
#pragma unroll
        for(int r = 0; r < R; ++r) {
            const int x = lane * R + r;
            const float4 v = *reinterpret_cast<const float4*>(tile + staged_offset<R>(x, (jl + c) >> 2));
            S[r][c] = v.x;
            S[r][c + 1] = v.y;
            S[r][c + 2] = v.z;
            S[r][c + 3] = v.w;
        }
    }
}

}  // namespace crit::kcore::scan
