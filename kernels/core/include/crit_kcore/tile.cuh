// crit's kernel core: K-major byte tiles for tensor-core GEMMs that accumulate in the MMA (kernels/core/README.md,
// "tile"). A tile is ROWS rows x BK bytes of a row-major operand (an E4M3 row of BK values, or a packed E2M1 / INT4 row
// of 2 BK), staged in shared memory with its 16-byte chunks swizzled by the row, so the eight rows an ldmatrix reads
// at one chunk column land in eight different bank groups; copies are cp.async (load.cuh) of 16-byte chunks.
//
//   KTileCopy<ROWS, BK, THREADS>  a CTA's copies of one tile a stage, each thread's rows fixed for the whole loop
//   ldsm_a / ldsm_b2              one k-step (32 bytes of every row: an 8-bit m16n8k32 or a 4-bit m16n8k64) as the
//                                 MMA's A fragment (16 rows) or as the B fragments of two 8-column tiles
//   copy_rows / row_copy_width    the rows' scale bytes of a stage (a few bytes a row) by the widest aligned cp.async
//   tile_of                       a CTA's (m, n) tile in a grouped order, so the CTAs that run together share rows
//
// Mechanisms after CUTLASS (BSD-3-Clause): its multistage mainloop's K-major swizzled shared-memory layouts for 8-bit
// and 4-bit operands and its threadblock swizzle; the code is written from the PTX ISA.
#pragma once
#include "crit_kcore/load.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace crit::kcore::tile {

// The byte offset of 16-byte chunk `chunk` of row `row` in a BK-byte-row tile (BK 32, 64, 128 or 256): the chunk index
// XORed with row bits, Swizzle<log2(min(BK, 128) / 16), 4, 3> on the unswizzled offset (load.cuh's swizzle; 256-byte
// rows swizzle each 128-byte half like a 128-byte row).
template <int BK>
__host__ __device__ __forceinline__ constexpr uint32_t offset(int row, int chunk) {
    static_assert(BK == 32 || BK == 64 || BK == 128 || BK == 256, "tile rows of 32, 64, 128 or 256 bytes");
    if constexpr(BK == 256) {
        return uint32_t(row) * 256u + 128u * uint32_t(chunk >> 3) + 16u * uint32_t((chunk ^ row) & 7);
    } else {
        constexpr int B = BK == 128 ? 3 : BK == 64 ? 2 : 1;
        return swizzle<B, 4, 3>(uint32_t(row) * uint32_t(BK) + 16u * uint32_t(chunk));
    }
}

// A CTA's copies of a ROWS x BK tile from a row-major operand (rows `ld` bytes apart, 16-byte aligned): thread t copies
// chunk t % (BK / 16) of rows t / (BK / 16) + u RS (RS = THREADS / (BK / 16), a multiple of 8, so the chunk's swizzle
// is the same for all its rows): one source pointer and stride a thread for the whole loop, a stage gives the k
// offset. Rows at or past `valid_rows` and chunks at or past `valid_bytes` of a row are zero-filled (nothing read).
template <int ROWS, int BK, int THREADS>
struct KTileCopy {
    static constexpr int kRowStep = THREADS / (BK / 16);
    static_assert(THREADS % (BK / 16) == 0 && kRowStep % 8 == 0, "THREADS a multiple of BK / 2");
    static constexpr int kPer = (ROWS + kRowStep - 1) / kRowStep;
    const unsigned char* src;
    std::size_t step;
    uint32_t dst;
    int row, at, valid_rows;

    __device__ __forceinline__ void init(const void* base, std::size_t ld, int valid_rows_, int tid) {
        row = tid / (BK / 16);
        at = 16 * (tid % (BK / 16));
        valid_rows = valid_rows_;
        src = static_cast<const unsigned char*>(base) + std::size_t(row) * ld + at;
        step = std::size_t(kRowStep) * ld;
        dst = offset<BK>(row, tid % (BK / 16));
    }
    // The tile at byte k0 of every row into `tile` (shared memory); bytes at or past valid_bytes of a row read as 0.
    __device__ __forceinline__ void issue(unsigned char* tile, int k0, int valid_bytes) const {
        const bool k_ok = k0 + at < valid_bytes;
        const unsigned char* p = src + k0;
#pragma unroll
        for(int u = 0; u < kPer; ++u) {
            const int r = row + u * kRowStep;
            if(ROWS % kRowStep == 0 || r < ROWS) {
                const bool ok = k_ok && r < valid_rows;
                cp_async16(tile + dst + uint32_t(u * kRowStep * BK), ok ? p : src, ok);
            }
            p += step;
        }
    }
};

// k-step `ks` (bytes 32 ks .. 32 ks + 31 of every row) of rows row0 .. row0 + 15 as an m16 A fragment: a0 row gid
// bytes 4 tig .. + 3, a1 row gid + 8, a2 row gid bytes 16 + 4 tig .. + 3, a3 row gid + 8 (mma.cuh's 8-bit k32 and
// 4-bit k64 A layouts). row0 a multiple of 8.
template <int BK>
__device__ __forceinline__ void ldsm_a(const unsigned char* tile, int row0, int ks, int lane, uint32_t (&a)[4]) {
    const int m = lane >> 3, row = row0 + (lane & 7) + 8 * (m & 1), chunk = 2 * ks + (m >> 1);
    ldmatrix_x4(a[0], a[1], a[2], a[3], tile + offset<BK>(row, chunk));
}

// k-step `ks` of columns (rows of the tile) col0 .. col0 + 15 as the B fragments of two n8 tiles: b[0], b[1] columns
// col0 + gid (bytes 4 tig .. + 3 and 16 + 4 tig .. + 3), b[2], b[3] columns col0 + 8 + gid. col0 a multiple of 8.
template <int BK>
__device__ __forceinline__ void ldsm_b2(const unsigned char* tile, int col0, int ks, int lane, uint32_t (&b)[4]) {
    const int m = lane >> 3, row = col0 + (lane & 7) + 8 * (m >> 1), chunk = 2 * ks + (m & 1);
    ldmatrix_x4(b[0], b[1], b[2], b[3], tile + offset<BK>(row, chunk));
}
// The same for one n8 tile: b[0], b[1] of columns col0 + gid.
template <int BK>
__device__ __forceinline__ void ldsm_b1(const unsigned char* tile, int col0, int ks, int lane, uint32_t (&b)[2]) {
    const int m = (lane >> 3) & 1, row = col0 + (lane & 7), chunk = 2 * ks + m;
    ldmatrix_x2(b[0], b[1], tile + offset<BK>(row, chunk));
}

// The widest cp.async (16, 8 or 4 bytes) that keeps every piece of a stage's SB bytes a row aligned: rows `ld` bytes
// apart from `base`, stages SB bytes apart. 0 when not even 4 bytes are.
__host__ __device__ __forceinline__ int row_copy_width(const void* base, std::size_t ld, int sb) {
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(base) | std::uintptr_t(ld) | std::uintptr_t(sb);
    return a % 16 == 0 ? 16 : a % 8 == 0 ? 8 : a % 4 == 0 ? 4 : 0;
}

// Bytes byte0 .. byte0 + SB of rows row0 .. row0 + ROWS - 1 (rows `ld` bytes apart, each `row_bytes` long) into dst
// [ROWS][SB], `width` bytes a cp.async (row_copy_width, a divisor of row_bytes); rows at or past valid_rows and
// pieces at or past row_bytes zero-filled.
template <int ROWS, int SB, int THREADS>
__device__ __forceinline__ void copy_rows(unsigned char* dst, const unsigned char* base, std::size_t ld, int valid_rows,
                                          int byte0, int row_bytes, int width, int tid) {
    static_assert(SB % 4 == 0, "a row's stage of whole words");
    auto run = [&](auto w) {
        constexpr int W = decltype(w)::value, kPieces = ROWS * (SB / W);
#pragma unroll
        for(int i = tid; i < kPieces; i += THREADS) {
            const int row = i / (SB / W), at = byte0 + W * (i % (SB / W));
            const bool ok = row < valid_rows && at < row_bytes;
            cp_async_ca<W>(dst + row * SB + W * (i % (SB / W)), base + std::size_t(ok ? row : 0) * ld + (ok ? at : 0),
                           ok);
        }
    };
    if constexpr(SB % 16 == 0) {
        if(width == 16) return run(std::integral_constant<int, 16>{});
    }
    if constexpr(SB % 8 == 0) {
        if(width >= 8) return run(std::integral_constant<int, 8>{});
    }
    run(std::integral_constant<int, 4>{});
}

// CTA `linear` of a tiles_m x tiles_n grid in a grouped order: `group` n-tiles at a time (the n index fastest, then m),
// every m-tile of them before the next group, so the CTAs resident together read a few n-tiles' rows and the m-tiles'.
__host__ __device__ __forceinline__ void tile_of(int linear, int tiles_m, int tiles_n, int group, int& tm, int& tn) {
    const int per = group * tiles_m, g = linear / per, first = g * group;
    const int width = tiles_n - first < group ? tiles_n - first : group;
    const int in = linear - g * per;
    tm = in / width;
    tn = first + in % width;
}

// A CTA's TM x TN FP32 accumulator tile (each warp's MT x NT m16n8 tiles at rows wm0 + 16 i, columns wn0 + 8 j: acc[i][j]
// as mma.cuh's C layout), times `scale`, into rows of `out` (row r of the tile at out + r ld, valid_m rows and valid_n
// columns kept) through shared memory: CHUNK rows at a time, padded by 4 floats a row (the warps' 8-byte writes free of
// bank conflicts), then each row written by consecutive threads in 16-byte stores where out and ld allow. `smem` holds
// at least CHUNK (TN + 4) floats and nothing the CTA still reads (the caller's mainloop is over: a barrier comes first).
// The CHUNK for store_tile: the most rows of TM (halving, down to 16) whose padded rows fit in smem_bytes.
__host__ __device__ constexpr int store_chunk(int tm, int tn, int smem_bytes) {
    return tm <= 16 || tm * (tn + 4) * 4 <= smem_bytes ? tm : store_chunk(tm / 2, tn, smem_bytes);
}

template <int TM, int TN, int MT, int NT, int CHUNK, int THREADS>
__device__ __forceinline__ void store_tile(const float (&acc)[MT][NT][4], float scale, int wm0, int wn0, int lane,
                                           float* smem, float* out, std::size_t ld, int valid_m, int valid_n,
                                           int tid) {
    static_assert(CHUNK % 16 == 0 && TM % CHUNK == 0 && TN % 4 == 0, "whole 16-row chunks of the tile");
    constexpr int kLd = TN + 4, kVec = TN / 4;
    const int gid = lane >> 2, tig = lane & 3;
    const bool vec = (reinterpret_cast<std::uintptr_t>(out) % 16 == 0) && ld % 4 == 0;
#pragma unroll
    for(int c0 = 0; c0 < TM; c0 += CHUNK) {
        if(c0 >= valid_m) break;
        __syncthreads();
#pragma unroll
        for(int i = 0; i < MT; ++i) {
            const int r = wm0 + 16 * i - c0;
            if(r < 0 || r >= CHUNK) continue;
#pragma unroll
            for(int j = 0; j < NT; ++j)
#pragma unroll
                for(int h = 0; h < 2; ++h)
                    *reinterpret_cast<float2*>(smem + (r + gid + 8 * h) * kLd + wn0 + 8 * j + 2 * tig) =
                        make_float2(acc[i][j][2 * h] * scale, acc[i][j][2 * h + 1] * scale);
        }
        __syncthreads();
        const int rows = valid_m - c0 < CHUNK ? valid_m - c0 : CHUNK;
        if(vec && valid_n == TN) {
            for(int e = tid; e < rows * kVec; e += THREADS) {
                const int r = e / kVec, v = e % kVec;
                *reinterpret_cast<float4*>(out + std::size_t(c0 + r) * ld + 4 * v) =
                    *reinterpret_cast<const float4*>(smem + r * kLd + 4 * v);
            }
        } else {
            for(int e = tid; e < rows * TN; e += THREADS) {
                const int r = e / TN, col = e % TN;
                if(col < valid_n) out[std::size_t(c0 + r) * ld + col] = smem[r * kLd + col];
            }
        }
    }
}

}  // namespace crit::kcore::tile
