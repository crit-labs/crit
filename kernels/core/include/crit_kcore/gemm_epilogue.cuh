// crit's kernel core: epilogues of tensor-core GEMMs that accumulate in the MMA (kernels/core/README.md,
// "gemm_epilogue"). A CTA's TM x TN FP32 accumulator tile (each warp's MT x NT m16n8 tiles at rows wm0 + 16 i, columns
// wn0 + 8 j, mma.cuh's C layout) scaled by a rank-1 factor, row factor x column factor (a token's activation scale and
// a weight row's scale times the matrix's post), rounded once to the output type (FP32, or BF16 / FP16 to nearest even)
// and stored: from registers (store_scaled) or staged through shared memory as whole rows (store_scaled_staged, 16-byte
// stores). The scales are read by the caller into registers (row_scales / col_scales: a lane's two rows of each m16
// tile and two columns of each n8 tile), so one epilogue serves any scale layout.
//
// Mechanisms after CUTLASS (BSD-3-Clause: its epilogue's per-row and per-column broadcast and shared-memory staged
// store); the code is written anew.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace crit::kcore::gemm_out {

// The output types: float, __nv_bfloat16 or __half (round to nearest even).
template <class OUT>
__device__ __forceinline__ OUT round_to(float v) {
    if constexpr(std::is_same_v<OUT, float>) return v;
    else if constexpr(std::is_same_v<OUT, __half>) return __float2half_rn(v);
    else return __float2bfloat16_rn(v);
}
// Two adjacent outputs as one 8-byte (float) or 4-byte (BF16, FP16) word.
template <class OUT>
__device__ __forceinline__ void store_pair(OUT* p, float v0, float v1) {
    if constexpr(std::is_same_v<OUT, float>) *reinterpret_cast<float2*>(p) = make_float2(v0, v1);
    else if constexpr(std::is_same_v<OUT, __half>) *reinterpret_cast<__half2*>(p) = __floats2half2_rn(v0, v1);
    else *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(v0, v1);
}

// The CHUNK (rows of the tile staged at a time) for store_scaled_staged: the most rows of TM (halving, down to 16)
// whose padded rows fit in smem_bytes.
template <class OUT>
__host__ __device__ constexpr int staged_chunk(int tm, int tn, int smem_bytes) {
    return tm <= 16 || tm * (tn + 16 / int(sizeof(OUT))) * int(sizeof(OUT)) <= smem_bytes
               ? tm : staged_chunk<OUT>(tm / 2, tn, smem_bytes);
}

// acc[i][j][2 h + c] (row wm0 + 16 i + gid + 8 h, column wn0 + 8 j + 2 tig + c) times rs[i][h] cs[j][c], into
// out (row r of the tile at out + r ld) for rows < valid_m and columns < valid_n, pairs where out and ld allow.
template <class OUT, int MT, int NT>
__device__ __forceinline__ void store_scaled(const float (&acc)[MT][NT][4], const float (&rs)[MT][2],
                                             const float (&cs)[NT][2], int wm0, int wn0, int lane, OUT* out,
                                             std::size_t ld, int valid_m, int valid_n) {
    const int gid = lane >> 2, tig = lane & 3;
    const bool pairs = ld % 2 == 0 && reinterpret_cast<std::uintptr_t>(out) % (2 * sizeof(OUT)) == 0;
#pragma unroll
    for(int i = 0; i < MT; ++i)
#pragma unroll
        for(int h = 0; h < 2; ++h) {
            const int t = wm0 + 16 * i + gid + 8 * h;
            if(t >= valid_m) continue;
            OUT* row = out + std::size_t(t) * ld;
#pragma unroll
            for(int j = 0; j < NT; ++j) {
                const int c = wn0 + 8 * j + 2 * tig;
                const float v0 = acc[i][j][2 * h] * rs[i][h] * cs[j][0];
                const float v1 = acc[i][j][2 * h + 1] * rs[i][h] * cs[j][1];
                if(pairs && c + 1 < valid_n) {
                    store_pair<OUT>(row + c, v0, v1);
                } else {
                    if(c < valid_n) row[c] = round_to<OUT>(v0);
                    if(c + 1 < valid_n) row[c + 1] = round_to<OUT>(v1);
                }
            }
        }
}

// The same through shared memory: CHUNK rows at a time (staged_chunk), padded by 16 bytes a row (the fragments'
// writes free of bank conflicts), then each row written by consecutive threads in 16-byte stores where out and ld
// allow. `smem` holds CHUNK (TN + 16 / sizeof(OUT)) outputs and nothing the CTA still reads (the caller's mainloop is
// over; the first barrier is here).
template <class OUT, int TM, int TN, int MT, int NT, int CHUNK, int THREADS>
__device__ __forceinline__ void store_scaled_staged(const float (&acc)[MT][NT][4], const float (&rs)[MT][2],
                                                    const float (&cs)[NT][2], int wm0, int wn0, int lane, OUT* smem,
                                                    OUT* out, std::size_t ld, int valid_m, int valid_n, int tid) {
    constexpr int kPer = 16 / int(sizeof(OUT));   // outputs a 16-byte store
    static_assert(CHUNK % 16 == 0 && TM % CHUNK == 0 && TN % kPer == 0, "whole 16-row chunks, 16-byte row pieces");
    constexpr int kLd = TN + kPer, kVec = TN / kPer;
    const int gid = lane >> 2, tig = lane & 3;
    const bool vec = reinterpret_cast<std::uintptr_t>(out) % 16 == 0 && ld % kPer == 0;
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
                    store_pair<OUT>(smem + (r + gid + 8 * h) * kLd + wn0 + 8 * j + 2 * tig,
                                    acc[i][j][2 * h] * rs[i][h] * cs[j][0], acc[i][j][2 * h + 1] * rs[i][h] * cs[j][1]);
        }
        __syncthreads();
        const int rows = valid_m - c0 < CHUNK ? valid_m - c0 : CHUNK;
        if(vec && valid_n == TN) {
            for(int e = tid; e < rows * kVec; e += THREADS) {
                const int r = e / kVec, v = e % kVec;
                *reinterpret_cast<uint4*>(out + std::size_t(c0 + r) * ld + kPer * v) =
                    *reinterpret_cast<const uint4*>(smem + r * kLd + kPer * v);
            }
        } else {
            for(int e = tid; e < rows * TN; e += THREADS) {
                const int r = e / TN, col = e % TN;
                if(col < valid_n) out[std::size_t(c0 + r) * ld + col] = smem[r * kLd + col];
            }
        }
    }
}

}  // namespace crit::kcore::gemm_out
