// crit's kernel core: the declared decode rows (kernels/core/README.md, "multiply"): 1 to 8 tokens a job times one
// matrix a job, on multiply.cuh's declared paths. The kernels are templates over the caller's matrix, activation and
// job types (the fields each names below), so int-block-experts and fp8-block-rows (and any kernel that multiplies
// those formats) run one implementation; the operations and their order are the ones those kernels declared, so an
// output's bits are theirs.
//
// int_rows_kernel (declared_int): a half-warp an output row, lane b % 16 takes blocks b in increasing order, each the
// exact dp4a dot of 16 codes and 16 int8 activations then the step; the half-warp's halving tree; times post.
//   Matrix: codes, scales (E4M3 bytes [rows][cols / 16]), rows, cols, post, int4. Acts: values, scales, ld, lds,
//   block (16 or 32). Job: matrix, tokens, act_row[], out_row[].
// fp8_rows_kernel (declared_fp8): 16 rows a CTA, the 16 lanes over its warps, one e4m3 MMA a 32-block then the step;
// one thread an output reduces its 16 lanes by the halving tree in shared memory; times post.
//   Matrix: codes, scales, rows, cols, scale_rows, scale_k, post, scale_e8m0. Acts: codes, scales, ld, lds, group.
//   Job as above.
// Knobs (speed only): THREADS, ROWS (rows a half-warp), STREAM (codes with L1::no_allocate), UNROLL (blocks a lane
// loads first) for int_rows_kernel; WARPS (4, 8, 16), AHEAD (rounds of loads in flight), STREAM for fp8_rows_kernel.
#pragma once
#include "crit_kcore/blockscaled_mma.cuh"
#include "crit_kcore/formats.cuh"
#include "crit_kcore/multiply.cuh"

#include <cstddef>
#include <cstdint>

namespace crit::kcore::declared {

// The records the declared rows take, as int-block-experts' and fp8-block-rows' Python layers pack them (struct
// "<QQiifi", "<QQiiiifi" and "<ii8i8i"): a kernel that runs these rows for those packages declares its own types with
// these layouts or uses these.
struct IntBlockMatrix {   // codes [rows][cols / 2], E4M3 scales [rows][cols / 16]; a weight is value * scale * post
    const std::uint8_t* codes = nullptr;
    const std::uint8_t* scales = nullptr;
    int rows = 0, cols = 0;
    float post = 1.0f;
    int int4 = 0;   // 1: INT4 values; 0: E2M1 x 2
};
struct IntBlockActs {     // values [row][ld] int8, scales [row][lds] FP32, one per `block` (16 or 32) values
    const std::int8_t* values = nullptr;
    const float* scales = nullptr;
    int ld = 0, lds = 0, block = 16;
    int rows = 0;   // activation rows (the tuning table's row count; 0: the job count)
};
struct Fp8BlockMatrix {   // codes [rows][cols] E4M3; scales [ceil(rows / scale_rows)][cols / scale_k], FP32 or E8M0
    const std::uint8_t* codes = nullptr;
    const void* scales = nullptr;
    int rows = 0, cols = 0, scale_rows = 1, scale_k = 32;
    float post = 1.0f;
    int scale_e8m0 = 0;
};
struct Fp8BlockActs {     // codes [row][ld] E4M3, scales [row][lds] FP32, one per `group` codes (a multiple of 32)
    const std::uint8_t* codes = nullptr;
    const float* scales = nullptr;
    int ld = 0, lds = 0, group = 32;
};
constexpr int kJobTokens = 8;
struct RowJob {           // one matrix and up to 8 activation rows; out[out_row[j]][r]
    int matrix = 0;
    int tokens = 0;
    int act_row[kJobTokens] = {};
    int out_row[kJobTokens] = {};
};

template <int STREAM>
__device__ __forceinline__ uint2 load_u2(const std::uint8_t* p) {
    if constexpr(STREAM != 0) {
        uint2 r;
        asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
        return r;
    } else {
        return *reinterpret_cast<const uint2*>(p);
    }
}

// The E4M3 scale bytes' values, decoded once a CTA into shared memory.
__device__ __forceinline__ void e4m3_table(float* table) {
    for(int i = int(threadIdx.x); i < 256; i += int(blockDim.x)) table[i] = e4m3_value(unsigned(i));
    __syncthreads();
}

template <int THREADS, int ROWS, int STREAM, int UNROLL, int TOKENS, class Matrix, class Acts, class Job>
__global__ void int_rows_kernel(const Matrix* matrices, const Job* jobs, Acts acts, float* out, int out_ld) {
    using P = multiply::DeclaredInt;
    constexpr int kHalfWarps = THREADS / 16;
    __shared__ float scale_of[256];
    e4m3_table(scale_of);
    const Job& job = jobs[blockIdx.y];
    const Matrix m = matrices[job.matrix];
    const int row0 = (int(blockIdx.x) * kHalfWarps + int(threadIdx.x) / 16) * ROWS;
    if(row0 >= m.rows) return;   // the whole half-warp leaves together: the shuffles stay within it
    const unsigned lane = threadIdx.x % 16u;
    const unsigned group = 0xffffu << (threadIdx.x % 32u & 16u);
    const blockscaled::NibbleTable table = blockscaled::nibble_table(m.int4 != 0);
    const int blocks = m.cols / 16, ratio = acts.block / 16, tokens = job.tokens;
    const std::uint8_t* codes[ROWS];
    const std::uint8_t* scales[ROWS];
#pragma unroll
    for(int r = 0; r < ROWS; ++r) {
        const int row = row0 + r < m.rows ? row0 + r : m.rows - 1;   // a row past the matrix reads its last, writes nothing
        codes[r] = m.codes + std::size_t(row) * std::size_t(m.cols / 2);
        scales[r] = m.scales + std::size_t(row) * std::size_t(blocks);
    }
    float acc[ROWS][TOKENS];
#pragma unroll
    for(int r = 0; r < ROWS; ++r)
#pragma unroll
        for(int t = 0; t < TOKENS; ++t) acc[r][t] = 0.0f;
    for(int b0 = int(lane); b0 < blocks; b0 += 16 * UNROLL) {
        uint2 q[UNROLL][ROWS];   // the blocks' 16 codes
        unsigned sb[UNROLL][ROWS];
#pragma unroll
        for(int u = 0; u < UNROLL; ++u) {
            const int b = b0 + 16 * u;
            if(UNROLL > 1 && b >= blocks) break;
#pragma unroll
            for(int r = 0; r < ROWS; ++r) {
                q[u][r] = load_u2<STREAM>(codes[r] + 8 * b);
                sb[u][r] = scales[r][b];
            }
        }
#pragma unroll
        for(int u = 0; u < UNROLL; ++u) {
            const int b = b0 + 16 * u;
            if(UNROLL > 1 && b >= blocks) break;
            int w[ROWS][4];
            float ws[ROWS];
#pragma unroll
            for(int r = 0; r < ROWS; ++r) {
                P::weights(q[u][r], table, w[r]);
                ws[r] = scale_of[sb[u][r]];
            }
#pragma unroll
            for(int t = 0; t < TOKENS; ++t) {
                if(t >= tokens) break;
                const int4 x = *reinterpret_cast<const int4*>(acts.values + std::size_t(job.act_row[t]) * acts.ld + 16 * b);
                const float as = acts.scales[std::size_t(job.act_row[t]) * acts.lds + b / ratio];
#pragma unroll
                for(int r = 0; r < ROWS; ++r) acc[r][t] = P::step(P::dot(w[r], x), ws[r], as, acc[r][t]);
            }
        }
    }
#pragma unroll
    for(int t = 0; t < TOKENS; ++t) {
        if(t >= tokens) break;
#pragma unroll
        for(int r = 0; r < ROWS; ++r) {
            const float v = P::tree(acc[r][t], group);
            if(lane == 0 && row0 + r < m.rows)
                out[std::size_t(job.out_row[t]) * out_ld + row0 + r] = __fmul_rn(v, m.post);
        }
    }
}

// Bytes 8q .. 8q + 7 of 32-block b of one row or token (crit's fragment order), weights read past L1 when STREAM.
__device__ __forceinline__ uint2 fp8_fragment(const std::uint8_t* row, int b, int q) {
    return *reinterpret_cast<const uint2*>(row + 32 * b + 8 * q);
}
template <int STREAM>
__device__ __forceinline__ uint2 fp8_weight_fragment(const std::uint8_t* row, int b, int q) {
    if constexpr(STREAM) {
        uint2 v;
        asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];"
                     : "=r"(v.x), "=r"(v.y) : "l"(row + 32 * b + 8 * q));
        return v;
    } else {
        return fp8_fragment(row, b, q);
    }
}

template <int WARPS, int AHEAD, int STREAM, int TOKENS, class Matrix, class Acts, class Job>
__global__ void __launch_bounds__(32 * WARPS) fp8_rows_kernel(const Matrix* matrices, const Job* jobs, Acts acts,
                                                              float* out, int out_ld) {
    using P = multiply::DeclaredFp8;
    constexpr int kBlock = P::kBlock, kLanes = P::kLanes;
    static_assert(kLanes % WARPS == 0 && 32 * WARPS >= 16 * TOKENS, "4, 8 or 16 warps");
    constexpr int kSlots = kLanes / WARPS;
    __shared__ float part[kLanes][16][TOKENS + 1];
    __shared__ int act_row[TOKENS], out_row[TOKENS];
    const Job* job = jobs + blockIdx.y;
    const Matrix m = matrices[job->matrix];
    const int row0 = int(blockIdx.x) * 16;
    if(row0 >= m.rows) return;   // the whole CTA
    const int tokens = job->tokens;
    if(threadIdx.x < TOKENS) {
        act_row[threadIdx.x] = int(threadIdx.x) < tokens ? job->act_row[threadIdx.x] : job->act_row[0];
        out_row[threadIdx.x] = job->out_row[threadIdx.x];
    }
    __syncthreads();
    const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32, g = lane / 4, q = lane % 4;
    const int ra = min(row0 + g, m.rows - 1), rb = min(row0 + g + 8, m.rows - 1);
    const std::uint8_t* wa = m.codes + std::size_t(ra) * std::size_t(m.cols);
    const std::uint8_t* wb = m.codes + std::size_t(rb) * std::size_t(m.cols);
    const std::uint8_t* xg = acts.codes + std::size_t(act_row[g]) * std::size_t(acts.ld);
    const multiply::ScaleGrid grid{static_cast<const std::uint8_t*>(m.scales), m.cols, m.scale_rows, m.scale_k,
                                   m.scale_e8m0};
    const std::uint8_t* wsa = grid.row(ra);
    const std::uint8_t* wsb = grid.row(rb);
    const float* xs0 = acts.scales + std::size_t(act_row[2 * q]) * std::size_t(acts.lds);
    const float* xs1 = acts.scales + std::size_t(act_row[2 * q + 1]) * std::size_t(acts.lds);
    const int blocks = m.cols / kBlock;
    float acc[kSlots][4];
#pragma unroll
    for(int s = 0; s < kSlots; ++s)
#pragma unroll
        for(int o = 0; o < 4; ++o) acc[s][o] = 0.0f;
    // AHEAD rounds of loads issued before their MMAs; each lane's blocks still in increasing order.
    for(int base = 0; base < blocks; base += kLanes * AHEAD) {
        uint2 fa[AHEAD][kSlots], fb[AHEAD][kSlots], fx[AHEAD][kSlots];
        float w0[AHEAD][kSlots], w1[AHEAD][kSlots], a0[AHEAD][kSlots], a1[AHEAD][kSlots];
#pragma unroll
        for(int u = 0; u < AHEAD; ++u)
#pragma unroll
            for(int s = 0; s < kSlots; ++s) {
                const int b = min(base + kLanes * u + warp + WARPS * s, blocks - 1);
                fa[u][s] = fp8_weight_fragment<STREAM>(wa, b, q);
                fb[u][s] = fp8_weight_fragment<STREAM>(wb, b, q);
                fx[u][s] = fp8_fragment(xg, b, q);
                const int wk = b * kBlock / m.scale_k, xk = b * kBlock / acts.group;
                w0[u][s] = multiply::ScaleGrid::at(wsa, wk, m.scale_e8m0);
                w1[u][s] = multiply::ScaleGrid::at(wsb, wk, m.scale_e8m0);
                a0[u][s] = xs0[xk];
                a1[u][s] = xs1[xk];
            }
#pragma unroll
        for(int u = 0; u < AHEAD; ++u)
#pragma unroll
            for(int s = 0; s < kSlots; ++s) {
                if(base + kLanes * u + warp + WARPS * s >= blocks) continue;
                float d[4];
                P::block(fa[u][s], fb[u][s], fx[u][s], d);
                acc[s][0] = P::step(d[0], w0[u][s], a0[u][s], acc[s][0]);
                acc[s][1] = P::step(d[1], w0[u][s], a1[u][s], acc[s][1]);
                acc[s][2] = P::step(d[2], w1[u][s], a0[u][s], acc[s][2]);
                acc[s][3] = P::step(d[3], w1[u][s], a1[u][s], acc[s][3]);
            }
    }
#pragma unroll
    for(int s = 0; s < kSlots; ++s)
#pragma unroll
        for(int o = 0; o < 4; ++o) part[warp + WARPS * s][g + 8 * (o / 2)][2 * q + o % 2] = acc[s][o];
    __syncthreads();
    if(int(threadIdx.x) >= 16 * TOKENS) return;
    const int pr = int(threadIdx.x) / TOKENS, pt = int(threadIdx.x) % TOKENS;
    if(row0 + pr >= m.rows || pt >= tokens) return;
    float v[kLanes];
#pragma unroll
    for(int j = 0; j < kLanes; ++j) v[j] = part[j][pr][pt];
    out[std::size_t(out_row[pt]) * std::size_t(out_ld) + std::size_t(row0 + pr)] = __fmul_rn(P::tree(v), m.post);
}

}  // namespace crit::kcore::declared
