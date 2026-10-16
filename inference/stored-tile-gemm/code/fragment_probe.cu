// Measure the fragment layout of the block-scaled NVFP4 MMA (m16n8k64 kind::mxf4nvf4, scale_vec::4X, selectors 0)
// on the device, instead of trusting a reading of the documentation.
//
// 1. Scale operands: with A = B = 1.0 everywhere and every scale 1.0 (E4M3 0x38), each output is 64. One warp per
//    (lane, byte) sets that single scale byte to 2.0 (0x40); the outputs that rise to 80 name the row (scale A) or
//    column (scale B) the byte belongs to. The expected mapping is printed beside the measured one.
// 2. Values: random E2M1 codes and E4M3 scales packed by the assumed fragment layout (a0 = row g, k 8t..8t+7;
//    a1 = row g+8; a2/a3 = k + 32; b0 = k 8t.., col g; b1 = k + 32; byte j of a scale word covers k 16j..16j+15;
//    row g's scales from lane 4g, row g+8's from lane 4g+1, column g's from lane 4g), compared with an FP64 dot
//    product. Products of E2M1 values and E4M3 scales are exact, so any difference means the layout is wrong.
//
// Build for an arch-specific target, e.g. nvcc -std=c++17 -gencode arch=compute_120a,code=sm_120a fragment_probe.cu
// Exit status 0 when both the scale mapping and the value layout match.
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define CRIT_PROBE_FP4 1
#else
#define CRIT_PROBE_FP4 0
#endif

namespace {

// Per warp w and lane l: inputs in[(w * 32 + l) * 8 + {a0..a3, b0, b1, sa, sb}], outputs out[(w * 32 + l) * 4 + i].
__global__ void probe_kernel(const std::uint32_t* in, float* out, int* compiled) {
    const int w = blockIdx.x, l = threadIdx.x;
    if(w == 0 && l == 0) *compiled = CRIT_PROBE_FP4;
    const std::uint32_t* p = in + (w * 32 + l) * 8;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#if CRIT_PROBE_FP4
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(p[0]), "r"(p[1]), "r"(p[2]), "r"(p[3]), "r"(p[4]), "r"(p[5]), "r"(p[6]), "r"(p[7]));
#endif
    for(int i = 0; i < 4; ++i) out[(w * 32 + l) * 4 + i] = c[i];
}

float e2m1(int code) {
    static const float v[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (code & 8 ? -1.f : 1.f) * v[code & 7];
}
float e4m3(std::uint8_t b) { __nv_fp8_e4m3 f; f.__x = b; return static_cast<float>(f); }

bool run(const std::vector<std::uint32_t>& in, int warps, std::vector<float>& out, int& compiled) {
    std::uint32_t* d_in = nullptr;
    float* d_out = nullptr;
    int* d_flag = nullptr;
    out.assign(static_cast<std::size_t>(warps) * 32 * 4, 0.f);
    bool ok = cudaMalloc(&d_in, in.size() * 4) == cudaSuccess && cudaMalloc(&d_out, out.size() * 4) == cudaSuccess &&
              cudaMalloc(&d_flag, sizeof(int)) == cudaSuccess &&
              cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess;
    if(ok) {
        probe_kernel<<<warps, 32>>>(d_in, d_out, d_flag);
        ok = cudaDeviceSynchronize() == cudaSuccess &&
             cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(&compiled, d_flag, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_flag);
    return ok;
}

// D[row][col] from the accumulator layout c0 = (g, 2t), c1 = (g, 2t+1), c2 = (g+8, 2t), c3 = (g+8, 2t+1).
void accumulators(const std::vector<float>& out, int w, float (&D)[16][8]) {
    for(int l = 0; l < 32; ++l) {
        const int g = l >> 2, t = l & 3;
        const float* c = out.data() + (w * 32 + l) * 4;
        D[g][2 * t] = c[0]; D[g][2 * t + 1] = c[1]; D[g + 8][2 * t] = c[2]; D[g + 8][2 * t + 1] = c[3];
    }
}

}  // namespace

int main() {
    // 1. scale operands: 128 warps for scale A (lane w / 4, byte w % 4), 128 for scale B
    const int W = 256;
    std::vector<std::uint32_t> in(static_cast<std::size_t>(W) * 32 * 8);
    for(int w = 0; w < W; ++w)
        for(int l = 0; l < 32; ++l) {
            std::uint32_t* p = in.data() + (w * 32 + l) * 8;
            for(int i = 0; i < 6; ++i) p[i] = 0x22222222u;  // E2M1 1.0 in every nibble
            p[6] = p[7] = 0x38383838u;                       // E4M3 1.0 in every scale byte
            const int e = w % 128, lane = e / 4, byte = e % 4;
            if(l == lane) {
                std::uint32_t& s = p[w < 128 ? 6 : 7];
                s = (s & ~(0xffu << (8 * byte))) | (0x40u << (8 * byte));
            }
        }
    std::vector<float> out;
    int compiled = 0;
    if(!run(in, W, out, compiled)) { std::printf("CUDA error\n"); return 2; }
    if(!compiled) { std::printf("not built for sm_120a / sm_121a: the block-scaled MMA is unavailable\n"); return 2; }
    bool scales_ok = true;
    for(int w = 0; w < W; ++w) {
        float D[16][8];
        accumulators(out, w, D);
        const int e = w % 128, lane = e / 4;
        int hit = -1, count = 0;
        if(w < 128) { for(int r = 0; r < 16; ++r) if(D[r][0] != 64.f) { hit = r; ++count; } }
        else { for(int c = 0; c < 8; ++c) if(D[0][c] != 64.f) { hit = c; ++count; } }
        // expected: scale A lane 4g -> row g, lane 4g+1 -> row g+8, other lanes unused; scale B lane 4g -> column g
        int expect = -1;
        if(w < 128) expect = (lane & 3) == 0 ? lane >> 2 : (lane & 3) == 1 ? (lane >> 2) + 8 : -1;
        else expect = (lane & 3) == 0 ? lane >> 2 : -1;
        if(hit != expect || count > 1) {
            scales_ok = false;
            std::printf("scale %s lane %d byte %d: measured %d, expected %d\n", w < 128 ? "A" : "B", lane, e % 4, hit, expect);
        }
    }
    std::printf("scale operands: %s\n", scales_ok ? "lane 4g -> row g, lane 4g+1 -> row g+8 (A); lane 4g -> column g (B)"
                                                 : "MISMATCH (see above)");
    // 2. values
    const int V = 32;
    std::mt19937 rng(1);
    std::vector<std::uint8_t> A(V * 16 * 64), B(V * 8 * 64), SA(V * 16 * 4), SB(V * 8 * 4);
    for(auto& x : A) x = static_cast<std::uint8_t>(rng() & 15);
    for(auto& x : B) x = static_cast<std::uint8_t>(rng() & 15);
    for(auto& x : SA) x = static_cast<std::uint8_t>(0x30 + rng() % 16);
    for(auto& x : SB) x = static_cast<std::uint8_t>(0x30 + rng() % 16);
    in.assign(static_cast<std::size_t>(V) * 32 * 8, 0u);
    for(int w = 0; w < V; ++w)
        for(int l = 0; l < 32; ++l) {
            const int g = l >> 2, t = l & 3;
            std::uint32_t* p = in.data() + (w * 32 + l) * 8;
            auto pack = [&](const std::vector<std::uint8_t>& m, int rows, int row, int k0) {
                std::uint32_t v = 0;
                for(int i = 0; i < 8; ++i) v |= static_cast<std::uint32_t>(m[(w * rows + row) * 64 + k0 + i]) << (4 * i);
                return v;
            };
            p[0] = pack(A, 16, g, 8 * t); p[1] = pack(A, 16, g + 8, 8 * t);
            p[2] = pack(A, 16, g, 32 + 8 * t); p[3] = pack(A, 16, g + 8, 32 + 8 * t);
            p[4] = pack(B, 8, g, 8 * t); p[5] = pack(B, 8, g, 32 + 8 * t);
            const int arow = t == 1 ? g + 8 : g;
            for(int j = 0; j < 4; ++j) {
                p[6] |= static_cast<std::uint32_t>(SA[(w * 16 + arow) * 4 + j]) << (8 * j);
                p[7] |= static_cast<std::uint32_t>(SB[(w * 8 + g) * 4 + j]) << (8 * j);
            }
        }
    if(!run(in, V, out, compiled)) { std::printf("CUDA error\n"); return 2; }
    double worst = 0.0;
    for(int w = 0; w < V; ++w) {
        float D[16][8];
        accumulators(out, w, D);
        for(int r = 0; r < 16; ++r)
            for(int c = 0; c < 8; ++c) {
                double want = 0.0;
                for(int k = 0; k < 64; ++k)
                    want += static_cast<double>(e2m1(A[(w * 16 + r) * 64 + k])) * e2m1(B[(w * 8 + c) * 64 + k]) *
                            e4m3(SA[(w * 16 + r) * 4 + k / 16]) * e4m3(SB[(w * 8 + c) * 4 + k / 16]);
                worst = std::fmax(worst, std::fabs(D[r][c] - want));
            }
    }
    std::printf("value layout: max abs difference %.3g over %d random tiles\n", worst, V);
    return scales_ok && worst == 0.0 ? 0 : 1;
}
