// blockscaled_mma.cuh's fast E4M3 block sum (the sm_120 family's block-scaled kind::mxf8f6f4 m16n8k32 with unit E8M0
// scales) against the plain e4m3 m16n8k32 from a zero accumulator, bit for bit (NaN payloads too), on this GPU: the
// test behind arch.cuh's CRIT_KCORE_FAST_E4M3_SUM, which lets block_sum_e4m3_fast take the fast MMA only on targets
// where this passed. Operands: codes cycling through every byte at every k position, uniformly random
// codes (NaNs and infinities of E4M3's encoding included), large magnitudes, subnormals, and blocks of near-one products
// that cancel (x against -x, plus small residuals), about 1.1e9 block sums in all. Where the target has no
// block-scaled MMA (plain sm_120, compute_80 PTX) both sides are the plain MMA, and below sm_89 there is no E4M3 MMA:
// the check is skipped. Also: block_sum_e4m3_fast gives the plain sum's bits on every target (the guard's fallback).
// tests/test_core.py builds it for compute_80 (PTX), plain sm_120 and sm_120a.
#include "check.cuh"
#include "crit_kcore/blockscaled_mma.cuh"

#include <cstdio>
#include <string>

namespace bs = crit::kcore::blockscaled;
using check::report;

namespace {

#if CRIT_KCORE_HAS_FP8_MMA
__device__ __forceinline__ uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// A code byte of distribution `kind` from a hash h.
__device__ __forceinline__ uint32_t code(int kind, uint32_t h, uint32_t pair) {
    switch(kind) {
        case 1: return h & 0xFFu;                                              // every code, NaN and inf included
        case 2: return (h & 0x80u) | (0x60u + (h >> 8) % 0x1Fu);               // large: exponents 12..15
        case 3: return (h & 0x80u) | ((h >> 8) & 0x0Fu);                      // subnormal and the smallest normals
        case 4: return (pair & 0x7Fu) % 0x7Eu | ((h & 1u) << 7);               // near one: magnitudes the pair shares
        default: return h & 0xFFu;
    }
}
#endif

// Each warp: `reps` 16 x 8 x 32 tiles of distribution `kind`, both sums of each; mismatches counted (bitwise).
// kind 0: byte (tile, lane, position) cycles through every value as the tile goes, A's and B's at different rates.
__global__ void sums_kernel(int kind, int reps, unsigned long long* bad, unsigned long long* fallback_bad,
                            unsigned long long* sums) {
#if CRIT_KCORE_HAS_FP8_MMA
    const int lane = threadIdx.x & 31;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned long long n_bad = 0, n_fb = 0;
    for(int rep = 0; rep < reps; ++rep) {
        const uint32_t tile = warp * uint32_t(reps) + uint32_t(rep);
        uint32_t w[6];
        for(int r = 0; r < 6; ++r) {
            uint32_t word = 0;
            for(int j = 0; j < 4; ++j) {
                const uint32_t pos = uint32_t(lane * 24 + r * 4 + j);
                uint32_t c;
                if(kind == 0) {
                    c = r < 4 ? (tile + 7u * uint32_t(lane) + 13u * pos) & 0xFFu
                              : (255u - ((tile >> 8) + 11u * uint32_t(lane) + 5u * pos)) & 0xFFu;
                } else {
                    const uint32_t h = mix(tile * 2654435761u ^ (pos * 40503u) ^ uint32_t(kind) * 0x9E3779B9u);
                    const uint32_t pair = mix(tile ^ (uint32_t(lane & 3) * 977u + uint32_t(j) * 31u + uint32_t(r & 1)));
                    c = code(kind, h, pair);
                    if(kind == 4) c = (r < 4 ? 0x38u : (j & 1) ? 0x38u : 0xB8u) ^ ((h >> 12) & 0x03u);   // +-1 products
                }
                word |= c << (8 * j);
            }
            w[r] = word;
        }
        float plain[4], fast[4], guarded[4];
        bs::block_sum_e4m3(make_uint2(w[0], w[2]), make_uint2(w[1], w[3]), make_uint2(w[4], w[5]), plain);
        bs::block_sum_e4m3_unguarded(w[0], w[1], w[2], w[3], w[4], w[5], fast);
        bs::block_sum_e4m3_fast(w[0], w[1], w[2], w[3], w[4], w[5], guarded);
        for(int i = 0; i < 4; ++i) {
            n_bad += __float_as_uint(plain[i]) != __float_as_uint(fast[i]);
            n_fb += __float_as_uint(plain[i]) != __float_as_uint(guarded[i]);
        }
    }
    atomicAdd(bad, n_bad);
    atomicAdd(fallback_bad, n_fb);
    atomicAdd(sums, 4ull * (unsigned long long)reps);
#else
    (void)kind, (void)reps, (void)bad, (void)fallback_bad, (void)sums;
#endif
}

}  // namespace

int main() {
    const check::Flags f = check::flags();
    if(!f.fp8_mma) {
        std::printf("no E4M3 MMA in this build's device code: skipped\n");
        return check::finish();
    }
    std::printf("CRIT_KCORE_FAST_E4M3_SUM on this target: %s\n", f.block_scaled ? "the block-scaled MMA where listed"
                                                                                : "the plain MMA (no block-scaled MMA)");
    const char* names[] = {"every code byte at every k position", "random codes (NaN and inf included)",
                           "large magnitudes", "subnormals", "near-one cancelling blocks"};
    const int blocks = 680, threads = 256, reps[] = {64, 512, 256, 256, 256};
    for(int kind = 0; kind < 5; ++kind) {
        unsigned long long* d = check::dev<unsigned long long>(3);
        sums_kernel<<<blocks, threads>>>(kind, reps[kind], d, d + 1, d + 2);
        const auto got = check::get(d, 3);
        std::printf("  %s: %llu sums, %llu differ (fast), %llu differ (guarded)\n", names[kind], got[2], got[0], got[1]);
        report((std::string("the fast E4M3 block sum gives the plain one's bits: ") + names[kind]).c_str(),
               (long long)got[0]);
        report((std::string("block_sum_e4m3_fast gives the plain one's bits: ") + names[kind]).c_str(),
               (long long)got[1]);
        cudaFree(d);
    }
    return check::finish();
}
