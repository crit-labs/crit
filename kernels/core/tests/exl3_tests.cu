// crit_kcore/exl3.cuh on the device against host references: every 16-bit state of each codebook (integer arithmetic,
// one FP16 rounding computed on the host in double and rounded once), the bitstream windows of every position at
// every bitrate (whole and half) from random tiles (a host bit reader over the stream, most significant bit first),
// and the fragment order's rows and columns, a permutation of the 16 x 16 tile. tests/test_core.py builds it for
// compute_80 (PTX), sm_120 and sm_120a.
#include "check.cuh"
#include "crit_kcore/exl3.cuh"

#include <cmath>
#include <random>
#include <set>

namespace kc = crit::kcore;
using check::dev;
using check::get;
using check::report;
using check::same;

namespace {

float host_half(uint32_t b) { __half_raw r; r.x = static_cast<unsigned short>(b); return __half2float(__half(r)); }
float host_round_half(double v) { return __half2float(__double2half(v)); }   // one rounding to nearest even

float host_value(int cb, uint32_t s) {
    if(cb == 2) {
        const uint32_t x = s * 0x83DCD12Du;
        const uint32_t sum = (x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24) + 0x6400u;
        return host_round_half(double(host_half(sum & 0xffffu)) * host_half(0x1eee) + host_half(0xc931));
    }
    uint32_t x = cb == 0 ? s * 89226354u + 64248484u : s * 0xCBAC1FEDu;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return host_round_half(double(host_half(x & 0xffffu)) + host_half(x >> 16));
}

// The 16 bits of the circular stream (bit i of word i / 32 at 31 - i % 32) ending at bit `end`.
uint32_t host_state(const std::vector<uint32_t>& words, long long end) {
    const long long total = 32LL * words.size();
    uint32_t v = 0;
    for(int b = 16; b >= 1; --b) {
        const long long i = ((end - b) % total + total) % total;
        v = (v << 1) | ((words[i / 32] >> (31 - i % 32)) & 1u);
    }
    return v;
}

__global__ void value_kernel(int cb, float* out) {
    const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= 65536u) return;
    out[s] = cb == 0 ? kc::exl3_value<0>(s) : cb == 1 ? kc::exl3_value<1>(s) : kc::exl3_value<2>(s);
}

__global__ void state_kernel(const uint32_t* words, int count, int bits, bool half_bit, uint32_t* out) {
    const int p = threadIdx.x;
    out[p] = kc::exl3_state(words, count, bits, half_bit, p);
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    for(int cb = 0; cb < 3; ++cb) {   // every state of each codebook
        float* d = dev<float>(65536);
        value_kernel<<<256, 256>>>(cb, d);
        const auto got = get(d, 65536);
        long long bad = 0;
        for(uint32_t s = 0; s < 65536u; ++s) bad += !same(got[s], host_value(cb, s));
        report(cb == 0 ? "exl3_value 3INST, every state" : cb == 1 ? "exl3_value MCG, every state"
                                                                   : "exl3_value MUL1, every state", bad);
    }
    {   // the windows of every position at every bitrate
        long long bad = 0;
        std::uniform_int_distribution<uint32_t> word(0, 0xFFFFFFFFu);
        for(int bits = 1; bits <= 8; ++bits)
            for(int half = 0; half < 2; ++half) {
                if(half && bits > 3) continue;
                const int count = kc::exl3_words(bits, half);
                std::vector<uint32_t> words(count);
                for(auto& w : words) w = word(rng);
                uint32_t* d = dev<uint32_t>(256);
                state_kernel<<<1, 256>>>(dev(words), count, bits, half, d);
                const auto got = get(d, 256);
                for(int p = 0; p < 256; ++p) {
                    const long long end = half ? (p / 2) * (2LL * bits + 1) + ((p & 1) ? 2 * bits + 1 : bits)
                                               : (p + 1LL) * bits;
                    bad += got[p] != host_state(words, end);
                }
            }
        report("exl3_state, every position at bitrates 1 to 8 and 1.5 to 3.5", bad);
    }
    {   // the fragment order fills the tile once
        std::set<int> seen;
        for(int lane = 0; lane < 32; ++lane)
            for(int j = 0; j < 8; ++j) seen.insert(16 * kc::exl3_row(lane, j) + kc::exl3_col(lane, j));
        report("exl3_row / exl3_col, a permutation of the tile", 256 - static_cast<long long>(seen.size()));
    }
    return check::finish();
}
