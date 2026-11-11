// crit_kcore/formats.cuh's INT4 blocks with a zero point on the device against host references: every code, zero and
// a spread of FP16 scales through int4_zp_value, int4x8_zp_to_float and int4x8_zp_to_bf16x2_split (hi + lo the exact
// value, hi the value rounded to BF16), AWQ's nibble order (awq_code, awq_to_sequential against the pack order of AWQ's
// checkpoints), GPTQ's sequential nibbles, zero offset and act-order groups. tests/test_core.py builds it for
// compute_80 (PTX), sm_120 and sm_120a.
#include "check.cuh"
#include "crit_kcore/formats.cuh"

#include <algorithm>
#include <cmath>
#include <random>

namespace kc = crit::kcore;
using check::dev;
using check::get;
using check::report;
using check::same;

namespace {

float host_half(uint16_t b) { __half_raw r; r.x = b; return __half2float(__half(r)); }
float host_bf16_lo(uint32_t w) { return check::from_bits(w << 16); }
float host_bf16_hi(uint32_t w) { return check::from_bits(w & 0xffff0000u); }
float host_round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// AWQ's packer: nibble p of a word holds element order[p] of 8 consecutive values along N.
constexpr int kAwqPack[8] = {0, 2, 4, 6, 1, 3, 5, 7};
uint32_t host_awq_pack(const uint32_t* v) {
    uint32_t w = 0;
    for(int p = 0; p < 8; ++p) w |= (v[kAwqPack[p]] & 15u) << (4 * p);
    return w;
}

// Per case i: word words[i], zero zeros[i] (0..16), scale bits scales[i]: the scalar value of each code, the eight
// values, the split pairs.
__global__ void zp_kernel(const uint32_t* words, const int* zeros, const uint32_t* scales, int n, float* scalar,
                          float* eight, uint32_t* hi, uint32_t* lo, float* scale_values) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    const float s = kc::f16_value(scales[i]);
    scale_values[i] = s;
    for(int j = 0; j < 8; ++j) scalar[8 * i + j] = kc::int4_zp_value(kc::nibble(words[i], j), zeros[i], s);
    kc::int4x8_zp_to_float(words[i], zeros[i], s, eight + 8 * i);
    kc::int4x8_zp_to_bf16x2_split(words[i], zeros[i], s, hi + 4 * i, lo + 4 * i);
}

__global__ void awq_kernel(const uint32_t* words, int n, uint32_t* codes, uint32_t* sequential) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    for(int j = 0; j < 8; ++j) codes[8 * i + j] = kc::awq_code(words[i], j);
    sequential[i] = kc::awq_to_sequential(words[i]);
}

__global__ void gptq_kernel(const uint32_t* words, const int* g_idx, int n, int group, int* zeros, int* groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    zeros[2 * i] = kc::gptq_zero(kc::nibble(words[i], i & 7), 1);
    zeros[2 * i + 1] = kc::gptq_zero(kc::nibble(words[i], i & 7), 0);
    groups[2 * i] = kc::gptq_group(g_idx, i, group);
    groups[2 * i + 1] = kc::gptq_group(nullptr, i, group);
}

}  // namespace

int main() {
    check::flags();
    std::mt19937 rng(7);

    {   // every code with every zero 0..16, at FP16 scales across the range (normal, subnormal, the largest)
        std::vector<uint16_t> pool = {0x3C00, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0xBC00, 0x2E66, 0x1234, 0x8001};
        std::uniform_int_distribution<int> any(0, 0xFFFF);
        while(pool.size() < 400) {
            const uint16_t b = uint16_t(any(rng));
            if(((b >> 10) & 31) != 31) pool.push_back(b);   // finite scales
        }
        std::vector<uint32_t> words, scales;
        std::vector<int> zeros;
        for(uint16_t s : pool)
            for(int z = 0; z <= 16; ++z)
                for(int base = 0; base < 16; base += 8) {   // two words cover codes 0..15
                    uint32_t w = 0;
                    for(int j = 0; j < 8; ++j) w |= uint32_t(base + j) << (4 * j);
                    words.push_back(w);
                    zeros.push_back(z);
                    scales.push_back(s);
                }
        std::uniform_int_distribution<uint32_t> word(0, 0xFFFFFFFFu);
        for(int i = 0; i < 1 << 14; ++i) {
            words.push_back(word(rng));
            zeros.push_back(int(word(rng) % 17));
            scales.push_back(pool[word(rng) % pool.size()]);
        }
        const int n = int(words.size());
        float *ds = dev<float>(8 * n), *de = dev<float>(8 * n), *dsv = dev<float>(n);
        uint32_t *dh = dev<uint32_t>(4 * n), *dl = dev<uint32_t>(4 * n);
        zp_kernel<<<(n + 127) / 128, 128>>>(dev(words), dev(zeros), dev(scales), n, ds, de, dh, dl, dsv);
        auto S = get(ds, 8 * n), E = get(de, 8 * n), SV = get(dsv, n);
        auto H = get(dh, 4 * n), L = get(dl, 4 * n);
        long long bad_scale = 0, bad_scalar = 0, bad_eight = 0, bad_split = 0, bad_hi = 0;
        for(int i = 0; i < n; ++i) {
            const float s = host_half(uint16_t(scales[i]));
            bad_scale += !same(SV[i], s);
            for(int j = 0; j < 8; ++j) {
                const int code = int((words[i] >> (4 * j)) & 15u);
                const double exact = double(code - zeros[i]) * double(s);
                const float want = float(exact);
                bad_scalar += !(double(want) == exact && same(S[8 * i + j], want));
                bad_eight += !same(E[8 * i + j], want);
                const uint32_t h = H[4 * i + j / 2], l = L[4 * i + j / 2];
                const float hv = (j & 1) ? host_bf16_hi(h) : host_bf16_lo(h);
                const float lv = (j & 1) ? host_bf16_hi(l) : host_bf16_lo(l);
                bad_split += double(hv) + double(lv) != exact;
                bad_hi += !same(hv, host_round_bf16(want));
            }
        }
        report("f16_value", bad_scale);
        report("int4_zp_value (exact)", bad_scalar);
        report("int4x8_zp_to_float", bad_eight);
        report("int4x8_zp_to_bf16x2_split: hi + lo exact", bad_split);
        report("int4x8_zp_to_bf16x2_split: hi the BF16 rounding", bad_hi);
    }

    {   // AWQ's nibble order against its packer
        const int n = 1 << 12;
        std::vector<uint32_t> values(8 * n), words(n);
        std::uniform_int_distribution<uint32_t> nib(0, 15);
        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < 8; ++j) values[8 * i + j] = nib(rng);
            words[i] = host_awq_pack(&values[8 * i]);
        }
        uint32_t *dc = dev<uint32_t>(8 * n), *dq = dev<uint32_t>(n);
        awq_kernel<<<(n + 127) / 128, 128>>>(dev(words), n, dc, dq);
        auto C = get(dc, 8 * n), Q = get(dq, n);
        long long bad = 0, bad_seq = 0, bad_host = 0;
        for(int i = 0; i < n; ++i) {
            uint32_t seq = 0;
            for(int j = 0; j < 8; ++j) {
                bad += C[8 * i + j] != values[8 * i + j];
                bad_host += kc::awq_code(words[i], j) != values[8 * i + j];
                seq |= values[8 * i + j] << (4 * j);
            }
            bad_seq += Q[i] != seq;
        }
        for(int j = 0; j < 8; ++j) bad += kAwqPack[kc::awq_nibble(j)] != j;
        report("awq_nibble / awq_code (device)", bad);
        report("awq_code (host)", bad_host);
        report("awq_to_sequential", bad_seq);
    }

    {   // GPTQ: sequential nibbles, the +1 zero offset, act-order groups from g_idx
        const int n = 1024, group = 128;
        std::vector<uint32_t> words(n);
        std::uniform_int_distribution<uint32_t> word(0, 0xFFFFFFFFu);
        for(auto& w : words) w = word(rng);
        std::vector<int> perm(n), g_idx(n);
        for(int i = 0; i < n; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for(int i = 0; i < n; ++i) g_idx[perm[i]] = i / group;   // GPTQ's act order: groups along the permutation
        int *dz = dev<int>(2 * n), *dg = dev<int>(2 * n);
        gptq_kernel<<<(n + 127) / 128, 128>>>(dev(words), dev(g_idx), n, group, dz, dg);
        auto Z = get(dz, 2 * n), G = get(dg, 2 * n);
        long long bad_z = 0, bad_g = 0;
        for(int i = 0; i < n; ++i) {
            const int stored = int((words[i] >> (4 * (i & 7))) & 15u);
            bad_z += Z[2 * i] != stored + 1 || Z[2 * i + 1] != stored;
            bad_g += G[2 * i] != g_idx[i] || G[2 * i + 1] != i / group;
        }
        report("nibble / gptq_zero", bad_z);
        report("gptq_group (act order and plain)", bad_g);
    }
    return check::finish();
}
