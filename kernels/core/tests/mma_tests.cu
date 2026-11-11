// crit_kcore/mma.cuh on the device: every mma.sync wrapper the device has, fed fragments packed by the layouts the
// header documents, against the host's product (values chosen so every sum is exact: the results must be equal), and
// the quad transpose. tests/test_core.py builds it for compute_80 (PTX), sm_89-class plain sm_120 and sm_120a.
#include "check.cuh"
#include "crit_kcore/formats.cuh"
#include "crit_kcore/mma.cuh"

#include <cmath>
#include <functional>
#include <random>

namespace kc = crit::kcore;
using check::report;

namespace {

enum Which { kBf16K16, kF16K16, kBf16K8, kS8K16, kS8K16Zero, kS8K32, kE4m3K32, kE4m3K32Zero, kNvf4, kMxf4, kMxf8 };

constexpr float kInitF = 0.5f;
constexpr int kInitI = 3;

__global__ void mma_kernel(int which, const uint32_t* a, const uint32_t* b, const uint32_t* sf, float* cf, int* ci) {
    const int lane = threadIdx.x;
    const uint32_t a0 = a[4 * lane], a1 = a[4 * lane + 1], a2 = a[4 * lane + 2], a3 = a[4 * lane + 3];
    const uint32_t b0 = b[2 * lane], b1 = b[2 * lane + 1], sfa = sf[2 * lane], sfb = sf[2 * lane + 1];
    float c[4] = {kInitF, kInitF, kInitF, kInitF};
    int d[4] = {kInitI, kInitI, kInitI, kInitI};
    switch(which) {
        case kBf16K16: kc::mma_bf16_m16n8k16(c, a0, a1, a2, a3, b0, b1); break;
        case kF16K16: kc::mma_f16_m16n8k16(c, a0, a1, a2, a3, b0, b1); break;
        case kBf16K8: kc::mma_bf16_m16n8k8(c, a0, a1, b0); break;
        case kS8K16: kc::mma_s8_m16n8k16(d, a0, a1, b0); break;
        case kS8K16Zero: kc::mma_s8_m16n8k16_zero(int(a0), int(a1), int(b0), d[0], d[1], d[2], d[3]); break;
        case kS8K32: kc::mma_s8_m16n8k32(d, a0, a1, a2, a3, b0, b1); break;
        case kE4m3K32: kc::mma_e4m3_m16n8k32(c, a0, a1, a2, a3, b0, b1); break;
        case kE4m3K32Zero: kc::mma_e4m3_m16n8k32_zero(c, a0, a1, a2, a3, b0, b1); break;
        case kNvf4: kc::mma_nvf4_m16n8k64(c, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
        case kMxf4: kc::mma_mxf4_m16n8k64(c, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
        case kMxf8: kc::mma_mxf8_e4m3_m16n8k32(c, a0, a1, a2, a3, b0, b1, sfa, sfb); break;
    }
    for(int i = 0; i < 4; ++i) {
        cf[4 * lane + i] = c[i];
        ci[4 * lane + i] = d[i];
    }
}

__global__ void transpose_kernel(const uint4* in, uint32_t* out) {
    const int lane = threadIdx.x;
    uint32_t y[4];
    kc::quad_transpose(in[lane], lane & 3, lane, y);
    for(int s = 0; s < 4; ++s) out[4 * lane + s] = y[s];
}

double e4m3(uint32_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double mag = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (b & 0x80u) ? -mag : mag;
}
double e2m1(uint32_t c) {
    const double t[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return (c & 8u) ? -t[c & 7] : t[c & 7];
}

// One case: K, element bits, the A and B register positions, codes and values, the scale block size (0: none) and
// scale decoding.
struct Case {
    const char* name;
    Which which;
    int K, bits, nA, nB;
    bool integer;
    int block;   // values of K a scale covers (0: unscaled)
    std::function<uint32_t(std::mt19937&)> code;          // a random element code
    std::function<double(uint32_t)> value;                // its value
    std::function<uint32_t(std::mt19937&)> scale_code;    // a random scale byte
    std::function<double(uint32_t)> scale_value;
};

void run(const Case& t, std::mt19937& rng) {
    const int M = 16, N = 8, K = t.K, per = 32 / t.bits;
    std::vector<uint32_t> A(M * K), B(K * N), SA(M * 4, 0), SB(N * 4, 0);
    for(auto& x : A) x = t.code(rng);
    for(auto& x : B) x = t.code(rng);
    const int blocks = t.block ? K / t.block : 0;
    for(int r = 0; r < M; ++r)
        for(int j = 0; j < blocks; ++j) SA[r * 4 + j] = t.scale_code(rng);
    for(int c = 0; c < N; ++c)
        for(int j = 0; j < blocks; ++j) SB[c * 4 + j] = t.scale_code(rng);
    // the documented layouts
    const int kbase_mul = per;   // a lane's first k is per * tig
    const int a_row[4] = {0, 8, 0, 8};
    const int a_k[4] = {0, 0, K / 2, K / 2};
    const int b_k[2] = {0, K / 2};
    std::vector<uint32_t> ra(32 * 4, 0), rb(32 * 2, 0), rs(32 * 2, 0);
    for(int lane = 0; lane < 32; ++lane) {
        const int gid = lane >> 2, tig = lane & 3;
        for(int i = 0; i < t.nA; ++i) {
            uint32_t w = 0;
            for(int e = 0; e < per; ++e)
                w |= A[(gid + a_row[i]) * K + (t.nA == 2 ? 0 : a_k[i]) + kbase_mul * tig + e] << (t.bits * e);
            ra[4 * lane + i] = w;
        }
        for(int i = 0; i < t.nB; ++i) {
            uint32_t w = 0;
            for(int e = 0; e < per; ++e) w |= B[((t.nB == 1 ? 0 : b_k[i]) + kbase_mul * tig + e) * N + gid] << (t.bits * e);
            rb[2 * lane + i] = w;
        }
        const int srow = gid + 8 * (tig & 1);
        for(int j = 0; j < blocks; ++j) {
            rs[2 * lane] |= SA[srow * 4 + j] << (8 * j);
            rs[2 * lane + 1] |= SB[gid * 4 + j] << (8 * j);
        }
    }
    uint32_t *da = check::dev(ra), *db = check::dev(rb), *ds = check::dev(rs);
    float* cf = check::dev<float>(128);
    int* ci = check::dev<int>(128);
    mma_kernel<<<1, 32>>>(t.which, da, db, ds, cf, ci);
    auto F = check::get(cf, 128);
    auto I = check::get(ci, 128);
    const bool zero = t.which == kS8K16Zero || t.which == kE4m3K32Zero;
    long long bad = 0;
    double worst = 0;
    for(int lane = 0; lane < 32; ++lane)
        for(int i = 0; i < 4; ++i) {
            const int r = (lane >> 2) + 8 * (i >> 1), c = 2 * (lane & 3) + (i & 1);
            double ref = zero ? 0.0 : (t.integer ? kInitI : kInitF);
            for(int k = 0; k < K; ++k) {
                double p = t.value(A[r * K + k]) * t.value(B[k * N + c]);
                if(t.block) p *= t.scale_value(SA[r * 4 + k / t.block]) * t.scale_value(SB[c * 4 + k / t.block]);
                ref += p;
            }
            const double got = t.integer ? double(I[4 * lane + i]) : double(F[4 * lane + i]);
            if(got != ref) {
                ++bad;
                worst = std::fmax(worst, std::fabs(got - ref));
            }
        }
    report(t.name, bad, bad ? "largest difference " + std::to_string(worst) : "");
    cudaFree(da), cudaFree(db), cudaFree(ds), cudaFree(cf), cudaFree(ci);
}

}  // namespace

// Never launched: instantiate the wgmma and tcgen05 wrappers so that test_every_arch_builds assembles them for
// sm_90a and sm_100a (no such device runs these tests).
__global__ void kcore_warpgroup_kernel(float* out, int accumulate) {
    __shared__ __align__(1024) uint16_t a[64 * 16], b[256 * 16];
    float d8[4] = {}, d64[32] = {}, d256[128] = {};
    const uint64_t da = kc::wgmma_desc(a, 128, 256, 1), db = kc::wgmma_desc(b, 128, 256, 1);
    kc::wgmma_fence();
    kc::Wgmma<8>::bf16<0, 0>(d8, da, db, accumulate);
    kc::Wgmma<16>::bf16<0, 1>(d64, da, db, accumulate);
    kc::Wgmma<32>::bf16<1, 0>(d64, da, db, accumulate);
    kc::Wgmma<64>::bf16<0, 0>(d64, da, db, accumulate);
    kc::Wgmma<128>::bf16<0, 0>(d256, da, db, accumulate);
    kc::Wgmma<256>::bf16<0, 0>(d256, da, db, accumulate);
    kc::wgmma_commit();
    kc::wgmma_wait<0>();
    out[threadIdx.x] = d8[0] + d64[31] + d256[127];
}
__global__ void kcore_tcgen05_kernel(float* out, int accumulate) {
    __shared__ __align__(1024) uint16_t a[128 * 16], b[128 * 16];
    __shared__ uint32_t taddr;
    __shared__ uint64_t bar;
    if(threadIdx.x < 32) kc::tcgen05_alloc(&taddr, 128);
    kc::tcgen05_fence_after();
    if(threadIdx.x == 0) {
        kc::tcgen05_mma_f16(taddr, kc::umma_desc(a, 128, 1024, 2), kc::umma_desc(b, 128, 1024, 2),
                            kc::umma_idesc_f16(128, 128), accumulate);
        kc::tcgen05_commit(&bar);
    }
    uint32_t r[8];
    kc::tcgen05_ld_32x32b_x8(taddr, r);
    kc::tcgen05_wait_ld();
    kc::tcgen05_fence_before();
    if(threadIdx.x < 32) {
        kc::tcgen05_dealloc(taddr, 128);
        kc::tcgen05_relinquish();
    }
    float s = 0.0f;
    for(int i = 0; i < 8; ++i) s += __uint_as_float(r[i]);
    out[threadIdx.x] = s;
}

int main() {
    const check::Flags f = check::flags();
    std::mt19937 rng(7);
    auto small_int16 = [](std::mt19937& g, bool bf) {
        const float v = float(int(g() % 17) - 8);
        if(bf) {
            const __nv_bfloat16 h = __float2bfloat16_rn(v);
            return uint32_t(*reinterpret_cast<const unsigned short*>(&h));
        }
        const __half h = __float2half_rn(v);
        return uint32_t(*reinterpret_cast<const unsigned short*>(&h));
    };
    auto bf16_value = [](uint32_t c) { return double(check::from_bits(c << 16)); };
    auto f16_value = [](uint32_t c) { __half_raw r; r.x = (unsigned short)c; return double(__half2float(__half(r))); };
    auto s8_code = [](std::mt19937& g) { return uint32_t(g() & 0xffu); };
    auto s8_value = [](uint32_t c) { return double(int8_t(c)); };
    // E4M3 codes of magnitude 2^-2 .. 2^2 (exponent fields 5 .. 9), either sign: every sum below is exact.
    auto e4m3_code = [](std::mt19937& g) { return uint32_t(((5 + g() % 5) << 3) | (g() & 7u) | ((g() & 1u) << 7)); };
    auto e2m1_code = [](std::mt19937& g) { return uint32_t(g() & 15u); };
    auto ue4m3_scale = [](std::mt19937& g) { return uint32_t(0x30 + g() % 17); };   // 0.5 .. 2
    auto ue8m0_scale = [](std::mt19937& g) { return uint32_t(125 + g() % 5); };    // 2^-2 .. 2^2
    auto ue8m0_value = [](uint32_t e) { return std::ldexp(1.0, int(e) - 127); };
    auto none = [](std::mt19937&) { return 0u; };
    auto unit = [](uint32_t) { return 1.0; };

    std::vector<Case> cases = {
        {"mma_bf16_m16n8k16", kBf16K16, 16, 16, 4, 2, false, 0, [&](std::mt19937& g) { return small_int16(g, true); },
         bf16_value, none, unit},
        {"mma_f16_m16n8k16", kF16K16, 16, 16, 4, 2, false, 0, [&](std::mt19937& g) { return small_int16(g, false); },
         f16_value, none, unit},
        {"mma_bf16_m16n8k8", kBf16K8, 8, 16, 2, 1, false, 0, [&](std::mt19937& g) { return small_int16(g, true); },
         bf16_value, none, unit},
        {"mma_s8_m16n8k16", kS8K16, 16, 8, 2, 1, true, 0, s8_code, s8_value, none, unit},
        {"mma_s8_m16n8k16_zero", kS8K16Zero, 16, 8, 2, 1, true, 0, s8_code, s8_value, none, unit},
        {"mma_s8_m16n8k32", kS8K32, 32, 8, 4, 2, true, 0, s8_code, s8_value, none, unit},
    };
    if(f.fp8_mma) {
        cases.push_back({"mma_e4m3_m16n8k32", kE4m3K32, 32, 8, 4, 2, false, 0, e4m3_code, e4m3, none, unit});
        cases.push_back({"mma_e4m3_m16n8k32_zero", kE4m3K32Zero, 32, 8, 4, 2, false, 0, e4m3_code, e4m3, none, unit});
    }
    if(f.block_scaled) {
        cases.push_back({"mma_nvf4_m16n8k64 (UE4M3 per 16)", kNvf4, 64, 4, 4, 2, false, 16, e2m1_code, e2m1,
                         ue4m3_scale, e4m3});
        cases.push_back({"mma_mxf4_m16n8k64 (UE8M0 per 32)", kMxf4, 64, 4, 4, 2, false, 32, e2m1_code, e2m1,
                         ue8m0_scale, ue8m0_value});
        cases.push_back({"mma_mxf8_e4m3_m16n8k32 (UE8M0 per 32)", kMxf8, 32, 8, 4, 2, false, 32, e4m3_code, e4m3,
                         ue8m0_scale, ue8m0_value});
    }
    for(int rep = 0; rep < 4; ++rep)
        for(const Case& t : cases) run(t, rng);

    {   // quad transpose
        std::vector<uint4> in(32);
        for(int l = 0; l < 32; ++l) in[l] = make_uint4(100 * l, 100 * l + 1, 100 * l + 2, 100 * l + 3);
        uint4* di = check::dev(in);
        uint32_t* dout = check::dev<uint32_t>(128);
        transpose_kernel<<<1, 32>>>(di, dout);
        auto out = check::get(dout, 128);
        long long bad = 0;
        for(int l = 0; l < 32; ++l)
            for(int s = 0; s < 4; ++s) bad += out[4 * l + s] != uint32_t(100 * ((l & ~3) | s) + (l & 3));
        report("quad_transpose", bad);
    }
    return check::finish();
}
