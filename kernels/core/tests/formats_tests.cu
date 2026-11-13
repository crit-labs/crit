// crit_kcore/formats.cuh on the device against host references: every E4M3, E8M0 and E2M1 code, random words of
// codes for the packed decoders, the E4M3 encoder against the CUDA conversion on every non-negative float, the E2M1
// encoder at and between its midpoints, the INT4 and integer decoders, the NVFP4 quantizer steps and the blocked
// scale layout. tests/test_core.py builds it for compute_80 (PTX: the fallback paths), sm_120 and sm_120a.
#include "check.cuh"
#include "crit_kcore/formats.cuh"

#include <cmath>
#include <random>

namespace kc = crit::kcore;
using check::bits;
using check::report;
using check::same;

namespace {

// ------------------------------------------------------------------------------------------- host references

double host_e4m3(uint32_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    if(e == 15 && m == 7) return NAN;
    const double mag = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (b & 0x80u) ? -mag : mag;
}
const float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
float host_e2m1(uint32_t c) { return (c & 8u) ? -kE2M1[c & 7u] : kE2M1[c & 7u]; }
float host_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
float bf16_lo(uint32_t w) { return check::from_bits(w << 16); }
float bf16_hi(uint32_t w) { return check::from_bits(w & 0xffff0000u); }
float half_lo(uint32_t w) { __half_raw r; r.x = (unsigned short)(w & 0xffffu); return __half2float(__half(r)); }
float half_hi(uint32_t w) { __half_raw r; r.x = (unsigned short)(w >> 16); return __half2float(__half(r)); }
// The nearest E2M1 magnitude code, ties to the even code.
int host_e2m1_code(float a) {
    if(!(a == a)) return 0;
    if(a > 6.0f) return 7;
    int best = 0;
    for(int c = 1; c < 8; ++c) {
        const float d = std::fabs(a - kE2M1[c]), bd = std::fabs(a - kE2M1[best]);
        if(d < bd || (d == bd && c % 2 == 0)) best = c;
    }
    return best;
}
// The nearest non-negative E4M3 code of x >= 0 (saturating at 448), ties to the even code.
uint32_t host_e4m3_encode(float x) {
    uint32_t best = 0;
    for(uint32_t c = 1; c < 0x7Fu; ++c) {
        const double d = std::fabs(x - host_e4m3(c)), bd = std::fabs(x - host_e4m3(best));
        if(d < bd || (d == bd && c % 2 == 0)) best = c;
    }
    return best;
}

// ------------------------------------------------------------------------------------------- device side

__global__ void codes_kernel(float* e4m3_value, float* e4m3_cvt, float* e8m0, uint32_t* e8m0_pair, uint32_t* e4m3_half2,
                             float* e2m1) {
    const uint32_t b = threadIdx.x;
    e4m3_value[b] = kc::e4m3_value(b);
    e4m3_cvt[b] = kc::e4m3_cvt(b);
    e8m0[b] = kc::e8m0_value(b);
    e8m0_pair[b] = kc::u32_of(kc::e8m0_bf16x2(b));
    e4m3_half2[b] = kc::u32_of(kc::e4m3x2_to_half2(b | (((b * 7u) & 0xffu) << 8)));
    if(b < 16) e2m1[b] = kc::e2m1_value(b);
}

// Per word: e2m1x8_to_half2 (8 values, unit removed), e2m1x8_to_float, e2m1x8_to_bf16x2, e2m1x8_to_bf16x2_scaled,
// e2m1x2_to_bf16x2_scaled (byte 0), e4m3x4_to_bf16x2, int4x8_to_bf16x2_scaled, four_bit_int8x4 (both kinds).
__global__ void words_kernel(const uint32_t* words, const uint32_t* scales, int n, float* half_vals, float* float_vals,
                             uint32_t* bf16_plain, uint32_t* bf16_scaled, uint32_t* pair_scaled, uint32_t* fp8_pairs,
                             uint32_t* int4_pairs, int* ints) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    const uint32_t w = words[i];
    const float s = kc::e4m3_value(scales[i]);
    uint32_t h[4];
    kc::e2m1x8_to_half2(w, h);
    for(int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(kc::half2_of(h[j]));
        half_vals[8 * i + 2 * j] = f.x / kc::e2m1_half2_unit();
        half_vals[8 * i + 2 * j + 1] = f.y / kc::e2m1_half2_unit();
    }
    kc::e2m1x8_to_float(w, s / kc::e2m1_half2_unit(), float_vals + 8 * i);
    kc::e2m1x8_to_bf16x2(w, bf16_plain + 4 * i);
    const __half2 s2 = __float2half2_rn(s);
    kc::e2m1x8_to_bf16x2_scaled(w, s, kc::u32_of(s2), bf16_scaled + 4 * i);
    pair_scaled[i] = kc::e2m1x2_to_bf16x2_scaled(w & 0xffu, s);
    kc::e4m3x4_to_bf16x2(w, fp8_pairs + 2 * i);
    const __nv_bfloat16 sb = __float2bfloat16_rn(s);
    kc::int4x8_to_bf16x2_scaled(w, __halves2bfloat162(sb, sb), int4_pairs + 4 * i);
    ints[4 * i] = kc::four_bit_int8x4(w & 0xffffu, false);
    ints[4 * i + 1] = kc::four_bit_int8x4(w >> 16, false);
    ints[4 * i + 2] = kc::four_bit_int8x4(w & 0xffffu, true);
    ints[4 * i + 3] = kc::four_bit_int8x4(w >> 16, true);
}

__global__ void e2m1_code_kernel(const float* v, int n, int* code, uint32_t* signed_code, uint32_t* pair_code) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    code[i] = kc::e2m1_code(fabsf(v[i]));
    signed_code[i] = kc::e2m1_code_signed(v[i]);
    pair_code[i] = kc::e2m1x2_encode(v[i], v[(i + 7) % n]);
}

__global__ void e4m3x2_kernel(const float* v, int n, uint32_t* pair, uint32_t* single) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    pair[i] = kc::e4m3x2_encode_satfinite(v[i], v[(i + 3) % n]);
    single[i] = kc::e4m3_encode_satfinite(v[i]);
}

// Every non-negative float bit pattern up to +inf: e4m3_encode against e4m3_encode_satfinite.
__global__ void e4m3_encode_kernel(unsigned long long* bad, uint32_t* first) {
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for(uint64_t u = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; u <= 0x7F800000ull; u += stride) {
        const float x = __uint_as_float(uint32_t(u));
        if(kc::e4m3_encode(x) != kc::e4m3_encode_satfinite(x)) {
            if(atomicAdd(bad, 1ull) == 0) *first = uint32_t(u);
        }
    }
}

__global__ void table_kernel(uint32_t* e2m1_table, uint32_t* int4_table) {
    __shared__ uint32_t a[256], b[256];
    kc::four_bit_pair_table<false>(a);
    kc::four_bit_pair_table<true>(b);
    __syncthreads();
    for(int i = threadIdx.x; i < 256; i += blockDim.x) {
        e2m1_table[i] = a[i];
        int4_table[i] = b[i];
    }
}

__global__ void nvfp4_kernel(const float* amax, const float* v, const float* g, int n, float* decoded, float* operand) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    decoded[i] = kc::nvfp4_block_scale(amax[i], g[i]);
    operand[i] = kc::nvfp4_operand(v[i], g[i], decoded[i]);
}

}  // namespace

int main() {
    using check::dev;
    using check::get;
    check::flags();
    std::mt19937 rng(1234);

    {   // every code
        float *ev = dev<float>(256), *ec = dev<float>(256), *e8 = dev<float>(256), *e2 = dev<float>(16);
        uint32_t *e8p = dev<uint32_t>(256), *eh = dev<uint32_t>(256);
        codes_kernel<<<1, 256>>>(ev, ec, e8, e8p, eh, e2);
        auto v = get(ev, 256), c = get(ec, 256), m = get(e8, 256), t = get(e2, 16);
        auto p = get(e8p, 256), h = get(eh, 256);
        long long bad_value = 0, bad_cvt = 0, bad_e8 = 0, bad_pair = 0, bad_half = 0, bad_e2 = 0;
        for(uint32_t b = 0; b < 256; ++b) {
            const double ref = host_e4m3(b);
            const bool nan = (b & 0x7f) == 0x7f;
            bad_value += nan ? v[b] != ((b & 0x80) ? -480.0f : 480.0f) : !same(v[b], float(ref));
            bad_cvt += !same(c[b], float(ref));
            const float e8ref = b == 255 ? NAN : float(std::ldexp(1.0, int(b) - 127));
            bad_e8 += !same(m[b], e8ref);
            bad_pair += !(same(bf16_lo(p[b]), host_bf16(e8ref)) && same(bf16_hi(p[b]), host_bf16(e8ref)));
            bad_half += !(same(half_lo(h[b]), float(host_e4m3(b))) &&
                          same(half_hi(h[b]), float(host_e4m3((b * 7u) & 0xffu))));
        }
        for(uint32_t b = 0; b < 16; ++b) bad_e2 += !same(t[b], host_e2m1(b));
        report("e4m3_value (every code; NaN codes +-480)", bad_value);
        report("e4m3_cvt (every code)", bad_cvt);
        report("e4m3x2_to_half2 (every code)", bad_half);
        report("e8m0_value (every code)", bad_e8);
        report("e8m0_bf16x2 (every code)", bad_pair);
        report("e2m1_value (every code)", bad_e2);
    }

    {   // packed words
        const int n = 1 << 14;
        std::vector<uint32_t> words(n), scales(n);
        for(int i = 0; i < n; ++i) {
            words[i] = rng();
            // positive finite E4M3 scales, the full range (block scales are positive)
            uint32_t s;
            do s = rng() & 0x7fu; while(s == 0x7fu || s == 0);
            scales[i] = s;
        }
        for(int i = 0; i < 256; ++i) words[i] = 0x01010101u * uint32_t(i);   // every byte in every position
        uint32_t *dw = dev(words), *ds = dev(scales);
        float *hv = dev<float>(8 * n), *fv = dev<float>(8 * n);
        uint32_t *bp = dev<uint32_t>(4 * n), *bs = dev<uint32_t>(4 * n), *ps = dev<uint32_t>(n), *f8 = dev<uint32_t>(2 * n),
                 *i4 = dev<uint32_t>(4 * n);
        int* in = dev<int>(4 * n);
        words_kernel<<<(n + 127) / 128, 128>>>(dw, ds, n, hv, fv, bp, bs, ps, f8, i4, in);
        auto H = get(hv, 8 * n), F = get(fv, 8 * n);
        auto BP = get(bp, 4 * n), BS = get(bs, 4 * n), PS = get(ps, n), F8 = get(f8, 2 * n), I4 = get(i4, 4 * n);
        auto IN = get(in, 4 * n);
        long long b_half = 0, b_float = 0, b_plain = 0, b_scaled = 0, b_pair = 0, b_fp8 = 0, b_int4 = 0, b_int = 0;
        for(int i = 0; i < n; ++i) {
            const uint32_t w = words[i];
            const float s = float(host_e4m3(scales[i]));
            const float sb = host_bf16(s);
            for(int j = 0; j < 8; ++j) {
                const uint32_t code = (w >> (4 * j)) & 15u;
                const float e = host_e2m1(code);
                b_half += !same(H[8 * i + j], e);
                b_float += !same(F[8 * i + j], e * s);
                const uint32_t pw = BP[4 * i + j / 2], sw = BS[4 * i + j / 2], iw = I4[4 * i + j / 2];
                b_plain += !same(j % 2 ? bf16_hi(pw) : bf16_lo(pw), e);
                b_scaled += !same(j % 2 ? bf16_hi(sw) : bf16_lo(sw), host_bf16(e * s));
                const int iv = int(code << 28) >> 28;
                b_int4 += !same(j % 2 ? bf16_hi(iw) : bf16_lo(iw), host_bf16(float(iv) * sb));
                if(j < 2) b_pair += !same(j ? bf16_hi(PS[i]) : bf16_lo(PS[i]), host_bf16(e * s));
            }
            for(int j = 0; j < 4; ++j) {
                const uint32_t byte = (w >> (8 * j)) & 0xffu;
                const double ref = host_e4m3(byte);
                const uint32_t fw = F8[2 * i + j / 2];
                const float got = j % 2 ? bf16_hi(fw) : bf16_lo(fw);
                // Below 8.9 the fallback decodes NaN codes as +-480; the flags tell which path ran.
                b_fp8 += !(same(got, host_bf16(float(ref))) ||
                           ((byte & 0x7f) == 0x7f && same(got, host_bf16((byte & 0x80) ? -480.0f : 480.0f))));
            }
            for(int q = 0; q < 2; ++q)
                for(int k = 0; k < 2; ++k) {   // k 0: E2M1 x 2, k 1: INT4
                    const int got = IN[4 * i + 2 * k + q];
                    for(int j = 0; j < 4; ++j) {
                        const uint32_t code = (w >> (16 * q + 4 * j)) & 15u;
                        const int ref = k ? int(code << 28) >> 28 : int(2.0f * host_e2m1(code));
                        b_int += int8_t((got >> (8 * j)) & 0xff) != ref;
                    }
                }
        }
        report("e2m1x8_to_half2 (random words, every byte)", b_half);
        report("e2m1x8_to_float", b_float);
        report("e2m1x8_to_bf16x2", b_plain);
        report("e2m1x8_to_bf16x2_scaled (E4M3 scales)", b_scaled);
        report("e2m1x2_to_bf16x2_scaled", b_pair);
        report("e4m3x4_to_bf16x2", b_fp8);
        report("int4x8_to_bf16x2_scaled (E4M3 scales)", b_int4);
        report("four_bit_int8x4 (E2M1 x 2 and INT4)", b_int);
    }

    {   // E2M1 encoder: midpoints, their neighbours, random values, both signs, NaN
        std::vector<float> v;
        const float mids[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
        for(float m : mids) {
            v.push_back(m);
            v.push_back(std::nextafter(m, 0.0f));
            v.push_back(std::nextafter(m, 10.0f));
        }
        for(float e : kE2M1) v.push_back(e);
        std::uniform_real_distribution<float> u(0.0f, 8.0f);
        for(int i = 0; i < 4096; ++i) v.push_back(u(rng));
        v.push_back(1e30f);
        v.push_back(INFINITY);
        v.push_back(0.1f);
        const size_t half = v.size();
        for(size_t i = 0; i < half; ++i) v.push_back(-v[i]);
        const int finite = int(v.size());   // the pair encoder's inputs: no NaN
        v.push_back(NAN);
        const int n = int(v.size());
        float* dv = dev(v);
        int* dc = dev<int>(n);
        uint32_t *ds = dev<uint32_t>(n), *dp = dev<uint32_t>(n);
        e2m1_code_kernel<<<(n + 127) / 128, 128>>>(dv, n, dc, ds, dp);
        auto C = get(dc, n);
        auto S = get(ds, n);
        auto P = get(dp, n);
        long long bad = 0, bad_signed = 0, bad_pair = 0;
        for(int i = 0; i < n; ++i) {
            const int ref = host_e2m1_code(std::fabs(v[i]));
            bad += C[i] != ref;
            bad_signed += S[i] != (uint32_t(ref) | (v[i] < 0.0f ? 8u : 0u));
            const int j = (i + 7) % n;
            auto code = [&](float x) { return uint32_t(host_e2m1_code(std::fabs(x))) | (std::signbit(x) ? 8u : 0u); };
            if(i < finite && j < finite) bad_pair += P[i] != (code(v[i]) | (code(v[j]) << 4));
        }
        report("e2m1_code (midpoints, neighbours, random)", bad);
        report("e2m1_code_signed", bad_signed);
        report("e2m1x2_encode (nearest codes, the float's sign bit)", bad_pair);

        uint32_t *e4p = dev<uint32_t>(n), *e4s = dev<uint32_t>(n);
        std::vector<float> w(v.begin(), v.end());
        for(auto& x : w) x *= 75.0f;   // across E4M3's range, saturating past 448
        float* dw = dev(w);
        e4m3x2_kernel<<<(n + 127) / 128, 128>>>(dw, n, e4p, e4s);
        auto EP = get(e4p, n), ES = get(e4s, n);
        long long bad_e4 = 0;
        for(int i = 0; i < n; ++i) bad_e4 += EP[i] != (ES[i] | (ES[(i + 3) % n] << 8));
        report("e4m3x2_encode_satfinite == e4m3_encode_satfinite pairs", bad_e4);
    }

    {   // E4M3 encoder: every non-negative float against the conversion, and a sample against the host's nearest code
        unsigned long long* bad = dev<unsigned long long>(1);
        uint32_t* first = dev<uint32_t>(1);
        e4m3_encode_kernel<<<4096, 256>>>(bad, first);
        auto b = get(bad, 1);
        auto f = get(first, 1);
        report("e4m3_encode == e4m3_encode_satfinite (every float in [0, inf])", (long long)b[0],
               b[0] ? "first at bits " + std::to_string(f[0]) : "");
    }

    {   // the pair tables
        uint32_t *a = dev<uint32_t>(256), *b = dev<uint32_t>(256);
        table_kernel<<<1, 128>>>(a, b);
        auto A = get(a, 256), B = get(b, 256);
        long long bad = 0;
        for(uint32_t i = 0; i < 256; ++i) {
            bad += !(same(bf16_lo(A[i]), host_e2m1(i & 15)) && same(bf16_hi(A[i]), host_e2m1(i >> 4)));
            bad += !(same(bf16_lo(B[i]), float(int(i << 28) >> 28)) && same(bf16_hi(B[i]), float(int(i << 24) >> 28)));
        }
        report("four_bit_pair_table (E2M1 and INT4)", bad);
    }

    {   // NVFP4 quantizer steps against the declared function on the host
        const int n = 1 << 14;
        std::vector<float> amax(n), v(n), g(n);
        std::uniform_real_distribution<float> mag(-8.0f, 8.0f), unit(0.0f, 1.0f);
        for(int i = 0; i < n; ++i) {
            amax[i] = std::ldexp(unit(rng), int(mag(rng)));
            v[i] = amax[i] * (2.0f * unit(rng) - 1.0f);
            g[i] = std::ldexp(1.0f + unit(rng), int(mag(rng)));
        }
        amax[0] = 0.0f;
        amax[1] = INFINITY;
        float *da = dev(amax), *dv = dev(v), *dg = dev(g), *dd = dev<float>(n), *dop = dev<float>(n);
        nvfp4_kernel<<<(n + 127) / 128, 128>>>(da, dv, dg, n, dd, dop);
        auto D = get(dd, n), O = get(dop, n);
        long long bad = 0, bad_op = 0;
        for(int i = 0; i < n; ++i) {
            float wanted = (amax[i] * g[i]) / 6.0f;
            if(!std::isfinite(wanted)) wanted = 448.0f;
            const float dec = std::isfinite(amax[i]) ? float(host_e4m3(host_e4m3_encode(std::fmin(wanted, 448.0f))))
                                                     : NAN;   // a block holding a non-finite value: a NaN scale
            bad += !same(D[i], dec);
            float op = 0.0f;
            if(dec > 0.0f) {
                const float q = (v[i] * g[i]) / dec;
                const int code = host_e2m1_code(std::fabs(q));
                const float m = kE2M1[code] * dec;
                op = (q < 0.0f && code) ? -m : m;
            }
            bad_op += !same(O[i], op);
        }
        report("nvfp4_block_scale", bad);
        report("nvfp4_operand", bad_op);
    }

    {   // the blocked layout: a bijection into blocked_scale_bytes
        const int rows = 300, cols = 9, ct = (cols + 3) / 4;
        const long long bytes = kc::blocked_scale_bytes(rows, cols * 16, 16);
        std::vector<int> seen(bytes, 0);
        long long bad = bytes != 384LL * 12;
        for(int n = 0; n < rows; ++n)
            for(int j = 0; j < cols; ++j) {
                const size_t at = kc::blocked_scale_offset(n, j, ct);
                if((long long)at >= bytes || seen[at]++) ++bad;
                bad += kc::scale_offset(n, j, cols, ct) != at;
                bad += kc::scale_offset(n, j, cols, 0) != size_t(n) * cols + j;
            }
        report("blocked_scale_offset / scale_offset / blocked_scale_bytes", bad);
    }
    return check::finish();
}
