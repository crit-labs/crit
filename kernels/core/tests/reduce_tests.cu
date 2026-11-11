// crit_kcore/reduce.cuh and crit_kcore/epilogue.cuh on the device against host emulations of the same orders: warp
// butterflies, the 16-lane halving tree (shuffled and folded), block sums and scans, selection keys and argmax, the
// online softmax merge, crit's declared exp and SwiGLU (bit for bit), and fragment stores.
#include "check.cuh"
#include "crit_kcore/epilogue.cuh"
#include "crit_kcore/mma.cuh"
#include "crit_kcore/reduce.cuh"

#include <algorithm>
#include <cmath>
#include <random>

namespace kc = crit::kcore;
using check::bits;
using check::report;
using check::same;

namespace {

__global__ void warp_kernel(const float* v, float* sum, float* sum8, float* mx, float* mn, float* tree, float* folded,
                            float* mxf) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float x = v[threadIdx.x];
    sum[threadIdx.x] = kc::warp_sum(x);
    sum8[threadIdx.x] = kc::warp_sum<8>(x);
    mx[threadIdx.x] = kc::warp_max(x);
    mn[threadIdx.x] = kc::warp_min(x);
    mxf[threadIdx.x] = kc::warp_max_from<4>(x);
    const float t = kc::tree_sum_down<16>(x, kc::half_warp_mask());
    if(lane % 16 == 0) tree[2 * w + lane / 16] = t;
    if(lane % 16 == 0) {   // the same tree, one thread folding its half-warp's values in kLaneOrder16
        float part[4];
        for(int s = 0; s < 16; ++s) kc::fold16(s, v[32 * w + lane + kc::kLaneOrder16[s]], part);
        folded[2 * w + lane / 16] = part[0];
    }
}

__global__ void block_kernel(const float* v, const int* iv, float* sum, int* scan, int* scan_max) {
    __shared__ float fb[32];
    __shared__ int ib[32];
    const float s = kc::block_sum(v[threadIdx.x], fb);
    if(threadIdx.x == 0) *sum = s;
    scan[threadIdx.x] = kc::block_scan<false>(iv[threadIdx.x], ib);
    scan_max[threadIdx.x] = kc::block_scan<true>(iv[threadIdx.x], ib);
}

__global__ void key_kernel(const float* v, int n, int* keys) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) keys[i] = kc::order_key(v[i]);
}

__global__ void argmax_kernel(const int* keys, const int* index, int* out) {
    int k = keys[threadIdx.x], i = index[threadIdx.x];
    kc::warp_argmax(k, i);
    out[2 * threadIdx.x] = k;
    out[2 * threadIdx.x + 1] = i;
}

__global__ void merge_kernel(const float* in, int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    const kc::SoftmaxMerge s = kc::softmax_merge(in[4 * i], in[4 * i + 1], in[4 * i + 2], in[4 * i + 3]);
    out[4 * i] = s.m, out[4 * i + 1] = s.l, out[4 * i + 2] = s.keep, out[4 * i + 3] = s.take;
}

__global__ void declared_kernel(const float* x, const float* y, int n, float* e, float* sw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    e[i] = kc::exp_declared(x[i]);
    sw[i] = kc::swiglu_declared(x[i], y[i]);
}

__global__ void fragment_kernel(__nv_bfloat16* out, float* out32, int ld, int r0, int c0, int rows, int cols) {
    const int lane = threadIdx.x;
    float c[4];
    for(int i = 0; i < 4; ++i) c[i] = float(100 * kc::mma_c_row(lane, i) + kc::mma_c_col(lane, i));
    kc::store_c_fragment(out, ld, r0, c0, rows, cols, c, lane);
    kc::store_c_fragment(out32, ld, r0, c0, rows, cols, c, lane);
}

// host emulations
std::vector<float> butterfly(std::vector<float> v, int width, bool is_max, bool is_min) {
    for(int o = width / 2; o > 0; o >>= 1) {
        std::vector<float> n(v.size());
        for(size_t l = 0; l < v.size(); ++l) {
            const float other = v[l ^ o];
            n[l] = is_max ? std::fmax(v[l], other) : is_min ? std::fmin(v[l], other) : v[l] + other;
        }
        v = n;
    }
    return v;
}
float host_tree16(const float* x) {
    float v[16];
    std::copy(x, x + 16, v);
    for(int w = 8; w >= 1; w /= 2)
        for(int l = 0; l < w; ++l) v[l] = v[l] + v[l + w];
    return v[0];
}
float host_exp_declared(float x) {
    x = x > 88.0f ? 88.0f : x < -87.0f ? -87.0f : x;
    const float n = std::rint(x * 1.44269504f);
    float r = std::fmaf(n, -0.693145752f, x);
    r = std::fmaf(n, -1.42860677e-06f, r);
    float p = 1.98412698e-04f;
    p = std::fmaf(p, r, 1.38888889e-03f);
    p = std::fmaf(p, r, 8.33333333e-03f);
    p = std::fmaf(p, r, 4.16666667e-02f);
    p = std::fmaf(p, r, 1.66666667e-01f);
    p = std::fmaf(p, r, 0.5f);
    p = std::fmaf(p, r, 1.0f);
    p = std::fmaf(p, r, 1.0f);
    return p * check::from_bits(uint32_t((int(n) + 127) << 23));
}

}  // namespace

int main() {
    check::flags();
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);

    {   // warp level, 4 warps
        const int n = 128;
        std::vector<float> v(n);
        for(auto& x : v) x = u(rng) * std::ldexp(1.0f, int(rng() % 20) - 10);
        float* dv = check::dev(v);
        float *s = check::dev<float>(n), *s8 = check::dev<float>(n), *mx = check::dev<float>(n), *mn = check::dev<float>(n);
        float *tr = check::dev<float>(8), *fo = check::dev<float>(8), *mf = check::dev<float>(n);
        warp_kernel<<<1, n>>>(dv, s, s8, mx, mn, tr, fo, mf);
        auto S = check::get(s, n), S8 = check::get(s8, n), MX = check::get(mx, n), MN = check::get(mn, n);
        auto TR = check::get(tr, 8), FO = check::get(fo, 8), MF = check::get(mf, n);
        long long b_sum = 0, b_sum8 = 0, b_max = 0, b_min = 0, b_tree = 0, b_fold = 0, b_from = 0;
        for(int w = 0; w < 4; ++w) {
            std::vector<float> lanes(v.begin() + 32 * w, v.begin() + 32 * w + 32);
            auto hs = butterfly(lanes, 32, false, false), hmx = butterfly(lanes, 32, true, false);
            auto hmn = butterfly(lanes, 32, false, true);
            std::vector<float> h8(32);
            for(int g = 0; g < 4; ++g) {
                auto part = butterfly(std::vector<float>(lanes.begin() + 8 * g, lanes.begin() + 8 * g + 8), 8, false, false);
                std::copy(part.begin(), part.end(), h8.begin() + 8 * g);
            }
            for(int l = 0; l < 32; ++l) {
                b_sum += bits(S[32 * w + l]) != bits(hs[l]);
                b_sum8 += bits(S8[32 * w + l]) != bits(h8[l]);
                b_max += bits(MX[32 * w + l]) != bits(hmx[l]);
                b_min += bits(MN[32 * w + l]) != bits(hmn[l]);
                float m = lanes[l & 3];
                for(int g = 0; g < 8; ++g) m = std::fmax(m, lanes[4 * g + (l & 3)]);
                b_from += MF[32 * w + l] != m;
            }
            for(int h = 0; h < 2; ++h) {
                const float t = host_tree16(&lanes[16 * h]);
                b_tree += bits(TR[2 * w + h]) != bits(t);
                b_fold += bits(FO[2 * w + h]) != bits(t);
            }
        }
        report("warp_sum (butterfly bits)", b_sum);
        report("warp_sum<8>", b_sum8);
        report("warp_max / warp_min", b_max + b_min);
        report("warp_max_from<4>", b_from);
        report("tree_sum_down<16> (halving tree bits)", b_tree);
        report("fold16 in kLaneOrder16 == tree_sum_down<16>", b_fold);
    }

    {   // block level, 256 threads
        const int n = 256;
        std::vector<float> v(n);
        std::vector<int> iv(n);
        for(auto& x : v) x = u(rng);
        for(auto& x : iv) x = int(rng() % 1000) - 300;
        float* dv = check::dev(v);
        int* di = check::dev(iv);
        float* s = check::dev<float>(1);
        int *sc = check::dev<int>(n), *sm = check::dev<int>(n);
        block_kernel<<<1, n>>>(dv, di, s, sc, sm);
        auto S = check::get(s, 1);
        auto SC = check::get(sc, n), SM = check::get(sm, n);
        float total = 0.0f;
        for(int w = 0; w < n / 32; ++w) {
            const float ws = butterfly(std::vector<float>(v.begin() + 32 * w, v.begin() + 32 * w + 32), 32, false, false)[0];
            total = w == 0 ? ws : total + ws;
        }
        long long b_scan = 0, b_max = 0;
        int run = 0, best = 0;
        for(int i = 0; i < n; ++i) {
            run += iv[i];
            best = i == 0 ? iv[0] : std::max(best, iv[i]);
            b_scan += SC[i] != run;
            b_max += SM[i] != best;
        }
        report("block_sum (warp butterflies, then warps in order)", bits(S[0]) != bits(total));
        report("block_scan<false> (inclusive sum)", b_scan);
        report("block_scan<true> (inclusive max)", b_max);
    }

    {   // selection keys: the order of floats, NaN below -inf
        std::vector<float> v = {-INFINITY, -1e30f, -2.0f, -1.0f, -1e-40f, -0.0f, 0.0f, 1e-40f, 1.0f, 2.0f, 1e30f, INFINITY};
        for(int i = 0; i < 200; ++i) v.push_back(u(rng));
        v.push_back(NAN);
        const int n = int(v.size());
        float* dv = check::dev(v);
        int* dk = check::dev<int>(n);
        key_kernel<<<1, 256>>>(dv, n, dk);
        auto K = check::get(dk, n);
        long long bad = 0;
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j) {
                const bool ni = v[i] != v[i], nj = v[j] != v[j];
                const bool less = ni ? !nj : nj ? false : v[i] < v[j];
                const bool equal = ni ? nj : nj ? false : v[i] == v[j];
                bad += (K[i] < K[j]) != less || (K[i] == K[j]) != equal;
            }
        bad += !(kc::kPastKey < K[n - 1]);
        report("order_key (total order, -0 == +0, NaN below -inf)", bad);

        std::vector<int> keys(32), index(32);
        long long bad_arg = 0;
        for(int rep = 0; rep < 50; ++rep) {
            for(int l = 0; l < 32; ++l) {
                keys[l] = int(rng() % 6);
                index[l] = int(rng() % 64);
            }
            int* dkk = check::dev(keys);
            int* dii = check::dev(index);
            int* out = check::dev<int>(64);
            argmax_kernel<<<1, 32>>>(dkk, dii, out);
            auto O = check::get(out, 64);
            int bk = keys[0], bi = index[0];
            for(int l = 1; l < 32; ++l)
                if(keys[l] > bk || (keys[l] == bk && index[l] < bi)) bk = keys[l], bi = index[l];
            for(int l = 0; l < 32; ++l) bad_arg += O[2 * l] != bk || O[2 * l + 1] != bi;
            cudaFree(dkk), cudaFree(dii), cudaFree(out);
        }
        report("warp_argmax (larger key, then lower index)", bad_arg);
    }

    {   // online softmax merge against double precision
        const int n = 4096;
        std::vector<float> in(4 * n);
        for(int i = 0; i < n; ++i) {
            in[4 * i] = i % 7 == 0 ? -INFINITY : u(rng);
            in[4 * i + 1] = in[4 * i] == -INFINITY ? 0.0f : std::fabs(u(rng));
            in[4 * i + 2] = i % 11 == 0 ? -INFINITY : u(rng);
            in[4 * i + 3] = in[4 * i + 2] == -INFINITY ? 0.0f : std::fabs(u(rng));
        }
        float *di = check::dev(in), *dout = check::dev<float>(4 * n);
        merge_kernel<<<(n + 127) / 128, 128>>>(di, n, dout);
        auto O = check::get(dout, 4 * n);
        long long bad = 0;
        for(int i = 0; i < n; ++i) {
            const double mr = in[4 * i], lr = in[4 * i + 1], m = in[4 * i + 2], l = in[4 * i + 3];
            const double mn = std::fmax(mr, m), sub = mn == -INFINITY ? 0.0 : mn;
            const double keep = std::exp(mr - sub), take = std::exp(m - sub), ln = lr * keep + l * take;
            bad += O[4 * i] != float(mn) || std::isnan(O[4 * i + 1]) ||
                   std::fabs(O[4 * i + 1] - ln) > 4e-6 * std::fmax(1.0, ln) || std::fabs(O[4 * i + 2] - keep) > 4e-7 ||
                   std::fabs(O[4 * i + 3] - take) > 4e-7;
        }
        report("softmax_merge (within FP32 rounding, no NaN for empty states)", bad);
    }

    {   // declared exp and SwiGLU, bit for bit
        std::vector<float> x, y;
        for(float a = -100.0f; a <= 100.0f; a += 0.0137f) x.push_back(a);
        for(int i = 0; i < 20000; ++i) x.push_back(std::ldexp(u(rng), int(rng() % 30) - 25));
        x.push_back(0.0f), x.push_back(-0.0f), x.push_back(88.0f), x.push_back(-87.0f), x.push_back(1e30f);
        for(size_t i = 0; i < x.size(); ++i) y.push_back(u(rng));
        const int n = int(x.size());
        float *dx = check::dev(x), *dy = check::dev(y), *de = check::dev<float>(n), *ds = check::dev<float>(n);
        declared_kernel<<<(n + 255) / 256, 256>>>(dx, dy, n, de, ds);
        auto E = check::get(de, n), SW = check::get(ds, n);
        long long b_exp = 0, b_sw = 0;
        for(int i = 0; i < n; ++i) {
            b_exp += !same(E[i], host_exp_declared(x[i]));
            b_sw += !same(SW[i], (x[i] / (1.0f + host_exp_declared(-x[i]))) * y[i]);
        }
        report("exp_declared (bits of the declared steps)", b_exp);
        report("swiglu_declared", b_sw);
    }

    {   // fragment stores: a tile at (3, 2) cut to 13 x 7
        const int ld = 12, rows = 13, cols = 7, r0 = 3, c0 = 2;
        __nv_bfloat16* out = check::dev<__nv_bfloat16>(ld * 20);
        float* out32 = check::dev<float>(ld * 20);
        fragment_kernel<<<1, 32>>>(out, out32, ld, r0, c0, rows, cols);
        auto O = check::get(out, ld * 20);
        auto O32 = check::get(out32, ld * 20);
        long long bad = 0;
        for(int r = 0; r < 20; ++r)
            for(int c = 0; c < ld; ++c) {
                const bool inside = r >= r0 && r < r0 + 16 && r < rows && c >= c0 && c < c0 + 8 && c < cols;
                const float want = inside ? float(100 * (r - r0) + (c - c0)) : 0.0f;
                bad += __bfloat162float(O[r * ld + c]) != __bfloat162float(__float2bfloat16_rn(want));
                bad += O32[r * ld + c] != want;
            }
        report("store_c_fragment (BF16 pairs and FP32, cut at the edges)", bad);
    }
    return check::finish();
}
