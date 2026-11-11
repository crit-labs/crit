// crit_kcore/blockscaled_*.cuh on the device: the tree's lane order (with reduce.cuh's fold16) against the 16-lane
// halving tree, the visiting orders (every block once, each lane's blocks in increasing order, the lanes in the
// tree's order), the nibble decoder against formats.cuh's for every 16-bit code word and both tables, the biased s8
// block sum and its conversion against the host's sum, the declared step's rounding and the scale slots.
// tests/test_core.py builds it for compute_80 (PTX), plain sm_120 and sm_120a.
#include "check.cuh"
#include "crit_kcore/blockscaled_mma.cuh"
#include "crit_kcore/blockscaled_order.cuh"
#include "crit_kcore/blockscaled_stage.cuh"
#include "crit_kcore/formats.cuh"

#include <cmath>
#include <cstring>
#include <random>

namespace kc = crit::kcore;
namespace bs = crit::kcore::blockscaled;
using check::report;

namespace {

// Lanes folded one at a time in lane_at's order by reduce.cuh's fold16 against the 16 lanes kept and halved: out[2 i]
// and out[2 i + 1]; lane_at against kLaneOrder16 (bad[0]).
__global__ void fold_kernel(const float* lanes, int n, float* out, int* bad) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i < 16 && bs::lane_at(i) != kc::kLaneOrder16[i]) atomicAdd(bad, 1);
    if(i >= n) return;
    const float* l = lanes + 16 * i;
    float x[4];
#pragma unroll
    for(int at = 0; at < 16; ++at) kc::fold16(at, l[bs::lane_at(at)], x);
    float v[16];
    for(int j = 0; j < 16; ++j) v[j] = l[j];
    for(int w = 8; w >= 1; w /= 2)
        for(int j = 0; j < w; ++j) v[j] = __fadd_rn(v[j], v[j + w]);
    out[2 * i] = x[0];
    out[2 * i + 1] = v[0];
}

// The visits of GroupVisits<G, RUN> over `blocks` blocks: out[] the blocks in visiting order, -1 for an empty slot.
template<int G, int RUN>
__global__ void visits_kernel(int blocks, int* out, int* count) {
    bs::GroupVisits<G, RUN> it(blocks);
    int n = 0;
    while(!it.done()) {
        for(int r = 0; r < RUN; ++r)
            for(int s = 0; s < G; ++s) out[n++] = it.block(s, r);
        it.next();
    }
    *count = n;
}

__global__ void nibble_kernel(int int4, uint32_t* out) {
    const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if(w >= 65536u) return;
    out[2 * w] = bs::nibbles_to_s8x4(w, bs::nibble_table(int4 != 0));
    out[2 * w + 1] = uint32_t(kc::four_bit_int8x4(w, int4 != 0));
}

// One s8 m16n8k16 block sum a warp, biased then converted, against the plain integer MMA from zero.
__global__ void s8_kernel(const uint32_t* a, const uint32_t* b, float* biased, int* plain) {
    const int lane = threadIdx.x;
    int d[4], z[4] = {0, 0, 0, 0};
    bs::block_sum_s8(a[2 * lane], a[2 * lane + 1], b[lane], d);
    kc::mma_s8_m16n8k16(z, a[2 * lane], a[2 * lane + 1], b[lane]);
    for(int i = 0; i < 4; ++i) {
        biased[4 * lane + i] = bs::biased_sum(d[i]);
        plain[4 * lane + i] = z[i];
    }
}

__global__ void step_kernel(const float* in, int n, float* out) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if(i < n) out[i] = bs::step(in[4 * i], in[4 * i + 1], in[4 * i + 2], in[4 * i + 3]);
}

template<int G, int RUN>
long long check_visits(int blocks) {
    int* d = check::dev<int>(1024);
    int* c = check::dev<int>(1);
    visits_kernel<G, RUN><<<1, 1>>>(blocks, d, c);
    const int n = check::get(c, 1)[0];
    const auto v = check::get(d, std::size_t(n));
    long long bad = 0;
    std::vector<int> seen(blocks, 0);
    int last_lane_step = -1;
    std::vector<int> last(16, -1);
    for(int i = 0; i < n; ++i) {
        const int b = v[std::size_t(i)];
        if(b < 0) continue;
        if(b >= blocks) { ++bad; continue; }
        ++seen[std::size_t(b)];
        int step = 0;
        while(bs::lane_at(step) != b % 16) ++step;
        if(b <= last[std::size_t(b % 16)]) ++bad;                    // a lane's blocks in increasing order
        last[std::size_t(b % 16)] = b;
        if(step / G < last_lane_step / G) ++bad;                       // the groups in the tree's order
        last_lane_step = step;
    }
    for(int b = 0; b < blocks; ++b) bad += seen[std::size_t(b)] != 1;   // every block once
    cudaFree(d);
    cudaFree(c);
    return bad;
}

}  // namespace

int main() {
    const check::Flags f = check::flags();
    (void)f;
    std::mt19937 rng(11);

    // lane_at is the tree's consumption order.
    {
        const int want[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
        long long bad = 0;
        for(int at = 0; at < 16; ++at) bad += bs::lane_at(at) != want[at];
        report("lane_at is the halving tree's lane order", bad);
    }
    // fold: the tree's bits, on random magnitudes and signs (rounding everywhere).
    {
        const int n = 4096;
        std::vector<float> lanes(std::size_t(16) * n);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::uniform_int_distribution<int> e(-20, 20);
        for(auto& x : lanes) x = std::ldexp(u(rng), e(rng));
        float* dl = check::dev(lanes);
        float* dout = check::dev<float>(2 * n);
        int* dbad = check::dev<int>(1);
        fold_kernel<<<(n + 127) / 128, 128>>>(dl, n, dout, dbad);
        const auto out = check::get(dout, std::size_t(2) * n);
        long long bad = check::get(dbad, 1)[0];
        for(int i = 0; i < n; ++i) bad += !check::same(out[2 * i], out[2 * i + 1]);
        report("fold16 in lane_at order gives the 16-lane halving tree's bits", bad);
        cudaFree(dbad);
        cudaFree(dl);
        cudaFree(dout);
    }
    // GroupVisits: every block once, each lane's blocks in increasing order, the groups in the tree's order.
    {
        long long bad = 0;
        for(int blocks : {1, 3, 15, 16, 17, 20, 40, 80, 81, 160, 513}) {
            bad += check_visits<1, 1>(blocks) + check_visits<2, 1>(blocks) + check_visits<4, 1>(blocks);
            bad += check_visits<1, 2>(blocks) + check_visits<2, 2>(blocks) + check_visits<2, 4>(blocks);
        }
        report("GroupVisits visits every block once in the declared lane order", bad);
    }
    // nibbles_to_s8x4 against formats.cuh's four_bit_int8x4, every 16-bit word, E2M1 x 2 and INT4.
    for(int int4 : {0, 1}) {
        uint32_t* d = check::dev<uint32_t>(2 * 65536);
        nibble_kernel<<<256, 256>>>(int4, d);
        const auto v = check::get(d, 2 * 65536);
        long long bad = 0;
        for(int w = 0; w < 65536; ++w) bad += v[std::size_t(2 * w)] != v[std::size_t(2 * w + 1)];
        report(int4 ? "nibbles_to_s8x4 decodes INT4 as four_bit_int8x4"
                    : "nibbles_to_s8x4 decodes E2M1 x 2 as four_bit_int8x4", bad);
        cudaFree(d);
    }
    // block_sum_s8 + biased_sum: the exact block sums as FP32, at the extremes of int8 too.
    {
        std::vector<uint32_t> a(64), b(32);
        long long bad = 0;
        for(int round = 0; round < 3; ++round) {
            for(auto& x : a) x = round == 0 ? 0x80808080u : round == 1 ? 0x7f7f7f7fu : uint32_t(rng());
            for(auto& x : b) x = round == 0 ? 0x80808080u : round == 1 ? 0x81818181u : uint32_t(rng());
            uint32_t* da = check::dev(a);
            uint32_t* db = check::dev(b);
            float* dbias = check::dev<float>(128);
            int* dplain = check::dev<int>(128);
            s8_kernel<<<1, 32>>>(da, db, dbias, dplain);
            const auto biased = check::get(dbias, 128);
            const auto plain = check::get(dplain, 128);
            for(int i = 0; i < 128; ++i) bad += biased[std::size_t(i)] != float(plain[std::size_t(i)]);
            for(void* p : {(void*)da, (void*)db, (void*)dbias, (void*)dplain}) cudaFree(p);
        }
        report("block_sum_s8 and biased_sum give the exact block sums", bad);
    }
    // step: fma(sum, round(ws * as), lane), each rounded once.
    {
        const int n = 4096;
        std::vector<float> in(std::size_t(4) * n);
        std::uniform_real_distribution<float> u(-4.0f, 4.0f);
        for(auto& x : in) x = u(rng);
        float* din = check::dev(in);
        float* dout = check::dev<float>(n);
        step_kernel<<<(n + 127) / 128, 128>>>(din, n, dout);
        const auto out = check::get(dout, std::size_t(n));
        long long bad = 0;
        for(int i = 0; i < n; ++i) {
            const float p = in[std::size_t(4 * i + 1)] * in[std::size_t(4 * i + 2)];
            bad += !check::same(out[std::size_t(i)], std::fmaf(in[std::size_t(4 * i)], p, in[std::size_t(4 * i + 3)]));
        }
        report("step rounds the scale product, then one FMA", bad);
        cudaFree(din);
        cudaFree(dout);
    }
    // pair_slot puts rows g and g + 8 of each 16-row tile side by side, a permutation of each tile.
    {
        long long bad = 0;
        for(int r = 0; r < 64; ++r) {
            const int s = bs::pair_slot(r);
            bad += s / 16 != r / 16 || s % 2 != (r % 16) / 8 || (s % 16) / 2 != r % 8;
        }
        report("pair_slot pairs rows g and g + 8", bad);
    }
    return check::finish();
}
