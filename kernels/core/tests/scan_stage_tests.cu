// scan_stage.cuh on the device: a [dk][dv] matrix's slot of 32 value columns staged by a block (cp.async where the
// rows are 16-byte aligned, plain loads where they are not) and read back lane by lane, for 1, 2, 4 and 8 key rows a
// lane and 4, 8 and 16 columns a lane, against the matrix's values (zeros past dk and dv), bit for bit.
#include "check.cuh"
#include "crit_kcore/scan_stage.cuh"

#include <random>
#include <string>

using namespace crit::kcore;

// One block of NT = 32 x 32 / CW threads stages slot `col0` and each lane writes its piece to out[x][column].
template <int R, int CW>
__global__ void stage_kernel(const float* head, int dk, int dv, int col0, float* out) {
    constexpr int NT = 32 * (32 / CW);
    __shared__ __align__(16) float tile[scan::staged_tile_floats<R>()];
    scan::stage_columns<R, NT>(tile, head, dk, dv, col0);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    const int lane = threadIdx.x & 31, jl = (threadIdx.x >> 5) * CW;
    float S[R][CW];
    scan::read_columns<R, CW>(S, tile, lane, jl);
    for(int r = 0; r < R; ++r)
        for(int j = 0; j < CW; ++j) out[(lane * R + r) * 32 + jl + j] = S[r][j];
}

template <int R, int CW>
void check_shape(std::mt19937& rng, int dk, int dv, int col0, int offset) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> M(std::size_t(dk) * dv + offset);
    for(auto& x : M) x = u(rng);
    float* dM = check::dev(M);
    float* dO = check::dev<float>(32 * R * 32);
    stage_kernel<R, CW><<<1, 32 * (32 / CW)>>>(dM + offset, dk, dv, col0, dO);
    const auto O = check::get(dO, 32 * R * 32);
    long long bad = 0;
    for(int x = 0; x < 32 * R; ++x)
        for(int j = 0; j < 32; ++j) {
            const float want = x < dk && col0 + j < dv ? M[offset + std::size_t(x) * dv + col0 + j] : 0.f;
            bad += !check::same(O[x * 32 + j], want);
        }
    const std::string name = "stage_columns / read_columns R " + std::to_string(R) + " CW " + std::to_string(CW) + " dk " +
                             std::to_string(dk) + " dv " + std::to_string(dv) + " col0 " + std::to_string(col0) +
                             (offset ? " unaligned" : "");
    check::report(name.c_str(), bad);
    cudaFree(dM);
    cudaFree(dO);
}

int main() {
    check::flags();
    std::mt19937 rng(7);
    check_shape<4, 4>(rng, 128, 128, 32, 0);    // a full slot, cp.async
    check_shape<4, 8>(rng, 128, 96, 64, 0);     // the last slot of a head
    check_shape<4, 16>(rng, 100, 128, 0, 0);    // key rows past dk
    check_shape<4, 4>(rng, 128, 100, 96, 0);    // columns past dv, rows not 16-byte aligned: plain loads
    check_shape<4, 8>(rng, 128, 128, 0, 1);     // an unaligned matrix: plain loads
    check_shape<1, 4>(rng, 20, 64, 32, 0);
    check_shape<2, 8>(rng, 64, 40, 32, 0);
    check_shape<8, 4>(rng, 200, 70, 64, 0);
    return check::finish();
}
