// crit_kcore/tile.cuh on the device: KTileCopy's swizzled tiles (zero fill past the rows and the row's bytes), the
// fragments ldsm_a / ldsm_b2 / ldsm_b1 read from them against mma.cuh's documented 8-bit and 4-bit layouts, the
// swizzle a bijection of a tile's chunks, copy_rows at every copy width, and tile_of a bijection of the grid.
#include "check.cuh"
#include "crit_kcore/tile.cuh"

#include <random>

namespace kc = crit::kcore;
namespace kt = crit::kcore::tile;
using check::report;

namespace {

constexpr int kRows = 64, kThreads = 128;

// A 64-row tile of BK bytes at byte k0 of a [rows][ld] matrix (valid_rows, valid_bytes as given), then every warp
// reads each k-step's fragments: out[warp][ks][frag][lane][4] (frag 0..3: A of rows 16 f, 4..7: B2 of columns 16 (f - 4),
// 8: B1 of columns 8), and the tile as stored, unswizzled, into tile_out [64][BK].
template <int BK>
__global__ void tile_kernel(const unsigned char* src, int ld, int valid_rows, int k0, int valid_bytes, uint32_t* out,
                            unsigned char* tile_out) {
    __shared__ __align__(128) unsigned char tile[kRows * BK];
    kt::KTileCopy<kRows, BK, kThreads> copy;
    copy.init(src, std::size_t(ld), valid_rows, int(threadIdx.x));
    copy.issue(tile, k0, valid_bytes);
    kc::cp_async_commit();
    kc::cp_async_wait<0>();
    __syncthreads();
    for(int i = int(threadIdx.x); i < kRows * BK; i += kThreads)
        tile_out[i] = tile[kt::offset<BK>(i / BK, (i % BK) / 16) + i % 16];
    const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
    if(warp != 0) return;
    for(int ks = 0; ks < BK / 32; ++ks) {
        uint32_t* o = out + std::size_t(ks) * 9 * 32 * 4;
        for(int f = 0; f < 4; ++f) {
            uint32_t a[4];
            kt::ldsm_a<BK>(tile, 16 * f, ks, lane, a);
            for(int r = 0; r < 4; ++r) o[(f * 32 + lane) * 4 + r] = a[r];
        }
        for(int f = 0; f < 4; ++f) {
            uint32_t b[4];
            kt::ldsm_b2<BK>(tile, 16 * f, ks, lane, b);
            for(int r = 0; r < 4; ++r) o[((4 + f) * 32 + lane) * 4 + r] = b[r];
        }
        uint32_t b1[2];
        kt::ldsm_b1<BK>(tile, 8, ks, lane, b1);
        o[(8 * 32 + lane) * 4 + 0] = b1[0];
        o[(8 * 32 + lane) * 4 + 1] = b1[1];
    }
}

template <int SB>
__global__ void rows_kernel(const unsigned char* src, int ld, int valid_rows, int byte0, int row_bytes, int width,
                            unsigned char* out) {
    __shared__ __align__(16) unsigned char s[kRows * SB];
    kt::copy_rows<kRows, SB, kThreads>(s, src, std::size_t(ld), valid_rows, byte0, row_bytes, width, int(threadIdx.x));
    kc::cp_async_commit();
    kc::cp_async_wait<0>();
    __syncthreads();
    for(int i = int(threadIdx.x); i < kRows * SB; i += kThreads) out[i] = s[i];
}

template <int BK>
long long check_tile(std::mt19937& rng, int valid_rows, int k0, int valid_bytes) {
    const int ld = 4 * BK + 48;   // rows 16-byte aligned, not a power of two
    std::vector<unsigned char> m(std::size_t(kRows) * ld);
    for(auto& b : m) b = static_cast<unsigned char>(rng());
    unsigned char* dm = check::dev(m);
    uint32_t* dout = check::dev<uint32_t>(std::size_t(BK / 32) * 9 * 32 * 4);
    unsigned char* dtile = check::dev<unsigned char>(std::size_t(kRows) * BK);
    tile_kernel<BK><<<1, kThreads>>>(dm, ld, valid_rows, k0, valid_bytes, dout, dtile);
    const auto out = check::get(dout, std::size_t(BK / 32) * 9 * 32 * 4);
    const auto tile = check::get(dtile, std::size_t(kRows) * BK);
    auto at = [&](int row, int byte) -> unsigned {   // the tile's byte as copied: zero past the rows and the bytes
        return row < valid_rows && k0 + byte < valid_bytes ? m[std::size_t(row) * ld + k0 + byte] : 0u;
    };
    auto word = [&](int row, int byte) {
        return at(row, byte) | at(row, byte + 1) << 8 | at(row, byte + 2) << 16 | unsigned(at(row, byte + 3)) << 24;
    };
    long long bad = 0;
    for(int r = 0; r < kRows; ++r)
        for(int b = 0; b < BK; ++b) bad += tile[std::size_t(r) * BK + b] != at(r, b);
    for(int ks = 0; ks < BK / 32; ++ks)
        for(int lane = 0; lane < 32; ++lane) {
            const int gid = lane / 4, tig = lane % 4, k = 32 * ks + 4 * tig;
            const uint32_t* o = out.data() + std::size_t(ks) * 9 * 32 * 4;
            for(int f = 0; f < 4; ++f) {   // A: a0 row gid, a1 row gid + 8, a2 / a3 the same 16 bytes on
                const uint32_t* a = o + (f * 32 + lane) * 4;
                const int r = 16 * f + gid;
                bad += a[0] != word(r, k) || a[1] != word(r + 8, k) || a[2] != word(r, k + 16) ||
                       a[3] != word(r + 8, k + 16);
            }
            for(int f = 0; f < 4; ++f) {   // B2: columns 16 f + gid, then 16 f + 8 + gid
                const uint32_t* b = o + ((4 + f) * 32 + lane) * 4;
                const int c = 16 * f + gid;
                bad += b[0] != word(c, k) || b[1] != word(c, k + 16) || b[2] != word(c + 8, k) ||
                       b[3] != word(c + 8, k + 16);
            }
            const uint32_t* b1 = o + (8 * 32 + lane) * 4;
            bad += b1[0] != word(8 + gid, k) || b1[1] != word(8 + gid, k + 16);
        }
    cudaFree(dm);
    cudaFree(dout);
    cudaFree(dtile);
    return bad;
}

template <int SB>
long long check_rows(std::mt19937& rng, int ld, int valid_rows, int byte0, int row_bytes) {
    std::vector<unsigned char> m(std::size_t(kRows) * ld + 64);
    for(auto& b : m) b = static_cast<unsigned char>(rng());
    unsigned char* dm = check::dev(m);
    unsigned char* dout = check::dev<unsigned char>(std::size_t(kRows) * SB);
    const int width = kt::row_copy_width(dm, std::size_t(ld), SB);
    long long bad = width == 0;
    rows_kernel<SB><<<1, kThreads>>>(dm, ld, valid_rows, byte0, row_bytes, width, dout);
    const auto out = check::get(dout, std::size_t(kRows) * SB);
    for(int r = 0; r < kRows; ++r)
        for(int b = 0; b < SB; ++b) {
            const unsigned want = r < valid_rows && byte0 + b < row_bytes ? m[std::size_t(r) * ld + byte0 + b] : 0u;
            bad += out[std::size_t(r) * SB + b] != want;
        }
    cudaFree(dm);
    cudaFree(dout);
    return bad;
}

}  // namespace

int main() {
    check::flags();
    std::mt19937 rng(7);
    report("KTileCopy and ldsm_* on 32-byte rows", check_tile<32>(rng, 64, 32, 1 << 20) + check_tile<32>(rng, 37, 0, 16));
    report("KTileCopy and ldsm_* on 64-byte rows", check_tile<64>(rng, 64, 64, 1 << 20) + check_tile<64>(rng, 50, 128, 160));
    report("KTileCopy and ldsm_* on 128-byte rows",
           check_tile<128>(rng, 64, 128, 1 << 20) + check_tile<128>(rng, 9, 0, 80));
    report("KTileCopy and ldsm_* on 256-byte rows", check_tile<256>(rng, 64, 0, 1 << 20) + check_tile<256>(rng, 63, 0, 208));
    {
        long long bad = 0;   // the swizzle permutes each tile's 16-byte chunks
        auto perm = [&](auto bk) {
            constexpr int BK = decltype(bk)::value;
            std::vector<int> seen(std::size_t(kRows) * BK / 16, 0);
            for(int r = 0; r < kRows; ++r)
                for(int c = 0; c < BK / 16; ++c) {
                    const uint32_t o = kt::offset<BK>(r, c);
                    bad += o % 16 != 0 || o / BK != uint32_t(r);
                    if(o / 16 < seen.size()) ++seen[o / 16];
                }
            for(int s : seen) bad += s != 1;
            // eight consecutive rows at one chunk column: eight different 16-byte bank groups
            for(int r0 = 0; r0 < kRows; r0 += 8)
                for(int c = 0; c < BK / 16; ++c) {
                    int groups = 0;
                    for(int r = r0; r < r0 + 8; ++r) groups |= 1 << ((kt::offset<BK>(r, c) / 16) % 8);
                    bad += groups != 0xff;
                }
        };
        perm(std::integral_constant<int, 32>{});
        perm(std::integral_constant<int, 64>{});
        perm(std::integral_constant<int, 128>{});
        perm(std::integral_constant<int, 256>{});
        report("offset permutes a tile's chunks, conflict free for ldmatrix", bad);
    }
    report("copy_rows at 16-byte width", check_rows<16>(rng, 160, 64, 32, 160) + check_rows<32>(rng, 160, 40, 128, 144));
    report("copy_rows at 8-byte width", check_rows<16>(rng, 40, 64, 32, 40) + check_rows<8>(rng, 40, 33, 8, 40));
    report("copy_rows at 4-byte width", check_rows<4>(rng, 20, 64, 4, 20) + check_rows<16>(rng, 84, 64, 64, 84));
    {
        long long bad = 0;   // tile_of visits every tile once
        for(int group : {1, 3, 8})
            for(int tm : {1, 5, 16})
                for(int tn : {1, 7, 40}) {
                    std::vector<int> seen(std::size_t(tm) * tn, 0);
                    for(int i = 0; i < tm * tn; ++i) {
                        int a, b;
                        kt::tile_of(i, tm, tn, group, a, b);
                        if(a < 0 || a >= tm || b < 0 || b >= tn) ++bad;
                        else ++seen[std::size_t(a) * tn + b];
                    }
                    for(int s : seen) bad += s != 1;
                }
        report("tile_of is a bijection of the grid", bad);
    }
    return check::finish();
}
