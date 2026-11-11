// crit_kcore/load.cuh on the device: streaming and read-only loads, cp.async (both cache modes, zero fill, the
// multi-stage ring), every ldmatrix form against the fragment layout it documents, the swizzles (involutions and
// bijections), mbarriers with the TMA's 1-D bulk copy and 2-D tensor tiles (plain and 128-byte swizzled, out-of-bounds
// fill, the store back) and a programmatic dependent launch.
#include "check.cuh"
#include "crit_kcore/load.cuh"

#include <cuda_bf16.h>

#include <random>

namespace kc = crit::kcore;
using check::report;

namespace {

__global__ void loads_kernel(const uint4* src, uint4* a, uint4* b, uint2* c) {
    const int i = threadIdx.x;
    a[i] = kc::ld_stream16(src + i);
    b[i] = kc::ldg16(src + i, i % 3 != 0);
    c[i] = kc::ld_stream8(reinterpret_cast<const uint2*>(src) + i);
}

// A STAGES ring over `chunks` chunks of 256 uint4: each stage copies one chunk (cp.async.cg, every 5th 16-byte piece
// zero-filled) and the block adds it into its output, in stage order.
template <int STAGES>
__global__ void ring_kernel(const uint4* src, int chunks, uint32_t* out) {
    __shared__ uint4 stage[STAGES][256];
    using Ring = kc::CpAsyncRing<STAGES>;
    const int t = threadIdx.x;   // 256 threads
    auto issue = [&](int s) {
        if(s < chunks) kc::cp_async16(&stage[Ring::slot(s)][t], src + s * 256 + t, (s * 256 + t) % 5 != 0);
        Ring::commit();
    };
    for(int s = 0; s < STAGES - 1; ++s) issue(s);
    uint4 acc = make_uint4(0, 0, 0, 0);
    for(int s = 0; s < chunks; ++s) {
        Ring::wait();
        const uint4 v = stage[Ring::slot(s)][(t + 1) % 256];   // another thread's copy: needs the ring's barrier
        acc.x += v.x, acc.y += v.y, acc.z += v.z, acc.w += v.w;
        __syncthreads();                                         // everyone read slot s before it is refilled
        issue(s + STAGES - 1);
    }
    out[4 * t] = acc.x, out[4 * t + 1] = acc.y, out[4 * t + 2] = acc.z, out[4 * t + 3] = acc.w;
}

__global__ void ca_kernel(const uint32_t* src, uint32_t* out) {
    __shared__ uint32_t s[3 * 128];
    const int t = threadIdx.x;   // 32 threads
    kc::cp_async_ca<4>(&s[t], src + t, t != 3);
    kc::cp_async_ca<8>(&s[64 + 2 * t], src + 64 + 2 * t, t != 5);
    kc::cp_async_ca<16>(&s[192 + 4 * t], src + 192 + 4 * t, t != 7);
    kc::cp_async_commit();
    kc::cp_async_wait_all();
    __syncthreads();
    for(int i = t; i < 3 * 128; i += 32) out[i] = s[i];
}

// ldmatrix: four 8 x 8 matrices of 16-bit values, value(m, r, c) = 1000 m + 10 r + c.
__global__ void ldmatrix_kernel(uint32_t* out) {
    __shared__ __align__(16) uint16_t m[4][8][8];
    const int lane = threadIdx.x;
    for(int i = lane; i < 256; i += 32) m[i / 64][(i / 8) % 8][i % 8] = uint16_t(1000 * (i / 64) + 10 * ((i / 8) % 8) + i % 8);
    __syncwarp();
    const void* row = &m[lane / 8][lane % 8][0];
    uint32_t r[16];
    kc::ldmatrix_x1(r[0], &m[0][lane % 8][0]);
    kc::ldmatrix_x2(r[1], r[2], &m[(lane / 8) % 2][lane % 8][0]);
    kc::ldmatrix_x4(r[3], r[4], r[5], r[6], row);
    kc::ldmatrix_x2_trans(r[7], r[8], &m[(lane / 8) % 2][lane % 8][0]);
    kc::ldmatrix_x4_trans(r[9], r[10], r[11], r[12], row);
    for(int i = 0; i < 13; ++i) out[13 * lane + i] = r[i];
}

__global__ void bulk_kernel(const uint4* src, uint4* out, int n) {
    __shared__ __align__(128) uint4 buf[1024];
    __shared__ uint64_t bar;
    if(threadIdx.x == 0) {
        kc::mbarrier_init(&bar, 1);
        kc::fence_barrier_init();
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        kc::mbarrier_arrive_expect_tx(&bar, n * 16);
        kc::tma_load_1d(buf, src, n * 16, &bar);
    }
    kc::mbarrier_wait(&bar, 0);
    for(int i = threadIdx.x; i < n; i += blockDim.x) out[i] = buf[i];
}

// Box (box_rows x box_cols BF16) at (x, y) of `map` into shared memory (row pitch box_cols), copied out raw, then the
// same box stored back through `store_map` at (x, y).
__global__ void tile_kernel(const __grid_constant__ CUtensorMap map, const __grid_constant__ CUtensorMap store_map,
                            int x, int y, int bytes, uint8_t* raw) {
    __shared__ __align__(1024) uint8_t buf[16384];
    __shared__ uint64_t bar;
    if(threadIdx.x == 0) {
        kc::tma_prefetch_map(&map);
        kc::mbarrier_init(&bar, 1);
        kc::fence_barrier_init();
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        kc::mbarrier_arrive_expect_tx(&bar, bytes);
        kc::tma_load_2d(buf, &map, x, y, &bar);
    }
    kc::mbarrier_wait(&bar, 0);
    for(int i = threadIdx.x; i < bytes; i += blockDim.x) raw[i] = buf[i];
    kc::fence_proxy_async();
    __syncthreads();
    if(threadIdx.x == 0) {
        kc::tma_store_2d(&store_map, buf, x, y);
        kc::tma_store_commit();
        kc::tma_store_wait();
    }
}

__global__ void producer_kernel(int* data, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) data[i] = 3 * i + 1;
    kc::griddep_launch_dependents();
}
__global__ void consumer_kernel(const int* data, int* out, int n) {
    kc::griddep_wait();
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = data[i] + 1;
}

}  // namespace

int main() {
    const check::Flags f = check::flags();
    std::mt19937 rng(11);

    {   // loads
        std::vector<uint4> v(64);
        for(auto& x : v) x = make_uint4(rng(), rng(), rng(), rng());
        uint4 *src = check::dev(v), *a = check::dev<uint4>(64), *b = check::dev<uint4>(64);
        uint2* c = check::dev<uint2>(64);
        loads_kernel<<<1, 64>>>(src, a, b, c);
        auto A = check::get(a, 64), B = check::get(b, 64);
        auto C = check::get(c, 64);
        long long bad = 0;
        const uint32_t* words = reinterpret_cast<const uint32_t*>(v.data());
        for(int i = 0; i < 64; ++i) {
            bad += A[i].x != v[i].x || A[i].y != v[i].y || A[i].z != v[i].z || A[i].w != v[i].w;
            const uint4 want = i % 3 ? v[i] : make_uint4(0, 0, 0, 0);
            bad += B[i].x != want.x || B[i].y != want.y || B[i].z != want.z || B[i].w != want.w;
            bad += C[i].x != words[2 * i] || C[i].y != words[2 * i + 1];
        }
        report("ld_stream16 / ldg16 (zero when invalid) / ld_stream8", bad);
    }

    {   // cp.async ring, 2 to 4 stages
        const int chunks = 7;
        std::vector<uint4> v(chunks * 256);
        for(auto& x : v) x = make_uint4(rng() & 0xffff, rng() & 0xffff, rng() & 0xffff, rng() & 0xffff);
        uint4* src = check::dev(v);
        std::vector<uint32_t> ref(1024, 0);
        for(int t = 0; t < 256; ++t)
            for(int s = 0; s < chunks; ++s) {
                const int j = s * 256 + (t + 1) % 256;
                if(j % 5 == 0) continue;
                ref[4 * t] += v[j].x, ref[4 * t + 1] += v[j].y, ref[4 * t + 2] += v[j].z, ref[4 * t + 3] += v[j].w;
            }
        for(int stages : {2, 3, 4}) {
            uint32_t* out = check::dev<uint32_t>(1024);
            if(stages == 2) ring_kernel<2><<<1, 256>>>(src, chunks, out);
            if(stages == 3) ring_kernel<3><<<1, 256>>>(src, chunks, out);
            if(stages == 4) ring_kernel<4><<<1, 256>>>(src, chunks, out);
            auto O = check::get(out, 1024);
            long long bad = 0;
            for(int i = 0; i < 1024; ++i) bad += O[i] != ref[i];
            report(("cp_async16 + CpAsyncRing<" + std::to_string(stages) + "> (zero fill)").c_str(), bad);
        }
    }

    {   // cp.async.ca, 4 / 8 / 16 bytes
        std::vector<uint32_t> v(384);
        for(auto& x : v) x = rng();
        uint32_t *src = check::dev(v), *out = check::dev<uint32_t>(384);
        ca_kernel<<<1, 32>>>(src, out);
        auto O = check::get(out, 384);
        long long bad = 0;
        for(int t = 0; t < 32; ++t) {
            bad += O[t] != (t == 3 ? 0u : v[t]);
            for(int j = 0; j < 2; ++j) bad += O[64 + 2 * t + j] != (t == 5 ? 0u : v[64 + 2 * t + j]);
            for(int j = 0; j < 4; ++j) bad += O[192 + 4 * t + j] != (t == 7 ? 0u : v[192 + 4 * t + j]);
        }
        report("cp_async_ca<4/8/16> (zero fill)", bad);
    }

    {   // ldmatrix
        uint32_t* out = check::dev<uint32_t>(13 * 32);
        ldmatrix_kernel<<<1, 32>>>(out);
        auto O = check::get(out, 13 * 32);
        auto val = [](int m, int r, int c) { return uint32_t(1000 * m + 10 * r + c); };
        auto pair = [&](int m, int r, int c) { return val(m, r, c) | (val(m, r, c + 1) << 16); };
        auto tpair = [&](int m, int r, int c) { return val(m, c, r) | (val(m, c + 1, r) << 16); };
        long long bad = 0;
        for(int l = 0; l < 32; ++l) {
            const int r = l / 4, c = 2 * (l % 4);
            const uint32_t* o = &O[13 * l];
            bad += o[0] != pair(0, r, c);
            bad += o[1] != pair(0, r, c) || o[2] != pair(1, r, c);
            for(int m = 0; m < 4; ++m) bad += o[3 + m] != pair(m, r, c);
            bad += o[7] != tpair(0, r, c) || o[8] != tpair(1, r, c);
            for(int m = 0; m < 4; ++m) bad += o[9 + m] != tpair(m, r, c);
        }
        report("ldmatrix_x1/x2/x4 and _trans (fragment layout)", bad);
    }

    {   // swizzles: involutions, bijections within each 1024-byte block (host and device share the constexpr code)
        long long bad = 0;
        std::vector<int> seen(1024);
        for(int mode = 0; mode < 3; ++mode) {
            std::fill(seen.begin(), seen.end(), 0);
            for(uint32_t o = 0; o < 1024; ++o) {
                const uint32_t s = mode == 0 ? kc::swizzle_128b(o) : mode == 1 ? kc::swizzle_64b(o) : kc::swizzle_32b(o);
                const uint32_t back = mode == 0 ? kc::swizzle_128b(s) : mode == 1 ? kc::swizzle_64b(s) : kc::swizzle_32b(s);
                bad += back != o || s >= 1024 || (s & 15) != (o & 15) || seen[s]++;
            }
        }
        bad += kc::swizzle_128b(1 * 128 + 0) != 128 + 16 || kc::swizzle_128b(7 * 128 + 3 * 16) != 7 * 128 + 4 * 16;
        report("swizzle_128b / 64b / 32b", bad);
    }

    if(f.tma) {
        {   // 1-D bulk copy
            const int n = 1000;
            std::vector<uint4> v(n);
            for(auto& x : v) x = make_uint4(rng(), rng(), rng(), rng());
            uint4 *src = check::dev(v), *out = check::dev<uint4>(n);
            bulk_kernel<<<1, 128>>>(src, out, n);
            auto O = check::get(out, n);
            long long bad = 0;
            for(int i = 0; i < n; ++i) bad += O[i].x != v[i].x || O[i].y != v[i].y || O[i].z != v[i].z || O[i].w != v[i].w;
            report("mbarrier + tma_load_1d", bad);
        }
        {   // 2-D tiles of a [40][72] BF16 matrix (values 1 + index as integers), plain and 128-byte swizzled
            const int rows = 40, cols = 72;
            std::vector<uint16_t> g(rows * cols);
            for(int i = 0; i < rows * cols; ++i) {
                const __nv_bfloat16 h = __float2bfloat16_rn(float(1 + i % 251));
                g[i] = *reinterpret_cast<const uint16_t*>(&h);
            }
            uint16_t* dg = check::dev(g);
            struct Box { int box_rows, box_cols, x, y; CUtensorMapSwizzle sw; const char* name; };
            const Box boxes[] = {{16, 32, 32, 16, CU_TENSOR_MAP_SWIZZLE_NONE, "tma_load_2d (inside)"},
                                 {16, 32, 64, 32, CU_TENSOR_MAP_SWIZZLE_NONE, "tma_load_2d (out-of-bounds zero fill)"},
                                 {16, 64, 0, 8, CU_TENSOR_MAP_SWIZZLE_128B, "tma_load_2d (128B swizzle) = swizzle_128b"}};
            for(const Box& b : boxes) {
                uint16_t* back = check::dev<uint16_t>(rows * cols);
                CUtensorMap map, store_map;
                const CUresult r1 = kc::make_tensor_map_2d(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, dg, rows, cols, cols * 2,
                                                           b.box_rows, b.box_cols, b.sw);
                const CUresult r2 = kc::make_tensor_map_2d(&store_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, back, rows, cols,
                                                           cols * 2, b.box_rows, b.box_cols, b.sw);
                if(r1 != CUDA_SUCCESS || r2 != CUDA_SUCCESS) {
                    report(b.name, 1, "make_tensor_map_2d failed: " + std::to_string(int(r1)) + " " + std::to_string(int(r2)));
                    continue;
                }
                const int bytes = b.box_rows * b.box_cols * 2;
                uint8_t* raw = check::dev<uint8_t>(bytes);
                tile_kernel<<<1, 128>>>(map, store_map, b.x, b.y, bytes, raw);
                auto R = check::get(raw, bytes);
                auto B = check::get(back, rows * cols);
                long long bad = 0, bad_store = 0;
                for(int r = 0; r < b.box_rows; ++r)
                    for(int c = 0; c < b.box_cols; ++c) {
                        const int gr = b.y + r, gc = b.x + c;
                        const uint16_t want = gr < rows && gc < cols ? g[gr * cols + gc] : 0;
                        uint32_t at = uint32_t(r * b.box_cols + c) * 2;
                        if(b.sw == CU_TENSOR_MAP_SWIZZLE_128B) at = kc::swizzle_128b(at);
                        bad += *reinterpret_cast<const uint16_t*>(&R[at]) != want;
                        if(gr < rows && gc < cols) bad_store += B[gr * cols + gc] != want;
                    }
                report(b.name, bad);
                report((std::string(b.name) + ": tma_store_2d back").c_str(), bad_store);
            }
        }
    } else {
        std::printf("SKIP the TMA checks (no TMA in this build)\n");
    }

    {   // programmatic dependent launch (an ordinary launch where the device has none)
        const int n = 1 << 20;
        int *data = check::dev<int>(n), *out = check::dev<int>(n);
        check::ok(kc::launch_programmatic(producer_kernel, dim3(256), dim3(256), 0, nullptr, data, n), "producer");
        check::ok(kc::launch_programmatic(consumer_kernel, dim3(256), dim3(256), 0, nullptr, (const int*)data, out, n),
                  "consumer");
        auto O = check::get(out, n);
        long long bad = 0;
        for(int i = 0; i < n; ++i) bad += O[i] != 3 * i + 2;
        report("launch_programmatic + griddep_wait / griddep_launch_dependents", bad);
    }
    return check::finish();
}
