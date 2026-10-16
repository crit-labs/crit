// crit eval device probe: free device memory, achievable copy bandwidth and dense GEMM rates per
// weight format. Built and run by `crit eval probe`; prints one JSON object on stdout.
//
// Memory: free and total device memory (cudaMemGetInfo) once the probe's CUDA context exists and
// before it allocates anything, which is what an engine process can use; Eval's max context starts
// from the free bytes.
//
// Bandwidth: device-to-device copies of a 2 GiB buffer, counting bytes read plus bytes written,
// mean over batches. GEMM: cuBLASLt with FP32 accumulation at prefill-like shapes, best rate per
// format: BF16, FP8 E4M3 and NVFP4 (E2M1 with 16-element E4M3 block scales). Operands hold
// pseudo-random bytes (constant operands draw less power and run faster than real weights);
// block scales are a fixed small power of two. A format cuBLASLt cannot run is omitted.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>

namespace {

__global__ void fill_random(uint8_t* data, size_t bytes, uint32_t seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < bytes; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = uint32_t(i) * 2654435761u ^ seed;
        h ^= h >> 15;
        h *= 2246822519u;
        h ^= h >> 13;
        data[i] = uint8_t(h);
    }
}

void fill(void* data, size_t bytes, uint32_t seed) { fill_random<<<1024, 256>>>(static_cast<uint8_t*>(data), bytes, seed); }

double copy_bandwidth_gbs(size_t bytes, int batches, int copies) {
    void *src = nullptr, *dst = nullptr;
    if (cudaMalloc(&src, bytes) != cudaSuccess || cudaMalloc(&dst, bytes) != cudaSuccess) return 0.0;
    cudaMemset(src, 1, bytes);
    cudaMemset(dst, 0, bytes);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);  // warmup
    double total = 0.0;
    for (int b = 0; b < batches; ++b) {
        cudaEventRecord(start);
        for (int i = 0; i < copies; ++i) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        total += 2.0 * bytes * copies / (ms * 1e-3) / 1e9;
    }
    cudaFree(src);
    cudaFree(dst);
    return total / batches;
}

// TFLOPS of D[N x M] = W[N x K] * X[K x M] (W row-major, op T), or 0 when unsupported.
double gemm_tflops(cublasLtHandle_t lt, cudaDataType_t type, bool block_scaled, int M, int N, int K) {
    const bool fp4 = type == CUDA_R_4F_E2M1;
    const size_t element_bits = fp4 ? 4 : (type == CUDA_R_8F_E4M3 ? 8 : 16);
    const size_t wbytes = size_t(N) * K * element_bits / 8, xbytes = size_t(M) * K * element_bits / 8;
    // Block-scale tensors are padded to 128 rows x 4 scale columns.
    const size_t sw_bytes = size_t((N + 127) / 128 * 128) * ((K / 16 + 3) / 4 * 4);
    const size_t sx_bytes = size_t((M + 127) / 128 * 128) * ((K / 16 + 3) / 4 * 4);
    const size_t ws_bytes = 64ull << 20;
    void *w = nullptr, *x = nullptr, *d = nullptr, *ws = nullptr, *sw = nullptr, *sx = nullptr;
    double result = 0.0;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heuristics[8];
    int found = 0;
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    const float one = 1.f, zero = 0.f;
    cudaEvent_t start, stop;
    if (cudaMalloc(&w, wbytes) || cudaMalloc(&x, xbytes) || cudaMalloc(&d, size_t(M) * N * 2) || cudaMalloc(&ws, ws_bytes))
        goto done;
    fill(w, wbytes, 1);
    fill(x, xbytes, 2);
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) goto done;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
    if (block_scaled) {
        const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        if (cudaMalloc(&sw, sw_bytes) || cudaMalloc(&sx, sx_bytes)) goto done;
        cudaMemset(sw, 0x38, sw_bytes);
        cudaMemset(sx, 0x38, sx_bytes);
        if (cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof mode) ||
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof mode) ||
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sw, sizeof sw) ||
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof sx))
            goto done;
    }
    if (cublasLtMatrixLayoutCreate(&a, type, K, N, K) || cublasLtMatrixLayoutCreate(&b, type, K, M, K) ||
        cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, M, N))
        goto done;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof ws_bytes);
    if (cublasLtMatmulAlgoGetHeuristic(lt, op, a, b, c, c, pref, 8, heuristics, &found) || !found) goto done;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int h = 0; h < found; ++h) {  // best of the offered algorithms
        bool ok = true;
        for (int i = 0; i < 3 && ok; ++i)
            ok = !cublasLtMatmul(lt, op, &one, w, a, x, b, &zero, d, c, d, c, &heuristics[h].algo, ws, ws_bytes, 0);
        if (!ok) continue;
        const int reps = 20;
        cudaEventRecord(start);
        for (int i = 0; i < reps; ++i)
            cublasLtMatmul(lt, op, &one, w, a, x, b, &zero, d, c, d, c, &heuristics[h].algo, ws, ws_bytes, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        const double tflops = 2.0 * M * N * K / (ms / reps * 1e-3) / 1e12;
        if (cudaGetLastError() == cudaSuccess && tflops > result) result = tflops;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
done:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (op) cublasLtMatmulDescDestroy(op);
    for (void* p : {w, x, d, ws, sw, sx})
        if (p) cudaFree(p);
    cudaGetLastError();
    return result;
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::printf("{\"error\": \"no CUDA device\"}\n");
        return 1;
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        std::printf("{\"error\": \"cudaMemGetInfo failed\"}\n");
        return 1;
    }
    const double bandwidth = copy_bandwidth_gbs(size_t(2) << 30, 5, 4);
    cublasLtHandle_t lt;
    if (cublasLtCreate(&lt)) {
        std::printf("{\"error\": \"cublasLtCreate failed\"}\n");
        return 1;
    }
    // Prefill-like shapes: M prompt rows against wide and square projection matrices.
    const int shapes[][3] = {{2048, 17408, 5120}, {4096, 17408, 5120}, {4096, 5120, 17408}, {4096, 8192, 8192}};
    struct Format { const char* name; cudaDataType_t type; bool scaled; };
    const Format formats[] = {{"bf16", CUDA_R_16BF, false}, {"fp8", CUDA_R_8F_E4M3, false}, {"nvfp4", CUDA_R_4F_E2M1, true}};
    std::string tflops;
    for (const Format& f : formats) {
        double best = 0.0;
        for (const auto& s : shapes) {
            const double t = gemm_tflops(lt, f.type, f.scaled, s[0], s[1], s[2]);
            if (t > best) best = t;
        }
        if (best > 0.0) {
            char item[96];
            std::snprintf(item, sizeof item, "%s\"%s\": %.1f", tflops.empty() ? "" : ", ", f.name, best);
            tflops += item;
        }
    }
    cublasLtDestroy(lt);
    int runtime = 0;
    cudaRuntimeGetVersion(&runtime);
    std::printf("{\"device\": \"%s\", \"compute_capability\": \"%d.%d\", \"cuda_runtime\": %d, "
                "\"memory_total_bytes\": %llu, \"memory_free_bytes\": %llu, "
                "\"bandwidth_gbs\": %.1f, \"tflops\": {%s}}\n",
                prop.name, prop.major, prop.minor, runtime, static_cast<unsigned long long>(total_bytes),
                static_cast<unsigned long long>(free_bytes), bandwidth, tflops.c_str());
    return 0;
}
