// Host execution of offloaded dense layers: BF16-weight GEMV on the CPU (scalar, AVX2 and AVX-512 paths chosen at run
// time through host_isa.h, rows split over a persistent thread pool), and the operations of a pre-norm transformer
// block on host activations: RMSNorm, split-half rotary embedding, grouped-query attention over a host KV cache and
// the SiLU product. Activations that a device kernel would round to BF16 are rounded to BF16 here too, so a layer's
// host and device results differ only by FP32 summation order.
#pragma once
#include "host_isa.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace crit::dense_offload {

constexpr int kMaxBatch = 8;

// The ISA files this build compiled in and the CPU and the OS report (host-expert-arithmetic's host_isa.h).
using Isa = crit::techniques::host_isa::Isa;
using crit::techniques::host_isa::best_isa;
using crit::techniques::host_isa::isa_name;
using crit::techniques::host_isa::supported;

inline float bf16_to_float(std::uint16_t b) {
    const std::uint32_t bits = std::uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}
// Round to the nearest BF16 (ties to even) and back.
float round_bf16(float x);

// y[b * ldy + r] = sum_c w[r * cols + c] * x[b * ldx + c] for rows [row0, row1) and b < batch (1..8); w holds BF16
// bits row-major. Each ISA accumulates a row in its own lane order; rows are independent, so any split over threads
// gives the same bits for one ISA. cols must be a multiple of 16. Throws invalid_argument otherwise or for an ISA the
// CPU does not support.
void gemv_rows(Isa, const std::uint16_t* w, std::size_t cols, const float* x, std::size_t ldx, int batch, std::size_t row0,
               std::size_t row1, float* y, std::size_t ldy);

// A fixed set of worker threads; run(n, fn) calls fn(begin, end) on contiguous slices of [0, n), one per worker, and
// returns when all have finished.
class ThreadPool {
public:
    explicit ThreadPool(unsigned threads);   // 0 = hardware concurrency
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    unsigned size() const noexcept { return unsigned(workers_.size()) + 1; }
    void run(std::size_t n, const std::function<void(std::size_t, std::size_t)>& fn);
private:
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(std::size_t, std::size_t)>* job_ = nullptr;
    std::size_t n_ = 0;
    std::uint64_t generation_ = 0;
    unsigned pending_ = 0;
    bool stop_ = false;
    void loop(unsigned index);
};

void gemv(ThreadPool&, Isa, const std::uint16_t* w, std::size_t rows, std::size_t cols, const float* x, std::size_t ldx, int batch,
          float* y, std::size_t ldy);

// One pre-norm block on the host for `batch` rows at positions pos0 .. pos0 + batch - 1 of one sequence (a prompt
// chunk or a decode step), with weights [out][in] in BF16 bits and a host KV cache [position][kv_heads][head_dim] of
// BF16-rounded values: x += O(attention(rope(Q), rope(K), V)); x += Down(silu(Gate) * Up). Matches the layer of the
// batched reference engine's synthetic model up to summation order.
struct BlockShape { int hidden, heads, kv_heads, head_dim, ffn; };
struct BlockWeights { const std::uint16_t *qkv, *o, *gate_up, *down; };
struct HostKv { float* k; float* v; int capacity; };
void host_block(ThreadPool&, Isa, const BlockShape&, const BlockWeights&, HostKv kv, const float* inv_freq, int pos0, int batch, float* x);

}  // namespace crit::dense_offload
