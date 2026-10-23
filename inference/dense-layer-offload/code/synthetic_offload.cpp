// Synthetic dense model under a layer placement; see synthetic_offload.h.
#include "synthetic_offload.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace crit::dense_offload {
namespace be = crit::batched_engine;
namespace pa = crit::paged_attention;

namespace {
void check(cudaError_t e, const char* what) {
    if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
std::uint64_t mix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
// BF16 bits of a uniform value in [-bound, bound] from (seed, tag, index): the same weights everywhere.
std::uint16_t weight(std::uint64_t seed, std::uint64_t tag, std::uint64_t i, float bound) {
    const double u = static_cast<double>(mix(seed ^ mix(tag * 0x100000001B3ull + i)) >> 11) * 0x1.0p-53;
    const float v = round_bf16(static_cast<float>((2.0 * u - 1.0) * bound));
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return static_cast<std::uint16_t>(bits >> 16);
}
template <class T> T* device_alloc(std::size_t n) {
    void* p = nullptr;
    check(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "device allocation");
    return static_cast<T*>(p);
}
}  // namespace

struct SyntheticOffloadModel::Device {
    __nv_bfloat16 *embedding = nullptr, *head = nullptr, *xn = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *att_bf16 = nullptr, *act = nullptr;
    float *x = nullptr, *qkv = nullptr, *att = nullptr, *gu = nullptr, *logits = nullptr, *scratch = nullptr, *partial = nullptr, *inv_freq = nullptr;
    std::int32_t *tokens = nullptr, *positions = nullptr, *table = nullptr, *kv_len = nullptr, *q_start = nullptr, *last = nullptr;
    std::vector<__nv_bfloat16*> kv_k, kv_v;
    be::GemmPlan qkv_plan, o_plan, gate_up_plan, down_plan, head_plan;
    int blocks = 1;
    std::vector<void*> owned;
    template <class T> T* own(std::size_t n) { T* p = device_alloc<T>(n); owned.push_back(p); return p; }
    ~Device() { for(void* p : owned) cudaFree(p); }
};

SyntheticOffloadModel::SyntheticOffloadModel(const be::Shape& shape, int context, cudaStream_t compute, unsigned threads)
    : shape_(shape), context_(context), compute_(compute), pool_(threads), isa_(best_isa()), d_(std::make_unique<Device>()) {
    const auto& s = shape_;
    if(!be::valid(s) || context <= 0) throw std::invalid_argument("invalid synthetic offload shape or context");
    const std::size_t H = static_cast<std::size_t>(s.hidden), F = static_cast<std::size_t>(s.ffn);
    qkv_elems_ = static_cast<std::size_t>(s.qkv()) * H;
    o_elems_ = H * static_cast<std::size_t>(s.heads) * s.head_dim;
    gate_up_elems_ = 2 * F * H;
    layer_bytes_ = (qkv_elems_ + o_elems_ + gate_up_elems_ + H * F) * 2;
    // Host copies of every layer, pinned so uploads run at full speed; the generator splits rows over the pool.
    host_weights_.assign(static_cast<std::size_t>(s.layers), nullptr);
    for(int l = 0; l < s.layers; ++l) {
        void* p = nullptr;
        check(cudaMallocHost(&p, layer_bytes_), "pinned layer weights");
        host_weights_[static_cast<std::size_t>(l)] = static_cast<std::uint16_t*>(p);
        std::uint16_t* w = host_weights_[static_cast<std::size_t>(l)];
        const std::size_t parts[4] = {qkv_elems_, o_elems_, gate_up_elems_, H * F};
        const float bounds[4] = {std::sqrt(3.f / float(H)), std::sqrt(3.f / float(s.heads * s.head_dim)), std::sqrt(3.f / float(H)), std::sqrt(3.f / float(F))};
        std::size_t at = 0;
        for(int t = 0; t < 4; ++t) {
            const std::uint64_t tag = 16 + 4 * static_cast<std::uint64_t>(l) + static_cast<std::uint64_t>(t);
            std::uint16_t* part = w + at;
            pool_.run(parts[t], [&](std::size_t b, std::size_t e) { for(std::size_t i = b; i < e; ++i) part[i] = weight(s.seed, tag, i, bounds[t]); });
            at += parts[t];
        }
    }
    host_k_.assign(static_cast<std::size_t>(s.layers), std::vector<float>(static_cast<std::size_t>(context) * s.kv_heads * s.head_dim));
    host_v_ = host_k_;
    inv_freq_.resize(static_cast<std::size_t>(s.head_dim / 2));
    for(std::size_t i = 0; i < inv_freq_.size(); ++i) inv_freq_[i] = static_cast<float>(std::pow(double(s.rope_theta), -2.0 * double(i) / s.head_dim));
    void* hx = nullptr;
    check(cudaMallocHost(&hx, sizeof(float) * kMaxRows * H), "pinned activations");
    host_x_ = static_cast<float*>(hx);
    // Device state: embedding and head, activations, per-layer KV (one page of `context` rows each), scratch.
    auto& d = *d_;
    int sms = 0, device = 0;
    check(cudaGetDevice(&device), "device");
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "attribute");
    d.qkv_plan = be::plan_gemm(s.qkv(), s.hidden, sms);
    d.o_plan = be::plan_gemm(s.hidden, s.heads * s.head_dim, sms);
    d.gate_up_plan = be::plan_gemm(2 * s.ffn, s.hidden, sms);
    d.down_plan = be::plan_gemm(s.hidden, s.ffn, sms);
    d.head_plan = be::plan_gemm(s.vocab, s.hidden, sms);
    d.blocks = std::clamp(sms / s.kv_heads, 1, 1024);
    const std::size_t V = static_cast<std::size_t>(s.vocab), R = kMaxRows;
    std::vector<std::uint16_t> table(V * H);
    pool_.run(table.size(), [&](std::size_t b, std::size_t e) { for(std::size_t i = b; i < e; ++i) table[i] = weight(s.seed, 1, i, 1.f); });
    d.embedding = d.own<__nv_bfloat16>(V * H);
    check(cudaMemcpy(d.embedding, table.data(), table.size() * 2, cudaMemcpyHostToDevice), "embedding");
    pool_.run(table.size(), [&](std::size_t b, std::size_t e) { for(std::size_t i = b; i < e; ++i) table[i] = weight(s.seed, 2, i, std::sqrt(3.f / float(H))); });
    d.head = d.own<__nv_bfloat16>(V * H);
    check(cudaMemcpy(d.head, table.data(), table.size() * 2, cudaMemcpyHostToDevice), "head");
    d.x = d.own<float>(R * H); d.xn = d.own<__nv_bfloat16>(R * std::max(H, F)); d.qkv = d.own<float>(R * s.qkv());
    d.q = d.own<__nv_bfloat16>(R * s.heads * s.head_dim); d.k = d.own<__nv_bfloat16>(R * s.kv_heads * s.head_dim);
    d.v = d.own<__nv_bfloat16>(R * s.kv_heads * s.head_dim); d.att = d.own<float>(R * s.heads * s.head_dim);
    d.att_bf16 = d.own<__nv_bfloat16>(R * s.heads * s.head_dim); d.gu = d.own<float>(R * 2 * F); d.act = d.own<__nv_bfloat16>(R * F);
    d.logits = d.own<float>(V);
    std::size_t scratch = 0;
    for(const auto& p : {d.qkv_plan, d.o_plan, d.gate_up_plan, d.down_plan, d.head_plan}) scratch = std::max(scratch, be::gemm_scratch_floats(p, kMaxRows));
    d.scratch = d.own<float>(scratch);
    d.partial = d.own<float>(pa::decode_partial_floats(1, s.kv_heads, s.head_dim, d.blocks, 1, s.heads));
    d.inv_freq = d.own<float>(inv_freq_.size());
    check(cudaMemcpy(d.inv_freq, inv_freq_.data(), inv_freq_.size() * 4, cudaMemcpyHostToDevice), "rope");
    d.tokens = d.own<std::int32_t>(R); d.positions = d.own<std::int32_t>(R); d.table = d.own<std::int32_t>(1);
    d.kv_len = d.own<std::int32_t>(1); d.q_start = d.own<std::int32_t>(2); d.last = d.own<std::int32_t>(1);
    check(cudaMemset(d.table, 0, 4), "table");
    for(int l = 0; l < s.layers; ++l) {
        d.kv_k.push_back(d.own<__nv_bfloat16>(static_cast<std::size_t>(context) * s.kv_heads * s.head_dim));
        d.kv_v.push_back(d.own<__nv_bfloat16>(static_cast<std::size_t>(context) * s.kv_heads * s.head_dim));
    }
    store_.device.assign(static_cast<std::size_t>(s.layers), nullptr);
    store_.host.assign(static_cast<std::size_t>(s.layers), nullptr);
    store_.bytes.assign(static_cast<std::size_t>(s.layers), layer_bytes_);
    for(int l = 0; l < s.layers; ++l) store_.host[static_cast<std::size_t>(l)] = host_weights_[static_cast<std::size_t>(l)];
    resident_.assign(static_cast<std::size_t>(s.layers), nullptr);
}

SyntheticOffloadModel::~SyntheticOffloadModel() {
    cudaStreamSynchronize(compute_);
    for(void* p : resident_) cudaFree(p);
    for(auto* p : host_weights_) cudaFreeHost(p);
    cudaFreeHost(host_x_);
}

const LayerStore& SyntheticOffloadModel::place(const Plan& plan) {
    check(cudaStreamSynchronize(compute_), "drain before placement");
    for(int l = 0; l < shape_.layers; ++l) {
        auto& slot = resident_[static_cast<std::size_t>(l)];
        const bool want = plan.places[static_cast<std::size_t>(l)] == Place::device;
        if(want && !slot) {
            check(cudaMalloc(&slot, layer_bytes_), "resident layer");
            check(cudaMemcpy(slot, host_weights_[static_cast<std::size_t>(l)], layer_bytes_, cudaMemcpyHostToDevice), "resident upload");
        } else if(!want && slot) {
            cudaFree(slot);
            slot = nullptr;
        }
        store_.device[static_cast<std::size_t>(l)] = slot;
    }
    return store_;
}

std::vector<float> SyntheticOffloadModel::step(const Plan& plan, WeightStream* stream, std::span<const std::int32_t> tokens) {
    const int n = static_cast<int>(tokens.size());
    if(n < 1 || n > kMaxRows || pos_ + n > context_) throw std::invalid_argument("step rows outside the model's bounds");
    auto& d = *d_;
    rows_ = n;
    std::vector<std::int32_t> positions(static_cast<std::size_t>(n));
    for(int i = 0; i < n; ++i) positions[static_cast<std::size_t>(i)] = pos_ + i;
    const std::int32_t meta[4] = {pos_ + n, 0, n, n - 1};
    check(cudaMemcpyAsync(d.tokens, tokens.data(), sizeof(std::int32_t) * n, cudaMemcpyHostToDevice, compute_), "tokens");
    check(cudaMemcpyAsync(d.positions, positions.data(), sizeof(std::int32_t) * n, cudaMemcpyHostToDevice, compute_), "positions");
    check(cudaMemcpyAsync(d.kv_len, meta, 4, cudaMemcpyHostToDevice, compute_), "length");
    check(cudaMemcpyAsync(d.q_start, meta + 1, 8, cudaMemcpyHostToDevice, compute_), "offsets");
    check(cudaMemcpyAsync(d.last, meta + 3, 4, cudaMemcpyHostToDevice, compute_), "last row");
    check(be::embed(d.embedding, shape_.hidden, d.tokens, n, d.x, compute_), "embed");
    run_layers(plan, store_, *this, stream, compute_);
    check(be::rmsnorm(d.x, d.last, 1, shape_.hidden, d.xn, compute_), "final norm");
    check(be::gemm(d.head_plan, d.xn, 1, d.head, d.logits, false, d.scratch, compute_), "head");
    std::vector<float> logits(static_cast<std::size_t>(shape_.vocab));
    check(cudaMemcpyAsync(logits.data(), d.logits, logits.size() * 4, cudaMemcpyDeviceToHost, compute_), "logits");
    check(cudaStreamSynchronize(compute_), "step");
    pos_ += n;
    return logits;
}

void SyntheticOffloadModel::device_layer(int layer, const void* weights, cudaStream_t compute) {
    const auto& s = shape_;
    auto& d = *d_;
    const auto* w = static_cast<const __nv_bfloat16*>(weights);
    const __nv_bfloat16 *wqkv = w, *wo = wqkv + qkv_elems_, *wgu = wo + o_elems_, *wdown = wgu + gate_up_elems_;
    const int n = rows_, width = s.kv_heads * s.head_dim, q_width = s.heads * s.head_dim;
    const pa::Pages pages{d.kv_k[static_cast<std::size_t>(layer)], d.kv_v[static_cast<std::size_t>(layer)], nullptr, nullptr, pa::Format::bf16,
                          context_, s.kv_heads, s.head_dim};
    const pa::Batch batch{1, d.table, 1, d.kv_len, d.q_start};
    check(be::rmsnorm(d.x, nullptr, n, s.hidden, d.xn, compute), "norm");
    check(be::gemm(d.qkv_plan, d.xn, n, wqkv, d.qkv, false, d.scratch, compute), "qkv");
    check(be::qkv_rope(d.qkv, d.positions, d.inv_freq, n, s, d.q, d.k, d.v, compute), "rope");
    check(pa::write(pages, d.k, d.v, width, n, d.positions, compute), "kv write");
    const float scale = 1.f / std::sqrt(static_cast<float>(s.head_dim));
    if(n == 1) {
        pa::Decode p;
        p.q = d.q; p.q_stride = q_width; p.pages = pages; p.batch = batch; p.out = d.att; p.out_stride = q_width;
        p.heads = s.heads; p.scale = scale; p.max_columns = 1; p.blocks = d.blocks; p.partial = d.partial;
        check(pa::decode(p, compute), "decode attention");
    } else {
        pa::Prefill p;
        p.q = d.q; p.q_stride = q_width; p.pages = pages; p.batch = batch; p.out = d.att; p.out_stride = q_width;
        p.heads = s.heads; p.scale = scale; p.max_queries = n;
        check(pa::prefill(p, compute), "prefill attention");
    }
    check(be::to_bf16(d.att, static_cast<std::size_t>(n) * q_width, d.att_bf16, compute), "attention");
    check(be::gemm(d.o_plan, d.att_bf16, n, wo, d.x, true, d.scratch, compute), "o");
    check(be::rmsnorm(d.x, nullptr, n, s.hidden, d.xn, compute), "norm");
    check(be::gemm(d.gate_up_plan, d.xn, n, wgu, d.gu, false, d.scratch, compute), "gate/up");
    check(be::silu_mul(d.gu, n, s.ffn, d.act, compute), "silu");
    check(be::gemm(d.down_plan, d.act, n, wdown, d.x, true, d.scratch, compute), "down");
}

void SyntheticOffloadModel::host_layer(int layer) {
    const auto& s = shape_;
    const std::uint16_t* w = host_weights_[static_cast<std::size_t>(layer)];
    const BlockWeights weights{w, w + qkv_elems_, w + qkv_elems_ + o_elems_, w + qkv_elems_ + o_elems_ + gate_up_elems_};
    const BlockShape block{s.hidden, s.heads, s.kv_heads, s.head_dim, s.ffn};
    const HostKv kv{host_k_[static_cast<std::size_t>(layer)].data(), host_v_[static_cast<std::size_t>(layer)].data(), context_};
    for(int r = 0; r < rows_; r += kMaxBatch) {
        const int rows = std::min(kMaxBatch, rows_ - r);
        host_block(pool_, isa_, block, weights, kv, inv_freq_.data(), pos_ + r, rows, host_x_ + static_cast<std::size_t>(r) * s.hidden);
    }
}

void SyntheticOffloadModel::to_host(cudaStream_t compute) {
    check(cudaMemcpyAsync(host_x_, d_->x, sizeof(float) * rows_ * shape_.hidden, cudaMemcpyDeviceToHost, compute), "activations down");
}
void SyntheticOffloadModel::to_device(cudaStream_t compute) {
    check(cudaMemcpyAsync(d_->x, host_x_, sizeof(float) * rows_ * shape_.hidden, cudaMemcpyHostToDevice, compute), "activations up");
}

}  // namespace crit::dense_offload
