// A batched serving engine for the synthetic transformer of synthetic_model.h: a Core ConcurrentModel with mixed
// batches whose every step runs all rows (decode rows and prompt chunks of several requests) through one projection
// GEMM per weight, keeps KV in StateCache pages with prefix reuse and attends with paged-attention's kernels.
//
// Ownership: Core's worker thread makes every call; all device work goes to the backend's compute stream and each
// step ends with that stream synchronized, so released pages are reusable at once (retire(epoch()) after a step).
#pragma once
#include "device_pages.h"
#include "paged_attention.h"
#include "synthetic_model.h"

#include <crit/core.h>
#include <crit/cuda.h>
#include <crit/sampling.h>
#include <crit/state.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace crit::batched_engine {
namespace cc = crit::core;
namespace pa = crit::paged_attention;

struct Config {
    Shape shape;
    pa::Format format = pa::Format::bf16;
    int page_tokens = 16;
    std::uint32_t context = 4096;        // tokens per request (prompt + output)
    std::uint32_t cache_tokens = 0;      // KV pages for this many tokens over all requests; 0 = context * concurrency
    std::uint32_t max_concurrency = 8;
    std::uint32_t max_batch_tokens = 256;
    std::uint32_t prefill_chunk = 256;
    bool prefix_cache = true;
    std::vector<cc::Token> stop_tokens = {};
};

// Counters the engine's tests and measurements read (written by the worker thread).
struct Stats {
    std::atomic<std::uint32_t> largest_batch{0}, mixed_steps{0}, steps{0};
    std::atomic<std::uint64_t> reused_tokens{0};
};

class Model final : public cc::ConcurrentModel {
    struct Receipt final : cc::Retirement { bool ready() override { return true; } };
    struct Session final : cc::ConcurrentSession {
        cc::RequestView request;
        cc::StateCache& cache;
        cc::SequenceId sequence;
        std::uint32_t reserved_pages = 0, reused = 0;
        bool reuse_reported = false;
        cc::TokenHistory history;
        cc::SamplingParams params;
        std::optional<cc::Round> round;
        std::vector<cc::Token> inputs, candidate;
        cc::Token last = 0;
        Session(const cc::RequestView& r, cc::StateCache& c, cc::SamplingParams p)
            : request(r), cache(c), sequence(c.open()), history(r.prompt), params(std::move(p)) {}
        ~Session() override { cache.close(sequence); }
        void close(cc::Finish) noexcept override { cache.close(sequence); }
    };

    Config config_;
    cc::CudaBackend& backend_;
    cudaStream_t stream_;
    std::shared_ptr<Stats> stats_;
    int rows_;  // most rows one step can hold
    std::vector<cc::CudaBuffer> buffers_;
    pa::DevicePages pages_;
    cc::Reservation page_reservation_;
    cc::StateCache cache_;
    pa::BatchTables decode_tables_, prefill_tables_;
    cc::Sampler sampler_;
    std::uint32_t reserved_pages_ = 0;
    // Device memory
    __nv_bfloat16 *embedding_ = nullptr, *head_ = nullptr;
    struct Layer { __nv_bfloat16 *qkv, *o, *gate_up, *down; };
    std::vector<Layer> layers_;
    GemmPlan qkv_plan_, o_plan_, gate_up_plan_, down_plan_, head_plan_;
    std::int32_t *tokens_ = nullptr, *positions_ = nullptr, *logit_rows_ = nullptr, *argmax_ = nullptr;
    float *inv_freq_ = nullptr, *x_ = nullptr, *qkv_ = nullptr, *att_ = nullptr, *gu_ = nullptr, *logits_ = nullptr, *scratch_ = nullptr, *partial_ = nullptr;
    __nv_bfloat16 *xn_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_bf16_ = nullptr, *act_ = nullptr;
    std::vector<float> host_logits_;
    std::vector<std::int32_t> host_argmax_;

    template <class T> T* allocate(std::size_t count) {
        buffers_.push_back(backend_.allocate(cc::Memory::device, std::max<std::size_t>(count, 1) * sizeof(T)));
        return static_cast<T*>(buffers_.back().data());
    }
    static int pages_for(const Config& c) {
        const std::uint64_t tokens = c.cache_tokens ? c.cache_tokens : static_cast<std::uint64_t>(c.context) * c.max_concurrency;
        return static_cast<int>((tokens + c.page_tokens - 1) / c.page_tokens);
    }
    static cc::StateCacheConfig cache_config(const Config& c, int rows) {
        cc::StateCacheConfig s;
        s.pages = static_cast<std::uint32_t>(pages_for(c));
        s.page_tokens = static_cast<std::uint32_t>(c.page_tokens);
        s.sequences = c.max_concurrency;
        s.max_tokens = c.context;
        s.max_round_rows = static_cast<std::uint32_t>(std::min<int>(rows, static_cast<int>(c.context)));
        s.prefix_cache = c.prefix_cache;
        // Names the weights (shape and seed) and the KV representation, as a prefix identity must.
        std::uint64_t id = 0x5157'4e7e'0000'0001ull ^ c.shape.seed;
        for(const auto v : {c.shape.vocab, c.shape.hidden, c.shape.layers, c.shape.heads, c.shape.kv_heads, c.shape.head_dim, c.shape.ffn,
                            static_cast<int>(c.format), static_cast<int>(c.shape.rope_theta)})
            id = id * 0x100000001B3ull ^ static_cast<std::uint64_t>(v);
        s.identity = id | 1;
        return s;
    }
    void ok(cudaError_t e, const char* what) const { pa::check(e, what); }
    static bool device_greedy(const cc::SamplingParams& p) {
        return p.temperature == 0.0 && p.repetition_penalty == 1.0 && p.frequency_penalty == 0.0 && p.presence_penalty == 0.0 &&
               p.logit_bias.empty() && p.min_tokens == 0;
    }

public:
    Model(const Config& config, cc::Services services, std::shared_ptr<Stats> stats)
        : config_(config), backend_(dynamic_cast<cc::CudaBackend&>(services.backend)), stream_(backend_.stream()),
          stats_(std::move(stats)),
          rows_(static_cast<int>(std::max({config.max_batch_tokens, config.prefill_chunk, config.max_concurrency}))),
          pages_(config.shape.layers, pages_for(config), config.page_tokens, config.shape.kv_heads, config.shape.head_dim, config.format, stream_),
          page_reservation_(services.resources.reserve(cc::Memory::device, pages_.bytes())),
          cache_(cache_config(config, rows_), pages_, services.resources),
          decode_tables_(static_cast<int>(config.max_concurrency), (static_cast<int>(config.context) + config.page_tokens - 1) / config.page_tokens,
                         static_cast<int>(config.max_concurrency)),
          prefill_tables_(static_cast<int>(config.max_concurrency), (static_cast<int>(config.context) + config.page_tokens - 1) / config.page_tokens, rows_),
          sampler_(static_cast<std::uint32_t>(config.shape.vocab)) {
        const Shape& s = config_.shape;
        if(!valid(s) || config.page_tokens <= 0 || !config.context || !config.max_concurrency) throw std::invalid_argument("Invalid synthetic engine configuration");
        const int sms = backend_.properties().multiProcessorCount;
        qkv_plan_ = plan_gemm(s.qkv(), s.hidden, sms);
        o_plan_ = plan_gemm(s.hidden, s.heads * s.head_dim, sms);
        gate_up_plan_ = plan_gemm(2 * s.ffn, s.hidden, sms);
        down_plan_ = plan_gemm(s.hidden, s.ffn, sms);
        head_plan_ = plan_gemm(s.vocab, s.hidden, sms);
        // Weights: uniform with variance 1 / fan-in (embedding: unit variance), one seed per tensor.
        auto weights = [&](std::size_t count, int fan_in, std::uint64_t tag) {
            auto* w = allocate<__nv_bfloat16>(count);
            ok(fill_uniform(w, count, std::sqrt(3.f / static_cast<float>(fan_in)), s.seed * 0x9E3779B97F4A7C15ull + tag, stream_), "weights");
            return w;
        };
        embedding_ = weights(static_cast<std::size_t>(s.vocab) * s.hidden, 1, 1);
        head_ = weights(static_cast<std::size_t>(s.vocab) * s.hidden, s.hidden, 2);
        for(int l = 0; l < s.layers; ++l) {
            const std::uint64_t t = 16 + 4 * static_cast<std::uint64_t>(l);
            layers_.push_back({weights(static_cast<std::size_t>(s.qkv()) * s.hidden, s.hidden, t),
                               weights(static_cast<std::size_t>(s.hidden) * s.heads * s.head_dim, s.heads * s.head_dim, t + 1),
                               weights(static_cast<std::size_t>(2) * s.ffn * s.hidden, s.hidden, t + 2),
                               weights(static_cast<std::size_t>(s.hidden) * s.ffn, s.ffn, t + 3)});
        }
        std::vector<float> inv(s.head_dim / 2);
        for(std::size_t i = 0; i < inv.size(); ++i) inv[i] = static_cast<float>(std::pow(static_cast<double>(s.rope_theta), -2.0 * static_cast<double>(i) / s.head_dim));
        inv_freq_ = allocate<float>(inv.size());
        ok(cudaMemcpyAsync(inv_freq_, inv.data(), inv.size() * sizeof(float), cudaMemcpyHostToDevice, stream_), "rope");
        const auto r = static_cast<std::size_t>(rows_);
        tokens_ = allocate<std::int32_t>(r); positions_ = allocate<std::int32_t>(r); logit_rows_ = allocate<std::int32_t>(r);
        argmax_ = allocate<std::int32_t>(r);
        x_ = allocate<float>(r * s.hidden); xn_ = allocate<__nv_bfloat16>(r * std::max(s.hidden, s.ffn));
        qkv_ = allocate<float>(r * s.qkv());
        q_ = allocate<__nv_bfloat16>(r * s.heads * s.head_dim); k_ = allocate<__nv_bfloat16>(r * s.kv_heads * s.head_dim);
        v_ = allocate<__nv_bfloat16>(r * s.kv_heads * s.head_dim);
        att_ = allocate<float>(r * s.heads * s.head_dim); att_bf16_ = allocate<__nv_bfloat16>(r * s.heads * s.head_dim);
        gu_ = allocate<float>(r * 2 * s.ffn); act_ = allocate<__nv_bfloat16>(r * s.ffn);
        logits_ = allocate<float>(r * s.vocab);
        std::size_t scratch = 0;
        for(const auto& p : {qkv_plan_, o_plan_, gate_up_plan_, down_plan_, head_plan_}) scratch = std::max(scratch, gemm_scratch_floats(p, rows_));
        scratch_ = allocate<float>(scratch);
        partial_ = allocate<float>(pa::decode_partial_floats(static_cast<int>(config.max_concurrency), s.kv_heads, s.head_dim, blocks(), 1, s.heads));
        host_logits_.resize(r * s.vocab);
        host_argmax_.resize(r);
        ok(cudaStreamSynchronize(stream_), "weights");
    }

    // Decode blocks per (sequence, key/value head): enough for one sequence to cover the multiprocessors.
    int blocks() const {
        return std::clamp(backend_.properties().multiProcessorCount / config_.shape.kv_heads, 1, 1024);
    }
    const cc::StateCache& cache() const { return cache_; }

    cc::ModelInfo info() const override {
        cc::ModelInfo i;
        i.implementation = "synthetic.batched.paged";
        i.vocabulary = static_cast<std::uint32_t>(config_.shape.vocab);
        i.context = config_.context;
        i.prefill_chunk = std::min(config_.prefill_chunk, config_.context);
        i.stop_tokens = config_.stop_tokens;
        i.max_concurrency = config_.max_concurrency;
        i.sampling = {true, true, true, true, true, true, true, false};
        i.mixed_batches = true;
        i.scheduling_hints = config_.prefix_cache;
        return i;
    }
    std::uint32_t pages_needed(const cc::RequestView& r) const {
        return static_cast<std::uint32_t>((r.prompt.size() + r.output_tokens + config_.page_tokens - 1) / config_.page_tokens);
    }
    // Admission reserves a request's whole page need (prompt plus output, no credit for shared pages), so a running
    // request never runs out of pages.
    cc::Admission admission(const cc::RequestView& r) override {
        if(r.prompt.empty() || r.prompt.size() + r.output_tokens > config_.context || pages_needed(r) > cache_.config().pages)
            return cc::Admission::infeasible;
        return reserved_pages_ + pages_needed(r) <= cache_.config().pages ? cc::Admission::ready : cc::Admission::deferred;
    }
    cc::SchedulingHint scheduling_hint(const cc::RequestView& r) override {
        cc::SchedulingHint hint;
        if(config_.prefix_cache && r.prompt.size() > 1)
            hint.reusable_tokens = cache_.reusable(r.prompt, static_cast<std::uint32_t>(r.prompt.size() - 1));
        return hint;
    }
    std::unique_ptr<cc::ConcurrentSession> open(const cc::RequestView& r) override {
        cc::SamplingParams defaults;
        defaults.temperature = 0.0;  // greedy unless the request sets a temperature
        defaults.seed = r.key.request;
        auto session = std::make_unique<Session>(r, cache_, cc::resolve_sampling(r.options->sampling, defaults, static_cast<std::uint32_t>(config_.shape.vocab)));
        if(config_.prefix_cache && r.prompt.size() > 1)
            session->reused = cache_.attach(session->sequence, r.prompt, static_cast<std::uint32_t>(r.prompt.size() - 1));
        session->reserved_pages = pages_needed(r);
        reserved_pages_ += session->reserved_pages;
        stats_->reused_tokens += session->reused;
        return session;
    }

    std::vector<cc::RowProgress> execute(std::span<const cc::WorkItem> items) override {
        const Shape& s = config_.shape;
        // Rows: decode items first (one row each, read by the decode kernel), then prompt chunks (read by the varlen
        // prefill kernel). The kernel follows the phase, never the row count, so a request's bits match a serial run.
        std::vector<const cc::WorkItem*> order;
        for(const auto& item : items) if(item.budget.prefill_complete) order.push_back(&item);
        const std::size_t decoding = order.size();
        for(const auto& item : items) if(!item.budget.prefill_complete) order.push_back(&item);
        decode_tables_.clear();
        prefill_tables_.clear();
        std::vector<std::int32_t> tokens, positions, logit_rows;
        std::vector<cc::RowProgress> progress;
        for(std::size_t i = 0; i < order.size(); ++i) {
            const auto& item = *order[i];
            auto& session = static_cast<Session&>(*item.session);
            const std::uint32_t start = cache_.committed(session.sequence);
            std::uint32_t n = 1;
            if(!item.budget.prefill_complete) {
                n = std::min<std::uint32_t>(item.budget.prefill_tokens, static_cast<std::uint32_t>(session.request.prompt.size()) - start);
                if(item.budget.token_work) n = std::min(n, item.budget.token_work);
            }
            if(!n) throw std::logic_error("A prompt step without rows");
            session.round = cache_.begin(session.sequence, n);
            session.inputs.clear();
            for(std::uint32_t j = 0; j < n; ++j) {
                const cc::Token t = item.budget.prefill_complete ? session.last : session.request.prompt[start + j];
                session.inputs.push_back(t);
                tokens.push_back(t);
                positions.push_back(static_cast<std::int32_t>(start + j));
            }
            auto& tables = i < decoding ? decode_tables_ : prefill_tables_;
            tables.add_slots(session.round->kv, config_.page_tokens);
            tables.add(cache_, session.sequence);
            if(item.budget.prefill_complete || start + n == session.request.prompt.size())
                logit_rows.push_back(static_cast<std::int32_t>(tokens.size() - 1));
        }
        const int rows = static_cast<int>(tokens.size());
        if(rows > rows_) throw std::length_error("Step exceeds the engine's row bound");
        ok(cudaMemcpyAsync(tokens_, tokens.data(), tokens.size() * 4, cudaMemcpyHostToDevice, stream_), "tokens");
        ok(cudaMemcpyAsync(positions_, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice, stream_), "positions");
        if(!logit_rows.empty()) ok(cudaMemcpyAsync(logit_rows_, logit_rows.data(), logit_rows.size() * 4, cudaMemcpyHostToDevice, stream_), "logit rows");
        const int decode_rows = decode_tables_.queries();
        const pa::Batch decode_batch = decode_tables_.upload(stream_), prefill_batch = prefill_tables_.upload(stream_);
        forward(rows, decode_rows, decode_batch, prefill_batch, static_cast<int>(logit_rows.size()));
        // Plain greedy rows take the device argmax; rows with temperature, penalties or bias copy their logits and use
        // Core's Sampler on the host (both pick the lowest id among equal maxima when greedy).
        if(!logit_rows.empty()) {
            ok(argmax(logits_, static_cast<int>(logit_rows.size()), s.vocab, argmax_, stream_), "argmax");
            ok(cudaMemcpyAsync(host_argmax_.data(), argmax_, logit_rows.size() * 4, cudaMemcpyDeviceToHost, stream_), "argmax");
            std::size_t row = 0;
            for(const auto* item : order) {
                const auto& session = static_cast<const Session&>(*item->session);
                if(!item->budget.prefill_complete && cache_.committed(session.sequence) + session.inputs.size() != session.request.prompt.size()) continue;
                if(!device_greedy(session.params))
                    ok(cudaMemcpyAsync(host_logits_.data() + row * s.vocab, logits_ + row * s.vocab, s.vocab * sizeof(float), cudaMemcpyDeviceToHost, stream_), "logits");
                ++row;
            }
        }
        ok(cudaStreamSynchronize(stream_), "step");
        std::size_t logit = 0;
        for(const auto* item : order) {
            auto& session = static_cast<Session&>(*item->session);
            session.candidate.clear();
            const auto n = static_cast<std::uint32_t>(session.inputs.size());
            const bool logits = item->budget.prefill_complete || cache_.committed(session.sequence) + n == session.request.prompt.size();
            if(logits) {
                const std::size_t row = logit++;
                if(device_greedy(session.params)) {
                    session.candidate.push_back(host_argmax_[row]);
                } else {
                    const auto values = std::span<const float>(host_logits_).subspan(row * s.vocab, static_cast<std::size_t>(s.vocab));
                    session.candidate.push_back(sampler_.sample(values, session.params, session.history).token);
                }
                session.history.stage(session.candidate);
            }
            cc::Progress p;
            p.prompt_tokens = item->budget.prefill_complete ? 0 : n;
            if(!item->budget.prefill_complete && !session.reuse_reported) {
                p.prompt_tokens += session.reused;
                p.reused_prompt_tokens = session.reused;
                session.reuse_reported = true;
            }
            p.prefill_complete = logits;
            p.tokens = session.candidate;
            p.token_work = n;
            progress.push_back({item->key, p});
        }
        stats_->steps++;
        stats_->largest_batch = std::max<std::uint32_t>(stats_->largest_batch, static_cast<std::uint32_t>(items.size()));
        if(decoding && decoding < order.size()) stats_->mixed_steps++;
        return progress;
    }

    void forward(int rows, int decode_rows, const pa::Batch& decode_batch, const pa::Batch& prefill_batch, int logit_rows) {
        const Shape& s = config_.shape;
        const int width = s.kv_heads * s.head_dim, q_width = s.heads * s.head_dim;
        const float scale = 1.f / std::sqrt(static_cast<float>(s.head_dim));
        ok(embed(embedding_, s.hidden, tokens_, rows, x_, stream_), "embed");
        for(int l = 0; l < s.layers; ++l) {
            const auto pages = pages_.layer(l);
            ok(rmsnorm(x_, nullptr, rows, s.hidden, xn_, stream_), "norm");
            ok(gemm(qkv_plan_, xn_, rows, layers_[l].qkv, qkv_, false, scratch_, stream_), "qkv");
            ok(qkv_rope(qkv_, positions_, inv_freq_, rows, s, q_, k_, v_, stream_), "rope");
            if(decode_rows) {
                ok(pa::write(pages, k_, v_, width, decode_rows, decode_tables_.device_slots(), stream_), "kv write");
                pa::Decode d;
                d.q = q_; d.q_stride = q_width; d.pages = pages; d.batch = decode_batch; d.out = att_; d.out_stride = q_width;
                d.heads = s.heads; d.scale = scale; d.max_columns = 1; d.blocks = blocks(); d.partial = partial_;
                ok(pa::decode(d, stream_), "decode attention");
            }
            if(rows > decode_rows) {
                const std::size_t off = static_cast<std::size_t>(decode_rows);
                ok(pa::write(pages, k_ + off * width, v_ + off * width, width, rows - decode_rows, prefill_tables_.device_slots(), stream_), "kv write");
                pa::Prefill p;
                p.q = q_ + off * q_width; p.q_stride = q_width; p.pages = pages; p.batch = prefill_batch; p.out = att_ + off * q_width;
                p.out_stride = q_width; p.heads = s.heads; p.scale = scale; p.max_queries = prefill_tables_.max_queries();
                ok(pa::prefill(p, stream_), "prefill attention");
            }
            ok(to_bf16(att_, static_cast<std::size_t>(rows) * q_width, att_bf16_, stream_), "attention");
            ok(gemm(o_plan_, att_bf16_, rows, layers_[l].o, x_, true, scratch_, stream_), "o");
            ok(rmsnorm(x_, nullptr, rows, s.hidden, xn_, stream_), "norm");
            ok(gemm(gate_up_plan_, xn_, rows, layers_[l].gate_up, gu_, false, scratch_, stream_), "gate/up");
            ok(silu_mul(gu_, rows, s.ffn, act_, stream_), "silu");
            ok(gemm(down_plan_, act_, rows, layers_[l].down, x_, true, scratch_, stream_), "down");
        }
        if(logit_rows) {
            ok(rmsnorm(x_, logit_rows_, logit_rows, s.hidden, xn_, stream_), "final norm");
            ok(gemm(head_plan_, xn_, logit_rows, head_, logits_, false, scratch_, stream_), "head");
        }
    }

    cc::RowProgress prefill(const cc::WorkItem& item) override {
        auto mixed = item;
        mixed.budget.token_work = mixed.budget.prefill_tokens;
        return execute(std::span<const cc::WorkItem>(&mixed, 1)).front();
    }
    std::vector<cc::RowProgress> decode(std::span<const cc::WorkItem> items) override {
        std::vector<cc::WorkItem> mixed(items.begin(), items.end());
        for(auto& item : mixed) item.budget.token_work = 1;
        return execute(mixed);
    }
    void commit(std::span<const cc::CommitDecision> decisions) override {
        for(const auto& d : decisions) {
            auto& session = static_cast<Session&>(*d.session);
            if(!session.round) continue;
            const auto round = std::exchange(session.round, std::nullopt)->id;
            if(d.cancelled || d.discard) {
                cache_.discard(session.sequence, round);
                session.history.rollback();
                continue;
            }
            cache_.commit(session.sequence, round, session.inputs);
            if(session.history.staged()) session.history.commit(d.accepted);
            if(d.accepted) session.last = session.candidate.front();
        }
        cache_.retire(cache_.epoch());  // every step ends synchronized
    }
    std::unique_ptr<cc::Retirement> retire(cc::RequestKey, cc::ConcurrentSession& session, cc::Finish) override {
        auto& s = static_cast<Session&>(session);
        if(s.round) { cache_.discard(s.sequence, s.round->id); s.round.reset(); }
        cache_.close(s.sequence);
        cache_.retire(cache_.epoch());
        reserved_pages_ -= std::exchange(s.reserved_pages, 0u);
        return std::make_unique<Receipt>();
    }
    void quiesce() override {}  // no host producers; every step is synchronized before it returns
    std::vector<cc::Token> tokenize(std::string_view) override { throw std::logic_error("Synthetic token-only engine"); }
    std::string detokenize(std::span<const cc::Token>) override { throw std::logic_error("Synthetic token-only engine"); }
};

// A Core engine around the model: mixed batches of at most max_batch_tokens token rows, max_concurrency requests.
inline std::unique_ptr<cc::Engine> make_engine(const Config& config, std::shared_ptr<Stats> stats, std::size_t host_limit = std::size_t{1} << 30,
                                               std::size_t device_limit = 0, int device = 0) {
    cc::Profile profile;
    profile.context = config.context;
    profile.prefill_chunk = std::min(config.prefill_chunk, config.context);
    profile.max_concurrency = config.max_concurrency;
    profile.max_pending = config.max_concurrency;
    profile.max_batch_tokens = config.max_batch_tokens;
    profile.host_limit = host_limit;
    if(!device_limit) {
        std::size_t free = 0, total = 0;
        pa::check(cudaSetDevice(device), "device");
        pa::check(cudaMemGetInfo(&free, &total), "device memory");
        device_limit = free;
    }
    profile.device_limit = device_limit;
    if(config.prefix_cache) profile.admission_policy = cc::AdmissionPolicy::reuse_aware;
    auto backend = std::make_shared<cc::CudaBackend>(device);
    return std::make_unique<cc::Engine>(profile, backend, [config, stats](const cc::Profile&, cc::Services services) -> std::unique_ptr<cc::Model> {
        return std::make_unique<Model>(config, services, stats);
    });
}

}  // namespace crit::batched_engine
