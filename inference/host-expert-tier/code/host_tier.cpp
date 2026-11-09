// The CPU expert tier; see host_tier.h. Compile without FP contraction (-ffp-contract=off): the SwiGLU and the
// quantizers are host_int's declared functions, called as written.
#include "host_tier.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace crit::techniques::host_tier {
namespace {

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
std::size_t up64(std::size_t n) { return (n + 63) / 64 * 64; }
int block_of(Intermediate kind, int int8_block) { return kind == Intermediate::e2m1x2 ? 16 : int8_block; }

// The rows [r0, r1) of one matrix for a chunk's tokens: the declared integer rows, times the matrix's post or, with
// per-row posts, times 1 and then each row's own (one rounding either way: (s * 1) * p == s * p).
void rows(hi::Isa isa, const hi::IntMatrix& m, const hn::ScaleDecoder& d, const hi::IntActs& acts, int tokens,
          std::size_t r0, std::size_t r1, float post, const float* row_post, float factor, float* out, std::size_t ldo) {
    hi::dot_rows_int(isa, {}, m, d, acts, tokens, 0xffu, r0, r1, row_post ? 1.0f : post, out, ldo);
    if(!row_post) return;
    for(std::size_t r = r0; r < r1; ++r) {
        const float p = factor * row_post[r];
        for(int t = 0; t < tokens; ++t) out[std::size_t(t) * ldo + r] = out[std::size_t(t) * ldo + r] * p;
    }
}

}  // namespace

void interleave_matrix(const hi::IntMatrix& m, std::uint8_t* out) {
    if(m.layout != hi::Layout::rows) throw std::invalid_argument("host_tier: interleave_matrix reads the rows layout");
    const std::size_t stride = m.code_stride ? m.code_stride : m.cols / 2;
    const std::size_t bytes = hi::row_bytes(hi::Layout::interleaved, m.cols);
    for(std::size_t r = 0; r < m.rows; ++r) hi::interleave_row(m.codes + r * stride, m.cols, out + r * bytes);
}

HostExpertTier::HostExpertTier(std::vector<TierExpert> experts, ExpertPool& pool, hi::Isa isa,
                               std::size_t rows_per_task)
    : pool_(pool), isa_(isa), rows_per_task_(std::max<std::size_t>(64, rows_per_task)) {
    if(experts.empty()) throw std::invalid_argument("host_tier: no experts");
    hidden_ = experts[0].gate_up.cols;
    inter_ = experts[0].gate_up.rows / 2;
    if(!hi::int_supported(isa_)) throw std::invalid_argument("host_tier: the integer path is not supported here");
    for(std::size_t i = 0; i < experts.size(); ++i) set_expert(i, experts[i]);
}

HostExpertTier::HostExpertTier(std::size_t hidden, std::size_t inter, ExpertPool& pool, hi::Isa isa,
                               std::size_t rows_per_task)
    : pool_(pool), isa_(isa), rows_per_task_(std::max<std::size_t>(64, rows_per_task)), hidden_(hidden), inter_(inter) {
    if(!hidden_ || !inter_ || hidden_ % 64 || inter_ % 64)
        throw std::invalid_argument("host_tier: hidden and inter are multiples of 64");
    if(!hi::int_supported(isa_)) throw std::invalid_argument("host_tier: the integer path is not supported here");
}

void HostExpertTier::check(const TierExpert& e) const {
    auto block_ok = [](Intermediate kind, int block, float global) {
        return kind == Intermediate::e2m1x2 ? global > 0.0f : (block == 16 || block == 32);
    };
    if(e.gate_up.cols != hidden_ || e.gate_up.rows != 2 * inter_ || e.down.rows != hidden_ || e.down.cols != inter_ ||
       !e.gate_up.codes || !e.down.codes || hidden_ % 64 || inter_ % 64 ||
       !block_ok(e.intermediate, e.intermediate_block, e.intermediate_global) ||
       !block_ok(e.input, e.input_block, e.input_global) ||
       (!e.gate_up_scales.decode && !e.gate_up_scales.byte_table) || (!e.down_scales.decode && !e.down_scales.byte_table))
        throw std::invalid_argument("host_tier: every expert shares [2 * inter][hidden] and [hidden][inter] (multiples of "
                                    "64), with scale decoders and declared quantizers");
}

void HostExpertTier::set_expert(std::size_t index, const TierExpert& expert) {
    check(expert);
    if(index >= experts_.size()) {
        experts_.resize(index + 1);
        present_.resize(index + 1, false);
    }
    experts_[index] = expert;
    present_[index] = true;
}

// Routes grouped by expert in route order, at most 8 tokens a chunk (dot_rows_int's token count).
void HostExpertTier::group(int count, const std::int32_t* expert) {
    chunks_.clear();
    for(int r = 0; r < count; ++r) {
        const int e = expert[r];
        if(e < 0 || !has_expert(std::size_t(e))) throw std::invalid_argument("host_tier: expert id out of range");
        auto found = std::find_if(chunks_.begin(), chunks_.end(),
                                  [&](const Chunk& c) { return c.expert == e && c.routes.size() < hi::kMaxTokens; });
        if(found == chunks_.end()) chunks_.push_back({e, {r}});
        else found->routes.push_back(r);
    }
}

void HostExpertTier::compute(const TierBatch& b) {
    const auto start = std::chrono::steady_clock::now();
    if(b.count < 0 || (b.count > 0 && (!b.expert || !b.values || !b.scales || !b.out)) || b.ld < hidden_ ||
       (b.block != 16 && b.block != 32) || b.lds < hidden_ / std::size_t(b.block) || b.ldo < hidden_)
        throw std::invalid_argument("host_tier: batch geometry");
    if(b.count == 0) return;
    group(b.count, b.expert);
    const std::size_t ldx = up64(hidden_), sblocks = hidden_ / 16;
    in_values_.resize(chunks_.size() * hi::kMaxTokens * ldx);   // only the chunks' tokens' columns are read
    in_scales_.resize(chunks_.size() * hi::kMaxTokens * sblocks);
    chunk_block_.assign(chunks_.size(), b.block);
    for(std::size_t c = 0; c < chunks_.size(); ++c)
        for(std::size_t t = 0; t < chunks_[c].routes.size(); ++t) {
            const std::size_t r = std::size_t(chunks_[c].routes[t]);
            std::memcpy(&in_values_[(c * hi::kMaxTokens + t) * ldx], b.values + r * b.ld, hidden_);
            std::memcpy(&in_scales_[(c * hi::kMaxTokens + t) * sblocks], b.scales + r * b.lds,
                        hidden_ / std::size_t(b.block) * sizeof(float));
        }
    TierStats call;
    call.routes = std::uint64_t(b.count);
    call_ = &call;
    run_chunks(b.out, b.ldo);
    call.ms_total = ms_since(start);
    add_stats(call);
}

void HostExpertTier::compute_inputs(const InputBatch& b) {
    const auto start = std::chrono::steady_clock::now();
    if(b.count < 0 || (b.count > 0 && (!b.expert || !b.token || !b.x || !b.out)) || b.ldx < hidden_ || b.ldo < hidden_)
        throw std::invalid_argument("host_tier: input batch geometry");
    if(b.count == 0) return;
    for(int r = 0; r < b.count; ++r)
        if(b.token[r] < 0) throw std::invalid_argument("host_tier: a route's token row is negative");
    group(b.count, b.expert);
    const std::size_t ldx = up64(hidden_), sblocks = hidden_ / 16;
    in_values_.resize(chunks_.size() * hi::kMaxTokens * ldx);   // only the chunks' tokens' columns are read
    in_scales_.resize(chunks_.size() * hi::kMaxTokens * sblocks);
    chunk_block_.resize(chunks_.size());
    for(std::size_t c = 0; c < chunks_.size(); ++c) {
        const TierExpert& e = experts_[std::size_t(chunks_[c].expert)];
        chunk_block_[c] = block_of(e.input, e.input_block);
    }
    // Each route's row quantized by its expert's declared quantizer, a route per task.
    auto phase = std::chrono::steady_clock::now();
    pool_.run(chunks_.size() * hi::kMaxTokens, [&](std::size_t task) {
        const std::size_t c = task / hi::kMaxTokens, t = task % hi::kMaxTokens;
        if(t >= chunks_[c].routes.size()) return;
        const TierExpert& e = experts_[std::size_t(chunks_[c].expert)];
        const float* x = b.x + std::size_t(b.token[std::size_t(chunks_[c].routes[t])]) * b.ldx;
        std::int8_t* values = &in_values_[(c * hi::kMaxTokens + t) * ldx];
        float* scales = &in_scales_[(c * hi::kMaxTokens + t) * sblocks];
        if(e.input == Intermediate::e2m1x2) hi::quantize_e2m1x2(x, hidden_, e.input_global, values, scales);
        else hi::quantize_int8(x, hidden_, e.input_block, values, scales);
    });
    TierStats call;
    call.ms_inputs = ms_since(phase);
    call.routes = std::uint64_t(b.count);
    call_ = &call;
    run_chunks(b.out, b.ldo);
    call.ms_total = ms_since(start);
    add_stats(call);
}

void HostExpertTier::add_stats(const TierStats& c) {
    const std::lock_guard lock(stats_mutex_);
    stats_.jobs += c.jobs;
    stats_.routes += c.routes;
    stats_.expert_chunks += c.expert_chunks;
    stats_.ms_inputs += c.ms_inputs;
    stats_.ms_gate_up += c.ms_gate_up;
    stats_.ms_down += c.ms_down;
    stats_.ms_total += c.ms_total;
    stats_.weight_bytes += c.weight_bytes;
}

void HostExpertTier::run_chunks(float* out, std::size_t ldo) {
    const std::size_t ldx = up64(hidden_), ldh = up64(inter_), sblocks = hidden_ / 16, hs_ld = inter_ / 16;
    const std::size_t gu_rows = 2 * inter_;
    // Gate/up and the intermediate in one phase: a task takes gate rows [k0, k1) and the up rows beside them, so it
    // has both halves of h[k0..k1) and quantizes them itself (k0, k1 multiples of 32: whole quantizer blocks).
    const std::size_t half = std::max<std::size_t>(32, rows_per_task_ / 2 / 32 * 32);
    const std::size_t gu_parts = (inter_ + half - 1) / half;
    gate_up_out_.resize(chunks_.size() * hi::kMaxTokens * gu_rows);   // written before read, a chunk's tokens only
    h_values_.resize(chunks_.size() * hi::kMaxTokens * ldh);
    h_scales_.resize(chunks_.size() * hi::kMaxTokens * hs_ld);
    auto phase = std::chrono::steady_clock::now();
    pool_.run(chunks_.size() * gu_parts, [&](std::size_t task) {
        const std::size_t c = task / gu_parts, part = task % gu_parts;
        const TierExpert& e = experts_[std::size_t(chunks_[c].expert)];
        const std::size_t k0 = part * half, k1 = std::min(inter_, k0 + half);
        const int tokens = int(chunks_[c].routes.size());
        const hi::IntActs acts{&in_values_[c * hi::kMaxTokens * ldx], ldx, &in_scales_[c * hi::kMaxTokens * sblocks],
                               sblocks, chunk_block_[c]};
        float* gu = &gate_up_out_[c * hi::kMaxTokens * gu_rows];
        rows(isa_, e.gate_up, e.gate_up_scales, acts, tokens, k0, k1, e.gate_up_post, e.gate_up_row_post,
             e.row_post_factor, gu, gu_rows);
        rows(isa_, e.gate_up, e.gate_up_scales, acts, tokens, inter_ + k0, inter_ + k1, e.gate_up_post,
             e.gate_up_row_post, e.row_post_factor, gu, gu_rows);
        thread_local std::vector<float> h;
        h.resize(k1 - k0);
        for(int t = 0; t < tokens; ++t) {
            const float* g = gu + std::size_t(t) * gu_rows;
            for(std::size_t k = k0; k < k1; ++k) h[k - k0] = hi::swiglu_declared(g[k], g[inter_ + k]);
            std::int8_t* values = &h_values_[(c * hi::kMaxTokens + std::size_t(t)) * ldh];
            float* scales = &h_scales_[(c * hi::kMaxTokens + std::size_t(t)) * hs_ld];
            if(e.intermediate == Intermediate::e2m1x2)
                hi::quantize_e2m1x2(h.data(), k1 - k0, e.intermediate_global, values + k0, scales + k0 / 16);
            else
                hi::quantize_int8(h.data(), k1 - k0, e.intermediate_block, values + k0,
                                  scales + k0 / std::size_t(e.intermediate_block));
        }
    });
    call_->ms_gate_up += ms_since(phase);
    // Down rows into every route's output row.
    phase = std::chrono::steady_clock::now();
    const std::size_t dn_parts = (hidden_ + rows_per_task_ - 1) / rows_per_task_;
    pool_.run(chunks_.size() * dn_parts, [&](std::size_t task) {
        const std::size_t c = task / dn_parts, part = task % dn_parts;
        const TierExpert& e = experts_[std::size_t(chunks_[c].expert)];
        const std::size_t r0 = part * rows_per_task_, r1 = std::min(hidden_, r0 + rows_per_task_);
        const int hblock = block_of(e.intermediate, e.intermediate_block);
        const hi::IntActs acts{&h_values_[c * hi::kMaxTokens * ldh], ldh, &h_scales_[c * hi::kMaxTokens * hs_ld], hs_ld,
                               hblock};
        thread_local std::vector<float> partial;
        partial.resize(hi::kMaxTokens * hidden_);   // the task's rows of its tokens, written before they are copied
        const int tokens = int(chunks_[c].routes.size());
        rows(isa_, e.down, e.down_scales, acts, tokens, r0, r1, e.down_post, e.down_row_post, e.row_post_factor,
             partial.data(), hidden_);
        for(int t = 0; t < tokens; ++t)
            std::memcpy(out + std::size_t(chunks_[c].routes[std::size_t(t)]) * ldo + r0,
                        &partial[std::size_t(t) * hidden_ + r0], (r1 - r0) * sizeof(float));
    });
    call_->ms_down += ms_since(phase);
    call_->jobs += 1;
    call_->expert_chunks += chunks_.size();
    for(const auto& c : chunks_) {
        const TierExpert& e = experts_[std::size_t(c.expert)];
        call_->weight_bytes += double(gu_rows) * double(hi::row_bytes(e.gate_up.layout, hidden_) + hidden_ / 16) +
                               double(hidden_) * double(hi::row_bytes(e.down.layout, inter_) + inter_ / 16);
    }
}

HostExpertTier::InputLayout HostExpertTier::input_layout(std::size_t hidden, std::size_t routes, std::size_t tokens) {
    InputLayout p{};
    p.routes = routes;
    p.tokens = tokens;
    p.experts = 64;
    p.token = up64(p.experts + routes * sizeof(std::int32_t));
    p.x = up64(p.token + routes * sizeof(std::int32_t));
    p.out = up64(p.x + tokens * hidden * sizeof(float));
    p.bytes = p.out + routes * hidden * sizeof(float);
    return p;
}

void HostExpertTier::serve_inputs(std::span<std::byte> payload, std::size_t routes, std::size_t tokens) {
    const InputLayout p = input_layout(hidden_, routes, tokens);
    if(payload.size() < p.bytes) throw std::invalid_argument("host_tier: payload smaller than its layout");
    std::uint32_t head[2];
    std::memcpy(head, payload.data(), sizeof(head));
    if(head[0] > routes || head[1] > tokens) throw std::invalid_argument("host_tier: payload route or token count");
    InputBatch b;
    b.count = int(head[0]);
    b.expert = reinterpret_cast<const std::int32_t*>(payload.data() + p.experts);
    b.token = reinterpret_cast<const std::int32_t*>(payload.data() + p.token);
    b.x = reinterpret_cast<const float*>(payload.data() + p.x);
    b.ldx = hidden_;
    b.out = reinterpret_cast<float*>(payload.data() + p.out);
    b.ldo = hidden_;
    for(int r = 0; r < b.count; ++r)
        if(std::uint32_t(b.token[r]) >= head[1]) throw std::invalid_argument("host_tier: a route's token is past the payload's");
    compute_inputs(b);
}

}  // namespace crit::techniques::host_tier
