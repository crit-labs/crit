// A ready CPU expert tier: a routed layer's misses computed on the CPU in host-expert-arithmetic's integer block order
// (host_int.h), so a CPU-computed expert gives the device's bits and the split between the tiers never changes a
// token. One call computes one layer's CPU routes: each distinct expert's gate/up rows once for all of its tokens,
// the declared SwiGLU, the intermediate quantized as declared, then the down rows; every route's unweighted output
// row, which the device sums with its routing weights in its own route order (async-expert-contributions). The work
// is split into row ranges of experts over an ExpertPool, so a layer with a few experts streams each at every
// core's rate; any split gives the same bits.
//
// Inputs come quantized (compute: a TierBatch of int8 values and block scales, as a device quantized them) or as FP32
// token rows (compute_inputs: an InputBatch), which the tier quantizes per route with that route's expert's declared
// quantizer and global (TierExpert::input): the same codes the device's quantizer gives, so either form gives the same
// bits. Experts live in a table the engine fills (set_expert), each a view of bytes the engine owns: a bundle in a
// host store read in place (the rows layout as the checkpoint stores it), or the interleaved layout of a CPU-only store.
// Posts are per matrix or per row (row_post): a row is computed with post 1 and then multiplied by its own, which is
// one rounding, the device's `v * post[n]`.
//
// Wiring (README): the device expert cache's admission limits (a quota by miss count from split.py on this tier's
// measured times) send the misses past the link's share here; the device posts them through a host-work-mailbox slot,
// the handler calls compute_inputs() on the service thread, and the stream's join (measure_joins()) waits only where
// the layer reads the rows.
#pragma once
#include "expert_pool.h"
#include "host_int.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

namespace crit::techniques::host_tier {

namespace hi = crit::techniques::host_int;
namespace hn = crit::techniques::host_nvfp4;

// How a matrix's input is quantized, as the device quantizes it: E2M1 x 2 per 16 under a global divisor (W4A4), or
// INT8 per block of 16 or 32 (W4A8).
enum class Intermediate { e2m1x2, int8 };

// One expert as the tier reads it: a fused gate/up matrix [2 * inter][hidden] (gate rows first) and a down matrix
// [hidden][inter], each with its block scale decoder and post factor (value = integer x scales x post), or a post per
// row (gate_up_row_post [2 * inter], down_row_post [hidden]; null: the matrix's post). A row's post is
// row_post_factor * row_post[n], rounded once (a bundle stores 1 / global per row; the factor is the integer values'
// 0.25 or 1), as the device computes post[n].
struct TierExpert {
    hi::IntMatrix gate_up, down;
    hn::ScaleDecoder gate_up_scales, down_scales;
    float gate_up_post = 1.0f, down_post = 1.0f;
    const float* gate_up_row_post = nullptr;
    const float* down_row_post = nullptr;
    float row_post_factor = 1.0f;
    Intermediate intermediate = Intermediate::e2m1x2;
    float intermediate_global = 1.0f;   // e2m1x2: crit::nvfp4's divisor of the down projection's input
    int intermediate_block = 32;        // int8: 16 or 32
    // compute_inputs() only: how the expert's FP32 input is quantized (the gate/up projection's input).
    Intermediate input = Intermediate::e2m1x2;
    float input_global = 1.0f;          // e2m1x2: the gate/up input's divisor
    int input_block = 32;               // int8: 16 or 32
};

// One layer's CPU routes, quantized. Route r: expert `expert[r]`, its token's quantized input at values + r * ld (int8,
// `hidden` of them) with scales at scales + r * lds (one per `block`), its unweighted output row at out + r * ldo (FP32).
struct TierBatch {
    int count = 0;
    const std::int32_t* expert = nullptr;
    const std::int8_t* values = nullptr;
    std::size_t ld = 0;
    const float* scales = nullptr;
    std::size_t lds = 0;
    int block = 16;
    float* out = nullptr;
    std::size_t ldo = 0;
};

// One layer's CPU routes over FP32 token rows. Route r: expert `expert[r]`, input row x + token[r] * ldx (`hidden`
// floats), quantized here by the expert's `input`; its unweighted output row at out + r * ldo.
struct InputBatch {
    int count = 0;
    const std::int32_t* expert = nullptr;
    const std::int32_t* token = nullptr;
    const float* x = nullptr;
    std::size_t ldx = 0;
    float* out = nullptr;
    std::size_t ldo = 0;
};

struct TierStats {
    std::uint64_t jobs = 0, routes = 0, expert_chunks = 0;
    double ms_inputs = 0, ms_gate_up = 0, ms_down = 0, ms_total = 0;
    double weight_bytes = 0;   // gate/up and down bytes read (codes and scales), each expert once a call
};

class HostExpertTier {
public:
    // isa: the integer path (default the fastest this CPU runs). rows_per_task: rows of one expert in a pool task.
    HostExpertTier(std::vector<TierExpert> experts, ExpertPool& pool, hi::Isa isa = hi::best_int_isa(),
                   std::size_t rows_per_task = 128);
    // An empty table for experts of [2 * inter][hidden] and [hidden][inter], filled by set_expert.
    HostExpertTier(std::size_t hidden, std::size_t inter, ExpertPool& pool, hi::Isa isa = hi::best_int_isa(),
                   std::size_t rows_per_task = 128);
    // Expert `index` of the table (grown as needed). Throws std::invalid_argument for other shapes.
    void set_expert(std::size_t index, const TierExpert& expert);
    bool has_expert(std::size_t index) const noexcept { return index < present_.size() && present_[index]; }
    std::size_t experts() const noexcept { return experts_.size(); }

    void compute(const TierBatch& batch);
    void compute_inputs(const InputBatch& batch);
    // A copy of the counters (taken under their lock: a mailbox's service thread computes while others read).
    TierStats stats() const {
        const std::lock_guard lock(stats_mutex_);
        return stats_;
    }
    void reset_stats() {
        const std::lock_guard lock(stats_mutex_);
        stats_ = {};
    }
    std::size_t hidden() const noexcept { return hidden_; }
    std::size_t inter() const noexcept { return inter_; }
    hi::Isa isa() const noexcept { return isa_; }

    // A host-work-mailbox payload of FP32 token rows (InputBatch): header {count, tokens, route capacity, token
    // capacity} (four uint32; serve_inputs reads the first two), experts int32 [routes], token
    // int32 [routes], x FP32 [tokens][hidden], out FP32 [routes][hidden], each part 64-byte aligned; `routes` and
    // `tokens` its capacities.
    struct InputLayout {
        std::size_t experts, token, x, out, bytes, routes, tokens;
    };
    static InputLayout input_layout(std::size_t hidden, std::size_t routes, std::size_t tokens);
    void serve_inputs(std::span<std::byte> payload, std::size_t routes, std::size_t tokens);

private:
    void check(const TierExpert& e) const;
    void group(int count, const std::int32_t* expert);
    // The two pooled phases over chunks_ once their inputs are in in_values_ / in_scales_ (block chunk_block_).
    void run_chunks(float* out, std::size_t ldo);
    std::vector<TierExpert> experts_;
    std::vector<bool> present_;
    ExpertPool& pool_;
    hi::Isa isa_;
    std::size_t rows_per_task_, hidden_ = 0, inter_ = 0;
    TierStats stats_;
    mutable std::mutex stats_mutex_;
    TierStats* call_ = nullptr;   // the counters of the call in flight, added to stats_ once at its end
    void add_stats(const TierStats& call);
    // Per call: routes grouped by expert in chunks of up to 8 tokens, their gathered inputs and intermediates.
    struct Chunk {
        int expert;
        std::vector<int> routes;
    };
    std::vector<Chunk> chunks_;
    std::vector<int> chunk_block_;   // each chunk's input block (a chunk's tokens share its expert's quantizer)
    std::vector<std::int8_t> in_values_, h_values_;
    std::vector<float> in_scales_, h_scales_, gate_up_out_;
};

// One matrix's codes from the rows layout into the interleaved one (host_int.h Layout::interleaved): the store a CPU
// tier reads. `out` holds rows * row_bytes(interleaved, cols) bytes.
void interleave_matrix(const hi::IntMatrix& rows_layout, std::uint8_t* out);

}  // namespace crit::techniques::host_tier
