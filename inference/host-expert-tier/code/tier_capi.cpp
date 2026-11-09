// A C interface over the CPU expert tier (host_tier.h) and its worker pool, for bindings that load the library at run
// time (ctypes), and the host-work-mailbox handler a device engine's dispatch wires to it. Every function returning
// int gives 0 on success or 1 with the error text in `error`.
#include "expert_pool.h"
#include "host_tier.h"
#include "nvfp4.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>

namespace hi = crit::techniques::host_int;
namespace hn = crit::techniques::host_nvfp4;
namespace ht = crit::techniques::host_tier;

namespace {

struct Tier {
    ht::ExpertPool pool;
    ht::HostExpertTier tier;
    Tier(std::size_t hidden, std::size_t inter, ht::PoolOptions options, hi::Isa isa, std::size_t rows_per_task)
        : pool(options), tier(hidden, inter, pool, isa, rows_per_task) {}
};

const float* e4m3_table() {
    static const std::array<float, 256> table = [] {
        std::array<float, 256> t{};
        for(int b = 0; b < 256; ++b) t[std::size_t(b)] = crit::nvfp4::e4m3(std::uint8_t(b));
        return t;
    }();
    return table.data();
}

int fail(const char* what, char* error, std::size_t bytes) {
    if(error && bytes) {
        const std::size_t n = std::min(std::strlen(what), bytes - 1);
        std::memcpy(error, what, n);
        error[n] = 0;
    }
    return 1;
}

template <class Fn> int guarded(char* error, std::size_t bytes, Fn&& fn) {
    try {
        fn();
        return 0;
    } catch(const std::exception& e) {
        return fail(e.what(), error, bytes);
    } catch(...) {
        return fail("host tier: an unknown exception", error, bytes);
    }
}

}  // namespace

extern "C" {

// A tier for experts [2 * inter][hidden] and [hidden][inter] on a pool of `workers` threads (-1: physical cores but the
// caller's), pinned when `pin`, idle workers spinning spin_us (or, with spin_max_us above it, as long as recent gaps
// between layers, at most spin_max_us) then sleeping after sleep_after_us. isa: -1 the fastest
// integer path, else host_isa's Isa (0 scalar, 2 avx2, 3 avx512). Null on failure.
void* crit_tier_create(std::size_t hidden, std::size_t inter, int workers, int pin, int spin_us, int sleep_after_us,
                       int spin_max_us, std::size_t rows_per_task, int isa, char* error, std::size_t error_bytes) {
    void* out = nullptr;
    guarded(error, error_bytes, [&] {
        ht::PoolOptions options;
        options.workers = workers;
        options.pin = pin != 0;
        options.spin = std::chrono::microseconds(std::max(0, spin_us));
        options.sleep_after = std::chrono::microseconds(std::max(spin_us, sleep_after_us));
        options.spin_max = std::chrono::microseconds(std::max(0, spin_max_us));
        out = new Tier(hidden, inter, options, isa < 0 ? hi::best_int_isa() : hi::Isa(isa), rows_per_task);
    });
    return out;
}

void crit_tier_destroy(void* tier) { delete static_cast<Tier*>(tier); }

int crit_tier_workers(void* tier) { return static_cast<Tier*>(tier)->pool.workers(); }
const char* crit_tier_isa(void* tier) { return hn::isa_name(static_cast<Tier*>(tier)->tier.isa()); }

// Expert `index` read in place: gate_up codes [2 * inter][hidden / 2], E4M3 scales [2 * inter][hidden / 16], inverse
// [2 * inter] FP32; down codes [hidden][inter / 2], scales [hidden][inter / 16], inverse [hidden]; post per row =
// factor * inverse[r]. weights: 0 E2M1 x 2, 1 INT4. kinds: 0 E2M1 x 2 under the global divisor, 1 INT8 per block.
int crit_tier_set_expert(void* tier, std::size_t index, const std::uint8_t* gu_codes, const std::uint8_t* gu_scales,
                         const float* gu_inverse, const std::uint8_t* dn_codes, const std::uint8_t* dn_scales,
                         const float* dn_inverse, int weights, float factor, int input_kind, float input_global,
                         int input_block, int inter_kind, float inter_global, int inter_block, char* error,
                         std::size_t error_bytes) {
    return guarded(error, error_bytes, [&] {
        Tier* t = static_cast<Tier*>(tier);
        const std::size_t hidden = t->tier.hidden(), inter = t->tier.inter();
        const hi::Weights kind = weights ? hi::Weights::int4 : hi::Weights::e2m1x2;
        hn::ScaleDecoder decoder;
        decoder.byte_table = e4m3_table();
        ht::TierExpert e;
        e.gate_up = {gu_codes, gu_scales, 2 * inter, hidden, 0, 0, kind, hi::Layout::rows};
        e.down = {dn_codes, dn_scales, hidden, inter, 0, 0, kind, hi::Layout::rows};
        e.gate_up_scales = decoder;
        e.down_scales = decoder;
        e.gate_up_row_post = gu_inverse;
        e.down_row_post = dn_inverse;
        e.row_post_factor = factor;
        e.input = input_kind ? ht::Intermediate::int8 : ht::Intermediate::e2m1x2;
        e.input_global = input_global;
        e.input_block = input_block;
        e.intermediate = inter_kind ? ht::Intermediate::int8 : ht::Intermediate::e2m1x2;
        e.intermediate_global = inter_global;
        e.intermediate_block = inter_block;
        if(!gu_inverse || !dn_inverse) throw std::invalid_argument("host tier: an expert's inverse rows");
        t->tier.set_expert(index, e);
    });
}

// One layer's routes over FP32 token rows x [tokens][ldx]: route r is expert[r] on row token[r]; out [routes][ldo].
int crit_tier_compute(void* tier, int count, const std::int32_t* expert, const std::int32_t* token, const float* x,
                      std::size_t ldx, int tokens, float* out, std::size_t ldo, char* error, std::size_t error_bytes) {
    return guarded(error, error_bytes, [&] {
        for(int r = 0; r < count; ++r)
            if(token[r] < 0 || token[r] >= tokens) throw std::invalid_argument("host tier: a route's token is outside x");
        ht::InputBatch b;
        b.count = count;
        b.expert = expert;
        b.token = token;
        b.x = x;
        b.ldx = ldx;
        b.out = out;
        b.ldo = ldo;
        static_cast<Tier*>(tier)->tier.compute_inputs(b);
    });
}

// jobs, routes, expert chunks, ms inputs, ms gate/up, ms down, ms total, weight bytes, pool batches, pool wakes,
// workers.
void crit_tier_stats(void* tier, double* out) {
    const Tier* t = static_cast<const Tier*>(tier);
    const ht::TierStats& s = t->tier.stats();
    const double v[11] = {double(s.jobs), double(s.routes), double(s.expert_chunks), s.ms_inputs, s.ms_gate_up, s.ms_down,
                          s.ms_total, s.weight_bytes, double(t->pool.batches()), double(t->pool.wakes()),
                          double(t->pool.workers())};
    std::memcpy(out, v, sizeof(v));
}
void crit_tier_reset_stats(void* tier) { static_cast<Tier*>(tier)->tier.reset_stats(); }

// A mailbox payload of FP32 rows for capacities `routes` and `tokens`: offsets of experts, token, x, out and its bytes.
void crit_tier_payload_layout(std::size_t hidden, std::size_t routes, std::size_t tokens, std::int64_t* out) {
    const auto p = ht::HostExpertTier::input_layout(hidden, routes, tokens);
    out[0] = std::int64_t(p.experts);
    out[1] = std::int64_t(p.token);
    out[2] = std::int64_t(p.x);
    out[3] = std::int64_t(p.out);
    out[4] = std::int64_t(p.bytes);
}

// The mailbox handler: the payload's header is {count, tokens, route capacity, token capacity} (uint32), then
// input_layout(hidden, route capacity, token capacity). 0: served (every route's row written); 1: the error in `error`.
int crit_tier_serve(void* tier, std::uint64_t, void* payload, std::uint64_t bytes, char* error, std::uint64_t error_bytes) {
    return guarded(error, std::size_t(error_bytes), [&] {
        if(!tier || !payload || bytes < 16) throw std::invalid_argument("host tier: payload");
        std::uint32_t head[4];
        std::memcpy(head, payload, sizeof(head));
        static_cast<Tier*>(tier)->tier.serve_inputs(std::span<std::byte>(static_cast<std::byte*>(payload), bytes), head[2],
                                                     head[3]);
    });
}

}  // extern "C"
