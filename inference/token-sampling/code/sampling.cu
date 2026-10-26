#include "sampling.h"

#include <crit/sampling.h>

#include <cfloat>
#include <climits>

namespace crit::techniques::token_sampling {
namespace {

using u64 = unsigned long long;

constexpr int kLaw = 1024;              // one block of kLaw threads computes one row's full law
constexpr int kWarps = kLaw / 32;
constexpr int kCand = kMaxTopK + 1;     // fast-path candidates per block and row: K + 1 for the exact boundary checks
// Fast-path stage blocks: kStageThreads threads x kStageItems logits, one contiguous chunk of a row per block, so a row
// takes ceil(vocab / kChunk) blocks. Law blocks: one per row, merging kMergeItems - 1 new candidates per thread and round.
constexpr int kStageThreads = 256;
constexpr int kStageItems = 16;
constexpr int kChunk = kStageThreads * kStageItems;
constexpr int kMergeThreads = 256;
constexpr int kMergeItems = 17;
constexpr std::uint32_t kBiasCap = 1024;   // Core's logit_bias_limit; longer device lists are read from global memory
constexpr std::uint32_t kListCap = 256;    // staged and suppressed ids held in shared memory

__host__ __device__ constexpr int blocks_per_row(int vocab) { return (vocab + kChunk - 1) / kChunk; }

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ bool masked(float v) { return v == neg_inf(); }

// ---- orderable keys -------------------------------------------------------------------------------
// Unsigned keys that order like the floats (-inf lowest).
__device__ __forceinline__ std::uint32_t order_key(float v) {
    const std::uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float key_float(std::uint32_t k) { return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }
// Packed rank key: a larger key ranks earlier (value descending, then id ascending). `inexact` marks a double value
// rounded down to the float: it ranks before an exact float of the same value, as the double does. 0 is the empty key
// (no real key is 0: the lowest, -inf's, is 0x007fffff).
__device__ __forceinline__ u64 pack(std::uint32_t key, std::uint32_t inexact, std::int32_t id) {
    return (u64(key) << 32) | (u64(inexact & 1u) << 31) | u64(0x7fffffffu - std::uint32_t(id));
}
__device__ __forceinline__ u64 raw_key(float x, std::int32_t id) { return pack(order_key(x), 0u, id); }
__device__ __forceinline__ std::int32_t key_id(u64 k) { return std::int32_t(0x7fffffffu - std::uint32_t(k & 0x7fffffffull)); }
__device__ __forceinline__ float key_value(u64 k) { return key_float(std::uint32_t(k >> 32)); }
__device__ __forceinline__ u64 umax(u64 a, u64 b) { return a > b ? a : b; }
__device__ __forceinline__ u64 umin(u64 a, u64 b) { return a < b ? a : b; }
// Two keys of distinct doubles rounded down to the same float: the key order between them is by id, not by value.
__device__ __forceinline__ bool same_inexact(u64 a, u64 b) { return a && b && (a >> 31) == (b >> 31) && ((a >> 31) & 1ull); }

// ---- Core's steps 2 to 5 --------------------------------------------------------------------------
// One row's lists in shared memory: bias entries (ids ascending), staged ids, suppressed ids (sorted here).
struct Lists {
    std::int32_t bias_id[kBiasCap];
    float bias_value[kBiasCap];
    std::int32_t staged[kListCap];
    std::int32_t suppress_in[kListCap];
    std::int32_t suppress[kListCap];
    std::int32_t path[64];               // a tree node's staged path (view_of)
};

// One row's processing: pointers into Lists, or global memory for lists longer than the shared ones.
struct Proc {
    const std::int32_t* bias_id = nullptr;
    const float* bias_value = nullptr;
    std::uint32_t bias_count = 0;
    const std::int32_t* staged = nullptr;
    std::uint32_t staged_count = 0;
    const std::int32_t* suppress = nullptr;
    std::uint32_t suppress_count = 0;
    bool suppress_sorted = false;
    bool penalties = false;
    const std::int32_t* counts = nullptr;
    const std::uint8_t* seen = nullptr;
    double repetition = 1., frequency = 0., presence = 0.;
};

// Every thread of the block calls it with the same arguments. It starts with a barrier (a tree path written just
// before) and ends with one: the lists are readable on return.
__device__ Proc load_proc(Lists& ls, const Request& q, const std::int32_t* staged, std::uint32_t staged_count,
                          bool penalties, bool suppressing) {
    __syncthreads();
    Proc P;
    P.bias_count = q.bias_count;
    P.bias_id = q.bias_ids;
    P.bias_value = q.bias_values;
    if(q.bias_count <= kBiasCap) {
        for(std::uint32_t i = threadIdx.x; i < q.bias_count; i += blockDim.x) {
            ls.bias_id[i] = q.bias_ids[i];
            ls.bias_value[i] = q.bias_values[i];
        }
        P.bias_id = ls.bias_id;
        P.bias_value = ls.bias_value;
    }
    if(penalties && staged_count) {
        P.staged_count = staged_count;
        P.staged = staged;
        if(staged_count <= kListCap) {
            for(std::uint32_t i = threadIdx.x; i < staged_count; i += blockDim.x) ls.staged[i] = staged[i];
            P.staged = ls.staged;
        }
    }
    if(suppressing && q.suppress_count) {
        P.suppress_count = q.suppress_count;
        P.suppress = q.suppress_ids;
        if(q.suppress_count <= kListCap)
            for(std::uint32_t i = threadIdx.x; i < q.suppress_count; i += blockDim.x) ls.suppress_in[i] = q.suppress_ids[i];
    }
    P.penalties = penalties;
    P.counts = q.counts;
    P.seen = q.prompt_seen;
    P.repetition = q.repetition;
    P.frequency = q.frequency;
    P.presence = q.presence;
    __syncthreads();
    if(P.suppress_count && P.suppress_count <= kListCap) {
        // Sorted by counting (ties by position), so every token's test is a binary search.
        const std::uint32_t n = P.suppress_count;
        for(std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            const std::int32_t x = ls.suppress_in[i];
            std::uint32_t rank = 0;
            for(std::uint32_t j = 0; j < n; ++j) rank += ls.suppress_in[j] < x || (ls.suppress_in[j] == x && j < i);
            ls.suppress[rank] = x;
        }
        P.suppress = ls.suppress;
        P.suppress_sorted = true;
        __syncthreads();
    }
    return P;
}

// Core's steps 2 to 5 for token t of raw logit x in IEEE double as Core computes them (core/src/sampling/sampling.cpp,
// prepare and penalize; explicit rounding, no contraction): bias, min_tokens suppression, repetition over the prompt
// and outputs, frequency and presence over committed plus staged outputs. touched: a step changed or masked the value.
__device__ double process_value(const Proc& P, std::int32_t t, float x, bool& touched) {
    double v = double(x);
    touched = false;
    if(P.bias_count) {
        std::uint32_t lo = 0, hi = P.bias_count;
        while(lo < hi) {
            const std::uint32_t mid = (lo + hi) >> 1;
            if(P.bias_id[mid] < t) lo = mid + 1;
            else hi = mid;
        }
        if(lo < P.bias_count && P.bias_id[lo] == t) {
            v = __dadd_rn(v, double(P.bias_value[lo]));
            touched = true;
        }
    }
    if(P.suppress_count) {
        bool hit = false;
        if(P.suppress_sorted) {
            std::uint32_t lo = 0, hi = P.suppress_count;
            while(lo < hi) {
                const std::uint32_t mid = (lo + hi) >> 1;
                if(P.suppress[mid] < t) lo = mid + 1;
                else hi = mid;
            }
            hit = lo < P.suppress_count && P.suppress[lo] == t;
        } else {
            for(std::uint32_t i = 0; i < P.suppress_count && !hit; ++i) hit = P.suppress[i] == t;
        }
        if(hit) { touched = true; return double(neg_inf()); }
    }
    if(v == double(neg_inf()) || !P.penalties) return v;
    std::int32_t n = P.counts ? P.counts[t] : 0;
    for(std::uint32_t i = 0; i < P.staged_count; ++i) n += P.staged[i] == t;
    if(P.repetition != 1. && (n > 0 || (P.seen && P.seen[t]))) {
        v = v > 0. ? fmin(__ddiv_rn(v, P.repetition), DBL_MAX) : fmax(__dmul_rn(v, P.repetition), -DBL_MAX);
        touched = true;
    }
    if(n > 0) {
        v = __dsub_rn(v, __dadd_rn(__dmul_rn(P.frequency, double(n)), P.presence));
        touched = true;
    }
    return v;
}

// A processed value for the FP32 full law: the nearest float, clamped to the finite range, so only -inf masks.
__device__ __forceinline__ float to_float(double v) {
    return v == double(neg_inf()) ? neg_inf() : float(fmin(fmax(v, -double(FLT_MAX)), double(FLT_MAX)));
}

// ---- The full-vocabulary law --------------------------------------------------------------------
// Rank order is (value descending, id ascending). Values map to unsigned keys that order like the
// floats, so a rank prefix is "key above a boundary, or equal key and id at most the boundary id".
// Prefix searches walk the key's eight 4-bit digits from the top, summing weights per digit, then
// resolve the boundary key's ties in id order. Every sum has a fixed order for a given vocabulary
// (per-thread FP32 over a fixed stride, then FP64 over threads), so a row's result never depends on
// timing or on the other rows of the launch.

// A sparse draft law (SparseLaw) sorted by id in shared memory, so q(t) is a binary search.
struct SparseQ {
    std::int32_t id[kMaxTopK];
    float p[kMaxTopK];
    int n;
};

// Block-wide (law.count > 0); ends with a barrier.
__device__ void load_sparse(const SparseLaw& law, SparseQ& q) {
    const int n = min(int(law.count), kMaxTopK), i = int(threadIdx.x);
    if(i < n) {
        const std::int32_t t = law.ids[i];
        int rank = 0;
        for(int j = 0; j < n; ++j) rank += law.ids[j] < t || (law.ids[j] == t && j < i);
        q.id[rank] = t;
        q.p[rank] = law.probabilities[i];
    }
    if(i == 0) q.n = n;
    __syncthreads();
}

__device__ __forceinline__ float sparse_at(const SparseQ& q, std::int32_t t) {
    int lo = 0, hi = q.n;
    while(lo < hi) {
        const int mid = (lo + hi) >> 1;
        if(q.id[mid] < t) lo = mid + 1;
        else hi = mid;
    }
    return lo < q.n && q.id[lo] == t ? q.p[lo] : 0.f;
}

// How a verify column's draft is checked: 0 the target's own draw (greedy drafts, cut columns: a draft id outside the
// vocabulary or an empty sparse law, and a full-law draft without a dense row), 1 against its sparse law, 2 against
// its dense row. Every thread reads the same value.
__device__ __forceinline__ int draft_mode(const Verify& v, std::size_t col, std::int32_t d, int vocab) {
    if(d < 0 || d >= vocab) return 0;
    const std::int32_t count = v.draft_laws ? v.draft_laws[col].count : -1;
    if(count > 0) return 1;
    return count < 0 && v.draft_probabilities ? 2 : 0;
}

struct Shared {
    double part[kWarps][16];
    double total[16];
    std::uint32_t key[kWarps];
    std::int32_t id[kWarps];
    std::int32_t value[kWarps];
    std::int32_t result;
};

__device__ double sum_block(double x, Shared& s) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for(int o = 16; o; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
    if(lane == 0) s.part[warp][0] = x;
    __syncthreads();
    if(threadIdx.x == 0) {
        double t = 0.;
        for(int w = 0; w < kWarps; ++w) t += s.part[w][0];
        s.total[0] = t;
    }
    __syncthreads();
    return s.total[0];
}

__device__ void sum16(const float (&acc)[16], Shared& s) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for(int b = 0; b < 16; ++b) {
        double x = acc[b];
        for(int o = 16; o; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
        if(lane == 0) s.part[warp][b] = x;
    }
    __syncthreads();
    if(threadIdx.x < 16) {
        double t = 0.;
        for(int w = 0; w < kWarps; ++w) t += s.part[w][threadIdx.x];
        s.total[threadIdx.x] = t;
    }
    __syncthreads();
}

// op 0: sum, op 1: minimum.
__device__ int int_block(int x, int op, Shared& s) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for(int o = 16; o; o >>= 1) {
        const int y = __shfl_xor_sync(0xffffffffu, x, o);
        x = op ? min(x, y) : x + y;
    }
    if(lane == 0) s.value[warp] = x;
    __syncthreads();
    if(threadIdx.x == 0) {
        int r = s.value[0];
        for(int w = 1; w < kWarps; ++w) r = op ? min(r, s.value[w]) : r + s.value[w];
        s.result = r;
    }
    __syncthreads();
    const int r = s.result;
    __syncthreads();
    return r;
}

// Whether (k, i) is preferred over (k2, i2): earlier in rank order, or later when `last`. id < 0 is empty.
__device__ __forceinline__ bool prefer(bool last, std::uint32_t k, std::int32_t i, std::uint32_t k2, std::int32_t i2) {
    if(i < 0) return false;
    if(i2 < 0) return true;
    const bool earlier = k > k2 || (k == k2 && i < i2);
    return last ? !earlier && i != i2 : earlier;
}

__device__ void rank_block(bool last, std::uint32_t& k, std::int32_t& i, Shared& s) {
    for(int o = 16; o; o >>= 1) {
        const std::uint32_t ok = __shfl_xor_sync(0xffffffffu, k, o);
        const std::int32_t oi = __shfl_xor_sync(0xffffffffu, i, o);
        if(prefer(last, ok, oi, k, i)) { k = ok; i = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if(lane == 0) { s.key[warp] = k; s.id[warp] = i; }
    __syncthreads();
    if(threadIdx.x == 0) {
        for(int w = 1; w < kWarps; ++w)
            if(prefer(last, s.key[w], s.id[w], k, i)) { k = s.key[w]; i = s.id[w]; }
        s.key[0] = k;
        s.id[0] = i;
    }
    __syncthreads();
    k = s.key[0];
    i = s.id[0];
    __syncthreads();
}

// The first token in rank order at which the cumulative weight exceeds `target` (reaches it when
// `reach`). Weight(t, v) is the token's weight, zero for tokens outside the walk. false when the
// whole mass stays below the target (rounding at the last bin).
template<class Weight>
__device__ bool descend(const float* values, int vocab, Weight weight, double target, bool reach, Shared& s,
                        std::int32_t& out) {
    std::uint32_t prefix = 0, mask = 0;
    double above = 0.;
    for(int shift = 28; shift >= 0; shift -= 4) {
        float acc[16];
#pragma unroll
        for(int b = 0; b < 16; ++b) acc[b] = 0.f;
        for(int t = threadIdx.x; t < vocab; t += kLaw) {
            const float v = values[t];
            const std::uint32_t k = order_key(v);
            if((k & mask) != prefix) continue;
            const float w = weight(t, v);
            if(!(w > 0.f)) continue;
            const std::uint32_t digit = (k >> shift) & 15u;
#pragma unroll
            for(int b = 0; b < 16; ++b)
                if(digit == std::uint32_t(b)) acc[b] += w;
        }
        sum16(acc, s);
        int pick = -1;
        double cum = above;
        for(int b = 15; b >= 0; --b) {
            const double mass = s.total[b];
            if(!(mass > 0.)) continue;
            if(reach ? cum + mass >= target : cum + mass > target) { pick = b; break; }
            cum += mass;
        }
        if(pick < 0) return false;
        above = cum;
        prefix |= std::uint32_t(pick) << shift;
        mask |= 15u << shift;
    }
    // The boundary key's tokens in id order, one block-wide inclusive scan per tile of kLaw ids.
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    double carry = above;
    for(int base = 0; base < vocab; base += kLaw) {
        const int t = base + int(threadIdx.x);
        float w = 0.f;
        if(t < vocab) {
            const float v = values[t];
            if(order_key(v) == prefix) w = fmaxf(weight(t, v), 0.f);
        }
        double x = w;
        for(int o = 1; o < 32; o <<= 1) {
            const double y = __shfl_up_sync(0xffffffffu, x, o);
            if(lane >= o) x += y;
        }
        if(lane == 31) s.part[warp][0] = x;
        __syncthreads();
        if(threadIdx.x == 0) {
            double run = 0.;
            for(int i = 0; i < kWarps; ++i) {
                const double sum = s.part[i][0];
                s.part[i][1] = run;
                run += sum;
            }
            s.total[0] = run;
        }
        __syncthreads();
        const double cum = carry + s.part[warp][1] + x;
        const bool hit = w > 0.f && (reach ? cum >= target : cum > target);
        const double tile = s.total[0];
        const int first = int_block(hit ? t : INT_MAX, 1, s);
        if(first != INT_MAX) { out = first; return true; }
        carry += tile;
    }
    return false;
}

// The last token in rank order with a positive weight, -1 when none has one.
template<class Weight>
__device__ std::int32_t last_positive(const float* values, int vocab, Weight weight, Shared& s) {
    std::uint32_t bk = 0;
    std::int32_t bi = -1;
    for(int t = threadIdx.x; t < vocab; t += kLaw) {
        const float v = values[t];
        if(!(weight(t, v) > 0.f)) continue;
        const std::uint32_t k = order_key(v);
        if(prefer(true, k, t, bk, bi)) { bk = k; bi = t; }
    }
    rank_block(true, bk, bi, s);
    return bi;
}

template<class Weight>
__device__ double mass(const float* values, int vocab, Weight weight, Shared& s) {
    float local = 0.f;
    for(int t = threadIdx.x; t < vocab; t += kLaw) local += weight(t, values[t]);
    return sum_block(double(local), s);
}

// Core's steps 6 to 11 over processed values.
struct Law {
    std::int32_t argmax = -1;       // -1: nothing selectable
    bool greedy = true;
    float top = 0.f, temperature = 1.f;
    bool bounded = false;           // kept tokens are the rank prefix ending at (bound_key, bound_id)
    std::uint32_t bound_key = 0;
    std::int32_t bound_id = 0;
    double min_weight = 0.;         // min_p (Core's rule in double), applied when positive
    double total = 0.;              // weight of the kept tokens
};

__device__ __forceinline__ float weight_of(const Law& L, float v) { return expf((v - L.top) / L.temperature); }

__device__ __forceinline__ bool in_prefix(const Law& L, std::uint32_t k, std::int32_t t) {
    return !L.bounded || k > L.bound_key || (k == L.bound_key && t <= L.bound_id);
}

__device__ __forceinline__ float prefix_weight(const Law& L, std::int32_t t, float v) {
    return masked(v) || !in_prefix(L, order_key(v), t) ? 0.f : weight_of(L, v);
}

__device__ __forceinline__ float kept_weight(const Law& L, std::int32_t t, float v) {
    const float w = prefix_weight(L, t, v);
    return L.min_weight > 0. && !(double(w) >= L.min_weight) ? 0.f : w;
}

// The rules are Core's doubles: top_p and min_p compare as Core compares them; the weights are FP32 at float(T).
__device__ Law compute_law(const float* values, int vocab, double temperature, std::int32_t top_k, double top_p,
                           double min_p, Shared& s) {
    Law L;
    std::uint32_t bk = 0;
    std::int32_t bi = -1;
    int selectable = 0;
    for(int t = threadIdx.x; t < vocab; t += kLaw) {
        const float v = values[t];
        if(masked(v)) continue;
        ++selectable;
        const std::uint32_t k = order_key(v);
        if(prefer(false, k, t, bk, bi)) { bk = k; bi = t; }
    }
    rank_block(false, bk, bi, s);
    selectable = int_block(selectable, 0, s);
    L.argmax = bi;
    if(bi < 0) return L;
    L.top = values[bi];
    if(!(temperature > 0.)) return L;
    L.greedy = false;
    L.temperature = float(temperature);
    if(top_k > 0 && top_k < selectable) {
        std::int32_t id = -1;
        if(descend(values, vocab, [](std::int32_t, float v) { return masked(v) ? 0.f : 1.f; }, double(top_k), true, s, id)) {
            L.bounded = true;
            L.bound_id = id;
            L.bound_key = order_key(values[id]);
        }
    }
    if(top_p < 1.) {
        const Law prefix = L;
        auto w = [&prefix](std::int32_t t, float v) { return prefix_weight(prefix, t, v); };
        const double S = mass(values, vocab, w, s);
        std::int32_t id = -1;
        if(descend(values, vocab, w, top_p * S, true, s, id)) {
            L.bounded = true;
            L.bound_id = id;
            L.bound_key = order_key(values[id]);
        }
    }
    L.min_weight = min_p > 0. ? min_p : 0.;
    const Law kept = L;
    L.total = mass(values, vocab, [&kept](std::int32_t t, float v) { return kept_weight(kept, t, v); }, s);
    return L;
}

// Core's step 12: the first kept token whose cumulative weight exceeds u times the kept weight.
template<class Weight>
__device__ std::int32_t walk(const float* values, int vocab, Weight weight, double target, Shared& s) {
    std::int32_t id = -1;
    if(descend(values, vocab, weight, target, false, s, id)) return id;
    return last_positive(values, vocab, weight, s);
}

__device__ std::int32_t draw(const float* values, int vocab, const Law& L, double u, Shared& s) {
    if(L.argmax < 0) return -1;
    if(L.greedy) return L.argmax;
    return walk(values, vocab, [&L](std::int32_t t, float v) { return kept_weight(L, t, v); }, u * L.total, s);
}

// Processed values of one row (Core's steps 2 to 5), rounded for the FP32 law. Block-wide; ends with a barrier.
__device__ void process_row(const float* logits, float* out, int vocab, const Proc& P) {
    for(int t = threadIdx.x; t < vocab; t += kLaw) {
        bool touched = false;
        out[t] = to_float(process_value(P, t, logits[t], touched));
    }
    __syncthreads();
}

struct RowView {
    const Request* q;
    const float* logits;
    float* values;
    const std::int32_t* staged;
    std::uint32_t staged_count;
    std::uint64_t position;
    std::size_t row;
};

// Row (r, c)'s staged tokens and draw position. `path` is shared memory for up to 63 staged tree tokens written by
// threads 1..63; load_proc's first barrier orders the writes before any read.
__device__ RowView view_of(const Rows& rows, std::uint32_t r, std::uint32_t c, const std::int32_t* fallback_staged,
                           std::uint32_t fallback_stride, std::int32_t* path) {
    RowView v;
    v.q = rows.request + r;
    v.row = std::size_t(r) * rows.columns + c;
    v.logits = rows.logits + v.row * rows.stride;
    v.values = rows.scratch + v.row * rows.vocab;
    if(rows.tree) {
        const std::uint64_t mask = rows.tree[v.row];
        const std::int32_t* tokens = rows.tree_tokens + std::size_t(r) * rows.columns;
        const std::uint32_t j = threadIdx.x;
        if(j >= 1 && j < 64 && ((mask >> j) & 1ull)) path[__popcll(mask & ((1ull << j) - 1ull) & ~1ull)] = tokens[j];
        v.staged = path;
        v.staged_count = std::uint32_t(__popcll(mask & ~1ull));
        v.position = v.q->position + rows.position_base + v.staged_count;
        return v;
    }
    if(rows.staged) {
        v.staged = rows.staged + std::size_t(r) * rows.staged_stride;
        v.staged_count = rows.staged_base + c;
    } else {
        v.staged = fallback_staged ? fallback_staged + std::size_t(r) * fallback_stride : nullptr;
        v.staged_count = fallback_staged ? c : 0;
    }
    v.position = v.q->position + rows.position_base + c;
    return v;
}

__device__ RowView view(const Rows& rows, const std::int32_t* fallback_staged, std::uint32_t fallback_stride, std::int32_t* path) {
    return view_of(rows, blockIdx.y, blockIdx.x, fallback_staged, fallback_stride, path);
}

__device__ __forceinline__ bool has_penalties(const Request& q) {
    return q.repetition != 1. || q.frequency != 0. || q.presence != 0.;
}

// ---- The fast path ----------------------------------------------------------------------------------
// Greedy rows and 1 <= top_k <= kMaxTopK: stage kernels (ceil(vocab / kChunk) blocks per row) write each block's
// top K = top_k + 1 packed keys of the processed values (block_top: exact, ties by id, no atomics); a law kernel (one
// block per row) merges them exactly, recomputes the candidates' processed values in double and runs Core's steps 7
// to 12 on them in double and in Core's summation order (rank order with top_k, id order when the row has at most
// top_k selectable tokens), the verify rule included. When the (top_k + 1)-th candidate exists the row has more than
// top_k selectable tokens, so Core's filter applies. Keys order like the doubles except between distinct doubles that
// round down to one float (both inexact): when such a pair straddles the boundary the row is left to the full law.
//
// Open rows (temperature > 0 without a top_k filter: top_k 0, or at least the vocabulary) take the same kernels with
// the exact top kCand candidates plus the whole row's weight at the row's temperature (FP32 exps, FP64 sums: Core's
// total to about 1e-7). They are settled there when the kept set closes within the first kMaxTopK candidates (top_p or
// min_p ends it, or the row has at most kMaxTopK selectable tokens), or, without any filter, when the draw lands among
// them. Every other row is left to crit's full law through RowResult::handled; rows with kMaxTopK < top_k < vocab go
// there at once (their stage and law blocks exit).

template<int THREADS> struct Reduce {
    static constexpr int kWarps = THREADS / 32;
    double d[kWarps];
    float f[kWarps];
    int i[kWarps];
};

// Sum over the block: a fixed warp tree, then the warps in order; every thread returns the same bits.
template<int THREADS> __device__ double block_sum(double x, Reduce<THREADS>& s) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for(int o = 16; o; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
    if(lane == 0) s.d[warp] = x;
    __syncthreads();
    double t = s.d[0];
#pragma unroll
    for(int w = 1; w < Reduce<THREADS>::kWarps; ++w) t += s.d[w];
    __syncthreads();
    return t;
}

template<int THREADS> __device__ float block_max(float x, Reduce<THREADS>& s) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for(int o = 16; o; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
    if(lane == 0) s.f[warp] = x;
    __syncthreads();
    float t = s.f[0];
#pragma unroll
    for(int w = 1; w < Reduce<THREADS>::kWarps; ++w) t = fmaxf(t, s.f[w]);
    __syncthreads();
    return t;
}

template<int THREADS> __device__ double block_max_double(double x, Reduce<THREADS>& s) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for(int o = 16; o; o >>= 1) x = fmax(x, __shfl_xor_sync(0xffffffffu, x, o));
    if(lane == 0) s.d[warp] = x;
    __syncthreads();
    double t = s.d[0];
#pragma unroll
    for(int w = 1; w < Reduce<THREADS>::kWarps; ++w) t = fmax(t, s.d[w]);
    __syncthreads();
    return t;
}

// Exclusive prefix of c over the threads in index order; total receives the block's sum.
template<int THREADS> __device__ int block_exclusive(int c, int& total, Reduce<THREADS>& s) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = c;
#pragma unroll
    for(int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if(lane >= o) x += y;
    }
    if(lane == 31) s.i[warp] = x;
    __syncthreads();
    int before = 0, all = 0;
#pragma unroll
    for(int w = 0; w < Reduce<THREADS>::kWarps; ++w) {
        const int v = s.i[w];
        before += w < warp ? v : 0;
        all += v;
    }
    __syncthreads();
    total = all;
    return before + x - c;
}

template<int THREADS, int E> struct TopShared {
    static constexpr int kWarps = THREADS / 32;
    static constexpr int kBuf = kCand * E > THREADS ? kCand * E : THREADS;
    Reduce<THREADS> red;
    u64 pick[kWarps];
    u64 tau;
    u64 buf[kBuf];
};

// The K (1..kCand) largest of the block's keys (E per thread; 0 is empty, nonzero keys are distinct) go to out[0, K)
// in descending order, zero padded; returns how many are nonzero. All threads call it with the same K. out is written
// by at most K threads and is visible to the block after the caller's next __syncthreads().
//
// A threshold first: each warp sorts its threads' maxima, and the smallest over the warps of each warp's
// ceil(K / warps)-th largest bounds at least K keys from below. The keys at or above it are gathered (thread order)
// and ranked by counting. When that gathers more than kBuf keys (adversarial rows), the exact K-th largest thread
// maximum replaces it: only the K threads at or above it hold candidates, so at most K * E remain.
template<int THREADS, int E>
__device__ int block_top(const u64 (&key)[E], int K, u64* out, TopShared<THREADS, E>& s) {
    constexpr int kTopWarps = THREADS / 32;
    constexpr int kBuf = TopShared<THREADS, E>::kBuf;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if(K <= 0) return 0;
    u64 m = 0;
#pragma unroll
    for(int i = 0; i < E; ++i) m = umax(m, key[i]);
    if(K == 1) {   // the maximum: a warp tree, then the warps
        u64 b = m;
#pragma unroll
        for(int o = 16; o; o >>= 1) b = umax(b, __shfl_xor_sync(0xffffffffu, b, o));
        if(lane == 0) s.pick[warp] = b;
        __syncthreads();
        if(threadIdx.x == 0) {
            u64 t = s.pick[0];
#pragma unroll
            for(int w = 1; w < kTopWarps; ++w) t = umax(t, s.pick[w]);
            out[0] = t;
            s.tau = t;
        }
        __syncthreads();
        return s.tau ? 1 : 0;
    }
    u64 v = m;   // descending bitonic sort of the warp's maxima over the lanes
#pragma unroll
    for(int size = 2; size <= 32; size <<= 1) {
#pragma unroll
        for(int stride = size >> 1; stride > 0; stride >>= 1) {
            const u64 o = __shfl_xor_sync(0xffffffffu, v, stride);
            const bool keep_max = ((lane & stride) == 0) == ((lane & size) == 0);
            v = keep_max ? umax(v, o) : umin(v, o);
        }
    }
    const int j = (K + kTopWarps - 1) / kTopWarps;
    if(lane == j - 1) s.pick[warp] = v;
    __syncthreads();
    u64 tau = s.pick[0];
#pragma unroll
    for(int w = 1; w < kTopWarps; ++w) tau = umin(tau, s.pick[w]);
    int count = 0;
#pragma unroll
    for(int i = 0; i < E; ++i) count += key[i] != 0 && key[i] >= tau;
    int total = 0;
    int offset = block_exclusive(count, total, s.red);
    if(total > kBuf) {
        s.buf[threadIdx.x] = m;
        if(threadIdx.x == 0) s.tau = 0;
        __syncthreads();
        if(m != 0) {
            int rank = 0;
            for(int u = 0; u < THREADS; ++u) rank += s.buf[u] > m;
            if(rank == K - 1) s.tau = m;
        }
        __syncthreads();
        tau = s.tau;
        count = 0;
#pragma unroll
        for(int i = 0; i < E; ++i) count += key[i] != 0 && key[i] >= tau;
        offset = block_exclusive(count, total, s.red);   // its barriers end every read of buf above
    }
#pragma unroll
    for(int i = 0; i < E; ++i)
        if(key[i] != 0 && key[i] >= tau) s.buf[offset++] = key[i];
    __syncthreads();
    if(total <= THREADS) {
        const u64 mine = int(threadIdx.x) < total ? s.buf[threadIdx.x] : 0;
        int rank = 0;
        for(int u = 0; u < total; ++u) rank += s.buf[u] > mine;
        if(int(threadIdx.x) < total && rank < K) out[rank] = mine;
    } else {
        constexpr int R = (kBuf + THREADS - 1) / THREADS;
        u64 mine[R];
        int rank[R];
#pragma unroll
        for(int r = 0; r < R; ++r) {
            const int at = int(threadIdx.x) + r * THREADS;
            mine[r] = at < total ? s.buf[at] : 0;
            rank[r] = 0;
        }
        for(int u = 0; u < total; ++u) {
            const u64 o = s.buf[u];
#pragma unroll
            for(int r = 0; r < R; ++r) rank[r] += o > mine[r];
        }
#pragma unroll
        for(int r = 0; r < R; ++r)
            if(int(threadIdx.x) + r * THREADS < total && rank[r] < K) out[rank[r]] = mine[r];
    }
    if(int(threadIdx.x) >= total && int(threadIdx.x) < K) out[threadIdx.x] = 0;
    return total < K ? total : K;
}

// Merges `blocks` per-block candidate lists (list + b * stride, K entries each, descending, zero padded) into the row's
// K largest keys in `top` (shared, kCand entries). Returns how many are nonzero; `top` is visible on return.
__device__ int merge_lists(const u64* list, int blocks, int stride, int K, u64* top, TopShared<kMergeThreads, kMergeItems>& s) {
    constexpr int kNew = (kMergeItems - 1) * kMergeThreads;
    const int count = blocks * K;
    int avail = 0;
    for(int base = 0;; base += kNew) {
        u64 key[kMergeItems];
#pragma unroll
        for(int i = 0; i < kMergeItems - 1; ++i) {
            const int at = base + i * kMergeThreads + int(threadIdx.x);
            key[i] = at < count ? list[std::size_t(at / K) * stride + at % K] : 0;
        }
        key[kMergeItems - 1] = base > 0 && int(threadIdx.x) < K ? top[threadIdx.x] : 0;
        __syncthreads();
        avail = block_top(key, K, top, s);
        __syncthreads();
        if(base + kNew >= count) break;
    }
    return avail;
}

// ---- Raw log-probabilities (Core's TokenLogprob over the raw row: crit/sampling.h log_softmax, top_logprobs) ----
// m = the row's largest finite logit, log_sum = log(sum of exp(x - m) over its finite entries), logprob = float((x - m)
// - log_sum) in double, alternatives ranked by float log-probability descending then id ascending. Core accumulates in
// double in id order; here the exps are FP32 (compensated per thread, FP64 over threads and blocks in a fixed order),
// so values agree to about 1e-7 relative in the sum and the rankings are Core's, ties at the boundary included.

// A stage block's raw maximum and sum of exp(x - max) over its finite entries.
struct Stat {
    float max;
    float unused;
    double sum;
};

// The raw part of a stage block: its Stat and, with K > 0, its best K raw keys (-inf entries included, padding not).
template<int E>
__device__ void raw_stage(const float (&v)[E], int lo, int vocab, int K, Stat* stat, u64* cand, TopShared<kStageThreads, E>& s) {
    float m = neg_inf();
#pragma unroll
    for(int i = 0; i < E; ++i) m = fmaxf(m, v[i]);
    m = block_max(m, s.red);
    double part = 0.;
    if(m != neg_inf()) {
        float sum = 0.f, comp = 0.f;
#pragma unroll
        for(int i = 0; i < E; ++i)
            if(!masked(v[i])) {
                const float y = expf(v[i] - m) - comp;
                const float t = sum + y;
                comp = (t - sum) - y;
                sum = t;
            }
        part = double(sum) - double(comp);
    }
    const double total = block_sum(part, s.red);
    if(threadIdx.x == 0) *stat = Stat{m, 0.f, total};
    if(K <= 0) return;
    u64 key[E];
#pragma unroll
    for(int i = 0; i < E; ++i) {
        const int t = lo + i * kStageThreads + int(threadIdx.x);
        key[i] = t < vocab ? raw_key(v[i], t) : 0;
    }
    block_top(key, K, cand, s);
    __syncthreads();
}

// Row normalization from its blocks' stats: max over the blocks and log(sum_b sum_b * exp(max_b - max)) in double, in a
// fixed order. Every thread returns the same values; max is -inf (and log_sum 0) for a row without a finite entry.
__device__ void merge_stats(const Stat* stat, int blocks, double& max, double& log_sum, Reduce<kMergeThreads>& s) {
    float m = neg_inf();
    for(int b = threadIdx.x; b < blocks; b += kMergeThreads) m = fmaxf(m, stat[b].max);
    m = block_max(m, s);
    double part = 0.;
    if(m != neg_inf())
        for(int b = threadIdx.x; b < blocks; b += kMergeThreads) {
            const Stat st = stat[b];
            if(st.max != neg_inf()) part += st.sum * exp(double(st.max) - double(m));
        }
    const double total = block_sum(part, s);
    max = double(m);
    log_sum = m != neg_inf() ? log(total) : 0.;
}

// Core's raw log-probability: (x - max) - log_sum in double, then rounded to float.
__device__ __forceinline__ float raw_logprob(float x, double max, double log_sum) { return float((double(x) - max) - log_sum); }

struct AltShared {
    float lp[kCand];
    std::int32_t id[kCand];
    float out_lp[kCand];
    std::int32_t out_id[kCand];
    int cut;              // answer entries ranked strictly above the boundary log-probability
    float lo, hi;         // tie scan: the raw values whose log-probability equals the boundary's
    Reduce<kMergeThreads> red;
};

// The n (0..kMaxTopK) best raw alternatives of a row by (float log-probability descending, id ascending), as Core's
// top_logprobs ranks them, from the row's raw top (avail = min(vocab, n + 1)) candidates `top` (descending raw keys).
// When the (n + 1)-th raw candidate's log-probability equals the n-th answer's (values collapsing to one float), a scan
// of the row in id order resolves that tie group exactly. Writes ids[0, n) and lps[0, n) (-1 and -inf for a row
// without a finite logit). Block-wide: every thread calls it; it starts and ends with a barrier.
__device__ void write_alternatives(const float* x, int vocab, const u64* top, int avail, int n, double max, double log_sum,
                                   std::int32_t* ids, float* lps, AltShared& s) {
    const int tid = int(threadIdx.x);
    __syncthreads();
    if(n <= 0) return;
    if(max == double(neg_inf())) {
        if(tid < n) { ids[tid] = -1; lps[tid] = neg_inf(); }
        __syncthreads();
        return;
    }
    const int cnt = avail < n + 1 ? avail : n + 1;
    if(tid < cnt) {
        s.lp[tid] = raw_logprob(key_value(top[tid]), max, log_sum);
        s.id[tid] = key_id(top[tid]);
    }
    __syncthreads();
    if(tid < cnt) {
        const float a = s.lp[tid];
        const std::int32_t ia = s.id[tid];
        int rank = 0;
        for(int j = 0; j < cnt; ++j) rank += s.lp[j] > a || (s.lp[j] == a && s.id[j] < ia);
        s.out_lp[rank] = a;
        s.out_id[rank] = ia;
    }
    __syncthreads();
    const bool exact = cnt <= n || s.lp[n] < s.out_lp[n - 1];
    if(exact) {
        if(tid < n) {
            ids[tid] = tid < cnt ? s.out_id[tid] : -1;
            lps[tid] = tid < cnt ? s.out_lp[tid] : neg_inf();
        }
        __syncthreads();
        return;
    }
    // Tie group at the boundary: the answer is every candidate above L, then the lowest ids whose log-probability
    // equals L, over the whole row. Values with log-probability L form one float interval [lo, hi] (monotone map).
    const float L = s.out_lp[n - 1];
    if(tid == 0) {
        int cut = 0;
        while(cut < n && s.out_lp[cut] > L) ++cut;
        s.cut = cut;
        // Smallest and largest keys whose value maps to L, by bisection over the ordered keys (-inf's key to +inf's).
        std::uint32_t lo = 0x007fffffu, hi = order_key(key_value(top[n]));   // lp(top[n]) == L
        while(lo < hi) {                                               // first key with lp >= L
            const std::uint32_t mid = lo + (hi - lo) / 2u;
            if(raw_logprob(key_float(mid), max, log_sum) >= L) hi = mid; else lo = mid + 1u;
        }
        s.lo = key_float(lo);
        std::uint32_t a = order_key(key_value(top[n])), b = 0xff800000u;  // last key with lp <= L (+inf's key bounds)
        while(a < b) {
            const std::uint32_t mid = a + (b - a + 1u) / 2u;
            if(raw_logprob(key_float(mid), max, log_sum) <= L) a = mid; else b = mid - 1u;
        }
        s.hi = key_float(a);
    }
    __syncthreads();
    const int cut = s.cut;
    const float lo = s.lo, hi = s.hi;
    if(tid < cut) { ids[tid] = s.out_id[tid]; lps[tid] = s.out_lp[tid]; }
    // Contiguous id ranges per thread: count the matches, prefix them, then emit the first n - cut in id order.
    const int span = (vocab + kMergeThreads - 1) / kMergeThreads;
    const int begin = tid * span, end = begin + span < vocab ? begin + span : vocab;
    int c = 0;
    for(int t = begin; t < end; ++t) {
        const float v = x[t];
        c += v >= lo && v <= hi;
    }
    int total = 0;
    int at = cut + block_exclusive(c, total, s.red);
    for(int t = begin; t < end && at < n; ++t) {
        const float v = x[t];
        if(v >= lo && v <= hi) {
            ids[at] = t;
            lps[at] = L;
            ++at;
        }
    }
    __syncthreads();
}

// A row's raw log-probabilities from its stage blocks (stat[0, B), lists of `stride` keys each holding its best n + 1):
// max and log_sum (returned), the chosen token's log-probability when chosen_out is set (-inf for an id outside
// [0, vocab)), and the n best alternatives. Block-wide.
__device__ void row_logprobs(const float* x, int vocab, const Stat* stat, const u64* cand, int B, int stride, int n,
                             std::int32_t chosen, float* chosen_out, std::int32_t* ids, float* lps, double& max,
                             double& log_sum, TopShared<kMergeThreads, kMergeItems>& s, u64* top, AltShared& alt) {
    merge_stats(stat, B, max, log_sum, alt.red);
    if(chosen_out && threadIdx.x == 0)
        *chosen_out = chosen >= 0 && chosen < vocab && max != double(neg_inf()) ? raw_logprob(x[chosen], max, log_sum) : neg_inf();
    if(n > 0) {
        const int avail = merge_lists(cand, B, stride, n + 1, top, s);
        write_alternatives(x, vocab, top, avail, n, max, log_sum, ids, lps, alt);
    }
}

// A stage block's processed maximum, the sum of exp((v - max) / T) over its selectable tokens (FP32 exps,
// compensated) and their count: the whole-row weight of open rows.
struct LawStat {
    double max;
    double sum;
    std::int32_t count;
    std::int32_t unused;
};

// Per-row results of the law kernel for the dense fill and the full law's dispatch.
struct RowResult {
    std::int32_t count;                 // kept pairs of a settled row
    std::int32_t handled;               // the law kernel settled the row: the full law skips it
    std::int32_t ids[kMaxTopK];
    float p[kMaxTopK];
    double max, log_sum;                // the raw row's normalization (log-probabilities of full-law rows)
};

struct Layout {
    int B = 0;                          // stage blocks per row
    u64* cand = nullptr;                // [rows][B][kCand] candidate keys
    u64* rawc = nullptr;                // [rows][B][kCand] raw candidate keys (log-probability alternatives)
    LawStat* lstat = nullptr;           // [rows][B]
    Stat* stat = nullptr;               // [rows][B] raw stats (log-probabilities)
    RowResult* result = nullptr;        // [rows]
};

__host__ __device__ inline std::size_t fast_bytes(std::uint32_t vocab, std::uint32_t rows) {
    const std::size_t B = std::size_t(blocks_per_row(int(vocab)));
    return std::size_t(rows) * (B * (2 * kCand * sizeof(u64) + sizeof(LawStat) + sizeof(Stat)) + sizeof(RowResult));
}

__host__ __device__ inline Layout layout(void* base, std::uint32_t vocab, std::uint32_t rows) {
    Layout L;
    L.B = blocks_per_row(int(vocab));
    const std::size_t blocks = std::size_t(rows) * std::size_t(L.B);
    L.cand = static_cast<u64*>(base);
    L.rawc = L.cand + blocks * kCand;
    L.lstat = reinterpret_cast<LawStat*>(L.rawc + blocks * kCand);
    L.stat = reinterpret_cast<Stat*>(L.lstat + blocks);
    L.result = reinterpret_cast<RowResult*>(L.stat + blocks);
    return L;
}

// The full law's processed rows come first in Rows::scratch, rounded up to 16 bytes.
__host__ __device__ inline std::size_t law_floats(std::uint32_t vocab, std::uint32_t rows) {
    return (std::size_t(vocab) * rows + 3) / 4 * 4;
}

// The packed key of processed value v of raw logit x: 0 when masked; the raw key when untouched; else the value
// rounded down to a float with the inexact bit when that rounding lost anything (below -FLT_MAX: -inf's key, inexact).
__device__ __forceinline__ u64 value_key(double v, bool touched, float x, std::int32_t t) {
    if(v == double(neg_inf())) return 0;
    if(!touched) return raw_key(x, t);
    const float f = __double2float_rd(v);
    return pack(order_key(f), double(f) != v, t);
}

// What both fast-path kernels decide for one row from its Request.
struct RowPlan {
    std::uint32_t r = 0, c = 0;
    bool fast = false;                  // greedy or 1 <= top_k <= kMaxTopK
    bool open = false;                  // temperature > 0 without a top_k filter (top_k <= 0 or >= vocab)
    bool penalties = false, suppressing = false, modified = false;
    int K = 0;                          // candidates per block and row; 0: the full law takes the row
    std::uint64_t position = 0;
    const std::int32_t* staged = nullptr;
    std::uint32_t staged_count = 0;
};

__device__ RowPlan plan_row(const Rows& rows, std::uint32_t row, const std::int32_t* fallback_staged, std::int32_t* path) {
    RowPlan p;
    p.r = row / rows.columns;
    p.c = row - p.r * rows.columns;
    const Request& q = rows.request[p.r];
    const RowView v = view_of(rows, p.r, p.c, fallback_staged, rows.columns - 1, path);
    p.position = v.position;
    p.staged = v.staged;
    p.staged_count = v.staged_count;
    p.fast = !(q.temperature > 0.) || (q.top_k >= 1 && q.top_k <= kMaxTopK);
    p.open = !p.fast && (q.top_k <= 0 || std::uint32_t(q.top_k) >= rows.vocab);
    p.penalties = has_penalties(q);
    p.suppressing = q.suppress_count > 0 && p.position < q.min_tokens;
    p.modified = q.bias_count > 0 || p.penalties || p.suppressing;
    p.K = p.fast ? (q.temperature > 0. ? q.top_k + 1 : 2) : (p.open ? kCand : 0);
    return p;
}

// LawStat of a stage block at temperature T; value_of(i) gives the thread's i-th processed value (-inf: not
// selectable). It is called twice per item (max, then the exps), so processed rows recompute their values instead of
// holding them in registers.
template<int E, class ValueOf>
__device__ void law_stats(ValueOf value_of, double temperature, LawStat* out, Reduce<kStageThreads>& red) {
    double m = double(neg_inf());
    int count = 0;
#pragma unroll
    for(int i = 0; i < E; ++i) {
        const double x = value_of(i);
        if(x != double(neg_inf())) { m = fmax(m, x); ++count; }
    }
    m = block_max_double(m, red);
    int total = 0;
    block_exclusive(count, total, red);
    double part = 0.;
    if(m != double(neg_inf())) {
        const float tf = float(temperature);
        float sum = 0.f, comp = 0.f;
#pragma unroll
        for(int i = 0; i < E; ++i) {
            const double x = value_of(i);
            if(x == double(neg_inf())) continue;
            const float y = expf(float(x - m) / tf) - comp;
            const float t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        part = double(sum) - double(comp);
    }
    const double s = block_sum(part, red);
    if(threadIdx.x == 0) *out = LawStat{m, s, total, 0};
}

// Stage for Rows (enqueue_sample, enqueue_verify): grid rows * B blocks.
// (The minimum of two blocks per SM bounds the registers at 128: without it ptxas for sm_80 to sm_90 chose 80 and
// spilled. The law kernel takes the same bound.)
// With kLp the block also writes its raw Stat and, lp_k > 0, its best lp_k raw keys (log-probabilities).
template<bool kLp>
__global__ void __launch_bounds__(kStageThreads, 2) stage_kernel(const Rows rows, const std::int32_t* fallback_staged, int lp_k, Layout L) {
    __shared__ TopShared<kStageThreads, kStageItems> s;
    __shared__ Lists ls;
    const int B = L.B;
    const std::uint32_t row = blockIdx.x / std::uint32_t(B);
    const int b = int(blockIdx.x - row * std::uint32_t(B));
    const RowPlan p = plan_row(rows, row, fallback_staged, ls.path);
    if(p.K == 0 && !kLp) return;
    const float* x = rows.logits + std::size_t(row) * rows.stride;
    const int vocab = int(rows.vocab), lo = b * kChunk;
    const std::size_t at = std::size_t(row) * B + b;
    const Request& q = rows.request[p.r];
    float v[kStageItems];
#pragma unroll
    for(int i = 0; i < kStageItems; ++i) {
        const int t = lo + i * kStageThreads + int(threadIdx.x);
        v[i] = t < vocab ? x[t] : neg_inf();
    }
    if constexpr(kLp) raw_stage(v, lo, vocab, lp_k, L.stat + at, L.rawc + at * kCand, s);
    if(p.K == 0) return;
    u64 key[kStageItems];
    if(!p.modified) {
        if(!p.fast) law_stats<kStageItems>([&](int i) { return double(v[i]); }, q.temperature, L.lstat + at, s.red);
#pragma unroll
        for(int i = 0; i < kStageItems; ++i) {
            const int t = lo + i * kStageThreads + int(threadIdx.x);
            key[i] = t < vocab && !masked(v[i]) ? raw_key(v[i], t) : 0;
        }
        block_top(key, p.K, L.cand + at * kCand, s);
        return;
    }
    const Proc P = load_proc(ls, q, p.staged, p.staged_count, p.penalties, p.suppressing);
#pragma unroll
    for(int i = 0; i < kStageItems; ++i) {
        const int t = lo + i * kStageThreads + int(threadIdx.x);
        bool touched = false;
        const double pv = t < vocab ? process_value(P, t, v[i], touched) : double(neg_inf());
        key[i] = t < vocab ? value_key(pv, touched, v[i], t) : 0;
    }
    if(!p.fast)
        law_stats<kStageItems>([&](int i) {
            const int t = lo + i * kStageThreads + int(threadIdx.x);
            bool touched = false;
            return t < vocab ? process_value(P, t, v[i], touched) : double(neg_inf());
        }, q.temperature, L.lstat + at, s.red);
    block_top(key, p.K, L.cand + at * kCand, s);
}

// The law of one fast row in shared memory: candidates in exact rank order, weights, the kept prefix.
struct LawShared {
    std::int32_t cid[kCand];        // candidates as merged
    double cval[kCand];
    std::int32_t id[kCand];         // selectable candidates in exact rank order
    double v[kCand];
    double w[kCand];
    double res[kCand];              // verify: residual weights
    std::int32_t order[kCand];      // id order of the candidates
    SparseQ q;                      // verify: the column's sparse draft law
    int n, kept;                    // kept -1: the whole row (an open row without filters)
    double kept_total;
    double pd, qd;
    std::int32_t token;
    int settled;
};

// The whole-row weight of an open row from its stage blocks: sum_b sum_b * exp((max_b - top) / T) in double in block
// order, relative to the row's top processed value, and the selectable count. Block-wide.
__device__ void merge_law_stats(const LawStat* st, int blocks, double T, double top, double& S, int& count,
                                Reduce<kMergeThreads>& red) {
    double part = 0.;
    int c = 0;
    for(int b = threadIdx.x; b < blocks; b += kMergeThreads) {
        const LawStat x = st[b];
        c += x.count;
        if(x.count > 0) part += x.sum * exp(__ddiv_rn(__dsub_rn(x.max, top), T));
    }
    S = block_sum(part, red);
    int total = 0;
    block_exclusive(c, total, red);
    count = total;
}

// Core's steps 8 to 12 for an open row (no top_k filter) with more than kMaxTopK selectable tokens: ls.id / ls.v hold
// its exact top kMaxTopK candidates (and the next one, exact unless `uncertain`) and S its whole weight. Settles the
// row (returns true) when top_p or min_p closes the kept set within the first kMaxTopK candidates (Core's sums: rank
// order with top_p, id order without), or, with no filter at all and `whole` allowed, when the draw lands among them
// (ls.kept = -1, kept_total = S). Block-wide.
__device__ bool open_law_draw(LawShared& ls, double T, double top_p, double min_p, double S, double u, bool whole,
                              bool uncertain) {
    const int tid = int(threadIdx.x);
    if(tid < kCand) ls.w[tid] = exp(__ddiv_rn(__dsub_rn(ls.v[tid], ls.v[0]), T));
    if(tid < kMaxTopK) {
        int pos = 0;
        for(int j = 0; j < kMaxTopK; ++j) pos += ls.id[j] < ls.id[tid];
        ls.order[pos] = tid;
    }
    __syncthreads();
    if(tid == 0) {
        int lm = 0;   // Core's min_p prefix (weights > 0 and >= min_p)
        while(lm < kCand && ls.w[lm] > 0. && ls.w[lm] >= min_p) ++lm;
        // A prefix closing at kMaxTopK is decided by the next token's weight, known only when that candidate is exact.
        const bool closes = lm < kMaxTopK || (lm == kMaxTopK && !uncertain);
        bool settled = true;
        int kept = 0;
        double kept_total = 0.;
        if(top_p < 1.) {
            const double target = top_p * S;
            double acc = 0.;
            int nucleus = 0;
            for(int r = 0; r < min(lm, kMaxTopK); ++r) {
                acc += ls.w[r];
                if(acc >= target) { nucleus = r + 1; break; }
            }
            if(nucleus) { kept = nucleus; kept_total = acc; }
            else if(closes) { kept = lm; kept_total = acc; }
            else settled = false;
        } else if(closes) {
            kept = lm;
            for(int j = 0; j < kMaxTopK; ++j) if(ls.order[j] < lm) kept_total += ls.w[ls.order[j]];
        } else if(!(min_p > 0.) && whole) {
            kept = -1;
            kept_total = S;
        } else {
            settled = false;
        }
        std::int32_t token = -1;
        if(settled) {
            const double target = u * kept_total;
            double acc = 0.;
            const int walk = kept > 0 ? kept : kMaxTopK;
            int chosen = kept > 0 ? kept - 1 : -1;
            for(int r = 0; r < walk; ++r) {
                acc += ls.w[r];
                if(target < acc) { chosen = r; break; }
            }
            if(chosen < 0) settled = false;   // the draw lies beyond the candidates
            else token = ls.id[chosen];
        }
        ls.kept = settled ? kept : 0;
        ls.kept_total = kept_total;
        ls.token = token;
        ls.settled = settled;
    }
    __syncthreads();
    return ls.settled != 0;
}

// Core's steps 7 to 12 over the row's n selectable candidates (ls.id / ls.v in exact rank order), in double and in
// Core's summation order (rank order with top_k, id order without), with draw u. Leaves the kept law (ls.kept, ls.w,
// ls.kept_total) and ls.token. Block-wide; ends with a barrier.
__device__ void fast_law_draw(LawShared& ls, int n, double T, std::int32_t top_k, double top_p, double min_p, double u) {
    const int tid = int(threadIdx.x);
    if(!(T > 0.) || n == 0) {
        if(tid == 0) {
            ls.kept = n > 0 ? 1 : 0;
            ls.w[0] = 1.;
            ls.kept_total = 1.;
            ls.token = n > 0 ? ls.id[0] : -1;
        }
        __syncthreads();
        return;
    }
    const bool sorted = n > top_k;                // Core's "0 < top_k < selectable"
    const int keff = sorted ? top_k : n;
    if(tid < keff) {
        ls.w[tid] = exp(__ddiv_rn(__dsub_rn(ls.v[tid], ls.v[0]), T));
        if(!sorted) {
            int pos = 0;
            for(int j = 0; j < keff; ++j) pos += ls.id[j] < ls.id[tid];
            ls.order[pos] = tid;
        }
    }
    __syncthreads();
    if(tid == 0) {
        double total = 0., kept_total = 0.;
        int kept = 0;
        if(sorted) {
            for(int i = 0; i < keff; ++i) {
                const double w = ls.w[i];
                total += w;
                if(kept == i && w > 0. && w >= min_p) { ++kept; kept_total += w; }
            }
        } else {
            for(int j = 0; j < keff; ++j) {
                const double w = ls.w[ls.order[j]];
                total += w;
                if(w > 0. && w >= min_p) { ++kept; kept_total += w; }
            }
        }
        if(top_p < 1.) {
            const double target = top_p * total;
            double acc = 0.;
            int nucleus = kept;
            for(int r = 0; r < kept; ++r) {
                acc += ls.w[r];
                if(acc >= target) { nucleus = r + 1; break; }
            }
            kept = nucleus;
            kept_total = acc;
        }
        const double target = u * kept_total;
        double acc = 0.;
        int chosen = kept - 1;
        for(int r = 0; r < kept; ++r) {
            acc += ls.w[r];
            if(target < acc) { chosen = r; break; }
        }
        ls.kept = kept;
        ls.kept_total = kept_total;
        ls.token = ls.id[chosen];
    }
    __syncthreads();
}

// Candidates of a merged row into ls.id / ls.v (selectable ones, exact rank order); returns n.
__device__ int exact_candidates(const u64* top, int K, bool modified, const Proc& P, const float* x, LawShared& ls) {
    const int tid = int(threadIdx.x);
    if(tid < K) {
        const u64 k = top[tid];
        std::int32_t id = -1;
        double val = double(neg_inf());
        if(k) {
            id = key_id(k);
            if(modified) {
                bool touched = false;
                val = process_value(P, id, x[id], touched);
            } else {
                val = double(key_value(k));
            }
        }
        ls.cid[tid] = id;
        ls.cval[tid] = val;
    }
    __syncthreads();
    if(tid == 0) {
        int n = 0;
        while(n < K && ls.cid[n] >= 0 && ls.cval[n] != double(neg_inf())) ++n;
        ls.n = n;
    }
    __syncthreads();
    const int n = ls.n;
    if(tid < n) {
        const double a = ls.cval[tid];
        const std::int32_t ia = ls.cid[tid];
        int rank = tid;
        if(modified) {
            rank = 0;
            for(int j = 0; j < n; ++j) rank += ls.cval[j] > a || (ls.cval[j] == a && ls.cid[j] < ia);
        }
        ls.id[rank] = ia;
        ls.v[rank] = a;
    }
    __syncthreads();
    return n;
}

struct LawOutputs {
    std::int32_t* out = nullptr;              // sample: [rows]
    SparseLaw* sparse = nullptr;              // sample: [rows]
    std::uint32_t purpose = 0;                // sample: the draw purpose
    Verify verify;                            // verify: drafts, draft laws, predictions
};

// The law of one fast-path row (merge, Core's steps 7 to 12, the verify rule) and, when it settles the row here, its
// outputs: the token or prediction, the sparse law and the dense fill's pairs. Returns whether it did; `token` is the
// row's token (or prediction). Block-wide.
template<bool kVerify>
__device__ bool settle(const Rows& rows, const RowPlan& p, std::uint32_t row, bool want_dense, const LawOutputs& o,
                       const Layout& L, LawShared& ls, Lists& lists, TopShared<kMergeThreads, kMergeItems>& s, u64* top,
                       Reduce<kMergeThreads>& red, std::int32_t& token) {
    const int tid = int(threadIdx.x);
    RowResult& R = L.result[row];
    const Request& q = rows.request[p.r];
    const int B = L.B, vocab = int(rows.vocab);
    const float* x = rows.logits + std::size_t(row) * rows.stride;
    const std::size_t at = std::size_t(row) * B;
    merge_lists(L.cand + at * kCand, B, kCand, p.K, top, s);
    Proc P;
    if(p.modified) P = load_proc(lists, q, p.staged, p.staged_count, p.penalties, p.suppressing);
    const int n = exact_candidates(top, p.K, p.modified, P, x, ls);
    // The candidates are the exact best ones unless two inexact keys of one float straddle the last needed one.
    const int need = p.fast ? p.K - 1 : kMaxTopK;
    const bool exact = !same_inexact(top[need - 1], top[need]);
    const crit::core::DrawPurpose purpose = kVerify ? crit::core::DrawPurpose::sample : crit::core::DrawPurpose(o.purpose);
    const double u = crit::core::uniform(q.seed, p.position, q.stream, purpose);
    bool settled = false;
    if(!exact) {
        settled = false;
    } else if(p.fast || n == 0) {
        fast_law_draw(ls, n, q.temperature, q.top_k, q.top_p, q.min_p, u);
        settled = true;
    } else {
        double S = 0.;
        int count = 0;
        merge_law_stats(L.lstat + at, B, q.temperature, ls.v[0], S, count, red);
        if(count <= kMaxTopK) {
            fast_law_draw(ls, n, q.temperature, n, q.top_p, q.min_p, u);   // every selectable token is a candidate
            settled = true;
        } else {
            settled = open_law_draw(ls, q.temperature, q.top_p, q.min_p, S, u, !want_dense, (top[kMaxTopK] >> 31) & 1ull);
        }
    }
    token = settled ? ls.token : -1;
    if constexpr(kVerify) {
        const std::uint32_t K = rows.columns - 1;
        if(settled && p.c < K && q.temperature > 0. && n > 0) {
            const std::size_t col = std::size_t(p.r) * K + p.c;
            const std::int32_t d = o.verify.drafts[col];
            // Sampled drafts with a law take the rejection rule; greedy drafts, cut columns and the bonus the draw above.
            const int mode = draft_mode(o.verify, col, d, vocab);
            if(mode && ls.kept < 0) {
                settled = false;   // a sampled draft against a whole-row law: crit's full law decides
            } else if(mode) {
                const float* qrow = mode == 2 ? o.verify.draft_probabilities + col * rows.vocab : nullptr;
                if(mode == 1) load_sparse(o.verify.draft_laws[col], ls.q);
                auto q_of = [&](std::int32_t t) { return double(qrow ? qrow[t] : sparse_at(ls.q, t)); };
                if(tid == 0) ls.pd = 0.;
                __syncthreads();
                const int kept = ls.kept;
                const double total = ls.kept_total;
                if(tid < kept) {
                    const double pi = ls.w[tid] / total;
                    ls.res[tid] = fmax(pi - q_of(ls.id[tid]), 0.);
                    if(ls.id[tid] == d) ls.pd = pi;
                }
                __syncthreads();
                if(tid == 0) {
                    const double qd = q_of(d);
                    const double ua = crit::core::uniform(q.seed, p.position, q.stream, crit::core::DrawPurpose::accept);
                    std::int32_t t = d;
                    if(!(ua * qd < ls.pd)) {
                        const double ur = crit::core::uniform(q.seed, p.position, q.stream, crit::core::DrawPurpose::residual);
                        double mass = 0.;
                        int last = -1;
                        for(int i = 0; i < kept; ++i) {
                            mass += ls.res[i];
                            if(ls.res[i] > 0.) last = i;
                        }
                        if(mass > 0.) {
                            const double target = ur * mass;
                            double acc = 0.;
                            int chosen = last;
                            for(int i = 0; i < kept; ++i) {
                                acc += ls.res[i];
                                if(target < acc) { chosen = i; break; }
                            }
                            t = ls.id[chosen];
                        }   // no residual mass: the draft carries the whole law and stands
                    }
                    ls.token = t;
                }
                __syncthreads();
                token = ls.token;
            }
        }
        if(settled && tid == 0) o.verify.predictions[std::size_t(p.r) * rows.columns + p.c] = token;
    } else {
        if(settled && tid == 0) o.out[row] = token;
    }
    // The kept law: the dense fill's source and the sparse output (count -1: a whole-row law, no sparse form).
    if(settled) {
        const int count = n > 0 ? ls.kept : 0;
        const double total = ls.kept_total;
        if(tid < count) {
            const float pr = !(q.temperature > 0.) ? 1.f : float(ls.w[tid] / total);
            R.ids[tid] = ls.id[tid];
            R.p[tid] = pr;
            if constexpr(!kVerify) {
                if(o.sparse) { o.sparse[row].ids[tid] = ls.id[tid]; o.sparse[row].probabilities[tid] = pr; }
            }
        }
        if(tid == 0) {
            R.count = count;
            if constexpr(!kVerify) {
                if(o.sparse) o.sparse[row].count = count;
            }
        }
    }
    return settled;
}

// Merge + law (+ verify rule) for Rows: blocks [0, rows) one per row, writing RowResult::handled for every row (set when
// the row's law was settled here, so crit's full law skips it). With log-probabilities, blocks [rows, 2 rows) run beside
// them: each writes its row's alternatives and raw normalization, from which the full-law kernel writes every row's
// chosen log-probability once the token is known.
template<bool kVerify, bool kLp>
__global__ void __launch_bounds__(kMergeThreads, 2)
law_kernel(const Rows rows, bool want_dense, LawOutputs o, Logprobs lpo, Layout L) {
    __shared__ TopShared<kMergeThreads, kMergeItems> s;
    __shared__ u64 top[kCand];
    __shared__ union {   // a block takes one role
        struct { LawShared ls; Lists lists; Reduce<kMergeThreads> red; } law;
        AltShared alt;
    } sh;
    const std::uint32_t count = rows.requests * rows.columns;
    if(kLp && blockIdx.x >= count) {
        const std::uint32_t row = blockIdx.x - count;
        const int B = L.B, n = int(lpo.n);
        const std::size_t at = std::size_t(row) * B;
        double mx = 0., log_sum = 0.;
        row_logprobs(rows.logits + std::size_t(row) * rows.stride, int(rows.vocab), L.stat + at, L.rawc + at * kCand, B, kCand,
                     n, -1, nullptr, lpo.top_ids + std::size_t(row) * n, lpo.top_logprobs + std::size_t(row) * n, mx, log_sum,
                     s, top, sh.alt);
        if(threadIdx.x == 0) {
            L.result[row].max = mx;
            L.result[row].log_sum = log_sum;
        }
        return;
    }
    const std::uint32_t row = blockIdx.x;
    const RowPlan p = plan_row(rows, row, kVerify ? o.verify.drafts : nullptr, sh.law.lists.path);
    std::int32_t token = -1;
    const bool settled = p.K > 0 && settle<kVerify>(rows, p, row, want_dense, o, L, sh.law.ls, sh.law.lists, s, top, sh.law.red, token);
    if(threadIdx.x == 0) L.result[row].handled = settled;
}

// Dense FP32 laws of the settled rows (the full law writes its own rows): zeros, then the kept pairs. (A settled row
// never has a whole-row law here: the law kernel leaves those to the full law when dense rows are asked for.)
__global__ void __launch_bounds__(kStageThreads) dense_kernel(const Rows rows, Layout L, float* probabilities) {
    const int B = L.B;
    const std::uint32_t row = blockIdx.x / std::uint32_t(B);
    const int b = int(blockIdx.x - row * std::uint32_t(B));
    const RowResult& R = L.result[row];
    if(!R.handled) return;
    const int vocab = int(rows.vocab), lo = b * kChunk, hi = min(vocab, lo + kChunk);
    float* out = probabilities + std::size_t(row) * rows.vocab;
    for(int t = lo + int(threadIdx.x); t < hi; t += kStageThreads) out[t] = 0.f;
    __syncthreads();
    if(int(threadIdx.x) < R.count) {
        const std::int32_t id = R.ids[threadIdx.x];
        if(id >= lo && id < hi) out[id] = R.p[threadIdx.x];
    }
}

// ---- enqueue_select's fast path (Params rows: raw logits, no processing) -----------------------------
__device__ __forceinline__ bool fast_select(const Params& P) { return !(P.temperature > 0.) || (P.top_k >= 1 && P.top_k <= kMaxTopK); }
// Candidates per block and row: top_k + 1 (1 when greedy), kCand for an open row, 0 for the full law.
__device__ __forceinline__ int select_candidates(const Params& P, int vocab) {
    if(fast_select(P)) return P.temperature > 0. ? P.top_k + 1 : 1;
    return P.top_k <= 0 || P.top_k >= vocab ? kCand : 0;
}

__global__ void __launch_bounds__(kStageThreads)
select_stage_kernel(const float* logits, int ld, int vocab, const Params* params, Layout L) {
    __shared__ TopShared<kStageThreads, kStageItems> s;
    const int B = L.B;
    const int row = int(blockIdx.x) / B, b = int(blockIdx.x) - row * B;
    const Params P = params[row];
    const int K = select_candidates(P, vocab);
    if(K == 0) return;
    const float* x = logits + std::size_t(row) * ld;
    const int lo = b * kChunk;
    const std::size_t at = std::size_t(row) * B + b;
    float v[kStageItems];
    u64 key[kStageItems];
#pragma unroll
    for(int i = 0; i < kStageItems; ++i) {
        const int t = lo + i * kStageThreads + int(threadIdx.x);
        v[i] = t < vocab ? x[t] : neg_inf();
        key[i] = !masked(v[i]) ? raw_key(v[i], t) : 0;
    }
    if(!fast_select(P)) law_stats<kStageItems>([&](int i) { return double(v[i]); }, P.temperature, L.lstat + at, s.red);
    block_top(key, K, L.cand + at * kCand, s);
}

__global__ void __launch_bounds__(kMergeThreads)
select_law_kernel(const float* logits, int ld, int vocab, Params* params, Layout L, std::int32_t* out) {
    __shared__ TopShared<kMergeThreads, kMergeItems> s;
    __shared__ u64 top[kCand];
    __shared__ LawShared ls;
    __shared__ Reduce<kMergeThreads> red;
    const int row = int(blockIdx.x);
    const Params P = params[row];
    RowResult& R = L.result[row];
    const int K = select_candidates(P, vocab);
    if(K == 0) {
        if(threadIdx.x == 0) R.handled = 0;
        return;
    }
    merge_lists(L.cand + std::size_t(row) * L.B * kCand, L.B, kCand, K, top, s);
    const Proc none;
    const int n = exact_candidates(top, K, false, none, logits + std::size_t(row) * ld, ls);
    const double u = crit::core::uniform(P.seed, P.position, P.stream, crit::core::DrawPurpose::sample);
    bool settled = true;
    if(fast_select(P) || n == 0) {
        fast_law_draw(ls, n, P.temperature, P.top_k, P.top_p, 0., u);
    } else {
        double S = 0.;
        int count = 0;
        merge_law_stats(L.lstat + std::size_t(row) * L.B, L.B, P.temperature, ls.v[0], S, count, red);
        if(count <= kMaxTopK) fast_law_draw(ls, n, P.temperature, n, P.top_p, 0., u);
        else settled = open_law_draw(ls, P.temperature, P.top_p, 0., S, u, true, false);
    }
    if(threadIdx.x == 0) {
        R.handled = settled;
        if(settled) {
            out[row] = ls.token;
            params[row].position = P.position + 1;
        }
    }
}

// ---- the full law's kernels -----------------------------------------------------------------------
// Each takes only the rows the law kernels left unsettled (RowResult::handled clear).
__global__ void __launch_bounds__(kLaw) exact_select_kernel(const float* logits, int ld, int vocab, Params* params,
                                                            const RowResult* result, std::int32_t* out) {
    __shared__ Shared s;
    const int row = int(blockIdx.x);
    if(result[row].handled) return;
    const Params P = params[row];
    const float* x = logits + std::size_t(row) * ld;
    const Law law = compute_law(x, vocab, P.temperature, P.top_k, P.top_p, 0., s);
    const double u = crit::core::uniform(P.seed, P.position, P.stream, crit::core::DrawPurpose::sample);
    const std::int32_t token = draw(x, vocab, law, u, s);
    if(threadIdx.x == 0) {
        out[row] = token;
        params[row].position = P.position + 1;
    }
}

// The full law's processing of row view r: Core's steps 2 to 5 into the row's scratch values.
__device__ void full_process(const RowView& r, const Request& q, int vocab, Lists& lists) {
    const bool penalties = has_penalties(q);
    const bool suppressing = q.suppress_count > 0 && r.position < q.min_tokens;
    const Proc P = load_proc(lists, q, r.staged, r.staged_count, penalties, suppressing);
    process_row(r.logits, r.values, vocab, P);
}

__global__ void __launch_bounds__(kLaw) sample_kernel(Rows rows, std::uint32_t purpose, std::int32_t* out, float* probabilities,
                                                      SparseLaw* sparse, float* chosen_lp, const RowResult* result) {
    __shared__ Shared s;
    __shared__ Lists lists;
    const RowView r = view(rows, nullptr, 0, lists.path);
    if(result[r.row].handled) {
        if(chosen_lp && threadIdx.x == 0) {
            const std::int32_t token = out[r.row];
            chosen_lp[r.row] = token >= 0 ? raw_logprob(r.logits[token], result[r.row].max, result[r.row].log_sum) : neg_inf();
        }
        return;
    }
    const Request q = *r.q;
    const int vocab = int(rows.vocab);
    full_process(r, q, vocab, lists);
    const Law law = compute_law(r.values, vocab, q.temperature, q.top_k, q.top_p, q.min_p, s);
    const double u = crit::core::uniform(q.seed, r.position, q.stream, crit::core::DrawPurpose(purpose));
    const std::int32_t token = draw(r.values, vocab, law, u, s);
    if(threadIdx.x == 0) {
        out[r.row] = token;
        if(sparse) sparse[r.row].count = -1;
        if(chosen_lp)
            chosen_lp[r.row] = token >= 0 ? raw_logprob(r.logits[token], result[r.row].max, result[r.row].log_sum) : neg_inf();
    }
    if(!probabilities) return;
    float* p = probabilities + r.row * rows.vocab;
    for(int t = threadIdx.x; t < vocab; t += kLaw) {
        float x = 0.f;
        if(law.argmax >= 0) x = law.greedy ? (t == law.argmax ? 1.f : 0.f) : float(double(kept_weight(law, t, r.values[t])) / law.total);
        p[t] = x;
    }
}

__global__ void __launch_bounds__(kLaw) verify_kernel(Verify v, const RowResult* result) {
    __shared__ Shared s;
    __shared__ Lists lists;
    __shared__ SparseQ sq;
    const Rows& rows = v.target;
    const std::uint32_t c = blockIdx.x, K = rows.columns - 1;
    const RowView r = view(rows, v.drafts, K, lists.path);
    if(result[r.row].handled) {
        if(v.logprobs.chosen && threadIdx.x == 0) {
            const std::int32_t token = v.predictions[std::size_t(blockIdx.y) * (K + 1) + c];
            v.logprobs.chosen[r.row] = token >= 0 ? raw_logprob(r.logits[token], result[r.row].max, result[r.row].log_sum) : neg_inf();
        }
        return;
    }
    const Request q = *r.q;
    const int vocab = int(rows.vocab);
    full_process(r, q, vocab, lists);
    const Law law = compute_law(r.values, vocab, q.temperature, q.top_k, q.top_p, q.min_p, s);
    std::int32_t token;
    const std::size_t col = std::size_t(blockIdx.y) * K + c;
    const std::int32_t d = c < K ? v.drafts[col] : -1;
    const int mode = c < K && law.argmax >= 0 && !law.greedy ? draft_mode(v, col, d, vocab) : 0;
    if(mode == 0) {
        // The target's own draw: a greedy draft is accepted exactly when it equals it; the bonus and cut columns too.
        token = draw(r.values, vocab, law, crit::core::uniform(q.seed, r.position, q.stream, crit::core::DrawPurpose::sample), s);
    } else {
        const float* qrow = mode == 2 ? v.draft_probabilities + col * rows.vocab : nullptr;
        if(mode == 1) load_sparse(v.draft_laws[col], sq);
        auto q_of = [qrow](std::int32_t t) { return qrow ? qrow[t] : sparse_at(sq, t); };   // sq: shared, not captured
        const double wd = double(kept_weight(law, d, r.values[d]));
        const double pd = wd / law.total;
        const double qd = double(q_of(d));
        const double ua = crit::core::uniform(q.seed, r.position, q.stream, crit::core::DrawPurpose::accept);
        if(ua * qd < pd) {
            token = d;
        } else {
            const double ur = crit::core::uniform(q.seed, r.position, q.stream, crit::core::DrawPurpose::residual);
            const double total = law.total;
            auto w = [&law, &q_of, total](std::int32_t t, float x) {
                return fmaxf(float(double(kept_weight(law, t, x)) / total) - q_of(t), 0.f);
            };
            token = walk(r.values, vocab, w, ur * mass(r.values, vocab, w, s), s);
            // No residual mass left: the draft carries the whole law, so it stands.
            if(token < 0) token = d;
        }
    }
    if(threadIdx.x == 0) {
        v.predictions[std::size_t(blockIdx.y) * (K + 1) + c] = token;
        if(v.logprobs.chosen)
            v.logprobs.chosen[r.row] = token >= 0 ? raw_logprob(r.logits[token], result[r.row].max, result[r.row].log_sum) : neg_inf();
    }
}

// ---- enqueue_logprobs: raw rows -> chosen log-probabilities and alternatives -------------------------------
__global__ void __launch_bounds__(kStageThreads)
raw_stage_kernel(const float* logits, int ld, int vocab, int B, int K, Stat* stat, u64* cand) {
    __shared__ TopShared<kStageThreads, kStageItems> s;
    const int row = int(blockIdx.x) / B, b = int(blockIdx.x) - row * B;
    const float* x = logits + std::size_t(row) * ld;
    const int lo = b * kChunk;
    float v[kStageItems];
#pragma unroll
    for(int i = 0; i < kStageItems; ++i) {
        const int t = lo + i * kStageThreads + int(threadIdx.x);
        v[i] = t < vocab ? x[t] : neg_inf();
    }
    raw_stage(v, lo, vocab, K, stat + blockIdx.x, cand + std::size_t(blockIdx.x) * std::size_t(K), s);
}

__global__ void __launch_bounds__(kMergeThreads, 2)
logprob_merge_kernel(const float* logits, int ld, int vocab, int B, int n, const Stat* stat, const u64* cand,
                     const std::int32_t* chosen, float* chosen_lp, std::int32_t* top_ids, float* top_lps) {
    __shared__ TopShared<kMergeThreads, kMergeItems> s;
    __shared__ u64 top[kCand];
    __shared__ AltShared alt;
    const int row = int(blockIdx.x), K = n > 0 ? n + 1 : 0;
    double mx = 0., log_sum = 0.;
    row_logprobs(logits + std::size_t(row) * ld, vocab, stat + std::size_t(row) * B, cand + std::size_t(row) * B * K, B, K, n,
                 chosen_lp ? chosen[row] : -1, chosen_lp ? chosen_lp + row : nullptr, top_ids + std::size_t(row) * n,
                 top_lps + std::size_t(row) * n, mx, log_sum, s, top, alt);
}

// One thread per logit: a clear bit in its row's mask word bans the token.
__global__ void mask_kernel(float* logits, int vocab, int stride, const std::uint32_t* mask, int mask_stride) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if(t >= vocab) return;
    if(!((mask[static_cast<std::size_t>(r) * mask_stride + t / 32] >> (t % 32)) & 1u))
        logits[static_cast<std::size_t>(r) * stride + t] = __int_as_float(0xff800000);  // -inf
}

__global__ void commit_kernel(Request* requests, std::uint32_t count, std::uint32_t vocab, const std::int32_t* tokens,
                              std::uint32_t token_stride, const std::int32_t* emitted, std::uint32_t emitted_stride) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= count) return;
    Request& q = requests[r];
    const std::int32_t n = emitted ? emitted[std::size_t(r) * emitted_stride] : 1;
    if(n <= 0) return;
    const std::int32_t* row = tokens + std::size_t(r) * token_stride;
    if(q.counts)
        for(std::int32_t i = 0; i < n; ++i) {
            const std::int32_t t = row[i];
            if(t >= 0 && std::uint32_t(t) < vocab) q.counts[t] += 1;
        }
    q.position += std::uint64_t(n);
}

bool aligned8(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 7u) == 0; }

// The fast path's grids: rows * B stage blocks and one law block per row in one grid dimension.
bool fast_grid(std::uint64_t rows, std::uint32_t vocab) {
    return rows * std::uint64_t(blocks_per_row(int(vocab))) <= std::uint64_t(INT_MAX);
}

bool valid_rows(const Rows& rows) {
    return rows.logits && rows.request && rows.scratch && aligned8(rows.scratch) && rows.vocab >= 1 &&
           rows.vocab <= 0x7fffffffu && rows.stride >= rows.vocab && rows.requests >= 1 && rows.requests <= 65535 &&
           rows.columns >= 1 && fast_grid(std::uint64_t(rows.requests) * rows.columns, rows.vocab) &&
           std::uint64_t(rows.requests) * rows.columns <= std::uint64_t(INT_MAX) / 2 &&
           (!rows.staged || rows.staged_stride >= rows.staged_base + rows.columns - 1) &&
           (!rows.tree || (rows.tree_tokens && rows.columns <= 64));
}

bool valid_logprobs(const Logprobs& lp, std::uint32_t vocab) {
    if(!lp.chosen && !lp.top_ids) return true;
    return lp.n <= std::uint32_t(kMaxTopK) && lp.n <= vocab && (lp.n == 0 || (lp.top_ids && lp.top_logprobs));
}

Layout rows_layout(const Rows& rows) {
    const std::uint32_t n = rows.requests * rows.columns;
    return layout(rows.scratch + law_floats(rows.vocab, n), rows.vocab, n);
}

} // namespace

std::size_t scratch_floats(std::uint32_t vocab, std::uint32_t rows) noexcept {
    return law_floats(vocab, rows) + (fast_bytes(vocab, rows) + sizeof(float) - 1) / sizeof(float);
}

std::size_t select_scratch_bytes(int vocab, int rows) noexcept {
    if(vocab < 1 || rows < 1) return 0;
    return fast_bytes(std::uint32_t(vocab), std::uint32_t(rows));
}

cudaError_t enqueue_mask(float* logits, int vocab, int stride, const std::uint32_t* mask, int mask_stride, int rows,
                         cudaStream_t stream) {
    if(!logits || !mask || vocab < 1 || rows < 1 || rows > 65535 || stride < vocab || mask_stride < (vocab + 31) / 32)
        return cudaErrorInvalidValue;
    constexpr int threads = 256;
    mask_kernel<<<dim3((vocab + threads - 1) / threads, rows), threads, 0, stream>>>(logits, vocab, stride, mask, mask_stride);
    return cudaGetLastError();
}

cudaError_t enqueue_select(const float* logits, int ld, int rows, int vocab, Params* params, void* scratch,
                           std::int32_t* out, cudaStream_t stream) {
    if(!logits || vocab < 1 || ld < vocab || rows < 1 || !params || !scratch || !aligned8(scratch) || !out ||
       !fast_grid(std::uint64_t(rows), std::uint32_t(vocab)))
        return cudaErrorInvalidValue;
    const Layout L = layout(scratch, std::uint32_t(vocab), std::uint32_t(rows));
    select_stage_kernel<<<unsigned(rows * L.B), kStageThreads, 0, stream>>>(logits, ld, vocab, params, L);
    select_law_kernel<<<unsigned(rows), kMergeThreads, 0, stream>>>(logits, ld, vocab, params, L, out);
    exact_select_kernel<<<unsigned(rows), kLaw, 0, stream>>>(logits, ld, vocab, params, L.result, out);
    return cudaGetLastError();
}

cudaError_t enqueue_sample(const Rows& rows, crit::core::DrawPurpose purpose, std::int32_t* out, float* probabilities,
                           cudaStream_t stream) {
    return enqueue_sample(rows, purpose, out, probabilities, nullptr, Logprobs{}, stream);
}

cudaError_t enqueue_sample(const Rows& rows, crit::core::DrawPurpose purpose, std::int32_t* out, float* probabilities,
                           SparseLaw* sparse, const Logprobs& logprobs, cudaStream_t stream) {
    if(!valid_rows(rows) || !out || !valid_logprobs(logprobs, rows.vocab)) return cudaErrorInvalidValue;
    const bool lp = logprobs.chosen || logprobs.top_ids;
    const int lp_k = lp && logprobs.n > 0 ? int(logprobs.n) + 1 : 0;
    const std::uint32_t count = rows.requests * rows.columns;
    const Layout L = rows_layout(rows);
    LawOutputs o;
    o.out = out;
    o.sparse = sparse;
    o.purpose = std::uint32_t(purpose);
    if(lp) stage_kernel<true><<<count * std::uint32_t(L.B), kStageThreads, 0, stream>>>(rows, nullptr, lp_k, L);
    else stage_kernel<false><<<count * std::uint32_t(L.B), kStageThreads, 0, stream>>>(rows, nullptr, 0, L);
    if(lp) law_kernel<false, true><<<2 * count, kMergeThreads, 0, stream>>>(rows, probabilities != nullptr, o, logprobs, L);
    else law_kernel<false, false><<<count, kMergeThreads, 0, stream>>>(rows, probabilities != nullptr, o, logprobs, L);
    sample_kernel<<<dim3(rows.columns, rows.requests), kLaw, 0, stream>>>(rows, std::uint32_t(purpose), out, probabilities, sparse,
                                                                          logprobs.chosen, L.result);
    if(probabilities) dense_kernel<<<count * std::uint32_t(L.B), kStageThreads, 0, stream>>>(rows, L, probabilities);
    return cudaGetLastError();
}

cudaError_t enqueue_verify(const Verify& v, cudaStream_t stream) {
    if(!valid_rows(v.target) || !v.predictions || (v.target.columns > 1 && !v.drafts) || v.target.tree ||
       !valid_logprobs(v.logprobs, v.target.vocab))
        return cudaErrorInvalidValue;
    const bool lp = v.logprobs.chosen || v.logprobs.top_ids;
    const int lp_k = lp && v.logprobs.n > 0 ? int(v.logprobs.n) + 1 : 0;
    const Rows& rows = v.target;
    const std::uint32_t count = rows.requests * rows.columns;
    const Layout L = rows_layout(rows);
    LawOutputs o;
    o.verify = v;
    if(lp) stage_kernel<true><<<count * std::uint32_t(L.B), kStageThreads, 0, stream>>>(rows, v.drafts, lp_k, L);
    else stage_kernel<false><<<count * std::uint32_t(L.B), kStageThreads, 0, stream>>>(rows, v.drafts, 0, L);
    if(lp) law_kernel<true, true><<<2 * count, kMergeThreads, 0, stream>>>(rows, false, o, v.logprobs, L);
    else law_kernel<true, false><<<count, kMergeThreads, 0, stream>>>(rows, false, o, v.logprobs, L);
    verify_kernel<<<dim3(rows.columns, rows.requests), kLaw, 0, stream>>>(v, L.result);
    return cudaGetLastError();
}

std::size_t logprob_scratch_bytes(int vocab, int rows, int n) noexcept {
    if(vocab < 1 || rows < 1 || n < 0 || n > kMaxTopK) return 0;
    const std::size_t B = std::size_t(blocks_per_row(vocab)), K = n > 0 ? std::size_t(n) + 1 : 0;
    return std::size_t(rows) * B * (sizeof(Stat) + K * sizeof(u64));
}

cudaError_t enqueue_logprobs(const float* logits, int ld, int rows, int vocab, const std::int32_t* chosen, int n,
                             float* chosen_logprob, std::int32_t* top_ids, float* top_logprobs, void* scratch,
                             cudaStream_t stream) {
    if(!logits || rows < 1 || vocab < 1 || ld < vocab || n < 0 || n > kMaxTopK || n > vocab || !scratch || !aligned8(scratch) ||
       (chosen_logprob && !chosen) || (n > 0 && (!top_ids || !top_logprobs)) || !fast_grid(std::uint64_t(rows), std::uint32_t(vocab)))
        return cudaErrorInvalidValue;
    const int B = blocks_per_row(vocab), K = n > 0 ? n + 1 : 0;
    auto* stat = static_cast<Stat*>(scratch);
    auto* cand = reinterpret_cast<u64*>(stat + std::size_t(rows) * B);
    raw_stage_kernel<<<unsigned(rows * B), kStageThreads, 0, stream>>>(logits, ld, vocab, B, K, stat, cand);
    logprob_merge_kernel<<<unsigned(rows), kMergeThreads, 0, stream>>>(logits, ld, vocab, B, n, stat, cand, chosen, chosen_logprob,
                                                                      top_ids, top_logprobs);
    return cudaGetLastError();
}

cudaError_t enqueue_commit(Request* requests, std::uint32_t count, std::uint32_t vocab, const std::int32_t* tokens,
                           std::uint32_t token_stride, const std::int32_t* emitted, std::uint32_t emitted_stride,
                           cudaStream_t stream) {
    if(!count) return cudaSuccess;
    if(!requests || !tokens || !vocab) return cudaErrorInvalidValue;
    constexpr unsigned threads = 128;
    commit_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(requests, count, vocab, tokens, token_stride,
                                                                           emitted, emitted_stride);
    return cudaGetLastError();
}

} // namespace crit::techniques::token_sampling
