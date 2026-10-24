#include "sampling.h"

#include <crit/sampling.h>

#include <cfloat>
#include <climits>

namespace crit::techniques::token_sampling {
namespace {

constexpr int kThreads = 256;
constexpr int kNone = 0x7fffffff;
constexpr int kLaw = 1024;              // one block of kLaw threads computes one row's law
constexpr int kWarps = kLaw / 32;

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ bool masked(float v) { return v == neg_inf(); }

__device__ __forceinline__ bool before(float v, int i, float w, int j) { return v > w || (v == w && i < j); }

// enqueue_select's two-stage path: greedy, or 1 <= top_k <= kMaxTopK. Everything else takes the full law.
__device__ __forceinline__ bool fast_path(const Params* p) {
    return !(p->temperature > 0.f) || (p->top_k >= 1 && p->top_k <= kMaxTopK);
}

__device__ int candidates(const Params* p) { return p->temperature > 0.f ? max(1, min(p->top_k, kMaxTopK)) : 1; }

// Block-wide best (value, index) in rank order; every thread receives it.
__device__ void block_best(float& v, int& i) {
    __shared__ float sv[32];
    __shared__ int si[32];
#pragma unroll
    for(int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if(before(ov, oi, v, i)) { v = ov; i = oi; }
    }
    __syncthreads();
    if((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = v; si[threadIdx.x >> 5] = i; }
    __syncthreads();
    if(threadIdx.x == 0) {
        for(int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w)
            if(before(sv[w], si[w], v, i)) { v = sv[w]; i = si[w]; }
        sv[0] = v;
        si[0] = i;
    }
    __syncthreads();
    v = sv[0];
    i = si[0];
}

// Stage 1: block b's k best of its slice, in rank order (k block-wide selections).
__global__ void __launch_bounds__(kThreads) local_kernel(const float* __restrict__ logits, int vocab, const Params* p,
                                                         float* cand_values, std::int32_t* cand_ids) {
    if(!fast_path(p)) return;
    const int k = candidates(p);
    const int chunk = (vocab + kStageBlocks - 1) / kStageBlocks;
    const int lo = blockIdx.x * chunk, hi = min(vocab, lo + chunk);
    float pv = FLT_MAX;
    int pi = -1;
    for(int r = 0; r < k; ++r) {
        float v = -FLT_MAX;
        int i = kNone;
        for(int j = lo + threadIdx.x; j < hi; j += kThreads) {
            const float x = logits[j];
            if(before(pv, pi, x, j) && before(x, j, v, i)) { v = x; i = j; }
        }
        block_best(v, i);
        if(threadIdx.x == 0) { cand_values[blockIdx.x * kMaxTopK + r] = v; cand_ids[blockIdx.x * kMaxTopK + r] = i; }
        pv = v;
        pi = i;
    }
}

// Stage 2: global k best from the candidates; then temperature, top-p and one draw.
__global__ void __launch_bounds__(1024) pick_kernel(const float* cand_values, const std::int32_t* cand_ids, Params* p,
                                                    std::int32_t* out) {
    if(!fast_path(p)) return;
    __shared__ float tv[kMaxTopK];
    __shared__ int ti[kMaxTopK];
    const int k = candidates(p);
    float pv = FLT_MAX;
    int pi = -1;
    for(int r = 0; r < k; ++r) {
        float v = -FLT_MAX;
        int i = kNone;
        for(int n = threadIdx.x; n < kStageBlocks * k; n += blockDim.x) {
            const int b = n / k, c = n % k;
            const float x = cand_values[b * kMaxTopK + c];
            const int j = cand_ids[b * kMaxTopK + c];
            if(j != kNone && before(pv, pi, x, j) && before(x, j, v, i)) { v = x; i = j; }
        }
        block_best(v, i);
        if(threadIdx.x == 0) { tv[r] = v; ti[r] = i; }
        pv = v;
        pi = i;
    }
    __syncthreads();
    if(threadIdx.x != 0) return;
    // No selectable logit (every one -inf or NaN): -1, as the full law answers; the position advances as for any select.
    if(ti[0] == kNone) { out[0] = -1; p->position += 1; return; }
    if(!(p->temperature > 0.f)) { out[0] = ti[0]; p->position += 1; return; }
    float w[kMaxTopK], total = 0.f;
    int n = 0;
    for(int r = 0; r < k && ti[r] != kNone; ++r, ++n) {
        w[r] = expf((tv[r] - tv[0]) / p->temperature);
        total += w[r];
    }
    const float top_p = p->top_p > 0.f && p->top_p < 1.f ? p->top_p : 1.f;
    int keep = n;
    float acc = 0.f;
    for(int r = 0; r < n; ++r) {
        acc += w[r] / total;
        if(acc >= top_p) { keep = r + 1; break; }
    }
    float kept = 0.f;
    for(int r = 0; r < keep; ++r) kept += w[r];
    // Core's draw: the first kept token whose cumulative weight exceeds u * kept.
    const double target = crit::core::uniform(p->seed, p->position, p->stream, crit::core::DrawPurpose::sample) * kept;
    p->position += 1;
    float c = 0.f;
    int chosen = ti[keep - 1];
    for(int r = 0; r < keep; ++r) {
        c += w[r];
        if(target < c) { chosen = ti[r]; break; }
    }
    out[0] = chosen;
}

// One thread per logit: a clear bit in its row's mask word bans the token.
__global__ void mask_kernel(float* logits, int vocab, int stride, const std::uint32_t* mask, int mask_stride) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if(t >= vocab) return;
    if(!((mask[static_cast<std::size_t>(r) * mask_stride + t / 32] >> (t % 32)) & 1u))
        logits[static_cast<std::size_t>(r) * stride + t] = __int_as_float(0xff800000);  // -inf
}

// ---- The full-vocabulary law --------------------------------------------------------------------
// Rank order is (value descending, id ascending). Values map to unsigned keys that order like the
// floats, so a rank prefix is "key above a boundary, or equal key and id at most the boundary id".
// Prefix searches walk the key's eight 4-bit digits from the top, summing weights per digit, then
// resolve the boundary key's ties in id order. Every sum has a fixed order for a given vocabulary
// (per-thread FP32 over a fixed stride, then FP64 over threads), so a row's result never depends on
// timing or on the other rows of the launch.

__device__ __forceinline__ std::uint32_t order_key(float v) {
    const std::uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
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
    float min_weight = 0.f;         // min_p, applied when positive
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
    return L.min_weight > 0.f && !(w >= L.min_weight) ? 0.f : w;
}

__device__ Law compute_law(const float* values, int vocab, float temperature, std::int32_t top_k, float top_p,
                           float min_p, Shared& s) {
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
    if(!(temperature > 0.f)) return L;
    L.greedy = false;
    L.temperature = temperature;
    if(top_k > 0 && top_k < selectable) {
        std::int32_t id = -1;
        if(descend(values, vocab, [](std::int32_t, float v) { return masked(v) ? 0.f : 1.f; }, double(top_k), true, s, id)) {
            L.bounded = true;
            L.bound_id = id;
            L.bound_key = order_key(values[id]);
        }
    }
    if(top_p < 1.f) {
        const Law prefix = L;
        auto w = [&prefix](std::int32_t t, float v) { return prefix_weight(prefix, t, v); };
        const double S = mass(values, vocab, w, s);
        std::int32_t id = -1;
        if(descend(values, vocab, w, double(top_p) * S, true, s, id)) {
            L.bounded = true;
            L.bound_id = id;
            L.bound_key = order_key(values[id]);
        }
    }
    L.min_weight = min_p > 0.f ? min_p : 0.f;
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

__device__ float bias_of(const Request& q, std::int32_t t) {
    std::uint32_t lo = 0, hi = q.bias_count;
    while(lo < hi) {
        const std::uint32_t mid = (lo + hi) / 2;
        const std::int32_t id = q.bias_ids[mid];
        if(id == t) return q.bias_values[mid];
        if(id < t) lo = mid + 1;
        else hi = mid;
    }
    return 0.f;
}

// Core's steps 2, 4 and 5 for one token with n committed-plus-staged occurrences.
__device__ float processed(const Request& q, std::int32_t t, float logit, std::int32_t n) {
    float v = logit;
    if(q.bias_count) v += bias_of(q, t);
    if(masked(v)) return v;
    const bool seen = n > 0 || (q.prompt_seen && q.prompt_seen[t]);
    if(seen && q.repetition != 1.f) v = v > 0.f ? fminf(v / q.repetition, FLT_MAX) : fmaxf(v * q.repetition, -FLT_MAX);
    if(n > 0) v -= q.frequency * float(n) + q.presence;
    return v;
}

// Processed values of one row; `outputs` is Core's outputs() + column for the min_tokens rule.
__device__ void process_row(const float* logits, float* out, int vocab, const Request& q, const std::int32_t* staged,
                            std::uint32_t staged_count, std::uint64_t outputs) {
    for(int t = threadIdx.x; t < vocab; t += kLaw) out[t] = processed(q, t, logits[t], q.counts ? q.counts[t] : 0);
    __syncthreads();
    // Staged tokens count as outputs of this column; duplicates write the same value.
    for(std::uint32_t j = threadIdx.x; j < staged_count; j += kLaw) {
        const std::int32_t t = staged[j];
        if(t < 0 || t >= vocab) continue;
        std::int32_t n = q.counts ? q.counts[t] : 0;
        for(std::uint32_t i = 0; i < staged_count; ++i) n += staged[i] == t;
        out[t] = processed(q, t, logits[t], n);
    }
    __syncthreads();
    if(outputs < q.min_tokens)
        for(std::uint32_t j = threadIdx.x; j < q.suppress_count; j += kLaw) {
            const std::int32_t t = q.suppress_ids[j];
            if(t >= 0 && t < vocab) out[t] = neg_inf();
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

// `path` is shared memory for up to 63 staged tree tokens; process_row's first barrier orders its writes.
__device__ RowView view(const Rows& rows, const std::int32_t* fallback_staged, std::uint32_t fallback_stride, std::int32_t* path) {
    const std::uint32_t c = blockIdx.x, r = blockIdx.y;
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

__global__ void __launch_bounds__(kLaw) exact_select_kernel(const float* logits, int vocab, Params* p, std::int32_t* out) {
    __shared__ Shared s;
    const Params params = *p;
    if(fast_path(&params)) return;
    const Law law = compute_law(logits, vocab, params.temperature, params.top_k, params.top_p, 0.f, s);
    const double u = crit::core::uniform(params.seed, params.position, params.stream, crit::core::DrawPurpose::sample);
    const std::int32_t token = draw(logits, vocab, law, u, s);
    if(threadIdx.x == 0) {
        out[0] = token;
        p->position = params.position + 1;
    }
}

__global__ void __launch_bounds__(kLaw) sample_kernel(Rows rows, std::uint32_t purpose, std::int32_t* out, float* probabilities) {
    __shared__ Shared s;
    __shared__ std::int32_t path[64];
    const RowView r = view(rows, nullptr, 0, path);
    const Request q = *r.q;
    const int vocab = int(rows.vocab);
    process_row(r.logits, r.values, vocab, q, r.staged, r.staged_count, r.position);
    const Law law = compute_law(r.values, vocab, q.temperature, q.top_k, q.top_p, q.min_p, s);
    const double u = crit::core::uniform(q.seed, r.position, q.stream, crit::core::DrawPurpose(purpose));
    const std::int32_t token = draw(r.values, vocab, law, u, s);
    if(threadIdx.x == 0) out[r.row] = token;
    if(!probabilities) return;
    float* p = probabilities + r.row * rows.vocab;
    for(int t = threadIdx.x; t < vocab; t += kLaw) {
        float x = 0.f;
        if(law.argmax >= 0) x = law.greedy ? (t == law.argmax ? 1.f : 0.f) : float(double(kept_weight(law, t, r.values[t])) / law.total);
        p[t] = x;
    }
}

__global__ void __launch_bounds__(kLaw) verify_kernel(Verify v) {
    __shared__ Shared s;
    __shared__ std::int32_t path[64];
    const Rows& rows = v.target;
    const std::uint32_t c = blockIdx.x, K = rows.columns - 1;
    const RowView r = view(rows, v.drafts, K, path);
    const Request q = *r.q;
    const int vocab = int(rows.vocab);
    process_row(r.logits, r.values, vocab, q, r.staged, r.staged_count, r.position);
    const Law law = compute_law(r.values, vocab, q.temperature, q.top_k, q.top_p, q.min_p, s);
    std::int32_t token;
    const float* qrow = c < K && v.draft_probabilities ? v.draft_probabilities + (std::size_t(blockIdx.y) * K + c) * rows.vocab : nullptr;
    if(!qrow || law.argmax < 0 || law.greedy) {
        // The target's own draw: a greedy draft is accepted exactly when it equals it, and the bonus column is one too.
        token = draw(r.values, vocab, law, crit::core::uniform(q.seed, r.position, q.stream, crit::core::DrawPurpose::sample), s);
    } else {
        const std::int32_t d = v.drafts[std::size_t(blockIdx.y) * K + c];
        const bool valid = d >= 0 && d < vocab;
        const double wd = valid ? double(kept_weight(law, d, r.values[d])) : 0.;
        const double pd = wd / law.total;
        const double qd = valid ? double(qrow[d]) : 0.;
        const double ua = crit::core::uniform(q.seed, r.position, q.stream, crit::core::DrawPurpose::accept);
        if(ua * qd < pd) {
            token = d;
        } else {
            const double ur = crit::core::uniform(q.seed, r.position, q.stream, crit::core::DrawPurpose::residual);
            const double total = law.total;
            auto w = [&law, qrow, total](std::int32_t t, float x) {
                return fmaxf(float(double(kept_weight(law, t, x)) / total) - qrow[t], 0.f);
            };
            token = walk(r.values, vocab, w, ur * mass(r.values, vocab, w, s), s);
            // No residual mass left: the draft carries the whole law, so it stands.
            if(token < 0) token = d;
        }
    }
    if(threadIdx.x == 0) v.predictions[std::size_t(blockIdx.y) * (K + 1) + c] = token;
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

bool valid_rows(const Rows& rows) {
    return rows.logits && rows.request && rows.scratch && rows.vocab >= 1 && rows.vocab <= 0x7fffffffu &&
           rows.stride >= rows.vocab && rows.requests >= 1 && rows.requests <= 65535 && rows.columns >= 1 &&
           (!rows.staged || rows.staged_stride >= rows.staged_base + rows.columns - 1) &&
           (!rows.tree || (rows.tree_tokens && rows.columns <= 64));
}

} // namespace

std::size_t scratch_floats(std::uint32_t vocab, std::uint32_t rows) noexcept { return std::size_t(vocab) * rows; }

cudaError_t enqueue_mask(float* logits, int vocab, int stride, const std::uint32_t* mask, int mask_stride, int rows,
                         cudaStream_t stream) {
    if(!logits || !mask || vocab < 1 || rows < 1 || rows > 65535 || stride < vocab || mask_stride < (vocab + 31) / 32)
        return cudaErrorInvalidValue;
    mask_kernel<<<dim3((vocab + kThreads - 1) / kThreads, rows), kThreads, 0, stream>>>(logits, vocab, stride, mask, mask_stride);
    return cudaGetLastError();
}

cudaError_t enqueue_select(const float* logits, int vocab, Params* params, float* cand_values, std::int32_t* cand_ids,
                           std::int32_t* out, cudaStream_t stream) {
    if(!logits || vocab < 1 || !params || !cand_values || !cand_ids || !out) return cudaErrorInvalidValue;
    local_kernel<<<kStageBlocks, kThreads, 0, stream>>>(logits, vocab, params, cand_values, cand_ids);
    pick_kernel<<<1, 1024, 0, stream>>>(cand_values, cand_ids, params, out);
    exact_select_kernel<<<1, kLaw, 0, stream>>>(logits, vocab, params, out);
    return cudaGetLastError();
}

cudaError_t enqueue_sample(const Rows& rows, crit::core::DrawPurpose purpose, std::int32_t* out, float* probabilities,
                           cudaStream_t stream) {
    if(!valid_rows(rows) || !out) return cudaErrorInvalidValue;
    sample_kernel<<<dim3(rows.columns, rows.requests), kLaw, 0, stream>>>(rows, std::uint32_t(purpose), out, probabilities);
    return cudaGetLastError();
}

cudaError_t enqueue_verify(const Verify& v, cudaStream_t stream) {
    if(!valid_rows(v.target) || !v.predictions || (v.target.columns > 1 && !v.drafts) || v.target.tree) return cudaErrorInvalidValue;
    verify_kernel<<<dim3(v.target.columns, v.target.requests), kLaw, 0, stream>>>(v);
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
