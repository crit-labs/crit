// Reference kernels for token trees; see draft_tree.h and tree.py.
#include "draft_tree.h"

#include <cfloat>
#include <climits>

namespace crit::techniques::draft_trees {
namespace {

namespace control = device_round::control;
constexpr int kThreads = 256;

__device__ __forceinline__ bool before(float v, int i, float w, int j) { return v > w || (v == w && i < j); }
// NaN ranks as -inf, so a row of NaN still yields in-range token ids.
__device__ __forceinline__ float ranked(float x) { return x == x ? x : -INFINITY; }

// Block-wide best (value, index) in rank order (value descending, index ascending); every thread receives it.
__device__ void block_best(float& v, int& i) {
    __shared__ float sv[kThreads / 32];
    __shared__ int si[kThreads / 32];
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
        for(int w = 1; w < kThreads / 32; ++w)
            if(before(sv[w], si[w], v, i)) { v = sv[w]; i = si[w]; }
        sv[0] = v;
        si[0] = i;
    }
    __syncthreads();
    v = sv[0];
    i = si[0];
}

__device__ double block_sum(double x) {
    __shared__ double part[kThreads / 32];
    for(int o = 16; o; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
    __syncthreads();
    if((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    double t = 0.;
    for(int w = 0; w < kThreads / 32; ++w) t += part[w];
    return t;
}

// Block (frontier entry f, request r): the entry's `branch` best tokens with log-softmax scores. A row without a
// finite logit gives its tokens a score of -inf, which selection never takes.
__global__ void __launch_bounds__(kThreads) expand_kernel(const float* logits, int stride, int vocab, const std::int32_t* frontier,
                                                          int width, int branch, Candidates c) {
    const int f = blockIdx.x, r = blockIdx.y;
    const std::int32_t parent = frontier[std::size_t(r) * width + f];
    const std::int32_t base = c.count[r] + f * branch;
    const std::size_t row0 = std::size_t(r) * c.capacity;
    if(parent < -1) {  // an unused entry leaves a hole that selection never takes
        for(int i = threadIdx.x; i < branch; i += kThreads) {
            c.token[row0 + base + i] = -1;
            c.parent[row0 + base + i] = -1;
            c.score[row0 + base + i] = -FLT_MAX;
            c.depth[row0 + base + i] = -1;
        }
        return;
    }
    const float* row = logits + (std::size_t(r) * width + f) * stride;
    float top = -INFINITY;
    int at = INT_MAX;
    for(int t = threadIdx.x; t < vocab; t += kThreads)
        if(before(ranked(row[t]), t, top, at)) { top = ranked(row[t]); at = t; }
    block_best(top, at);
    const bool finite = top > -INFINITY;
    float local = 0.f;
    if(finite)
        for(int t = threadIdx.x; t < vocab; t += kThreads) local += expf(ranked(row[t]) - top);
    const float lse = finite ? top + float(log(block_sum(double(local)))) : -INFINITY;
    const float base_score = parent >= 0 ? c.score[row0 + parent] : 0.f;
    const std::int32_t depth = parent >= 0 ? c.depth[row0 + parent] + 1 : 1;
    float pv = FLT_MAX;
    int pi = -1;
    for(int k = 0; k < branch; ++k) {
        float v = -INFINITY;
        int i = INT_MAX;
        for(int t = threadIdx.x; t < vocab; t += kThreads) {
            const float x = ranked(row[t]);
            if(before(pv, pi, x, t) && before(x, t, v, i)) { v = x; i = t; }
        }
        block_best(v, i);
        if(threadIdx.x == 0) {
            const std::size_t at_k = row0 + base + k;
            c.token[at_k] = i;
            c.parent[at_k] = parent;
            c.score[at_k] = finite && v > -INFINITY ? base_score + (v - lse) : -INFINITY;
            c.depth[at_k] = depth;
        }
        pv = v;
        pi = i;
    }
}

__global__ void advance_kernel(std::int32_t* count, int add, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < rows) count[r] += add;
}

// Block per request: m rounds of block-wide best among unchosen candidates of the depth.
__global__ void __launch_bounds__(kThreads) select_kernel(Candidates c, int depth, int m, std::int32_t* out) {
    const int r = blockIdx.x;
    const std::size_t row0 = std::size_t(r) * c.capacity;
    const int n = c.count[r];
    float pv = FLT_MAX;
    int pi = -1;
    for(int k = 0; k < m; ++k) {
        float v = -FLT_MAX;
        int i = INT_MAX;
        for(int j = threadIdx.x; j < n; j += kThreads) {
            const int d = c.depth[row0 + j];
            if(d < 1 || (depth > 0 && d != depth)) continue;
            const float s = ranked(c.score[row0 + j]);
            if(before(pv, pi, s, j) && before(s, j, v, i)) { v = s; i = j; }
        }
        block_best(v, i);
        if(threadIdx.x == 0) out[std::size_t(r) * m + k] = i == INT_MAX ? -1 : i;
        if(i == INT_MAX) {
            for(int rest = k + 1 + int(threadIdx.x); rest < m; rest += kThreads) out[std::size_t(r) * m + rest] = -1;
            return;
        }
        pv = v;
        pi = i;
    }
}

// One thread per request: nodes by depth then rank, parents through the candidate links, masks and depths.
__global__ void build_kernel(Candidates c, Build b) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= b.rows) return;
    const int m = b.nodes - 1;
    const std::size_t row0 = std::size_t(r) * c.capacity;
    const std::int32_t* kept = b.kept + std::size_t(r) * m;
    std::int32_t order[kMaxNodes];   // kept ranks in node order
    int n = 0;
    for(int k = 0; k < m && kept[k] >= 0; ++k) {
        // Insertion by depth keeps rank order within a depth.
        const int d = c.depth[row0 + kept[k]];
        int at = n;
        while(at > 0 && c.depth[row0 + kept[order[at - 1]]] > d) { order[at] = order[at - 1]; --at; }
        order[at] = k;
        ++n;
    }
    std::uint64_t* masks = b.masks + std::size_t(r) * b.nodes;
    std::int32_t* depths = b.depths + std::size_t(r) * b.nodes;
    std::int32_t* parents = b.parents + std::size_t(r) * m;
    std::int32_t* drafts = b.drafts + std::size_t(r) * m;
    std::int32_t* tokens = b.tokens ? b.tokens + std::size_t(r) * b.nodes : nullptr;
    masks[0] = 1ull;
    depths[0] = 0;
    if(tokens) tokens[0] = b.roots[std::size_t(r) * b.root_stride];
    for(int i = 0; i < n; ++i) {
        const std::int32_t cand = kept[order[i]], link = c.parent[row0 + cand];
        int parent = 0;
        for(int j = 0; j < i; ++j)
            if(kept[order[j]] == link) { parent = j + 1; break; }
        const int node = i + 1;
        parents[i] = parent;
        drafts[i] = c.token[row0 + cand];
        masks[node] = masks[parent] | (1ull << node);
        depths[node] = depths[parent] + 1;
        if(tokens) tokens[node] = drafts[i];
    }
    // Fewer kept candidates than nodes: the rest are padding nodes (token 0 under the root, so they embed and attend
    // like any node) with parent -1, which the walk never follows; their rows are never kept.
    for(int i = n; i < m; ++i) {
        parents[i] = -1;
        drafts[i] = 0;
        masks[i + 1] = 1ull | (1ull << (i + 1));
        depths[i + 1] = 1;
        if(tokens) tokens[i + 1] = 0;
    }
}

__device__ bool is_stop(const std::int32_t* stops, std::uint32_t count, std::int32_t token) {
    for(std::uint32_t i = 0; i < count; ++i)
        if(stops[i] == token) return true;
    return false;
}

// One thread per request; the control and result writes mirror device_round's greedy accept kernel.
__global__ void tree_accept_kernel(device_round::AcceptRows rows, const std::int32_t* parents_all, std::int32_t* path_all) {
    const auto row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows.rows) return;
    const auto k = rows.draft_tokens, nodes = k + 1;
    const auto* tokens = rows.drafts + std::size_t(row) * k;          // node i's token at tokens[i - 1]
    const auto* parents = parents_all + std::size_t(row) * k;         // node i's parent at parents[i - 1]
    const auto* predictions = rows.predictions + std::size_t(row) * nodes;
    auto* control = rows.control + std::size_t(row) * control::fields;
    auto* result = rows.result + std::size_t(row) * device_round::result_width(k);
    auto* out_tokens = result + 1;
    std::int32_t* path = path_all + std::size_t(row) * nodes;
    const std::uint32_t stop_count = rows.device_stop_count ? std::uint32_t(*rows.device_stop_count) : rows.stop_count;
    const auto budget = control[control::output_budget];
    const std::uint32_t remaining = budget > 0 ? std::uint32_t(budget) : 0u;
    std::uint32_t out = 0, matched = 0, v = 0;
    path[0] = 0;
    while(out < remaining) {
        const std::int32_t t = predictions[v];
        std::uint32_t child = 0;
        for(std::uint32_t i = 1; i < nodes; ++i)
            if(std::uint32_t(parents[i - 1]) == v && tokens[i - 1] == t) { child = i; break; }
        out_tokens[out++] = t;
        if(!child) break;
        path[++matched] = std::int32_t(child);
        v = child;
        if(is_stop(rows.stops, stop_count, t)) break;
    }
    for(auto i = matched + 1; i < nodes; ++i) path[i] = -1;
    for(auto i = out; i < k + 1; ++i) out_tokens[i] = -1;
    result[0] = std::int32_t(out);
    control[control::n_emitted] = std::int32_t(out);
    control[control::n_matched] = std::int32_t(matched);
    control[control::base_frontier] = control[control::frontier];
    control[control::base_last] = control[control::last_token];
    control[control::frontier] += std::int32_t(out);
    if(out) control[control::last_token] = out_tokens[out - 1];
    control[control::commit_tokens] = std::int32_t(out);
}

// Block (array, request): moves path[i] -> i in increasing i; 16-byte words when aligned. Thread t owns word t of
// every row, so for each word the moves run in program order on one thread: move j reads node path[j] before a later
// move could write position path[j].
__global__ void __launch_bounds__(kThreads) compact_kernel(const RowArray* arrays, const std::int32_t* control_all,
                                                           const std::int32_t* path_all, int nodes) {
    const RowArray a = arrays[blockIdx.x];
    const int r = blockIdx.y;
    const std::int32_t* control = control_all + std::size_t(r) * control::fields;
    const std::int32_t* path = path_all + std::size_t(r) * nodes;
    const int m = min(control[control::n_emitted], nodes);
    const std::int32_t base = control[control::base_frontier];
    auto slot = [&](int node) { return a.slots ? a.slots[std::size_t(r) * nodes + node] : base + node; };
    auto* bytes = static_cast<unsigned char*>(a.base);
    const bool wide = a.row_bytes % 16 == 0 && a.row_stride % 16 == 0 && (reinterpret_cast<std::uintptr_t>(a.base) & 15u) == 0;
    for(int i = 1; i < m; ++i) {
        const int from = path[i];
        if(from == i || from < 0) continue;
        unsigned char* dst = bytes + std::size_t(slot(i)) * a.row_stride;
        const unsigned char* src = bytes + std::size_t(slot(from)) * a.row_stride;
        if(wide) {
            for(std::size_t w = threadIdx.x; w < a.row_bytes / 16; w += kThreads)
                reinterpret_cast<uint4*>(dst)[w] = reinterpret_cast<const uint4*>(src)[w];
        } else {
            for(std::size_t w = threadIdx.x; w < a.row_bytes; w += kThreads) dst[w] = src[w];
        }
    }
}

__global__ void settle_kernel(const std::int32_t* control_all, std::int32_t* path_all, int nodes, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= rows) return;
    const int m = min(control_all[std::size_t(r) * control::fields + control::n_emitted], nodes);
    for(int i = 0; i < m; ++i) path_all[std::size_t(r) * nodes + i] = i;
}

// Block (position d, request r): the number of tokens ranked before the target's prediction in the drafter's row
// (logit desc, id asc), counted in the histogram when the round compared position d.
__global__ void __launch_bounds__(kThreads) draft_ranks_kernel(DraftRanks p) {
    const int d = blockIdx.x, r = blockIdx.y;
    const std::int32_t* control_row = p.control + std::size_t(r) * control::fields;
    const std::int32_t matched = control_row[control::n_matched], emitted = control_row[control::n_emitted];
    const std::int32_t token = p.predictions[std::size_t(r) * (p.draft_tokens + 1) + d];
    // Position matched was compared only when the walk emitted a token there (a correction); a round that ended on
    // its output budget or after an accepted stop emitted exactly the accepted drafts and never reached it.
    const bool reached = d < matched || (d == matched && emitted > matched);
    if(!reached || token < 0 || token >= p.vocab) return;
    const float* row = p.logits + (std::size_t(r) * p.draft_tokens + d) * p.stride;
    const float x = row[token];
    double ahead = 0.;
    for(int t = threadIdx.x; t < p.vocab; t += kThreads) ahead += before(row[t], t, x, token) ? 1. : 0.;
    const auto rank = static_cast<int>(block_sum(ahead));
    if(threadIdx.x == 0) atomicAdd(p.histogram + std::size_t(d) * (p.ranks + 1) + min(rank, p.ranks), 1ull);
}

bool valid(const Candidates& c) { return c.token && c.parent && c.score && c.depth && c.count && c.capacity > 0; }

}  // namespace

cudaError_t enqueue_expand(const float* logits, int stride, int vocab, const std::int32_t* frontier, int width, int branch,
                           const Candidates& c, int rows, cudaStream_t stream) {
    if(!valid(c) || !logits || !frontier || vocab < 1 || stride < vocab || width < 1 || width > 65535 || branch < 1 ||
       branch > kMaxBranch || rows < 1 || rows > 65535)
        return cudaErrorInvalidValue;
    expand_kernel<<<dim3(width, rows), kThreads, 0, stream>>>(logits, stride, vocab, frontier, width, branch, c);
    advance_kernel<<<(rows + 127) / 128, 128, 0, stream>>>(c.count, width * branch, rows);
    return cudaGetLastError();
}

cudaError_t enqueue_select(const Candidates& c, int depth, int m, std::int32_t* out, int rows, cudaStream_t stream) {
    if(!valid(c) || !out || m < 1 || depth < 0 || rows < 1) return cudaErrorInvalidValue;
    select_kernel<<<rows, kThreads, 0, stream>>>(c, depth, m, out);
    return cudaGetLastError();
}

cudaError_t enqueue_build(const Candidates& c, const Build& b, cudaStream_t stream) {
    if(!valid(c) || !b.kept || !b.drafts || !b.parents || !b.masks || !b.depths || (b.tokens && !b.roots) || b.nodes < 2 ||
       b.nodes > kMaxNodes || b.rows < 1 || b.root_stride < 1)
        return cudaErrorInvalidValue;
    build_kernel<<<(b.rows + 63) / 64, 64, 0, stream>>>(c, b);
    return cudaGetLastError();
}

cudaError_t enqueue_tree_accept(const device_round::AcceptRows& rows, const std::int32_t* parents, std::int32_t* path,
                                cudaStream_t stream) {
    if(!rows.rows) return cudaSuccess;
    if(!rows.drafts || !rows.predictions || !rows.control || !rows.result || !parents || !path || rows.draft_tokens < 1 ||
       rows.draft_tokens + 1 > std::uint32_t(kMaxNodes) || ((rows.stop_count || rows.device_stop_count) && !rows.stops))
        return cudaErrorInvalidValue;
    tree_accept_kernel<<<(rows.rows + 127) / 128, 128, 0, stream>>>(rows, parents, path);
    return cudaGetLastError();
}

cudaError_t enqueue_draft_ranks(const DraftRanks& p, cudaStream_t stream) {
    if(!p.rows) return cudaSuccess;
    if(!p.logits || !p.predictions || !p.control || !p.histogram || p.vocab < 1 || p.stride < p.vocab || p.draft_tokens < 1 ||
       p.draft_tokens > 65535 || p.rows < 0 || p.rows > 65535 || p.ranks < 1)
        return cudaErrorInvalidValue;
    draft_ranks_kernel<<<dim3(p.draft_tokens, p.rows), kThreads, 0, stream>>>(p);
    return cudaGetLastError();
}

cudaError_t enqueue_compact(const RowArray* arrays, int count, const std::int32_t* control, const std::int32_t* path, int nodes,
                            int rows, cudaStream_t stream) {
    if(!count) return cudaSuccess;
    if(!arrays || !control || !path || count < 0 || nodes < 1 || nodes > kMaxNodes || rows < 1 || rows > 65535)
        return cudaErrorInvalidValue;
    compact_kernel<<<dim3(count, rows), kThreads, 0, stream>>>(arrays, control, path, nodes);
    return cudaGetLastError();
}

cudaError_t enqueue_settle(const std::int32_t* control, std::int32_t* path, int nodes, int rows, cudaStream_t stream) {
    if(!control || !path || nodes < 1 || rows < 1) return cudaErrorInvalidValue;
    settle_kernel<<<(rows + 127) / 128, 128, 0, stream>>>(control, path, nodes, rows);
    return cudaGetLastError();
}

}  // namespace crit::techniques::draft_trees
