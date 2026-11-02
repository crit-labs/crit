#pragma once
#include <crit/state.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// The batched reference engine's paged attention over Core's StateCache: the host half. A batch of sequences whose keys and values live in fixed-size
// pages addressed by block tables (one PageId per page_tokens rows, shared prefixes, copy-on-write tails). This header
// holds what does not need a device: the page format, the step's packed batch tables, host codecs for BF16 and E4M3,
// a host page store in the kernels' layout and an FP64 reference of the kernels' attention. The CUDA kernels and the
// device page store are in paged_attention.h.
//
// Page layout, per attention layer: row r of page p is physical row p * page_tokens + r, and physical rows are laid out
// [row][kv_heads][head_dim]. BF16 pages store values; E4M3 pages store one-byte codes with one FP32 scale per row and
// key/value head ([row][kv_heads]), value = code * scale with scale = max |x| over the head vector / 448 (1 for an
// all-zero vector). The kernels round every key and value to BF16 (E4M3: code * scale rounded once) before use.
//
// Visibility: sequence s holds kv_len keys and its newest n rows are its queries; query i sits at position
// p = kv_len - n + i and sees keys 0 .. p (its own key included, so a step writes its keys first). Variants several
// model families use: a sliding window (keys p - window + 1 .. p), per-head sink logits that join each row's softmax
// denominator, a logit soft-cap (score = cap * tanh(score / cap) after the scale) and, for decode and verify, tree
// masks: bit j of column i's mask is set when column i sees column j of its round (the key at kv_len - n + j); a column
// sees exactly its ancestors and itself, in topological order, and with a window its window is measured from its
// depth (popcount - 1).
namespace crit::paged_attention {
enum class Format : int { bf16 = 0, e4m3 = 1 };

// Round to nearest even, as __float2bfloat16 does; NaN stays NaN.
std::uint16_t to_bf16(float) noexcept;
float from_bf16(std::uint16_t) noexcept;
// OCP E4M3 (FN: no infinities, 0x7f and 0xff are NaN), round to nearest even with saturation to +-448, as
// __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3) does.
std::uint8_t to_e4m3(float) noexcept;
float from_e4m3(std::uint8_t) noexcept;
// The E4M3 scale of one head vector: max |x| / 448, or 1 when every value is zero.
float e4m3_scale(std::span<const float> head) noexcept;

// One step's batch tables, packed into the int32 words the kernels read: block tables [max_seqs][max_pages], lengths
// [max_seqs], query starts [max_seqs + 1] and write slots [max_rows]. The device side (BatchTables in
// paged_attention.h) uploads the used part of each array. Bounds are fixed at construction; filling never
// allocates.
class Tables {
public:
    // invalid_argument unless max_seqs and max_pages are positive and max_rows nonnegative.
    Tables(int max_seqs, int max_pages, int max_rows);
    void clear() noexcept;
    // A sequence holding kv_len rows through `table`, whose newest `queries` rows are queried. With page_tokens > 0 the
    // table must also address every row (kv_len <= table.size() * page_tokens). length_error past max_seqs;
    // invalid_argument for a longer table, more queries than rows, a negative count or a short table.
    void add(std::span<const core::PageId> table, std::uint32_t kv_len, int queries, int page_tokens = 0);
    // A sequence's committed and staged rows, querying its staged rows (a round's new tokens).
    void add(const core::StateCache&, core::SequenceId);
    // Padding for a captured batch size: no queries, no keys.
    void add_idle();
    // Physical rows (page * page_tokens + offset) where new rows go, in query-row order: each round's Round::kv.
    // length_error past max_rows.
    void add_slots(std::span<const core::RowSlot>, int page_tokens);
    int seqs() const noexcept;
    int queries() const noexcept;
    int rows() const noexcept;
    int max_queries() const noexcept;     // the launch's max_columns (decode) or max_queries (prefill) at least
    int max_seqs() const noexcept;
    int max_pages() const noexcept;       // the table stride
    int max_rows() const noexcept;
    // The packed words and where each array starts in them.
    std::span<const std::int32_t> words() const noexcept;
    std::size_t lens_at() const noexcept;
    std::size_t starts_at() const noexcept;
    std::size_t slots_at() const noexcept;
private:
    int max_seqs_, max_pages_, max_rows_;
    std::vector<std::int32_t> words_;
    int seqs_ = 0, queries_ = 0, rows_ = 0, max_queries_ = 0;
};

// Pages of `layers` attention layers in host memory, laid out as the device pages are: a StateStorage for a
// StateCache, so the cache's copy-on-write copies reach it. For CPU engines, tests and oracles of device kernels.
class HostPages final : public core::StateStorage {
public:
    // invalid_argument unless every count is positive and head_dim is 64, 128 or 256 (the kernels' widths; the
    // reference itself takes any).
    HostPages(int layers, int pages, int page_tokens, int kv_heads, int head_dim, Format);
    // Store `rows` rows of keys and values (BF16 bits, row t at k + t * stride, head g at column g * head_dim) at
    // physical rows slots[t] of `layer`, encoded as the device write kernel encodes them; a negative slot skips its row.
    // out_of_range for a slot outside the pages.
    void write(int layer, std::span<const std::uint16_t> k, std::span<const std::uint16_t> v, int stride, int rows,
               std::span<const std::int32_t> slots);
    // Key or value `d` of head `g` at physical row `row` as the kernels read it: the BF16 value, or bf16(code * scale).
    float key(int layer, std::size_t row, int g, int d) const;
    float value(int layer, std::size_t row, int g, int d) const;
    // Raw storage of one layer: BF16 bits (two bytes per element) or E4M3 codes, and the E4M3 scales.
    std::span<const std::byte> keys(int layer) const;
    std::span<const std::byte> values(int layer) const;
    std::span<const float> key_scales(int layer) const;
    std::span<const float> value_scales(int layer) const;
    int layers() const noexcept;
    int pages() const noexcept;
    int page_tokens() const noexcept;
    int kv_heads() const noexcept;
    int head_dim() const noexcept;
    Format format() const noexcept;
    void copy_rows(core::PageId from, core::PageId to, std::uint32_t rows) override;
    void copy_slot(std::uint32_t, core::SlotId, core::SlotId) override;   // logic_error: no recurrent state
private:
    int layers_, pages_, page_tokens_, kv_heads_, head_dim_;
    Format format_;
    std::size_t layer_rows_, element_bytes_;
    std::vector<std::byte> k_, v_;
    std::vector<float> k_scale_, v_scale_;
};

// One attention problem over host pages, the reference of the device kernels' Decode and Prefill problems. Queries
// are FP32 values (pass the BF16-rounded values the kernels read), [rows][q_stride] with head h at column h * head_dim;
// the result is FP32 [rows][heads * head_dim]. Query heads h use key/value head h / (heads / kv_heads).
struct Problem {
    std::span<const float> q;
    std::size_t q_stride = 0;
    int layer = 0, heads = 0;
    float scale = 0.f;                    // multiplies q . k before the softmax
    int window = 0;                       // > 0: sliding window of `window` keys, the query's own included
    std::span<const float> sinks = {};    // [heads] sink logits in scaled-score units; empty: none
    float softcap = 0.f;                  // > 0: score = softcap * tanh(score / softcap)
    std::span<const std::uint64_t> tree = {};  // [queries] tree masks (decode and verify); empty: the causal chain
};
// FP64 softmax attention of every query row of `tables` over `pages`, keys and values as the kernels read them. Each
// result depends only on its query, its position and the keys it sees: never on the batch or other queries. A row that
// sees no key and has no sink is zero. invalid_argument for a problem the kernels reject (heads not a multiple of
// kv_heads, a short q_stride, a negative window or soft-cap, sinks or tree masks of the wrong size, tree masks with a
// window of 64 keys or fewer) or tables addressing pages outside `pages`.
std::vector<float> attend(const HostPages&, const Tables&, const Problem&);
}
