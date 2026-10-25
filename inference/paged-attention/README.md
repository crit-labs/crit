# Batched paged attention

CUDA attention for a ragged batch of sequences whose keys and values live in fixed-size
pages addressed through block tables: the pages of Core's
[StateCache](../../core/src/state/README.md), with shared prefixes and copy-on-write tails.
One decode launch covers every sequence of a step, whatever their lengths, plus a varlen
causal prefill for ragged prompt chunks over a paged prefix. Pages hold BF16 values or
E4M3 codes with one scale per head vector (the representation of
[fp8-kv-attention](../fp8-kv-attention/README.md)). The launch shape depends only on host
counts, so one captured CUDA graph serves every length and table.

## When to use it

When an engine serves several requests at once (`concurrent_requests`, Core's
`ConcurrentModel` with mixed batches) or reuses prompt prefixes across requests: each
sequence keeps its own block table into one page pool, so memory follows the tokens held
rather than a per-request maximum, and forks and reused prefixes cost no copy beyond a
partial tail page. Not needed for a single contiguous sequence (use
[fp8-kv-attention](../fp8-kv-attention/README.md) or
[flash-prefill-attention](../flash-prefill-attention/README.md)), for attention-free
models, or for head dimensions other than 64, 128 and 256.

## How it works

**Pages.** Row `r` of page `p` is physical row `p * page_tokens + r`; rows are
`[kv_heads][head_dim]`, so a page id from `StateCache::block_table` addresses the same
rows in every layer. Sequence `s`'s key `j` is at
`table[s][j / page_tokens] * page_tokens + j % page_tokens`. Any `page_tokens` works: a
key tile gathers its rows one by one, so tiles may span pages.

**Batch.** Device arrays give each sequence its block table, `kv_lens[s]` (keys held,
the step's new keys included) and `q_start[s] .. q_start[s + 1]` (its query rows). With
`n` queries, query `i` sits at position `kv_len - n + i` and sees keys up to it. A
sequence with no queries is an idle slot, so a captured batch size can be padded.
The counts live on the device, so the launch cannot check them: a sequence with more
queries than the launch's `max_columns` (decode) or `max_queries` (prefill), more
queries than `kv_lens[s]`, or more keys than its `table_stride * page_tokens` table
addresses is not attended. Its rows read no keys and come out NaN in every head, and
the optional `Batch::fault` flag is set to 1 (the engine clears it), so a too-small
bound shows up instead of leaving rows unwritten. `BatchTables` rejects more queries
than keys and, given `page_tokens` (always from the StateCache), short tables on the
host; pass its `max_queries()` or more as the launch bound.

**Write.** `write` stores a step's new rows at the physical rows Core's `Round::kv`
names (`slot = page * page_tokens + offset`; negative slots are skipped). E4M3 pages take
`scale = max |x| / 448` per head vector and `code = e4m3(x / scale)` saturated.

**Decode and verify.** Blocks are `(key/value head, block, sequence)`. Block `b` of
`blocks` merges 128-key pieces `b, b + blocks, ...` of its sequence in key order, so its
work follows the tokens held, never the capacity. Each piece is walked in 32-key tiles
fetched into registers while the previous tile is used, stored (BF16) or dequantized
(E4M3: `code * scale` rounded to BF16) into a swizzled shared stage, and multiplied on
BF16 tensor cores with an FP32 online softmax. The rows of a block are the sequence's
query columns times the query heads of the key/value head (at most 64), so speculative
verify columns share every tile. Decode rows are few, so a block runs two key streams
(half its warps take even tiles, half odd tiles), each with its own softmax state.
A combine adds the `(block, stream)` partial results in that order with weights
`l exp(m - M)`.

**Determinism and invariance.** A tile after a row's position leaves that row's maximum,
sum and output bit for bit unchanged, and which tile goes to which block and stream
depends only on key positions. A query's result therefore depends only on its own
position and `blocks`: a column decoded alone equals the same column inside a
multi-column verify, and a sequence's result does not depend on the rest of the batch.

**Prefill.** Blocks own 128 queries of one head of one sequence (the sequence's latest
tile first, which has the most keys) and walk keys `0 ..` their last query in the same
tiles, masking keys after each query. Ragged chunk lengths, reused prefix pages and a
fork's own tail all read through the table.

**Pages behind the StateCache.** `DevicePages` (in [device_pages.h](code/device_pages.h))
holds the pages of all attention layers and implements `StateStorage::copy_rows` as one
2-D device copy per array on the engine's stream. `BatchTables` packs one step's tables,
lengths, offsets and write slots and uploads the used part.

## Sliding windows, sinks, soft-caps and trees

- **Window** (`window > 0`): a query at position `p` sees keys `p - window + 1 .. p`. Read
  the window group's pages through Core's `window_table`: pages a sequence has passed are
  `no_page` there, and the kernels never fetch a key before the oldest query's window, so
  those entries are never dereferenced. Decode starts each block's walk at the first piece
  inside the window (pieces stay assigned by position), so its work follows the window.
- **Sinks** (`sinks[h]`, scaled-score units): `exp(sink)` joins each row's denominator,
  in decode after the blocks' weights in the ordered combine.
- **Soft-cap** (`softcap > 0`): `score = softcap * tanh(score / softcap)` after the scale.
- **Trees** (`Decode::tree`): verify column `i` of sequence `s` sees the round's column `j`
  only when bit `j` of `tree[q_start[s] + i]` is set: its ancestors and itself, listed in
  topological order, for speculative token trees (see [draft trees](../draft-trees/README.md)).
  With a window, a column's window is measured from its depth (`popcount - 1`), so trees
  need `window > 64`.

Masked keys contribute exact zeros, so a window, a tree mask and skipped pieces change no
bits of the visible keys' contribution: a windowed column decoded alone still equals the
same column in a verify.

## Worked example

From `code/`:

```python
import math, random
from reference import PIECE, attend, batch, physical_row

random.seed(3)
page_tokens, dim = 4, 8
storage = [([random.gauss(0, 1) for _ in range(dim)], [random.gauss(0, 1) for _ in range(dim)]) for _ in range(400)]
assert physical_row([7, 2], page_tokens, 5) == 2 * page_tokens + 1   # key 5: second page (id 2), offset 1
# Two sequences share their first page (a fork or a reused prefix) and diverge in their own tails.
shared = [9]
seq_a = (shared + list(range(10, 90)), 300, [[random.gauss(0, 1) for _ in range(dim)]])           # one decode column
seq_b = (shared + list(range(90, 100)), 37, [[random.gauss(0, 1) for _ in range(dim)] for _ in range(3)])  # a 3-column verify
one = batch(storage, page_tokens, [seq_a, seq_b], 1 / math.sqrt(dim), blocks=1)
split = batch(storage, page_tokens, [seq_a, seq_b], 1 / math.sqrt(dim), blocks=3)  # 300 keys: three 128-key pieces
assert len(one) == 4 and 300 > 2 * PIECE
assert all(abs(x - y) < 1e-12 for r1, r2 in zip(one, split) for x, y in zip(r1, r2))
# The last verify column alone (its own position, same keys) equals the column inside the verify.
alone = attend(storage, page_tokens, seq_b[0], 37, seq_b[2][2:], 1 / math.sqrt(dim))
assert all(abs(x - y) < 1e-12 for x, y in zip(alone[0], one[3]))
# A chain tree mask is the causal default; with a window, pages before it are never read (no_page in Core's
# window table: a bogus id here would raise if it were).
assert batch(storage, page_tokens, [seq_b], 1 / math.sqrt(dim), trees=[0b1, 0b11, 0b111]) == batch(storage, page_tokens, [seq_b], 1 / math.sqrt(dim))
passed = [10 ** 6] * 50 + seq_a[0][50:]   # pages 0..49 hold keys 0..199, all before the window of key 299
windowed = attend(storage, page_tokens, passed, 300, seq_a[2], 1 / math.sqrt(dim), window=100, sink=0.5, softcap=4.0)
assert windowed == attend(storage, page_tokens, seq_a[0], 300, seq_a[2], 1 / math.sqrt(dim), window=100, sink=0.5, softcap=4.0)
```

The CUDA interface ([paged_attention.h](code/paged_attention.h),
[paged_attention.cu](code/paged_attention.cu), [device_pages.h](code/device_pages.h)):

```cpp
namespace pa = crit::paged_attention;
pa::DevicePages pages(layers, cache_pages, page_tokens, kv_heads, head_dim, pa::Format::e4m3, stream);
crit::core::StateCache cache(config, pages, resources);      // config.page_tokens == page_tokens
pa::BatchTables tables(max_seqs, max_pages_per_seq, max_new_rows);
for(auto seq : step) {                                       // after cache.begin(seq, rows) for each sequence
    tables.add_slots(round[seq].kv, page_tokens);
    tables.add(cache, seq);                                  // its staged rows are its queries
}
const pa::Batch batch = tables.upload(stream);
pa::write(pages.layer(l), k_rows, v_rows, kv_heads * head_dim, tables.rows(), tables.device_slots(), stream);
pa::decode({q, heads * head_dim, pages.layer(l), batch, out, heads * head_dim, heads, scale,
            max_columns, blocks, partial}, stream);
// or a varlen prompt step: pa::prefill({q, ..., pages.layer(l), batch, out, ..., heads, scale, tables.max_queries()}, stream)
```

`partial` holds `decode_partial_floats(seqs, kv_heads, head_dim, blocks, max_columns, heads)`
floats. Choose `blocks` from the device's SM count so that `seqs * kv_heads * blocks`
fills it (about the SM count divided by `seqs * kv_heads`, at least 1), and keep it fixed
for a captured graph.

## Invariants

- Write a step's keys and values before the attention that reads them, on the same stream.
- Pages the cache shares or indexes are never written: write only at `Round::kv` slots,
  which the cache guarantees are private; copy-on-write copies go through `copy_rows`.
- Released pages stay readable until `retire()` covers their epoch; call `advance_epoch`
  or `retire(epoch())` only after the kernels reading them are ordered before reuse.
- Piece-to-block and tile-to-stream assignment and the combine order depend only on key
  positions; keep them so when changing tile sizes, or verify columns stop matching
  single-column decode.
- E4M3 scales are per row and key/value head and applied when a tile is dequantized; the
  KV representation belongs in the cache's prefix `identity`.

## Costs and validation

The kernels need sm_80 or later (BF16 `mma.sync`, `ldmatrix`, `cp.async`; E4M3
conversions are native from sm_89 and emulated before) and up to 96 KiB of shared
memory per block at head dimension 256. They build for sm_80, sm_86, sm_89 and
sm_120; the tests below ran on sm_120 only.

`tests/inference/paged_attention_tests.cpp` drives Core's StateCache with prefix
caching, ragged prefills of 64, 300, 130, 9 (after 64 reused rows) and 1 rows, a fork
whose shared tail is copied on write, an idle slot, 3-column verifies and decode steps
across page boundaries, for head dimension / query / KV heads 64/8/2, 128/4/4, 256/16/2 and
128/12/2, and both formats. Every output row is compared with the attention-state CPU
reference (`PagedAttentionPlan`, FP64 accumulation) over the same pages: within 2^-8 of
the output magnitude (BF16 probabilities; at most 2.6e-3 measured on an RTX 5090). E4M3
codes and scales equal a host encoder exactly; decode columns alone equal verify columns
and lone sequences equal batched ones bit for bit, and its `widths` case checks every verify
width from 1 to the 64 rows' limit (16, 64, 8, 21 and 10 columns at 4, 1, 8, 3 and 6 query
heads per key/value head, head dimensions 128, 64, 256, 128 and 256) against columns decoded alone at 2,048, 3,001, 7,168 and 16,384 keys over
shuffled pages, with and without a window, for 1, 5 and SM-count / `kv_heads` blocks, and
with `max_columns` 1 or the largest width; a captured graph replays new tables
and lengths with the bits of a direct launch. Sequences whose device counts break the
batch (too many queries for `max_columns` or `max_queries`, more queries than keys,
more keys than the table addresses) come out NaN with the fault flag set, next to a
valid sequence attended as usual, in decode and prefill.
[attention_variants_tests.cu](../../tests/inference/cuda/attention_variants_tests.cu) checks
windowed prefill and a tree verify with sinks and soft-caps against an FP64 reference for
both formats, with every page before the window pointing at a page of NaN, and windowed
decode columns alone against the verify bit for bit.

Measured on an RTX 5090 (170 SMs, 1,438 GB/s from `crit eval probe`), decode of one
query per sequence at 4,096 keys, head dimension 128, 32 query and 8 key/value heads,
16-token pages in shuffled order: 64 sequences took 0.80 ms with BF16 pages (1,336 GB/s
of KV read) and 0.57 ms with E4M3 pages (974 GB/s); one sequence took 26 µs (BF16) and
32 µs (E4M3), the contiguous fp8-kv-attention decode 36 µs at the same length on that
run. These numbers illustrate one device and shape, not settings to copy. E4M3 decode is
bound by dequantization and the softmax of few rows rather than by bandwidth.
