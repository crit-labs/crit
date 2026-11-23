# Transactional KV, shared pages and explicit-prefix attention

Stage attention state provisionally, expose the intended causal prefix to a query,
and commit only the accepted part. A Python host formula makes that behavior
executable without a page layout or kernel. The plugin's device form is a paged
pool in torch: every layer's keys and values in pages, each sequence's page
table, copy-on-write forks and captured rounds.

## When to use it

Use this when request state may be partially accepted, rolled back or
checkpointed, especially during speculative verification, or as a small attention
reference. It does not describe recurrent state or every attention variant. Measure
paging against contiguous state for single-request workloads; it pays off with
variable-length concurrent state, prefix sharing or branches.

## How it works

`KVState` holds ordered `committed` and `pending` rows (key width `K`, value width
`V`). `stage` validates and appends to pending within capacity; `commit(n)` keeps
the first `n` pending rows and drops the rest (no argument commits all; a bad count
changes nothing); `rollback()` drops pending; `attend(query)` reads committed rows,
or all rows with `include_pending=True`:

```text
score_i = dot(query, key_i) * scale       default scale = 1 / sqrt(K)
weight_i = exp(score_i - max(score)) / sum_j exp(score_j - max(score))
output[c] = sum_i weight_i * value_i[c]   for c in 0..V-1
```

`include_pending=True` is not a per-query causal mask: an earlier query in a
verification block must see only its own prefix, or use a backend with an
explicit mask.

`rope_table(rope, head_dim, max_position_embeddings, sequence_length)`
([rope_table.py](code/rope_table.py)) gives the rotary frequencies from the model's
`config.json`: the inverse frequency of each rotated pair (`rotary_dim(rope,
head_dim)` dimensions rotate) and the attention factor that scales cos and sin,
as Hugging Face's `ROPE_INIT_FUNCTIONS` derive them, for `default`, `mrope`,
`linear`, dynamic NTK, YaRN (with `mscale`, `attention_factor` and `truncate`),
`llama3`, `longrope` and `proportional` (a table as wide as the head, of which
the first `partial_rotary_factor` turns and the rest keeps its values). `rope` is
`crit_core.model`'s `RopeConfig` (`load_model_config(...).rope`, or one layer
type's of `rope_by_layer_type` where the config keys its rope by layer type), `max_position_embeddings` the declared length dynamic
NTK needs and YaRN, Llama 3 and LongRoPE read when the rope object leaves factor or
original length out; `sequence_length` picks dynamic NTK's base and LongRoPE's long
factors past the original length. Other types and missing parameters are refused.
`tests/test_rope_tables.py` checks it against Transformers.

## What the code supplies

| File | Supplies |
| --- | --- |
| [implementation.py](code/implementation.py) | `KVState`, its transaction and a stable-softmax formula; optional `backend(query, rows, scale)` |
| [rope_table.py](code/rope_table.py) | `rope_table`, `rotary_dim`, `RopeTable`: rotary frequencies and attention factor from `config.json`'s rope parameters |
| [pages.py](code/pages.py) | `KVPool`: the device form, a paged pool in torch ([below](#on-the-device)) |
| [device_attention.py](code/device_attention.py) | The attention slot's device form over the pool |

Device attention over paged KV can also come from a provider
([Kernels' libraries](../../../../kernels/README.md#libraries)): FlashInfer's paged and ragged attention
(BF16 and FP8 KV, prefill, decode and a verify's query block) or FlashAttention's paged KV,
which read a block table like the pool's. Store device pages in a format the provider reads
(BF16 or FP8 rows per head).

**Sliding-window layers as rings (`kv:sliding`).** A pool made with `pool(..., sliding=W, span=S, positions=N)` holds
the layers whose tokens see only their last W positions (local layers): each sequence keeps a ring of
ceil((W + S) / page) + 1 pages (S: the most positions a row writes in one step, a prompt chunk or a verify window; N:
the most positions a sequence reaches, the page tables' width), logical page p in ring page p mod ring. The tables the
attention reads are the ring expanded to the logical pages, so a position older than the window aliases a newer
one's page, which the window never shows; softmax-attention reads such a pool with its window (`attention:sliding`)
and every kernel path of paged-kv-attention reads it unchanged. A sequence holds at most its ring whatever its length,
`bind` takes no page past it for a captured round, a fork shares the ring until either writes, `keep` frees no page
of a sequence that has wrapped, and the plugin's facts count these pools as `kv_sliding_bytes_per_sequence`. Tested
in softmax-attention's `test_sliding_pool.py`: prompt chunks, decode steps, verify windows rolled back, forks and 120
captured rounds over a ring read bit for bit as a full pool under the same window (BF16 on crit's kernels; FP32 on
the torch fallback within 1e-5), and a captured round replays over it.

**Lending the pool's unused pages.** The device pool ([pages.py](code/pages.py) `KVPool`) keeps its page tensors in
one allocation, with `spare` bytes after them for a borrower, and takes the lowest free page first, so every page at
or past its `floor` holds no KV. A device expert cache borrows that memory: `lendable()` gives the allocation, each
page tensor's place in it, the spare bytes and the floor, and `on_reserve(callback)` calls the borrower before a page
at or past the bound it returned is handed out (it reclaims the slots under the new floor first) and after the floor
falls (pages freed). Readers see the same tensors as without a borrower.

**An elastic pool.** A pool sized for the longest context holds device memory its sequences rarely use. With the
plugin's `elastic` option the page tensors lie in one reserved address range
(`crit_plugins.memory`'s elastic ranges) of which only the pages sequences hold are backed, 256 pages (4,096 tokens) of
every layer at a time, and each tensor's dump page: every address is the one a pool of one allocation has, so
readers, writers and captured rounds are unchanged. Taking a page past the backed ones first asks the device's
givers for the bytes (the device memory's `need()`: an expert cache gives up its coldest slots, and keeps free what
the step in flight still takes). A device that has no memory left refuses the page as a pool whose pages are all in
use does, after asking what keeps sequences it can do without to close one: the driver is never asked for pages
it would put in system memory. When a closed
sequence lowers the floor by more than 256 pages their backing goes back and the givers grow into it. Such a pool
lends nothing: what it does not hold is simply the device's. Where the driver has no virtual memory API, or with
spare bytes asked for, the pool is one allocation as before. `backed_bytes()` says what the pages take now.

**Pages in host memory.** With `host_pages_mib`, an elastic pool that finds the device full and nothing left to
close backs its next pages from host memory instead of refusing the request, up to that many MiB a pool and
never into the host's last 2 GiB. The driver pins that memory and maps it into the pool's address range, so every
reader, writer and captured round uses the addresses it always used and each row has the bits it has on the
device: a context the device does not hold is served, slower, where it was refused. What it costs is the link:
the device reads those pages over it every time attention reads them (`host_bytes` says how many bytes lie
there, the device memory's report lists them as the range's `host`). Pages in host memory are given back as soon
as the sequences above them close, so the next ones try the device first. Checked on an RTX 5090 under Linux (WSL)
and Windows: the mapping works under both drivers. Measured there (512 MiB, PCIe 5.0): kernels read and write
pages in host memory at 28 GB/s where pages on the device go at 690, and a query's scores over FP16 keys in host
memory read them at 26 GB/s. A pool with 4 GiB of its pages in host memory therefore adds about 150 ms to every
pass that reads them all.

## What the engine supplies

Keep one state image per request or owned branch. Apply position transforms and
key/value construction before staging, and pass the model's own attention scale.
The engine rolls back provisional KV on failure or cancellation and commits only after
all layers succeed. Map acceptance to the session's committed cursor; a produced token need
not have its own KV row yet, so keep the engine's decode convention.

Prefix tokens describe state **already materialized** in KV. A full prompt hit
supplies no next-token logits or other model state: recompute the last position,
look up an earlier prefix, or retain those results under a compatible identity.
The identity must name weights, layer, adapter, positional conventions and KV
representation (including the codec for encoded pages). A prompt cache verifies hits by
tokens, never KV values, so an incomplete identity silently reuses wrong state.

On a GPU, a speculative round can keep the KV frontier on the device: verify
writes K+1 rows at `frontier..frontier+K`, acceptance advances the frontier by
`n_accepted` without a host read, and the next round overwrites rejected rows.
See [device-speculative-round](../../speculation/device-speculative-round/README.md).

## On the device

The plugin's device form is a paged pool in torch ([pages.py](code/pages.py)):
every layer's keys and values in pages of `page` positions, each sequence's
page table, copy-on-write forks, and page types BF16, FP16, FP32, FP8 E4M3 with
the layer's key and value scales (`fp8`), FP8 E4M3 with an FP32 scale per
position and KV head (`fp8:row`: the row's absolute maximum over 448, the
quotient rounded to the nearest E4M3 value) and INT8 with an FP32 scale per
position and KV head (`int8`: over 127, rounded to the nearest code, ties to
even). `fp8_rows` and `int8_rows` are the encoding rules; crit's kernel
`paged-kv-attention` (built for the engine's device, with the engine's tuned knobs
or its defaults) writes a step's keys and values in one launch with the same
codes and scales, and torch writes them where it does not build.

Captured decode rounds bind a sequence to a round slot (`rounds:kv_cache`):
`bind(slot, sequence, capacity)` takes its pages up to `capacity` positions
(prompt, output limit and the widest verify window) and puts its page table at
a fixed device address (`round_tables`), copying a page a fork still shares;
`reserve(RoundBatch)` then computes every token's slot on the device from those
tables and the batch's positions, so a captured round never allocates or waits
for the host; a round wider than the pool's `round_rows` is refused. A token
outside every bound capacity (an unbound slot, a position past the capacity)
and a write's slot below 0 go to the dump page, one page past the sequences'
that no table holds for a sequence and no borrower is lent; every unused entry
of a round's or a step's page table points there. `start_round(view)` notes
each slot's length as the round finds it, and `commit_round(view)` sets it to
the round's frontier where the row emitted and keeps the noted one where it
emitted nothing (a held row, or a `keep()` between rounds), on the device; with
a tree verifier's accepted nodes (`view.tree_path`) it first moves their keys
and values to consecutive positions, once a round. `unbind(slot)` reads the
length back (one wait) and frees the pages past it; `move(source, target)`
keeps the decoding rows packed at the front.

Captured prompt chunks (`prompts:kv_cache`, crit_plugins.prompt_graphs):
`stage_prompt(batch)`, outside the graph before each replay, takes the
chunk's pages up to its end and puts its real tokens' KV slots and its page
table in fixed buffers (`prompt_slots`, `prompt_table`), the padding's slots at
the dump page; `reserve(batch)` inside the graph returns those slots, and
`commit_prompt(batch)` sets the length after. A chunk of padding alone (a
capture's warm-up) takes no page. The attention plugins read the prompt table
on the device: crit's prompt reader at a position read on the device
(one graph for every frontier), else the bucket as window rows through the
split-KV reader or its torch fallback; no FlashInfer plan runs in a capture.

[device_attention.py](code/device_attention.py) is the attention slot's device
form over this pool, which softmax-attention and partitioned-attention share:
crit's paged-kv-attention kernels, built for the engine's device with tuned or default knobs
(a split-KV reader of decode and verify windows over every page type, whose partitions walk each row's own length read on the
device, with tree masks and block-sparse lists, tiles every GPU multiplies
with mma.sync and one partition count per pool, so a column's bits are the same
alone, in a window, in a step and in a round; and a prompt reader over every
page type, eager prompt rows one row a launch), then
torch: rows gathered and multiplied by SDPA, or a masked FP32 softmax over the
page tables that captures in a CUDA graph (in a capture it reads every position
the tables hold, so its cost follows the pool's capacity; in an eager step only
the positions its rows reach). The pool's `window` option (16 tokens) is the one
knob of what is a window row: a longer row is a prompt row. No library kernel
reads the pages by default: the plugins' option `flashinfer` (default `off`)
set to `auto` puts FlashInfer's paged prefill on eager prompt rows of 16-bit and
FP8 pages (and with softmax-attention's `decode: flashinfer` its decode on a
step's window rows), a yardstick to compare crit's kernels with in an engine.

The attention plugins' option `reader` picks which of the kernel's structures read
the pages: `split` (the default, the split-KV reader above and crit's earlier prompt
reader) or `flash`, the kernel's flash structures: a row's positions in contiguous
splits of one chunk per pool (`flash_chunk()`, the engine's tuning or 1,024
positions), a window's columns times a KV head's query heads packed into
tensor-core tiles of up to 128 rows, the splits merged (or, for `prompt_reader:
decode`'s prompt pieces, folded in the program) in split order. A column's bits are
then the same alone, in a verify window, in a step and in a round (one reduction,
another than `split`'s), and the facts say so (`reader`). Crit's prompt reader
becomes the flash prompt chunk, which splits a chunk's positions by its length: a
prompt token's bits follow the chunk it is read in, so the facts carry
`prompt_rows: not row-invariant` (as FlashInfer's prefill, within the readers'
rounding of decode rows); `reader: split` keeps the earlier reader. Block lists stay
on the split launch, and the options `partitions` and `piece` do not apply.
`reader: cuda` reads the same launches through the kernel's CUDA flash tile
(`cuda-window`, `cuda-fused`, `cuda-prefill`: the queries in shared memory, 16 rows
a warp, each warp's online softmax in registers, one chunk and block per pool from
`cuda_chunk()`): a column's bits are again the same alone, in a window, in a step and
in a round (its own reduction), its prompt chunk not row-invariant as the flash
one's; int8_qk prompt chunks stay on the Triton flash prompt chunk.

The plugin's `facts()` report `kv_bytes_per_token`: the device bytes one token's
slot takes in every page tensor of the pools the plugin made (keys, values and an
INT8 or row-scaled pool's scales, every layer), summed over the pools. The engine's
facts carry it to crit serve's `engine_facts` and to Eval, which sizes the KV cache
and prices the KV term of its floors by it.

## Worked example

Run from `code/`. The provisional query is allowed to see both staged rows.

```python
from implementation import KVState

state = KVState(capacity=3, key_width=2, value_width=1)
state.stage((1.0, 0.0), (2.0,))
state.stage((0.0, 1.0), (10.0,))
assert state.attend((0.0, 0.0), include_pending=True) == (6.0,)
state.commit(1)
assert state.attend((0.0, 0.0)) == (2.0,)
assert state.pending == []
state.stage((1.0, 1.0), (99.0,))
state.rollback()
assert len(state.committed) == 1
assert state.attend((0.0, 0.0)) == (2.0,)
```

## Adapting it

Preserve causal indexing, the accepted prefix and the separation of logical
visibility from physical lifetime: a rewind and a completion fence are different
facts. Paged or compressed storage can replace lists and fused attention can
replace the formula, with representation, masks, scaling and arithmetic explicit
and freshly checked. Tie graph bindings to the storage owner and physical address;
a page id alone is insufficient.
[Draft verification](../../speculation/mtp-decode/README.md) supplies an accepted count and
[checkpoint storage](../../memory/checkpoint-storage/README.md) can retain a committed image
once its reads settle.

## Costs and validation

Host work is `O(N * (K + V))` for `N` visible rows.

Host fixtures (`tests/test_techniques.py`) cover the Python transaction, and the
plugin's tests the device pool and its attention. None qualify GPU fences,
complete-model equivalence or throughput. The paged design follows
[PagedAttention](https://arxiv.org/abs/2309.06180).
