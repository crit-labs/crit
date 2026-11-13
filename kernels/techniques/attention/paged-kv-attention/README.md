# paged-kv-attention

Triton kernels for attention over paged KV caches, used by crit's attention-state KV pool and its attention plugins
(softmax-attention, partitioned-attention). Operations: `paged-attention` (attention-state) and
`partial-attention-merge` (partitioned-attention).

- **`write`**: a step's keys and values into their page slots in one launch, encoded for the page kind: BF16 or FP16,
  FP8 E4M3 with one multiplier per layer (`fp8`), FP8 E4M3 with an FP32 scale per position and KV head (`fp8:row`,
  scale = max |x| / 448, codes rounded to the nearest E4M3 value) or INT8 with an FP32 scale per position and KV head
  (`int8`, scale = max |x| / 127, codes rounded to nearest, ties to even). Slots below 0 are skipped. The codes and
  scales equal torch's encoding of the same rule bit for bit (IEEE divisions).
- **`attend`**: decode tokens and speculative-verify windows (up to 64 columns a row, causal inside the window) over the
  pages through per-row page tables, split over KV partitions: piece `i` (16 to 128 positions, set by the head width
  and the page kind) goes to partition `i mod P`, each partition walks only up to its row's last position (read on
  the device), and the partials fold in partition order (a running maximum, the sum and output rescaled to it). The
  work follows each row's length, not the cache's capacity, and one CUDA graph replays at every length. Tree masks
  (`parents`: a draft tree's columns see their ancestors and themselves), block-sparse lists (`blocks`, `block_size`:
  a token sees only its listed KV blocks), log-sum-exp outputs (`lse`), and partials packed by the rows' tokens
  (`offsets`), so a launch's workspace follows its tokens. `fused`: each program reads every partition of its columns
  in turn and folds them as the merge does, writing the output itself: the same bits as the split launch, no
  workspace, so a prompt chunk read as windows of 64 columns has the bits of its tokens decoded one by one (its value
  columns split over `vsplit` programs, each computing the scores: fewer registers, the same bits). `fresh`: each
  row's positions from a first one on read from 16-bit rows given beside the pages (a prompt chunk's own keys and
  values as the projections gave them), the earlier ones from the pages (whose bits stay as without them).
- **`prefill`**: a causal prompt chunk of one sequence over its pages (any page kind; `int8_qk` multiplies INT8 queries
  with INT8 keys on integer tensor cores); its position an int or a device tensor (one CUDA graph for every position);
  with `fresh` the chunk's own keys and values as the projections gave them (16-bit), the earlier positions from the
  pages at their kind's precision (over FP8 and INT8 pages the chunk then does not see its own quantization); a first
  fresh position (an int or a device tensor) reads the chunk's positions before it from the pages too. Another
  reduction than `attend`'s (FP32 row sums in one pass over 32- or 64-position blocks): its outputs and a decode's
  differ by the readers' rounding (up to 1.6e-2 in BF16 outputs over a few dozen positions, 2.4e-4 over 12K); a chunk
  that must have its tokens' decode bits reads through `attend(..., fused=True)` as windows of 64 columns.
- **`attend(..., flash=True)`, `prefill(..., flash=True)`**: the flash structures (`_flash.py`, its online softmax
  the kernel core's `crit_kcore.attention_softmax`), the mechanisms of FlashInfer's batch prefill over paged KV
  (Apache-2.0) in Triton. A program's query tile packs a row's columns times a KV head's query heads (tile row `j`:
  column `j // G`, head `j % G`; 16 to 128 rows, no row left empty between columns, each key block read once for
  every head that shares it, the tile's shape one per cache, `flash_tile()`); a row's positions split into contiguous
  ranges of `chunk` positions (`flash_chunk()`, one per cache) walked in blocks of `BLOCK_N` with the softmax scale
  folded with log2(e) into one multiply before exp2. Windows (`attend`): each split's state into
  the partials and a merge in split order, or `fused` (each program folds the splits in order itself: the same bits,
  no partials); trees, `lse`, `offsets`, `counts` and `fresh` rows as the split launch takes them, block lists not
  (they stay on the split launch). A row's FP32 sum is the tile layout's reduction, the same for every row of every
  tile of the cache's shape, and its products are mma.sync dots, so a column's bits depend on its own keys and query,
  the chunk, the block, the tile and the page kind: the same alone, in any window, packed, fused or split (another
  reduction than the split launch's, whose sums run on the tensor cores so that its tiles may differ).
  Window launches mask every block (an unmasked block may round otherwise: the scale's multiply and the maximum's
  subtraction contract into one FMA there). Prompt chunks (`prefill`): the chunk as one row of its tokens, its
  positions split on the device over `SPLITS` programs a tile (merged) or one (written directly), the blocks every
  row of a tile sees in full without the causal mask; a token's bits follow its chunk's length.
- **`attend(..., flash="cuda")`, `prefill(..., flash="cuda")`**: the CUDA flash tile (`cuda_window.cu`, kernel.json's
  `cuda` section: compiled by crit's build like a CUDA kernel's sources, for the build's GPU, with the build's
  `crit_tuning.h`, one library a variant (query dtype, page kind, head widths) on the variant's first call, called
  through ctypes on the caller's stream, so CUDA graphs capture it; the structures' knobs are read by the library from
  the tuning, the configurations it holds instantiated from `CRIT_CONFIGS_*`). The query tile (a row's columns times a
  KV head's query heads, as the flash tile packs them) stays in shared memory, swizzled, in row groups of 16 rows
  (`WARPS` of them): their fragments come from shared memory a k-step at a time (ldmatrix), a block of `BLOCK_N`
  positions' keys and values is staged through a cp.async ring the CTA's warps share (one stage: keys and values in
  buffers of their own, each loaded while the other is read), the scores and the output are FP32 mma.sync m16n8k16
  accumulators, and each row group runs its rows' online softmax in registers (the kernel core's
  `attention_softmax.cuh`): no state crosses row groups. A row group takes `VWARPS` warps: each computes some of a
  block's score tiles (n8 tiles of positions, shared through shared memory), runs the group's softmax on the same scores
  and holds a part of the value columns (`NV / VWARPS` n8 tiles of the output), so a window of a few query rows (a
  decode token: 12 rows of Qwen3.8-27B's heads) spreads its products and its value columns over four warps instead of
  one. 8-bit pages' codes stay in their stage (16-byte chunks XOR-swizzled) and each warp converts the ones its
  fragments take in registers (`CONVERT` 2, the rule: the keys' code pairs read from their rows, the values' by an
  ldmatrix `.trans` of 16-bit pairs and a quad shuffle, converted by the same function as the shared block), so no
  16-bit block, no pass converting it and no sync for it (`CONVERT` 1, and every launch with fresh rows: the codes
  converted once a block into a shared 16-bit buffer, as before); the operands, and so the bits, are the same. Prompt
  chunks (`cuda-prefill`) take it too: without fresh rows in registers; with them, in one range, in three passes,
  each continuing every thread's state the one before left at full precision (`workspace=`, or the package's own
  buffer): the blocks below the first fresh position's block in registers, that block through the 16-bit block, the
  blocks past it (all fresh rows) through the 16-bit pages' tile, the one range's blocks in its order (the same bits).
  On an RTX 5090, Qwen3.8-27B's heads (256 wide, 24 over 4), a 2,048-token chunk at 32,768 positions over FP8 pages
  went from 92 to 144 TFLOPS without fresh rows and from 84 to 132 with them (FP8 with a scale per layer). With
  `splits=1` a chunk reads in one range (a token's bits follow its own positions only: `cuda_prompt()`), against the
  Triton prompt chunk 1.2-1.65x as fast (1.55x at 131,072 positions, 1.18x for a first chunk). A row's
  positions split into contiguous ranges of `CHUNK` positions (`cuda_chunk()`, one per cache), one program a split,
  each split's state written with its output rounded to BF16 and folded in split order by the merge: the staged merge
  (`cw_merge2`), one program a column's query head, its splits' outputs streamed through shared memory in batches of 32
  (16 where its programs outnumber the SMs), a factor warp folding each batch's maxima and sums a batch ahead of the
  threads folding the outputs, the steps of the merge before it (whose programs read a few partials at a time and
  bounded long rows: a 131K-position decode row took 50 us to merge). Fused launches run the same
  programs with no partials a split: each writes its state into a ring of two groups of slots its tile owns, and a
  group's states are folded in split order into the tile's running state by the next group's programs, each its
  share of the tile's rows (it waits for the group before its own to be complete, which also frees its slot, stores
  its state, folds its rows of that group, then arrives); the last arrival of the last group folds that group over
  every row into the outputs. The merge's steps and rounding a row, group by group: a fused column has the split
  launch's bits; its workspace follows the tiles (`FUSED_GROUP` splits a group at most, fewer past `FUSED_WORKSPACE`
  bytes: the group changes no bit), not the row's length; the programs take their work by ticket in split order
  (every tile's first split, the longest tiles first, then the next), so a program only waits for programs of the
  group before its own, which hold earlier tickets and run already. A fused launch whose partials would fit
  `FUSED_PARTIALS` bytes (workspace_floats of every split its page tables hold, no `offsets`, no caller workspace) runs
  as the window launch and the staged merge instead (the same bits; `FUSED_PARTIALS` 0 keeps every fused launch on its
  ring): its tile's folds run one after another, where the merge spreads over the GPU. Trees, `lse`, `offsets`, `counts`, `fresh` rows, every page kind and FP16 queries as the
  flash launch takes them; head widths 64, 128 and 256. A column's bits depend on its own query, keys and values, the
  page kind, `BLOCK_N` and `CHUNK` only: not on the rows beside it, the warps or the tile (the same alone, in any
  window, packed, fused or split). Prompt chunks (`prefill`): the chunk as one row, its positions split on the device
  over `SPLITS` programs a tile (merged) or one; a token's bits follow its chunk's length (int8_qk chunks stay on the
  Triton flash prompt chunk). A fused launch's sync words live in a module workspace of the device: fused launches of
  one device run on one stream at a time.

- **Sliding windows** (`attend(..., sliding=W)`, `prefill(..., sliding=W)`, every structure but block lists): a
  token at position t sees positions t - W + 1 .. t, as models with local layers define their window. The pieces,
  blocks and splits wholly below a tile's window are skipped (a position no row sees changes no bit), so a decode row
  over a long cache reads only its window's blocks: on an RTX 5090, 8 query and 2 KV heads of 128 over 60,000
  positions, a window of 128 takes 12 us against 49 (split launch), 10 against 38 (flash) and 10 against 31 (CUDA
  tile: its splits wholly below the window write nothing and its merge starts after them). A fused launch keeps its split launch's bits; a
  window at least the row's length keeps every bit except on the flash prompt chunk, which masks every block under a
  window (no unmasked fast path).
- **Sinks** (`attend(..., sinks=)`, `prefill(..., sinks=)`, every structure): each query head's sink logit (FP32
  `[heads]`, natural log, in the scaled scores' units; -inf for none) joins every row's softmax as one more position
  that carries no value, `out = sum p v / (sum p + e^sink)`, as attention with learned sinks computes it. It is folded
  into a row's final state once, after its partials (the kernel core's `sink` / `softmax_sink2`), so a fused launch
  keeps its split launch's bits; the log-sum-exp counts it; sinks of -inf keep every bit of a launch without them.

## Numerics

Codes convert exactly to BF16 (FP16 for FP16 pages); scores and the softmax in FP32 with BF16 probabilities on the
tensor cores. A window launch's query tile (`tile_for`) holds its columns of a KV head's query heads: 16 or 32 rows in
two warps, and past 32 rows 64 rows in four warps where the GPU multiplies that with mma.sync too (sm_80 to sm_89,
sm_120); on sm_90 (wgmma), sm_100 and sm_110 (tcgen05) such windows take 32-row two-warp tiles. So on each GPU every
tile is multiplied with mma.sync (the tests compile each GPU's own choice for sm_80, sm_90, sm_100, sm_110 and sm_120
and check), and the row sums run on the tensor cores in a fixed order too. A column's result depends only on its own
keys and query, the partition count, the piece size and the page kind: not on the rows beside it, not on how many
columns or rows its launch reads (a token decoded alone equals the same token inside a verify window, bit for bit). A
launch given another tile or warp count keeps that only where the GPU multiplies it with mma.sync as well. Errors
against float64 attention over the stored values: within 8e-3 absolute (BF16 queries and probabilities), log-sum-exps
within 1e-3. E4M3 codes convert with the hardware conversion from sm_89 and with integer arithmetic before (the same
bits).

## Structures and knobs

`kernel.json` lists thirteen structures; their knobs are read at each call from the build's tuning
(`_crit_tuning.knobs()`). Every default reproduces the kernel's hand-set rules, so an untuned build runs exactly what
it ran before; a knob of 0 means "the rule".

| Structure | Key | Rows | Knobs |
| --- | --- | --- | --- |
| `write` | D, DV, HKV, KIND | tokens | `WARPS` (0: one up to 128 values a head, two past) |
| `split` | D, DV, HKV, KIND | decode rows | `PIECE`, `PARTS` (order knobs) |
| `window` | D, DV, HQ, HKV, PAGE, KIND | columns | `WARPS` (0: two, four over a 64-row tile), `STAGES` (0: three, four) |
| `merge` | D, DV, HQ, HKV, PAGE, KIND | columns | `WARPS` (4), `CHUNK` (16 partitions a turn) |
| `fused` | D, DV, HQ, HKV, PAGE, KIND | columns | `WARPS` (4), `STAGES` (3), `VSPLIT` (0: two past 128 value columns) |
| `sparse` | D, DV, HQ, HKV, PAGE, KIND | tokens | `WARPS` (2), `STAGES` (3) |
| `prefill` | D, HQ, HKV, PAGE, KIND, INT8 | chunk tokens | `BLOCK_M` (64), `BLOCK_N` (0: 64 up to 128-wide heads, 32 past), `WARPS` (4): order knobs; `STAGES` (0: three with int8_qk, two) |
| `flash-window` | D, DV, HQ, HKV, PAGE, KIND | columns | `CHUNK` (0: 1,024), `BLOCK_N` (0: 32), `ROWS` (0: 32), `WARPS` (0: four): order knobs; `STAGES` (0: one past 128-wide heads, two up to them) |
| `flash-fused` | D, DV, HQ, HKV, PAGE, KIND | columns | `STAGES` as `flash-window`'s, `VSPLIT` (0: two from 128 value columns, else one); the tile, block and chunk `flash-window`'s |
| `flash-prefill` | D, HQ, HKV, PAGE, KIND, INT8 | chunk tokens | `BLOCK_M`, `BLOCK_N` (0: 32), `WARPS` (0: four), `STAGES` (0: one past 128-wide heads, two up to them), `SPLITS` (0: about two programs an SM); `row_invariant: false` |
| `cuda-window` | D, DV, HQ, HKV, PAGE, KIND | columns | `CHUNK` (0: the GPU's rule, below), `BLOCK_N` (0: 16 over 16-bit pages, 32 over 8-bit pages): order knobs; `WARPS` (row groups of 16 rows; 0: the rule's tile shape, below), `STAGES` (0: one over 16-bit pages and 8-bit pages read in registers, keys and values in buffers of their own; two over 8-bit pages), `VWARPS` (warps a row group; 0: the rule's), `CONVERT` (8-bit pages; 0: in registers, 1: a shared 16-bit block), `RAMP` (0: none; an order knob, below) |
| `cuda-fused` | D, DV, HQ, HKV, PAGE, KIND | columns | `WARPS`, `STAGES`, `VWARPS`, `CONVERT` as `cuda-window`'s, `VSPLIT` (0: one); the block and chunk `cuda-window`'s |
| `cuda-prefill` | D, HQ, HKV, PAGE, KIND, INT8 | chunk tokens | `BLOCK_N` (0: 32), `WARPS` (0: four), `STAGES` (0: two), `SPLITS` (0: about two programs an SM); `row_invariant: false` |

KIND is the page kind's number (0 16-bit, 1 `fp8`, 2 `fp8:row`, 3 `int8`), INT8 whether the chunk takes `int8_qk`.

- `split` decides a column's bits (piece size and partition count): `partitions(device, kv_heads, kind, head_dim,
  value_dim)` and `piece_size(head_dim, value_dim, kind, kv_heads)` return the tuning's for the cache's geometry
  (without the geometry, the rule's), and `attend` takes them when the caller gives none. A cache keeps one of each for
  every launch over it (decode, verify windows, fused prompt rows, block lists), so a tuned engine's columns still
  agree in a step, in a round, alone and in a window. A caller's own `partitions` or `piece` wins.
- `window`, `merge`, `fused` and `sparse` change speed only: warps and stages of mma.sync tiles, the merge's warps and
  partials loaded a turn (folded one by one in partition order), the fused launch's value split. A warps knob never
  makes a tile of 64 rows or more four warps or more on a GPU whose tensor cores take that as warpgroup MMA (sm_90,
  sm_100, sm_110): there the rule's warps stay, so every tile a tuning picks is mma.sync too (the tests compile the
  knobs' choices for sm_80, sm_90, sm_100 and sm_120 and check).
- `prefill`'s blocks and warps are order knobs: the online softmax's steps (`BLOCK_N`) set its summation order, and
  on GPUs with warpgroup MMA the query block and warps decide the instruction. One value of each per shape, for every
  chunk length.
- `flash-window`'s `CHUNK`, `BLOCK_N`, `ROWS` and `WARPS` decide a flash column's bits: `flash_chunk(...)`,
  `flash_block(...)` and `flash_tile(...)` return the tuning's for the cache's geometry (head widths, heads, page,
  kind), and every flash launch over the cache (decode, verify windows, fused prompt rows) takes them: one tile shape
  per cache, so a row's FP32 sum (the tile layout's reduction) is the same in every tile, and the mma.sync products
  do not depend on the rows beside them (on GPUs with warpgroup MMA every flash launch takes one 64-row four-warp
  tile). `STAGES` and `VSPLIT` change speed only. A flash launch's workspace holds `flash_parts(table positions,
  chunk)` splits a partial row.
- `flash-prefill` gives up row invariance for speed (its splits follow the chunk's length): its knobs are all speed
  knobs; the earlier `prefill` stays selectable (`flash=False`, the default).
- `cuda-window`'s untuned `CHUNK` comes from the GPU: a decode row of 90,112 positions spread over two programs an
  SM, the KV heads counted by their square root, `90,112 x sqrt(KV heads) / (2 x SMs)` to the nearest 128 positions,
  128 to 2,048 (on an RTX 5090: 384 positions for 2 KV heads, 768 for 8). Small splits fill the GPU on short rows,
  large ones keep long rows' merges short (at 131K positions a split of 256 is 1.3x slower than 1,024); more KV
  heads already give more programs. Fixed per cache (the GPU and the cache's KV heads), never by the batch.
- `cuda-window`'s `CHUNK` and `BLOCK_N` decide a cuda column's bits: `cuda_chunk(...)` and `cuda_block(...)` return
  the tuning's for the cache's geometry, and every cuda launch over the cache (decode, verify windows, fused prompt
  rows) takes them. The warps, stages and value split change speed only (each warp owns 16 query rows and their
  softmax; a warp whose rows end before a block skips its math, a block a row does not see changes none of its
  bits), and a launch keeps fewer stages, then fewer warps, where the GPU's shared memory does not hold them.
  `cuda-prefill` gives up row invariance as `flash-prefill` does; `flash=False` stays the default.
- `cuda-window`'s `RAMP` (an order knob, 0 by default: the splits before it) splits a row's first positions finer: R
  bands of `CUDA_RAMP_SPLITS` (16) splits, band b's of `CHUNK / 2^(R - b)` positions, then splits of `CHUNK`
  (`cuda_ramp()`, `cuda_splits()`, `cuda_parts()` the splits a row of the tables' positions takes; a ramp that would
  not split the chunk into whole blocks is lowered). The splits are a function of the position only, so a column's
  bits stay its own (alone, in any window, fused or split); a short row gets several times the programs (Qwen3.8-27B
  at 4K under RAMP 2: 30 splits a KV head, not 11), a long one about as many (131K: 362, not 342).
- The untuned tile shape (`WARPS` and `VWARPS` 0) follows the call's query rows a tile (columns times the query heads
  a KV head, `crit_cw_plan`): up to 16 rows one row group of four warps, up to 32 two of two warps, more four row groups
  of one warp (a 64-row tile); a tuning's `WARPS` without `VWARPS` takes one warp a row group. The row groups' warps,
  `CONVERT` and the stages change no bit (each output's accumulator is its own chain of mma.sync steps whichever warp
  holds it; every warp of a group runs its softmax on the same scores; the codes convert to the same operands).
- A caller's `tile`, `num_warps`, `num_stages` and `vsplit` still override the tuning (measurements).

## Build, test, sweep

```sh
crit kernels build paged-kv-attention                  # stages the package with its default knobs (no compile step)
python -m pytest kernels/techniques/attention/paged-kv-attention/tests -q
crit kernels sweep paged-kv-attention --model CHECKPOINT --out TUNING_DIR
crit kernels sweep paged-kv-attention --shapes shapes.json --structure window --out TUNING_DIR
```

The tests load the kernel with `crit_kernels.build.load("paged-kv-attention")` (and `load(..., knobs=...)` for the
knob tests) and check against torch's encoding and float64 attention: page writes, windows of every page kind at
real shapes, row and column invariance bit for bit, every tile's lowering, trees, block lists, log-sum-exps, CUDA
graph replay at other lengths, work that follows a row's length, prompt chunks, fresh rows, and that speed knobs keep
every bit while the split knobs reach every launch over a cache. `tests/test_flash.py` checks the flash structures
the same way: windows of every page kind against float64, a column's bits alone, in windows of 2 to 64 columns,
packed, fused or split (log-sum-exps too), trees, fresh rows, a graph replayed at other lengths over poisoned
positions past the row, prompt chunks split or not (a device position, fresh rows, int8_qk), the tile's lowering
and warp layout (split and fused alike) on sm_80, sm_89, sm_120 and one tile on sm_90, sm_100, sm_110, and the
knobs. `tests/test_cuda.py` checks the CUDA flash tile the same way (windows of every page kind and FP16 against
float64 at five head shapes and two page sizes, a column's bits alone, in windows of 2 to 64 columns, packed, fused or
split and in tiles of 2, 4 or 8 warps, trees, fresh rows, a graph replayed over poisoned positions, prompt chunks split
or not, at a device position, with fresh rows, and its knobs); the kernel core's `attention_tests.cu` its softmax.

`bench.py` holds one bench a structure: the model's attention shape (head widths, query and KV heads, 16-position
pages) for BF16, `fp8:row` and INT8 pages (other kinds through `--shapes`, e.g. `{"window": [{"D": 128, "DV": 128,
"HQ": 56, "HKV": 8, "PAGE": 16, "KIND": 1, "kind": "fp8"}]}`); a shape may also set the context `N`, the batch rows `R`
and a chunk's position `P`. Each bench checks a configuration against float64 attention over the stored values (the
write bench: codes and scales equal to the encoding rule) and row 0 against row 0 alone.

## GPUs and shapes

Head dims 16..256 (powers of two; keys and values may differ), any page size, grouped query heads (query heads a
multiple of the KV heads), any number of rows. GPUs: CUDA GPUs Triton supports (sm_80 and newer); tested on an RTX 5090
(sm_120).

## Measured

On one RTX 5090 (CUDA graph replays, the GPU alone, default knobs), 24 query / 4 KV heads x 256, FP8 pages with a scale
per position and head, 64-position pages, one row: 16.4 us at 4K positions and about 350 us at 200K for a decode token
(about 1,200 GB/s), 344 us for a 5-column verify window, 512 us for 8 columns; eight rows at 200K, 2.53 ms. FlashInfer
0.7.3 over FP8 pages with a layer scale on the same shapes: 283 us decode, 388 us for a 4-column window (its batch
prefill), 2.03 ms for eight decode rows.

24 query / 2 KV heads x 256, BF16 pages, one row at 200K: 290 us decode, 300 us for 5 columns, 519 us for 8
(FlashInfer: 276 us decode, 294 for 4 columns, 347 for 8). 56 query / 8 KV heads x 128, INT8 pages with a scale per
position and head: 15.3 us at 4K, 349 us at 200K, 522 us for 5 columns.

On sm_90, sm_100 and sm_110 a window of more than 32 query rows keeps the cost of 32-row tiles: it is read by several
of them, each reading the row's pages again (about the tile count times a tile's time: on sm_120 the 5-column BF16
window above took 670 us that way). The reason: their tensor cores take four warps over 64 rows as warpgroup MMA
(sm_90's wgmma, sm_100's and sm_110's tcgen05), another summation order than the mma.sync of the decode tiles, so a
64-row tile there would give a window's column other bits than the same column decoded alone; a two-warp 64-row tile
stays mma.sync but runs about 2x slower, and two 32-row dots in four warps spill. Wide windows at short contexts over
many rows are bound by the partials (FP32, value width per query head, column and partition).

## Yardstick

Each structure's `yardstick` (kernel.json) is FlashInfer 0.7.3's call on the same pages, page tables and queries,
which `crit kernels sweep` times beside the tuned kernel (`library_us`, `vs_library` = library time / tuned time, above
1 where the template is faster): `append_paged_kv_cache` into the same slots for `write` (16-bit pages only: no library
encodes FP8 or INT8 pages, so those kinds report no library time); `BatchDecodeWithPagedKVCacheWrapper` (tensor-core
kernels) for `split`, and for `sparse` over page lists holding exactly each token's listed blocks;
`BatchPrefillWithPagedKVCacheWrapper`, causal, for `window`, `merge`, `fused` (the merge is timed inside the whole
attend call it closes, FlashInfer's own split and merge inside its call) and `prefill` (BF16 queries: FlashInfer has no
INT8 query path). FlashInfer reads 16-bit pages and FP8 pages with one scale a layer; FP8 and INT8 pages with a scale
per position and head are widened to BF16 once outside the timed call (the same values, twice the bytes). Plans are made
once outside the timed call; the library runs eagerly behind the sweep's GPU spin, the template as graph replays.

Measured on an RTX 5090 (one short sweep, `--shapes`, the GPU alone), Qwen3.8-27B's attention (24 query / 2 KV heads x
256, BF16 pages of 16) unless named, tuned / FlashInfer in us:

| Structure | Workload | Rows: tuned / library (vs_library) |
| --- | --- | --- |
| `write` | tokens into 8,192 shuffled slots | 1: 4.1 / 4.1 (1.0); 256: 4.1 / 4.1 (1.0); 4,096: 10.2 / 10.2 (1.0) |
| `split` | decode rows at 16K positions | 1: 34.8 / 34.8 (1.0); 8: 186 / 172 (0.92) |
| `split` | the same, Nanbeige4.3-33B's 56 / 8 heads x 128 over INT8 pages (library on BF16) | 1: 34.8 / 51.2 (1.47); 8: 209 / 331 (1.59) |
| `window` | 4 rows of N columns at 4K positions | 1: 32.7 / 32.7 (1.0); 4: 43.0 / 32.7 (0.76); 16: 127 / 53.2 (0.42); 64: 580 / 178 (0.31) |
| `merge` | the same windows | 1: 36.8 / 32.7 (0.89); 8: 65.5 / 36.8 (0.56); 64: 580 / 178 (0.31) |
| `fused` | the same windows | 1: 215 / 32.7 (0.15); 16: 248 / 53.2 (0.21); 64: 440 / 178 (0.40) |
| `sparse` | Nanbeige4.3-33B's heads, INT8 pages (library on BF16), 16 blocks of 64 | 1: 16.4 / 14.3 (0.87); 8: 22.2 / 30.7 (1.39); 64: 75.8 / 81.9 (1.08) |
| `prefill` | one chunk after 1,024 positions | 16: 51.2 / 18.4 (0.36); 256: 113 / 67.6 (0.60); 2,048: 952 / 643 (0.68) |

The decode path matches FlashInfer on 16-bit pages and beats it over INT8 pages (it reads half the bytes); verify
windows past a few columns, the fused window launch and causal prompt chunks are 1.5x to 6.6x slower than FlashInfer's
batch prefill: the gap the templates have to close. The sweep's event timer has about 2 us steps at these times.

The flash structures, one sweep of them alone (`--structure flash-window --structure flash-fused --structure
flash-prefill`, the same workloads; flash-window's order knobs one group for every column count, here a chunk of 256
positions, 32-row four-warp tiles, blocks of 32):

| Structure | Workload | Rows: tuned / library (vs_library) |
| --- | --- | --- |
| `flash-window` | 4 rows of N columns at 4K positions | 1: 30.7 / 32.7 (1.07); 4: 32.7 / 32.7 (1.0); 16: 84.0 / 53.2 (0.63); 64: 297 / 178 (0.60) |
| `flash-fused` | the same windows | 1: 190 / 32.7 (0.17); 16: 195 / 53.2 (0.27); 64: 332 / 178 (0.54) |
| `flash-prefill` | one chunk after 1,024 positions | 16: 14.3 / 18.4 (1.29); 256: 88.1 / 67.6 (0.77); 2,048: 743 / 645 (0.87) |
| `flash-window` | Nanbeige4.3-33B's 56 / 8 heads x 128, INT8 pages (library on BF16), the same windows | 1: 36.9 / 92.1 (2.50); 4: 38.9 / 94.2 (2.42); 16: 104 / 104 (0.99); 64: 391 / 252 (0.65) |
| `flash-fused` | the same | 1: 156 / 53.2 (0.34); 16: 180 / 67.6 (0.38); 64: 389 / 253 (0.65) |
| `flash-prefill` | the same heads and pages, one chunk after 1,024 positions | 16: 12.3 / 18.6 (1.52); 256: 88.1 / 83.3 (0.95); 2,048: 768 / 717 (0.93) |

(The Nanbeige sweep's one order group: a chunk of 256 positions and blocks of 64 for every column count; with the
default chunk of 1,024 a 64-column window took 326 us, 0.77 of the library.)

What is left at 256-wide heads: the flash tiles hold the queries in registers (mma.sync reads both operands from
them), so a 32-row tile in four warps is the widest that does not spill: every 32 rows of a window read the row's keys
and values again (a 64-column window of 12 heads a KV head reads them 24 times, from L2), the warps split the value
columns, and each block's maximum and sum cross the four warps through shared memory. FlashInfer keeps the query
tile in shared memory and loads its fragments a k-step at a time, so 64 rows of 16 a warp fit; Triton gives no hold
on that (on sm_120: no warpgroup MMA from shared memory). Wide windows over a few thousand positions stay at about
0.6 of FlashInfer, the fused launch (no KV split by design: one program walks every split of its tile) at 0.17 to
0.54. A CUDA flash tile with the queries in shared memory (the kernel core's mma and cp.async blocks) is the next step
for 256-wide heads.

The CUDA flash tile, one sweep of its structures alone (`--structure cuda-window --structure cuda-fused --structure
cuda-prefill`, the same workloads, the library timed as graph replays; report in
`/llm/engines/crit/kt-1109/yard/paged-kv-attention-cuda/`), before the build integration and the fused redesign below.
Order knobs, one group for every column count: Qwen3.8-27B a chunk of 256 positions and blocks of 16, Nanbeige4.3-33B
the defaults then (512, 32):

| Structure | Workload | Rows: tuned / library (vs_library) |
| --- | --- | --- |
| `cuda-window` | 4 rows of N columns at 4K positions | 1: 32.8 / 30.7 (0.94); 4: 32.7 / 30.7 (0.94); 16: 75.8 / 51.2 (0.68); 64: 213 / 178 (0.84) |
| `cuda-prefill` | one chunk after 1,024 positions | 16: 16.4 / 16.4 (1.0); 256: 71.7 / 65.5 (0.91); 2,048: 686 / 647 (0.94) |
| `cuda-window` | Nanbeige4.3-33B's 56 / 8 heads x 128, INT8 pages (library on BF16), the same windows | 1: 45.1 / 51.2 (1.14); 4: 49.1 / 55.3 (1.13); 16: 91.9 / 65.5 (0.71); 64: 270 / 248 (0.92) |
| `cuda-prefill` | the same heads and pages, one chunk after 1,024 positions | 16: 16.4 / 16.4 (1.0); 256: 73.7 / 77.8 (1.06); 2,048: 850 / 709 (0.83) |

Against the Triton flash tile at 256-wide heads: 64-column windows 0.60 -> 0.84, 16 columns 0.63 -> 0.68, prompt
chunks of 256 tokens 0.77 -> 0.91.

The fused launch, redesigned (one program a split, the states folded by a group's last program, tickets in split
order) against its first form (one program walking every split of its tile, 0.16 to 0.74 of FlashInfer), and the
untuned rules (a split from the GPU, blocks and stages from the page kind), on one RTX 5090, one timing script (the
sweep's timer: L2 flushed by a write, the template and the library as graph replays), untuned build, untuned / library
in us:

| Structure | Workload | Columns: untuned / library (vs_library) |
| --- | --- | --- |
| `cuda-window` | Qwen3.8-27B (24 / 2 x 256, BF16 pages of 16), 4 rows of N columns at 4K | 1: 47.1 / 34.8 (0.74); 4: 49.1 / 34.5 (0.70); 16: 65.5 / 53.8 (0.82); 64: 211 / 180 (0.85) |
| `cuda-fused` | the same | 1: 51.2 / 34.8 (0.68); 4: 61.4 / 34.8 (0.57); 16: 81.9 / 53.2 (0.65); 64: 221 / 180 (0.81) |
| `cuda-window` | Nanbeige4.3-33B (56 / 8 x 128, INT8 pages, library on BF16), the same | 1: 61.4 / 53.2 (0.87); 4: 67.6 / 57.3 (0.85); 16: 96.3 / 67.6 (0.70); 64: 266 / 250 (0.94) |
| `cuda-fused` | the same | 1: 61.4 / 53.2 (0.87); 4: 69.6 / 57.3 (0.82); 16: 98.0 / 67.6 (0.69); 64: 270 / 250 (0.92) |
| `cuda-window` | Qwen3.8-27B, one row at 32K / 131K | 1: 69.7 / 61.4 (0.88), 262 / 187 (0.71); 16: 137 / 91.5 (0.67), 521 / 322 (0.62) |
| `cuda-fused` | the same | 1: 79.9 / 61.4 (0.77), 211 / 186 (0.88); 16: 182 / 92.2 (0.51), 516 / 322 (0.62) |
| `cuda-window` | Nanbeige4.3-33B, one row at 32K / 131K | 1: 107 / 96.2 (0.90), 336 / 344 (1.03); 16: 180 / 111 (0.61), 618 / 387 (0.63) |
| `cuda-fused` | the same | 1: 102 / 96.2 (0.94), 277 / 344 (1.24); 16: 184 / 111 (0.60), 569 / 389 (0.68) |

The untuned split before (512 positions, blocks of 32, two stages) against now, Qwen3.8-27B at 4K: 16 columns 92.2 ->
65.5 us, 64 columns 213 -> 211; at 131K one column a split of 256 took 285 us, 1,024 221 us, the rule's 384 262 us.

One sweep of `cuda-window` and `cuda-fused` on these two shapes (CHUNK 128 to 2,048 positions, 448 configurations in
19 builds, 176 s to build, 199 s to measure; report in `/llm/engines/crit/ktb/sweep/`), tuned / untuned in us:
Qwen3.8-27B picked a chunk of 384 (the rule's) with blocks of 16: windows 1: 36.9 / 47.1 (three stages), 4: 38.9 /
49.1, 16: 65.5 / 65.5, 64: 210 / 210; fused 1: 40.9 / 51.2, 16: 81.9, 64: 223. Nanbeige4.3-33B picked 512 positions,
blocks of 32, two stages: windows 1: 45.0 / 61.4, 4: 49.2 / 65.5, 16: 94.2, 64: 274 / 264; fused 61.4, 98.3, 270.
(The sweep's own library times for Qwen3.8-27B's windows came out higher than the script's, 57 to 192 us; the
script's are above.)

Before this round the window form measured 0.62 to 1.02 of FlashInfer on these shapes and the fused form 0.49 to 0.89;
16 columns stayed at 0.7 to 0.82 (192 or 112 query rows a KV head in 2 to 3 tiles that each read the row's pages
again) and the fused launch's group fold (one program after the last split) cost 5 to 12 us more than the merge.

Row groups of several warps, codes converted in registers, the staged merge and the fused launch's window form, on one
RTX 5090 shared with other builders, untuned build, the start of the round's build ("before") and this one timed by
one script (the sweep's timer: L2 flushed by a write, template and library as graph replays; each case the template
forms and the library in turn, three rounds, the lower of rounds; the library's spread between rounds 0 to 8%, 0.7% on
average), FlashInfer 0.7.3 the yardstick as above (BatchDecode with tensor cores for one column, BatchPrefill causal
for more; Nanbeige4.3-33B's INT8 pages widened to BF16 for it, Qwen3.8-27B's FP8 pages read with their layer scale),
24 query / 2 KV heads x 256 and 56 / 8 x 128, pages of 16, column 0 of row r at position N - 7r; vs FlashInfer
(above 1 crit faster):

| Shape | Positions, rows | Window: before -> now (vs FlashInfer), columns 1 / 4 / 16 | Fused: before -> now | FlashInfer us |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B BF16 | 4K, 1 | 0.42 -> 0.44 / 0.50 -> 0.53 / 0.58 -> 0.60 | 0.39 -> 0.44 / 0.37 -> 0.53 / 0.41 -> 0.60 | 16 / 20 / 24 |
| Qwen3.8-27B BF16 | 4K, 4 | 0.74 -> 0.74 / 0.71 -> 0.67 / 0.81 -> 0.81 | 0.69 -> 0.74 / 0.55 -> 0.67 / 0.65 -> 0.81 | 35 / 33 / 55 |
| Qwen3.8-27B BF16 | 32K, 1 | 0.87 -> 1.07 / 0.78 -> 0.94 / 0.67 -> 0.73 | 0.75 -> 1.07 / 0.43 -> 0.94 / 0.49 -> 0.73 | 61 / 59 / 92 |
| Qwen3.8-27B BF16 | 32K, 4 | 0.90 -> 0.99 / 0.84 -> 0.90 / 0.76 -> 0.76 | 0.90 -> 0.97 / 0.77 -> 0.89 / 0.78 -> 0.78 | 184 / 185 / 326 |
| Qwen3.8-27B BF16 | 128K, 1 | 0.71 -> 0.95 / 0.68 -> 0.81 / 0.62 -> 0.75 | 0.88 -> 0.95 / 0.45 -> 0.81 / 0.62 -> 0.62 | 187 / 186 / 326 |
| Qwen3.8-27B BF16 | 128K, 4 | 0.85 -> 0.95 / 0.77 -> 0.86 / 0.74 -> 0.75 | 0.98 -> 0.95 / 0.94 -> 0.95 / 0.81 -> 0.80 | 678 / 685 / 1208 |
| Qwen3.8-27B FP8 | 4K, 1 | 0.33 -> 0.67 / 0.42 -> 0.69 / 0.64 -> 0.94 | 0.32 -> 0.67 / 0.31 -> 0.69 / 0.46 -> 0.94 | 20 / 25 / 33 |
| Qwen3.8-27B FP8 | 4K, 4 | 0.52 -> 0.87 / 0.66 -> 0.94 / 0.67 -> 0.86 | 0.49 -> 0.89 / 0.53 -> 0.96 / 0.55 -> 0.87 | 27 / 35 / 66 |
| Qwen3.8-27B FP8 | 32K, 1 | 0.51 -> 1.24 / 0.69 -> 1.07 / 0.62 -> 0.87 | 0.54 -> 1.23 / 0.42 -> 1.07 / 0.54 -> 0.87 | 43 / 59 / 108 |
| Qwen3.8-27B FP8 | 32K, 4 | 0.45 -> 0.94 / 0.62 -> 0.95 / 0.59 -> 0.80 | 0.50 -> 0.94 / 0.62 -> 0.94 / 0.58 -> 0.78 | 104 / 146 / 358 |
| Qwen3.8-27B FP8 | 128K, 1 | 0.34 -> 0.85 / 0.43 -> 0.85 / 0.50 -> 0.79 | 0.46 -> 0.85 / 0.39 -> 0.85 / 0.56 -> 0.69 | 106 / 149 / 355 |
| Qwen3.8-27B FP8 | 128K, 4 | 0.42 -> 0.94 / 0.55 -> 0.85 / 0.56 -> 0.77 | 0.47 -> 0.93 / 0.62 -> 0.84 / 0.58 -> 0.79 | 357 / 489 / 1329 |
| Nanbeige4.3-33B INT8 | 4K, 1 | 0.40 -> 0.67 / 0.39 -> 0.61 / 0.50 -> 0.70 | 0.39 -> 0.66 / 0.37 -> 0.61 / 0.44 -> 0.70 | 23 / 22 / 29 |
| Nanbeige4.3-33B INT8 | 4K, 4 | 0.87 -> 1.24 / 0.85 -> 1.19 / 0.73 -> 0.75 | 0.87 -> 1.24 / 0.80 -> 1.21 / 0.70 -> 0.75 | 53 / 57 / 68 |
| Nanbeige4.3-33B INT8 | 32K, 1 | 0.87 -> 1.69 / 0.89 -> 1.48 / 0.61 -> 0.67 | 0.92 -> 1.68 / 0.92 -> 1.48 / 0.60 -> 0.67 | 96 / 100 / 112 |
| Nanbeige4.3-33B INT8 | 32K, 4 | 1.20 -> 1.59 / 1.17 -> 1.53 / 0.70 -> 0.72 | 1.27 -> 1.59 / 1.19 -> 1.53 / 0.71 -> 0.72 | 342 / 358 / 404 |
| Nanbeige4.3-33B INT8 | 128K, 1 | 1.01 -> 1.54 / 0.98 -> 1.49 / 0.63 -> 0.70 | 1.24 -> 1.54 / 1.19 -> 1.49 / 0.68 -> 0.70 | 344 / 360 / 394 |
| Nanbeige4.3-33B INT8 | 128K, 4 | 1.35 -> 1.71 / 1.26 -> 1.59 / 0.69 -> 0.71 | 1.40 -> 1.71 / 1.30 -> 1.59 / 0.70 -> 0.72 | 1319 / 1360 / 1513 |

Where the gains come from, each measured alone on the same script: 8-bit pages read in registers instead of a shared
16-bit block (FP8 pages at 131K positions, one row: 208 -> 125 us for a decode token, both with the staged merge; the
block took 33 KB of shared memory, one CTA an SM, and a pass and a sync a block); the staged merge (Qwen3.8-27B one row
at 131K: 242 -> 197 us, its 342 splits a column were folded by 6 programs reading 4 partials at a time); a decode
token's row group in four warps (scores and value columns shared: 12 query rows used one warp of four before) changed
little alone (38.9 -> 38.7 us at 4K, one row): the short rows' bound is the 22 programs (11 splits x 2 KV heads) each
walking 24 blocks.

One sweep of cuda-window on each model's shape (4 rows of N columns at 4K, the knobs of kernel.json: 1,536
configurations each, 484 and 682 s to build, 229 and 184 s to measure; reports in /llm/engines/crit/att4/sweep-q16 and
sweep-n8), tuned / untuned / library in us: Qwen3.8-27B BF16 kept the rule's split and block (one order group for every
column count) and picked three stages for 1 and 4 columns: 1: 34.8 / 47.1 / 34.8 (1.0), 4: 38.9 / 49.1 / 34.8 (0.90),
16: 67.6 / 67.6 / 55.3 (0.82), 64: 221 / 221 / 182 (0.82); Nanbeige4.3-33B INT8 picked a split of 384 (the rule's 768)
and two stages: 1: 32.7 / 43.0 / 53.2 (1.63), 4: 36.8 / 47.1 / 57.3 (1.56), 16: 91.5 / 90.1 / 114.7 (1.25), 64: 281 /
283 / 254 (0.91). (The sweep's own library time at Nanbeige's 16 columns, 114.7 us, is above the script's 67.6 in the
table; the script's are the yardstick above.)

A second round (the start's build ddd50147 and this one: every default path's bits equal, 930 A/B cases with every fused
launch on its ring too; the tests above): fused launches fold a group over the next group's programs instead of one
program after the last split, blocks every row of a row group sees whole skip the mask, the outputs' rescale is skipped
where no row's maximum moved, and the ldmatrix offsets are kept from block to block. The fused ring at long rows, one
row of 16 columns at 131K: 524 -> 422 us (the window form's 416); four rows of 4 columns: 717 us (0.95 of FlashInfer,
the window form 788). The rest changed within 3%.

The ramp (`RAMP`, not the default; the same timing script, Qwen3.8-27B's chunk the rule's 384, Nanbeige4.3-33B's
768), window form vs FlashInfer, columns 1 / 4 / 16 at 4K:

| Shape, rows | RAMP 0 (default) | RAMP 1 | RAMP 2 | RAMP 3 |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B BF16, 1 | 0.44 / 0.53 / 0.60 | 0.44 / 0.53 / 0.58 | 0.67 / 0.77 / 0.76 | 0.62 / 0.71 / 0.64 |
| Qwen3.8-27B BF16, 4 | 0.73 / 0.71 / 0.84 | 0.74 / 0.74 / 0.63 | 1.00 / 0.82 / 0.66 | 0.89 / 0.74 / 0.59 |
| Qwen3.8-27B FP8, 1 | 0.61 / 0.59 / 0.89 | 0.61 / 0.59 / 0.89 | 0.89 / 0.83 / 0.89 | 0.89 / 0.83 / 0.89 |
| Qwen3.8-27B FP8, 4 | 0.93 / 0.94 / 0.86 | 0.94 / 0.94 / 0.78 | 1.08 / 0.90 / 0.74 | 1.08 / 0.90 / 0.75 |
| Nanbeige4.3-33B INT8, 1 | 0.65 / 0.61 / 0.72 | 0.92 / 0.86 / 0.88 | 0.92 / 0.89 / 0.83 | 1.22 / 1.10 / 0.80 |
| Nanbeige4.3-33B INT8, 4 | 1.24 / 1.22 / 0.76 | 1.62 / 1.61 / 0.75 | 1.43 / 1.17 / 0.70 | 1.39 / 1.27 / 0.67 |

(FP8 pages' 32-position blocks hold Qwen3.8-27B's 384 to a ramp of 2.) RAMP 2 costs 5 to 8% at 32K (Qwen3.8-27B one row,
one column: 1.07 -> 1.00) and 0 to 3% at 131K (RAMP 3 up to 20%: Nanbeige4.3-33B one row, 4 columns at 32K), and 16
columns at 4K lose most (four rows: 0.84 -> 0.66): their programs each load their 64 queries again for a split of 96
positions. A split of 768 under a ramp of 3 (96-position splits first, 768 past 10,752 positions) keeps the short rows'
gain, thins the long rows' partials (Qwen3.8-27B 131K, four rows, 16 columns: 0.76 -> 0.82) and costs 32K rows (one row:
1.03 -> 0.73): one order group for every column count and length, so the sweep weighs them; a band of 32 splits measured
slower than 16 at nearly every point.

A third round. Split launches store their partials lane-major (a lane's pairs of a row contiguous, 16 bytes a store,
eight stores a row a thread where 32 were; both merges read that layout): the same bits (930 A/B cases against
ddd50147), the times within 2%. CHUNK takes 192, 288 and 320 too (order-knob values; the default is the rule's): a split
that fills the GPU for one launch shape, Qwen3.8-27B four rows of 16 columns at 4K in 13 splits of 320 (312 CTAs of 340
slots, against 264): 0.84 -> 0.90, but a cache keeps one split for every launch (a launch's programs are its splits x
tiles x KV heads x rows, and only the first is the cache's), so no split fills every launch.

The sweep (`crit_kernels.sweep` on cuda-window, one run a model's shape at the bench's 4 rows over 4K positions, each
under one lock hold of 62 to 93 s measuring after 370 to 502 s of builds; its configurations limited to CHUNK 0, 192,
256, 288, 320, 512 or 768, BLOCK_N 0 or 32, WARPS 0, every RAMP, STAGES, VWARPS and CONVERT: 896 a shape; reports in
/llm/engines/crit/att4/sweep3-q16, -q8, -n8) picked a split of 320 and no ramp for all three shapes, three stages for 1
and 4 columns over 16-bit pages, blocks of 32 over 8-bit ones. Its own vs_library at 1 / 4 / 16 / 64 columns:
Qwen3.8-27B BF16 1.75 / 1.71 / 1.31 / 0.91, FP8 1.49 / 1.44 / 1.25 / 1.02, Nanbeige4.3-33B INT8 3.00 / 2.94 / 1.37 /
0.91 (its library times are above the graph-replayed ones of the script, 57 us against 35 at Qwen3.8-27B's 4 columns:
the sweep times the library eagerly behind its spin). The same tunings on the script (window form, vs FlashInfer graph
replays), untuned -> swept:

| Shape, rows | 4K: 1 / 4 / 16 columns | 32K | 131K |
| --- | --- | --- | --- |
| Qwen3.8-27B BF16, 1 | 0.44 -> 0.57 / 0.53 -> 0.62 / 0.62 -> 0.69 | 1.05 -> 0.88 / 0.98 -> 0.82 / 0.75 -> 0.83 | 0.94 -> 0.90 / 0.83 -> 0.85 / 0.78 -> 0.76 |
| Qwen3.8-27B BF16, 4 | 0.74 -> 1.07 / 0.74 -> 1.00 / 0.83 -> 0.88 | 0.97 -> 0.95 / 0.90 -> 0.90 / 0.80 -> 0.76 | 0.96 -> 0.96 / 0.87 -> 0.85 / 0.78 -> 0.75 |
| Qwen3.8-27B FP8, 1 | 0.71 -> 0.77 / 0.67 -> 0.75 / 0.89 -> 1.00 | 1.18 -> 0.88 / 1.04 -> 0.94 / 0.85 -> 0.96 | 0.84 -> 0.81 / 0.82 -> 0.89 / 0.78 -> 0.85 |
| Qwen3.8-27B FP8, 4 | 0.90 -> 1.00 / 0.94 -> 1.02 / 0.84 -> 1.00 | 0.92 -> 0.88 / 0.92 -> 0.94 / 0.79 -> 0.84 | 0.93 -> 0.87 / 0.84 -> 0.82 / 0.75 -> 0.81 |
| Nanbeige4.3-33B INT8, 1 | 0.66 -> 1.00 / 0.58 -> 0.92 / 0.68 -> 0.93 | 1.68 -> 1.42 / 1.48 -> 1.36 / 0.67 -> 0.69 | 1.55 -> 1.47 / 1.50 -> 1.36 / 0.71 -> 0.66 |
| Nanbeige4.3-33B INT8, 4 | 1.24 -> 1.63 / 1.17 -> 1.65 / 0.77 -> 0.83 | 1.59 -> 1.59 / 1.55 -> 1.47 / 0.73 -> 0.68 | 1.72 -> 1.69 / 1.58 -> 1.49 / 0.72 -> 0.65 |

The sweep tunes at the bench's one length (a shape's key holds no length: shapes that differ only in N count as one), so
its split serves 4K rows and costs 32K ones (Qwen3.8-27B one row, one column: 1.05 -> 0.88): a tuning for long contexts
wants the sweep at a long N.

The tile's code against FlashInfer's (cuobjdump, the 64-row 16-bit tile's block loop): the same 64 HMMA and 48 LDSM, the
rescale's FMULs alike (160 against 134), and crit's with 61 ISETP against 23, 46 MOV against 21, 48 branches and 12
divergence regions against a few; its branches are the masked block's path (taken only on a row group's last blocks),
the page lookup's power-of-two test (uniform) and the loads past the split's end. Votes on the block's branches,
unmasked full blocks and the skipped rescale (round 2) left the times within 3%.

What is left (measured, not built):

- 16 columns (untuned 0.62 to 0.89 at 4K, 0.71 to 0.82 at 131K; the sweep's split 0.69 to 1.00 at 4K). FlashInfer's
  batch prefill on these shapes is the same tile (64 rows in four warps of 16, blocks of 16 positions, one stage with
  keys and values in buffers of their own, 254 registers, 49 KB of shared memory, two CTAs an SM), with the same
  tensor-core instructions (1.58M against 1.60M at 4K, four rows): at 4K it fills the GPU (336 CTAs against crit's 264:
  11 splits of 384 positions, fixed per cache, the rest of the 340 slots idle), at 131K its tensor pipe is busy 90% of
  the time against crit's 70% (32K: its stalls on fixed-latency dependencies 1,405 samples against 2,372, on LSU
  throttling 5 against 391), with 25% fewer instructions. Masks only where a block needs them, kept ldmatrix offsets
  (19M -> 14.5M instructions) and the skipped rescale changed the times by 0 to 3%. Two warps a row group (scores
  shared, value columns split) are slower at every point (four rows at 4K: 108 us against 66; with 128 registers forced
  for two CTAs an SM, spills). The cross-warp overlap of the next block's scores with this block's values was not built:
  the same tile without it reaches 90% on FlashInfer's side. Next: the partials' stores (64 32-bit stores a thread at
  the end of a split: LSU throttling) in 16-byte rows, and the loop's control flow (373 branches and 283 divergence
  regions in the 64-row tile's kernel).
- One row over short rows: the default keeps today's splits (0.44 to 0.69 at 4K, one row); a ramp is an order knob
  for a tuning to choose (above).
- The fused ring: its folds now run over the next group's programs; the last group's fold is one program after the
  last split (16 states over the tile's rows), and the ring keeps two groups of slots (a program waits for the group
  before its own before storing).
- The attention plugins read decode and verify windows through the window form (split workspace given), and prompt
  pieces (prompt_reader decode, 64 columns) through the fused launch; with a ramp their partials take `cuda_parts()`
  splits a row (device_attention sizes its launches by it). Where a step's window rows need more than one launch of
  the plugin's split workspace (about 7 tokens a launch at 262K positions and 128 MB), one fused launch would read
  them at once, but its ring workspace grows with the rows and may not grow inside a capture.
