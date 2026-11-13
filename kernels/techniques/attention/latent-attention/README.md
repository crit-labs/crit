# latent-attention

Latent attention (multi-head latent attention in its absorbed form) in Triton: decode tokens and verify windows of
every query head over a paged cache of compressed rows, one row a position holding the latent (rank R) and then the
rotary key (width P), shared by all heads. The latent is each position's value too, so a piece of cached rows is read
once and serves the scores and the values. Operation `latent-attention` (attention-state's
[operations.json](../../../../inference/attention-state/operations.json), reference
[latent_attention.py](../../../../inference/attention-state/code/latent_attention.py)).

**Why it exists.** A model with latent attention keeps one compressed row a position instead of a key and a value a
KV head (576 values a position at rank 512, against 128 heads' keys and values), and its decode reads that row from
every query head. paged-kv-attention takes head widths up to 256 and separate key and value pages, so without this
kernel an engine expands the latents into per-head keys and values (reading and writing them a token) or writes its
own kernel.

## Applicability

| | |
| --- | --- |
| Cache | `[pages][page][R + P]` BF16 or FP16, the latent then the rotary key a row (`write(latent, rope, slots, cache)`), or E4M3 (float8_e4m3fn) with FP32 scales `[pages][page][2]`, one for each part of a position (`write(..., scales)`: max \|x\| / 448, the nearest codes); page tables `[sequences][width]` int32 |
| Queries | `[tokens][heads][R + P]` in the cache's dtype (BF16 or FP16 over an E4M3 cache): the query's latent part (its non-rotary part times the key up-projection) then its rotary part; out `[tokens][heads][R]` in the latent space (the value up-projection follows outside) |
| Geometry | R and P powers of two (16 to 1024, 16 to 256), any head count; decode rows and verify windows of up to 64 columns (causal inside the window, `attend`), and prompt chunks of any length (`prefill`), one launch for any lengths (positions read on the device) |
| Extras | `lse` (each token's natural log-sum-exp), `sinks` (each head's sink logit, one more position with no value, folded in last: the kernel core's `attention_softmax.sink`), `counts` (columns a row) |
| GPUs | NVIDIA sm_80 and newer (Triton's tensor-core dots) |
| Numerics | scores `scale (q_latent . latent + q_rope . rope)` in FP32 from the operands on the tensor cores, probabilities in the operands' dtype for the value product, the online softmax and the merge in FP32; within 8e-3 absolute of the float64 reference over the cached rows (BF16 probabilities), log-sum-exps within 1e-3 |
| Determinism | same bits on a repeat; a token's bits depend on its query, its positions, PIECE and the partition count only: the same alone, inside a verify window and beside other rows; the speed knobs keep every bit |

## How it computes

A program takes a tile of BLOCK_M query rows (a row's columns times its heads, row j: column j // H, head j % H) and
partition b of the row's positions: pieces b, b + PARTS, ... of PIECE positions, up to the tile's last column. Each
piece's rows are loaded once; the scores are two tensor-core dots (the latents, then the rotary keys into the same
accumulator), masked at each column's own position, and the latents again are the values of the next dot. The online
softmax and its row sums (a dot with ones: a fixed order whatever the tile) are the kernel core's `attention_softmax`.
VSPLIT programs share a tile, each with R / VSPLIT of the output columns (each computes the scores: fewer registers,
the same bits). The partitions' states fold in partition order in the merge, the sink last, and the output rounds
once.

**Prompt chunks** (`prefill(q, cache, table, position, page, rank=R)`, structure `prompt`): a chunk of one sequence
at positions `position ..` (an int or a one-element device tensor: one CUDA graph for every position), causal. The
chunk's tokens times their heads are packed into tiles of BLOCK_M rows (row j: token j // H, head j % H), so each piece
of cached rows is loaded once for BLOCK_M of them; the pieces go in order through one online softmax and the output is
written directly, with no partitions and no partials (a decode reader's partials over a 2,048-token chunk at 128 heads
would take gigabytes). A token's bits do not depend on the chunk's length; another reduction than `attend`'s.
Sinks, lse and the E4M3 cache as `attend`. An E4M3 cache's scales may also be one a position for the whole row
(`[pages][page][1]`: a KV pool's fp8:row pages of one head).

## Knobs (structure `window`, keyed by R, P, heads, page and the cache kind)

| Knob | Values | Effect |
| --- | --- | --- |
| `BLOCK_M` | 16, 32, 64 (64) | query rows a program, at most the window's columns times the heads rounded up to a power of two |
| `PIECE` | 16, 32, 64 (16) | positions a piece (order) |
| `PARTS` | 0, 1, 2, ..., 128 (0) | partitions (order); 0: about two programs a multiprocessor for one decode token, 2 x SMs over the heads' 16-row tiles, 1 to 128 |
| `WARPS` | 2, 4, 8 (4) | warps a program |
| `STAGES` | 1, 2, 3 (3) | Triton's pipeline stages over the pieces |
| `VSPLIT` | 1, 2, 4 (2) | programs sharing a tile's output columns |

A configuration whose tiles the GPU's shared memory cannot hold fails to launch and the sweep drops it (BLOCK_M 64
with VSPLIT 1 at rank 512 does on 100 KB GPUs).

## Measured

RTX 5090, rank 512, rotary width 64, page 64, BF16, one decode token a row (one column), default knobs, against
FlashInfer 0.7.3's `BatchMLAPagedAttentionWrapper` on the same cache split into its latent and rotary tensors (CUDA
graphs, best of five; the two agree to 1.2e-4):

| Heads | Rows x positions | latent-attention | FlashInfer |
| --- | --- | --- | --- |
| 16 | 1 x 4,096 | 25.1 us | 32.0 us |
| 16 | 1 x 32,768 | 65.2 us (579 GB/s) | 72.3 us |
| 16 | 4 x 8,192 | 62.4 us | 56.4 us |
| 16 | 8 x 32,768 | 371 us (813 GB/s) | 366 us |
| 16 | 1 x 8,192, 4 columns | 41.1 us | 48.1 us |
| 128 | 1 x 4,096 | 24.8 us | 44.9 us |
| 128 | 1 x 32,768 | 98.0 us | 112 us |
| 128 | 4 x 8,192 | 118 us | 104 us |
| 128 | 8 x 32,768 | 857 us | 678 us |
| 128 | 1 x 8,192, 4 columns | 116 us | 106 us |

An E4M3 cache holds twice the positions in the same memory. Its codes convert exactly to the queries' dtype for the
dots (each part's scores take their part's scale per position, the values the latent's, folded into the
probabilities), so its speed is the conversion's, not the bytes': with FP16 queries the same as the BF16 cache (16
heads, 8 x 32,768: 376 against 372 us; 128 heads, 1 x 32,768: 98 against 98 us), with BF16 queries about 0.85 times
it (the conversion to BF16 is not one instruction).

At 128 heads the long rows are compute bound (the scores and values are about 280 KFLOP a position): there the tile
of 64 heads reaches about 85 TFLOPS against FlashInfer's 107. `crit kernels sweep latent-attention` tunes the knobs
for a model's heads.

Prompt chunks (structure `prompt`, default knobs BLOCK_M 64, PIECE 16, WARPS 4, STAGES 2, VSPLIT 2) against
FlashInfer's MLA wrapper with causal prompt rows on the same cache (the two agree to 2e-3):

| Heads | Chunk x cached before | latent-attention | FlashInfer |
| --- | --- | --- | --- |
| 16 | 512 x 0 | 80.6 us | 62.6 us |
| 16 | 2,048 x 0 | 776 us (94 TFLOPS) | 656 us (111) |
| 16 | 1,024 x 7,168 | 3.20 ms | 2.78 ms |
| 128 | 512 x 0 | 414 us | 340 us |
| 128 | 2,048 x 0 | 5.47 ms (107 TFLOPS) | 4.78 ms (122) |
| 128 | 1,024 x 7,168 | 20.7 ms | 19.0 ms |

That is 0.78 to 0.92 times FlashInfer. Both run the absorbed form, about 1,090 multiply-adds a head and position;
engines that expand the latents into per-head keys and values for prompts (192-wide keys, 128-wide values) do about
a third of that arithmetic but expand and hold the expansion first.

## How it was checked

`tests/test_latent_attention.py` on an RTX 5090: four geometries (rank 512 with 16 and 128 heads, rank 256 with 8,
rank 128 with 20 heads and a rotary width of 32) over shuffled pages and rows of 1 to 517 positions, with and without
sinks, against the float64 reference, outputs and log-sum-exps; an E4M3 cache against the reference over its stored
values (codes times scales), its tokens alone and in a window bit for bit; every token alone, inside its verify window and
beside other rows bit for bit; the speed knobs bit for bit; one CUDA graph replayed at five lengths; the inputs it
refuses; prompt chunks at three geometries and three positions against the float64 reference, with and without
sinks, over an E4M3 cache, a token's bits the same at five chunk lengths, and one CUDA graph replayed at three device
positions. The kernel gates (nonfinite, repeat, edges) on both structures at 16 and 128 heads over both cache kinds.

## Yardstick

FlashInfer's `BatchMLAPagedAttentionWrapper` (bench.py's `yardstick`), on the cache's latent and rotary parts as
separate tensors.
