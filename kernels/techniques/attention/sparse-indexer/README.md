# sparse-indexer

The selection step of a sparse attention over compressed keys, from [crit](https://github.com/crit-labs/crit)'s
attention-state technique (operation `sparse-index`): every query token scores each compressed key it may see, the
sum over its index heads of the head's weight times the ReLU of the head's query-key product, and keeps the positions
of its k best keys, which the attention then reads instead of every position.

```python
s = scores(q, keys, weights, ends)                 # FP32 [tokens][keys], -inf at or past a token's end
picked = topk(s, ends, k)                          # int32 [tokens][k]: ascending positions, then -1
picked = select(q, keys, weights, ends, k)         # both, tokens in groups whose scores fit a workspace
s = scores(q, store, weights, ends, slots=slots)   # the keys in place: key j is store[slots[j]]
```

**Why it exists.** At long contexts the selection is most of a sparse layer's work: a token's score over N keys is
H x D x N products (64 heads of 128 over 50,000 keys at a 200,000-token context: 410 million multiply-adds a token),
and a prompt chunk scores every token. Done as torch ops it writes every head's products to memory before the ReLU
and the head sum, and the selection sorts. Here a tile of a few tokens times their heads multiplies a block of keys on
the tensor cores and reduces the heads in registers, so only one FP32 score a token and key leaves the tile, and a
radix select finds the k best without sorting.

**Keys in place.** A pool keeps a sequence's keys in pages, wherever its pages are, so a caller that wants them
side by side gathers them: one copy of every key a sequence has, for every layer and round, and at a 200,000-token
context that copy is 50,000 keys a row (12 MB in BF16, 24 MB once made FP32) written to score them once. With
`slots` the kernel reads each key from the pool's store instead: `keys` is the store `[rows][dim]`, `slots` the
sequence's slots `[n]` (int32 or int64), or `[groups][n]` for a round whose rows are sequences (the tokens in equal
runs, a run a group), and key j of a token is `store[slots[...][j]]`. A program loads its block of keys through the
slots, so nothing is gathered, and the scores have the bits the gathered keys give: the same products in the same
order. E4M3 stores take `key_scales` over the store's rows.

## Applicability

| | |
| --- | --- |
| Queries | BF16 or FP16 [tokens][heads][dim], each head row contiguous; heads a power of two up to 128, dim 64, 128 or 256 |
| Slots | optional: int32 or int64 [n], or [groups][n] with tokens % groups == 0: the store's row of each key, so `keys` is a pool's store read in place |
| Keys | [n][dim] in the queries' dtype, or E4M3 codes (`float8_e4m3fn`) with FP32 key_scales [n] (a key's value its code times its scale, converted exactly to the queries' type; the scale after the product) |
| Weights | FP32 [tokens][heads], each head's weight (a projection of the token's hidden state, with any scaling folded in); `scale` (default dim ** -0.5) multiplies every head's ReLU |
| Ends | int32 [tokens]: the keys token t sees are 0 .. ends[t] - 1 (a causal bound: for keys compressing 4 positions each, (position + 1) // 4) |
| Scores | sum over heads of weight x scale x max(0, q . key) in FP32: the products' FP32 sums on the tensor cores, ReLU, the head factor and the head sum in FP32; -inf at or past the end |
| Picks | each token's k largest scores below its end (all of them where it has k or fewer), ties to the lower position, a NaN below every number (-inf included); positions ascending, then -1 |
| Determinism | a token's scores depend on its query, weights and the keys: never on the tokens beside it, nor on a knob (the head sum is an explicit tree, heads 2 i and 2 i + 1 added, then the pairs' sums, so the tile's shape and warps change no bit); its picks on its scores alone. The same bits on a repeat |
| Capture | no host synchronization and no allocation in `scores` and `topk` given `out` (and `select` given `out` and `workspace`): a CUDA graph captures them |
| GPUs | NVIDIA sm_80 and newer (Triton's tensor-core dots); the hardware E4M3 conversion from sm_89 |

## Structures and knobs

| Structure | Key | Rows | Knobs |
| --- | --- | --- | --- |
| `scores` | H, D, KIND (0 16-bit keys, 1 E4M3) | tokens | `TM` (tokens a program, TM x H tile rows up to 128, no more than the call's tokens), `BN` (keys a program), `WARPS`, `STAGES`: speed only |
| `topk` | K | tokens | `BLOCK` (scores a step of each pass), `WARPS`, `SLICE` (positions a program selects among, 0 the rule), `CUDA` (1: calls of few tokens in the CUDA kernel) |

The radix select reads a token's valid scores five times: four passes of 8-bit digit histograms of their
order-preserving 32-bit keys (the highest digit first, among the keys matching the digits found), which give the k-th
largest key and how many of its ties to keep, then one pass writing the positions of every key above it and of the
first ties, a running count giving each its slot.

**The head sum.** Until 2026-11-15 the heads were summed by `tl.sum` over the tile's head axis, whose tree followed
the tile's register layout: TM, BN and the warps changed a token's bits, so they were order knobs, one value a shape
for every token count, and a one-token decode call multiplied a tile padded with a second, absent token (half its
tensor-core work). The sum is now written out as a tree, heads 2 i and 2 i + 1 added, then the pairs' sums, the same
order whatever the tile: the knobs are speed knobs, a sweep picks them per token count, and a call takes no more
tokens a program than it has. A token's scores differ from the earlier kernel's in their last bits (another order of
the same FP32 terms). RTX 5090, 64 heads of 128 over 50,000 keys, one call in a captured graph: one token 13.4 us
before and 10.2 now; 2,048 tokens 9.7 ms with the default tile either way (9.2 with TM 1 and BN 128).

**Few tokens.** A decode call's selection is a handful of programs, and the Triton program reads its scores from
global memory in each of the five passes: one token's 50,000 scores took 26.5 us, twice their scoring. Calls of fewer
tokens than half the device's multiprocessors select in a CUDA kernel instead (`cuda_select.cu`, knob `CUDA`), with
the same slices: a CTA copies its slice to shared memory once (one bulk copy where the device has them), turns the
scores into their order-preserving keys in place and runs the five passes there. A digit's histogram is 256
shared-memory counters with no branch on the data (a key that does not count adds to a spare bin); every warp then
reads the bins itself, a lane eight of them, and finds the digit from a suffix sum over the lanes, so a pass costs one
barrier; the write takes each thread's run of the slice as bit masks and stores only the set bits. A row longer than
two slices goes in two launches, the slices' k best (positions and scores) and then the k best of those candidates, as
the Triton programs slice; where the candidates would not fit a block's shared memory the slices are fewer and wider,
and where a slice would not, the call takes the Triton programs. The picks are the same by definition (no arithmetic
in a select), and the tests compare the two paths.

## Measured speed

RTX 5090, a call's GPU time in a captured graph, 64 index heads of 128, k 512, BF16 keys, against the same arithmetic
as torch ops (one batched matmul, ReLU, the weighted head sum, torch.topk, the positions sorted):

| Keys | Tokens | scores | topk | torch | |
| --- | --- | --- | --- | --- | --- |
| 8,192 | 1 | 8.2 us | 14.3 us | 77.8 us | 3.5x |
| 8,192 | 2,048 | 1.67 ms (165 TFLOPS) | 0.14 ms | 17.9 ms | 9.9x |
| 50,000 | 1 | 18.4 us | 30.7 us | 116 us | 2.4x |
| 50,000 | 4 | 32.7 us | 30.7 us | 222 us | 3.5x |
| 50,000 | 256 | 1.30 ms (161 TFLOPS) | 0.13 ms | 13.7 ms | 9.6x |
| 50,000 | 2,048 | 10.2 ms (165 TFLOPS) | 0.83 ms | (several seconds: its products do not fit the memory) | |

A few tokens over many keys select in slices (`SLICE`): 82 us a token at 50,000 keys in one program, 31 us in
slices.

The topk column above is the Triton programs' (`CUDA` 0). With the CUDA select, the default for few tokens, the same
calls back to back in a captured graph (2026-11-15, the same GPU; the Triton programs timed the same way beside them):

| Keys | Tokens | topk, Triton programs | topk, CUDA select |
| --- | --- | --- | --- |
| 8,192 | 1 | 13.3 us | 10.7 us (one launch) |
| 8,192 | 4 | 13.3 us | 11.0 us |
| 50,000 | 1 | 26.5 us | 14.5 us |
| 50,000 | 4 | 27.3 us | 14.7 us |
| 200,000 | 1 | 65.1 us | 19.5 us |

A launch of the select costs about 5 us before its passes (the launch, the slice's copy and its barriers), so two
launches bound a long row at about 10 us; slices of 2,048 or 8,192 positions are slower than the rule's 4,096 (more
candidates for the second launch, or more scores a multiprocessor copies: one takes small copies at about 8 GB/s).

## Tests

`tests/test_sparse_indexer.py`: scores against attention-state's float64 reference (`sparse_index.py`) at several head
counts and widths over 16-bit and E4M3 keys, within FP32 accumulation; picks with many ties, fewer valid keys than k,
empty rows and NaN against the reference's; a token's bits the same alone and beside others; `select` in groups the
same as in one call; the CUDA select's picks against the Triton programs' (one launch, slices, fewer and wider
slices, the fallback) with ties, NaNs, -inf and rows cut short; CUDA graph capture; refusals. `bench.py` is the sweep's bench (the model's `index_n_heads`,
`index_head_dim`, `index_topk`).
