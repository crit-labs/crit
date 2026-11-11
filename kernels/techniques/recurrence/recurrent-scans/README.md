CUDA kernels for the recurrent layers of hybrid language models, crit's kernel `recurrent-scans` (operation
`gated-delta`, recurrent-state-recovery's):

- **Gated delta** (Qwen3-Next / Qwen3.5 gated DeltaNet): `gated_delta_prefill`, a chunked two-pass prefill (64-token
  chunks) in FP32 on CUDA cores (register-blocked for 128-dim keys) or, with `tensor_cores=True`, the chunked WY form on
  tensor cores (FP16 operands, FP32 accumulation and state), and `gated_delta_scan`, a token-by-token scan for decode
  and speculative verify.
- **Mamba-2** state-space recurrence: `ssd_prefill` and `ssd_scan`.
- **Short causal depthwise convolution** with its carried window: `causal_conv1d`.
- **Pools of sequences** for serving many requests and speculative rounds: `pool_delta` runs the decode and verify
  rows of many sequences in one launch, each row from its own sequence's state in a pool
  `[layers][sequences][v_heads][dk][dv]` (FP32), keys and queries L2-normalized inside the scan, BF16, FP16 or FP32
  inputs, grouped key heads, dk up to 256 and any dv; a row either stashes each token's normalized key, decay and delta
  (a deferred commit: the sequence's next row replays the accepted prefix first), writes the state before and after
  every token (per-column checkpoints), keeps its start state (a recompute commit, `pool_delta_replay`), or keeps its
  final state (a prompt row). Rows come from a table the host writes or from a captured round's device tensors and a
  slot table, so a round is captured once. `pool_conv` / `conv_segments`: the causal convolution over pooled windows (2 to 8 taps, and 2 to 4
  taps at dilation 2 to 4, the dilation read from the window's width, `pool_conv_supported`),
  deferred the same way, and long rows split into 32-row segments with the same bits. `pool_conv_prompt`: one
  sequence's prompt chunk padded to a captured graph's bucket, its sequence and real length read on the device, the
  padding output as zeros and the window moved past the real inputs only (with `pool_delta`'s FINAL row written into
  the live state, a chunk replays for any sequence and length). The commit's bookkeeping (writing the accepted count,
  selecting a checkpoint) is the caller's.

The scans are column invariant (a token decoded alone equals the same token inside a multi-token verify, bit for bit)
and can write the state after every token (`checkpoints`), which speculative decoding needs to roll back rejected
drafts. The pool rows add row invariance (a row's bits do not depend on the rows beside it) and one update expression on
every path (verify, checkpoint, stash replay, recompute), so every commit strategy gives the same bits as decoding the
accepted tokens one by one. Per-token decays are FP64 `exp` rounded once to FP32, so a decay near 1 does not drift the state over a long
decode. The state and the arithmetic are FP32. The single-sequence functions take FP32 inputs and update the state in
place, with key dims 64 or 128 and value dims a multiple of 64 (Mamba-2: d_state 64 or 128, head_dim a multiple of 32);
the pool rows take FP32, BF16 or FP16 inputs, key dims 1 to 256 and any value dim.

GPUs: NVIDIA sm_80 and newer (kernel.json's capabilities: 8.0 to 12.1). FP32 SIMT, and for the tensor-core prefill
mma.sync m16n8k16 (FP16, FP32 accumulation), ldmatrix and cp.async, all from sm_80 on; no architecture-specific
instruction. The prefills take up to 98,816 bytes of shared memory a block (the tensor-core prefill's two stages at
dk 128; the FP32 passes up to 96 KB), within the 99 KB a block every one of those GPUs offers.

Measured on one RTX 5090: the gated-delta prefill of a 2,048-token chunk at Qwen3.8-27B's shape (16 key and 48 value
heads, dk = dv = 128) takes 1.38 ms per layer (1.32 ms measured again 2026-11-07, 5.2 ms per 8,192 tokens). At the same
shape the pool rows take 5.4 us per layer for one sequence's decode token, 12.0 us for an 8-column verify row and
56.4 us for 8 sequences x 8 columns (deferred commit; a 48-layer round of convolution, scan and commit captured as one
CUDA graph: 0.51, 1.06 and 4.28 ms); per-column checkpoints cost up to 2.5x that, the recompute commit up to 1.7x. The
tests compare every function with a float64 recurrence, and every commit strategy of the pool rows with decoding the
accepted tokens one by one, bit for bit.

## The prefill on tensor cores (`delta-prefill-tc`)

`gated_delta_prefill(..., tensor_cores=True)` computes the chunked WY (UT-transform) form of the gated delta rule, the
algorithm of flash-linear-attention's `chunk_gated_delta_rule` (MIT) and FlashInfer's chunked gated-delta prefill
(Apache-2.0), written here from the algorithm (no code adapted). Per value head and 64-token chunk with start state S,
cumulative log decays G, keys K, scaled queries Q and values V:

```text
A = strictly lower(beta_i e^{G_i - G_j} k_i . k_j),  T = (I + A)^-1,  Tb = T diag(beta)
delta = Tb (V - diag(e^G) K S)          O = diag(e^G) Q S + P delta,  P = lower(e^{G_i - G_j} q_i . k_j)
S' = e^{G_last} S + (diag(e^{G_last - G}) K)^T delta
```

- **Pass 1** (block per key head and chunk, every chunk at once): K and Q to FP16 (copies pass 2 streams), `K K^T` and
  `Q K^T` by mma.sync once for all the key head's value heads, then per value head A, `T` a column a thread by forward
  substitution in FP32 (the core's `unit_lower_inverse_column`), `Tb`, `P` and G.
- **Pass 2** (block per value head and `WARPS` x 16 value columns, the chunks in order): each warp keeps its slice of
  the state transposed, `S^T` [16 values][dk], in FP32 MMA accumulators (the core's `TransposedState`). The same
  registers are the FP16 A operand of `S^T K^T` and `S^T Q^T` and take the update `S^T += delta'^T K`; the deltas'
  accumulators become the A operand of `delta^T P^T` and of the update the same way (flash attention's reuse of its
  score tile, `f16_a_of_accumulators`), so no data moves between lanes. The chunk's K, Q, Tb and P stream through
  `STAGES` cp.async shared-memory stages (16-byte chunks XOR-swizzled by row, read by ldmatrix, the update's K by
  ldmatrix.trans); the values come from global memory into registers while `S^T K^T` runs. The causal products skip the
  tiles above the diagonal.
- **Overlap** (`OVERLAP` 1; the default is 0, two launches): one launch holds both passes. Pass 2's blocks (96 at the
  yardstick's shape) leave most multiprocessors free; the launch's other blocks are pass-1 workers taking the next
  (chunk, key head) item from a counter, in chunk order, and pass 2 waits on a per-chunk flag (an acquire load) before
  reading a chunk, so pass 1, which mostly moves memory, can run beside pass 2, which mostly multiplies. The host takes
  this launch only where every block of it fits on the device at once (occupancy times multiprocessors above pass 2's
  blocks), so a waiting pass-2 block never holds the place a worker needs; elsewhere, and for a single chunk, the two
  passes run as two launches.

Built from the kernel core (`kernels/core`): `mma.cuh`'s FP16 m16n8k16, `load.cuh`'s cp.async, ldmatrix and swizzle,
`scan_chunk.cuh`'s inverse, accumulator reuse and transposed state tile.

Numerics: products of FP16 operands (K, Q, the state, Tb, P, the deltas: each rounded once a chunk) summed in FP32; the
inverse, the gates and the state in FP32. FP16 has TF32's 11-bit significand at twice TF32's tensor-core rate (an RTX
5090's dense figures: 209.5 against 104.8 TFLOPS), and BF16's 8 bits failed the bench's tolerance in a simulation of the
same algorithm (the final state past 1e-3 + 1e-3|x| of the FP64 recurrence at 2,048 tokens). Values and states must lie
within FP16's range (|x| < 65504; keys and queries are normalized). On the bench's prompts the outputs are within 8.3e-5
and the final state within 8.5e-4 of the FP64 recurrence (1e-3 + 1e-3|x| allowed). Another summation order and rounding
than `delta-prefill-rb`, so `row_invariant` is false like the FP32 prefill's: a prompt row's bits follow the prompt's
length. dk 64 or 128, dv a multiple of 16; scratch `delta_tc_scratch_bytes` (FP16 K, Q, Tb, P and FP32 G per chunk).

## Structures and knobs

kernel.json lists each way the kernel computes as a structure; four of them have knobs a sweep tunes per layer shape
(key and value heads, dk, dv) and row count. The default knobs are the kernel as it was before it had knobs: an untuned
build compiles to the same instructions (checked against the earlier build's SASS); `pool-delta-pipe`, new, defaults to
its fastest untuned configuration.

| Structure | Knobs (default) | What they change |
| --- | --- | --- |
| `delta-prefill-rb` (`gated_delta_prefill`, dk 128) | `COLS` 16 / 32 / 64 (64), `THREADS` 128 to 1024 (512), `MINB` 0 / 2 (0) | pass 2's block: the value columns it owns (so the grid, value heads x dv / COLS), its threads and so each thread's tile rows, the minimum blocks a multiprocessor in `__launch_bounds__`. The tuning is looked up by the prompt's tokens |
| `delta-prefill-tc` (`gated_delta_prefill(tensor_cores=True)`, dk 64 / 128) | `WARPS` 1 / 2 / 4 / 8 (4), `STAGES` 1 / 2 (2), `OVERLAP` 0 / 1 (0) | pass 2's block: its warps (16 value columns each, so the grid, value heads x dv / (16 WARPS)), its shared-memory stages (2: the next chunk's tiles load during the current one), and whether both passes share one launch. Looked up by the prompt's tokens |
| `pool-delta` (`pool_delta`) | `SLOTS` 1 / 2 / 4 (1), `PREFETCH` 0 / 1 (1), `MINB` 0 to 6 (0), `LANE_COLS` 4 / 8 / 16 (4) | the 32-column slots a block covers, whether the next token's inputs load while the current one updates the state, the minimum blocks a multiprocessor, the value columns a lane holds (a slot then takes 32 / LANE_COLS warps, a lane reading 16 to 64 contiguous bytes a key row). The tuning is looked up by the launch's rows (sequences) |
| `pool-delta-pipe` (`pool_delta(pipelined=True)`) | `LANE_COLS` 4 / 8 / 16 (8), `ITEMS` 1 / 2 / 4 / 8 (1), `STAGES` 1 / 2 (1), `MINB` 0 / 2 / 4 / 6 (0) | the value columns a lane holds (a block then has 32 / LANE_COLS warps), the slots a block runs one after another, the slots' state tiles in flight ahead of the one running, the minimum blocks a multiprocessor. Looked up by the launch's rows |

No knob changes a value. `pool-delta`'s pending counts and stash parities belong to each 32 value columns whatever the
block, so a row run in one configuration continues one run in another, and a row's bits are the same in every
configuration and beside any rows, and `pool-delta-pipe`'s are `pool-delta`'s, in any mix from call to call
(`tests/test_knobs.py`; `tests/test_recurrent_pools.py` runs every pool test on both structures): `LANE_COLS`' wider lanes sum the same pairs (the core's
`column_sum`, the 32-lane butterfly's bits for 8 to 32 values). `delta-prefill-rb` sums the same terms in the same order
in every configuration; a tile also adds the exact zeros above its triangular matrices up to its last row, which can
only change the sign of a zero sum. `delta-prefill-tc` gives the same bits in every configuration (a warp's instructions
do not depend on its block, the stages and the overlap move the same bytes at other times). None has an order knob.

The other structures (the prefill at dk 64, the commit replay, the pool convolutions, the single-sequence scans and
convolution, Mamba-2) are listed without knobs: their blocks are either the pending counts' slots, which a tuning must
not change between calls, or a path no plugin times.

## Yardstick

The tuned structures' `yardstick` (kernel.json) is FlashInfer 0.7.3, which `crit kernels sweep` times on the same inputs
beside the tuned kernel (`library_us`, `vs_library` = library time / tuned time); flash-linear-attention (`fla`) is not
installed in crit-venv, so it is not timed. `delta-prefill-rb` and `delta-prefill-tc`: `chunk_gated_delta_rule` with
log-domain gates and the keys and queries normalized beforehand as here, on q, k and v rounded to BF16 once (it takes
16-bit inputs only: half the input bytes) and the start state transposed once to its V-major layout. `pool-delta`:
`gated_delta_rule_decode_pretranspose` for one token a row (it also writes the updated state back, which the STASH rows
defer) or `gated_delta_rule_mtp` without the state update for verify windows, on the same FP32 states transposed once,
the gates given as its pre-activation inputs (`A_log` and `dt_bias` zero, `a` and `b` the inverse softplus and logit of
the same decay and beta, made once), BF16 out.

Measured on an RTX 5090 (short sweeps, `--shapes`; `delta-prefill-tc` and `pool-delta` swept again 2026-11-10 with
every knob above), Qwen3.8-27B's gated-delta layer (16 key / 48 value heads, dk = dv = 128), tuned / FlashInfer in us
(the timer's ticks are 2.05 us on this machine, so the shortest rows move in whole ticks):

| Structure | Workload | Rows: tuned / library (vs_library) |
| --- | --- | --- |
| `delta-prefill-rb` | a prompt from a random state | 512: 305 / 153 (0.50); 2,048: 1,296 / 296 (0.23); 8,192: 4,979 / 847 (0.17) |
| `delta-prefill-tc` | the same prompts, FP32 inputs (FlashInfer: BF16) | 512: 65 / 153 (2.34); 2,048: 221 / 292 (1.32); 8,192: 829 / 841 (1.01) |
| `pool-delta` | decode rows (one token each) from FP32 states | 1: 8.2 / 6.1 (0.75); 2: 9.9 / 6.1 to 8.2 (0.62 to 0.83); 4: 14.3 / 10.2 (0.71); 8: 22.5 / 16.1 (0.72); 16: 38.9 / 30.7 (0.79); 32: 69.4 / 123 (1.77) |
| `pool-delta-pipe` | the same decode rows (sweep of 2026-11-10, beside `pool-delta`'s) | 1: 8.2 / 6.1 (0.75); 2: 9.6 / 6.1 (0.64); 4: 14.3 / 10.2 (0.72); 8: 22.5 / 16.4 (0.73); 16: 36.8 / 30.7 (0.83); 32: 67.6 / 123 (1.82) |

The FP32 CUDA-core prefill is 2x to 6x slower than FlashInfer's tensor-core chunked prefill on BF16 inputs; the
tensor-core prefill (its defaults, `WARPS` 4, `STAGES` 2, `OVERLAP` 0, which the sweep of its 16 configurations picked
at every length) is 1.01x to 2.3x faster than FlashInfer while reading twice its input bytes, and 4.8x to 6x faster than
`delta-prefill-rb`. Its time per 8,192 tokens splits into pass 1, 222 us (bound by memory: 300 MB of FP32 keys and
queries read and FP16 tiles written), and pass 2, 648 us (warm L2). `OVERLAP` 1 runs both in one launch but was slower
at every length here (pass 1 on the 74 multiprocessors pass 2 leaves free is slower than pass 2). The decode rows stay
0.62 to 0.79 of FlashInfer from 1 to 16 rows and 1.8x faster at 32: tuning picks `MINB` 4 or 6 and `LANE_COLS` 8 or 16
at some row counts (28.0 / 48.4 us untuned at 8 / 16 rows), but the tuned times equal the earlier sweep's. FlashInfer's
times are those of repeated runs that day; the 2026-11-10 sweep's own pass timed it at 8.2 / 10.2 / 15.3 / 22.6 / 53.3 /
236 us (vs_library 1.00 to 3.4), slower than in every other run, so the table does not use them. The gap is each block's
chain of dependent loads and barriers at low occupancy (120 registers a thread untuned), not the read pattern: lanes
along the value columns (each warp 128 contiguous bytes of a key row, which a read probe ran at 1.4x to 1.7x this
kernel's pattern's bandwidth) and loading the first token's inputs with the state were built, tested and timed, and came
out no faster (24.6 us both at 8 rows), so they are not kept.

## The pool rows staged (`pool-delta-pipe`)

`pool_delta(..., pipelined=True)` runs `pool-delta`'s rows with its bits on another schedule. A block of 32 / `LANE_COLS`
warps runs `ITEMS` slots (value head, 32 value columns, row) one after another. Each slot's state tile, 128 key rows x
32 columns at dk 128, goes into shared memory by cp.async `STAGES` slots ahead of the one the block runs, a key row's
128 contiguous bytes by 8 neighbouring threads (whole lines), the 16-byte chunks swizzled by the lane that reads them
(the core's `scan_stage.cuh`: `stage_columns`, `read_columns`). The same copy group stages the row's first 8 tokens as
stored (q and k of the key head, v of the slot's columns, beta). The next slot's pending count, parity and decay inputs
load into registers while the current one runs. `pool-delta` read the pending count, synchronized, then loaded the token
and the decay, then took the exponential; here no load of a slot waits on another. The lane's tile then sits in the
registers `pool-delta`'s lane holds and goes through the same `token_step` and `tile_replay`. The mechanism follows
FlashInfer's gated-delta decode (Apache-2.0: cp.async-staged state tiles, gate inputs read before the barrier),
written here for crit's [dk][dv] layout and sums; no code adapted. Shared memory a stage: the tile plus 8 tokens
(21,024 bytes at dk 128, BF16).

Measured on one RTX 5090 at Qwen3.8-27B's layer (16 key / 48 value heads, dk = dv = 128), BF16 decode rows from FP32
states, in us:

- Swept (the table above): tuned, as fast as `pool-delta` tuned or faster at every row count (8.2 / 9.6 / 14.3 / 22.5 /
  36.8 / 67.6 against 8.2 / 9.9 / 14.3 / 23.9 / 38.9 / 69.6 the same day). Untuned (`LANE_COLS` 8): 8.2 / 9.6 / 14.3 /
  22.5 / 38.9 / 69.6 against `pool-delta`'s defaults 8.2 / 12.3 / 18.1 / 28.3 / 47.1 / 89.1.
- The sweep's FlashInfer is warm: the timer runs every call on the same state, and FlashInfer's decode writes its state
  back, and its state evidently survives the read that flushes the L2 (timed warm 30.7 us at 16 rows, cold 55.9), while `pool_delta`'s STASH rows only read it. On cold states
  (the sweep's timer rotating over enough copies that each call's state is out of the L2, as a model's 48 layers are),
  FlashInfer takes 6.1 / 8.2 / 14.3 / 20.5 / 55.9 us at 1 / 2 / 4 / 8 / 16 rows and `pool-delta-pipe` 8.2 / 10.2 /
  14.3 / 22.5 / 38.9 (vs_library 0.74 / 0.80 / 1.00 / 0.91 / 1.44; `pool-delta` with `MINB` 4, `LANE_COLS` 8: 8.2 / 10.2 / 16.4 / 26.3 / 41.5).
  Back to back (16 calls in one graph on cold states, per call): 5.1 / 7.6 / 13.3 / 22.1 / 37.9 against FlashInfer's
  5.1 / 8.1 / 16.1 / 32.9 / 66.5.
- Verify rows of 4 tokens (FlashInfer: `gated_delta_rule_mtp` without the state update, cold): 1 / 4 / 16 rows 10.2 /
  22.5 / 53.2 against 10.2 / 19.6 / 48.3 (0.87 to 1.0); `pool-delta` 14.3 / 26.6 to 36.8 / 73.7 to 88.6.
- Scanned prompt rows of 64 tokens: one sequence 84 (`pool-delta` 88); 4 sequences 178 untuned (`pool-delta` 135 to
  248 by configuration): a long row runs its tokens past the 8 staged as `pool-delta` does, a tuning picked on decode
  rows may suit it less.

What remains at 1 and 2 rows is one 2-us tick of the timer (8.2 against 6.1): 192 slots fill the device once, so each
call is one chain of dependent latencies (the row table, then the tile; the decay's FP64 exponential is taken while the
tile loads). Not built: programmatic dependent launch, which would overlap that chain with the previous kernel's tail
in a model's graph but not in the sweep's one-call timer.

## Build, test, sweep

```sh
crit kernels build recurrent-scans                       # this GPU, default knobs (cached in CRIT_KERNEL_CACHE)
crit kernels build recurrent-scans --arch 80 --arch 86 --arch 89 --arch 90a --arch 100a --arch 120a   # every GPU it claims
python -m pytest kernels/techniques/recurrence/recurrent-scans/tests -q
crit kernels sweep recurrent-scans --model CHECKPOINT --out TUNING_DIR       # or --shapes FILE
```

The sweep reads the gated-delta layer's shape from the checkpoint's config (`linear_num_key_heads`,
`linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`) and times `bench.py`'s benches: the prefill
on 512, 2,048 and 8,192 tokens of FP32 inputs (normalized keys and queries, as the plugins pass them) against an FP64
token-by-token recurrence (outputs and final state within 1e-3 + 1e-3|x|), and the pool rows on 1 to 32 sequences of
one BF16 token each (`"tokens"` in a shapes file for verify rows) against the FP64 recurrence within 1e-5 + 1e-4|x|, each
row's bits checked alone and beside the others. A sweep build compiles only those two paths, BF16 pool rows, and
returns cudaErrorNotSupported from the rest. A plugin builds the kernel with the engine's tuning
(`CRIT_KERNEL_TUNING=TUNING_DIR`) or with the default knobs.

```python
from crit_kernels.build import load
import torch

rs = load("recurrent-scans")   # built for this GPU on first use
dev = torch.device("cuda")
# A gated-delta layer of 16 key and 48 value heads (dk = dv = 128) over a 1,000-token prompt, then one decode step.
T, kh, vh, d = 1000, 16, 48, 128
q = torch.nn.functional.normalize(torch.randn(T + 1, kh, d, device=dev), dim=-1)
k = torch.nn.functional.normalize(torch.randn(T + 1, kh, d, device=dev), dim=-1)
v = torch.randn(T + 1, vh, d, device=dev)
log_decay = -torch.rand(T + 1, vh, device=dev) * 0.1   # log of the per-token decay, <= 0
beta = torch.rand(T + 1, vh, device=dev)
state = torch.zeros(vh, d, d, device=dev)               # updated in place
prompt_out = rs.gated_delta_prefill(q[:T], k[:T], v[:T], log_decay[:T], beta[:T], state, scale=d ** -0.5)
step_out = rs.gated_delta_scan(q[T:], k[T:], v[T:], log_decay[T:], beta[T:], state, scale=d ** -0.5)
```

## Functions

`causal_conv1d`, `gated_delta_prefill`, `gated_delta_scan`, `ssd_prefill`, `ssd_scan`, `pool_delta`,
`pool_delta_replay`, `pool_conv`, `conv_segments`, `pool_conv_prompt`, and the pool layout helpers `pool_blocks`,
`pool_half_floats`, `pool_conv_blocks`, `pool_conv_supported` (row kinds `STASH`, `CHECKPOINTS`, `RECOMPUTE`, `FINAL`,
`FLUSH`; replay modes `REPLAY`, `FLUSH_PENDING`). Their arguments are in `torch-ext/recurrent_scans/__init__.py`.
