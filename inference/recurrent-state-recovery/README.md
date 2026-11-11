# Recover accepted recurrent state without per-token matrix snapshots

Speculative verification of a gated-delta recurrent layer can keep the starting
matrix and step inputs, compute verification outputs without saving a matrix per
token, then reconstruct only the accepted state. This trades state storage for
repair work. The entry supplies a bounded single-head host POC, the CUDA device
commit of a captured round's accepted prefix, the recurrence slot's device form (a
pool of every sequence's state and convolution window with three commit
strategies), and a two-pass chunked prefill reference.

## When to use it

Use it when verification needs a large recurrent state but only a short accepted
prefix is committed. Measure total proposal, verification and recovery cost
before choosing it. The code has no draft head, tokenizer, sampling policy or
transformer; it verifies one selected rooted chain at a time and does not evaluate
a branching tree in parallel.

## How it works

For key width `K` and value width `V`, the matrix holds `K * V` elements in
`matrix[k * V + v]` order. Each step supplies query `q[K]`, key `k[K]`, value
`v[V]`, `log_decay <= 0` and `beta` in `[0, 1]`; the engine computes projections,
convolutions, gating and head mappings beforehand. With optional query/key
normalization `x / sqrt(sum(x*x) + epsilon)`:

```text
G      = exp(log_decay) * H
delta  = beta * (v - transpose(k) * G)
H_next = G + outer(k, delta)
output = query_scale * transpose(q) * H_next
```

The native POC uses FP64 (the Python formula FP64 with `math.fsum`). A device
adaptation must state where activations, beta, outputs and state are rounded and
reproduce its ordinary decode contract during verification and recovery; equality
in real arithmetic does not survive different intermediate casts.

**Output without a second pass.** `output = query_scale * (exp(log_decay) *
transpose(q) * H + (q . k) * delta)`. `transpose(q) * H` and `transpose(k) * H`
read the same matrix, and `q . k` is state independent, so each column needs one
state read plus one update write. Its rounding differs from the formula above, so
compare a kernel using it against its own decode contract.

**One update expression everywhere.** Decode, verify, prefill and deferred replay
must evaluate the step with the same operations in the same order.
`fmaf(S, decay, k * delta)` and `fmaf(k, delta, S * decay)` round differently,
and the drift only shows near ties, which a short exactness check can miss.
crit's kernel recurrent-scans routes every pool path (verify, checkpoint, stash replay,
recompute) through one `fmaf(k, delta, S * decay)`, and the device form's torch
formula through one expression of its own.

**Prepare and commit.** `Recovery::prepare(nodes, path)` owns a copy of the
initial matrix, validated inputs and the physical path, and evaluates outputs with
only a running and a replacement matrix. `commit(id, accepted_count,
checkpoint_after, prepare_siblings, cancellation)` consumes the round and
reconstructs the accepted matrix (and optionally one checkpoint) from the starting
matrix. Both states and sibling preparation must succeed before the committed
bundle is replaced; failure, cancellation or invalid acceptance preserves the
original. `discard(id)` drops a round without publication.

**After-token counts.** Starting at `P`, verify ordinal `i` is the state after
`P + i + 1` materialized tokens. `checkpoint_node(P, accepted_nodes, B)` maps a
checkpoint count `B > P` to its physical node (`B == P` is the initial state);
natively this is Core's `crit::core::checkpoint_node`
([crit/state.h](../../core/include/crit/state.h)). Nodes are topologically
numbered and a chain may skip siblings, so an ordinal never equals a node by
assumption.

### Two-pass chunked prefill

Within a chunk of `C` tokens with inclusive cumulative log decay `g`, normalized
keys `k` and queries `q`, define `L_ij = b_i e^{g_i - g_j} k_i . k_j` for `j < i`
and `T = (I + L)^-1`. Corrected values are `U = Ub - W S0` with
`Ub = T diag(b) V` and `W = T diag(b e^g) K`; outputs are
`o_t = s (e^{g_t} q_t . S0 + sum_{i <= t} e^{g_t - g_i} (q_t . k_i) U_i)` and the
state after the chunk is `S_C = e^{g_C} S0 + sum_i e^{g_C - g_i} k_i U_i^T`. `T`,
`W`, `Ub` and the decayed `q . k` matrix are state independent, so pass 1 computes
them for all chunks in parallel and pass 2 walks chunks in order with matrix
products only (use FP32-equivalent accuracy, for example split TF32, for long
prompts). It only helps where matrix units run the needed precision much faster
than scalar code; otherwise a per-column kernel holding state in registers is as
fast. In one measured engine a two-pass kernel cut the recurrence from 58 to 42 ms
per 2,048 tokens without a measurable change in time to first token.

## What the code supplies

| File | Supplies |
| --- | --- |
| [recurrent_recovery.h](code/recurrent_recovery.h) | Header-only C++20 `Recovery`, `Geometry`, `Limits`, `Snapshot`, `Node`; includes `<crit/state.h>`, so users link `crit::core` |
| [example.cpp](code/example.cpp), [CMakeLists.txt](code/CMakeLists.txt) | Native example and regression target |
| [reference.py](code/reference.py) | Python formula, output-only verification and accepted-prefix reconstruction |
| [chunked_prefill.py](code/chunked_prefill.py) | `prepare_chunk`, `apply_chunk`, `two_pass`, `sequential`, `max_difference` |
| [stochastic_round.h](code/stochastic_round.h) | Host and device `bf16_stochastic` and `state_rounding_bits`: unbiased BF16 state storage from Core's keyed Philox draws |
| [recurrent_cuda.h](code/recurrent_cuda.h) / [.cu](code/recurrent_cuda.cu) | The device commit: `enqueue_select_accepted` (the state after the accepted columns from per-column checkpoints) and `enqueue_record_accepted` (the count of a deferred commit, one or many per-block counts), from a device-held count |
| [device_states.py](code/device_states.py) | The recurrence slot's device form: `StatePool` (states, stashes, convolution windows, round slot table), `forward`, `convolve`, `accept`, `commit_round` by strategy, over recurrent-scans' pool kernels or a capture-safe torch formula; `FLA_TILES` and `fla_pinned`, the stated tiles fla's chunked kernels run on |

Default native limits (65,536 state elements, 1,024 nodes, chain length 64,
magnitude `1e12`, and others in `Limits`) are mechanism choices an engine may
tighten. Invalid shapes, gates, parents, paths and overflowing intermediates are
rejected.

## What the engine supplies

Bind the starting matrix and inputs to the exact model representation and
committed token count. Calls, including sibling hooks, must be serialized;
borrowed `state()` views are valid only under serialized access.

`Snapshot::siblings` holds opaque byte vectors for every other continuation
component (convolution windows, embedding context, target/draft KV); the sibling
hook sees a private copy and the selected prefix and must derive restored state
only from those. The helper cannot discover omitted state.
[Checkpoint bundles](../checkpoint-storage/README.md) offer a checked component
inventory. Only materialized tokens count as accepted; a correction or bonus may
need another target forward first. [MTP execution](../mtp-decode/README.md) owns
the acceptance policy.

An asynchronous adaptation must retain the starting state, inputs, gates, path,
destinations and scratch until completion, reconstruct a boundary before
overwriting its `H`, and order side-stream recovery between verification input
production and the next target forward.

### The scans themselves come from a provider

The layer's prompt and decode arithmetic (the chunked prefill and the token-by-token scan
of gated delta, Mamba-2 and the short convolution before them) is a kernel provider's, not
this entry's: [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
(`fla.ops.gated_delta_rule`, listed in [Kernels' libraries](../../kernels/README.md#libraries)) for
prompt chunks, and crit's kernel recurrent-scans
([kernels/techniques/recurrence/recurrent-scans](../../kernels/techniques/recurrence/recurrent-scans), built for
the engine's device with the engine's tuned knobs or its defaults) for the rest: its
pool rows (`pool_delta`: a launch for the decode and verify rows of many sequences, each
from its own state, writing a stash, per-column checkpoints or its start state;
`pool_delta_replay`: the recompute commit and the flush; `pool_conv`, `conv_segments`: the
causal convolution over pooled windows) and its chunked prefill (`gated_delta_prefill`: on
tensor cores, structure `delta-prefill-tc`, the chunked WY form with FP16 operands and FP32
accumulation and state; or in FP32 on CUDA cores, `delta-prefill-rb`; both plugins' `prefill`
option, `tensor_cores` by default). What this entry adds is the speculative part: which state a
verify keeps, and how the accepted prefix becomes the live state without a host read
(below), and the pool that ties a slot's rows to them.

### Device commit pattern

On a GPU the accepted count comes from a device acceptance kernel, for example a
[device speculative round](../device-speculative-round/README.md)'s
`control[n_emitted]`. Keep the accepted state with no host read:

| Strategy | Verify row | Commit |
| --- | --- | --- |
| Per-column checkpoints | writes the state before its first column and after each column | copies the one after the accepted count (`enqueue_select_accepted`; in the pool a gather) |
| Recompute | keeps its start state and stashes each column's key, decay and delta | replays the first `n` stashed steps from the start state into the live state |
| Deferred | first replays its sequence's pending prefix from the stash (writing the live state), then verifies from it and stashes | records the count (`enqueue_record_accepted`; in the pool a fill of the per-block counts) |

Checkpoints cost `(columns + 1) * S` state memory per round row and write it every
round; recompute keeps one `S` per round row and reads and writes the state once more
at commit; deferred keeps no round-start copy and a stash of
`columns * (key_heads * keys + heads + heads * values)` values per layer, folds the
replay into the next row's state pass, and its commit is a fill of counts. Every
commit can run again with a smaller count before the sequence's next row (a host
truncation): checkpoints keep the state before column 0, recompute keeps its start,
and deferred overwrites the count. Apply a pending deferred prefix (a flush) before
reading or serializing state outside a row; the pool does it in `state()`, `window()`
and `fork()`. A flush makes a deferred commit final: rerun `commit_round` with a smaller
count (a truncation) before any flush, never after one (checkpoints and recompute stay
rerunnable until the row's next round). Each deferred row consumes its block's count after applying it, so a
flush, a repeated flush or a verified row that is never committed does not replay a
prefix twice. All three strategies give bitwise-identical state and outputs.

**Which strategy suits which regime.** A decode or verify row of a large state is
bound by reading and writing that state: deferred reads and writes it once per row,
recompute once more at commit, checkpoints writes it `columns + 1` times. Measured on
one RTX 5090 at 48 layers of 48 value heads x 128 x 128 (16 key heads):

| Rows x columns | Deferred: round (commit) | Recompute: round (commit) | Checkpoints: round (commit) |
| --- | --- | --- | --- |
| 1 x 1 | 0.51 ms (0.01) | 0.71 ms (0.23) | 1.01 ms (0.42) |
| 1 x 4 | 0.76 ms (0.01) | 0.90 ms (0.23) | 1.44 ms (0.43) |
| 1 x 8 | 1.06 ms (0.01) | 1.14 ms (0.22) | 1.94 ms (0.42) |
| 4 x 1 | 1.19 ms (0.02) | 2.05 ms (0.84) | 3.19 ms (1.61) |
| 4 x 4 | 1.79 ms (0.01) | 2.53 ms (0.84) | 4.49 ms (1.61) |
| 4 x 8 | 2.60 ms (0.01) | 3.13 ms (0.85) | 6.50 ms (1.61) |
| 8 x 8 | 4.28 ms (0.01) | | |

A round here is every layer's convolution, scan and the commit, captured as one CUDA graph,
with recurrent-scans' pool rows. The scan alone is 5.4 us a layer for one row of one
column (read and write 3 MiB) and 12.0 us for one row of eight columns; the same shape's
single-sequence column scan in place takes 6.4 and 10.4 us (24.6 with checkpoints). The
memory each strategy keeps beside the states: deferred a stash of 25 MB per sequence at 8
columns; recompute one state per round row (144 MiB at 48 layers); checkpoints `columns + 1`
of them (1.3 GiB per row at 8 columns). The torch formula (no kernel) takes 92 us a layer for
one row of one column and 1.2 ms for four rows of four columns: a fallback, not a decode path.

Deferred is faster in every regime measured (one to eight rows, one to eight columns) and
is the default of both plugins (`commit: deferred`). Checkpoints and recompute keep the
live state committed between rounds (no flush before reading it) at the memory and time
above: recompute suits an engine that reads the live state right after each commit (one
more state pass), checkpoints a small state (a few KiB per layer) where `columns + 1` copies
cost nothing. All three give the same bits, so the choice never changes a reply.

**Prompt chunks.** Per layer at the same shape (BF16 inputs): fla's chunked kernel takes 0.56
ms per 2,048-token chunk in BF16 and 0.77 ms in FP32 (3.2 ms per 8,192), the plugin's fla path
with its FP32 casts 0.88 ms; recurrent-scans' two-pass prefill (FP32 SIMT) 1.32 ms (5.2 ms per
8,192), this plugin's path with its normalization 1.55 ms; the column scan over the whole
chunk 2.05 ms; the convolution 0.03 ms. recurrent-scans' prefill on tensor cores (`prefill:
tensor_cores`, the default) takes 0.21 ms per 2,048-token chunk and 0.78 ms per 8,192 on FP32
inputs, ahead of fla and of FlashInfer's `chunk_gated_delta_rule` on BF16 inputs (0.29 and
0.84 ms; the kernel's README has the sweep). `prefill: fp32` keeps the FP32 two-pass prefill.
`sequential-recurrence` runs fla where it is installed and the same prefill otherwise. Every
chunked kernel sums in another order than the scan, so a prompt row's bits follow the prompt's
length and chunking: the facts' `recurrence_prompts` pools say which kernel runs
(`chunked`, `chunked_arithmetic`) and `chunked_row_invariant: false`.

**Short prompts.** A chunked kernel pays a fixed cost per call (fla: ~15 Triton launches a
layer, 0.12 ms of GPU time at 16 tokens and ~0.7 ms of host time), the column scan one launch
whose time grows with the tokens (0.03 ms at 16, 0.12 at 128, 0.45 at 512 a layer, with the
convolution). A prompt row of up to `exact_rows` tokens runs the scan, a longer one the
chunked kernel, in an eager call and in a captured prompt bucket alike (`exact_rows: 0`: every
prompt row the chunked kernel). The two paths give other bits, so the configuration states
the value (default 512, the largest of `crit_plugins.prompt_graphs`' default buckets) and no
pool measures it when it is made: a crossover measured at load moved with the machine's load
(eager 861 to 1,044 tokens from one process to the next on one RTX 5090), and where it fell
between a prompt's last chunk and its full chunks (953 and 1,024 rows) a prefix's rows parted
from the whole prompt's by 0.17 nats on average (crit smoke's `causal_long`). A model whose
prompt chunks are at most `exact_rows` tokens long runs every prompt row through the scan: a
prefix's rows are then the whole prompt's, bit for bit, and a prompt row's recurrence has the
bits of decoding it. `device_states.scan_rows()` measures where the chunked kernel gets
faster, over prompts of 16 to 4,096 tokens on a pool's shape and device, to recommend a
value: for an eager call, which costs the longer of its host time (Python and launches: the
stream waits for them when the host is behind, as in a short prompt) and its GPU time, and for
a call inside a captured graph (a prompt-graph bucket, or any capture), which costs its GPU
time. On the RTX 5090 at the shape above against fla in FP32: eager 987 tokens, captured
106-156 (the two GPU lines stay within ~15% of each other from 100 to 200 tokens); against the
two-pass prefill (a few launches) 625 both ways (32 value heads: 923 / 167 against fla; 16
value heads of 64: 1,190 / 246). Over 48 layers (convolution and recurrence) eager prompts take:

| Tokens | fla past the GPU crossover: GPU / wall | the scan (eager crossover): GPU / wall |
| --- | --- | --- |
| 160 | 8.2 / 37.7 ms | 7.5 / 8.1 ms |
| 213 | 9.0 / 32.6 ms | 9.6 / 10.1 ms |
| 256 | 8.6 / 32.5 ms | 11.3 / 11.9 ms |
| 512 | 11.4 / 32.7 ms | 21.4 / 22.3 ms |

fla's ~33 ms of host time over 48 layers is the same at every length; the scan's is ~3 ms. A
prompt row through the scan has the bits of decoding its tokens one by one (state, window,
outputs); one through a chunked kernel agrees with it within `2e-3 + 2e-2 |x|`. On the 27B
engine (prompt chunks of 1,024 tokens) `exact_rows` 1,024 against 512 cost 12-13% of
prefill (6,213 against 7,040 tokens/s at 8K, 4,994 against 5,511 at 32K) and nothing below
512 tokens.

**fla's tiles.** fla autotunes the six Triton kernels of its chunked forward (the L2 norm,
the decay cumsum, the fused `K K^T` solve, the WY recompute, the state pass and the output)
by timing their configs at a shape's first call, and keeps the pick in Triton's cache for
later processes. Not every tile gives the same bits: on the RTX 5090 at the shape above,
over 3,000 tokens with FP32 or BF16 rows and an FP32 initial state, the norm's 8-row blocks
on 8 or 16 warps (and its 16-row blocks on 16) and the output's 64-key block part from the
other tiles (outputs by up to 2.6e-5 with FP32 rows and 2.4e-4 with BF16 ones, the final
state by up to 6.6e-5), while every tile of the other four keeps the bits. So the plugin pins
each kernel to one stated tile (`device_states.FLA_TILES`, one of fla's own configs, the
autotuner's FP32 pick at that shape): fla then times nothing and a prompt row gets the same
bits on every run, at the autotuner's speed in FP32 and about 3% more kernel time in BF16.

With Core's `StateCache` ([Core state cache](../../core/src/state/README.md)),
declare the layers as a `StateComponent` whose `RecurrentCommit` matches the
strategy: `column_checkpoints` (verify writes `scratch[i]`, commit swaps
`scratch[n - 1]` live by id), `recompute` (writes `start` advanced by `n` into
`scratch[0]`) or `in_place` (deferred). Core's commit takes `n` on the host; the
device-count kernels remain for captured rounds that never read `n` back.

## The device form (the recurrence slot)

[device_states.py](code/device_states.py) is the slot's device form
([crit_plugins.device](../../core/python/crit_plugins/device.py)), which this
technique's plugin and the base `sequential-recurrence` share. A `StatePool` keeps per
layer and sequence the live FP32 state `[heads][keys][values]`, a two-half stash of
`widest + 1` columns and its per-block counts, and with `conv=(channels, kernel)` or
`(channels, kernel, dilation)` the convolution's window (the last `(kernel - 1) * dilation`
inputs of each channel: tap `i` of token `t` reads `x[t - (kernel - 1 - i) * dilation]`) and
its input stash. recurrent-scans' pool rows run 2 to 8 taps undilated and 2 to 4 taps at
dilation 2 to 4 (a model's dilated n-gram convolution, Flash-Next's PLE: 4 taps at dilation
3); another shape runs the torch formula, which the engine's facts record. A pool for a
convolution alone takes a one-value state (`heads = keys = values = 1`). Queries and keys
come at the checkpoint's key heads (`key_heads`, value head `h` reading key head
`h // (heads / key_heads)`).

- **step() rows.** `forward` and `convolve` run every row of a `StepBatch` in one launch:
  a row of up to `widest + 1` tokens (decode, verify) applies its sequence's pending
  count, scans from the live state and stashes; `accept(sequence, count)` records the
  count (any prefix). A longer row (a prompt chunk) keeps its final state and window,
  which `accept(sequence, its tokens)` makes live and `accept(sequence, 0)` drops; it
  runs the scan up to the pool's `scan_rows` tokens (`scan_rows_captured` while the stream
  captures a graph; both the plugin's `exact_rows`) and the chunked kernel past it (this
  plugin: recurrent-scans' chunked prefill, on tensor cores or in FP32 by `prefill`;
  `sequential-recurrence`: fla in FP32, on its stated tiles, else the same prefill).
- **captured prompt chunks** (`prompts:recurrence`, Core's `crit_plugins.prompt_graphs`).
  A `PromptBatch` (one sequence's chunk padded to a bucket) reads its sequence and count on
  the device from the rows the pool's `stage_prompt(batch)` wrote before the replay, and
  writes the final state and window over the live ones (no accept follows;
  `commit_prompt(batch)` after the replay). A bucket of up to `scan_rows_captured` tokens runs the scan
  over the chunk's real tokens: the bits of the step path's scan rows over the same tokens. A
  larger bucket runs the chunked kernel over the whole bucket with the padding's decay 1 and
  beta 0 (a step that keeps the state's bits), within the chunked tolerance of the scan. The
  convolution (recurrent-scans' `pool_conv_prompt`) convolves the real inputs only and moves
  the window past them. Padding outputs zeros; padding alone (a capture's warm-ups) changes no
  sequence.
- **captured rounds.** `bind(slot, sequence)` writes the sequence into the pool's device
  slot table; a `RoundBatch`'s rows read their sequences through it and run the
  strategy's kind; `commit_round(view)` keeps `view.n_emitted` columns per row by the
  plugin's `commit` option. Nothing reads the host, nothing allocates but the output,
  so a round of (rows, width) is captured once.
- **arithmetic.** recurrent-scans' `pool_delta` / `pool_delta_replay` / `pool_conv` /
  `conv_segments` / `gated_delta_prefill`, crit's kernel (declared in the plugin's
  manifest) built for the GPU on first use, with the engine's tuning
  (`CRIT_KERNEL_TUNING`) or its default knobs; the engine's facts say `tuned`, `built` or
  `fallback`. Where it does not build, the same semantics in torch, a row at a time (CPU,
  or a GPU without the kernel), capture-safe.

Each decode or verify row's bits are its own (the rows beside it change nothing), a
verify row's accepted columns have the bits of decoding them one by one, and a row
rolled back to any count continues as the serial decode, state and window, by every
strategy and on both paths. Prompt rows through a chunked kernel follow the FP64
recurrence within `2e-3 + 2e-2 |x|` (another summation order), the scan within
`1e-5 + 1e-4 |x|`.

## Worked example

Run from `code/` with Python 3.11+. Node 1 is a rejected sibling; the checkpoint at
64 is the state after physical node 0, not node 2.

```python
from reference import Geometry, Node, Step, checkpoint_node, verify_chain

geometry = Geometry(2, 1)
first = Step((0.3, 0.7), (0.4, -0.2), (1.0,), -0.2, 0.6)
second = Step((0.5, -0.1), (0.1, 0.8), (-0.5,), -0.1, 0.4)
round = verify_chain((0.0, 0.0),
                     (Node(None, first), Node(0, first), Node(0, second)),
                     (0, 2), geometry, prefix_after=63)
assert checkpoint_node(63, round.path, 64) == 0
assert round.recover(0) == round.initial
checkpoint = round.recover(1)
assert checkpoint.after_tokens == 64
restored = verify_chain(checkpoint.matrix, (Node(None, second),), (0,),
                        geometry, prefix_after=checkpoint.after_tokens)
assert restored.recover(1).matrix == round.recover(2).matrix
assert round.initial.matrix == (0.0, 0.0)

# Prompt prefill: the two-pass chunked form equals the token-by-token recurrence.
from chunked_prefill import max_difference, sequential, two_pass
steps = [first, second, first, second, second]
direct = sequential((0.1, -0.2), steps, geometry)
chunked = two_pass((0.1, -0.2), steps, geometry, chunk_size=2)
assert max_difference(direct, chunked) < 1e-12
```

Native POC from the repository root:

```text
cmake -S inference/recurrent-state-recovery/code -B build/recurrent-poc -DBUILD_TESTING=ON
cmake --build build/recurrent-poc --config Release
ctest --test-dir build/recurrent-poc -C Release --output-on-failure
```

## Adapting it

Preserve one update expression on every path, after-token checkpoint identity,
all-or-nothing commit, and idempotent device commit from round-start state. The
strategy, storage precision and bounds are adaptable; an engine with BF16 state or
activations must state its own cast points.

An engine that stores state in BF16 rounds it once per token, and round to nearest
pushes a slowly changing value the same way each time. `bf16_stochastic` rounds up
with probability equal to the discarded fraction, so each stored value is an unbiased
estimate of the FP32 one. Its 16 random bits come from `state_rounding_bits(seed,
position, layer, element)`, Core's Philox draw, so a replay or recompute of the same
step stores the same bits and checkpoint and recompute stay bitwise identical. It is
opt-in: whether it brings a BF16-state engine closer to the reference over long
outputs is a crit-eval measurement, not yet made.

## Costs and validation

For state size `S`, step input size `I`, output width `V` and chain length `D`, a
retained round uses `O(S + nodes*I + D*V)` instead of `O(D*S)` snapshots, but
recovery still recomputes the recurrence. The device strategies' costs are measured
above (which strategy suits which regime).

The [CUDA fixture](../../tests/inference/recurrent_cuda_tests.cpp) under
`CRIT_INFERENCE_CUDA` checks the device commit: the select chooses the start or the
accepted checkpoint for every count (clamped below 0 and above the columns) bit for
bit, and the many-count record writes every group's counts and nothing else. The
reference scans its strategies were proven with (checkpoints, recompute and deferred
against the FP64 host oracle over 96 geometries, deferred for 96 rounds bit for bit
against checkpoints) moved to crit's kernel recurrent-scans with their fixture. The
plugin's tests check the device form as stated above, on CPU, on CUDA with
recurrent-scans and with the torch formula; `sequential-recurrence`'s, that every
autotuned kernel fla's chunked path reaches holds its stated tile, so nothing times a tile
and a prompt row keeps its bits with the autotuners' picks cleared.
[Native fixtures](../../tests/inference/recurrent_recovery_tests.cpp) compare
every accepted length with a long-double oracle, covering nonsequential paths,
63/64/65 boundaries, stale commits, sibling failure, cancellation, reentry and
malformed inputs. Full-model continuation, device casts and performance remain
engine qualification work.
