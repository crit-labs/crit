# Run hybrid models' recurrent layers on the device

CUDA kernels for the recurrent layers of hybrid models: a short causal depthwise
convolution with its carried window, gated-delta attention (Qwen3-Next and Qwen3.5-style
gated DeltaNet, Kimi-style delta layers with per-head gates) and Mamba-2's state-space
recurrence (Nemotron-H, Granite 4, Falcon-H1). Each has a chunked two-pass prefill for
prompts and a column scan for decode and speculative verify that can write the state after
every column, the form Core's `RecurrentCommit::column_checkpoints` and
[recurrent-state-recovery](../recurrent-state-recovery/README.md) consume, or defer the commit
of the accepted prefix to the next verify (recurrent-state-recovery's Strategy C).

## When to use it

For a model with gated-delta or Mamba-2 layers, before writing these kernels for one
engine: an engine for a hybrid model otherwise rebuilds them (the first case-study engine
did, for gated delta). Use the column scan for decode and verify, and the chunked prefill
for prompts long enough to fill several 64-token chunks. Not for other recurrences (RWKV,
Mamba-1's selective scan with per-channel decay, lightning attention) without adapting the
math, and not when state widths fall outside the supported shapes below.

## How it works

**Convolution.** `out[t] = act(bias + sum_i w[i] * x[t - (width - 1) + i])`, one thread per
channel walking the tokens, the window of the last `width - 1` inputs carried in registers,
read from `state_in` and written to `state_out` (optionally after every token, for verify
columns). `state_out` may be `state_in`, the window updated in place, or another buffer, so a
verify reads its round-start window without a copy. Out of place (`out` is not `x`), a prompt
of more than 32 rows runs a thread per (channel, 32 rows) instead: each segment starts from
the `width - 1` inputs before it and computes every output with the same expression, so the
bits equal the row-by-row form's.

**Gated delta.** Per value head with state `S[dk][dv]` and per token unit key `k`, query
`q`, value `v`, log decay `l <= 0` and `beta`:

```text
delta = beta (v - e^l S^T k);   S' = e^l S + k delta^T;   out = e^l S^T (s q) + ((s q) . k) delta
```

The column scan holds `S` in registers (lane `l` of a warp owns `dk / 32` key rows, the warp
four value columns), reduces each dot product once and broadcasts it (the eight of a column,
`k . S` and `q . S` for the warp's four values, together: a reduce-scatter whose pair sums
are those of a warp tree to lane 0, 17 shuffles instead of 48), and applies one update
expression, `fmaf(k, delta, S * e^l)`, to every column, so decode and verify agree bit for
bit. Each token's decay `e^l` is FP64 `exp` rounded once to FP32, computed once per block:
CUDA's `expf` is not correctly rounded and errs upward more often than downward, and a decay
near 1 applied token after token compounds that bias into the long-memory state. The prefill
is the two-pass chunked form of recurrent-state-recovery's `chunked_prefill.py`: pass 1
builds every 64-token chunk's `T = (I + L)^-1`, `T diag(beta)`, `W = T diag(beta e^G) K` and
decayed `q . k` in parallel; pass 2 walks the chunks in order per (head, 64 value columns)
with the state slice in shared memory.

**Mamba-2.** Per head with state `H[p][n]`, `dt > 0`, `A < 0` and the group's `B`, `C`:

```text
H' = e^{dt A} H + dt x B^T;   y = H' C + D x
```

The column scan gives each warp one row `p` and each lane `d_state / 32` state columns, with
`e^{dt A}` rounded once from FP64 as for gated delta.
The prefill's pass 1 builds `M_ts = (C_t . B_s) e^{a_t - a_s} dt_s` and the cumulative
decays `a` for every chunk in parallel; pass 2 walks the chunks per (head, 32 rows):
`y_t = e^{a_t} H C_t + sum_{s <= t} M_ts x_s + D x_t` and
`H' = e^{a_Q} H + sum_s e^{a_Q - a_s} dt_s x_s B_s^T`.

The chunked prefills equal the recurrence up to rounding; they are a different summation,
which prompts tolerate and decode never sees.

**Deferred commit.** recurrent-state-recovery's Strategy C on the column scan and the
convolution: one live state and window, no round-start copy and no checkpoints. A verify
first applies the previous round's accepted prefix from a stash (for gated delta the scan's
update expression on each accepted step's stashed key, `e^l` and delta, which are the operands
the verify used, so the state is the verify's running state bit for bit; for the convolution
the stashed raw rows shifted into the window), writes it, then verifies the new columns
without writing the state, stashing their operands. Each block keeps its own pending count
and consumes it, so no second launch and no atomics are needed and a flush (zero columns), a
repeated flush or a verify that is never recorded applies nothing twice. The gated-delta stash
is double-buffered with a per-block parity, since the blocks of a key head share its stashed
key. recurrent-state-recovery's `enqueue_record_accepted` writes the accepted count, read on
the device, into every block's count of every layer in one launch.

## What the code supplies

| File | Supplies |
| --- | --- |
| [recurrent_kernels.h](code/recurrent_kernels.h) / [.cu](code/recurrent_kernels.cu) | `enqueue_conv`, `enqueue_conv_deferred`; `enqueue_delta_prefill`, `enqueue_delta_columns`, `enqueue_delta_columns_deferred` (`dk` 64 or 128, `dv` a multiple of 64); `enqueue_ssd_prefill`, `enqueue_ssd_columns` (`d_state` 64 or 128, `head_dim` a multiple of 32); scratch, stash and count sizing |
| [ssd.py](code/ssd.py) | Host references: the convolution, Mamba-2 token by token and chunked |

The gated-delta host reference is recurrent-state-recovery's `chunked_prefill.py`.

## What the engine supplies

The projections into `q`, `k`, `v`, gates, `B`, `C` and `dt`; activations before the
recurrence (the convolution's SiLU is here, the gates' softplus, `exp` and `sigmoid` and the
key/query normalization are the engine's); the output gate and norm; state storage in
Core's slots; and the scratch (`delta_scratch_floats`, `ssd_scratch_floats`). For verify,
pass the round's columns to the column scan with checkpoints and let Core's commit select
the checkpoint of the accepted count, or use the deferred commit for gated-delta layers:
verify with `enqueue_conv_deferred` and `enqueue_delta_columns_deferred`, record the
device-held accepted count with recurrent-state-recovery's `enqueue_record_accepted`
(`conv_deferred_slots` and `delta_deferred_slots` counts per layer, zeroed once), and flush
(zero columns) before reading, serializing or prefilling the state outside a round; with
Core's StateCache that is `RecurrentCommit::in_place`. A chunked prefill writes only the
state after its last row: begin its round with `RoundStates::final_row` and write that state
into `scratch[0]`, so a prefix-cache commit never snapshots a page boundary inside the chunk.

## Worked example

From `code/`: the chunked Mamba-2 evaluation the prefill kernels use equals the token-by-token
recurrence the column scan uses, for chunks that do and do not divide the prompt.

```python
import random
from ssd import conv, max_difference, ssd_chunked, ssd_sequential

rng = random.Random(0)
H0 = [[rng.gauss(0, 0.2) for _ in range(4)] for _ in range(2)]           # head_dim 2, d_state 4
steps = [([rng.gauss(0, 1) for _ in range(2)], rng.uniform(0.01, 0.3),   # x, dt, B, C per token
          [rng.gauss(0, 0.5) for _ in range(4)], [rng.gauss(0, 0.5) for _ in range(4)]) for _ in range(10)]
direct = ssd_sequential(H0, steps, A=-1.2, D=0.5)
assert max_difference(direct, ssd_chunked(H0, steps, A=-1.2, D=0.5, chunk=4)) < 1e-12   # 4 + 4 + 2 tokens
out, state = conv([1.0, 2.0, 3.0], weight=[0.5, 0.25, 1.0], state=[0.0, 0.0], silu=False)
assert out == [1.0, 2.25, 4.0] and state == [2.0, 3.0]                  # the window carries the last two inputs
```

## Adapting it

Preserve: one update expression per recurrence across decode, verify and replay; per-token
decays rounded once from FP64 `exp` (an `expf` decay drifts over long decodes); the
dot products reduced once and broadcast; partial chunks padded with tokens that neither
update nor output; checkpoints that are the state after each column; and a deferred replay
that applies the update expression to the verify's own stashed operands. Adaptable: tensor-core
matrix products in the prefill passes (keep FP32-equivalent accuracy, for example split TF32),
BF16 inputs, fusing the convolution into the input projection's epilogue, and tile shapes.

## Costs and validation

The column scan reads each state once per call and the inputs once per token; the
prefill's pass 1 is parallel over (head, chunk) and pass 2 over (head, value tile), with
`O(64^2 dk)` work per chunk. Shared memory: up to 96 KiB per block at `dk = 128` or
`d_state = 128`, within sm_80+ limits. Reference SIMT speed; no speed is claimed. The FP64
decay costs little where FP64 runs at half rate; where it runs at 1/64 (consumer GPUs) it
adds its latency to short calls (one RTX 5090, one token: 3.3 to 3.8 us for 32 gated-delta
heads of 128 x 128, 3.6 to 4.6 us for 128 Mamba-2 heads of 64 x 128; unchanged or faster
from 16 tokens). The prompt convolution's segments give a prompt `rows / 32` times the
parallelism of a thread per channel for `(width - 1) / 32` extra reads (one RTX 5090, 8192
rows of 10,240 channels, width 4: 0.45 ms out of place, 1.2 ms in place). The deferred
commit keeps a stash of `2 capacity (k_heads dk + v_heads + v_heads dv)` floats per layer
instead of a state per column, and its verify costs the plain scan plus the replay (one RTX
5090, 32 gated-delta heads of 128 x 128, 9 columns with 4 pending: 9.6 us with the record,
against 7.0 us for the plain scan and 15.7 us with checkpoints).

[recurrent_kernels_tests.cu](../../tests/inference/recurrent_kernels_tests.cu) compares the
convolution (outputs, state, checkpoints, a `state_in` left unwritten, and the prompt form bit for
bit with the row-by-row form for widths 2 to 8 and up to 8192 rows), both column scans (outputs,
state, checkpoints, and every verify width from 1 to 16 equal bit for bit to the same tokens one at
a time; every checkpoint of a 1-, 2-, 4- or 8-column scan bit for bit the final state of a separate
shorter scan, for 33 shapes up to 64 value heads, with the plane after the round left unwritten),
8192-token decodes with slow decays (`e^l` within about 1e-6 to 1e-3 of 1) whose state stays within
twice the error of an FP32 recurrence with correctly rounded decays (`expf` gave 4 to 10 times), and
both chunked prefills over 300 tokens (four full chunks and a partial one) with FP64 token-by-token
references, for `dk` 64 and 128 and `d_state` 64 and 128. The deferred commit runs 160 rounds for
each of four shapes (`dk` 64 and 128, `dv` 64 and 128, 1 to 4 value heads per key head, convolution
widths 2 to 8 over two blocks) with random widths, accepted counts of 0 to all, abandoned rounds,
truncation reruns, flushes and prompt chunks between rounds, two layers per record: every accepted
column's outputs and every flushed state and window equal plain decoding bit for bit.
[test_recurrent_kernels.py](../../tests/test_recurrent_kernels.py) checks the host references.
