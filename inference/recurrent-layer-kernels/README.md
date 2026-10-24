# Run hybrid models' recurrent layers on the device

CUDA kernels for the recurrent layers of hybrid models: a short causal depthwise
convolution with its carried window, gated-delta attention (Qwen3-Next and Qwen3.5-style
gated DeltaNet, Kimi-style delta layers with per-head gates) and Mamba-2's state-space
recurrence (Nemotron-H, Granite 4, Falcon-H1). Each has a chunked two-pass prefill for
prompts and a column scan for decode and speculative verify that can write the state after
every column, the form Core's `RecurrentCommit::column_checkpoints` and
[recurrent-state-recovery](../recurrent-state-recovery/README.md) consume.

## When to use it

For a model with gated-delta or Mamba-2 layers, before writing these kernels for one
engine: an engine for a hybrid model otherwise rebuilds them (the first case-study engine
did, for gated delta). Use the column scan for decode and verify, and the chunked prefill
for prompts long enough to fill several 64-token chunks. Not for other recurrences (RWKV,
Mamba-1's selective scan with per-channel decay, lightning attention) without adapting the
math, and not when state widths fall outside the supported shapes below.

## How it works

**Convolution.** `out[t] = act(bias + sum_i w[i] * x[t - (width - 1) + i])`, one thread per
channel walking the tokens, the window of the last `width - 1` inputs carried in registers
and stored back as the state (optionally after every token, for verify columns).

**Gated delta.** Per value head with state `S[dk][dv]` and per token unit key `k`, query
`q`, value `v`, log decay `l <= 0` and `beta`:

```text
delta = beta (v - e^l S^T k);   S' = e^l S + k delta^T;   out = e^l S^T (s q) + ((s q) . k) delta
```

The column scan holds `S` in registers (lane `l` of a warp owns `dk / 32` key rows, the warp
four value columns), reduces each dot product to lane 0 and broadcasts it, and applies one
update expression, `fmaf(k, delta, S * e^l)`, to every column, so decode and verify agree bit
for bit. The prefill is the two-pass chunked form of recurrent-state-recovery's
`chunked_prefill.py`: pass 1 builds every 64-token chunk's `T = (I + L)^-1`,
`T diag(beta)`, `W = T diag(beta e^G) K` and decayed `q . k` in parallel; pass 2 walks the
chunks in order per (head, 64 value columns) with the state slice in shared memory.

**Mamba-2.** Per head with state `H[p][n]`, `dt > 0`, `A < 0` and the group's `B`, `C`:

```text
H' = e^{dt A} H + dt x B^T;   y = H' C + D x
```

The column scan gives each warp one row `p` and each lane `d_state / 32` state columns.
The prefill's pass 1 builds `M_ts = (C_t . B_s) e^{a_t - a_s} dt_s` and the cumulative
decays `a` for every chunk in parallel; pass 2 walks the chunks per (head, 32 rows):
`y_t = e^{a_t} H C_t + sum_{s <= t} M_ts x_s + D x_t` and
`H' = e^{a_Q} H + sum_s e^{a_Q - a_s} dt_s x_s B_s^T`.

The chunked prefills equal the recurrence up to rounding; they are a different summation,
which prompts tolerate and decode never sees.

## What the code supplies

| File | Supplies |
| --- | --- |
| [recurrent_kernels.h](code/recurrent_kernels.h) / [.cu](code/recurrent_kernels.cu) | `enqueue_conv`; `enqueue_delta_prefill`, `enqueue_delta_columns` (`dk` 64 or 128, `dv` a multiple of 64); `enqueue_ssd_prefill`, `enqueue_ssd_columns` (`d_state` 64 or 128, `head_dim` a multiple of 32); scratch sizing |
| [ssd.py](code/ssd.py) | Host references: the convolution, Mamba-2 token by token and chunked |

The gated-delta host reference is recurrent-state-recovery's `chunked_prefill.py`.

## What the engine supplies

The projections into `q`, `k`, `v`, gates, `B`, `C` and `dt`; activations before the
recurrence (the convolution's SiLU is here, the gates' softplus, `exp` and `sigmoid` and the
key/query normalization are the engine's); the output gate and norm; state storage in
Core's slots; and the scratch (`delta_scratch_floats`, `ssd_scratch_floats`). For verify,
pass the round's columns to the column scan with checkpoints and let Core's commit select
the checkpoint of the accepted count.

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

Preserve: one update expression per recurrence across decode, verify and replay; the
dot products reduced once and broadcast; partial chunks padded with tokens that neither
update nor output; and checkpoints that are the state after each column. Adaptable: tensor-core
matrix products in the prefill passes (keep FP32-equivalent accuracy, for example split TF32),
BF16 inputs, fusing the convolution into the input projection's epilogue, and tile shapes.

## Costs and validation

The column scan reads each state once per call and the inputs once per token; the
prefill's pass 1 is parallel over (head, chunk) and pass 2 over (head, value tile), with
`O(64^2 dk)` work per chunk. Shared memory: up to 96 KiB per block at `dk = 128` or
`d_state = 128`, within sm_80+ limits. Reference SIMT speed; no speed is claimed.

[recurrent_kernels_tests.cu](../../tests/inference/recurrent_kernels_tests.cu) compares the
convolution (outputs, state, checkpoints), both column scans (outputs, state, checkpoints,
and five columns at once equal bit for bit to one at a time) and both chunked prefills over
300 tokens (four full chunks and a partial one) with FP64 token-by-token references, for
`dk` 64 and 128 and `d_state` 64 and 128. [test_recurrent_kernels.py](../../tests/test_recurrent_kernels.py)
checks the host references.
