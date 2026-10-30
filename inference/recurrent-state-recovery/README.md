# Recover accepted recurrent state without per-token matrix snapshots

Speculative verification of a gated-delta recurrent layer can keep the starting
matrix and step inputs, compute verification outputs without saving a matrix per
token, then reconstruct only the accepted state. This trades state storage for
repair work. The entry supplies a bounded single-head host POC, an FP32 CUDA
reference of the device commit for multi-head state, and a two-pass chunked
prefill reference.

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
`code/recurrent_cuda.cu` routes every path through one `updated()` function.

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
| [recurrent_cuda.h](code/recurrent_cuda.h) / [.cu](code/recurrent_cuda.cu) | FP32 CUDA reference for `heads` independent heads and the device commit below (not for performance) |

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

### Device commit pattern

On a GPU the accepted count comes from a device acceptance kernel, for example a
[device speculative round](../device-speculative-round/README.md)'s
`control[n_emitted]`. Select the accepted state with no host read:

| Strategy | Verify | Commit |
| --- | --- | --- |
| Per-column checkpoints | `enqueue_verify(..., checkpoints, nullptr)` writes the state after every column | `enqueue_select_accepted` copies `checkpoints[n - 1]`, or the start state when `n == 0` |
| Recompute | `enqueue_verify(..., nullptr, work)` keeps one running copy | `enqueue_recompute_accepted` replays the first `n` steps from the start state |
| Deferred | `enqueue_verify_deferred` applies the previous round's pending prefix from a compact stash and consumes the count, then verifies and overwrites the stash | `enqueue_record_accepted` writes only the count |

Checkpoints cost `columns * S` state memory; recompute repeats `n` steps; deferred
keeps no round-start copy and a stash of `columns * heads * (keys + 1 + values)`
values, folds the replay into the next verify, and its commit is a scalar write.
Apply a pending deferred prefix (`columns = 0`) before reading or serializing state
outside a round. Each deferred verify zeroes the pending count after applying it, so
a flush, a repeated flush or a verified round that is never recorded does not replay
a prefix twice; rerun `enqueue_record_accepted` only before the next deferred verify.
Counts are read through a device pointer and clamped to
`[0, columns]`; keeping round-start and live state separate lets commit rerun with
a smaller count after a host truncation. One block per head, one thread per value
column; all three strategies produce bitwise-identical state. A second
`enqueue_record_accepted` writes the count into `count` slots of each of `groups` groups in
one launch, for a deferred commit that keeps a count per block: an engine's column scan and
convolution run Strategy C with them, each block applying the accepted prefix with the scan's own
update expression.

With Core's `StateCache` ([Core state cache](../../core/src/state/README.md)),
declare the layers as a `StateComponent` whose `RecurrentCommit` matches the
strategy: `column_checkpoints` (verify writes `scratch[i]`, commit swaps
`scratch[n - 1]` live by id), `recompute` (writes `start` advanced by `n` into
`scratch[0]`) or `in_place` (deferred). Core's commit takes `n` on the host; the
device-count kernels remain for captured rounds that never read `n` back.

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
recovery still recomputes the recurrence. No throughput or GPU memory saving is
measured.

The [CUDA fixture](../../tests/inference/recurrent_cuda_tests.cpp) under
`CRIT_INFERENCE_CUDA` compares outputs and committed state for every accepted
count against the FP64 host oracle over 96 geometries within
`|gpu - ref| <= 2e-5 + 2e-5 * |ref|`, requires bitwise equality between
checkpoint and recompute, and runs the deferred strategy for 96 rounds (including
truncation reruns) bit for bit against checkpoints, plus a flush between rounds, a
repeated flush and an abandoned round bit for bit against recompute.
[Native fixtures](../../tests/inference/recurrent_recovery_tests.cpp) compare
every accepted length with a long-double oracle, covering nonsequential paths,
63/64/65 boundaries, stale commits, sibling failure, cancellation, reentry and
malformed inputs. Full-model continuation, device casts and performance remain
engine qualification work.
