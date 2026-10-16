# Partitioned attention

Split a long KV sequence across ranks so each evaluates the same queries over
disjoint keys, then merge the partial results exactly. This is a host reference,
not a distributed runtime or device kernel.

## When to use it

Use it when replicating KV wastes memory, for dense or globally selected sparse
keys. Short contexts may lose to communication overhead.

## How it works

`SequencePartition` assigns global position `p` to rank `p % world`, at local
position `p // world`; counts differ by at most one. A causal query at `q` sees
only positions `p <= q`, using **global** positions. For local output `o[r]` and
log normalizer `L[r]`:

```text
M = max(L)
w[r] = exp(L[r] - M) / sum(exp(L[s] - M))
output = sum(w[r] * o[r])
global_L = M + log(sum(exp(L[r] - M)))
```

An empty shard supplies zero output and `L = -inf`; all-empty attention, NaN and
positive infinity are errors. The merge equals attention over the union of keys
up to reduction-order rounding.

## What the code and engine supply

[implementation.py](code/implementation.py) supplies `SequencePartition` (owner,
local/global position, lengths, page addresses), `attention_partial` (stable
single-query attention with causal positions and optional selected keys),
`merge_partials`, and `combine_and_scatter`, which reduces `[rank][query][head]`
partials into `[rank][local_head][query]` (heads must divide evenly).

The engine applies query/key transforms, passes rows as
`(global_position, key, value)` with identical query, scale and selection on every
rank, guarantees disjoint, complete ownership, and supplies collectives and
buffers (a native path all-gathers normalizers and reduce-scatters weighted outputs).

## Worked example

Run from `code/`. Four ranks share two tokens; two ranks contribute empty shards.

```python
from implementation import SequencePartition, attention_partial, merge_partials

rows = ((0, (1., 0.), (2.,)), (1, (0., 1.), (8.,)))
parts = []
for rank in range(4):
    layout = SequencePartition(4, rank)
    local = [row for row in rows if layout.owner(row[0]) == rank]
    parts.append(attention_partial((1., 1.), local, value_width=1,
                                   query_position=1))
assert merge_partials(parts).output == (5.,)
assert SequencePartition(4, 1).global_position(3) == 13
assert attention_partial((1., 1.), rows, value_width=1,
                         query_position=0).output == (2.,)
```

## Adapting it

Every visible key contributes once. Preserve global causal positions, query/head
order, scale and sparse selection. Changing partition count requires remapping
live KV and invalidating graph bindings. Interleaving, page size, tiles and
transport are adaptable.

## Costs and validation

Sharded KV approaches `1/world` per rank. Missing
ranks, duplicated keys or mismatched masks give plausible wrong answers; failed
collectives must block publication. [Host fixtures](../../tests/test_reusable_techniques.py)
compare dense and sparse partitions with direct softmax, including uneven and
empty shards and extreme normalizers. Native attention, collectives and
performance are unqualified.
