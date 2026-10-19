# Asynchronous expert contributions

Join independently launched work, such as host and device experts, into one
routed result only after every required producer has physically finished, while
keeping each producer's sources alive.

## When to use it

Use it when one logical routed operation launches independent work and must
produce one result. The unit is a **route occurrence**: two positions can select
the same expert and still owe two contributions. A single fused projection may
need no host-side group, but the ownership and completion rules still apply if the
join moves into a device stream; the [host-work mailbox](../host-work-mailbox/README.md)
supplies that join for host producers.

## How it works

The caller declares ordered route positions and accepts one completion receipt and
source owner per position. The group is ready only when every position is
submitted and every receipt reports physical completion. Settlement retrieves
every result, raises if any failed, and otherwise sums vectors in declared
position order. Vectors must already include their routing weight.

Physical completion and successful arithmetic are different facts: a failed
receipt can be finished and safe to release, while the first failure does not
authorize releasing sources another producer still reads.

## What the code and engine supply

[`ContributionGroup`](code/implementation.py) supplies accepted-work bookkeeping,
strong references to owners, readiness polling, ordered reduction and an abort
boundary. `settle()` consumes the group even on failure. Receipts need `done()`
and `result()`. Widths are checked; nonfinite values are not. There is no
executor, kernel, routing policy or transfer pipeline.

The engine:

1. Launches work, then calls `accept(position, receipt, owner)`; the interval
   before acceptance, including a launch that throws after enqueueing, is its own.
2. Calls `settle()` before exposing the result.
3. On cancellation, calls `abort(quiesce)`, which must stop producers, settle
   accepted reads and handle launched but unaccepted work. If quiescence fails,
   owners stay held.

Cancelling a request or host future does not prove a device stopped reading.

## Worked example

Run this from `code/`. Two contributions finish independently; neither an
incomplete submission nor a single finished receipt permits a join.

```python
from concurrent.futures import Future
from implementation import ContributionGroup

left, right = Future(), Future()
group = ContributionGroup((2, 7))
group.accept(2, left, owner=bytearray(b"left weights"))
left.set_result((1.0, 3.0))
assert not group.ready()  # position 7 has not even been submitted

group.accept(7, right, owner=bytearray(b"right weights"))
assert not group.ready()
try:
    group.settle()
except ValueError:
    pass
else:
    raise AssertionError("premature publication")

right.set_result((2.0, -1.0))
assert group.ready()
assert group.settle() == (3.0, 2.0)
assert not group.accepted  # both physical reads ended before release
```

## Adapting it

Preserve one contribution per occurrence, source retention through physical
reads, all-producer settlement on failure and publication only after success.
Containers, reduction, polling and executor can change; a different reduction
order changes rounding, so test against a numerical criterion. With
[expert residency](../expert-residency/README.md) an owner can hold a cache lease;
[mixed-source dispatch](../heterogeneous-expert-dispatch/README.md) may resolve
all sources in one projection and remove the group entirely.

## Costs and validation

Reduction costs O(PW) for P positions of width W. A split that duplicates
activation preparation, launches or transfers may be slower than one device
operation; measure the whole operation including the slowest producer. Common
errors: dropping a repeated occurrence, weighting twice and releasing all owners
on the first failure. Host fixtures cover joins, errors and ownership, not native
overlap or speedup; a native port should inject failures while another producer
is live.
