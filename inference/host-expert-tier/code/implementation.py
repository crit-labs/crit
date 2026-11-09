"""Python model of the CPU expert tier's two contracts (expert_pool.h, host_tier.h), standard library only.

The pool: a batch's head is one word, epoch << 32 | next task, and a claim is a compare-and-swap that succeeds only
for the epoch the worker woke for, so a worker still leaving an earlier batch claims nothing of the next one.

The tier: routes are grouped by expert in route order, at most 8 tokens a chunk, and every route gets its own
unweighted output row. The device multiplies by the routing weight and sums a token's experts in its plan order, so
which experts the CPU computed never changes the order of that sum.
"""

import struct

MAX_TOKENS = 8


def f32(x):
    """Round a double to FP32 (correct for a product or a sum of two FP32 values)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


class Head:
    """The pool's claim word: run() publishes epoch << 32 for a new batch; claim() is one compare-and-swap."""

    def __init__(self):
        self.word, self.tasks = 0, 0

    def publish(self, epoch, tasks):
        self.tasks = tasks
        self.word = (epoch & 0xFFFFFFFF) << 32

    def claim(self, epoch, seen):
        """A worker's claim with the word it read: None when its epoch is over, the swap failed or the batch is drained."""
        if seen >> 32 != epoch & 0xFFFFFFFF or self.word != seen or seen & 0xFFFFFFFF >= self.tasks:
            return None
        self.word = seen + 1
        return seen & 0xFFFFFFFF


class UntaggedHead(Head):
    """The same claim without the epoch: a worker that read the word in one batch can claim a task of the next."""

    def claim(self, epoch, seen):
        if self.word & 0xFFFFFFFF != seen & 0xFFFFFFFF or seen & 0xFFFFFFFF >= self.tasks:
            return None
        self.word += 1
        return seen & 0xFFFFFFFF


def chunks(experts, max_tokens=MAX_TOKENS):
    """Routes grouped by expert in route order, at most max_tokens a chunk: [(expert, [route, ...]), ...]."""
    out = []
    for route, expert in enumerate(experts):
        for e, routes in out:
            if e == expert and len(routes) < max_tokens:
                routes.append(route)
                break
        else:
            out.append((expert, [route]))
    return out


def tier_rows(experts, expert_row):
    """Every route's unweighted row, computed chunk by chunk as the tier does; expert_row(expert, route) -> row."""
    rows = [None] * len(experts)
    for expert, routes in chunks(experts):
        for route in routes:
            rows[route] = expert_row(expert, route)
    return rows


def device_sum(plan, weights, rows):
    """A token's output: weight times row, summed in the device's plan order, one FP32 rounding per operation."""
    width = len(rows[plan[0]])
    out = [0.0] * width
    for route in plan:
        for i in range(width):
            out[i] = f32(out[i] + f32(weights[route] * rows[route][i]))
    return out


def host_partial(routes, weights, rows):
    """A host-side sum of the CPU's routes: the shape a tier must not have, since it reorders the device's sum."""
    width = len(rows[routes[0]])
    out = [0.0] * width
    for route in routes:
        for i in range(width):
            out[i] = f32(out[i] + f32(weights[route] * rows[route][i]))
    return out


__all__ = ["Head", "UntaggedHead", "chunks", "tier_rows", "device_sum", "host_partial", "bits", "f32", "MAX_TOKENS"]
