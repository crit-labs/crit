"""Check that a batched operator computes every column identically whatever the batch width.

Greedy speculative output equals plain greedy output only if the verify pass (K + 1 columns) produces, for each
column, exactly the logits the one-token step would. Floating-point sums depend on their order, so a kernel
whose reduction split or tile choice depends on the number of columns (split-K chosen by T, a different tile
for wide batches, a width-dependent reduction tree) changes low-order bits with the width, and an argmax near a
tie can flip. Exactness then depends on luck. A kernel that sums each column in a fixed order for every width
(fixed K split, fixed-order partial reductions) makes exactness a property of the arithmetic.

`check` runs an operator on the same columns at several widths, embedded among different neighbors, and
reports any column whose result is not bitwise identical. The two reference operators show both behaviors.
"""
import struct


def f32(x):
    """Round to the nearest float32, as a device accumulator does."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def fixed_order_dot(weights):
    """op(columns) -> per column, per row: sum over k in a fixed order (two fixed halves), whatever the width."""
    def op(columns):
        out = []
        for x in columns:
            rows = []
            for w in weights:
                half = len(w) // 2
                a = b = 0.0
                for k in range(half):
                    a = f32(a + f32(w[k] * x[k]))
                for k in range(half, len(w)):
                    b = f32(b + f32(w[k] * x[k]))
                rows.append(f32(a + b))
            out.append(rows)
        return out
    return op


def width_split_dot(weights, columns_per_split=2):
    """A split-K choice that depends on the batch width (more columns, fewer K parts): the per-column sum order
    changes with the width."""
    def op(columns):
        parts = max(1, 4 // max(1, len(columns) // columns_per_split))
        out = []
        for x in columns:
            rows = []
            for w in weights:
                step = -(-len(w) // parts)
                total = 0.0
                for p in range(parts):
                    acc = 0.0
                    for k in range(p * step, min(len(w), (p + 1) * step)):
                        acc = f32(acc + f32(w[k] * x[k]))
                    total = f32(total + acc)
                rows.append(total)
            out.append(rows)
        return out
    return op


def check(op, columns, widths, filler):
    """Run `op` on every column alone and inside batches of each width (padded with `filler` columns). Returns
    a list of (column index, width) whose results differ bitwise from the one-column result; empty when the
    operator is column invariant for these inputs."""
    differences = []
    for i, column in enumerate(columns):
        alone = op([column])[0]
        for width in widths:
            if width < 1:
                raise ValueError("widths are positive")
            batch = [column] + [filler[(i + j) % len(filler)] for j in range(width - 1)]
            if op(batch)[0] != alone:
                differences.append((i, width))
    return differences
