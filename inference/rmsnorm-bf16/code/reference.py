"""Host formula reference; neither CUDA accumulation nor BF16 rounding proof."""
import math


def normalize(rows, gain, epsilon, mode="plain", gate=None):
    rows, gain = tuple(tuple(row) for row in rows), tuple(gain)
    if not gain or not rows or any(len(row) != len(gain) for row in rows):
        raise ValueError("nonempty rectangular input/gain required")
    if mode not in {"plain", "unit_offset", "gated"} or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid normalization parameters")
    if (gate is not None) != (mode == "gated"):
        raise ValueError("gate required exactly for gated mode")
    if gate is not None:
        gate = tuple(tuple(row) for row in gate)
        if len(gate) != len(rows) or any(len(row) != len(gain) for row in gate):
            raise ValueError("gate shape differs")
    values = gain + tuple(x for row in rows for x in row) + (() if gate is None else tuple(x for row in gate for x in row))
    if any(not math.isfinite(x) for x in values):
        raise ValueError("finite reference values required")
    output = []
    for index, row in enumerate(rows):
        inverse = 1 / math.sqrt(sum(x*x for x in row) / len(row) + epsilon)
        result = []
        for column, x in enumerate(row):
            value = x * inverse * (gain[column] + (1 if mode == "unit_offset" else 0))
            if gate is not None:
                g = gate[index][column]
                sigmoid = 1 / (1 + math.exp(-g)) if g >= 0 else math.exp(g) / (1 + math.exp(g))
                value *= g * sigmoid
            result.append(value)
        output.append(tuple(result))
    return tuple(output)


def fold_partials(residual, partials):
    """Deterministic split reduction: residual + partials[0] + partials[1] + ...,
    added element by element in split order. A producer split along its reduction
    axis writes one partial per split instead of accumulating atomically; the
    consumer adds them in this fixed order, so the result does not depend on
    which split finished first."""
    residual = tuple(tuple(row) for row in residual)
    partials = tuple(tuple(tuple(row) for row in part) for part in partials)
    if not residual or any(len(part) != len(residual) or any(len(a) != len(b) for a, b in zip(part, residual))
                           for part in partials):
        raise ValueError("partials must match the residual shape")
    folded = []
    for index, row in enumerate(residual):
        out = []
        for column, value in enumerate(row):
            for part in partials:
                value = value + part[index][column]
            out.append(value)
        folded.append(tuple(out))
    return tuple(folded)


def fold_and_normalize(residual, partials, gain, epsilon, mode="plain", gate=None):
    """Fold the partials into the residual, then normalize the updated rows.
    Returns (updated residual, normalized rows): one pass over the residual
    replaces a separate accumulate step before the normalization."""
    updated = fold_partials(residual, partials)
    return updated, normalize(updated, gain, epsilon, mode, gate)
