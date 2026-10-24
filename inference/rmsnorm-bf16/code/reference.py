"""Host formula reference; no proof of CUDA accumulation. rounding=None keeps real
arithmetic; "single" and "normalize_first" round to BF16 where the kernel does."""
import math

ROUNDINGS = (None, "single", "normalize_first")
BF16_MAX = float.fromhex("0x1.fep127")


def bf16(value):
    """Round a finite float to the nearest BF16 value, ties to even."""
    if value == 0 or not math.isfinite(value):
        return value
    _, exponent = math.frexp(value)
    quantum = math.ldexp(1.0, max(exponent, -125) - 8)  # 8 significant bits; subnormals below 2^-126
    result = round(value / quantum) * quantum
    return result if abs(result) <= BF16_MAX else math.copysign(math.inf, value)


def normalize(rows, gain, epsilon, mode="plain", gate=None, rounding=None):
    """rounding "single" rounds x * r * gain (times the gate's SiLU) once, as Gemma does;
    "normalize_first" rounds x * r, then its product with the gain and then the gated
    product, as the HF Llama-family norms do in the input dtype."""
    rows, gain = tuple(tuple(row) for row in rows), tuple(gain)
    if not gain or not rows or any(len(row) != len(gain) for row in rows):
        raise ValueError("nonempty rectangular input/gain required")
    if mode not in {"plain", "unit_offset", "gated"} or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid normalization parameters")
    if rounding not in ROUNDINGS:
        raise ValueError("unsupported rounding")
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
            scale = gain[column] + (1 if mode == "unit_offset" else 0)
            value = bf16(bf16(x * inverse) * scale) if rounding == "normalize_first" else x * inverse * scale
            if gate is not None:
                g = gate[index][column]
                sigmoid = 1 / (1 + math.exp(-g)) if g >= 0 else math.exp(g) / (1 + math.exp(g))
                value *= g * sigmoid
            result.append(value if rounding is None else bf16(value))
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


def fold_and_normalize(residual, partials, gain, epsilon, mode="plain", gate=None, rounding=None):
    """Fold the partials into the residual, then normalize the updated rows.
    Returns (updated residual, normalized rows): one pass over the residual
    replaces a separate accumulate step before the normalization."""
    updated = fold_partials(residual, partials)
    return updated, normalize(updated, gain, epsilon, mode, gate, rounding)
