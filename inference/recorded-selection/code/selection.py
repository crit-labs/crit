"""Recorded selection among interchangeable implementations.

A choice is keyed by an operation shape and a power-of-two width bucket. Recorded
choices replay exactly; timing happens only when tuning is requested, and the
result is exported as a table that belongs in the frozen configuration. Standard
library only.
"""
import math

RECORDED, TUNED, FALLBACK = "recorded", "tuned", "fallback"


def bucket(width, minimum=1, cap=None):
    """Smallest power of two >= width, at least `minimum`, clipped to `cap`."""
    if not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    if not isinstance(minimum, int) or minimum < 1 or minimum & (minimum - 1):
        raise ValueError("minimum must be a power of two")
    if cap is not None and width > cap:
        raise ValueError("width exceeds the bucket cap")
    b = minimum
    while b < width:
        b *= 2
    return min(b, cap) if cap is not None else b


def key(operation, dims, width_bucket):
    """Canonical key: operation name, shape dimensions and the width bucket."""
    if not operation or ":" in operation or any(not isinstance(d, int) or d < 0 for d in dims):
        raise ValueError("operation name without ':' and non-negative integer dimensions required")
    return "%s:%s:%d" % (operation, "x".join(str(d) for d in dims), width_bucket)


def validate(table):
    """A recorded table maps nonempty string keys to non-negative integer indices."""
    if not isinstance(table, dict):
        raise ValueError("selection table must be an object")
    for k, v in table.items():
        if not isinstance(k, str) or not k or "\\" in k or '"' in k:
            raise ValueError("selection keys must be nonempty strings without quotes or backslashes")
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError("selection indices must be non-negative integers")
    return dict(table)


class Table:
    def __init__(self, recorded=None, tuning=False):
        self.recorded = validate(recorded or {})
        self.tuning = tuning
        self.chosen = {}

    def choose(self, k, count, time=None):
        """Return (index, source). `time(i)` returns a duration or None when candidate
        i cannot run; it is called only when tuning and `k` has no recorded choice.
        Ties keep the lower index; the first choice for a key is reused."""
        if count < 1:
            raise ValueError("selection needs at least one candidate")
        if k in self.chosen:
            return self.chosen[k]
        if k in self.recorded:
            if self.recorded[k] >= count:
                raise ValueError("recorded choice for %s exceeds the candidate count" % k)
            choice = (self.recorded[k], RECORDED)
        elif self.tuning and time is not None:
            choice, best = (0, FALLBACK), math.inf
            for i in range(count):
                t = time(i)
                if t is not None and t < best:
                    choice, best = (i, TUNED), t
        else:
            choice = (0, FALLBACK)
        self.chosen[k] = choice
        return choice

    def export(self):
        """Every choice made so far, sorted by key, for the frozen configuration."""
        return {k: self.chosen[k][0] for k in sorted(self.chosen)}
