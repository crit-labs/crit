"""The CPU/link split of a routed layer's misses, from times measured on the target machine.

With m misses in a plan, k of them are copied over the host link into cache slots (and
admitted, so the cache keeps adapting) and m - k are computed on the CPU from host
memory, at the same time. Both paths read the same host memory, so their combined time
is measured, not taken as the slower of the two alone. quota_table picks k per miss
count; the planner reads it as Directory::admit_by_misses (ClockDirectory admit_by_misses),
admits the first k misses in route order and bypasses the rest to the CPU.

The engine supplies the measurements (calibrate takes its timing callables): ms for one
routed layer of a decode step with k link misses alone, j CPU misses alone, and k and j
together. Unmeasured points are never interpolated: a miss count without a measured
split keeps the planner's default (-1, every miss admitted). Measure again when the hit
rate moves away from the one the table was measured at (needs_recalibration): fewer
misses per layer change which split is fastest.
"""
from dataclasses import dataclass, field
import math
from statistics import median

MAX_MISSES = 32  # routes per plan (device_cache.h kMaxRoutes)


def _ms(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative milliseconds required")
    return float(value)


@dataclass(frozen=True)
class SplitTimings:
    """Median ms per routed layer: link[k] with k link misses alone, cpu[j] with j CPU
    misses alone, both[(k, j)] with both at once. hit_rate: the cache's hit rate while
    measuring."""
    link: dict = field(default_factory=dict)
    cpu: dict = field(default_factory=dict)
    both: dict = field(default_factory=dict)
    hit_rate: float | None = None

    def __post_init__(self):
        for table in (self.link, self.cpu):
            for n, t in table.items():
                if type(n) is not int or not 0 <= n <= MAX_MISSES:
                    raise ValueError("miss counts from 0 to 32")
                _ms(t)
        for (k, j), t in self.both.items():
            if type(k) is not int or type(j) is not int or k < 1 or j < 1 or k + j > MAX_MISSES:
                raise ValueError("combined points need both paths and at most 32 misses")
            _ms(t)

    def layer_ms(self, link, cpu):
        """Measured time of a layer with `link` link and `cpu` CPU misses, or None."""
        if link and cpu:
            return self.both.get((link, cpu))
        return self.link.get(link) if not cpu else self.cpu.get(cpu)


def best_split(timings, misses):
    """(link misses, ms) of the fastest measured split of `misses`, or None. Equal times
    take more link misses, which keeps the cache adapting."""
    best = None
    for k in range(misses, -1, -1):
        t = timings.layer_ms(k, misses - k)
        if t is not None and (best is None or t < best[1]):
            best = (k, t)
    return best


def quota_table(timings):
    """Link misses per miss count 0-32, -1 where no split was measured (every miss
    admitted): the Directory::admit_by_misses array."""
    table = []
    for m in range(MAX_MISSES + 1):
        best = best_split(timings, m)
        table.append(-1 if best is None else best[0])
    return table


def calibrate(measure_link, measure_cpu, measure_both, misses, repeats=5, hit_rate=None):
    """SplitTimings from the engine's timing callables (each returns ms for one routed
    layer), for every split of each miss count in `misses`, `repeats` times each."""
    if repeats < 1:
        raise ValueError("at least one repetition")
    link, cpu, both = {0: 0.0}, {0: 0.0}, {}

    def timed(call, *args):
        return median(_ms(call(*args)) for _ in range(repeats))

    for m in misses:
        if not 0 < m <= MAX_MISSES:
            raise ValueError("miss counts from 1 to 32")
        for k in range(m + 1):
            j = m - k
            if k and k not in link:
                link[k] = timed(measure_link, k)
            if j and j not in cpu:
                cpu[j] = timed(measure_cpu, j)
            if k and j and (k, j) not in both:
                both[(k, j)] = timed(measure_both, k, j)
    return SplitTimings(link, cpu, both, hit_rate)


def needs_recalibration(timings, observed_hit_rate, tolerance=0.05):
    """Whether the hit rate moved more than `tolerance` from the one the table was measured at."""
    if timings.hit_rate is None:
        return True
    return abs(observed_hit_rate - timings.hit_rate) > tolerance
