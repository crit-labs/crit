"""Chunk size and layer schedule for reading a prompt through routed experts kept off the device.

When most experts live in host memory, a prompt chunk pays twice per routed layer: the
arithmetic, and the bytes of every touched expert that is not resident, over the link.
A small chunk re-sends the same experts for every chunk; a large one sends each expert
once per chunk and gives the grouped projection enough rows per expert to run near
its rate. The schedule therefore picks the smallest chunk whose rows per active expert
reach the knee of the projection's measured rate curve, and overlaps each step's
expert copies with the step before it through two staging buffers.

A step is one routed layer of one chunk, in execution order (chunk 0 layers 0..L-1,
then chunk 1, ...). Step i + 1's copies run while step i computes, into the buffer
step i - 1 released, so a step starts when its own copies are complete and the step
before it is done. Whole-layer copies need no routing prediction: once a chunk touches
nearly every expert of a layer (fetch_whole_layer), the schedule copies the layer's
whole non-resident set before its router runs.

Rates, compute times and bytes are the engine's measurements on the target; nothing
here is interpolated between measured points.
"""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class RateCurve:
    """The grouped projection's measured rate (TFLOPS, or any rate where more is
    faster) by rows per expert, on the target device and the engine's shapes."""
    points: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.points:
            raise ValueError("a measured rate curve needs points")
        for rows, rate in self.points.items():
            if type(rows) is not int or rows < 1 or type(rate) not in (int, float) or not rate > 0:
                raise ValueError("positive integer rows per expert and positive rates required")

    def knee(self, fraction=0.9):
        """The fewest measured rows per expert whose rate reaches `fraction` of the best measured rate."""
        if not 0 < fraction <= 1:
            raise ValueError("fraction in (0, 1]")
        best = max(self.points.values())
        return min(rows for rows, rate in self.points.items() if rate >= fraction * best)


def expected_touched(tokens, top_k, experts):
    """Distinct experts a chunk's tokens touch in one layer, under uniform routing."""
    if tokens < 0 or not 0 < top_k <= experts:
        raise ValueError("tokens >= 0 and 0 < top_k <= experts")
    return experts * (1.0 - (1.0 - top_k / experts) ** tokens)


def rows_per_active_expert(tokens, top_k, experts):
    touched = expected_touched(tokens, top_k, experts)
    return tokens * top_k / touched if touched else 0.0


def choose_chunk(curve, top_k, experts, max_tokens, fraction=0.9, granularity=256):
    """The smallest chunk (a multiple of `granularity`, at most `max_tokens`, the
    activation workspace's bound) whose rows per active expert reach the curve's knee;
    `max_tokens` when none does."""
    if max_tokens < granularity or granularity < 1:
        raise ValueError("max_tokens of at least one granule")
    target = curve.knee(fraction)
    for tokens in range(granularity, max_tokens + 1, granularity):
        if rows_per_active_expert(tokens, top_k, experts) >= target:
            return tokens
    return max_tokens - max_tokens % granularity


def fetch_whole_layer(tokens, top_k, experts, threshold=0.9):
    """Whether a chunk touches at least `threshold` of a layer's experts, so copying the
    layer's whole non-resident set ahead of its router wastes little."""
    return expected_touched(tokens, top_k, experts) >= threshold * experts


@dataclass(frozen=True)
class Step:
    compute_ms: float  # the step's arithmetic, measured at the chunk's size
    fetch_bytes: int   # non-resident expert bytes it reads, copied into its staging buffer


def timeline(steps, link_gbs):
    """(total ms, [(fetch end, compute start, compute end)] per step) with two staging
    buffers: step i's copies start once step i - 1's copies are done and step i - 2,
    the buffer's last user, has finished computing."""
    if not link_gbs > 0:
        raise ValueError("a positive link rate")
    out, fetch_end, compute_end = [], 0.0, []
    for i, s in enumerate(steps):
        if s.compute_ms < 0 or s.fetch_bytes < 0:
            raise ValueError("nonnegative compute and bytes")
        free = compute_end[i - 2] if i >= 2 else 0.0
        start = max(fetch_end, free)
        fetch_end = start + s.fetch_bytes / (link_gbs * 1e6)  # bytes over GB/s, in ms
        begin = max(fetch_end, compute_end[-1] if compute_end else 0.0)
        compute_end.append(begin + s.compute_ms)
        out.append((fetch_end, begin, compute_end[-1]))
    return (compute_end[-1] if compute_end else 0.0), out


def prefill_estimate(prompt_tokens, chunk, layers, compute_ms, nonresident_bytes, link_gbs):
    """Prompt tok/s and ms for `prompt_tokens` read in chunks of `chunk`: compute_ms(tokens)
    and nonresident_bytes(tokens) give one routed layer's measured time and copied bytes at a
    chunk of that many tokens. Returns (tok/s, ms, link-bound ms): the last is the time the
    copies alone take, the floor this schedule cannot beat."""
    if prompt_tokens < 1 or chunk < 1 or layers < 1:
        raise ValueError("positive prompt, chunk and layer counts")
    steps = []
    for start in range(0, prompt_tokens, chunk):
        n = min(chunk, prompt_tokens - start)
        steps += [Step(compute_ms(n), nonresident_bytes(n))] * layers
    total, _ = timeline(steps, link_gbs)
    link = sum(s.fetch_bytes for s in steps) / (link_gbs * 1e6)
    return prompt_tokens / (total / 1e3), total, link


def staging_bytes(expert_bytes, tokens, top_k, experts, resident_share=0.0):
    """Bytes of one staging buffer for a chunk: the expected non-resident touched experts
    of a layer (or the whole non-resident layer when fetch_whole_layer). Two are needed."""
    touched = experts if fetch_whole_layer(tokens, top_k, experts) else expected_touched(tokens, top_k, experts)
    return math.ceil(touched * (1.0 - resident_share)) * expert_bytes
