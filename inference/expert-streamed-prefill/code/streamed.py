"""The schedule on the device: an experts plugin's grouped kernel measured by rows per active expert, the prompt chunk
chosen at the knee of that curve, and a prompt's routed layer run chunk by chunk over the bundles a staging source puts
where the kernel reads them.

Any experts plugin with the experts:bundles device form (forward_bundles over a store [slots][bundle_bytes], a slot
table [experts] and the store's format) runs here, so the routed experts of a prompt chunk may sit in a residency
cache's slots, a staging ring a copy engine fills while the layer before computes, or the plugin's own bundles. A
timing may choose only between bit-identical paths, so Streamed takes the chunk from a curve only when the experts
plugin's facts say a route keeps its bits whatever rows share its call (experts row_invariance prompt_rows, as
grouped-experts' do): the chunk the curve picks then changes no token. For any other kernel the chunk is stated (no
curve, chunks of max_tokens).

    curve = rate_curve(experts, layer, store, slots, "nvfp4", hidden=2048, inter=768, top=8, experts_count=128,
                       tokens=(64, 256, 1024, 4096))   # at load; None for a store it cannot time
    prefill = Streamed(experts, curve, top=8, experts_count=128, max_tokens=8192)
    out = prefill.layer(layer, x, ids, weights, stage, "nvfp4")   # stage(layer, ids) -> (store, slots)

No plugin of the library runs this yet: the plugin that stages a prompt's experts (a residency cache's prompt staging)
measures the curve at its load and runs its prompt layers through Streamed. Measurements are the engine's on the
target: nothing here is interpolated between measured points.
"""
import time

try:
    from .schedule import RateCurve, choose_chunk, rows_per_active_expert
except ImportError:   # loaded as a file of its own, its folder on the path
    from schedule import RateCurve, choose_chunk, rows_per_active_expert


def _routes(torch, tokens, top, experts, generator, device):
    """Uniform routes: each token's `top` distinct experts."""
    scores = torch.rand(tokens, experts, generator=generator)
    return scores.topk(top, dim=-1).indices.to(device)


def _timer(torch, device):
    """A wall clock of `device`'s work: CUDA events on a CUDA device, the host clock on the CPU (its work runs in the
    call); None for a device it cannot time."""
    if device.type == "cuda":
        def cuda(call):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) * 1e-3
        return cuda
    if device.type == "cpu":
        def host(call):
            start = time.perf_counter()
            call()
            return time.perf_counter() - start
        return host
    return None


def rate_curve(experts, layer, store, slots, format, hidden, inter, top, experts_count, tokens, repeats=3,
               device=None):
    """The grouped kernel's measured rate (TFLOPS) by rows per active expert, the measure choose_chunk compares with the
    knee: one call of each chunk size in `tokens` over uniform routes, the median of `repeats` timed calls after a
    first one, keyed by rows_per_active_expert(count, top, experts_count) rounded down (at least 1). None when the
    store sits where nothing here can time it (Streamed then runs one chunk of its largest size)."""
    import torch
    device = torch.device(device or store.device)
    timer = _timer(torch, device)
    if timer is None:
        return None
    generator = torch.Generator().manual_seed(0)
    points = {}
    for count in tokens:
        x = torch.randn(count, hidden, generator=generator).to(device, torch.bfloat16)
        ids = _routes(torch, count, top, experts_count, generator, device)
        weights = torch.rand(count, top, generator=generator).to(device)

        def call():
            experts.forward_bundles(layer, x, ids, weights, store, slots, format)

        call()
        seconds = sorted(timer(call) for _ in range(repeats))[repeats // 2]
        rows = max(1, int(rows_per_active_expert(count, top, experts_count)))
        flops = 2.0 * count * top * 3 * hidden * inter
        points[rows] = max(points.get(rows, 0.0), flops / seconds / 1e12)
    return RateCurve(points)


def row_invariant(experts):
    """Whether the experts plugin's facts say a route's bits are the same whatever prompt rows share its call
    (facts()["experts"]["row_invariance"]["prompt_rows"]), so a chunk size changes no token."""
    facts = experts.facts() if callable(getattr(experts, "facts", None)) else {}
    return bool(((facts.get("experts") or {}).get("row_invariance") or {}).get("prompt_rows"))


class Streamed:
    """A prompt's routed layers in chunks of the size the curve's knee gives (choose_chunk), each chunk's experts
    staged by the engine's source; without a curve, chunks of max_tokens (whole granules), the stated chunk. A curve is
    refused for an experts plugin that is not row_invariant: its chunk would be a timing between paths of other bits."""

    def __init__(self, experts, curve, top, experts_count, max_tokens, fraction=0.9, granularity=256):
        self.experts = experts
        if curve is None:
            if max_tokens < granularity or granularity < 1:
                raise ValueError("max_tokens of at least one granule")
            self.chunk = max_tokens - max_tokens % granularity
        elif not row_invariant(experts):
            raise ValueError("a chunk timed from a rate curve needs an experts plugin whose facts say experts "
                             "row_invariance prompt_rows (a route's bits whatever rows share its call): state the "
                             "chunk instead (curve None, max_tokens the chunk) or use a row-invariant experts kernel "
                             "such as grouped-experts")
        else:
            self.chunk = choose_chunk(curve, top, experts_count, max_tokens, fraction, granularity)

    def layer(self, layer, x, ids, weights, stage, format):
        """out [tokens][hidden] FP32 of one routed layer of a prompt: chunk by chunk, stage(layer, ids of the chunk)
        -> (store, slots) puts the chunk's routed experts where the kernel reads them (a residency cache's prompt(),
        a staging ring), then the experts plugin's forward_bundles runs the chunk over that store of `format`
        bundles."""
        import torch
        tokens = x.shape[0]
        out = torch.empty(tokens, x.shape[1], dtype=torch.float32, device=x.device)
        for first in range(0, tokens, self.chunk):
            last = min(tokens, first + self.chunk)
            store, slots = stage(layer, ids[first:last])
            out[first:last] = self.experts.forward_bundles(layer, x[first:last], ids[first:last], weights[first:last],
                                                           store, slots, format)
        return out
