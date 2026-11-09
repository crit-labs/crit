"""Core's sampling law over logits rows on a device, in torch: the sampler slot's device form (crit_plugins.device).

The same rules as sampling.py's mirror: candidates rank by (value descending, index ascending), greedy takes the
first, a sampled row keeps exp((value - best) / temperature) weights through top_k, top_p and min_p, and Core's draw
u = uniform(seed, position, stream) picks the first kept token whose cumulative weight exceeds u times the kept
weight. Weights are FP64 on the device as in the mirror; the draw is computed on the host (Philox4x32-10, one per
row), so a draw within rounding of a boundary may differ from Core's FP64 walk, as sampling.py's own note says.

All rows of a step are ranked together: the greedy rows in one argmax, the sampled rows in one sort, each row's kept
prefix a length (top_k, then top_p's cut, then min_p) over that sort, so a step's tokens cost a few kernels whatever
its row count, and the host reads the device once (a row without a selectable token is refused).
"""
from . import sampling


def _checked(p):
    top_p, top_k, min_p = float(p.get("top_p", 1.0)), int(p.get("top_k", 0)), float(p.get("min_p", 0.0))
    if not 0 < top_p <= 1 or top_k < 0 or not 0 <= min_p <= 1:
        raise ValueError("top_p in (0, 1], top_k non-negative and min_p in [0, 1]")
    return top_p, top_k, min_p


def _masked(logits, found):
    """The logits with each constrained row's disallowed tokens at -inf (the logits themselves without any)."""
    import torch
    rows = [r for r, p in enumerate(found) if p.get("allowed") is not None]
    if not rows:
        return logits
    values = logits.clone()
    for r in rows:
        allowed = found[r]["allowed"]
        if hasattr(allowed, "mask"):
            keep = allowed.mask(values.device, values.shape[-1])
        else:
            keep = torch.zeros(values.shape[-1], dtype=torch.bool, device=values.device)
            keep[torch.tensor(sorted(allowed), dtype=torch.int64, device=values.device)] = True
        values[r] = values[r].masked_fill(~keep, float("-inf"))
    return values


def _rows(tensor, rows, count):
    """tensor's rows `rows` (a slice when they are all of them)."""
    import torch
    if len(rows) == count:
        return tensor
    return tensor.index_select(0, torch.tensor(rows, dtype=torch.int64, device=tensor.device))


def sample(logits, params):
    """One token per row of logits ([rows][vocabulary]) as a [rows] int64 tensor on its device; params one dict per
    row (or one for every row): temperature, top_p, top_k, min_p, seed, position, stream, allowed (a constrained row's
    crit_plugins.Allowed, masked on the device, or sorted ids)."""
    import torch
    rows = logits.shape[0]
    found = params if isinstance(params, list) else [params] * rows
    if len(found) != rows:
        raise ValueError("params is one dict for every row or a list of one dict per row")
    values = _masked(logits, found)
    device = logits.device
    out = torch.empty(rows, dtype=torch.int64, device=device)
    temperatures = [float(p.get("temperature", 0.0)) for p in found]
    greedy = [r for r, t in enumerate(temperatures) if not t > 0]
    sampled = [r for r, t in enumerate(temperatures) if t > 0]
    if greedy:
        # The first of equal maxima: the lower id (exact in the logits' own precision, as in FP64).
        chosen = torch.argmax(_rows(values, greedy, rows), dim=-1)
        if len(greedy) == rows:
            return chosen
        out.index_copy_(0, torch.tensor(greedy, dtype=torch.int64, device=device), chosen)
    if not sampled:
        return out
    rules = {r: _checked(found[r]) for r in sampled}
    # The rank in the logits' own precision (the FP64 values order the same, being exact copies), weights in FP64.
    v = _rows(values, sampled, rows)
    if v.dtype == torch.float64 or v.dtype.itemsize > 4:
        v = v.double()
    elif v.dtype != torch.float32:
        v = v.float()
    ranked, order = torch.sort(v, dim=-1, descending=True, stable=True)
    if all(0 < rules[r][1] for r in sampled):
        # Every row keeps at most its top_k: the rest of the rank is never read.
        width = max(rules[r][1] for r in sampled)
        ranked, order = ranked[:, :width], order[:, :width]
    ranked = ranked.double()
    finite = torch.isfinite(ranked).sum(-1)
    if bool((finite == 0).any()):
        raise ValueError("no selectable token")
    # Every sampled row's temperature, top_k, top_p, min_p and draw in one copy to the device.
    table = torch.tensor([[temperatures[r], rules[r][1], rules[r][0], rules[r][2],
                           sampling.uniform(int(found[r].get("seed", 0)), int(found[r].get("position", 0)),
                                            int(found[r].get("stream", 0)))] for r in sampled],
                         dtype=torch.float64).to(device, non_blocking=True)
    temperature, top_k, top_p, min_p, u = (table[:, c:c + 1] for c in range(5))
    # Tokens at -inf weigh exp(-inf) = 0, so a row keeps its finite tokens without a mask.
    weights = torch.exp((ranked - ranked[:, :1]) / temperature)
    length = finite
    position = None

    def kept_prefix(weights, length):
        return torch.where(position < length[:, None], weights, torch.zeros_like(weights))

    if any(rules[r][1] > 0 for r in sampled):
        position = torch.arange(ranked.shape[-1], device=device)[None, :]
        k = top_k[:, 0].to(torch.int64)
        length = torch.where((k > 0) & (k < length), k, length)
        weights = kept_prefix(weights, length)
    if any(rules[r][0] < 1 for r in sampled):
        position = torch.arange(ranked.shape[-1], device=device)[None, :] if position is None else position
        cumulative = torch.cumsum(weights, -1)
        total = cumulative.gather(1, (length - 1)[:, None])
        cut = torch.minimum(torch.searchsorted(cumulative, top_p * total)[:, 0], length - 1) + 1
        length = torch.where(top_p[:, 0] < 1, cut, length)
        weights = kept_prefix(weights, length)
    if any(rules[r][2] > 0 for r in sampled):
        position = torch.arange(ranked.shape[-1], device=device)[None, :] if position is None else position
        kept = ((weights >= min_p) & (position < length[:, None])).sum(-1)   # weights fall along the rank: a prefix
        length = torch.where(min_p[:, 0] > 0, kept, length)
        weights = kept_prefix(weights, length)
    cumulative = torch.cumsum(weights, -1)
    total = cumulative.gather(1, (length - 1)[:, None])
    pick = torch.minimum(torch.searchsorted(cumulative, u * total, right=True), (length - 1)[:, None])
    chosen = order.gather(1, pick)[:, 0]
    if len(sampled) == rows:
        return chosen
    out.index_copy_(0, torch.tensor(sampled, dtype=torch.int64, device=device), chosen)
    return out
