"""Sparse index selection: each query token scores every compressed key it may see, the sum over its index heads of
the head's weight times the ReLU of the head's query-key product, and keeps its k best keys for a sparse attention to
read. The reference of the operation `sparse-index` (operations.json), in float64.

    score[t, s] = sum_h weights[t, h] * scale * max(0, q[t, h] . keys[s])     for s < ends[t]
    picked[t] = the positions of the k largest score[t, s], ascending (all of them where ends[t] <= k), then -1

with ties broken toward the lower position and a NaN ranked below every number. Requires torch.
"""


def sparse_scores(q, keys, weights, ends, scale):
    """q [T][H][D], keys [N][D], weights [T][H], ends [T] -> score [T][N] float64, -inf at or past ends[t]."""
    import torch
    s = torch.einsum("thd,nd->thn", q.double(), keys.double()).clamp_min(0.0)
    out = (s * (weights.double() * scale)[..., None]).sum(1)
    n = torch.arange(keys.shape[0], device=keys.device)
    return out.masked_fill(n[None, :] >= ends.long()[:, None], float("-inf"))


def sparse_index(score, ends, k):
    """score [T][N] -> picked int64 [T][k]: each token's k best positions below ends[t], ascending, then -1."""
    import torch
    t, n = score.shape
    picked = torch.full((t, k), -1, dtype=torch.int64, device=score.device)
    for i in range(t):
        end = max(0, min(int(ends[i]), n))
        row = score[i, :end].double()
        if end <= k:
            chosen = torch.arange(end, device=score.device)
        else:
            # the larger score first, then the lower position (stable sorts), every NaN after every number
            order = torch.sort(-row.nan_to_num(nan=float("-inf")), stable=True).indices
            order = order[torch.sort(row[order].isnan().to(torch.int8), stable=True).indices][:k]
            chosen = torch.sort(order).values
        picked[i, :chosen.numel()] = chosen
    return picked
