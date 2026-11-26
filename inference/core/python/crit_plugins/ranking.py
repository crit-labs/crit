"""The k highest values of each row in Core's order, at a top-k's cost.

Core ranks a row's alternatives (a token's log-probabilities, a request's top_logprobs) highest first, equal values
by the lower index first: a stable descending sort. Sorting a whole vocabulary row for its first twenty entries costs
its values, an int64 order and the sort's own workspace, about 24 bytes an entry: for 128 rows of a 248,320-token
vocabulary 0.76 GB, the largest temporary of a long score. top_ranked() gives the same first k entries, bit for bit,
from a top-k and a sort of the few entries at or above each row's k-th value.

    from crit_plugins.ranking import top_ranked
    ranked, order = top_ranked(logp, 20)   # == logp.sort(dim=-1, descending=True, stable=True)[...][:, :20]

It reads the device (how many entries tie at a row's k-th value), so it runs outside a capture; a captured path
sorts the row.
"""


def top_ranked(values, k):
    """The first k entries of each row of `values` ([rows, n], or [n]) in a stable descending sort: (their values,
    their indices), [rows, k] each ([k] for a row). Equal values keep the lower index first, as the sort does; a row
    with a NaN, which the sort places first, is sorted whole."""
    import torch
    row = values.dim() == 1
    if row:
        values = values[None]
    n = values.shape[-1]
    k = max(0, min(int(k), n))
    if k == 0:
        ranked, order = values[:, :0], torch.empty(values.shape[0], 0, dtype=torch.int64, device=values.device)
        return (ranked[0], order[0]) if row else (ranked, order)
    if k == n:
        ranked, order = values.sort(dim=-1, descending=True, stable=True)
        ranked, order = ranked[:, :k], order[:, :k]
        return (ranked[0], order[0]) if row else (ranked, order)
    rows = values.shape[0]
    ranked = values.new_empty(rows, k)
    order = torch.empty(rows, k, dtype=torch.int64, device=values.device)
    # The k + 1 highest: a row has exactly k candidates where its (k + 1)-th value is below its k-th. No temporary
    # wider than the rows' masks (a count of them would be int64 a value: twice an FP32 row's bytes).
    top = values.topk(k + 1, dim=-1, sorted=True).values
    threshold = top[:, k - 1:k]
    at = values >= threshold                  # every entry the first k can hold: those past the k-th value and its ties
    bad = torch.isnan(values).any(-1) | (top[:, k] == top[:, k - 1])
    plain = (~bad).nonzero()[:, 0]
    if plain.numel():                         # exactly k candidates: in index order, then sorted stably
        whole = plain.numel() == rows         # every row: its masks as they are; never a copy of the rows' values
        index = (at if whole else at[plain]).nonzero()[:, 1].view(-1, k)
        found = values[plain[:, None], index]
        found, by = found.sort(dim=-1, descending=True, stable=True)
        ranked[plain], order[plain] = found, index.gather(1, by)
    for r in bad.nonzero()[:, 0].tolist():   # ties at the k-th value (or a NaN): the row's candidates, or the row
        if torch.isnan(values[r]).any():
            found, index = values[r].sort(descending=True, stable=True)
        else:
            index = at[r].nonzero()[:, 0]
            found, by = values[r, index].sort(descending=True, stable=True)
            index = index[by]
        ranked[r], order[r] = found[:k], index[:k]
    return (ranked[0], order[0]) if row else (ranked, order)
