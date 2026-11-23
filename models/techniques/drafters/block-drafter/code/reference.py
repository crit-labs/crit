"""block-drafter's span attention in float64: the reference of its operation (operations.json), which crit's kernel
row-ops (structure span-attention) is held to."""
import torch


def span_attention(q, k, v, first, last):
    """[rows][kv heads][G][D]: each row's queries q [rows][kv heads][G][D] over its keys and values k, v
    [rows][kv heads][T][D] at positions first[r] .. last[r], every query seeing all of them, scaled by D^-1/2."""
    t = k.shape[2]
    at = torch.arange(t, device=q.device)[None, :]
    keep = (at >= first.long()[:, None]) & (at <= last.long()[:, None])                  # [rows][T]
    scores = torch.einsum("rkgd,rktd->rkgt", q.double(), k.double()) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~keep[:, None, None, :], float("-inf"))
    return torch.einsum("rkgt,rktd->rkgd", torch.softmax(scores, dim=-1), v.double())
