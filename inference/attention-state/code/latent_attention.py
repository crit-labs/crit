"""Latent attention (multi-head latent attention in its absorbed form): every query head attends one cached row a
position, the row's latent (rank R) and its rotary key (width P) shared by every head, and the latent is the value
too. The reference of the operation `latent-attention` (operations.json), in float64.

    out[t, h] = sum_n p[t, h, n] * latent[n]          p = softmax over the positions token t sees of
    s[t, h, n] = scale * (q_latent[t, h] . latent[n] + q_rope[t, h] . rope[n])

with, where given, a sink logit a head as one more position that carries no value. q_latent is the query's
non-rotary part already multiplied into the latent space (the key up-projection absorbed); out is in the latent space
(the value up-projection follows, outside the attention). Requires torch.
"""


def latent_attention(q_latent, q_rope, latent, rope, seen, scale, sinks=None):
    """q_latent [T][H][R], q_rope [T][H][P] over latent [N][R], rope [N][P], seen [T][N] bool (the positions each
    token sees) -> (out [T][H][R], lse [T][H]) in float64; sinks [H] (or None) one more score a head with no value."""
    import torch
    ql, qr = q_latent.double(), q_rope.double()
    c, k = latent.double(), rope.double()
    s = (torch.einsum("thr,nr->thn", ql, c) + torch.einsum("thp,np->thn", qr, k)) * scale
    s = s.masked_fill(~seen[:, None, :], float("-inf"))
    if sinks is not None:
        sink = sinks.double().view(1, -1, 1).expand(s.shape[0], -1, 1)
        s = torch.cat([s, sink], dim=-1)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None])
    if sinks is not None:
        p = p[..., :-1]
    return torch.einsum("thn,nr->thr", p, c), lse
