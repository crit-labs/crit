"""block-attention's row operations in float64: the references of its operations (operations.json), which crit's
kernel row-ops (structures summary-mix, pair-norm, gated-join, low-rank-gates) is held to. The roundings to BF16 that
the operations name are part of them; everything else is exact arithmetic on the inputs as given."""
import torch

EPS = 1e-6


def _rms(x):
    x = x.double()
    return x / torch.sqrt(x.square().mean(-1, keepdim=True) + EPS)


def summary_mix(blocks, h, query):
    """[t][Q][H]: per token, softmax over its N summaries blocks [t][N][H] and its row h [t][H] (all normed without a
    weight) of query . rms(v) / sqrt(H), for each query [Q][H], and the weighted sum of the rms(v)."""
    stacked = _rms(torch.cat([blocks.double(), h.double()[:, None, :]], dim=1))          # [t][N + 1][H]
    scores = torch.einsum("qh,tnh->tqn", query.double(), stacked) * stacked.shape[-1] ** -0.5
    return torch.einsum("tqn,tnh->tqh", torch.softmax(scores, dim=-1), stacked)


def pair_norm(x, message):
    """[t][2H]: [rms(x), rms(message)] before the output's BF16 rounding."""
    return torch.cat([_rms(x), _rms(message)], dim=-1)


def gated_join(x, message, f=None, a=None, down=None, up=None):
    """(1 - tanh(f)) * x + tanh(a) * message. Without f and a, from the low-rank gates: x and message [t][G][H],
    g = bf16(down_i . bf16([rms(x), rms(message)])) (down [G][rank][2H]), [f; a] = up_i . g (up [G][2H][rank])."""
    if f is None:
        joined = torch.cat([_rms(x), _rms(message)], dim=-1).to(torch.bfloat16).double()  # [t][G][2H]
        g = torch.einsum("grk,tgk->tgr", down.double(), joined).to(torch.bfloat16).double()
        fa = torch.einsum("gkr,tgr->tgk", up.double(), g)
        hidden = x.shape[-1]
        f, a = fa[..., :hidden], fa[..., hidden:]
    return (1.0 - torch.tanh(f.double())) * x.double() + torch.tanh(a.double()) * message.double()
