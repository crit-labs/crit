"""relative-attention's row operations in float64: the references of its operations (operations.json), which crit's
kernel row-ops (structures head-clip, small-product, sink-gate, scaled-add) is held to. The roundings to BF16 that
the operations name are part of them; everything else is exact arithmetic on the inputs as given."""
import torch


def head_clip(x, threshold, scale=None, tau=None, round_bf16=False):
    """x [t][heads][d] (times scale [heads][d]) shortened per head to `threshold` where its L2 norm + 1e-6 is longer;
    with tau [t], rounded to BF16 first where round_bf16 and times its token's tau."""
    x = x.double() if scale is None else x.double() * scale.double()
    norm = x.square().sum(-1).sqrt()
    y = x * (threshold / (norm + 1e-6)).clamp(max=1.0)[..., None]
    if tau is not None:
        if round_bf16:
            y = y.to(torch.bfloat16).double()
        y = y * tau.double()[:, None, None]
    return y


def small_product(r, proj):
    """r [t][heads][R] rounded to BF16, times proj [R][E]: [t][heads][E] before the output's BF16 rounding."""
    return r.to(torch.bfloat16).double() @ proj.double()


def sink_gate(a, lse, sink, gate):
    """[t][heads * d]: a [t][heads][d] times sigmoid(lse - sink) [t][heads] times sigmoid(gate) [t][heads * d]."""
    t, heads, d = a.shape
    keep = torch.sigmoid(lse.double() - sink.double())[..., None]
    return (a.double() * keep).reshape(t, heads * d) * torch.sigmoid(gate.double())


def scaled_add(y, x, scale):
    """(y + x) * scale."""
    return (y.double() + x.double()) * scale
