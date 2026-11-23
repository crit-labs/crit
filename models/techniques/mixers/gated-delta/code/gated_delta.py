"""Gated delta-rule linear attention through the recurrence slot, either projection layout a checkpoint stores.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `gated_delta`
and composes the block gated-delta a layout names (README.md)."""
from ..template.blocks import _tensor


class GatedDelta:
    """A gated delta-rule layer (linear attention) over the recurrence slot: q, k and v through the slot's causal
    depthwise convolution, the decay -exp(A_log) * softplus(a + dt_bias) and beta = sigmoid(b) from the norms slot, the
    delta rule in the recurrence slot's pool, the gated RMSNorm over SiLU(z) (or, with the spec's `norm_gate` naming a
    config key, that key's activation: silu or sigmoid), the output projection. The checkpoint's projections are read
    in either layout it stores: one grouped in_proj_qkvz and in_proj_ba (per key head: its q, k, then its value heads'
    v and z; its value heads' b, then a) or separate in_proj_qkv, in_proj_z, in_proj_b and in_proj_a."""

    pool = "states"

    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        self.layer = layer
        linear = m.config.linear
        if linear is None:
            raise ValueError("model-template: a gated-delta layer needs the config's linear_* sizes")
        self.nk, self.nv = linear.num_key_heads, linear.num_value_heads
        self.dk, self.dv = linear.key_head_dim, linear.value_head_dim
        self.kernel = linear.conv_kernel_dim
        if self.nv % self.nk:
            raise ValueError(f"model-template: {self.nv} value heads over {self.nk} key heads")
        self.grouped = m.matrices.has(prefix + "in_proj_qkvz")
        # Each the fewest loaded matrices whose rows side by side are the projection's (a checkpoint may store the
        # halves in two formats: a GGUF's in_proj_qkv Q5_K beside its in_proj_z Q6_K)
        if self.grouped:
            self.qkvz = m.matrices.together([prefix + "in_proj_qkvz"], "qkvz")
            self.ba = m.matrices.together([prefix + "in_proj_ba"], "ba")
        else:
            self.qkvz = m.matrices.together([prefix + "in_proj_qkv", prefix + "in_proj_z"], "qkvz")
            self.ba = m.matrices.together([prefix + "in_proj_b", prefix + "in_proj_a"], "ba")
        self.out = m.matrices.load([prefix + "out_proj"], "linear_out")
        self.conv = _tensor(m, prefix + "conv1d.weight")[:, 0].float().contiguous()
        self.rate = _tensor(m, prefix + "A_log").float().exp()
        self.dt_bias = _tensor(m, prefix + "dt_bias").float()
        self.norm = _tensor(m, prefix + "norm.weight").float()
        self.channels = 2 * self.nk * self.dk + self.nv * self.dv
        self.norm_gate = (m.setting(spec["norm_gate"]) if spec.get("norm_gate") else None) or "silu"
        if self.norm_gate not in ("silu", "sigmoid"):
            raise ValueError(f"model-template: the gated norm's activation is {self.norm_gate} (silu or sigmoid)")

    @staticmethod
    def geometry(m):
        linear = m.config.linear
        return {"keys": linear.key_head_dim, "values": linear.value_head_dim, "epsilon": 1e-6,
                "query_scale": linear.key_head_dim ** -0.5, "normalize_keys": True, "normalize_queries": True}

    def forward(self, h, ctx):
        torch, m, t = self.m.torch, self.m, h.shape[0]
        nk, nv, dk, dv = self.nk, self.nv, self.dk, self.dv
        r = nv // nk
        qkvz, ba = (torch.cat([ctx.mm(h, w) for w in found], -1) if len(found) > 1 else ctx.mm(h, found[0])
                    for found in (self.qkvz, self.ba))
        if self.grouped:
            q, k, v, z = torch.split(qkvz.view(t, nk, 2 * dk + 2 * dv * r), [dk, dk, r * dv, r * dv], dim=-1)
            qkv = torch.cat([q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)], -1)
            beta, a = torch.split(ba.view(t, nk, 2 * r), [r, r], dim=-1)
            beta, a = beta.reshape(t, nv), a.reshape(t, nv)
        else:
            width = 2 * nk * dk + nv * dv
            qkv, z = qkvz[:, :width].contiguous(), qkvz[:, width:]
            beta, a = ba[:, :nv], ba[:, nv:]
        states = m.pools["states"]
        mixed = m.recurrence.convolve(self.layer, qkv, self.conv, None, states, ctx.batch)
        q, k, v = torch.split(mixed, [nk * dk, nk * dk, nv * dv], dim=-1)
        q, k, v = q.reshape(t, nk, dk), k.reshape(t, nk, dk), v.reshape(t, nv, dv)   # value head h reads key head h // r
        decay, beta = m.norms.softplus_decay(a.contiguous(), self.dt_bias, self.rate, beta.contiguous())
        out = m.recurrence.forward(self.layer, q, k, v, decay, beta, states, ctx.batch, m.geometry)
        gated = m.norms.gated_rms_norm(out.reshape(t, nv, dv), z.reshape(t, nv, dv), self.norm, m.eps,
                                       activation=self.norm_gate, out_dtype=h.dtype)
        return ctx.mm(gated.reshape(t, -1), self.out)


BLOCKS = {"gated-delta": GatedDelta}
