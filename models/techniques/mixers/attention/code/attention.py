"""Grouped-query softmax attention: a stacked q, k, v projection, biases, query/key norms, rotary positions (per layer type, partial rotation), sliding windows, sinks and a sigmoid output gate, through the attention slot.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `attention`
and composes the block attention a layout names (README.md)."""
from ..template.blocks import _tensor


class Attention:
    """Softmax attention over the kv_cache slot's paged pool, grouped query heads: one stacked q|k|v matrix (biases
    where the checkpoint stores them), per-head query and key RMSNorms where it stores them, rotary positions (the
    config's rope table, its layer type's where the config keys its rope by layer type; partial rotation), a sliding
    window (the layer kind sliding_attention: the config's sliding_window, or the spec's `window` key), attention
    sinks where the checkpoint stores `sinks`, and with the spec's `output_gate: sigmoid` a gated output (each head's
    query and gate from one projection, the output times sigmoid of the gate). Spec keys for families whose layer
    kinds differ in shape: `head_dim` and `kv_heads` name the config keys of this kind's head width and key/value
    heads (a config's own per-layer entry first, the model's by default), `scale` is the attention's scale where it is not head_dim ** -0.5, and
    `value_norm: true` norms each value head (an RMSNorm without a weight). A layer that stores no v_proj takes its
    values from the key projection's rows, before the keys' norm and rotation."""

    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        self.layer = layer
        self.heads = m.heads
        self.kv_heads, self.head_dim = m.kv_shape(layer, spec)
        self.gate = spec.get("output_gate")
        if self.gate not in (None, "sigmoid"):
            raise ValueError(f"model-template: attention output_gate {self.gate!r} (sigmoid or none)")
        names = [prefix + n for n in spec.get("matrices", ("q_proj", "k_proj", "v_proj"))]
        self.shared_values = len(names) == 3 and not m.matrices.has(names[2])   # values: the key projection's rows
        if self.shared_values:
            names = names[:2]
        self.value_norm = None
        if spec.get("value_norm"):
            self.value_norm = m.torch.ones(self.head_dim, dtype=m.torch.float32, device=m.device)
        width = self.heads * self.head_dim * (2 if self.gate else 1)
        if m.matrices.rows(names[0]) != width:
            raise ValueError(f"model-template: {names[0]} has {m.matrices.rows(names[0])} rows, the layout's attention "
                             f"takes {width} ({self.heads} heads of {self.head_dim}" + (", query and gate" if self.gate
                                                                                         else "") + ")")
        self.qkv = m.matrices.load(names, "qkv")
        self.qkv_bias = m.matrices.bias(names)
        out = prefix + spec.get("output", "o_proj")
        self.o, self.o_bias = m.matrices.load([out], "o"), m.matrices.bias([out])
        self.q_norm = _tensor(m, prefix + "q_norm.weight", False)
        self.k_norm = _tensor(m, prefix + "k_norm.weight", False)
        if (self.q_norm is None) != (self.k_norm is None):
            raise ValueError(f"model-template: {prefix} stores a norm for queries or keys but not both")
        self.q_norm = None if self.q_norm is None else self.q_norm.float()
        self.k_norm = None if self.k_norm is None else self.k_norm.float()
        sinks = _tensor(m, prefix + "sinks", False)
        self.sinks = None if sinks is None else sinks.float()
        if self.sinks is not None and "attention:sinks" not in m.parts.facts("attention"):
            raise ValueError(f"model-template: {prefix} has attention sinks and the attention plugin does not provide "
                             "attention:sinks")
        window = spec.get("window")
        if window is None and spec.get("kind") == "sliding_attention":
            window = m.config.sliding_window
        self.window = int(window or 0)
        if self.window and "attention:sliding" not in m.parts.facts("attention"):
            raise ValueError(f"model-template: layer {layer} attends a sliding window and the attention plugin does not "
                             "provide attention:sliding")
        self.pool = spec.get("pool") or (("kv", self.window) if self.window else "kv")
        self.scale = float(spec["scale"]) if spec.get("scale") is not None else self.head_dim ** -0.5
        self.rope = m.rope_for(layer, self.head_dim)
        if m.kv_scales and self.pool != "mtp":   # the draft layer's pool keeps mtp_kv pages, which take no static scales
            self.kv_scales = tuple(float(_tensor(m, prefix + n).float()) for n in ("k_scale", "v_scale"))

    def qkv_rows(self, h, ctx, positions=None):
        """The layer's normed, rotated queries and keys, its values and (gated) the output gate."""
        torch, m, t = self.m.torch, self.m, h.shape[0]
        positions = ctx.positions if positions is None else positions
        qkv = ctx.mm(h, self.qkv)
        if self.qkv_bias is not None:
            qkv = (qkv.float() + self.qkv_bias).to(h.dtype)
        width = self.heads * self.head_dim * (2 if self.gate else 1)
        kv = self.kv_heads * self.head_dim
        gate = None
        if self.gate:
            q, gate = torch.chunk(qkv[:, :width].reshape(t, self.heads, 2 * self.head_dim), 2, dim=-1)
        else:
            q = qkv[:, :width].reshape(t, self.heads, self.head_dim)
        k = qkv[:, width:width + kv].reshape(t, self.kv_heads, self.head_dim)
        v = k if self.shared_values else qkv[:, width + kv:].reshape(t, self.kv_heads, self.head_dim)
        norms, rope = m.norms, self.rope
        if self.value_norm is not None:
            v = norms.rms_norm(v.contiguous(), self.value_norm, m.eps)
        elif self.shared_values:
            v = v.clone()   # the keys are normed and rotated next: the values keep the projection's rows
        if self.q_norm is not None:
            if m.fused_norms:
                q, k = norms.rms_norm_rope_pair(q, k, self.q_norm, self.k_norm, m.eps, positions, rope.freqs,
                                                offset=m.norm_offset, scale=rope.scale)
            else:
                q = norms.rms_norm_rope(q, self.q_norm, m.eps, positions, rope.freqs, offset=m.norm_offset,
                                        scale=rope.scale)
                k = norms.rms_norm_rope(k, self.k_norm, m.eps, positions, rope.freqs, offset=m.norm_offset,
                                        scale=rope.scale)
        else:
            q = norms.rope(q.contiguous(), positions, rope.freqs, scale=rope.scale)
            k = norms.rope(k.contiguous(), positions, rope.freqs, scale=rope.scale)
        return q, k, v.contiguous(), gate

    def attend(self, q, gate, pool, batch, ctx, t, dtype, **extra):
        if self.window:
            extra["sliding"] = self.window
        if self.sinks is not None:
            extra["sinks"] = self.sinks
        a = self.m.attention.forward(self.layer, q, pool, batch, scale=self.scale, **extra)
        if gate is not None:
            a = self.m.norms.sigmoid_mul(a, gate, out_dtype=dtype)       # each head's output times sigmoid(its gate)
        y = ctx.mm(a.reshape(t, -1).to(dtype), self.o)
        if self.o_bias is not None:
            y = (y.float() + self.o_bias).to(dtype)
        return y

    def write(self, h, ctx):
        """The rows' keys and values into the layer's pool, and nothing read: (q, gate) for a forward to attend with.
        A draft layer's prompt positions are written so (their attention is never read)."""
        q, k, v, gate = self.qkv_rows(h, ctx)
        self.m.pools[self.pool].write(self.layer, ctx.slots[self.pool], k, v)
        return q, gate

    def forward(self, h, ctx):
        q, gate = self.write(h, ctx)
        return self.attend(q, gate, self.m.pools[self.pool], ctx.batch, ctx, h.shape[0], h.dtype)


BLOCKS = {"attention": Attention}
