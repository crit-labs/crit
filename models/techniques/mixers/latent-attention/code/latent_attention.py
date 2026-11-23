"""Multi-head latent attention in its absorbed form: one cached row a position (a latent and a shared rotary key), low-rank queries multiplied into the latent space.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `latent_attention`
and composes the block latent-attention a layout names (README.md)."""
import math
from ..template.blocks import _tensor


class LatentAttention:
    """Multi-head latent attention: every position caches one row, its latent (rank R) and one rotary key (width P)
    that every head shares, and the latent is the value too.

        q      = q_b(rms_norm(q_a(h)))  (or q(h))     per head [N + P]: a part without positions, a rotary part
        c, k   = kv_a(h)                               c = rms_norm of its first R values, k the rest, rotated
        key_h  = [U_h c | k],  value_h = V_h c        kv_b's rows per head: U_h [N][R], V_h [values][R]
        out_h  = softmax(scale * q_h . key_h) values, then o_proj over the heads' outputs

    run in its absorbed form: the query's first part is multiplied into the latent space (U_h^T q_h), the attention
    slot attends the cached rows with latent=R and returns each head's mix of latents, and V_h maps it to the head's
    values; no key or value per head is ever stored. The layers keep no KV in the model's pools: one pool of keys
    only, a row [R + P] a position (`pools`). The per-head maps U_h and V_h are plain matrices (the checkpoint's
    kv_b_proj, 16-bit or FP32), multiplied a token at a time for one-token rows and a chunk at a time for a prompt
    chunk, so a decode row's bits are its own. Rotary pairs are interleaved where the config says so
    (rope_interleave; the spec's `interleave` renames the key), and the scale is (N + P) ** -0.5, times the squared
    YaRN magnitude scale where the config's rope parameters give mscale_all_dim. Config: kv_lora_rank,
    qk_rope_head_dim, qk_nope_head_dim, v_head_dim, q_lora_rank (none: one q_proj)."""

    keeps_kv = False   # its layers' state is the pool below, not a KV pool of the model's

    @classmethod
    def _config(cls, model):
        mla = model.config.mla
        if mla is None or not (mla.qk_rope_head_dim and mla.qk_nope_head_dim and mla.v_head_dim):
            raise ValueError("model-template: latent attention takes the config's kv_lora_rank, qk_rope_head_dim, "
                             "qk_nope_head_dim and v_head_dim")
        return mla

    @classmethod
    def pools(cls, model, spec, layers):
        mla = cls._config(model)
        return [{"key": spec.get("pool") or "latent", "kind": "keys", "layers": list(layers), "heads": 1,
                 "dim": mla.kv_lora_rank + mla.qk_rope_head_dim}]

    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        self.layer, self.pool = layer, spec.get("pool") or "latent"
        mla = self._config(m)
        self.heads, self.rank, self.rotary = m.heads, mla.kv_lora_rank, mla.qk_rope_head_dim
        self.plain, self.values = mla.qk_nope_head_dim, mla.v_head_dim
        if "attention:latent" not in m.parts.facts("attention"):
            raise ValueError(f"model-template: layer {layer} is latent attention and the attention plugin does not "
                             "provide attention:latent")
        self.q_a = self.q_a_norm = self.q_a_bias = None
        if m.matrices.has(prefix + "q_a_proj"):
            self.q_a, self.q_a_bias = m.matrices.load([prefix + "q_a_proj"], "q_a"), m.matrices.bias([prefix + "q_a_proj"])
            self.q_a_norm = _tensor(m, prefix + "q_a_layernorm.weight").float()
            self.q = m.matrices.load([prefix + "q_b_proj"], "q_b")
        else:
            self.q = m.matrices.load([prefix + "q_proj"], "q")
        self.kv_a = m.matrices.load([prefix + "kv_a_proj_with_mqa"], "kv_a")
        self.kv_a_bias = m.matrices.bias([prefix + "kv_a_proj_with_mqa"])
        self.kv_norm = _tensor(m, prefix + "kv_a_layernorm.weight").float()
        if m.matrices.format(prefix + "kv_b_proj") != "bf16":
            raise ValueError(f"model-template: {prefix}kv_b_proj is stored as {m.matrices.format(prefix + 'kv_b_proj')}"
                             "; latent attention multiplies its per-head rows as plain matrices")
        up = _tensor(m, prefix + "kv_b_proj.weight").float()
        if tuple(up.shape) != (self.heads * (self.plain + self.values), self.rank):
            raise ValueError(f"model-template: {prefix}kv_b_proj is {list(up.shape)}, {self.heads} heads of "
                             f"{self.plain} + {self.values} over a latent of {self.rank} take "
                             f"{[self.heads * (self.plain + self.values), self.rank]}")
        up = up.view(self.heads, self.plain + self.values, self.rank)
        self.keys_up, self.values_up = up[:, :self.plain].contiguous(), up[:, self.plain:].contiguous()
        self.o, self.o_bias = m.matrices.load([prefix + "o_proj"], "o"), m.matrices.bias([prefix + "o_proj"])
        self.rope = m.rope_of(self.rotary)
        self.style = "interleaved" if m.setting(spec.get("interleave", "rope_interleave")) else "half"
        self.scale = (self.plain + self.rotary) ** -0.5
        rope = m.config.get("rope_parameters") or m.config.get("rope_scaling") or {}
        if (rope.get("rope_type") or rope.get("type") or "default") != "default" and rope.get("mscale_all_dim"):
            factor = float(rope.get("factor") or 1.0)
            mscale = 0.1 * float(rope["mscale_all_dim"]) * math.log(factor) + 1.0 if factor > 1 else 1.0
            self.scale *= mscale * mscale

    def _biased(self, y, bias, dtype):
        return y if bias is None else (y.float() + bias).to(dtype)

    def _rotated(self, x, ctx):
        return self.m.norms.rope(x.contiguous(), ctx.positions, self.rope.freqs, style=self.style,
                                 scale=self.rope.scale)

    def _each(self, equation, x, weights, ctx):
        """The per-head products of rows x ([tokens][heads][n]) with a matrix a head (FP32): a prompt chunk in one
        call, one-token rows a token at a time, so each such row has the bits it has alone."""
        torch = self.m.torch
        x, found = x.float(), []
        for first, end, prompt in ctx.groups:
            if prompt:
                found.append(torch.einsum(equation, x[first:end], weights))
            else:
                found += [torch.einsum(equation, x[i:i + 1], weights) for i in range(first, end)]
        return found[0] if len(found) == 1 else torch.cat(found)

    def write(self, h, ctx):
        """Each row's latent and rotary key into the layer's pool; nothing read."""
        m, tokens = self.m, h.shape[0]
        row = self._biased(ctx.mm(h, self.kv_a), self.kv_a_bias, h.dtype)
        latent = m._norm(row[:, :self.rank], self.kv_norm)
        key = self._rotated(row[:, None, self.rank:], ctx)
        m.pools[self.pool].write(self.layer, ctx.slots[self.pool],
                                 m.torch.cat([latent.reshape(tokens, 1, self.rank), key], dim=-1).contiguous(), None)

    def forward(self, h, ctx):
        m, tokens = self.m, h.shape[0]
        self.write(h, ctx)
        if self.q_a is not None:
            low = self._biased(ctx.mm(h, self.q_a), self.q_a_bias, h.dtype)
            q = ctx.mm(m._norm(low, self.q_a_norm), self.q)
        else:
            q = ctx.mm(h, self.q)
        q = q.reshape(tokens, self.heads, self.plain + self.rotary)
        absorbed = self._each("thd,hdr->thr", q[..., :self.plain], self.keys_up, ctx).to(h.dtype)
        queries = m.torch.cat([absorbed, self._rotated(q[..., self.plain:], ctx)], dim=-1).contiguous()
        mixed = m.attention.forward(self.layer, queries, m.pools[self.pool], ctx.batch, scale=self.scale,
                                    latent=self.rank)
        out = self._each("thr,hvr->thv", mixed, self.values_up, ctx).to(h.dtype)
        return self._biased(ctx.mm(out.reshape(tokens, -1), self.o), self.o_bias, h.dtype)


BLOCKS = {"latent-attention": LatentAttention}
