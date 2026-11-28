"""Full-attention layers whose values are their keys' own rows (the config's attention_k_eq_v), held once a position:
latent rows through the attention slot's latent attention.

Such a layer stores no v_proj: its values are the key projection's rows r through the value norm (an RMSNorm without
a weight), v = r / rms(r), and its keys the same rows through the key norm and the layer's partial rotation,
k = rope(v * w_k) (w_k the key norm's weight, plus the layout's norm offset). The keys' unrotated dimensions are the
values' times w_k, and only the rotated ones (the dimensions the layer's rotary table turns) carry anything else. So
each KV head keeps one row a position, [v | k_rot] (the head's D values, then its rotated key dimensions), read
through latent attention (the attention slot's attention:latent, rank R = D):

    query row  [q * w_k on the unrotated dimensions, 0 on the rotated ones | q's rotated dimensions]
    score      q_lat . v + q_rot . k_rot  =  q . k                (the layer's own scores, at its own scale)
    out        the mix of the rows' first R values: the mix of v  (the layer's own output)

one latent call a KV head over its query heads (grouped queries: head h reads KV head h // (heads / kv_heads)). A
position takes kv_heads x (D + rotated) elements where attention keeps keys and values apart, kv_heads x 2D.

The layout names it for the layers that share keys and values with `"otherwise": "attention"`: where the config does
not set attention_k_eq_v, the attention plugin provides no attention:latent (the split-KV reader), or its latent rows
are of a shape the slot's latent readers do not take on the device (a rank or rotary width not a power of two), the
layer runs the library's attention, which takes its values from the key projection's rows itself (serves()). Its layers keep no
KV in the model's pools (keeps_kv False): one pool of keys only (`pools`, kind keys, value_dim 0, at the model's KV
page type, fp8 pages by a scale a row since a static key scale does not scale values) of one head a row, a pseudo
layer a KV head past the model's own layer numbers. A drafter that reads this layer's keys and values calls read().

A model technique of the gemma4-model package (models/cache/README.md, a family's own pieces): its manifest mounts
this code as its source `shared_kv_attention`, the block shared-kv-attention."""
from ..core.attention.attention import Attention

POOL = "kv-once"


def _latent_width(width, most):
    """Whether a latent row's rank (most 1024) or rotary width (most 256) is one the attention slot's latent readers
    take on the device (crit's latent-attention kernel, eager and captured): a power of two from 16."""
    return 16 <= width <= most and not width & (width - 1)


class SharedKVAttention(Attention):
    keeps_kv = False

    @classmethod
    def serves(cls, shape, spec):
        """None where the layer's keys and values are one row (the config's attention_k_eq_v) and the attention plugin
        reads latent rows of their shape; else why not (the layout's `otherwise` block serves it)."""
        if not shape.setting("attention_k_eq_v"):
            return "the config keeps keys and values apart (attention_k_eq_v)"
        if "attention:latent" not in shape.slot_facts("attention"):
            return "the attention plugin provides no attention:latent"
        layer = shape.kinds.index(spec["kind"])
        _, head_dim = shape.kv_shape(layer, spec)
        rotated = cls._rotated(shape, layer, head_dim)
        if not (_latent_width(head_dim, 1024) and _latent_width(rotated, 256)):
            return (f"its latent rows (rank {head_dim}, rotary width {rotated}) are not a shape the attention slot's "
                    "latent readers take on the device: powers of two, rank 16 to 1024, rotary width 16 to 256")
        return None

    @staticmethod
    def _rotated(model, layer, head_dim):
        """The dimensions the layer's rotary table turns: twice its nonzero inverse frequencies (a proportional table
        keeps zeros for the pairs it leaves unturned)."""
        from ..core.rope.rope_table import rope_table
        kind = model.types[layer] if layer < len(model.types) else None
        rope = next((e.rope for e in model.config.rope_by_layer_type if e.layer_type == kind), None)
        table = rope_table(model.config.rope if rope is None else rope, head_dim, model.config.max_positions)
        return 2 * sum(1 for f in table.inv_freq if f > 0)

    @staticmethod
    def _ids(model, layer, kv_heads):
        """The pool's pseudo layers of a layer: one a KV head, past the model's layers and its draft layer."""
        base = model.layers + 1
        return [base + layer * kv_heads + g for g in range(kv_heads)]

    @staticmethod
    def _dtype(model):
        return "fp8:row" if model.kv_dtype == "fp8" else model.kv_dtype

    @classmethod
    def pools(cls, model, spec, layers):
        kv_heads, head_dim = model.kv_shape(layers[0], spec)
        rotated = cls._rotated(model, layers[0], head_dim)
        return [{"key": spec.get("pool") or POOL, "kind": "keys", "heads": 1, "dim": head_dim + rotated,
                 "dtype": cls._dtype(model),
                 "layers": [i for layer in layers for i in cls._ids(model, layer, kv_heads)]}]

    def __init__(self, model, layer, prefix, spec):
        super().__init__(model, layer, prefix, spec)
        m, torch = self.m, model.torch
        if not self.shared_values or self.value_norm is None or self.k_norm is None or self.window or self.gate:
            raise ValueError(f"model-template: shared-kv-attention takes a full layer whose values are its key "
                             f"projection's rows through a value norm, with a key norm (layer {layer})")
        if "attention:latent" not in m.parts.facts("attention"):
            raise ValueError("model-template: shared-kv-attention reads latent rows: the attention plugin provides no "
                             "attention:latent")
        self.pool = spec.get("pool") or POOL
        self.ids = self._ids(m, layer, self.kv_heads)
        self.group = self.heads // self.kv_heads
        # The rotated dimensions, as the norms slot's rotation turns them: a row turned at position 1 against the same
        # row at 0, under the table's pairs at one radian each (the table's own low frequencies turn a BF16 row by less
        # than its rounding at position 1, which hid them: a partial table over 96- or 512-wide heads), of a row of
        # ones (a random row loses a dimension whose turn rounds back to it)
        D, rope = self.head_dim, self.rope
        probe = torch.ones(2, 1, D, device=m.device, dtype=torch.bfloat16)   # (1, 1) turns to (cos-sin, sin+cos)
        ones = torch.ones(D, device=m.device, dtype=torch.float32)
        at = torch.tensor([0, 1], dtype=torch.int64, device=m.device)
        turned = m.norms.rms_norm_rope(probe, ones, m.eps, at, (rope.freqs > 0).float(),
                                       offset=0.0, scale=rope.scale).float()
        self.rot = torch.nonzero((turned[0, 0] - turned[1, 0]).abs() > 0).flatten()
        if self.rot.numel() != self._rotated(m, layer, D):
            raise ValueError(f"model-template: layer {layer}'s rotation turns {self.rot.numel()} dimensions, its table "
                             f"{self._rotated(m, layer, D)}")
        plain = torch.ones(D, dtype=torch.bool, device=m.device)
        plain[self.rot] = False
        # The query's latent part: q times the key norm's weight on the unrotated dimensions, nothing on the rotated
        self.q_weight = torch.where(plain, self.k_norm + m.norm_offset, torch.zeros_like(self.k_norm))

    def _rows(self, k, v):
        """[tokens][kv_heads][D + rotated]: each KV head's values, then its rotated key dimensions."""
        return self.m.torch.cat([v, k.index_select(-1, self.rot)], dim=-1).contiguous()

    def write(self, h, ctx):
        q, k, v, gate = self.qkv_rows(h, ctx)
        rows = self._rows(k, v.to(k.dtype))
        pool, slots = self.m.pools[self.pool], ctx.slots[self.pool]
        for g, i in enumerate(self.ids):
            pool.write(i, slots, rows[:, g:g + 1].contiguous(), None)
        return q, gate

    def read(self, q, batch):
        """The layer's attention of queries q [tokens][heads][D] (normed and rotated as the layer's own) over the rows
        the batch's sequences hold: [tokens][heads][D]."""
        torch, m = self.m.torch, self.m
        lat = (q.float() * self.q_weight).to(q.dtype)
        queries = torch.cat([lat, q.index_select(-1, self.rot)], dim=-1)
        pool = m.pools[self.pool]
        if "attention:latent-layers" in m.parts.facts("attention"):   # every KV head's pseudo layer in one call
            return m.attention.forward_latent_layers(self.ids, queries, pool, batch, self.head_dim, scale=self.scale)
        outs = [m.attention.forward(i, queries[:, g * self.group:(g + 1) * self.group].contiguous(), pool, batch,
                                    scale=self.scale, latent=self.head_dim)
                for g, i in enumerate(self.ids)]
        return torch.cat(outs, dim=1)

    def forward(self, h, ctx):
        q, gate = self.write(h, ctx)
        t, dtype = h.shape[0], h.dtype
        a = self.read(q, ctx.batch)
        y = ctx.mm(a.reshape(t, -1).to(dtype), self.o)
        if self.o_bias is not None:
            y = (y.float() + self.o_bias).to(dtype)
        return y


BLOCKS = {"shared-kv-attention": SharedKVAttention}
