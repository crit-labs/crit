"""The blocks a composed model is made of: each the arithmetic of one kind of layer part over the other slots' plugins.

A block is made once per layer, `Block(model, layer, prefix, spec)`: `model` is the ComposedModel (its matrices,
norms, pools and the slots' plugins), `prefix` the layer's tensor prefix for the block ("model.layers.3.self_attn."),
`spec` the layout's entry for it. A mixer block (attention, gated-delta) has `pool`, the key of the pool its layer
state lives in ("kv", ("kv", window), "states"), and `forward(h, ctx)`; an FFN block (swiglu, moe) has
`forward(h, ctx)`. ctx is the step's Context: the batch, positions, row groups and each pool's slots. Every matmul goes
through ctx.mm (RowGroups: exact rows), every norm, rotary position and gated product through the norms slot, so a row's
bits never depend on the rows beside it.

A block reads what varies between checkpoints from the checkpoint and its config (biases, query/key norms, the
recurrence's projection layout, shared experts, the router's rule) and what a family fixes from the layout (the norm
offset, an attention output gate). An engine adds a kind of block by passing its class to compose() under a name its
layout uses (README: adding a block).

A block that keeps state of its own beside its layer's pool declares it before anything is made, so the model plans
its bytes with every other tensor: the class method `pools(model, spec, layers)` gives [{key, kind, layers, ...}] for
the layers that run it, kind "keys" (a pool of keys only, a row a position: heads, dim) or "window" (the last inputs
of a causal depthwise convolution: channels, kernel, dilation), which the model makes and follows through steps,
rounds and the prompt cache as it does its KV and state pools; `tables(model, prefix, spec)` gives {plan name: bytes}
for lookup tables it reads rows of, which may lie off the device. A `before` block (the layout's `before`) has
`forward(residual, ctx)` and returns the residual with its rows added."""
import math


class Context:
    """One forward's step: batch (StepBatch, RoundBatch or PromptBatch), positions, row groups, the slots each pool
    reserved for the batch's tokens; ids, the tokens' ids where the forward has them (a draft layer's rows have
    none), and windows, each token's ids with the ones before it (model.py, _windows), for the blocks that read them;
    shared, what the blocks of one forward compute once for all of them (a step's block lists are every indexed
    layer's)."""

    def __init__(self, model, batch, groups, slots, ids=None, windows=None):
        self.model, self.batch, self.groups, self.slots = model, batch, groups, slots
        self.ids, self.windows, self.shared = ids, windows, {}
        self.residual = None   # the stream a layer's FFN output is added to (a residual that gives it: sandwich)
        self.positions = batch.positions_t
        self.prompt = bool(getattr(batch, "prompt", False)) or (not batch.round and any(p for _, _, p in groups))

    def mm(self, x, matrix, out_dtype=None):
        return self.model.rows.matmul(x, matrix, self.groups, out_dtype)

    def mm_each(self, x, matrix, out_dtype=None):
        """The matmul of x [tokens][n][in], n rows a token (its residual streams): [tokens][n][out], a prompt chunk's
        rows in one prompt call and one-token rows in calls of at most the invariant rows, as mm's."""
        tokens, n = x.shape[0], x.shape[1]
        step, groups = self.model.rows.decode_rows, []
        for first, end, prompt in self.groups:
            if prompt:
                groups.append((first * n, end * n, True))
            else:
                groups += [(at, min(at + step, end * n), False) for at in range(first * n, end * n, step)]
        return self.model.rows.matmul(x.reshape(tokens * n, -1), matrix, groups, out_dtype).view(tokens, n, -1)


def row_sum(x):
    """The sum of each last-dimension row of x in one fixed order whatever the row count: adjacent pairs added, then
    the pairs' sums, an odd value joining the next level. Elementwise adds only, where torch's own sum over a row
    takes another order in a call of other rows (the norms slot's reductions are the norms and the stream sums; this
    is a plain dot product's)."""
    import torch
    while x.shape[-1] > 1:
        n = x.shape[-1]
        pairs = x[..., 0:n - n % 2:2] + x[..., 1:n:2]
        x = torch.cat([pairs, x[..., n - 1:]], dim=-1) if n % 2 else pairs
    return x[..., 0]


def _tensor(model, name, required=True):
    if name in model.ck:
        return model.ck.get(name, model.device)
    if required:
        raise KeyError(f"model-template: the checkpoint has no {name}")
    return None


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


INDEXER = {"heads": "indexer_n_heads", "kv_heads": "indexer_kv_heads", "dim": "indexer_head_dim",
           "budget": "indexer_budget", "block": "indexer_compress_ratio"}


class _BlockPlan:
    """One step's block bookkeeping, shared by its indexed layers (they keep one pool, so one set of slots): the
    tokens that end a block (`at`: their indices, None for every token), the slots of each such block's earlier
    positions (`previous` [n][block - 1], oldest first), the position it starts at (`starts`), the slot its key is
    written at (`target`: the token's own; -1 where a round's token ends none), and what selection reads: per row of
    a step (first token, end, blocks complete after it, the slots of its blocks' keys), or for a round `keys` [rows]
    [blocks] and `ends` [tokens]."""
    __slots__ = ("at", "previous", "starts", "target", "rows", "keys", "ends")


class IndexedAttention(Attention):
    """Block-sparse softmax attention: an indexer picks the KV blocks each token reads.

    The sequence's positions lie in blocks of `block` positions from position 0. A token at position p reads the
    complete blocks an indexer picks for it, up to budget / block of them, and the block it lies in (its tail), so a
    token with no more complete blocks than that reads every position up to its own: plain causal attention below
    the budget. The indexer projects each token's row to index queries (several heads) and one raw index key; a
    complete block's key is the mean of its positions' raw keys, normed and rotated at the block's first position; a
    token scores a block by the sum over its index heads of relu(query . key) / sqrt(dim), queries normed and rotated
    at the token's position, and keeps its highest-scoring blocks (crit's canonical choice: ties to the lower block).

    Keys: the layers keep one pool of keys only, a row a position (`pools`): a position's raw key, and at a block's
    last position, once the block is complete, the block's key in its place (the raw key of a block's last position
    is read by nothing else). So the pool follows a sequence exactly as its KV does, through steps, rounds, forks and
    truncations, and a block's key is rewritten whenever its last position is computed again.

    Selection runs on crit's kernel sparse-indexer where it loads and takes the indexer's shape (the keys read in
    place through their slots), else in torch a token at a time, so a token's blocks never depend on the tokens
    beside it. A prompt step's tokens are scored in groups whose scores fit `score_bytes` (a token's scores are one
    FP32 a block, and the choice works on a wider copy of them: a 4,096-token step over 50,000 blocks would hold
    gigabytes at once), which changes no choice: each token's is its own. Where no sequence the engine serves can pass the budget (its KV pool holds no more blocks) the block is
    its base: no index keys, no block lists. Otherwise every token's attention reads a block list, below the budget
    too, so a row's arithmetic is one whatever shares its step. Config keys (the spec's `config` renames them):
    indexer_n_heads, indexer_kv_heads (1), indexer_head_dim, indexer_budget, indexer_compress_ratio; the spec's
    `indexer` is the indexer's tensor prefix under the attention's (indexer.)."""

    score_bytes = 64 << 20   # the most a group of a step's tokens holds in scores and their sort keys at once

    @classmethod
    def _config(cls, model, spec):
        names = {**INDEXER, **(spec.get("config") or {})}
        found = {key: int(model.setting(name) or 0) for key, name in names.items()}
        if min(found.values()) < 1 or found["kv_heads"] != 1 or found["budget"] % found["block"]:
            raise ValueError("model-template: an indexed attention takes the config's " + ", ".join(
                f"{name}={found[key]}" for key, name in names.items()) + ": positive, one index key head, the budget "
                "a multiple of the block")
        found["sparse"] = model.kv_tokens // found["block"] > found["budget"] // found["block"]
        return found

    @classmethod
    def pools(cls, model, spec, layers):
        found = cls._config(model, spec)
        if not found["sparse"]:
            return []
        key = "index" if not spec.get("pool") else f"{spec['pool']}-index"
        return [{"key": key, "kind": "keys", "layers": list(layers), "heads": 1, "dim": found["dim"]}]

    def __init__(self, model, layer, prefix, spec):
        super().__init__(model, layer, prefix, spec)
        m = self.m
        if self.window:
            raise ValueError("model-template: an indexed attention reads block lists, never a sliding window")
        found = self._config(m, spec)
        self.sparse, self.block, self.top = found["sparse"], found["block"], found["budget"] // found["block"]
        self.index_heads, self.index_dim = found["heads"], found["dim"]
        self.captures_prompts = not self.sparse   # a chunk's blocks are planned from its sequence's page table
        if not self.sparse:
            return
        for slot, fact in (("attention", "attention:blocks"), ("kv_cache", "kv:keys")):
            if fact not in m.parts.facts(slot):
                raise ValueError(f"model-template: layer {layer} attends the blocks an indexer picks, and the "
                                 f"{slot} plugin does not provide {fact}")
        ip = prefix + spec.get("indexer", "indexer.")
        if m.matrices.rows(ip + "index_qk_proj") != (self.index_heads + 1) * self.index_dim:
            raise ValueError(f"model-template: {ip}index_qk_proj has {m.matrices.rows(ip + 'index_qk_proj')} rows, "
                             f"{self.index_heads} index heads and a key of {self.index_dim} take "
                             f"{(self.index_heads + 1) * self.index_dim}")
        self.index_qk = m.matrices.load([ip + "index_qk_proj"], "index_qk")
        self.index_q_norm = _tensor(m, ip + "q_layernorm.weight").float()
        self.index_k_norm = _tensor(m, ip + "k_layernorm.weight").float()
        self.index = "index" if self.pool == "kv" else f"{self.pool}-index"
        self.index_scale = self.index_dim ** -0.5
        self.kernel = m.kernel("sparse-indexer") if self._kernel_shape() else None

    def _kernel_shape(self):
        """Whether crit's kernel sparse-indexer takes this indexer: a power of two of heads up to 128 of 64, 128 or
        256 values, on a CUDA device (its keys are the pool's own BF16 rows)."""
        heads = self.index_heads
        return (self.m.device.type == "cuda" and heads <= 128 and heads & (heads - 1) == 0
                and self.index_dim in (64, 128, 256))

    # -- the step's blocks ---------------------------------------------------------------------------------------------
    def _plan(self, ctx):
        key = ("blocks", self.index, self.block, self.top)
        plan = ctx.shared.get(key)
        if plan is None:
            plan = ctx.shared[key] = (self._round_plan if ctx.batch.round else self._step_plan)(ctx)
        return plan

    def _step_plan(self, ctx):
        m, torch, batch, size = self.m, self.m.torch, ctx.batch, self.block
        pool, plan = m.pools[self.index], _BlockPlan()
        at, previous, starts, plan.rows = [], [], [], []
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            table, page = pool.table(sequence), pool.page
            done = (position + count) // size   # the blocks complete once the row's tokens are in
            for b in range(position // size, done):
                last = b * size + size - 1
                at.append(first + last - position)
                previous.append([table[q // page] * page + q % page for q in range(b * size, last)])
                starts.append(b * size)
            keys = None
            if done > self.top:   # past the budget: the row's tokens choose among its blocks' keys
                pages = torch.tensor(table, dtype=torch.int64, device=m.device)
                where = torch.arange(done, dtype=torch.int64, device=m.device) * size + (size - 1)
                keys = pages[where // page] * page + where % page
            plan.rows.append((first, first + count, done, keys))
        long = dict(dtype=torch.int64, device=m.device)
        plan.at = torch.tensor(at, **long)
        plan.previous = torch.tensor(previous, **long).view(len(at), size - 1)
        plan.starts = torch.tensor(starts, **long)
        plan.target = ctx.slots[self.index].index_select(0, plan.at)
        plan.keys = plan.ends = None
        return plan

    def _round_plan(self, ctx):
        """A round's tokens at fixed shapes, on the device alone: every token is treated as the end of the block of
        positions up to it, and writes that block's key only where it does end one."""
        m, torch, batch, size = self.m, self.m.torch, ctx.batch, self.block
        pool, plan = m.pools[self.index], _BlockPlan()
        p, tokens, rows = ctx.positions, ctx.positions.shape[0], batch.rows
        slot = batch.slots_t.index_select(0, batch.rows_t)
        back = torch.arange(size - 1, 0, -1, dtype=torch.int64, device=m.device)
        plan.at = None
        plan.previous = pool.slots_at(slot[:, None].expand(tokens, size - 1), p[:, None] - back[None, :])
        plan.starts = (p - (size - 1)).clamp(min=0)
        plan.target = torch.where((p + 1) % size == 0, ctx.slots[self.index], -1)
        blocks = m.kv_tokens // size
        where = torch.arange(blocks, dtype=torch.int64, device=m.device) * size + (size - 1)
        plan.keys = pool.slots_at(batch.slots_t[:rows, None].expand(rows, blocks), where[None, :].expand(rows, blocks))
        plan.ends = ((p + 1) // size).to(torch.int32)
        plan.rows = None
        return plan

    def _index_rows(self, h, ctx):
        """(queries, raw): the indexer's queries [tokens][index heads][dim], normed and rotated at their positions,
        and the tokens' raw index keys [tokens][dim]."""
        m, t = self.m, h.shape[0]
        qk = ctx.mm(h, self.index_qk)
        width = self.index_heads * self.index_dim
        q = m.norms.rms_norm_rope(qk[:, :width].reshape(t, self.index_heads, self.index_dim), self.index_q_norm,
                                  m.eps, ctx.positions, m.rope.freqs, offset=m.norm_offset, scale=m.rope.scale)
        return q, qk[:, width:]

    def _write_index(self, raw, ctx):
        """The step's raw index keys at their positions, then the keys of the blocks its tokens complete at the
        blocks' last positions: the mean of the block's raw keys in position order (the earlier ones read back from
        the pool), normed and rotated at the block's first position."""
        m, torch = self.m, self.m.torch
        pool, plan, size, dim = m.pools[self.index], self._plan(ctx), self.block, self.index_dim
        pool.write(self.layer, ctx.slots[self.index], raw.reshape(-1, 1, dim).contiguous(), None)
        n = plan.target.shape[0]
        if not n:
            return
        earlier, _ = pool.rows_at(self.layer, plan.previous)
        own = raw if plan.at is None else raw.index_select(0, plan.at)
        group = torch.cat([earlier.view(n, size - 1, dim).to(raw.dtype), own[:, None, :]], dim=1)
        pooled = m.norms.stream_sum(group, scale=1.0 / size, out_dtype=raw.dtype)
        key = m.norms.rms_norm_rope(pooled[:, None, :], self.index_k_norm, m.eps, plan.starts, m.rope.freqs,
                                    offset=m.norm_offset, scale=m.rope.scale)
        pool.write(self.layer, plan.target, key.contiguous(), None)

    # -- selection ---------------------------------------------------------------------------------------------------
    def _tail(self, picked, positions):
        """The attention's block list of each token, [tokens][top + 1] int32: its chosen blocks (ascending, -1 past
        them) and the block it lies in where that one is not complete, in block order, -1 last."""
        torch, size = self.m.torch, self.block
        tail = torch.where((positions + 1) % size != 0, positions // size, -1)
        both = torch.cat([picked.long(), tail[:, None]], dim=1)
        big = torch.iinfo(torch.int64).max
        both = torch.where(both < 0, big, both).sort(dim=-1).values
        return torch.where(both == big, -1, both).to(torch.int32)

    def _scores(self, q, keys):
        """FP32 [tokens][blocks]: each token's scores of `keys` ([blocks][dim], or [rows][blocks][dim] for a round's
        rows of equal token counts), a token at a time and its heads summed in order, so a token's scores are its
        own."""
        torch = self.m.torch
        tokens = q.shape[0]
        run = tokens // keys.shape[0] if keys.dim() == 3 else 0
        q, keys = q.float(), keys.float()
        out = []
        for t in range(tokens):
            found = torch.relu(q[t] @ (keys[t // run] if run else keys).transpose(0, 1))
            total = found[0]
            for head in range(1, found.shape[0]):
                total = total + found[head]
            out.append(total * self.index_scale)
        return torch.stack(out)

    def _select(self, q, ctx):
        """The block list of every token of the step ([tokens][top + 1] int32)."""
        from crit_plugins.device import topk_blocks
        m, torch, size = self.m, self.m.torch, self.block
        pool, plan, p = m.pools[self.index], self._plan(ctx), ctx.positions
        store = pool.layer_pages(self.layer)[0].view(-1, self.index_dim) if self.kernel is not None else None
        if ctx.batch.round:
            if self.kernel is not None:
                ones = torch.ones(q.shape[0], self.index_heads, dtype=torch.float32, device=q.device)
                scores = self.kernel.scores(q.contiguous(), store, ones, plan.ends, slots=plan.keys)
                return self._tail(self.kernel.topk(scores, plan.ends, self.top), p)
            keys, _ = pool.rows_at(self.layer, plan.keys)
            return topk_blocks(self._scores(q, keys.view(*plan.keys.shape, self.index_dim)), self.top, p, size)
        found = []
        for first, end, done, slots in plan.rows:
            positions = p[first:end]
            if slots is None:   # no more blocks than the budget: every complete block, whatever its score
                zeros = torch.zeros(end - first, max(done, 1), dtype=torch.float32, device=q.device)
                found.append(topk_blocks(zeros, self.top, positions, size))
            elif self.kernel is not None:
                ones = torch.ones(end - first, self.index_heads, dtype=torch.float32, device=q.device)
                ends = ((positions + 1) // size).to(torch.int32)
                picked = self.kernel.select(q[first:end].contiguous(), store, ones, ends, self.top, slots=slots,
                                            workspace_bytes=self.score_bytes)
                found.append(self._tail(picked, positions))
            else:
                keys, _ = pool.rows_at(self.layer, slots)
                keys = keys.view(done, self.index_dim)
                # a token's scores and the choice's keys of them are 12 bytes a block: tokens in groups within the
                # bound, each token's choice its own whatever its group
                group = max(1, self.score_bytes // (12 * done))
                for at in range(first, end, group):
                    stop = min(at + group, end)
                    found.append(topk_blocks(self._scores(q[at:stop], keys), self.top, p[at:stop], size))
        return found[0] if len(found) == 1 else torch.cat(found)

    def write(self, h, ctx):
        q, gate = super().write(h, ctx)
        if self.sparse:
            self._write_index(self._index_rows(h, ctx)[1], ctx)
        return q, gate

    def forward(self, h, ctx):
        if not self.sparse:
            return super().forward(h, ctx)
        q, gate = super().write(h, ctx)
        queries, raw = self._index_rows(h, ctx)
        self._write_index(raw, ctx)
        return self.attend(q, gate, self.m.pools[self.pool], ctx.batch, ctx, h.shape[0], h.dtype,
                           blocks=self._select(queries, ctx), block_size=self.block)


NGRAM = {"size": "ngram_size", "heads": "heads_per_ngram", "base": "ngram_vocab_size_base", "dim": "ple_embed_dim",
         "kernel": "ple_conv_kernel_size", "seed": "seed", "layers": "ple_layer_ids"}
_MASK64 = (1 << 64) - 1
_GAMMA, _M1, _M2, _PRIME = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 10007


def _splitmix64(value):
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _M1) & _MASK64
    value = ((value ^ (value >> 27)) * _M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def ngram_hash(vocabulary, size, heads, base, index, seed):
    """(multipliers, sizes) of the `index`-th hashed n-gram table of a model, as its reference derives them where the
    checkpoint does not store them: one odd multiplier a token of the n-gram (splitmix64 of the seed, below the bound
    that keeps token * multiplier inside 63 bits) and each head's table size, the next primes from `base` on (the
    tables of a model take them in turn)."""
    bound = max(1, (((1 << 63) - 1) // max(vocabulary, 1)) // 2)
    start = seed + _PRIME * index
    multipliers = [2 * (_splitmix64((start + _GAMMA * (i + 1)) & _MASK64) % bound) + 1 for i in range(size)]

    def prime(n):
        return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))
    sizes, at = [], base - 1
    for _ in range((index + 1) * heads):
        at += 1
        while not prime(at):
            at += 1
        sizes.append(at)
    return multipliers, sizes[index * heads:]


class NgramRows:
    """Rows of a hashed n-gram table gated into every residual stream ahead of a layer (a `before` block over a
    streams residual).

    A token's n-grams, of 2 .. size tokens ending at it, are hashed to table rows, several heads an n-gram: the tokens
    times one odd multiplier each, XORed, modulo each head's own prime size, at the head's offset in one table. An
    n-gram never reaches past the end token before it: the tokens from that one back count as end tokens, as those
    before the sequence do. The rows side by side are the token's embedding e, and with n the streams' norms

        key = rms_norm(key_proj(e))           one a stream            value = value_proj(e)
        g = key . rms_norm(stream) / sqrt(hidden),  then sign(g) * sqrt(max(|g|, 1e-6))
        v = sigmoid(g) * value                one a stream
        stream += v + silu(conv(rms_norm(v)))  a causal depthwise convolution over the token's sequence, dilated by
                                               the n-gram size (the recurrence slot's, its window a pool the block
                                               declares)

    The table (hundreds of millions of rows where the n-gram vocabulary is large) is planned as a lookup table: on
    the device where the placement plugin puts it there, else read where the checkpoint stores it (its shards'
    files) through an embeddings plugin that provides embeddings:host; stored NVFP4 rows (their global scale
    weight_scale_2) and FP8 rows (E4M3 codes with one multiplier for the table, weight_scale) are decoded on lookup. The
    multipliers, sizes and offsets are the checkpoint's where it stores them, else derived as its reference derives
    them. Config keys (the spec's `config` renames them): ngram_size, heads_per_ngram, ngram_vocab_size_base,
    ple_embed_dim (the hidden width by default), ple_conv_kernel_size, seed, ple_layer_ids (the layers, for the
    table's turn among a model's tables), and eos_token_id. Spec: embedding (ple_embedding.) and table
    (ngram_embedding.), the tensors' prefixes under the block's."""

    @classmethod
    def _config(cls, model, spec):
        names = {**NGRAM, **(spec.get("config") or {})}
        found = {key: model.setting(name) for key, name in names.items()}
        found["dim"] = int(found["dim"] or model.hidden)
        for key in ("size", "heads", "base", "kernel"):
            if not isinstance(found[key], int) or found[key] < 1:
                raise ValueError(f"model-template: hashed n-gram rows take the config's {names[key]} (a positive "
                                 f"integer), not {found[key]!r}")
        if found["size"] < 2 or found["dim"] % ((found["size"] - 1) * found["heads"]):
            raise ValueError(f"model-template: {found['dim']} n-gram values over {found['size'] - 1} n-grams of "
                             f"{found['heads']} heads")
        return found

    @classmethod
    def _names(cls, model, prefix, spec):
        """The table's row tensors in row order: one tensor, or its shards."""
        ck = model.ck
        at = prefix + spec.get("embedding", "ple_embedding.") + spec.get("table", "ngram_embedding.")
        if at + "weight" in ck:
            return at, [at + "weight"]
        names = []
        while f"{at}shard_{len(names)}.weight" in ck:
            names.append(f"{at}shard_{len(names)}.weight")
        if not names:
            raise KeyError(f"model-template: the checkpoint has no {at}weight or {at}shard_0.weight")
        return at, names

    @classmethod
    def tables(cls, model, prefix, spec):
        """{plan name: (bytes, the tensor-name prefix the plan covers)}."""
        at, _ = cls._names(model, prefix, spec)
        return {"tables." + at.rstrip("."): (sum(model.ck.nbytes(n) for n in model.ck.names() if n.startswith(at)),
                                             at)}

    @classmethod
    def pools(cls, model, spec, layers):
        found = cls._config(model, spec)
        count = getattr(model.residual, "count", 0)
        if not count:
            raise ValueError("model-template: hashed n-gram rows join residual streams (the layout's residual: "
                             "streams)")
        return [{"key": "ngram", "kind": "window", "layers": list(layers), "channels": count * model.hidden,
                 "kernel": found["kernel"], "dilation": found["size"]}]

    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        torch = m.torch
        self.layer, self.pool = layer, "ngram"
        found = self._config(m, spec)
        self.size, self.per, self.dim = found["size"], found["heads"], found["dim"]
        self.token_window = self.size   # the ids a token's rows are hashed from: its own and the size - 1 before it
        self.captures_prompts = False   # a chunk's windows of ids come from the request, on the host
        heads = (self.size - 1) * self.per
        self.count, hidden = m.residual.count, m.hidden
        eos = m.config.get("eos_token_id")
        eos = eos[0] if isinstance(eos, (list, tuple)) and eos else eos
        if not isinstance(eos, int):
            raise ValueError("model-template: hashed n-gram rows stop at the end token: the config's eos_token_id")
        self.eos = eos
        ep = prefix + spec.get("embedding", "ple_embedding.")
        stored = [ep + n for n in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")]
        if all(n in m.ck for n in stored):
            multipliers, sizes, offsets = (m.ck.get(n, "cpu").tolist() for n in stored)
        else:
            turn = sorted(found["layers"] or [layer + 1]).index(layer + 1)
            multipliers, sizes = ngram_hash(m.vocabulary, self.size, heads, found["base"], turn,
                                            int(found["seed"] if found["seed"] is not None else 1234))
            offsets = [sum(sizes[:i]) for i in range(heads)]
        if len(multipliers) != self.size or len(sizes) != heads or len(offsets) != heads:
            raise ValueError(f"model-template: {ep} hashes {len(multipliers)} tokens into {len(sizes)} heads, the "
                             f"config's n-grams take {self.size} and {heads}")
        long = dict(dtype=torch.int64, device=m.device)
        self.multipliers = torch.tensor(multipliers, **long)
        self.sizes, self.offsets = torch.tensor(sizes, **long), torch.tensor(offsets, **long)
        self.table = self._table(prefix, spec, sizes[-1] + offsets[-1])
        self.key_value = m.matrices.together([prefix + "key_proj", prefix + "value_proj"], "ngram")
        if m.matrices.rows(prefix + "key_proj") != self.count * hidden:
            raise ValueError(f"model-template: {prefix}key_proj gives {m.matrices.rows(prefix + 'key_proj')} values, "
                             f"a key a stream is {self.count * hidden}")
        self.norm_key, self.norm_query, self.norm_conv = (
            _tensor(m, prefix + name + ".weight").float().view(self.count, hidden).contiguous()
            for name in ("norm_key", "norm_query", "norm_conv"))
        self.conv = _tensor(m, prefix + "conv1d.weight")[:, 0].float().contiguous()

    def _table(self, prefix, spec, rows):
        """The table through the embeddings slot: on the device where its plan put it, else read in place from the
        checkpoint's files (embeddings:host)."""
        m, ck, torch = self.m, self.m.ck, self.m.torch
        at, names = self._names(m, prefix, spec)
        width = self.dim // ((self.size - 1) * self.per)
        if sum(ck.shape(n)[0] for n in names) < rows:
            raise ValueError(f"model-template: {at} stores {sum(ck.shape(n)[0] for n in names)} rows, its heads hash "
                             f"into {rows}")
        if getattr(ck.headers[names[0]], "stored", False):   # a GGUF's quantized rows, decoded on lookup
            return self._gguf_table(names, width)
        stored = ck.headers[names[0]].dtype
        nvfp4, fp8 = stored == "U8", stored == "F8_E4M3"
        scales = [n + "_scale" for n in names] if nvfp4 else []
        scale = 1.0
        if nvfp4 or fp8:   # NVFP4's global scale, or FP8's one multiplier for the whole table (ModelOpt's weight_scale)
            name = at + ("weight_scale_2" if nvfp4 else "weight_scale")
            if name not in ck or ck.headers[name].nbytes > 8:
                raise ValueError(f"model-template: {at} stores {'NVFP4' if nvfp4 else 'FP8'} rows, read with one "
                                 f"scale for the table ({name}), which the checkpoint does not store as one value")
            scale = float(ck.get(name, "cpu").float().reshape(-1)[0])
        if stored not in ("U8", "F8_E4M3", "BF16", "F16", "F32"):
            raise ValueError(f"model-template: {at} stores {stored} rows; a table is BF16, F16, F32, NVFP4 or FP8")
        if m.placed.get("tables." + at.rstrip("."), "device") == "device":
            found = torch.cat([ck.get(n, m.device) for n in names])
            if nvfp4:
                return m.embeddings.store(found, format="nvfp4", scale=scale, dtype=torch.bfloat16,
                                          scales=torch.cat([ck.get(n, m.device) for n in scales]))
            if fp8:
                return m.embeddings.store(found, format="fp8", scale=scale, dtype=torch.bfloat16)
            return m.embeddings.store(found.to(torch.bfloat16))
        if "embeddings:host" not in m.parts.facts("embeddings"):
            raise ValueError(f"model-template: the table {at} lies off the device, and the embeddings plugin does "
                             "not provide embeddings:host (lookup-table-rows reads a table where the checkpoint "
                             "stores it)")

        def files(tensors):
            return {"paths": [str(ck.headers[n].shard) for n in tensors],
                    "offsets": [int(ck.headers[n].data_begin) for n in tensors],
                    "rows": [int(ck.shape(n)[0]) for n in tensors]}
        if nvfp4:
            return m.embeddings.open(files(names), width, torch.bfloat16, m.device, format="nvfp4",
                                     scales=files(scales), scale=scale)
        if fp8:
            return m.embeddings.open(files(names), width, torch.bfloat16, m.device, format="fp8", scale=scale)
        kind = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[ck.headers[names[0]].dtype]
        return m.embeddings.open(files(names), width, kind, m.device)

    def _gguf_table(self, names, width):
        """The table as a GGUF stores it (rows of one block format), decoded a lookup (the model's gguf_table)."""
        at = names[0].rsplit("weight", 1)[0]
        return self.m.gguf_table(names, width, "tables." + at.rstrip("."))

    def rows(self, windows):
        """The table rows of each token ([tokens][heads] int64) from its window of ids ([tokens][size], the token's
        own last, -1 before the sequence)."""
        torch = self.m.torch
        ids = torch.where(windows < 0, self.eos, windows)
        mixed = ids[:, -1] * self.multipliers[0]
        ended = torch.zeros_like(mixed, dtype=torch.bool)
        found = []
        for back in range(1, self.size):
            earlier = ids[:, -1 - back]
            ended = ended | (earlier == self.eos)   # an end token from here back: the n-gram stops at it
            mixed = mixed ^ (torch.where(ended, self.eos, earlier) * self.multipliers[back])
            heads = slice((back - 1) * self.per, back * self.per)
            found.append(torch.remainder(mixed[:, None], self.sizes[None, heads]) + self.offsets[None, heads])
        return torch.cat(found, dim=1)

    def forward(self, streams, ctx):
        m, norms = self.m, self.m.norms
        tokens, count, hidden = streams.shape
        if ctx.windows is None:
            raise RuntimeError("model-template: hashed n-gram rows read each token's window of ids, and this forward "
                               "has none")
        e = self.table.lookup(self.rows(ctx.windows).reshape(-1)).reshape(tokens, self.dim)
        found = [ctx.mm(e, matrix) for matrix in self.key_value]
        kv = found[0] if len(found) == 1 else m.torch.cat(found, dim=-1)
        key = norms.rms_norm(kv[:, :count * hidden].reshape(tokens, count, hidden), self.norm_key, m.eps,
                             offset=m.norm_offset)
        query = norms.rms_norm(streams, self.norm_query, m.eps, offset=m.norm_offset)
        gate = row_sum(key.float() * query.float()) * hidden ** -0.5
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        value = norms.sigmoid_mul(kv[:, None, count * hidden:].expand(tokens, count, hidden).contiguous(),
                                  gate[:, :, None].expand(tokens, count, hidden).contiguous(),
                                  out_dtype=streams.dtype)
        normed = norms.rms_norm(value, self.norm_conv, m.eps, offset=m.norm_offset)
        conv = m.recurrence.convolve(self.layer, normed.reshape(tokens, count * hidden), self.conv, None,
                                     m.pools[self.pool], ctx.batch)
        return (streams.float() + value.float() + conv.view(tokens, count, hidden).float()).to(streams.dtype)


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


class SwiGLU:
    """The dense gated MLP: act(gate) * up through one stacked gate|up matrix, then down. The gate is the model's
    activation (ComposedModel.activation, the config's hidden_act or hidden_activation): SiLU (SwiGLU), or GELU in
    its tanh form."""

    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        self.gate_up = m.matrices.load([prefix + "gate_proj", prefix + "up_proj"], "gate_up")
        self.down = m.matrices.load([prefix + "down_proj"], "down")

    def forward(self, h, ctx):
        m, both = self.m, ctx.mm(h, self.gate_up)
        gated = m.norms.silu_mul(both) if m.activation == "silu" else m.norms.gelu_mul(both, approximate="tanh")
        return ctx.mm(gated, self.down)


def _mx_values(torch, stored):
    """An MXFP8 matrix's exact values in BF16: each E4M3 code times 2 ** (its E8M0 scale - 127), one scale a 32."""
    codes, scales = stored["weight"], stored["scale"]
    factors = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)[..., :codes.shape[-1]]
    values = codes.float() * factors
    found = values.to(torch.bfloat16)
    if not torch.equal(found.float(), values):
        raise ValueError("model-template: an MXFP8 matrix's values do not fit BF16 exactly")
    return found.contiguous()


def router_rule(moe, defaults=None):
    """The routing rule a checkpoint's config states, in topk-router's option names; defaults: what the family's
    reference assumes where config.json is silent (the layout's `defaults`: norm_topk_prob)."""
    defaults = defaults or {}
    scoring = moe.scoring_func or defaults.get("scoring_func") or "softmax"
    renormalize = moe.norm_topk_prob if moe.norm_topk_prob is not None else defaults.get("norm_topk_prob")
    rule = {"scoring": scoring, "score_before_selection": True, "renormalize": bool(renormalize),
            "scaling": float(moe.routed_scaling_factor or 1.0)}
    if moe.n_group and moe.n_group > 1:
        rule.update(groups=moe.n_group, selected_groups=moe.topk_group or 1)
        if defaults.get("group_score_top"):   # a group's score: the sum of its this many best experts
            rule["group_score_top"] = int(defaults["group_score_top"])
    return rule


class MoE:
    """A mixture of experts: the router slot's selection over the router's logits (the config's rule, checked against
    the router plugin's), the experts slot's routed SwiGLU block over every expert the layer loads (per-expert gate,
    up and down matrices, or fused [experts] tensors), and the shared expert where the checkpoint has one (gated by
    sigmoid of its own projection where it stores that gate). The experts plugin may hold the experts anywhere
    (expert-offload: device, host memory or a file): with `experts:offload` the model plans no device bytes for them
    and calls begin() before the loads and open() after its pools."""

    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        self.layer = layer
        self.slot = m.expert_ids[layer]   # the experts slot's id of the layer
        moe = m.config.moe
        self.top, self.count = moe.experts_per_token, moe.experts
        self.router = m.matrices.load([prefix + spec.get("router", "gate")], "router")
        bias = _tensor(m, prefix + spec.get("router", "gate") + ".e_score_correction_bias", False)
        self.bias = None if bias is None else bias.float()
        self.shared = self.shared_gate = None
        for name in ("shared_expert.", "shared_experts."):
            if m.matrices.has(prefix + name + "gate_proj"):
                self.shared = SwiGLU(m, layer, prefix + name, {})
        if m.matrices.has(prefix + "shared_expert_gate"):
            self.shared_gate = m.matrices.load([prefix + "shared_expert_gate"], "shared_gate")
        self.prefix = prefix + spec.get("experts", "experts.")
        # A gate_up stack that keeps each gate row beside its up row (the spec's gate_up: interleaved)
        self.interleaved = spec.get("gate_up") == "interleaved"
        # The routing weights at the hidden dtype, as most references round them, or FP32 (the spec's weights)
        self.fp32_weights = spec.get("weights") == "fp32"

    def load_experts(self):
        """Every routed expert of the layer into the experts slot: per-expert matrices, or a fused gate_up_proj
        [experts][2 * inter][hidden] and down_proj [experts][hidden][inter]."""
        m = self.m
        # An experts plugin that packs its bundles on the host (host_loads: expert-offload) takes host tensors: on
        # the device first, every expert crossed the link twice on its way to host memory.
        device = "cpu" if getattr(m.experts, "host_loads", False) else m.device
        hold = getattr(m.ck, "hold", None)
        if hold is not None:   # the layer's experts, read once and packed elsewhere: past the page cache, at once
            hold(self.prefix)
        try:
            self._load_experts(device)
        finally:
            if hold is not None:
                m.ck.release()

    def _load_experts(self, device):
        m, torch = self.m, self.m.torch
        fused = self.prefix + "gate_up_proj"
        if fused in m.ck:
            gate_up, down = m.ck.get(fused, device), m.ck.get(self.prefix + "down_proj", device)
            if gate_up.shape[1] == m.hidden != gate_up.shape[2]:   # stored [experts][hidden][2 * inter]: transposed
                gate_up, down = gate_up.transpose(1, 2), down.transpose(1, 2)
            for e in range(self.count):
                m.experts.load(self.slot, e, gate_up[e].contiguous().to(torch.bfloat16),
                               down[e].contiguous().to(torch.bfloat16))
            return
        if m.matrices.has(fused):   # two stacks as matrices (fused.weight with its scales): each expert as stored
            for e in range(self.count):
                fmt, gate_up, down = m.matrices.stacked_expert(fused, self.prefix + "down_proj", e, self.interleaved,
                                                               device)
                if fmt == "mxfp8" and "experts:mxfp8" not in m.parts.facts("experts"):
                    # an experts plugin without MXFP8 bundles: the values exactly, in BF16 (an E4M3 code times a
                    # power of two is a BF16 value)
                    fmt, gate_up, down = "bf16", _mx_values(torch, gate_up), _mx_values(torch, down)
                    m.__dict__.setdefault("experts_decoded", set()).add(self.layer)
                if fmt == "bf16":
                    m.experts.load(self.slot, e, gate_up, down)
                else:
                    m.experts.load(self.slot, e, gate_up, down, fmt)
            return
        # The draft layer's experts in the format the slot holds the others in, re-encoded where stored otherwise.
        from crit_plugins.encodings import SOURCES
        want = m.expert_format if self.layer == m.layers else None
        stored = m.matrices.format(f"{self.prefix}0.gate_proj")
        if stored not in SOURCES:   # read as its own format reads it (a GGUF's layers keep theirs)
            want = None
        if want is not None and stored != want:
            m.draft_experts = f"stored {stored}, re-encoded to {want}"
        for e in range(self.count):
            p = f"{self.prefix}{e}."
            fmt, gate_up, down = m.matrices.expert(p + "gate_proj", p + "up_proj", p + "down_proj", want, device)
            if fmt == "bf16":
                m.experts.load(self.slot, e, gate_up, down)
            else:
                m.experts.load(self.slot, e, gate_up, down, fmt)

    def routed(self, h, ids, weights, ctx):
        """The experts' block of rows h over their routes (FP32), through the experts slot."""
        torch, m = self.m.torch, self.m
        if m.offload:
            # decode rows and prompt chunks apart: a decode row keeps its decode arithmetic whatever shares the step
            # (a step's decode rows come first, StepLayout)
            split = next((first for first, _, prompt in ctx.groups if prompt), None)
            if ctx.prompt and split and not getattr(ctx.batch, "prompt", False):
                routed = torch.cat([m.experts.forward(self.slot, h[:split], ids[:split], weights[:split], prompt=False),
                                    m.experts.forward(self.slot, h[split:], ids[split:], weights[split:],
                                                      prompt=True)])
            else:
                # Each token's row for the cache's per-row reads and hits, which a round's count_round hands to its
                # requests: a round's rows only (a step's rows are not round slots, and their counts would wait in
                # the cache for the next round's requests).
                routed = m.experts.forward(self.slot, h, ids, weights, prompt=ctx.prompt,
                                           rows=ctx.batch.rows_t if ctx.batch.round else None)
        else:
            routed = m.experts.forward(self.slot, h, ids, weights)
        return routed

    def forward(self, h, ctx):
        torch, m = self.m.torch, self.m
        ids, weights = m.router.select(ctx.mm(h, self.router).float(), self.top, bias=self.bias)
        if not self.fp32_weights:
            weights = weights.to(h.dtype).float()   # the routing weights at the hidden dtype, as the reference rounds them
        routed = self.routed(h, ids, weights, ctx)
        if self.shared is None:
            return routed.to(h.dtype)
        shared = self.shared.forward(h, ctx).float()
        if self.shared_gate is not None:
            shared = torch.sigmoid(ctx.mm(h, self.shared_gate).float()) * shared
        return (routed + shared).to(h.dtype)


class ParallelMoE(MoE):
    """A dense gated MLP and a mixture of experts side by side, each normed, summed:

        dense   = norm_1(mlp(h))                              h: the layer's normed input, as any FFN's
        r       = the residual the layer adds into            (Context.residual: a sandwich residual gives it)
        routes  = top-k of softmax(proj(rms_norm(r) * scale / sqrt(hidden))), renormalized by the router slot,
                  each weight times its expert's scale
        routed  = norm_3(experts(norm_2(r), routes))
        out     = dense + routed

    The block sits at its layer's prefix: the MLP under `mlp.`, the router's `proj`, `scale` and `per_expert_scale`
    under `router.`, the experts fused under `experts.`, the three norms `post_feedforward_layernorm_1`,
    `pre_feedforward_layernorm_2` and `post_feedforward_layernorm_2` (the spec's `dense`, `router`, `experts` and
    `norms` rename them). The MLP's and the experts' gate is the model's activation."""

    at_layer = True   # made with its layer's prefix, not the FFN's

    def __init__(self, model, layer, prefix, spec):
        spec = {**spec, "router": spec.get("router", "router.") + "proj"}
        super().__init__(model, layer, prefix, spec)
        m, torch = self.m, self.m.torch
        self.dense = SwiGLU(m, layer, prefix + spec.get("dense", "mlp."), {})
        at = prefix + spec["router"][:-len("proj")]
        self.route_scale = _tensor(m, at + "scale").float() * m.hidden ** -0.5
        self.expert_scale = _tensor(m, at + "per_expert_scale").float()
        names = spec.get("norms", ("post_feedforward_layernorm_1", "pre_feedforward_layernorm_2",
                                   "post_feedforward_layernorm_2"))
        self.norm_dense, self.norm_in, self.norm_out = (_tensor(m, prefix + n + ".weight").float() for n in names)
        self.unit = torch.ones(m.hidden, dtype=torch.float32, device=m.device)   # an RMSNorm without a weight

    def forward(self, h, ctx):
        m, r = self.m, ctx.residual
        if r is None:
            raise RuntimeError("model-template: a parallel-moe block reads the stream its layer adds into, which "
                               "this residual does not give (the layout's residual: sandwich)")
        dense = m._norm(self.dense.forward(h, ctx), self.norm_dense)
        scored = (m.norms.rms_norm(r, self.unit, m.eps).float() * self.route_scale).to(r.dtype)
        ids, weights = m.router.select(ctx.mm(scored, self.router).float(), self.top, bias=self.bias)
        weights = weights.float() * self.expert_scale[ids]
        routed = self.routed(m._norm(r, self.norm_in), ids, weights, ctx)
        return (dense.float() + m._norm(routed.to(r.dtype), self.norm_out).float()).to(h.dtype)

def _clipped(torch, x, threshold, scale=None):
    """Each head vector of x [.., heads, dim] (times its learned scale [heads][dim], where given) shortened to
    `threshold` where its L2 norm is longer (norm + 1e-6), in FP32: the norm a row sum in one fixed order."""
    x = x.float() if scale is None else x.float() * scale
    norm = row_sum(x * x).sqrt()
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)[..., None]


class RelativeAttention:
    """Softmax attention with a relative-position bias a token computes for itself, learned sinks a token, short
    convolutions over the keys and values, and a gated output. Its rows: queries, keys and values may each come from
    rows of their own (forward((q_in, k_in, v_in), ctx), each normed by the residual; a single row for all three):

        q = clip(q_proj(q_in) * q_scale)           each head shortened to the config's qk_l2_clip_threshold
        k = clip(conv(k_proj(k_in)))               the convolution: (taps over the last `kernel` rows + the row) /
        v = rms_norm(conv(v_proj(v_in)))           sqrt(kernel + 1); with stored static taps and a projection of
                                                   the input (dynamic), each token's taps are the static ones plus
                                                   its own, one set a key/value head, / sqrt(2 * kernel + 1)
        r = clip(relative_proj(q_in) * r_scale)    each head's relative state, d_rel values, clipped to the config's
        bias[d] = r . proj[:, d]                     rel_l2_clip_threshold: its score for the position d behind it
        out = sigmoid(lse - sink) * attention(q, k, v, bias)    sink = sink_proj(q_in), a logit with no value
        y = o_proj(out * sigmoid(gate_proj(q_in)))

    With `log_scaling` (the full layers) the queries and the relative states are scaled after their clip by
    1 + alpha * log(max((position + 1) / floor, 1)) (the config's log_scaling_alpha and log_scaling_n_floor). The
    bias reaches `extent` positions back: the sliding window, or the config's rel_extent. The block also holds the
    convolutions its layer's residual runs after the attention and after the FFN (`post_convolution(name, y, ctx)`):
    each convolution keeps its window in a pool the block declares (the recurrence slot's convolve), one a layer, so a
    looped layer's passes keep their own. Tensor names are transformers'; spec keys: heads, kv_heads and head_dim (the
    config keys of this kind's shape), log_scaling, post (the residual's convolutions: {name: tensor prefix under the
    layer})."""

    keeps_kv = True

    @classmethod
    def _config(cls, model, spec):
        found = {key: model.setting(name) for key, name in (
            ("kernel", "conv_kernel_size"), ("d_rel", "d_rel"), ("qk", "qk_l2_clip_threshold"),
            ("rel", "rel_l2_clip_threshold"), ("alpha", "log_scaling_alpha"), ("floor", "log_scaling_n_floor"),
            ("rel_extent", "rel_extent"))}
        if not found["kernel"]:
            found["kernel"] = model.setting("sconv_kernel_size")
        missing = [key for key in ("kernel", "d_rel", "qk", "rel") if not found[key]]
        if missing:
            raise ValueError(f"model-template: relative attention takes the config's {', '.join(missing)}")
        return found

    @classmethod
    def _dynamic(cls, model, layers, spec):
        p = model.layer_prefix(layers[0]) + spec.get("prefix", "self_attn.")
        return p + "k_proj.sconv.static_weight" in model.ck

    @classmethod
    def pools(cls, model, spec, layers):
        """The windows of the layers' convolutions: keys and values (a dynamic convolution keeps `kernel` copies of
        each input, one a tap), and the residual's after the attention and after the FFN."""
        found = cls._config(model, spec)
        kv_heads, head_dim = model.kv_shape(layers[0], spec)
        taps = found["kernel"] if cls._dynamic(model, layers, spec) else 1
        kind = spec.get("kind", "")
        windows = [(f"{kind}.k", taps * kv_heads * head_dim), (f"{kind}.v", taps * kv_heads * head_dim)]
        windows += [(f"{kind}.{name}", model.hidden) for name in (spec.get("post") or {})]
        return [{"key": key, "kind": "window", "layers": list(layers), "channels": channels,
                 "kernel": found["kernel"]} for key, channels in windows]

    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        torch = m.torch
        self.layer = layer
        found = self._config(m, spec)
        self.kernel, self.d_rel = int(found["kernel"]), int(found["d_rel"])
        self.qk, self.rel = float(found["qk"]), float(found["rel"])
        self.log_scaling = bool(spec.get("log_scaling"))
        self.alpha, self.floor = float(found["alpha"] or 0.0), float(found["floor"] or 1.0)
        self.heads = int(m.setting(spec["heads"]) if spec.get("heads") else m.heads)
        self.kv_heads, self.head_dim = m.kv_shape(layer, spec)
        kind = spec.get("kind", "")
        window = spec.get("window")
        if window is None and kind == "sliding_attention":
            window = m.config.sliding_window
        self.window = int(window or 0)
        self.extent = self.window or int(found["rel_extent"] or 0)
        if not self.extent:
            raise ValueError("model-template: relative attention's bias reaches the sliding window or the config's "
                             "rel_extent")
        for fact in ("attention:distance-bias", "attention:lse") + (("attention:sliding",) if self.window else ()):
            if fact not in m.parts.facts("attention"):
                raise ValueError(f"model-template: relative attention needs the attention plugin's {fact}")
        if m.recurrence is None:
            raise ValueError("model-template: relative attention's convolutions keep their windows in the "
                             "recurrence slot: the selection leaves it empty")
        self.pool = ("kv", self.window) if self.window else "kv"
        names = ["q_proj.linear", "o_proj.gate_proj", "sink_proj", "relative_position_bias.linear"]
        q_rows = self.heads * self.head_dim
        rows = [q_rows, q_rows, self.heads, self.heads * self.d_rel]
        for name, want in zip(names, rows):
            if m.matrices.rows(prefix + name) != want:
                raise ValueError(f"model-template: {prefix}{name} gives {m.matrices.rows(prefix + name)} rows, the "
                                 f"layer's {self.heads} heads take {want}")
        self.query_rows = m.matrices.together([prefix + n for n in names], "qkv")   # one input: the query rows'
        self.k, self.v = (m.matrices.load([prefix + n + "_proj.linear"], "qkv") for n in "kv")
        self.o = m.matrices.load([prefix + "o_proj.linear"], "o")
        self.q_scale = _tensor(m, prefix + "q_proj.norm.scale").float().view(self.heads, self.head_dim)
        self.r_scale = _tensor(m, prefix + "relative_position_bias.norm.scale").float().view(self.heads, self.d_rel)
        proj = _tensor(m, prefix + "relative_position_bias.proj").float()
        if tuple(proj.shape) != (self.d_rel, self.extent):
            raise ValueError(f"model-template: {prefix}relative_position_bias.proj is {list(proj.shape)}, the bias "
                             f"takes [{self.d_rel}, {self.extent}]")
        self.proj = proj.contiguous()
        channels = self.kv_heads * self.head_dim
        self.dynamic = prefix + "k_proj.sconv.static_weight" in m.ck
        self.taps = {}
        for n in "kv":
            if self.dynamic:   # static taps [kernel][channels] by lag (0: the token's own), a projection of the input
                static = _tensor(m, prefix + n + "_proj.sconv.static_weight").float()
                self.taps[n] = (static.contiguous(), m.matrices.load([prefix + n + "_proj.sconv.weight_proj"], "conv"))
            else:              # [channels][kernel], the last tap the token's own: the recurrence slot's order
                self.taps[n] = _tensor(m, prefix + n + "_proj.sconv.conv1d.weight").float().reshape(channels, -1)
        if self.dynamic:   # x repeated once a tap, each copy read through one tap: copy i the input i rows back
            k = self.kernel
            pick = torch.zeros(k, channels, k, dtype=torch.float32, device=m.device)
            for lag in range(k):
                pick[lag, :, k - 1 - lag] = 1.0
            self.pick = pick.reshape(k * channels, k).contiguous()
        self.post = {name: _tensor(m, m.layer_prefix(layer) + at + "conv1d.weight").float().reshape(m.hidden, -1)
                     for name, at in (spec.get("post") or {}).items()}
        self.windows = [f"{kind}.k", f"{kind}.v"] + [f"{kind}.{name}" for name in self.post]
        self.ones = torch.ones(self.head_dim, dtype=torch.float32, device=m.device)

    def _log_scale(self, x, ctx):
        """x [tokens][heads][dim] times 1 + alpha * log(max((position + 1) / floor, 1)) of its token."""
        torch = self.m.torch
        tau = 1.0 + self.alpha * torch.log(((ctx.positions.float() + 1.0) / self.floor).clamp(min=1.0))
        return x * tau[:, None, None]

    def _convolve(self, key, x, weight, ctx):
        """(convolution + x) / sqrt(kernel + 1) of rows x [tokens][channels] over the window `key`, in FP32."""
        m = self.m
        y = m.recurrence.convolve(self.layer, x.float().contiguous(), weight, None, m.pools[key], ctx.batch,
                                  activation=None)
        return (y.float() + x.float()) / math.sqrt(self.kernel + 1)

    def _projected(self, n, x_in, ctx):
        """Keys or values: the projection, its convolution, then per head the clip (keys) or the norm (values)."""
        m, torch = self.m, self.m.torch
        rows = ctx.mm(x_in, self.k if n == "k" else self.v)
        t, channels = rows.shape[0], self.kv_heads * self.head_dim
        kind_key = self.windows[0 if n == "k" else 1]
        if self.dynamic:
            static, proj = self.taps[n]
            own = ctx.mm(x_in, proj).float().view(t, self.kv_heads, 1, self.kernel)
            copies = rows.float().repeat(1, self.kernel)                                  # [t][kernel * channels]
            back = m.recurrence.convolve(self.layer, copies.contiguous(), self.pick, None, m.pools[kind_key],
                                         ctx.batch, activation=None).float()
            back = back.view(t, self.kernel, self.kv_heads, self.head_dim)               # back[:, lag]: lag rows back
            taps = static.view(self.kernel, self.kv_heads, self.head_dim) + own.permute(0, 3, 1, 2)
            y = (row_sum((back * taps).permute(0, 2, 3, 1)).reshape(t, channels) + rows.float()) \
                / math.sqrt(2 * self.kernel + 1)
        else:
            y = self._convolve(kind_key, rows, self.taps[n], ctx)
        y = y.to(rows.dtype).view(t, self.kv_heads, self.head_dim)
        if n == "k":
            return _clipped(torch, y, self.qk).to(rows.dtype)
        return m.norms.rms_norm(y.contiguous(), None, m.eps).to(rows.dtype)

    def post_convolution(self, name, y, ctx):
        """The residual's convolution `name` (after the attention, after the FFN) of rows y, in y's dtype."""
        return self._convolve(f"{self.windows[0].rsplit('.', 1)[0]}.{name}", y, self.post[name], ctx).to(y.dtype)

    def forward(self, h, ctx):
        m, torch = self.m, self.m.torch
        q_in, k_in, v_in = h if isinstance(h, tuple) else (h, h, h)
        t, dtype = q_in.shape[0], q_in.dtype
        found = [ctx.mm(q_in, matrix) for matrix in self.query_rows]
        rows = found[0] if len(found) == 1 else torch.cat(found, dim=-1)
        qw = self.heads * self.head_dim
        q = _clipped(torch, rows[:, :qw].view(t, self.heads, self.head_dim), self.qk, self.q_scale)
        r = _clipped(torch, rows[:, 2 * qw + self.heads:].view(t, self.heads, self.d_rel).to(dtype).float(), self.rel,
                     self.r_scale)
        if self.log_scaling:
            q = self._log_scale(q.to(dtype).float(), ctx)
            r = self._log_scale(r.to(dtype).float(), ctx)
        q = q.to(dtype).contiguous()
        bias = (r.to(dtype).float() @ self.proj).to(dtype).float().contiguous()          # [t][heads][extent]
        sink = rows[:, 2 * qw:2 * qw + self.heads].float()
        k, v = self._projected("k", k_in, ctx), self._projected("v", v_in, ctx)
        pool = m.pools[self.pool]
        pool.write(self.layer, ctx.slots[self.pool], k, v)
        a, lse = m.attention.forward(self.layer, q, pool, ctx.batch, scale=self.head_dim ** -0.5, lse=True,
                                     bias=bias, **({"sliding": self.window} if self.window else {}))
        a = a.float() * torch.sigmoid(lse - sink)[..., None]                              # the sink: no value
        gated = (a.reshape(t, qw) * torch.sigmoid(rows[:, qw:2 * qw].float())).to(dtype)
        return ctx.mm(gated, self.o)



BLOCKS = {"attention": Attention, "indexed-attention": IndexedAttention, "latent-attention": LatentAttention,
          "ngram-rows": NgramRows, "gated-delta": GatedDelta, "swiglu": SwiGLU, "moe": MoE,
          "parallel-moe": ParallelMoE, "relative-attention": RelativeAttention}
