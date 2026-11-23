"""Attention residuals over blocks of layers, with a looped span of layers run several times, its later passes restarted and gated by the last pass's messages.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `block_attention`
and composes the residual block-attention a layout names (README.md)."""
from ..template.blocks import row_sum


class BlockAttention:
    """Attention residuals over blocks of layers, with the layers of a looped span run several times (layer passes).

    The layers are grouped in blocks of `block` (the config's n_layers_per_attn_res_block): at a block's first layer
    the row so far becomes one more block summary, and the layer's own sum starts again from its attention's output.
    Each block input is an attention over the summaries and the row so far, all normed without a weight, by a learned
    query (FP32):

        mix(blocks, h, query) = softmax(query . rms(s) / sqrt(hidden)) . rms(s)   s: the summaries, then h

        q_in = k_in = v_in = mix(blocks, h, attention_query)     (layer 0: h; a full layer has three queries)
        a = mixer(norm(q_in), norm(k_in), norm(v_in)); a = conv_after_attention(a)
        r = a at a block's first layer (h joins the summaries), else h + a
        y = conv_after_ffn(ffn(norm(mix(blocks, r, ffn_query))));  the layer's row: r + y

    A looped span's later passes start again from the row and summaries before the span, and each of its layers
    takes messages from the pass before: one attention over that pass's own summaries and last row by four learned
    queries a layer (the query, key, value and FFN messages), each joined to its input by a gate,

        gate(x, m) = (1 - tanh(forget(g))) * x + tanh(gain(g)) * m,   g = down([rms(x), rms(m)])

    The last row is the head's input: mix(blocks, h, final query), then the final norm. Each pass of a looped layer is
    a layer of its own here (its own KV and convolution windows: shape.py's layer applications). Tensor names are
    transformers' (a layout's names rules rename a checkpoint's own); the mixer is a relative-attention block, whose
    post convolutions the residual runs. Layout keys (beside `block`): embed_norm (the embedding's norm, after the
    lookup), block_key (the config key of the block size, n_layers_per_attn_res_block where it gives none).

    On CUDA the mixes and the message gates run crit's kernel row-ops (the model's kernel("row-ops"): summary_mix,
    pair_norm, gated_join, low_rank_gates, scaled_add; its knobs tuned by crit kernels sweep), else the torch forms
    here, which are its references' arithmetic."""

    name = "block-attention"

    def __init__(self, model, spec):
        m = self.m = model
        self.size = int(m.setting(spec.get("block_key", "n_layers_per_attn_res_block")) or 0)
        if self.size < 1:
            raise ValueError("model-template: attention residuals group layers in blocks of the config's "
                             "n_layers_per_attn_res_block")
        self.embed_norm_name = spec.get("embed_norm", "embed_tokens.embed_norm.weight")
        self.width = model.hidden
        spans = m.config.layer_passes
        if len(spans) > 1:
            raise ValueError("model-template: attention residuals loop one span of layers, not "
                             f"{len(spans)}")
        self.span = spans[0] if spans else None
        if self.span and (self.span[0] % self.size or self.span[1] % self.size or not self.span[0]):
            raise ValueError(f"model-template: a looped span {self.span[:2]} starts and ends at a block's first "
                             f"layer, after layer 0 (blocks of {self.size})")
        self.final = self.embed_norm = self.final_query = self.messages = None
        # crit's row operations on CUDA, None for the torch forms (and in the dry shape, which computes nothing)
        self.ops = m.kernel("row-ops") if callable(getattr(m, "kernel", None)) else None

    def anchor(self):
        return "post_attention_layernorm.weight"

    def names(self, prefix):
        return [prefix + "norm.weight", prefix + "final_input.query", prefix + self.embed_norm_name,
                prefix + "message_input.query", prefix + "message_input.query_mlp", prefix + "message_input.key_value"]

    def _vector(self, name):
        return self.m.ck.get(name, self.m.device).float()

    def load(self, prefix):
        m = self.m
        self.final = self._vector(prefix + "norm.weight")
        self.final_query = self._vector(prefix + "final_input.query").view(1, -1)
        self.embed_norm = self._vector(prefix + self.embed_norm_name) if prefix + self.embed_norm_name in m.ck \
            else None
        if self.span:
            if prefix + "message_input.query" in m.ck:                     # [loop layers][4][hidden]
                self.messages = self._vector(prefix + "message_input.query")
            else:   # kept as two tensors: each loop layer's query and FFN queries, then its key and value queries
                torch = m.torch
                pairs = self._vector(prefix + "message_input.query_mlp").unflatten(0, (-1, 2))
                kv = self._vector(prefix + "message_input.key_value").unflatten(0, (-1, 2))
                self.messages = torch.cat([pairs[:, :1], kv, pairs[:, 1:]], dim=1).contiguous()
            if tuple(self.messages.shape) != (self.span[1] - self.span[0], 4, m.hidden):
                raise ValueError(f"model-template: {prefix}message_input.query is {list(self.messages.shape)}, the "
                                 f"loop's {self.span[1] - self.span[0]} layers take four queries each")
        # Where each layer stands: (source layer, pass, its place in the span, the pass's first layer)
        self.where = []
        for i, source in enumerate(m.sources):
            looped = self.span is not None and self.span[0] <= source < self.span[1]
            before = sum(1 for s in m.sources[:i] if s == source)
            self.where.append((source, before if looped else 0, source - self.span[0] if looped else None,
                               looped and source == self.span[0]))

    def layer(self, p):
        """Per layer: its queries (attention input: None at layer 0, [1 or 3][hidden]; FFN input [1][hidden]), its
        norms (one for the mixer's rows, or one each for queries, keys and values; the FFN's) and, in a looped layer,
        its four message gates (down, forget, gain matrices)."""
        m, ck = self.m, self.m.ck
        attention = self._vector(p + "attention_input.query").view(-1, m.hidden) \
            if p + "attention_input.query" in ck else None
        ffn = self._vector(p + "mlp_input.query").view(1, m.hidden)
        if p + "q_input_layernorm.weight" in ck:
            norms = tuple(self._vector(p + f"{n}_input_layernorm.weight") for n in "qkv")
        else:
            norms = (self._vector(p + "input_layernorm.weight"),)
        post = self._vector(p + "post_attention_layernorm.weight")
        gates = None
        if p + "q_message_gate.down_proj.weight" in ck:
            # down, then forget and gain as one product of the low-rank rows where they are stored alike
            gates = {n: (m.matrices.load([f"{p}{n}_message_gate.down_proj"], "gate"),
                         m.matrices.together([f"{p}{n}_message_gate.forget_gate_proj",
                                              f"{p}{n}_message_gate.message_gate_proj"], "gate"))
                     for n in ("q", "k", "v", "mlp")}
            if self.ops is not None:
                # and as one stack each, for row-ops' low_rank_gates: down [4][rank][2 hidden], up [4][2 hidden][rank]
                bf16 = m.torch.bfloat16
                raw = lambda n, part: ck.get(f"{p}{n}_message_gate.{part}.weight", m.device).to(bf16)  # noqa: E731
                names = ("q", "k", "v", "mlp")
                gates["down"] = m.torch.stack([raw(n, "down_proj") for n in names]).contiguous()
                gates["up"] = m.torch.stack([m.torch.cat([raw(n, "forget_gate_proj"), raw(n, "message_gate_proj")])
                                             for n in names]).contiguous()
        return attention, ffn, norms, post, gates

    def carry(self, itemsize):
        """The state's bytes a token: the row, the summaries a layer may hold (a block a layer at most, the embedding
        row's), and in a loop the pre-loop row and summaries and the last pass's."""
        blocks = -(-self.m.config.num_hidden_layers // self.size) + 1
        return (2 + 3 * blocks) * self.m.hidden * itemsize

    # -- the arithmetic -------------------------------------------------------------------------------------------------
    def _rms(self, x):
        return self.m.norms.rms_norm(x.float().contiguous(), None, 1e-6)

    def mix(self, blocks, h, query):
        """[tokens][queries][hidden] FP32: each query's attention over the summaries [tokens][n][hidden] and the row
        h, the scores' sums in one fixed order."""
        torch = self.m.torch
        if self.ops is not None:   # one program a token and chunk of the row (row-ops' summary_mix)
            return self.ops.summary_mix(blocks, h, query)
        stacked = self._rms(torch.cat([blocks, h[:, None, :]], dim=1))               # [t][n + 1][hidden]
        scores = row_sum(query[None, :, None, :] * stacked[:, None, :, :]) * self.m.hidden ** -0.5
        p = torch.softmax(scores, dim=-1)                                            # [t][queries][n + 1]
        return row_sum((p[..., None] * stacked[:, None]).transpose(-1, -2))          # summed over the n + 1

    def _gate(self, x, message, gate, ctx):
        torch = self.m.torch
        down, both = gate
        hidden = x.shape[-1]
        if self.ops is not None:
            g = ctx.mm(self.ops.pair_norm(x, message), down)
            found = [ctx.mm(g, matrix, out_dtype=torch.float32) for matrix in both]
            fa = found[0] if len(found) == 1 else torch.cat(found, dim=-1)
            return self.ops.gated_join(x, message, fa[:, :hidden], fa[:, hidden:])
        joined = torch.cat([self._rms(x), self._rms(message)], dim=-1).to(torch.bfloat16)
        g = ctx.mm(joined, down)
        found = [ctx.mm(g, matrix, out_dtype=torch.float32).float() for matrix in both]
        fa = found[0] if len(found) == 1 else torch.cat(found, dim=-1)
        f, a = torch.tanh(fa[:, :hidden]), torch.tanh(fa[:, hidden:])
        return (1.0 - f) * x.float() + a * message.float()

    def _norm(self, x, weight, dtype):
        x = x.float()
        return self.m.norms.rms_norm(x if x.is_contiguous() else x.contiguous(), weight, self.m.eps, out_dtype=dtype)

    def _layer(self, layer, source, h, blocks, messages, ctx):
        torch = self.m.torch
        attention, ffn, norms, post, gates = layer.res
        dtype = h.dtype
        boundary = source % self.size == 0
        if attention is None:
            inputs = [h.float()] * 3
        else:
            found = self.mix(blocks, h, attention)
            inputs = [found[:, i] for i in range(found.shape[1])] if found.shape[1] == 3 else [found[:, 0]] * 3
        # A step's decode and verify rows: a looped layer's q, k and v gates in one fused call (one program a row:
        # a round's few rows, where the products' launches would cost more than their arithmetic); a step with prompt
        # rows takes them as GEMMs (each its tensor-core product over the step's rows)
        few = messages is not None and not ctx.prompt and self.ops is not None and "down" in gates
        if few:
            shared = attention is None or found.shape[1] != 3
            x = inputs[0][:, None, :].expand(-1, 3, -1) if shared else found
            joined = self.ops.low_rank_gates(x, messages[:, :3], gates["down"][:3], gates["up"][:3])
            inputs = [joined[:, i] for i in range(3)]
        elif messages is not None:
            inputs = [self._gate(x, messages[:, i], gates[n], ctx) for i, (x, n) in enumerate(zip(inputs, "qkv"))]
        if len(norms) == 3:
            rows = tuple(self._norm(x, w, dtype) for x, w in zip(inputs, norms))
        elif messages is None:
            rows = (self._norm(inputs[0], norms[0], dtype),) * 3
        else:
            rows = tuple(self._norm(x, norms[0], dtype) for x in inputs)
        a = layer.mixer.forward(rows, ctx)
        a = layer.mixer.post_convolution("attention", a, ctx)
        quick = self.ops is not None
        r = a if quick else a.float()   # a's values: its BF16 rows read as FP32 where they are added
        if boundary:
            blocks = torch.cat([blocks, h[:, None, :]], dim=1)
        else:
            r = self.ops.scaled_add(h, a, 1.0, torch.float32) if quick else h.float() + r
        x = self.mix(blocks, r, ffn)[:, 0]
        if few:
            x = self.ops.low_rank_gates(x[:, None], messages[:, 3:4], gates["down"][3:], gates["up"][3:])[:, 0]
        elif messages is not None:
            x = self._gate(x, messages[:, 3], gates["mlp"], ctx)
        y = layer.ffn.forward(self._norm(x, post, dtype), ctx)
        y = layer.mixer.post_convolution("ffn", y, ctx)
        if quick:
            return self.ops.scaled_add(r, y, 1.0, dtype)
        return (r + y.float()).to(dtype)

    def begin(self, layers, x, ctx):
        """(h, summaries, pre-loop h, pre-loop summaries, last pass's summaries, last pass's h) before layer 0."""
        if any(layer.before for layer in layers):
            raise ValueError("model-template: attention residuals have no rows added before a layer")
        if self.embed_norm is not None:
            x = self._norm(x, self.embed_norm, x.dtype)
        empty = x.new_zeros(x.shape[0], 0, x.shape[1])
        return x, empty, None, None, None, None

    def row(self, state):
        """The row a layer reads on entry (the row so far), which a drafter technique may tap (residual.py)."""
        return state[0]

    def advance(self, layers, i, state, ctx):
        torch = self.m.torch
        h, blocks, pre_h, pre_blocks, last_blocks, last_h = state
        source, passed, place, first = self.where[i]
        if first and passed == 0:          # the loop's first pass: the row and summaries every pass starts from
            pre_h, pre_blocks = h, blocks
        elif first:                        # a later pass: the last pass's own summaries and row are its messages
            last_blocks, last_h = blocks[:, pre_blocks.shape[1] + 1:], h
            h, blocks = pre_h, pre_blocks
        messages = None
        if place is not None and passed:
            messages = self.mix(last_blocks, last_h, self.messages[place])          # [t][4][hidden]
        out = self._layer(layers[i], source, h, blocks, messages, ctx)
        if source % self.size == 0:
            blocks = torch.cat([blocks, h[:, None, :]], dim=1)
        return out, blocks, pre_h, pre_blocks, last_blocks, last_h

    def end(self, layers, state, ctx):
        h, blocks = state[0], state[1]
        x = self.mix(blocks, h, self.final_query)[:, 0]
        out = self._norm(x, self.final, h.dtype)
        return out, out

    def forward(self, layers, x, ctx):
        state = self.begin(layers, x, ctx)
        for i in range(len(layers)):
            state = self.advance(layers, i, state, ctx)
        return self.end(layers, state, ctx)

    def draft_stored(self, p):
        return False   # the family's draft layer is not composed yet

RESIDUALS = {BlockAttention.name: BlockAttention}
