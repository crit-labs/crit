"""How a composed model's blocks join its residual: the layout's `residual` names one of these.

    pre-norm   one residual stream: an RMSNorm before each block, the block's output added to the stream (the norm
               after a block's output and its add are one call of the norms slot), a final norm before the head
    sandwich   one residual stream with a norm on either side of each block: the stream's norm goes into the block,
               the block's output is normed again and then added, and the stream is scaled once a layer where the
               checkpoint stores a scalar for it; a final norm before the head
    streams    several residual streams (hyper-connections): each block reads one row mixed from the streams' norms by
               weights the streams compute themselves, and its output is added to every stream by weights of its
               own; a last mix, without a block, gives the row the head reads

A residual makes the per-layer parameters a decoder layer keeps beside its mixer and FFN (`layer`), runs the layers
over a step's rows (`forward`: the head's rows and the draft feature, what a draft layer reads of the target), and
runs the checkpoint's draft layer (`draft`, `draft_input`, `draft_keys`, `draft_forward`). Every reduction is the
norms slot's and every matmul goes through the step's row groups, so a row's bits never depend on the rows beside it.

The layers run one at a time over a state the residual defines: `begin(layers, x, ctx)` gives the state of the
embedding rows before the first layer, `advance(layers, i, state, ctx)` the state after layer i, `end(layers, state,
ctx)` the head's rows and the draft feature, and `forward` is the three in turn. The state is a tuple of tensors of
one row a token (None for what the first layer has not made yet), which is all a layer leaves the next: a prompt span
run layer by layer (passes.py) keeps each chunk's state between its layers and gets the bits of the chunk run through
every layer at once. `carry(itemsize)` is the state's bytes a token, for the memory such a span plans.
"""
from .blocks import row_sum


def _vector(model, name):
    return model.ck.get(name, model.device).float()


class PreNorm:
    """One pre-norm residual stream. Layout keys (beside `block`): pre_norm and post_norm, the layer norms' names
    (input_layernorm and post_attention_layernorm by default; the layout's own pre_norm / post_norm are read too)."""

    name = "pre-norm"

    def __init__(self, model, spec):
        self.m = model
        self.pre = spec.get("pre_norm") or model.layout.get("pre_norm", "input_layernorm")
        self.post = spec.get("post_norm") or model.layout.get("post_norm", "post_attention_layernorm")
        self.width = model.hidden   # a draft feature's values: the final-normed hidden state
        self.final = None

    def anchor(self):
        """A tensor every decoder layer stores, by its name inside the layer: layer 0's finds the decoder's prefix."""
        return self.pre + ".weight"

    def names(self, prefix):
        """The tensors it reads outside the layers."""
        return [prefix + "norm.weight"]

    def load(self, prefix):
        self.final = _vector(self.m, prefix + "norm.weight")

    def layer(self, p):
        return _vector(self.m, p + self.pre + ".weight"), _vector(self.m, p + self.post + ".weight")

    def carry(self, itemsize):
        """The state's bytes a token: the stream and its norm for the next layer."""
        return 2 * self.m.hidden * itemsize

    def begin(self, layers, x, ctx):
        """The state before layer 0, (h, x): the stream x and its norm for the layer's mixer."""
        if any(layer.before for layer in layers):
            raise ValueError("model-template: a pre-norm residual has no rows added before a layer (the layout's "
                             "`before` blocks join residual streams)")
        return self.m._norm(x, layers[0].res[0]), x

    def advance(self, layers, i, state, ctx):
        """The state after layer i: each norm after a block's output advances the stream, the last one with the next
        layer's weight (the final norm's after the last layer)."""
        m, layer = self.m, layers[i]
        h, x = state
        h, x = m._norm(layer.mixer.forward(h, ctx), layer.res[1], residual=x)
        following = layers[i + 1].res[0] if i + 1 < len(layers) else self.final
        return m._norm(layer.ffn.forward(h, ctx), following, residual=x)

    def end(self, layers, state, ctx):
        """(h, feature): the final-normed rows, which are a draft layer's feature too."""
        return state[0], state[0]

    def forward(self, layers, x, ctx):
        """(h, feature) of the embedding rows x through every layer."""
        state = self.begin(layers, x, ctx)
        for i in range(len(layers)):
            state = self.advance(layers, i, state, ctx)
        return self.end(layers, state, ctx)

    # -- the draft layer: fc over [norm(embedding of the next token), norm(the target's hidden)], one decoder layer,
    #    the final norm, whose rows are the next step's feature and the draft head's input --------------------------
    def draft_stored(self, p):
        return p + "fc.weight" in self.m.ck

    def draft(self, p, encode):
        m, lp = self.m, p + "layers.0."
        return {"norm_e": _vector(m, p + "pre_fc_norm_embedding.weight"),
                "norm_h": _vector(m, p + "pre_fc_norm_hidden.weight"),
                "fc": m.matrices.load([p + "fc"], "mtp_fc", encode), "final": _vector(m, p + "norm.weight"),
                "pre": _vector(m, lp + self.pre + ".weight"), "post": _vector(m, lp + self.post + ".weight")}

    def draft_input(self, d, embedded, feature, ctx):
        m = self.m
        joined = m.torch.cat([m._norm(embedded, d["norm_e"]), m._norm(feature, d["norm_h"])], dim=-1)
        return ctx.mm(joined, d["fc"])

    def draft_keys(self, d, x, ctx):
        """The draft layer's mixer input of rows whose KV alone is written (prompt positions)."""
        return self.m._norm(x, d["pre"])

    def draft_forward(self, d, layer, x, ctx):
        """(out, feature) of the draft layer over its input rows: the rows the draft head reads, the next step's
        feature."""
        m = self.m
        h = m._norm(x, d["pre"])
        h, x = m._norm(layer.mixer.forward(h, ctx), d["post"], residual=x)
        out, _ = m._norm(layer.ffn.forward(h, ctx), d["final"], residual=x)
        return out, out


class Sandwich:
    """One residual stream, each block between two norms:

        x += norm(mixer(norm(x)))        input_layernorm, post_attention_layernorm
        x += norm(ffn(norm(x)))          pre_feedforward_layernorm, post_feedforward_layernorm
        x *= layer_scalar                where the layer stores one

    The stream before the FFN's norm is the context's `residual` while the FFN runs, for an FFN that reads it beside
    its normed input (a router that scores the stream itself). Layout keys (beside `block`): norms, the four norms'
    names in that order, and scalar, the scalar's name (layer_scalar)."""

    name = "sandwich"

    def __init__(self, model, spec):
        self.m = model
        self.norms = tuple(spec.get("norms", ("input_layernorm", "post_attention_layernorm",
                                              "pre_feedforward_layernorm", "post_feedforward_layernorm")))
        if len(self.norms) != 4:
            raise ValueError("model-template: a sandwich residual has four norms a layer: before and after the mixer, "
                             "before and after the FFN")
        self.scalar = spec.get("scalar", "layer_scalar")
        self.width = model.hidden
        self.final = None

    def anchor(self):
        return self.norms[0] + ".weight"

    def names(self, prefix):
        return [prefix + "norm.weight"]

    def load(self, prefix):
        self.final = _vector(self.m, prefix + "norm.weight")

    def layer(self, p):
        m = self.m
        scalar = float(m.ck.get(p + self.scalar, "cpu").float().reshape(-1)[0]) if p + self.scalar in m.ck else 1.0
        return tuple(_vector(m, p + name + ".weight") for name in self.norms) + (scalar,)

    def carry(self, itemsize):
        """The state's bytes a token: the stream."""
        return self.m.hidden * itemsize

    def begin(self, layers, x, ctx):
        if any(layer.before for layer in layers):
            raise ValueError("model-template: a sandwich residual has no rows added before a layer (the layout's "
                             "`before` blocks join residual streams)")
        return (x,)

    def _added(self, x, y, weight):
        """The stream with a block's normed output added, at the stream's precision."""
        return (x.float() + self.m._norm(y.to(x.dtype), weight).float()).to(x.dtype)

    def advance(self, layers, i, state, ctx):
        m, layer = self.m, layers[i]
        before, after, pre, post, scalar = layer.res
        x = state[0]
        x = self._added(x, layer.mixer.forward(m._norm(x, before), ctx), after)
        ctx.residual = x
        x = self._added(x, layer.ffn.forward(m._norm(x, pre), ctx), post)
        ctx.residual = None
        if scalar != 1.0:
            x = (x.float() * scalar).to(x.dtype)
        return (x,)

    def end(self, layers, state, ctx):
        h = self.m._norm(state[0], self.final)
        return h, h

    def forward(self, layers, x, ctx):
        """(h, feature) of the embedding rows x through every layer: the final-normed rows."""
        state = self.begin(layers, x, ctx)
        for i in range(len(layers)):
            state = self.advance(layers, i, state, ctx)
        return self.end(layers, state, ctx)

    def draft_stored(self, p):
        return False   # a draft layer over this residual is not composed yet


class _Mix:
    """One mix of the streams: the streams' norm weight [streams][hidden], the matrices its normed streams multiply
    (the low-rank down projection and, for a block's mix, the injection weights' rows after it), the up projection."""
    __slots__ = ("norm", "down", "rank", "up", "inject")

    def __init__(self, norm, down, rank, up, inject):
        self.norm, self.down, self.rank, self.up, self.inject = norm, down, rank, up, inject


class Streams:
    """Several residual streams. A mix of the streams s (S of them, each the model's hidden width) is

        n = rms_norm(s)                                     each stream by its own weight (the layout's norm offset)
        g = sigmoid(up(silu(down(n) / S)))                  one gate a value: the low-rank maps read every stream
        h = mean over the streams of g * n                  the block's input row
        w = 2 * sigmoid(inject(n) / S)                      one weight a stream: s += w * block(h)

    A layer mixes before its mixer and before its FFN; the last mix, without a block or weights, is the row the head
    reads. The streams start as S copies of the embedding row. Layout keys (beside `block`): count (the config key of
    S, hc_count by default), mixer, ffn and final (the three mixes' prefixes: attn_hyper_connection.,
    mlp_hyper_connection., hyper_connection_mixer.), norm, down, up and inject (their tensors' names: hc_norm,
    input_mix_weight_down, input_mix_weight_up, block_inject_weight). A draft layer's feature is the streams before
    the last mix, [S * hidden] a row."""

    name = "streams"

    def __init__(self, model, spec):
        m = self.m = model
        self.count = int(m.setting(spec.get("count", "hc_count")) or 0)
        if self.count < 2:
            raise ValueError(f"model-template: a streams residual has two streams or more (the config's "
                             f"{spec.get('count', 'hc_count')}), not {self.count}")
        self.attn = spec.get("mixer", "attn_hyper_connection.")
        self.ffn = spec.get("ffn", "mlp_hyper_connection.")
        self.out = spec.get("final", "hyper_connection_mixer.")
        names = {"norm": "hc_norm", "down": "input_mix_weight_down", "up": "input_mix_weight_up",
                 "inject": "block_inject_weight"}
        self.tensors = {key: spec.get(key, name) for key, name in names.items()}
        self.width = self.count * m.hidden
        self.final = None

    def anchor(self):
        return self.attn + self.tensors["norm"] + ".weight"

    def names(self, prefix):
        return [n for n in self.m.ck.names() if n.startswith(prefix + self.out)]

    def load(self, prefix):
        self.final = self._mix(prefix + self.out, inject=False)

    def layer(self, p):
        return self._mix(p + self.attn), self._mix(p + self.ffn)

    def _mix(self, p, inject=True, encode=None):
        m, names = self.m, self.tensors
        norm = _vector(m, p + names["norm"] + ".weight")
        if norm.numel() != self.width:
            raise ValueError(f"model-template: {p}{names['norm']} has {norm.numel()} weights, {self.count} streams of "
                             f"{m.hidden} take {self.width}")
        bases = [p + names["down"]] + ([p + names["inject"]] if inject else [])
        if inject and m.matrices.rows(bases[1]) != self.count:
            raise ValueError(f"model-template: {bases[1]} gives {m.matrices.rows(bases[1])} weights, one a stream is "
                             f"{self.count}")
        return _Mix(norm.view(self.count, m.hidden).contiguous(), m.matrices.together(bases, "streams_down", encode),
                    m.matrices.rows(bases[0]), m.matrices.load([p + names["up"]], "streams_up", encode), inject)

    def _each(self, x, matrices, ctx):
        """The matrices' rows side by side over rows x (matrices that read one input, stacked where stored alike)."""
        found = [ctx.mm(x, matrix) for matrix in matrices]
        return found[0] if len(found) == 1 else self.m.torch.cat(found, dim=-1)

    def mix(self, normed, mix, ctx):
        """(h, weights) of the normed streams [tokens][streams][hidden]: the block's input rows, and the weights its
        output is added to each stream with ([tokens][streams] FP32; None for the last mix)."""
        m, norms, torch = self.m, self.m.norms, self.m.torch
        tokens = normed.shape[0]
        low = self._each(normed.reshape(tokens, self.width), mix.down, ctx)
        down = low[:, :mix.rank].float() / self.count
        gate = ctx.mm(norms.sigmoid_mul(down, down, out_dtype=normed.dtype), mix.up)   # silu(v) = v * sigmoid(v)
        h = norms.stream_sum(normed, gate=gate.view(tokens, self.count, m.hidden), activation="sigmoid",
                             scale=1.0 / self.count, out_dtype=normed.dtype)
        if not mix.inject:
            return h, None
        raw = low[:, mix.rank:].float() / self.count
        return h, norms.sigmoid_mul(torch.full_like(raw, 2.0), raw, out_dtype=torch.float32)

    def _norm(self, streams, weight, pending=None):
        """The streams' norms, after a block's output joined them (pending: its rows and weights): one call where the
        norms plugin fuses the two."""
        m, norms = self.m, self.m.norms
        if pending is None:
            return norms.rms_norm(streams, weight, m.eps, offset=m.norm_offset)
        if m.fused_norms:
            return norms.stream_add_norm(streams, pending[0], pending[1], weight, m.eps, offset=m.norm_offset)[0]
        norms.stream_add(streams, pending[0], pending[1])
        return norms.rms_norm(streams, weight, m.eps, offset=m.norm_offset)

    def _layer(self, streams, mixes, layer, ctx, pending=None):
        """One layer over the streams: (the FFN's output rows, its weights), still to be added."""
        a, f = mixes
        h, weights = self.mix(self._norm(streams, a.norm, pending), a, ctx)
        pending = (layer.mixer.forward(h, ctx), weights)
        h, weights = self.mix(self._norm(streams, f.norm, pending), f, ctx)
        return layer.ffn.forward(h, ctx), weights

    def carry(self, itemsize):
        """The state's bytes a token: the streams, the last block's output and its weight a stream (FP32)."""
        return (self.count + 1) * self.m.hidden * itemsize + 4 * self.count

    def begin(self, layers, x, ctx):
        """The state before layer 0, (streams, output, weights): S copies of the embedding row, and no block's output
        waiting to be added."""
        return x[:, None, :].repeat(1, self.count, 1), None, None

    def advance(self, layers, i, state, ctx):
        """The state after layer i: the streams (advanced in place by the layers' norms) and the layer's FFN output
        with its weights, which the next layer's first norm, or the last mix, adds."""
        layer = layers[i]
        streams, pending = state[0], (None if state[1] is None else (state[1], state[2]))
        if layer.before:   # rows a layer adds into the streams before it mixes them (hashed n-gram table rows)
            if pending is not None:
                self.m.norms.stream_add(streams, pending[0], pending[1])
                pending = None
            for block in layer.before:
                streams = block.forward(streams, ctx)
        out, weights = self._layer(streams, layer.res, layer, ctx, pending)
        return streams, out, weights

    def end(self, layers, state, ctx):
        """(h, feature): the last mix's rows, which the head reads, and the streams before it, [tokens][S * hidden]."""
        streams, pending = state[0], (None if state[1] is None else (state[1], state[2]))
        h, _ = self.mix(self._norm(streams, self.final.norm, pending), self.final, ctx)
        return h, streams.reshape(streams.shape[0], self.width)

    def forward(self, layers, x, ctx):
        """(h, feature) of the embedding rows x through every layer."""
        state = self.begin(layers, x, ctx)
        for i in range(len(layers)):
            state = self.advance(layers, i, state, ctx)
        return self.end(layers, state, ctx)

    # -- the draft layer: every stream of the target's feature through fc_hidden (each normed by its own weight), the
    #    next token's normed embedding through fc_embedding added to each; one decoder layer over those streams; its
    #    own last mix. The streams before that mix are the next step's feature ----------------------------------------
    def draft_stored(self, p):
        return p + "fc_hidden.weight" in self.m.ck

    def draft(self, p, encode):
        m, lp = self.m, p + "layers.0."
        norm_h = _vector(m, p + "pre_fc_norm_hidden.weight")
        if norm_h.numel() != self.width:
            raise ValueError(f"model-template: {p}pre_fc_norm_hidden has {norm_h.numel()} weights, a feature of "
                             f"{self.count} streams takes {self.width}")
        return {"norm_e": _vector(m, p + "pre_fc_norm_embedding.weight"),
                "norm_h": norm_h.view(self.count, m.hidden).contiguous(),
                "fc_e": m.matrices.load([p + "fc_embedding"], "mtp_fc", encode),
                "fc_h": m.matrices.load([p + "fc_hidden"], "mtp_fc", encode),
                "mixes": (self._mix(lp + self.attn, encode=encode), self._mix(lp + self.ffn, encode=encode)),
                "final": self._mix(p + self.out, inject=False, encode=encode)}

    def draft_input(self, d, embedded, feature, ctx):
        m, norms = self.m, self.m.norms
        tokens = embedded.shape[0]
        normed = norms.rms_norm(feature.view(tokens, self.count, m.hidden), d["norm_h"], m.eps, offset=m.norm_offset)
        streams = ctx.mm_each(normed, d["fc_h"])
        norms.stream_add(streams, ctx.mm(m._norm(embedded, d["norm_e"]), d["fc_e"]))
        return streams

    def draft_keys(self, d, streams, ctx):
        a = d["mixes"][0]
        return self.mix(self._norm(streams, a.norm), a, ctx)[0]

    def draft_forward(self, d, layer, streams, ctx):
        pending = self._layer(streams, d["mixes"], layer, ctx)
        out, _ = self.mix(self._norm(streams, d["final"].norm, pending), d["final"], ctx)
        return out, streams.reshape(streams.shape[0], self.width)

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
    lookup), block (the config key of the block size)."""

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
            gates = {n: (m.matrices.load([f"{p}{n}_message_gate.down_proj"], "gate"),
                         m.matrices.load([f"{p}{n}_message_gate.forget_gate_proj"], "gate"),
                         m.matrices.load([f"{p}{n}_message_gate.message_gate_proj"], "gate"))
                     for n in ("q", "k", "v", "mlp")}
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
        stacked = self._rms(torch.cat([blocks, h[:, None, :]], dim=1))               # [t][n + 1][hidden]
        scores = row_sum(query[None, :, None, :] * stacked[:, None, :, :]) * self.m.hidden ** -0.5
        p = torch.softmax(scores, dim=-1)                                            # [t][queries][n + 1]
        return row_sum((p[..., None] * stacked[:, None]).transpose(-1, -2))          # summed over the n + 1

    def _gate(self, x, message, gate, ctx):
        torch = self.m.torch
        down, forget, gain = gate
        joined = torch.cat([self._rms(x), self._rms(message)], dim=-1).to(torch.bfloat16)
        g = ctx.mm(joined, down)
        f = torch.tanh(ctx.mm(g, forget, out_dtype=torch.float32).float())
        a = torch.tanh(ctx.mm(g, gain, out_dtype=torch.float32).float())
        return (1.0 - f) * x.float() + a * message.float()

    def _norm(self, x, weight, dtype):
        return self.m.norms.rms_norm(x.float().contiguous(), weight, self.m.eps).to(dtype)

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
        if messages is not None:
            inputs = [self._gate(x, messages[:, i], gates[n], ctx) for i, (x, n) in enumerate(zip(inputs, "qkv"))]
        if len(norms) == 3:
            rows = tuple(self._norm(x, w, dtype) for x, w in zip(inputs, norms))
        elif messages is None:
            rows = (self._norm(inputs[0], norms[0], dtype),) * 3
        else:
            rows = tuple(self._norm(x, norms[0], dtype) for x in inputs)
        a = layer.mixer.forward(rows, ctx)
        a = layer.mixer.post_convolution("attention", a, ctx)
        r = a.float()
        if boundary:
            blocks = torch.cat([blocks, h[:, None, :]], dim=1)
        else:
            r = h.float() + r
        x = self.mix(blocks, r, ffn)[:, 0]
        if messages is not None:
            x = self._gate(x, messages[:, 3], gates["mlp"], ctx)
        y = layer.ffn.forward(self._norm(x, post, dtype), ctx)
        y = layer.mixer.post_convolution("ffn", y, ctx)
        return (r + y.float()).to(dtype)

    def begin(self, layers, x, ctx):
        """(h, summaries, pre-loop h, pre-loop summaries, last pass's summaries, last pass's h) before layer 0."""
        if any(layer.before for layer in layers):
            raise ValueError("model-template: attention residuals have no rows added before a layer")
        if self.embed_norm is not None:
            x = self._norm(x, self.embed_norm, x.dtype)
        empty = x.new_zeros(x.shape[0], 0, x.shape[1])
        return x, empty, None, None, None, None

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



RESIDUALS = {PreNorm.name: PreNorm, Sandwich.name: Sandwich, Streams.name: Streams,
             BlockAttention.name: BlockAttention}
