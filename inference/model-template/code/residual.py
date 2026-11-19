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


RESIDUALS = {PreNorm.name: PreNorm, Sandwich.name: Sandwich, Streams.name: Streams}
