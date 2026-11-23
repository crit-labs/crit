"""Several residual streams (hyper-connections) mixed by weights the streams compute themselves.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `streams`
and composes the residual streams a layout names (README.md)."""
from ..template.residual import _vector


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


RESIDUALS = {Streams.name: Streams}
