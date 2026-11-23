"""One residual stream: an RMSNorm before each block, its output added, a final norm.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `pre_norm`
and composes the residual pre-norm a layout names (README.md)."""
from ..template.residual import _vector


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

    def row(self, state):
        """The row a layer reads on entry (the stream), which a drafter technique may tap (residual.py)."""
        return state[1]

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


RESIDUALS = {PreNorm.name: PreNorm}
