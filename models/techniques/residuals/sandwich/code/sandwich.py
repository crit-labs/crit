"""One residual stream, each block between two norms, a scalar a layer.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `sandwich`
and composes the residual sandwich a layout names (README.md)."""
from ..template.residual import _vector


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

    def row(self, state):
        """The row a layer reads on entry (the stream), which a drafter technique may tap (residual.py)."""
        return state[0]

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


RESIDUALS = {Sandwich.name: Sandwich}
