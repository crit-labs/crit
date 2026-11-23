"""The gated MLP: SiLU or tanh-GELU gate by the config's activation.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `swiglu`
and composes the block swiglu a layout names (README.md)."""


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


BLOCKS = {"swiglu": SwiGLU}
