# swiglu

The gated MLP: SiLU or tanh-GELU gate by the config's activation.

## How it works

The dense gated MLP: act(gate) * up through one stacked gate|up matrix, then down. The gate is the model's
activation (ComposedModel.activation, the config's hidden_act or hidden_activation): SiLU (SwiGLU), or GELU in
its tanh form.

## Use it

A layout names it as a mixer, an FFN or a `before` block (`swiglu`); the model plugin mounts [code/swiglu.py](code/swiglu.py) as its source `swiglu`. What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
