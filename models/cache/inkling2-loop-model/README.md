# inkling2-loop-model

The model plugin of Inkling-2 looped checkpoints: config.json `model_type` `inkling2_loop`, `inkling2_loop_text`.

A decoder whose middle span of layers runs several times with the same weights (each pass its own KV and convolution windows, the later passes taking messages from the pass before), joined by attention residuals over blocks of layers; sliding-window and full layers of relative attention (a bias a token computes for the positions just behind it, learned sinks a token, short convolutions over the keys and values, L2-clipped queries and keys, a gated output) with short convolutions after the attention and after the FFN, experts routed by sigmoid scores with a correction bias whose SwiGLU caps its linear gate and its up projection, a normed embedding, a head of fewer rows than its stored ones and capped logits.

A finished model plugin, built once and kept: `crit plugins add inkling2-loop-model` copies it into an engine with the models core it extends ([model-template](../../core/model-template/README.md)) and the techniques that core mounts.

| File | What it holds |
| --- | --- |
| [plugin.json](plugin.json) | The model types it serves (`needs`), the slots its step calls (`requires`, `requires_when`), and the core it extends, whose options it takes |
| [layout.json](layout.json) | The architecture: the model technique of each layer kind and FFN (`auto`, `block-attention`, `relative-attention`; `auto`: `moe` on the layers the config routes, `swiglu` elsewhere), the residual, the draft layer, the config values the reference assumes where config.json is silent, and the rules that rename the checkpoint's tensors to the names the techniques read |
| [plugin.py](plugin.py) | The plugin's three functions, the core's over this layout |
| [tests/](tests) | A small random checkpoint of the architecture served by an engine of this plugin, against the reference's log-probabilities and greedy choices |
