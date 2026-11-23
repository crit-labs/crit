# deepseek-v3-model

The model plugin of DeepSeek-V3 checkpoints: config.json `model_type` `deepseek_v3`.

A pre-norm decoder of multi-head latent attention (each position one cached latent row and one rotary key every head shares, low-rank queries, interleaved rotary pairs) with a SwiGLU MLP on its first layers and, on the rest, a mixture of experts routed by sigmoid scores within the best groups, with shared experts.

A finished model plugin, built once and kept: `crit plugins add deepseek-v3-model` copies it into an engine with the models core it extends ([model-template](../../core/model-template/README.md)) and the techniques that core mounts.

| File | What it holds |
| --- | --- |
| [plugin.json](plugin.json) | The model types it serves (`needs`), the slots its step calls (`requires`, `requires_when`), and the core it extends, whose options it takes |
| [layout.json](layout.json) | The architecture: the model technique of each layer kind and FFN (`auto`, `latent-attention`, `pre-norm`; `auto`: `moe` on the layers the config routes, `swiglu` elsewhere), the residual, the draft layer, the config values the reference assumes where config.json is silent, and the rules that rename the checkpoint's tensors to the names the techniques read |
| [plugin.py](plugin.py) | The plugin's three functions, the core's over this layout |
| [tests/](tests) | A small random checkpoint of the architecture served by an engine of this plugin, against the reference's log-probabilities and greedy choices |
