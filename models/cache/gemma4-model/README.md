# gemma4-model

The model plugin of Gemma 4 checkpoints: config.json `model_type` `gemma4`, `gemma4_text`.

A decoder whose blocks sit between two RMSNorms each (plain weights), its attention with query, key and value norms at scale 1, sliding-window layers beside full ones of another head shape and rotary table (the full layers' values the key projection's rows where the checkpoint stores no value projection), a GELU-gated MLP, and where the config routes experts a mixture of experts beside the MLP with a router that scores the residual stream; embeddings scaled by the square root of the hidden width, logits capped by a tanh.

A finished model plugin, built once and kept: `crit plugins add gemma4-model` copies it into an engine with the models core it extends ([model-template](../../core/model-template/README.md)) and the techniques that core mounts.

| File | What it holds |
| --- | --- |
| [plugin.json](plugin.json) | The model types it serves (`needs`), the slots its step calls (`requires`, `requires_when`), and the core it extends, whose options it takes |
| [layout.json](layout.json) | The architecture: the model technique of each layer kind and FFN (`attention`, `auto`, `sandwich`; `auto`: `moe` on the layers the config routes, `swiglu` elsewhere), the residual, the draft layer, the config values the reference assumes where config.json is silent, and the rules that rename the checkpoint's tensors to the names the techniques read |
| [plugin.py](plugin.py) | The plugin's three functions, the core's over this layout |
| [tests/](tests) | A small random checkpoint of the architecture served by an engine of this plugin, against the reference's log-probabilities and greedy choices |
