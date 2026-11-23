# llama-qwen3-model

The model plugin of Llama, Qwen2 and Qwen3 checkpoints and their mixtures of experts: config.json `model_type` `llama`, `qwen2`, `qwen3`, `qwen2_moe`, `qwen3_moe`.

A pre-norm decoder of grouped-query softmax attention (rotary positions, query/key norms and biases where stored, sliding-window layers where the config's layer_types name them) and a SwiGLU MLP, or a mixture of experts on the layers the config routes (a shared expert and its gate where stored).

A finished model plugin, built once and kept: `crit plugins add llama-qwen3-model` copies it into an engine with the models core it extends ([model-template](../../core/model-template/README.md)) and the techniques that core mounts.

| File | What it holds |
| --- | --- |
| [plugin.json](plugin.json) | The model types it serves (`needs`), the slots its step calls (`requires`, `requires_when`), and the core it extends, whose options it takes |
| [layout.json](layout.json) | The architecture: the model technique of each layer kind and FFN (`attention`, `auto`, `pre-norm`; `auto`: `moe` on the layers the config routes, `swiglu` elsewhere), the residual, the draft layer, the config values the reference assumes where config.json is silent, and the rules that rename the checkpoint's tensors to the names the techniques read |
| [plugin.py](plugin.py) | The plugin's three functions, the core's over this layout |
| [tests/](tests) | A small random checkpoint of the architecture served by an engine of this plugin, against the reference's log-probabilities and greedy choices |
