# qwen3-5-model

The model plugin of Qwen3-Next and Qwen3.5 checkpoints, dense and with a mixture of experts: config.json `model_type` `qwen3_next`, `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text`.

A hybrid decoder of gated delta-rule layers and gated softmax attention (an output gate per head, query/key norms, partial rotary positions) with zero-centred RMSNorms (1 + weight), a SwiGLU MLP or a mixture of experts with a gated shared expert, and a multi-token-prediction layer the model drafts from.

A finished model plugin, built once and kept: `crit plugins add qwen3-5-model` copies it into an engine with the models core it extends ([model-template](../../core/model-template/README.md)) and the techniques that core mounts.

| File | What it holds |
| --- | --- |
| [plugin.json](plugin.json) | The model types it serves (`needs`), the slots its step calls (`requires`, `requires_when`), and the core it extends, whose options it takes |
| [layout.json](layout.json) | The architecture: the model technique of each layer kind and FFN (`attention`, `auto`, `gated-delta`, `pre-norm`; `auto`: `moe` on the layers the config routes, `swiglu` elsewhere), the residual, the draft layer, the config values the reference assumes where config.json is silent, and the rules that rename the checkpoint's tensors to the names the techniques read |
| [plugin.py](plugin.py) | The plugin's three functions, the core's over this layout |
| [tests/](tests) | A small random checkpoint of the architecture served by an engine of this plugin, against the reference's log-probabilities and greedy choices |
