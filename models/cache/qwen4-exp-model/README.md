# qwen4-exp-model

The model plugin of Qwen4-Exp checkpoints: config.json `model_type` `qwen4_exp`, `qwen4_exp_text`.

A hybrid decoder over several residual streams (hyper-connections: each block reads a row mixed from the streams and its output is added to every stream by weights of its own) of gated delta-rule layers and gated block-sparse attention (an indexer picks the KV blocks a token reads past its budget), with zero-centred RMSNorms (1 + weight), a mixture of experts with a gated shared expert, and a multi-token-prediction layer the model drafts from.

A finished model plugin, built once and kept: `crit plugins add qwen4-exp-model` copies it into an engine with the models core it extends ([model-template](../../core/model-template/README.md)) and the techniques that core mounts.

| File | What it holds |
| --- | --- |
| [plugin.json](plugin.json) | The model types it serves (`needs`), the slots its step calls (`requires`, `requires_when`), and the core it extends, whose options it takes |
| [layout.json](layout.json) | The architecture: the model technique of each layer kind and FFN (`auto`, `gated-delta`, `indexed-attention`, `ngram-rows`, `streams`; `auto`: `moe` on the layers the config routes, `swiglu` elsewhere), the residual, the draft layer, the config values the reference assumes where config.json is silent, and the rules that rename the checkpoint's tensors to the names the techniques read |
| [plugin.py](plugin.py) | The plugin's three functions, the core's over this layout |
| [tests/](tests) | A small random checkpoint of the architecture served by an engine of this plugin, against the reference's log-probabilities and greedy choices |
