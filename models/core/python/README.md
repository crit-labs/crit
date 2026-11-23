# crit Models (Python)

`crit-models`: a decoder's mechanisms as its config.json states them, and the rules over them that Eval's limits and
the model plugins share (`crit_models.config`): layer kinds, looped spans, routed experts, latent attention,
recurrent state, rotary positions, the KV elements a token adds and the state that does not grow with the context.
`architecture(config)` gives a Core ModelConfig together with them. `crit_models.gguf` checks a GGUF's metadata against
the config.json beside it for these mechanisms, through Core's registry.

`crit_models.plugin` reads a model plugin's folder without a checkpoint (the layout it composes, checked against the
techniques the models core mounts and the plugin's own), and `crit_models.testing` is the harness a model plugin's
tests run on: `Manifest` (its manifest and layout on the host) and `Engines` / `Layout` (a small random checkpoint
served by an engine of the plugin, against transformers).

A family that spells a field its own way adds its key to `crit_models/config_keys.json`, never a branch in a rule.
Core reads only the generic fields (`crit_core.model.config`).

Install from a checkout with `python -m pip install -e inference/core/python -e models/core/python -e .`.

## The fields and rules

**Typed sections** beside Core's `ModelConfig` (sizes, head counts, positions,
norm epsilon, `dtype`, `rope`, `rope_by_layer_type`, `layer_types`, `moe`, `mla`,
`linear`, token ids and `quantization`; see [config.py](crit_models/config.py)). The
quantization subtree is kept raw, never interpreted; unknown keys stay reachable
through `get`. No architecture defaults are invented.

`RopeConfig` reads `rope_scaling`, else `rope_parameters`, with `rope_type`
before `type`; `theta` is the section's `rope_theta`, else the rope object's. A
rope object whose members are all objects is keyed by layer type (transformers
v5 Gemma 3): `rope_by_layer_type` then holds one `RopeLayer(layer_type, rope)`
per entry in file order, each read on its own with no fallback to the section.

**Rules Eval's limits read.** These take a config mapping (usually Core's
`text_section(...)`), return what crit eval's protocol `crit-eval-v2` rules
returned for every well-typed config, and are what Eval calls. `Architecture`
has a method for each.

| Function | Result |
| --- | --- |
| `attention_shape(config)` | `(full layers, heads, head dim, KV heads)` or `None`. Full layers count `layer_types` entries `full_attention`/`attention`, else `layers // full_attention_interval`, else all layers. Head dim is `head_dim`, else `hidden_size // heads`. |
| `kv_elements_per_token(config)` | K and V elements per token over full-attention layers; with `kv_lora_rank`, `layers * (kv_lora_rank + qk_rope_head_dim)`. |
| `fixed_state_bytes(config, kv_bytes_per_element)` | Linear-attention recurrence (FP32 state, BF16 window of `linear_conv_kernel_dim - 1`, default kernel 4) plus the KV cache of `sliding_attention` layers. |

A rule reads values in crit eval's order and only where crit eval does, and
validates each value it reads: a wrongly typed, negative or out-of-range count,
or a malformed `layer_types`, raises `ConfigError` where crit eval would compute
with it. A YaRN product of 2^64 or more, or one that is not finite, raises
`ConfigError` naming the factor's key.

**Strict helpers.** `layer_kinds(config)` returns `FULL_ATTENTION`,
`SLIDING_ATTENTION`, `LINEAR_ATTENTION` or `OTHER` per layer, from `layer_types`,
else `full_attention_interval` (as Hugging Face derives for Qwen3-Next and
Qwen3.5), else all full. `routed_layers(config)` lists the layers whose MLP routes experts and
`layer_applications(config)` the checkpoint layer each of a token's layer
applications runs (a looped span once a pass). Core's `effective_head_dim`,
`effective_kv_heads` and `require_int(key)` raise instead of rounding or
defaulting.

**Limits of the rules.** A `sliding_window` without `layer_types` (Mistral style)
counts as full attention, so KV estimates are upper bounds. `fixed_state_bytes`
counts recurrent state only for layers `layer_types` lists as `linear_attention`,
so a hybrid declared only through `full_attention_interval` reports none. Only
`text_config` nests.

