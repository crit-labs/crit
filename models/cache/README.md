# Finished model plugins

Model plugins that are already built, one for each architecture, so the next engine for it takes the plugin instead
of building it again. Each folder is a whole plugin, added to an engine like any other library plugin:

```sh
crit plugins list --model MODEL --device cuda      # the model plugin that fits the checkpoint, beside the other slots'
crit plugins add ENGINE/plugins qwen4-exp-model    # the plugin, the models core it extends and its techniques' code
```

| Model plugin | Model types (config.json `model_type`) | Its tests serve, against transformers |
| --- | --- | --- |
| [llama-qwen3-model](llama-qwen3-model/README.md) | llama, qwen2, qwen3, qwen2_moe, qwen3_moe | Llama; Qwen3 with a sliding-window layer; Qwen3-MoE; the Llama checkpoint written as GGUF |
| [qwen3-5-model](qwen3-5-model/README.md) | qwen3_next, qwen3_5, qwen3_5_text, qwen3_5_moe, qwen3_5_moe_text | Qwen3-Next and Qwen3.5-MoE (gated delta, gated attention, a gated shared expert); its MTP layer drafts |
| [qwen4-exp-model](qwen4-exp-model/README.md) | qwen4_exp, qwen4_exp_text | Qwen4Exp: four residual streams, gated delta layers, hashed n-gram rows, gated block-sparse attention, routed experts with a gated shared expert, its draft layer |
| [deepseek-v3-model](deepseek-v3-model/README.md) | deepseek_v3 | DeepSeek V3: latent attention, experts routed by sigmoid scores within the best groups, shared experts |
| [gemma4-model](gemma4-model/README.md) | gemma4, gemma4_text | Gemma 4 with its mixture-of-experts block: blocks between two norms, sliding and full layers of two head shapes, a GELU-gated MLP beside GELU-gated experts |
| [inkling2-loop-model](inkling2-loop-model/README.md) | inkling2_loop, inkling2_loop_text | Inkling 2 loop: a middle span of layers run several times, attention residuals over blocks of layers, relative attention, capped SwiGLU experts |

| File | What it holds |
| --- | --- |
| `plugin.json` | The plugin's manifest ([Plugins](../../docs/PLUGINS.md)): it fills `model`, serves the config model_types its `needs` names, calls the slots in `requires` and `requires_when` (a mixture of experts' `router` and `experts`, a linear-attention layer's `recurrence`), provides the draft head where its layout has a draft layer, and extends the models core ([model-template](../core/model-template/README.md)), whose options it takes |
| `layout.json` | The architecture (`crit-model-layout-v1`, [compose.py](../core/model-template/code/compose.py)): the model technique each layer kind and FFN runs, the residual, the draft layer, the config values the family's reference assumes where config.json is silent, and the rules that rename a checkpoint's tensors to the names the techniques read |
| `plugin.py` | The plugin's three functions (`resolve`, `memory`, `create`), the core's over this layout; and, where the layout runs a block or residual no library technique computes, that technique's class |
| `tests/` | A small random checkpoint of the architecture served by an engine of the plugin, against the reference's log-probabilities and greedy choices ([crit_models.testing](../core/python/crit_models/testing/__init__.py)), in at least two checkpoint layouts that `tests/checkpoints.json` lists with their test classes: transformers' safetensors, the same model as GGUF (`GgufEngines`), or under a release's own tensor names that the layout's `names` map back (`Respelled`) |
| `README.md` | What the architecture is and which techniques compose it |

A model_type is the only place a family's name appears: it is what the checkpoint's config says, used to match it.

## Keeping a plugin a run built

A model plugin an engine's agent built ([a new model plugin](../core/model-template/README.md#a-new-model-plugin))
lies in its engine at `ENGINE/plugins/ID`. `crit models cache add ENGINE/plugins/ID` copies it as it is into the
user's cache (`CRIT_CACHE_HOME/models/ID`), and from then on `crit plugins list` shows it and `crit plugins add ID`
puts it into any engine, with the core it extends and the library plugins and code it uses. So an architecture is
built once on a machine. `crit models cache add ENGINE/plugins/ID --library` writes it into this folder of a checkout
instead, its sources pointing at the library: how a plugin becomes one crit ships.
