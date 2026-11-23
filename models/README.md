# crit Models

Models is where model plugins live: what an engine computes for one architecture, built from parts an agent picks
instead of written from scratch, and kept once built. It is built like [Inference](../inference/README.md) and
[Kernels](../kernels/README.md): a core, techniques composed on it, and what was built from them.

| Layer | What it holds | Where |
| --- | --- | --- |
| Models core | What every model plugin extends: the step model over Core's interface, the serving contracts (exact rows, prompt spans within the cancel grace, captured rounds and prompt chunks, the prompt cache), compose() and the layout format, the checkpoint readers and the shape before the weights; and `crit_models`, the mechanisms of a decoder as its config.json states them | [core/](core/README.md) |
| Model techniques | The blocks and residuals an architecture is composed of: attention kinds, recurrences, FFNs and experts, rows added into the residual, residual schemes, each with what it reads | [techniques/](techniques/README.md) |
| Cache | Finished model plugins, one an architecture: crit ships some, and the one a run builds is kept so that it is built once | [cache/](cache/README.md) |

A model plugin fills the engine's `model` slot ([Plugins](../docs/PLUGINS.md)); every other slot is an Inference
technique's or the engine's own. Core knows no model: what an architecture computes is the model plugin's, and the
general mechanism behind it is here.

## An engine's model plugin

```sh
crit models inspect MODEL                          # the finished model plugin that serves it, or what one needs
crit plugins add ENGINE/plugins qwen4-exp-model    # a finished one: the plugin, the core it extends, its techniques
```

Where no finished plugin serves the checkpoint, `inspect` names the techniques that read its config keys and tensors
and what no technique reads, and the engine's agent builds the plugin
([how](core/model-template/README.md#a-new-model-plugin)): a folder `ENGINE/plugins/ID` with a `plugin.json` that
extends the core, a `layout.json` that names the technique of each layer kind, and a class only for a layer part no
technique computes. It is checked on a tiny random checkpoint against the architecture's reference, and kept:

```sh
crit models tiny MODEL --out TINY                              # a small random checkpoint from the reference's code
crit models diff --model TINY --plugin ENGINE/plugins/ID       # the plugin's engine against the reference
crit models cache add ENGINE/plugins/ID                        # kept: crit plugins add ID puts it into any engine
```
