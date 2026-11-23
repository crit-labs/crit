# Models core

What every model plugin needs and no architecture owns: [model-template](model-template/README.md), the core a model
plugin extends, which composes a checkpoint's architecture from tested blocks by the plugin's layout and keeps the
serving contracts every model plugin must keep; and [crit_models](python/README.md), a decoder's mechanisms as its
config.json states them and the harness a model plugin's tests run whole engines on. model-template fills no slot: an
engine selects a model plugin that extends it, a finished one from [models/cache](../cache/README.md) or its own.

| Part | What it does | File |
| --- | --- | --- |
| The step model | Core's model interface over the blocks: placement, pools, `step`, `commit`, `retire`, `score`, admission, captured rounds and prompt chunks, the prompt cache, constrained, replayed and plain requests | [model.py](model-template/code/model.py) |
| The serving contracts | Exact rows and prompt spans within the cancel grace | [rows.py](model-template/code/rows.py) |
| compose() | The model plugin's object from a layout and the blocks it names, and its dry twin resolve() | [compose.py](model-template/code/compose.py) |
| The shape | The model before its weights: layer kinds, mixers, residual, draft layer, KV pools | [shape.py](model-template/code/shape.py) |
| Layer passes | Prompts run layer by layer over several steps | [passes.py](model-template/code/passes.py) |
| The checkpoint | Safetensors or GGUF read under the names the blocks read | [checkpoint.py](model-template/code/checkpoint.py), [gguf.py](model-template/code/gguf.py) |

`plugins.json` names [inference/techniques](../../inference/techniques) and [models/cache](../cache) as the folders
of plugins this one is used with, so a library plugin's tests and `crit plugins` see all three as one library.
