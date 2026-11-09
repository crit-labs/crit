# Core for Python

`crit_core` is the Python side of Core: it reads model files by name, parses
their JSON strictly, types `config.json`, checks safetensors headers and indexes,
and turns chat messages into prompt ids and generated ids back into reply text
with Core's stop rules. It ships as the `crit-core` distribution (`pip install crit-core`;
`pip install crit` brings it along), and `crit_core.__version__` is crit's version. From a checkout, put
`core/python` on `sys.path`.

`crit_plugins`, in the same distribution, is the contract every engine loads with, since every engine is Core plus
plugins: plugin manifests, crit's slots and their interfaces (the required ones and the model plugin, Core's step
model), the slots' device forms and the device extensions plugins provide (`device`), the assembly that checks a
selection as a whole engine, creates its plugins and measures their memory, the registry kernel loader (`kernels`),
Core's round driver for captured decode rounds (`rounds`) and captured prompt chunks (`prompt_graphs`), the helpers a
plugin's own tests use and the variants an ablation serves ([crit plugins](../../docs/PLUGINS.md)). Core's loader,
`crit_core.engine`, assembles a package with it and runs the result. It holds no technique and no arithmetic: the
plugins are crit's [Inference](../../inference/README.md) techniques and an engine's own.

| Module | Contents | Native counterpart |
| --- | --- | --- |
| [files.py](crit_core/files.py) | `ModelFiles` sources (`DirectoryFiles`, `MappingFiles`, `CallableFiles`), `model_files`, `FileRecord`, `ModelFilesError` | `ReadOnlyFile` ([io](../src/io/README.md)) |
| [jsonfile.py](crit_core/jsonfile.py) | `loads`, `read`, `JsonError`: strict JSON | [crit/json.h](../include/crit/json.h) |
| [model](crit_core/model/README.md) | `load_model_config`, `ModelConfig`, crit eval's config rules, `inspect_safetensors`, `inspect_model`, `TensorReader` (tensor bytes), `matrix_parts` and `layer_prefix` (stored weight matrices) | [crit/model_config.h](../include/crit/model_config.h), [crit/safetensors.h](../include/crit/safetensors.h) |
| [text](crit_core/text/README.md) | `Tokenizer`, chat templates, `GenerationConfig`, `IncrementalDetokenizer`, `PromptTokenCache`, `StopMatcher`, `TextStream`, `load_model_text` | [crit/text.h](../include/crit/text.h) |
| [engine.py](crit_core/engine.py) | Core's engine loader, the one way an engine runs: `load` (a plugin package assembled as a whole engine and run on Core's concurrent runtime), `check_package`, `capabilities`, `serving_problems`, `LoadContext`, and Core's adapters over it, `resident` (chat-resident-v1, crit serve) and `eval_v1` (eval-v1, crit-eval's `--package`, through [eval_adapter.py](crit_core/eval_adapter.py)) ([crit plugins](../../docs/PLUGINS.md#the-engine)) | |
| [native.py](crit_core/native.py) | `StepEngine`: a model plugin's step model on Core's concurrent runtime, the `crit_core_steps` library ([step engines](../src/runtime/README.md#step-engines)), with `NativeRounds` for decode rounds Core launches itself | [crit/native_engine.h](../include/crit/native_engine.h), [crit/step_engine.h](../include/crit/step_engine.h) |
| [batch.py](crit_core/batch.py) | `StepLayout`, `step_item`, `row_pages`, `PageReservations`: a batched step's rows for a model plugin | [crit/batch.h](../include/crit/batch.h) |
| [buffers.py](crit_core/buffers.py) | `plan_buffers`, `BufferRequest`, `BufferPlan` (views, slots and scratch as slices of one allocation), `ScratchArena`: [fixed buffer plans](../src/backends/cuda/README.md#fixed-buffer-plans) | [crit/device_memory.h](../include/crit/device_memory.h) |
| [constraints](crit_core/constraints) | `compile_language`, `CompilerCache`, `PreparationService`, `byte_vocabulary`, `matches`, `json_object_matches` ([guide](../src/constraints/README.md)) | [crit/constraints.h](../include/crit/constraints.h) |

Everything but `crit_core.text` uses the standard library only. The text layer
needs `tokenizers` and `jinja2` (CI pins `tokenizers==0.23.3` and
`jinja2==3.1.6`) and imports them lazily, only when a name that needs them is
first used.

## Model files

Loaders take a model directory path or a `ModelFiles` source; anything else
raises `TypeError`. Each read passes a size cap (16 MiB for `config.json` and
`generation_config.json`, as the native loaders use). A source records a
`FileRecord` (path, bytes, sha256) for each file read, in the shape of crit
run-record model manifests, and refuses a file whose bytes change between reads.
Names are relative `/` paths; absolute names, `.`/`..`, backslashes, NUL, `:` and
Windows device names are refused on every platform.

## Parity with the native parts

The same files give the same values, or are refused, in both languages: strict
JSON follows the native reader (non-finite numbers only for the config readers),
`crit_core.model` uses the native loaders' rules and limits, and `StopMatcher` and
`IncrementalDetokenizer` follow the native stop rule and `offset_decoder`.
Deliberate differences are listed in [Differences from the Python
port](../src/model/README.md#differences-from-the-python-port) and [Differences
from the C++ port](crit_core/model/README.md#differences-from-the-c-port). The
tokenizer, chat templates, OpenAI message conversion and `GenerationConfig.from_files`
exist only in Python; a native engine tokenizes through its `TextFrontend`
([text services](../src/runtime/README.md#text-services)).

## Tests

[test_core_files.py](../../tests/test_core_files.py),
[test_core_model.py](../../tests/test_core_model.py) and
[test_core_text.py](../../tests/test_core_text.py) run offline with the host
suite in `tests/`. The text tests build tiny tokenizers in-process
([core_text_fixtures.py](../../tests/core_text_fixtures.py)) and need `tokenizers`
and `jinja2`; the model tests compare the config rules with crit-eval-v2's own
([eval_v2_rules.py](../../tests/eval_v2_rules.py)). `crit_core.native`'s `StepEngine` is
tested by Core's CTest with a toy step model on `crit_core_steps`
([step_engine_check.py](../../tests/core/step_engine_check.py)), and `crit_core.engine` with the toy engine's
plugin package through Core's loader and adapters ([test_core_engine.py](../../tests/test_core_engine.py));
[test_core_batch.py](../../tests/test_core_batch.py) checks `crit_core.batch`
against the C++ library's cases, and [test_core_buffers.py](../../tests/test_core_buffers.py) `crit_core.buffers`.
