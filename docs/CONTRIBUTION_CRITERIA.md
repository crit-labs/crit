# What belongs in crit

crit is a vessel for any LLM on any hardware backend, and it has to stay maintainable
while the number of supported architectures grows. Nobody can maintain a toolkit
that carries a code path for each model family. Core is the general foundation that
most architectures, ideally all of them, build on. Engines are where one model on one
machine gets fast.

Every change to crit, whether it comes from a user, a contributor or an agent
salvaging lessons from an engine run, must meet all three criteria below. A change
that misses one of them belongs in an engine project, a run record or crit-hub, not
in crit. [Contributing](CONTRIBUTING.md) covers the mechanics (folders, tests,
licensing).

## The three criteria

1. **It objectively improves the platform.** The gain can be shown: a bug that is
   reproduced and then fixed, a test that failed before and passes after, a
   measurement taken before and after, or a capability that engines needed and
   previously had to rebuild themselves. "Seemed useful during a run" is not
   evidence. The change keeps the host and native suites green on Windows and Linux.
2. **It does not overfit to one hardware configuration or one model.** Nothing in
   it is true only for the model or machine where it was discovered. That includes
   constants tuned on one GPU, shortcuts that work for one checkpoint's tensor names,
   and docs that turn one run's measured numbers into general advice.
3. **It is applicable and valuable to almost any model and any hardware.** A
   different architecture (dense, MoE, hybrid recurrent, multimodal), a different
   GPU (consumer or datacenter, another generation) and a different backend (CUDA, ROCm,
   Metal, Vulkan, CPU) either use it unchanged or are unaffected by it. Something that is correct everywhere but only ever helps one
   family still fails this criterion.

## Interfaces, not instances

When a model needs something crit does not offer, add the **general mechanism** to
crit and put the model's **instance** in its family's package or its engine package
([a family's own pieces](#a-familys-own-pieces)).

| A model needs | Don't add to crit | Add to crit instead |
| --- | --- | --- |
| Its own tool-call syntax | A parser for that model | A hook the package fills (`ResidentChat.tool_format` with a `ToolFormat`) |
| A config key spelled its own way | An `if` for that family in each reader | One alias table that every reader shares |
| Special handling of some tensors | A list of that checkpoint's tensor names | A role the tensors declare, or a rule derived from the config |
| A kernel tuned for its shapes | The tuned kernel as a default | A kernel template whose knobs `crit kernels sweep` tunes for each model and GPU, the plugin's dispatch and its torch fallback |
| A faster setting on one GPU | The constant | A measurement (`crit eval probe`) the setting is derived from |

**Kernels are templates.** A technique holds its host reference and its mechanisms; its arithmetic comes from a
library where one covers it, and otherwise from one of crit's kernel templates (`kernels/techniques/<family>/<id>/`): source
built for the engine's device, with every constant that only changes speed (warps, tiles, splits, stages) a knob that
`crit kernels sweep` tunes for the model and GPU. A sweep's picks may ship with the kernel as the table for the
compute capability it ran on (`crit kernels ship`: `tunings/<capability>.json`, used only on that capability, after an
engine's own tuning), a measurement and never a default: a table is written from sweeps, not by hand, and names no
model. A contribution to a kernel adds a structure or a knob value, never a
constant chosen on one GPU, and never removes a structure because a new one beat it on one GPU: the sweep picks
between them on each device ([guarding against one GPU](../kernels/README.md#one-gpu)). [Kernels and plugins](PLUGINS.md#kernels-and-plugins) gives the rule: what stays in a
technique as mechanism, how a plugin declares and loads a kernel with its fallback, and what a technique keeps
instead of a list of kernels.

The tool-call work from the Gemma 4 runs shows the pattern. A Gemma-only parser
(`gemma-call`) went into Core first and was then replaced by `ResidentChat.tool_format`,
which lets any package supply its own syntax. The second version is what belongs in
crit's shared code; Gemma 4's syntax itself lives in its family's package
(`models/cache/gemma4-model/tools.py`), so the next Gemma 4 engine takes it instead of
writing it again.

## A family's own pieces

A piece only one model family uses (its tool-call syntax, a drafter published beside its checkpoints, a technique only
its layout composes) has a home in crit: that family's finished model plugin, `models/cache/<family>`
([a family's own pieces](../models/cache/README.md#a-familys-own-pieces)), where the family's name is already data. It
holds code only that family uses, and holds it to the library's standard, not an engine's:

- **Built from crit's interfaces.** A model technique of its own (`techniques/ID`: an `entry.json`, a README, `code/`,
  laid out as [models/techniques](../models/techniques/README.md)' are, its manifest mounting it as its own source),
  or a hook crit defines (`tool_format`, `reasoning_format`). It reads its tensors and config keys through the layout
  and declares them (`reads`, held by [test_technique_reads.py](../tests/test_technique_reads.py)), takes its kernels
  from crit's templates with their torch fallback, and holds no constant measured on one GPU.
- **Engine-tested.** Its package's tests serve tiny random checkpoints of the family in its two checkpoint layouts
  against the official implementation (transformers where it has one, else a host reference written from the published
  config, which the test says), as every model technique is.
- **Its family's only.** Its package's layout composes it and no other family's does. Shared code still names no
  model and imports no family package. When a second family needs a piece, the change that makes it use it moves the
  piece to [models/techniques](../models/techniques/README.md) (or `inference/techniques`) under its general name, in
  that same change.

[test_model_neutral.py](../tests/test_model_neutral.py) holds the placement (family code only inside its own package,
no shared module importing one). So an engine build that writes a family-only piece that works proposes it back to the
family's package as toolkit feedback, and the next build of that family starts from it instead of writing it again
(the next build cannot learn from a sealed engine).

## Shared knowledge

Knowledge that is shared across model families is fine to put in crit. Examples are
published checkpoint formats (NVFP4 in ModelOpt or compressed-tensors layout), config
conventions used by several model families, tool syntaxes several families write, and
layer kinds such as Mamba-2 state or sliding-window attention that several
architectures use. Such layer kinds are [model techniques](../models/techniques/README.md), which the models core
([model-template](../models/core/model-template/README.md)) composes, and a family's arrangement of them is the layout
of its [finished model plugin](../models/cache/README.md): a JSON file, not a code path. A technique names no model;
the model plugins' model_types, what the checkpoints' configs say, are the only names. Keep that knowledge in data,
such as tables, declared formats and
registries, rather than in branches spread through the code, and keep one copy of it.
When Python and C++ both need the same table, they read one source.

## Model-neutral code

crit's shared code names no model. Shared code is everything but the finished model plugins in
[models/cache](../models/cache/README.md) (which hold [a family's own pieces](#a-familys-own-pieces)): Core and the Inference techniques, Kernels, the models core, the model
techniques, the CLI, the API, Eval, the MCP server, smoke and tools, every text file there whatever its suffix
(`.py`, `.cu`, `.inc`, `.in`, `.toml`, a file with none) but Markdown. A model name, a model type or one checkpoint's
tensor spelling may appear only as data, in a model plugin's `plugin.json` and `layout.json` (and its README and
tests), and in test fixtures: the files of a `tests/` folder, and the modules of `crit_models.testing` that only tests
import (a module of it that runtime code imports is read like any other). Comments are code: evidence that names a
model goes in a doc or a run record, and a comment describes the shape or kind instead (a multimodal wrapper's
language-model section, the wavelength-banded rope type).

- **No model name.** No family name and no model_type in shared code, its comments included. A family's name counts
  at the start of a word whatever follows, in any case: `LlamaForCausalLM`, `deepseekv3`, `isQwen` and `ChatGLM` each
  name one. The only exception is an identifier a published format or a library defines and crit does not choose (a
  config value such as a rope type, a GGUF metadata value, a library function's name), listed in
  [model_neutral.json](../tests/support/model_neutral.json) with the field it is a value of and the definitions that
  hold it (the table or the function that reads the field). It is allowed only there and only as written: the whole
  of a string literal (`"llama3"`), or a library's function as that library's attribute (`fi.gemma_rmsnorm`). The
  same word elsewhere in the file, in a comment or a comparison outside those definitions, is a hit.
- **No checkpoint's own spelling.** A technique's tensor vocabulary is the transformers spelling of its layer kind
  where one exists (its own where none does), whichever family brought it. A checkpoint that spells it otherwise gets
  `names` rules in its layout; the technique never learns that spelling. Its `entry.json` declares what its code
  reads (`reads`: tensor names, config keys, layout fields), which
  [test_technique_reads.py](../tests/test_technique_reads.py) holds to the code. The guard below finds a spelling a
  layout renames in shared code, not a technique that took one model's names where transformers spells the layer kind
  otherwise: that is checked when the technique is added
  ([tensor vocabulary](../models/techniques/README.md#tensor-vocabulary), which lists the techniques not held to it
  yet).
- **Built for any model.** A technique (a model technique, an Inference technique, a kernel template) may serve one
  model today: what matters is that it is built so another model can use it unchanged. It names no model, reads its
  tensors and config keys under its own names through the layout (prefixes, `names` rules, the keys a layout maps),
  takes its kernels from tunable templates, and is tested on tiny random checkpoints.
- **Two checkpoint layouts.** Each finished model plugin declares in `tests/checkpoints.json` the checkpoint layouts
  its engine tests serve against the reference, at least two, one of them the model under another spelling that names
  rules map back (`crit_models.testing.Respelled`: another decoder prefix, other part names, layers numbered
  otherwise; the same model written as GGUF counts beside it, not instead, since the GGUF reader hands the
  techniques their own names), and every model technique is composed by such a plugin's layout. Reading the same
  model through a second spelling is what shows a technique takes its names from the layout.
- **Kernels the same way.** A fused kernel a technique needs is a kernel template with its knobs and its torch
  fallback ([Kernels are templates](#interfaces-not-instances)), never a kernel with one model's block sizes in the
  technique.

[test_model_neutral.py](../tests/test_model_neutral.py) holds these rules in crit's CI (the lint job, and the host
tests) and as a fast CPU test: `python tests/test_model_neutral.py`.

## Hardware

- Derive hardware-dependent choices from measurements (`crit eval probe`, device
  queries), never from constants measured on one GPU or CPU.
- A platform workaround (WSL2, WDDM, a driver quirk) is fine when it is detected and
  scoped to that platform, and when it does no harm elsewhere.
- A measured number in a doc is evidence, not a rule. Name the device, model and
  workload it came from, and state the general principle it supports separately.

## Docs and Inference entries

- An Inference entry states a technique and its contract for any model that fits the
  technique's assumptions. Measurements from one engine illustrate it. They never set
  thresholds or shares that a reader would copy as tuning constants.
- An entry that fills an engine slot ships as a [plugin](PLUGINS.md) (`plugin.json`, `plugin.py`, `tests/`)
  whose tests pass against the slot's base plugin, so an engine takes it with `crit plugins add` and an
  ablation can measure it. A plugin that runs on a device keeps its host form as the oracle, and its device form
  is tested against it: exactly where the slot promises (token ids, row invariance, capture), within a documented
  tolerance otherwise.
- Lessons from one run go into that run's records, its `summary.md` and its crit-hub
  entry. Only the general principle behind them goes into crit.
- Naming a model as an example is fine ("for example Gemma 4's empty thought
  channel"), but only when the rule itself does not depend on that model.

## Eval

Eval is the common yardstick, so its changes must be general and comparable. Keep the
suite, protocol, reference and timing identity comparable, and state the effect on
existing results. The reference runs the model's official implementation. Overriding
that implementation needs a rule that holds for every model it touches, recorded in
the reference's identity, with the evidence (for example, a value that `config.json`
declares and the official code ignores).

## Maintainability

- One code path per concern. When a general mechanism replaces a specific one, delete
  the specific one in the same change.
- No duplicated logic between modules. Two functions that answer the same question
  (for example, which tool format a template teaches) become one.
- Generated files, caches and run data never enter the repository.

## Checklist for a change

- Which criterion does the evidence satisfy, and where is the evidence (test,
  measurement, reproduced bug)?
- Would a model of a different architecture, a different GPU and a different backend use
  this unchanged or be unaffected by it?
- Is anything named after one model, checkpoint or GPU? In code it may not be, comments included
  ([model-neutral code](#model-neutral-code)); in a doc only as an example or as a measurement's source.
- For a model technique: does it read only the names its entry declares, through the layout, so another model's
  layout could compose it? Is it engine-tested on tiny random checkpoints in two checkpoint layouts?
- For a kernel: is every speed-only constant a knob and every device limit read from
  the device; is a new instruction behind a feature check with the other GPUs' path
  beside it and tested on the reference GPU; does a new structure sit beside the one
  it beat; does every number name its device?
- Is the model-specific part, if there is one, behind a general interface, in its family's package
  (built and tested to the library's standard) or in the engine package?
- Does the change add a second copy of something crit already has?
- Does it decide how an engine computes or schedules on the device (a level, a default, a budget, the order of
  requests)? Then it is Core's ([what Core decides](../inference/core/README.md#what-core-decides)), and every other place
  reads Core's answer.
- Do the host suite (`tests/test_model_neutral.py` among it) and, for native code, the CMake suite pass?

When the answer to any of these is unclear, keep the change in the engine and record
it as toolkit feedback in the run's `summary.md`. A family-only piece that meets the
standard above moves into its family's package; shared code takes it once a second
model or machine shows that it generalizes.
