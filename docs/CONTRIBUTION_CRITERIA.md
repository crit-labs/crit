# What belongs in crit

crit is a vessel for any LLM on any CUDA hardware, and it has to stay maintainable
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
   different architecture (dense, MoE, hybrid recurrent, multimodal) and a different
   CUDA GPU (consumer or datacenter, another generation) either use it unchanged or
   are unaffected by it. Something that is correct everywhere but only ever helps one
   family still fails this criterion.

## Interfaces, not instances

When a model needs something crit does not offer, add the **general mechanism** to
crit and put the model's **instance** in its engine package.

| A model needs | Don't add to crit | Add to crit instead |
| --- | --- | --- |
| Its own tool-call syntax | A parser for that model | A hook the package fills (`ResidentChat.tool_format` with a `ToolFormat`) |
| A config key spelled its own way | An `if` for that family in each reader | One alias table that every reader shares |
| Special handling of some tensors | A list of that checkpoint's tensor names | A role the tensors declare, or a rule derived from the config |
| A kernel tuned for its shapes | The tuned kernel as a default | A dispatch interface, with the tuned kernel kept in the engine or the Kernel Hub |
| A faster setting on one GPU | The constant | A measurement (`crit eval probe`) the setting is derived from |

The tool-call work from the Gemma 4 run shows the pattern. A Gemma-only parser
(`gemma-call`) went into Core first and was then replaced by `ResidentChat.tool_format`,
which lets any package supply its own syntax. The second version is what belongs in
crit.

Knowledge that is shared across model families is fine to put in crit. Examples are
published checkpoint formats (NVFP4 in ModelOpt or compressed-tensors layout), config
conventions used by several model families, tool syntaxes several families write, and
layer kinds such as Mamba-2 state or sliding-window attention that several
architectures use. Keep that knowledge in data, such as tables, declared formats and
registries, rather than in branches spread through the code, and keep one copy of it.
When Python and C++ both need the same table, they read one source.

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
- Would a model of a different architecture, and a different CUDA GPU, use this
  unchanged or be unaffected by it?
- Is anything named after one model, checkpoint or GPU? If so, is that only an
  example, or does the behavior depend on it?
- Is the model-specific part, if there is one, in the engine package behind a general
  interface?
- Does the change add a second copy of something crit already has?
- Do the host suite and, for native code, the CMake suite pass?

When the answer to any of these is unclear, keep the change in the engine and record
it as toolkit feedback in the run's `summary.md`. It can move into crit once a second
model or machine shows that it generalizes.
