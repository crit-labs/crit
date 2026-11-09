# crit plugins

Every crit engine is Core plus plugins, and nothing else. Core is what every model and every device share
(requests, sessions, admission, the concurrent step scheduler, stop rules, sampling control, chat templates, the
serving and Eval contracts); it knows no architecture and no hardware, so it is never an engine on its own. Each
part that depends on the model or the device is a plugin in a slot: the architecture in `model`, where tensors live
in `placement`, the matrix kernels in `projections`, the attention kernels in `attention`, the KV format in
`kv_cache`, the sampler in `sampler`, the drafter in `drafter`, and so on. Each plugin is a folder of the engine with
a manifest that says what it fills, needs, provides and runs on, and its own tests. Core's loader puts the plugins
together once, when the engine loads, so a plugin costs nothing per token: the model plugin calls the other plugins'
objects as it would call its own code, and CUDA graphs capture them like any other code.

That gives a build three things before Eval measures anything:

- **Each plugin is tested alone,** against a reference, by `crit plugins test`.
- **The selection is checked** when the engine loads: one plugin per slot, every need met, no conflict, one
  memory budget for all of them.
- **Each plugin's effect is measured on the one engine** by `crit run ... ablate`. The candidate is served with its
  plugins switched off one at a time and on one at a time, a short bench and [crit smoke](../smoke/README.md) run
  on each variant, and the report says which plugin pays, which slows the engine and which breaks it.

A plugin is the engine's own code. Plugins follow the engine's design: an engine can fuse two techniques into one
plugin that fills both slots, and it writes its own code in any slot. What it cannot do is run outside them: a
package is its plugins and the configuration that selects them, and crit serve, crit-eval and `crit run` run nothing
else ([the engine](#the-engine)).

crit's [Inference](../inference/README.md) is its [library of plugins](#the-library): every technique that fills a
slot is a plugin, tested against a reference, and every slot crit tests has a plain base plugin. An engine starts from
them. `crit plugins add` copies plugins into the engine with what they need, and the engine then replaces their
insides with its own device code while the same tests keep passing.

```sh
crit plugins list                                   # crit's library: its techniques and base plugins
crit plugins list --model MODEL --device cuda       # per slot, the library plugins that fit the checkpoint and the GPU
crit plugins add ENGINE/plugins token-sampling draft-trees
crit plugins add ENGINE/plugins token-sampling --update   # the library's current copy over the engine's
crit plugins list ENGINE/plugins
crit plugins check ENGINE/plugins --selection configuration.json
crit plugins test ENGINE/plugins
crit plugins variants ENGINE/plugins --selection configuration.json
crit run --id RUN_ID ablate --candidate CANDIDATE_ID
```

## A plugin

A plugin is a folder whose name is its id, under the package's `plugins/` folder:

```text
plugins/
  eagle3-drafter/
    plugin.json
    plugin.py
    tests/test_drafter.py
```

`plugin.json`:

```json
{
  "schema": "crit-plugin-v1",
  "id": "eagle3-drafter",
  "title": "EAGLE-3 drafter from the checkpoint's companion head",
  "fills": ["drafter"],
  "needs": ["taps:hidden-states"],
  "provides": ["draft:hidden-states"],
  "conflicts": ["sampler:host"],
  "options": {"width": {"type": "int", "default": 6, "min": 1, "max": 16}},
  "entry": "plugin.py:create"
}
```

| Field | Meaning |
| --- | --- |
| `fills` | The slots it fills. Two plugins in one slot exclude each other; a fused plugin fills several |
| `needs` | Facts the rest of the engine must provide: another plugin's `provides` or the engine's own facts (its GPU, its model's layout). An item that is a list is met by any one of its facts |
| `provides` | Facts it gives the plugins that need them. Two plugins never provide the same fact |
| `needs_when` | Needs that hold only where a fact does: `{"stored:nvfp4": ["cuda:fp4"]}` for a plugin that decodes NVFP4 only on FP4 tensor cores |
| `conflicts` | Facts it cannot run beside: a selection where another plugin provides one is refused |
| `requires` | Other slots its code calls, which a selection using it must fill: a `model` plugin of a hybrid architecture requires `recurrence`, one of a mixture of experts the expert slots |
| `devices` | The devices it runs on: `cuda`, `rocm`, `metal`, `vulkan`, `cpu`, or `host` for a plugin of a policy slot (`placement`, `graphs`, `expert_residency`, `expert_source`, `drafter`, `verifier`), whose code decides rather than computes. A plugin without devices is a host reference: the oracle device plugins are tested against, which no engine serves |
| `options` | Its settings, each `{type, default}` (`int`, `float`, `bool`, `str` or `choice` with `choices`; numbers may have `min` and `max`) |
| `entry` | `FILE.py:FUNCTION`, its factory: `create(context, options, parts)` returns the plugin's object |
| `reference` | The plugins its tests compare it with, which fill its slots between them, each once: the plugin it replaces (its slot's plain plugin), or for a fused plugin the separate plugins it must equal |
| `test_with` | The other plugins its tests assemble beside it, which must be in the engine too |
| `sources` | Folders of code it imports, by name: `{"sampling": "code"}` makes the plugin's `code/` folder the subpackage `.sampling`, so `plugin.py` reads `from .sampling import sampling`; `"../token-sampling/code"` imports another technique's |
| `kernels` | The registry kernels its code loads, each `{name, version, revision, fallback}` and optionally `backend` ([kernels and plugins](#kernels-and-plugins)) |
| `tests` | Its test folder (default `tests`) |

The factory gets the engine's `context` (crit serve's [model context](SERVING.md#cores-resident-adapter), so
device weights it leases through `context.device` stay loaded when the engine is built again), its checked
options and `parts`: the plugins already created, by slot. A plugin is created after every plugin whose facts it
needs, so `parts["kv_cache"]` is there for an attention plugin that needs a KV fact. A module-level
`memory(context, options)` returning `{"device": bytes, "host": bytes}` declares the workspaces the factory
allocates, which Core's loader checks against the engine's budget before any plugin is created and measures while
the factory runs ([memory](#memory)). An object with `close()` is closed when the engine closes, in the reverse of
creation; one with `facts()` adds the engine facts it knows of its own work, and one with `telemetry()` reports its
work so far ([facts and telemetry](#facts-and-telemetry)).

[crit's slot names](../core/python/crit_plugins/manifest.py) (`SLOTS`) give plugins of different engines one
meaning per slot: `model`, `placement`, `attention`, `kv_cache`, `projections`, `experts`, `expert_residency`,
`expert_source`, `router`, `recurrence`, `norms`, `drafter`, `verifier`, `sampler`, `graphs`, `embeddings`. An
engine may name others; `crit plugins list` shows the ones crit does not. Scheduling is no slot: Core's concurrent
runtime schedules every engine's steps (mixed prefill and decode, admission, the prefill chunk and batch tokens the
model's info sets), since every engine batches. `graphs` is a slot the model plugin calls: it asks the graphs plugin
which capture to run for a step's shape (the [captured prompt chunks](#captured-prompt-chunks) it runs through
Core's prompt graphs ask for it), and Core's scheduler never does.

### Every engine is plugins

Core is the part of an engine that is the same for every model and every device: requests, sessions, admission,
the concurrent step scheduler, stop rules, sampling control, chat templates and the serving and Eval contracts. It
knows no architecture and no hardware, so it is never an engine on its own: an engine is Core plus a selection of
plugins, and Core runs nothing a selection does not assemble. Every engine fills the required slots
(`REQUIRED_SLOTS`): `model`, `placement`, `projections`, `attention`, `kv_cache` and `sampler`, and any slot its
plugins `require`.

- **`model`** is the architecture: its layer plan, its weight names and the forward pass, written over the other
  slots' plugins. Its object is the step model Core's concurrent runtime runs: `step(rows)` computes one row for
  each scheduled request, prompt and decode rows of many requests together, and its info declares
  `max_concurrency` of 2 or more. One request at a time is a batch of one; a model that cannot batch is refused
  (`crit_plugins.model_problems`). A model plugin `needs` the architectures it implements, which the loader
  provides as facts from the checkpoint's `config.json` (`model_type:qwen3_5`), fills only `model`, and is created
  last, after every plugin its step calls; the facts it provides (its draft head) are the engine's.
- **`placement`** says where each tensor lives at the context the engine serves (the device, host memory, files),
  before anything loads. The base `device-placement` puts everything on the device and refuses a model that does
  not fit. The plan is the placement, not advice: the model plugin plans every tensor it keeps on the device
  (weights, KV, state), and on CUDA Core's loader refuses an engine that holds more device memory than its plans
  put there (`plan_problems`, with `PLAN_SLACK` for small tensors and workspace).

A selection for an engine is checked as a whole (`check(..., device="cuda")`): every required slot is filled, and
every plugin runs on the engine's device, or on the host in a policy slot. A host reference never serves.

### Slot interfaces

Each slot crit names has an interface in
[crit_plugins/slots.py](../core/python/crit_plugins/slots.py) (`INTERFACES`), the methods the engine calls on the
slot's plugin, with their arguments and what they return:

| Slot | Methods |
| --- | --- |
| `model` | `info`, `open`, `step`, `commit`, `retire`, `score` (and optionally `admission`, `native_rounds`) |
| `placement` | `plan` |
| `sampler` | `select`, `law`, `predictions` |
| `drafter` | `start`, `propose`, `observe`, `release` |
| `verifier` | `accept` |
| `kv_cache` | `open`, `stage`, `commit`, `rollback`, `rows`, `fork`, `release` |
| `attention` | `attend` |
| `projections` | `prepare`, `project`, `dense` |
| `router` | `route` |
| `experts` | `run` |
| `expert_source` | `add`, `matrices` |
| `expert_residency` | `plan`, `resident` |
| `recurrence` | `prefill` |
| `graphs` | `choose` |
| `embeddings` | `load`, `rows` |
| `norms` | `rms_norm_rows`, `gated_rms_norm_rows`, `rope_rows`, `act_mul_rows`, `l2_norm_rows`, `stream_sum_rows`, `stream_add_rows` |

The interface is the host form of the slot, the one its host references and every plugin's tests speak: token ids,
rows of numbers and dicts. A drafter's proposal is
`{tokens, parents, laws}` (node 0 the pending token, `parents[i] < i`, so a chain has parents `-1, 0, 1, ...`), and a
verifier's `accept` returns `{tokens, accepted, path}`. Facts the engine supplies are named the same way across
plugins: `draft:head` (the draft model's `head(paths, rows=None)`), `draft:head-rows` (it scores a subset of the
vocabulary), `taps:hidden-states`.

### Device form

A plugin that runs on a device (`cuda`, `rocm`, `metal`, `vulkan` or `cpu`) adds its slot's device form
([crit_plugins/device.py](../core/python/crit_plugins/device.py), `DEVICE_INTERFACES`) to the host form: the methods
the model plugin calls inside its batched step, with torch tensors on the engine's device. Both forms run the same
arithmetic, so the plugin's tests compare its host form with the slot's host reference on crit's cases and its device
form with its host form, and `assertImplements` checks it has both. Assembling an engine refuses a plugin without its
slot's device form as the plugin is created: the host form is an oracle, and declaring `cpu` does not make one serve.

| Slot | Device form |
| --- | --- |
| `projections` | `load(tensors, format, role=None, encode=None)` (bf16, fp8:channel, fp8:tensor, mxfp8, nvfp4 as stored, `{weight, scale, global_scale, input_scale}`; role, the matrix's part in the model's own words, where the model plugin gives one; encode, for a matrix only a drafter reads, the format the plugin re-encodes it to, recorded in its facts as `draft_encoding`), `matmul(x, matrix, out_dtype=None)` (each row's bits those of the row alone in a call of up to `invariant_rows(plugin)` rows: [row invariance](#row-invariance)) |
| `kv_cache` | `pool(layers, kv_heads, head_dim, tokens, device, dtype, *, value_dim=None)`: a KV pool (`crit_plugins.device.KV_POOL`) with `open`, `close`, `fork`, `length`, `reserve(batch)` (each token's KV slot), `write(layer, slots, keys, values)` (a slot below 0 skipped), `keep(sequence, length)` and `gather(layer, sequence, count)` (a sequence's keys and values), and `kv_heads`, `head_dim`, `value_dim` (0: a pool of keys only, with `kv:keys`); `assertImplements` writes, forks, rolls back and gathers through it |
| `attention` | `forward(layer, queries, pool, batch, scale=None)`: each token attends its sequence's KV up to its position, over a StepBatch or (with `rounds:attention`) a RoundBatch; with the attention facts also `blocks` and `block_size` (`attention:blocks`: each token reads only its listed KV blocks, chosen canonically by `crit_plugins.device.topk_blocks`), `parents` (`attention:tree`: a verify window's columns see their ancestors in the draft tree) and `lse=True` (`attention:lse`: each token's log-sum-exp beside its output) |
| `recurrence` | `pool(layers, heads, keys, values, device, sequences=None, widest=0, conv=None, key_heads=None, rows=None)`: a state pool with `open`, `close`, `fork`, `accept(sequence, count)` (any prefix of a row of up to widest + 1 tokens), `state`, `window` and `flush`; `forward(layer, query, key, value, log_decay, beta, pool, batch, geometry)` over a StepBatch or RoundBatch, each decode or verify row's bits its own; `convolve(layer, x, weight, bias, pool, batch, activation='silu')`, the causal depthwise convolution over each sequence's window in the pool (`conv=(channels, kernel)` or `(channels, kernel, dilation)`: the window the last (kernel - 1) * dilation inputs) |
| `router` | `select(logits, top, bias=None)`: each token's routed experts and their weights, by the host form's rule |
| `experts` | `load(layer, expert, gate_up, down, format='bf16')` (bfloat16 matrices, or the stored tensors as projections' `load` takes them: fp8:channel, fp8:block, nvfp4, int4, mxfp4; a plugin refuses a format it does not compute), `forward(layer, x, ids, weights)`: the routed SwiGLU block of each token row over the layer's loaded experts, FP32 out, a token's result independent of the rows beside it (a store of bundles goes through `experts:bundles`) |
| `embeddings` | `store(rows, format=None, scales=None, scale=1.0, dtype=None)` (a table on the device, or stored NVFP4 rows with their E4M3 scales and global scale, decoded on lookup), `lookup(ids)`: the rows of a step's ids; `store` returns a table handle with its own `lookup(ids)` and `close()`, so one plugin holds any number of tables; with `embeddings:host` also `open(source, width, dtype, device)` for a table in host memory or files and `prefetch(ids)` for ids known ahead |
| `sampler` | `sample(logits, params)`: one token per logits row by Core's law, inside a constrained row's `allowed` mask (`crit_plugins.Allowed.mask(device, width)`) |
| `norms` | `rms_norm(x, weight, eps, offset=0.0, residual=None, out_dtype=None)` (with a residual, `residual += x` in place and the sum normalized: `(y, residual)`; a weight `[dim]` or one per head or residual stream), `rms_norm_heads`, `rms_norm_rope`, `gated_rms_norm`, `rope` (half or interleaved pairs, partial, inverse frequencies or a table), `silu_mul`, `gelu_mul`, `sigmoid_mul`, `l2_norm`, `stream_sum` and `stream_add` (residual streams mixed and added into): inputs in their own dtype, FP32 inside, one rounding to `out_dtype`, strided views read in place, each row's bits independent of the rows beside it and of the row count |

A step's tokens are one `StepBatch` (`crit_plugins.StepBatch.build(rows, device)`): rows of consecutive tokens of one
sequence each (a prompt chunk, a decode token, a verify window), packed in row order with each token's position, so
a token's KV slot, its attention window and its recurrent state follow from the one batch every plugin reads. A
captured decode round reads a `RoundBatch` instead, and a captured prompt chunk a `PromptBatch`: the same fields at
fixed device addresses, with no host lists (`batch.round`, `batch.prompt`; [captured rounds](#captured-rounds),
[captured prompt chunks](#captured-prompt-chunks)).

#### Row invariance

A batched engine's reply must not depend on the requests beside it, so the device forms promise each row's bits alone
where a model plugin batches rows. A projections plugin's `matmul` keeps every row's bits those of the row alone in a
call of up to `invariant_rows(plugin)` rows (`crit_plugins.device.invariant_rows`: the plugin's facts' `row_invariance`
`rows`, at least `DECODE_ROWS`, 8); `assertImplements` checks it at as many rows as the facts promise (up to 32). A
model plugin splits a step's decode rows, verify windows included, into calls of at most that many rows; a longer call
is prompt rows, which may sum in another order than a shorter one unless the facts also say `prompt_rows`
(quantized-projections' `prompt_gemm: pinned` or `fixed_tiles`); even then a prompt row is not a decode row's bits,
which only the decode kernel gives ([building engines](BUILDING_ENGINES.md#3-build-check-and-compare) on preserve's
honesty check). The attention, recurrence, router, experts and norms device forms keep a row's bits whatever rows share
the call.

### Device extensions

A plugin may offer device forms beyond its slot's, each named by the fact it provides
([crit_plugins/device.py](../core/python/crit_plugins/device.py), `DEVICE_EXTENSIONS`). A plugin that provides one
of these facts has its methods (`assertImplements` checks them, and Core's loader refuses a plugin that provides the
fact without them), and a plugin that needs the fact may call them. The methods of a pool's facts (`POOL_EXTENSIONS`:
`rounds:kv_cache`, `rounds:recurrence`, `prompts:kv_cache`, `prompts:recurrence`, `kv:scales`, `kv:keys`) live on the
pool the slot's device form returns, the others on the plugin.

| Fact | Methods | What it says |
| --- | --- | --- |
| `rounds:<slot>` | `rounds:kv_cache`: `bind(slot, sequence, capacity)`, `unbind`, `commit_round(view)`; `rounds:recurrence`: `bind`, `unbind`, `move`, `commit_round(view)`; `rounds:sampler`: `reserve`, `stage`, `laws`, `draft_device`, `predictions_device`, `logprobs_device`; `rounds:attention`, `rounds:projections`, `rounds:experts`, `rounds:router`, `rounds:embeddings`, `rounds:norms`: none beyond the slot's | The slot's device form runs inside a [captured round](#captured-rounds) over a RoundBatch, capture-safely, and its per-sequence state follows the round slots |
| `prompts:<slot>` | `prompts:kv_cache`, `prompts:recurrence`: `stage_prompt(batch)`, `commit_prompt(batch)`; `prompts:attention`, `prompts:projections`, `prompts:experts`, `prompts:router`, `prompts:embeddings`, `prompts:norms`: none beyond the slot's | The slot's prompt form runs inside a [captured prompt chunk](#captured-prompt-chunks) over a PromptBatch: it reads the batch's device tensors only, launches the same work for a bucket whatever the context, and its padding changes no state |
| `draft:device` | `draft(view, head, sampler)`, `commit_round(view)` (and `reserve`, `bind`, `unbind`, `move`, `seed` where it keeps per-slot state) | A drafter that drafts inside a captured round from the model's device draft head (`draft:head-device`) |
| `verify:device` | `accept_device(view)` | A verifier that accepts inside a captured round: each row's emitted tokens and counts, after a stop and within its allowance |
| `experts:bundles` | `layout(format, hidden, inter)`, `pack(gate_up, down, format, layer=None)`, `forward_bundles(layer, x, ids, weights, store, slots, format)` | Experts as bundles in a slot store: `layout` is the one definition of a bundle, which store builders and the CPU expert tier call; with `layer`, a bundle declares that layer's activations as a load does |
| `experts:routes` | `route_outputs(layer, x, ids, store, slots, format)`, `combine(y, ids, weights, slots=None)` | A layer's routes as rows, so another tier's rows join a token's sum in route order |
| `experts:cache` | `cache(layers, experts, layout, slots, device, source, lend=None)` -> an expert cache (`store`, `plan`, `prompt`, `end_prompt`, `start_round`, `count_round`, `telemetry`) | A device expert cache in the expert_residency slot |
| `experts:store` | `open_store(layout, layers, experts, device, path=None, identity=None)`, the store's `put(layer, expert, bundle, compressed=None)` | Bundles of the experts plugin's slot layout in host memory or files, which a device expert cache fills its slots from; a store file is bound to the checkpoint's identity; with the expert_source plugin's `csf_scales` option the store keeps the named scale components CSF-compressed (its own layout, a fill program the cache runs to rebuild the slot layout) |
| `embeddings:host` | `open(source, width, dtype, device)`, `prefetch(ids)` | Lookup-table rows kept in host memory or files, staged to the device |
| `kv:scales` | `set_scales(layer, key_scale, value_scale)` | An FP8 pool's static per-layer scales (a checkpoint's k_scale and v_scale) |
| `kv:keys` | `rows_at(layer, slots)`, `slots_at(round_slots, positions)` | Rows at chosen positions: a pool of keys only (an indexer's keys) or a per-block row cache, read and written inside a captured round |
| `attention:blocks`, `attention:tree`, `attention:lse` | forward's keywords | Block-sparse attention, draft-tree masks, log-sum-exps (the attention row above) |
| `norms:fused` | `rms_norm_rope_pair`, `rope_pair`, `stream_add_norm` | One call where a layer's separate norms calls take two, each output with the separate calls' bits |

Two optional methods let a plugin's configuration speak for it. A plugin whose configuration can keep it out of a
captured round or prompt chunk (a device expert cache over a file tier, a table read through Python) has
`round_problems() -> [reason, ...]` (no arguments; `[]` when it can be captured), which the round driver and the
prompt graphs ask before any capture: a reason refuses the captures, naming the plugin and slot, and the model runs
those rows eagerly through `step()`; an answer that is not a list of non-empty strings is the plugin's error. A pool
that holds a fixed number of round rows says how many as `round_rows` (an int), and `crit_plugins.rounds.Rounds`
refuses rounds of more rows when it is built.

### Kernels and plugins

crit hosts no compute kernels. Kernels are not plugins: they are the arithmetic plugins run, a GEMM or GEMV,
attention, a norm, a scan, a fused MoE, any arithmetic a library or the kernel registry could provide. A plugin takes
its arithmetic in this order:

1. a provider: torch, cuBLASLt through torch's scaled matmul, FlashInfer, fla, CUTLASS
   ([Kernels' sources](../kernels/SOURCES.md));
2. a crit-labs kernel from the Hugging Face kernel registry (the Kernel Hub), for an operation no provider covers.
   Each is a kernel-builder project in the crit-kernels repository, uploaded as `crit-labs/NAME`. The plugin declares
   it in its plugin.json as a dependency with its pin, `"kernels": [{"name": "crit-labs/NAME", "version": 1,
   "revision": null, "fallback": "what runs without it"}]` (revision null until the kernel is published, then the
   registry commit it is pinned to; `backend` where the build is not torch's own device backend, `cpu` for a CPU
   kernel beside a CUDA torch), and loads it once, when it is created, with
   `crit_plugins.kernels.registry_kernel("crit-labs/NAME", folder=Path(__file__).parent)`, which reads that pin and
   refuses a kernel the manifest does not declare;
3. last, a plain torch fallback with the same contract, where the kernel is not there (not published, no build for
   the machine, offline): the plugin still serves correctly, and the engine's facts say so.

Every request is recorded (`crit_plugins.kernels.loaded()`), and the engine's facts carry the record under `kernels`:
per kernel the source it came from (`local`, `hub` or `fallback`), its revision and, for a fallback, why. A fresh clone
with no published kernel serves on its fallbacks and says so. To test a kernel before it is published, build its
project for the machine's GPU and torch only and set `LOCAL_KERNELS=crit-labs/NAME=/path/to/project` (several entries
joined by `:`): `registry_kernel` then loads that build through the same `kernels.get_kernel` call a published kernel
takes, its source recorded as `local`, and the registry is never asked for a repository named there. Run the plugin's
tests twice, with `LOCAL_KERNELS` (the registry path) and without it (the fallback path).

Mechanism code that runs on the device is no compute kernel and stays in its technique, built and tested there: a
sampling law bit-identical to Core's host sampler, a round's accept, draft trees and hidden-state drafters, a recurrent
commit, a host-work mailbox, an expert cache's directory and copy ring, a CPU SIMD codec. A file that mixes the two is
split. A technique keeps no list of kernels: its `operations.json` says what each device operation computes, the CPU
reference and how a kernel is verified against it ([Inference](../inference/README.md#read-and-adapt-an-entry)), and the
plugin's plugin.json is the only place a pin lives. A plugin may hold several kernels and pick one by device, shape and
format; a selection never names a kernel. Kernels are measured by the kernel workbench against a CPU reference (`crit
kernels`, `crit eval workbench`), plugins by their tests and an ablation, engines by Eval. An engine's own kernel lives
in the engine's own plugin.

### When two plugins only pay together

Plugins in different slots that depend on each other say so with facts: a KV format provides `kv:fp8-paged`, the
attention kernel that reads it needs it. When two techniques only get fast fused, for example a dequantizing
attention kernel that reads the KV pages directly, they become one plugin that fills both slots, and its
`reference` names the separate pair. Its tests compare it with them:

```python
from crit_plugins.testing import PluginTestCase


class FusedTests(PluginTestCase):
    def test_it_equals_its_reference(self):
        fused, reference = self.assemble(), self.reference()
        for batch in fixed_batches():
            self.assertClose(fused["attention"].run(batch), reference["attention"].run(batch), atol=1e-3)
```

## The engine

An engine package is a folder of two things: `plugins/` (its plugins, the plain ones in `plugins/base/`) and
`configuration.json`, whose `plugins` names the plugin of every slot:

```text
package/
  plugins/
    qwen3_5-model/            fills model, requires recurrence
    fp8-paged-attention/      fills attention
    fp8-pages/                fills kv_cache
    nvfp4-gemm/               fills projections
    gated-delta-scan/         fills recurrence
    device-sampler/           fills sampler
    eagle3-drafter/           fills drafter
    base/device-placement/    fills placement
  configuration.json
```

```json
{"plugins": {
  "use":     {"model": "qwen3_5-model", "placement": "device-placement", "projections": "nvfp4-gemm",
              "attention": "fp8-paged-attention", "kv_cache": "fp8-pages", "recurrence": "gated-delta-scan",
              "sampler": "device-sampler", "drafter": "eagle3-drafter"},
  "base":    {"attention": "sdpa-attention", "kv_cache": "bf16-pages", "drafter": null},
  "options": {"eagle3-drafter": {"width": 6}}
}}
```

`use` gives the plugin filling each slot (a fused plugin is named in every slot it fills, `null` leaves an optional
slot empty). `base` gives each slot its plain choice, the plugin an ablation falls back to when it switches the
slot's plugin off: `null` for a slot the engine can run without, and by default the plugin in use for a slot every
engine fills, which the ablation then never switches. `options` sets plugins' options. A base is a plugin the engine
serves, so it runs on the device too; a base that cannot load (16-bit weights of a model that only fits quantized)
leaves its slot's plugin unmeasured and, since the `base` variant fails with it, every plugin's `alone`.

The package has no entry of its own. Core's loader, `crit_core.engine.load(package, model)`, is the one way an
engine runs:

1. It reads the selection (the run's configuration over the package's, so an ablation's variant is a selection
   too) and the facts plugins need: from the checkpoint's `config.json` its `model_type` and `architectures`
   (`model_type:qwen3_5`, `architecture:...`) and how its matrices are stored (`stored:plain`, `stored:nvfp4`,
   `stored:fp8`, `stored:mxfp8`, `stored:mxfp4`, `stored:awq`, ...; a checkpoint of mixed formats has each format its
   layers use; `crit_core.engine.stored_facts`), and from the device its kind
   (`device:cuda`) and, for CUDA, its compute capability (`cuda:sm_120`) and tensor-core formats (`cuda:bf16`,
   `cuda:fp8`, `cuda:fp4`; `crit_core.engine.machine_facts`, from a crit eval probe device file or `nvidia-smi`).
   `crit run candidate` and `package` check against the run's device file. A fact Core cannot read is left out,
   so a plugin that needs it does not fit.
2. It checks the selection as a whole engine on the device its model plugin names (`check(..., device=)`): every
   required slot filled, every plugin's needs met, every plugin running on that device.
3. It creates the plugins (`assemble`), the model plugin last, with a `LoadContext`: the package, the checkpoint
   (`model`, `config`), the configuration, the served context (`context_tokens`), the device, whether the engine
   drafts, and under crit serve its model context (`serving`: verified model files, device leases that stay loaded
   when the engine is built again).
4. It refuses a model plugin that cannot batch (`crit_plugins.model_problems`) or falls short of what every engine
   serves (`crit_core.engine.serving_problems`): token log-probabilities with 20 alternatives, fixed-text scores
   (`scoring`, `query_scoring`, `replay`), Core's sampling fields (`temperature`, `top_p`, `top_k`, `min_p`, `seed`,
   plus the ones its sampler provides as `sampling:penalties`, `sampling:logit_bias`, `sampling:min_tokens`) and,
   when the selection drafts, plain decoding on request (`plain`).
5. Once every factory has run, it hands torch's cached, unused device memory back (the load's temporaries,
   [memory](#memory)) and runs the model plugin's object on Core's concurrent runtime (`crit_core.native.StepEngine`
   over the `crit_core_steps` library, `CRIT_CORE_STEP_LIBRARY`, which exports Core's engine C ABI version 3), whose
   facts carry `{"plugins": use}` and the plugins' own facts ([facts and telemetry](#facts-and-telemetry)). A model
   plugin with `native_rounds(library)` hands Core its captured decode rounds as the engine opens.
6. It checks that the engine is its plugins. Every plugin's factory got the other slots' plugins behind counting
   proxies (`crit_plugins.observe`: a call to a plugin, or to an object its methods made such as a KV pool or a
   loaded matrix, counts for its slot). The model plugin must have asked the placement plugin where its tensors live,
   and one short greedy request must reach every computing slot of the selection (`slot_problems`): a call through
   its proxy, or a replay of a captured graph whose capture called it (a decode round's, a native round's, a prompt
   chunk's: the round driver and the prompt graphs note the slots each capture called and count every replay as one
   call of each). A model plugin that does a slot's work itself, and fills the slot with a plugin it never calls, is
   refused. The same request is then kept to one token its greedy choice is not: a model plugin whose rows do not
   hand Core's mask to the sampler is refused too, since no constrained reply of it would keep to its language. On
   CUDA, last, the device memory the engine holds after those requests must stay within what its placement plans put
   on the device (`crit_core.engine.plan_problems`). Once the checks pass the proxies leave the hot path: every proxy
   a plugin keeps (in its attributes, containers, closures, partials, torch modules and the objects it made) is
   replaced by its target (`Assembly.unobserve`), and one held where Core cannot reach it passes calls through,
   counting nothing.

Loading compiles a package's plugin modules from their source and never writes or runs bytecode beside it, so a sealed
package keeps and runs the bytes its manifest lists, even when a script without `-B` loads it.

A crit engine is its plugins, run through Core's loader, and nothing else. crit serve reaches the engine through Core's
resident adapter (`crit_core.engine.resident`, chat-resident-v1) and crit-eval through Core's eval-v1 adapter (`crit
eval run --package DIR`, `crit_core/eval_adapter.py`); both serve the same engine, and what it serves is Core's to say
(`crit_core.engine.capabilities`). crit serve serves a run's candidates and the engines `crit pull` installed, nothing
else: a package becomes servable as a run's candidate (`crit run ... candidate` and `package`), which `crit run ...
eval` and `crit run ... serve` serve before it is sealed and crit serve once it is ([serving](SERVING.md)), so an
engine measured outside a run, a proof of a model's plugins among them, is built and sealed through crit's run tools
too. `crit run candidate` and `crit run package` check every source and package with `check_package` before they
record it, without running any of its code, and refuse anything else.

crit serve's `/v1/models` card carries the facts in `engine_facts`, and the ablation checks them to know the engine
runs the variant it was given. A package keeps its plugins in `plugins/` and its base plugins in `plugins/base/`, as
crit's library does: they are part of its bytes, like the rest of its source, so a plugin from another engine or
from crit-hub is copied in.

### Memory

The placement plan is the placement, not advice: the model plugin plans every tensor it keeps on the device (weights,
KV, state, an optional draft-only copy) through the placement plugin, and on CUDA the loader refuses an engine that
holds more device memory after its first requests than its plans and the plugins' declarations put there
(`plan_problems`, with `PLAN_SLACK`, 5% and 64 MiB, for small tensors). Three things keep that count honest:

- **Declared memory is measured.** A plugin's `memory(context, options) -> {"device", "host"}` declares the workspaces
  its factory allocates when it is made. The loader measures each factory's device growth while it runs
  (`crit_plugins.assemble.device_bytes`: torch's allocated bytes on CUDA, read without initializing it), less the
  device bytes of the placement plans made meanwhile. A declaration counts as planned only up to what the factory was
  measured to take (`Assembly.declared_device`), so a workspace declared and never allocated covers no other memory,
  and a plugin that takes more than it declares (beyond `DECLARED_SLACK`, 5% and 64 MiB: small tables, rounding, a
  math library's workspace per stream) is refused, the error naming the plugin, its declaration and the measurement.
  On a device whose memory Core cannot read, a declaration counts as given and the plan check names the plugin.
  Declare what the factory allocates: a workspace allocated only when an optional kernel loads is declared then, not
  otherwise, and one allocated at the first request belongs in a placement plan. `crit plugins test` runs the same
  check through `PluginTestCase.assemble`.
- **The load's temporaries go back.** Once every factory has run, the loader hands back the device memory torch's
  caching allocator holds for nothing (checkpoint conversions, warm-ups; `crit_plugins.assemble.release_cached`, CUDA
  only), before the engine opens and its captures are measured and before the plan check, so neither sees memory the
  engine does not keep. The engine's telemetry reports the device memory allocated and reserved at each step of the
  load (`"load"`: `made` after the factories, `released_cached`, `opened` after the release, `released_after_warm`
  after the warm requests, `checked` after the first requests; None where the device's memory cannot be read).
- **Captures are planned.** Core's round driver and prompt graphs measure the device memory their captures take (the
  warm-ups' lasting allocations, the graph pools, the capture stream's workspaces) and plan it through the placement
  plugin as their own entries, `"rounds:captures"` and `"prompts:captures"`; a placement that puts them off the device
  refuses the captures. A model plans no fixed capture workspace of its own.

### Compiling at load

Every kernel an engine runs compiles while it loads, none on a request. A plugin compiles what it launches when it is
made or at its first eager call, which is at load (a prompt chunk's or a round's warm-ups, or Core's first requests):
attention-state launches each window width class and table class a step can take once there. The round driver runs
one eager round of every (rows, width) it has no graph of, every row held (`SpeculativeRounds.warm`), and Core serves
at load one request of each kind the first request is not: a sampled one, one with log-probabilities, a plain one (an
engine that drafts), a replay with log-probabilities and a score (Eval's scored continuation), prompts of 37 and
601 tokens (past an attention window and the default buckets: the eager prompt readers, a recurrence's chunked kernel),
and two distinct ones together
(`crit_core.engine.warm_requests`; the kinds run are in the load telemetry, `"warmed"`). A kernel whose variants grow
with a request's shape keeps them in classes a load can run (a launch width rounded to its tile, page tables in
multiples of 16 pages), and its plugin's tests check it with `crit_plugins.testing.compiles()` or
`PluginTestCase.assertNoCompileAfter`; the device engine tests check whole engines at new prompt lengths and round
widths.

### Facts and telemetry

An engine's facts are what it states about itself for crit serve's `/v1/models` card (`engine_facts`), Eval and the
ablation: the model plugin's info, then each plugin object's optional `facts()` for the keys the info states none of
(the first plugin in creation order winning a key), then `plugins` (the plugin filling each slot) and `kernels` (the
registry kernels its plugins asked for and where each came from). The facts Eval reads
(`crit_core.serving.ENGINE_FACTS`) include a projections plugin's `prompt_activations`, which crit-eval prices
prefill by, and the kv_cache plugin's `kv_bytes_per_token`: the device bytes one more token of context takes in the
pools it made (every layer's keys and values at its pool's precision, their scales too, summed over every pool, a
draft layer's among them), which crit-eval sizes the KV cache and prices the KV term of its floors by. Without that
count Eval leaves the KV term out and names it missing rather than assume a precision.

A generation's result carries its rounds: Core counts every decode row as one round (`rounds`); when the request
drafted it lists per round `accepted` and `drafted`, and it sums each of the engine's named round counters (the model
info's `round_counters`, such as `expert_reads` and `expert_hits`) over its native rounds. The model plugin's
optional `request_facts(session)` (keys of `crit_core.serving.REQUEST_FACTS`), which Core asks for when a session
ends with a length or stop finish, is merged in round by round: its step() rounds' entries take their places in Core's
lists, its count of a round counter adds to the native rounds', and its other facts stand as given.

`Assembly.telemetry()` gathers each plugin object's optional `telemetry()` and the model's, section by section (a
device expert cache's `{"expert_cache": {slots, resident, lent, reads, hits, fills, fill_bytes}}`, the round
driver's `{"drafts": {rounds, drafted, offered_by_depth, accepted_by_depth}}`, the first plugin in creation order
winning a key); Core exposes it as `StepEngine.telemetry()` with its own `"core"` counters (`steps`, `rows`,
`native_rounds`, `step_seconds`) and the load's memory (`"load"`, above). crit serve's `/health` answers it as
`engine_telemetry` (None when the engine reports none or its report fails), and the eval-v1 adapter's `telemetry()`
returns it to Eval. `telemetry()` runs on a thread of its own while the engine steps and its native rounds run, so it
reads counters only: no state a step changes, no device synchronization and no wait on the device (device counters
reach host memory through each round's own copies, and a report may be a round behind), no allocation a round reads.
Raising is allowed.

### The model plugin

The `model` plugin is the architecture: it loads the checkpoint's tensors where the placement plugin puts them and
computes Core's batched step over the other slots' plugins. Its object implements the `model` interface
(`crit_plugins.slots.INTERFACES`), the step model Core's concurrent runtime drives:

| Method | What Core calls it for |
| --- | --- |
| `info()` | Core's settings (`vocabulary`, `context`, `max_concurrency` of 2 or more, `max_pending`, `max_batch_tokens`, `prefill_chunk`, `max_row_tokens`, `stop_tokens`, `sampling`, `token_logprobs`, `max_top_logprobs`, `scoring`, `score_paths`) and facts (`implementation`, `speculative`, `plain`, `replay`, `prompt_activations`, ...) |
| `open(request)` | A session for a `StepRequest` (key, prompt, output limit, sampling, stop tokens, log-probabilities, replay, plain) |
| `step(rows)` | One `StepResult` per scheduled `StepRow`: prompt chunks and decode rows of many requests in one step, each row's tokens independent of the other rows; a constrained row's `allowed` goes to the sampler as its `params["allowed"]` |
| `commit(decisions)` | Core's decision per row of the last step: the tokens kept, an end, a cancel, a discard |
| `retire(session, finish)` | The request ended; its state goes |
| `score(ids, first_target, path, queries, top_k)` | Fixed text scored alone on the engine |
| `admission(request)`, `native_rounds(library)` | Optional: page-reserving admission, decode rounds Core launches without calling `step()` ([captured rounds](#captured-rounds)) |
| `request_facts(session)`, `telemetry()` | Optional: the request's facts as its session retires, merged into its result with Core's facts of its rounds, and the engine's work so far by section ([facts and telemetry](#facts-and-telemetry)) |
| `prepare(ids, images, cancelled)` | With the fact `inputs:images` (`crit_plugins.IMAGES`): a prompt's images (`crit_plugins.Image`, decoded bytes in order) laid out on the request's thread; returns the ids to prefill, each image's positions in place of its placeholder, and the inputs `open()` gets as `StepRequest.inputs` |

Structured output and forced tool calls are Core's, for every engine. A constrained request (a `response_format`, a
`tool_choice` that forces calls) decodes plainly (`StepRequest.plain`, one token per row), and before each step Core's
matcher of its constraint (json mode or a compiled grammar over the model's byte vocabulary, `crit/constraints.h`)
puts the tokens it may take next on its row as `StepRow.allowed`, a `crit_plugins.Allowed`. The model plugin hands it
to the sampler with the row's params, the sampler masks every other token before its rules, and Core checks the token
the row produced: one outside the mask, or more than one, fails that request.

A model plugin `needs` the architectures it implements, `requires` the slots its architecture calls beyond the
required ones, fills only `model`, and is created last; the facts it `provides` (`draft:head`, `taps:hidden-states`)
are the engine's, which the drafter and verifier read. Its own tests compare it with the transformers reference on a
small checkpoint and check that a batch of requests decodes what each decodes alone.

Core puts the prompt chunks of several requests into one step. A GEMM's sums for a row depend on how many rows it
multiplies (cuBLAS picks its kernel by shape), so a model plugin that runs those chunks through its matrices
together gives a request another reply than it gets alone, and crit-eval's concurrency gate fails it: run each
prompt chunk on its own, unless the projections plugin's facts say `row_invariance` `prompt_rows` (quantized-projections
with `prompt_gemm: pinned` or `fixed_tiles`): then prompt chunks of more than `invariant_rows(projections)` rows may share a
pass, and a shorter one keeps a pass of its own. Decode rows may always share one, split into calls of at most
`invariant_rows(projections)` rows ([row invariance](#row-invariance)).

`row_invariance.prompt_exact` (quantized-projections' `prompt_gemm: tile`, with `gemv: mma`): every prompt row's
output has the bits the same row gets as a decode row, alone or batched, at every prompt length, so a token scored
in a prompt has the served token's log-probability exactly. A preserve-mode engine that must pass exact honesty
selects `prompt_gemm: tile`, `gemv: mma` (weight-only NVFP4: `activations: bf16`, `prompt: bf16`) and
softmax-attention's `prompt_reader: decode`; a Nanbeige4.3-33B engine prefills about 2,300 tok/s that way
(395 with prompt rows through the decode GEMV in 16-row calls).

A model plugin's norms, rotary positions and gated products belong in the `norms` slot (its manifest `requires` it),
named by what they compute: `h, x = norms.rms_norm(block_output, next_weight, eps, residual=x)` adds a block's output
to the residual stream and normalizes it for the next block in one call, `norms.rms_norm_rope(q, q_norm, eps,
batch.positions_t, inv_freq)` does a query's head norm and its rotary positions, `norms.silu_mul(proj.matmul(h,
gate_up))` the MLP's product over one gate|up matrix, and a model with several residual streams mixes them with
`rms_norm` (a weight per stream), `stream_sum` and `stream_add`. torch's own reductions over a row are no substitute:
they do not keep a row's bits when the row count changes.
The norms slot also computes a gated delta or state-space layer's discretized log-decay and beta gate from its
projection's rows (`softplus_decay`, host `softplus_decay_rows`). A norms plugin that provides `norms:fused` also has
fused forms of calls a layer runs back to back, each output with the bits of the separate calls: `rms_norm_rope_pair`
and `rope_pair` (a layer's queries and keys in one call) and `stream_add_norm` (a row added into the residual
streams, then the streams' norms). A model checks the fact and otherwise calls the separate forms.

A mixture-of-experts layer calls `ids, w = router.select(logits, top)`, then `experts.forward(layer, x, ids, w)` over
the experts plugin's loaded experts. Experts that do not all fit the device move as bundles: an experts plugin that
provides `experts:bundles` defines one expert as one contiguous block of bytes in its stored format (`layout`, the
one definition every store builder and CPU tier calls; `pack`), the expert_source's `open_store` holds every bundle in
host memory the device reads (or files), and a device expert cache (`experts:cache`) keeps the hot ones in the slots
of a store `[slots][bundle_bytes]`. Per routed layer `slots = cache.plan(layer, ids)` in decode and verify rows
(`store, slots = cache.prompt(layer, ids)` in a prompt chunk, `cache.end_prompt()` after the step), then
`experts.forward_bundles(layer, x, ids, w, cache.store, slots, format)`. A slot table entry is a slot >= 0, -1 for an
expert not routed, -2 for one another tier computes (its routes are skipped); with `experts:routes` a CPU tier's rows
join a token's sum in route order (`route_outputs`, then `combine`), so a token's bits do not depend on which tier
computed which route.

A hybrid model's gated-delta layer calls the recurrence slot twice: `y = recurrence.convolve(layer, qkv, conv_weight,
None, pool, batch)` (the pool made with `conv=(channels, kernel)`), then `recurrence.forward(layer, q, k, v,
log_decay, beta, pool, batch, geometry)` with q and k at the checkpoint's key heads (`key_heads=` at pool creation).
It commits with `pool.accept(sequence, count)` after each step, or in captured rounds through the pool's
`commit_round(view)`; a pool made with `widest` = the drafter's widest width commits any prefix of a verify window.

A model plugin with draft layers (an MTP layer, an EAGLE head) provides its draft head on the device
(`draft:head-device`: `catch_up(view)`, `logits(depth, tokens, view, parents=None)`, `commit_round(view)`, optionally
`rows` and `d2t`) beside its host head (`draft:head`), so a drafter with `draft:device` drafts from it inside
captured rounds. Sampled requests draft from the request's own processed law (Core's `draft` draws) and verify by
rejection sampling, which keeps the target's distribution; greedy rounds equal plain greedy decoding. The host forms
draft greedily, so a sampled request's device rounds and its host decoding emit different, equally distributed text.

### Captured rounds

A model plugin decodes its rows in captured rounds through Core's round driver, `crit_plugins.rounds.Rounds(model,
sampler, rows=, widest=, device=, drafter=, head=, verifier=, pools=[...], extra_parts=[...], parts=, uses=,
vocabulary=, sampled=, counters=, widths=, width_policy=)`, so Core launches them natively with no Python between
rounds. The model plugin
adds one method, `verify(view, batch)`: the target pass over a `RoundBatch` of R rows of W + 1 columns (each bound
row's pending token and its W drafts at its frontier, at fixed device addresses), returning the logits of every
column, `[rows][width + 1][vocabulary]` FP32. One round of R rows at width W then runs, inside one CUDA graph:

- start: `start_round(view)` on every part that has it;
- draft: the drafter's `draft(view, head, sampler)` (`draft:device`, called at every width so a stateful head never
  misses a round's tokens) writes W drafts per row from the model's draft head;
- verify and predict: the model's `verify(view, batch)`, then the sampler's `predictions_device(view, logits,
  draft_laws)` (`rounds:sampler`: greedy argmax or rejection sampling with Core's draws);
- accept: the verifier's `accept_device(view)` (`verify:device`) or the driver's chain accept writes each row's
  emitted tokens and counts, and with log-probabilities the sampler's `logprobs_device` scores them;
- advance and commit: the driver advances each row's own state from what it emitted, then the model's, every pool's,
  the drafter's and the head's `commit_round(view)` keep each row's accepted prefix (idempotent from the round's
  start: a truncation commits again);
- count and pack: `count_round(view)` on every part that has it adds the part's counts of the round into the engine's
  named counters (`view.counter(name)`), and one block of results reaches pinned host memory.

The parts are the model, sampler, drafter, head, verifier, every pool and the extra parts the model registers
(`extra_parts`: a device expert cache), each once; with no Python between native rounds, a part's per-round
bookkeeping runs in its two hooks. `bind(slot, sequence, frontier, token, allowance, capacity, stops=, params=,
session=, tokens=, outputs=)` puts a sequence into a round slot in every pool (a KV pool reserves its pages up to
`capacity`, so a captured round never allocates; pools and the drafter with `seed(slot, tokens)` get its tokens so
far; the sampler stages its rules), `unbind` and `move` keep the decoding rows packed at the front, and a row a step
does not schedule sits a round out (`view.hold`: it emits nothing, and every commit keeps it as it was). Each (rows,
width, mode) is one graph with its own memory pool, captured at load on one capture stream per device, two eager
warm-ups first; the memory the captures take is planned as `"rounds:captures"` ([memory](#memory)). Inputs a model
computes from the tokens before each column (hashed n-gram ids) come from a `TokenRing` among its pools.

A drafting model may let its rounds choose their draft width: the driver captures rounds at several widths
(`widths=`, each in 0 .. widest) and a width policy (`width_policy=`, or `crit_plugins.rounds.width_policy(name,
**options)` from a model plugin's option) picks each round's width from the rounds before it, inside the captured
round. `FixedWidth` (the default) never changes it; `StepWidth` steps one captured width up after a round that kept
every draft and one down after one that kept fewer; `RateWidth` picks the width with the most tokens per unit of
round time from each draft depth's acceptance rate and the captured rounds' measured times (`calibrate()`). The
policy writes the next width per row count into the driver's width table, which each round's pack copies to host
memory, so Core's native rounds read it before the next launch (`crit_core.native.NativeRounds` takes the engine's
widths array by address and its widest round, and the engine may rewrite the array between steps within it), and a
model's own launches ask `rounds.width(rows)`. The width changes which drafts are verified, never what a row emits:
greedy rounds emit the target's greedy tokens at any width, and sampled rounds keep the target's law. A model plugin
sizes every buffer and grant for the widest round and puts `rounds.width_facts()` (`draft_widths`, `width_policy`)
in its info; the driver's telemetry gives the policy and the rounds run at each width (`drafts.rounds_by_width`).

Before capturing, the driver checks the selection (`rounds.problems`, `crit_plugins.rounds.round_problems`): every slot
the model's verify calls (`uses`, by default every computing slot) provides `rounds:<slot>`, the sampler
`rounds:sampler`, the drafter `draft:device`, a tree drafter's verifier `verify:device` and its attention
`attention:tree`, and none of them gives a reason of its own (`round_problems()`). When one fails it refuses
(`RoundsRefused`, naming the plugin), `native_rounds` returns None and the model decodes through `step()`: correct,
slower; the driver's `problems` lists the reasons after loading. A model plugin serving rounds returns
`rounds.native_rounds(library, widths)` from its `native_rounds(library)`, declares `max_row_tokens` and
`decode_token_work` of widest + 1 and `round_counters` naming the driver's counters in its info (Core refuses to open an
engine whose settings cannot hold its rounds), and keeps `step()` for prompts and whatever rounds do not cover
(constrained requests, replays, rows past the captured widths); bind a session to a round slot when its prompt completes
and unbind it as it retires. [test_device_engine_rounds.py](../tests/test_device_engine_rounds.py) shows it: the
Llama-layout model plugin decodes the same tokens in plain and drafting rounds as through `step()`, batched as alone,
each result counting its rounds and accepted drafts.

### Captured prompt chunks

A model plugin runs a sequence's short prompt chunk as one captured graph through Core's
`crit_plugins.prompt_graphs.PromptGraphs(forward, device=, buckets=, parts=, uses=, pools=, extra_parts=)`:
`forward(batch)` is the chunk's forward over a `PromptBatch` (the chunk's tokens first, then padding to the bucket,
`valid_t` marking the real ones, positions frontier + i for all), the same prompt arithmetic as the model's eager
path, its output at a fixed address. `prepare()` captures every bucket at load (16, 64, 128 and 512 tokens by
default), and `run(sequence, frontier, tokens)` in `step()` replays the bucket the graphs plugin chooses and returns
its output, or None for the eager path. Every plugin the forward calls provides `prompts:<slot>`: its prompt form reads
the batch's device tensors only and its padding changes no state; pools stage the sequence before each replay
(`stage_prompt`) and settle it after (`commit_prompt`). A run's host work is one copy of the chunk from pinned staging.
The graphs slot's plugin chooses between the eager path and the padded buckets (eager-graphs: always eager;
recorded-selection: the arithmetic its options name, the smallest padded bucket with `allow_padding`, else the eager
path, then the least measured time among the offers that compute it; no graphs plugin: the smallest bucket that
holds the chunk). Each offer names the arithmetic it computes (`bits`): a replay gives exactly what the eager path
gives for the same padded chunk, and the unpadded eager path's output only where every plugin's prompt form keeps a
row's bits whatever the padding, so each bucket is its own and a timing never chooses which a chunk runs. The
captures' memory is planned as `"prompts:captures"`.

## Each plugin's own tests

`crit plugins test ENGINE/plugins` runs each plugin's `tests/test*.py` in a process of its own (no plugin shares
another's imports or device state) and fails a plugin that has none. `PluginTestCase` gives a test:

| Helper | What it does |
| --- | --- |
| `assemble(use, options, context, facts, use_options=)` | The plugin in its own slots, with the other plugins (`use`, their options by id in `use_options`) and engine facts it needs |
| `reference(use, options, context, facts)` | Its `reference` plugins in its slots |
| `assertImplements()` | Its object in each slot it fills has the slot's interface methods, its device form and the methods of every device extension it provides; a projections plugin keeps its rows' bits (`assertKeepsRows`) and a KV pool writes, forks, rolls back and gathers (`assertKeepsKV`) |
| `device()` | The device its tests run on, from the manifest's `devices` (a test of a CUDA-only plugin skips without CUDA) |
| `assertMatchesReference(use, options, atol=, rtol=, cases=)` | It and its reference give the same values on crit's cases for its slots (`CASES` in [slots.py](../core/python/crit_plugins/slots.py)): token ids and counts exactly, numbers within `atol + rtol * abs(expected)`. `cases` names other slots whose cases see it, as the `experts` cases see an `expert_source` |
| `assertClose(actual, expected)` | Numbers, or nested lists of them, within a tolerance |

Test against the reference the plugin replaces: the slot's plain plugin, the technique's CPU reference in
[Inference](../inference/README.md), or a fused plugin's separate pair. A device plugin keeps its library twin's
tests and tightens or loosens only the tolerance its arithmetic needs (FP32 against FP64, a quantized format against
the dense one). Tests of one plugin should run in seconds; the engine's quality as a whole is
[crit smoke](../smoke/README.md)'s and Eval's.

## The library

crit's library is its [Inference](../inference/README.md) folder, which crit's wheel ships as package data
(`crit/library/inference`), so an installed crit lists, tests and adds its plugins as a checkout does. Every technique
that fills a slot is one plugin:
its entry folder holds `plugin.json`, `plugin.py` and `tests/` beside its README and `code/`, and the plugin's id is
the technique's (`inference/draft-trees/`). A technique's variants are its options (`quantized-projections`'
`format`, `device-expert-cache`'s `policy`), and parts that only work together are one fused plugin (`draft-trees`
fills `drafter` and `verifier`). Each plugin is built on the technique's own code (its `sources`: `"code"`, and
another technique's `"../ID/code"` when it combines two), with tests against the slot's base plugin or the
technique's exact reference: one that names its `devices` adds its slot's device form and serves there, one that
names none is a host reference, the oracle that device plugins are tested against.

The base plugins in `inference/base/` are the plain plugin of each slot: what a technique is tested against and what
a selection's `base` falls back to. They are no technique, so they have no entry of their own; some are built on a
technique's code (`softmax-attention` reads attention-state's rows). `crit plugins list` and `crit plugins test`
without a folder act on the whole library, and `crit inference show ID` lists the technique's plugin and the plugins
that use its code.

| Slot | Base plugin | Techniques (the plugin each is compared with) |
| --- | --- | --- |
| `model` | | One model plugin per architecture family, checked against the transformers reference |
| `placement` | `device-placement` (every tensor on the device, runs on the host) | |
| `attention` | `softmax-attention` | `partitioned-attention` (against `softmax-attention`) |
| `kv_cache` | | `attention-state` |
| `projections` | `projections-dense` | `quantized-projections` (`format` int8, int4 or nvfp4, against `projections-dense`); `host-expert-arithmetic` (against `quantized-projections` in nvfp4) |
| `experts` | `experts-reference` | `grouped-experts` (against `experts-reference`); `heterogeneous-expert-dispatch` (with async-expert-contributions' code, bitwise against `experts-reference`); `host-expert-tier` (against `experts-reference`) |
| `expert_residency` | | `device-expert-cache` (`policy` clock or ranked); `expert-residency` |
| `expert_source` | `expert-tensors` | `ssd-expert-materialization` (exactly against `expert-tensors`) |
| `router` | `topk-router` | |
| `recurrence` | `sequential-recurrence` | `recurrent-state-recovery` (against `sequential-recurrence`) |
| `drafter` | `greedy-head-drafter` | `mtp-decode` (prompt and history copies); `draft-vocabulary-subset` (against `greedy-head-drafter`); `hidden-state-drafters` |
| `verifier` | `chain-verifier` | |
| `drafter` and `verifier` | | `draft-trees`, fused (against `greedy-head-drafter` with `chain-verifier`) |
| `sampler` | | `token-sampling`; `token-constraints` (against `token-sampling`) |
| `graphs` | `eager-graphs` | `recorded-selection` |
| `embeddings` | `memory-table-rows` | `lookup-table-rows` (against `memory-table-rows`) |
| `norms` | `torch-norms` | `fused-norms` (against `torch-norms`) |

Three entries fill no slot themselves and lend their code to plugins that do: async-expert-contributions,
expert-streamed-prefill (whose moe_block is `experts-reference`) and host-work-mailbox (heterogeneous-expert-dispatch's
in-stream join of its CPU routes). Seven entries are no plugin (`ENGINE_SERVICES` in
[cli/crit/plugins.py](../cli/crit/plugins.py)): artifact-loading, model-import-audit, checkpoint-storage,
host-memory-admission and device-speculative-round, since a model plugin calls them and no slot holds them, and
session-scheduling and concurrent-request-execution, since Core's concurrent runtime does what they describe for every
engine.
The plugin system itself is Core's, since an engine loads with it: `crit_plugins` in
[core/python](../core/python/README.md) holds the manifest, the assembly, the slot interfaces and device forms, the
registry kernel loader, Core's round driver and prompt graphs, the test helpers and the ablation's variants, and
holds no technique. The library and `crit plugins` are crit's command line
([cli/crit/plugins.py](../cli/crit/plugins.py)).

`crit plugins add ENGINE/plugins ID...` copies library plugins into an engine with their `reference` and
`test_with` plugins, so their tests run there. Each plugin's folder is copied whole to its place in the library: a
technique to `ENGINE/plugins/ID/` with its README, code and operations.json, a base plugin to `ENGINE/plugins/base/ID/`.
Code a plugin imports from another plugin of the engine is that plugin's copy, as in the library; the code of a
technique that is no plugin of the engine goes to `ENGINE/plugins/_sources/` at its path in Inference
(`_sources/expert-streamed-prefill/code`), and the copied manifest points there. A plugin the engine has already is
kept as the engine's own, and so is a source it has a copy of already; naming a plugin the engine has is refused,
unless `--update` asks to replace the engine's copy with the library's current one (where `crit plugins add` put
it; the engine's own plugins that read it or name it as their reference stay as they are). The command lists the
needs no plugin of the engine provides:
the engine's own facts (`draft:head`), or a slot whose plugin is still to be chosen. The copy is the engine's from
then on: it changes it like the rest of its source.

Every slot but `model` has library plugins that run on the device, each with its host form as before and the slot's
[device form](#device-form) in torch, over providers where one covers the arithmetic and crit-labs' registry kernels
where none does, each with its torch fallback ([kernels and plugins](#kernels-and-plugins)). Their facts say which
[device extensions](#device-extensions) they provide (each plugin.json's `provides`):

| Slot | Library plugin | Device form |
| --- | --- | --- |
| `kv_cache` | `attention-state` | A paged KV pool of every sequence: BF16, FP16, FP32 and FP8 E4M3 pages with the layer's scales (static ones from the checkpoint through `kv:scales`), FP8 E4M3 (`fp8:row`) or INT8 pages with one scale per position and KV head, copy-on-write forks, pools of keys only and rows at chosen slots (`kv:keys`); rows of up to its `window` tokens read as window rows; with `step_rows`, a step's slots and window rows' metadata in device buffers at fixed addresses, so a model replays its decode steps from CUDA graphs; round slots for captured rounds (up to `round_rows`, pages reserved at bind, slots computed and lengths committed on the device) and prompt buffers for captured prompt chunks; over FP8 and INT8 pages a fresh buffer of a prompt chunk's own 16-bit keys and values (`fresh_mib`, off by default: 0 reads the chunk from the pages, which keeps a scored continuation on the served decode's reads; facts `prompt_kv`, `prompt_kv_fresh_tokens`); its unused pages lent to a device expert cache (`spare`, `lendable()`, `on_reserve()`); pages written by the registry kernel crit-labs/paged-kv-attention where it loads; `kv_bytes_per_token` in its facts |
| `attention` | `softmax-attention` (base), `partitioned-attention` | Each row's queries over its sequence's pages read in place (one device form, attention-state's code/device_attention.py): FlashInfer's paged prefill over 16-bit and FP8 prompt rows when FlashInfer is installed (the KV split pinned, so a row does not depend on the rows beside it); decode and verify windows of every page type through the registry kernel crit-labs/paged-kv-attention's split-KV reader (KV partitions walking each row's length on the device, merged by their log-sum-exps; block lists take PIECE_PARTITIONS in every launch, so prompt, step and round rows sum a list in one order), in a step as in a captured round, so a decode row's bits are the same in both and plain, log-probability and replay requests decode what a drafted round decodes (softmax-attention's `decode: flashinfer` reads a step's window rows with FlashInfer's decode instead, for engines that decode in steps only), with tree masks and block lists; fp8:row and INT8 prompt rows through its prompt reader; otherwise the rows gathered and torch's SDPA, or a masked softmax over the page tables that captures in a CUDA graph. Captured prompt chunks read the staged prompt table on the device. `prompt_reader`: `prefill` (the default) reads prompt rows through a prompt reader, the fastest and another reduction than decode rows'; `decode` reads a step's prompt rows and captured chunks in the decode's reduction (window rows of up to 64 columns through the split-KV reader, or its torch fallback), so a scored token equals the same token served, bit for bit (facts `prompt_reads`, `served_vs_scored`) partitioned-attention is the same on one GPU, with log-sum-exps and `merge` for partials of other devices |
| `projections` | `projections-dense` (base), `quantized-projections`, `host-expert-arithmetic` (cpu) | BF16 matrices (`out_dtype` FP32 for logits, on the CPU too). quantized-projections: stored BF16, FP8 (per channel or tensor), MXFP8 and NVFP4 matrices; prompt rows through cuBLASLt (torch's scaled matmul) with FP8 or NVFP4 activations, or with 16-bit activations on the matrix decoded a chunk of rows at a time (BF16, FP16 accumulation or FP8 rows by `prompt` and `prompt_roles`; FP8 matrices' W8A16 with `fp8_activations: bf16`); with `prompt_gemm: pinned` every format's prompt GEMMs on one split-K-free cuBLASLt algorithm per shape and format, chosen at load (crit-labs/pinned-gemm), and with `fixed_tiles` the BF16 and weight-only ones through crit-labs/fixed-tile-gemm, so a prompt row's bits are the same at every prompt length, chunk and padded bucket (`row_invariance` `prompt_rows`); decode rows (up to `decode_rows`, 16) through crit-labs/mma-rows' tensor-core tile or crit-labs/decode-gemv's GEMVs at the prompt rows' activations (`decode_activations: prompt`) or 16-bit ones, each row's bits alone and batched; short prompts through the tile as far as it measured faster (`tile_rows: auto`); every call path of a matrix run once at load, so no request compiles; the prompt arithmetic of every format in its facts (`prompt_arithmetic`); draft-only matrices re-encoded at load (`encode`). host-expert-arithmetic: stored NVFP4 / INT4 matrices weight only (the FP32 order) or W4A4 / W4A8 (the integer block order) on the CPU, BF16 a row at a time, every row's bits those of the row alone at any count |
| `recurrence` | `sequential-recurrence` (base), `recurrent-state-recovery` | Every sequence's FP32 state and convolution window in a pool; decode tokens and verify windows of a step or a captured round in one column-scan launch (crit-labs/recurrent-scans where installed, else torch), the accepted prefix kept on the device (`commit`: deferred, checkpoints or recompute); prompt rows of up to `exact_rows` tokens (a stated value, 512 by default: the largest default prompt bucket; 0: none) through the same scan, so their bits are a decode's, and longer ones through fla's chunked kernel (sequential-recurrence; each of its autotuned Triton kernels pinned to one stated tile, `FLA_TILES`, since fla's autotune picks among tiles of other bits) or recurrent-scans' two-pass prefill (recurrent-state-recovery); which one a chunk runs and on which tiles follows from the configuration, never from a timing (a model whose chunk grid stays within `exact_rows` keeps every prompt row exact and a prefix's rows equal to the whole prompt's) |
| `router` | `topk-router` (base) | Softmax or sigmoid routing, correction bias, group limits, renormalization: one launch of the registry kernel crit-labs/topk-routing where it loads and the rule has no group limit, torch otherwise; capture-safe (a row of NaN logits still picks valid distinct experts, NaN ranking below -inf) |
| `experts` | `experts-reference` (base), `grouped-experts`, `host-expert-tier` (cpu), `heterogeneous-expert-dispatch` | experts-reference: the routed SwiGLU block over the experts' dense matrices, FP64 products in fixed chunks, capture-safe. grouped-experts: each expert one bundle in its stored format (bf16, fp8 channel or block, nvfp4 W4A16, or with `activations: nvfp4` each NVFP4 layer as its stored matrices declare, W4A4 where they carry input scales, int4, mxfp4), the routes grouped by expert on the device over the plugin's bundles or a residency cache's slot store, through the registry kernel crit-labs/grouped-experts or its torch fallback; a route's bits independent of the rows beside it and of decode against prompt; a long prompt chunk runs in parts, each combined into its tokens' rows as it goes, one prompt workspace serving the shorter prompt-sized calls of its shape. host-expert-tier: NVFP4 W4A4 / INT4 W4A8 experts in host-expert-arithmetic's declared integer block order on the CPU (a native tier built at load over a pinned worker pool, or torch with the same bits). heterogeneous-expert-dispatch: on CUDA, the routes an expert cache sends to the CPU (slot -2) computed by host-expert-tier's tier and joined inside the stream through host-work-mailbox, the split never changing a bit |
| `expert_residency` | `device-expert-cache` | The directory on the device (`experts:cache`): one capture-safe plan launch and one fill launch per routed layer (a slot table: slot, -1 not routed, -2 the CPU tier), fills by SM loads from a registered host store, prompt chunks through a copy-engine ring that never admits, a KV pool's unused pages as slots, clock or ranked policy, telemetry and per-row counters through `count_round` |
| `expert_source` | `expert-tensors` (base), `ssd-expert-materialization` | Bundle stores (`experts:store`, `open_store`): bundles in registered host memory with a device view, or a file tier read by a native reader behind a host tier, which a cache fills from on the host; `csf_scales` keeps NVFP4 scale components CSF-compressed in the store |
| `embeddings` | `memory-table-rows` (base), `lookup-table-rows` | A device table, or (memory-table-rows' `host`, lookup-table-rows' `open`) a table in registered host memory gathered over the link; lookup-table-rows also reads tables in files (native positioned reads, prefetch into a hot-row cache, the gather a host function a CUDA graph captures) and stored NVFP4 rows decoded on the device |
| `norms` | `torch-norms` (base), `fused-norms` | torch ops with every reduction through torch's fused rms_norm (torch's mean over a row is not row-invariant at hidden widths); fused-norms: FlashInfer for the calls whose function it computes exactly, one launch per call of the registry kernel crit-labs/fused-norms for the rest, torch-norms' arithmetic where it is not there |
| `sampler` | `token-sampling`, `token-constraints` | Core's sampling law over logits rows, Core's Philox draws; `staged(rows, device)` keeps a step's rules in device buffers and draws on the device, so a model captures its sampling in its decode graph; inside captured rounds (`rounds:sampler`) its sampling law (sampling.cu, built at load) draws drafts with their sparse laws, predicts every verify column by rejection sampling and scores the emitted tokens, or the torch law. token-constraints masks each constrained row to its allowed set first (in captured rounds with each slot's mask staged beside its rules) |
| `placement` | `device-placement` (base), `dense-layer-offload` | Every tensor on the device (`reserve_mib` kept free beside the plan), or with `off_device` the groups of those name prefixes (routed experts' bundles, lookup tables) in host memory, then files, and `slots()` to size a device expert cache; whole layers on the device while they fit and the rest in host memory, by dense-layer-offload's planner over measured rates (host policy code) |
| `graphs` | `eager-graphs` (base), `recorded-selection` | The capture a step or prompt chunk runs, from the model plugin's offers: always eager, or the least run time per use among the measured offers that compute the arithmetic its options name (host policy code) |
| `drafter`, `verifier` | `greedy-head-drafter`, `chain-verifier` (base); `mtp-decode`, `hidden-state-drafters`, `draft-vocabulary-subset`, `draft-trees` | Host policy over the engine's draft head (`draft:head`), and inside captured rounds `draft:device` / `verify:device`: the chain over the model's device draft head, copies from device token histories (mtp-decode, `override` putting long copies over the head's chain), d2t mapping and the confidence cut on the device, subset rows chosen again on the device, trees; the chain accept and the tree walk on the device |

A model plugin for a checkpoint's architecture is the engine's: the layer arithmetic of one family is no general
mechanism (see [What belongs in crit](CONTRIBUTION_CRITERIA.md)), so the library holds none. It calls the other slots'
plugins, and [test_device_engine.py](../tests/test_device_engine.py) shows the whole: the KV pool, attention,
projections, sampling, fused-norms and placement plugins copied with `crit plugins add` beside a Llama-layout model
plugin, run by Core's loader on a GPU, give transformers' log-probabilities and greedy choices for the same checkpoint,
decode the same tokens batched as alone, and decode the same tokens with torch-norms in fused-norms' place.
[test_hybrid_engine.py](../tests/test_hybrid_engine.py) does the same for a hybrid mixture of experts: a
Qwen3-Next-layout model plugin over the library's gated-delta recurrence, top-k router, experts block, expert store and
norms beside the attention, KV, projection and sampling plugins, against transformers' FP32 arithmetic on the same BF16
weights, and again with grouped-experts in the experts slot.
[test_device_engine_rounds.py](../tests/test_device_engine_rounds.py) decodes through Core's captured rounds.

One library technique plugin is still a host reference (it names no `devices`): expert-residency, a host planning
tool whose device counterpart is device-expert-cache. An engine does not serve a host reference: Core's loader refuses
one in a served selection. Host references and every plugin's host form are the oracles a device plugin is tested
against. An engine's own plugin in a slot keeps the slot's interface, options and tests, puts its own device code
behind the device form (`forward`, `matmul`, `sample`, `select`, ...), declares the device it runs on, and is
compared with the library plugin it came from.

## Ablation

`crit run --id RUN_ID ablate --candidate ID` serves a packaged candidate built from plugins under a check attempt,
then for each variant builds the engine again in the same resident host with that variant's `plugins` over the
candidate's configuration (the model's files and device weights stay loaded), checks that the engine reports the
variant's plugins, sends the run's cases and runs crit smoke. An engine built again in the host runs its first
requests cold, so each case is sent until two requests in a row take the same time per decode round (within 3%, at
most 8 requests) before its `--repetitions` (default 3); a case whose workload has no warmup is not warmed. The
variants (`crit plugins variants` lists them):

| Variant | Plugins |
| --- | --- |
| `full` | As built |
| `base` | Every slot at its base |
| `without:ID` | As built, with ID's slots at their base |
| `only:ID` | At the base, with ID in its slots (when two or more plugins switch) |

A variant that equals an earlier one is served once and lists the other names in `also`; one its plugins refuse
(an attention kernel without the KV format it reads) is reported as refused and not served. Variants are checked
against the engine's own facts as the candidate's selection shows them: what its plugins need and none of them
provides, which the engine as built supplies. `--variant NAME` serves only some of them, `--no-smoke` leaves the
suite out, and `--smoke-test TEST` runs some of its tests.

The ablation is one check attempt, so it has the run's check process cap (an hour by default) for two variants
and two more per plugin, each built again, benched and smoke-tested (crit smoke takes a minute or two on a working
engine). When they do not fit, serve `full` and the `without:` variants in one attempt and `base` with the `only:`
ones in another: each report judges the plugins from the variants it served. An attempt that runs out of time keeps
the report of the variants it served and leaves the rest unmeasured.

The report, `candidates/ID/logs/ablation-ATTEMPT.json`, holds each variant's per-case medians (decode and prefill
tokens per second, time to first token, tokens per round) and warmup requests, their geometric mean over the cases
with the time per decode round they give, its crit smoke items, and a row per plugin:

| Field | Meaning |
| --- | --- |
| `in_engine` | Each KPI's relative change, `full` against `without:ID`: what the plugin adds to the engine as built |
| `alone` | `only:ID` against `base`: what it adds on its own |
| `breaks` | crit smoke tests that fail with it and pass without it (`load` when the engine only loads without it) |
| `verdict` | `breaks`, `pays` (better by more than the noise, 3%, on some KPI and worse on none), `slows`, `trades` (better on one KPI, worse on another), `neutral` or `unmeasured` |

Where both variants of a comparison draft (more than 1.05 tokens a round), the row also compares their time per
decode round (`round_ms`). A plugin outside the drafter changes how long a round takes, while the tokens a short bench
accepts per round move with what it samples: two variants that accept 3.2 and 3.8 tokens a round can decode at the
same speed with one round 20% faster.

A short bench is not Eval: its numbers rank variants of one engine on one machine in one session, and a change
within the noise is no result. Use the ablation to decide what goes into the candidate Eval measures: drop or fix a
plugin that breaks or slows the engine, keep one that pays, and give each plan lever that fills a slot its fate
from it (that lever is one plugin, by the same id). The ablation records no check and no benchmark; its report is
the attempt's evidence.

## Tests of crit plugins

`tests/test_plugins.py` covers manifests (references, sources, test_with), selections (fused plugins, needs,
conflicts, options, cycles), the assembly (order, memory budget, closing on failure), the test runner and the
ablation's variants and verdicts. `tests/test_plugin_library.py` runs every library plugin's tests, holds every
Inference entry to being a plugin, lending one its code or being an engine service, checks that each kernel a plugin
declares is one its code asks `registry_kernel` for and that no kernel-builder project (`build.toml`) lives in crit,
and adds plugins into a temporary engine where their tests pass again. `tests/test_device_contracts.py` holds
plugins to the device extensions they provide, and `tests/test_core_rounds.py`, `tests/test_python_rounds.py` and
`tests/test_device_engine_rounds.py` cover Core's round driver.
`tests/test_run_ablate.py` packages a toy engine built from three plugins as a run's candidate, serves it through
crit serve and ablates it: drafting three tokens a round pays, the slow attention plugin slows the engine, and an
engine that does not report its plugins fails every variant.
