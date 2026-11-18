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
crit plugins memory ENGINE --model MODEL            # what each device buffer really holds (the memory audit)
crit plugins decisions ENGINE --model MODEL         # record the choices its plugins time at load (measured decisions)
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
| `kernels` | crit's kernels its code loads, each `{name, fallback}` and optionally `backend` ([kernels and plugins](#kernels-and-plugins)) |
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
| `projections` | `load(tensors, format, role=None, encode=None)` (bf16, fp8:channel, fp8:tensor, mxfp8, nvfp4 as stored, `{weight, scale, global_scale, input_scale}`; awq and gptq as stored, `{qweight, qzeros, scales}` (gptq also `g_idx`); role, the matrix's part in the model's own words, where the model plugin gives one; encode, for a matrix only a drafter reads, the format the plugin re-encodes it to, recorded in its facts as `draft_encoding`), `matmul(x, matrix, out_dtype=None, prompt=False)` (each row's bits those of the row alone in a call of up to `invariant_rows(plugin)` rows: [row invariance](#row-invariance); `prompt=True` for a prompt chunk, which takes the prompt rows' arithmetic at every length, so a short prompt is not multiplied as decode rows) |
| `kv_cache` | `pool(layers, kv_heads, head_dim, tokens, device, dtype, *, value_dim=None)`: a KV pool (`crit_plugins.device.KV_POOL`) with `open`, `close`, `fork`, `length`, `reserve(batch)` (each token's KV slot), `write(layer, slots, keys, values)` (a slot below 0 skipped), `keep(sequence, length)` and `gather(layer, sequence, count)` (a sequence's keys and values), and `kv_heads`, `head_dim`, `value_dim` (0: a pool of keys only, with `kv:keys`); `assertImplements` writes, forks, rolls back and gathers through it |
| `attention` | `forward(layer, queries, pool, batch, scale=None)`: each token attends its sequence's KV up to its position, over a StepBatch or (with `rounds:attention`) a RoundBatch; with the attention facts also `blocks` and `block_size` (`attention:blocks`: each token reads only its listed KV blocks, chosen canonically by `crit_plugins.device.topk_blocks`), `parents` (`attention:tree`: a verify window's columns see their ancestors in the draft tree), `lse=True` (`attention:lse`: each token's log-sum-exp beside its output) , `sinks` (`attention:sinks`: each query head's sink logit, one more position with no value in every token's softmax), `sliding` (`attention:sliding`: a sliding window, a token's last positions) and `latent=R` (`attention:latent`: latent attention over a pool of keys only whose rows are each position's latent then its rotary key, the latent the value too) |
| `recurrence` | `pool(layers, heads, keys, values, device, sequences=None, widest=0, conv=None, key_heads=None, rows=None)`: a state pool with `open`, `close`, `fork`, `accept(sequence, count)` (any prefix of a row of up to widest + 1 tokens), `state`, `window` and `flush`; `forward(layer, query, key, value, log_decay, beta, pool, batch, geometry)` over a StepBatch or RoundBatch, each decode or verify row's bits its own; `convolve(layer, x, weight, bias, pool, batch, activation='silu')`, the causal depthwise convolution over each sequence's window in the pool (`conv=(channels, kernel)` or `(channels, kernel, dilation)`: the window the last (kernel - 1) * dilation inputs) |
| `router` | `select(logits, top, bias=None)`: each token's routed experts and their weights, by the host form's rule |
| `experts` | `load(layer, expert, gate_up, down, format='bf16')` (bfloat16 matrices, or the stored tensors as projections' `load` takes them: fp8:channel, fp8:block, nvfp4, int4, mxfp4; a plugin refuses a format it does not compute), `forward(layer, x, ids, weights)`: the routed SwiGLU block of each token row over the layer's loaded experts, FP32 out, a token's result independent of the rows beside it (a store of bundles goes through `experts:bundles`) |
| `embeddings` | `store(rows, format=None, scales=None, scale=1.0, dtype=None, device=None)` (a table on the device, or stored NVFP4 rows with their E4M3 scales and global scale, decoded on lookup; `device` the one it serves, by default the one `rows` are on, so a host table's rows read on the CPU need not pass through the device), `lookup(ids)`: the rows of a step's ids; `store` returns a table handle with its own `lookup(ids)` and `close()`, so one plugin holds any number of tables; with `embeddings:host` also `open(source, width, dtype, device)` for a table in host memory or files and `prefetch(ids)` for ids known ahead |
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
model plugin splits a step's decode rows, verify windows included, into calls of at most that many rows, and passes each
prompt chunk with `prompt=True`, so a prompt of a few tokens is not multiplied as decode rows (crit smoke's causal test
compares a prompt's prefix with the whole prompt); a longer call is prompt rows, which may sum in another order than a shorter one unless the facts also say `prompt_rows`
(quantized-projections' `prompt_gemm: pinned` or `fixed_tiles`); even then a prompt row is not a decode row's bits,
which only the decode kernel gives ([building engines](BUILDING_ENGINES.md#3-build-check-and-compare) on preserve's
honesty check). The attention, recurrence, router, experts and norms device forms keep a row's bits whatever rows share
the call.

#### Measured decisions

Nothing an engine computes may depend on how fast something ran in this process. A plugin that chooses between two
ways by timing both as it loads (who computes a missed expert, how a layer's work splits) resolves a near-tie
differently in the next process, and where the two ways give other bits the same request gets another text there:
no repeat check inside one process sees it. A choice that only changes speed (a spin time, a thread count) is free.
One that can change a bit is a decision, settled through
[crit_plugins.decisions](../core/python/crit_plugins/decisions.py):

```python
from crit_plugins import decisions
way = decisions.settle("my-plugin", "misses",
                       key={"device": name, "hidden": hidden, "format": format},   # what else the choice depends on
                       measure=lambda: {"cpu_us": cpu, "link_us": link},           # runs only when no record answers
                       decide=lambda found: "cpu" if found["cpu_us"] < 0.8 * found["link_us"] else "link")
```

`settle` replays the package's record where it holds the decision with the same key and measures otherwise.
`crit plugins decisions PACKAGE --model MODEL` loads the engine once and writes the record into the package; the
configuration names it (`"decisions": "decisions.json"`), as it names the kernel tuning, and Core's loader applies it
before any plugin is made, so the package Eval measured decides as the package served does. The engine's telemetry
carries every decision under `decisions` (key, choice, `recorded` or `measured`, the measurement) with one digest
over the choices: two processes whose texts part show different digests when a decision is why. Kernel choices have
their own record, the [tuning tables](#kernels-and-plugins) and a prompt GEMM's
[recorded algorithm](../inference/recorded-selection/README.md); this is the same rule for everything else. The
same holds while an engine serves: a placement or a width that follows a measurement changes at a boundary the
requests fix (a round's end, a count of rounds), never when a copy or a timer happens to finish.

### Device extensions

A plugin may offer device forms beyond its slot's, each named by the fact it provides
([crit_plugins/device.py](../core/python/crit_plugins/device.py), `DEVICE_EXTENSIONS`). A plugin that provides one
of these facts has its methods (`assertImplements` checks them, and Core's loader refuses a plugin that provides the
fact without them), and a plugin that needs the fact may call them. The methods of a pool's facts (`POOL_EXTENSIONS`:
`rounds:kv_cache`, `rounds:recurrence`, `prompts:kv_cache`, `prompts:recurrence`, `kv:scales`, `kv:keys`, `kv:sliding`) live on the
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
| `kv:keys` | `rows_at(layer, slots)`, `slots_at(round_slots, positions)` | Rows at chosen positions: a pool of keys only (an indexer's keys) or a per-block row cache, read and written inside a captured round. `rows_at` gathers a copy; a kernel reads the rows where they lie, from `layer_pages(layer)` through the slots (crit's sparse-indexer takes them as `slots=`), which copies nothing |
| `kv:sliding` | `pool(..., sliding=W, span=, positions=)` | A pool of sliding-window layers: each sequence a ring of pages holding its last positions, whatever its length; the attention reads it with its window |
| `attention:blocks`, `attention:tree`, `attention:lse`, `attention:sinks`, `attention:sliding`, `attention:latent` | forward's keywords | Block-sparse attention, draft-tree masks, log-sum-exps, attention sinks, sliding windows, latent attention (the attention row above) |
| `norms:fused` | `rms_norm_rope_pair`, `rope_pair`, `stream_add_norm` | One call where a layer's separate norms calls take two, each output with the separate calls' bits |

Two optional methods let a plugin's configuration speak for it. A plugin whose configuration can keep it out of a
captured round or prompt chunk (a device expert cache over a file tier, a table read through Python) has
`round_problems() -> [reason, ...]` (no arguments; `[]` when it can be captured), which the round driver and the
prompt graphs ask before any capture: a reason refuses the captures, naming the plugin and slot, and the model runs
those rows eagerly through `step()`; an answer that is not a list of non-empty strings is the plugin's error. A pool
that holds a fixed number of round rows says how many as `round_rows` (an int), and `crit_plugins.rounds.Rounds`
refuses rounds of more rows when it is built.

### Kernels and plugins

Kernels are not plugins: they are the arithmetic plugins run, a GEMM or GEMV, attention, a norm, a scan, a fused MoE.
A plugin takes its arithmetic in this order:

1. a provider: torch, cuBLASLt through torch's scaled matmul, FlashInfer, fla, CUTLASS
   ([Kernels' libraries](../kernels/README.md#libraries));
2. one of crit's own kernels, for an operation no provider covers well. They are templates in crit's
   [kernel library](../kernels/README.md#crits-kernels) (`kernels/techniques/<family>/<id>/`, a `kernel.json` each), source that
   crit builds for the engine's device the first time a plugin asks, with the engine's tuning table
   (`crit kernels sweep` tunes each kernel's knobs for the model on that GPU) or the kernel's default knobs. Nothing
   is downloaded. The plugin declares it in its plugin.json, `"kernels": [{"name": "ID", "fallback": "what runs
   without it"}]` (`backend` where the build is not torch's own device backend, `cpu` for a CPU kernel beside a CUDA
   torch), and loads it once, when it is created, with
   `crit_plugins.kernels.load_kernel("ID", folder=Path(__file__).parent)`, which refuses a kernel the manifest does not
   declare;
3. last, a plain torch fallback with the same contract, where the kernel cannot be built here (no CUDA toolkit, an
   architecture it does not cover): the plugin still serves correctly, and the engine's facts say so.

Every request is recorded (`crit_plugins.kernels.loaded()`), and the engine's facts carry the record under `kernels`:
per kernel its source (`tuned`: built with the engine's tuning, `built`: with its default knobs, or `fallback`), the
tuning file and, for a fallback, why. The engine's tuning is a folder of `<ID>.json` files from `crit kernels sweep`,
carried in the package: the configuration's `kernel_tuning` names that folder inside it (`"kernel_tuning": "tuning"`
with the sweep's `--out tuning` in the candidate's source), which Core's loader applies before any plugin is made, so
the package served, sealed and published is the one Eval measured. Without it `CRIT_KERNEL_TUNING` names the folder
(or `crit_plugins.kernels.use_tuning`), which no package carries. A plugin's tests cover both paths: the kernel
where it builds, and the torch fallback.

Mechanism code that runs on the device is no compute kernel and stays in its technique, built and tested there: a
sampling law bit-identical to Core's host sampler, a round's accept, draft trees and hidden-state drafters, a recurrent
commit, a host-work mailbox, an expert cache's directory and copy ring, a CPU SIMD codec. A file that mixes the two is
split. A technique keeps no list of kernels: its `operations.json` says what each device operation computes, the CPU
reference and how a kernel is verified against it ([Inference](../inference/README.md#read-and-adapt-an-entry)), and the
plugin's plugin.json is the only place a pin lives. A plugin may hold several kernels and pick one by device, shape and
format; a selection never names a kernel. Kernels are measured by the kernel workbench against a CPU reference (`crit
kernels`, `crit eval workbench`), plugins by their tests and an ablation, engines by Eval. An engine's own kernel lives
in the engine's own plugin.

<a id="exactness"></a>

### Exactness

The fastest way to compute a phase's rows is often not the most exact one, and an engine says which it wants. Its
configuration asks for a level of [the kernels' scale](../kernels/README.md#exactness) (`declared`, `exact`,
`row-exact`, `repeatable`), one for both phases or one each:

```json
"exactness": {"decode": "row-exact", "prompt": "repeatable"}
```

`decode` is the decode and verify rows, `prompt` the prompt rows; for the decode rows `exact` and `row-exact` are one
level, the bits a prompt row is compared with. Core's loader reads it before any plugin is made
(`crit_plugins.kernels.use_exactness`), as it does the kernel tuning, and a package that asks for a level that does
not exist is refused when it is checked. A plugin that can compute a phase's rows in more than one way then follows
two rules:

1. Where its own option leaves the choice open (`auto`), it takes its fastest way that keeps the level asked:
   `crit_plugins.kernels.asks("prompt")` is the level, or None where the configuration asks nothing, and
   `allows("prompt", "repeatable")` says whether rows of a level may run there. An option set to one way is that way,
   whatever is asked.
2. Its object's `exactness()` states what its rows keep as it is loaded, `{"decode": level, "prompt": level}` (a
   phase it computes nothing in left out): the level of the kernel structures it runs
   (`crit_kernels.exactness.level`), or of its provider's path.

The engine's facts carry both under `exactness`: `asked` (the configuration's levels, or null), the lowest level any
plugin keeps in each phase, and each plugin's own. Core refuses an engine whose plugin keeps less than the
configuration asks, naming the plugin, so an engine never serves below the level it states. A configuration that asks
nothing is served as its plugins' options say, and its facts still state what the plugins keep.

The library's plugins that follow the level, each through the option it leaves open (`auto`, the default), and what
that option becomes for the prompt rows:

| Plugin | Option | Nothing asked, or `repeatable` | `row-exact` | `exact`, `declared` |
|---|---|---|---|---|
| quantized-projections | `prompt_gemm` | the native GEMM | `pinned` | `tile` |
| gguf-projections | `prompt` | `dequantize` | `exact` | `exact` |
| softmax-attention, partitioned-attention | `prompt_reader` | `prefill` | `prefill` (the split reader on the kernel, no FlashInfer), else `decode` | `decode` |
| host-expert-arithmetic | `prompt_gemm` | `native` | `declared` | `declared` |
| expert-offload | `prompt_arithmetic` | `tensor` when `repeatable` is asked, `declared` with nothing asked | `tensor` | `declared` |
| heterogeneous-expert-dispatch | `prompt_arithmetic` | `declared` (`tensor` when `repeatable` is asked and the device kernel is there) | `declared` | `declared` |
| sequential-recurrence, recurrent-state-recovery | `prompt_rows` | `chunked` | `scan` | `scan` |

With nothing asked each is what it was before the option had an `auto`. The others state what they keep and have one
way: projections-dense (decode `row-exact`, prompt `repeatable`: its prompt GEMM is the library's), fused-norms
(decode `row-exact`, prompt `exact`), grouped-experts (decode `row-exact` and prompt `exact` with tensor arithmetic,
`declared` with the declared one, which is another precision and so never chosen for a level), attention-state
(`repeatable` prompt rows while a quantized pool keeps fresh chunk keys, `fresh_mib`). An engine over one of these
that cannot keep the level it is asked is refused with the plugin named, and asks the level it keeps instead.

A plugin whose prompt rows run the arithmetic of its decode rows, in an order that never follows the row count,
states its prompt rows `exact`, not `row-exact`: a prompt row then has the bits of the same row decoded, which is
what `exact` asks. fused-norms on its kernel, grouped-experts' tensor arithmetic and host-expert-arithmetic's 16-bit
matrices are such, so an engine of a `preserve` run loads over them. `row-exact` is for prompt rows with an order of
their own (a pinned GEMM beside a decode GEMV, the split reader's prompt launch).

A candidate of a run always asks: `crit run ... candidate` completes its configuration's `exactness` to a level for
each phase, with the run's [quality mode's](RUN_RECORDS.md#quality-modes) level where the configuration states none
(`preserve`: `exact`; `speed`: `repeatable`), and the package carries it.

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
   over the `crit_core_steps` library, `CRIT_CORE_STEP_LIBRARY`, which exports Core's engine C ABI), whose
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
(`plan_problems`, with `PLAN_SLACK`, 5% and 64 MiB, for small tensors). The device bytes a plan is given are
`crit_plugins.device.free_device_bytes()` (the device memory's `free(release=True)`): the free memory after torch's allocator hands back what it caches for
nothing, since a model plugin that re-encodes or converts weights before planning its pools leaves those temporaries
cached, and `torch.cuda.mem_get_info` alone counts them as taken (one engine's KV pool refused to fit for it). Where
a request's temporaries end (a prompt's last chunk, a long score), `crit_plugins.device.release_cached_under(free_below)`
hands the cache back only when the device has fewer than `free_below` bytes free, so one request's cached blocks
leave room for the next one's without a release, and its wait on the device, after every step. Three things keep
the plan's count honest:

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
  load (`"load"`: `made` after the factories, `released_cached`, `opened` after the release, `warm` and
  `released_after_warm` after the warm requests, `checked` after the first requests; None where the device's memory
  cannot be read), each with `other`: the device memory neither torch nor an elastic range holds and that is not
  free (the process's CUDA context, libraries, modules and graphs, and other processes), so what a stage took
  outside torch shows.
- **Captures are planned.** Core's round driver and prompt graphs measure the device memory their captures take (the
  warm-ups' lasting allocations, the graph pools, the capture stream's workspaces) and plan it through the placement
  plugin as their own entries, `"rounds:captures"` and `"prompts:captures"`; a placement that puts them off the device
  refuses the captures. A model plans no fixed capture workspace of its own.

**The device's memory.** Every byte of a device is somebody's, and one object knows whose:
`crit_plugins.memory.of(device)`, the same for Core and for every plugin (a plugin's factory also finds it as
`context.memory`). Core's loader and the plugins read and move device memory through it and nowhere else.

- **Readings.** `total`, `free()` (the driver's free bytes; blocks torch's allocator holds unused count as taken),
  `torch()` (its allocator here: allocated, reserved, and `unused`, the whole blocks nothing uses and
  `release()` hands back; the empty segments of captured graphs' own pools are not among them), `release()`
  (hand those back; `release(below=n)` only when fewer than n bytes are free), `mark()` and `peak()` (the most
  torch held since the mark: what a step took at its peak).
- **Elastic ranges.** `elastic(owner, name, max_bytes)` reserves an address range and backs it granule by granule
  through the driver's virtual memory API: `grow(n)` and `shrink(n)` back more or less of it while every address
  stays, so a tensor over it, a captured round and a kernel's arguments stay valid; `back(first, end)` and
  `release(first, end)` back it in places. `elastic_ranges` says whether the device and driver have the API; a
  plugin keeps one fixed allocation where they do not. `back(first, end, host=True)` backs granules from host
  memory, which the driver pins and maps into the range: kernels read and write them at the same addresses, over
  the link and at no device memory (a range's `host_mapped`, its `host` bytes in the report).
- **Paged ranges.** A pool's units are what its sequences take as they come: a KV pool's pages, a state pool's
  sequences. `paged(owner, name, nbytes, planes)` is an elastic range of planes (a layer's tensor each: first
  byte, bytes a unit, units) whose units are backed together: `back(units)` asks the givers first and says
  whether the device had the memory, `trim(units)` gives back what lies past and lets the givers grow into it.
  A range made with `host_bytes` may back units from host memory when the device has none (`back(units,
  host=True)`): attention-state's `host_pages_mib` serves a context the device does not hold that way, exact
  and at the link's rate, instead of refusing it.
  attention-state's and the recurrence plugins' `elastic` option put their pools on it, so a pool sized for the
  longest context and every sequence at once takes device memory only for what is open.
- **Growing buffers.** A workspace a plugin makes for each call shape it meets (a prompt chunk of every size)
  adds up, and a captured graph keeps each one's address for good. `growing(owner, name, max_bytes)` is one
  buffer at a fixed address: `take(nbytes)` backs its first `nbytes`, the givers making room first, and returns
  that prefix. Calls of one stream run one after another, so the shapes share its rows and cost the largest's
  bytes, not their sum. An engine's scratch is one.
- **Scratch.** What a call only computes in (a kernel's partial sums, route rows, a decoded tile) and nothing
  reads once it has returned is scratch, and an engine's calls run one after another on its stream, so they all
  compute in the same bytes. `context.scratch` is the engine's (`crit_plugins.memory.Scratch`, made by Core's
  loader and closed with the engine; `memory.scratch()` makes one for code that runs without an engine):
  `take(nbytes)` gives the first bytes of one growing buffer, `take(a, b, ...)` a call's buffers together. A call
  takes its scratch every time it runs, all of it in one take, and holds nothing there across calls: the next
  call's take is the same bytes. The scratch therefore costs the engine its largest call's bytes, where a
  workspace per plugin costs their sum, and its address never moves, so a captured call keeps it. It grows
  outside captures only (a capture's warm-ups take a call's size first; a take that would grow it inside a
  capture raises). `crit plugins test` runs with `CRIT_SCRATCH_POISON=1`: every take then hands out bytes
  filled with 0xFF (NaN as a float, -1 as an index), inside captured graphs too, so a call that reads what an
  earlier call left there fails its test. A plugin keeps a buffer of its own only for what outlives a call
  (state, results, tables). The split-KV reader's partials, the sampler's launch scratch, the route rows of
  expert-offload's and heterogeneous-expert-dispatch's device part and quantized-projections' tile activations
  and partial sums are taken from it.
- **State is read where it lies.** A pool's rows are its own (KV pages, index keys, recurrent states), at the
  slots its sequences hold. A kernel takes the pool's store and the slots (`layer_pages(layer)` and a reservation's
  or `slots_at()`'s slots, as the attention reads KV through page tables) and loads what it needs from there. A
  gathered copy of a sequence's rows for a kernel to read is a buffer the size of the state, written every call,
  and inside a captured graph it stays in the graph's pool for good: on Qwen3.8-Flash-Next the gathers of whole
  sequences' index keys were 146 of the 174 MiB the largest round's graph took.
- **The ledger.** A plugin that can give memory back is a giver, `give(name, release, regrow, keep)`, and says what
  it keeps free (bytes, or a callable that follows the requests in flight); whoever is about to take device memory
  it cannot do without calls `need(bytes)` first, and the givers release what the device lacks to hold those bytes
  with what they keep still free; `settle()` lets them grow back down to what they keep. Nothing is taken from a
  giver while the device has room.
- **Load, then serve.** While an engine loads, what its plugins left free at their creation is the load's (graphs,
  workspaces, first prompts) and `settle()` does nothing. Once the warm requests ran, Core calls `serve()`: torch's
  unused blocks go back to the driver and the givers take what is left. No plugin is called for it.
- **Other programs.** The device is not the engine's alone. When a desktop or another process takes memory while
  the engine serves, nothing in the engine asked for it, and under WDDM the driver then moves allocations to system
  memory without an error. Core calls `relieve()` before each step it runs: where less is free than the givers
  keep, they release the difference at once, and when what other programs hold has grown since the last look they
  wait five seconds before `settle()` lets them grow back. The report counts it (`outside`). A tier of copies
  loses speed by this and never a token. Rounds Core launches without the engine's Python run no step: a
  thread of Core's (`watch()`) reads the free memory five times a second and, when it fell under what the givers
  keep, asks the engine for a step (`NativeRounds.ask_step()`): the next decode step then runs through `step()`
  once, where `relieve()` gives the memory back, and the rounds go on.
- **The census.** When the load is over Core walks the plugin objects and counts each live tensor of torch's
  allocator for the plugin whose code made the object holding it (`census(plugins)`). The engine's telemetry
  carries the device as it stands (`"memory"`, from `report()`): free, torch's allocator, each elastic range's
  backed bytes, each plugin's bytes, the largest tensors one by one (`tensors`: plugin, bytes, shape and dtype,
  so a table a pass gathers a few rows of, or a pool sized past what is served, shows by its shape), what captured
  graphs' own pools hold (torch keeps a pool for its graph's replays, in use or not), what no plugin object holds,
  and what is neither torch's nor a range's nor free (the CUDA context, other processes, a desktop). The walk
  (`holdings(plugins)`) starts at the plugin objects and at the globals of the plugins' own modules, so a buffer
  a module keeps for all its objects is its plugin's too.
- **One pool for an engine's graphs.** A captured graph's temporaries live in a memory pool torch keeps for its
  replays. An engine's graphs never run at once, so they share one: `context.graphs`, the engine's pool, which
  Core's round driver and prompt graphs capture into by default (they find it through the factory's `parts`).
  Each capture reuses the bytes of the temporaries captured before it, so the pool holds far less than their
  sum (four rounds' pools of 494 MiB became one of 270 on Qwen3.8-Flash-Next, where the largest graph alone
  takes 174), and after the load nothing a replay touches is allocated again. It asks one thing of the plugins: a tensor made inside a capture that anything
  reads later is kept for good, never dropped or replaced (`share_pool=False` gives a driver's graphs pools of
  their own). Pools are made through `pool(owner, name)` and captures run inside `capturing(pool)`, so the
  census lists each pool's bytes by name (`pools`: `core/graphs`, `captures/matrix workspaces`).
- **Steps that allocate.** `made()` counts the allocations torch's allocator has made on the device. A step that
  only replays graphs makes none, and Core holds an engine to that where asked: with
  `CRIT_CORE_STEP_ALLOCATIONS=0` a step that allocates on the device once the load is over ends the engine
  with the count, and a probed engine (`CRIT_CORE_STEP_PROBE=1`) reports each kind of step's allocations in its
  telemetry (`core.probe`: `allocations`, `most_allocations`). Native rounds run no Python and allocate nothing.
- **The audit.** The census says who holds a byte; the audit says what the engine does with it.
  `crit plugins memory PACKAGE --model MODEL` loads the engine in process, serves two short prompt sets of its
  own and fingerprints every page of the device memory the process holds before and after
  (`crit_plugins.audit`): torch's blocks and every backed granule of the elastic ranges, read by the engine's own
  thread as a request ends, before its sequence closes (`StepEngine.audit`). It reports each buffer
  by the plugin and attribute that hold it with the bytes the workload wrote, how far into it the writes reach
  and what stayed zero, and draws the verdicts that are certain (dead: all zero at every reading; oversized: a
  part never written) apart from those that are not (unchanged bytes that hold something: weights, a table, or a
  workspace only the load wrote). The bytes Core knows to be scratch (a capture's temporaries, growing buffers)
  it fills with a mark first, so there every unwritten byte is certain, and a greedy reply repeated after the
  workload must come out the same. Each graph's pool is listed with the allocations its captures made, by the
  code that made them (`record()` turns torch's allocation history on for the audit's load), each paged range
  with the units backed and written, and each kind of step with its graph replays, allocations and transient
  peak. It needs nothing from the engine's plugins. Run it when a plan is about device memory (a larger expert
  cache, a longer context, a workspace to shrink) and after a change that adds a buffer; its verdicts hold for
  its workload, one request at a time, which the report states.

Some memory is worth more in use than in reserve: a device expert cache hits more with every slot it has, yet a
prompt step needs room for its temporaries, and under WDDM (Windows, WSL) an allocation past the device's memory
does not fail: it moves to system memory, and every kernel that reads it slows down. So that such a plugin holds
every byte the requests in flight do not need and none they do, Core also tells it of every step it runs:

- **Step frames** (`crit_plugins.device.STEP_FRAMES`). A plugin object of any slot that has the method is told of
  every step Core runs: `begin_step(prompt_tokens)` before the model's `step()` (the prompt tokens its rows
  prefill, 0 for decode rows alone) or its `score()`, and `end_step(prompts_open)` after it (whether a row is
  still prefilling, so another prompt step follows). The model plugin makes neither call. Native rounds run no
  Python, so a frame is a step the model runs: prompt chunks, scores, eager decode.

expert-offload's cache is the first giver: it opens with a reserve, takes every free byte but its margin (the
run's VRAM headroom) once the load is over, makes room for a prompt step out of what a chunk of its size was
measured to take, and grows back after the prompt's last step. A probed step's free device memory (the run's
full-context checks) is read when the model's step returns, before the end frames.

### The host

**The host's processors.** An engine's threads share one machine: the workers that compute what the device does not
hold, the thread that answers the device's doorbells, the loop that drives the device. Left to each plugin, every one
starts its own pool and picks its own cores, and the threads of an engine outnumber them. So the host is one object,
as a device's memory is: `crit_plugins.host.current()`, which every plugin also gets as `context.host`.

- **The plan.** `cores` are the physical cores the process may run on, `plan()` says who runs where: the first core
  is the service's (the thread that answers the device's doorbells and runs their jobs), each other core has one
  worker of the pool, pinned to it (Linux). Nothing else of the engine computes on the host while the device waits
  for a job, so a job has every core, each once.
- **The pool.** `pool()` is the process's worker pool, one for Core and every plugin: one batch at a time, claimed
  by the workers and the caller without a lock, batches of different callers one after the other. Native code built
  apart from Core runs on it through two addresses, `pool().native` (the pool's C entry and the pool: `PoolRef` in
  Core's `pool.h`, whose folder is `crit_plugins.host.INCLUDE`); Python has `pool().run(tasks, fn)`. The first
  `pool(workers, pin)` may size it; what it is serves everyone after that, since any split of a batch gives the same
  result. A plugin never starts worker threads of its own.
- **File reads.** `reads(owner, path)` reads one file at a queue depth: positioned reads from 8 threads at once,
  each with its own handle, opened past the OS's page cache where the file system allows (`direct=False` for
  buffered reads). It is for a tier that keeps bytes of a file larger than host memory itself, so that the drive is
  busy and nothing is held twice. The threads wait in the kernel and compute nothing, so they are no workers of the
  pool; the report lists each open file's reads, bytes and whether they are direct.
- **Handoffs.** Work the device decides and the host does inside one CUDA stream, captured graphs included
  (the [host-work-mailbox](../inference/host-work-mailbox/README.md) technique): `handoff(device, owner, name,
  slots, payload_bytes, serve, context)` gives slots in mapped pinned memory; a plugin's kernel writes a job into a
  slot's payload and raises its doorbell (`handoff_device.cuh`, with the slot's device addresses `words(slot)`), the
  service thread runs the handler (a C function of the plugin's host library) and answers, and `join(slot)` makes
  the stream wait where the device reads the result. The join is the cheaper capturable one measured on the device
  (`joins(device)`): the driver's stream wait or a one-thread poll kernel. A stream's wait has no clock, so a
  job that never returned would hang the device for good: the service has a second thread that only watches, and
  a job that runs past the job budget (20 s; `job_budget(ms)`) fails its handoff. Its waits are answered at once,
  `failures()` names it with the reason, and Core ends the engine at its next step (also between rounds it
  launches itself) instead of serving on what such a window computed. The plan and the pool know no device; a
  handoff's device side is a runtime service of the device's backend (CUDA is the one built today, and
  `handoff()` raises `NativeUnavailable` for any other device).
- **The report.** The engine's telemetry carries the host as it stands (`"host"`, from `report()`): cores and
  memory (total, available), the pool's workers and batches, the service thread's CPU, each handoff's join and jobs,
  and what was built on the machine.

**Native code built at load** (`crit_plugins.native`). A plugin's C++ and device code is source, compiled on the
machine the first time it is needed, cached by content and recorded: `library(name, units, includes)` builds a shared
library with the system's C++ compiler and opens it with ctypes; `extension(name, sources, includes)` builds a torch
extension for the device's architecture. Both raise `NativeUnavailable` where they cannot build, and the plugin runs
its fallback or refuses to load; `built()` (in the host's report as `"native"`) says what was built and what was not.
A plugin does not carry a build of its own.

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
(the first plugin in creation order winning a key), then `plugins` (the plugin filling each slot), `kernels` (crit's kernels its plugins asked for and how each was built: tuned, built or fallback) and `exactness` (the levels the configuration asks and the ones its plugins keep: [exactness](#exactness)). The facts Eval reads
(`crit_core.serving.ENGINE_FACTS`) include a projections plugin's `prompt_activations`, which crit-eval prices
prefill by, and the kv_cache plugin's `kv_bytes_per_token`: the device bytes one more token of context takes in the
pools it made (every layer's keys and values at its pool's precision, their scales too, summed over every pool, a
draft layer's among them), which crit-eval sizes the KV cache and prices the KV term of its floors by. Without that
count Eval leaves the KV term out and names it missing rather than assume a precision.

A generation's result carries its rounds: Core counts every decode row as one round (`rounds`); when the request
drafted it lists per round `accepted` and `drafted`, and it sums each of the engine's named round counters (the model
info's `round_counters`, such as `expert_reads` and `expert_hits`) over its native rounds. A model that names
`nonfinite` among them has the round driver count, row by row, the verified columns whose logits held a NaN or an
infinity (one reduction over the logits a round), and Core fails a request that saw one instead of returning tokens
that are not the model's; the engine's telemetry counts them (`core.nonfinite_requests`). The model plugin's
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
| `open(request)` | A session for a `StepRequest` (key, prompt, output limit, sampling, stop tokens, log-probabilities, replay, plain, and `prompt_cache`: whether it may be served from the [prompt cache](#the-prompt-cache)) |
| `step(rows)` | One `StepResult` per scheduled `StepRow`: prompt chunks and decode rows of many requests in one step, each row's tokens independent of the other rows; a constrained row's `allowed` goes to the sampler as its `params["allowed"]` |
| `commit(decisions)` | Core's decision per row of the last step: the tokens kept, an end, a cancel |
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

#### The prompt cache

The prompt cache is Core's, as the scheduler is: `crit_plugins.prompt_cache.PromptCache` keeps ended requests'
sequences in the model plugin's pools, so a prompt that begins as one of them did (a conversation's next turn, an
agent's next tool step) prefills only the rest. It is a ledger over the pools' own `open`, `fork`, `keep` and
`close` and touches no KV byte. A model plugin gives it its pools and calls it where it opened and closed
sequences:

```python
self.cache = PromptCache(pools=[kv_pool], states=[state_pool], entries=8)
reused = self.cache.open(session.handle, request.prompt, request.prompt_cache)   # open(): the prefix it reuses
self.cache.mark(session.handle, position)        # optional: the state pools stand at `position` of the prompt
self.cache.retire(session.handle, tokens)        # retire(): the tokens whose rows the pools hold; None drops it
```

Prefill starts after the reused prefix, and the prompt's first step reports it as
`StepResult.reused_prompt_tokens` with the rows it computed, which Core counts as a retained prefix and crit serve
as the request's cached prompt tokens. `pools` are cut anywhere: the kept sequence is forked, its pages shared until
one of the two writes, and the fork kept to the shared prefix. `states` hold one state per sequence that nothing
rolls back (a recurrence's state pool, a sliding window's ring), so they resume only where they stood: at a kept
sequence's end, or at a position the model marked while it prefilled, a copy of the state per mark (`marks`, 2 by
default, the newest). A sequence that resumed at a mark holds that mark too, the same copy, beside the ones it
makes, so a conversation keeps the place its last turn resumed from. Which marks stay is the cache's rule, since the
model marks every chunk's end: the newest by default, or with `spread` (a token count) marks that lie about twice as
far apart with each step back from the newest, so a long prompt changed in its middle resumes near the change
instead of nowhere. With `shared_least` the cache learns where conversations part: a prompt that shares that many
tokens more with a kept sequence than the place it can resume from (a system prompt and tool list every
conversation of a client repeats) gets that place as its first stop (`cache.stop(handle, frontier)`), and the mark
the model makes there is a shared one, kept outside the count of `marks` and held by every sequence that resumes
from it, so it outlives the conversations that made it. A later prompt rarely repeats a prompt to its last token: the same
prompt sent again computes its last token anew, and a chat template may end a prompt otherwise than it renders
that turn in the next one (a thinking model's opening tag). A model with states therefore gives the cache a
`tail` (8 tokens, say) and passes `stop=cache.stop(session.handle)` to `step_item`: the prompt chunk then ends
that many tokens before the prompt's end, the model marks there as it commits, and the tail is the next step's
chunk, which marks the prompt's end. The shorter chunk costs the prompt a step (about 45 ms on a 27B model's
engines), so `stop()` is None where it would not pay: for a prompt that resumed at a kept prompt's end, whose
conversation repeats its prompts whole and which holds that place for the turn sent again, and for a prompt
shorter than `tail_least` (a prefill chunk, say), prefilled whole in about a step's time. The last prompt token is
always computed: its row gives the first output token. Kept sequences leave least recently used first, past
`entries` or `tokens`, and on `evict()`: the model calls it when its page reservations run short, and a pool that
offers `on_exhausted(callback)` (attention-state's) calls it itself before it refuses a page, so a kept sequence
never costs a request its pages. One that a newer sequence extends leaves too. A request that decoded in Core's
native rounds, which publish its output without the model seeing it, is kept to its prompt: with states it retires
with `stands=False`, since its state is past the tokens it can name, and later prompts resume at its marks. A model
with state of its own beside the pools (a hidden row its draft layer reads) hands it over where it stood,
`mark(sequence, position, payload)` and `retire(sequence, tokens, payload)`, gets it back from
`resumed(sequence)` after `open()`, and resumes at those places only (`at_marks`). A state is large where a KV
page is small, so `marking` bounds the open sequences that hold marks at once, and a model need not size its
state pool for the marks: they may stand in the states no open request uses, a mark the pool has no room for
takes a kept sequence's (and is refused when none is left), and the model takes the copies back when a request
or a score needs a state, with `evict()` for the
kept sequences' and then `unmark()` for the open ones'. The cache then costs no device memory, and keeps
nothing while every state is a request's.

A request allows the cache with `StepRequest.prompt_cache`. crit serve's requests do, unless they carry
`"crit": {"prompt_cache": false}` (`chat-prompt-cache-v1`, which every plugin package declares); a request that
does not reuses nothing, marks nothing and is not kept, so it runs as on an engine without a cache. Everything that
measures sends it so: Eval's timed, scored and concurrent requests, crit smoke, `crit run measure` and the
ablation's benches. Eval's follow-up turns allow the cache: a model plugin that keeps one states `prompt_cache:
true` in its info, Eval measures the turns against their floor with a prompt cache (its goal `followup_ttft`) and
its prompt_cache gate holds the cached path to the answers the fresh one gave.
[test_prompt_cache.py](../tests/test_prompt_cache.py) shows the ledger and the toy engine that keeps one
(`toy-model`'s `prompt_cache` option): a follow-up turn reuses the kept turn, answers as fresh, and requests at
once each fork the kept sequence. [test_device_prompt_cache.py](../tests/test_device_prompt_cache.py) does the same
on a GPU over attention-state's paged pool, through `step()`, captured rounds and drafting rounds, and for a hybrid
model whose recurrent state resumes at the marks it set as it committed each prompt chunk.

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

The parts are the model, sampler, drafter, head, verifier, every pool, the plugins of the slots the round uses
(expert-offload in the experts slot) and the extra parts the model registers (`extra_parts`: a device expert cache
the model made itself), each once; with no Python between native rounds, a part's per-round
bookkeeping runs in its two hooks. `bind(slot, sequence, frontier, token, allowance, capacity, stops=, params=,
session=, tokens=, outputs=)` puts a sequence into a round slot in every pool (a KV pool reserves its pages up to
`capacity`, so a captured round never allocates; pools and the drafter with `seed(slot, tokens)` get its tokens so
far; the sampler stages its rules), `unbind` and `move` keep the decoding rows packed at the front, and a row a step
does not schedule sits a round out (`view.hold`: it emits nothing, and every commit keeps it as it was). Each (rows,
width, mode) is one graph, captured at load into the engine's graph pool ([memory](#memory); a pool of its own with
`share_pool=False`, or where no loader made the engine) on one capture stream per device, two eager
warm-ups first, with cuBLAS workspaces made in a pool Core keeps for them (`capture_workspaces`: torch frees a
stream's workspaces whenever a graph captured on it goes, and an engine closed beside another must not free what the
other's graphs write); the memory the captures take is planned as `"rounds:captures"` ([memory](#memory)). Inputs a
model computes from the tokens before each column (hashed n-gram ids) come from a `TokenRing` among its pools.

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
another's imports or device state) and fails a plugin that has none. A plugin whose tests passed before under the
same key (its files, the plugins its tests use, crit's `crit_plugins` and `crit_core`, and the environment: Python,
torch, the device, `CRIT_KERNEL_TUNING` and the provider packages' versions) is reported `cached` without running them
again; `--no-cache` runs every plugin's tests. `PluginTestCase` gives a test:

| Helper | What it does |
| --- | --- |
| `assemble(use, options, context, facts, use_options=)` | The plugin in its own slots, with the other plugins (`use`, their options by id in `use_options`) and engine facts it needs |
| `reference(use, options, context, facts)` | Its `reference` plugins in its slots |
| `assertImplements()` | Its object in each slot it fills has the slot's interface methods, its device form and the methods of every device extension it provides; a projections plugin keeps its rows' bits (`assertKeepsRows`) and a KV pool writes, forks, rolls back and gathers (`assertKeepsKV`) |
| `device()` | The device its tests run on, from the manifest's `devices` (a test of a CUDA-only plugin skips without CUDA) |
| `core_runtime()` | The `crit_core_steps` library a test that runs the engine on Core's runtime loads, found as Core's loader finds it (`CRIT_CORE_STEP_LIBRARY`, else the checkout's build, else what `crit core build` built); the test skips, saying to run `crit core build`, when there is none. Gate on it, never on the environment variable alone |
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
| `experts` | `experts-reference` | `expert-offload` (bitwise against `experts-reference` on the host; experts of any size against the device); `grouped-experts` (against `experts-reference`); `heterogeneous-expert-dispatch` (with async-expert-contributions' code, bitwise against `experts-reference`); `host-expert-tier` (against `experts-reference`) |
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

Two entries fill no slot themselves and lend their code to plugins that do: async-expert-contributions and
expert-streamed-prefill (whose moe_block is `experts-reference`). Eight entries are no plugin (`ENGINE_SERVICES` in
[cli/crit/plugins.py](../cli/crit/plugins.py)): artifact-loading, model-import-audit, checkpoint-storage,
host-memory-admission and device-speculative-round, since a model plugin calls them and no slot holds them, and
session-scheduling, concurrent-request-execution and host-work-mailbox, since Core does what they describe for every
engine (its concurrent runtime, and the host's handoffs).
The plugin system itself is Core's, since an engine loads with it: `crit_plugins` in
[core/python](../core/python/README.md) holds the manifest, the assembly, the slot interfaces and device forms, the
kernel loader, Core's round driver and prompt graphs, the test helpers and the ablation's variants, and
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

The library goes on changing after the copy. `crit plugins status ENGINE/plugins` says how each of the engine's
plugins, and each folder of code under `_sources`, stands to it: `current` (the library's files as they are now),
`behind` (a copy of an earlier library version, which the engine did not change: it names that version's commit
and date and counts the library's changes since), `changed` (a copy of no library version, so the engine changed
it) or `own`. It reads the versions from the history of the crit checkout the library lies in; with the library
of an installed wheel, which has none, a copy that is not current `differs`. `--json` prints the rows.
`crit plugins update ENGINE/plugins` replaces every copy that is behind with the library's current one, the code
under `_sources` too, and leaves the copies the engine changed and its own plugins as they are (`--dry-run` says
what it would replace). An engine that took a library plugin unchanged gets the library's fixes that way; its
tests and its Eval then say whether it keeps its results.

Every slot but `model` has library plugins that run on the device, each with its host form as before and the slot's
[device form](#device-form) in torch, over providers where one covers the arithmetic and crit's own kernels
where none does, each with its torch fallback ([kernels and plugins](#kernels-and-plugins)). Their facts say which
[device extensions](#device-extensions) they provide (each plugin.json's `provides`):

| Slot | Library plugin | Device form |
| --- | --- | --- |
| `kv_cache` | `attention-state` | A paged KV pool of every sequence: BF16, FP16, FP32 and FP8 E4M3 pages with the layer's scales (static ones from the checkpoint through `kv:scales`), FP8 E4M3 (`fp8:row`) or INT8 pages with one scale per position and KV head, copy-on-write forks, pools of keys only and rows at chosen slots (`kv:keys`); rows of up to its `window` tokens read as window rows; with `step_rows`, a step's slots and window rows' metadata in device buffers at fixed addresses, so a model replays its decode steps from CUDA graphs; round slots for captured rounds (up to `round_rows`, pages reserved at bind, slots computed and lengths committed on the device) and prompt buffers for captured prompt chunks; over FP8 and INT8 pages a fresh buffer of a prompt chunk's own 16-bit keys and values (`fresh_mib`, off by default: 0 reads the chunk from the pages, which keeps a scored continuation on the served decode's reads; facts `prompt_kv`, `prompt_kv_fresh_tokens`); its unused pages lent to a device expert cache (`spare`, `lendable()`, `on_reserve()`); pools of sliding-window layers whose sequences are rings of pages (`kv:sliding`: `sliding`, `span`, `positions`; fact `kv_sliding_bytes_per_sequence`); pages written by crit's kernel paged-kv-attention where it loads; `kv_bytes_per_token` in its facts |
| `attention` | `softmax-attention` (base), `partitioned-attention` | Each row's queries over its sequence's pages read in place (one device form, attention-state's code/device_attention.py): FlashInfer's paged prefill over 16-bit and FP8 prompt rows when FlashInfer is installed (the KV split pinned, so a row does not depend on the rows beside it); decode and verify windows of every page type through crit's kernel paged-kv-attention's split-KV reader (KV partitions walking each row's length on the device, merged by their log-sum-exps; block lists take PIECE_PARTITIONS in every launch, so prompt, step and round rows sum a list in one order), in a step as in a captured round, so a decode row's bits are the same in both and plain, log-probability and replay requests decode what a drafted round decodes (softmax-attention's `decode: flashinfer` reads a step's window rows with FlashInfer's decode instead, for engines that decode in steps only), with tree masks and block lists; fp8:row and INT8 prompt rows through its prompt reader; otherwise the rows gathered and torch's SDPA, or a masked softmax over the page tables that captures in a CUDA graph. Captured prompt chunks read the staged prompt table on the device. `prompt_reader`: `prefill` (the default) reads prompt rows through a prompt reader, the fastest and another reduction than decode rows'; `decode` reads a step's prompt rows and captured chunks in the decode's reduction (window rows of up to 64 columns through the split-KV reader, or its torch fallback), so a scored token equals the same token served, bit for bit (facts `prompt_reads`, `served_vs_scored`) partitioned-attention is the same on one GPU, with log-sum-exps and `merge` for partials of other devices |
| `projections` | `projections-dense` (base), `quantized-projections`, `host-expert-arithmetic` (cpu) | BF16 matrices (`out_dtype` FP32 for logits, on the CPU too). quantized-projections: stored BF16, FP8 (per channel or tensor), MXFP8, NVFP4 and AWQ / GPTQ INT4 matrices (INT4 with a zero point a group, repacked at load); with `prompt_gemm: native` the quantized prompt GEMMs on crit's own kernels, and with `auto` (the default) only where they measured faster than the provider path; MXFP8 checkpoints' own activations with `mxfp8_activations: mxfp8`; prompt rows through cuBLASLt (torch's scaled matmul) with FP8 or NVFP4 activations, or with 16-bit activations on the matrix decoded a chunk of rows at a time (BF16, FP16 accumulation or FP8 rows by `prompt` and `prompt_roles`; FP8 matrices' W8A16 with `fp8_activations: bf16`); with `prompt_gemm: pinned` every format's prompt GEMMs on one split-K-free cuBLASLt algorithm per shape and format, chosen at load (pinned-gemm), and with `fixed_tiles` the BF16 and weight-only ones through fixed-tile-gemm, so a prompt row's bits are the same at every prompt length, chunk and padded bucket (`row_invariance` `prompt_rows`); decode rows (up to `decode_rows`, 16) through mma-rows' tensor-core tile or decode-gemv's GEMVs at the prompt rows' activations (`decode_activations: prompt`) or 16-bit ones, each row's bits alone and batched; short prompts through the tile as far as it measured faster (`tile_rows: auto`); every call path of a matrix run once at load, so no request compiles; the prompt arithmetic of every format in its facts (`prompt_arithmetic`); draft-only matrices re-encoded at load (`encode`). host-expert-arithmetic: stored NVFP4 / INT4 matrices weight only (the FP32 order) or W4A4 / W4A8 (the integer block order) on the CPU, BF16 a row at a time, every row's bits those of the row alone at any count |
| `recurrence` | `sequential-recurrence` (base), `recurrent-state-recovery` | Every sequence's FP32 state and convolution window in a pool; decode tokens and verify windows of a step or a captured round in one column-scan launch (recurrent-scans where installed, else torch), the accepted prefix kept on the device (`commit`: deferred, checkpoints or recompute); prompt rows of up to `exact_rows` tokens (a stated value, 512 by default: the largest default prompt bucket; 0: none) through the same scan, so their bits are a decode's, and longer ones through fla's chunked kernel (sequential-recurrence; each of its autotuned Triton kernels pinned to one stated tile, `FLA_TILES`, since fla's autotune picks among tiles of other bits) or recurrent-scans' two-pass prefill (recurrent-state-recovery); which one a chunk runs and on which tiles follows from the configuration, never from a timing (a model whose chunk grid stays within `exact_rows` keeps every prompt row exact and a prefix's rows equal to the whole prompt's) |
| `router` | `topk-router` (base) | Softmax or sigmoid routing, correction bias, group limits, renormalization: one launch of crit's kernel topk-routing where it loads and the rule has no group limit, torch otherwise; capture-safe (a row of NaN logits still picks valid distinct experts, NaN ranking below -inf) |
| `experts` | `experts-reference` (base), `expert-offload`, `grouped-experts`, `host-expert-tier` (cpu), `heterogeneous-expert-dispatch` | expert-offload: experts larger than the device in one plugin, the model calling it as for experts that fit (begin, load, open, forward), its decision tree placing them on the device, in host memory behind a device cache in every byte the device has free while the engine serves, or in a file, misses computed by the CPU tier while the link copies them in beside the round (or copied over the link), prompt chunks staged, a diagnosis in its telemetry. experts-reference: the routed SwiGLU block over the experts' dense matrices, FP64 products in fixed chunks, capture-safe. grouped-experts: each expert one bundle in its stored format (bf16, fp8 channel or block, nvfp4 W4A16, or with `activations: nvfp4` each NVFP4 layer as its stored matrices declare, W4A4 where they carry input scales, int4, mxfp4), the routes grouped by expert on the device over the plugin's bundles or a residency cache's slot store, through crit's kernel grouped-experts or its torch fallback; a route's bits independent of the rows beside it and of decode against prompt; a long prompt chunk runs in parts, each combined into its tokens' rows as it goes, one prompt workspace serving the shorter prompt-sized calls of its shape. host-expert-tier: NVFP4 W4A4 / INT4 W4A8 experts in host-expert-arithmetic's declared integer block order on the CPU (a native tier built at load over a pinned worker pool, or torch with the same bits). heterogeneous-expert-dispatch: on CUDA, the routes an expert cache sends to the CPU (slot -2) computed by host-expert-tier's tier and joined inside the stream through host-work-mailbox, the split never changing a bit |
| `expert_residency` | `device-expert-cache` | The directory on the device (`experts:cache`): one capture-safe plan launch and one fill launch per routed layer (a slot table: slot, -1 not routed, -2 the CPU tier), fills by SM loads from a registered host store, prompt chunks through a copy-engine ring that never admits, a KV pool's unused pages as slots, clock or ranked policy, telemetry and per-row counters through `count_round` |
| `expert_source` | `expert-tensors` (base), `ssd-expert-materialization` | Bundle stores (`experts:store`, `open_store`): bundles in registered host memory with a device view, or a file tier read by a native reader behind a host tier, which a cache fills from on the host; `csf_scales` keeps NVFP4 scale components CSF-compressed in the store |
| `embeddings` | `memory-table-rows` (base), `lookup-table-rows` | A device table, or (memory-table-rows' `host`, lookup-table-rows' `open`) a table in registered host memory gathered over the link; lookup-table-rows also reads tables in files (native positioned reads, prefetch into a hot-row cache, the gather a host function a CUDA graph captures) and stored NVFP4 rows decoded on the device |
| `norms` | `torch-norms` (base), `fused-norms` | torch ops with every reduction through torch's fused rms_norm (torch's mean over a row is not row-invariant at hidden widths); fused-norms: FlashInfer for the calls whose function it computes exactly, one launch per call of crit's kernel fused-norms for the rest, torch-norms' arithmetic where it is not there |
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

The run's cases are the workload the user captured, often one prompt. A plugin whose effect depends on the text it
decodes (a drafter, copied drafts, a draft width, a vocabulary subset) can look neutral there and still cost decode on
long prompts or other languages. `--data EVAL_DATA` adds crit-eval's speed cases and held-out prompts to every
variant (`eval:ID`, `eval:heldout:ID`): each request is the case's messages at the model's sampling as Eval sends them,
`--seeds N` requests each (default 3, one seed apiece where the package declares seeded sampling), and the report
judges each plugin on their geometric mean with the run's cases. Use it on every ablation of a drafting engine.

`--set PLUGIN.OPTION=VALUE[;PLUGIN.OPTION=VALUE...]` (repeatable) serves an option variant `set:...`: the full
selection with those options changed, on the same cases. With `--set` and no `--variant`, only `full` and the option
variants run, so `ablate --data EVAL_DATA --set mtp-decode.override=0 --set attention-state.fresh_mib=64` is an
in-process A/B of two options against the engine as built, on Eval's text, without writing a probe script. Each
option variant gets a row like a plugin's, its `in_engine` changes against `full`.

The level the engine asks of its rows ([exactness](#exactness)) is set the same way: `--set exactness.prompt=repeatable`
(or `exactness.decode=LEVEL`, or `exactness=LEVEL` for both phases) serves the candidate asking that level over its
own, so the report says what a level costs or gains before a candidate is built for it. Each variant's row carries the
engine's `exactness` fact, what was asked and what each plugin kept; a variant whose plugins keep less than it asks
fails with Core's refusal.

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
| `judged` | The KPIs the verdict read (below: the time per decode round in place of decode where both variants draft) |

Where both variants of a comparison draft (more than 1.05 tokens a round), the row also compares their time per
decode round (`round_ms`). A plugin outside the drafter changes how long a round takes, while the tokens a short bench
accepts per round move with what it samples: two variants that accept 3.2 and 3.8 tokens a round can decode at the
same speed with one round 20% faster. So such a plugin's verdict reads `round_ms` in place of decode; a plugin in
the drafter or verifier slot, whose drafts are what a round keeps, is still judged by decode.

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
declares is one its code asks `load_kernel` for and is one of crit's kernels (`kernels/techniques/<family>/<id>/kernel.json`),
and adds plugins into a temporary engine where their tests pass again. `tests/test_device_contracts.py` holds
plugins to the device extensions they provide, and `tests/test_core_rounds.py`, `tests/test_python_rounds.py` and
`tests/test_device_engine_rounds.py` cover Core's round driver.
`tests/test_run_ablate.py` packages a toy engine built from three plugins as a run's candidate, serves it through
crit serve and ablates it: drafting three tokens a round pays, the slow attention plugin slows the engine, and an
engine that does not report its plugins fails every variant.
