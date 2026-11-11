# The smallest plugin engine

Every crit engine is Core plus plugins, one per slot ([Plugins](../../../docs/PLUGINS.md)). This folder is the
smallest such engine: a plugin package whose plugins run on the CPU in torch, which Core's loader assembles and runs
on Core's concurrent runtime. It is a teaching and test engine, with no pretrained weights and no speed to measure:
its model's next token is a hash of every token its row has seen, so any wrong bookkeeping (a lost KV row, a row that
borrows another's state, a skipped commit) changes the output. crit's own tests serve it end to end
([tests/support/engines.py](../../../tests/support/engines.py)), so it stays in step with the loader.

```text
plugin-engine/
  configuration.json        the selection: one plugin per slot
  plugins/
    toy-model/              model: the batched step model
    toy-placement/          placement: where each tensor lives
    toy-projections/        projections: the matrix arithmetic
    toy-attention/          attention: reads the KV rows
    toy-kv/                 kv_cache: the KV pool
    toy-sampler/            sampler: a greedy draw over each row's logits
    toy-drafter/            drafter and verifier: optional, they make a decode round emit several tokens
    toy-verifier/
    toy-vision/             a second model plugin, toy-model taking images
```

## Check and test it

From the repository root, with crit installed (`python -m pip install -e .`) and torch:

```sh
crit plugins list inference/examples/plugin-engine/plugins
crit plugins check inference/examples/plugin-engine/plugins --selection inference/examples/plugin-engine/configuration.json
crit plugins test inference/examples/plugin-engine/plugins
```

`check` applies Core's loader rules without running any code: every slot an engine fills has one plugin, every
`needs` is met by another plugin's `provides`, nothing conflicts, and the plugins are created in dependency order.
`test` runs each plugin's own tests.

## Follow the engine

Read the plugins in this order:

1. **Each manifest says what the plugin fills, needs and provides.** `toy-kv/plugin.json` fills `kv_cache` and
   provides `kv:toy-rows`; `toy-attention/plugin.json` fills `attention` and needs that fact, so a selection without
   a KV plugin that provides it is refused when it loads.
2. **The model plugin is Core's step model.** `toy-model/plugin.py` receives the other plugins as `parts` and calls
   each one in its slot: it asks `placement` where its table lives, loads its one weight through `projections`, keeps
   every token in the `kv_cache` pool, runs `attention` over the step's rows and hands each row's logits to the
   `sampler`. Core's loader refuses an engine whose model never calls a computing slot's plugin.
3. **Every step holds many requests' rows.** `step(rows)` computes prompt chunks and decode rows together, and a row's
   token never depends on the rows beside it, so a reply is the same batched as alone. Its `info()` declares
   `max_concurrency` and its limits.
4. **Drafting is two more slots.** With `toy-drafter` and `toy-verifier` in the selection, a decode round emits the
   drafter's tokens per round, the shape a real drafter and verifier take.

Some options of `toy-model` (`skip`, `prompt_graphs`, `prompt_width`) exist only so crit's tests can check the loader's
refusals and captured prompt chunks; a real engine has no such switches.

## From here to a real engine

A real engine keeps the same shape and replaces the insides: a `model` plugin for the checkpoint's architecture, and
for each slot a plugin from crit's library (`crit plugins list --model MODEL --device cuda`, `crit plugins add`) or the
engine's own. Building, measuring and serving it go through `crit run` ([AGENTS.md](../../../AGENTS.md)).
