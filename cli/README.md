# crit CLI

The `crit` command is installed with the Python distribution (`python -m pip install -e .` from
the checkout, or `python -m pip install .`) and drives every part of crit. Each subcommand prints its own options with
`--help`. Both installs build engines: the wheel carries crit's plugin library (Inference) as package data, so
`crit plugins` and `crit inference` act on it from an installed crit as from a checkout; `crit run` keeps its records
against a crit checkout (`--project`) or, from an installed crit without one, in any folder (`--project`, default the
current one), whose `settings.toml` it reads and whose `runs/` is the default root. Such a run's brief names the
installed release's guides, crit's repository at its tag, instead of a checkout.

| Command | What it does | Guide |
| --- | --- | --- |
| `crit run` | Keeps a run's records: create, recon, plan, candidate, package, serve, measure, ablate, eval, seal, select, stop, cost | [Run records](../docs/RUN_RECORDS.md) |
| `crit eval` | The common suite: probe the device, pin outputs, score quality references, run, compare, A/B, profile kernels | [Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md) |
| `crit serve` | Serves a sealed candidate over the OpenAI, Anthropic, Responses and `/api/chat` APIs | [API](../api/README.md) |
| `crit pull` | Installs a crit-hub engine (requirements, pinned source, build, checked checkpoint) for `crit serve --engine` | [Installed engines](../docs/SERVING.md#installed-engines) |
| `crit bench` | Measures an installed crit-hub engine with crit Eval beside its card's numbers | [Benchmark your machine](../docs/SERVING.md#benchmark-your-machine) |
| `crit hub` | Adds a `crit bench` result to its crit-hub card (`crit hub report`) for a pull request | [Benchmark your machine](../docs/SERVING.md#benchmark-your-machine) |
| `crit smoke` | Checks a served engine's quality paths in minutes: prefill that never depends on prompt length, decode against prefill, request isolation, recall | [Smoke tests](../smoke/README.md) |
| `crit kernels` | Finds, inspects and retains external kernels for the target hardware | [Kernels](../kernels/README.md) |
| `crit models` | Quantizes weights, distills draft heads and records each transform | [Models](../models/README.md) |
| `crit plugins` | Lists, checks and tests an engine's plugins and shows the variants `crit run ... ablate` serves | [Plugins](../docs/PLUGINS.md) |
| `crit inference` | Lists, searches, shows and checks the Inference techniques | [Inference](../inference/README.md) |
| `crit mcp` | Serves these commands to an agent as MCP tools, long ones as jobs it waits on | [MCP](../mcp/README.md) |
| `crit contracts` | Prints where Core's native contracts live | [Core](../core/README.md) |

## A run, start to finish

```text
crit eval probe --out device.json
crit run --project CRIT_ROOT create --record request.json
crit run --project CRIT_ROOT --id RUN_ID recon
crit run --project CRIT_ROOT --id RUN_ID resolve --record inputs.json
crit run --project CRIT_ROOT --id RUN_ID plan --record plan.json
crit run --project CRIT_ROOT --id RUN_ID candidate --source SOURCE_DIR --record proposal.json
crit run --project CRIT_ROOT --id RUN_ID package --candidate ID --source PACKAGE_DIR --record recipe.json
crit run --project CRIT_ROOT --id RUN_ID serve --check --candidate ID
crit run --project CRIT_ROOT --id RUN_ID ablate --candidate ID
crit run --project CRIT_ROOT --id RUN_ID eval --candidate ID --data EVAL_DATA_DIR
crit run --project CRIT_ROOT --id RUN_ID seal --candidate ID --lifecycle completed
crit run --project CRIT_ROOT --id RUN_ID select --candidate ID
crit serve --workspace CRIT_ROOT/runs
```

Plan, candidate, package, the checks, eval and seal repeat once per candidate, with builds recorded in between; [Building engines](../docs/BUILDING_ENGINES.md)
gives the full sequence and [AGENTS.md](../AGENTS.md) the rules an agent follows.

## How it is built

[cli.py](crit/cli.py) routes each subcommand to its package. Beside the packages, two leaf modules serve all of
them: [records.py](crit/records.py), strict JSON records, SHA-256 identities, file inventories and canonical
paths, and [lease.py](crit/lease.py), the single-writer lease the run coordinator and crit serve's runtime
ownership take. [run/](crit/run/) holds the run records: a
single-writer coordinator ([store.py](crit/run/store.py)) that writes every record and refuses an invalid
one, strict record schemas ([records.py](crit/run/records.py)), candidate plans and their checks
([plans.py](crit/run/plans.py)), model reconnaissance ([recon.py](crit/run/recon.py)), captured case requests
([capture.py](crit/run/capture.py)), one way to reserve, log and settle an attempt of a candidate
([attempt.py](crit/run/attempt.py)), builds under a build attempt ([build.py](crit/run/build.py)), checks from Eval
results and the standard serving checks ([checks.py](crit/run/checks.py)), Eval and measurement
through `crit serve` ([evaluation.py](crit/run/evaluation.py), [measure.py](crit/run/measure.py)), ablations of a
candidate built from plugins ([ablate.py](crit/run/ablate.py)) and build
cost from the agent's transcripts ([cost.py](crit/run/cost.py)). [pull/](crit/pull/) installs crit-hub engines:
the pipeline ([cli.py](crit/pull/cli.py)), the requirements check ([requirements.py](crit/pull/requirements.py))
and the engine store `crit serve --engine` reads ([store.py](crit/pull/store.py)). [process/](crit/process/) owns the engine
processes it launches: a per-user ownership record, a guardian that stops children when their parent dies,
and process trees that Windows owns before they run,
so a crashed run's engine processes are retired, not left holding the GPU.
