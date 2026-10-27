# crit CLI

The `crit` command is installed with the Python distribution (`python -m pip install -e .` from
the checkout) and drives every part of crit. Each subcommand prints its own options with `--help`.

| Command | What it does | Guide |
| --- | --- | --- |
| `crit run` | Keeps a run's records: create, recon, plan, candidate, package, serve, measure, eval, seal, select, stop, cost | [Run records](../docs/RUN_RECORDS.md) |
| `crit eval` | The common suite: probe the device, pin outputs, score quality references, run, compare, A/B, profile kernels | [Eval](https://github.com/crit-labs/crit-eval/blob/main/docs/README.md) |
| `crit serve` | Serves a sealed candidate over the OpenAI, Anthropic, Responses and Ollama APIs | [API](../api/README.md) |
| `crit kernels` | Finds, inspects and retains external kernels for the target hardware | [Kernels](../kernels/README.md) |
| `crit models` | Quantizes weights, distills draft heads and records each transform | [Models](../models/README.md) |
| `crit inference` | Lists, searches, shows and checks the Inference techniques | [Inference](../inference/README.md) |
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
crit run --project CRIT_ROOT --id RUN_ID eval --candidate ID --data EVAL_DATA_DIR
crit run --project CRIT_ROOT --id RUN_ID seal --candidate ID --lifecycle completed
crit run --project CRIT_ROOT --id RUN_ID select --candidate ID
crit serve --workspace CRIT_ROOT/runs
```

Plan, candidate, package, eval and seal repeat once per candidate, with builds and checks recorded in between; [Building engines](../docs/BUILDING_ENGINES.md)
gives the full sequence and [AGENTS.md](../AGENTS.md) the rules an agent follows.

## How it is built

[cli.py](crit/cli.py) routes each subcommand to its package. [run/](crit/run/) holds the run records: a
single-writer coordinator ([store.py](crit/run/store.py)) that writes every record and refuses an invalid
one, strict record schemas ([records.py](crit/run/records.py)), candidate plans and their checks
([plans.py](crit/run/plans.py)), model reconnaissance ([recon.py](crit/run/recon.py)), captured case requests
([capture.py](crit/run/capture.py)), builds under a build attempt ([build.py](crit/run/build.py)), checks from Eval
results and the standard serving checks ([checks.py](crit/run/checks.py)), Eval and measurement
through `crit serve` ([evaluation.py](crit/run/evaluation.py), [measure.py](crit/run/measure.py)) and build
cost from the agent's transcripts ([cost.py](crit/run/cost.py)). [process/](crit/process/) owns the engine
processes it launches: a per-user ownership record, a guardian that stops children when their parent dies,
and process trees that Windows owns before they run,
so a crashed run's engine processes are retired, not left holding the GPU.
