# crit MCP

`crit_mcp` is `crit mcp`: a [Model Context Protocol](https://modelcontextprotocol.io) server that offers crit's
command line to an agent as tools, over stdio. An agent whose harness speaks MCP (Claude Code, Codex, Cline and
most others) then drives a run through typed tools instead of composing shell commands from
[AGENTS.md](../AGENTS.md), and waits on the long ones (builds, Eval, references) through jobs the server owns.
It needs nothing beyond crit itself.

```sh
python -m pip install -e .        # from the crit checkout
crit mcp --root /path/to/runs     # what the client starts; it speaks MCP on stdin and stdout
```

Register it with the client, for example Claude Code:

```sh
claude mcp add crit -- crit mcp --root /path/to/runs
```

or a client's JSON configuration:

```json
{"mcpServers": {"crit": {"command": "crit", "args": ["mcp", "--root", "/path/to/runs"]}}}
```

Run the server where the engines run. A Windows client driving crit in WSL starts it there, with the venv's
`crit`: `"command": "wsl.exe", "args": ["-d", "DISTRO", "-e", "/path/to/venv/bin/crit", "mcp", "--root", "/path/to/runs"]`.
Paths in tool calls are then that machine's paths: give them absolute.

## Tools

Every subcommand of `crit run`, `crit eval`, `crit kernels`, `crit models` and `crit inference` is a tool, and so
are `crit pull` and `crit serve`: `crit run plan` is `crit_run_plan`, `crit eval status` is `crit_eval_status`. A
tool's input schema is read from the command's own parser when the server starts, so its options, types, choices
and help are the command line's and never drift from it. Arguments are the options' long names with underscores
(`--from-engine` is `from_engine`); a switch is `true`, a repeatable option a list. Every `record` option takes the
JSON object itself (the server writes it to a file for the command) or a JSON file's path, and a JSON-text option
such as `quality_budget` takes an object. `project` defaults to the checkout the server runs from (a crit installed
without one: the folder the commands run in, `--cwd`), and `root` (the
run root) of `crit run` and `workspace` of `crit serve` to `--root`. `crit_run_next` gives the run's next step
and `crit_run_show` its records.

`crit plugins`, `crit smoke`, `crit bench` and `crit hub` are not tools: an engine's plugins are files the agent
writes, so it runs `crit plugins list|add|test` and `crit smoke --url` in its own shell beside the server.
`crit_run_serve` with `check` runs a candidate's plugin tests and crit smoke as part of the serving check.

| Tool | What it does |
| --- | --- |
| `crit_run_*` | A run's records: next, create, recon, resolve, plan (`check` to check it without recording, `draft` for its skeleton), candidate, build, package, measure, eval, serve, check, seal, select, stop, cost, replay, show, ... |
| `crit_eval_*` | crit Eval: probe, prepare, reference, run, status, kernels, workbench, compare, ab, ... |
| `crit_pull`, `crit_serve` | Install a crit-hub engine; serve sealed candidates or an installed engine |
| `crit_kernels_*` | List, build and sweep crit's kernel templates on this device; scaffold a workbench for a kernel the engine writes |
| `crit_models_*` | Plan, quantize and verify a weight transform, bind its Eval; distill a drafter (only on the user's word: a run never trains one) |
| `crit_inference_*` | List, search, show and check the Inference techniques |
| `crit_serve_status` | A running `crit serve`'s `/health` and `/v1/models`; its `api_key` goes only to this machine or a private network address |
| `crit_job_wait`, `crit_job_output`, `crit_job_stop`, `crit_jobs` | Follow the commands' jobs |

Every tool carries MCP annotations: its `title`, `readOnlyHint`, `destructiveHint` and `openWorldHint`, so a
client can run the read-only ones without asking and confirm the rest. Read-only tools only read records, logs,
files or the device (`crit_run_next`, `crit_run_show`, `crit_eval_status`, `crit_kernels_list`, `crit_inference_*`,
`crit_job_wait`, ...). Destructive tools end something or cannot be undone: `crit_run_stop`, `crit_run_seal`,
`crit_run_select`, `crit_run_recover` and `crit_job_stop`. Every other tool adds records or files and removes
nothing. `crit_pull`, `crit_eval_fetch`, `crit_eval_run` and `crit_run_eval` are open-world: they can
download from crit-hub, an engine's git repository and Hugging Face. [tools.py](crit_mcp/tools.py) holds the table, and a tool it does
not name (a newer crit Eval's subcommand) takes the protocol's cautious defaults, destructive and open-world.

## Jobs

Each call runs its command as a job: an owned process tree (a process group, or a Windows job object) whose output
and errors go to files. The call waits `wait_seconds` (default `--wait`, 30) and then returns the result when the
command has ended: its exit code, its JSON output parsed as `result` (else its `output` text), the end of its
errors (where crit prints reports, such as `crit run eval`'s status and priorities) and the paths of both logs.
A command still running returns its job instead, and keeps running: `crit_job_wait` waits on it again (at most an
hour per call, with progress notifications to a client that asks for them), `crit_job_output` reads either log from
any offset and `crit_job_stop` interrupts it as Ctrl+C would, so crit settles the attempt as interrupted and
releases its leases (an interrupted Eval goes on with `crit_run_eval` and `resume`), ending the tree if it has not
stopped after `grace_seconds`. On Windows the stop ends the tree at once. Keep `wait_seconds` below the client's
own timeout for a tool call. While Eval or a build holds the GPU, an agent does the next round's CPU work and waits
on the job afterwards, as AGENTS.md's long jobs ask.

The jobs belong to the server: when the client closes it (or ends it with SIGTERM) it stops every job still
running, so no command outlives the session unseen. A server killed outright cannot: on Linux its jobs then run
on, and `crit_run_recover` proves an abandoned lease.

## Resources

AGENTS.md and the guides it links (Building engines, Run records, Plugins, Serving, the contribution criteria,
Contributing and the CLI, Inference, Kernels, Models, smoke and MCP READMEs) are resources, `crit://AGENTS.md`,
`crit://docs/PLUGINS.md` and so on, for an agent that does not share the checkout's filesystem. The server's instructions tell the agent to read AGENTS.md first. A crit
installed from its wheel carries them all (in `crit/library/guides`), so `crit mcp` serves the guides of the
release it runs without a checkout.

## Options

| Option | Purpose |
| --- | --- |
| `--root DIR` | Run root for `crit run` and `crit serve` calls that name none (default: `PROJECT/runs`) |
| `--project DIR` | crit checkout the commands use, whose `settings.toml` gives the device and headroom (default: this checkout; a crit installed without one takes any folder, default the folder the commands run in) |
| `--cwd DIR` | Folder the commands run in (default: the current one) |
| `--jobs DIR` | Folder of the jobs' logs and record files (default: a new temporary folder) |
| `--wait SECONDS` | How long a call waits for its command by default (30) |
| `--only FAMILY` | Offer only these command families' tools (repeatable): `run`, `eval`, `pull`, `serve`, `kernels`, `models`, `inference` |
| `--without TOOL` | Leave out one tool (repeatable), such as `crit_run_cost`, which reads the agent's transcript files for their token counts |

The server speaks protocol versions 2024-11-05 to 2025-11-25 over stdio, one JSON message per line. It has no
HTTP transport: the agent's harness starts it beside crit.

## What it runs and what it reaches

`crit mcp` runs crit's own commands on the machine it starts on, as the command line would: they build and run
native code (candidates' builds, `crit serve`, crit Eval's kernels and probes) and keep their records under the
run root. The tools reach only these places:

| Where | When |
| --- | --- |
| The local filesystem | Every command: run records, logs, checkpoints and the folders a call names |
| `127.0.0.1` | `crit serve`, `crit_run_measure`, `crit_run_eval` and `crit_serve_status` talk to a local crit serve |
| huggingface.co | `crit_pull` downloads an engine's checkpoint or prebuilt artifact; `crit_eval_fetch`, and `crit_eval_run` or `crit_run_eval` when the data is missing, download crit Eval's reference package |
| crit-hub.dev and the engine's git repository | `crit_pull` reads crit-hub's index and the engine's card, and fetches its source at the card's commit, which it builds and runs locally |

## Privacy

crit collects no data and sends nothing to crit-labs: no telemetry, analytics or accounts. The downloads above
send the requests any download does (the address and the file asked for) to those hosts, under their own
privacy policies. Everything the commands write stays in the folders on your machine that they name, for as
long as you keep it. `crit_run_cost` reads only the token counts and timestamps of the transcript files a call
names, on this machine; leave it out with `--without crit_run_cost`. Questions go to
[crit's issues](https://github.com/crit-labs/crit/issues).

## Examples

Prompts an agent with `crit mcp` handles end to end:

- "Build me a fast engine for Qwen3.8-27B on this GPU, and serve the best candidate."
- "Find the inference techniques for a hybrid model's KV cache and show me how candidates use them."
- "Install the crit-hub engine OWNER/NAME, serve it, and check that it is healthy."
- "Run Eval on candidate-0003 and tell me which gates it fails and why."

Report problems in [crit's issues](https://github.com/crit-labs/crit/issues).
