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
run root) of `crit run` and `workspace` of `crit serve` to `--root`. `crit_run_show` gives the run's state.

| Tool | What it does |
| --- | --- |
| `crit_run_*` | A run's records: create, recon, resolve, plan (`check` to check it without recording), candidate, build, package, measure, eval, serve, check, seal, select, stop, cost, replay, show, ... |
| `crit_eval_*` | crit Eval: probe, prepare, reference, run, status, kernels, workbench, compare, ab, ... |
| `crit_pull`, `crit_serve` | Install a crit-hub engine; serve sealed candidates or an installed engine |
| `crit_kernels_*`, `crit_models_*`, `crit_inference_*` | Find and retain kernels and scaffold workbenches; transform weights; search the Inference techniques |
| `crit_serve_status` | A running `crit serve`'s `/health` and `/v1/models` |
| `crit_job_wait`, `crit_job_output`, `crit_job_stop`, `crit_jobs` | Follow the commands' jobs |

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

AGENTS.md and the guides it links (Building engines, Run records, Serving, the contribution criteria and the
toolkit READMEs) are resources, `crit://AGENTS.md`, `crit://docs/RUN_RECORDS.md` and so on, for an agent that does
not share the checkout's filesystem. The server's instructions tell the agent to read AGENTS.md first.

## Options

| Option | Purpose |
| --- | --- |
| `--root DIR` | Run root for `crit run` and `crit serve` calls that name none (default: `PROJECT/runs`) |
| `--project DIR` | crit checkout the commands use, whose `settings.toml` gives the device and headroom (default: this checkout; a crit installed without one takes any folder, default the folder the commands run in) |
| `--cwd DIR` | Folder the commands run in (default: the current one) |
| `--jobs DIR` | Folder of the jobs' logs and record files (default: a new temporary folder) |
| `--wait SECONDS` | How long a call waits for its command by default (30) |
| `--only FAMILY` | Offer only these command families' tools (repeatable): `run`, `eval`, `pull`, `serve`, `kernels`, `models`, `inference` |

The server speaks protocol versions 2024-11-05 to 2025-11-25 over stdio, one JSON message per line. It has no
HTTP transport: the agent's harness starts it beside crit.
