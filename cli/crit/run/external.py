"""External candidates: an engine crit did not assemble from plugins, recorded by crit run like any candidate.

An external engine takes crit's techniques and kernel templates into its own code (charter, vision A) and runs as
its own OpenAI-compatible server. Its candidate's configuration names it in `engine`:

    "engine": {
      "kind": "external",
      "launch": {"command": ["python", "server.py", "--port", "{port}", "--model", "{model}",
                             "--context", "{context_tokens}"],
                 "cwd": ".", "env": {}, "health": "/v1/models", "ready_seconds": 900},
      "capabilities": ["chat-tools-v1"],
      "max_concurrency": 8,
      "options": {"draft_width": {"flag": "--draft-width", "default": 4},
                  "graphs": {"flag": "--enforce-eager", "switch": true, "default": false}}
    }

The package holds the engine's files and no plugins/: Core's loader never runs it. crit run serve, eval, measure,
ablate and kernels start it from `launch.command` under the attempt's reservation (placeholders {port}, {host},
{model}, {context_tokens} and {package}, in the command and in `env`), in its own process tree, wait until `health`
answers 200, run their requests against its URL, and stop the whole tree when the work ends, fails or passes the
reservation's deadline, as a build's command is stopped (crit.process.tree). A cleanup that cannot prove the tree
gone leaves the attempt's cleanup unsettled. The engine's output goes to candidates/ID/logs/engine-ATTEMPT.log, kept
with the attempt.

`options` are the switches the engine declares: each a command-line flag with its value as built (`default`), or a
bare flag (`switch`) given when the value is true. crit run ablate varies only these (--set NAME=VALUE), the A/B of
an external engine; an engine that declares none has no ablation, and the ablation report says so. A launch command
without {context_tokens} serves at the engine's own context: a part that asks for another context is served at it,
and the attempt's log says so.
"""
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request

from .attempt import Expired, remaining
from .records import keys, load_run, read_json, safe_path, text
from .served import Served

KIND = "external"
PLACEHOLDERS = ("port", "host", "model", "context_tokens", "package")
HEALTH = "/v1/models"
READY_SECONDS = 900
LAUNCH_KEYS = {"command"}, {"cwd", "env", "health", "ready_seconds"}
ENGINE_KEYS = {"kind", "launch"}, {"capabilities", "max_concurrency", "options"}
SERVED_BY = "the candidate's launch command"


def spec_of(configuration):
    """The external engine a candidate's configuration declares (`engine`, validated), or None for a plugin package."""
    engine = configuration.get("engine") if isinstance(configuration, dict) else None
    if engine is None:
        return None
    if not isinstance(engine, dict) or engine.get("kind") != KIND:
        raise ValueError(f'configuration.engine declares an external engine: {{"kind": "{KIND}", "launch": {{...}}}} '
                         "(a plugin package names its plugins in configuration.plugins instead)")
    if configuration.get("plugins") is not None:
        raise ValueError("configuration declares an external engine (engine) and a plugin selection (plugins): an "
                         "engine is one or the other")
    keys(engine, *ENGINE_KEYS)
    launch = engine["launch"]
    if not isinstance(launch, dict):
        raise ValueError("configuration.engine.launch is {command, cwd, env, health, ready_seconds}")
    keys(launch, *LAUNCH_KEYS)
    command = launch["command"]
    if not isinstance(command, list) or not command or not all(isinstance(p, str) and p for p in command):
        raise ValueError("configuration.engine.launch.command is the engine server's command line, a list of strings")
    env = launch.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError("configuration.engine.launch.env maps variable names to strings")
    if not any("{port}" in part for part in command + list(env.values())):
        raise ValueError("configuration.engine.launch.command takes {port}: crit picks the port the engine listens on, "
                         "so two attempts never meet on one")
    if "cwd" in launch:
        text(launch["cwd"], "configuration.engine.launch.cwd (a folder of the package)", 400)
    health = launch.get("health", HEALTH)
    if not isinstance(health, str) or not health.startswith("/"):
        raise ValueError("configuration.engine.launch.health is the path that answers 200 once the engine serves")
    ready = launch.get("ready_seconds", READY_SECONDS)
    if isinstance(ready, bool) or not isinstance(ready, (int, float)) or ready <= 0:
        raise ValueError("configuration.engine.launch.ready_seconds is how long the engine may take to answer health")
    capabilities = engine.get("capabilities", [])
    if not isinstance(capabilities, list) or not all(isinstance(c, str) for c in capabilities):
        raise ValueError("configuration.engine.capabilities lists what the engine serves beside plain chat (crit "
                         "serve's names, such as chat-tools-v1 and chat-logprobs-v1)")
    concurrency = engine.get("max_concurrency")
    if concurrency is not None and (type(concurrency) is not int or concurrency < 1):
        raise ValueError("configuration.engine.max_concurrency is how many requests the engine runs at once")
    options = engine.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("configuration.engine.options maps each switch the engine declares to {flag, default, switch}")
    for name, option in options.items():
        text(name, "configuration.engine.options name", 100)
        if not isinstance(option, dict):
            raise ValueError(f"configuration.engine.options.{name} is {{flag, default, switch}}")
        keys(option, {"flag"}, {"default", "switch"})
        if not isinstance(option["flag"], str) or not option["flag"].startswith("-"):
            raise ValueError(f"configuration.engine.options.{name}.flag is the command-line flag it sets (--name)")
        if "switch" in option and type(option["switch"]) is not bool:
            raise ValueError(f"configuration.engine.options.{name}.switch is true for a bare flag")
    return {"kind": KIND, "launch": {"command": list(command), "cwd": launch.get("cwd", "."), "env": dict(env),
                                     "health": health, "ready_seconds": float(ready)},
            "capabilities": list(capabilities), "max_concurrency": concurrency, "options": dict(options)}


def check_package(folder, configuration, what, launchable=True):
    """Refuse `folder` unless it is the external engine `configuration` declares: no plugins/ (Core's loader never
    runs an external candidate) and, once `launchable` (a package, not yet a candidate's source a build may complete),
    the launch command's folder inside it."""
    spec = spec_of(configuration)
    folder = Path(folder)
    if (folder / "plugins").exists():
        raise ValueError(f"{what} declares an external engine (configuration.engine) but holds plugins/: an engine is "
                         "either Core plus a plugin selection or an external engine started from its launch command")
    if launchable and not launch_folder(folder, spec).is_dir():
        raise ValueError(f"{what} has no folder {spec['launch']['cwd']!r}, where configuration.engine.launch.cwd starts "
                         "the engine")
    return spec


def launch_folder(package, spec):
    """The folder of the package the launch command starts in (configuration.engine.launch.cwd, default the package)."""
    cwd = spec["launch"]["cwd"]
    return Path(package) if cwd in (".", "./", "") else safe_path(package, cwd)


def candidate_spec(folder):
    """The external engine of a candidate's folder (candidate.json's configuration), or None for a plugin package."""
    return spec_of(read_json(Path(folder) / "candidate.json")["configuration"])


def option_values(spec, overrides=None):
    """Every declared option's value: as built (its default), `overrides` over it."""
    values = {name: option.get("default") for name, option in spec["options"].items()}
    for name, value in (overrides or {}).items():
        if name not in spec["options"]:
            raise ValueError(f"the engine declares no option {name!r} (configuration.engine.options: "
                             + (", ".join(sorted(spec["options"])) or "none") + ")")
        values[name] = value
    return values


def launch_command(spec, values, places):
    """(argv, env): the launch command with its placeholders filled from `places` and the options' flags appended."""
    def fill(part):
        for name in PLACEHOLDERS:
            part = part.replace("{" + name + "}", str(places[name]))
        return part
    argv = [fill(part) for part in spec["launch"]["command"]]
    for name, option in spec["options"].items():
        value = values.get(name)
        if option.get("switch"):
            if value:
                argv.append(option["flag"])
        elif value is not None:
            argv += [option["flag"], value if isinstance(value, str) else json.dumps(value)]
    return argv, {key: fill(value) for key, value in spec["launch"]["env"].items()}


def free_port(host):
    """A port no process listens on at `host` now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def _get(url, timeout):
    """GET `url` directly (no proxy): (status, JSON body or None)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(urllib.request.Request(url), timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        return error.code, None
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, None


class ExternalEngine:
    """What crit run's measurements read of an engine while it is served (crit serve's ServedEngine offers the same):
    its capabilities, the process it runs in and how many requests it runs at once."""

    def __init__(self, spec, tree, url, tokens, values):
        self.tree, self.url, self.tokens, self.values = tree, url, tokens, values
        self.model, self.card = None, {}
        status, models = _get(url + "/v1/models", 30)
        if status == 200 and isinstance(models, dict) and models.get("data"):
            self.card = models["data"][0] if isinstance(models["data"][0], dict) else {}
            self.model = self.card.get("id")
        status, health = _get(url + "/health", 30)
        self.health = health if status == 200 and isinstance(health, dict) else {}
        self.declared = spec

    def spec(self):
        """The engine's capabilities: the ones its configuration declares, else its model card's."""
        return {"capabilities": list(self.declared["capabilities"] or self.card.get("capabilities") or [])}

    def host_pid(self):
        return self.tree.process.pid

    def capacity(self):
        """max_concurrent_requests from the engine's /health or card, else its configuration's max_concurrency."""
        for source in (self.health, self.card):
            if isinstance(source.get("max_concurrent_requests"), int):
                return source["max_concurrent_requests"]
        return self.declared["max_concurrency"]


class External(Served):
    """The external candidate served from its launch command under one attempt, as Served serves a plugin package:
    open() starts it (again, at another context or with other options), until() bounds it by the reservation's
    deadline, stop() stops its whole process tree."""

    def __init__(self, store, run_id, candidate_id, attempt, kv_tokens, log, name, timeout=600,
                 address=("127.0.0.1", 0), environment=None):
        super().__init__(store, run_id, candidate_id, getattr(attempt, "id", attempt), kv_tokens, log, name, timeout,
                         address)
        root = store.directory(run_id)
        self.folder = safe_path(root, f"candidates/{candidate_id}")
        self.package = self.folder / "package"
        self.spec = spec_of(read_json(self.package / "configuration.json"))
        self.model_path = load_run(root)["spec"]["model_path"]
        self.server = f"external {run_id}/{candidate_id}"
        self.environment = dict(environment or {})   # crit's own variables for the engine (kernels: the tracer)
        self.output = self.folder / "logs" / f"engine-{self.attempt}.log"
        if hasattr(attempt, "evidence"):
            attempt.evidence.append(self.output)
        self.fixed_context_noted = False

    def open(self, tokens=None, configuration=None):
        """Start the engine at `tokens` of context (default the run's) with `configuration`'s options ({"options":
        {NAME: VALUE}}) over the ones it was built with. An engine already running at the same context and options is
        the one returned; otherwise it is stopped and started again. Returns (ExternalEngine, an object naming the
        served model, port)."""
        tokens = tokens or self.kv_tokens
        values = option_values(self.spec, (configuration or {}).get("options"))
        fixed = not any("{context_tokens}" in part
                        for part in self.spec["launch"]["command"] + list(self.spec["launch"]["env"].values()))
        if fixed and tokens != self.kv_tokens:
            if not self.fixed_context_noted:
                self.log(f"external engine: the launch command takes no {{context_tokens}}, so the engine serves at its "
                         f"own context where {tokens:,} tokens were asked for")
                self.fixed_context_noted = True
            tokens = self.kv_tokens
        with self.lock:
            current = self.current
        if current is not None and not self.expired:
            engine = current[0]
            if engine.tokens == tokens and engine.values == values and engine.tree.process.poll() is None:
                return engine, engine, self.port
        self.stop()
        with self.lock:
            if self.expired:
                raise Expired(f"{self.name} passed its reserved deadline")
        host = self.address[0]
        self.port = self.address[1] or free_port(host)
        cwd = launch_folder(self.package, self.spec)
        argv, env = launch_command(self.spec, values, {"port": self.port, "host": host, "model": self.model_path,
                                                       "context_tokens": tokens, "package": self.package})
        from crit.process.tree import ProcessTree
        self.output.parent.mkdir(parents=True, exist_ok=True)
        stream = self.output.open("ab")
        stream.write(f"$ {' '.join(argv)}\n".encode("utf-8"))
        stream.flush()
        shown = ", ".join(f"{k}={v}" for k, v in values.items())
        self.log(f"external engine: starting {self.candidate_id} at {tokens:,} tokens of context"
                 + (f" ({shown})" if shown else "") + f" on port {self.port}")
        try:
            tree = ProcessTree(argv, cwd, stream, env=dict(os.environ, **env, **self.environment))
        except BaseException:
            stream.close()
            raise
        with self.lock:
            self.current = (None, tree, stream)
        try:
            url = f"http://127.0.0.1:{self.port}" if host in ("0.0.0.0", "") else f"http://{host}:{self.port}"
            self._ready(tree, url)
            engine = ExternalEngine(self.spec, tree, url, tokens, values)
        except BaseException:
            self.stop()
            raise
        with self.lock:
            self.current = (engine, tree, stream)
        self.log(f"external engine: serving {engine.model or 'its model'} at {url}/v1 (process {tree.process.pid})")
        return engine, engine, self.port

    def _ready(self, tree, url):
        """Wait until the engine answers its health path, it exits, or its ready time or the deadline passes."""
        limit = time.monotonic() + min(self.spec["launch"]["ready_seconds"], max(0.0, remaining_seconds(self)))
        path = self.spec["launch"]["health"]
        while True:
            code = tree.process.poll()
            if code is not None:
                raise ValueError(f"the engine exited {code} before {path} answered; its output: "
                                 f"{self.output.relative_to(self.folder.parent.parent).as_posix()}: " + tail(self.output))
            if self.expired:
                raise Expired(f"{self.name} passed its reserved deadline while the engine started")
            try:
                status, _ = _get(url + path, 5)
                if status == 200:
                    return
            except OSError:
                pass
            if time.monotonic() >= limit:
                raise TimeoutError(f"the engine did not answer {path} within its ready time "
                                   f"({self.spec['launch']['ready_seconds']:,.0f} s, or the reservation's end)")
            time.sleep(.1)

    def stop(self):
        """Stop the engine's whole process tree, if one runs; a cleanup that cannot prove it gone raises with
        `settled` False."""
        with self.lock:
            current, self.current = self.current, None
        if current is None:
            return
        _, tree, stream = current
        try:
            tree.close()
        except BaseException as exc:
            self.failure = f"stopping the external engine's process tree: {type(exc).__name__}: {exc}"
            exc.settled = False
            raise
        finally:
            stream.close()
        self.log("external engine: stopped (its process tree is gone)")


def remaining_seconds(served):
    """Seconds left of the attempt's reservation (attempt.Attempt keeps it, Served knows only its id)."""
    deadline = getattr(served, "deadline_at", None)
    return remaining(deadline) if deadline else float("inf")


def tail(path, size=2000):
    try:
        return path.read_bytes()[-size:].decode("utf-8", "replace").strip() or "(no output)"
    except OSError:
        return "(no output)"


def serving(store, run_id, candidate_id, attempt, kv_tokens, log, name, timeout=600, address=("127.0.0.1", 0),
            environment=None):
    """The candidate served under `attempt` (attempt.Attempt): through crit serve for a plugin package, from its launch
    command for an external candidate."""
    folder = safe_path(store.directory(run_id), f"candidates/{candidate_id}")
    if candidate_spec(folder) is None:
        return Served(store, run_id, candidate_id, attempt.id, kv_tokens, log, name, timeout, address)
    served = External(store, run_id, candidate_id, attempt, kv_tokens, log, name, timeout, address, environment)
    served.deadline_at = attempt.deadline_at
    return served


# ----------------------------------------------------------------------------------------------- kernel profiling

def kernels(store, run_id, candidate_id, seconds=None, nvcc=None):
    """crit run ... kernels: trace the external candidate's own process tree while it serves the run's cases, the way
    crit eval kernels traces a package in process: crit-eval's CUPTI injection library is in the engine's environment
    (CUDA_INJECTION64_PATH), so every kernel any of its processes launches is recorded, and each case's prefill and
    decode are told apart by when its first and last reply pieces arrived. Where tracing is not possible (not Linux,
    no CUDA toolkit with CUPTI, an engine whose processes record no kernel) the report says so instead. Keeps
    candidates/ID/logs/kernels-ATTEMPT.json under a check attempt; records no check and no benchmark."""
    from .attempt import Attempt, packaged, unsealed
    from .measure import _request, case_body, case_request
    from .records import inputs, load_run, write_json
    root = store.directory(run_id)
    run = load_run(root)
    folder = unsealed(root, candidate_id, "The candidate is sealed: trace it before sealing it")
    packaged(folder, "tracing", "crit run kernels traces")
    if candidate_spec(folder) is None:
        raise ValueError("crit run kernels traces an external candidate's own processes; a plugin package runs in "
                         "process, where crit eval kernels --package PACKAGE traces it")
    resolved = inputs(root, run)
    cases = [{"case_id": case["case_id"], "request": case_request(root, resolved["case_manifests"][case["case_id"]]),
              "output_tokens": case["workload"]["output_tokens"], "prompt_tokens": case["workload"]["prompt_tokens"]}
             for case in run["spec"]["cases"] if case["workload"]["mode"] == "resident"]
    attempt = Attempt(store, run_id, candidate_id, "check", seconds, "kernels", "crit run kernels", "The trace")
    report = folder / "logs" / f"kernels-{attempt.id}.json"
    found = {"attempt_id": attempt.id, "candidate_id": candidate_id, "traced": "the engine's own process tree"}
    with attempt.running() as log:
        reason, toolkit = _toolkit(nvcc)
        if reason is not None:
            log(f"crit run kernels: tracing is unavailable: {reason}")
            found.update(available=False, reason=reason)
        else:
            from crit_eval.limits.kernels import analyze, build, read_traces
            with tempfile.TemporaryDirectory() as directory:
                library = build(toolkit, directory)
                environment = {"CUDA_INJECTION64_PATH": str(library), "CRIT_KTRACE_DIR": directory,
                               "CRIT_KTRACE_BACKEND": "cuda"}
                served = serving(store, run_id, candidate_id, attempt, run["spec"]["hardware_constraints"]["kv_tokens"],
                                 log, "The trace", timeout=min(86400, max(1.0, remaining(attempt.deadline_at))),
                                 environment=environment)
                # Marks on the tracer's own clock (CUPTI's, which its kernel records carry), read in this process
                # through the library the engine loads: the host's realtime clock drifts from it by milliseconds.
                now, clock = _tracer_clock(library)
                marks = {"info": {"name": candidate_id}, "clock": clock, "cases": []}
                with served.until(attempt.deadline_at):
                    _, _, port = served.open()
                    for case in cases:
                        body = case_body(case)
                        _request(port, body, served.timeout)   # warmup: graphs and workspaces for this length
                        offset = now() - time.perf_counter_ns()
                        started, first, last, usage, telemetry = _request(port, body, served.timeout)
                        stamp = lambda t: offset + round(t * 1e9)   # noqa: E731
                        marks["cases"].append({"id": case["case_id"], "prompt_tokens": usage["prompt_tokens"],
                                               "output_tokens": usage["completion_tokens"],
                                               "rounds": telemetry.get("decode_rounds"),
                                               "start": stamp(started), "first": stamp(first), "end": stamp(last)})
                        log(f"crit run kernels: traced {case['case_id']}")
                # The tracer writes its records as each process exits: read them once the tree is gone.
                traced, device = read_traces(directory, clock)
            if traced:
                found.update(available=True, toolkit=toolkit["nvcc"], **analyze(traced, device, marks))
            else:
                found.update(available=False, reason="no kernels were recorded: the engine may not run on CUDA, or a "
                                                     "process of it dropped the tracer (a static CUDA runtime, or an "
                                                     "environment it does not pass to its workers)")
                log("crit run kernels: " + found["reason"])
    write_json(report, found, immutable=True)
    result = attempt.settle(evidence=(report,))
    attempt.finish()
    return {"attempt": result, "report": report.relative_to(root).as_posix(), "available": found["available"],
            **({"reason": found["reason"]} if not found["available"] else {})}


def _tracer_clock(library):
    """(now(), name): the tracer library's CUPTI clock when it exports crit_ktrace_timestamp, else the host's realtime
    clock, which read_traces converts kernel times to through the trace's clock pairs."""
    import ctypes
    try:
        now = ctypes.CDLL(str(library)).crit_ktrace_timestamp
    except (OSError, AttributeError):
        return time.time_ns, "realtime"
    now.restype, now.argtypes = ctypes.c_ulonglong, []
    return now, "cupti"


def _toolkit(nvcc=None):
    """(None, toolkit) when crit-eval's CUPTI tracer can be built here, else (why not, None)."""
    if not sys.platform.startswith("linux"):
        return f"kernel tracing injects CUPTI into the engine's processes, which needs Linux (this is {sys.platform})", None
    try:
        from crit_eval.limits.kernels import find_toolkit
        return None, find_toolkit(nvcc)
    except SystemExit as missing:
        return str(missing), None
    except ImportError as missing:
        return f"crit-eval is not installed: {missing}", None

