"""A packaged candidate served through crit serve under one of its attempts, for crit run eval, crit run measure and
crit run serve: crit serve's resident host loads the candidate's resident chat adapter, packaged and not yet sealed,
behind its chat API. One engine is served at a time, and the reservation's deadline stops it, cancelling the request
in flight.
"""
from contextlib import contextmanager
import os
import threading
import time

from crit_core.serving import (JSON_CAPABILITY, LOGPROBS_CAPABILITY, RESIDENT_CAPABILITY, SCORE_CAPABILITY,
                               STREAM_CAPABILITY)

from .attempt import Attempt, Expired, packaged, remaining, unsealed
from .records import file_sha, load_run, now, quality_mode, read_json

# What a measurement needs crit serve to offer, which Core's contract gives every engine (crit_core.engine): a resident
# chat engine that streams.
SERVED = (JSON_CAPABILITY, RESIDENT_CAPABILITY, STREAM_CAPABILITY)
# What Eval's quality needs beside them: the score call and, for its honesty check, every served token's
# log-probability.
QUALITY = (SCORE_CAPABILITY, LOGPROBS_CAPABILITY)


class Served:
    """The candidate served through crit serve. `name` names the measurement in its deadline's message, and `log`
    receives a line when an engine loads and stops; `timeout` bounds each request."""

    def __init__(self, store, run_id, candidate_id, attempt, kv_tokens, log, name, timeout=600, address=("127.0.0.1", 0)):
        self.store, self.run_id, self.candidate_id, self.attempt = store, run_id, candidate_id, attempt
        self.kv_tokens, self.log, self.name, self.timeout = kv_tokens, log, name, timeout
        self.address = address  # where the chat API listens: a free loopback port unless given
        self.server = f"crit serve {run_id}/{candidate_id}"
        self.lock = threading.Lock()
        self.current = None  # (ServedEngine, HTTP server) while one is served
        self.expired = False
        self.failure = None  # a cleanup that failed, also on the deadline's thread

    def open(self, tokens=None, configuration=None):
        """Serve the candidate at `tokens` of context (default the run's), with `configuration`'s keys over its
        configuration (an ablation's variant; default its own). Returns the ServedEngine, crit serve's Api and the
        loopback port it listens on. An engine already served is built again at the new context and configuration in
        the same resident host, which keeps the model's files, device weights and the adapter's host-side weights."""
        from crit_api.engine import ServedEngine
        from crit_api.server import Api, Server
        tokens = tokens or self.kv_tokens
        with self.lock:
            current = self.current
        if current is not None and current[1] is not None and not self.expired:
            engine, server = current
            self.log(f"crit serve: reloading {self.candidate_id} at {tokens:,} tokens of context in the same host")
            try:
                engine.reopen(None if tokens == self.kv_tokens else tokens, configuration=configuration)
            except BaseException:
                self.stop()
                raise
            return engine, server.api, server.server_address[1]
        self.stop()  # an engine that failed to open
        engine = ServedEngine(self.store.root, self.run_id, self.candidate_id, project=self.store.project,
                              timeout=self.timeout, attempt=self.attempt,
                              kv_tokens=None if tokens == self.kv_tokens else tokens, configuration=configuration)
        with self.lock:
            if self.expired:
                raise Expired(f"{self.name} passed its reserved deadline")
            self.current = (engine, None)
        self.log(f"crit serve: loading {self.candidate_id} at {tokens:,} tokens of context")
        engine.open()
        api = Api(engine)
        server = Server(self.address, api)
        with self.lock:
            self.current = (engine, server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return engine, api, server.server_address[1]

    @contextmanager
    def until(self, deadline_at):
        """Run the block with the reservation's deadline stopping the served engine, and stop it when the block ends.
        Raises Expired when the deadline stopped it, and an error with `settled` False when its cleanup failed."""
        timer = threading.Timer(max(0.0, remaining(deadline_at)), self.expire)
        timer.daemon = True
        timer.start()
        try:
            yield self
        except BaseException as exc:
            if self.expired and getattr(exc, "settled", True):
                raise Expired(f"{self.name} passed its reserved deadline") from exc
            raise
        finally:
            timer.cancel()
            timer.join()  # an expiry stopping the engine finishes first
            self.stop()  # an engine the block did not stop
        if self.failure:
            error = ValueError("Engine cleanup failed: " + self.failure)
            error.settled = False
            raise error
        if self.expired:
            raise Expired(f"{self.name} passed its reserved deadline")

    def expire(self):
        """At the reservation's deadline: stop the engine, which cancels its request in flight."""
        with self.lock:
            self.expired = True
        self.stop()

    def stop(self):
        """Stop the served engine, if one is; a failed cleanup raises with `settled` False."""
        with self.lock:
            current, self.current = self.current, None
        if current is None:
            return
        engine, server = current
        if server is not None:
            server.shutdown()
            server.server_close()
        try:
            engine.close()
            if engine.ownership.blocked:
                raise ValueError("Engine cleanup failed: " + engine.ownership.blocked)
        except BaseException as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            exc.settled = False
            raise
        self.log("crit serve: stopped")


class Reopening:
    """crit-eval's engines (its adapter) for the gate pre-check on the engine crit run serve serves: the engine at
    the run's context is the one already listening; another context reloads it in the same resident host."""

    def __init__(self, served, prepared, port):
        self.served, self.prepared = served, prepared
        self.port, self.tokens = port, served.kv_tokens
        self.server = served.server

    def create_eval_engine(self, context):
        from crit_eval.run import chat
        tokens = context["context_tokens"]
        if tokens != self.tokens:
            _, _, self.port = self.served.open(tokens)
            self.tokens = tokens
        engine = chat.Server("http://127.0.0.1:%d/v1" % self.port, self.prepared, None).create_eval_engine(context)
        engine.close = lambda: None   # the engine stays served; crit run serve stops it
        return engine


def serve(store, run_id, candidate_id, host="127.0.0.1", port=8000, seconds=None, ready=None, stop=None, check=False,
          data=None, smoke=True, plugin_tests=True):
    """crit run serve: the packaged candidate, not yet sealed, served through crit serve under a check attempt until
    Ctrl+C, `stop` (a threading.Event) or the reservation's deadline, for the checks a candidate passes before it is
    sealed (streaming, stop strings, tool calls, cancellation, Ctrl+C). `ready(port)` runs once the API listens.
    With `check`, crit first runs the package's plugin tests (crit plugins test, cached by files and environment;
    `plugin_tests` False leaves them out), then the standard serving checks (crit.run.checks.serve_checks), stops the
    engine, keeps their report as candidates/ID/logs/serve-check-ATTEMPT.json and records a functional check per case:
    the one check a candidate passes before its rung's Eval or its seal. With
    `smoke` (the default) the report adds crit smoke (_smoke): the engine's quality components at the run's context
    and at a quarter of it, in a minute or two each. With `data` too (crit-eval's prepared data for the run's model), the report adds crit-eval's gate pre-check
    (crit_eval.run.precheck): honesty of short replies and of one long reply, the request history around them, the
    device memory after the load and the speed cases at the lower context, the gates whole Eval rounds most often
    fail on, in minutes.
    An external candidate (crit.run.external) is started from its launch command instead, in its own process tree,
    and has no plugin tests; every other check is the same.
    Settles the attempt and returns its result; crit serve serves the candidate once it is sealed."""
    from .external import candidate_spec, serving
    root = store.directory(run_id)
    folder = unsealed(root, candidate_id, "The candidate is sealed: serve it with crit serve")
    if data is not None and not check:
        raise ValueError("--data is for crit run serve --check: the gate pre-check runs beside the serving checks")
    manifest = packaged(folder, "serving", "crit run serve serves")
    run = load_run(root)
    kv_tokens = run["spec"]["hardware_constraints"]["kv_tokens"]
    # The engine is served until the reservation's deadline: reaching it is how a serve without --check ends.
    attempt = Attempt(store, run_id, candidate_id, "check", seconds, "serve", "crit run serve", "crit run serve",
                      deadline=False)
    report = None
    external = candidate_spec(folder) is not None
    with attempt.running() as log:
        tested = _plugin_tests(folder, log) if check and plugin_tests and not external else []
        served = serving(store, run_id, candidate_id, attempt, kv_tokens, log, "crit run serve", address=(host, port))
        with served.until(attempt.deadline_at):
            with _probed(check and not external):
                engine, _, bound = served.open()
            log(f"Serving {run_id}/{candidate_id} (not sealed, {attempt.id}) at http://{host}:{bound}/v1; Ctrl+C stops")
            if ready is not None:
                ready(bound)
            if check:
                from .checks import full_context_checks, serve_checks
                variables = (read_json(folder / "candidate.json")["configuration"].get("recon") or {}).get(
                    "template_variables") or {}
                report = tested + serve_checks(bound, engine.spec()["capabilities"], variables, served.timeout, log)
                if not external:
                    from crit_api.residency import CANCEL_GRACE_SECONDS
                    report += full_context_checks(bound, engine, kv_tokens,
                                                  run["spec"]["hardware_constraints"]["vram_headroom_bytes"],
                                                  CANCEL_GRACE_SECONDS, served.timeout, log)
                if smoke:
                    report += _smoke(served, bound, log)
                if data is not None:
                    report += _precheck(served, run, data, bound, log)
                log("crit run serve --check: stopping the engine")
            try:
                while not check and not served.expired and not (stop is not None and stop.wait(.25)):
                    if stop is None:
                        time.sleep(.25)
            except KeyboardInterrupt:
                log("crit run serve: interrupted, stopping the engine")
    if report is not None:
        completed = attempt.outcome == "completed"
        report.append({"name": "stop_engine", "state": "passed" if completed else "failed",
                       "detail": ("the engine's whole process tree stopped, as on Ctrl+C" if external else
                                  "the engine stopped and its resident host exited, as on Ctrl+C") if completed
                       else f"stopping the engine: {attempt.failure}"})
    result = attempt.settle()
    if report is not None:
        result = {"attempt": result, "checks": _record_serve_checks(store, run_id, candidate_id, root, folder, attempt.id,
                                                                     manifest, report, result["logs"])}
    attempt.finish()
    return result


@contextmanager
def _probed(on):
    """With `on`, the engine host started in the block probes its Core steps (crit_core.native.STEP_PROBE): the host
    process inherits the environment it starts with."""
    from crit_core.native import STEP_PROBE
    before = os.environ.get(STEP_PROBE)
    if on:
        os.environ[STEP_PROBE] = "1"
    try:
        yield
    finally:
        if on:
            if before is None:
                os.environ.pop(STEP_PROBE, None)
            else:
                os.environ[STEP_PROBE] = before


def _plugin_tests(folder, log):
    """The packaged plugins' own tests against their references (crit plugins test, one plugin per process, cached by
    files and environment), as serving-check items plugin_test_ID."""
    from crit_plugins.testing import run_tests
    plugins = folder / "package" / "plugins"
    if not plugins.is_dir():
        return [{"name": "plugin_tests", "state": "failed", "detail": "the package holds no plugins/ folder"}]
    log("crit run serve --check: running the package's plugin tests (crit plugins test)")
    try:
        items = run_tests(plugins, log=lambda line: log("crit plugins test: " + str(line).strip()), cache=True)
    except Exception as error:   # a registry that does not load is the check's failure, not the attempt's
        return [{"name": "plugin_tests", "state": "failed", "detail": f"{type(error).__name__}: {error}"[:2000]}]
    return [{"name": f"plugin_test_{item['name']}", "state": item["state"], "detail": item.get("detail", "")}
            for item in items]


# crit smoke's second pass opens the engine at crit-eval's lower context, a quarter of the run's (lower_fraction,
# rounded down to its 1024-token granularity), where buckets, graphs and caches sized for a smaller context run; never
# below what the suite's longest prompt (4096 tokens) needs.
SMOKE_LOWER_FRACTION, SMOKE_GRANULARITY, SMOKE_LOWER_FLOOR = 0.25, 1024, 5120


def _smoke(served, port, log):
    """crit smoke (smoke/README.md) on the served engine at the run's context, then reopened at the lower context in
    the same resident host and back, its items in the serving checks' shape: smoke_TEST, and smoke_TEST@TOKENS at the
    lower context. A package that offers none of what the suite reads skips it at once, without reopening."""
    try:
        from crit_smoke import Client, run
    except ImportError:   # a checkout installed before smoke/ existed
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "smoke"))
        from crit_smoke import Client, run
    lower = max(int(served.kv_tokens * SMOKE_LOWER_FRACTION) // SMOKE_GRANULARITY * SMOKE_GRANULARITY,
                SMOKE_LOWER_FLOOR)
    items, contexts = [], [served.kv_tokens] + ([lower] if lower < served.kv_tokens else [])
    for tokens in contexts:
        suffix = "" if tokens == served.kv_tokens else f"@{tokens}"
        try:
            if suffix:
                log(f"crit smoke: reopening at {tokens:,} tokens of context")
                _, _, port = served.open(tokens)
            found = run(Client(f"http://127.0.0.1:{port}", timeout=min(served.timeout, 120)),
                        log=lambda line: log("crit smoke" + suffix + ": " + line.strip()))
        except Exception as error:   # an engine that cannot open or answer fails the suite, the item says why
            found = [{"name": "open", "state": "failed",
                      "detail": f"at {tokens:,} tokens: {type(error).__name__}: {error}"[:2000]}]
        items += [dict(item, name=f"smoke_{item['name']}{suffix}") for item in found]
        if not suffix and all(item["state"] == "skipped" for item in found):
            break
    if any(item["name"].endswith(f"@{lower}") for item in items):
        log(f"crit smoke: reopening at {served.kv_tokens:,} tokens of context")
        try:
            served.open()   # back at the run's context for what follows
        except Exception as error:
            items.append({"name": "smoke_reopen", "state": "failed",
                          "detail": f"opening again at {served.kv_tokens:,} tokens: {type(error).__name__}: {error}"[:2000]})
            raise
    return items


def _precheck(served, run, data, port, log):
    """crit-eval's gate pre-check on the served engine (Reopening), its items in the serving checks' shape."""
    from pathlib import Path
    from crit_eval import load_json, machine_settings
    from crit_eval.quality.prepare import PREPARED
    from crit_eval.run.precheck import run_precheck
    data = Path(data).absolute()
    if not (data / PREPARED).is_file():
        raise ValueError(f"{data} holds no {PREPARED}: prepare it from the run's model first (crit eval prepare)")
    headroom = run["spec"]["hardware_constraints"].get("vram_headroom_bytes")
    if headroom is None:
        headroom = round(machine_settings().get("vram_headroom_gib", 1.0) * (1 << 30))
    try:
        return run_precheck(Reopening(served, load_json(data / PREPARED), port), run["spec"]["model_path"], data,
                            served.kv_tokens, quality_mode(run["spec"]["correctness"]), log=log,
                            vram_headroom_bytes=headroom)
    except Exception as error:   # a pre-check that cannot run is a failed check, the report says why
        return [{"name": "precheck", "state": "failed", "detail": f"{type(error).__name__}: {error}"[:2000]}]


def _record_serve_checks(store, run_id, candidate_id, root, folder, attempt, manifest, report, logs):
    """Keep the serving checks' report and record a functional check per case citing it and the serve log."""
    from .checks import verdict
    from .records import digest, write_json
    path = folder / "logs" / f"serve-check-{attempt}.json"
    write_json(path, {"attempt_id": attempt, "checks": report}, immutable=True)
    status, rationale = verdict(report)
    run = load_run(root)
    evidence = [{"path": path.relative_to(root).as_posix(), "sha256": file_sha(path)}] + logs
    return [store.append(run_id, candidate_id, "check", {
        "case_id": case["case_id"], "package_sha256": digest(manifest), "gate": "functional",
        "criteria_sha256": digest(run["spec"]["correctness"]), "status": status, "rationale": rationale[:8000],
        "evidence": evidence, "evidence_level": "artifact_verified", "recorded_at": now(),
        "extensions": {"recorded_by": "crit run serve --check"}}) for case in run["spec"]["cases"]]
