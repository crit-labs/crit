"""A packaged candidate served through crit serve under one of its attempts, for crit run eval, crit run measure and
crit run serve: crit serve's resident host loads the candidate's resident chat adapter, packaged and not yet sealed,
behind its chat API. One engine is served at a time, and the reservation's deadline stops it, cancelling the request
in flight.
"""
from contextlib import contextmanager
import threading
import time

from crit_core.serving import (JSON_CAPABILITY, LOGPROBS_CAPABILITY, RESIDENT_CAPABILITY, SCORE_CAPABILITY,
                               STREAM_CAPABILITY)

from .attempt import Attempt, Expired, packaged, remaining, unsealed
from .records import file_sha, load_run, now, read_json

# What crit serve needs from a package to serve it for a measurement: a resident chat adapter that streams. Every
# resident package declares the temperature (crit run package), which greedy decoding sets.
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

    def open(self, tokens=None):
        """Serve the candidate at `tokens` of context (default the run's). Returns the ServedEngine, crit serve's Api
        and the loopback port it listens on. An engine already served is built again at the new context in the same
        resident host, which keeps the model's files, device weights and the adapter's host-side weights."""
        from crit_api.engine import ServedEngine
        from crit_api.server import Api, Server
        tokens = tokens or self.kv_tokens
        with self.lock:
            current = self.current
        if current is not None and current[1] is not None and not self.expired:
            engine, server = current
            self.log(f"crit serve: reloading {self.candidate_id} at {tokens:,} tokens of context in the same host")
            try:
                engine.reopen(None if tokens == self.kv_tokens else tokens)
            except BaseException:
                self.stop()
                raise
            return engine, server.api, server.server_address[1]
        self.stop()  # an engine that failed to open
        engine = ServedEngine(self.store.root, self.run_id, self.candidate_id, project=self.store.project,
                              timeout=self.timeout, attempt=self.attempt,
                              kv_tokens=None if tokens == self.kv_tokens else tokens)
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


def serve(store, run_id, candidate_id, host="127.0.0.1", port=8000, seconds=None, ready=None, stop=None, check=False):
    """crit run serve: the packaged candidate, not yet sealed, served through crit serve under a check attempt until
    Ctrl+C, `stop` (a threading.Event) or the reservation's deadline, for the checks a candidate passes before it is
    sealed (streaming, stop strings, tool calls, cancellation, Ctrl+C). `ready(port)` runs once the API listens.
    With `check`, crit runs the standard serving checks itself (crit.run.checks.serve_checks), stops the engine,
    keeps their report as candidates/ID/logs/serve-check-ATTEMPT.json and records a functional check per case.
    Settles the attempt and returns its result; crit serve serves the candidate once it is sealed."""
    root = store.directory(run_id)
    folder = unsealed(root, candidate_id, "The candidate is sealed: serve it with crit serve")
    manifest = packaged(folder, SERVED, "serving", "crit run serve serves")
    kv_tokens = load_run(root)["spec"]["hardware_constraints"]["kv_tokens"]
    # The engine is served until the reservation's deadline: reaching it is how a serve without --check ends.
    attempt = Attempt(store, run_id, candidate_id, "check", seconds, "serve", "crit run serve", "crit run serve",
                      deadline=False)
    report = None
    with attempt.running() as log:
        served = Served(store, run_id, candidate_id, attempt.id, kv_tokens, log, "crit run serve", address=(host, port))
        with served.until(attempt.deadline_at):
            _, _, bound = served.open()
            log(f"Serving {run_id}/{candidate_id} (not sealed, {attempt.id}) at http://{host}:{bound}/v1; Ctrl+C stops")
            if ready is not None:
                ready(bound)
            if check:
                from .checks import serve_checks
                variables = (read_json(folder / "candidate.json")["configuration"].get("recon") or {}).get(
                    "template_variables") or {}
                report = serve_checks(bound, manifest["capabilities"], variables, served.timeout, log)
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
                       "detail": "the engine stopped and its resident host exited, as on Ctrl+C" if completed
                       else f"stopping the engine: {attempt.failure}"})
    result = attempt.settle()
    if report is not None:
        result = {"attempt": result, "checks": _record_serve_checks(store, run_id, candidate_id, root, folder, attempt.id,
                                                                     manifest, report, result["logs"])}
    attempt.finish()
    return result


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
