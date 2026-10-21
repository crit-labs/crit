"""A packaged candidate served through crit serve under one of its benchmark attempts, for crit run eval and crit run
measure: crit serve's resident host loads the candidate's resident chat adapter, packaged and not yet sealed, behind
its chat API on a loopback port. One engine is served at a time, and the reservation's deadline stops it, cancelling
the request in flight.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import threading

from .records import timestamp

# What crit serve needs from a package to serve it for a measurement: a resident chat adapter that streams. Every
# resident package declares the temperature (crit run package), which greedy decoding sets.
SERVED = ("chat-json-v1", "chat-resident-v1", "chat-stream-v1")


class Expired(Exception):
    """The reservation's deadline passed while the candidate was served."""


def remaining(deadline_at):
    """Seconds left before the reservation's deadline."""
    return (timestamp(deadline_at, "deadline") - datetime.now(timezone.utc)).total_seconds()


class Served:
    """The candidate served through crit serve. `name` names the measurement in its deadline's message, and `log`
    receives a line when an engine loads and stops; `timeout` bounds each request."""

    def __init__(self, store, run_id, candidate_id, attempt, kv_tokens, log, name, timeout=600):
        self.store, self.run_id, self.candidate_id, self.attempt = store, run_id, candidate_id, attempt
        self.kv_tokens, self.log, self.name, self.timeout = kv_tokens, log, name, timeout
        self.server = f"crit serve {run_id}/{candidate_id}"
        self.lock = threading.Lock()
        self.current = None  # (ServedEngine, HTTP server) while one is served
        self.expired = False
        self.failure = None  # a cleanup that failed, also on the deadline's thread

    def open(self, tokens=None):
        """Serve the candidate at `tokens` of context (default the run's). Returns the ServedEngine, crit serve's Api
        and the loopback port it listens on."""
        from crit_api.engine import ServedEngine
        from crit_api.server import Api, Server
        self.stop()  # an engine that failed to open
        tokens = tokens or self.kv_tokens
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
        server = Server(("127.0.0.1", 0), api)
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
