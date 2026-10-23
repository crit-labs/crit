"""Resident host with bounded request isolation and exclusive model lifecycle work."""
from pathlib import Path
import importlib.util
import os
import sys
import types
import threading

if __package__ in {None, ""}:
    # Launched by path: make crit_api and, in a checkout, crit and crit_core importable.
    root = Path(__file__).resolve().parents[2]
    for folder in (Path(__file__).resolve().parents[1], root / "cli", root / "core" / "python"):
        if any((folder / name).is_dir() for name in ("crit_api", "crit", "crit_core")):
            sys.path.insert(0, str(folder))
from crit.run.records import MAX_JSON_BYTES, digest, read_json, safe_path
from crit_api import channel
from crit_api.logprobs import RESPONDING
from crit_api.models import ModelContext, ModelPool
from crit_api.stream import CAPABILITY as STREAM_CAPABILITY, StreamWriter
from crit_api.options import MAX_CHAT_RECORD, TOOLS_CAPABILITY, STRICT_TOOLS_CAPABILITY, TOOL_STREAM_CAPABILITY
from crit_core.serving import LOGPROBS_CAPABILITY, SCORE_CAPABILITY

CONFIG_CAPABILITY = "chat-config-v1"


def effective_configuration_problem(engine, configuration):
    """chat-config-v1: the engine reports the configuration it actually applied.
    Every frozen configuration key must be present with an equal value."""
    report = getattr(engine, "effective_configuration", None)
    if not callable(report):
        return CONFIG_CAPABILITY + " requires effective_configuration()"
    effective = report()
    if not isinstance(effective, dict):
        return "effective_configuration() must return a dictionary"
    ignored = sorted(k for k in configuration if k not in effective)
    changed = sorted(k for k in configuration if k in effective and effective[k] != configuration[k])
    if ignored or changed:
        return ("Engine did not apply its frozen configuration (ignored: %s; changed: %s)"
                % (", ".join(ignored) or "none", ", ".join(changed) or "none"))
    return None

CAPABILITY = "chat-resident-v1"


class Cancelled(Exception):
    pass


class Cancellation:
    """Set when crit serve cancels the command."""
    def __init__(self, event=None):
        self.event = event or threading.Event()

    def is_set(self):
        return self.event.is_set()

    def check(self):
        if self.is_set():
            raise Cancelled("Stopped")


class Host:
    def __init__(self, root, host_limit, device_limit, send):
        self.root, self.send = root, send
        self.pool = ModelPool(root / "models", host_limit, device_limit)
        self.pools = {}
        self.host_limit, self.device_limit = self.pool.host_limit, self.pool.device_limit
        self.engine = self.context = self.identity = None
        self.blocked = False
        self.loads = 0
        self.capacity = 1
        self.concurrent = False
        self.active = 0
        self.requests = set()
        self.state_lock = threading.RLock()

    def status(self, state, **extra):
        with self.state_lock:
            if self.active and state in {"ready", "draining"}:
                state = "generating"
            self.send({"kind": "status", "status": {"schema_version": 1, "state": state,
                "engine": self.identity, "model_id": self.pool.model_id, "model_generation": self.pool.generation,
                "host_bytes": sum(p.host_bytes for p in self.pools.values()),
                "device_bytes": sum(p.device_bytes for p in self.pools.values()),
                "engine_loads": self.loads, "active_requests": self.active,
                "max_concurrent_requests": self.capacity, **extra}})

    def detach(self):
        if self.engine is not None:
            self.status("draining")
            try:
                self.engine.close()
                self.context.close()
            except BaseException:
                self.blocked = True
                raise
            self.engine = self.context = self.identity = None

    def attach(self, spec):
        if self.blocked:
            raise ValueError("Resident host is quarantined")
        identity = {k: spec[k] for k in ("run_id", "candidate_id", "package_sha256")}
        if identity == self.identity:
            self.pool.verify_sources()
            return
        self.detach()
        model_id = digest(spec['model_manifest'])
        if self.pool.model_id is not None and self.pool.model_id != model_id:
            # One model at a time: settle leases and free old weights before any
            # new model allocation. A failed fence/free quarantines the host.
            try:
                for pool in self.pools.values():
                    pool.close()
                self.pools.clear()
            except BaseException:
                self.blocked = True
                raise
        if model_id not in self.pools:
            self.pools[model_id] = ModelPool(self.root / 'models' / model_id, self.host_limit, self.device_limit)
        self.pool = self.pools[model_id]
        self.pool.host_limit = self.host_limit - sum(p.host_bytes for k, p in self.pools.items() if k != model_id)
        self.pool.device_limit = self.device_limit - sum(p.device_bytes for k, p in self.pools.items() if k != model_id)
        self.status("loading_model")
        loaded = self.pool.load(spec["model_path"], spec["model_manifest"],
                       native_files='chat-model-file-v1' in spec.get('capabilities', []),
                       verify=bool(spec.get('verify_model')),
                       progress=lambda value: self.status("loading_model", progress=value))
        def loading(state):
            self.status(state, model_reused=not loaded, model_bytes=self.pool.host_bytes,
                        model_device_bytes=self.pool.device_bytes)
        loading("loading_engine")
        package = Path(spec["package_path"])
        entry = safe_path(package, spec["launch"][0])
        if entry.suffix.lower() != ".py":
            raise ValueError("chat-resident-v1 requires a Python factory entry; native code may be wrapped by it")
        namespace = "crit_resident_" + spec["package_sha256"]
        if namespace not in sys.modules:
            module = types.ModuleType(namespace)
            module.__path__ = [str(package)]
            sys.modules[namespace] = module
        name = namespace + ".entry"
        # A sealed package keeps its bytes: importing it must not write a __pycache__ into it. crit serve launches
        # the host with -B; a host run in-process (a test, an embedding) may not have been.
        sys.dont_write_bytecode = True
        module_spec = importlib.util.spec_from_file_location(name, entry)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        self.context = ModelContext(self.pool, spec["configuration"], spec["launch"][1:], progress=loading)
        self.context.limits = spec.get('limits', {})
        try:
            capabilities = spec.get("capabilities", [])
            if (STRICT_TOOLS_CAPABILITY in capabilities or TOOL_STREAM_CAPABILITY in capabilities) and TOOLS_CAPABILITY not in capabilities:
                raise ValueError("Strict or streaming tools require chat-tools-v1")
            if TOOL_STREAM_CAPABILITY in capabilities and STREAM_CAPABILITY not in capabilities:
                raise ValueError("chat-tool-stream-v1 requires chat-stream-v1")
            self.engine = module.create_engine(self.context)
            if not all(callable(getattr(self.engine, name, None)) for name in ("generate", "reset", "close")):
                raise ValueError("Resident engine must implement generate, reset and close")
            if STREAM_CAPABILITY in spec.get("capabilities", []) and not callable(getattr(self.engine, "generate_stream", None)):
                raise ValueError("chat-stream-v1 requires generate_stream(request, cancellation, emit)")
            self.concurrent = "chat-concurrent-v1" in capabilities
            self.capacity = getattr(self.engine, "max_concurrent_requests", 1) if self.concurrent else 1
            if type(self.capacity) is not int or not 1 <= self.capacity <= 1024:
                raise ValueError("max_concurrent_requests must be an integer in [1, 1024]")
            if self.concurrent and not callable(getattr(self.engine, "create_request", None)):
                raise ValueError("chat-concurrent-v1 requires create_request(request)")
            if LOGPROBS_CAPABILITY in capabilities and getattr(self.engine, "logprobs", False) is not True:
                raise ValueError(LOGPROBS_CAPABILITY + " requires an engine that reports log-probabilities (logprobs "
                                 "True)")
            if SCORE_CAPABILITY in capabilities and (getattr(self.engine, "scores", False) is not True
                                                     or not callable(getattr(self.engine, "score", None))):
                raise ValueError(SCORE_CAPABILITY + " requires an engine that scores fixed text (scores True, "
                                 "score(request, cancellation))")
        except BaseException:
            self.blocked = True  # Partial construction may own native work.
            raise
        finally:
            self.context.progress = None  # Loading updates must not replace generation state.
        if CONFIG_CAPABILITY in spec.get("capabilities", []):
            problem = effective_configuration_problem(self.engine, spec["configuration"])
            if problem:
                self.detach()  # close the constructed engine and its context before rejecting
                raise ValueError(problem)
        self.identity = identity
        self.loads += 1
        self.status("ready")

    def score(self, spec, request, cancellation):
        """A score request (chat-score-v1): alone on the engine, as Core runs scores, then reset like a turn."""
        if SCORE_CAPABILITY not in spec.get("capabilities", []):
            raise ValueError("This engine package does not declare " + SCORE_CAPABILITY)
        with self.state_lock:
            self.active += 1
        self.status("generating")
        try:
            cancellation.check()
            response = self.engine.score(request, cancellation)
            cancellation.check()
            self.pool.verify_sources()
        finally:
            self.status("draining")
            try:
                self.engine.reset()
                self.pool.verify_sources()
            except BaseException:
                self.blocked = True
                raise
            finally:
                with self.state_lock:
                    self.active -= 1
        self.status("ready")
        return response

    def handle(self, command, cancellation):
        """Run one command; a generation returns its response."""
        if self.blocked:
            raise ValueError("Resident host is quarantined")
        action, payload = command["action"], command["payload"]
        if action == "unload":
            self.detach()
            for pool in self.pools.values():
                pool.close()
            self.pools.clear()
            self.status("unloaded")
            return None
        self.attach(payload["spec"])
        if action == "load":
            return None
        if action not in ("generate", "score"):
            raise ValueError("Unknown resident command")
        request = payload["request"]
        if any(request[key] != self.identity[key] for key in self.identity):
            raise ValueError("Resident request engine identity mismatch")
        if action == "score":
            return self.score(payload["spec"], request, cancellation)
        stream = (StreamWriter(self.send, command["command_id"], request, cancellation,
                               tools=TOOL_STREAM_CAPABILITY in payload["spec"].get("capabilities", []))
                  if STREAM_CAPABILITY in payload["spec"].get("capabilities", []) else None)
        target = self.engine
        owned = False
        with self.state_lock:
            self.active += 1
        self.status("generating")
        try:
            cancellation.check()
            if self.concurrent:
                target = self.engine.create_request(request)
                with self.state_lock:
                    if target is self.engine or id(target) in self.requests:
                        raise ValueError("create_request must return a distinct request object")
                    if not all(callable(getattr(target, name, None)) for name in
                               ("generate", "reset", "close", *(("generate_stream",) if stream else ()))):
                        raise ValueError("Request object must implement generation, reset and close")
                    self.requests.add(id(target))
                    owned = True
            try:
                if stream:
                    response = target.generate_stream(request, cancellation, stream.emit)
                    stream.finish(response)
                else:
                    response = target.generate(request, cancellation)
            except Exception:
                cancellation.check()
                raise
            cancellation.check()
            self.pool.verify_sources()
        finally:
            if stream:
                stream.close()
            self.status("draining")
            try:
                if not self.concurrent:
                    self.engine.reset()
                elif target is not self.engine:
                    # Creation/validation failures must never reset a peer's object.
                    if owned:
                        # A failed reset has not proved settlement. Quarantine
                        # and retire instead of freeing possibly borrowed state.
                        target.reset()
                        target.close()
                        with self.state_lock:
                            self.requests.remove(id(target))
                self.pool.verify_sources()
            except BaseException:
                self.blocked = True
                raise
            finally:
                with self.state_lock:
                    self.active -= 1
        self.status("ready")
        return response


def main():
    if os.name != "nt":
        import resource
        # No core file: a crash of a host holding tens of GB would write one that large (a soft limit of 1 byte also
        # stops pipe-based core handlers, which ignore 0).
        hard = resource.getrlimit(resource.RLIMIT_CORE)[1]
        if hard == resource.RLIM_INFINITY or hard >= 1:
            resource.setrlimit(resource.RLIMIT_CORE, (1, hard))
    config = read_json(Path(sys.argv[1]))
    link = channel.connect(config["port"], config["token"])
    host = Host(Path(config["root"]), config["host_limit"], config["device_limit"], link.send)
    host.status("unloaded")
    lock, wake = threading.Lock(), threading.Event()
    pending, running, cancels, closed = [], set(), {}, []

    def receive():
        """Queue commands and set cancellations as crit serve sends them; any failure ends the host."""
        try:
            while True:
                message = link.receive(MAX_CHAT_RECORD + MAX_JSON_BYTES)
                with lock:
                    if message["kind"] == "command":
                        pending.append(message)
                    elif message["kind"] == "cancel":
                        cancels.setdefault(message["command_id"], Cancellation()).event.set()
                    else:
                        raise ValueError("Unknown resident message")
                wake.set()
        except BaseException as exc:
            closed.append(exc)
            wake.set()

    def execute(command):
        cid = command["command_id"]
        with lock:
            cancellation = cancels.setdefault(cid, Cancellation())
        status, error, response = "completed", None, None
        try:
            response = host.handle(command, cancellation)
        except Cancelled as exc:
            status, error = "cancelled", str(exc)
        except BaseException as exc:
            status, error = "failed", str(exc)
            host.blocked = True
        host.blocked = host.blocked or host.pool.blocked
        host.status("blocked" if host.blocked else ("ready" if host.engine else "unloaded"), error=error)
        link.send({"kind": "result", "command_id": cid, "status": status, "error": error,
                   "max_concurrent_requests": host.capacity, "cleanup_settled": not host.blocked,
                   **({"response": response} if command["action"] in RESPONDING and status == "completed" else {})})
        with lock:
            cancels.pop(cid, None)
            running.discard(cid)
        wake.set()

    threading.Thread(target=receive, name="crit-host-receive", daemon=True).start()
    while not closed:
        wake.clear()
        with lock:
            queued = list(pending)
        for command in queued:
            parallel = command["action"] == "generate" and host.concurrent
            identity = {k: command["payload"].get("spec", {}).get(k)
                        for k in ("run_id", "candidate_id", "package_sha256")}
            with lock:
                if running and (not parallel or identity != host.identity or len(running) >= host.capacity):
                    continue
                pending.remove(command)
                if parallel:
                    running.add(command["command_id"])
            if parallel:
                threading.Thread(target=execute, args=(command,), daemon=True).start()
            else:
                execute(command)
        wake.wait()
    # crit serve closed the channel: it is retiring this host.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
