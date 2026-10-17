"""One sealed candidate, loaded once and kept resident; requests run one at a time."""
from pathlib import Path
import shutil
import threading
import uuid

from crit.process.ownership import RuntimeOwnership
from crit.run.records import (digest, keys, now, number, read_json, record, safe_path, text,
                              verification_scope, write_json)
from crit.run.store import ROOT, RunStore

from .residency import CAPABILITY as RESIDENT_CAPABILITY, ResidentRuntime
from .stream import CAPABILITY as STREAM_CAPABILITY

CAPABILITY = "chat-json-v1"
# The adapter applies a request's response_format (crit_core.constraints / crit/constraints.h); see docs/SERVING.md.
CONSTRAINTS_CAPABILITY = "chat-constraints-v1"
MAX_REPLY = 65536
MAX_CONVERSATION = 1_000_000
LOAD_BYTES_PER_SECOND = 50e6  # model bytes per second assumed by the default load bound (copy, hash, engine load)


class EngineError(Exception):
    """The engine failed a request. `status` is the HTTP status the API reports."""
    def __init__(self, message, status=500):
        super().__init__(message)
        self.status = status


def telemetry(value, request):
    """Rates use the engine's timed phase token counts, never string lengths."""
    keys(value, {"prompt_tokens", "output_tokens", "prefill_tokens", "prefill_ms",
                 "decode_tokens", "decode_ms", "ttft_ms", "missing"})
    prompt = number(value["prompt_tokens"], "prompt tokens", 1, request["limits"]["kv_tokens"], integer=True)
    output = number(value["output_tokens"], "output tokens", 1, request["limits"]["max_output_tokens"], integer=True)
    if prompt + output > request["limits"]["kv_tokens"]:
        raise ValueError("Reply exceeds the requested KV capacity")
    keys(value["missing"], set(), {"prefill", "decode", "ttft"})
    rates = {}
    for phase, maximum in (("prefill", prompt), ("decode", output)):
        tokens, ms = value[phase + "_tokens"], value[phase + "_ms"]
        if tokens is None and ms is None:
            text(value["missing"].get(phase), phase + " missing reason", 1000)
            rates[phase + "_tps"] = None
        else:
            number(tokens, phase + " tokens", 0, maximum, integer=True)
            number(ms, phase + " milliseconds", 0, request["limits"]["timeout_seconds"] * 1000)
            if tokens and not ms:
                raise ValueError("A measured token phase requires positive elapsed time")
            if phase in value["missing"]:
                raise ValueError("A measured phase cannot also be missing")
            rates[phase + "_tps"] = tokens * 1000 / ms if tokens else None
            if rates[phase + "_tps"] is not None:
                number(rates[phase + "_tps"], phase + " rate", 0, 1_000_000_000)
    if value["ttft_ms"] is None:
        text(value["missing"].get("ttft"), "TTFT missing reason", 1000)
    else:
        number(value["ttft_ms"], "first token milliseconds", 0, request["limits"]["timeout_seconds"] * 1000)
        if "ttft" in value["missing"]:
            raise ValueError("A measured TTFT cannot also be missing")
    return value | rates


def reply(value, request):
    keys(value, {"schema_version", "record_kind", "request_id", "candidate_id", "package_sha256", "assistant", "telemetry"})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["record_kind"] != "chat_response":
        raise ValueError("Invalid engine chat response")
    if any(value[key] != request[key] for key in ("request_id", "candidate_id", "package_sha256")):
        raise ValueError("Reply belongs to a different request or engine")
    return text(value["assistant"], "Reply", MAX_REPLY), telemetry(value["telemetry"], request)


def package_signature(folder):
    result = []
    for path in sorted(Path(folder).rglob("*")):
        stat = path.stat()
        result.append((str(path), stat.st_size, stat.st_mtime_ns))
    return result


def single_run(store):
    runs = sorted(path.parent.name for path in store.root.glob("run-*/run.json"))
    if len(runs) != 1:
        raise ValueError("Choose a run with --run; the workspace holds " + (", ".join(runs) or "no runs"))
    return runs[0]


class ServedEngine:
    """Verifies sealed candidates of one run, keeps the requested one loaded in the resident host and runs requests
    one at a time. Switching candidates reuses model bytes and compatible device allocations.

    The server owns the machine's crit runtime while it runs: no other crit process can load an engine.
    """
    def __init__(self, workspace, run_id=None, candidate_id=None, project=ROOT, runtime_directory=None,
                 host_limit=None, device_limit=None, timeout=600, load_timeout=None):
        self.store = RunStore(workspace, project)
        self.run_id = run_id or single_run(self.store)
        self.requested_candidate = candidate_id
        self.default = None
        self.timeout = number(timeout, "request timeout", .01, 86400)
        self.load_timeout = None if load_timeout is None else number(load_timeout, "load timeout", .01, 7 * 86400)
        self.ownership = RuntimeOwnership(runtime_directory)
        self.runtime = ResidentRuntime(self.ownership, host_limit, device_limit)
        self.lock = threading.Lock()
        self.specs = {}
        self.listed = None
        self.closed = False

    def _verify(self, candidate_id):
        """The candidate's retained bytes, seal, chat interface, model file and settled attempts."""
        view = self.store.get(self.run_id)
        if view["errors"]:
            raise ValueError("Retained project verification failed: " + str(view["errors"]))
        candidate = next((c for c in view["candidates"] if c["candidate_id"] == candidate_id), None)
        if candidate is None:
            raise EngineError("Unknown candidate: " + str(candidate_id), 404)
        if not candidate["sealed"] or not candidate["artifacts"].get("package"):
            raise ValueError("Serving requires a sealed candidate with a runnable package")
        root = Path(view["folder"])
        package = safe_path(root, candidate["artifacts"]["package"])
        manifest = read_json(package / "manifest.json")
        if CAPABILITY not in manifest["capabilities"]:
            raise ValueError("This package has no chat-json-v1 interface. Its agent must add a chat adapter.")
        if RESIDENT_CAPABILITY not in manifest["capabilities"]:
            raise ValueError("Serving needs a resident adapter: publish the package with chat-resident-v1.")
        if not Path(view["run"]["spec"]["model_path"]).exists():
            raise ValueError("Model file is unavailable at its recorded path: " + view["run"]["spec"]["model_path"])
        # External agents cannot be stopped from here. Their recorded attempts must have settled cleanup.
        for reservation in self.store._attempts(root):
            owner = ("candidates/" + reservation["candidate_id"] if reservation["candidate_id"]
                     else "segments/" + reservation["segment_id"])
            result = safe_path(root, owner + "/attempts/" + reservation["attempt_id"] + "/result.json")
            if not result.is_file() or not read_json(result)["cleanup_settled"]:
                raise ValueError("Settle the external agent/build attempt and its cleanup before serving")
        return dict(run_id=self.run_id, candidate_id=candidate_id, package_sha256=digest(manifest),
                    package_path=str(package), model_path=view["run"]["spec"]["model_path"],
                    model_manifest=read_json(safe_path(root, "inputs.json"))["model_manifest"],
                    configuration=candidate["configuration"], launch=manifest["launch"],
                    capabilities=manifest["capabilities"], limits=view["run"]["spec"]["hardware_constraints"],
                    label=candidate["label"], signature=package_signature(package))

    def candidates(self, refresh=False):
        """Sealed candidates of the run whose packages declare a resident chat adapter (listed once, then on refresh)."""
        if self.listed is not None and not refresh:
            return self.listed
        with verification_scope():
            view = self.store.get(self.run_id)
        result = []
        for candidate in view["candidates"]:
            if candidate["sealed"] and candidate["artifacts"].get("package"):
                manifest = read_json(safe_path(Path(view["folder"]), candidate["artifacts"]["package"] + "/manifest.json"))
                if {CAPABILITY, RESIDENT_CAPABILITY} <= set(manifest["capabilities"]):
                    result.append({"candidate_id": candidate["candidate_id"], "label": candidate["label"]})
        self.listed = result
        return result

    def spec(self, candidate_id=None):
        candidate_id = candidate_id or self.default
        if candidate_id not in self.specs:
            with verification_scope():
                self.specs[candidate_id] = self._verify(candidate_id)
        return self.specs[candidate_id]

    def model_id(self, candidate_id=None):
        return self.run_id + "/" + (candidate_id or self.default)

    @property
    def kv_tokens(self):
        return self.spec()["limits"]["kv_tokens"]

    @property
    def package_sha256(self):
        return self.spec()["package_sha256"]

    def open(self, stop=None):
        """Verify the default candidate, take runtime ownership and load the engine."""
        with self.lock:
            if self.closed:
                raise ValueError("Engine is closed")
            view = self.store.get(self.run_id)
            self.default = self.requested_candidate or view["state"]["selected_candidate_id"]
            if self.default is None:
                raise ValueError("The run has no selected candidate; choose one with --candidate")
            spec = self.spec()
            self.candidates(refresh=True)
            self.ownership.acquire()
            try:
                result = self.runtime.call("load", {"spec": spec}, stop or threading.Event(), self.load_seconds(spec))
                if result["status"] != "completed":
                    raise ValueError(result["error"] or "Engine loading failed")
            except BaseException:
                self._release()
                raise

    def load_seconds(self, spec):
        """Bound on loading a candidate: --load-timeout, else 600 s plus the model's bytes at 50 MB/s. The host
        copies and hashes every model file into its private snapshot before the engine reads them, so loading
        grows with the checkpoint; a fixed bound fails large models on ordinary disks."""
        if self.load_timeout is not None:
            return self.load_timeout
        return 600 + sum(row["bytes"] for row in spec["model_manifest"]) / LOAD_BYTES_PER_SECOND

    def _release(self):
        try:
            if self.ownership.failed_tree:
                self.ownership.failed_tree.close()
                self.ownership.failed_tree = None
            self.runtime.close()
        except BaseException as exc:
            self.ownership.block("Engine cleanup failed: " + str(exc))
            raise
        self.ownership.release(settled=True)

    def status(self):
        if self.ownership.blocked:
            return {"state": "blocked", "error": self.ownership.blocked}
        return self.runtime.status()

    def generate(self, messages, max_tokens, stop, on_text=None, candidate_id=None, response_format=None):
        """Run one request. `on_text(text)` receives the cumulative accepted text when the adapter streams.
        `response_format` (from crit_api.server.response_format) goes into the request record for adapters that
        declare chat-constraints-v1.

        Returns (status, assistant, telemetry): status is "completed", "cancelled" or "timed_out"; a cancelled
        or timed-out request has no reply or telemetry. Raises EngineError when the engine fails.
        """
        if sum(len(m["content"]) for m in messages) > MAX_CONVERSATION:
            raise EngineError("Conversation exceeds 1,000,000 characters", 400)
        with self.lock:
            if self.closed:
                raise EngineError("Engine is closed", 503)
            if self.ownership.blocked:
                raise EngineError("Runtime cleanup is unresolved: " + self.ownership.blocked, 503)
            spec = self.spec(candidate_id)
            limits = spec["limits"]
            if not 1 <= max_tokens < limits["kv_tokens"]:
                raise EngineError("max_tokens must be between 1 and %d" % (limits["kv_tokens"] - 1), 400)
            if response_format is not None and CONSTRAINTS_CAPABILITY not in spec["capabilities"]:
                raise EngineError("This engine package does not declare " + CONSTRAINTS_CAPABILITY, 400)
            extra = {} if response_format is None else {"response_format": response_format}
            self.runtime.start()
            folder = self.runtime.root / "requests" / uuid.uuid4().hex
            folder.mkdir(parents=True)
            request = record("chat_request", self.run_id, chat_id="chat-" + uuid.uuid4().hex, request_id=uuid.uuid4().hex,
                candidate_id=spec["candidate_id"], package_sha256=spec["package_sha256"], created_at=now(),
                model_path=spec["model_path"], package_path=spec["package_path"],
                model_manifest=spec["model_manifest"], configuration=spec["configuration"], messages=messages,
                limits={"max_output_tokens": max_tokens, "kv_tokens": limits["kv_tokens"], "timeout_seconds": self.timeout,
                        **{k: v for k, v in limits.items() if k.endswith("headroom_bytes")}}, **extra)
            write_json(folder / "request.json", request, immutable=True)
            streams = STREAM_CAPABILITY in spec["capabilities"]
            try:
                result = self.runtime.call("generate", {"spec": spec, "request_path": str(folder / "request.json"),
                    "response_path": str(folder / "response.json")}, stop, self.timeout,
                    progress=on_text if streams and on_text else None, request=request)
            except (ValueError, OSError) as exc:
                self._retire(exc)
                raise EngineError("Engine host failed: " + str(exc)) from exc
            if result["status"] in {"cancelled", "timed_out"}:
                # Settled cooperative cancellation keeps the weights resident.
                shutil.rmtree(folder, ignore_errors=True)
                return result["status"], None, None
            if result["status"] != "completed":
                self._retire(result["error"])
                raise EngineError(result["error"] or "Engine failed the request")
            try:
                assistant, measured = reply(read_json(folder / "response.json"), request)
                self._check_sealed(spec)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self._retire(exc)
                raise EngineError("Engine returned an invalid response: " + str(exc)) from exc
            # A long-running server would otherwise keep every conversation on disk; failed requests stay for diagnosis.
            shutil.rmtree(folder, ignore_errors=True)
            return "completed", assistant, measured

    def _check_sealed(self, spec):
        """The package must still be the sealed one after the engine ran. Unchanged file metadata skips rehashing;
        any change verifies the candidate's retained bytes again."""
        if package_signature(spec["package_path"]) == spec["signature"]:
            return
        self.specs.pop(spec["candidate_id"], None)
        if self.spec(spec["candidate_id"])["package_sha256"] != spec["package_sha256"]:
            raise ValueError("Engine artifacts changed during the request")

    def _retire(self, error):
        """A failed request retires the host process; the next request loads the engine again."""
        try:
            self.runtime.close()
        except BaseException as exc:
            self.ownership.block("Engine cleanup failed after an error (%s): %s" % (error, exc))

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if not self.ownership.token:
                return
            if self.runtime.tree:
                try:
                    self.runtime.call("unload", {}, threading.Event(), 5)
                except (ValueError, OSError):
                    pass  # Retiring the whole process below also proves release.
            self._release()
