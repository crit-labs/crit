"""Verified sealed candidates, bounded request admission and resident lifetimes."""
from pathlib import Path
import json
import re
import threading
import uuid

from crit.process.ownership import RuntimeOwnership
from crit.run.records import (digest, keys, now, number, read_json, record, safe_path, text,
                              verification_scope, write_json)
from crit.run.store import ROOT, RunStore

from .residency import CAPABILITY as RESIDENT_CAPABILITY, ResidentRuntime
from .admission import Admission
from .content import response_content, tool_size
from .logprobs import logprob_entries, logprob_options, response_bytes
from .stream import CAPABILITY as STREAM_CAPABILITY
from .options import (MAX_CHAT_RECORD, RequestError, TOOL_STREAM_CAPABILITY, sampling_options, serving_capabilities,
                      tool_options as validate_tool_options, validate_tool_reply)

CAPABILITY = "chat-json-v1"
# The adapter applies a request's response_format (crit_core.constraints / crit/constraints.h); see docs/SERVING.md.
CONSTRAINTS_CAPABILITY = "chat-constraints-v1"
MAX_CONVERSATION = 8 * 1024 * 1024
LOAD_BYTES_PER_SECOND = 200e6  # model bytes per second assumed by the default load bound (the engine's own load)
VERIFY_BYTES_PER_SECOND = 50e6  # with --verify-model: read, hash and copy every model file, then the engine load


class EngineError(Exception):
    """The engine failed a request. `status` is the HTTP status the API reports."""
    def __init__(self, message, status=500):
        super().__init__(message)
        self.status = status


class _CommandStop:
    """Observe client cancellation and engine shutdown without changing either event."""
    def __init__(self, client, shutdown):
        self.client, self.shutdown = client, shutdown

    def is_set(self):
        return self.client.is_set() or self.shutdown.is_set()


def telemetry(value, request):
    """Rates use the engine's timed phase token counts, never string lengths. Optional decode_rounds, the target
    passes after prefill (a speculative engine's verify rounds), gives tokens_per_round: output tokens per round."""
    keys(value, {"prompt_tokens", "output_tokens", "prefill_tokens", "prefill_ms",
                 "decode_tokens", "decode_ms", "ttft_ms", "missing"}, {"decode_rounds"})
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
    if "decode_rounds" in value:
        if value["decode_tokens"] is None:
            raise ValueError("Decode rounds need a measured decode phase")
        rates["tokens_per_round"] = output / number(value["decode_rounds"], "decode rounds", 1, output, integer=True)
    return value | rates


def reply(value, request):
    keys(value, {"schema_version", "record_kind", "request_id", "candidate_id", "package_sha256", "assistant", "telemetry"},
         {"tool_calls", "reasoning", "logprobs"})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["record_kind"] != "chat_response":
        raise ValueError("Invalid engine chat response")
    if any(value[key] != request[key] for key in ("request_id", "candidate_id", "package_sha256")):
        raise ValueError("Reply belongs to a different request or engine")
    measured = telemetry(value["telemetry"], request)
    calls = validate_tool_reply(value.get("tool_calls"), request,
                                measured["output_tokens"] == request["limits"]["max_output_tokens"])
    assistant, reasoning = response_content(value, tool_size(calls))
    if calls:
        assistant = {"content": assistant, "tool_calls": calls}
    if reasoning:
        # Reasoning the model wrote apart from its reply (crit_core.text.reasoning); clients get reasoning_content.
        assistant = dict(assistant if isinstance(assistant, dict) else {"content": assistant}, reasoning=reasoning)
    if ("logprobs" in value) != (request.get("logprobs") is not None):
        raise ValueError("Reply logprobs do not match the request")
    if "logprobs" in value:
        entries = logprob_entries(value["logprobs"], request)
        if len(entries) > measured["output_tokens"]:
            raise ValueError("Reply logprobs exceed its output tokens")
        assistant = dict(assistant if isinstance(assistant, dict) else {"content": assistant}, logprobs=entries)
    return assistant, measured


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
    within its declared capacity. Switching candidates reuses compatible model allocations.

    With `attempt` (crit run eval), it serves only `candidate_id`, packaged but not yet sealed, for Eval under that
    benchmark attempt of the candidate, whose reservation settles after the server stops; `kv_tokens` then serves it
    at a lower context than the run's (Eval's lower-context gate).

    The server owns the machine's crit runtime while it runs: no other crit process can load an engine.
    """
    def __init__(self, workspace, run_id=None, candidate_id=None, project=ROOT, runtime_directory=None,
                 host_limit=None, device_limit=None, timeout=600, load_timeout=None, verify_model=False, attempt=None,
                 kv_tokens=None):
        self.store = RunStore(workspace, project)
        self.run_id = run_id or single_run(self.store)
        self.requested_candidate = candidate_id
        if attempt is not None and candidate_id is None:
            raise ValueError("Measuring under an attempt needs its candidate")
        if kv_tokens is not None and attempt is None:
            raise ValueError("Only a measurement serves a candidate at another context")
        self.attempt, self.context_tokens = attempt, kv_tokens
        self.default = None
        self.timeout = number(timeout, "request timeout", .01, 86400)
        self.load_timeout = None if load_timeout is None else number(load_timeout, "load timeout", .01, 7 * 86400)
        self.verify_model = bool(verify_model)
        self.ownership = RuntimeOwnership(runtime_directory)
        self.runtime = ResidentRuntime(self.ownership, host_limit, device_limit)
        self.lock = Admission()
        self.runtime.on_retire = self.lock.invalidate
        self._stopping = threading.Event()
        self.specs = {}
        self.listed = None
        self.closed = False

    def _verify(self, candidate_id, *, store=None, run_id=None):
        """The candidate's retained bytes, seal, chat interface, model file and settled attempts."""
        store, run_id = store or self.store, run_id or self.run_id
        view = store.get(run_id)
        if view["errors"]:
            raise ValueError("Retained project verification failed: " + str(view["errors"]))
        candidate = next((c for c in view["candidates"] if c["candidate_id"] == candidate_id), None)
        if candidate is None or (self.attempt is not None and candidate_id != self.requested_candidate):
            raise EngineError("Unknown candidate: " + str(candidate_id), 404)
        if self.attempt is not None and candidate["sealed"]:
            raise ValueError("A sealed candidate is not measured again; allocate a new candidate")
        if not candidate["artifacts"].get("package") or not (candidate["sealed"] or self.attempt is not None):
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
        for reservation in store._attempts(root):
            if (reservation["candidate_id"], reservation["attempt_id"]) == (candidate_id, self.attempt):
                continue  # the attempt this engine is measured under settles after it stops
            owner = ("candidates/" + reservation["candidate_id"] if reservation["candidate_id"]
                     else "segments/" + reservation["segment_id"])
            result = safe_path(root, owner + "/attempts/" + reservation["attempt_id"] + "/result.json")
            if not result.is_file() or not read_json(result)["cleanup_settled"]:
                raise ValueError("Settle the external agent/build attempt and its cleanup before serving")
        limits = view["run"]["spec"]["hardware_constraints"]
        if self.context_tokens is not None:
            if type(self.context_tokens) is not int or not 1 < self.context_tokens <= limits["kv_tokens"]:
                raise ValueError("A measurement serves at most the run's context of %d tokens" % limits["kv_tokens"])
            limits = dict(limits, kv_tokens=self.context_tokens)
        return dict(run_id=run_id, candidate_id=candidate_id, selector=candidate_id, package_sha256=digest(manifest),
                    package_path=str(package), model_path=view["run"]["spec"]["model_path"],
                    model_manifest=read_json(safe_path(root, "inputs.json"))["model_manifest"],
                    configuration=candidate["configuration"], launch=manifest["launch"],
                    capabilities=serving_capabilities(manifest["capabilities"]), limits=limits,
                    label=candidate["label"], signature=package_signature(package), verify_model=self.verify_model)

    def serves(self, candidate):
        """Sealed candidates; under an attempt only its packaged candidate."""
        if self.attempt is not None:
            return candidate["candidate_id"] == self.requested_candidate and not candidate["sealed"]
        return candidate["sealed"]

    def candidates(self, refresh=False):
        """Served candidates of the run whose packages declare a resident chat adapter (listed once, then on
        refresh)."""
        if self.listed is not None and not refresh:
            return self.listed
        with verification_scope():
            view = self.store.get(self.run_id)
        result = []
        for candidate in view["candidates"]:
            if self.serves(candidate) and candidate["artifacts"].get("package"):
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

    def resolve_model(self, model):
        """Legacy single-run names permit arbitrary client aliases for the default."""
        if model is None:
            return self.default
        candidate = model.removeprefix(self.run_id + "/")
        if candidate == model and not re.fullmatch(r"candidate-\d+", model):
            return self.default
        known = lambda refresh=False: candidate in {c["candidate_id"] for c in self.candidates(refresh)}
        if not known() and not known(refresh=True):
            raise EngineError("Unknown model " + model + "; GET /v1/models lists the served candidates", 404)
        return candidate

    @property
    def kv_tokens(self):
        return self.spec()["limits"]["kv_tokens"]

    @property
    def package_sha256(self):
        return self.spec()["package_sha256"]

    def open(self, stop=None):
        """Verify the default candidate, take runtime ownership and load the engine."""
        with self.lock:
            if self.closed or self._stopping.is_set():
                raise ValueError("Engine is closed")
            view = self.store.get(self.run_id)
            self.default = self.requested_candidate or view["state"]["selected_candidate_id"]
            if self.default is None:
                raise ValueError("The run has no selected candidate; choose one with --candidate")
            spec = self.spec()
            self.candidates(refresh=True)
            self.ownership.acquire()
            try:
                result = self.runtime.call("load", {"spec": spec},
                                           _CommandStop(stop or threading.Event(), self._stopping), self.load_seconds(spec))
                if result["status"] != "completed":
                    raise ValueError(result["error"] or "Engine loading failed")
                self.lock.configure(self.default, result.get("max_concurrent_requests", 1))
            except BaseException:
                self._release()
                raise

    def load_seconds(self, spec):
        """Bound on loading a candidate: --load-timeout, else 600 s plus the model's bytes at 200 MB/s, which the
        engine's own load reads. With --verify-model the host first reads and hashes every model file (copying it
        for engines without chat-model-file-v1), so the bound assumes 50 MB/s. Loading grows with the checkpoint;
        a fixed bound fails large models on ordinary disks."""
        if self.load_timeout is not None:
            return self.load_timeout
        rate = VERIFY_BYTES_PER_SECOND if spec.get("verify_model") else LOAD_BYTES_PER_SECOND
        return 600 + sum(row["bytes"] for row in spec["model_manifest"]) / rate

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

    def host_pid(self):
        """The process the engine runs in, crit serve's resident host, once it has connected."""
        return self.runtime.host_pid()

    def status(self):
        if self.ownership.blocked:
            return {"state": "blocked", "error": self.ownership.blocked}
        return self.runtime.status()

    def generate(self, messages, max_tokens, stop, on_text=None, candidate_id=None, response_format=None,
                 sampling=None, tool_options=None, on_tools=None, on_reasoning=None, on_started=None, logprobs=None):
        """Run one request. `on_text(text)` receives each piece of accepted text as the adapter streams it.
        `response_format` (from crit_api.server.response_format) goes into the request record for adapters that
        declare chat-constraints-v1. Sampling and tool options are capability-gated request overrides; logprobs n
        asks a chat-logprobs-v1 adapter for every token's log-probability with n alternatives.
        `on_started(deadline)` receives the command's absolute monotonic deadline
        after dispatch, so response delivery can honor the same execution budget.

        Returns (status, assistant, telemetry): status is "completed", "cancelled" or "timed_out"; a cancelled
        or timed-out request has no reply or telemetry. Assistant is text, or {content, tool_calls, reasoning,
        logprobs (OpenAI's logprobs content)} with the parts the reply has. Raises EngineError when the engine fails.
        """
        image_input = any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
                          for m in messages)
        # The HTTP body's bound, in bytes: a character count would refuse a prompt that fits the model's context.
        if len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) > MAX_CONVERSATION:
            raise EngineError("Conversation exceeds 8 MiB", 400)
        # A disconnected waiter must leave promptly without taking ownership of
        # the active turn or eventually loading/switching an engine for dead work.
        candidate_id = candidate_id or self.default
        needs_load = self.lock.admit(candidate_id, _CommandStop(stop, self._stopping))
        if needs_load is None:
            if self._stopping.is_set():
                raise EngineError("Engine is stopping", 503)
            return "cancelled", None, None
        try:
            if stop.is_set():
                return "cancelled", None, None
            if self._stopping.is_set():
                raise EngineError("Engine is stopping", 503)
            if self.ownership.blocked:
                raise EngineError("Runtime cleanup is unresolved: " + self.ownership.blocked, 503)
            spec = self.spec(candidate_id)
            if image_input and "chat-images-v1" not in spec["capabilities"]:
                raise EngineError("This engine package does not declare chat-images-v1", 400)
            limits = spec["limits"]
            if not 1 <= max_tokens < limits["kv_tokens"]:
                raise EngineError("max_tokens must be between 1 and %d" % (limits["kv_tokens"] - 1), 400)
            if response_format is not None and CONSTRAINTS_CAPABILITY not in spec["capabilities"]:
                raise EngineError("This engine package does not declare " + CONSTRAINTS_CAPABILITY, 400)
            extra = {} if response_format is None else {"response_format": response_format}
            try:
                applied = sampling_options(sampling or {}, spec["capabilities"])
                extra.update(validate_tool_options(tool_options or {}, spec["capabilities"]))
                if logprobs is not None:
                    extra["logprobs"] = logprob_options({"logprobs": True, "top_logprobs": logprobs},
                                                        spec["capabilities"])
            except RequestError as exc:
                raise EngineError(str(exc), 400) from exc
            if applied:
                extra["sampling"] = applied
            if stop.is_set():
                return "cancelled", None, None
            if self._stopping.is_set():
                raise EngineError("Engine is stopping", 503)
            if needs_load:
                try:
                    loaded = self.runtime.call("load", {"spec": spec}, _CommandStop(stop, self._stopping),
                                               self.load_seconds(spec))
                    if loaded["status"] in {"cancelled", "timed_out"}:
                        self.lock.invalidate()
                        if self._stopping.is_set():
                            raise EngineError("Engine is stopping", 503)
                        return loaded["status"], None, None
                    if loaded["status"] != "completed":
                        raise EngineError(loaded.get("error") or "Engine loading failed")
                    self.lock.configure(candidate_id, loaded.get("max_concurrent_requests", 1))
                except BaseException as exc:
                    if not self._stopping.is_set():
                        self._retire(exc)
                    raise
            request = record("chat_request", spec["run_id"], chat_id="chat-" + uuid.uuid4().hex, request_id=uuid.uuid4().hex,
                candidate_id=spec["candidate_id"], package_sha256=spec["package_sha256"], created_at=now(),
                model_path=spec["model_path"], package_path=spec["package_path"],
                model_manifest=spec["model_manifest"], configuration=spec["configuration"], messages=messages,
                limits={"max_output_tokens": max_tokens, "kv_tokens": limits["kv_tokens"], "timeout_seconds": self.timeout,
                        **{k: v for k, v in limits.items() if k.endswith("headroom_bytes")}}, **extra)
            streams = STREAM_CAPABILITY in spec["capabilities"]
            try:
                result = self.runtime.call("generate", {"spec": spec, "request": request},
                    _CommandStop(stop, self._stopping), self.timeout,
                    progress=on_text if streams and on_text else None, request=request,
                    progress_tools=on_tools if TOOL_STREAM_CAPABILITY in spec["capabilities"] else None,
                    progress_reasoning=on_reasoning if streams else None, on_started=on_started)
            except BaseException as exc:
                # Once the supervisor unwinds, no observer remains to forward
                # cancellation. Retire even for consumer failures/interrupts,
                # while the request lock still excludes another command.
                self._keep(request)
                self._retire(exc)
                if isinstance(exc, (ValueError, OSError)):
                    if self._stopping.is_set():
                        raise EngineError("Engine is stopping", 503) from exc
                    raise EngineError("Engine host failed: " + str(exc)) from exc
                raise
            if result["status"] in {"cancelled", "timed_out"}:
                # Settled cooperative cancellation keeps the weights resident.
                if self._stopping.is_set():
                    raise EngineError("Engine is stopping", 503)
                return result["status"], None, None
            if result["status"] != "completed":
                self._keep(request)
                self._retire(result["error"])
                raise EngineError(result["error"] or "Engine failed the request")
            try:
                assistant, measured = reply(result["response"], request)
                self._check_sealed(spec)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self._keep(request, result["response"])
                self._retire(exc)
                raise EngineError("Engine returned an invalid response: " + str(exc)) from exc
            return "completed", assistant, measured
        finally:
            self.lock.release()

    def _check_sealed(self, spec):
        """The package must still be the sealed one after the engine ran. Unchanged file metadata skips rehashing;
        any change verifies the candidate's retained bytes again."""
        if package_signature(spec["package_path"]) == spec["signature"]:
            return
        selector = spec["selector"]
        self.specs.pop(selector, None)
        if self.spec(selector)["package_sha256"] != spec["package_sha256"]:
            raise ValueError("Engine artifacts changed during the request")

    def _keep(self, request, response=None):
        """A failed request's records stay in the runtime directory for diagnosis."""
        try:
            folder = self.runtime.root / "requests" / request["request_id"]
            write_json(folder / "request.json", request, immutable=True, max_bytes=MAX_CHAT_RECORD)
            if response is not None:
                write_json(folder / "response.json", response, immutable=True, max_bytes=response_bytes(request))
        except (OSError, ValueError, TypeError):
            pass  # diagnostics never mask the failure

    def _retire(self, error):
        """A failed request retires the host process; the next request loads the engine again."""
        try:
            self.runtime.close()
        except BaseException as exc:
            self.ownership.block("Engine cleanup failed after an error (%s): %s" % (error, exc))

    def close(self):
        # Publish before waiting for the request lock: its owner must be able to
        # cancel and settle, and queued requests must not dispatch behind it.
        self._stopping.set()
        with self.lock:
            if self.closed:
                return
            if self.ownership.token:
                try:
                    if self.runtime.tree and not self.ownership.blocked:
                        try:
                            self.runtime.call("unload", {}, threading.Event(), 5)
                        except (ValueError, OSError):
                            pass  # Retiring the whole process below also proves release.
                finally:
                    self._release()
            # A failed release leaves shutdown requested and ownership blocked,
            # but allows a later close to retry physical retirement.
            self.closed = True
