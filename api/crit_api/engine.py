"""Verified sealed candidates, bounded request admission and resident lifetimes."""
from pathlib import Path
import json
import re
import threading
import time
import uuid

from crit.process.ownership import RuntimeOwnership
from crit.run.records import (digest, keys, now, number, read_json, record, safe_path, text,
                              verification_scope, write_json)
from crit.run.store import ROOT, RunStore

from crit_core.serving import (COMPLETION_CAPABILITY, CONTINUE_CAPABILITY, FIT_OUTPUT_CAPABILITY, MIN_LOGPROB,
                               PLAIN_CAPABILITY, SCORE_CAPABILITY, TOKENIZE_CAPABILITY)

from .residency import CAPABILITY as RESIDENT_CAPABILITY, ResidentRuntime
from .admission import Admission, Refused
from .content import response_content, tool_size
from .logprobs import logprob_entries, logprob_options, prompt_entries, response_bytes
from .stream import CAPABILITY as STREAM_CAPABILITY
from .options import (MAX_CHAT_RECORD, RequestError, TOOL_STREAM_CAPABILITY, sampling_options, serving_capabilities,
                      template_options, tool_options as validate_tool_options, validate_tool_reply)

CAPABILITY = "chat-json-v1"
# The adapter applies a request's response_format (crit_core.constraints / crit/constraints.h); see docs/SERVING.md.
CONSTRAINTS_CAPABILITY = "chat-constraints-v1"
# With it, also a response_format of type grammar: any compiled grammar (structured_outputs regex, choice, grammar).
GRAMMAR_CAPABILITY = "chat-constraints-grammar-v1"
MAX_CONVERSATION = 8 * 1024 * 1024
LOAD_BYTES_PER_SECOND = 200e6  # model bytes per second assumed by the default load bound (the engine's own load)
VERIFY_BYTES_PER_SECOND = 50e6  # with --verify-model: read, hash and copy every model file, then the engine load
# What a tokenize request (chat-tokenize-v1) returns for each operation.
TOKENIZED = {"tokenize": ("tokens",), "render": ("prompt", "tokens"), "detokenize": ("prompt",)}


class EngineError(Exception):
    """The engine failed a request. `status` is the HTTP status the API reports."""
    def __init__(self, message, status=500):
        super().__init__(message)
        self.status = status


class RequestRejected(EngineError):
    """The engine refused the request before generating (crit_core.serving.RequestRejected): a 400 that leaves the
    engine loaded. `reason` names the adapter's rule and `details` its facts."""
    def __init__(self, message, reason, details=None):
        super().__init__(message, 400)
        self.reason, self.details = reason, details or {}


class Overloaded(EngineError):
    """The queue refused the request (admission.Refused) before the engine saw it: a full queue or an expired wait.
    crit serve answers 429 (Anthropic 529, Ollama 503) with Retry-After."""
    def __init__(self, message):
        super().__init__(message, 429)


class ContextLengthError(RequestRejected):
    """The prompt plus the requested output exceeds the engine's context (crit_core.serving.ContextExceeded)."""
    def __init__(self, message, prompt_tokens, output_tokens, capacity):
        super().__init__(message, "context_exceeded", {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                                                       "capacity": capacity})
        self.prompt_tokens, self.output_tokens, self.capacity = prompt_tokens, output_tokens, capacity


def rejected(result):
    """The RequestRejected for a host result of status "rejected"."""
    found, message = result["rejection"], result["error"] or "The engine rejected the request"
    counts = [found.get(key) for key in ("prompt_tokens", "output_tokens", "capacity")]
    if found["reason"] == "context_exceeded" and all(type(n) is int and n >= 0 for n in counts):
        return ContextLengthError(message, *counts)
    return RequestRejected(message, found["reason"], {k: v for k, v in found.items() if k != "reason"})


class _CommandStop:
    """Observe client cancellation and engine shutdown without changing either event."""
    def __init__(self, client, shutdown):
        self.client, self.shutdown = client, shutdown

    def is_set(self):
        return self.client.is_set() or self.shutdown.is_set()


def telemetry(value, request):
    """Rates use the engine's timed phase token counts, never string lengths. Optional decode_rounds, the target
    passes after prefill (a speculative engine's verify rounds), gives tokens_per_round: output tokens per round.
    Optional accepted (drafts accepted in each of those rounds), verify_ms and draft_ms (device time in verify passes
    and draft steps) pass through for Eval's headroom."""
    keys(value, {"prompt_tokens", "output_tokens", "prefill_tokens", "prefill_ms",
                 "decode_tokens", "decode_ms", "ttft_ms", "missing"}, {"decode_rounds", "accepted", "verify_ms", "draft_ms"})
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
    if "accepted" in value:
        accepted = value["accepted"]
        if (value.get("decode_rounds") is None or not isinstance(accepted, list) or len(accepted) != value["decode_rounds"]
                or not all(type(a) is int and 0 <= a < output for a in accepted)):
            raise ValueError("Accepted drafts are one count per decode round")
    for key in ("verify_ms", "draft_ms"):
        if key in value:
            number(value[key], key.replace("_ms", " milliseconds"), 0, request["limits"]["timeout_seconds"] * 1000)
    return value | rates


def reply(value, request):
    keys(value, {"schema_version", "record_kind", "request_id", "candidate_id", "package_sha256", "assistant", "telemetry"},
         {"tool_calls", "reasoning", "logprobs", "max_output_tokens", "prompt"})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["record_kind"] != "chat_response":
        raise ValueError("Invalid engine chat response")
    if any(value[key] != request[key] for key in ("request_id", "candidate_id", "package_sha256")):
        raise ValueError("Reply belongs to a different request or engine")
    measured = telemetry(value["telemetry"], request)
    limit = request["limits"]["max_output_tokens"]
    if ("max_output_tokens" in value) != bool(request["limits"].get("fit_output")):
        raise ValueError("Reply max_output_tokens does not match the request")
    if "max_output_tokens" in value:
        # chat-fit-output-v1: the cap the engine applied, the room the prompt left in the context.
        limit = number(value["max_output_tokens"], "applied output tokens", measured["output_tokens"], limit,
                       integer=True)
        if measured["prompt_tokens"] + limit > request["limits"]["kv_tokens"]:
            raise ValueError("The applied output cap exceeds the requested KV capacity")
        measured = dict(measured, max_output_tokens=limit)
    calls = validate_tool_reply(value.get("tool_calls"), request, measured["output_tokens"] == limit)
    raw = "prompt" in request  # completion-v1: the text continues the prompt, which may echo back
    if raw and "reasoning" in value:
        raise ValueError("A raw prompt's reply has no reasoning")
    if ("prompt" in value) != bool(request.get("echo")):
        raise ValueError("Reply prompt does not match the request")
    assistant, reasoning = response_content(value, tool_size(calls),
                                            continued=bool(request.get("continue_final_message")), blank=raw)
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
        if request.get("echo"):
            prompt = prompt_entries(value["logprobs"], request)
            if len(prompt) != measured["prompt_tokens"]:
                raise ValueError("Reply prompt logprobs need one row per prompt token")
            entries = prompt + entries
        assistant = dict(assistant if isinstance(assistant, dict) else {"content": assistant}, logprobs=entries)
    if request.get("echo"):
        echo = value["prompt"]
        if (not isinstance(echo, str) or len(echo.encode("utf-8")) > MAX_CONVERSATION
                or ("text" in request["prompt"] and echo != request["prompt"]["text"])):
            raise ValueError("Reply prompt is not the request's prompt text")
        assistant = dict(assistant if isinstance(assistant, dict) else {"content": assistant}, prompt=echo)
    return assistant, measured


def score_reply(value, request):
    """A score response record (chat-score-v1) checked against its request: one row per prompt position after the
    first and per continuation token, each naming its own token, with exactly the requested ranked, distinct
    alternatives and log-probabilities in [MIN_LOGPROB, 0]."""
    keys(value, {"schema_version", "record_kind", "request_id", "candidate_id", "package_sha256", "prompt",
                 "continuation", "telemetry"})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["record_kind"] != "score_response":
        raise ValueError("Invalid engine score response")
    if any(value[key] != request[key] for key in ("request_id", "candidate_id", "package_sha256")):
        raise ValueError("Score belongs to a different request or engine")

    def checked(logprob):
        if type(logprob) not in (int, float) or not MIN_LOGPROB <= logprob <= 0:
            raise ValueError(f"Score log-probabilities must be numbers in [{MIN_LOGPROB}, 0]")
        return float(logprob)

    def rows(found, tokens, part):
        if part == "prompt" and not request["prompt_rows"]:
            tokens = []
        if not isinstance(found, list) or len(found) != len(tokens):
            raise ValueError(f"Score {part} rows must number {len(tokens)}")
        out = []
        for row, token in zip(found, tokens):
            if not isinstance(row, list) or len(row) != 3 or row[0] != token or not isinstance(row[2], list):
                raise ValueError(f"Score {part} rows are [token, logprob, alternatives] for the scored tokens")
            top = [[pair[0], checked(pair[1])] for pair in row[2]
                   if isinstance(pair, list) and len(pair) == 2 and type(pair[0]) is int and pair[0] >= 0]
            if (len(top) != len(row[2]) or len(top) != request["top_logprobs"] or len({p[0] for p in top}) != len(top)
                    or any(a[1] < b[1] for a, b in zip(top, top[1:]))):
                raise ValueError(f"Score {part} rows need {request['top_logprobs']} distinct, ranked alternatives")
            out.append([token, checked(row[1]), top])
        return out
    telemetry = value["telemetry"]
    if not isinstance(telemetry, dict) or set(telemetry) != {"prefill_ms", "decode_ms"} or not all(
            v is None or (type(v) in (int, float) and v >= 0) for v in telemetry.values()):
        raise ValueError("Score telemetry holds prefill_ms and decode_ms")
    return {"prompt": rows(value["prompt"], request["prompt_ids"][1:], "prompt"),
            "continuation": rows(value["continuation"], request["continuation_ids"], "continuation"),
            "telemetry": telemetry}


def tokenize_reply(value, request):
    """A tokenize response record (chat-tokenize-v1) checked against its request: the text's ids for tokenize, the
    prompt's text and ids for render, the ids' text for detokenize."""
    parts = TOKENIZED[request["operation"]]
    keys(value, {"schema_version", "record_kind", "request_id", "candidate_id", "package_sha256", *parts})
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["record_kind"] != "tokenize_response"):
        raise ValueError("Invalid engine tokenize response")
    if any(value[key] != request[key] for key in ("request_id", "candidate_id", "package_sha256")):
        raise ValueError("Tokens belong to a different request or engine")
    if "tokens" in parts and (not isinstance(value["tokens"], list)
                              or not all(type(t) is int and 0 <= t < 2**31 for t in value["tokens"])):
        raise ValueError("Tokens must be a list of token ids")
    if "prompt" in parts and not isinstance(value["prompt"], str):
        raise ValueError("A prompt must be text")
    return {part: value[part] for part in parts}


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

    Requests the engine cannot take yet wait in its queue: `max_queue` of them at most, each for `queue_timeout`
    seconds at most (None: no bound); past either a request raises Overloaded.

    The server owns the machine's crit runtime while it runs, also while its engine is unloaded: no other crit process
    can load an engine. With `idle_unload` (crit serve --idle-unload) the engine unloads after that many seconds
    without a request, and the next request loads it again.
    """
    def __init__(self, workspace, run_id=None, candidate_id=None, project=ROOT, runtime_directory=None,
                 host_limit=None, device_limit=None, timeout=600, load_timeout=None, verify_model=False, attempt=None,
                 kv_tokens=None, max_queue=None, queue_timeout=None, idle_unload=None):
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
        self.idle_unload = None if idle_unload is None else number(idle_unload, "idle unload", .01, 7 * 86400)
        self.ownership = RuntimeOwnership(runtime_directory)
        self.runtime = ResidentRuntime(self.ownership, host_limit, device_limit)
        self.lock = Admission(
            None if max_queue is None else number(max_queue, "queue size", 0, 1_000_000, integer=True),
            None if queue_timeout is None else number(queue_timeout, "queue timeout", 0, 86400))
        self.runtime.on_retire = self.lock.invalidate
        self._stopping = threading.Event()
        self.specs = {}
        self.listed = None
        self.closed = False
        self.loads = self.switches = 0  # completed engine loads, and those that replaced another candidate's engine
        self.last_load = None

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
                self._loaded(self.default)
                self.lock.configure(self.default, result.get("max_concurrent_requests", 1))
            except BaseException:
                self._release()
                raise
            if self.idle_unload is not None:
                threading.Thread(target=self._unload_idle, name="crit-idle-unload", daemon=True).start()

    def reopen(self, kv_tokens=None, stop=None):
        """Serve the measured candidate at `kv_tokens` of context (None: the run's) on the loaded runtime: the
        resident host closes the engine and builds it again at the new limits in the same process, keeping the
        model's mapped files and device weights (and whatever host-side weights the adapter keeps in the process)."""
        if self.attempt is None:
            raise ValueError("Only a measurement serves a candidate at another context")
        with self.lock:
            if self.closed or self._stopping.is_set():
                raise ValueError("Engine is closed")
            self.context_tokens = kv_tokens
            self.specs.clear()
            spec = self.spec()
            result = self.runtime.call("load", {"spec": spec}, _CommandStop(stop or threading.Event(), self._stopping),
                                       self.load_seconds(spec))
            if result["status"] != "completed":
                raise ValueError(result["error"] or "Engine loading failed")
            self._loaded(self.default)
            self.lock.configure(self.default, result.get("max_concurrent_requests", 1))

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

    def ready(self, status=None):
        """Whether a request for the loaded engine runs without loading one: the host has an engine ready or
        generating and no switch, reload or retire is under way. After a retire it stays false until a request loads
        the engine again."""
        status = status if status is not None else self.status()
        return (status.get("state") in {"ready", "generating"} and bool(status.get("engine"))
                and not self.lock.loading() and not self._stopping.is_set())

    def facts(self, candidate_id=None):
        """The loaded engine's facts for Eval (crit_core.serving.engine_facts) when `candidate_id` (default: the
        default candidate) is the one loaded, else None."""
        status = self.runtime.status()
        return status.get("engine_facts") if self.loaded(candidate_id, status) else None

    def loaded(self, candidate_id=None, status=None):
        """Whether the host's engine is `candidate_id`'s (default: the default candidate's) package: the same run,
        candidate and package digest, which a catalog name resolves to."""
        engine = (status if status is not None else self.runtime.status()).get("engine") or {}
        spec = self.spec(candidate_id)
        return all(engine.get(k) == spec[k] for k in ("run_id", "candidate_id", "package_sha256"))

    def generate(self, messages, max_tokens, stop, on_text=None, candidate_id=None, response_format=None,
                 sampling=None, tool_options=None, on_tools=None, on_reasoning=None, on_started=None, logprobs=None,
                 decode=None, template_kwargs=None, fit_output=False, on_admitted=None, priority=0, request_id=None,
                 timeout=None, continue_final_message=False, prompt=None, echo=False):
        """Run one request. `on_text(text)` receives each piece of accepted text as the adapter streams it.
        `response_format` (from crit_api.server.response_format or structured_outputs) goes into the request record
        for adapters that declare chat-constraints-v1, and also chat-constraints-grammar-v1 for type grammar.
        Sampling and tool options are capability-gated request overrides; logprobs n asks a chat-logprobs-v1 adapter
        for every token's log-probability with n alternatives; decode "plain" asks a chat-plain-v1 adapter to decode
        one token per target pass; template_kwargs set the chat-template variables a chat-template-kwargs-v1 adapter
        declares; fit_output asks a chat-fit-output-v1 adapter to cut max_tokens to the room the prompt leaves in the
        context, and the telemetry then carries the cap it applied as max_output_tokens; continue_final_message asks
        a chat-continue-v1 adapter to continue the final assistant message, its reply only what follows it.
        `prompt`, {"text"} or {"ids"} in place of messages (None), asks a completion-v1 adapter to continue a raw
        prompt without the chat template; with `echo` the reply also returns the prompt, and with logprobs its rows
        (from a chat-score-v1 adapter's score path).
        `priority` orders the request among those waiting for the engine, lower first (Admission.admit). `timeout`
        (seconds) bounds this request in place of the server's timeout, which caps it.
        `on_started(deadline)` receives the command's absolute monotonic deadline
        after dispatch, so response delivery can honor the same execution budget.
        `on_admitted()` runs once the request leaves the queue, before the engine loads or sees it. `request_id` (crit
        serve's id of the HTTP request; default a new one) is the request record's id, which names its records when it
        fails.

        Returns (status, assistant, telemetry): status is "completed", "cancelled" or "timed_out"; a cancelled
        or timed-out request has no reply or telemetry. Assistant is text, or {content, tool_calls, reasoning,
        logprobs (OpenAI's logprobs content, an echoed prompt's first), prompt (the echoed prompt)} with the parts the
        reply has. Raises EngineError when the engine fails.
        """
        image_input = any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
                          for m in messages or ())
        # The HTTP body's bound, in bytes: a character count would refuse a prompt that fits the model's context.
        sent = messages if prompt is None else prompt
        if len(json.dumps(sent, ensure_ascii=False).encode("utf-8")) > MAX_CONVERSATION:
            raise EngineError(("Conversation" if prompt is None else "Prompt") + " exceeds 8 MiB", 400)
        # A disconnected waiter must leave promptly without taking ownership of
        # the active turn or eventually loading/switching an engine for dead work.
        candidate_id = candidate_id or self.default
        needs_load = self._admit(candidate_id, stop, priority)
        if needs_load is None:
            if self._stopping.is_set():
                raise EngineError("Engine is stopping", 503)
            return "cancelled", None, None
        try:
            if on_admitted:
                on_admitted()
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
            if (response_format or {}).get("type") == "grammar" and GRAMMAR_CAPABILITY not in spec["capabilities"]:
                raise EngineError("This engine package does not declare " + GRAMMAR_CAPABILITY, 400)
            if decode is not None and (decode != "plain" or PLAIN_CAPABILITY not in spec["capabilities"]):
                raise EngineError("This engine package does not declare " + PLAIN_CAPABILITY, 400)
            if fit_output and FIT_OUTPUT_CAPABILITY not in spec["capabilities"]:
                raise EngineError("This engine package does not declare " + FIT_OUTPUT_CAPABILITY, 400)
            if continue_final_message and CONTINUE_CAPABILITY not in spec["capabilities"]:
                raise EngineError("This engine package does not declare " + CONTINUE_CAPABILITY, 400)
            if timeout is not None and (type(timeout) not in (int, float) or not 0 < timeout < float("inf")):
                raise EngineError("timeout must be a positive number of seconds", 400)
            if prompt is not None and COMPLETION_CAPABILITY not in spec["capabilities"]:
                raise EngineError("This engine package does not declare " + COMPLETION_CAPABILITY, 400)
            if echo and logprobs is not None and SCORE_CAPABILITY not in spec["capabilities"]:
                raise EngineError("This engine package does not declare " + SCORE_CAPABILITY, 400)
            timeout = self.timeout if timeout is None else min(timeout, self.timeout)
            extra = {} if response_format is None else {"response_format": response_format}
            if decode is not None:
                extra["decode"] = decode
            if continue_final_message:
                extra["continue_final_message"] = True
            try:
                applied = sampling_options(sampling or {}, spec["capabilities"])
                extra.update(validate_tool_options(tool_options or {}, spec["capabilities"]))
                if logprobs is not None:
                    extra["logprobs"] = logprob_options({"logprobs": True, "top_logprobs": logprobs},
                                                        spec["capabilities"])
                if template_kwargs:
                    extra["template_kwargs"] = template_options({"chat_template_kwargs": template_kwargs},
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
                loaded = self._load(spec, candidate_id, stop)
                if loaded is not None:
                    return loaded, None, None
            text = {"messages": messages} if prompt is None else {"prompt": prompt, **({"echo": True} if echo else {})}
            request = record("chat_request", spec["run_id"], chat_id="chat-" + uuid.uuid4().hex,
                request_id=request_id or uuid.uuid4().hex,
                candidate_id=spec["candidate_id"], package_sha256=spec["package_sha256"], created_at=now(),
                model_path=spec["model_path"], package_path=spec["package_path"],
                model_manifest=spec["model_manifest"], configuration=spec["configuration"], **text,
                limits={"max_output_tokens": max_tokens, "kv_tokens": limits["kv_tokens"], "timeout_seconds": timeout,
                        **{k: v for k, v in limits.items() if k.endswith("headroom_bytes")},
                        **({"fit_output": True} if fit_output else {})}, **extra)
            streams = STREAM_CAPABILITY in spec["capabilities"]
            try:
                result = self.runtime.call("generate", {"spec": spec, "request": request},
                    _CommandStop(stop, self._stopping), timeout,
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
            if result["status"] == "rejected":
                raise rejected(result)  # refused before generating: the host settled and keeps the engine
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

    def _admit(self, candidate_id, stop, priority=0):
        """Wait in the queue (Admission.admit), lower priority values first: whether the request must load, None when
        cancelled."""
        try:
            return self.lock.admit(candidate_id, _CommandStop(stop, self._stopping), priority)
        except Refused as exc:
            raise Overloaded(str(exc)) from None

    def _load(self, spec, candidate_id, stop):
        """Load the candidate's engine on the host: None once loaded, else "cancelled" or "timed_out"."""
        try:
            loaded = self.runtime.call("load", {"spec": spec}, _CommandStop(stop, self._stopping),
                                       self.load_seconds(spec))
            if loaded["status"] in {"cancelled", "timed_out"}:
                self.lock.invalidate()
                if self._stopping.is_set():
                    raise EngineError("Engine is stopping", 503)
                return loaded["status"]
            if loaded["status"] != "completed":
                raise EngineError(loaded.get("error") or "Engine loading failed")
            self._loaded(candidate_id)
            self.lock.configure(candidate_id, loaded.get("max_concurrent_requests", 1))
        except BaseException as exc:
            if not self._stopping.is_set():
                self._retire(exc)
            raise
        return None

    def _loaded(self, candidate_id):
        """Count a completed load for GET /metrics, before admission lets other requests in."""
        if self.last_load not in (None, candidate_id):
            self.switches += 1
        self.loads += 1
        self.last_load = candidate_id

    def load(self, stop, candidate_id=None):
        """Load the candidate's engine (default: the default candidate's) without generating, as a request would:
        "completed" once it is loaded, else "cancelled" or "timed_out"."""
        candidate_id = candidate_id or self.default
        needs_load = self._admit(candidate_id, stop)
        if needs_load is None:
            if self._stopping.is_set():
                raise EngineError("Engine is stopping", 503)
            return "cancelled"
        try:
            if self._stopping.is_set():
                raise EngineError("Engine is stopping", 503)
            if self.ownership.blocked:
                raise EngineError("Runtime cleanup is unresolved: " + self.ownership.blocked, 503)
            loaded = self._load(self.spec(candidate_id), candidate_id, stop) if needs_load else None
            return loaded or "completed"
        finally:
            self.lock.release()

    def unload(self, candidate_id=None, idle=None):
        """Release the loaded engine's weights once the requests it runs drain, while later requests wait: the resident
        host closes the engine and frees the model's pool, as at shutdown, but keeps its process and the runtime's
        ownership, and the next request loads the engine again as a switch does. With `candidate_id`, only when its
        engine is the one loaded; with `idle` seconds (--idle-unload), only when no request has used the engine for
        that long and none is waiting. Returns whether an engine was unloaded."""
        if idle is None:
            self.lock.acquire()
        elif not self.lock.acquire(blocking=False):
            return False
        try:
            if self.closed or self._stopping.is_set():
                raise EngineError("Engine is stopping", 503)
            if self.ownership.blocked:
                raise EngineError("Runtime cleanup is unresolved: " + self.ownership.blocked, 503)
            status = self.runtime.status()
            if (not status.get("engine") or (candidate_id is not None and not self.loaded(candidate_id, status))
                    or (idle is not None and time.monotonic() - self.lock.used < idle)):
                return False
            try:
                result = self.runtime.call("unload", {}, _CommandStop(threading.Event(), self._stopping), self.timeout)
            except BaseException as exc:
                self._retire(exc)  # retiring the host process releases the weights too
                if not isinstance(exc, (ValueError, OSError)):
                    raise
            else:
                if result["status"] != "completed":
                    self._retire(result["error"])
            if self.ownership.blocked:
                raise EngineError("Runtime cleanup is unresolved: " + self.ownership.blocked, 503)
            self.lock.invalidate()
            return True
        finally:
            self.lock.release()

    def _unload_idle(self):
        """--idle-unload: unload the engine once no request has used it for idle_unload seconds."""
        while not self._stopping.wait(min(1, self.idle_unload)):
            if time.monotonic() - self.lock.used >= self.idle_unload and self.runtime.status().get("engine"):
                try:
                    self.unload(idle=self.idle_unload)
                except EngineError:
                    pass  # the next request reports the runtime's state

    def unloads_at(self):
        """With idle_unload, the time (seconds since the epoch) the loaded engine unloads unless a request uses it;
        else None."""
        if self.idle_unload is None:
            return None
        return time.time() + max(0, self.lock.used + self.idle_unload - time.monotonic())

    def score(self, prompt_ids, continuation_ids, top_logprobs, stop, candidate_id=None, prompt_rows=True,
              request_id=None, on_started=None):
        """Score fixed text through the engine's serving paths (chat-score-v1): every prompt position from its
        prefill (none with prompt_rows false) and every continuation token from its decode, forced to the
        continuation, each with the token's log-probability and top_logprobs ranked alternatives. Returns (status,
        scored, telemetry) as generate() does, scored {"prompt": rows, "continuation": rows} with rows
        [token, logprob, [[token, logprob], ...]]; `request_id` is the request record's id and `on_started(deadline)`
        runs once the command goes to the engine, as for generate()."""
        candidate_id = candidate_id or self.default
        needs_load = self._admit(candidate_id, stop)
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
            if SCORE_CAPABILITY not in spec["capabilities"]:
                raise EngineError("This engine package does not declare " + SCORE_CAPABILITY, 400)
            limits = spec["limits"]
            if len(prompt_ids) < 1 or len(prompt_ids) + len(continuation_ids) > limits["kv_tokens"]:
                raise EngineError("prompt_ids and continuation_ids must hold 1 to %d tokens" % limits["kv_tokens"], 400)
            if needs_load:
                loaded = self._load(spec, candidate_id, stop)
                if loaded is not None:
                    return loaded, None, None
            request = record("score_request", spec["run_id"], request_id=request_id or uuid.uuid4().hex,
                             candidate_id=spec["candidate_id"], package_sha256=spec["package_sha256"], created_at=now(),
                             prompt_ids=list(prompt_ids), continuation_ids=list(continuation_ids),
                             top_logprobs=top_logprobs, prompt_rows=bool(prompt_rows),
                             limits={"kv_tokens": limits["kv_tokens"], "timeout_seconds": self.timeout})
            try:
                result = self.runtime.call("score", {"spec": spec, "request": request},
                                           _CommandStop(stop, self._stopping), self.timeout, request=request,
                                           on_started=on_started)
            except BaseException as exc:
                self._retire(exc)
                if isinstance(exc, (ValueError, OSError)):
                    if self._stopping.is_set():
                        raise EngineError("Engine is stopping", 503) from exc
                    raise EngineError("Engine host failed: " + str(exc)) from exc
                raise
            if result["status"] in {"cancelled", "timed_out"}:
                if self._stopping.is_set():
                    raise EngineError("Engine is stopping", 503)
                return result["status"], None, None
            if result["status"] == "rejected":
                raise rejected(result)
            if result["status"] != "completed":
                self._retire(result["error"])
                raise EngineError(result["error"] or "Engine failed the request")
            try:
                scored = score_reply(result["response"], request)
                self._check_sealed(spec)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self._retire(exc)
                raise EngineError("Engine returned an invalid response: " + str(exc)) from exc
            return "completed", {"prompt": scored["prompt"], "continuation": scored["continuation"]}, scored["telemetry"]
        finally:
            self.lock.release()

    def tokenize(self, operation, fields, stop, candidate_id=None):
        """Run a tokenize request (chat-tokenize-v1) on the candidate's engine: "tokenize" a text ({"prompt",
        "add_special_tokens"}), "render" a conversation as a reply would prefill it ({"messages"}, the tool options
        and any "template_kwargs" and "continue_final_message") or "detokenize" ids ({"tokens"}). While the
        candidate's engine is loaded it runs at once, beside any generation and outside admission; otherwise it is
        admitted and loads the engine as a reply would (a full queue refuses it as one, Overloaded), and `stop` (the
        client's disconnect) cancels that wait and load as a reply's, a 499. Returns TOKENIZED[operation]:
        {"tokens"}, {"prompt", "tokens"} or {"prompt"}."""
        candidate_id = candidate_id or self.default
        if self._stopping.is_set():
            raise EngineError("Engine is stopping", 503)
        spec = self.spec(candidate_id)
        if TOKENIZE_CAPABILITY not in spec["capabilities"]:
            raise EngineError("This engine package does not declare " + TOKENIZE_CAPABILITY, 404)
        request = record("tokenize_request", spec["run_id"], request_id=uuid.uuid4().hex,
                         candidate_id=spec["candidate_id"], package_sha256=spec["package_sha256"], created_at=now(),
                         operation=operation, **fields)
        if not self.ownership.blocked and self.loaded(candidate_id):
            found = self._tokenize(spec, request)
            if found is not None:
                return found
        # Not loaded, or switched away meanwhile: admitted and loaded as for a reply, never for a client that left; a
        # full queue refuses it as a reply (Overloaded).
        needs_load = self._admit(candidate_id, stop)
        if needs_load is None:
            if self._stopping.is_set():
                raise EngineError("Engine is stopping", 503)
            raise EngineError("Request cancelled", 499)
        try:
            if stop.is_set():
                raise EngineError("Request cancelled", 499)
            if self._stopping.is_set():
                raise EngineError("Engine is stopping", 503)
            if self.ownership.blocked:
                raise EngineError("Runtime cleanup is unresolved: " + self.ownership.blocked, 503)
            loaded = self._load(spec, candidate_id, stop) if needs_load else None
            if loaded == "timed_out":
                raise EngineError("The engine did not load within its load timeout", 504)
            if loaded is not None:
                raise EngineError("Request cancelled", 499)
            found = self._tokenize(spec, request)
            if found is None:
                raise EngineError("The resident host has not loaded the request's engine")
            return found
        finally:
            self.lock.release()

    def _tokenize(self, spec, request):
        """The host's checked answer to a tokenize request, or None when it has not loaded the candidate's engine. A
        tokenize leaves the engine loaded whatever its outcome; the client's disconnect does not cancel it."""
        try:
            result = self.runtime.call("tokenize", {"spec": spec, "request": request}, threading.Event(),
                                       self.timeout, request=request)
        except (ValueError, OSError) as exc:
            raise EngineError("Engine host failed: " + str(exc)) from exc
        if result["status"] == "rejected":
            if result["rejection"]["reason"] == "unloaded":   # crit_api.host.Unloaded
                return None
            raise rejected(result)
        if result["status"] == "timed_out":
            raise EngineError("The engine did not finish within the request timeout", 504)
        if result["status"] != "completed":
            raise EngineError(result["error"] or "Engine failed the request")
        try:
            return tokenize_reply(result["response"], request)
        except (ValueError, KeyError, TypeError) as exc:
            raise EngineError("Engine returned an invalid response: " + str(exc)) from exc

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
