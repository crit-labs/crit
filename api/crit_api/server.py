"""OpenAI-compatible HTTP endpoints over one ServedEngine (standard library and crit_core only)."""
import functools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import re
import threading
import time
import uuid

from crit_core.constraints import UnsupportedConstraint, compile_language, json_object_matches, matches
from crit_core.text import StopMatcher, TextError

from .engine import CONSTRAINTS_CAPABILITY, EngineError

MAX_BODY = 8 * 1024 * 1024
MAX_STOPS = 4
DEFAULT_MAX_TOKENS = 1024
# Accepted for client compatibility. The engine applies the sampling frozen in its candidate configuration until
# the adapter contract carries per-request sampling; responses name the parameters that were ignored.
IGNORED = {"temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
           "repetition_penalty", "logit_bias", "user", "metadata", "store", "service_tier", "parallel_tool_calls",
           "reasoning_effort", "prediction", "modalities"}
UNSUPPORTED = {"tools": "tool calls", "tool_choice": "tool calls", "functions": "function calls",
               "function_call": "function calls",
               "logprobs": "logprobs", "top_logprobs": "logprobs", "audio": "audio output"}
KNOWN = {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens", "stop", "n",
         "response_format"}


class RequestError(Exception):
    def __init__(self, message, status=400, param=None, code=None):
        super().__init__(message)
        self.status, self.param, self.code = status, param, code


def content_text(content, index):
    """Message content as text: a string, or a list of text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(p, dict) for p in content):
        if any(p.get("type") != "text" or not isinstance(p.get("text"), str) for p in content):
            raise RequestError("This engine accepts text content only", param=f"messages[{index}].content")
        return "".join(p["text"] for p in content)
    raise RequestError("Message content must be a string or a list of text parts", param=f"messages[{index}].content")


def conversation(messages):
    """An optional leading system message, then user/assistant turns ending with the user.

    Consecutive messages from the same role are joined with a blank line, since the adapter contract alternates.
    """
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list", param="messages")
    result = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer", "user", "assistant"}:
            raise RequestError("Each message needs a role of system, user or assistant", param=f"messages[{index}].role")
        role = "system" if message["role"] == "developer" else message["role"]
        content = content_text(message.get("content"), index)
        if role == "system" and any(m["role"] != "system" for m in result):
            raise RequestError("System messages must come before the conversation", param=f"messages[{index}]")
        if result and result[-1]["role"] == role:
            result[-1]["content"] += "\n\n" + content
        else:
            result.append({"role": role, "content": content})
    turns = result[1:] if result[0]["role"] == "system" else result
    if not turns or turns[0]["role"] != "user" or turns[-1]["role"] != "user":
        raise RequestError("The conversation must start and end with a user message", param="messages")
    return result


def stops(value):
    """The request's stop strings: up to 4 (OpenAI's limit), each within Core's StopMatcher limits."""
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or len(values) > MAX_STOPS:
        raise RequestError("stop must be a non-empty string or a list of up to 4", param="stop")
    try:
        StopMatcher(values)
    except TextError as exc:
        raise RequestError("stop: " + str(exc), param="stop") from None
    return values


@functools.lru_cache(maxsize=32)
def _schema_grammar(source):
    return compile_language("schema", source)


def response_format(value):
    """The request's constraint for the engine (None for plain text), compiled with Core's constraint compiler.

    json_object passes through; a json_schema's schema is compiled into a crit-byte-grammar-v1 artifact, so the
    engine needs no compiler. Schemas outside the supported subset are refused with the compiler's reason.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("type") not in {"text", "json_object", "json_schema"}:
        raise RequestError("response_format.type must be text, json_object or json_schema", param="response_format")
    if value["type"] == "text":
        return None
    if value["type"] == "json_object":
        return {"type": "json_object"}
    spec = value.get("json_schema")
    if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
        raise RequestError("response_format.json_schema.schema must be a JSON schema object", param="response_format")
    name = spec.get("name", "response")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise RequestError("response_format.json_schema.name must be 1 to 64 letters, digits, _ or -",
                           param="response_format")
    try:
        grammar = _schema_grammar(json.dumps(spec["schema"], ensure_ascii=False, allow_nan=False))
    except (UnsupportedConstraint, ValueError) as exc:
        raise RequestError("response_format.json_schema.schema is outside the supported JSON schema subset "
                           "(docs/SERVING.md#structured-output): " + str(exc), param="response_format",
                           code="unsupported_schema") from None
    return {"type": "json_schema", "name": name, "schema": spec["schema"], "grammar": grammar}


def conforms(constraint, text):
    """Whether a complete reply belongs to the constraint's language (Core's host checks)."""
    data = text.encode("utf-8")
    return json_object_matches(data) if constraint["type"] == "json_object" else matches(constraint["grammar"], data)


class Api:
    """Parses OpenAI chat-completion requests and runs them on the engine."""
    def __init__(self, engine, model_name=None, api_key=None, cors_origin=None, log=None):
        self.engine = engine
        self.model = model_name or engine.model_id()
        self.api_key = api_key
        self.cors_origin = cors_origin
        self.created = int(time.time())
        self.log = log

    def completion_request(self, body):
        if not isinstance(body, dict):
            raise RequestError("The request body must be a JSON object")
        for key, feature in UNSUPPORTED.items():
            if key in body and body[key] not in (None, False, [], "auto", "none") and not (
                    key == "response_format" and isinstance(body[key], dict) and body[key].get("type") == "text"):
                raise RequestError(f"This engine does not support {feature} yet", param=key, code="unsupported_parameter")
        if body.get("n", 1) != 1:
            raise RequestError("This engine returns one choice per request", param="n")
        candidate = self.candidate(body.get("model"))
        constraint = response_format(body.get("response_format"))
        if constraint and CONSTRAINTS_CAPABILITY not in self.engine.spec(candidate)["capabilities"]:
            raise RequestError(f"This engine cannot apply response_format {constraint['type']}: its package does not "
                               f"declare {CONSTRAINTS_CAPABILITY}", param="response_format", code="unsupported_parameter")
        messages = conversation(body.get("messages"))
        limit = body.get("max_completion_tokens", body.get("max_tokens"))
        if limit is None:
            limit = min(DEFAULT_MAX_TOKENS, self.engine.kv_tokens - 1)
        if type(limit) is not int or not 1 <= limit < self.engine.kv_tokens:
            raise RequestError(f"max_tokens must be between 1 and {self.engine.kv_tokens - 1}", param="max_tokens")
        options = body.get("stream_options") or {}
        if not isinstance(options, dict):
            raise RequestError("stream_options must be an object", param="stream_options")
        ignored = sorted(k for k in body if k in IGNORED)
        unknown = sorted(k for k in body if k not in IGNORED | set(UNSUPPORTED) | KNOWN)
        return {"candidate": candidate, "model": self.model if candidate == self.engine.default else self.engine.model_id(candidate),
                "messages": messages, "max_tokens": limit, "stream": bool(body.get("stream")),
                "include_usage": bool(options.get("include_usage")), "stop": stops(body.get("stop")),
                "response_format": constraint, "ignored": ignored + unknown}

    def candidate(self, model):
        """The candidate a request's model names: RUN_ID/CANDIDATE_ID or a bare candidate ID. Any other name (clients
        often send a fixed one) selects the default candidate."""
        if model is None or model == self.model:
            return self.engine.default
        if not isinstance(model, str):
            raise RequestError("model must be a string", param="model")
        candidate = model.removeprefix(self.engine.run_id + "/")
        if candidate == model and not re.fullmatch(r"candidate-\d+", model):
            return self.engine.default
        known = lambda refresh=False: candidate in {c["candidate_id"] for c in self.engine.candidates(refresh)}
        if not known() and not known(refresh=True):
            raise RequestError("Unknown model " + model + "; GET /v1/models lists the served candidates", 404,
                               param="model", code="model_not_found")
        try:
            self.engine.spec(candidate)
        except (ValueError, EngineError) as exc:
            raise RequestError("Cannot serve " + model + ": " + str(exc), 409, param="model") from exc
        return candidate

    def models(self):
        cards = []
        for candidate in self.engine.candidates():
            default = candidate["candidate_id"] == self.engine.default
            cards.append({"id": self.model if default else self.engine.model_id(candidate["candidate_id"]),
                          "object": "model", "created": self.created, "owned_by": "crit", "label": candidate["label"],
                          "context_length": self.engine.kv_tokens, "default": default})
        return cards

    def usage(self, measured):
        if measured is None:
            return None
        return {"prompt_tokens": measured["prompt_tokens"], "completion_tokens": measured["output_tokens"],
                "total_tokens": measured["prompt_tokens"] + measured["output_tokens"]}

    def finish_reason(self, stopped, measured, limit):
        if stopped:
            return "stop"
        return "length" if measured and measured["output_tokens"] >= limit else "stop"

    def run(self, request, emit, cancel):
        """Generate; `emit(piece)` gets released text. Returns (finish_reason, measured or None). Stop strings follow
        Core's rule (crit_core.text.StopMatcher): the first stop to complete wins, and only text that could still
        begin a stop is held back."""
        scanner = StopMatcher(request["stop"])
        def progress(text):
            piece = scanner.advance(text)
            if piece:
                emit(piece)
            if scanner.stopped is not None:
                cancel.set()  # Stop sequence reached: end the engine's turn early.
        try:
            status, assistant, measured = self.engine.generate(request["messages"], request["max_tokens"], cancel,
                                                               on_text=progress if request["stream"] or request["stop"] else None,
                                                               candidate_id=request["candidate"],
                                                               response_format=request.get("response_format"))
        except (ValueError, OSError) as exc:
            raise EngineError(str(exc)) from exc
        if status == "completed":
            try:
                piece = scanner.advance(assistant, final=True)
            except TextError as exc:
                raise EngineError("The engine's reply does not extend its streamed text: " + str(exc)) from exc
            finish = self.finish_reason(scanner.stopped is not None, measured, request["max_tokens"])
            constraint = request.get("response_format")
            # A reply the engine ended itself must be in the requested language; one cut by the output cap or a
            # stop string is a prefix and is returned as cut, like OpenAI's finish_reason "length".
            if constraint and finish == "stop" and scanner.stopped is None and not conforms(constraint, assistant):
                raise EngineError("The engine's reply does not match response_format " + constraint["type"])
            if piece:
                emit(piece)
            return finish, measured
        if scanner.stopped is not None:
            return "stop", None
        if status == "timed_out":
            raise EngineError("The engine did not finish within the request timeout", 504)
        raise EngineError("Request cancelled", 499)

    def telemetry(self, measured):
        if measured is None:
            return None
        return {key: measured[key] for key in ("prompt_tokens", "output_tokens", "prefill_tps", "decode_tps", "ttft_ms")}


class Handler(BaseHTTPRequestHandler):
    server_version = "crit-serve"
    protocol_version = "HTTP/1.1"

    @property
    def api(self):
        return self.server.api

    def log_message(self, format, *args):
        if self.api.log:
            self.api.log("%s %s" % (self.address_string(), format % args))

    def send_json(self, status, value, headers=()):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.cors()
        for name, content in headers:
            self.send_header(name, content)
        self.end_headers()
        self.wfile.write(body)

    def error(self, status, message, kind="invalid_request_error", param=None, code=None):
        self.send_json(status, {"error": {"message": message, "type": kind, "param": param, "code": code}})

    def cors(self):
        if self.api.cors_origin:
            self.send_header("Access-Control-Allow-Origin", self.api.cors_origin)
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def authorized(self):
        if not self.api.api_key:
            return True
        supplied = self.headers.get("Authorization", "")
        if supplied.startswith("Bearer ") and hmac.compare_digest(supplied[7:].encode(), self.api.api_key.encode()):
            return True
        self.error(401, "Missing or invalid API key", "authentication_error", code="invalid_api_key")
        return False

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            status = self.api.engine.status()
            ok = status.get("state") not in {"blocked", "interrupted"}
            return self.send_json(200 if ok else 503, {"status": "ok" if ok else "unavailable", "engine": status})
        if not self.authorized():
            return
        if path == "/v1/models":
            return self.send_json(200, {"object": "list", "data": self.api.models()})
        if path.startswith("/v1/models/"):
            card = next((c for c in self.api.models() if c["id"] == path[len("/v1/models/"):]), None)
            if card:
                return self.send_json(200, card)
            return self.error(404, "Unknown model", code="model_not_found")
        self.error(404, "Unknown endpoint " + path, code="not_found")

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY:
            self.close_connection = True  # An unread body must not be parsed as the next request.
            return self.error(400 if length <= MAX_BODY else 413, "The request body must be between 1 byte and 8 MiB")
        raw = self.rfile.read(length)
        if not self.authorized():
            return
        if path != "/v1/chat/completions":
            if path == "/v1/completions":
                return self.error(404, "Raw-prompt completions need the engine's tokenizer in crit; use /v1/chat/completions",
                                  code="not_supported")
            return self.error(404, "Unknown endpoint " + path, code="not_found")
        try:
            request = self.api.completion_request(json.loads(raw))
        except (ValueError, UnicodeDecodeError) as exc:
            return self.error(400, "Invalid JSON body: " + str(exc))
        except RequestError as exc:
            return self.error(exc.status, str(exc), param=exc.param, code=exc.code)
        (self.stream if request["stream"] else self.complete)(request)

    def headers_for(self, request):
        return [("X-Crit-Ignored-Parameters", ",".join(request["ignored"]))] if request["ignored"] else []

    def complete(self, request):
        pieces = []
        try:
            finish, measured = self.api.run(request, pieces.append, threading.Event())
        except EngineError as exc:
            return self.error(exc.status, str(exc), "server_error" if exc.status >= 500 else "invalid_request_error")
        value = {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                 "model": request["model"], "system_fingerprint": self.api.engine.spec(request["candidate"])["package_sha256"][:16],
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(pieces)},
                              "logprobs": None, "finish_reason": finish}]}
        usage = self.api.usage(measured)
        if usage:
            value["usage"] = usage
        if measured:
            value["crit_telemetry"] = self.api.telemetry(measured)
        self.send_json(200, value, self.headers_for(request))

    def stream(self, request):
        cancel = threading.Event()
        ident, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        fingerprint = self.api.engine.spec(request["candidate"])["package_sha256"][:16]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.cors()
        for name, content in self.headers_for(request):
            self.send_header(name, content)
        self.end_headers()
        self.close_connection = True

        def event(value):
            try:
                self.wfile.write(b"data: " + json.dumps(value).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            except OSError:
                cancel.set()  # The client went away: stop the engine.

        def chunk(delta, finish=None, usage=None, choices=True):
            value = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": request["model"],
                     "system_fingerprint": fingerprint,
                     "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}] if choices else []}
            if usage is not None or (request["include_usage"] and not choices):
                value["usage"] = usage
            return value

        event(chunk({"role": "assistant", "content": ""}))
        try:
            finish, measured = self.api.run(request, lambda piece: event(chunk({"content": piece})), cancel)
        except EngineError as exc:
            event({"error": {"message": str(exc), "type": "server_error", "param": None, "code": None}})
        else:
            event(chunk({}, finish))
            if request["include_usage"]:
                event(chunk({}, usage=self.api.usage(measured), choices=False))
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, api):
        super().__init__(address, Handler)
        self.api = api
