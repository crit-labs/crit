"""OpenAI-compatible HTTP endpoints over one ServedEngine (standard library and crit_core only)."""
import functools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import re
import time
import uuid

from crit_core.constraints import UnsupportedConstraint, compile_language, json_object_matches, matches
from crit_core.text import StopMatcher, TextError, message_content

from .disconnect import ClientDisconnect
from .delivery import StreamDelivery
from .engine import CONSTRAINTS_CAPABILITY, EngineError
from .logprobs import logprob_options
from .options import (RequestError, SAMPLING, TOOLS_CAPABILITY, TOOL_STREAM_CAPABILITY, complete_arguments, require,
                      sampling_capability, sampling_options, tool_calls, tool_options)

MAX_BODY = 8 * 1024 * 1024
MAX_STOPS = 4
DEFAULT_MAX_TOKENS = 1024
# Only non-generation metadata is accepted without applying it.
IGNORED = {"user", "metadata", "store", "service_tier"}
UNSUPPORTED = {"functions": "legacy function calls (use tools)",
               "function_call": "function calls", "audio": "audio output",
               "reasoning_effort": "reasoning effort", "prediction": "predicted output", "modalities": "modalities"}
KNOWN = {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens", "stop", "n",
         "response_format", "tools", "tool_choice", "parallel_tool_calls", "logprobs", "top_logprobs"} | SAMPLING


def content_text(content, index, capabilities=()):
    param = f"messages[{index}].content"
    if content is None:
        raise RequestError("Message content must be a string or a list of parts", param=param)
    try:
        return message_content(content, param, images="chat-images-v1" in capabilities)
    except (TextError, ValueError) as exc:
        raise RequestError(str(exc), param=param) from None


def conversation(messages, capabilities=()):
    """An optional leading system message, then user/assistant turns ending with the user.

    Consecutive messages from the same role are joined with a blank line, since the adapter contract alternates.
    """
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list", param="messages")
    if any(isinstance(m, dict) and (m.get("role") == "tool" or m.get("tool_calls")) for m in messages):
        require(capabilities, TOOLS_CAPABILITY, "messages")
        return tool_conversation(messages, capabilities)
    result = []
    for index, message in enumerate(messages):
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or message["role"] not in {"system", "developer", "user", "assistant"}):
            raise RequestError("Each message needs a role of system, user or assistant", param=f"messages[{index}].role")
        role = "system" if message["role"] == "developer" else message["role"]
        content = content_text(message.get("content"), index, capabilities)
        if role == "system" and any(m["role"] != "system" for m in result):
            raise RequestError("System messages must come before the conversation", param=f"messages[{index}]")
        if result and result[-1]["role"] == role:
            previous = result[-1]["content"]
            if isinstance(previous, str) and isinstance(content, str):
                result[-1]["content"] += "\n\n" + content
            else:
                parts = lambda value: [{"type": "text", "text": value}] if isinstance(value, str) else value
                result[-1]["content"] = parts(previous) + [{"type": "text", "text": "\n\n"}] + parts(content)
        else:
            result.append({"role": role, "content": content})
    turns = result[1:] if result[0]["role"] == "system" else result
    if not turns or turns[0]["role"] != "user" or turns[-1]["role"] != "user":
        raise RequestError("The conversation must start and end with a user message", param="messages")
    return result


def tool_conversation(messages, capabilities=()):
    """Preserve call ids and message boundaries; every call must have exactly one result."""
    result, pending, seen = [], set(), set()
    for index, message in enumerate(messages):
        param = f"messages[{index}]"
        if not isinstance(message, dict):
            raise RequestError("Each message must be an object", param=param)
        role = message.get("role")
        role = "system" if role == "developer" else role
        if not isinstance(role, str) or role not in {"system", "user", "assistant", "tool"}:
            raise RequestError("Invalid message role", param=param + ".role")
        if role == "system" and any(m["role"] != "system" for m in result):
            raise RequestError("System messages must come first", param=param)
        if pending and role != "tool":
            raise RequestError("Supply results for all pending tool calls before the next turn", param=param)
        calls = message.get("tool_calls")
        if calls and role != "assistant":
            raise RequestError("Only assistant messages may contain tool_calls", param=param)
        content = message.get("content")
        row = {"role": role, "content": "" if content is None and calls else content_text(content, index, capabilities)}
        if calls:
            try:
                row["tool_calls"] = tool_calls(calls)
            except (ValueError, TypeError) as exc:
                raise RequestError(str(exc), param=param + ".tool_calls") from None
            pending = {c["id"] for c in calls}
            if pending & seen:
                raise RequestError("Tool call ids must be unique across the conversation", param=param)
            seen.update(pending)
        if role == "tool":
            ident = message.get("tool_call_id")
            if not isinstance(ident, str) or ident not in pending:
                raise RequestError("tool_call_id must reference an unanswered assistant call", param=param)
            pending.remove(ident)
            row["tool_call_id"] = ident
        for key in ("name", "reasoning_content"):
            if key in message:
                if not isinstance(message[key], str):
                    raise RequestError(key + " must be a string", param=param)
                row[key] = message[key]
        result.append(row)
    turns = [m for m in result if m["role"] != "system"]
    if pending or not turns or turns[0]["role"] != "user" or turns[-1]["role"] not in {"user", "tool"}:
        raise RequestError("Conversation must start with user and end with user or completed tool results", param="messages")
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
            if key in body and body[key] is not None:
                raise RequestError(f"This engine does not support {feature} yet", param=key, code="unsupported_parameter")
        if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
            raise RequestError("This engine returns one choice per request", param="n")
        if "stream" in body and type(body["stream"]) is not bool:
            raise RequestError("stream must be a boolean", param="stream")
        candidate = self.candidate(body.get("model"))
        spec = self.engine.spec(candidate)
        capabilities = spec["capabilities"]
        sampling = sampling_options(body, capabilities)
        logprobs = logprob_options(body, capabilities)
        if logprobs is not None and body.get("stop"):
            # The turn is cancelled where a stop string completes, before the engine reports its tokens.
            raise RequestError("Stop strings cannot be combined with logprobs", param="stop")
        tools = tool_options(body, capabilities)
        constraint = response_format(body.get("response_format"))
        if constraint and CONSTRAINTS_CAPABILITY not in self.engine.spec(candidate)["capabilities"]:
            raise RequestError(f"This engine cannot apply response_format {constraint['type']}: its package does not "
                               f"declare {CONSTRAINTS_CAPABILITY}", param="response_format", code="unsupported_parameter")
        messages = conversation(body.get("messages"), capabilities)
        if tools and (constraint or body.get("stop")):
            raise RequestError("Tool requests cannot combine stop strings or response_format",
                               param="response_format" if constraint else "stop")
        limit = body.get("max_completion_tokens", body.get("max_tokens"))
        if limit is None:
            limit = min(DEFAULT_MAX_TOKENS, spec["limits"]["kv_tokens"] - 1)
        if type(limit) is not int or not 1 <= limit < spec["limits"]["kv_tokens"]:
            raise RequestError(f"max_tokens must be between 1 and {spec['limits']['kv_tokens'] - 1}", param="max_tokens")
        options = body.get("stream_options") or {}
        if not isinstance(options, dict):
            raise RequestError("stream_options must be an object", param="stream_options")
        ignored = sorted(k for k in body if k in IGNORED)
        unknown = sorted(k for k in body if k not in IGNORED | set(UNSUPPORTED) | KNOWN)
        if unknown:
            raise RequestError("Unknown request parameter: " + unknown[0], param=unknown[0], code="unsupported_parameter")
        return {"candidate": candidate, "model": self.model if candidate == self.engine.default else self.engine.model_id(candidate),
                "messages": messages, "max_tokens": limit, "stream": bool(body.get("stream")),
                "include_usage": bool(options.get("include_usage")), "stop": stops(body.get("stop")),
                "response_format": constraint, "sampling": sampling, "tool_options": tools, "logprobs": logprobs,
                "ignored": ignored}

    def candidate(self, model):
        """The candidate a request's model names: RUN_ID/CANDIDATE_ID or a bare candidate ID. Any other name (clients
        often send a fixed one) selects the default candidate."""
        if model is None or model == self.model:
            return self.engine.default
        if not isinstance(model, str):
            raise RequestError("model must be a string", param="model")
        try:
            candidate = self.engine.resolve_model(model)
        except EngineError as exc:
            raise RequestError(str(exc), exc.status, param="model", code="model_not_found") from exc
        try:
            self.engine.spec(candidate)
        except (ValueError, EngineError) as exc:
            raise RequestError("Cannot serve " + model + ": " + str(exc), 409, param="model") from exc
        return candidate

    def models(self):
        cards = []
        for candidate in self.engine.candidates():
            default = candidate["candidate_id"] == self.engine.default
            spec = self.engine.spec(candidate["candidate_id"])
            cards.append({"id": self.model if default else self.engine.model_id(candidate["candidate_id"]),
                          "object": "model", "created": self.created, "owned_by": "crit", "label": candidate["label"],
                          "context_length": spec["limits"]["kv_tokens"], "default": default,
                          "capabilities": spec["capabilities"],
                          "supported_sampling_parameters": sorted(k for k in SAMPLING if sampling_capability(k)
                                                                  in spec["capabilities"])})
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

    def run(self, request, emit, cancel, emit_tools=None, emit_reasoning=None, on_started=None):
        """Generate; emit receives text, emit_tools indexed deltas, emit_reasoning reasoning text (apart from the
        reply). Returns (finish_reason, measured, calls, reasoning, logprobs), logprobs being OpenAI's logprobs
        content when the request asked for it.
        Stop strings follow
        Core's rule (crit_core.text.StopMatcher): the first stop to complete wins, and only text that could still
        begin a stop is held back."""
        scanner = StopMatcher(request["stop"])
        streamed = []

        def reasoned(text):
            streamed.append(text)
            if emit_reasoning:
                emit_reasoning(text)

        def progress(text):
            piece = scanner.push(text)
            if piece:
                emit(piece)
            if scanner.stopped is not None:
                cancel.set()  # Stop sequence reached: end the engine's turn early.
        try:
            tool_incremental = (request["stream"] and request["tool_options"] and
                                TOOL_STREAM_CAPABILITY in self.engine.spec(request["candidate"])["capabilities"])
            incremental = (request["stream"] or request["stop"]) and (not request["tool_options"] or tool_incremental)
            status, assistant, measured = self.engine.generate(request["messages"], request["max_tokens"], cancel,
                                                               on_text=progress if incremental else None,
                                                               candidate_id=request["candidate"],
                                                               response_format=request.get("response_format"),
                                                               sampling=request["sampling"], tool_options=request["tool_options"],
                                                               on_tools=emit_tools if tool_incremental else None,
                                                               on_reasoning=reasoned if request["stream"] else None,
                                                               on_started=on_started, logprobs=request.get("logprobs"))
        except (ValueError, OSError) as exc:
            raise EngineError(str(exc)) from exc
        if status == "completed":
            calls = assistant.get("tool_calls") if isinstance(assistant, dict) else None
            reasoning = assistant.get("reasoning", "") if isinstance(assistant, dict) else ""
            logprobs = assistant.get("logprobs") if isinstance(assistant, dict) else None
            assistant = assistant["content"] if isinstance(assistant, dict) else assistant
            sent = "".join(streamed)
            if not reasoning.startswith(sent):
                raise EngineError("The engine's reasoning does not extend its streamed reasoning")
            if emit_reasoning and reasoning != sent:
                emit_reasoning(reasoning[len(sent):])
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
            if calls:
                if emit_tools and not tool_incremental:
                    emit_tools([dict(call, index=i) for i, call in enumerate(calls)])
                # A call the output limit cut keeps the arguments written so far, and the reply finishes as cut.
                return ("tool_calls" if complete_arguments(calls[-1]) else "length"), measured, calls, reasoning, logprobs
            return finish, measured, [], reasoning, logprobs
        if scanner.stopped is not None:
            return "stop", None, [], "".join(streamed), None
        if status == "timed_out":
            raise EngineError("The engine did not finish within the request timeout", 504)
        raise EngineError("Request cancelled", 499)

    def telemetry(self, measured):
        if measured is None:
            return None
        return {key: measured[key] for key in ("prompt_tokens", "output_tokens", "prefill_tps", "decode_tps", "ttft_ms",
                                               "tokens_per_round") if key in measured}


class Handler(BaseHTTPRequestHandler):
    server_version = "crit-serve"
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True  # each streamed chunk leaves at once, not after the client's delayed ACK

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
        with ClientDisconnect(self.connection) as client:
            try:
                (self.stream if request["stream"] else self.complete)(request, client.cancel)
            except OSError:
                # A response write can fail between transport polls.
                client.cancel.set()
                self.close_connection = True
        if client.disconnected.is_set():
            self.close_connection = True

    def headers_for(self, request):
        return [("X-Crit-Ignored-Parameters", ",".join(request["ignored"]))] if request["ignored"] else []

    def complete(self, request, cancel):
        pieces = []
        try:
            finish, measured, calls, reasoning, logprobs = self.api.run(request, pieces.append, cancel)
        except EngineError as exc:
            return self.error(exc.status, str(exc), "server_error" if exc.status >= 500 else "invalid_request_error")
        value = {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                 "model": request["model"], "system_fingerprint": self.api.engine.spec(request["candidate"])["package_sha256"][:16],
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(pieces)},
                              "logprobs": None if logprobs is None else {"content": logprobs}, "finish_reason": finish}]}
        if calls:
            value["choices"][0]["message"].update(content="".join(pieces) or None, tool_calls=calls)
        if reasoning:
            value["choices"][0]["message"]["reasoning_content"] = reasoning
        usage = self.api.usage(measured)
        if usage:
            value["usage"] = usage
        if measured:
            value["crit_telemetry"] = self.api.telemetry(measured)
        self.send_json(200, value, self.headers_for(request))

    def stream(self, request, cancel):
        with StreamDelivery(self.connection, cancel, self.api.engine._stopping, self.api.engine.timeout) as delivery:
            previous, self.wfile = self.wfile, delivery
            try:
                self._stream(request, cancel, delivery.set_deadline)
            finally:
                self.wfile = previous

    def _stream(self, request, cancel, on_started):
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

        def chunk(delta, finish=None, usage=None, choices=True, logprobs=None, telemetry=None):
            value = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": request["model"],
                     "system_fingerprint": fingerprint,
                     "choices": [{"index": 0, "delta": delta, "logprobs": None if logprobs is None else {"content": logprobs},
                                  "finish_reason": finish}] if choices else []}
            if usage is not None or (request["include_usage"] and not choices):
                value["usage"] = usage
            if telemetry is not None:
                value["crit_telemetry"] = telemetry
            return value

        event(chunk({"role": "assistant", "content": ""}))
        try:
            finish, measured, _, _, logprobs = self.api.run(request, lambda piece: event(chunk({"content": piece})),
                cancel, lambda deltas: event(chunk({"tool_calls": deltas})),
                lambda piece: event(chunk({"reasoning_content": piece})), on_started=on_started)
        except EngineError as exc:
            event({"error": {"message": str(exc), "type": "server_error", "param": None, "code": None}})
        else:
            # Text streams as the engine emits it; the reply's logprobs come from its final record, with the finish chunk.
            event(chunk({}, finish, logprobs=logprobs))
            if request["include_usage"]:
                event(chunk({}, usage=self.api.usage(measured), choices=False, telemetry=self.api.telemetry(measured)))
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
