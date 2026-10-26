"""OpenAI-compatible HTTP endpoints over one ServedEngine (standard library and crit_core only)."""
import array
import base64
from contextlib import ExitStack, contextmanager
import functools
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import re
import sys
import threading
import time
from urllib.parse import parse_qs
import uuid

from crit_core.constraints import UnsupportedConstraint, compile_language, json_object_matches, matches
from crit_core.serving import (COMPLETION_CAPABILITY, CONTINUE_CAPABILITY, FIT_OUTPUT_CAPABILITY, LOGPROBS_CAPABILITY,
                                SCORE_CAPABILITY, TOKENIZE_CAPABILITY, TOP_LOGPROBS)
from crit_core.text import StopMatcher, TextError, message_content
from crit_core.text.stops import MAX_STOP_BYTES, MAX_STOPS

from crit import __version__

from . import anthropic, ollama, responses
from .content import MAX_CONTENT
from .disconnect import ClientDisconnect
from .delivery import StreamDelivery
from .engine import CONSTRAINTS_CAPABILITY, GRAMMAR_CAPABILITY, ContextLengthError, EngineError, Overloaded
from .logprobs import completion_logprobs, logprob_options
from .metrics import Metrics
from .options import (EXTRA_SAMPLING, MAX_TOOLS, RequestError, SAMPLING, TEMPLATE_KWARGS_CAPABILITY, TOOLS_CAPABILITY,
                      TOOL_STREAM_CAPABILITY, complete_arguments, crit_options, require, sampling_capability,
                      sampling_options, template_options, thinking_kwargs, tool_calls, tool_options, unknown_parameters,
                      validate_tool_reply)

MAX_BODY = 8 * 1024 * 1024
BODY_SIZE = "The request body must be between 1 byte and 8 MiB"
TLS_HANDSHAKE_SECONDS = 10
GET_PATHS = {"", "/health", "/api/version", "/v1/models", "/metrics", "/api/tags", "/api/ps"}
# Tokenizer and chat template of the engine (chat-tokenize-v1): vLLM's /tokenize and /detokenize, llama-server's
# /apply-template and Anthropic's token count.
TOKENIZE_PATHS = ("/tokenize", "/detokenize", "/apply-template", "/v1/messages/count_tokens")
POST_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/crit/score", "/v1/crit/abort", "/v1/crit/load",
              "/v1/crit/unload", "/v1/messages", "/v1/responses", "/api/chat", "/api/generate", "/api/show",
              *TOKENIZE_PATHS}
EMBEDDINGS = "; run a separate embedding server for embeddings and reranking"
# Endpoints clients look for beside chat, with the paths under them: a 404 not_supported names the endpoint.
NOT_SERVED = {"/v1/embeddings": EMBEDDINGS, "/v1/rerank": EMBEDDINGS, "/v2/rerank": EMBEDDINGS, "/rerank": EMBEDDINGS,
              "/v1/score": EMBEDDINGS + " (crit's fixed-text scores are /v1/crit/score)", "/api/embed": EMBEDDINGS,
              "/api/embeddings": EMBEDDINGS, "/v1/audio": "", "/v1/images": "", "/v1/moderations": "", "/v1/files": "",
              "/v1/batches": ""}
DEFAULT_MAX_TOKENS = 1024
RETRY_AFTER = 1  # seconds a client the queue refused is asked to wait before it retries (Retry-After)
PING = b": ping\n\n"  # an SSE comment: clients skip it, while proxies and idle timers see the stream is alive
# Only non-generation metadata is accepted without applying it.
IGNORED = {"user", "metadata", "store", "service_tier", "prompt_cache_key", "prompt_cache_retention",
           "safety_identifier"}
UNSUPPORTED = {"functions": "legacy function calls (use tools)",
               "function_call": "function calls", "audio": "audio output",
               "prediction": "predicted output", "modalities": "modalities"}
# What --lenient ignores of UNSUPPORTED: without them the reply is still the one asked for.
HINTS = {"prediction"}
STRUCTURED_OUTPUTS = ("regex", "choice", "grammar", "json")
# vLLM's earlier names for one structured_outputs field.
GUIDED = {"guided_regex": "regex", "guided_choice": "choice", "guided_grammar": "grammar", "guided_json": "json"}
KNOWN = {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens", "stop", "n",
         "include_stop_str_in_output", "response_format", "structured_outputs", "tools", "tool_choice",
         "parallel_tool_calls", "logprobs", "top_logprobs", "crit", "chat_template_kwargs", "reasoning_effort",
         "priority", "continue_final_message", "add_generation_prompt"} | set(GUIDED) | SAMPLING | set(EXTRA_SAMPLING)
# POST /v1/completions (completion-v1); suffix is refused until fill-in-the-middle exists.
COMPLETION_KNOWN = {"model", "prompt", "suffix", "echo", "stream", "stream_options", "max_tokens", "stop", "n", "best_of",
                    "include_stop_str_in_output", "logprobs", "crit", "priority"} | SAMPLING | set(EXTRA_SAMPLING)
SCORE_FORMATS = ("json", "packed")
# Marks a system or developer message after the first turn, which reaches the engine as user text.
SYSTEM_MARK = "System: "
# A chat request's fields its prompt comes from, which /tokenize and /apply-template render (chat-tokenize-v1).
RENDERED = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls", "chat_template_kwargs",
            "reasoning_effort", "continue_final_message", "add_generation_prompt"}


def content_text(content, index, capabilities=()):
    param = f"messages[{index}].content"
    if content is None:
        raise RequestError("Message content must be a string or a list of parts", param=param)
    try:
        return message_content(content, param, images="chat-images-v1" in capabilities)
    except (TextError, ValueError) as exc:
        raise RequestError(str(exc), param=param) from None


def joined(first, second):
    """Two message contents as one, a blank line between: a string, or parts when either has images."""
    if isinstance(first, str) and isinstance(second, str):
        return first + "\n\n" + second
    parts = lambda value: [{"type": "text", "text": value}] if isinstance(value, str) else value
    return parts(first) + [{"type": "text", "text": "\n\n"}] + parts(second)


def late_system(content):
    """A system or developer message after the first turn as user content marked with SYSTEM_MARK: the adapter
    contract takes system text only before the conversation, and role-play UIs, RAG injectors and agent handoffs send
    it later."""
    return SYSTEM_MARK + content if isinstance(content, str) else [{"type": "text", "text": SYSTEM_MARK}] + content


def continuation(body, capabilities):
    """Whether the request continues its final assistant message: vLLM's continue_final_message true with
    add_generation_prompt false, for a package that declares chat-continue-v1. Their defaults change nothing."""
    for key in ("continue_final_message", "add_generation_prompt"):
        if body.get(key) is not None and type(body[key]) is not bool:
            raise RequestError(key + " must be a boolean", param=key)
    continuing = body.get("continue_final_message") is True
    if continuing and body.get("add_generation_prompt") is not False:
        raise RequestError("continue_final_message needs add_generation_prompt false", param="add_generation_prompt")
    if not continuing and body.get("add_generation_prompt") is False:
        raise RequestError("add_generation_prompt false needs continue_final_message true",
                           param="add_generation_prompt")
    if continuing:
        require(capabilities, CONTINUE_CAPABILITY, "continue_final_message")
    return continuing


def conversation(messages, capabilities=(), continuing=False):
    """An optional leading system message, then user/assistant turns ending with the user, or with the assistant
    message the request is `continuing`.

    Consecutive messages from the same role are joined with a blank line, since the adapter contract alternates; a
    later system or developer message becomes marked user text (late_system), joined with the user message next to
    it or a user turn of its own.
    """
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list", param="messages")
    if any(isinstance(m, dict) and (m.get("role") == "tool" or m.get("tool_calls")) for m in messages):
        require(capabilities, TOOLS_CAPABILITY, "messages")
        return tool_conversation(messages, capabilities, continuing)
    result = []
    for index, message in enumerate(messages):
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or message["role"] not in {"system", "developer", "user", "assistant"}):
            raise RequestError("Each message needs a role of system, user or assistant", param=f"messages[{index}].role")
        role = "system" if message["role"] == "developer" else message["role"]
        content = content_text(message.get("content"), index, capabilities)
        if role == "system" and any(m["role"] != "system" for m in result):
            role, content = "user", late_system(content)
        if result and result[-1]["role"] == role:
            result[-1]["content"] = joined(result[-1]["content"], content)
        else:
            result.append({"role": role, "content": content})
    turns = result[1:] if result[0]["role"] == "system" else result
    if not turns or turns[0]["role"] != "user" or turns[-1]["role"] != ("assistant" if continuing else "user"):
        raise RequestError("The conversation must start with a user message and end with the assistant message to "
                           "continue" if continuing else "The conversation must start and end with a user message",
                           param="messages")
    return result


def tool_conversation(messages, capabilities=(), continuing=False):
    """Preserve call ids and message boundaries; every call must have exactly one result. A later system or
    developer message becomes marked user text (late_system) that opens the user message after it, else ends the
    one before it, else is a user turn of its own. The conversation ends with user text or completed tool results,
    or with the assistant message the request is `continuing`."""
    result, pending, seen, note = [], set(), set(), None
    for index, message in enumerate(messages):
        param = f"messages[{index}]"
        if not isinstance(message, dict):
            raise RequestError("Each message must be an object", param=param)
        role = message.get("role")
        role = "system" if role == "developer" else role
        if not isinstance(role, str) or role not in {"system", "user", "assistant", "tool"}:
            raise RequestError("Invalid message role", param=param + ".role")
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
        # reasoning is vLLM's and OpenRouter's name for reasoning_content; their replies give null when there is none.
        for key in ("name", "reasoning_content", "reasoning"):
            if key in message and not (key == "reasoning" and message[key] is None):
                if not isinstance(message[key], str):
                    raise RequestError(key + " must be a string", param=param)
                if row.setdefault("reasoning_content" if key == "reasoning" else key, message[key]) != message[key]:
                    raise RequestError("reasoning must equal reasoning_content when both are given",
                                       param=param + ".reasoning")
        if role == "system" and any(m["role"] != "system" for m in result):
            note = late_system(row["content"]) if note is None else joined(note, late_system(row["content"]))
            ahead = messages[index + 1] if index + 1 < len(messages) else None
            if not isinstance(ahead, dict) or ahead.get("role") not in {"system", "developer", "user"}:
                if result[-1]["role"] == "user":
                    result[-1]["content"] = joined(result[-1]["content"], note)
                else:
                    result.append({"role": "user", "content": note})
                note = None
            continue
        if note is not None:
            row["content"], note = joined(note, row["content"]), None
        result.append(row)
    turns = [m for m in result if m["role"] != "system"]
    final = {"assistant"} if continuing else {"user", "tool"}
    if pending or not turns or turns[0]["role"] != "user" or turns[-1]["role"] not in final:
        raise RequestError("Conversation must start with user and end with " + ("the assistant message to continue"
                           if continuing else "user or completed tool results"), param="messages")
    return result


def thinking_template(thinking, capabilities, template, lenient=None):
    """(template variables, parameters ignored) of a chat request whose template_options are `template` once another
    dialect's `thinking` controls apply ({params, controls}, Api.completion_request): with chat-template-kwargs-v1
    they set the package's thinking variables in place of `template` (thinking_kwargs), listing the parameters they
    leave unused and, given crit serve --lenient's `lenient` list, theirs when the template cannot take them; without
    it they are ignored and listed whatever they hold."""
    if thinking is None:
        return template, set()
    if TEMPLATE_KWARGS_CAPABILITY not in capabilities:
        return template, set(thinking["params"])
    enabled, effort, unused = thinking["controls"]()
    refused = None if lenient is None else []
    return (thinking_kwargs(capabilities, enabled, effort, thinking["params"][0], refused),
            set(unused) | set(thinking["params"] if refused else ()))


def completion_prompt(value):
    """A /v1/completions prompt as the engine takes it: {"text": ...} for a string, {"ids": [...]} for token ids, each
    also as a list of one; a list of several prompts is refused, since a request completes one."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (str, list)):
        value = value[0]
    if isinstance(value, str) and value:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise RequestError("prompt must be valid Unicode text", param="prompt") from None
        return {"text": value}
    if isinstance(value, list) and value and all(type(t) is int and t >= 0 for t in value):
        return {"ids": value}
    if isinstance(value, list) and len(value) > 1 and all(isinstance(p, (str, list)) for p in value):
        raise RequestError("This engine completes one prompt per request", param="prompt")
    raise RequestError("prompt must be a non-empty string or list of token ids", param="prompt")


def output_limit(value, kv_tokens, fits=False):
    """(max_tokens, fit_output) for a request's output cap: 1 to kv_tokens - 1, by default DEFAULT_MAX_TOKENS or less
    to fit. A package declaring chat-fit-output-v1 (`fits`) fits a cap at or above its context: the engine stops where
    the prompt leaves no room (Ollama's num_predict -1 and -2, ollama.FILL_CONTEXT, ask for that), so the request
    sends kv_tokens - 1 with fit_output. A cap below the context stays exact."""
    if value is ollama.FILL_CONTEXT:
        value = kv_tokens if fits else None
    if value is None:
        value = min(DEFAULT_MAX_TOKENS, kv_tokens - 1)
    if type(value) is not int or value < 1 or (value >= kv_tokens and not fits):
        raise RequestError("max_tokens must be a positive integer" if fits else
                           f"max_tokens must be between 1 and {kv_tokens - 1}", param="max_tokens")
    return min(value, kv_tokens - 1), value >= kv_tokens


def stops(value):
    """The request's stop strings, within Core's StopMatcher limits (up to MAX_STOPS, each at most 256 bytes)."""
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or len(values) > MAX_STOPS:
        raise RequestError(f"stop must be a non-empty string or a list of up to {MAX_STOPS}", param="stop")
    try:
        StopMatcher(values)
    except TextError as exc:
        raise RequestError("stop: " + str(exc), param="stop") from None
    return values


@functools.lru_cache(maxsize=32)
def _grammar(kind, source):
    return compile_language(kind, source)


def schema_constraint(schema, name, field, param):
    """The json_schema constraint of `schema`, or a 400 on `param` naming `field` with the compiler's reason."""
    try:
        grammar = _grammar("schema", json.dumps(schema, ensure_ascii=False, allow_nan=False))
    except (UnsupportedConstraint, ValueError) as exc:
        raise RequestError(field + " is outside the supported JSON schema subset (docs/SERVING.md#structured-output): "
                           + str(exc), param=param, code="unsupported_schema") from None
    return {"type": "json_schema", "name": name, "schema": schema, "grammar": grammar}


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
    return schema_constraint(spec["schema"], name, "response_format.json_schema.schema", "response_format")


def structured_outputs(body):
    """(constraint, param) for vLLM's structured_outputs, or (None, None) without one: exactly one of regex (Core's
    byte regex over the whole reply), choice (the reply is one of the strings), grammar (Core's line EBNF) or json (a
    JSON schema, as response_format's), compiled with Core's constraint compiler. vLLM's earlier guided_regex,
    guided_choice, guided_grammar and guided_json name one the same way. Null fields count as absent.

    regex, choice and grammar reach the engine as {"type": "grammar", "language", "source", "grammar"}, the
    compiled crit-byte-grammar-v1 artifact applied like a schema's; json as response_format's json_schema.
    """
    given = [(key, body[key]) for key in GUIDED if body.get(key) is not None]
    value = body.get("structured_outputs")
    if value is not None:
        if not isinstance(value, dict):
            raise RequestError("structured_outputs must be an object", param="structured_outputs")
        fields = {key: source for key, source in value.items() if source is not None}
        unknown = sorted(set(fields) - set(STRUCTURED_OUTPUTS))
        if unknown:
            raise RequestError(f"structured_outputs.{unknown[0]} is not supported: give one of regex, choice, grammar "
                               "or json", param="structured_outputs." + unknown[0], code="unsupported_parameter")
        if not fields:
            raise RequestError("structured_outputs needs one of regex, choice, grammar or json",
                               param="structured_outputs")
        given += [("structured_outputs." + key, source) for key, source in fields.items()]
    if not given:
        return None, None
    if len(given) > 1:
        raise RequestError("A request takes one constraint, not " + " and ".join(param for param, _ in given),
                           param=given[-1][0])
    (param, source), = given
    key = GUIDED.get(param) or param[len("structured_outputs."):]
    if key == "json":
        if isinstance(source, str):
            try:
                source = json.loads(source)
            except ValueError:
                pass
        if not isinstance(source, dict):
            raise RequestError(param + " must be a JSON schema object", param=param)
        return schema_constraint(source, "response", param, param), param
    if key == "choice":
        if not isinstance(source, list) or not source or not all(isinstance(c, str) and c for c in source):
            raise RequestError(param + " must be a non-empty list of non-empty strings", param=param)
        # One rule of quoted literals, on one line: json.dumps escapes line breaks and every non-ASCII character.
        language, kind, text = "choice", "ebnf", "root ::= " + " | ".join(map(json.dumps, source))
    else:
        if not isinstance(source, str) or not source:
            raise RequestError(param + " must be a non-empty string", param=param)
        language = kind = "regex" if key == "regex" else "ebnf"
        text = source
    try:
        grammar = _grammar(kind, text)
    except (UnsupportedConstraint, ValueError) as exc:
        raise RequestError(f"{param} cannot be compiled (docs/SERVING.md#structured-output): {exc}", param=param,
                           code="unsupported_constraint") from None
    return {"type": "grammar", "language": language, "source": source, "grammar": grammar}, param


def packed_rows(scores):
    """Checked score rows (crit_api.engine.PackedScores) in a score request's "packed" format: base64 of little-endian
    arrays, the tokens and alternative tokens int32, the log-probabilities float32, the alternatives row by row
    (count x alternatives)."""
    def encoded(typecode, values):
        data = array.array(typecode, values)   # float64 to float32 rounds each value to nearest, as C casts do
        if sys.byteorder != "little":
            data.byteswap()
        return base64.b64encode(data.tobytes()).decode("ascii")
    return {"count": len(scores.tokens), "alternatives": scores.alternatives,
            "tokens": encoded("i", scores.tokens), "logprobs": encoded("f", scores.logprobs),
            "top_tokens": encoded("i", scores.top_tokens), "top_logprobs": encoded("f", scores.top_logprobs)}


def not_served(path):
    """The 404 message for a path under one of NOT_SERVED, or None."""
    for endpoint, hint in NOT_SERVED.items():
        if path == endpoint or path.startswith(endpoint + "/"):
            return f"{path} is not available on a crit engine, which serves chat" + hint
    return None


def conforms(constraint, text):
    """Whether a complete reply belongs to the constraint's language (Core's host checks)."""
    data = text.encode("utf-8")
    return json_object_matches(data) if constraint["type"] == "json_object" else matches(constraint["grammar"], data)


def key_digest(key):
    """An API key's SHA-256, which the server keeps and compares in place of the key."""
    return hashlib.sha256(key.encode()).digest()


class Api:
    """Parses OpenAI chat-completion requests and runs them on the engine. api_keys: the keys a request may carry
    (one string or several); without any, requests need none. With `lenient` (crit serve --lenient), the parameters
    crit does not apply are ignored and reported instead of refused."""
    def __init__(self, engine, model_name=None, api_keys=(), cors_origin=None, log=None, reasoning_alias=False,
                 lenient=False, ping_interval=0):
        self.engine = engine
        self.model = model_name or engine.model_id()
        self.key_digests = [key_digest(key) for key in ([api_keys] if isinstance(api_keys, str) else api_keys)]
        self.cors_origin = cors_origin
        # Seconds an OpenAI or Anthropic stream stays quiet before a ping (crit serve's --sse-ping-interval). 0 sends
        # none, as crit run eval and measure serve: Eval must see an engine that stalls.
        self.ping_interval = ping_interval
        self.created = int(time.time())
        self.log = log
        self.lenient = lenient
        self.logged = set()  # the ignored parameters --lenient has logged
        self.logging = threading.Lock()
        self.metrics = Metrics()
        self.reasoning_alias = reasoning_alias
        self.active = {}  # crit's request id -> (the client's X-Request-Id, the cancel event) of each request in flight
        self.active_lock = threading.Lock()

    def log_ignored(self, keys):
        """With --lenient, log each ignored parameter the first time a request has it."""
        if not (self.lenient and self.log):
            return
        with self.logging:
            new = sorted(set(keys) - self.logged)
            self.logged.update(new)
        for key in new:
            self.log(f"warning: ignoring request parameter {key}; responses list it in X-Crit-Ignored-Parameters")

    def accepts(self, key):
        """Whether `key` is one of the API keys. Every key is compared, in constant time, with digests of one length,
        so the time taken says nothing about the keys or which one matched."""
        digest = key_digest(key)
        return any([hmac.compare_digest(digest, known) for known in self.key_digests])

    def score_request(self, body):
        """A fixed-text score request (chat-score-v1, POST /v1/crit/score): {"model", "prompt_ids",
        "continuation_ids", "top_logprobs", "prompt_rows", "format"}, ids as the model's tokenizer numbers them;
        prompt_rows false (default true) returns no prompt rows; format "packed" (default "json") returns each part's
        rows as packed arrays (packed_rows)."""
        if not isinstance(body, dict):
            raise RequestError("The request body must be a JSON object")
        unknown = sorted(set(body) - {"model", "prompt_ids", "continuation_ids", "top_logprobs", "prompt_rows",
                                      "format"})
        if unknown:
            raise RequestError("Unknown score parameter " + unknown[0], param=unknown[0])
        candidate = self.candidate(body.get("model"))
        for key in ("prompt_ids", "continuation_ids"):
            value = body.get(key, [])
            if not isinstance(value, list) or not all(type(t) is int and t >= 0 for t in value):
                raise RequestError(f"{key} must be a list of token ids", param=key)
        if not body.get("prompt_ids"):
            raise RequestError("prompt_ids must hold at least one token", param="prompt_ids")
        top = body.get("top_logprobs", 0)
        if type(top) is not int or not 0 <= top <= TOP_LOGPROBS:
            raise RequestError(f"top_logprobs must be an integer from 0 to {TOP_LOGPROBS}", param="top_logprobs")
        prompt_rows = body.get("prompt_rows", True)
        if type(prompt_rows) is not bool:
            raise RequestError("prompt_rows must be a boolean", param="prompt_rows")
        if body.get("format", "json") not in SCORE_FORMATS:
            raise RequestError("format must be json or packed", param="format")
        return {"model": body.get("model") or self.model, "candidate": candidate, "prompt_ids": body["prompt_ids"],
                "continuation_ids": body.get("continuation_ids", []), "top_logprobs": top, "prompt_rows": prompt_rows,
                "format": body.get("format", "json")}

    @contextmanager
    def tracked(self, request_id, client_id, cancel):
        """A chat, score, load or tokenize request in flight, which POST /v1/crit/abort finds by crit's id or its
        client's."""
        with self.active_lock:
            self.active[request_id] = (client_id, cancel)
        try:
            yield
        finally:
            with self.active_lock:
                del self.active[request_id]

    def abort(self, body):
        """POST /v1/crit/abort: {"request_id"}, crit's id of a request in flight (its X-Request-Id) or the X-Request-Id
        its client sent, or {"all": true}. Sets each matching request's cancel event, as its client closing the
        connection does: a queued request leaves the queue, a running one ends its turn. Returns crit's ids found."""
        if not isinstance(body, dict) or set(body) not in ({"request_id"}, {"all"}):
            raise RequestError('The abort request is {"request_id": ID} or {"all": true}')
        if body.get("all", True) is not True:
            raise RequestError("all must be true", param="all")
        wanted = body.get("request_id")
        if "request_id" in body and (not isinstance(wanted, str) or not wanted):
            raise RequestError("request_id must be a non-empty string", param="request_id")
        with self.active_lock:
            found = [ident for ident, (client, _) in self.active.items() if wanted is None or wanted in (ident, client)]
            for ident in found:
                self.active[ident][1].set()
        return {"object": "crit.abort", "found": bool(found), "request_ids": found}

    def tokenize_request(self, body, path):
        """(candidate, operation, fields, ignored) of a tokenize request (chat-tokenize-v1, a TOKENIZE_PATHS path):
        /tokenize takes "prompt" and "add_special_tokens" (default true), or "messages" with the other fields a chat
        request's prompt comes from (RENDERED: the tool options, chat_template_kwargs, reasoning_effort, and
        continue_final_message with add_generation_prompt); /detokenize "tokens"; /apply-template "messages" and
        those fields; /v1/messages/count_tokens a Messages request's model, system, messages, tools, tool_choice and
        thinking, a final assistant message continued as prefill. Messages, tools, continuation and template
        variables are validated as a chat request's (completion_request), so their prompt is the one a reply to them
        prefills. "ignored" lists, as a chat request's does, the thinking controls the package does not apply and,
        with --lenient, what crit does not apply."""
        if not isinstance(body, dict):
            raise RequestError("The request body must be a JSON object")
        candidate = self.candidate(body.get("model"))
        capabilities = self.engine.spec(candidate)["capabilities"]
        if TOKENIZE_CAPABILITY not in capabilities:
            raise RequestError(f"{path} needs the engine's tokenizer: its package does not declare "
                               f"{TOKENIZE_CAPABILITY}", 404, code="not_supported")
        ignored, thinking = [], None
        if path == "/v1/messages/count_tokens":
            body, ignored, thinking = anthropic.count_body(body, self.lenient)
        allowed = {"/tokenize": RENDERED | {"prompt", "add_special_tokens"}, "/detokenize": {"model", "tokens"}}
        ignored += unknown_parameters(body, allowed.get(path, RENDERED), self.lenient)
        if path == "/detokenize":
            tokens = body.get("tokens")
            if not isinstance(tokens, list) or not all(type(t) is int and 0 <= t < 2**31 for t in tokens):
                raise RequestError("tokens must be a list of token ids", param="tokens")
            return candidate, "detokenize", {"tokens": tokens}, ignored
        special = body.get("add_special_tokens")
        if special is not None and type(special) is not bool:
            raise RequestError("add_special_tokens must be a boolean", param="add_special_tokens")
        if "prompt" in body:
            if not isinstance(body["prompt"], str):
                raise RequestError("prompt must be a string", param="prompt")
            if set(body) & (RENDERED - {"model"}):
                raise RequestError("A prompt is tokenized as it is: send messages, tools and template variables "
                                   "without it", param="prompt")
            fields = {"prompt": body["prompt"], "add_special_tokens": special is not False}
            return candidate, "tokenize", fields, ignored
        if special:
            raise RequestError("add_special_tokens applies to a prompt: a conversation's special tokens come from the "
                               "chat template", param="add_special_tokens")
        lenient = [] if self.lenient else None
        tools = tool_options(body, capabilities)
        template = template_options(body, capabilities, lenient)
        continuing = continuation(body, capabilities)
        fields = {"messages": conversation(body.get("messages"), capabilities, continuing), **tools}
        if continuing and tools:
            raise RequestError("Continuing the final assistant message cannot combine tools", param="tools")
        template, unapplied = thinking_template(thinking, capabilities, template, lenient)
        if template:
            fields["template_kwargs"] = template
        if continuing:
            fields["continue_final_message"] = True
        return candidate, "render", fields, sorted(set(ignored) | unapplied | set(lenient or ()))

    def completion_request(self, body, thinking=None):
        """The validated request for an OpenAI chat body. `thinking`, another dialect's thinking controls ({params,
        controls}: the dialect's parameters that carry them, and a function that validates their values and returns
        enabled, effort and the parameters it never applies), sets the package's chat-template variables
        (thinking_kwargs); without chat-template-kwargs-v1 its params are ignored and listed whatever they hold.
        Its "ignored" lists the client metadata and, with --lenient, what crit does not apply: unknown keys
        (stream_options' too), HINTS, sampling fields the package does not apply once their values are checked, and
        thinking controls and template variables its chat template cannot take (template_options)."""
        if not isinstance(body, dict):
            raise RequestError("The request body must be a JSON object")
        lenient = [] if self.lenient else None
        for key, feature in UNSUPPORTED.items():
            if key in body and body[key] is not None:
                if lenient is not None and key in HINTS:
                    lenient.append(key)
                    continue
                raise RequestError(f"This engine does not support {feature} yet", param=key, code="unsupported_parameter")
        if body.get("n") is not None and (type(body["n"]) is not int or body["n"] != 1):
            raise RequestError("This engine returns one choice per request", param="n")
        if "stream" in body and type(body["stream"]) is not bool:
            raise RequestError("stream must be a boolean", param="stream")
        # vLLM's extension: lower values leave the queue for the engine first (Admission.admit).
        if body.get("priority") is not None and type(body["priority"]) is not int:
            raise RequestError("priority must be an integer", param="priority")
        if type(body.get("include_stop_str_in_output", False)) is not bool:
            raise RequestError("include_stop_str_in_output must be a boolean", param="include_stop_str_in_output")
        candidate = self.candidate(body.get("model"))
        spec = self.engine.spec(candidate)
        capabilities = spec["capabilities"]
        sampling = sampling_options(body, capabilities, lenient)
        logprobs = logprob_options(body, capabilities)
        if logprobs is not None and body.get("stop"):
            # The turn is cancelled where a stop string completes, before the engine reports its tokens.
            raise RequestError("Stop strings cannot be combined with logprobs", param="stop")
        tools = tool_options(body, capabilities)
        template = template_options(body, capabilities, lenient)
        extension = crit_options(body.get("crit"), capabilities)
        constraint, param = response_format(body.get("response_format")), "response_format"
        structured, field = structured_outputs(body)
        if structured is not None:
            if constraint is not None:
                raise RequestError("A request takes one constraint: response_format or structured_outputs",
                                   param=field)
            constraint, param = structured, field
        if constraint:
            needed = [CONSTRAINTS_CAPABILITY] + ([GRAMMAR_CAPABILITY] if constraint["type"] == "grammar" else [])
            missing = next((c for c in needed if c not in capabilities), None)
            if missing:
                applied = f"response_format {constraint['type']}" if param == "response_format" else param
                raise RequestError(f"This engine cannot apply {applied}: its package does not declare {missing}",
                                   param=param, code="unsupported_parameter")
        continuing = continuation(body, capabilities)
        messages = conversation(body.get("messages"), capabilities, continuing)
        if tools and tools["tool_choice"] != "none":
            if constraint:
                raise RequestError(param + " cannot be combined with tools unless tool_choice is none", param=param)
            if body.get("stop"):
                # A stop ends the reply in its text: the calls finished before it come from the ordered tool stream.
                require(capabilities, TOOL_STREAM_CAPABILITY, "stop")
        if continuing and (tools or constraint):
            # Calls and constraints apply from the start of a reply, not from inside the message it continues.
            raise RequestError("Continuing the final assistant message cannot combine tools, response_format or "
                               "structured_outputs", param="tools" if tools else param)
        limit, fit = output_limit(body.get("max_completion_tokens", body.get("max_tokens")),
                                  spec["limits"]["kv_tokens"], FIT_OUTPUT_CAPABILITY in capabilities)
        options = body.get("stream_options") or {}
        if not isinstance(options, dict):
            raise RequestError("stream_options must be an object", param="stream_options")
        unknown = unknown_parameters(body, IGNORED | set(UNSUPPORTED) | KNOWN, self.lenient)
        unknown += unknown_parameters(options, {"include_usage"}, self.lenient, prefix="stream_options.")
        ignored = {k for k in body if k in IGNORED} | set(unknown)
        template, unapplied = thinking_template(thinking, capabilities, template, lenient)
        ignored = sorted(ignored | unapplied | set(lenient or ()))
        return {"candidate": candidate, "model": self.model if candidate == self.engine.default else self.engine.model_id(candidate),
                "messages": messages, "max_tokens": limit, "fit_output": fit, "stream": bool(body.get("stream")),
                "include_usage": bool(options.get("include_usage")), "stop": stops(body.get("stop")),
                "include_stop": body.get("include_stop_str_in_output", False), "response_format": constraint,
                "sampling": sampling, "tool_options": tools, "logprobs": logprobs, "decode": extension["decode"],
                "timeout": extension["timeout"], "template_kwargs": template, "priority": body.get("priority") or 0,
                "continue_final_message": continuing, "ignored": ignored}

    def prompt_request(self, body, thinking=None, param="prompt"):
        """A raw-prompt completion (completion-v1): POST /v1/completions, or an Ollama /api/generate with raw true as
        ollama.generate_body translates it (param "raw", the field that asked for it). One prompt, text or token ids,
        continued without the chat template; logprobs n gives each token n alternatives, and echo returns the prompt
        before the reply, with logprobs its rows from a chat-score-v1 engine's score path. Without completion-v1,
        /v1/completions is a 404 not_supported and Ollama's raw a 400. max_tokens, stop, include_stop_str_in_output,
        priority, crit and --lenient apply as to a chat request (completion_request); `thinking`, another dialect's
        thinking controls, sets template variables a raw prompt never renders, so its params are ignored and listed."""
        if not isinstance(body, dict):
            raise RequestError("The request body must be a JSON object")
        candidate = self.candidate(body.get("model"))
        spec = self.engine.spec(candidate)
        capabilities = spec["capabilities"]
        if COMPLETION_CAPABILITY not in capabilities:
            message = ("This engine renders the model's chat template: raw prompts need a package declaring "
                       + COMPLETION_CAPABILITY)
            if param == "prompt":
                raise RequestError(message + "; use /v1/chat/completions", 404, code="not_supported")
            raise RequestError(message, param=param, code="unsupported_parameter")
        if body.get("suffix") is not None:
            raise RequestError("suffix (fill-in-the-middle) is not supported yet", param="suffix",
                               code="unsupported_parameter")
        lenient = [] if self.lenient else None
        for key in ("n", "best_of"):
            if body.get(key) is not None and (type(body[key]) is not int or body[key] != 1):
                raise RequestError("This engine returns one choice per request", param=key)
        for key in ("stream", "echo"):
            if key in body and type(body[key]) is not bool:
                raise RequestError(key + " must be a boolean", param=key)
        if body.get("priority") is not None and type(body["priority"]) is not int:
            raise RequestError("priority must be an integer", param="priority")
        if type(body.get("include_stop_str_in_output", False)) is not bool:
            raise RequestError("include_stop_str_in_output must be a boolean", param="include_stop_str_in_output")
        prompt = completion_prompt(body.get("prompt"))
        sampling = sampling_options(body, capabilities, lenient)
        logprobs, echo = body.get("logprobs"), body.get("echo", False)
        if logprobs is not None:
            if type(logprobs) is not int or not 0 <= logprobs <= TOP_LOGPROBS:
                raise RequestError(f"logprobs must be an integer from 0 to {TOP_LOGPROBS}", param="logprobs")
            require(capabilities, LOGPROBS_CAPABILITY, "logprobs")
            if body.get("stop"):
                raise RequestError("Stop strings cannot be combined with logprobs", param="stop")
        if echo and (body.get("stream") or body.get("stop")):
            # The prompt comes back with the engine's final record: after a stream's text, never past a stop string.
            raise RequestError("echo cannot be combined with stream or stop", param="echo")
        if echo and logprobs is not None:
            require(capabilities, SCORE_CAPABILITY, "echo")
        extension = crit_options(body.get("crit"), capabilities)
        limit, fit = output_limit(body.get("max_tokens"), spec["limits"]["kv_tokens"],
                                  FIT_OUTPUT_CAPABILITY in capabilities)
        options = body.get("stream_options") or {}
        if not isinstance(options, dict):
            raise RequestError("stream_options must be an object", param="stream_options")
        unknown = unknown_parameters(body, IGNORED | COMPLETION_KNOWN, self.lenient)
        unknown += unknown_parameters(options, {"include_usage"}, self.lenient, prefix="stream_options.")
        ignored = {k for k in body if k in IGNORED} | set(unknown) | set(thinking["params"] if thinking else ())
        return {"candidate": candidate, "model": self.model if candidate == self.engine.default else self.engine.model_id(candidate),
                "prompt": prompt, "echo": echo, "max_tokens": limit, "fit_output": fit, "stream": bool(body.get("stream")),
                "include_usage": bool(options.get("include_usage")), "stop": stops(body.get("stop")),
                "include_stop": body.get("include_stop_str_in_output", False), "response_format": None,
                "sampling": sampling, "tool_options": {}, "logprobs": logprobs, "decode": extension["decode"],
                "timeout": extension["timeout"], "priority": body.get("priority") or 0,
                "ignored": sorted(ignored | set(lenient or ()))}

    def generate_request(self, body, thinking=None):
        """An Ollama /api/generate request as ollama.generate_body translates it: a raw prompt (raw true) is a
        completion-v1 request, any other a chat turn."""
        if "prompt" in body:
            return self.prompt_request(body, thinking, "raw")
        return self.completion_request(body, thinking)

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
        return [self.card(candidate) for candidate in self.engine.candidates()]

    def card(self, candidate):
        """The /v1/models card of an engine.candidates() entry: its display_name (a catalog's) or label as display_name,
        its context as context_length and as max_model_len, vLLM's name, which LiteLLM reads, and the limits a request
        to it is held to."""
        default = candidate["candidate_id"] == self.engine.default
        spec = self.engine.spec(candidate["candidate_id"])
        context = spec["limits"]["kv_tokens"]
        return {"id": self.model if default else self.engine.model_id(candidate["candidate_id"]),
                "object": "model", "created": self.created, "owned_by": "crit", "label": candidate["label"],
                "display_name": candidate.get("display_name") or candidate["label"],
                "context_length": context, "max_model_len": context,
                "default": default, "loaded": self.engine.loaded(candidate["candidate_id"]),
                "capabilities": spec["capabilities"],
                "supported_sampling_parameters": sorted(k for k in SAMPLING if sampling_capability(k)
                                                        in spec["capabilities"]),
                "limits": {"context_length": context, "max_output_tokens": context - 1, "max_body_bytes": MAX_BODY,
                           "max_stops": MAX_STOPS, "max_stop_bytes": MAX_STOP_BYTES, "max_tools": MAX_TOOLS,
                           "max_top_logprobs": TOP_LOGPROBS, "max_reply_characters": MAX_CONTENT},
                # What Eval's headroom and exactness read, while this candidate's engine is loaded.
                "engine_facts": self.engine.facts(candidate["candidate_id"])}

    def ollama_card(self, candidate, status=None):
        """The /api/tags entry of an engine.candidates() entry, its digest the package's; with the engine's status,
        its /api/ps entry."""
        card, digest = self.card(candidate), self.engine.spec(candidate["candidate_id"])["package_sha256"]
        if status is None:
            return ollama.model_card(card, self.created, digest)
        return ollama.loaded_card(card, self.created, digest, status, self.engine.unloads_at())

    def usage(self, measured):
        """OpenAI usage; prompt_tokens_details.cached_tokens counts the prompt tokens the engine did not compute
        (its telemetry's prompt_tokens minus prefill_tokens), when it reports them."""
        if measured is None:
            return None
        usage = {"prompt_tokens": measured["prompt_tokens"], "completion_tokens": measured["output_tokens"],
                 "total_tokens": measured["prompt_tokens"] + measured["output_tokens"]}
        if measured.get("prefill_tokens") is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": measured["prompt_tokens"] - measured["prefill_tokens"]}
        return usage

    def reasoning_fields(self, text):
        """Reasoning apart from the reply as a message or delta carries it: reasoning_content, and with
        reasoning_alias the same text as reasoning too, the name vLLM and OpenRouter use."""
        return {"reasoning_content": text, "reasoning": text} if self.reasoning_alias else {"reasoning_content": text}

    def finish_reason(self, stopped, measured, limit):
        if stopped or not measured:
            return "stop"
        # A fitted request ends at the limit the engine applied (its telemetry's max_output_tokens).
        return "length" if measured["output_tokens"] >= measured.get("max_output_tokens", limit) else "stop"

    def run(self, request, emit, cancel, emit_tools=None, emit_reasoning=None, on_started=None, on_admitted=None,
            timing=None):
        """Generate; emit receives text, emit_tools indexed deltas, emit_reasoning reasoning text (apart from the
        reply). Returns (finish_reason, measured, calls, reasoning, logprobs, stopped), logprobs being OpenAI's
        logprobs content when the request asked for it and stopped the stop string that ended the reply, or None.
        on_admitted() runs once the engine takes the request from its queue (ServedEngine.generate).
        A raw prompt (prompt_request) with echo emits its prompt first, and its logprobs start with the prompt's.
        Stop strings follow
        Core's rule (crit_core.text.StopMatcher): the first stop to complete wins, and only text that could still
        begin a stop is held back. They are matched in the reply's text, never in its calls: a reply a stop ends
        keeps the calls streamed before it, which the tool stream saw whole before any later text. With include_stop
        (include_stop_str_in_output) the stop ends the emitted text.
        `timing` (Metrics.begin()) is marked dispatched when the engine starts the request."""
        scanner = StopMatcher(request["stop"])
        streamed, begun = [], []  # reasoning streamed; calls as they streamed before any stop

        def started(deadline):
            if timing is not None:
                self.metrics.dispatched(timing)
            if on_started:
                on_started(deadline)

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

        def called(deltas):
            if scanner.stopped is not None:
                return  # Written after the stop: not part of the reply.
            for delta in deltas:
                if delta["index"] == len(begun):
                    begun.append({"id": delta["id"], "type": "function",
                                  "function": {"name": delta["function"]["name"], "arguments": ""}})
                begun[delta["index"]]["function"]["arguments"] += delta["function"].get("arguments", "")
            if emit_tools:
                emit_tools(deltas)

        def finished():
            if not begun:
                return []
            try:
                return validate_tool_reply(begun, dict(request["tool_options"], messages=request["messages"]))
            except ValueError as exc:
                raise EngineError("The engine streamed an invalid tool call: " + str(exc)) from exc
        try:
            tool_incremental = ((request["stream"] or request["stop"]) and request["tool_options"] and
                                TOOL_STREAM_CAPABILITY in self.engine.spec(request["candidate"])["capabilities"])
            incremental = (request["stream"] or request["stop"]) and (not request["tool_options"] or tool_incremental)
            status, assistant, measured = self.engine.generate(request.get("messages"), request["max_tokens"], cancel,
                                                               on_text=progress if incremental else None,
                                                               candidate_id=request["candidate"],
                                                               response_format=request.get("response_format"),
                                                               sampling=request["sampling"], tool_options=request["tool_options"],
                                                               on_tools=called if tool_incremental else None,
                                                               on_reasoning=reasoned if request["stream"] else None,
                                                               on_started=started, logprobs=request.get("logprobs"),
                                                               decode=request.get("decode"),
                                                               template_kwargs=request.get("template_kwargs"),
                                                               fit_output=request.get("fit_output", False),
                                                               on_admitted=on_admitted,
                                                               priority=request.get("priority", 0),
                                                               request_id=request.get("request_id"),
                                                               timeout=request.get("timeout"),
                                                               continue_final_message=request.get(
                                                                   "continue_final_message", False),
                                                               prompt=request.get("prompt"),
                                                               echo=request.get("echo", False))
        except (ValueError, OSError) as exc:
            raise EngineError(str(exc)) from exc
        if status == "completed":
            calls = assistant.get("tool_calls") if isinstance(assistant, dict) else None
            reasoning = assistant.get("reasoning", "") if isinstance(assistant, dict) else ""
            logprobs = assistant.get("logprobs") if isinstance(assistant, dict) else None
            echo = assistant.get("prompt") if isinstance(assistant, dict) else None
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
            if scanner.stopped is not None and request.get("include_stop"):
                piece += scanner.stopped
            if echo:
                emit(echo)  # an echo neither streams nor stops, so none of the reply has gone out
            if piece:
                emit(piece)
            if scanner.stopped is not None:
                calls = finished()
            if calls:
                if emit_tools and not tool_incremental:
                    emit_tools([dict(call, index=i) for i, call in enumerate(calls)])
                # A call the output limit cut keeps the arguments written so far, and the reply finishes as cut.
                return ("tool_calls" if complete_arguments(calls[-1]) else "length"), measured, calls, reasoning, logprobs, None
            return finish, measured, [], reasoning, logprobs, scanner.stopped
        if scanner.stopped is not None:
            if request.get("include_stop"):
                emit(scanner.stopped)
            calls = finished()
            if calls:
                return "tool_calls", None, calls, "".join(streamed), None, None
            return "stop", None, [], "".join(streamed), None, scanner.stopped
        if status == "timed_out":
            raise EngineError("The engine did not finish within the request timeout", 504)
        raise EngineError("Request cancelled", 499)

    def telemetry(self, measured):
        if measured is None:
            return None
        return {key: measured[key] for key in ("prompt_tokens", "output_tokens", "prefill_tokens", "prefill_ms",
                                               "decode_tokens", "decode_ms", "prefill_tps", "decode_tps", "ttft_ms",
                                               "tokens_per_round", "accepted", "verify_ms", "draft_ms") if key in measured}


class Handler(BaseHTTPRequestHandler):
    server_version = "crit-serve"
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True  # each streamed chunk leaves at once, not after the client's delayed ACK

    @property
    def api(self):
        return self.server.api

    def handle_one_request(self):
        # Every request on a kept-alive connection gets its own id, before http.server can answer a malformed one.
        self.request_id, self.client_request_id = uuid.uuid4().hex, None
        super().handle_one_request()

    def parse_request(self):
        """The client's X-Request-Id, when it is 1 to 128 visible ASCII characters, is logged and found by an abort;
        crit's own id names the request's records."""
        if not super().parse_request():
            return False
        supplied = self.headers.get("X-Request-Id", "")
        self.client_request_id = supplied if re.fullmatch(r"[\x21-\x7e]{1,128}", supplied) else None
        return True

    def send_response(self, code, message=None):
        """Every response carries crit's id of its request, which error bodies repeat and a failed request's records
        are named by."""
        super().send_response(code, message)
        self.send_header("X-Request-Id", self.request_id)

    def log_message(self, format, *args):
        if self.api.log:
            client = " client_request_id=" + self.client_request_id if self.client_request_id else ""
            self.api.log("%s %s request_id=%s%s" % (self.address_string(), format % args, self.request_id, client))

    def send_json(self, status, value, headers=()):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.cors()
        for name, content in headers:
            self.send_header(name, content)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def error(self, status, message, kind="invalid_request_error", param=None, code=None, headers=()):
        self.send_json(status, {"error": {"message": message, "type": kind, "param": param, "code": code},
                                "request_id": self.request_id}, headers)

    def cors(self):
        if self.api.cors_origin:
            self.send_header("Access-Control-Allow-Origin", self.api.cors_origin)
            self.send_header("Access-Control-Allow-Headers",
                             "Authorization, Content-Type, x-api-key, anthropic-version, X-Request-Id")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Expose-Headers", "X-Request-Id")

    def send_text(self, status, text, content_type="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def dialect(self):
        """The API dialect a path belongs to: anthropic (/v1/messages), responses (/v1/responses, OpenAI's errors),
        ollama (/api/...), else openai."""
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.startswith("/v1/messages"):
            return "anthropic"
        if path == "/v1/responses" or path.startswith("/v1/responses/"):
            return "responses"
        if path.startswith("/api/") or path == "":
            return "ollama"
        return "openai"

    def fail(self, status, message, kind="invalid_request_error", param=None, code=None, headers=()):
        """An error in the shape of the dialect the request used."""
        dialect = self.dialect()
        if dialect == "anthropic":
            return self.send_json(status, anthropic.error_body(status, message, self.request_id), headers)
        if dialect == "ollama":
            return self.send_json(status, ollama.error_body(message, self.request_id), headers)
        return self.error(status, message, kind, param, code, headers)

    def text_completion(self):
        """Whether the request is a raw-prompt completion (POST /v1/completions), whose errors name the prompt."""
        return self.path.split("?", 1)[0].rstrip("/") == "/v1/completions"

    def error_words(self, exc):
        """(message, code) of an engine error in the request's dialect: a context overflow in the API's own words,
        which clients match to compact the conversation."""
        if not isinstance(exc, ContextLengthError):
            return str(exc), None
        dialect = self.dialect()
        if dialect == "anthropic":
            return anthropic.context_message(exc.prompt_tokens, exc.output_tokens, exc.capacity), None
        if dialect == "ollama":
            return str(exc), None
        if self.text_completion():
            return (f"This model's maximum context length is {exc.capacity} tokens, however you requested "
                    f"{exc.prompt_tokens + exc.output_tokens} tokens ({exc.prompt_tokens} in your prompt; "
                    f"{exc.output_tokens} for the completion). Please reduce your prompt; or completion length.",
                    "context_length_exceeded")
        return (f"This model's maximum context length is {exc.capacity} tokens. However, you requested "
                f"{exc.prompt_tokens + exc.output_tokens} tokens ({exc.prompt_tokens} in the messages, "
                f"{exc.output_tokens} in the completion). Please reduce the length of the messages or completion.",
                "context_length_exceeded")

    def engine_error(self, exc):
        """An engine error as a response in the request's dialect."""
        if isinstance(exc, Overloaded):
            return self.busy(exc)
        message, code = self.error_words(exc)
        return self.fail(exc.status, message, "server_error" if exc.status >= 500 else "invalid_request_error",
                         ("prompt" if self.text_completion() else "messages") if code else None, code)

    def busy(self, exc):
        """A request the queue refused, as each dialect says the server is busy: OpenAI's rate limit (429, its type
        requests and code rate_limit_exceeded), Anthropic's overloaded_error (529) and Ollama's 503, with Retry-After,
        which clients wait before retrying."""
        status = {"anthropic": 529, "ollama": 503}.get(self.dialect(), 429)
        return self.fail(status, str(exc), "requests", code="rate_limit_exceeded",
                         headers=[("Retry-After", str(RETRY_AFTER))])

    def unknown(self, path):
        """404 for a path crit does not serve: not_supported, naming the endpoint, for the ones clients look for beside
        chat, else not_found."""
        message = not_served(path)
        if message:
            return self.fail(404, message, code="not_supported")
        return self.fail(404, "Unknown endpoint " + path, code="not_found")

    def authorized(self):
        """With API keys configured, every request carries one as a Bearer token or, as Anthropic clients send it,
        in x-api-key."""
        if not self.api.key_digests:
            return True
        supplied = self.headers.get("Authorization", "")
        if supplied.startswith("Bearer ") and self.api.accepts(supplied[7:]):
            return True
        key = self.headers.get("x-api-key", "")
        if key and self.api.accepts(key):
            return True
        self.fail(401, "Missing or invalid API key", "authentication_error", code="invalid_api_key")
        return False

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        """GET's status and headers without the body, for clients that probe a path before using it."""
        self.do_GET()

    def do_DELETE(self):
        """Any other method: 405 with the methods a known path takes, else 404, in the path's dialect."""
        path = self.path.split("?", 1)[0].rstrip("/")
        if self.headers.get("Content-Length", "0") != "0" or self.headers.get("Transfer-Encoding"):
            self.discard_body()
        if path not in {"", "/health", "/api/version"} and not self.authorized():
            return None
        if path.startswith("/v1/responses/"):
            return self.fail(404, responses.STATELESS, code="not_supported")
        allowed = "GET, HEAD" if path in GET_PATHS or path.startswith("/v1/models/") else "POST" if path in POST_PATHS else None
        if allowed is None:
            return self.unknown(path)
        return self.fail(405, f"{self.command} is not allowed on {path}; it takes {allowed}", code="method_not_allowed",
                         headers=[("Allow", allowed + ", OPTIONS")])

    do_PUT = do_PATCH = do_DELETE

    def discard_body(self):
        """Read and drop a body the answer will not use, so the connection stays usable; one that cannot be read
        closes it. Closing with unread bytes would reset the connection on Windows before the client reads the answer."""
        try:
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                self.read_chunks()
                return
            length = int(self.headers.get("Content-Length", "0"))
            if 0 <= length <= MAX_BODY and len(self.rfile.read(length)) == length:
                return
        except (RequestError, ValueError):
            pass
        self.close_connection = True  # An unread body must not be parsed as the next request.

    def read_body(self, empty=False):
        """The request body, sent with Content-Length or Transfer-Encoding: chunked, or None once the error is sent;
        with `empty`, the request may send none (b""), but a malformed or negative Content-Length is still an error."""
        try:
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                raw = self.read_chunks()
            else:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                if length > MAX_BODY:
                    raise RequestError(BODY_SIZE, 413)
                if length < 0:
                    raise RequestError(BODY_SIZE)
                raw = self.rfile.read(length) if length > 0 else b""
            if not raw and not empty:
                raise RequestError(BODY_SIZE)
        except RequestError as exc:
            self.close_connection = True  # An unread body must not be parsed as the next request.
            self.fail(exc.status, str(exc))
            return None
        return raw

    def read_chunks(self):
        """A Transfer-Encoding: chunked body of at most MAX_BODY bytes; its trailer fields are read and dropped."""
        raw = bytearray()
        while True:
            line = re.fullmatch(rb"([0-9A-Fa-f]{1,8})(;[^\r\n]*)?\r?\n", self.rfile.readline(1024))
            if line is None:
                raise RequestError("The chunked request body is malformed")
            size = int(line.group(1), 16)
            if size == 0:
                break
            if len(raw) + size > MAX_BODY:
                raise RequestError(BODY_SIZE, 413)
            chunk = self.rfile.read(size)
            if len(chunk) != size or self.rfile.readline(3) not in (b"\r\n", b"\n"):
                raise RequestError("The chunked request body is malformed")
            raw += chunk
        for _ in range(64):
            if self.rfile.readline(1024) in (b"\r\n", b"\n", b""):
                return bytes(raw)
        raise RequestError("The chunked request body is malformed")

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            # ?ready=1: also 503 until an engine is loaded with no switch, reload or retire under way.
            status = self.api.engine.status()
            ok, ready = status.get("state") not in {"blocked", "interrupted"}, self.api.engine.ready(status)
            wanted = parse_qs(self.path.partition("?")[2]).get("ready", [""])[-1] in {"1", "true"}
            return self.send_json(200 if ok and (ready or not wanted) else 503,
                                  {"status": "ok" if ok else "unavailable", "ready": ready,
                                   "loaded": bool(status.get("engine")), "engine": status})
        if path == "":
            return self.send_text(200, "Ollama is running")  # what Ollama clients probe first
        if path == "/api/version":
            return self.send_json(200, {"version": ollama.OLLAMA_COMPAT_VERSION, "crit_version": __version__})
        if not self.authorized():
            return
        if path == "/v1/models":
            return self.send_json(200, {"object": "list", "data": self.api.models()})
        if path.startswith("/v1/models/"):
            card = next((c for c in self.api.models() if c["id"] == path[len("/v1/models/"):]), None)
            if card:
                return self.send_json(200, card)
            return self.error(404, "Unknown model", code="model_not_found")
        if path.startswith("/v1/responses/"):
            return self.error(404, responses.STATELESS, code="not_supported")
        if path == "/metrics":
            return self.send_text(200, self.api.metrics.render(self.api.engine), "text/plain; version=0.0.4; charset=utf-8")
        if path == "/api/tags":
            return self.send_json(200, {"models": [self.api.ollama_card(c) for c in self.api.engine.candidates()]})
        if path == "/api/ps":
            status = self.api.engine.status()
            loaded = [c for c in self.api.engine.candidates() if self.api.engine.loaded(c["candidate_id"], status)]
            return self.send_json(200, {"models": [self.api.ollama_card(c, status) for c in loaded]})
        self.unknown(path)

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.startswith("/v1/responses/"):
            return self.do_DELETE()  # The SDKs post .../cancel with no body, which read_body would refuse first.
        raw = self.read_body(empty=path in {"/v1/crit/load", "/v1/crit/unload"})
        if raw is None or not self.authorized():
            return
        if path == "/v1/crit/abort":
            try:
                return self.send_json(200, self.api.abort(json.loads(raw)))
            except (ValueError, UnicodeDecodeError) as exc:
                return self.error(400, "Invalid JSON body: " + str(exc))
            except RequestError as exc:
                return self.error(exc.status, str(exc), param=exc.param, code=exc.code)
        if path == "/v1/crit/score":
            try:
                request = self.api.score_request(json.loads(raw))
            except (ValueError, UnicodeDecodeError) as exc:
                self.api.metrics.count("crit", "rejected")
                return self.error(400, "Invalid JSON body: " + str(exc))
            except RequestError as exc:
                self.api.metrics.count("crit", self.outcome(exc))
                return self.error(exc.status, str(exc), param=exc.param, code=exc.code)
            with (ClientDisconnect(self.connection) as client,
                  self.api.tracked(self.request_id, self.client_request_id, client.cancel)):
                try:
                    self.scored(request, client.cancel)
                except OSError:
                    client.cancel.set()
                    self.close_connection = True
            if client.disconnected.is_set():
                self.close_connection = True
            return None
        if path in {"/v1/crit/load", "/v1/crit/unload"}:
            return self.crit_lifecycle(raw, path[len("/v1/crit/"):])
        if path == "/v1/messages":
            return self.dispatch(raw, anthropic.chat_body, self.anthropic_complete, self.anthropic_stream)
        if path == "/v1/responses":
            return self.dispatch(raw, responses.chat_body, self.responses_complete, self.responses_stream)
        if path == "/api/chat":
            return self.ollama_chat(raw)
        if path == "/api/generate":
            return self.ollama_chat(raw, generate=True)
        if path == "/api/show":
            return self.ollama_show(raw)
        if path in TOKENIZE_PATHS:
            return self.tokenized(raw, path)
        if path.startswith("/api/") and not_served(path) is None:
            return self.fail(404, f"{path} is not available on a crit engine: it serves /api/chat, /api/generate, /api/show, "
                                  "/api/tags, /api/ps and /api/version")
        if path == "/v1/completions":
            return self.dispatch(raw, None, self.text_complete, self.text_stream, validate=self.api.prompt_request)
        if path != "/v1/chat/completions":
            return self.unknown(path)
        self.dispatch(raw, None, self.complete, self.stream)

    def dispatch(self, raw, translate, complete, stream, validate=None, parsed=None, **extra):
        """Parse a request in its dialect (translate: (body, lenient) -> (OpenAI body, ignored, thinking controls),
        None for OpenAI itself; parsed: the body when the caller has parsed it), validate it (validate: (body,
        thinking) -> request, by default as every chat request is validated), then run it as a reply or a stream. The
        body as sent stays the request's source, which a reply may echo. Returns the request once it ran."""
        try:
            body = source = json.loads(raw) if parsed is None else parsed
            ignored, thinking = [], None
            if translate is not None:
                body, ignored, thinking = translate(body, self.api.lenient)
            request = (validate or self.api.completion_request)(body, thinking)
            request["ignored"] = sorted(set(request["ignored"]) | set(ignored))
            request["requested_model"] = body.get("model") if isinstance(body, dict) else None
            request["request_id"] = self.request_id
            request["source"] = source
        except (ValueError, UnicodeDecodeError) as exc:
            self.api.metrics.count(self.dialect(), "rejected")
            return self.fail(400, "Invalid JSON body: " + str(exc))
        except RequestError as exc:
            self.api.metrics.count(self.dialect(), self.outcome(exc))
            return self.fail(exc.status, str(exc), param=exc.param, code=exc.code)
        self.api.log_ignored(request["ignored"])
        with (ClientDisconnect(self.connection) as client,
              self.api.tracked(self.request_id, self.client_request_id, client.cancel)):
            try:
                (stream if request["stream"] else complete)(request, client.cancel, **extra)
            except OSError:
                # A response write can fail between transport polls.
                client.cancel.set()
                self.close_connection = True
        if client.disconnected.is_set():
            self.close_connection = True
        return request

    def crit_lifecycle(self, raw, action):
        """POST /v1/crit/load loads a model's engine ({"model"}, default the default model) as its next request would;
        POST /v1/crit/unload releases the loaded engine's weights (with "model", only that model's) once its requests
        drain. The body may be empty. Either answers with the engine's status after it."""
        try:
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                raise RequestError("The request body must be a JSON object")
            unknown = sorted(set(body) - {"model"})
            if unknown:
                raise RequestError("Unknown parameter " + unknown[0], param=unknown[0])
            model = body.get("model")
            candidate = self.api.candidate(model) if action == "load" or model is not None else None
        except (ValueError, UnicodeDecodeError) as exc:
            return self.error(400, "Invalid JSON body: " + str(exc))
        except RequestError as exc:
            return self.error(exc.status, str(exc), param=exc.param, code=exc.code)

        def answer(outcome):
            found = {"model": model or self.api.model} if action == "load" else {"unloaded": outcome}
            return {"object": "crit." + action, **found, "engine": self.api.engine.status()}
        self.lifecycle(action, candidate, answer)

    def lifecycle(self, action, candidate, answer):
        """Load `candidate`'s engine ("load") as its next request would, or unload the loaded engine ("unload"; with a
        candidate, only its own) once its requests drain, then reply answer(outcome): "completed" for a load, whether
        an engine was unloaded for an unload. Errors use the path's dialect. A load waits in the queue as a request
        does, so POST /v1/crit/abort cancels it as it cancels one."""
        with ExitStack() as stack:
            client = stack.enter_context(ClientDisconnect(self.connection))
            if action == "load":
                stack.enter_context(self.api.tracked(self.request_id, self.client_request_id, client.cancel))
            try:
                self.lifecycle_reply(action, candidate, answer, client.cancel)
            except OSError:
                client.cancel.set()
                self.close_connection = True
        if client.disconnected.is_set():
            self.close_connection = True

    def lifecycle_reply(self, action, candidate, answer, cancel):
        try:
            outcome = self.api.engine.load(cancel, candidate) if action == "load" else self.api.engine.unload(candidate)
        except EngineError as exc:
            return self.engine_error(exc)
        if outcome in {"cancelled", "timed_out"}:
            return self.fail(504 if outcome == "timed_out" else 499, "Request " + outcome.replace("_", " "),
                             "server_error")
        self.send_json(200, answer(outcome))

    def tokenized(self, raw, path):
        """A tokenize request (chat-tokenize-v1): on the candidate's loaded engine it runs beside any generation,
        never queued behind it; one that waits to load the candidate leaves when its client disconnects or POST
        /v1/crit/abort cancels it, and a full queue refuses it as it refuses a reply."""
        try:
            candidate, operation, fields, ignored = self.api.tokenize_request(json.loads(raw), path)
        except (ValueError, UnicodeDecodeError) as exc:
            return self.fail(400, "Invalid JSON body: " + str(exc))
        except RequestError as exc:
            return self.fail(exc.status, str(exc), param=exc.param, code=exc.code)
        self.api.log_ignored(ignored)
        with (ClientDisconnect(self.connection) as client,
              self.api.tracked(self.request_id, self.client_request_id, client.cancel)):
            try:
                self.tokens(path, candidate, operation, fields, ignored, client.cancel)
            except OSError:
                client.cancel.set()
                self.close_connection = True
        if client.disconnected.is_set():
            self.close_connection = True

    def tokens(self, path, candidate, operation, fields, ignored, cancel):
        try:
            found = self.api.engine.tokenize(operation, fields, cancel, candidate)
        except EngineError as exc:
            return self.engine_error(exc)
        if path == "/tokenize":
            value = {"tokens": found["tokens"], "count": len(found["tokens"]),
                     "max_model_len": self.api.engine.spec(candidate)["limits"]["kv_tokens"]}
        elif path == "/v1/messages/count_tokens":
            value = {"input_tokens": len(found["tokens"])}
        else:
            value = {"prompt": found["prompt"]}
        self.send_json(200, value, [("X-Crit-Ignored-Parameters", ",".join(ignored))] if ignored else [])

    def measure(self, dialect, timing, outcome, measured=None):
        self.api.metrics.end(dialect, outcome, measured, timing)

    @staticmethod
    def outcome(exc):
        return {499: "cancelled", 504: "timed_out"}.get(exc.status, "rejected" if exc.status < 500 else "error")

    def headers_for(self, request, measured=None):
        """The ignored parameters and, for a reply whose max_tokens crit serve fitted to the context, the limit the
        engine applied; a stream sends its headers before the engine counts the prompt."""
        headers = [("X-Crit-Ignored-Parameters", ",".join(request["ignored"]))] if request["ignored"] else []
        if measured and "max_output_tokens" in measured:
            headers.append(("X-Crit-Max-Tokens-Applied", str(measured["max_output_tokens"])))
        return headers

    def scored(self, request, cancel):
        timing = self.api.metrics.begin()
        try:
            status, scored, telemetry = self.api.engine.score(request["prompt_ids"], request["continuation_ids"],
                                                              request["top_logprobs"], cancel, request["candidate"],
                                                              prompt_rows=request["prompt_rows"],
                                                              packed=request["format"] == "packed",
                                                              request_id=self.request_id,
                                                              on_started=lambda _: self.api.metrics.dispatched(timing))
        except EngineError as exc:
            self.measure("crit", timing, self.outcome(exc))
            if isinstance(exc, Overloaded):
                return self.busy(exc)
            return self.error(exc.status, str(exc), "server_error" if exc.status >= 500 else "invalid_request_error")
        self.measure("crit", timing, status)
        if status != "completed":
            return self.error(504 if status == "timed_out" else 499, "Request " + status.replace("_", " "),
                              "server_error")
        if request["format"] == "packed":
            scored = {"format": "packed", **{part: packed_rows(scores) for part, scores in scored.items()}}
        self.send_json(200, {"object": "crit.score", "model": request["model"],
                             "system_fingerprint": self.api.engine.spec(request["candidate"])["package_sha256"][:16],
                             **scored, "crit_telemetry": telemetry})

    def complete(self, request, cancel):
        pieces = []
        timing = self.api.metrics.begin()
        try:
            finish, measured, calls, reasoning, logprobs, _ = self.api.run(request, pieces.append, cancel, timing=timing)
        except EngineError as exc:
            self.measure("openai", timing, self.outcome(exc))
            return self.engine_error(exc)
        self.measure("openai", timing, "completed", measured)
        value = {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                 "model": request["model"], "system_fingerprint": self.api.engine.spec(request["candidate"])["package_sha256"][:16],
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(pieces)},
                              "logprobs": None if logprobs is None else {"content": logprobs}, "finish_reason": finish}]}
        if calls:
            value["choices"][0]["message"].update(content="".join(pieces) or None, tool_calls=calls)
        if reasoning:
            value["choices"][0]["message"].update(self.api.reasoning_fields(reasoning))
        usage = self.api.usage(measured)
        if usage:
            value["usage"] = usage
        if measured:
            value["crit_telemetry"] = self.api.telemetry(measured)
        self.send_json(200, value, self.headers_for(request, measured))

    def start_stream(self, content_type, request, cancel):
        """A stream's status and headers, sent once the engine takes the request (Api.run's on_admitted), so a request
        refused while queued gets its own status instead."""
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.cors()
            for name, content in self.headers_for(request):
                self.send_header(name, content)
            self.end_headers()
        except OSError:
            cancel.set()  # The client went away: stop the engine.
        self.close_connection = True

    def stream(self, request, cancel):
        with StreamDelivery(self.connection, cancel, self.api.engine._stopping, self.api.engine.timeout) as delivery:
            previous, self.wfile = self.wfile, delivery
            try:
                self._stream(request, cancel, delivery)
            finally:
                self.wfile = previous

    def _stream(self, request, cancel, delivery):
        ident, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        fingerprint = self.api.engine.spec(request["candidate"])["package_sha256"][:16]
        opened, pings = False, ExitStack()

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

        def admitted():
            nonlocal opened
            opened = True
            self.start_stream("text/event-stream", request, cancel)
            event(chunk({"role": "assistant", "content": ""}))
            # Pings keep proxies and idle clients from closing the open stream while the engine has nothing to send.
            pings.enter_context(delivery.pinging(PING, self.api.ping_interval))

        timing = self.api.metrics.begin()
        try:
            with pings:  # they end before the closing events
                finish, measured, _, _, logprobs, _ = self.api.run(request, lambda piece: event(chunk({"content": piece})),
                    cancel, lambda deltas: event(chunk({"tool_calls": deltas})),
                    lambda piece: event(chunk(self.api.reasoning_fields(piece))), on_started=delivery.set_deadline,
                    on_admitted=admitted, timing=timing)
        except EngineError as exc:
            self.measure("openai", timing, self.outcome(exc))
            if not opened:
                return self.engine_error(exc)  # refused before the stream began, such as by a full queue
            message, code = self.error_words(exc)
            event({"error": {"message": message, "type": "server_error" if exc.status >= 500 else "invalid_request_error",
                             "param": "messages" if code else None, "code": code}, "request_id": self.request_id})
        else:
            self.measure("openai", timing, "completed", measured)
            # Text streams as the engine emits it; the reply's logprobs come from its final record, with the finish chunk.
            event(chunk({}, finish, logprobs=logprobs))
            if request["include_usage"]:
                event(chunk({}, usage=self.api.usage(measured), choices=False, telemetry=self.api.telemetry(measured)))
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass

    # OpenAI legacy completions of a raw prompt (completion-v1)

    def text_complete(self, request, cancel):
        pieces = []
        timing = self.api.metrics.begin()
        try:
            finish, measured, _, _, logprobs, _ = self.api.run(request, pieces.append, cancel, timing=timing)
        except EngineError as exc:
            self.measure("openai", timing, self.outcome(exc))
            return self.engine_error(exc)
        self.measure("openai", timing, "completed", measured)
        value = {"id": "cmpl-" + uuid.uuid4().hex, "object": "text_completion", "created": int(time.time()),
                 "model": request["model"], "system_fingerprint": self.api.engine.spec(request["candidate"])["package_sha256"][:16],
                 "choices": [{"text": "".join(pieces), "index": 0,
                              "logprobs": None if logprobs is None else completion_logprobs(logprobs),
                              "finish_reason": finish}]}
        usage = self.api.usage(measured)
        if usage:
            value["usage"] = usage
        if measured:
            value["crit_telemetry"] = self.api.telemetry(measured)
        self.send_json(200, value, self.headers_for(request, measured))

    def text_stream(self, request, cancel):
        with StreamDelivery(self.connection, cancel, self.api.engine._stopping, self.api.engine.timeout) as delivery:
            previous, self.wfile = self.wfile, delivery
            try:
                self._text_stream(request, cancel, delivery)
            finally:
                self.wfile = previous

    def _text_stream(self, request, cancel, delivery):
        ident, created = "cmpl-" + uuid.uuid4().hex, int(time.time())
        fingerprint = self.api.engine.spec(request["candidate"])["package_sha256"][:16]
        opened, pings = False, ExitStack()

        def event(value):
            try:
                self.wfile.write(b"data: " + json.dumps(value).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            except OSError:
                cancel.set()  # The client went away: stop the engine.

        def chunk(text, finish=None, usage=None, choices=True, logprobs=None, telemetry=None):
            value = {"id": ident, "object": "text_completion", "created": created, "model": request["model"],
                     "system_fingerprint": fingerprint,
                     "choices": [{"text": text, "index": 0, "logprobs": logprobs, "finish_reason": finish}] if choices else []}
            if usage is not None or (request["include_usage"] and not choices):
                value["usage"] = usage
            if telemetry is not None:
                value["crit_telemetry"] = telemetry
            return value

        def admitted():
            nonlocal opened
            opened = True
            self.start_stream("text/event-stream", request, cancel)
            # As in a chat stream, pings keep the open stream alive while the engine has nothing to send.
            pings.enter_context(delivery.pinging(PING, self.api.ping_interval))

        timing = self.api.metrics.begin()
        try:
            with pings:  # they end before the closing events
                finish, measured, _, _, logprobs, _ = self.api.run(request, lambda piece: event(chunk(piece)), cancel,
                                                                   on_started=delivery.set_deadline,
                                                                   on_admitted=admitted, timing=timing)
        except EngineError as exc:
            self.measure("openai", timing, self.outcome(exc))
            if not opened:
                return self.engine_error(exc)  # refused before the stream began, such as by a full queue
            message, code = self.error_words(exc)
            event({"error": {"message": message, "type": "server_error" if exc.status >= 500 else "invalid_request_error",
                             "param": "prompt" if code else None, "code": code}, "request_id": self.request_id})
        else:
            self.measure("openai", timing, "completed", measured)
            # As in chat streams, the reply's logprobs come from the engine's final record, with the finish chunk.
            event(chunk("", finish, logprobs=None if logprobs is None else completion_logprobs(logprobs)))
            if request["include_usage"]:
                event(chunk("", usage=self.api.usage(measured), choices=False, telemetry=self.api.telemetry(measured)))
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass

    # Anthropic Messages (crit_api.anthropic)

    def anthropic_complete(self, request, cancel):
        pieces = []
        timing = self.api.metrics.begin()
        try:
            finish, measured, calls, reasoning, _, stopped = self.api.run(request, pieces.append, cancel, timing=timing)
        except EngineError as exc:
            self.measure("anthropic", timing, self.outcome(exc))
            return self.engine_error(exc)
        self.measure("anthropic", timing, "completed", measured)
        value = anthropic.message(request["requested_model"] or request["model"], finish, measured, "".join(pieces), calls,
                                  reasoning, stopped)
        self.send_json(200, value, self.headers_for(request, measured))

    def anthropic_stream(self, request, cancel):
        with StreamDelivery(self.connection, cancel, self.api.engine._stopping, self.api.engine.timeout) as delivery:
            previous, self.wfile = self.wfile, delivery
            try:
                self._anthropic_stream(request, cancel, delivery)
            finally:
                self.wfile = previous

    def _anthropic_stream(self, request, cancel, delivery):
        opened, pings = False, ExitStack()

        def send(event, data):
            try:
                self.wfile.write(f"event: {event}\n".encode("utf-8") + b"data: " + json.dumps(data).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            except OSError:
                cancel.set()

        stream = anthropic.Stream(send, request["requested_model"] or request["model"])

        def admitted():
            nonlocal opened
            opened = True
            self.start_stream("text/event-stream", request, cancel)
            stream.start()
            pings.enter_context(delivery.pinging(anthropic.PING, self.api.ping_interval))

        timing = self.api.metrics.begin()
        try:
            with pings:
                finish, measured, _, _, _, stopped = self.api.run(request, stream.text, cancel, stream.tools,
                                                                 stream.thinking, on_started=delivery.set_deadline,
                                                                 on_admitted=admitted, timing=timing)
        except EngineError as exc:
            self.measure("anthropic", timing, self.outcome(exc))
            if not opened:
                return self.engine_error(exc)
            stream.error(self.error_words(exc)[0], exc.status, self.request_id)
        else:
            self.measure("anthropic", timing, "completed", measured)
            stream.finish(finish, measured, stopped)

    # OpenAI Responses (crit_api.responses)

    def responses_complete(self, request, cancel):
        pieces = []
        timing = self.api.metrics.begin()
        try:
            finish, measured, calls, reasoning, _, _ = self.api.run(request, pieces.append, cancel, timing=timing)
        except EngineError as exc:
            self.measure("responses", timing, self.outcome(exc))
            return self.engine_error(exc)
        self.measure("responses", timing, "completed", measured)
        value = responses.reply(request["source"], request["model"], finish, measured, "".join(pieces), calls, reasoning)
        if measured:
            value["crit_telemetry"] = self.api.telemetry(measured)
        self.send_json(200, value, self.headers_for(request, measured))

    def responses_stream(self, request, cancel):
        with StreamDelivery(self.connection, cancel, self.api.engine._stopping, self.api.engine.timeout) as delivery:
            previous, self.wfile = self.wfile, delivery
            try:
                self._responses_stream(request, cancel, delivery)
            finally:
                self.wfile = previous

    def _responses_stream(self, request, cancel, delivery):
        opened, pings = False, ExitStack()

        def send(event, data):
            try:
                self.wfile.write(f"event: {event}\n".encode("utf-8") + b"data: " + json.dumps(data).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            except OSError:
                cancel.set()

        stream = responses.Stream(send, request["source"], request["model"])

        def admitted():
            nonlocal opened
            opened = True
            self.start_stream("text/event-stream", request, cancel)
            stream.start()
            # As in a chat stream, pings keep the open stream alive while the engine has nothing to send.
            pings.enter_context(delivery.pinging(PING, self.api.ping_interval))

        timing = self.api.metrics.begin()
        try:
            with pings:  # they end before the closing events
                finish, measured, _, _, _, _ = self.api.run(request, stream.text, cancel, stream.tools, stream.reasoning,
                                                             on_started=delivery.set_deadline, on_admitted=admitted,
                                                             timing=timing)
        except EngineError as exc:
            self.measure("responses", timing, self.outcome(exc))
            if not opened:
                return self.engine_error(exc)  # refused before the stream began, such as by a full queue
            # Clients read the code: context_length_exceeded to compact, server_error to retry.
            message, code = self.error_words(exc)
            stream.fail(message, code or ("server_error" if exc.status >= 500 else "invalid_prompt"), self.request_id)
        else:
            self.measure("responses", timing, "completed", measured)
            stream.finish(finish, measured, self.api.telemetry(measured))

    # Ollama (crit_api.ollama)

    def ollama_chat(self, raw, generate=False):
        """/api/chat, or /api/generate. Without messages (or a prompt) the request loads its model and answers
        done_reason "load", with keep_alive 0 unloads it and answers "unload"; keep_alive 0 on a reply unloads the model
        once the reply is sent, as Ollama expires its runner."""
        try:
            body = json.loads(raw)
            action = ollama.lifecycle(body, generate, self.api.lenient)
            candidate = self.api.candidate(body.get("model")) if action else None
        except (ValueError, UnicodeDecodeError) as exc:
            self.api.metrics.count("ollama", "rejected")
            return self.fail(400, "Invalid JSON body: " + str(exc))
        except RequestError as exc:
            self.api.metrics.count("ollama", self.outcome(exc))
            return self.fail(exc.status, str(exc))
        if action is None:
            request = self.dispatch(raw, ollama.generate_body if generate else ollama.chat_body, self.ollama_complete,
                                    self.ollama_stream, validate=self.api.generate_request if generate else None,
                                    parsed=body, generate=generate)
            if request is not None and ollama.unloads(body.get("keep_alive")):
                # In the background: the client has its reply and must not wait for the requests the unload drains.
                threading.Thread(target=self.unload_quietly, args=(request["candidate"],), daemon=True).start()
            return None
        self.lifecycle(action, candidate, lambda outcome: ollama.done(body.get("model") or self.api.model, action,
                                                                      generate))

    def unload_quietly(self, candidate):
        """Unload `candidate`'s engine if it is still the loaded one; a failure is the next request's to report."""
        try:
            self.api.engine.unload(candidate)
        except EngineError:
            pass

    def ollama_show(self, raw):
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise RequestError("The request body must be a JSON object")
            name = body.get("model", body.get("name"))
            candidate = self.api.candidate(name)
        except (ValueError, UnicodeDecodeError) as exc:
            return self.fail(400, "Invalid JSON body: " + str(exc))
        except RequestError as exc:
            return self.fail(exc.status, str(exc))
        wanted = self.api.engine.model_id(candidate)
        card = next((c for c in self.api.models() if c["id"] == wanted or (candidate == self.api.engine.default and c["default"])), None)
        if card is None:
            return self.fail(404, f"model '{name}' not found")
        self.send_json(200, ollama.show(card))

    def ollama_complete(self, request, cancel, generate=False):
        pieces, thoughts = [], []
        model = request["requested_model"] or request["model"]
        started = time.perf_counter()
        timing = self.api.metrics.begin()
        try:
            finish, measured, calls, reasoning, logprobs, _ = self.api.run(request, pieces.append, cancel, None,
                                                                           thoughts.append, timing=timing)
        except EngineError as exc:
            self.measure("ollama", timing, self.outcome(exc))
            return self.engine_error(exc)
        self.measure("ollama", timing, "completed", measured)
        seconds = time.perf_counter() - started
        text = "".join(pieces)
        if generate:
            value = ollama.final(model, finish, measured, seconds, response=text, thinking=reasoning or None,
                                 entries=logprobs)
        else:
            message = {"role": "assistant", "content": text}
            if reasoning:
                message["thinking"] = reasoning
            if calls:
                message["tool_calls"] = [ollama.tool_call(call) for call in calls]
            value = ollama.final(model, finish, measured, seconds, message=message, entries=logprobs)
        self.send_json(200, value, self.headers_for(request, measured))

    def ollama_stream(self, request, cancel, generate=False):
        with StreamDelivery(self.connection, cancel, self.api.engine._stopping, self.api.engine.timeout) as delivery:
            previous, self.wfile = self.wfile, delivery
            try:
                self._ollama_stream(request, cancel, delivery.set_deadline, generate)
            finally:
                self.wfile = previous

    def _ollama_stream(self, request, cancel, on_started, generate):
        model = request["requested_model"] or request["model"]
        opened = False

        def send(value):
            try:
                self.wfile.write(json.dumps(value).encode("utf-8") + b"\n")
                self.wfile.flush()
            except OSError:
                cancel.set()

        def admitted():
            nonlocal opened
            opened = True
            self.start_stream("application/x-ndjson", request, cancel)

        started = time.perf_counter()
        timing = self.api.metrics.begin()
        try:
            finish, measured, calls, _, logprobs, _ = self.api.run(
                request, lambda piece: send(ollama.chunk(model, piece, generate=generate)), cancel, None,
                lambda piece: send(ollama.chunk(model, "", thinking=piece, generate=generate)), on_started=on_started,
                on_admitted=admitted, timing=timing)
        except EngineError as exc:
            self.measure("ollama", timing, self.outcome(exc))
            if not opened:
                return self.engine_error(exc)
            send(ollama.error_body(self.error_words(exc)[0], self.request_id))
            return
        self.measure("ollama", timing, "completed", measured)
        if calls and not generate:
            send(ollama.chunk(model, "", tool_calls=[ollama.tool_call(call) for call in calls]))
        # As in the OpenAI stream, the reply's logprobs come from the engine's final record, with the last object.
        send(ollama.final(model, finish, measured, time.perf_counter() - started, response="" if generate else None,
                          entries=logprobs))


class Server(ThreadingHTTPServer):
    """The HTTP server, or HTTPS with `tls`, an ssl.SSLContext holding the server's certificate."""
    daemon_threads = True

    def __init__(self, address, api, tls=None):
        super().__init__(address, Handler)
        self.api = api
        self.tls = tls

    def get_request(self):
        request, address = super().get_request()
        if self.tls is not None:
            # The handshake waits for the connection's own thread (finish_request), so a client stalling in it holds
            # up no other.
            request = self.tls.wrap_socket(request, server_side=True, do_handshake_on_connect=False)
        return request, address

    def finish_request(self, request, client_address):
        if self.tls is not None:
            timeout = request.gettimeout()
            try:
                request.settimeout(TLS_HANDSHAKE_SECONDS)
                request.do_handshake()
            except OSError as exc:
                if self.api.log:
                    self.api.log("%s TLS handshake failed: %s" % (client_address[0], exc))
                return
            request.settimeout(timeout)
        super().finish_request(request, client_address)
