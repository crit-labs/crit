"""The Ollama API over the same engine (GET /, /api/version, /api/tags, /api/ps; POST /api/chat, /api/generate,
/api/show), so the local chat UIs and tools that look for an Ollama server use a crit engine.

/api/chat and /api/generate are translated into the chat request crit serve already validates
(server.Api.completion_request): Ollama's messages, images (base64 without a media type, sniffed from the bytes),
tool calls (arguments as objects, ids assigned when missing) and tool results, a final assistant message (which the
reply continues, as Ollama's does), `tools`, `format` ("json" or a JSON schema), `options` (temperature, top_p, top_k,
min_p, seed, num_predict, stop, repeat_penalty, presence_penalty, frequency_penalty; the rest are Ollama's own runtime
settings and are ignored), `think`, which sets the package's thinking switches, and `logprobs` and `top_logprobs`.
/api/generate with `raw` true is a raw-prompt completion instead (server.Api.prompt_request, completion-v1): the
prompt as written, without the template or a system message, so `think` is ignored. Replies stream as
newline-delimited JSON objects, each with the model, a timestamp and a message (or `response` for /api/generate), the
last with `done`, `done_reason`, the durations and token counts Ollama reports and the reply's `logprobs` when asked
for. A request without messages (or a prompt) only loads its model, or with `keep_alive` 0 unloads it;
`keep_alive` 0 on a reply unloads the model after it.
"""
import json
import re
import time
import uuid

from crit_core.serving import TOP_LOGPROBS

from .options import RequestError, parameter_name, template_variables, unknown_parameters

CHAT_KNOWN = {"model", "messages", "tools", "format", "options", "stream", "keep_alive", "think", "logprobs",
              "top_logprobs"}
GENERATE_KNOWN = {"model", "prompt", "suffix", "system", "template", "images", "format", "options", "stream", "raw",
                  "keep_alive", "think", "context", "logprobs", "top_logprobs"}
OPTIONS = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p", "seed": "seed",
           "repeat_penalty": "repetition_penalty", "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty"}
# Ollama's runtime settings have no meaning for a crit engine: its context, threads and memory are the candidate's.
IGNORED_OPTIONS = {"num_ctx", "num_batch", "num_gpu", "main_gpu", "num_thread", "num_keep", "low_vram", "f16_kv",
                   "vocab_only", "use_mmap", "use_mlock", "mirostat", "mirostat_eta", "mirostat_tau", "tfs_z",
                   "typical_p", "repeat_last_n", "penalize_newline", "numa", "logits_all"}
MAGIC = (("/9j/", "image/jpeg"), ("iVBOR", "image/png"), ("R0lGOD", "image/gif"), ("UklGR", "image/webp"))
DONE_REASONS = {"stop": "stop", "length": "length", "tool_calls": "stop"}
THINK_LEVELS = ("low", "medium", "high")  # think's levels, which set a template's reasoning_effort
# num_predict -1 (no limit) and -2 (until the context is full): all the output the context leaves room for. crit serve
# fits it for a package declaring chat-fit-output-v1 and gives the default cap otherwise; no JSON value is this.
FILL_CONTEXT = object()
# What /api/version reports: clients compare it with the Ollama release a feature needs, not with crit's version.
OLLAMA_COMPAT_VERSION = "0.12.0"
LOADED_UNTIL = "2318-01-01T00:00:00Z"  # /api/ps expires_at of a model that stays loaded


def error_body(message, request_id=None):
    return {"error": message} if request_id is None else {"error": message, "request_id": request_id}


def _image(data, where):
    if not isinstance(data, str) or not data:
        raise RequestError(f"{where}: images are base64 strings", param=where)
    media = next((kind for prefix, kind in MAGIC if data.startswith(prefix)), "image/png")
    return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}


def _content(text, images, where):
    if not isinstance(text, str):
        raise RequestError(f"{where}.content must be a string", param=where)
    if not images:
        return text
    if not isinstance(images, list):
        raise RequestError(f"{where}.images must be a list of base64 strings", param=where)
    return [{"type": "text", "text": text}] + [_image(image, f"{where}.images[{i}]") for i, image in enumerate(images)]


def messages(value):
    """crit's chat messages for Ollama's: tool calls get ids when Ollama left them out, and tool results are matched
    to the latest assistant's calls in order."""
    if not isinstance(value, list) or not value:
        raise RequestError("messages must be a non-empty list", param="messages")
    result, pending = [], []
    for index, message in enumerate(value):
        where = f"messages[{index}]"
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise RequestError(f"{where}.role must be system, user, assistant or tool", param=where)
        role = message["role"]
        if role == "tool":
            if not pending:
                raise RequestError(f"{where}: a tool message needs a preceding assistant tool call", param=where)
            result.append({"role": "tool", "tool_call_id": pending.pop(0), "content": _content(message.get("content", ""), None, where)})
            continue
        row = {"role": role, "content": _content(message.get("content", ""), message.get("images"), where)}
        calls = message.get("tool_calls")
        if calls and role == "assistant":
            if not isinstance(calls, list):
                raise RequestError(f"{where}.tool_calls must be a list", param=where)
            row["tool_calls"], pending = [], []
            for number, call in enumerate(calls):
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    raise RequestError(f"{where}.tool_calls[{number}] needs function.name", param=where)
                arguments = function.get("arguments", {})
                ident = call.get("id") if isinstance(call.get("id"), str) else f"call_{index}_{number}"
                row["tool_calls"].append({"id": ident, "type": "function", "function": {
                    "name": function["name"],
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)}})
                pending.append(ident)
        elif role == "assistant":
            pending = []
        result.append(row)
    return result


def options(value, result, lenient=False):
    """Apply Ollama's options to the chat body; returns the ignored option names (with `lenient`, unknown ones too)."""
    if value is None:
        return []
    if not isinstance(value, dict):
        raise RequestError("options must be an object", param="options")
    ignored = []
    for key, item in value.items():
        if key in OPTIONS:
            if item is not None:
                result[OPTIONS[key]] = item
        elif key == "num_predict":
            if item is not None and (type(item) is not int):
                raise RequestError("options.num_predict must be an integer", param="options.num_predict")
            if item in (-1, -2):
                result["max_tokens"] = FILL_CONTEXT
            elif item is not None and item > 0:
                result["max_tokens"] = item
        elif key == "stop":
            if isinstance(item, str):
                result["stop"] = [item]
            elif isinstance(item, list):
                result["stop"] = item
            elif item is not None:
                raise RequestError("options.stop must be a string or a list", param="options.stop")
        elif key in IGNORED_OPTIONS or lenient and parameter_name(key):
            ignored.append("options." + key)
        else:
            raise RequestError("Unknown option " + key, param="options." + key, code="unsupported_parameter")
    return ignored


def logprobs(body, result):
    """Apply logprobs and top_logprobs to the chat body; returns the ignored names (top_logprobs without logprobs,
    which Ollama ignores)."""
    top = body.get("top_logprobs")
    if top is not None and (type(top) is not int or not 0 <= top <= TOP_LOGPROBS):
        raise RequestError(f"top_logprobs must be between 0 and {TOP_LOGPROBS}", param="top_logprobs")
    if body.get("logprobs") is not None:
        result["logprobs"] = body["logprobs"]
    if body.get("logprobs") is True:
        result["top_logprobs"] = top
        return []
    return ["top_logprobs"] if top else []


def response_format(value):
    if value is None or value == "":
        return None
    if value == "json":
        return {"type": "json_object"}
    if isinstance(value, dict):
        return {"type": "json_schema", "json_schema": {"name": "response", "schema": value}}
    raise RequestError('format must be "json" or a JSON schema object', param="format")


def thinking(value):
    """Ollama's think as crit serve's thinking controls (server.Api.completion_request), or None. Its value is validated
    (think_controls) only for a package that applies it, so any other package ignores and lists think whatever it
    holds."""
    return None if value is None else {"params": ["think"], "controls": lambda: think_controls(value)}


def think_controls(value):
    """(enabled, effort, unused parameters) for think: true, false or one of THINK_LEVELS."""
    if type(value) is not bool and value not in THINK_LEVELS:
        raise RequestError("think must be true, false, " + ", ".join(THINK_LEVELS), param="think")
    return (value, None, []) if type(value) is bool else (None, value, [])


def unloads(keep_alive):
    """Whether Ollama's keep_alive asks to unload the model after the request: zero, as seconds or a duration such
    as "0" or "0s"."""
    if type(keep_alive) in (int, float):
        return keep_alive == 0
    return isinstance(keep_alive, str) and re.fullmatch(r"[+-]?0+(\.0*)?(ns|us|ms|s|m|h)?", keep_alive) is not None


def lifecycle(body, generate=False, lenient=False):
    """"load" for an /api/chat request without messages, or an /api/generate request without a prompt, which only
    loads its model; "unload" when its keep_alive is 0; None for a request to answer. An unknown parameter is refused
    as on a reply, or with `lenient` (crit serve --lenient) ignored."""
    if not isinstance(body, dict):
        return None
    key, empty = ("prompt", (None, "")) if generate else ("messages", (None, []))
    if body.get(key) not in empty:
        return None
    unknown_parameters(body, GENERATE_KNOWN if generate else CHAT_KNOWN, lenient)
    return "unload" if unloads(body.get("keep_alive")) else "load"


def _common(body, known, result, lenient):
    unknown = unknown_parameters(body, known, lenient)
    if "stream" in body and type(body["stream"]) is not bool:
        raise RequestError("stream must be a boolean", param="stream")
    result["stream"] = body.get("stream", True)
    result["stream_options"] = {"include_usage": True}
    ignored = options(body.get("options"), result, lenient) + logprobs(body, result)
    constraint = response_format(body.get("format"))
    if constraint is not None:
        result["response_format"] = constraint
    # keep_alive 0 is applied: crit_api.server unloads the model after the reply.
    ignored += (["keep_alive"] if "keep_alive" in body and not unloads(body["keep_alive"]) else []) + unknown
    return ignored, thinking(body.get("think"))


def chat_body(body, lenient=False):
    """(OpenAI-shaped chat body, ignored parameters, thinking controls) for an /api/chat request; with `lenient` (crit
    serve --lenient) unknown parameters and options are ignored too."""
    if not isinstance(body, dict):
        raise RequestError("The request body must be a JSON object")
    result = {"model": body.get("model"), "messages": messages(body.get("messages"))}
    if result["messages"][-1]["role"] == "assistant":
        result.update(continue_final_message=True, add_generation_prompt=False)  # the reply continues it
    if body.get("tools") is not None:
        result["tools"] = body["tools"]
    return (result, *_common(body, CHAT_KNOWN, result, lenient))


def generate_body(body, lenient=False):
    """(OpenAI-shaped chat body, ignored parameters, thinking controls) for an /api/generate request: the prompt is
    one user turn, rendered with the model's chat template like any request; `lenient` as for chat_body. With raw true
    it is a /v1/completions body instead, the prompt continued as written, which a system message, images or a format
    cannot join: its logprobs are the number of alternatives (top_logprobs) when logprobs is true, and think, which
    sets template variables a raw prompt never renders, is ignored and listed."""
    if not isinstance(body, dict):
        raise RequestError("The request body must be a JSON object")
    if body.get("raw") is not None and type(body["raw"]) is not bool:
        raise RequestError("raw must be a boolean", param="raw")
    for key in ("suffix", "template", "context"):
        if body.get(key):
            raise RequestError(f"{key} is not supported", param=key, code="unsupported_parameter")
    if not isinstance(body.get("prompt"), str) or not body["prompt"]:
        raise RequestError("prompt must be a non-empty string", param="prompt")
    if body.get("raw"):
        for key in ("system", "images", "format"):
            if body.get(key):
                raise RequestError(f"{key} does not apply to a raw prompt", param=key, code="unsupported_parameter")
        result = {"model": body.get("model"), "prompt": body["prompt"]}
        ignored, thinking = _common(body, GENERATE_KNOWN, result, lenient)
        wanted, top = result.pop("logprobs", None), result.pop("top_logprobs", None)
        if wanted is not None and type(wanted) is not bool:
            raise RequestError("logprobs must be a boolean", param="logprobs")
        if wanted:
            result["logprobs"] = top or 0
        return result, ignored + (thinking["params"] if thinking else []), None
    rows = []
    if isinstance(body.get("system"), str) and body["system"]:
        rows.append({"role": "system", "content": body["system"]})
    rows.append({"role": "user", "content": _content(body["prompt"], body.get("images"), "prompt")})
    result = {"model": body.get("model"), "messages": rows}
    return (result, *_common(body, GENERATE_KNOWN, result, lenient))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1e9):09d}Z"


def tool_call(call):
    try:
        arguments = json.loads(call["function"]["arguments"] or "{}")
    except ValueError:
        arguments = {"arguments": call["function"]["arguments"]}
    return {"function": {"name": call["function"]["name"], "arguments": arguments}}


def final(model, finish, measured, seconds, message=None, response=None, thinking=None, entries=None):
    """The last object of a reply: done, why, Ollama's durations (nanoseconds) and token counts, the prompt tokens
    the engine did not compute as prompt_eval_cached_count when it reports its prefill, and with `entries` (OpenAI's
    logprobs content) the reply's logprobs."""
    value = {"model": model, "created_at": now()}
    if response is not None:
        value["response"] = response
        if thinking:
            value["thinking"] = thinking
    else:
        value["message"] = message if message is not None else {"role": "assistant", "content": ""}
    value.update({"done": True, "done_reason": DONE_REASONS.get(finish, "stop"), "load_duration": 0})
    phases = 0
    if measured:
        value.update({"prompt_eval_count": measured["prompt_tokens"], "eval_count": measured["output_tokens"]})
        if measured.get("prefill_tokens") is not None:
            value["prompt_eval_cached_count"] = measured["prompt_tokens"] - measured["prefill_tokens"]
        if measured.get("prefill_ms") is not None:
            value["prompt_eval_duration"] = int(measured["prefill_ms"] * 1e6)
        if measured.get("decode_ms") is not None:
            value["eval_duration"] = int(measured["decode_ms"] * 1e6)
        phases = value.get("prompt_eval_duration", 0) + value.get("eval_duration", 0)
    # The whole request, never less than the engine's own phases (the server's clock may be coarser).
    value["total_duration"] = max(int(seconds * 1e9), phases)
    if entries is not None:
        # The same token, logprob, bytes and crit_token_id; Ollama leaves out an empty top_logprobs.
        value["logprobs"] = [{key: item for key, item in entry.items() if key != "top_logprobs" or item}
                             for entry in entries]
    return value


def done(model, reason, generate=False):
    """The one object answering a request that only loads ("load") or unloads ("unload") its model."""
    value = {"model": model, "created_at": now()}
    value.update({"response": ""} if generate else {"message": {"role": "assistant", "content": ""}})
    return dict(value, done=True, done_reason=reason)


def chunk(model, content="", thinking=None, tool_calls=None, generate=False):
    if generate:
        value = {"model": model, "created_at": now(), "response": content, "done": False}
        if thinking is not None:
            value["thinking"] = thinking
        return value
    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": model, "created_at": now(), "message": message, "done": False}


def model_card(card, created, digest=""):
    """An /api/tags entry for a /v1/models card; `digest` is its package's sha256."""
    return {"name": card["id"], "model": card["id"], "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created)),
            "size": 0, "digest": digest,
            "details": {"parent_model": "", "format": "crit", "family": card.get("label", ""), "families": None,
                        "parameter_size": "", "quantization_level": ""}}


def loaded_card(card, created, digest, status, until=None):
    """An /api/ps entry. `until` is when crit serve --idle-unload unloads the engine (seconds since the epoch); without
    it the engine stays loaded until it is unloaded or crit serve stops, which Ollama shows as a far-future expires_at
    (keep_alive -1)."""
    expires = LOADED_UNTIL if until is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(until))
    return dict(model_card(card, created, digest), expires_at=expires, size_vram=status.get("device_bytes", 0),
                context_length=card["context_length"])


def show(card):
    """An /api/show reply: what a client reads to size its context and pick its features, and the card's limits."""
    capabilities = ["completion"]
    if "chat-tools-v1" in card.get("capabilities", []):
        capabilities.append("tools")
    if template_variables(card.get("capabilities", [])):
        capabilities.append("thinking")
    if "chat-images-v1" in card.get("capabilities", []):
        capabilities.append("vision")
    return {"modelfile": "", "parameters": "", "template": "", "details": model_card(card, time.time())["details"],
            "model_info": {"general.architecture": "crit", "crit.context_length": card["context_length"],
                           "crit.label": card.get("label", ""), "crit.capabilities": card.get("capabilities", [])},
            "capabilities": capabilities, "limits": card["limits"]}


def tool_call_id():
    return "call_" + uuid.uuid4().hex[:12]
