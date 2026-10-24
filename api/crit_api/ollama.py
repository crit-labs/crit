"""The Ollama API over the same engine (GET /, /api/version, /api/tags, /api/ps; POST /api/chat, /api/generate,
/api/show), so the local chat UIs and tools that look for an Ollama server use a crit engine.

/api/chat and /api/generate are translated into the chat request crit serve already validates
(server.Api.completion_request): Ollama's messages, images (base64 without a media type, sniffed from the bytes),
tool calls (arguments as objects, ids assigned when missing) and tool results, `tools`, `format` ("json" or a JSON
schema) and `options` (temperature, top_p, top_k, min_p, seed, num_predict, stop, repeat_penalty, presence_penalty,
frequency_penalty; the rest are Ollama's own runtime settings and are ignored). Replies stream as newline-delimited
JSON objects, each with the model, a timestamp and a message (or `response` for /api/generate), the last with
`done`, `done_reason` and the durations and token counts Ollama reports.
"""
import json
import time
import uuid

from .options import RequestError

CHAT_KNOWN = {"model", "messages", "tools", "format", "options", "stream", "keep_alive", "think"}
GENERATE_KNOWN = {"model", "prompt", "suffix", "system", "template", "images", "format", "options", "stream", "raw",
                  "keep_alive", "think", "context"}
OPTIONS = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p", "seed": "seed",
           "repeat_penalty": "repetition_penalty", "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty"}
# Ollama's runtime settings have no meaning for a crit engine: its context, threads and memory are the candidate's.
IGNORED_OPTIONS = {"num_ctx", "num_batch", "num_gpu", "main_gpu", "num_thread", "num_keep", "low_vram", "f16_kv",
                   "vocab_only", "use_mmap", "use_mlock", "mirostat", "mirostat_eta", "mirostat_tau", "tfs_z",
                   "typical_p", "repeat_last_n", "penalize_newline", "numa", "logits_all"}
MAGIC = (("/9j/", "image/jpeg"), ("iVBOR", "image/png"), ("R0lGOD", "image/gif"), ("UklGR", "image/webp"))
DONE_REASONS = {"stop": "stop", "length": "length", "tool_calls": "stop"}


def error_body(message):
    return {"error": message}


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


def options(value, result):
    """Apply Ollama's options to the chat body; returns the ignored option names."""
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
            if item is not None and item > 0:
                result["max_tokens"] = item
        elif key == "stop":
            if isinstance(item, str):
                result["stop"] = [item]
            elif isinstance(item, list):
                result["stop"] = item
            elif item is not None:
                raise RequestError("options.stop must be a string or a list", param="options.stop")
        elif key in IGNORED_OPTIONS:
            ignored.append("options." + key)
        else:
            raise RequestError("Unknown option " + key, param="options." + key, code="unsupported_parameter")
    return ignored


def response_format(value):
    if value is None or value == "":
        return None
    if value == "json":
        return {"type": "json_object"}
    if isinstance(value, dict):
        return {"type": "json_schema", "json_schema": {"name": "response", "schema": value}}
    raise RequestError('format must be "json" or a JSON schema object', param="format")


def _common(body, known, result):
    unknown = sorted(k for k in body if k not in known)
    if unknown:
        raise RequestError("Unknown request parameter: " + unknown[0], param=unknown[0], code="unsupported_parameter")
    if "stream" in body and type(body["stream"]) is not bool:
        raise RequestError("stream must be a boolean", param="stream")
    result["stream"] = body.get("stream", True)
    result["stream_options"] = {"include_usage": True}
    ignored = options(body.get("options"), result)
    constraint = response_format(body.get("format"))
    if constraint is not None:
        result["response_format"] = constraint
    ignored += sorted(k for k in ("keep_alive", "think") if k in body)
    return ignored


def chat_body(body):
    """(OpenAI-shaped chat body, ignored parameters) for an /api/chat request."""
    if not isinstance(body, dict):
        raise RequestError("The request body must be a JSON object")
    result = {"model": body.get("model"), "messages": messages(body.get("messages"))}
    if body.get("tools") is not None:
        result["tools"] = body["tools"]
    return result, _common(body, CHAT_KNOWN, result)


def generate_body(body):
    """(OpenAI-shaped chat body, ignored parameters) for an /api/generate request: the prompt is one user turn,
    rendered with the model's chat template like any request."""
    if not isinstance(body, dict):
        raise RequestError("The request body must be a JSON object")
    if body.get("raw"):
        raise RequestError("raw prompts are not supported: a crit engine renders the model's chat template", param="raw",
                           code="unsupported_parameter")
    for key in ("suffix", "template", "context"):
        if body.get(key):
            raise RequestError(f"{key} is not supported", param=key, code="unsupported_parameter")
    if not isinstance(body.get("prompt"), str) or not body["prompt"]:
        raise RequestError("prompt must be a non-empty string", param="prompt")
    rows = []
    if isinstance(body.get("system"), str) and body["system"]:
        rows.append({"role": "system", "content": body["system"]})
    rows.append({"role": "user", "content": _content(body["prompt"], body.get("images"), "prompt")})
    result = {"model": body.get("model"), "messages": rows}
    return result, _common(body, GENERATE_KNOWN, result)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1e9):09d}Z"


def tool_call(call):
    try:
        arguments = json.loads(call["function"]["arguments"] or "{}")
    except ValueError:
        arguments = {"arguments": call["function"]["arguments"]}
    return {"function": {"name": call["function"]["name"], "arguments": arguments}}


def final(model, finish, measured, seconds, message=None, response=None, thinking=None):
    """The last object of a reply: done, why, and Ollama's durations (nanoseconds) and token counts."""
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
        if measured.get("prefill_ms") is not None:
            value["prompt_eval_duration"] = int(measured["prefill_ms"] * 1e6)
        if measured.get("decode_ms") is not None:
            value["eval_duration"] = int(measured["decode_ms"] * 1e6)
        phases = value.get("prompt_eval_duration", 0) + value.get("eval_duration", 0)
    # The whole request, never less than the engine's own phases (the server's clock may be coarser).
    value["total_duration"] = max(int(seconds * 1e9), phases)
    return value


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


def model_card(card, created):
    """An /api/tags entry for a /v1/models card."""
    return {"name": card["id"], "model": card["id"], "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created)),
            "size": 0, "digest": card.get("system_fingerprint", ""),
            "details": {"parent_model": "", "format": "crit", "family": card.get("label", ""), "families": None,
                        "parameter_size": "", "quantization_level": ""}}


def show(card):
    """An /api/show reply: what a client reads to size its context and pick its features."""
    capabilities = ["completion"]
    if "chat-tools-v1" in card.get("capabilities", []):
        capabilities.append("tools")
    if "chat-images-v1" in card.get("capabilities", []):
        capabilities.append("vision")
    return {"modelfile": "", "parameters": "", "template": "", "details": model_card(card, time.time())["details"],
            "model_info": {"general.architecture": "crit", "crit.context_length": card["context_length"],
                           "crit.label": card.get("label", ""), "crit.capabilities": card.get("capabilities", [])},
            "capabilities": capabilities}


def tool_call_id():
    return "call_" + uuid.uuid4().hex[:12]
