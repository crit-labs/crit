"""The Anthropic Messages API over the same engine: POST /v1/messages, so clients that speak it (Claude Code and
other agent CLIs, SDKs with a base URL) run on a crit engine.

A Messages request is translated into the chat request crit serve already validates (server.Api.completion_request):
`system` becomes the leading system message, content blocks become text and image parts, `tool_use` blocks become
tool calls and `tool_result` blocks tool messages, `tools` with an `input_schema` become functions, `tool_choice`
and `stop_sequences` map one to one, and `output_config.format` of type json_schema becomes `response_format`. The
reply comes back as a Messages response (text, thinking and tool_use content blocks, Anthropic's stop reasons and
usage), or as the Messages event stream: message_start, content_block_start/delta/stop per block, message_delta
with the stop reason and usage, message_stop.
"""
import json
import uuid

from .options import RequestError

KNOWN = {"model", "messages", "max_tokens", "system", "tools", "tool_choice", "stop_sequences", "stream", "temperature",
         "top_p", "top_k", "output_config"}
# Accepted without effect: they configure Anthropic's service, not the model's generation.
IGNORED = {"metadata", "thinking", "service_tier", "betas", "inference_geo", "speed", "diagnostics", "fallbacks",
           "context_management"}
UNSUPPORTED = {"mcp_servers": "MCP servers", "container": "containers"}
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error",
               409: "invalid_request_error", 413: "request_too_large", 429: "rate_limit_error", 499: "api_error",
               500: "api_error", 504: "api_error", 529: "overloaded_error"}


def error_body(status, message):
    return {"type": "error", "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": message}}


def _text_blocks(value, where):
    """The text of a string or a list of text blocks."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise RequestError(f"{where} must be a string or a list of content blocks", param=where)
    parts = []
    for index, block in enumerate(value):
        if not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise RequestError(f"{where}[{index}] must be a text block", param=where)
        parts.append(block["text"])
    return "\n\n".join(parts)


def _image_part(block, where):
    source = block.get("source")
    if not isinstance(source, dict):
        raise RequestError(f"{where}: an image block needs a source", param=where)
    if source.get("type") == "base64":
        if not isinstance(source.get("media_type"), str) or not isinstance(source.get("data"), str):
            raise RequestError(f"{where}: a base64 image source needs media_type and data", param=where)
        url = f"data:{source['media_type']};base64,{source['data']}"
    elif source.get("type") == "url" and isinstance(source.get("url"), str):
        url = source["url"]
    else:
        raise RequestError(f"{where}: image sources are base64 or url", param=where)
    return {"type": "image_url", "image_url": {"url": url}}


def _parts(parts):
    """Content parts as crit's chat request takes them: one string when every part is text."""
    if all(part["type"] == "text" for part in parts):
        return "\n\n".join(part["text"] for part in parts)
    return parts


def messages(system, value):
    """crit's chat messages for a Messages request's system and messages."""
    if not isinstance(value, list) or not value:
        raise RequestError("messages must be a non-empty list", param="messages")
    result = []
    if system is not None:
        text = _text_blocks(system, "system")
        if text:
            result.append({"role": "system", "content": text})
    for index, message in enumerate(value):
        where = f"messages[{index}]"
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            raise RequestError(f"{where}.role must be user or assistant", param=where)
        role, content = message["role"], message.get("content")
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue
        if not isinstance(content, list) or not content:
            raise RequestError(f"{where}.content must be a string or a non-empty list of blocks", param=where)
        parts, calls, results = [], [], []
        for number, block in enumerate(content):
            at = f"{where}.content[{number}]"
            kind = block.get("type") if isinstance(block, dict) else None
            if kind == "text":
                if not isinstance(block.get("text"), str):
                    raise RequestError(f"{at}.text must be a string", param=at)
                parts.append({"type": "text", "text": block["text"]})
            elif kind == "image":
                parts.append(_image_part(block, at))
            elif kind == "tool_use" and role == "assistant":
                if not isinstance(block.get("id"), str) or not isinstance(block.get("name"), str) or not isinstance(block.get("input"), dict):
                    raise RequestError(f"{at}: a tool_use block needs id, name and an input object", param=at)
                calls.append({"id": block["id"], "type": "function",
                              "function": {"name": block["name"], "arguments": json.dumps(block["input"], ensure_ascii=False)}})
            elif kind == "tool_result" and role == "user":
                if not isinstance(block.get("tool_use_id"), str):
                    raise RequestError(f"{at}: a tool_result block needs tool_use_id", param=at)
                text = _text_blocks(block.get("content", ""), at + ".content")
                if block.get("is_error"):
                    text = text or "The tool call failed."
                results.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": text})
            elif kind in {"thinking", "redacted_thinking"} and role == "assistant":
                continue  # the engine's own reasoning is not replayed; Core's template decides what a turn keeps
            elif kind == "document":
                raise RequestError(f"{at}: document blocks are not supported; pass the text", param=at)
            else:
                raise RequestError(f"{at}: unknown or misplaced block type {kind!r}", param=at)
        # Tool results come first, as Anthropic orders them, then the user's text and images.
        result.extend(results)
        if role == "assistant":
            row = {"role": "assistant", "content": _parts(parts) if parts else ("" if calls else None)}
            if calls:
                row["tool_calls"] = calls
            if row["content"] is None:
                raise RequestError(f"{where}.content holds no text, image or tool_use blocks", param=where)
            result.append(row)
        elif parts:
            result.append({"role": "user", "content": _parts(parts)})
        elif not results:
            raise RequestError(f"{where}.content holds no text, image or tool_result blocks", param=where)
    return result


def tools(value):
    if value is None:
        return None
    if not isinstance(value, list):
        raise RequestError("tools must be a list", param="tools")
    result = []
    for index, tool in enumerate(value):
        where = f"tools[{index}]"
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise RequestError(f"{where} needs a name", param=where)
        if tool.get("type") not in (None, "custom"):
            raise RequestError(f"{where}: Anthropic-defined tools ({tool['type']}) are not available on a crit engine; "
                               "define the tool with an input_schema", param=where, code="unsupported_parameter")
        schema = tool.get("input_schema")
        if not isinstance(schema, dict):
            raise RequestError(f"{where}.input_schema must be a JSON schema object", param=where)
        function = {"name": tool["name"], "parameters": schema}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        result.append({"type": "function", "function": function})
    return result


def tool_choice(value):
    """(OpenAI tool_choice, parallel_tool_calls) for Anthropic's tool_choice object."""
    if value is None:
        return None, None
    if not isinstance(value, dict) or value.get("type") not in {"auto", "any", "tool", "none"}:
        raise RequestError("tool_choice.type must be auto, any, tool or none", param="tool_choice")
    parallel = None
    if "disable_parallel_tool_use" in value:
        if type(value["disable_parallel_tool_use"]) is not bool:
            raise RequestError("tool_choice.disable_parallel_tool_use must be a boolean", param="tool_choice")
        parallel = not value["disable_parallel_tool_use"]
    kind = value["type"]
    if kind == "tool":
        if not isinstance(value.get("name"), str):
            raise RequestError("tool_choice of type tool needs the tool's name", param="tool_choice")
        return {"type": "function", "function": {"name": value["name"]}}, parallel
    return {"auto": "auto", "any": "required", "none": "none"}[kind], parallel


def response_format(output_config):
    if output_config is None:
        return None
    if not isinstance(output_config, dict):
        raise RequestError("output_config must be an object", param="output_config")
    fmt = output_config.get("format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
        raise RequestError("output_config.format must be {type: json_schema, schema: {...}}", param="output_config")
    return {"type": "json_schema", "json_schema": {"name": "response", "schema": fmt["schema"]}}


def chat_body(body):
    """The OpenAI-shaped chat body crit serve validates, for a Messages request body, and the parameters ignored."""
    if not isinstance(body, dict):
        raise RequestError("The request body must be a JSON object")
    for key, feature in UNSUPPORTED.items():
        if body.get(key) is not None:
            raise RequestError(f"A crit engine does not provide {feature}", param=key, code="unsupported_parameter")
    unknown = sorted(k for k in body if k not in KNOWN | IGNORED | set(UNSUPPORTED))
    if unknown:
        raise RequestError("Unknown request parameter: " + unknown[0], param=unknown[0], code="unsupported_parameter")
    if type(body.get("max_tokens")) is not int:
        raise RequestError("max_tokens is required and must be an integer", param="max_tokens")
    if "stream" in body and type(body["stream"]) is not bool:
        raise RequestError("stream must be a boolean", param="stream")
    stops = body.get("stop_sequences")
    if stops is not None and (not isinstance(stops, list) or not all(isinstance(s, str) and s for s in stops)):
        raise RequestError("stop_sequences must be a list of non-empty strings", param="stop_sequences")
    result = {"model": body.get("model"), "messages": messages(body.get("system"), body.get("messages")),
              "max_tokens": body["max_tokens"], "stream": bool(body.get("stream")),
              "stream_options": {"include_usage": True}}
    if stops:
        result["stop"] = stops
    for key in ("temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            result[key] = body[key]
    functions = tools(body.get("tools"))
    if functions is not None:
        result["tools"] = functions
    choice, parallel = tool_choice(body.get("tool_choice"))
    if choice is not None:
        if functions is None:
            raise RequestError("tool_choice needs tools", param="tool_choice")
        result["tool_choice"] = choice
    if parallel is not None:
        result["parallel_tool_calls"] = parallel
    constraint = response_format(body.get("output_config"))
    if constraint is not None:
        result["response_format"] = constraint
    ignored = sorted(k for k in body if k in IGNORED)
    if isinstance(body.get("output_config"), dict):
        ignored += sorted(f"output_config.{k}" for k in body["output_config"] if k != "format")
    return result, ignored


def _input(call):
    try:
        value = json.loads(call["function"]["arguments"] or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {"value": value}


def usage(measured):
    """Anthropic usage: input tokens the engine computed, cache reads apart, and the output tokens."""
    if not measured:
        return {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}
    cached = measured["prompt_tokens"] - measured["prefill_tokens"] if measured.get("prefill_tokens") is not None else 0
    return {"input_tokens": measured["prompt_tokens"] - cached, "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cached, "output_tokens": measured["output_tokens"]}


def stop_reason(finish, stopped):
    if finish == "stop" and stopped is not None:
        return "stop_sequence"
    return STOP_REASONS.get(finish, "end_turn")


def message(model, finish, measured, text, calls, reasoning, stopped, ident=None):
    """A complete Messages response."""
    content = []
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if text or not calls:
        content.append({"type": "text", "text": text})
    for call in calls or []:
        content.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": _input(call)})
    return {"id": ident or "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop_reason(finish, stopped), "stop_sequence": stopped, "usage": usage(measured)}


class Stream:
    """Writes the Messages event stream for one request through `send(event, data)`: blocks open as text, thinking
    or tool_use content arrives, in the order the engine produces it, and close before the next kind or the end."""

    def __init__(self, send, model):
        self.send = send
        self.model = model
        self.ident = "msg_" + uuid.uuid4().hex
        self.next_index = 0
        self.open = None          # (kind, index) of the text or thinking block being written
        self.tool_blocks = {}     # tool call index -> content block index
        self.started = False

    def start(self):
        self.started = True
        self.send("message_start", {"type": "message_start", "message": {
            "id": self.ident, "type": "message", "role": "assistant", "model": self.model, "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": usage(None)}})

    def _close(self):
        if self.open is not None:
            self.send("content_block_stop", {"type": "content_block_stop", "index": self.open[1]})
            self.open = None

    def _ensure(self, kind):
        if self.open is not None and self.open[0] == kind:
            return self.open[1]
        self._close()
        index, self.next_index = self.next_index, self.next_index + 1
        block = {"type": "text", "text": ""} if kind == "text" else {"type": "thinking", "thinking": "", "signature": ""}
        self.send("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})
        self.open = (kind, index)
        return index

    def text(self, piece):
        index = self._ensure("text")
        self.send("content_block_delta", {"type": "content_block_delta", "index": index,
                                          "delta": {"type": "text_delta", "text": piece}})

    def thinking(self, piece):
        index = self._ensure("thinking")
        self.send("content_block_delta", {"type": "content_block_delta", "index": index,
                                          "delta": {"type": "thinking_delta", "thinking": piece}})

    def tools(self, deltas):
        """OpenAI tool-call deltas (indexed; the first for a call carries id and name, every one may carry an
        arguments fragment) become tool_use blocks with input_json_delta events."""
        self._close()
        for delta in deltas:
            position = delta.get("index", 0)
            function = delta.get("function") or {}
            if position not in self.tool_blocks:
                index, self.next_index = self.next_index, self.next_index + 1
                self.tool_blocks[position] = index
                self.send("content_block_start", {"type": "content_block_start", "index": index, "content_block": {
                    "type": "tool_use", "id": delta.get("id") or "toolu_" + uuid.uuid4().hex[:24],
                    "name": function.get("name") or "", "input": {}}})
            fragment = function.get("arguments")
            if fragment:
                self.send("content_block_delta", {"type": "content_block_delta", "index": self.tool_blocks[position],
                                                  "delta": {"type": "input_json_delta", "partial_json": fragment}})

    def finish(self, finish, measured, stopped):
        self._close()
        for index in self.tool_blocks.values():
            self.send("content_block_stop", {"type": "content_block_stop", "index": index})
        self.tool_blocks.clear()
        self.send("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason(finish, stopped),
                                                                       "stop_sequence": stopped}, "usage": usage(measured)})
        self.send("message_stop", {"type": "message_stop"})

    def error(self, message):
        self.send("error", error_body(500, message))
