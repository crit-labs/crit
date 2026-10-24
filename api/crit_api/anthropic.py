"""The Anthropic Messages API over the same engine: POST /v1/messages, so clients that speak it (Claude Code and
other agent CLIs, SDKs with a base URL) run on a crit engine.

A Messages request is translated into the chat request crit serve already validates (server.Api.completion_request):
`system` becomes the leading system message, content blocks become text and image parts, `tool_use` blocks become
tool calls, `thinking` blocks the turn's reasoning_content and `tool_result` blocks tool messages (text and images,
an error's text after "Error: "), `tools` with an `input_schema` become functions (`strict` included), `tool_choice`
becomes OpenAI's, `stop_sequences` becomes `stop`, `output_config.format` of type json_schema becomes
`response_format`, `thinking` and `output_config.effort` set the package's thinking switches, and a final assistant
message is prefill, which the reply continues (continue_final_message), each then under crit's limits and capability
gates (docs/SERVING.md). The reply comes back as a Messages response (text, thinking and tool_use content blocks,
Anthropic's stop reasons and usage), or as the Messages event stream: message_start, content_block_start/delta/stop
per block, one block open at a time, message_delta with the stop reason and usage, message_stop, with ping events
while the engine has nothing to send. POST /v1/messages/count_tokens counts the prompt of a request's model,
system, messages, tools, tool_choice and thinking, for packages declaring chat-tokenize-v1.
"""
import json
import uuid

from .options import RequestError, complete_arguments, parameter_name, unknown_parameters

KNOWN = {"model", "messages", "max_tokens", "system", "tools", "tool_choice", "stop_sequences", "stream", "temperature",
         "top_p", "top_k", "output_config", "thinking"}
# Accepted without effect: they configure Anthropic's service, not the model's generation.
IGNORED = {"metadata", "service_tier", "betas", "inference_geo", "speed", "diagnostics", "fallbacks",
           "context_management"}
UNSUPPORTED = {"mcp_servers": "MCP servers", "container": "containers"}
# What POST /v1/messages/count_tokens counts: a Messages request's prompt, with the thinking that sets its template
# variables, without its output settings.
COUNTED = {"model", "messages", "system", "tools", "tool_choice", "thinking"}
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
# thinking.type: whether it turns thinking on.
THINKING = {"enabled": True, "adaptive": True, "disabled": False, "between_tools": False}
# output_config.effort as reasoning_effort: xhigh and max take the highest a template takes.
EFFORTS = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
# The type Anthropic gives a status; a 503 (a stopping or blocked engine) is the overload its clients retry.
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error",
               405: "invalid_request_error", 409: "invalid_request_error", 413: "request_too_large", 429: "rate_limit_error",
               499: "api_error", 500: "api_error", 503: "overloaded_error", 504: "timeout_error",
               529: "overloaded_error"}
PING = b'event: ping\ndata: {"type": "ping"}\n\n'  # Anthropic's keep-alive event, which its clients skip


def error_body(status, message, request_id=None):
    body = {"type": "error", "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": message}}
    return body if request_id is None else dict(body, request_id=request_id)


def context_message(prompt_tokens, output_tokens, capacity):
    """A context overflow in Anthropic's words, which clients such as Claude Code match to compact the conversation."""
    if prompt_tokens >= capacity:
        return f"prompt is too long: {prompt_tokens} tokens > {capacity} maximum"
    return (f"input length and `max_tokens` exceed context limit: {prompt_tokens} + {output_tokens} > {capacity}, "
            "decrease input length or `max_tokens` and try again")


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


def _result(block, where):
    """A tool_result block's content for its tool message: its text and image blocks in order, an error's (is_error)
    text after "Error: "."""
    content = block.get("content", "")
    if isinstance(content, str):
        parts = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        parts = []
        for index, item in enumerate(content):
            kind = item.get("type") if isinstance(item, dict) else None
            if kind == "text" and isinstance(item.get("text"), str):
                parts.append({"type": "text", "text": item["text"]})
            elif kind == "image":
                parts.append(_image_part(item, f"{where}[{index}]"))
            else:
                raise RequestError(f"{where}[{index}] must be a text or image block", param=where)
    else:
        raise RequestError(f"{where} must be a string or a list of content blocks", param=where)
    if block.get("is_error"):
        if not parts or parts[0]["type"] != "text":
            parts.insert(0, {"type": "text", "text": ""})
        parts[0] = {"type": "text", "text": "Error: " + (parts[0]["text"] or "The tool call failed.")}
    return _parts(parts)


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
        parts, calls, results, thoughts = [], [], [], []
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
                results.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                                "content": _result(block, at + ".content")})
            elif kind == "thinking" and role == "assistant":
                if not isinstance(block.get("thinking"), str):
                    raise RequestError(f"{at}.thinking must be a string", param=at)
                thoughts.append(block["thinking"])
            elif kind == "redacted_thinking" and role == "assistant":
                continue  # encrypted by Anthropic, so no engine can read it
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
            if thoughts:
                row["reasoning_content"] = "\n\n".join(thoughts)  # as OpenAI clients send a turn's reasoning back
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
        if tool.get("strict") is not None:
            function["strict"] = tool["strict"]
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


def thinking(body):
    """A request's thinking and output_config.effort as crit serve's thinking controls (server.Api.completion_request),
    or None: the parameters that carry them and `controls`, which validates their values (thinking_controls) and runs
    only for a package that applies them, so any other package ignores and lists them whatever they hold."""
    value, config = body.get("thinking"), body.get("output_config")
    effort = config.get("effort") if isinstance(config, dict) else None
    params = (["thinking"] if value is not None else []) + (["output_config.effort"] if effort is not None else [])
    return {"params": params, "controls": lambda: thinking_controls(value, effort)} if params else None


def thinking_controls(value, effort):
    """(enabled, effort, unused parameters) for thinking ({type: one of THINKING}) and output_config.effort (EFFORTS).
    Its other thinking fields, such as budget_tokens, are unused and listed, so each needs a name parameter_name
    takes."""
    kind = value.get("type") if isinstance(value, dict) else None
    if value is not None and (not isinstance(kind, str) or kind not in THINKING):
        raise RequestError("thinking.type must be " + ", ".join(THINKING), param="thinking")
    if effort is not None and (not isinstance(effort, str) or effort not in EFFORTS):
        raise RequestError("output_config.effort must be " + ", ".join(EFFORTS), param="output_config")
    unused = sorted(k for k in value or {} if k != "type")
    for key in unused:
        if not parameter_name(key):
            raise RequestError("Unknown request parameter: thinking." + key, param="thinking",
                               code="unsupported_parameter")
    return THINKING.get(kind), EFFORTS.get(effort), ["thinking." + k for k in unused]


def _checked(body, known, lenient):
    """The unknown parameters of a Messages body whose keys are `known`, ignored or refused with a reason: with
    `lenient` (crit serve --lenient) the ones it ignores, else the first is refused."""
    if not isinstance(body, dict):
        raise RequestError("The request body must be a JSON object")
    for key, feature in UNSUPPORTED.items():
        if body.get(key) is not None:
            raise RequestError(f"A crit engine does not provide {feature}", param=key, code="unsupported_parameter")
    return unknown_parameters(body, known | IGNORED | set(UNSUPPORTED), lenient)


def _prompt(body):
    """The OpenAI chat fields a Messages request's prompt comes from: model, messages (system first), tools,
    tool_choice and parallel_tool_calls, and a final assistant message as prefill, which the reply continues."""
    result = {"model": body.get("model"), "messages": messages(body.get("system"), body.get("messages"))}
    if result["messages"][-1]["role"] == "assistant":
        result.update(continue_final_message=True, add_generation_prompt=False)  # prefill: the reply continues it
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
    return result


def count_body(body, lenient=False):
    """The OpenAI-shaped chat body whose prompt POST /v1/messages/count_tokens counts, for its Messages body (model,
    system, messages, tools, tool_choice and thinking), the parameters ignored (with `lenient`, unknown ones too) and
    the thinking controls of the counted ones, as chat_body gives them."""
    unknown = _checked(body, COUNTED, lenient)
    counted = {key: value for key, value in body.items() if key in COUNTED}
    return _prompt(body), sorted(k for k in body if k in IGNORED) + unknown, thinking(counted)


def chat_body(body, lenient=False):
    """The OpenAI-shaped chat body crit serve validates for a Messages request body, the parameters ignored (with
    `lenient`, crit serve --lenient, unknown ones too) and the thinking controls."""
    unknown = _checked(body, KNOWN, lenient)
    if type(body.get("max_tokens")) is not int:
        raise RequestError("max_tokens is required and must be an integer", param="max_tokens")
    if "stream" in body and type(body["stream"]) is not bool:
        raise RequestError("stream must be a boolean", param="stream")
    stops = body.get("stop_sequences")
    if stops is not None and (not isinstance(stops, list) or not all(isinstance(s, str) and s for s in stops)):
        raise RequestError("stop_sequences must be a list of non-empty strings", param="stop_sequences")
    result = dict(_prompt(body), max_tokens=body["max_tokens"], stream=bool(body.get("stream")),
                  stream_options={"include_usage": True})
    if stops:
        result["stop"] = stops
    for key in ("temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            result[key] = body[key]
    constraint = response_format(body.get("output_config"))
    if constraint is not None:
        result["response_format"] = constraint
    ignored = sorted(k for k in body if k in IGNORED) + unknown
    if isinstance(body.get("output_config"), dict):
        for key in sorted(k for k in body["output_config"] if k not in {"format", "effort"}):
            if not parameter_name(key):
                raise RequestError("Unknown request parameter: output_config." + key, param="output_config",
                                   code="unsupported_parameter")
            ignored.append("output_config." + key)
    return result, ignored, thinking(body)


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
    """Writes the Messages event stream for one request through `send(event, data)`: one content block is open at a
    time, text, thinking or a tool call's tool_use, opened as its content arrives in the order the engine produces
    it and closed when the next one opens or at the end. Engines may interleave their calls' deltas, so a call's
    block stays open until its arguments are whole JSON, and what arrives for other blocks meanwhile waits."""

    def __init__(self, send, model):
        self.send = send
        self.model = model
        self.ident = "msg_" + uuid.uuid4().hex
        self.next_index = 0
        self.open = None          # (kind, content block index, tool call index or None) of the block being written
        self.calls = {}           # tool call index -> the call as its block received it
        self.held = []            # (kind, piece or tool-call delta) waiting for the open call's arguments
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

    def _unfinished(self):
        return self.open is not None and self.open[2] is not None and not complete_arguments(self.calls[self.open[2]])

    def _write(self, kind, value):
        """A text or thinking piece, or a tool-call delta, into its block, which replaces the open block."""
        call = value.get("index", 0) if kind == "tool_use" else None
        if self.open is None or self.open[0] != kind or self.open[2] != call:
            if call in self.calls:
                return  # a closed call's arguments are whole JSON: only whitespace can follow
            if self.held or self._unfinished():
                self.held.append((kind, value))
                return
            self._close()
            index, self.next_index = self.next_index, self.next_index + 1
            if kind == "tool_use":
                function = value.get("function") or {}
                block = {"type": "tool_use", "id": value.get("id") or "toolu_" + uuid.uuid4().hex[:24],
                         "name": function.get("name") or "", "input": {}}
                self.calls[call] = {"function": {"arguments": ""}}
            elif kind == "text":
                block = {"type": "text", "text": ""}
            else:
                block = {"type": "thinking", "thinking": "", "signature": ""}
            self.send("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})
            self.open = (kind, index, call)
        if kind == "tool_use":
            fragment = (value.get("function") or {}).get("arguments")
            if not fragment:
                return
            self.calls[call]["function"]["arguments"] += fragment
            delta = {"type": "input_json_delta", "partial_json": fragment}
        elif kind == "text":
            delta = {"type": "text_delta", "text": value}
        else:
            delta = {"type": "thinking_delta", "thinking": value}
        self.send("content_block_delta", {"type": "content_block_delta", "index": self.open[1], "delta": delta})
        if self.held and not self._unfinished():
            self._release()

    def _release(self):
        held, self.held = self.held, []
        for kind, value in held:
            self._write(kind, value)

    def text(self, piece):
        self._write("text", piece)

    def thinking(self, piece):
        self._write("thinking", piece)

    def tools(self, deltas):
        """OpenAI tool-call deltas (indexed; the first for a call carries id and name, every one may carry an
        arguments fragment) become tool_use blocks with input_json_delta events."""
        for delta in deltas:
            self._write("tool_use", delta)

    def finish(self, finish, measured, stopped):
        while self.held:
            self._close()  # a call whose arguments the reply left unfinished, as a cut reply does
            self._release()
        self._close()
        self.send("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason(finish, stopped),
                                                                       "stop_sequence": stopped}, "usage": usage(measured)})
        self.send("message_stop", {"type": "message_stop"})

    def error(self, message, status=500, request_id=None):
        self.send("error", error_body(status, message, request_id))
