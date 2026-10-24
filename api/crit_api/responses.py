"""The OpenAI Responses API over the same engine, statelessly: POST /v1/responses, so clients that speak it (Codex
CLI, the OpenAI Agents SDK, LangChain and LibreChat in their Responses modes) run on a crit engine.

A Responses request is translated into the chat request crit serve already validates (server.Api.completion_request):
`instructions` becomes the leading system message, `input` (a string, or message, function_call,
function_call_output and reasoning items) becomes the conversation, function tools and `tool_choice` map across,
`text.format` becomes `response_format`, `max_output_tokens` becomes `max_tokens` and `reasoning.effort` becomes
`reasoning_effort`. crit serve keeps no responses, so stored state (`previous_response_id`, `conversation`,
`background`) and OpenAI's built-in tools are refused. The reply comes back as a Response (reasoning, message and
function_call output items, status completed or incomplete, usage), or as the Responses event stream: response.created
and response.in_progress, the output item, content part and delta events per item, then response.completed,
response.incomplete or response.failed, every event numbered by sequence_number.
"""
import time
import uuid

from .options import RequestError, unknown_parameters

KNOWN = {"model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls", "text", "max_output_tokens",
         "reasoning", "stream", "temperature", "top_p"}
# Accepted without effect: storage, caching and client metadata, not the model's generation.
IGNORED = {"store", "metadata", "include", "truncation", "user", "service_tier", "safety_identifier",
           "prompt_cache_key", "prompt_cache_retention", "stream_options", "max_tool_calls"}
UNSUPPORTED = {"previous_response_id": "crit serve keeps no responses: send the whole conversation as input",
               "conversation": "crit serve keeps no conversations: send the whole conversation as input",
               "prompt": "crit serve keeps no prompt templates: send the instructions and input",
               "background": "Background responses need stored responses, which crit serve does not keep",
               "top_logprobs": "Responses carry no log-probabilities on a crit engine; use /v1/chat/completions"}
STATELESS = "crit serve keeps no responses: /v1/responses is stateless and serves only POST /v1/responses"


def _content(value, where):
    """crit's chat content for an item's content: a string, or text and image parts."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise RequestError(f"{where} must be a string or a non-empty list of content parts", param=where)
    parts = []
    for number, part in enumerate(value):
        at = f"{where}[{number}]"
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in {"input_text", "output_text"} and isinstance(part.get("text"), str):
            parts.append({"type": "text", "text": part["text"]})
        elif kind == "refusal" and isinstance(part.get("refusal"), str):
            parts.append({"type": "text", "text": part["refusal"]})
        elif kind == "input_image":
            if not isinstance(part.get("image_url"), str):
                raise RequestError(f"{at}: an input_image needs an image_url (crit serve keeps no files)", param=at)
            image = {"url": part["image_url"]}
            if part.get("detail") is not None:
                image["detail"] = part["detail"]
            parts.append({"type": "image_url", "image_url": image})
        elif kind == "input_file":
            raise RequestError(f"{at}: file inputs are not supported; pass the text", param=at,
                               code="unsupported_parameter")
        else:
            raise RequestError(f"{at}: unknown content part type {kind!r}", param=at)
    return parts


def _text(value, where):
    parts = _content(value, where)
    if isinstance(parts, str):
        return parts
    if any(part["type"] != "text" for part in parts):
        raise RequestError(f"{where}: an assistant message holds text only", param=where)
    return "".join(part["text"] for part in parts)


def _reasoning(item, where):
    """A reasoning item's text: its reasoning_text content, else its summary (what crit serve returns)."""
    parts = item.get("content") or item.get("summary") or []
    if not isinstance(parts, list) or not all(isinstance(p, dict) and isinstance(p.get("text"), str) for p in parts):
        raise RequestError(f"{where}: a reasoning item's summary and content are lists of text parts", param=where)
    return "\n\n".join(part["text"] for part in parts)


def _assistant(result, reasoning):
    """The assistant row an assistant message or function call joins: the last row when it is the assistant's (one
    turn holds its text and calls), else a new one. Reasoning items before it become its reasoning_content."""
    if not result or result[-1]["role"] != "assistant":
        result.append({"role": "assistant", "content": ""})
    row = result[-1]
    if reasoning:
        earlier = [row["reasoning_content"]] if "reasoning_content" in row else []
        row["reasoning_content"] = "\n\n".join(earlier + reasoning)
    return row


def messages(instructions, value):
    """crit's chat messages for a Responses request's instructions and input."""
    result = []
    if instructions is not None:
        if not isinstance(instructions, str):
            raise RequestError("instructions must be a string", param="instructions")
        if instructions:
            result.append({"role": "system", "content": instructions})
    if isinstance(value, str):
        return result + [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise RequestError("input must be a string or a non-empty list of items", param="input")
    reasoning = []  # reasoning texts waiting for the assistant turn they belong to
    for index, item in enumerate(value):
        where = f"input[{index}]"
        if not isinstance(item, dict):
            raise RequestError(f"{where} must be an object", param=where)
        kind, role = item.get("type", "message"), item.get("role")
        if kind == "reasoning":
            text = _reasoning(item, where)
            if text:
                reasoning.append(text)
            continue
        if kind == "message" and role in {"user", "system", "developer"}:
            result.append({"role": role, "content": _content(item.get("content"), where + ".content")})
        elif kind == "message" and role == "assistant":
            text, row = _text(item.get("content"), where + ".content"), _assistant(result, reasoning)
            row["content"] = "\n\n".join(part for part in (row["content"], text) if part)
        elif kind == "message":
            raise RequestError(f"{where}.role must be user, assistant, system or developer", param=where)
        elif kind == "function_call":
            if not all(isinstance(item.get(key), str) for key in ("call_id", "name", "arguments")):
                raise RequestError(f"{where}: a function_call needs call_id, name and arguments strings", param=where)
            function = {"name": item["name"], "arguments": item["arguments"]}
            _assistant(result, reasoning).setdefault("tool_calls", []).append(
                {"id": item["call_id"], "type": "function", "function": function})
        elif kind == "function_call_output":
            if not isinstance(item.get("call_id"), str):
                raise RequestError(f"{where}: a function_call_output needs call_id", param=where)
            result.append({"role": "tool", "tool_call_id": item["call_id"],
                           "content": _content(item.get("output"), where + ".output")})
        elif kind == "item_reference":
            raise RequestError(f"{where}: item references need stored responses, which crit serve does not keep; "
                               "send the items", param=where, code="unsupported_parameter")
        else:
            raise RequestError(f"{where}: {kind!r} items are not supported; send messages, function calls, their "
                               "outputs and reasoning", param=where, code="unsupported_parameter")
        reasoning = []
    return result


def tools(value):
    if value is None:
        return None
    if not isinstance(value, list):
        raise RequestError("tools must be a list", param="tools")
    result = []
    for index, tool in enumerate(value):
        where = f"tools[{index}]"
        if not isinstance(tool, dict):
            raise RequestError(f"{where} must be an object", param=where)
        if tool.get("type") != "function":
            raise RequestError(f"{where}: OpenAI's built-in tools ({tool.get('type')}) are not available on a crit "
                               "engine; define a function tool", param=where, code="unsupported_parameter")
        function = {k: tool[k] for k in ("name", "description", "parameters", "strict") if tool.get(k) is not None}
        result.append({"type": "function", "function": function})
    return result


def tool_choice(value):
    """The chat tool_choice for a Responses one: auto, none, required or {type: function, name}."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("type") == "function" and isinstance(value.get("name"), str):
        return {"type": "function", "function": {"name": value["name"]}}
    raise RequestError("tool_choice must be auto, none, required or {type: function, name}", param="tool_choice",
                       code="unsupported_parameter")


def response_format(text):
    """The chat response_format for the request's text settings, or None."""
    if text is None:
        return None
    if not isinstance(text, dict) or set(text) - {"format", "verbosity"}:
        raise RequestError("text takes an object with format and verbosity", param="text")
    if text.get("verbosity") is not None:
        raise RequestError("This engine does not support text verbosity", param="text.verbosity",
                           code="unsupported_parameter")
    fmt = text.get("format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") not in {"text", "json_object", "json_schema"}:
        raise RequestError("text.format.type must be text, json_object or json_schema", param="text.format")
    if fmt["type"] == "json_schema":
        return {"type": "json_schema", "json_schema": {k: v for k, v in fmt.items() if k != "type"}}
    return {"type": fmt["type"]}


def chat_body(body, lenient=False):
    """The OpenAI-shaped chat body crit serve validates for a Responses request body, the parameters ignored (with
    `lenient`, crit serve --lenient, unknown ones too) and its thinking controls: none, since reasoning.effort maps
    onto the chat body's reasoning_effort, which the chat request applies (server.Api.completion_request)."""
    if not isinstance(body, dict):
        raise RequestError("The request body must be a JSON object")
    for key, message in UNSUPPORTED.items():
        if body.get(key) not in (None, False):
            raise RequestError(message, param=key, code="unsupported_parameter")
    unknown = unknown_parameters(body, KNOWN | IGNORED | set(UNSUPPORTED), lenient)
    result = {"model": body.get("model"), "messages": messages(body.get("instructions"), body.get("input"))}
    for key, name in (("stream", "stream"), ("max_output_tokens", "max_tokens"), ("temperature", "temperature"),
                      ("top_p", "top_p"), ("parallel_tool_calls", "parallel_tool_calls")):
        if body.get(key) is not None:
            result[name] = body[key]
    functions = tools(body.get("tools"))
    if functions is not None:
        result["tools"] = functions
    choice = tool_choice(body.get("tool_choice"))
    if choice is not None and (functions or choice != "auto"):  # auto without tools, as clients echo it, is none
        result["tool_choice"] = choice
    constraint = response_format(body.get("text"))
    if constraint is not None:
        result["response_format"] = constraint
    ignored = sorted(k for k in body if k in IGNORED) + unknown
    reasoning = body.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict) or set(reasoning) - {"effort", "summary", "generate_summary"}:
            raise RequestError("reasoning takes an object with effort and summary", param="reasoning")
        if reasoning.get("effort") is not None:
            result["reasoning_effort"] = reasoning["effort"]
        # The engine's reasoning always comes back as the reasoning item's summary text.
        ignored += sorted("reasoning." + k for k in ("summary", "generate_summary") if reasoning.get(k) is not None)
    return result, ignored, None


def usage(measured):
    """Responses usage: input tokens, the ones the engine's prompt cache served as cached_tokens, and output tokens.
    Engines report neither cache writes nor the reasoning share of their one output count, so both are 0."""
    prompt, output = (measured["prompt_tokens"], measured["output_tokens"]) if measured else (0, 0)
    cached = prompt - measured["prefill_tokens"] if measured and measured.get("prefill_tokens") is not None else 0
    return {"input_tokens": prompt, "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
            "output_tokens": output, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": prompt + output}


def output_text(text):
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def reasoning_item(ident, text, status="completed"):
    return {"id": ident, "type": "reasoning", "summary": [{"type": "summary_text", "text": text}], "status": status}


def message_item(ident, text, status="completed"):
    return {"id": ident, "type": "message", "role": "assistant", "status": status, "content": [output_text(text)]}


def call_item(ident, call, status="completed"):
    return {"id": ident, "type": "function_call", "status": status, "call_id": call["id"],
            "name": call["function"]["name"], "arguments": call["function"]["arguments"]}


def response(ident, created, source, model, status, output, measured=None, error=None):
    """A Response object; `source` is the request body, whose settings it echoes. Usage comes with a finished one."""
    reasoning, text = source.get("reasoning") or {}, source.get("text") or {}
    return {"id": ident, "object": "response", "created_at": created, "status": status, "error": error,
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "instructions": source.get("instructions"), "max_output_tokens": source.get("max_output_tokens"),
            "model": model, "output": output, "parallel_tool_calls": source.get("parallel_tool_calls") is not False,
            "previous_response_id": None,
            "reasoning": {"effort": reasoning.get("effort"), "summary": reasoning.get("summary")}, "store": False,
            "temperature": source.get("temperature"), "text": {"format": text.get("format") or {"type": "text"}},
            "tool_choice": source.get("tool_choice") or "auto", "tools": source.get("tools") or [],
            "top_p": source.get("top_p"), "truncation": "disabled",
            "usage": usage(measured) if status in {"completed", "incomplete"} else None,
            "metadata": source.get("metadata") or {}}


def reply(source, model, finish, measured, text, calls, reasoning):
    """A complete Response: the reasoning, the message (also when the reply is empty), then the function calls; a
    reply the output cap cut is incomplete, and so is its last item."""
    items = [reasoning_item("rs_" + uuid.uuid4().hex, reasoning)] if reasoning else []
    if text or not (calls or reasoning):
        items.append(message_item("msg_" + uuid.uuid4().hex, text))
    items += [call_item("fc_" + uuid.uuid4().hex, call) for call in calls or []]
    if finish == "length":
        items[-1]["status"] = "incomplete"
    return response("resp_" + uuid.uuid4().hex, int(time.time()), source, model,
                    "incomplete" if finish == "length" else "completed", items, measured)


class Stream:
    """Writes the Responses event stream for one request through `send(event, data)`, numbering the events by
    sequence_number: output items open as reasoning, text or function calls arrive, in the order the engine produces
    them; a reasoning or message item closes before the next kind, function calls at the end."""

    def __init__(self, send, source, model):
        self.send, self.source, self.model = send, source, model
        self.ident, self.created = "resp_" + uuid.uuid4().hex, int(time.time())
        self.sequence = 0
        self.output = []   # each output item once done, None while it is written
        self.open = None   # (kind, output index, item id, pieces) of the reasoning or message item being written
        self.calls = {}    # tool call index -> (output index, item id, call, argument pieces)

    def event(self, kind, **fields):
        self.send(kind, {"type": kind, **fields, "sequence_number": self.sequence})
        self.sequence += 1

    def response(self, status, measured=None, error=None):
        return response(self.ident, self.created, self.source, self.model, status,
                        [item for item in self.output if item is not None], measured, error)

    def start(self):
        for kind in ("response.created", "response.in_progress"):
            self.event(kind, response=self.response("in_progress"))

    def _add(self, item):
        self.output.append(None)
        self.event("response.output_item.added", output_index=len(self.output) - 1, item=item)
        return len(self.output) - 1

    def _done(self, index, item):
        self.output[index] = item
        self.event("response.output_item.done", output_index=index, item=item)

    def _ensure(self, kind):
        if self.open is not None and self.open[0] == kind:
            return self.open
        self._close()
        if kind == "reasoning":
            ident = "rs_" + uuid.uuid4().hex
            index = self._add({"id": ident, "type": "reasoning", "summary": [], "status": "in_progress"})
            self.event("response.reasoning_summary_part.added", item_id=ident, output_index=index, summary_index=0,
                       part={"type": "summary_text", "text": ""})
        else:
            ident = "msg_" + uuid.uuid4().hex
            index = self._add({"id": ident, "type": "message", "role": "assistant", "status": "in_progress",
                               "content": []})
            self.event("response.content_part.added", item_id=ident, output_index=index, content_index=0,
                       part=output_text(""))
        self.open = (kind, index, ident, [])
        return self.open

    def _close(self, status="completed"):
        if self.open is None:
            return
        (kind, index, ident, pieces), self.open = self.open, None
        text = "".join(pieces)
        if kind == "reasoning":
            self.event("response.reasoning_summary_text.done", item_id=ident, output_index=index, summary_index=0,
                       text=text)
            self.event("response.reasoning_summary_part.done", item_id=ident, output_index=index, summary_index=0,
                       part={"type": "summary_text", "text": text})
            self._done(index, reasoning_item(ident, text, status))
        else:
            self.event("response.output_text.done", item_id=ident, output_index=index, content_index=0, text=text,
                       logprobs=[])
            self.event("response.content_part.done", item_id=ident, output_index=index, content_index=0,
                       part=output_text(text))
            self._done(index, message_item(ident, text, status))

    def text(self, piece):
        _, index, ident, pieces = self._ensure("message")
        pieces.append(piece)
        self.event("response.output_text.delta", item_id=ident, output_index=index, content_index=0, delta=piece,
                   logprobs=[])

    def reasoning(self, piece):
        _, index, ident, pieces = self._ensure("reasoning")
        pieces.append(piece)
        self.event("response.reasoning_summary_text.delta", item_id=ident, output_index=index, summary_index=0,
                   delta=piece)

    def tools(self, deltas):
        """OpenAI tool-call deltas (indexed; the first for a call carries id and name, every one may carry an
        arguments fragment) become function_call items with function_call_arguments.delta events."""
        self._close()
        for delta in deltas:
            position = delta.get("index", 0)
            function = delta.get("function") or {}
            if position not in self.calls:
                ident = "fc_" + uuid.uuid4().hex
                call = {"id": delta.get("id") or "call_" + uuid.uuid4().hex[:24],
                        "function": {"name": function.get("name") or "", "arguments": ""}}
                self.calls[position] = (self._add(call_item(ident, call, "in_progress")), ident, call, [])
            index, ident, _, pieces = self.calls[position]
            if function.get("arguments"):
                pieces.append(function["arguments"])
                self.event("response.function_call_arguments.delta", item_id=ident, output_index=index,
                           delta=function["arguments"])

    def _close_call(self, position, status):
        index, ident, call, pieces = self.calls.pop(position)
        arguments = "".join(pieces)
        self.event("response.function_call_arguments.done", item_id=ident, output_index=index, arguments=arguments)
        call = {"id": call["id"], "function": dict(call["function"], arguments=arguments)}
        self._done(index, call_item(ident, call, status))

    def finish(self, finish, measured, telemetry=None):
        """Close the open items in output order, the last one incomplete when the output cap cut the reply, then end
        with the whole Response (an empty message when the engine wrote nothing)."""
        incomplete = finish == "length"
        if not self.output:
            self._ensure("message")
        last = len(self.output) - 1
        writing = [(entry[0], position) for position, entry in self.calls.items()]
        for index, position in sorted(writing + ([(self.open[1], None)] if self.open else [])):
            status = "incomplete" if incomplete and index == last else "completed"
            if position is None:
                self._close(status)
            else:
                self._close_call(position, status)
        value = self.response("incomplete" if incomplete else "completed", measured)
        if telemetry is not None:
            value["crit_telemetry"] = telemetry
        self.event("response.incomplete" if incomplete else "response.completed", response=value)

    def fail(self, message, code, request_id=None):
        """End with the failed Response; its event carries crit's id of the request as request_id, as every dialect's
        stream error does."""
        fields = {} if request_id is None else {"request_id": request_id}
        self.event("response.failed", response=self.response("failed", error={"code": code, "message": message}),
                   **fields)
