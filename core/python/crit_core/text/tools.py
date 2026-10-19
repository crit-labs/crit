"""Tool calls in generated text, in the format the model's own chat template teaches (standard library only).

A chat template that renders `tools` also tells the model how to write a call. `tool_call_format(source)` names
that format from the template source, and `ToolCallParser` splits streamed reply text into content and complete
calls. The parser never executes anything and never invents a call: a block that does not parse, names an unknown
function or exceeds `parallel=False` stays in the content as written, so nothing malformed is exposed as a usable
call. FORMATS holds the syntaxes several model families share. An engine package whose model writes another syntax
supplies it as a `ToolFormat` (its parse function and markers) on `crit_core.serving.ResidentChat.tool_format`
instead of registering it here, so crit carries no model-specific parser; `ToolReply` assembles the calls for it.

- "xml-function" (Qwen3-Coder, Qwen3.5+, Nemotron 3): <tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE
  \\n</parameter>\\n</function>\\n</tool_call>. A value is a JSON value unless the function's schema declares the
  parameter a string (or does not declare it); an undecodable value stays a string, as the template writes strings
  verbatim.
- "json-object" (Hermes, Qwen2.5/Qwen3): <tool_call>\\n{"name": NAME, "arguments": {...}}\\n</tool_call>.
"""
import json
import re
import uuid
from typing import Callable, NamedTuple

from .errors import TextError

OPEN, CLOSE = "<tool_call>", "</tool_call>"
_FUNCTION = re.compile(r"\A\s*<function=([^>\n]+)>\n?(.*?)\n?</function>\s*\Z", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _reject_constant(value):
    raise ValueError("non-finite JSON number: " + value)


def _json(text):
    return json.loads(text, parse_constant=_reject_constant)


def _schema_type(tool, key):
    properties = ((tool.get("parameters") or {}).get("properties") or {}) if isinstance(tool, dict) else {}
    kind = properties[key].get("type") if isinstance(properties.get(key), dict) else None
    if isinstance(kind, list):
        return "string" if "string" in kind else next((k for k in kind if isinstance(k, str) and k != "null"), None)
    return kind if isinstance(kind, str) else None


def _xml_function(body, tools):
    match = _FUNCTION.match(body)
    if not match:
        return None
    name, inner = match.group(1).strip(), match.group(2)
    if name not in tools or _PARAMETER.sub("", inner).strip():
        return None
    arguments = {}
    for key, value in _PARAMETER.findall(inner):
        key = key.strip()
        if key in arguments:
            return None
        if _schema_type(tools[name], key) in (None, "string"):
            arguments[key] = value
            continue
        try:
            arguments[key] = _json(value)
        except ValueError:
            arguments[key] = value
    return name, arguments


def _json_object(body, tools):
    try:
        value = _json(body)
    except ValueError:
        return None
    if not isinstance(value, dict) or set(value) - {"name", "arguments"} or value.get("name") not in tools:
        return None
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = _json(arguments)
        except ValueError:
            return None
    return (value["name"], arguments) if isinstance(arguments, dict) else None


# name: (detect(template source), parse(block body, {function name: function spec}) -> (name, arguments) | None,
#        opening marker, closing marker)
FORMATS = {
    "xml-function": (lambda source: OPEN in source and "<function=" in source and "<parameter=" in source,
                     _xml_function, OPEN, CLOSE),
    "json-object": (lambda source: OPEN in source and CLOSE in source and "<function=" not in source,
                    _json_object, OPEN, CLOSE),
}


class ToolFormat(NamedTuple):
    """A tool-call syntax: `parse(block body, {function name: function spec}) -> (name, arguments) | None` for the
    text between `opening` and `closing`, and `markers`, every other literal marker the syntax writes inside calls.
    A tokenizer may hold markers as special tokens, which the reply text then has to keep (ResidentChat does)."""
    parse: Callable
    opening: str
    closing: str
    markers: tuple = ()


def _format(format):
    if isinstance(format, ToolFormat):
        return format
    if format not in FORMATS:
        raise TextError(f"unknown tool-call format {format!r}; registered: {', '.join(FORMATS)}")
    _, parse, opening, closing = FORMATS[format]
    return ToolFormat(parse, opening, closing)


def format_markers(format):
    """Every literal marker a format (a registered name or a ToolFormat) writes around and inside calls."""
    f = _format(format)
    return (f.opening, f.closing, *f.markers)


def tool_call_format(source):
    """The registered tool-call format a chat template teaches, or None when it renders no recognized one."""
    if not isinstance(source, str):
        raise TextError("chat template source must be a string")
    return next((name for name, (detect, *_) in FORMATS.items() if detect(source)), None)


def templates_tool_format(sources):
    """The registered tool-call format a model's chat template sources (an iterable of strings) teach, or None when
    they teach none; TextError when they teach different ones. The one lookup crit_core.serving and crit run package
    use."""
    formats = {tool_call_format(source) for source in sources} - {None}
    if len(formats) > 1:
        raise TextError("the model's chat templates disagree on the tool-call format: " + ", ".join(sorted(formats)))
    return formats.pop() if formats else None


def _held(buffer, opening):
    """Length of the buffer's suffix that could still start the opening marker."""
    for size in range(min(len(buffer), len(opening) - 1), 0, -1):
        if opening.startswith(buffer[-size:]):
            return size
    return 0


class ToolCallParser:
    """Incremental split of reply text into ("text", str) and ("call", {"name", "arguments"}) events.

    Text that could still begin a call is held back, and so is trailing whitespace: the space before, between and
    after calls belongs to the format, not the reply, and is dropped around every parsed call. Text that stays text
    keeps it. `tools` is the request's OpenAI tool list; calls must name one of its functions. `parallel=False`
    keeps only the first call.
    """

    def __init__(self, format, tools, *, parallel=True):
        f = _format(format)
        self._parse, self._open, self._close = f.parse, f.opening, f.closing
        self.tools = {t["function"]["name"]: t["function"] for t in tools}
        self.parallel = parallel
        self.calls = 0
        self._buffer, self._gap, self._inside, self._finished, self._after_call = "", "", False, False, False

    def _text(self, text, events):
        if self._after_call:
            text = text.lstrip()
        if text:
            events.append(("text", text))
            self._after_call = False

    def _block(self, body, events):
        call = self._parse(body, self.tools) if (self.parallel or not self.calls) else None
        if call is None:
            self._text(self._gap + self._open + body + self._close, events)
            return
        self.calls += 1
        self._after_call = True
        events.append(("call", {"name": call[0], "arguments": call[1]}))

    def push(self, text):
        if self._finished:
            raise TextError("the tool-call parser has finished")
        self._buffer += text
        events = []
        while True:
            if self._inside:
                end = self._buffer.find(self._close)
                if end < 0:
                    return events
                body, self._buffer, self._inside = self._buffer[:end], self._buffer[end + len(self._close):], False
                self._block(body, events)
                continue
            start = self._buffer.find(self._open)
            if start >= 0:
                before = self._buffer[:start]
                content = before.rstrip()
                self._text(content, events)
                self._gap, self._buffer, self._inside = before[len(content):], self._buffer[start + len(self._open):], True
                continue
            keep = _held(self._buffer, self._open)
            ready = self._buffer[:len(self._buffer) - keep]
            stripped = ready.rstrip()
            self._text(stripped, events)
            self._buffer = ready[len(stripped):] + self._buffer[len(ready):]
            return events

    def finish(self):
        """Flush held text. An unclosed call (a reply cut by its output limit) stays text."""
        events = self.push("")
        self._finished = True
        rest = (self._gap + self._open + self._buffer) if self._inside else self._buffer
        self._text(rest, events)
        self._buffer = ""
        return events


class ToolReply:
    """One reply's text split into content and OpenAI tool calls (`{id, type, function: {name, arguments}}`, JSON
    string arguments). `emit` (optional) receives content text and, as each call's block closes, a
    `{"tool_calls": [call with its index]}` delta, the chat-tool-stream-v1 shape. `request` is the chat request with
    its `tools` and `parallel_tool_calls`."""

    def __init__(self, format, request, emit=None):
        self.parser = ToolCallParser(format, request["tools"], parallel=request.get("parallel_tool_calls", True))
        self.emit, self.content, self.calls = emit, "", []

    def _apply(self, events):
        for kind, value in events:
            if kind == "text":
                self.content += value
                if self.emit:
                    self.emit(value)
                continue
            call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                    "function": {"name": value["name"],
                                 "arguments": json.dumps(value["arguments"], ensure_ascii=False, allow_nan=False)}}
            self.calls.append(call)
            if self.emit:
                self.emit({"tool_calls": [dict(call, index=len(self.calls) - 1)]})

    def push(self, text):
        self._apply(self.parser.push(text))

    def finish(self):
        """Flush; returns (content, calls)."""
        self._apply(self.parser.finish())
        return self.content, self.calls


__all__ = ["FORMATS", "ToolCallParser", "ToolFormat", "ToolReply", "format_markers", "templates_tool_format",
           "tool_call_format"]
