"""Tool calls in generated text, in the format the model's own chat template teaches (standard library only).

A chat template that renders `tools` also tells the model how to write a call. `tool_call_format(source)` names
that format from the template source, and `ToolCallParser` splits streamed reply text into content and complete
calls. The parser never executes anything and never invents a call: a block that does not parse, names an unknown
function or exceeds `parallel=False` stays in the content as written, so nothing malformed is exposed as a usable
call. FORMATS holds the syntaxes several model families share. An engine package whose model writes another syntax
supplies it as a `ToolFormat` (its parse function and markers) on `crit_core.serving.ResidentChat.tool_format`
instead of registering it here, so crit carries no model-specific parser; `ToolReply` assembles the calls for it.

`ToolReply` streams a call as the model writes it when its format reads an unfinished block (`ToolFormat.stream`;
both registered formats do): the call's id and name as soon as the name is written, then its arguments as they
become certain, so a client sees the call from its first tokens, not when its block closes. From then on the block
is that call: one that closes without parsing, or that the model ends its reply inside, fails the reply, and a reply
cut by its output limit inside it keeps the call with the arguments written so far, as OpenAI's API returns a call
cut by finish_reason "length".

`call_grammar` writes a registered format's calls as a grammar (crit_core.constraints), so an engine that keeps its
decoding to it makes the model call: tool_choice "required" or a named function.

- "xml-function" (Qwen3-Coder, Qwen3.5+, Nemotron 3): <tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE
  \\n</parameter>\\n</function>\\n</tool_call>. A value is a JSON value unless the function's schema declares the
  parameter a string (or does not declare it); an undecodable value stays a string, as the template writes strings
  verbatim.
- "json-object" (Hermes, Qwen2.5/Qwen3): <tool_call>\\n{"name": NAME, "arguments": {...}}\\n</tool_call>. A call
  that repeats a key is no call. It streams when written name first with an arguments object.
"""
import json
import re
import uuid
from typing import Callable, NamedTuple

from .errors import TextError
from .markers import MarkerBuffer, marker_ids

OPEN, CLOSE = "<tool_call>", "</tool_call>"
_FUNCTION = re.compile(r"\A\s*<function=([^>\n]+)>\n?(.*?)\n?</function>\s*\Z", re.S)
_FUNCTION_START = re.compile(r"\A\s*<function=([^>\n]+)>")
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
_PARAMETER_START = re.compile(r"<parameter=([^>\n]+)>\n?")
_PARAMETER_END = "\n</parameter>"
_NAME = re.compile(r'\A\s*\{\s*"name"\s*:\s*"([A-Za-z0-9_-]{1,64})"')
_ARGUMENTS = re.compile(r'\s*,\s*"arguments"\s*:\s*\{')


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


def _dumps(value):
    """JSON as a call's arguments carry it."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _xml_value(tool, key, value):
    if _schema_type(tool, key) in (None, "string"):
        return value
    try:
        return _json(value)
    except ValueError:
        return value


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
        arguments[key] = _xml_value(tools[name], key, value)
    return name, arguments


def _xml_function_stream(body, tools):
    """The name and the start of the arguments of an unfinished xml-function block: every finished parameter, and
    the text so far of a string parameter."""
    match = _FUNCTION_START.match(body)
    if not match or match.group(1).strip() not in tools:
        return None
    name, inner, parts, end = match.group(1).strip(), body[match.end():], [], 0
    keys = set()
    for parameter in _PARAMETER.finditer(inner):
        key = parameter.group(1).strip()
        if key in keys:  # the block will not parse
            return name, "{" + ", ".join(parts)
        keys.add(key)
        parts.append(_dumps(key) + ": " + _dumps(_xml_value(tools[name], key, parameter.group(2))))
        end = parameter.end()
    arguments = "{" + ", ".join(parts)
    opening = _PARAMETER_START.search(inner, end)
    if opening is None or opening.group(1).strip() in keys:
        return name, arguments
    key = opening.group(1).strip()
    arguments += (", " if parts else "") + _dumps(key) + ": "
    if _schema_type(tools[name], key) not in (None, "string"):
        return name, arguments  # a JSON value is written when its parameter closes
    value = inner[opening.end():]
    # Hold what may still turn out to be the newline and closing tag the value ends with.
    held = next((size for size in range(min(len(value), len(_PARAMETER_END)), 0, -1)
                 if _PARAMETER_END.startswith(value[-size:]) or _PARAMETER_END[1:].startswith(value[-size:])), 0)
    return name, arguments + _dumps(value[:len(value) - held])[:-1]


def _json_object(body, tools):
    try:
        value = json.loads(body, parse_constant=_reject_constant, object_pairs_hook=_unique)
    except ValueError:
        return None
    if not isinstance(value, dict) or set(value) - {"name", "arguments"} or value.get("name") not in tools:
        return None
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments, parse_constant=_reject_constant, object_pairs_hook=_unique)
        except ValueError:
            return None
    return (value["name"], arguments) if isinstance(arguments, dict) else None


def _unique(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("a repeated key")
    return value


_DECODER = json.JSONDecoder(parse_constant=_reject_constant, object_pairs_hook=_unique)


def _json_object_stream(body, tools):
    """The name and the start of the arguments of an unfinished json-object block written name first: every finished
    member of an arguments object, then the key of the member being written and, for a string, its text so far."""
    match = _NAME.match(body)
    if not match or match.group(1) not in tools:
        return None
    opening = _ARGUMENTS.match(body, match.end())
    if opening is None:
        return match.group(1), "{"  # arguments not yet begun, or a JSON string decoded once the block closes
    text, position, parts, keys = body, opening.end(), [], set()
    while True:
        position = _space(text, position + (1 if parts and text.startswith(",", position) else 0))
        try:
            key, after = _DECODER.raw_decode(text, position)
        except ValueError:
            break
        colon = _space(text, after)
        if not isinstance(key, str) or key in keys or not text.startswith(":", colon):
            break
        member = _dumps(key) + ": "
        try:
            value, after = _DECODER.raw_decode(text, _space(text, colon + 1))
        except ValueError:
            partial = _partial_string(text, _space(text, colon + 1))
            parts.append(member + ("" if partial is None else _dumps(partial)[:-1]))
            break
        following = _space(text, after)
        if following < len(text) and text[following] not in ",}":
            break  # the block will not parse
        if following == len(text) and type(value) in (int, float):
            parts.append(member)  # a number may still go on
            break
        keys.add(key)
        parts.append(member + _dumps(value))
        if following == len(text) or text[following] == "}":
            break
        position = following
    return match.group(1), "{" + ", ".join(parts)


def _partial_string(text, start):
    """The text so far of a JSON string opened at `start`, without an escape it may still be writing; None when no
    string opens there."""
    if not text.startswith('"', start):
        return None
    content = text[start + 1:]
    for cut in range(min(len(content), 12) + 1):  # \ud83d\ude00, the longest escape, is 12 characters
        try:
            value = json.loads('"' + content[:len(content) - cut] + '"')
        except ValueError:
            continue
        if not value or not "\ud800" <= value[-1] <= "\udbff":  # half a surrogate pair is not yet a character
            return value
    return None


def _space(text, position):
    while position < len(text) and text[position] in " \t\n\r":
        position += 1
    return position


# Text in a call grammar: tab, newline, carriage return and printable ASCII, other characters as whole UTF-8 sequences.
# A JSON string holds the printable ASCII but " and \ as it is, UTF-8 sequences and escapes.
_TEXT = (9, 10, 13, *range(0x20, 0x7f))
_STRING = tuple(byte for byte in range(0x20, 0x7f) if byte not in (0x22, 0x5c))


class _Grammar:
    """A call grammar built with crit_core.constraints' rule builder: rules are ids, productions tuples of bytes and
    rules. Its text and JSON strings never hold `forbidden`, the markers that would end a call's block early (the
    parser takes the first one)."""

    def __init__(self, forbidden):
        from crit_core.constraints.compiler import Builder
        self.b, self.classes = Builder(), {}
        b, chars = self.b, self.chars
        tail = chars(range(0x80, 0xc0))
        self.utf8 = b.rule([(chars(range(0xc2, 0xe0)), tail), (0xe0, chars(range(0xa0, 0xc0)), tail),
                            (chars([*range(0xe1, 0xed), 0xee, 0xef]), tail, tail),
                            (0xed, chars(range(0x80, 0xa0)), tail), (0xf0, chars(range(0x90, 0xc0)), tail, tail),
                            (chars(range(0xf1, 0xf4)), tail, tail, tail), (0xf4, chars(range(0x80, 0x90)), tail, tail)])
        self.text = self.excluding(_TEXT, [self.utf8], forbidden)
        self.key = b.repeat(b.rule([(chars(set(range(0x21, 0x7f)) - {0x3c, 0x3e}),), (self.utf8,)]), 1)
        x, u = chars(b"0123456789ABCDEFabcdef"), (0x5c, 0x75)   # a hex digit; \u
        escape = b.rule([(0x5c, chars(b'"\\/bfnrt')), (*u, chars(b"0123456789ABCabcEeFf"), x, x, x),
                         (*u, chars(b"Dd"), chars(b"01234567"), x, x),   # no unpaired surrogates, as the compiler
                         (*u, chars(b"Dd"), chars(b"89ABab"), x, x, *u, chars(b"Dd"), chars(b"CDEFcdef"), x, x)])
        self.string = b.rule([(0x22, self.excluding(_STRING, [self.utf8, escape], forbidden), 0x22)])
        digit = chars(b"0123456789")
        natural = b.rule([(0x30,), (chars(b"123456789"), b.repeat(digit))])
        self.integer = b.rule([(natural,), (0x2d, natural)])
        self.number = b.rule([(self.integer, b.rule([(), (0x2e, b.repeat(digit, 1))]),
                               b.rule([(), (chars(b"Ee"), b.rule([(), (chars(b"+-"),)]), b.repeat(digit, 1))]))])
        self.boolean, self.null = b.rule([tuple(b"true"), tuple(b"false")]), b.rule([tuple(b"null")])
        self.comma = (0x2c, b.rule([(), (0x20,)]))   # "," and an optional space, as json.dumps and compact JSON write
        self.value, self.object = b.new(), b.new()
        self.array = self.sequence(0x5b, self.value, 0x5d)
        member = b.rule([(self.string, 0x3a, self.comma[1], self.value)])
        b.define(self.object, [(0x7b, 0x7d), (0x7b, self.items(member), 0x7d)])
        b.define(self.value, [(self.object,), (self.array,), (self.string,), (self.number,), (self.boolean,),
                              (self.null,)])

    def chars(self, values):
        key = frozenset(values)
        if key not in self.classes:
            self.classes[key] = self.b.chars(key)
        return self.classes[key]

    def items(self, item):
        """One or more `item`s apart by commas."""
        rule = self.b.new()
        return self.b.define(rule, [(item,), (rule, *self.comma, item)])

    def sequence(self, opening, item, closing):
        """A bracketed list of `item`s, maybe empty."""
        return self.b.rule([(opening, closing), (opening, self.items(item), closing)])

    def excluding(self, alphabet, others, forbidden):
        """Text of the `alphabet` bytes and the rules `others` (characters with none of the forbidden bytes) that
        holds no `forbidden` string (ASCII): one left-linear rule per state, the longest end of the text that may
        still grow into one of them."""
        states = sorted({""} | {word[:size] for word in forbidden for size in range(1, len(word))}, key=len)
        rules = {state: self.b.new() for state in states}
        choices = {state: [] for state in states}
        choices[""].append(())
        letters = set("".join(forbidden))
        for state in states:
            targets = {}
            for byte in alphabet:
                text = state + chr(byte)
                if chr(byte) not in letters:
                    targets.setdefault("", []).append(byte)
                elif not any(text.endswith(word) for word in forbidden):
                    targets.setdefault(next(text[start:] for start in range(len(text) + 1) if text[start:] in rules),
                                       []).append(byte)
            for target, values in targets.items():
                choices[target].append((rules[state], self.chars(values)))
            choices[""] += [(rules[state], other) for other in others]
        for state in states:
            self.b.define(rules[state], choices[state])
        return self.b.rule([(rules[state],) for state in states])

    def schema(self, schema, depth=0):
        """JSON values of `schema` as far as its structure goes: its const or enum values, its type or types, an
        object's declared properties and an array's items; any JSON value where it says no more."""
        if not isinstance(schema, dict) or depth > 32:
            return self.value
        values = [schema["const"]] if "const" in schema else schema.get("enum")
        if isinstance(values, list) and values:
            return self.b.rule([tuple(_dumps(value).encode("utf-8")) for value in values])
        kinds = schema.get("type", "object" if isinstance(schema.get("properties"), dict) else None)
        parts = []
        for kind in kinds if isinstance(kinds, list) else [kinds]:
            if kind == "object":
                parts.append(self.properties(schema, depth))
            elif kind == "array":
                parts.append(self.sequence(0x5b, self.schema(schema.get("items"), depth + 1), 0x5d))
            elif kind in ("string", "integer", "number", "boolean", "null"):
                parts.append(getattr(self, kind))
            else:
                return self.value
        return self.b.rule([(part,) for part in parts]) if len(parts) > 1 else parts[0] if parts else self.value

    def properties(self, schema, depth=0):
        """An object of the declared properties in their order, the required ones always; any object when it declares
        none."""
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return self.object
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        tails = {False: (), True: ()}   # the members after one, when a member came before it (True) or none did
        for key, value in reversed(properties.items()):
            member = (*(_dumps(key) + ":").encode("utf-8"), self.comma[1], self.schema(value, depth + 1))
            following = {}
            for seen in (False, True):
                choices = [(*(self.comma if seen else ()), *member, *tails[True])]
                following[seen] = (self.b.rule(choices + ([] if key in required else [tails[seen]])),)
            tails = following
        return self.b.rule([(0x7b, *tails[False], 0x7d)])


def _xml_function_call(grammar, function):
    """One xml-function call to `function`: its declared parameters in their order (the required ones always), any
    parameters when it declares none; a value is text unless its schema declares another type (_xml_value)."""
    parameters = function.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None

    def parameter(key, value):
        return (*b"<parameter=", *key, *b">\n", value, *b"\n</parameter>\n")

    if isinstance(properties, dict):
        required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
        body = []
        for key, schema in properties.items():
            if key and key == key.strip() and ">" not in key and "\n" not in key:  # a name the syntax can write
                value = grammar.text if _schema_type(function, key) in (None, "string") else grammar.schema(schema, 1)
                block = parameter(key.encode("utf-8"), value)
                body += block if key in required else (grammar.b.rule([(), block]),)
    else:
        body = (grammar.b.repeat(grammar.b.rule([parameter((grammar.key,), grammar.text)])),)
    return (*(OPEN + "\n<function=" + function["name"] + ">\n").encode("utf-8"), *body,
            *("</function>\n" + CLOSE).encode("utf-8"))


def _json_object_call(grammar, function):
    """One json-object call to `function`, written name first, its arguments shaped by its parameters schema."""
    parameters = function.get("parameters")
    arguments = grammar.properties(parameters, 1) if isinstance(parameters, dict) else grammar.object
    return (*(OPEN + '\n{"name": ' + _dumps(function["name"]) + ', "arguments": ').encode("utf-8"), arguments,
            *("}\n" + CLOSE).encode("utf-8"))


# name: (detect(template source), parse(block body, {function name: function spec}) -> (name, arguments) | None,
#        opening marker, closing marker, stream(unfinished block body, functions) -> (name, arguments start) | None,
#        (call(_Grammar, function spec) -> one call block's production, the strings no text or string in it holds))
FORMATS = {
    "xml-function": (lambda source: OPEN in source and "<function=" in source and "<parameter=" in source,
                     _xml_function, OPEN, CLOSE, _xml_function_stream, (_xml_function_call, ("</parameter>", CLOSE))),
    "json-object": (lambda source: OPEN in source and CLOSE in source and "<function=" not in source,
                    _json_object, OPEN, CLOSE, _json_object_stream, (_json_object_call, (CLOSE,))),
}


class ToolFormat(NamedTuple):
    """A tool-call syntax: `parse(block body, {function name: function spec}) -> (name, arguments) | None` for the
    text between `opening` and `closing`, and `markers`, every other literal marker the syntax writes inside calls.
    A tokenizer may hold markers as special tokens, which the reply text then has to keep (ResidentChat does).
    `delimiters="auto"` uses token identity for single special opening/closing tokens when a tokenizer is supplied;
    `delimiters="text"` always matches their spelling. The body parser still receives text.

    `stream(unfinished block body, functions) -> (name, arguments) | None` (optional) lets the call stream as it
    decodes: None until the body shows the function's name (one of `functions`), then that name and the start of its
    arguments as JSON text, which must begin every later return and the arguments `parse` gives the whole block,
    serialized as json.dumps(arguments, ensure_ascii=False) writes them ("{" when nothing more is certain). Without it
    a call streams when its block closes."""
    parse: Callable
    opening: str
    closing: str
    markers: tuple = ()
    delimiters: str = "auto"
    stream: Callable = None


def _format(format):
    if isinstance(format, ToolFormat):
        return format
    if format not in FORMATS:
        raise TextError(f"unknown tool-call format {format!r}; registered: {', '.join(FORMATS)}")
    _, parse, opening, closing, stream, _ = FORMATS[format]
    return ToolFormat(parse, opening, closing, stream=stream)


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


def call_grammar(format, tools, choice="required", parallel=True, *, reasoning=None, opened=False):
    """A reply that only calls tools, in registered format `format`, as a crit-byte-grammar-v1 artifact
    (crit_core.constraints): calls to `tools` (the request's OpenAI tools), or to the function a named `choice` names,
    one or with `parallel` one or more, written as the chat template writes them. Arguments follow the function's
    parameters schema as far as its structure goes (declared properties in their order, the required ones always,
    types, enums), and no text or string in a call holds a marker that would end its block early. `reasoning`, the
    (opening, closing) reasoning markers (ASCII), lets the model think before it calls; with `opened` the generation
    prompt ends with the opening marker, so the reply starts inside the reasoning. None for a format crit does not
    register (a package's own ToolFormat): it cannot write that syntax."""
    if not isinstance(format, str) or format not in FORMATS:
        return None
    call, forbidden = FORMATS[format][5]
    named = choice["function"]["name"] if isinstance(choice, dict) else None
    functions = [tool["function"] for tool in tools if named is None or tool["function"]["name"] == named]
    if not functions:
        raise TextError(f"tool_choice names {named!r}, which the tools do not offer")
    if opened and reasoning is None:
        raise TextError("a reply that starts inside its reasoning needs the reasoning markers")
    if reasoning is not None and not "".join(reasoning).isascii():
        raise TextError("a call grammar writes ASCII reasoning markers only")
    grammar = _Grammar(forbidden)
    b = grammar.b
    calls = b.rule([call(grammar, function) for function in functions])
    if parallel:   # one or more, a newline between two allowed
        one, calls = calls, b.new()
        b.define(calls, [(one,), (calls, one), (calls, 0x0a, one)])
    root = [(calls,)]
    if reasoning is not None:
        opening, closing = reasoning
        thought = (grammar.excluding(_TEXT, [grammar.utf8], (closing,)), *closing.encode("utf-8"),
                   b.rule([(), (0x0a,), (0x0a, 0x0a)]), calls)
        root = [thought] if opened else root + [(*opening.encode("utf-8"), *thought)]
    return b.artifact(b.rule(root))


class ToolCallParser:
    """Incremental split of reply text into ("text", str) and ("call", {"name", "arguments"}) events.

    Text that could still begin a call is held back, and so is trailing whitespace: the space before, between and
    after calls belongs to the format, not the reply, and is dropped around every parsed call. Text that stays text
    keeps it. `tools` is the request's OpenAI tool list; calls must name one of its functions. `parallel=False`
    keeps only the first call. With `tokenizer`, special delimiters require TokenText events from MarkerDecoder.
    """

    def __init__(self, format, tools, *, parallel=True, tokenizer=None):
        f = _format(format)
        self.token_ids = marker_ids(tokenizer, (f.opening, f.closing), f.delimiters)
        self._parse, self._open, self._close = f.parse, f.opening, f.closing
        self.tools = {t["function"]["name"]: t["function"] for t in tools}
        self.parallel = parallel
        self.calls = 0
        self._buffer, self._gap, self._inside, self._finished, self._after_call = MarkerBuffer(), "", False, False, False

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
        self._buffer.append(text)
        events = []
        while True:
            if self._inside:
                end = self._buffer.find(self._close, self.token_ids)
                if end < 0:
                    return events
                body = self._buffer.text[:end]
                self._buffer, self._inside = self._buffer.slice(end + len(self._close)), False
                self._block(body, events)
                continue
            start = self._buffer.find(self._open, self.token_ids)
            if start >= 0:
                before = self._buffer.text[:start]
                content = before.rstrip()
                self._text(content, events)
                self._gap = before[len(content):]
                self._buffer, self._inside = self._buffer.slice(start + len(self._open)), True
                continue
            keep = self._buffer.held(self._open, self.token_ids)
            ready = self._buffer.text[:len(self._buffer.text) - keep]
            stripped = ready.rstrip()
            self._text(stripped, events)
            self._buffer = self._buffer.slice(len(stripped))
            return events

    def pending(self):
        """The body so far of the open call block, without text that may begin its closing marker; None outside a
        block or in one that cannot become a call (a second call with parallel=False)."""
        if not self._inside or not (self.parallel or not self.calls):
            return None
        text = self._buffer.text
        return text[:len(text) - self._buffer.held(self._close, self.token_ids)]

    def finish(self, unclosed=True):
        """Flush held text. An unclosed call (a reply cut by its output limit) stays text, or with `unclosed` False is
        dropped (ToolReply keeps a call it has streamed)."""
        events = self.push("")
        self._finished = True
        rest = ((self._gap + self._open + self._buffer.text if unclosed else "") if self._inside
                else self._buffer.text)
        self._text(rest, events)
        self._buffer = MarkerBuffer()
        return events


class ToolReply:
    """One reply's text split into content and OpenAI tool calls (`{id, type, function: {name, arguments}}`, JSON
    string arguments). `emit` (optional) receives content text and the calls as chat-tool-stream-v1 deltas: a call's
    first delta `{"tool_calls": [call with its index]}`, then `{"tool_calls": [{"index", "function": {"arguments"}}]}`
    with each further piece of its arguments. With the format's `stream` a call goes out as the model writes it,
    otherwise when its block closes. `request` is the chat request with its `tools` and `parallel_tool_calls`."""

    def __init__(self, format, request, emit=None, *, tokenizer=None):
        self.format = _format(format)
        self.parser = ToolCallParser(self.format, request["tools"], parallel=request.get("parallel_tool_calls", True),
                                     tokenizer=tokenizer)
        self.emit, self.content, self.calls = emit, "", []
        self.open = None  # the call streaming while its block is open

    def _send(self, delta):
        if self.emit:
            self.emit({"tool_calls": [json.loads(json.dumps(delta))]})

    def _extend(self, name, arguments):
        """Stream what the open call's arguments add to what has gone out."""
        function = self.open["function"]
        if name != function["name"] or not arguments.startswith(function["arguments"]):
            raise TextError(f"the call to {function['name']} changed after it streamed")
        if len(arguments) > len(function["arguments"]):
            self._send({"index": len(self.calls), "function": {"arguments": arguments[len(function["arguments"]):]}})
            function["arguments"] = arguments

    def _apply(self, events):
        for kind, value in events:
            if kind == "text":
                if self.open is not None:
                    raise TextError(f"the model wrote a malformed call to {self.open['function']['name']}")
                self.content += value
                if self.emit:
                    self.emit(value)
                continue
            arguments = _dumps(value["arguments"])
            if self.open is not None:
                self._extend(value["name"], arguments)
                call, self.open = self.open, None
            else:
                call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                        "function": {"name": value["name"], "arguments": arguments}}
                self._send(dict(call, index=len(self.calls)))
            self.calls.append(call)

    def _stream(self):
        body = self.parser.pending()
        found = self.format.stream(body, self.parser.tools) if body is not None and self.format.stream else None
        if found is None:
            return
        name, arguments = found
        if self.open is not None:
            self._extend(name, arguments)
        elif name in self.parser.tools:
            self.open = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                         "function": {"name": name, "arguments": arguments}}
            self._send(dict(self.open, index=len(self.calls)))

    def push(self, text):
        self._apply(self.parser.push(text))
        self._stream()

    def finish(self, cut=False):
        """Flush; returns (content, calls). A reply that ends inside a streamed call keeps it with the arguments
        written so far when the reply was `cut` by its output limit, and otherwise wrote a malformed call."""
        self._apply(self.parser.finish(unclosed=self.open is None))
        if self.open is not None:
            if not cut:
                raise TextError(f"the model ended its reply inside its call to {self.open['function']['name']}")
            self.calls.append(self.open)
            self.open = None
        return self.content, self.calls


__all__ = ["FORMATS", "ToolCallParser", "ToolFormat", "ToolReply", "call_grammar", "format_markers",
           "templates_tool_format", "tool_call_format"]
