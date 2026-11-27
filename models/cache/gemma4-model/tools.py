"""Gemma 4's tool calls and reasoning as its chat template writes them, for crit serve's parsers (the manifest's
tool_format and reasoning_format).

A call (the template's format_argument):

    <|tool_call>call:NAME{key:VALUE,key:VALUE}<tool_call|>

VALUE is a string between <|"|> markers, true or false, null, a number written bare, an object {key:VALUE,...} (keys
bare in a call's arguments; the template quotes them inside nested objects of a declaration, which parse too) or an
array [VALUE,...]. A block that does not parse in full, or names a function the request does not offer, is no call
(crit_core.text.tools keeps it as content). The reasoning is the thought channel the template opens before a turn's
calls, <|channel>thought ... <channel|>: REASONING_FORMAT, the channel's name its label.

Standard library and crit_core only: the resident engine imports this file to parse its replies."""
import json
import re

from crit_core.text.reasoning import ReasoningFormat
from crit_core.text.tools import ToolFormat

OPEN, CLOSE, QUOTE = "<|tool_call>", "<tool_call|>", '<|"|>'


class _Text:
    def __init__(self, text):
        self.text, self.at = text, 0

    def peek(self, token):
        return self.text.startswith(token, self.at)

    def take(self, token):
        if not self.peek(token):
            raise ValueError(f"expected {token!r} at {self.at}")
        self.at += len(token)

    def spaces(self):
        while self.at < len(self.text) and self.text[self.at] in " \t\r\n":
            self.at += 1


def _string(t):
    t.take(QUOTE)
    end = t.text.find(QUOTE, t.at)
    if end < 0:
        raise ValueError("an unclosed string")
    value, t.at = t.text[t.at:end], end + len(QUOTE)
    return value


def _bare(t, stops):
    start = t.at
    while t.at < len(t.text) and t.text[t.at] not in stops:
        t.at += 1
    word = t.text[start:t.at].strip()
    if not word:
        raise ValueError(f"an empty value at {start}")
    if word in ("true", "false"):
        return word == "true"
    if word == "null":
        return None
    try:
        return json.loads(word)   # a number as JSON reads it
    except ValueError:
        return word               # the template writes any other value as it is


def _value(t, stops):
    t.spaces()
    if t.peek(QUOTE):
        return _string(t)
    if t.peek("{"):
        return _object(t)
    if t.peek("["):
        return _array(t)
    return _bare(t, stops)


def _key(t):
    t.spaces()
    if t.peek(QUOTE):
        return _string(t)
    start = t.at
    while t.at < len(t.text) and t.text[t.at] not in ":{}[],":
        t.at += 1
    key = t.text[start:t.at].strip()
    if not key:
        raise ValueError(f"an empty key at {start}")
    return key


def _object(t):
    t.take("{")
    found = {}
    t.spaces()
    if t.peek("}"):
        t.take("}")
        return found
    while True:
        key = _key(t)
        t.spaces()
        t.take(":")
        if key in found:
            raise ValueError(f"the key {key!r} twice")
        found[key] = _value(t, ",}")
        t.spaces()
        if t.peek(","):
            t.take(",")
            continue
        t.take("}")
        return found


def _array(t):
    t.take("[")
    found = []
    t.spaces()
    if t.peek("]"):
        t.take("]")
        return found
    while True:
        found.append(_value(t, ",]"))
        t.spaces()
        if t.peek(","):
            t.take(",")
            continue
        t.take("]")
        return found


_NAME = re.compile(r"\A\s*call:([^{]+)\{")


def parse(body, functions):
    """(name, arguments) of a call block's body `call:NAME{...}`, or None where it does not parse in full or names no
    function of `functions`."""
    found = _NAME.match(body)
    if not found:
        return None
    t = _Text(body)
    t.at = found.end(1)
    try:
        arguments = _object(t)
        t.spaces()
    except ValueError:
        return None
    name = found.group(1).strip()
    if t.at != len(t.text) or name not in functions:
        return None
    return name, arguments


def stream(body, functions):
    """(name, "{") of an unfinished call once its name is written (`call:NAME{`); its arguments stream when the block
    closes (parse)."""
    found = _NAME.match(body)
    if not found or found.group(1).strip() not in functions:
        return None
    return found.group(1).strip(), "{"


TOOL_FORMAT = ToolFormat(parse, OPEN, CLOSE, markers=(QUOTE,), stream=stream)

# The thought channel the template writes before a turn's calls: <|channel>thought\n...\n<channel|>.
REASONING_FORMAT = ReasoningFormat("<|channel>", "<channel|>", label="thought")
