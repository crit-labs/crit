"""The tool calls this model's chat template teaches: a call is its own reply message,
NAME<|content_invoke_tool_json|>{"name": NAME, "args": {...}}<|end_message|> (the template renders the arguments with
sorted keys and no spaces); crit serve parses them through TOOL_FORMAT (crit_core.text.ToolFormat, the manifest's
tool_format), and its reasoning, a message of its own between <|content_thinking|> and
<|end_message|>, through REASONING_FORMAT (the manifest's reasoning_format). A block whose JSON is not an object with a string name and an object of args, or names no offered
function, stays text."""
import json
import re

from crit_core.text.reasoning import ReasoningFormat
from crit_core.text.tools import ToolFormat

OPENING, CLOSING = "<|content_invoke_tool_json|>", "<|end_message|>"
NAME = r"[A-Za-z_][A-Za-z0-9_.\-]*"


def parse(body, functions):
    """(name, arguments) of a call's body, or None."""
    try:
        found = json.loads(body)
    except ValueError:
        return None
    if not isinstance(found, dict) or not isinstance(found.get("name"), str) or not isinstance(found.get("args", {}), dict):
        return None
    if found["name"] not in functions:
        return None
    return found["name"], found.get("args", {})


_STREAM = re.compile(r'\A\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"args"\s*:\s*')


def stream(body, functions):
    """(name, the arguments' start as JSON text) of an unfinished call: "{" once its name is written; the arguments
    stream when the block closes (parse)."""
    found = _STREAM.match(body)
    if not found:
        return None
    name = json.loads(f'"{found.group(1)}"')
    return (name, "{") if name in functions else None


TOOL_FORMAT = ToolFormat(parse, OPENING, CLOSING, stream=stream, lead=NAME)

# The reasoning the template writes as a message of its own: <|content_thinking|> ... <|end_message|>.
REASONING_FORMAT = ReasoningFormat("<|content_thinking|>", "<|end_message|>", repeated=True)
