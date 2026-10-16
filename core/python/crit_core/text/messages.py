"""OpenAI chat messages normalized for Hugging Face chat templates (standard library only)."""
from collections.abc import Mapping

from crit_core import jsonfile

from .errors import TextError

KEPT = ("tool_call_id", "name", "reasoning_content")


def _content(content, where):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        texts = []
        for index, part in enumerate(content):
            if not isinstance(part, Mapping) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise TextError(f"{where}[{index}] must be a text part; only text content is supported")
            texts.append(part["text"])
        return "".join(texts)
    raise TextError(f"{where} must be a string, a list of text parts or null")


def _tool_call(call, where):
    if not isinstance(call, Mapping):
        raise TextError(f"{where} must be an object")
    call = dict(call)
    function = call.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise TextError(f"{where}.function must be an object with a name string")
    function = dict(function)
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = jsonfile.loads(arguments, f"{where}.function.arguments")
        except jsonfile.JsonError as error:
            raise TextError(str(error)) from None
    if not isinstance(arguments, Mapping):
        raise TextError(f"{where}.function.arguments must be a JSON object or a string holding one")
    function["arguments"] = dict(arguments)
    call["function"] = function
    return call


def openai_messages(messages):
    """Return new message dicts for a chat template: "developer" becomes "system"; content that is null becomes ""
    and a list of text parts is joined into one string; tool_calls[*].function.arguments given as a JSON string
    becomes a dict (TextError if it is not a JSON object); tool_call_id, name and reasoning_content are kept and
    other keys dropped. Role order is left to the template, which rejects what it cannot render."""
    if isinstance(messages, (str, bytes, Mapping)) or not isinstance(messages, (list, tuple)):
        raise TextError("messages must be a list of message objects")
    result = []
    for index, message in enumerate(messages):
        where = f"messages[{index}]"
        if not isinstance(message, Mapping):
            raise TextError(f"{where} must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise TextError(f"{where}.role must be a non-empty string")
        normalized = {"role": "system" if role == "developer" else role,
                      "content": _content(message.get("content"), f"{where}.content")}
        calls = message.get("tool_calls")
        if calls is not None:
            if isinstance(calls, (str, bytes, Mapping)) or not isinstance(calls, (list, tuple)):
                raise TextError(f"{where}.tool_calls must be a list")
            normalized["tool_calls"] = [_tool_call(call, f"{where}.tool_calls[{number}]")
                                        for number, call in enumerate(calls)]
        for key in KEPT:
            value = message.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TextError(f"{where}.{key} must be a string")
            normalized[key] = value
        result.append(normalized)
    return result
