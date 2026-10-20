"""OpenAI chat messages normalized for Hugging Face chat templates (standard library only)."""
from collections.abc import Mapping
import base64
from urllib.parse import urlsplit

from crit_core import jsonfile

from .errors import TextError

KEPT = ("tool_call_id", "name", "reasoning_content")


def message_content(content, where="content", *, images=False):
    """Validate OpenAI text/image parts without fetching URLs or opening files.

    Text-only lists retain the earlier joined-string form. Image lists preserve
    order and use the OpenAI image_url shape in request records.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts, has_images = [], False
        for index, part in enumerate(content):
            if isinstance(part, Mapping) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append({"type": "text", "text": part["text"]})
                continue
            image = part.get("image_url") if isinstance(part, Mapping) else None
            if not images:
                raise TextError(f"{where}[{index}] must be a text part; only text content is supported")
            if not images or not isinstance(part, Mapping) or part.get("type") != "image_url" or not isinstance(image, Mapping):
                raise TextError(f"{where}[{index}] must be a supported text or image_url part")
            url, detail = image.get("url"), image.get("detail", "auto")
            if not isinstance(url, str) or not url or detail not in ("auto", "low", "high"):
                raise TextError(f"{where}[{index}] needs an image URL and detail auto, low or high")
            if url.startswith("data:image/"):
                header, separator, data = url.partition(",")
                if not separator or not header.endswith(";base64") or not data:
                    raise TextError("Image data URLs require base64 image data")
                try:
                    base64.b64decode(data, validate=True)
                except ValueError as exc:
                    raise TextError("Invalid base64 image data") from exc
            else:
                parsed = urlsplit(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                    raise TextError("Image URLs must use http, https or base64 data:image/")
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
            has_images = True
        return parts if has_images else "".join(p["text"] for p in parts)
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


def openai_messages(messages, *, images=False):
    """Return new message dicts for a chat template: "developer" becomes "system"; content that is null becomes ""
    and a list of text parts is joined into one string; tool_calls[*].function.arguments given as a JSON string
    becomes a dict (TextError if it is not a JSON object); tool_call_id, name and reasoning_content are kept and
    other keys dropped. With images=True, all content becomes ordered parts, with image_url sources represented
    as {"type": "image", "image": URL, "detail": ...} for the checkpoint template. No URLs are fetched.
    Role order is left to the template, which rejects what it cannot render."""
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
        content = message_content(message.get("content"), f"{where}.content", images=images)
        if images:
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
            content = [({"type": "image", "image": p["image_url"]["url"], "detail": p["image_url"]["detail"]}
                        if p["type"] == "image_url" else dict(p)) for p in parts]
        normalized = {"role": "system" if role == "developer" else role, "content": content}
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
