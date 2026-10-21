"""Serving content policy; identities, telemetry and tool-call semantics live with their validators."""
import json

MAX_CONTENT = 65536


def content_text(value, name, *, allow_empty=False):
    """Preserve text exactly, including whitespace and grapheme fragments."""
    if not isinstance(value, str) or (not value and not allow_empty) or "\0" in value:
        raise ValueError(name + " must be " + ("" if allow_empty else "nonempty ") + "text without NUL")
    if len(value) > MAX_CONTENT:
        raise ValueError(f"{name} exceeds {MAX_CONTENT} characters")
    value.encode("utf-8")
    return value


def tool_size(calls):
    """The wire representation's character count, shared by final and partial calls."""
    if not calls:
        return 0
    encoded = json.dumps(calls, ensure_ascii=False, allow_nan=False)
    encoded.encode("utf-8")
    return len(encoded)


def content_size(assistant, tool_characters=0, reasoning=""):
    """Assistant text and tools share one budget; reasoning has its own. Text may be given by its length."""
    size = (assistant if type(assistant) is int else len(assistant)) + tool_characters
    if size > MAX_CONTENT:
        raise ValueError(f"Reply content exceeds {MAX_CONTENT} characters")
    if (reasoning if type(reasoning) is int else len(reasoning)) > MAX_CONTENT:
        raise ValueError(f"Reply reasoning exceeds {MAX_CONTENT} characters")
    return size


def response_content(value, tool_characters=0, *, final=True):
    """Validate content channels; callers validate complete or partial tool-call semantics.

    A completed plain-text reply must contain nonblank text. Reasoning or tools
    may be the entire output. Stream prefixes may consist solely of whitespace;
    their eventual completed response still goes through the final check.
    """
    assistant = content_text(value["assistant"], "Reply assistant", allow_empty=True)
    reasoning = content_text(value["reasoning"], "Reply reasoning") if "reasoning" in value else ""
    content_size(assistant, tool_characters, reasoning)
    if not (tool_characters or reasoning or (assistant.strip() if final else assistant)):
        raise ValueError("Reply requires assistant text, reasoning or tool calls")
    return assistant, reasoning
