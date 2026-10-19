"""Bounded provisional text for an opt-in, serialized resident request."""
import errno
import sys
import threading
import time

from crit.run.records import keys, number, read_json, write_json
from .options import validate_tool_reply
from .tool_stream import ToolCallStream

CAPABILITY = "chat-stream-v1"
MAX_TEXT = 65536
IDENTITY = ("request_id", "candidate_id", "package_sha256")


def stream_text(value):
    # Whitespace and grapheme fragments are valid incremental text. A caller
    # must concatenate them without trimming or decoding them independently.
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT or "\0" in value:
        raise ValueError("Stream text must contain 1–65536 characters without NUL")
    value.encode("utf-8")  # Reject unpaired surrogates before publication.
    return value


class StreamWriter:
    """Synchronous adapter callback; keep one coalesced snapshot, never a queue."""
    def __init__(self, path, command, request, cancellation, tools=False):
        self.path, self.cancellation = path, cancellation
        self.identity = {key: request[key] for key in IDENTITY}
        self.identity.update(command_id=command["command_id"], token=command["token"])
        self.thread = threading.get_ident()
        self.text, self.reasoning, self.sequence, self.published = "", "", 0, 0
        self.next_write, self.closed, self.error = 0, False, None
        self.tools = ToolCallStream(request) if tools else None
        self.request = self.tools.request if self.tools is not None else request

    def emit(self, chunk):
        try:
            if self.closed or threading.get_ident() != self.thread:
                raise ValueError("Emit stream text only on the generation thread before it returns")
            if self.error:
                raise ValueError(self.error)
            self.cancellation.check()
            if isinstance(chunk, dict) and "reasoning" in chunk:
                # Reasoning apart from the reply (crit_core.text.reasoning), with its own size budget.
                keys(chunk, {"reasoning"})
                self.reasoning += stream_text(chunk["reasoning"])
                if len(self.reasoning) > MAX_TEXT:
                    raise ValueError("Stream reasoning exceeds 65536 characters")
            elif isinstance(chunk, dict) and self.tools is not None:
                keys(chunk, {"tool_calls"})
                self.tools.update(chunk["tool_calls"])
            else:
                self.text += stream_text(chunk)
            if len(self.text) + (self.tools.size if self.tools else 0) > MAX_TEXT:
                raise ValueError("Stream text exceeds 65536 characters")
            self.sequence += 1
            if time.monotonic() >= self.next_write:
                self.flush()
        except BaseException as exc:
            self.error = str(exc) or type(exc).__name__
            raise

    def flush(self):
        if self.sequence != self.published:
            write_json(self.path, {"schema_version": 1, "record_kind": "chat_progress",
                **self.identity, "sequence": self.sequence, "assistant": self.text,
                **({"tool_calls": self.tools.calls} if self.tools else {}),
                **({"reasoning": self.reasoning} if self.reasoning else {})})
            self.published = self.sequence
            self.next_write = time.monotonic() + .05

    def finish(self, response):
        if self.error:
            raise ValueError(self.error)
        if not isinstance(response, dict) or response.get("assistant") != self.text:
            raise ValueError("Final reply differs from emitted stream text")
        if response.get("reasoning", "") != self.reasoning:
            raise ValueError("Final reasoning differs from emitted stream reasoning")
        if self.tools is not None:
            calls = validate_tool_reply(response.get("tool_calls"), self.request)
            if calls != self.tools.calls:
                raise ValueError("Final tool calls differ from emitted tool calls")
        self.flush()

    def close(self):
        self.closed = True


class StreamReader:
    """Reject stale identities and rewrites; dropped intermediate snapshots are OK."""
    def __init__(self, path, command_id, token, request, publish, publish_tools=None, publish_reasoning=None):
        self.path, self.publish, self.publish_reasoning = path, publish, publish_reasoning
        self.identity = {key: request[key] for key in IDENTITY}
        self.identity.update(command_id=command_id, token=token)
        self.sequence, self.text, self.reasoning = 0, "", ""
        self.tools, self.publish_tools = ToolCallStream(request), publish_tools

    def poll(self):
        if not self.path.exists():
            return
        deadline = time.monotonic() + .25
        while True:
            try:
                value = read_json(self.path)
                break
            except PermissionError as exc:
                code = getattr(exc, "winerror", None)
                retryable = code in {5, 32, 33} or (code is None and exc.errno == errno.EACCES)
                if sys.platform != "win32" or not retryable or time.monotonic() >= deadline:
                    raise
                time.sleep(.005)
        keys(value, {"schema_version", "record_kind", "sequence", "assistant", *self.identity},
             {"tool_calls", "reasoning"})
        if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["record_kind"] != "chat_progress":
            raise ValueError("Invalid resident stream record")
        if any(value[key] != expected for key, expected in self.identity.items()):
            raise ValueError("Resident stream identity mismatch")
        sequence = number(value["sequence"], "Stream sequence", 1, MAX_TEXT, integer=True)
        text = value["assistant"]
        reasoning = stream_text(value["reasoning"]) if "reasoning" in value else ""
        if text != "" or not (value.get("tool_calls") or reasoning):
            text = stream_text(text)
        calls = value.get("tool_calls", [])
        if (sequence == self.sequence and text == self.text and calls == self.tools.calls
                and reasoning == self.reasoning):
            return
        if len(reasoning) > MAX_TEXT or not reasoning.startswith(self.reasoning):
            raise ValueError("Resident stream reasoning moved backwards")
        grown = len(reasoning) - len(self.reasoning)
        old_size = self.tools.size
        deltas = self.tools.snapshot(calls)
        size = len(text) + self.tools.size
        if (size > MAX_TEXT or sequence <= self.sequence or not text.startswith(self.text)
                or size - len(self.text) - old_size + grown < sequence - self.sequence):
            raise ValueError("Resident stream text or sequence moved backwards")
        added = reasoning[len(self.reasoning):]
        self.sequence, self.text, self.reasoning = sequence, text, reasoning
        if added and self.publish_reasoning:
            self.publish_reasoning(added)
        if text:
            self.publish(text)
        if deltas and self.publish_tools:
            self.publish_tools(deltas)
