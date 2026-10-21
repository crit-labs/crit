"""Bounded provisional text for one isolated resident request, sent over the host's channel as it is emitted."""
import threading

from crit.run.records import keys
from .content import content_size, content_text, response_content, tool_size
from .options import validate_tool_reply
from .tool_stream import ToolCallStream

CAPABILITY = "chat-stream-v1"
PARTS = ("text", "reasoning", "tool_calls")


class StreamWriter:
    """Synchronous adapter callback: checks each chunk and sends it at once as one stream piece."""
    def __init__(self, send, command_id, request, cancellation, tools=False):
        self.send, self.command_id, self.cancellation = send, command_id, cancellation
        self.thread = threading.get_ident()
        self.text, self.reasoning = [], []  # pieces, joined only to compare the final reply
        self.text_size = self.reasoning_size = 0
        self.closed, self.error = False, None
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
                piece = {"reasoning": content_text(chunk["reasoning"], "Stream reasoning")}
                self.reasoning.append(piece["reasoning"])
                self.reasoning_size += len(piece["reasoning"])
            elif isinstance(chunk, dict) and self.tools is not None:
                keys(chunk, {"tool_calls"})
                piece = {"tool_calls": self.tools.update(chunk["tool_calls"])}
            else:
                piece = {"text": content_text(chunk, "Stream text")}
                self.text.append(piece["text"])
                self.text_size += len(piece["text"])
            content_size(self.text_size, self.tools.size if self.tools else 0, self.reasoning_size)
            self.send({"kind": "progress", "command_id": self.command_id, **piece})
        except BaseException as exc:
            self.error = str(exc) or type(exc).__name__
            raise

    def finish(self, response):
        if self.error:
            raise ValueError(self.error)
        if not isinstance(response, dict) or response.get("assistant") != "".join(self.text):
            raise ValueError("Final reply differs from emitted stream text")
        if response.get("reasoning", "") != "".join(self.reasoning):
            raise ValueError("Final reasoning differs from emitted stream reasoning")
        telemetry = response.get("telemetry")
        output = telemetry.get("output_tokens") if isinstance(telemetry, dict) else None
        calls = validate_tool_reply(response.get("tool_calls"), self.request,
                                    output == self.request["limits"]["max_output_tokens"])
        if self.tools is not None and calls != self.tools.calls:
            raise ValueError("Final tool calls differ from emitted tool calls")
        response_content(response, tool_size(calls))

    def close(self):
        self.closed = True


class StreamReader:
    """crit serve's side: checks each piece as the writer did, in order, and publishes it."""
    def __init__(self, request, publish, publish_tools=None, publish_reasoning=None):
        self.publish, self.publish_tools, self.publish_reasoning = publish, publish_tools, publish_reasoning
        self.text = self.reasoning = 0  # characters so far
        self.tools = ToolCallStream(request)

    def apply(self, piece):
        keys(piece, {"kind", "command_id"}, PARTS)
        if sum(part in piece for part in PARTS) != 1:
            raise ValueError("A resident stream piece carries one of " + ", ".join(PARTS))
        if "text" in piece:
            text = content_text(piece["text"], "Stream text")
            self.text += len(text)
            content_size(self.text, self.tools.size, self.reasoning)
            self.publish(text)
        elif "reasoning" in piece:
            reasoning = content_text(piece["reasoning"], "Stream reasoning")
            self.reasoning += len(reasoning)
            content_size(self.text, self.tools.size, self.reasoning)
            if self.publish_reasoning:
                self.publish_reasoning(reasoning)
        else:
            deltas = self.tools.update(piece["tool_calls"])
            content_size(self.text, self.tools.size, self.reasoning)
            if self.publish_tools:
                self.publish_tools(deltas)
