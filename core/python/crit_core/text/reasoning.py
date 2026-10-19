"""Reasoning in generated text, split from the reply in the markers the model's own chat template uses (standard
library only).

Reasoning models write their thinking between markers before the reply, and their chat templates know the markers:
they strip thinking from history turns, and many open the generation prompt with the opening marker already in
place. `template_reasoning_format(sources)` names the registered format a model's templates use, and
`ReasoningSplitter` splits streamed reply text into reasoning and reply text. REASONING_FORMATS holds the syntaxes
several model families share; a package whose model writes another one supplies a `ReasoningFormat` on
`crit_core.serving.ResidentChat.reasoning_format`, so crit carries no model-specific syntax.

- "think-tags" (Qwen3 and later, DeepSeek-R1, GLM-4.5, Nemotron): <think>REASONING</think>REPLY.

Reasoning begins where the reply begins: when the generation prompt ends with the opening marker, or when the reply's
first text (after whitespace) is the opening marker. A marker anywhere else is reply text. A reply cut before the
closing marker is all reasoning. Whitespace around the markers belongs to the format and is dropped.
"""
from typing import NamedTuple

from .errors import TextError


class ReasoningFormat(NamedTuple):
    """A reasoning syntax: the literal markers around the thinking. A tokenizer may hold them as special tokens, which
    the reply text then has to keep (ResidentChat does)."""
    opening: str
    closing: str


REASONING_FORMATS = {
    "think-tags": ReasoningFormat("<think>", "</think>"),
}


def reasoning_format(format):
    """A ReasoningFormat for a registered name or a ReasoningFormat."""
    if isinstance(format, ReasoningFormat):
        return format
    if format not in REASONING_FORMATS:
        raise TextError(f"unknown reasoning format {format!r}; registered: {', '.join(REASONING_FORMATS)}")
    return REASONING_FORMATS[format]


def template_reasoning_format(sources):
    """The registered reasoning format a model's chat template sources (an iterable of strings) use, or None; a
    format counts when a template writes both of its markers. TextError when the templates use different ones."""
    found = set()
    for source in sources:
        if not isinstance(source, str):
            raise TextError("chat template source must be a string")
        found |= {name for name, f in REASONING_FORMATS.items() if f.opening in source and f.closing in source}
    if len(found) > 1:
        raise TextError("the model's chat templates disagree on the reasoning format: " + ", ".join(sorted(found)))
    return found.pop() if found else None


def _held(buffer, marker):
    """Length of the buffer's suffix that could still start `marker`."""
    for size in range(min(len(buffer), len(marker) - 1), 0, -1):
        if marker.startswith(buffer[-size:]):
            return size
    return 0


class ReasoningSplitter:
    """Incremental split of reply text into ("reasoning", str) and ("text", str) events.

    `opened`: the generation prompt already ends with the opening marker, so the reply starts inside the reasoning.
    Text that could still be a marker is held back, and so is whitespace that may turn out to sit next to one."""

    def __init__(self, format, *, opened=False):
        self.format = reasoning_format(format)
        self.state = "reasoning" if opened else "start"   # start -> reasoning -> text, or start -> text
        self.buffer, self.finished, self._leading = "", False, True

    def _reasoning(self, text, events):
        if self._leading:
            text = text.lstrip()
        if text:
            events.append(("reasoning", text))
            self._leading = False

    def _text(self, text, events):
        if text:
            events.append(("text", text))

    def push(self, text):
        if self.finished:
            raise TextError("the reasoning splitter has finished")
        self.buffer += text
        events = []
        opening, closing = self.format
        if self.state == "start":
            stripped = self.buffer.lstrip()
            if stripped.startswith(opening):
                self.state, self.buffer = "reasoning", stripped[len(opening):]
            elif not stripped or opening.startswith(stripped):
                return events   # still whitespace or a partial opening marker
            else:
                self.state = "text"
        if self.state == "reasoning":
            end = self.buffer.find(closing)
            if end < 0:
                keep = _held(self.buffer, closing)
                ready = self.buffer[:len(self.buffer) - keep]
                keep += len(ready) - len(ready.rstrip())   # whitespace that may precede the closing marker
                self._reasoning(self.buffer[:len(self.buffer) - keep], events)
                self.buffer = self.buffer[len(self.buffer) - keep:]
                return events
            self._reasoning(self.buffer[:end].rstrip(), events)
            self.state, self.buffer = "after", self.buffer[end + len(closing):]
        if self.state == "after":
            stripped = self.buffer.lstrip()
            if not stripped:
                self.buffer = ""
                return events
            self.state, self.buffer = "text", stripped
        if self.state == "text":
            self._text(self.buffer, events)
            self.buffer = ""
        return events

    def finish(self):
        """Flush held text: a reply cut inside the reasoning keeps it as reasoning."""
        events = self.push("")
        self.finished = True
        if self.state == "reasoning":
            self._reasoning(self.buffer.rstrip(), events)
        elif self.state == "start":
            self._text(self.buffer, events)
        self.buffer = ""
        return events


__all__ = ["REASONING_FORMATS", "ReasoningFormat", "ReasoningSplitter", "reasoning_format", "template_reasoning_format"]
