"""Bounded provisional text for an opt-in, serialized resident request."""
import errno
import sys
import threading
import time

from crit.run.records import keys, number, read_json, write_json

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
    def __init__(self, path, command, request, cancellation):
        self.path, self.cancellation = path, cancellation
        self.identity = {key: request[key] for key in IDENTITY}
        self.identity.update(command_id=command["command_id"], token=command["token"])
        self.thread = threading.get_ident()
        self.text, self.sequence, self.published = "", 0, 0
        self.next_write, self.closed, self.error = 0, False, None

    def emit(self, chunk):
        try:
            if self.closed or threading.get_ident() != self.thread:
                raise ValueError("Emit stream text only on the generation thread before it returns")
            if self.error:
                raise ValueError(self.error)
            self.cancellation.check()
            chunk = stream_text(chunk)
            if len(self.text) + len(chunk) > MAX_TEXT:
                raise ValueError("Stream text exceeds 65536 characters")
            self.text += chunk
            self.sequence += 1
            if time.monotonic() >= self.next_write:
                self.flush()
        except BaseException as exc:
            self.error = str(exc) or type(exc).__name__
            raise

    def flush(self):
        if self.sequence != self.published:
            write_json(self.path, {"schema_version": 1, "record_kind": "chat_progress",
                **self.identity, "sequence": self.sequence, "assistant": self.text})
            self.published = self.sequence
            self.next_write = time.monotonic() + .05

    def finish(self, response):
        if self.error:
            raise ValueError(self.error)
        if not isinstance(response, dict) or response.get("assistant") != self.text:
            raise ValueError("Final reply differs from emitted stream text")
        self.flush()

    def close(self):
        self.closed = True


class StreamReader:
    """Reject stale identities and rewrites; dropped intermediate snapshots are OK."""
    def __init__(self, path, command_id, token, request, publish):
        self.path, self.publish = path, publish
        self.identity = {key: request[key] for key in IDENTITY}
        self.identity.update(command_id=command_id, token=token)
        self.sequence, self.text = 0, ""

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
        keys(value, {"schema_version", "record_kind", "sequence", "assistant", *self.identity})
        if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["record_kind"] != "chat_progress":
            raise ValueError("Invalid resident stream record")
        if any(value[key] != expected for key, expected in self.identity.items()):
            raise ValueError("Resident stream identity mismatch")
        sequence = number(value["sequence"], "Stream sequence", 1, MAX_TEXT, integer=True)
        text = stream_text(value["assistant"])
        if sequence == self.sequence and text == self.text:
            return
        if sequence <= self.sequence or not text.startswith(self.text) or len(text) - len(self.text) < sequence - self.sequence:
            raise ValueError("Resident stream text or sequence moved backwards")
        self.sequence, self.text = sequence, text
        self.publish(text)
