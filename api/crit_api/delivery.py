"""Bounded socket delivery that cannot strand resident command supervision."""
import select
import socket
import time


class StreamDelivery:
    """An unbuffered writer for one streaming HTTP response.

    Keep the command's absolute deadline, not a fresh timeout per write. Before
    dispatch, the same duration bounds delivery of the headers/initial event.
    Only the handler writes; the disconnect watcher may still peek at input.
    """
    def __init__(self, connection, cancel, shutdown, timeout):
        self.connection, self.cancel, self.shutdown = connection, cancel, shutdown
        self.deadline = time.monotonic() + timeout
        self.failed = False

    def __enter__(self):
        self.previous_timeout = self.connection.gettimeout()
        self.connection.setblocking(False)
        return self

    def __exit__(self, kind, value, traceback):
        self.connection.settimeout(self.previous_timeout)

    def set_deadline(self, deadline):
        self.deadline = deadline

    def _remaining(self):
        if self.cancel.is_set() or self.shutdown.is_set():
            return 0
        return max(0, self.deadline - time.monotonic())

    def _abort(self):
        self.failed = True
        self.cancel.set()
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def write(self, data):
        if self.failed:
            raise ConnectionAbortedError("Stream delivery was abandoned")
        view = memoryview(data)
        sent = 0
        try:
            while sent < len(view):
                # One immediate attempt permits a small terminal error/[DONE]
                # even after timeout/cancellation, but never waits for a reader.
                if sent and not self._remaining():
                    raise ConnectionAbortedError("Stream delivery stopped")
                try:
                    count = self.connection.send(view[sent:sent + 16384])
                    if count == 0:
                        raise ConnectionAbortedError("Client stopped receiving")
                    sent += count
                except (BlockingIOError, InterruptedError):
                    remaining = self._remaining()
                    if not remaining:
                        raise ConnectionAbortedError("Stream delivery stopped")
                    select.select([], [self.connection], [], min(.05, remaining))
            return sent
        except OSError:
            # Never append an error or another event after a partial frame.
            # Returning control lets the existing supervisor cancel and settle.
            self._abort()
            raise

    def flush(self):
        pass  # Successful writes have already handed all bytes to the socket.
