"""Bounded socket delivery that cannot strand resident command supervision."""
from contextlib import contextmanager
import select
import socket
import ssl
import threading
import time


class StreamDelivery:
    """An unbuffered writer for one streaming HTTP response.

    Keep the command's absolute deadline, not a fresh timeout per write. Before
    dispatch, the same duration bounds delivery of the headers/initial event.
    The handler writes, and while it waits on the engine a ping thread may; a
    lock keeps every write whole. The disconnect watcher may still peek at input.
    """
    def __init__(self, connection, cancel, shutdown, timeout):
        self.connection, self.cancel, self.shutdown = connection, cancel, shutdown
        self.written = time.monotonic()  # when the last write ended
        self.deadline = self.written + timeout
        self.failed = False
        self.lock = threading.Lock()

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
        with self.lock:
            return self._write(data)

    def _write(self, data):
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
                except (BlockingIOError, InterruptedError, ssl.SSLWantWriteError):  # a TLS socket's would-block
                    remaining = self._remaining()
                    if not remaining:
                        raise ConnectionAbortedError("Stream delivery stopped")
                    select.select([], [self.connection], [], min(.05, remaining))
            self.written = time.monotonic()
            return sent
        except OSError:
            # Never append an error or another event after a partial frame.
            # Returning control lets the existing supervisor cancel and settle.
            self._abort()
            raise

    def flush(self):
        pass  # Successful writes have already handed all bytes to the socket.

    @contextmanager
    def pinging(self, ping, interval):
        """Write `ping` whenever `interval` seconds pass without a write, until the block ends (0: never).

        Pings stop with cancellation or shutdown. One that cannot be delivered
        abandons the stream as any write does, and the handler's next write fails.
        """
        if not interval:
            yield
            return
        done = threading.Event()
        thread = threading.Thread(target=self._ping, args=(ping, interval, done), name="crit-stream-ping", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()

    def _ping(self, ping, interval, done):
        while not done.wait(max(0, self.written + interval - time.monotonic())):
            with self.lock:
                if done.is_set() or self.failed or self.cancel.is_set() or self.shutdown.is_set():
                    return
                if time.monotonic() - self.written < interval:
                    continue  # the handler wrote meanwhile
                try:
                    self._write(ping)
                except OSError:
                    return
