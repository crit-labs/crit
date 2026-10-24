"""Request-scoped connection monitoring without consuming HTTP input."""
import select
import socket
import ssl
import threading


class ClientDisconnect:
    """Cancel when the client closes the connection, including during quiet generation.

    The handler has read the request body and must not read again until this scope
    exits. Peeking leaves pipelined requests intact. Serving treats EOF as request
    abandonment: clients must keep both sides open until the response completes.
    A write-half-close is indistinguishable from an ordinary client close and
    therefore also cancels the turn. Over TLS any input cancels the turn, and
    only EOF or a socket error marks the client disconnected.
    """
    def __init__(self, connection):
        self.connection = connection
        self.cancel = threading.Event()
        self.disconnected = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, name="crit-client-disconnect", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.cancel.set()
        self.done.set()
        # No blocking reads or writes in the watcher. Join before the handler
        # reuses this connection for another request or closes its descriptor.
        self.thread.join()

    def _watch(self):
        while not self.done.is_set():
            try:
                # SO_ERROR also detects resets when unread pipelined bytes hide
                # the error from MSG_PEEK. Reading it clears it, so latch failure.
                if self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR):
                    self._disconnect()
                    return
                readable, _, _ = select.select([self.connection], [], [], 0)
                # Peek beneath TLS: only the end of the stream is a disconnect.
                if readable and not socket.socket.recv(self.connection, 1, socket.MSG_PEEK):
                    self._disconnect()
                    return
                # Encrypted records cannot be told apart, so on HTTPS any input, the client's close_notify or a
                # pipelined request, ends the turn but keeps the connection: a keep-alive client's next request can
                # arrive before this scope exits.
                if readable and isinstance(self.connection, ssl.SSLSocket):
                    self.cancel.set()
            except (BlockingIOError, InterruptedError):
                pass
            except (OSError, ValueError):
                self._disconnect()
                return
            # Pending request bytes remain readable; never spin on them.
            if self.done.wait(.05):
                return

    def _disconnect(self):
        self.disconnected.set()
        self.cancel.set()
