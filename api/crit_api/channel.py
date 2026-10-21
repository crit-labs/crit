"""The resident host's channel: length-prefixed JSON messages over one authenticated loopback connection.

crit serve's supervisor listens on 127.0.0.1 before it launches the host. The host connects, opens with the runtime's
ownership token and its process id, and the connection then carries every command and cancellation to the host and every status,
stream piece and result back. A message is delivered as soon as it is sent (TCP_NODELAY, no polling), so serving
adds no delay to a reply's first token or to its stream.
"""
import hmac
import json
import os
import queue
import socket
import struct
import threading

HEADER = struct.Struct("!I")
STATUS_BYTES = 1 << 20  # a status, stream piece or result without a response
HELLO_SECONDS = 5


def encode(message, limit):
    data = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > limit:
        raise ValueError(f"Resident {message.get('kind', 'message')} exceeds {limit} bytes")
    return HEADER.pack(len(data)) + data


class Connection:
    """One end of the channel; sends are serialized, receives belong to one thread."""

    def __init__(self, sock):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock, self.reader = sock, sock.makefile("rb")
        self.lock = threading.Lock()

    def send(self, message, limit=1 << 31):
        self.write(encode(message, limit))

    def write(self, data):
        with self.lock:
            self.sock.sendall(data)

    def receive(self, limit):
        """The next message, a dictionary with a "kind"; EOFError once the other end closed. `limit` bounds its
        bytes, or is a function returning the bound when the message arrives."""
        header = self.reader.read(HEADER.size)
        if len(header) < HEADER.size:
            raise EOFError("Resident channel closed")
        size, = HEADER.unpack(header)
        limit = limit() if callable(limit) else limit
        if size > limit:
            raise ValueError(f"Resident message of {size} bytes exceeds {limit}")
        data = self.reader.read(size)
        if len(data) < size:
            raise EOFError("Resident channel closed")
        message = json.loads(data)
        if not isinstance(message, dict) or not isinstance(message.get("kind"), str):
            raise ValueError("Invalid resident message")
        return message

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)  # wakes a receive blocked in another thread
        except OSError:
            pass
        self.reader.close()
        self.sock.close()


def connect(port, token):
    """The host's end: connect to the supervisor, authenticate with the ownership token and name its process."""
    connection = Connection(socket.create_connection(("127.0.0.1", port)))
    connection.send({"kind": "hello", "token": token, "pid": os.getpid()})
    return connection


class Mailbox:
    """What one command's host sent back: stream pieces in order, then its result."""

    def __init__(self, limit):
        self.limit = limit
        self.pieces, self.result = [], None
        self.ready = threading.Event()


class Supervisor:
    """crit serve's end of one host's channel. send() never blocks: a sender thread writes commands once the host
    has connected. A receiver thread routes stream pieces and results to the mailbox of the command they answer and
    keeps the host's latest status; on any failure it records why and wakes every waiting command. `pid` is the host's
    process id once it has connected."""

    def __init__(self, token):
        self.token = token
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.listener.settimeout(.25)
        self.port = self.listener.getsockname()[1]
        self.lock = threading.Lock()
        self.boxes, self.status, self.failure = {}, None, None
        self.connection = self.pid = None
        self.connected = threading.Event()
        self.outbox = queue.SimpleQueue()
        threading.Thread(target=self._receive, name="crit-resident-receive", daemon=True).start()
        threading.Thread(target=self._send, name="crit-resident-send", daemon=True).start()

    def register(self, command_id, limit=STATUS_BYTES):
        """A mailbox for a command's answers, whose result may be up to `limit` bytes."""
        with self.lock:
            box = self.boxes[command_id] = Mailbox(max(limit, STATUS_BYTES))
        return box

    def unregister(self, command_id):
        with self.lock:
            self.boxes.pop(command_id, None)

    def take(self, box):
        """The stream pieces received since the last take, and the result once it arrived."""
        with self.lock:
            pieces, box.pieces = box.pieces, []
            box.ready.clear()
            return pieces, box.result

    def send(self, message, limit=STATUS_BYTES):
        data = encode(message, limit)
        if self.failure:
            raise ValueError("Resident host channel failed: " + self.failure)
        self.outbox.put(data)

    def close(self):
        self._fail("closed")

    def _fail(self, reason):
        with self.lock:
            if self.failure is None:
                self.failure = reason
            boxes, connection = list(self.boxes.values()), self.connection
        for box in boxes:
            box.ready.set()
        self.listener.close()
        if connection is not None:
            connection.close()
        self.connected.set()
        self.outbox.put(None)

    def _accept(self):
        while self.failure is None:
            try:
                sock, _ = self.listener.accept()
            except TimeoutError:
                continue
            sock.settimeout(HELLO_SECONDS)
            connection = Connection(sock)
            try:
                hello = connection.receive(STATUS_BYTES)
                token = hello.get("token") if hello["kind"] == "hello" else None
                pid = hello.get("pid")
                if (isinstance(token, str) and hmac.compare_digest(token.encode(), self.token.encode())
                        and type(pid) is int and pid > 0):
                    sock.settimeout(None)
                    self.pid = pid
                    return connection
            except (OSError, EOFError, ValueError):
                pass
            connection.close()  # not the host
        return None

    def _receive(self):
        try:
            connection = self._accept()
            if connection is None:
                return
            with self.lock:
                if self.failure is not None:
                    connection.close()
                    return
                self.connection = connection
            self.listener.close()
            self.connected.set()
            while True:
                message = connection.receive(self._limit)
                kind = message["kind"]
                if kind == "status":
                    if not isinstance(message.get("status"), dict):
                        raise ValueError("Invalid resident status")
                    with self.lock:
                        self.status = message["status"]
                    continue
                if kind not in {"progress", "result"}:
                    raise ValueError("Unknown resident message " + repr(kind)[:40])
                with self.lock:
                    box = self.boxes.get(message.get("command_id"))
                    if box is None:
                        continue  # the command already returned (cancelled)
                    if kind == "result":
                        box.result = message
                    else:
                        box.pieces.append(message)
                box.ready.set()
        except BaseException as exc:
            self._fail(str(exc) or type(exc).__name__)

    def _limit(self):
        with self.lock:
            return max((box.limit for box in self.boxes.values()), default=STATUS_BYTES)

    def _send(self):
        self.connected.wait()
        try:
            while True:
                data = self.outbox.get()
                if data is None or self.failure is not None:
                    return
                self.connection.write(data)
        except BaseException as exc:
            self._fail(str(exc) or type(exc).__name__)
