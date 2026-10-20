"""Real TCP backpressure must not strand resident cancellation or ownership."""
import json
import socket
import threading
import time
import unittest
from unittest.mock import Mock, patch

from test_serve_shutdown import ShutdownFixture
from crit_api.delivery import StreamDelivery
from crit_api.server import Api, Server
from crit.process.ownership import process_identity


PRESSURED_ADAPTER = r'''
from pathlib import Path
_generate_stream = Engine.generate_stream
def pressured_stream(self, request, cancellation, emit):
    content = request["messages"][-1]["content"]
    if not content.startswith("pressure:"):
        return _generate_stream(self, request, cancellation, emit)
    gate = Path(content[len("pressure:"):])
    text = "\U0001f600" * 60000
    self.private.append(request["request_id"])
    try:
        gate.with_suffix(".started").touch()
        for offset in range(0, len(text), 6000):
            emit(text[offset:offset + 6000])
            time.sleep(.06)
        while not gate.with_suffix(".release").exists():
            if not gate.with_suffix(".ignore").exists():
                cancellation.check()
            time.sleep(.01)
        return self.response(request, text)
    finally:
        gate.with_suffix(".settled").touch()
Engine.generate_stream = pressured_stream
'''


class SmallBufferServer(Server):
    def get_request(self):
        connection, address = super().get_request()
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        return connection, address


class BackpressureTests(ShutdownFixture, unittest.TestCase):
    adapter_suffix = PRESSURED_ADAPTER

    def setUp(self):
        super().setUp()
        self.served = self.engine(timeout=2)
        self.api = Api(self.served)
        run = self.api.run
        self.write_lock = threading.Lock()
        self.writes = {"started": 0, "finished": 0, "since": None}
        self.admitting = threading.Event()

        def observed_run(request, emit, cancel, *args, **kwargs):
            self.admitting.set()
            def observed_emit(piece):
                with self.write_lock:
                    self.writes["started"] += 1
                    self.writes["since"] = time.monotonic()
                try:
                    emit(piece)  # Actual socket delivery; no injected stall.
                finally:
                    with self.write_lock:
                        self.writes["finished"] += 1
            return run(request, observed_emit, cancel, *args, **kwargs)

        self.api.run = observed_run
        self.server = SmallBufferServer(("127.0.0.1", 0), self.api)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server_thread.join, 5)
        self.addCleanup(self.server.shutdown)
        self.gate = self.root / "producer"
        self.addCleanup(self.gate.with_suffix(".release").touch)

    def connect(self, message=None):
        client = socket.socket()
        self.addCleanup(client.close)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        client.settimeout(8)
        client.connect(self.server.server_address)
        body = json.dumps({"messages": [{"role": "user", "content": message or "pressure:" + str(self.gate)}],
                           "max_tokens": 128, "stream": True}).encode()
        client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                       b"Content-Type: application/json\r\nContent-Length: " +
                       str(len(body)).encode() + b"\r\n\r\n" + body)
        return client

    def blocked(self):
        def waiting():
            with self.write_lock:
                return (self.writes["started"] > self.writes["finished"]
                        and time.monotonic() - self.writes["since"] > .15)
        self.until(waiting, "client did not backpressure an actual streaming write")
        self.assertTrue(self.gate.with_suffix(".started").exists())

    def wire(self, client):
        data = bytearray()
        try:
            while piece := client.recv(65536):
                data.extend(piece)
        except ConnectionResetError:
            pass  # A reset after abandonment is also a closed response.
        return bytes(data)

    def test_stalled_stream_timeout_cancels_and_preserves_healthy_weights(self):
        before = self.ask(self.served)
        client = self.connect()
        self.blocked()
        self.until(self.gate.with_suffix(".settled").exists,
                   "request timeout did not cancel the backpressured producer")
        self.until(lambda: not self.served.lock.locked(), "request lock was not released")
        after = self.ask(self.served)
        self.assertEqual((after["pid"], after["device"], after["generation"]),
                         (before["pid"], before["device"], before["generation"]))
        self.assertEqual(after["private"], 1)
        self.assertEqual(list((self.served.runtime.root / "requests").iterdir()), [])
        # No client read/close/release above could have allowed cancellation.
        self.assertFalse(self.gate.with_suffix(".release").exists())
        wire = self.wire(client)
        self.assertIn(b"HTTP/1.1 200", wire)
        self.assertNotIn(b'data: {"error"', wire)
        self.assertNotIn(b"data: [DONE]", wire)

    def test_shutdown_interrupts_stalled_write_and_releases_ownership(self):
        self.served.timeout = 60
        pid = self.served.runtime.tree.process.pid
        client = self.connect()
        self.blocked()
        closer = self.launch(self.served.close)
        try:
            self.assertNotIn("error", self.finish(closer))
            self.assertTrue(self.gate.with_suffix(".settled").exists())
            self.assertFalse(self.gate.with_suffix(".release").exists())
            self.assertTrue(self.served.closed)
            self.assertIsNone(process_identity(pid))
            self.assertFalse(self.served.ownership.path.exists())
        finally:
            client.close()
            self.finish(closer)

    def test_stalled_stream_timeout_retires_unresponsive_producer(self):
        # Finish emitting before timeout, then deliberately ignore cancellation.
        self.gate.with_suffix(".ignore").touch()
        pid = self.served.runtime.tree.process.pid
        client = self.connect()
        self.blocked()
        self.until(lambda: self.served.runtime.tree is None,
                   "unresponsive backpressured producer was not physically retired")
        self.assertIsNone(process_identity(pid))
        self.assertFalse(self.gate.with_suffix(".settled").exists())
        self.assertFalse(self.gate.with_suffix(".release").exists())
        after = self.ask(self.served)
        self.assertNotEqual(after["pid"], pid)
        self.assertEqual(after["private"], 1)
        self.wire(client)

    def assert_complete_stream(self, wire):
        header, body = wire.split(b"\r\n\r\n", 1)
        self.assertTrue(header.startswith(b"HTTP/1.1 200"))
        frames = [line[len(b"data: "):] for line in body.splitlines() if line.startswith(b"data: ")]
        self.assertEqual(frames[-1], b"[DONE]")
        self.assertEqual(frames.count(b"[DONE]"), 1)
        events = [json.loads(frame) for frame in frames[:-1]]
        self.assertFalse(any("error" in event for event in events))
        choices = [event["choices"][0] for event in events]
        self.assertEqual(choices[0]["delta"]["role"], "assistant")
        self.assertEqual("".join(choice["delta"].get("content", "") for choice in choices), "\U0001f600" * 60000)
        self.assertEqual([choice["finish_reason"] for choice in choices if choice["finish_reason"]], ["stop"])

    def test_reader_resumes_before_deadline_without_losing_or_duplicating_text(self):
        self.served.timeout = 15
        before = self.ask(self.served)
        client = self.connect()
        self.blocked()
        self.gate.with_suffix(".release").touch()
        self.assert_complete_stream(self.wire(client))
        after = self.ask(self.served)
        self.assertEqual((after["pid"], after["device"]), (before["pid"], before["device"]))

    def test_queued_stream_gets_the_execution_budget_after_admission(self):
        self.served.timeout = 3
        self.gate.with_suffix(".release").touch()
        reader = None
        try:
            # Hold admission longer than the delivery's initial header budget.
            with self.served.lock:
                client = self.connect()
                reader = self.launch(lambda: self.wire(client))
                self.assertTrue(self.admitting.wait(5))
                self.assertFalse(reader[1].wait(self.served.timeout + .1))
                self.assertFalse(self.gate.with_suffix(".started").exists())
            outcome = self.finish(reader)
            self.assertNotIn("error", outcome)
            self.assert_complete_stream(outcome["result"])
        finally:
            if reader:
                client.close()
                self.finish(reader)

    def test_responsive_reader_receives_timeout_error_and_done(self):
        self.served.timeout = 1
        before = self.ask(self.served)
        client = self.connect("slow")
        wire = self.wire(client)
        self.assertIn(b"HTTP/1.1 200", wire)
        self.assertIn(b'"message": "The engine did not finish within the request timeout"', wire)
        self.assertEqual(wire.count(b"data: [DONE]"), 1)
        self.assertNotIn(b'"finish_reason": "stop"', wire)
        self.assertEqual(self.ask(self.served)["pid"], before["pid"])


class DeliveryTests(unittest.TestCase):
    def scripted_socket(self, steps=()):
        connection = Mock()
        connection.gettimeout.return_value = 7
        wire = bytearray()
        actions = iter(steps)

        def send(data):
            action = next(actions, len(data))
            if isinstance(action, BaseException):
                raise action
            wire.extend(data[:action])
            return min(action, len(data))

        connection.send.side_effect = send
        return connection, wire

    def test_partial_sends_and_waits_preserve_bytes_and_restore_socket_mode(self):
        connection, wire = self.scripted_socket([3, BlockingIOError(), 2, InterruptedError(), 1])
        cancel, shutdown = threading.Event(), threading.Event()
        payload = bytes(range(256)) * 200
        with patch("crit_api.delivery.time.monotonic", return_value=100), \
                patch("crit_api.delivery.select.select", return_value=([], [connection], [])) as wait:
            with StreamDelivery(connection, cancel, shutdown, 5) as writer:
                self.assertEqual(writer.write(payload), len(payload))
                writer.flush()
        self.assertEqual(wire, payload)
        self.assertFalse(cancel.is_set())
        self.assertEqual(wait.call_count, 2)
        self.assertTrue(all(call.args[3] <= .05 for call in wait.call_args_list))
        connection.setblocking.assert_called_once_with(False)
        connection.settimeout.assert_called_once_with(7)
        connection.shutdown.assert_not_called()

    def test_absolute_budget_does_not_restart_for_writes_or_progress(self):
        connection, wire = self.scripted_socket()
        cancel, shutdown = threading.Event(), threading.Event()
        clock = [0.]

        def slow_progress(data):
            wire.extend(data[:1])
            clock[0] += .2
            return 1

        connection.send.side_effect = slow_progress
        with patch("crit_api.delivery.time.monotonic", side_effect=lambda: clock[0]):
            with StreamDelivery(connection, cancel, shutdown, .5) as writer:
                writer.write(b"A")
                with self.assertRaises(ConnectionAbortedError):
                    writer.write(b"BCDE")
                count = connection.send.call_count
                with self.assertRaises(ConnectionAbortedError):
                    writer.write(b"data: [DONE]\n\n")
                self.assertEqual(connection.send.call_count, count)
        self.assertEqual(wire, b"ABC")
        self.assertTrue(cancel.is_set())
        connection.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        connection.settimeout.assert_called_once_with(7)

    def test_terminal_event_after_deadline_must_fit_without_waiting(self):
        connection, wire = self.scripted_socket()
        cancel, shutdown = threading.Event(), threading.Event()
        with patch("crit_api.delivery.time.monotonic", return_value=100), \
                patch("crit_api.delivery.select.select") as wait:
            with StreamDelivery(connection, cancel, shutdown, 1) as writer:
                writer.set_deadline(99)
                terminal = b'data: {"error": {"message": "timeout"}}\n\n'
                writer.write(terminal)
                writer.write(b"data: [DONE]\n\n")
                connection.send.side_effect = BlockingIOError()
                with self.assertRaises(ConnectionAbortedError):
                    writer.write(b"blocked")
        self.assertEqual(wire, terminal + b"data: [DONE]\n\n")
        wait.assert_not_called()
        self.assertTrue(cancel.is_set())

    def test_cancellation_or_shutdown_interrupts_a_waiting_write(self):
        for cause in ("cancel", "shutdown"):
            with self.subTest(cause=cause):
                connection, _ = self.scripted_socket()
                connection.send.side_effect = BlockingIOError()
                cancel, shutdown = threading.Event(), threading.Event()

                def wait(*args):
                    (cancel if cause == "cancel" else shutdown).set()
                    return [], [], []

                with patch("crit_api.delivery.select.select", side_effect=wait) as select_wait:
                    with StreamDelivery(connection, cancel, shutdown, 60) as writer:
                        with self.assertRaises(ConnectionAbortedError):
                            writer.write(b"waiting")
                self.assertEqual(select_wait.call_count, 1)
                self.assertTrue(cancel.is_set())
                connection.shutdown.assert_called_once_with(socket.SHUT_RDWR)


if __name__ == "__main__":
    unittest.main()
