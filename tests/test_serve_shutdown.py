"""Shutdown admission, cancellation and physical retirement with real resident processes."""
import ctypes
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from test_api_cancellation import GATED_ADAPTER
from test_chat_residency import ServeFixture
from crit_api.engine import EngineError
from crit_api.server import Api, Server
from crit.process.ownership import RuntimeOwnership, process_identity
from crit.process.tree import ProcessTree
from crit.run.records import read_json


class ShutdownFixture(ServeFixture):
    stream = True
    adapter_suffix = GATED_ADAPTER

    def until(self, predicate, message):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail(message)

    def launch(self, action):
        done, outcome = threading.Event(), {}

        def run():
            try:
                outcome["result"] = action()
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, done, outcome

    def finish(self, task):
        thread, done, outcome = task
        self.assertTrue(done.wait(10), "shutdown operation did not finish")
        thread.join(1)
        self.assertFalse(thread.is_alive())
        return outcome

    def assert_stopping(self, task):
        error = self.finish(task).get("error")
        self.assertIsInstance(error, EngineError)
        self.assertEqual(error.status, 503)
        self.assertIn("stopping", str(error))

    def request(self, address, gate, stream):
        connection = http.client.HTTPConnection(*address, timeout=15)
        self.addCleanup(connection.close)
        connection.request("POST", "/v1/chat/completions", json.dumps({
            "messages": [{"role": "user", "content": "gate:" + str(gate)}],
            "stream": stream, "max_tokens": 128,
        }), {"Content-Type": "application/json"})
        response = None
        if stream:
            response = connection.getresponse()
            self.addCleanup(response.close)
            self.assertEqual(response.status, 200)
            self.assertIn(b'"role": "assistant"', response.readline())
        self.until(gate.with_suffix(".started").exists, "producer did not reach its gate")
        return connection, response


class EngineShutdownTests(ShutdownFixture, unittest.TestCase):
    def test_shutdown_during_command_preparation_does_not_dispatch_or_switch(self):
        # Exercise both entry to call() and the publication boundary after start().
        for method in ("call", "_call"):
            with self.subTest(method=method):
                engine = self.engine(timeout=60)
                root = engine.runtime.root
                accepted = set((root / "accepted").iterdir())
                preparing, release = threading.Event(), threading.Event()
                call = getattr(engine.runtime, method)
                producer = closer = None

                def held_call(*args, **kwargs):
                    action = args[0 if method == "call" else 1]
                    if action in {"load", "generate"}:
                        preparing.set()
                        if not release.wait(10):
                            raise RuntimeError("test command gate was not released")
                    return call(*args, **kwargs)

                try:
                    with patch.object(engine.runtime, method, side_effect=held_call):
                        producer = self.launch(lambda: engine.generate(
                            [{"role": "user", "content": "hello"}], 128, threading.Event(), candidate_id=self.b))
                        self.assertTrue(preparing.wait(5))
                        closer = self.launch(engine.close)
                        self.assertTrue(engine._stopping.wait(5))
                        release.set()
                        self.assert_stopping(producer)
                        self.assertNotIn("error", self.finish(closer))
                    added = set((root / "accepted").iterdir()) - accepted
                    self.assertEqual([read_json(path)["action"] for path in added], ["unload"])
                    self.assertEqual(list((root / "commands").iterdir()), [])
                    # Cancellation may now precede even the request directory.
                    self.assertEqual(list((root / "requests").glob("*")), [])
                    self.assertFalse(engine.ownership.path.exists())
                finally:
                    release.set()
                    if producer:
                        self.finish(producer)
                    if closer:
                        self.finish(closer)

    def test_close_cancels_active_work_and_releases_only_after_settlement(self):
        engine = self.engine(timeout=60)
        gate, stop = self.root / "generation", threading.Event()
        producer = self.launch(lambda: engine.generate(
            [{"role": "user", "content": "gate:" + str(gate)}], 128, stop))
        closer = None
        try:
            self.until(gate.with_suffix(".started").exists, "producer did not reach its gate")
            tree = engine.runtime.tree
            pid = tree.process.pid
            retire = tree.close
            retired = []

            def checked_retirement():
                self.assertTrue(gate.with_suffix(".settled").exists())
                self.assertTrue(engine.ownership.path.exists())
                retired.append(True)
                retire()

            with patch.object(tree, "close", side_effect=checked_retirement):
                closer = self.launch(engine.close)
                self.assert_stopping(producer)
                self.assertNotIn("error", self.finish(closer))
            self.assertTrue(retired)
            self.assertFalse(stop.is_set(), "shutdown must not mutate the caller's event")
            self.assertTrue(engine.closed)
            self.assertIsNone(engine.runtime.tree)
            self.assertIsNone(process_identity(pid))
            self.assertFalse(engine.ownership.path.exists())
            engine.close()  # Successful close is idempotent.
            with self.assertRaisesRegex(ValueError, "closed"):
                engine.open()
        finally:
            stop.set()
            gate.with_suffix(".release").touch()
            self.finish(producer)
            if closer:
                self.finish(closer)

    def test_shutdown_rejects_queued_switch_while_unresponsive_host_is_retiring(self):
        engine = self.engine(timeout=60)
        stop, queued_stop = threading.Event(), threading.Event()
        producer = self.launch(lambda: engine.generate([{"role": "user", "content": "hang"}], 128, stop))
        retiring, release, attempted = threading.Event(), threading.Event(), threading.Event()
        queued = closer = None
        retire = engine.runtime.close

        def held_retirement():
            retiring.set()
            if not release.wait(10):
                raise RuntimeError("test retirement gate was not released")
            retire()

        def queued_request():
            attempted.set()
            return engine.generate([{"role": "user", "content": "hello"}], 128, queued_stop, candidate_id=self.b)

        try:
            self.generating(engine)
            tree, root = engine.runtime.tree, engine.runtime.root
            accepted = set((root / "accepted").iterdir())
            queued = self.launch(queued_request)
            self.assertTrue(attempted.wait(5))
            self.assertFalse(queued[1].wait(.1))
            with patch.object(engine.runtime, "close", side_effect=held_retirement):
                try:
                    closer = self.launch(engine.close)
                    self.assertTrue(retiring.wait(5), "unresponsive producer was not retired")
                    self.assert_stopping(queued)
                    with self.assertRaisesRegex(EngineError, "stopping") as failure:
                        engine.generate([{"role": "user", "content": "hello"}], 128, threading.Event())
                    self.assertEqual(failure.exception.status, 503)
                    self.assertFalse(closer[1].is_set())
                    self.assertFalse(engine.closed)
                    self.assertTrue(engine.lock.locked())
                    self.assertTrue(engine.ownership.path.exists())
                    self.assertIsNone(tree.process.poll())
                    self.assertEqual(set((root / "accepted").iterdir()), accepted)
                    self.assertEqual(list((root / "commands").iterdir()), [])
                finally:
                    release.set()
                    self.assert_stopping(producer)
                    self.assertNotIn("error", self.finish(closer))
            self.assertIsNotNone(tree.process.poll())
            self.assertFalse(engine.ownership.path.exists())
        finally:
            stop.set()
            queued_stop.set()
            release.set()
            self.finish(producer)
            if queued:
                self.finish(queued)
            if closer:
                self.finish(closer)

    def test_failed_shutdown_retirement_keeps_ownership_and_can_be_retried(self):
        engine = self.engine(timeout=60)
        stop = threading.Event()
        producer = self.launch(lambda: engine.generate([{"role": "user", "content": "hang"}], 128, stop))
        closer = None
        try:
            self.generating(engine)
            tree, root = engine.runtime.tree, engine.runtime.root
            accepted = set((root / "accepted").iterdir())
            with patch.object(ProcessTree, "close", side_effect=OSError("synthetic shutdown retirement failure")):
                closer = self.launch(engine.close)
                self.assertIsInstance(self.finish(producer).get("error"), EngineError)
                self.assertIsInstance(self.finish(closer).get("error"), OSError)
                self.assertFalse(engine.closed)
                self.assertEqual(engine.status()["state"], "blocked")
                self.assertTrue(engine.ownership.path.exists())
                self.assertIsNone(tree.process.poll())
                with self.assertRaisesRegex(EngineError, "stopping"):
                    self.ask(engine)
                with self.assertRaisesRegex(ValueError, "Another CRIT runtime"):
                    self.engine(open=False).open()
                self.assertEqual(set((root / "accepted").iterdir()), accepted)
            engine.close()
            self.assertTrue(engine.closed)
            self.assertIsNone(engine.runtime.tree)
            self.assertIsNotNone(tree.process.poll())
            self.assertFalse(engine.ownership.path.exists())
            self.ask(self.engine())
        finally:
            stop.set()
            self.finish(producer)
            if closer:
                self.finish(closer)


class HttpShutdownTests(ShutdownFixture, unittest.TestCase):
    def check_shutdown(self, stream):
        engine = self.engine(timeout=60)
        server = Server(("127.0.0.1", 0), Api(engine))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        gate = self.root / "generation"
        connection = response = None
        closer = None

        def shutdown():
            server.shutdown()
            server_thread.join(5)
            server.server_close()
            engine.close()

        try:
            connection, response = self.request(server.server_address, gate, stream)
            pid = engine.runtime.tree.process.pid
            closer = self.launch(shutdown)
            self.assertNotIn("error", self.finish(closer))
            # Neither the request timeout nor a client disconnect can unblock this test.
            self.assertTrue(gate.with_suffix(".settled").exists())
            self.assertFalse(gate.with_suffix(".release").exists())
            self.assertIsNone(process_identity(pid))
            self.assertFalse(engine.ownership.path.exists())
            self.assertFalse(server_thread.is_alive())
            if stream:
                body = response.read()
                self.assertIn(b'"message": "Engine is stopping"', body)
                self.assertIn(b"data: [DONE]", body)
                self.assertNotIn(b'"finish_reason": "stop"', body)
            else:
                response = connection.getresponse()
                self.assertEqual(response.status, 503)
                self.assertEqual(json.loads(response.read())["error"]["message"], "Engine is stopping")
        finally:
            gate.with_suffix(".release").touch()
            if response:
                response.close()
            if connection:
                connection.close()
            server.shutdown()
            server_thread.join(5)
            server.server_close()
            if closer:
                self.finish(closer)

    def test_shutdown_cancels_connected_nonstreaming_client(self):
        self.check_shutdown(False)

    def test_shutdown_cancels_connected_streaming_client(self):
        self.check_shutdown(True)


class CliShutdownTests(ShutdownFixture, unittest.TestCase):
    def check_signal(self, signum, stream):
        gate = self.root / ("generation-%s-%s" % (signum, stream))
        log_path = gate.with_suffix(".log")
        options = {}
        if os.name == "nt":
            # A private hidden console confines CTRL_C_EVENT to this server.
            # CREATE_NEW_PROCESS_GROUP would disable Ctrl+C, so do not use it.
            # Children inherit an ignored Ctrl+C even with a new console, and CI
            # runners start tests that way, so re-enable it before the launch.
            ctypes.WinDLL("kernel32").SetConsoleCtrlHandler(None, False)
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            options.update(creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([
                sys.executable, "-B", str(Path(__file__).with_name("run_crit.py")), "serve",
                "--workspace", str(self.store.root), "--run", self.run_id, "--port", "0", "--timeout", "60",
            ], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options)
            connection = response = None
            try:
                def listening():
                    output = log_path.read_text(encoding="utf-8")
                    self.assertIsNone(process.poll(), output)
                    return re.search(r"http://127\.0\.0\.1:(\d+)/v1", output)

                self.until(listening, "CLI server did not start")
                port = int(listening().group(1))
                connection, response = self.request(("127.0.0.1", port), gate, stream)
                owner_path = self.root / "runtime-owner" / "owner.json"
                owner = read_json(owner_path)
                if os.name == "nt":
                    # The helper attaches only to the server's private console.
                    # Ignore Ctrl+C in the helper after AttachConsole resets its handlers.
                    helper = r'''
import ctypes
import sys
import time
api = ctypes.WinDLL("kernel32", use_last_error=True)
api.FreeConsole()
if not api.AttachConsole(int(sys.argv[1])):
    raise ctypes.WinError(ctypes.get_last_error())
try:
    if not api.SetConsoleCtrlHandler(None, True):
        raise ctypes.WinError(ctypes.get_last_error())
    if not api.GenerateConsoleCtrlEvent(0, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    time.sleep(.2)
finally:
    api.FreeConsole()
'''
                    delivered = subprocess.run(
                        [sys.executable, "-B", "-c", helper, str(process.pid)],
                        creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True, text=True, timeout=5)
                    self.assertEqual(delivered.returncode, 0, delivered.stderr)
                else:
                    process.send_signal(signum)
                self.assertEqual(process.wait(10), 0, log_path.read_text(encoding="utf-8"))
                self.assertTrue(gate.with_suffix(".settled").exists())
                self.assertFalse(gate.with_suffix(".release").exists())
                self.assertFalse(owner_path.exists())
                self.assertIsNone(process_identity(owner["process"]["pid"]))
                # The clients stay connected until after exit and ownership release.
            finally:
                gate.with_suffix(".release").touch()
                if response:
                    response.close()
                if connection:
                    connection.close()
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(10)
                if (self.root / "runtime-owner" / "owner.json").exists():
                    RuntimeOwnership().recover()

    def test_console_shutdown_cancels_active_connected_clients(self):
        signals = ("CTRL_C_EVENT",) if os.name == "nt" else (signal.SIGINT, signal.SIGTERM)
        for signum in signals:
            for stream in (False, True):
                with self.subTest(signal=signum, stream=stream):
                    self.check_signal(signum, stream)


if __name__ == "__main__":
    unittest.main()
