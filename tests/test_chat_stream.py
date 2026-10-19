"""Provisional Unicode text, final validation and real guarded-process streaming."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "eval", "kernels", "cli"): sys.path.insert(0, str(ROOT / _folder))
from crit_api.host import Cancellation
from crit_api.stream import StreamReader, StreamWriter
from crit.run.records import read_json, write_json
from crit_api.engine import EngineError
from test_chat_residency import ADAPTER, ServeFixture


class StreamContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "stream.json"
        self.request = dict(request_id="request", candidate_id="candidate", package_sha256="package")
        self.command = dict(command_id="command", token="owner")

    def writer(self):
        return StreamWriter(self.path, self.command, self.request, Cancellation(self.root / "cancel"))

    def reader(self, updates):
        return StreamReader(self.path, "command", "owner", self.request, updates.append)

    def test_coalesced_snapshots_preserve_whitespace_and_split_graphemes(self):
        updates, writer = [], self.writer()
        reader = self.reader(updates)
        chunks = ["界 e", "\u0301 ", "👩", "🏽\u200d", "💻\n"]
        with patch("crit_api.stream.time.monotonic", return_value=10), patch("crit_api.stream.write_json", wraps=write_json) as writes:
            for chunk in chunks:
                writer.emit(chunk)
                reader.poll()
            self.assertEqual(updates, [chunks[0]])
            self.assertEqual(writes.call_count, 1)
            writer.finish({"assistant": "".join(chunks)})
            writer.close()
            reader.poll()
            reader.poll()
            self.assertEqual(writes.call_count, 2)
        self.assertEqual(updates, [chunks[0], "".join(chunks)])
        self.assertEqual(read_json(self.path)["sequence"], len(chunks))
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_invalid_chunks_cannot_be_swallowed_or_exceed_the_reply_bound(self):
        for chunk in ("", None, "x\0y", "\ud800", "x" * 65537):
            with self.subTest(chunk=repr(chunk)[:35]):
                writer = self.writer()
                with self.assertRaises((ValueError, UnicodeError)):
                    writer.emit(chunk)
                with self.assertRaises(ValueError):
                    writer.finish({"assistant": ""})
        writer = self.writer()
        writer.emit(" " * 65536)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            writer.emit("x")

    def test_callback_lifetime_thread_and_final_reply_are_checked(self):
        writer = self.writer()
        errors = []
        def wrong_thread():
            try:
                writer.emit("wrong")
            except ValueError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=wrong_thread)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIn("generation thread", errors[0])
        with self.assertRaisesRegex(ValueError, "generation thread"):
            writer.finish({"assistant": ""})
        writer = self.writer()
        writer.emit("correct")
        with self.assertRaisesRegex(ValueError, "differs"):
            writer.finish({"assistant": "rewritten"})
        writer.close()
        with self.assertRaisesRegex(ValueError, "generation thread"):
            writer.emit("late")

    def test_reader_rejects_stale_identity_rewrites_and_invalid_sequences(self):
        writer = self.writer()
        writer.emit("first")
        original = read_json(self.path)
        for change in ({"command_id": "stale"}, {"request_id": "stale"}, {"candidate_id": "other"},
                       {"package_sha256": "other"}, {"token": "other"}, {"schema_version": True},
                       {"sequence": True}, {"sequence": 0}, {"sequence": 65537},
                       {"sequence": 1, "assistant": "rewrite"}, {"sequence": 2, "assistant": "first"}):
            with self.subTest(change=change):
                write_json(self.path, original)
                updates = []
                reader = self.reader(updates)
                reader.poll()
                write_json(self.path, original | change)
                with self.assertRaises(ValueError):
                    reader.poll()
                self.assertEqual(updates, ["first"])


STREAM_ADAPTER = ADAPTER + r'''
from pathlib import Path

class StreamingEngine(Engine):
    def generate_stream(self, request, cancellation, emit):
        response = super().generate(request, cancellation)
        chunks = ['界 e', '\u0301 ', '👩', '🏽\u200d', '💻\n', response['assistant']]
        for chunk in chunks:
            emit(chunk)
            time.sleep(.06)
        if self.mode == 'hold':
            (Path(self.context.arguments[0]) / 'holding').touch()
            while not (Path(self.context.arguments[0]) / 'release').exists():
                cancellation.check()
                time.sleep(.01)
        response['assistant'] = ''.join(chunks)
        if self.mode == 'badfinal':
            response['assistant'] += 'unemitted'
        if self.mode == 'badtelemetry':
            response['telemetry']['output_tokens'] = 10000
        if self.mode == 'streamcrash':
            os._exit(19)
        if self.mode == 'swallowed':
            try:
                emit('')
            except ValueError:
                pass
        return response

def create_engine(context):
    return StreamingEngine(context)
'''


class StreamingFixture(ServeFixture):
    def package(self, candidate):
        package = self.root / ("stream-package-" + candidate["candidate_id"])
        package.mkdir(exist_ok=True)
        source = ADAPTER + "\ndel Engine.generate_stream\n" if candidate["configuration"].get("missing_stream") else STREAM_ADAPTER
        (package / "engine.py").write_text(source, encoding="utf-8")
        self.control = self.root / "stream-control"
        self.control.mkdir(exist_ok=True)
        return self.store.package(self.run_id, candidate["candidate_id"], package,
            {"build": {"toolchain": {"fixture": "host-python"}, "dependencies": [], "command": ["fixture"]},
             "launch": ["engine.py", str(self.control)],
             "capabilities": ["prefill", "decode", "chat-json-v1", "chat-resident-v1", "chat-stream-v1",
                              "chat-sampling-temperature-v1"]})

    def streaming(self, engine, message, stop=None, candidate=None):
        """Generate in a thread, collecting the cumulative text the engine streams."""
        seen, outcome, stop = [], {}, stop or threading.Event()
        def run():
            try:
                outcome["result"] = engine.generate([{"role": "user", "content": message}], 128, stop,
                                                    on_text=seen.append, candidate_id=candidate)
            except EngineError as exc:
                outcome["error"] = exc
        thread = threading.Thread(target=run)
        thread.start()
        def wait():
            thread.join(10)
            self.assertFalse(thread.is_alive())
            return outcome.get("result"), outcome.get("error")
        return seen, stop, wait

    def partial(self, seen):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if seen:
                return seen[-1]
            time.sleep(.01)
        self.fail("No provisional text appeared")


class StreamingTests(StreamingFixture, unittest.TestCase):
    def test_live_text_arrives_before_completion_then_commits_and_reuses_model(self):
        engine = self.engine()
        seen, _, wait = self.streaming(engine, "hold")
        self.assertTrue(self.partial(seen).startswith("界"))
        (self.control / "release").touch()
        (status, assistant, measured), error = wait()
        self.assertIsNone(error)
        self.assertEqual(status, "completed")
        self.assertTrue(assistant.startswith("界 e\u0301 👩🏽\u200d💻\n"))
        self.assertEqual(seen[-1], assistant)
        self.assertTrue(all(b.startswith(a) for a, b in zip(seen, seen[1:])))
        data = json.loads(assistant.split("\n", 1)[1])
        for candidate in (self.a, self.b, self.a):
            (status, assistant, _), error = self.streaming(engine, "hello", candidate=candidate)[2]()
            self.assertEqual(status, "completed", error)
            next_data = json.loads(assistant.split("\n", 1)[1])
            self.assertEqual((next_data["pid"], next_data["device"]), (data["pid"], data["device"]))
            self.assertEqual(next_data["private"], 1)
        self.assertFalse((engine.runtime.root / "stream.json").exists())

    def test_cancelled_partial_is_discarded_and_next_request_is_isolated(self):
        engine = self.engine()
        seen, stop, wait = self.streaming(engine, "hold")
        self.partial(seen)
        stop.set()
        (status, assistant, measured), error = wait()
        self.assertEqual((status, assistant, measured, error), ("cancelled", None, None, None))
        (status, assistant, _), _ = self.streaming(engine, "hello")[2]()
        self.assertEqual(status, "completed")
        self.assertEqual(json.loads(assistant.split("\n", 1)[1])["history"], 1)

    def test_failed_final_telemetry_reset_or_swallowed_emit_fails_the_request(self):
        engine = self.engine()
        for message, error in (("badfinal", "differs"), ("badtelemetry", "output tokens"),
                               ("badreset", "drain failure"), ("swallowed", "Stream text")):
            with self.subTest(message=message):
                result, failure = self.streaming(engine, message)[2]()
                self.assertIsNone(result)
                self.assertIn(error, str(failure))
                self.assertIsNone(engine.runtime.tree)
                self.assertFalse((engine.runtime.root / "stream.json").exists())

    def test_malformed_progress_retires_the_producer_before_another_request(self):
        engine = self.engine()
        seen, _, wait = self.streaming(engine, "hold")
        self.partial(seen)
        path = engine.runtime.root / "stream.json"
        # Coalescing need not publish each chunk. Wait for a producer barrier,
        # then corrupt the latest snapshot while no more emits can replace it.
        deadline = time.monotonic() + 5
        while not (self.control / "holding").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue((self.control / "holding").exists())
        value = read_json(path)
        write_json(path, value | {"request_id": "stale-request"})
        result, failure = wait()
        self.assertIsNone(result)
        self.assertIn("stream identity", str(failure))
        self.assertIsNone(engine.runtime.tree)
        self.assertFalse(path.exists())
        self.assertEqual(self.streaming(engine, "hello")[2]()[0][0], "completed")

    def test_timeout_and_host_crash_discard_observed_text(self):
        engine = self.engine()
        engine.timeout = 2
        for message, status in (("hold", "timed_out"), ("streamcrash", "failed")):
            with self.subTest(message=message):
                seen, _, wait = self.streaming(engine, message)
                self.partial(seen)
                result, failure = wait()
                if status == "failed":
                    self.assertIsNotNone(failure)
                else:
                    self.assertEqual(result, ("timed_out", None, None))
                self.assertFalse((engine.runtime.root / "stream.json").exists())

    def test_declared_streaming_requires_the_method_and_retires_failed_factory(self):
        engine = self.engine()
        candidate, *_ = self.measured(label="Missing method", configuration={"missing_stream": True})
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        result, failure = self.streaming(engine, "hello", candidate=candidate["candidate_id"])[2]()
        self.assertIn("requires generate_stream", str(failure))
        self.assertIsNone(engine.runtime.tree)
        self.assertEqual(self.streaming(engine, "hello")[2]()[0][0], "completed")
