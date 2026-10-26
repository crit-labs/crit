"""Provisional Unicode text, final validation and real guarded-process streaming."""
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "../crit-eval", "kernels", "cli"): sys.path.insert(0, str(ROOT / _folder))
from crit_api.host import Cancellation
from crit_api.stream import StreamReader, StreamWriter
from crit_api.engine import EngineError
from test_chat_residency import ADAPTER, ServeFixture


class StreamContractTests(unittest.TestCase):
    def setUp(self):
        self.request = dict(request_id="request", candidate_id="candidate", package_sha256="package",
                            limits={"kv_tokens": 64, "max_output_tokens": 16, "timeout_seconds": 60})
        self.sent = []

    def writer(self, tools=False):
        return StreamWriter(self.sent.append, "command", self.request, Cancellation(), tools=tools)

    def reader(self, updates, tools=None, reasoning=None):
        return StreamReader(self.request, updates.append, tools, reasoning)

    def piece(self, **part):
        return {"kind": "progress", "command_id": "command", **part}

    def test_each_chunk_is_sent_at_once_and_published_exactly(self):
        updates, writer = [], self.writer()
        reader = self.reader(updates)
        chunks = ["界 e", "\u0301 ", "👩", "🏽\u200d", "💻\n"]
        for chunk in chunks:
            writer.emit(chunk)
            reader.apply(self.sent[-1])
            self.assertEqual(updates[-1], chunk)
        writer.finish({"assistant": "".join(chunks)})
        writer.close()
        self.assertEqual(self.sent, [self.piece(text=chunk) for chunk in chunks])
        self.assertEqual(updates, chunks)

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

    def test_reader_rejects_malformed_or_oversized_pieces(self):
        for piece in (self.piece(), self.piece(text="a", reasoning="b"), self.piece(text=""), self.piece(text=3),
                      self.piece(text="x\0y"), self.piece(text="\ud800"), self.piece(reasoning=None),
                      self.piece(other="x"), {"kind": "progress", "text": "a"},
                      self.piece(tool_calls=[{"index": 0, "id": "call_1", "type": "function",
                                              "function": {"name": "lookup", "arguments": "{}"}}])):
            with self.subTest(piece=piece):
                updates = []
                with self.assertRaises(ValueError):
                    self.reader(updates).apply(piece)
                self.assertEqual(updates, [])
        updates = []
        reader = self.reader(updates)
        reader.apply(self.piece(text=" " * 65536))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            reader.apply(self.piece(text="x"))
        self.assertEqual(len(updates), 1)


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
        if self.mode == 'malformed':
            # A piece that bypasses the writer's checks: crit serve checks every piece again.
            writer = emit.__self__
            writer.send({'kind': 'progress', 'command_id': writer.command_id, 'text': ''})
            while True:
                cancellation.check()
                time.sleep(.01)
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
        """Generate in a thread, collecting the pieces of text the engine streams."""
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
                return "".join(seen)
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
        self.assertEqual(seen[:5], ["界 e", "\u0301 ", "👩", "🏽\u200d", "💻\n"])  # each chunk as emitted
        self.assertEqual("".join(seen), assistant)
        data = json.loads(assistant.split("\n", 1)[1])
        for candidate in (self.a, self.b, self.a):
            (status, assistant, _), error = self.streaming(engine, "hello", candidate=candidate)[2]()
            self.assertEqual(status, "completed", error)
            next_data = json.loads(assistant.split("\n", 1)[1])
            self.assertEqual((next_data["pid"], next_data["device"]), (data["pid"], data["device"]))
            self.assertEqual(next_data["private"], 1)

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

    def check_consumer_failure(self, error):
        engine = self.engine()
        self.addCleanup(engine._retire, "stream consumer regression cleanup")
        root, tree = engine.runtime.root, engine.runtime.tree
        held, observed = [], []
        close = engine.runtime.close

        def retire():
            held.append(engine.lock.locked())
            close()

        def consume(text):
            observed.append(text)
            raise error

        with patch.object(engine.runtime, "close", side_effect=retire):
            with self.assertRaises(type(error)) as caught:
                engine.generate([{"role": "user", "content": "hold"}], 128,
                                threading.Event(), on_text=consume)
        self.assertIs(caught.exception, error)
        self.assertTrue(observed)
        self.assertIsNone(engine.runtime.tree)
        self.assertIsNotNone(tree.process.poll())
        self.assertEqual(held, [True])
        self.assertFalse(engine.lock.locked())
        self.assertFalse((root / "models").exists())
        (status, assistant, _), failure = self.streaming(engine, "hello")[2]()
        self.assertEqual(status, "completed", failure)
        self.assertNotEqual(engine.runtime.root, root)
        data = json.loads(assistant.split("\n", 1)[1])
        self.assertEqual((data["calls"], data["private"], data["history"]), (1, 1, 1))

    def test_consumer_exception_retires_before_unlock_and_preserves_original_error(self):
        self.check_consumer_failure(RuntimeError("consumer failed"))

    def test_consumer_interrupt_retires_before_unlock_and_preserves_interrupt(self):
        self.check_consumer_failure(KeyboardInterrupt("consumer interrupted"))

    def test_consumer_failure_with_failed_retirement_blocks_reuse(self):
        engine = self.engine()
        self.addCleanup(engine._retire, "failed retirement regression cleanup")
        root, tree = engine.runtime.root, engine.runtime.tree
        error = RuntimeError("consumer failed")

        def consume(text):
            raise error

        with patch.object(engine.runtime, "close", side_effect=RuntimeError("retirement failed")):
            with self.assertRaises(RuntimeError) as caught:
                engine.generate([{"role": "user", "content": "hold"}], 128,
                                threading.Event(), on_text=consume)
            self.assertIs(caught.exception, error)
            self.assertEqual(engine.status()["state"], "blocked")
            self.assertIn("retirement failed", engine.ownership.blocked)
            self.assertIs(engine.runtime.tree, tree)
            self.assertIsNone(tree.process.poll())
            with self.assertRaisesRegex(EngineError, "cleanup is unresolved") as blocked:
                self.ask(engine)
            self.assertEqual(blocked.exception.status, 503)
            with self.assertRaisesRegex(ValueError, "Another CRIT runtime"):
                self.engine(open=False).open()
        engine._retire("retry after synthetic cleanup failure")
        self.assertIsNotNone(tree.process.poll())
        engine.close()
        self.assertFalse(engine.ownership.path.exists())
        self.assertEqual(self.streaming(self.engine(), "hello")[2]()[0][0], "completed")

    def test_queued_request_waits_for_consumer_failure_retirement(self):
        engine = self.engine()
        self.addCleanup(engine._retire, "queued retirement regression cleanup")
        root, tree = engine.runtime.root, engine.runtime.tree
        retiring, release = threading.Event(), threading.Event()
        attempted, finished = threading.Event(), threading.Event()
        outcome = {}
        error = RuntimeError("consumer failed")
        close = engine.runtime.close

        def retire():
            retiring.set()
            if not release.wait(10):
                raise RuntimeError("test did not release retirement")
            close()

        def consume(text):
            raise error

        def first_request():
            try:
                engine.generate([{"role": "user", "content": "hold"}], 128,
                                threading.Event(), on_text=consume)
            except BaseException as exc:
                outcome["first_error"] = exc

        def queued_request():
            attempted.set()
            try:
                outcome["second"] = engine.generate([{"role": "user", "content": "hello"}],
                                                    128, threading.Event())
            except BaseException as exc:
                outcome["second_error"] = exc
            finally:
                finished.set()

        first, second = threading.Thread(target=first_request), threading.Thread(target=queued_request)
        link = engine.runtime.link
        with patch.object(engine.runtime, "close", side_effect=retire), \
                patch.object(link, "send", wraps=link.send) as send:
            try:
                first.start()
                self.assertTrue(retiring.wait(5), "consumer failure did not begin retirement")
                second.start()
                self.assertTrue(attempted.wait(5))
                self.assertFalse(finished.wait(.1), "queued request passed unfinished retirement")
                self.assertTrue(engine.lock.locked())
                self.assertIsNone(tree.process.poll())
                self.assertEqual([call.args[0]["action"] for call in send.call_args_list
                                  if call.args[0]["kind"] == "command"], ["generate"])
            finally:
                release.set()
                first.join(10)
                if second.ident is not None:
                    second.join(10)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertIs(outcome.get("first_error"), error)
        self.assertNotIn("second_error", outcome)
        self.assertIsNotNone(tree.process.poll())
        self.assertNotEqual(engine.runtime.root, root)
        status, assistant, _ = outcome["second"]
        self.assertEqual(status, "completed")
        data = json.loads(assistant.split("\n", 1)[1])
        self.assertEqual((data["calls"], data["private"], data["history"]), (1, 1, 1))

    def test_failed_final_telemetry_reset_or_swallowed_emit_fails_the_request(self):
        engine = self.engine()
        for message, error in (("badfinal", "differs"), ("badtelemetry", "output tokens"),
                               ("badreset", "drain failure"), ("swallowed", "Stream text")):
            with self.subTest(message=message):
                result, failure = self.streaming(engine, message)[2]()
                self.assertIsNone(result)
                self.assertIn(error, str(failure))
                self.assertIsNone(engine.runtime.tree)

    def test_malformed_progress_retires_the_producer_before_another_request(self):
        engine = self.engine()
        seen, _, wait = self.streaming(engine, "malformed")
        result, failure = wait()
        self.assertIsNone(result)
        self.assertIn("Stream text must be nonempty", str(failure))
        self.assertEqual(seen[:5], ["界 e", "\u0301 ", "👩", "🏽\u200d", "💻\n"])
        self.assertIsNone(engine.runtime.tree)
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

    def test_declared_streaming_requires_the_method_and_retires_failed_factory(self):
        engine = self.engine()
        candidate, *_ = self.measured(label="Missing method", configuration={"missing_stream": True})
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        result, failure = self.streaming(engine, "hello", candidate=candidate["candidate_id"])[2]()
        self.assertIn("requires generate_stream", str(failure))
        self.assertIsNone(engine.runtime.tree)
        self.assertEqual(self.streaming(engine, "hello")[2]()[0][0], "completed")
