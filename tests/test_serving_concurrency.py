"""Concurrent ResidentChat requests share weights, never request state or stream files."""
import json
import http.client
import threading
import unittest

from test_serve_shutdown import ShutdownFixture
from crit.process.ownership import process_identity
from crit_api.server import Api, Server

CONCURRENT_ADAPTER = r'''
from pathlib import Path
from types import SimpleNamespace
from crit_core.serving import ResidentChat

class Text:
    templates = {}
    tokenizer = SimpleNamespace(special_ids=set(), decode=lambda ids: ''.join(map(chr, ids)))
    def encode_chat(self, messages, **options):
        return list(map(ord, messages[-1]["content"]))
    def text_stream(self, ids, **options):
        return SimpleNamespace(push=lambda tokens: SimpleNamespace(text=''.join(map(chr, tokens))),
                               finish=lambda: SimpleNamespace(text=''))

class Request(ResidentChat):
    sampling = ("temperature",)
    text = Text()
    capacity = 8192
    stop_ids = set()
    template_variables = {}
    def __init__(self, parent, request):
        self.parent = parent
        self.private = [request["request_id"]]
        message = request["messages"][-1]["content"]
        self.gate = Path(message[5:]) if message.startswith("gate:") else None
    def generate_tokens(self, ids, maximum, sampling, on_tokens, cancelled):
        if self.gate:
            self.gate.with_suffix(".started").write_text(str(os.getpid()))
            while not self.gate.with_suffix(".release").exists():
                if cancelled() and not self.gate.with_suffix(".ignore").exists():
                    return None, {}
                time.sleep(.01)
        value = json.dumps(dict(pid=os.getpid(), device=self.parent.lease.buffer["id"],
                                private=len(self.private)))
        tokens = list(map(ord, value))
        for token in tokens:
            on_tokens([token])
        return tokens, dict(prefill_ms=2, decode_ms=3, first_token_ms=2)
    def reset(self):
        if self.gate and self.gate.with_suffix(".badreset").exists():
            raise RuntimeError("request reset did not settle")
        self.private.clear()
    def close(self):
        if self.gate:
            self.gate.with_suffix(".closed").touch()

Engine.max_concurrent_requests = 2
def create_request(self, request):
    if request["messages"][-1]["content"] == "badfactory":
        return self
    return Request(self, request)
Engine.create_request = create_request
'''


class ConcurrentTests(ShutdownFixture, unittest.TestCase):
    extra_capabilities = ("chat-concurrent-v1",)
    adapter_suffix = CONCURRENT_ADAPTER

    def turn(self, engine, name, candidate=None):
        gate = self.root / name
        stop = threading.Event()
        chunks = []
        task = self.launch(lambda: engine.generate([{"role": "user", "content": "gate:" + str(gate)}],
                                                  256, stop, candidate_id=candidate, on_text=chunks.append))
        self.addCleanup(self.finish, task)
        self.addCleanup(stop.set)
        return gate, stop, task, chunks

    def started(self, turn):
        self.until(turn[0].with_suffix(".started").exists, "request did not start")

    def release(self, turn):
        turn[0].with_suffix(".release").touch()
        result = self.finish(turn[2])
        self.assertNotIn("error", result)
        return result["result"]

    def test_overlap_capacity_private_state_streams_and_one_request_cancellation(self):
        engine = self.engine()
        first, second = self.turn(engine, "first"), self.turn(engine, "second")
        self.started(first)
        self.started(second)
        self.assertEqual(engine.status()["active_requests"], 2)
        third = self.turn(engine, "third")
        self.assertFalse(third[2][1].wait(.15))
        self.assertFalse(third[0].with_suffix(".started").exists())
        abandoned = self.turn(engine, "abandoned")
        abandoned[1].set()
        self.assertEqual(self.finish(abandoned[2])["result"][0], "cancelled")
        self.assertFalse(abandoned[0].with_suffix(".started").exists())
        first[1].set()
        self.assertEqual(self.finish(first[2])["result"][0], "cancelled")
        self.started(third)
        self.assertFalse(second[2][1].is_set())
        replies = [self.release(turn)[1] for turn in (second, third)]
        values = [json.loads(reply) for reply in replies]
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[0]["private"], 1)
        self.assertEqual("".join(second[3]), replies[0])
        self.assertEqual("".join(third[3]), replies[1])
        self.assertTrue(first[0].with_suffix(".closed").exists())
        self.assertFalse((engine.runtime.root / "requests").exists())

    def test_forced_retirement_fails_peer_before_new_admission(self):
        engine = self.engine()
        first = self.turn(engine, "ignored")
        first[0].with_suffix(".ignore").touch()
        second = self.turn(engine, "peer")
        self.started(first)
        self.started(second)
        pid = engine.runtime.tree.process.pid
        first[1].set()
        self.assertEqual(self.finish(first[2])["result"][0], "cancelled")
        self.assertIn("error", self.finish(second[2]))
        self.assertIsNone(process_identity(pid))
        after = self.ask(engine)
        self.assertNotEqual(after["pid"], pid)
        self.assertEqual(after["private"], 1)

    def test_candidate_switch_drains_peers_and_cannot_be_starved(self):
        engine = self.engine()
        first = self.turn(engine, "old")
        self.started(first)
        switch = self.turn(engine, "switch", self.b)
        self.until(lambda: len(engine.lock.waiters) == 1, "switch did not queue")
        later = self.turn(engine, "later")
        self.release(first)
        self.started(switch)
        self.assertFalse(later[0].with_suffix(".started").exists())
        self.assertEqual(engine.status()["engine"]["candidate_id"], self.b)
        self.release(switch)
        self.started(later)
        self.release(later)
        self.assertEqual(engine.status()["engine_loads"], 3)

    def test_shutdown_settles_every_request_and_rejects_shared_request_object(self):
        engine = self.engine()
        first, second = self.turn(engine, "close1"), self.turn(engine, "close2")
        self.started(first)
        self.started(second)
        engine.close()
        self.assert_stopping(first[2])
        self.assert_stopping(second[2])
        self.assertTrue(first[0].with_suffix(".closed").exists())
        self.assertTrue(second[0].with_suffix(".closed").exists())
        self.assertFalse(engine.ownership.path.exists())
        other = self.engine()
        with self.assertRaisesRegex(Exception, "distinct request object"):
            self.ask(other, "badfactory")
        self.assertIsNone(other.runtime.tree)

    def test_http_clients_overlap_with_independent_streaming_and_json_responses(self):
        engine = self.engine()
        server = Server(("127.0.0.1", 0), Api(engine))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        tasks = []
        for index, streaming in enumerate((False, True)):
            gate = self.root / ("http" + str(index))
            self.addCleanup(gate.with_suffix(".release").touch)

            def send(gate=gate, streaming=streaming):
                client = http.client.HTTPConnection(*server.server_address, timeout=20)
                try:
                    client.request("POST", "/v1/chat/completions", json.dumps({
                        "messages": [{"role": "user", "content": "gate:" + str(gate)}],
                        "stream": streaming, "max_tokens": 256}))
                    response = client.getresponse()
                    return response.status, response.read()
                finally:
                    client.close()

            tasks.append((gate, self.launch(send)))
        try:
            for gate, _ in tasks:
                self.until(gate.with_suffix(".started").exists, "HTTP request did not overlap")
            self.assertEqual(engine.status()["active_requests"], 2)
            for gate, _ in tasks:
                gate.with_suffix(".release").touch()
            responses = [self.finish(task)["result"] for _, task in tasks]
            self.assertTrue(all(status == 200 for status, _ in responses))
            plain = json.loads(json.loads(responses[0][1])["choices"][0]["message"]["content"])
            events = [json.loads(line[6:]) for line in responses[1][1].splitlines()
                      if line.startswith(b"data: ") and line != b"data: [DONE]"]
            streamed = json.loads("".join(e["choices"][0]["delta"].get("content", "") for e in events))
            self.assertEqual(plain, streamed)
            self.assertEqual(plain["private"], 1)
        finally:
            for gate, task in tasks:
                gate.with_suffix(".release").touch()
                self.finish(task)

    def test_failed_request_reset_retires_without_closing_unsettled_resources(self):
        engine = self.engine()
        turn = self.turn(engine, "badreset")
        turn[0].with_suffix(".badreset").touch()
        self.started(turn)
        pid = engine.runtime.tree.process.pid
        turn[0].with_suffix(".release").touch()
        self.assertIn("reset did not settle", str(self.finish(turn[2]).get("error")))
        self.assertIsNone(process_identity(pid))
        self.assertFalse(turn[0].with_suffix(".closed").exists())
        self.assertNotEqual(self.ask(engine)["pid"], pid)
