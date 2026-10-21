"""Real HTTP -> resident subprocess -> stock native adapter -> Core scheduler."""
import http.client
import json
from pathlib import Path
import shutil
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_serve_shutdown import ShutdownFixture
from crit_api.engine import ServedEngine
from crit_api.server import Api, Server

LIBRARY = Path(sys.argv[1]).resolve()


class NativeServing(ShutdownFixture, unittest.TestCase):
    model_files = {
        "tokenizer.json": json.dumps({"version": "1.0", "added_tokens": [], "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
            "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": False},
            "model": {"type": "WordLevel", "vocab": {chr(97+i): i for i in range(16)}, "unk_token": "a"}}),
        "tokenizer_config.json": json.dumps({"chat_template": "{{ messages[-1]['content'] }}",
                                             "eos_token": None, "bos_token": None}),
        "chat_template.jinja": "{{ messages[-1]['content'] }}",
        "config.json": json.dumps({"vocab_size": 16, "eos_token_id": None}),
        "generation_config.json": json.dumps({"eos_token_id": []}),
    }

    def package(self, candidate):
        directory = self.root / ("native-" + candidate["candidate_id"])
        directory.mkdir()
        shutil.copy2(LIBRARY, directory / LIBRARY.name)
        attention = candidate["candidate_id"].endswith("2")
        defaults = {"control_directory": str(self.root / ("control-" + candidate["candidate_id"])),
                    "attention": attention}
        (directory / "engine.py").write_text(
            "from pathlib import Path\nfrom crit_core.native import resident_engine\n"
            "def create_engine(context):\n"
            f"    return resident_engine(context, Path(__file__).with_name({LIBRARY.name!r}), defaults={defaults!r})\n")
        return self.store.package(self.run_id, candidate["candidate_id"], directory,
            {"build": {"toolchain": {"fixture": "compiled-Core"}, "dependencies": [], "command": ["cmake"]},
             "launch": ["engine.py"], "capabilities": ["prefill", "decode", "chat-json-v1", "chat-resident-v1",
                 "chat-concurrent-v1", "chat-stream-v1", "chat-sampling-temperature-v1"]})

    def served(self, candidate):
        engine = ServedEngine(self.store.root, self.run_id, candidate_id=candidate, host_limit=64 << 20, device_limit=64 << 20)
        self.addCleanup(engine.close)
        engine.open()
        server = Server(("127.0.0.1", 0), Api(engine))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        control = self.root / ("control-" + candidate)
        control.mkdir(exist_ok=True)
        return engine, server, control

    def send(self, server, stream=False, prompt="b c d e f g", output=5):
        connection = http.client.HTTPConnection(*server.server_address, timeout=20)
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps({
                "messages": [{"role": "user", "content": prompt}], "max_tokens": output, "stream": stream}),
                {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def check_overlap(self, candidate):
        engine, server, control = self.served(candidate)
        self.addCleanup((control / "release").touch)
        first = self.launch(lambda: self.send(server, stream=True))
        second = self.launch(lambda: self.send(server))
        try:
            self.until(lambda: len(list(control.glob("queued-*"))) == 2, "HTTP requests did not share native Core")
            self.assertEqual(engine.status()["active_requests"], 2)
            pid = engine.runtime.tree.process.pid
            (control / "release").touch()
            one, two = self.finish(first), self.finish(second)
            self.assertNotIn("error", one)
            self.assertNotIn("error", two)
            self.assertEqual(one["result"][0], 200, one)
            self.assertEqual(two["result"][0], 200, two)
            self.assertIn(b"[DONE]", one["result"][1])
            self.assertEqual(json.loads(two["result"][1])["usage"]["completion_tokens"], 5)
            results = [json.loads(p.read_text()) for p in control.glob("result-*")]
            self.assertEqual(len(results), 2)
            self.assertTrue(any(r["preemptions"] for r in results), results)
            self.assertEqual(engine.runtime.tree.process.pid, pid)
            self.assertEqual(self.send(server)[0], 200)
        finally:
            (control / "release").touch()
            self.finish(first)
            self.finish(second)

    def test_recurrent_native_http_recovery(self):
        self.check_overlap(self.a)

    def test_attention_native_http_recovery(self):
        self.check_overlap(self.b)

    def test_normal_http_disconnect_cancels_native_prefill_and_keeps_peer(self):
        engine, server, control = self.served(self.a)
        (control / "release").touch()
        connection = http.client.HTTPConnection(*server.server_address, timeout=20)
        self.addCleanup(connection.close)
        connection.request("POST", "/v1/chat/completions", json.dumps({
            "messages": [{"role": "user", "content": " ".join(["b"] * 50)}], "max_tokens": 5}),
            {"Content-Type": "application/json"})
        self.until((control / "computed-1").exists, "native prefill did not begin")
        pid = engine.runtime.tree.process.pid
        peer = self.launch(lambda: self.send(server, prompt="c", output=3))
        connection.close()  # ordinary HTTP client close, no reset or special cancellation endpoint
        retired = control / "retired-1"
        self.until(lambda: retired.exists() and retired.read_text() == "2",
                   "disconnect did not retire native generation as cancelled")  # Finish::cancelled
        self.assertEqual(self.finish(peer)["result"][0], 200)
        self.assertEqual(engine.runtime.tree.process.pid, pid)
        self.assertEqual(self.send(server)[0], 200)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
