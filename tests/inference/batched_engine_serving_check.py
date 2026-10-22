"""Real HTTP -> crit serve's resident host -> stock native adapter -> the batched synthetic engine.

Concurrent chat requests share the engine's steps (chat-concurrent-v1), each reply equals the same request served
alone, and a repeated prompt is served on reused prefix pages with the same reply."""
import http.client
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_serve_shutdown import ShutdownFixture
from crit_api.engine import ServedEngine
from crit_api.server import Api, Server

LIBRARY = Path(sys.argv[1]).resolve()
VOCAB = {f"t{i}": i for i in range(64)}


class BatchedEngineServing(ShutdownFixture, unittest.TestCase):
    model_files = {
        "tokenizer.json": json.dumps({"version": "1.0", "added_tokens": [], "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
            "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": False},
            "model": {"type": "WordLevel", "vocab": VOCAB, "unk_token": "t0"}}),
        "tokenizer_config.json": json.dumps({"chat_template": "{{ messages[-1]['content'] }}",
                                             "eos_token": None, "bos_token": None}),
        "chat_template.jinja": "{{ messages[-1]['content'] }}",
        "config.json": json.dumps({"vocab_size": 64, "eos_token_id": None}),
        "generation_config.json": json.dumps({"eos_token_id": []}),
    }

    def package(self, candidate):
        directory = self.root / ("batched-" + candidate["candidate_id"])
        directory.mkdir()
        shutil.copy2(LIBRARY, directory / LIBRARY.name)
        defaults = {"vocab": 64, "hidden": 256, "layers": 2, "heads": 4, "kv_heads": 2, "head_dim": 64, "ffn": 512,
                    "context": 512, "max_concurrency": 4, "max_batch_tokens": 64, "prefill_chunk": 32}
        (directory / "engine.py").write_text(
            "from pathlib import Path\nfrom crit_core.native import resident_engine\n"
            "def create_engine(context):\n"
            f"    return resident_engine(context, Path(__file__).with_name({LIBRARY.name!r}), defaults={defaults!r})\n")
        return self.store.package(self.run_id, candidate["candidate_id"], directory,
            {"build": {"toolchain": {"fixture": "compiled-Core"}, "dependencies": [], "command": ["cmake"]},
             "launch": ["engine.py"], "capabilities": ["prefill", "decode", "chat-json-v1", "chat-resident-v1",
                 "chat-concurrent-v1", "chat-stream-v1", "chat-sampling-temperature-v1"]})

    def served(self, candidate):
        engine = ServedEngine(self.store.root, self.run_id, candidate_id=candidate, host_limit=1 << 30, device_limit=1 << 30)
        self.addCleanup(engine.close)
        engine.open()
        server = Server(("127.0.0.1", 0), Api(engine))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        return engine, server

    def send(self, server, prompt, output=96):
        connection = http.client.HTTPConnection(*server.server_address, timeout=60)
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps({
                "messages": [{"role": "user", "content": prompt}], "max_tokens": output, "temperature": 0}),
                {"Content-Type": "application/json"})
            response = connection.getresponse()
            body = json.loads(response.read())
            self.assertEqual(response.status, 200, body)
            return body["choices"][0]["message"]["content"], body["usage"]
        finally:
            connection.close()

    def test_concurrent_replies_equal_solo_replies(self):
        engine, server = self.served(self.a)
        prompts = [" ".join(f"t{(7 * i + j) % 64}" for j in range(5 + 11 * i)) for i in range(4)]
        solo = [self.send(server, p) for p in prompts]
        busiest, stop = [0], threading.Event()

        def watch():
            while not stop.is_set():
                busiest[0] = max(busiest[0], engine.status()["active_requests"])
                time.sleep(.002)

        watcher = threading.Thread(target=watch)
        watcher.start()
        try:
            tasks = [self.launch(lambda p=p: self.send(server, p)) for p in prompts]
            replies = [self.finish(task) for task in tasks]
        finally:
            stop.set()
            watcher.join()
        for reply, alone in zip(replies, solo):
            self.assertNotIn("error", reply)
            self.assertEqual(reply["result"][0], alone[0])
            self.assertEqual(reply["result"][1]["completion_tokens"], 96)
        self.assertGreaterEqual(busiest[0], 2, "requests did not run concurrently")
        # The longest prompt again: its full pages are reused, and the reply is unchanged.
        self.assertEqual(self.send(server, prompts[-1])[0], solo[-1][0])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
