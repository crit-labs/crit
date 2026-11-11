"""crit serve end to end, the one way it runs an engine: a sealed candidate's plugin package (the toy engine's plugins,
inference/examples/plugin-engine/plugins) opened by Core's loader in crit serve's resident host on Core's concurrent runtime, behind the
HTTP API, with no test hook. A model whose chat template teaches a call syntax crit writes and whose tokenizer's bytes
Core can match, so Core's adapter serves every capability it derives: chat, streaming, tool calls forced by
tool_choice, structured output, token log-probabilities, fixed-text scores, tokenizing and concurrent requests.
The protocol tests drive crit serve with a scripted adapter (support/serve.py); this module is the same API over the
real path. Needs Core's crit_core_steps (CRIT_CORE_STEP_LIBRARY) and torch."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
import threading
import unittest

from core_text_fixtures import llama_files
from support.engines import selection
from support.runs import with_plugins
from test_core_serving import TEMPLATE
from test_run_measure import MeasureFixture

try:
    import torch  # noqa: F401
    READY = bool(os.environ.get("CRIT_CORE_STEP_LIBRARY"))
except ImportError:
    READY = False

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}, "n": {"enum": [3, 7]}},
          "required": ["ok", "n"], "additionalProperties": False}
PING = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}]


@unittest.skipUnless(READY, "Core's runtime (CRIT_CORE_STEP_LIBRARY) and torch")
class ServeEndToEndTests(MeasureFixture, unittest.TestCase):
    model_files = {**{k: v for k, v in llama_files().items() if k != "model.fixture"},
                   "chat_template.jinja": TEMPLATE,
                   "config.json": {"model_type": "llama", "eos_token_id": 1,
                                   "text_config": {"eos_token_id": 4, "vocab_size": 420}}}

    def setUp(self):
        super().setUp()
        from crit_core.model import open_model
        from crit_core.text import load_model_text
        prompt = len(load_model_text(open_model(self.model).files, clock=None).encode_chat(
            self.case["messages"], add_generation_prompt=True))
        self.view = self.store.create(self.data | {"workload": {"prompt_tokens": prompt, "output_tokens": 8,
                                                                "repetitions": 1}})
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        self.store.recon(self.run_id)
        record = self.candidate(configuration={"plugins": selection(options={"toy-model": {"concurrency": 4}})})
        candidate = record["candidate_id"]
        package = with_plugins(self.root / ("package-" + candidate))
        manifest = self.store.package(self.run_id, candidate, package, {"build": {"toolchain": {"fixture": "host-python"},
                                                                                  "dependencies": [], "command": ["fixture"]}})
        self.checks(record, manifest)   # recorded before measure, as check --from-eval records them
        from crit.run import measure
        measure.measure(self.store, self.run_id, candidate)   # what sealing asks for: crit serve serves sealed ones
        self.store.seal(self.run_id, candidate, "completed")

        from crit_api.engine import ServedEngine
        from crit_api.server import Api, Server
        engine = ServedEngine(self.store.root, self.run_id, candidate_id=candidate, host_limit=1 << 30,
                              device_limit=1 << 30)
        self.addCleanup(engine.close)
        engine.open()
        server = Server(("127.0.0.1", 0), Api(engine))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        self.address = server.server_address

    def call(self, path, body=None, method=None):
        connection = http.client.HTTPConnection(*self.address, timeout=60)
        try:
            data = None if body is None else json.dumps(body).encode()
            connection.request(method or ("POST" if body is not None else "GET"), path, data,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()

    def chat(self, **body):
        status, text = self.call("/v1/chat/completions", {"model": "default", "max_tokens": 8, "temperature": 0,
                                                          "messages": [{"role": "user", "content": "hi"}], **body})
        self.assertEqual(status, 200, text)
        return json.loads(text) if not body.get("stream") else text

    def test_the_card_lists_what_core_derives_for_the_package(self):
        status, text = self.call("/v1/models")
        self.assertEqual(status, 200, text)
        (card,) = json.loads(text)["data"]
        self.assertGreater(card["max_concurrent_requests"], 1)
        self.assertEqual(card["engine_facts"]["plugins"]["model"], "toy-model")

    def test_a_chat_reply_and_its_stream(self):
        reply = self.chat()
        self.assertEqual(reply["choices"][0]["finish_reason"], "length")
        self.assertEqual(reply["usage"]["completion_tokens"], 8)
        streamed = self.chat(stream=True)
        events = [line[6:] for line in streamed.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        text = "".join(json.loads(e)["choices"][0]["delta"].get("content") or "" for e in events[:-1]
                       if json.loads(e)["choices"])
        self.assertEqual(text, reply["choices"][0]["message"]["content"])

    def test_structured_output_keeps_to_its_schema(self):
        reply = self.chat(max_tokens=32, response_format={"type": "json_schema", "json_schema": {
            "name": "r", "schema": SCHEMA, "strict": True}})
        value = json.loads(reply["choices"][0]["message"]["content"])
        self.assertEqual((set(value), value["n"] in (3, 7), type(value["ok"])), ({"ok", "n"}, True, bool))

    def test_a_forced_call(self):
        reply = self.chat(max_tokens=64, tools=PING, tool_choice="required")
        (call,) = reply["choices"][0]["message"]["tool_calls"]
        self.assertEqual((call["function"]["name"], json.loads(call["function"]["arguments"])), ("ping", {}))

    def test_token_log_probabilities(self):
        reply = self.chat(logprobs=True, top_logprobs=3)
        rows = reply["choices"][0]["logprobs"]["content"]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(len(row["top_logprobs"]) == 3 for row in rows))

    def test_fixed_text_scores(self):
        status, text = self.call("/v1/crit/score", {"model": "default", "prompt_ids": [5, 6, 7, 8],
                                                    "continuation_ids": [9, 10], "top_logprobs": 2})
        self.assertEqual(status, 200, text)
        scored = json.loads(text)
        self.assertEqual((len(scored["prompt"]), len(scored["continuation"])), (3, 2))
        self.assertEqual([row[0] for row in scored["continuation"]], [9, 10])
        self.assertTrue(all(len(row[2]) == 2 for row in scored["prompt"] + scored["continuation"]))

    def test_tokenizing_and_requests_in_flight_together(self):
        status, text = self.call("/tokenize", {"model": "default", "prompt": "hello there"})
        self.assertEqual(status, 200, text)
        self.assertTrue(json.loads(text)["tokens"])
        alone = self.chat()["choices"][0]["message"]["content"]
        with ThreadPoolExecutor(4) as pool:
            together = list(pool.map(lambda _: self.chat()["choices"][0]["message"]["content"], range(4)))
        self.assertEqual(together, [alone] * 4)


if __name__ == "__main__":
    unittest.main()
