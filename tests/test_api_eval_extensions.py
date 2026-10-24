"""crit serve's extensions for Eval: the loaded engine's facts on /v1/models, plain decoding on request
(chat-plain-v1), draft telemetry, and packed fixed-text scores."""
import array
import base64
import http.client
import json
import threading
import unittest

from test_api_score import SCORE_ADAPTER
from test_chat_residency import ServeFixture
from crit_api.engine import telemetry
from crit_api.server import Api, Server
from crit_core.serving import ResidentChat, capabilities, engine_facts

# The engine speculates three drafts wide and decodes plainly on request; a message "decode" echoes the request's
# decode mode with a reply of seven tokens in two rounds and their draft telemetry.
EVAL_ADAPTER = SCORE_ADAPTER + r'''
Engine.facts = {'speculative': True, 'plain': True, 'draft_width': 3, 'bytes_per_pass': 1000,
                'kv_bytes_per_token': 64, 'implementation': 'fixture', 'draft_ms': float('nan')}
Engine.plain = True
_generate = Engine.generate

def _eval_generate(self, request, cancellation):
    if request['messages'][-1]['content'] != 'decode':
        return _generate(self, request, cancellation)
    response = self.response(request, json.dumps({'decode': request.get('decode')}), output_tokens=7, rounds=2)
    response['telemetry'].update(accepted=[3, 2], verify_ms=4.0, draft_ms=1.5)
    return response

Engine.generate = _eval_generate
'''


class Served(ServeFixture):
    def setUp(self):
        super().setUp()
        self.server = Server(("127.0.0.1", 0), Api(self.engine()))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, method, path, body=None):
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=30)
        self.addCleanup(connection.close)
        connection.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response, json.loads(response.read())

    def chat(self, message, **extra):
        return self.call("POST", "/v1/chat/completions",
                         {"model": "default", "messages": [{"role": "user", "content": message}], **extra})


class EvalExtensionTests(Served, unittest.TestCase):
    extra_capabilities = ("chat-score-v1", "chat-plain-v1")
    adapter_suffix = EVAL_ADAPTER

    def test_the_loaded_engines_facts_are_listed_with_its_model(self):
        response, value = self.call("GET", "/v1/models")
        self.assertEqual(response.status, 200, value)
        cards = {card["id"].rsplit("/", 1)[-1]: card for card in value["data"]}
        loaded = next(card for card in value["data"] if card["default"])
        # Numbers and booleans of the known facts only: the implementation name and a NaN draft time stay out.
        self.assertEqual(loaded["engine_facts"], {"speculative": True, "plain": True, "draft_width": 3,
                                                  "bytes_per_pass": 1000, "kv_bytes_per_token": 64})
        self.assertIsNone(cards[self.b]["engine_facts"])   # not loaded

    def test_plain_decoding_reaches_the_adapter_only_when_asked(self):
        response, value = self.chat("decode", crit={"decode": "plain"})
        self.assertEqual(response.status, 200, value)
        self.assertEqual(json.loads(value["choices"][0]["message"]["content"]), {"decode": "plain"})
        response, value = self.chat("decode")
        self.assertEqual(json.loads(value["choices"][0]["message"]["content"]), {"decode": None})
        for extension, param in (({"decode": "greedy"}, "crit.decode"), ({"speculative": False}, "crit"), ("plain", "crit")):
            with self.subTest(extension=extension):
                response, value = self.chat("decode", crit=extension)
                self.assertEqual(response.status, 400, value)
                self.assertEqual(value["error"]["param"], param)

    def test_draft_telemetry_passes_through(self):
        response, value = self.chat("decode")
        self.assertEqual(response.status, 200, value)
        found = value["crit_telemetry"]
        self.assertEqual((found["accepted"], found["verify_ms"], found["draft_ms"]), ([3, 2], 4.0, 1.5))
        self.assertEqual(found["tokens_per_round"], 3.5)

    def test_packed_scores_hold_the_json_rows(self):
        body = {"prompt_ids": [5, 6, 7], "continuation_ids": [8, 9], "top_logprobs": 2}
        _, plain = self.call("POST", "/v1/crit/score", body)
        response, packed = self.call("POST", "/v1/crit/score", {**body, "format": "packed"})
        self.assertEqual(response.status, 200, packed)
        self.assertEqual(packed["format"], "packed")

        def unpack(part):
            numbers = lambda kind, key: array.array(kind, base64.b64decode(part[key])).tolist()
            tops, values = numbers("i", "top_tokens"), numbers("f", "top_logprobs")
            k = part["alternatives"]
            return [[t, v, [[a, b] for a, b in zip(tops[i * k:(i + 1) * k], values[i * k:(i + 1) * k])]]
                    for i, (t, v) in enumerate(zip(numbers("i", "tokens"), numbers("f", "logprobs")))]
        for part in ("prompt", "continuation"):
            self.assertEqual(packed[part]["count"], len(plain[part]))
            self.assertEqual(unpack(packed[part]), plain[part])
        response, value = self.call("POST", "/v1/crit/score", {**body, "format": "npz"})
        self.assertEqual((response.status, value["error"]["param"]), (400, "format"))


class WithoutPlainTests(Served, unittest.TestCase):
    def test_plain_decoding_needs_its_capability(self):
        response, value = self.chat("hello", crit={"decode": "plain"})
        self.assertEqual(response.status, 400, value)
        self.assertEqual((value["error"]["param"], value["error"]["code"]), ("crit.decode", "unsupported_parameter"))
        _, models = self.call("GET", "/v1/models")
        self.assertTrue(all(card["engine_facts"] is None for card in models["data"]))


class TelemetryTests(unittest.TestCase):
    def test_accepted_drafts_are_one_count_per_round(self):
        request = {"limits": {"kv_tokens": 100, "max_output_tokens": 10, "timeout_seconds": 60}}
        base = {"prompt_tokens": 5, "output_tokens": 7, "prefill_tokens": 5, "prefill_ms": 1.0, "decode_tokens": 6,
                "decode_ms": 2.0, "ttft_ms": 1.0, "missing": {}, "decode_rounds": 2}
        self.assertEqual(telemetry({**base, "accepted": [3, 2], "draft_ms": 0.5}, request)["accepted"], [3, 2])
        for bad in ({"accepted": [3]}, {"accepted": [3, -1]}, {"accepted": [3, 2.0]}, {"verify_ms": -1},
                    {"accepted": [1, 1], "decode_rounds": None}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                telemetry({**base, **bad}, request)


class ResidentPlainTests(unittest.TestCase):
    def test_capabilities_and_facts_of_a_plain_engine(self):
        class Text:
            templates = {}
        self.assertIn("chat-plain-v1", capabilities(Text(), ("temperature",), plain=True))
        self.assertNotIn("chat-plain-v1", capabilities(Text(), ("temperature",)))

        class Engine(ResidentChat):
            plain = True
            facts = {"speculative": True, "plain": True, "draft_width": 2}
        self.assertEqual(engine_facts(Engine.__new__(Engine)), {"speculative": True, "plain": True, "draft_width": 2})
        Engine.plain = False   # facts alone do not make a resident engine decode plainly
        self.assertEqual(engine_facts(Engine.__new__(Engine))["plain"], False)
        self.assertIsNone(engine_facts(object()))


if __name__ == "__main__":
    unittest.main()
