"""Token log-probabilities through crit serve (chat-logprobs-v1) with a synthetic resident adapter."""
import http.client
import json
import threading
import unittest

from test_chat_residency import ServeFixture
from crit_api.engine import reply
from crit_api.logprobs import logprob_entries, logprob_options, response_bytes
from crit_api.options import RequestError
from crit_api.server import Api, Server

# Two reply tokens: "h" (104) and the first byte of "é" (195, which alone is no text).
LOGPROB_ADAPTER = r'''
Engine.logprobs = True
_reply = Engine.generate

def _generate(self, request, cancellation):
    response = _reply(self, request, cancellation)
    n = request.get('logprobs')
    if n is not None:
        rows = [[104, -0.25, [[104, -0.25], [195, -1.5], [233, -2.0]][:n]],
                [195, -0.5, [[233, -0.125], [195, -0.5], [104, -3.0]][:n]]]
        if request['messages'][-1]['content'] == 'unranked':
            rows[0][2].reverse()
        response['logprobs'] = {'tokens': rows, 'bytes': {'104': '68', '195': 'c3', '233': 'c3a9'}}
    return response

Engine.generate = _generate
'''


def record(rows, named=None):
    return {"tokens": rows, "bytes": {"104": "68", "195": "c3"} if named is None else named}


class LogprobRecordTests(unittest.TestCase):
    request = {"logprobs": 1, "limits": {"max_output_tokens": 4}}

    def test_options_follow_openai_and_the_capability(self):
        declared = ["chat-logprobs-v1"]
        self.assertIsNone(logprob_options({}, declared))
        self.assertIsNone(logprob_options({"logprobs": False, "top_logprobs": 0}, []))
        self.assertEqual(logprob_options({"logprobs": True}, declared), 0)
        self.assertEqual(logprob_options({"logprobs": True, "top_logprobs": 20}, declared), 20)
        for body, param in (({"logprobs": True}, "logprobs"), ({"logprobs": "yes"}, "logprobs"),
                            ({"top_logprobs": 2}, "top_logprobs"), ({"logprobs": True, "top_logprobs": 21}, "top_logprobs"),
                            ({"logprobs": True, "top_logprobs": 1.0}, "top_logprobs")):
            with self.subTest(body=body), self.assertRaises(RequestError) as caught:
                logprob_options(body, [] if param == "logprobs" and body["logprobs"] is True else declared)
            self.assertEqual(caught.exception.param, param)

    def test_rows_become_openai_entries_with_token_text_and_bytes(self):
        entries = logprob_entries(record([[104, -0.25, [[104, -0.25]]], [195, -9999.0, [[104, -1]]]]), self.request)
        self.assertEqual(entries[0], {"token": "h", "logprob": -0.25, "bytes": [104],
                                      "top_logprobs": [{"token": "h", "logprob": -0.25, "bytes": [104]}]})
        self.assertEqual((entries[1]["token"], entries[1]["bytes"], entries[1]["logprob"]), ("�", [195], -9999.0))

    def test_malformed_rows_are_invalid_responses(self):
        cases = ([[104, -0.25, []]],                       # fewer alternatives than requested
                 [[104, 0.5, [[104, -0.25]]]],             # a positive log-probability
                 [[104, -10000.0, [[104, -0.25]]]],        # below what a zero probability reads as
                 [[7, -0.25, [[104, -0.25]]]],             # a token without bytes
                 [[104, -0.25, [[104, "x"]]]],
                 [[104, -0.25, [[104, -0.25]]]] * 5)       # more rows than output tokens
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                logprob_entries(record(rows), self.request)
        two = dict(self.request, logprobs=2)
        for top in ([[104, -1.0], [195, -0.5]], [[104, -0.5], [104, -0.5]]):   # unranked, repeated
            with self.subTest(top=top), self.assertRaises(ValueError):
                logprob_entries(record([[104, -0.25, top]]), two)
        for named in ({"104": "6"}, {"104": "6G"}, {"x": "68"}, {"104": "C3"}):
            with self.subTest(named=named), self.assertRaises(ValueError):
                logprob_entries(record([[104, -0.25, [[104, -0.25]]]], named), self.request)

    def test_a_reply_carries_logprobs_exactly_when_the_request_asked(self):
        request = {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "logprobs": 0,
                   "limits": {"max_output_tokens": 4, "kv_tokens": 64, "timeout_seconds": 60}}
        value = {"schema_version": 1, "record_kind": "chat_response", "request_id": "r", "candidate_id": "c",
                 "package_sha256": "p", "assistant": "h",
                 "telemetry": {"prompt_tokens": 3, "output_tokens": 1, "prefill_tokens": 3, "prefill_ms": 1,
                               "decode_tokens": None, "decode_ms": None, "ttft_ms": 1,
                               "missing": {"decode": "the reply has a single token"}},
                 "logprobs": record([[104, -0.25, []]])}
        assistant, _ = reply(value, request)
        self.assertEqual((assistant["content"], assistant["logprobs"][0]["token"]), ("h", "h"))
        with self.assertRaisesRegex(ValueError, "do not match the request"):
            reply(value, {k: v for k, v in request.items() if k != "logprobs"})
        with self.assertRaisesRegex(ValueError, "do not match the request"):
            reply({k: v for k, v in value.items() if k != "logprobs"}, request)
        with self.assertRaisesRegex(ValueError, "exceed its output tokens"):
            reply(dict(value, logprobs=record([[104, -0.25, []]] * 2)), request)

    def test_the_response_record_may_grow_with_the_rows_asked_for(self):
        self.assertEqual(response_bytes({"limits": {"max_output_tokens": 10}}), 1 << 20)
        self.assertGreater(response_bytes({"logprobs": 20, "limits": {"max_output_tokens": 10}}), (1 << 20) + 200_000)


class LogprobApiTests(ServeFixture, unittest.TestCase):
    extra_capabilities = ("chat-logprobs-v1",)
    adapter_suffix = LOGPROB_ADAPTER
    stream = True

    def setUp(self):
        super().setUp()
        self.server = Server(("127.0.0.1", 0), Api(self.engine()))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def chat(self, content="hello", **options):
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=30)
        self.addCleanup(connection.close)
        connection.request("POST", "/v1/chat/completions", json.dumps(
            {"model": "default", "messages": [{"role": "user", "content": content}], **options}),
            {"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        return response, data if options.get("stream") else json.loads(data)

    def test_invalid_logprob_requests_are_rejected_in_openai_shape(self):
        for options, param in (({"logprobs": True, "top_logprobs": 21}, "top_logprobs"),
                               ({"top_logprobs": 2}, "top_logprobs"), ({"logprobs": 1}, "logprobs"),
                               ({"logprobs": True, "stop": "x"}, "stop")):
            with self.subTest(options=options):
                response, value = self.chat(**options)
                self.assertEqual(response.status, 400, value)
                self.assertEqual(value["error"]["param"], param)

    def test_a_reply_lists_each_token_with_its_bytes_and_alternatives(self):
        response, value = self.chat(logprobs=True, top_logprobs=2)
        self.assertEqual(response.status, 200, value)
        entries = value["choices"][0]["logprobs"]["content"]
        self.assertEqual([(e["token"], e["bytes"], e["logprob"]) for e in entries], [("h", [104], -0.25),
                                                                                        ("�", [195], -0.5)])
        self.assertEqual(entries[1]["top_logprobs"], [{"token": "é", "logprob": -0.125, "bytes": [195, 169]},
                                                      {"token": "�", "logprob": -0.5, "bytes": [195]}])
        response, value = self.chat(logprobs=True)
        self.assertEqual([e["top_logprobs"] for e in value["choices"][0]["logprobs"]["content"]], [[], []])
        response, value = self.chat()
        self.assertIsNone(value["choices"][0]["logprobs"])

    def test_streamed_logprobs_come_with_the_finish_chunk(self):
        response, data = self.chat("hello", stream=True, logprobs=True, top_logprobs=1)
        chunks = [json.loads(line[6:]) for line in data.decode().split("\n\n") if line.startswith("data: {")]
        carrying = [c for c in chunks if c["choices"][0]["logprobs"] is not None]
        self.assertEqual(len(carrying), 1)
        self.assertEqual(carrying[0]["choices"][0]["finish_reason"], "stop")
        self.assertEqual([e["token"] for e in carrying[0]["choices"][0]["logprobs"]["content"]], ["h", "�"])

    def test_malformed_logprobs_from_the_adapter_are_a_server_error(self):
        response, value = self.chat("unranked", logprobs=True, top_logprobs=2)
        self.assertEqual(response.status, 500, value)
        self.assertIn("ranked", value["error"]["message"])


class UndeclaredLogprobTests(ServeFixture, unittest.TestCase):
    extra_capabilities = ("chat-logprobs-v1",)

    def test_the_capability_needs_an_engine_that_reports_logprobs(self):
        with self.assertRaisesRegex(ValueError, "chat-logprobs-v1 requires an engine that reports log-probabilities"):
            self.engine()


if __name__ == "__main__":
    unittest.main()
