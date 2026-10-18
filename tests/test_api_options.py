"""Serving options travel across the actual resident process boundary and back over HTTP."""
import json
import unittest

import test_api
from crit_api.options import SAMPLING, RequestError, sampling_capability, sampling_options
from crit_api.engine import reply


TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
CALL = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}


class SamplingValidationTests(unittest.TestCase):
    def test_ranges_types_and_capabilities(self):
        caps = list(map(sampling_capability, SAMPLING))
        for name, value in (("temperature", -1), ("temperature", True), ("top_p", 1.1),
                            ("top_p", 0), ("seed", -1), ("seed", 2**64), ("seed", 1.5), ("top_k", 2**32),
                            ("min_p", float("nan")), ("repetition_penalty", 0),
                            ("frequency_penalty", float("inf")), ("presence_penalty", -3),
                            ("logit_bias", {"01": 1}), ("logit_bias", {"1": float("nan")}),
                            ("logit_bias", {"1": True}), ("logit_bias", {"2147483648": 1})):
            with self.subTest(name=name, value=value), self.assertRaises(RequestError):
                sampling_options({name: value}, caps)
        for name in SAMPLING:
            with self.subTest(name=name), self.assertRaises(RequestError):
                sampling_options({name: {} if name == "logit_bias" else 1}, [])
        self.assertEqual(sampling_options({"temperature": None}, []), {})

    def test_tool_reply_allows_whitespace_and_requires_new_call_ids(self):
        request = {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "tools": TOOLS,
                   "tool_choice": "auto", "limits": {"kv_tokens": 100, "max_output_tokens": 10, "timeout_seconds": 10}}
        response = {"schema_version": 1, "record_kind": "chat_response", "request_id": "r",
                    "candidate_id": "c", "package_sha256": "p", "assistant": "\n", "tool_calls": [CALL],
                    "telemetry": {"prompt_tokens": 2, "output_tokens": 2, "prefill_tokens": 2, "prefill_ms": 1,
                                  "decode_tokens": 1, "decode_ms": 1, "ttft_ms": 1, "missing": {}}}
        self.assertEqual(reply(response, request)[0], {"content": "\n", "tool_calls": [CALL]})
        request["messages"] = [{"role": "assistant", "tool_calls": [CALL]}]
        with self.assertRaisesRegex(ValueError, "reused a tool call id"):
            reply(response, request)


class RequestOptionsTests(test_api.ApiTests):
    request_options = True

    def test_sampling_reaches_adapter_and_does_not_leak_to_next_turn_or_switch(self):
        options = dict(temperature=.7, top_p=.9, top_k=12, min_p=.1, seed=42,
                       presence_penalty=.5, frequency_penalty=-.5, repetition_penalty=1.1, logit_bias={"12": -2})
        response, value = self.chat("echo-request", **options)
        self.assertEqual(response.status, 200, value)
        self.assertIsNone(response.getheader("X-Crit-Ignored-Parameters"))
        self.assertEqual(json.loads(value["choices"][0]["message"]["content"])["sampling"], options)
        _, data = self.call("GET", "/v1/models")
        card = json.loads(data)["data"][0]
        self.assertEqual(card["supported_sampling_parameters"], sorted(SAMPLING))
        self.assertIn("chat-tools-v1", card["capabilities"])
        for candidate in (self.a, self.b, self.a):
            response, value = self.chat("echo-request", model=candidate)
            self.assertEqual(response.status, 200, value)
            self.assertIsNone(json.loads(value["choices"][0]["message"]["content"])["sampling"])

    def test_tools_round_trip_with_results_and_streaming(self):
        response, value = self.chat("tool:" + json.dumps([CALL]), tools=TOOLS, tool_choice="required")
        self.assertEqual(response.status, 200, value)
        choice = value["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"], {"role": "assistant", "content": None, "tool_calls": [CALL]})
        history = [{"role": "user", "content": "Read a.py"}, choice["message"],
                   {"role": "tool", "tool_call_id": "call_1", "content": "echo-request"}]
        response, data = self.call("POST", "/v1/chat/completions", {"messages": history, "tools": TOOLS,
                                                                   "parallel_tool_calls": False})
        self.assertEqual(response.status, 200, data)
        echoed = json.loads(json.loads(data)["choices"][0]["message"]["content"])
        self.assertEqual(echoed["messages"][1]["tool_calls"], [CALL])
        self.assertEqual(echoed["messages"][-1], history[-1])
        self.assertEqual((echoed["tools"], echoed["tool_choice"], echoed["parallel_tool_calls"]), (TOOLS, "auto", False))
        response, data = self.chat("tool:" + json.dumps([CALL]), tools=TOOLS, stream=True,
                                  stream_options={"include_usage": True},
                                  tool_choice={"type": "function", "function": {"name": "read_file"}})
        chunks = [json.loads(e[6:]) for e in data.decode().split("\n\n") if e.startswith("data: {")]
        self.assertEqual(chunks[1]["choices"][0]["delta"]["tool_calls"], [dict(CALL, index=0)])
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 4)
        self.assertTrue(data.endswith(b"data: [DONE]\n\n"))

    def test_tool_policy_and_malformed_replies_are_errors(self):
        cases = [("hello", {"tool_choice": "required"}),
                 ("tool:" + json.dumps([CALL]), {"tool_choice": "none"}),
                 ("tool:" + json.dumps([CALL, dict(CALL, id="call_2")]), {"parallel_tool_calls": False}),
                 ("tool:" + json.dumps([dict(CALL, function={"name": "other", "arguments": "{}"})]), {}),
                 ("tool:" + json.dumps([dict(CALL, function={"name": "read_file", "arguments": "broken"})]), {})]
        for content, options in cases:
            with self.subTest(options=options):
                response, value = self.chat(content, tools=TOOLS, **options)
                self.assertEqual(response.status, 500, value)
        response, value = self.chat("hello", tools=TOOLS, tool_choice="none")
        self.assertEqual(response.status, 200, value)

    def test_invalid_options_fail_before_sse_headers(self):
        for options in ({"temperature": -1}, {"seed": True}, {"typo": 1}, {"tools": {}},
                        {"tools": TOOLS, "parallel_tool_calls": "false"},
                        {"tools": TOOLS, "tool_choice": "bogus"}, {"tool_choice": "required"},
                        {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": []}}},
                        {"tools": TOOLS, "stop": "STOP"},
                        {"tools": [{"type": "function", "function": {"name": "x", "strict": True}}]}):
            with self.subTest(options=options):
                response, data = self.chat(stream=True, **options)
                self.assertEqual(response.status, 400, data)
                self.assertEqual(response.getheader("Content-Type"), "application/json")

    def test_tool_history_requires_correlated_results(self):
        for history in ([{"role": "tool", "tool_call_id": "unknown", "content": "x"}],
                        [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [CALL]},
                         {"role": "user", "content": "missing result"}],
                        [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [CALL]},
                         {"role": "tool", "tool_call_id": "wrong", "content": "x"}]):
            response, data = self.call("POST", "/v1/chat/completions", {"messages": history})
            self.assertEqual(response.status, 400, data)


class LegacyOptionsTests(test_api.ApiTests):
    def test_legacy_adapter_rejects_generation_options_instead_of_ignoring_them(self):
        for options in ({"temperature": 0}, {"seed": 0}, {"tools": TOOLS}, {"frequency_penalty": 0}):
            response, value = self.chat(**options)
            self.assertEqual(response.status, 400, value)
            self.assertEqual(value["error"]["code"], "unsupported_parameter")
        response, value = self.chat(temperature=None)
        self.assertEqual(response.status, 200, value)


if __name__ == "__main__":
    unittest.main()
