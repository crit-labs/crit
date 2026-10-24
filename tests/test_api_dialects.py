"""crit serve's other dialects over the synthetic resident engine: the Anthropic Messages API, the OpenAI Responses
API, the Ollama API, the Prometheus metrics endpoint and unloading and loading the engine."""
import http.client
import json
from pathlib import Path
import socket
import sys
import threading
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "eval", "kernels", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit import __version__
from crit_api import anthropic, ollama
from crit_api.engine import EngineError
from crit_api.metrics import Histogram, Metrics
from crit_api.server import Api, Server
from crit_eval import chat
from crit_eval.deadline import Hung
from test_api_cancellation import GATED_ADAPTER
from test_chat_residency import ServeFixture
from test_core_reasoning import ADAPTER as REASONING_ADAPTER
from test_tool_stream import ADAPTER as TOOL_ADAPTER

TOOLS = [{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                                                                             "required": ["path"]}}]
OPENAI_TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": TOOLS[0]["input_schema"]}}]
RESPONSES_TOOLS = [{"type": "function", "name": "read_file", "description": "Read a file", "parameters": TOOLS[0]["input_schema"]}]
CALL = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"}}
STRICT_TOOLS = [dict(TOOLS[0], strict=True, input_schema=dict(TOOLS[0]["input_schema"], additionalProperties=False))]
IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"}}


def samples(text):
    """{sample: value} of a Prometheus exposition text."""
    return {line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
            if line and not line.startswith("#")}


# A message "steps:[[kind, value], ...]" streams text and calls in that order (chat-tool-stream-v1); "hold" waits to
# be cancelled.
STEPS = r'''
class StepsEngine(Engine):
    def generate_stream(self, request, cancellation, emit):
        message = request['messages'][-1]['content']
        if not message.startswith('steps:'):
            return super().generate_stream(request, cancellation, emit)
        text, calls = '', []
        for kind, value in json.loads(message[len('steps:'):]):
            if kind == 'text':
                emit(value)
                text += value
            elif kind == 'call':
                emit({'tool_calls': [dict(value, index=len(calls))]})
                calls.append(value)
            else:
                for _ in range(500):
                    cancellation.check()
                    time.sleep(.01)
        response = self.response(request, text)
        if calls:
            response['tool_calls'] = calls
        return response
def create_engine(context):
    return StepsEngine(context)
'''


def events(data):
    """[(event, data)] of an Anthropic event stream."""
    found = []
    for block in data.decode().split("\n\n"):
        lines = block.split("\n")
        if len(lines) >= 2 and lines[0].startswith("event: ") and lines[1].startswith("data: "):
            found.append((lines[0][7:], json.loads(lines[1][6:])))
    return found


class DialectFixture(ServeFixture):
    stream = True
    request_options = True
    api_keys = ()
    model_name = None
    lenient = False

    def setUp(self):
        super().setUp()
        self.served = self.engine()
        self.server = Server(("127.0.0.1", 0), Api(self.served, model_name=self.model_name, api_keys=self.api_keys,
                                                         cors_origin="*", lenient=self.lenient))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=30)
        self.addCleanup(connection.close)
        payload = None if body is None else json.dumps(body)
        connection.request(method, path, payload, {"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        return response, response.read()

    def messages(self, content, **options):
        body = {"model": "claude-opus-5-5", "max_tokens": 50, "messages": [{"role": "user", "content": content}], **options}
        response, data = self.call("POST", "/v1/messages", body)
        return response, (json.loads(data) if not options.get("stream") else data)

    def blocks(self, data):
        """[(content block, its deltas joined)] of an Anthropic event stream, which opens one block at a time."""
        found, open_index = [], None
        for name, event in events(data):
            if name == "content_block_start":
                self.assertIsNone(open_index, "a block opened before the previous one closed")
                open_index = event["index"]
                found.append((event["content_block"], []))
            elif name in {"content_block_delta", "content_block_stop"}:
                self.assertEqual(event["index"], open_index)
                if name == "content_block_stop":
                    open_index = None
                else:
                    found[-1][1].append(next(v for k, v in event["delta"].items() if k != "type"))
        self.assertIsNone(open_index)
        return [(block, "".join(deltas)) for block, deltas in found]


class AnthropicTests(DialectFixture, unittest.TestCase):
    def test_a_reply_is_a_message_with_text_usage_and_the_requested_model(self):
        response, value = self.messages("reply:hello world", system="Be brief.", metadata={"user_id": "u"})
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value["type"], "message")
        self.assertEqual(value["role"], "assistant")
        self.assertEqual(value["model"], "claude-opus-5-5")
        self.assertEqual(value["content"], [{"type": "text", "text": "hello world"}])
        self.assertEqual(value["stop_reason"], "end_turn")
        self.assertIsNone(value["stop_sequence"])
        self.assertEqual(value["usage"], {"input_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                                          "output_tokens": 4})
        self.assertTrue(value["id"].startswith("msg_"))
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "metadata")

    def test_the_system_prompt_and_blocks_reach_the_engine_as_one_conversation(self):
        system = [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "Answer in JSON."}]
        response, value = self.messages([{"type": "text", "text": "echo-request"}], system=system)
        self.assertEqual(response.status, 200, value)
        echoed = json.loads(value["content"][0]["text"])
        self.assertEqual(echoed["messages"][0], {"role": "system", "content": "Be brief.\n\nAnswer in JSON."})
        self.assertEqual(echoed["messages"][-1], {"role": "user", "content": "echo-request"})

    def test_stop_sequences_and_the_output_cap_set_the_stop_reason(self):
        response, value = self.messages("reply:hello world", stop_sequences=["wor"])
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value["content"], [{"type": "text", "text": "hello "}])
        self.assertEqual(value["stop_reason"], "stop_sequence")
        self.assertEqual(value["stop_sequence"], "wor")
        response, value = self.messages("reply:hello world", stop_sequences=[str(i) for i in range(15)] + ["wor"])
        self.assertEqual((response.status, value["stop_sequence"]), (200, "wor"), value)
        response, value = self.messages("reply:hello world", stop_sequences=["wor"] * 17)
        self.assertEqual((response.status, value["error"]["type"]), (400, "invalid_request_error"))
        body = {"model": "x", "max_tokens": 2, "messages": [{"role": "user", "content": "words:one two three four"}]}
        response, data = self.call("POST", "/v1/messages", body)
        self.assertEqual(json.loads(data)["stop_reason"], "max_tokens")

    def test_tools_become_tool_use_blocks_and_results_go_back_as_tool_messages(self):
        response, value = self.messages("tool:" + json.dumps([CALL]), tools=TOOLS, tool_choice={"type": "any"})
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value["stop_reason"], "tool_use")
        self.assertEqual(value["content"], [{"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.py"}}])
        history = [{"role": "user", "content": "Read a.py"},
                   {"role": "assistant", "content": [{"type": "text", "text": "Reading."}, value["content"][0]]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "print(1)"},
                                                {"type": "text", "text": "echo-request"}]}]
        body = {"model": "x", "max_tokens": 50, "messages": history, "tools": TOOLS,
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
        response, data = self.call("POST", "/v1/messages", body)
        self.assertEqual(response.status, 200, data)
        echoed = json.loads(json.loads(data)["content"][0]["text"])
        self.assertEqual(echoed["messages"][1]["tool_calls"], [CALL])
        self.assertEqual(echoed["messages"][1]["content"], "Reading.")
        self.assertEqual(echoed["messages"][2], {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"})
        self.assertEqual(echoed["messages"][3], {"role": "user", "content": "echo-request"})
        self.assertEqual(echoed["tools"], OPENAI_TOOLS)
        self.assertEqual((echoed["tool_choice"], echoed["parallel_tool_calls"]), ("auto", False))

    def test_thinking_and_error_results_reach_the_engine_as_the_same_openai_conversation_does(self):
        history = [{"role": "user", "content": "Read a.py"},
                   {"role": "assistant", "content": [
                       {"type": "thinking", "thinking": "The user wants a.py.", "signature": "c2ln"},
                       {"type": "redacted_thinking", "data": "b3BhcXVl"}, {"type": "text", "text": "Reading."},
                       {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.py"}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "is_error": True,
                                                 "content": [{"type": "text", "text": "No such file"}]},
                                                {"type": "text", "text": "echo-request"}]}]
        response, data = self.call("POST", "/v1/messages", {"model": "x", "max_tokens": 50, "messages": history,
                                                            "tools": TOOLS})
        self.assertEqual(response.status, 200, data)
        echoed = json.loads(json.loads(data)["content"][0]["text"])
        self.assertEqual(echoed["messages"][1]["reasoning_content"], "The user wants a.py.")
        self.assertEqual(echoed["messages"][2], {"role": "tool", "tool_call_id": "call_1", "content": "Error: No such file"})
        openai = [{"role": "user", "content": "Read a.py"},
                  {"role": "assistant", "content": "Reading.", "reasoning_content": "The user wants a.py.",
                   "tool_calls": [CALL]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "Error: No such file"},
                  {"role": "user", "content": "echo-request"}]
        response, data = self.call("POST", "/v1/chat/completions", {"messages": openai, "tools": OPENAI_TOOLS,
                                                                    "max_tokens": 50})
        self.assertEqual(echoed, json.loads(json.loads(data)["choices"][0]["message"]["content"]))
        failed = {"type": "tool_result", "tool_use_id": "call_1", "is_error": True}
        result = lambda block: anthropic.messages(None, history[:2] + [{"role": "user", "content": [block]}])[2]["content"]
        self.assertEqual(result(failed), "Error: The tool call failed.")
        self.assertEqual(result(dict(failed, content=[IMAGE])), [{"type": "text", "text": "Error: The tool call failed."}, {
            "type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}}])

    def test_strict_tools_and_tool_result_images_need_the_package_capabilities(self):
        response, value = self.messages("hi", tools=STRICT_TOOLS)
        self.assertEqual((response.status, value["error"]["type"]), (400, "invalid_request_error"))
        self.assertIn("chat-tools-strict-v1", value["error"]["message"])
        history = [{"role": "user", "content": "Look"},
                   {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "read_file", "input": {}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": [IMAGE]}]}]
        response, data = self.call("POST", "/v1/messages", {"model": "x", "max_tokens": 50, "messages": history, "tools": TOOLS})
        self.assertEqual((response.status, json.loads(data)["error"]["type"]), (400, "invalid_request_error"))

    def test_invalid_requests_fail_in_anthropic_shape(self):
        response, value = self.messages("hello", max_tokens=None)
        self.assertEqual(response.status, 400)
        self.assertEqual(value["type"], "error")
        self.assertEqual(value["error"]["type"], "invalid_request_error")
        self.assertIn("max_tokens", value["error"]["message"])
        response, value = self.messages("hello", mcp_servers=[{"url": "x"}])
        self.assertEqual((response.status, value["error"]["type"]), (400, "invalid_request_error"))
        response, value = self.messages("hello", tools=[{"type": "web_search_20260209", "name": "web_search"}])
        self.assertIn("Anthropic-defined tools", value["error"]["message"])
        response, data = self.call("POST", "/v1/messages/count_tokens", {"model": "x", "messages": []})
        self.assertEqual((response.status, json.loads(data)["error"]["type"]), (404, "not_found_error"))
        response, data = self.call("POST", "/v1/messages/batches", {"requests": []})
        self.assertEqual((response.status, json.loads(data)["error"]["type"]), (404, "not_found_error"))
        # Anthropic's messages have no system role; the system prompt is the top-level field.
        body = {"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}]}
        response, data = self.call("POST", "/v1/messages", body)
        self.assertEqual((response.status, json.loads(data)["error"]["type"]), (400, "invalid_request_error"))

    def test_output_config_settings_are_listed_by_plain_name_and_other_names_refused(self):
        response, value = self.messages("reply:hi", output_config={"effort": "low"})
        self.assertEqual((response.status, response.getheader("X-Crit-Ignored-Parameters")),
                         (200, "output_config.effort"), value)
        response, value = self.messages("reply:hi", output_config={"effort\r\nX-Injected: yes": "low"})
        self.assertEqual((response.status, value["error"]["type"]), (400, "invalid_request_error"))
        self.assertIsNone(response.getheader("X-Injected"))

    def test_a_context_overflow_is_a_400_in_anthropic_words_and_keeps_the_engine_loaded(self):
        error = {"type": "error", "error": {"type": "invalid_request_error",
                                            "message": "prompt is too long: 9000 tokens > 8192 maximum"}}
        response, value = self.messages("overflow:9000")
        self.assertEqual((response.status, value), (400, dict(error, request_id=response.getheader("X-Request-Id"))))
        response, data = self.messages("overflow:9000", stream=True)
        self.assertEqual([name for name, _ in events(data)], ["message_start", "error"])
        self.assertEqual(events(data)[-1][1], dict(error, request_id=response.getheader("X-Request-Id")))
        response, value = self.messages("overflow:8150")
        self.assertEqual(value["error"]["message"], "input length and `max_tokens` exceed context limit: 8150 + 50 > "
                                                    "8192, decrease input length or `max_tokens` and try again")
        response, value = self.messages("reply:hi")
        self.assertEqual(response.status, 200, value)
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_the_event_stream_opens_closes_blocks_and_ends_with_the_stop_reason_and_usage(self):
        response, data = self.messages("words:one two three", stream=True)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        found = events(data)
        names = [name for name, _ in found]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(found[0][1]["message"]["model"], "claude-opus-5-5")
        self.assertEqual(found[1], ("content_block_start", {"type": "content_block_start", "index": 0,
                                                             "content_block": {"type": "text", "text": ""}}))
        deltas = [d["delta"]["text"] for name, d in found if name == "content_block_delta"]
        self.assertEqual("".join(deltas), "one two three ")
        self.assertEqual(names[-3:], ["content_block_stop", "message_delta", "message_stop"])
        delta = found[-2][1]
        self.assertEqual(delta["delta"], {"stop_reason": "end_turn", "stop_sequence": None})
        self.assertEqual(delta["usage"]["output_tokens"], 3)
        self.assertEqual(delta["usage"]["input_tokens"], 10)

    def test_streamed_tool_calls_become_tool_use_blocks_with_json_deltas(self):
        second = dict(CALL, id="call_2", function={"name": "read_file", "arguments": "{\"path\": \"b.py\"}"})
        response, data = self.messages("tool:" + json.dumps([CALL, second]), tools=TOOLS, stream=True)
        self.assertEqual(response.status, 200)
        self.assertEqual([(block, json.loads(partial)) for block, partial in self.blocks(data)],
                         [({"type": "tool_use", "id": "call_1", "name": "read_file", "input": {}}, {"path": "a.py"}),
                          ({"type": "tool_use", "id": "call_2", "name": "read_file", "input": {}}, {"path": "b.py"})])
        self.assertEqual(events(data)[-2][1]["delta"]["stop_reason"], "tool_use")


class AnthropicCapabilityTests(DialectFixture, unittest.TestCase):
    """A package with strict tools, streamed calls (written as test_tool_stream's adapter writes them, every call's
    start before any arguments) and images."""
    strict_tools = tool_stream = True
    extra_capabilities = ("chat-images-v1",)
    adapter_suffix = TOOL_ADAPTER

    def test_strict_tools_and_tool_result_images_reach_the_engine(self):
        response, value = self.messages("echo-request", tools=STRICT_TOOLS)
        self.assertEqual(response.status, 200, value)
        echoed = json.loads(value["content"][0]["text"])
        self.assertIs(echoed["tools"][0]["function"]["strict"], True)
        self.assertIn("read_file", echoed["tool_grammars"])
        history = [{"role": "user", "content": "Take a screenshot"},
                   {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "read_file", "input": {}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1",
                                                 "content": [{"type": "text", "text": "Taken"}, IMAGE]},
                                                {"type": "text", "text": "echo-request"}]}]
        response, data = self.call("POST", "/v1/messages", {"model": "x", "max_tokens": 50, "messages": history, "tools": TOOLS})
        self.assertEqual(response.status, 200, data)
        echoed = json.loads(json.loads(data)["content"][0]["text"])
        self.assertEqual(echoed["messages"][2]["content"], [{"type": "text", "text": "Taken"}, {
            "type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj", "detail": "auto"}}])

    def test_interleaved_streamed_calls_open_one_tool_use_block_at_a_time(self):
        calls = [dict(CALL, function={"name": "read_file", "arguments": '{"path":"a.py"}'}),
                 dict(CALL, id="call_2", function={"name": "read_file", "arguments": '{"path":"b.py"}'})]
        response, data = self.messages("stream-tools:" + json.dumps({"calls": calls}), tools=STRICT_TOOLS, stream=True)
        self.assertEqual(response.status, 200)
        self.assertEqual([(block["id"], partial) for block, partial in self.blocks(data)],
                         [("call_1", '{"path":"a.py"}'), ("call_2", '{"path":"b.py"}')])
        self.assertEqual(events(data)[-2][1]["delta"]["stop_reason"], "tool_use")


class AnthropicKeyTests(DialectFixture, unittest.TestCase):
    api_keys = ("secret",)

    def test_x_api_key_authenticates_like_a_bearer_token(self):
        response, value = self.messages("reply:hi")
        self.assertEqual((response.status, value["error"]["type"]), (401, "authentication_error"))
        body = {"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "reply:hi"}]}
        response, data = self.call("POST", "/v1/messages", body, {"x-api-key": "secret", "anthropic-version": "2023-06-01"})
        self.assertEqual(response.status, 200, data)
        response, data = self.call("GET", "/api/tags", headers={"x-api-key": "secret"})
        self.assertEqual(response.status, 200)
        response, data = self.call("GET", "/metrics")
        self.assertEqual(response.status, 401)


class ForcedToolsTests(DialectFixture, unittest.TestCase):
    """A package without chat-tool-choice-v1, such as one built on ResidentChat, offers tool_choice auto and none."""
    tool_choice = False

    def test_a_forced_call_is_a_400_in_each_dialect_and_the_engine_stays_loaded(self):
        for choice in ("required", {"type": "function", "function": {"name": "read_file"}}):
            body = {"messages": [{"role": "user", "content": "hi"}], "tools": OPENAI_TOOLS, "tool_choice": choice}
            response, data = self.call("POST", "/v1/chat/completions", body)
            error = json.loads(data)["error"]
            self.assertEqual((response.status, error["param"], error["code"]), (400, "tool_choice", "unsupported_parameter"))
            self.assertIn("chat-tool-choice-v1", error["message"])
        for choice in ({"type": "any"}, {"type": "tool", "name": "read_file"}):
            response, value = self.messages("hi", tools=TOOLS, tool_choice=choice)
            self.assertEqual((response.status, value["error"]["type"]), (400, "invalid_request_error"))
        for choice in ("required", {"type": "function", "name": "read_file"}):
            response, data = self.call("POST", "/v1/responses", {"input": "hi", "tools": RESPONSES_TOOLS, "tool_choice": choice})
            error = json.loads(data)["error"]
            self.assertEqual((response.status, error["param"], error["code"]), (400, "tool_choice", "unsupported_parameter"))
        response, value = self.messages("tool:" + json.dumps([CALL]), tools=TOOLS, tool_choice={"type": "auto"})
        self.assertEqual((response.status, value["stop_reason"]), (200, "tool_use"))
        self.assertEqual(self.served.status()["engine_loads"], 1)


class ThinkingFixture(DialectFixture):
    def sent(self, path, **options):
        """(response, the template_kwargs the engine received, or the error) for an echo request on a chat path."""
        turn = [{"role": "user", "content": "echo-request"}]
        body = {"model": "default", "stream": False, **({"input": turn} if path == "/v1/responses" else {"messages": turn}),
                **({"max_tokens": 50} if path == "/v1/messages" else {}), **options}
        response, data = self.call("POST", path, body)
        value = json.loads(data)
        if response.status != 200:
            return response, value
        text = (value["choices"][0]["message"]["content"] if path == "/v1/chat/completions" else
                value["content"][0]["text"] if path == "/v1/messages" else
                value["output"][-1]["content"][0]["text"] if path == "/v1/responses" else value["message"]["content"])
        return response, json.loads(text)["template_kwargs"]


class UndeclaredThinkingTests(ThinkingFixture, unittest.TestCase):
    """A package without chat-template-kwargs-v1: OpenAI's thinking controls are a 400, Anthropic's and Ollama's are
    ignored and listed whatever they hold."""

    def test_openai_refuses_the_controls_and_the_other_dialects_list_them(self):
        for options, param in (({"reasoning_effort": "high"}, "reasoning_effort"),
                               ({"chat_template_kwargs": {"enable_thinking": False}}, "chat_template_kwargs")):
            response, value = self.sent("/v1/chat/completions", **options)
            self.assertEqual((response.status, value["error"]["param"], value["error"]["code"]),
                             (400, param, "unsupported_parameter"))
            self.assertIn("chat-template-kwargs-v1", value["error"]["message"])
        response, value = self.sent("/v1/responses", reasoning={"effort": "high"})
        self.assertEqual((response.status, value["error"]["param"], value["error"]["code"]),
                         (400, "reasoning_effort", "unsupported_parameter"))
        for path, options, listed in (
                ("/v1/messages", {"thinking": {"type": "enabled", "budget_tokens": 2048},
                                  "output_config": {"effort": "high"}}, "output_config.effort,thinking"),
                ("/v1/messages", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
                 "output_config.effort,thinking"),
                ("/v1/messages", {"thinking": {"type": "sometimes"}, "output_config": {"effort": ["extreme"]}},
                 "output_config.effort,thinking"),
                ("/v1/messages", {"thinking": "on"}, "thinking"),
                ("/api/chat", {"think": False}, "think"),
                ("/api/chat", {"think": "max"}, "think"),
                ("/api/chat", {"think": {"level": 9}}, "think")):
            with self.subTest(path=path, options=options):
                response, sent = self.sent(path, **options)
                self.assertEqual((response.status, sent, response.getheader("X-Crit-Ignored-Parameters")),
                                 (200, None, listed))
        response, data = self.call("POST", "/api/show", {"model": "default"})
        self.assertNotIn("thinking", json.loads(data)["capabilities"])


class ThinkingSwitchTests(ThinkingFixture, unittest.TestCase):
    """A package whose chat template reads enable_thinking, as Qwen3's does."""
    extra_capabilities = ("chat-template-kwargs-v1", "chat-template-enable-thinking-v1")

    def test_each_dialect_turns_thinking_on_or_off_and_a_request_without_controls_sets_nothing(self):
        off, on = {"enable_thinking": False}, {"enable_thinking": True}
        for path, options, expected, listed in (
                ("/v1/chat/completions", {}, None, None),
                ("/v1/chat/completions", {"chat_template_kwargs": off}, off, None),
                ("/v1/chat/completions", {"reasoning_effort": "none"}, off, None),
                ("/v1/chat/completions", {"reasoning_effort": "high"}, on, None),
                ("/v1/responses", {"reasoning": {"effort": "none"}}, off, None),
                ("/v1/responses", {"reasoning": {"effort": "medium", "summary": "auto"}}, on, "reasoning.summary"),
                ("/v1/messages", {"thinking": {"type": "disabled"}}, off, None),
                ("/v1/messages", {"thinking": {"type": "between_tools"}}, off, None),
                ("/v1/messages", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}}, on, None),
                ("/v1/messages", {"thinking": {"type": "enabled", "budget_tokens": 2048}}, on, "thinking.budget_tokens"),
                ("/v1/messages", {"output_config": {"effort": "low"}}, on, None),
                ("/api/chat", {"think": False}, off, None),
                ("/api/chat", {"think": "high"}, on, None)):
            with self.subTest(path=path, options=options):
                response, sent = self.sent(path, **options)
                self.assertEqual((response.status, sent, response.getheader("X-Crit-Ignored-Parameters")),
                                 (200, expected, listed))
        for path, options in (("/v1/chat/completions", {"chat_template_kwargs": {"reasoning_effort": "low"}}),
                              ("/v1/chat/completions", {"chat_template_kwargs": {"enable_thinking": "no"}}),
                              ("/v1/responses", {"reasoning": {"effort": "max"}}),
                              ("/v1/messages", {"thinking": {"type": "sometimes"}}),
                              ("/v1/messages", {"thinking": {"type": ["enabled"]}}),
                              ("/v1/messages", {"output_config": {"effort": "extreme"}}),
                              ("/v1/messages", {"thinking": {"type": "enabled", "a\r\nX-Injected: yes": 1}}),
                              ("/api/chat", {"think": "max"})):
            with self.subTest(path=path, options=options):
                response, value = self.sent(path, **options)
                self.assertEqual(response.status, 400, value)
                self.assertIsNone(response.getheader("X-Injected"))
        response, data = self.call("POST", "/api/show", {"model": "default"})
        self.assertIn("thinking", json.loads(data)["capabilities"])
        self.assertEqual(self.served.status()["engine_loads"], 1)


class ReasoningEffortTests(ThinkingFixture, unittest.TestCase):
    """A package whose chat template always thinks and reads reasoning_effort, as gpt-oss's does."""
    extra_capabilities = ("chat-template-kwargs-v1", "chat-template-reasoning-effort-v1")

    def test_efforts_set_reasoning_effort_and_thinking_cannot_be_turned_off(self):
        for path, options, expected in (("/v1/chat/completions", {"reasoning_effort": "minimal"}, "low"),
                                        ("/v1/chat/completions", {"reasoning_effort": "xhigh"}, "high"),
                                        ("/v1/responses", {"reasoning": {"effort": "low"}}, "low"),
                                        ("/v1/messages", {"output_config": {"effort": "xhigh"}}, "high"),
                                        ("/v1/messages", {"output_config": {"effort": "max"}}, "high"),
                                        ("/api/chat", {"think": "medium"}, "medium")):
            with self.subTest(path=path, options=options):
                response, sent = self.sent(path, **options)
                self.assertEqual((response.status, sent), (200, {"reasoning_effort": expected}))
        response, sent = self.sent("/api/chat", think=True)
        self.assertEqual((response.status, sent), (200, None))   # it thinks anyway
        for path, options in (("/v1/chat/completions", {"reasoning_effort": "none"}),
                              ("/v1/messages", {"thinking": {"type": "disabled"}}),
                              ("/v1/messages", {"thinking": {"type": "between_tools"}}),
                              ("/api/chat", {"think": False})):
            with self.subTest(path=path, options=options):
                response, value = self.sent(path, **options)
                self.assertEqual(response.status, 400)
                self.assertIn("cannot turn thinking off", json.dumps(value))
        response, data = self.call("POST", "/api/show", {"model": "default"})
        self.assertIn("thinking", json.loads(data)["capabilities"])


class LenientThinkingTests(ThinkingFixture, unittest.TestCase):
    """crit serve --lenient over a package whose chat template always thinks and reads reasoning_effort: a control or
    variable it cannot apply is ignored and listed, once its value is checked."""
    extra_capabilities = ("chat-template-kwargs-v1", "chat-template-reasoning-effort-v1")
    lenient = True

    def test_controls_the_template_cannot_take_are_ignored_and_listed_and_values_are_still_checked(self):
        for path, options, expected, listed in (
                ("/v1/chat/completions", {"reasoning_effort": "none"}, None, "reasoning_effort"),
                ("/v1/chat/completions", {"reasoning_effort": "high"}, {"reasoning_effort": "high"}, None),
                ("/v1/chat/completions",
                 {"reasoning_effort": "none", "chat_template_kwargs": {"reasoning_effort": "low"}},
                 {"reasoning_effort": "low"}, "reasoning_effort"),
                ("/v1/chat/completions",
                 {"chat_template_kwargs": {"enable_thinking": False, "bos_token": "<s>", "reasoning_effort": "low"}},
                 {"reasoning_effort": "low"}, "chat_template_kwargs.bos_token,chat_template_kwargs.enable_thinking"),
                ("/v1/messages", {"thinking": {"type": "disabled", "budget_tokens": 0}}, None,
                 "thinking,thinking.budget_tokens"),
                ("/v1/messages", {"thinking": {"type": "disabled"}, "output_config": {"effort": "high"}}, None,
                 "output_config.effort,thinking"),
                ("/v1/messages", {"output_config": {"effort": "max"}}, {"reasoning_effort": "high"}, None),
                ("/api/chat", {"think": False}, None, "think"),
                ("/api/chat", {"think": "low"}, {"reasoning_effort": "low"}, None)):
            with self.subTest(path=path, options=options):
                response, sent = self.sent(path, **options)
                self.assertEqual((response.status, sent, response.getheader("X-Crit-Ignored-Parameters")),
                                 (200, expected, listed))
        for path, options in (("/v1/chat/completions", {"reasoning_effort": "max"}),
                              ("/v1/chat/completions", {"chat_template_kwargs": {"enable_thinking": "no"}}),
                              ("/v1/chat/completions", {"chat_template_kwargs": {"a b": True}}),
                              ("/v1/chat/completions", {"reasoning_effort": "high",
                                                        "chat_template_kwargs": {"reasoning_effort": "low"}}),
                              ("/v1/messages", {"thinking": {"type": "sometimes"}}),
                              ("/api/chat", {"think": "max"})):
            with self.subTest(path=path, options=options):
                self.assertEqual(self.sent(path, **options)[0].status, 400)


class ToolStopTests(DialectFixture, unittest.TestCase):
    """Stop strings end a tool reply in its text with the calls finished before them; a response format applies when
    tool_choice is none. The package streams its calls (chat-tool-stream-v1) and applies formats."""
    tool_stream = constraints = True
    adapter_suffix = STEPS
    STOPPED = [["text", "Reading. "], ["call", CALL], ["text", "Observation: done"], ["hold", None]]

    def chat(self, steps, **options):
        body = {"messages": [{"role": "user", "content": "steps:" + json.dumps(steps)}], "tools": OPENAI_TOOLS, **options}
        response, data = self.call("POST", "/v1/chat/completions", body)
        if not options.get("stream"):
            return response, json.loads(data)
        chunks = [json.loads(e[6:]) for e in data.decode().split("\n\n") if e.startswith("data: {")]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        return response, ("".join(d.get("content", "") for d in deltas), [t for d in deltas for t in d.get("tool_calls", [])],
                          chunks[-1]["choices"][0]["finish_reason"])

    def test_a_stop_ends_the_text_and_keeps_the_calls_finished_before_it(self):
        response, value = self.chat(self.STOPPED, stop=["Observation:"])
        self.assertEqual(response.status, 200, value)
        self.assertEqual((value["choices"][0]["finish_reason"], value["choices"][0]["message"]),
                         ("tool_calls", {"role": "assistant", "content": "Reading. ", "tool_calls": [CALL]}))
        response, streamed = self.chat(self.STOPPED, stop=["Observation:"], stream=True)
        self.assertEqual(streamed, ("Reading. ", [dict(CALL, index=0)], "tool_calls"))
        # include_stop_str_in_output ends the text with the stop and keeps the calls before it.
        kept = {"role": "assistant", "content": "Reading. Observation:", "tool_calls": [CALL]}
        response, value = self.chat(self.STOPPED, stop=["Observation:"], include_stop_str_in_output=True)
        self.assertEqual((value["choices"][0]["finish_reason"], value["choices"][0]["message"]), ("tool_calls", kept))
        response, streamed = self.chat(self.STOPPED, stop=["Observation:"], include_stop_str_in_output=True, stream=True)
        self.assertEqual(streamed, ("Reading. Observation:", [dict(CALL, index=0)], "tool_calls"))
        # A stop before the call ends the reply without it, whether the call was written or not.
        steps = [["text", "Reading. Observation: done"], ["call", CALL]]
        response, value = self.chat(steps, stop="Observation:")
        self.assertEqual((value["choices"][0]["finish_reason"], value["choices"][0]["message"]),
                         ("stop", {"role": "assistant", "content": "Reading. "}))
        response, streamed = self.chat(steps, stop="Observation:", stream=True)
        self.assertEqual(streamed, ("Reading. ", [], "stop"))
        # Stops are never matched in a call's arguments.
        response, value = self.chat([["call", CALL], ["text", "Done."]], stop=["a.py"])
        self.assertEqual(value["choices"][0]["message"], {"role": "assistant", "content": "Done.", "tool_calls": [CALL]})
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_text_inside_an_unfinished_call_fails_the_turn_rather_than_cutting_the_call(self):
        cut = dict(CALL, id="call_2", function={"name": "read_file", "arguments": "{\"path\": "})
        steps = [["call", CALL], ["call", cut], ["text", "Observation: done"]]
        response, value = self.chat(steps, stop=["Observation:"])
        self.assertEqual((response.status, value["error"]["type"]), (500, "server_error"), value)
        body = {"messages": [{"role": "user", "content": "steps:" + json.dumps(steps)}], "tools": OPENAI_TOOLS,
                "stop": ["Observation:"], "stream": True}
        response, data = self.call("POST", "/v1/chat/completions", body)
        chunks = [json.loads(e[6:]) for e in data.decode().split("\n\n") if e.startswith("data: {")]
        # The cut call's start has gone out, but the turn ends in an error, never in a finish to act on.
        deltas = [t for c in chunks if c.get("choices") for t in c["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual([t["id"] for t in deltas], ["call_1", "call_2"])
        self.assertIn("unfinished tool call", chunks[-1]["error"]["message"])
        self.assertFalse(any(c["choices"][0]["finish_reason"] for c in chunks if c.get("choices")))

    def test_anthropic_and_ollama_stops_keep_the_calls_before_them(self):
        content = "steps:" + json.dumps(self.STOPPED)
        response, value = self.messages(content, tools=TOOLS, stop_sequences=["Observation:"])
        self.assertEqual(response.status, 200, value)
        self.assertEqual((value["content"], value["stop_reason"], value["stop_sequence"]),
                         ([{"type": "text", "text": "Reading. "},
                           {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.py"}}],
                          "tool_use", None))
        response, data = self.messages(content, tools=TOOLS, stop_sequences=["Observation:"], stream=True)
        found = events(data)
        self.assertEqual([d["content_block"]["type"] for name, d in found if name == "content_block_start"],
                         ["text", "tool_use"])
        self.assertEqual(found[-2][1]["delta"], {"stop_reason": "tool_use", "stop_sequence": None})
        response, data = self.call("POST", "/api/chat", {"model": "default", "stream": False, "tools": OPENAI_TOOLS,
                                                         "options": {"stop": "Observation:"},
                                                         "messages": [{"role": "user", "content": content}]})
        self.assertEqual(response.status, 200, data)
        self.assertEqual(json.loads(data)["message"], {"role": "assistant", "content": "Reading. ", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]})

    def test_a_response_format_applies_with_tool_choice_none_in_every_dialect(self):
        response, value = self.chat([["text", '{"a": 1}']], tool_choice="none", response_format={"type": "json_object"})
        self.assertEqual((response.status, value["choices"][0]["message"]["content"]), (200, '{"a": 1}'), value)
        response, value = self.chat([["text", '{"a": 1}']], response_format={"type": "json_object"})
        self.assertEqual((response.status, value["error"]["param"]), (400, "response_format"), value)
        self.assertIn("tool_choice is none", value["error"]["message"])
        schema = {"type": "object", "properties": {"a": {"type": "integer", "minimum": 0, "maximum": 9}},
                  "required": ["a"], "additionalProperties": False}
        # vLLM's structured_outputs alike.
        response, value = self.chat([["text", '{"a":1}']], tool_choice="none", structured_outputs={"json": schema})
        self.assertEqual((response.status, value["choices"][0]["message"]["content"]), (200, '{"a":1}'), value)
        response, value = self.chat([["text", '{"a":1}']], structured_outputs={"json": schema})
        self.assertEqual((response.status, value["error"]["param"]), (400, "structured_outputs.json"), value)
        self.assertIn("tool_choice is none", value["error"]["message"])
        output = {"format": {"type": "json_schema", "schema": schema}}
        response, value = self.messages('reply:{"a":1}', tools=TOOLS, tool_choice={"type": "none"}, output_config=output)
        self.assertEqual((response.status, value["content"]), (200, [{"type": "text", "text": '{"a":1}'}]), value)
        response, value = self.messages('reply:{"a":1}', tools=TOOLS, output_config=output)
        self.assertEqual((response.status, value["error"]["type"]), (400, "invalid_request_error"))
        # Ollama has no tool_choice: its tools are always offered.
        response, data = self.call("POST", "/api/chat", {"model": "default", "stream": False, "tools": OPENAI_TOOLS,
                                                         "format": "json", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(response.status, 400)
        self.assertIn("response_format", json.loads(data)["error"])
        self.assertEqual(self.served.status()["engine_loads"], 1)


class BufferedToolStopTests(DialectFixture, unittest.TestCase):
    """A package that returns its calls only with the reply (no chat-tool-stream-v1)."""

    def test_a_stop_with_offered_tools_is_a_400_and_applies_with_tool_choice_none(self):
        body = {"messages": [{"role": "user", "content": "reply:hello world"}], "tools": OPENAI_TOOLS, "stop": ["wor"]}
        response, data = self.call("POST", "/v1/chat/completions", body)
        error = json.loads(data)["error"]
        self.assertEqual((response.status, error["param"], error["code"]), (400, "stop", "unsupported_parameter"))
        self.assertIn("chat-tool-stream-v1", error["message"])
        response, value = self.messages("reply:hello world", tools=TOOLS, stop_sequences=["wor"])
        self.assertEqual((response.status, value["error"]["type"]), (400, "invalid_request_error"))
        response, data = self.call("POST", "/v1/chat/completions", dict(body, tool_choice="none"))
        choice = json.loads(data)["choices"][0]
        self.assertEqual((response.status, choice["message"]["content"], choice["finish_reason"]), (200, "hello ", "stop"))
        response, value = self.messages("reply:hello world", tools=TOOLS, tool_choice={"type": "none"},
                                        stop_sequences=["wor"])
        self.assertEqual((value["content"], value["stop_reason"]), ([{"type": "text", "text": "hello "}], "stop_sequence"))


class ContinueTests(DialectFixture, unittest.TestCase):
    """A package that declares chat-continue-v1, as ResidentChat's do, continues a final assistant message."""
    constraints = True
    extra_capabilities = ("chat-continue-v1",)
    prefill = [{"role": "user", "content": "Write"}, {"role": "assistant", "content": "echo-request"}]
    flags = {"continue_final_message": True, "add_generation_prompt": False}

    def test_each_dialect_continues_a_final_assistant_message(self):
        response, data = self.call("POST", "/v1/chat/completions", {"messages": self.prefill, **self.flags})
        self.assertEqual(response.status, 200, data)
        echoed = json.loads(json.loads(data)["choices"][0]["message"]["content"])
        self.assertEqual((echoed["messages"], echoed["continue_final_message"]), (self.prefill, True))
        # Anthropic's prefill and an Ollama chat that ends with the assistant are continued without a flag.
        response, data = self.call("POST", "/v1/messages", {"model": "x", "max_tokens": 50, "messages": self.prefill})
        self.assertEqual(response.status, 200, data)
        echoed = json.loads(json.loads(data)["content"][0]["text"])
        self.assertEqual((echoed["messages"], echoed["continue_final_message"]), (self.prefill, True))
        response, data = self.call("POST", "/api/chat", {"model": "default", "messages": self.prefill, "stream": False})
        self.assertEqual(response.status, 200, data)
        echoed = json.loads(json.loads(data)["message"]["content"])
        self.assertEqual((echoed["messages"], echoed["continue_final_message"]), (self.prefill, True))
        prefill = [self.prefill[0], {"role": "assistant", "content": "words:upon a time"}]
        response, data = self.call("POST", "/v1/messages", {"model": "x", "max_tokens": 50, "messages": prefill,
                                                            "stream": True})
        found = events(data)
        self.assertEqual("".join(d["delta"]["text"] for name, d in found if name == "content_block_delta"), "upon a time ")
        self.assertEqual(found[-2][1]["delta"]["stop_reason"], "end_turn")

    def test_an_empty_continuation_completes_the_message_and_keeps_the_engine_loaded(self):
        body = {"messages": [self.prefill[0], {"role": "assistant", "content": "reply:"}], **self.flags}
        response, data = self.call("POST", "/v1/chat/completions", body)
        self.assertEqual(response.status, 200, data)
        choice = json.loads(data)["choices"][0]
        self.assertEqual((choice["message"]["content"], choice["finish_reason"]), ("", "stop"))
        response, data = self.call("POST", "/v1/chat/completions", dict(body, stream=True))
        chunks = [json.loads(e[6:]) for e in data.decode().split("\n\n") if e.startswith("data: {")]
        self.assertEqual([c["choices"][0]["finish_reason"] for c in chunks], [None, "stop"])
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_a_final_assistant_message_without_the_flags_or_with_tools_or_a_format_is_a_400(self):
        cases = (({"messages": self.prefill}, "messages"),
                 ({"messages": self.prefill, "continue_final_message": True}, "add_generation_prompt"),
                 ({"messages": self.prefill[:1], **self.flags}, "messages"),
                 ({"messages": self.prefill, "tools": OPENAI_TOOLS, **self.flags}, "tools"),
                 ({"messages": self.prefill, "response_format": {"type": "json_object"}, **self.flags}, "response_format"),
                 ({"messages": self.prefill, "guided_json": {"type": "null"}, **self.flags}, "guided_json"))
        for body, param in cases:
            with self.subTest(body=body):
                response, data = self.call("POST", "/v1/chat/completions", body)
                self.assertEqual((response.status, json.loads(data)["error"]["param"]), (400, param), data)
        response, data = self.call("POST", "/v1/messages", {"model": "x", "max_tokens": 50, "messages": self.prefill,
                                                            "tools": TOOLS})
        self.assertEqual((response.status, json.loads(data)["error"]["type"]), (400, "invalid_request_error"))
        self.assertEqual(self.served.status()["engine_loads"], 1)


class WithoutContinueTests(DialectFixture, unittest.TestCase):
    """A package that does not declare chat-continue-v1 refuses a final assistant message before the engine sees it."""

    def test_a_continuation_is_a_400_in_each_dialect(self):
        response, data = self.call("POST", "/v1/chat/completions", {"messages": ContinueTests.prefill,
                                                                   **ContinueTests.flags})
        error = json.loads(data)["error"]
        self.assertEqual((response.status, error["param"], error["code"]),
                         (400, "continue_final_message", "unsupported_parameter"))
        self.assertIn("chat-continue-v1", error["message"])
        response, data = self.call("POST", "/v1/messages", {"model": "x", "max_tokens": 50,
                                                            "messages": ContinueTests.prefill})
        self.assertEqual((response.status, json.loads(data)["error"]["type"]), (400, "invalid_request_error"))
        response, data = self.call("POST", "/api/chat", {"model": "default", "messages": ContinueTests.prefill,
                                                         "stream": False})
        self.assertEqual(response.status, 400)
        self.assertIn("chat-continue-v1", json.loads(data)["error"])
        with self.assertRaises(EngineError) as raised:
            self.served.generate(ContinueTests.prefill, 50, threading.Event(), continue_final_message=True)
        self.assertEqual(raised.exception.status, 400)


class ResponsesTests(DialectFixture, unittest.TestCase):
    def respond(self, input, **options):
        response, data = self.call("POST", "/v1/responses", {"model": "gpt-5.5", "input": input, **options})
        return response, (json.loads(data) if not options.get("stream") else data)

    def test_a_reply_is_a_response_with_a_message_item_and_usage(self):
        response, value = self.respond("reply:hello world", instructions="Be brief.", store=False, metadata={"k": "v"})
        self.assertEqual(response.status, 200, value)
        self.assertTrue(value["id"].startswith("resp_"))
        self.assertEqual((value["object"], value["status"], value["error"], value["incomplete_details"]),
                         ("response", "completed", None, None))
        self.assertEqual(value["model"], self.run_id + "/" + self.a)
        [message] = value["output"]
        self.assertTrue(message.pop("id").startswith("msg_"))
        self.assertEqual(message, {"type": "message", "role": "assistant", "status": "completed", "content": [
            {"type": "output_text", "text": "hello world", "annotations": [], "logprobs": []}]})
        self.assertEqual(value["usage"], {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                                          "output_tokens": 4, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 14})
        self.assertEqual((value["instructions"], value["metadata"], value["store"], value["text"]),
                         ("Be brief.", {"k": "v"}, False, {"format": {"type": "text"}}))
        self.assertEqual(value["crit_telemetry"]["prompt_tokens"], 10)
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "metadata,store")

    def test_input_items_become_one_conversation_and_function_tools_map_across(self):
        items = [{"role": "developer", "content": "Use the tools."},
                 {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Read a.py"}]},
                 {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "The file first."}]},
                 {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Reading.", "annotations": []}]},
                 {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": CALL["function"]["arguments"]},
                 {"type": "function_call_output", "call_id": "call_1", "output": "print(1)"},
                 {"role": "user", "content": "echo-request"}]
        response, value = self.respond(items, instructions="Be brief.", tools=RESPONSES_TOOLS, tool_choice="auto",
                                       parallel_tool_calls=False, include=["reasoning.encrypted_content"],
                                       reasoning={"summary": "auto"})
        self.assertEqual(response.status, 200, value)
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "include,reasoning.summary")
        echoed = json.loads(value["output"][0]["content"][0]["text"])
        self.assertEqual([m["role"] for m in echoed["messages"]], ["system", "system", "user", "assistant", "tool", "user"])
        self.assertEqual([m["content"] for m in echoed["messages"][:2]], ["Be brief.", "Use the tools."])
        self.assertEqual(echoed["messages"][3], {"role": "assistant", "content": "Reading.", "tool_calls": [CALL],
                                                 "reasoning_content": "The file first."})
        self.assertEqual(echoed["messages"][4], {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"})
        self.assertEqual(echoed["tools"], OPENAI_TOOLS)
        self.assertEqual((echoed["tool_choice"], echoed["parallel_tool_calls"]), ("auto", False))
        self.assertEqual((value["tools"], value["parallel_tool_calls"]), (RESPONSES_TOOLS, False))

    def test_tool_calls_are_function_call_items(self):
        response, value = self.respond("tool:" + json.dumps([CALL]), tools=RESPONSES_TOOLS)
        self.assertEqual((response.status, value["status"]), (200, "completed"), value)
        [call] = value["output"]
        self.assertTrue(call.pop("id").startswith("fc_"))
        self.assertEqual(call, {"type": "function_call", "status": "completed", "call_id": "call_1", "name": "read_file",
                                "arguments": CALL["function"]["arguments"]})

    def test_the_output_cap_ends_a_response_incomplete_and_text_format_maps_to_response_format(self):
        response, value = self.respond("words:one two three four", max_output_tokens=2)
        self.assertEqual((value["status"], value["incomplete_details"]), ("incomplete", {"reason": "max_output_tokens"}))
        self.assertEqual((value["output"][-1]["status"], value["max_output_tokens"]), ("incomplete", 2))
        response, data = self.respond("words:one two three four", max_output_tokens=2, stream=True)
        found = events(data)
        self.assertEqual(found[-1][0], "response.incomplete")
        self.assertEqual(found[-1][1]["response"]["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(found[-1][1]["response"]["output"][-1]["status"], "incomplete")
        response, value = self.respond("reply:hi", text={"format": {"type": "text"}})
        self.assertEqual(response.status, 200, value)
        # This package does not declare chat-constraints-v1, so a structured format is the chat path's 400.
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}
        response, value = self.respond("hi", text={"format": {"type": "json_schema", "name": "r", "schema": schema}})
        self.assertEqual((response.status, value["error"]["param"], value["error"]["code"]),
                         (400, "response_format", "unsupported_parameter"))

    def test_stored_state_and_built_in_tools_are_refused_in_openai_shape(self):
        for options in ({"previous_response_id": "resp_1"}, {"conversation": "conv_1"}, {"background": True},
                        {"tools": [{"type": "web_search"}]}, {"text": {"verbosity": "low"}}, {"mystery": 1},
                        {"reasoning": {"effort": "low"}}):
            with self.subTest(options=options):
                response, value = self.respond("reply:hi", **options)
                self.assertEqual((response.status, value["error"]["code"]), (400, "unsupported_parameter"), value)
        response, value = self.respond([{"type": "item_reference", "id": "msg_1"}])
        self.assertEqual((response.status, value["error"]["param"]), (400, "input[0]"))
        # The SDKs post .../cancel with no body.
        for method, path, body in (("GET", "/v1/responses/resp_1", None), ("DELETE", "/v1/responses/resp_1", None),
                                   ("POST", "/v1/responses/resp_1/cancel", None), ("POST", "/v1/responses/resp_1/cancel", {}),
                                   ("GET", "/v1/responses/resp_1/input_items", None)):
            response, data = self.call(method, path, body)
            self.assertEqual((response.status, json.loads(data)["error"]["code"]), (404, "not_supported"), (method, path))
        response, data = self.call("DELETE", "/v1/responses")
        self.assertEqual((response.status, response.getheader("Allow")), (405, "POST, OPTIONS"))
        response, value = self.respond("reply:hi", background=False, previous_response_id=None, tool_choice="auto")
        self.assertEqual(response.status, 200, value)

    def test_the_event_stream_numbers_its_events_and_ends_with_the_completed_response(self):
        response, data = self.respond("words:one two three", stream=True)
        self.assertEqual((response.status, response.getheader("Content-Type")), (200, "text/event-stream"))
        found = events(data)
        self.assertEqual([d["sequence_number"] for _, d in found], list(range(len(found))))
        self.assertTrue(all(d["type"] == name for name, d in found))
        self.assertEqual([name for name, _ in found], [
            "response.created", "response.in_progress", "response.output_item.added", "response.content_part.added",
            *["response.output_text.delta"] * 3, "response.output_text.done", "response.content_part.done",
            "response.output_item.done", "response.completed"])
        self.assertEqual(found[0][1]["response"]["status"], "in_progress")
        self.assertEqual(len({d["response"]["id"] for _, d in found if "response" in d}), 1)
        self.assertEqual("".join(d["delta"] for name, d in found if name == "response.output_text.delta"), "one two three ")
        self.assertEqual(found[-4][1]["text"], "one two three ")
        completed = found[-1][1]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["output"], [found[-2][1]["item"]])
        self.assertEqual(completed["output"][0]["content"][0]["text"], "one two three ")
        self.assertEqual(completed["usage"]["output_tokens"], 3)

    def test_streamed_tool_calls_become_function_call_items_with_argument_deltas(self):
        response, data = self.respond("tool:" + json.dumps([CALL]), tools=RESPONSES_TOOLS, stream=True)
        self.assertEqual(response.status, 200)
        found = events(data)
        self.assertEqual([name for name, _ in found], [
            "response.created", "response.in_progress", "response.output_item.added",
            "response.function_call_arguments.delta", "response.function_call_arguments.done", "response.output_item.done",
            "response.completed"])
        added, done = found[2][1]["item"], found[5][1]["item"]
        self.assertEqual((added["type"], added["call_id"], added["name"], added["arguments"], added["status"]),
                         ("function_call", "call_1", "read_file", "", "in_progress"))
        self.assertEqual(json.loads(found[3][1]["delta"]), {"path": "a.py"})
        self.assertEqual(found[4][1]["arguments"], CALL["function"]["arguments"])
        self.assertEqual((found[3][1]["item_id"], found[3][1]["output_index"]), (added["id"], 0))
        self.assertEqual(done, dict(added, status="completed", arguments=CALL["function"]["arguments"]))
        self.assertEqual(found[-1][1]["response"]["output"], [done])

    def test_a_context_overflow_is_a_400_or_a_failed_response_and_keeps_the_engine_loaded(self):
        response, value = self.respond("overflow:9000")
        self.assertEqual((response.status, value["error"]["code"]), (400, "context_length_exceeded"))
        self.assertIn("maximum context length is 8192 tokens", value["error"]["message"])
        response, data = self.respond("overflow:9000", stream=True)
        found = events(data)
        self.assertEqual([name for name, _ in found], ["response.created", "response.in_progress", "response.failed"])
        failed = found[-1][1]["response"]
        self.assertEqual((failed["status"], failed["error"]["code"], failed["usage"]), ("failed", "context_length_exceeded", None))
        self.assertEqual(found[-1][1]["request_id"], response.getheader("X-Request-Id"))
        response, value = self.respond("reply:hi")
        self.assertEqual(response.status, 200, value)
        self.assertEqual(self.served.status()["engine_loads"], 1)


class ResponsesReasoningTests(DialectFixture, unittest.TestCase):
    """An engine's reasoning comes back as a reasoning item, its summary text, before the message."""
    adapter_suffix = REASONING_ADAPTER

    def test_reasoning_is_a_reasoning_item_before_the_message_in_replies_and_streams(self):
        body = {"model": "x", "input": "think:why|Because."}
        response, data = self.call("POST", "/v1/responses", body)
        value = json.loads(data)
        self.assertEqual(response.status, 200, value)
        self.assertEqual([(item["type"], item["status"]) for item in value["output"]],
                         [("reasoning", "completed"), ("message", "completed")])
        self.assertEqual(value["output"][0]["summary"], [{"type": "summary_text", "text": "why"}])
        self.assertTrue(value["output"][0]["id"].startswith("rs_"))
        self.assertEqual(value["output"][1]["content"][0]["text"], "Because.")
        response, data = self.call("POST", "/v1/responses", dict(body, stream=True))
        found = events(data)
        self.assertEqual([name for name, _ in found][2:9], [
            "response.output_item.added", "response.reasoning_summary_part.added", "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done", "response.reasoning_summary_part.done", "response.output_item.done",
            "response.output_item.added"])
        self.assertEqual((found[4][1]["delta"], found[5][1]["text"]), ("why", "why"))
        completed = found[-1][1]["response"]
        self.assertEqual([(item["type"], item["status"]) for item in completed["output"]],
                         [("reasoning", "completed"), ("message", "completed")])
        self.assertEqual(completed["output"][0], found[7][1]["item"])
        self.assertEqual(completed["output"][1]["content"][0]["text"], "Because.")


class OllamaTests(DialectFixture, unittest.TestCase):
    def chat(self, content, **options):
        body = {"model": "default", "messages": [{"role": "user", "content": content}], "stream": False, **options}
        response, data = self.call("POST", "/api/chat", body)
        return response, json.loads(data) if body.get("stream") is False else data

    def test_the_server_answers_what_ollama_clients_probe(self):
        response, data = self.call("GET", "/")
        self.assertEqual((response.status, data), (200, b"Ollama is running"))
        response, data = self.call("GET", "/api/version")
        self.assertEqual(json.loads(data), {"version": ollama.OLLAMA_COMPAT_VERSION, "crit_version": __version__})
        response, data = self.call("GET", "/api/tags")
        models = json.loads(data)["models"]
        self.assertEqual([m["name"] for m in models], [self.run_id + "/" + self.a, self.run_id + "/" + self.b])
        self.assertEqual(models[0]["details"]["format"], "crit")
        response, data = self.call("POST", "/api/show", {"model": self.run_id + "/" + self.b})
        shown = json.loads(data)
        self.assertEqual(shown["model_info"]["crit.context_length"], 8192)
        self.assertIn("tools", shown["capabilities"])
        response, data = self.call("POST", "/api/pull", {"model": "x"})
        self.assertEqual(response.status, 404)
        self.assertIn("error", json.loads(data))
        response, data = self.call("POST", "/api/embed", {"model": "x", "input": "hello"})
        self.assertEqual((response.status, json.loads(data)), (404, {"error": "/api/embed is not available on a crit "
                         "engine, which serves chat; run a separate embedding server for embeddings and reranking",
                         "request_id": response.getheader("X-Request-Id")}))

    def test_ps_lists_the_loaded_model_and_tags_carry_package_digests(self):
        response, data = self.call("GET", "/api/tags")
        self.assertEqual([m["digest"] for m in json.loads(data)["models"]],
                         [self.served.spec(c)["package_sha256"] for c in (self.a, self.b)])
        response, data = self.call("GET", "/api/ps")
        loaded = json.loads(data)["models"]
        self.assertEqual([(m["name"], m["digest"], m["context_length"], m["expires_at"]) for m in loaded],
                         [(self.model_name or self.run_id + "/" + self.a, self.served.spec(self.a)["package_sha256"],
                           8192, ollama.LOADED_UNTIL)])
        self.assertEqual(loaded[0]["size_vram"], self.served.status()["device_bytes"])
        response, data = self.call("POST", "/api/chat", {"model": self.run_id + "/" + self.b, "stream": False,
                                                         "messages": [{"role": "user", "content": "reply:hi"}]})
        self.assertEqual(response.status, 200, data)
        response, data = self.call("GET", "/api/ps")
        self.assertEqual([m["name"] for m in json.loads(data)["models"]], [self.run_id + "/" + self.b])

    def test_a_chat_reply_carries_ollama_counts_and_durations(self):
        response, value = self.chat("reply:hello")
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value["message"], {"role": "assistant", "content": "hello"})
        self.assertTrue(value["done"])
        self.assertEqual(value["done_reason"], "stop")
        self.assertEqual((value["prompt_eval_count"], value["eval_count"]), (10, 4))
        self.assertEqual(value["prompt_eval_duration"], 5_000_000)
        self.assertGreaterEqual(value["total_duration"], value["prompt_eval_duration"] + value["eval_duration"])
        self.assertEqual(value["model"], "default")

    def test_the_cached_prompt_count_is_what_the_engine_did_not_prefill(self):
        response, value = self.chat("reply:hello")
        self.assertEqual(value["prompt_eval_cached_count"], 0)  # the engine computed all 10 prompt tokens
        measured = {"prompt_tokens": 10, "output_tokens": 4, "prefill_tokens": 7, "prefill_ms": 5, "decode_ms": 10}
        self.assertEqual(ollama.final("m", "stop", measured, 0.1)["prompt_eval_cached_count"], 3)
        unmeasured = dict(measured, prefill_tokens=None, prefill_ms=None)
        self.assertNotIn("prompt_eval_cached_count", ollama.final("m", "stop", unmeasured, 0.1))

    def test_logprobs_need_a_package_that_reports_them(self):
        response, value = self.chat("reply:hi", logprobs=True)
        self.assertEqual(response.status, 400, value)
        self.assertEqual(value, {"error": "This engine package does not declare chat-logprobs-v1",
                                 "request_id": response.getheader("X-Request-Id")})

    def test_options_format_and_tools_map_onto_the_chat_request(self):
        response, value = self.chat("echo-request", options={"temperature": 0.5, "top_k": 20, "num_predict": 7, "num_ctx": 4096,
                                                             "repeat_penalty": 1.1, "stop": ["END"]}, keep_alive="5m")
        self.assertEqual(response.status, 200, value)
        echoed = json.loads(value["message"]["content"])
        self.assertEqual(echoed["sampling"], {"repetition_penalty": 1.1, "temperature": 0.5, "top_k": 20})
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "keep_alive,options.num_ctx")
        response, value = self.chat("tool:" + json.dumps([CALL]), tools=OPENAI_TOOLS)
        self.assertEqual(value["message"]["tool_calls"], [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}])
        history = [{"role": "user", "content": "Read a.py"},
                   {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]},
                   {"role": "tool", "content": "print(1)"},
                   {"role": "user", "content": "echo-request"}]
        response, data = self.call("POST", "/api/chat", {"model": "default", "messages": history, "tools": OPENAI_TOOLS, "stream": False})
        self.assertEqual(response.status, 200, data)
        echoed = json.loads(json.loads(data)["message"]["content"])
        self.assertEqual(echoed["messages"][1]["tool_calls"][0]["function"], CALL["function"])
        self.assertEqual(echoed["messages"][2]["tool_call_id"], echoed["messages"][1]["tool_calls"][0]["id"])
        response, data = self.call("POST", "/api/chat", {"model": "default", "messages": [{"role": "user", "content": "hi"}],
                                                        "options": {"mystery": 1}, "stream": False})
        self.assertEqual(response.status, 400)
        self.assertIn("mystery", json.loads(data)["error"])

    def test_a_later_system_message_reaches_the_engine_as_marked_user_text(self):
        history = [{"role": "user", "content": "hi"}, {"role": "system", "content": "Answer in French."},
                   {"role": "assistant", "content": "Bonjour"}, {"role": "user", "content": "echo-request"}]
        response, data = self.call("POST", "/api/chat", {"model": "default", "messages": history, "stream": False})
        self.assertEqual(response.status, 200, data)
        echoed = json.loads(json.loads(data)["message"]["content"])
        self.assertEqual(echoed["messages"], [{"role": "user", "content": "hi\n\nSystem: Answer in French."},
                                              history[2], history[3]])

    def test_streams_are_newline_delimited_objects_ending_with_done(self):
        response, data = self.call("POST", "/api/chat", {"model": "default", "messages": [{"role": "user", "content": "words:a b c"}]})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "application/x-ndjson")
        rows = [json.loads(line) for line in data.decode().splitlines() if line]
        self.assertEqual("".join(r["message"]["content"] for r in rows[:-1]), "a b c ")
        self.assertTrue(all(r["done"] is False for r in rows[:-1]))
        self.assertEqual((rows[-1]["done"], rows[-1]["done_reason"], rows[-1]["eval_count"]), (True, "stop", 3))

    def test_a_context_overflow_is_a_400_error_and_keeps_the_engine_loaded(self):
        error = {"error": "prompt of 9000 tokens plus 100 output tokens exceeds the context of 8192 tokens"}
        for stream in (False, True):
            with self.subTest(stream=stream):
                response, data = self.call("POST", "/api/chat", {"model": "default", "stream": stream, "messages": [
                    {"role": "user", "content": "overflow:9000"}], "options": {"num_predict": 100}})
                self.assertEqual((response.status, json.loads(data)),
                                 (200 if stream else 400, dict(error, request_id=response.getheader("X-Request-Id"))))
        response, value = self.chat("reply:hi")
        self.assertEqual(response.status, 200, value)
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_num_predict_minus_one_or_two_keeps_the_default_cap_without_chat_fit_output(self):
        for value in (-1, -2):
            with self.subTest(num_predict=value):
                response, reply = self.chat("hello", options={"num_predict": value})
                self.assertEqual(response.status, 200, reply)
                self.assertEqual(json.loads(reply["message"]["content"])["limit"], 1024)
                self.assertIsNone(response.getheader("X-Crit-Max-Tokens-Applied"))
        response, value = self.chat("hello", options={"num_predict": 8192})
        self.assertEqual(response.status, 400, value)

    def test_generate_renders_the_prompt_as_a_user_turn(self):
        response, data = self.call("POST", "/api/generate", {"model": "default", "prompt": "echo-request", "system": "Be brief.", "stream": False})
        self.assertEqual(response.status, 200, data)
        value = json.loads(data)
        echoed = json.loads(value["response"])
        self.assertEqual(echoed["messages"], [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "echo-request"}])
        self.assertTrue(value["done"])
        response, data = self.call("POST", "/api/generate", {"model": "default", "prompt": "hi", "raw": True})
        self.assertEqual(response.status, 400)
        self.assertIn("raw", json.loads(data)["error"])


class FitOutputTests(DialectFixture, unittest.TestCase):
    """A package declaring chat-fit-output-v1 fits Anthropic's max_tokens and Ollama's num_predict to the context."""
    fit_output = True

    def test_anthropic_max_tokens_past_the_context_is_fitted_and_a_prompt_alone_still_overflows(self):
        response, value = self.messages("overflow:8150", max_tokens=32000)
        self.assertEqual(response.status, 200, value)
        self.assertEqual((value["stop_reason"], value["usage"]["output_tokens"]), ("max_tokens", 42))
        self.assertEqual(response.getheader("X-Crit-Max-Tokens-Applied"), "42")
        response, data = self.messages("overflow:8150", max_tokens=32000, stream=True)
        self.assertEqual(events(data)[-2][1]["delta"]["stop_reason"], "max_tokens")
        response, value = self.messages("overflow:9000", max_tokens=32000)
        self.assertEqual((response.status, value["error"]["message"]), (400, "prompt is too long: 9000 tokens > 8192 maximum"))
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_ollama_num_predict_minus_one_or_two_or_past_the_context_is_fitted(self):
        for value in (-1, -2, 9000):
            with self.subTest(num_predict=value):
                response, data = self.call("POST", "/api/chat", {"model": "default", "stream": False, "messages": [
                    {"role": "user", "content": "hello"}], "options": {"num_predict": value}})
                reply = json.loads(json.loads(data)["message"]["content"])
                self.assertEqual((reply["limit"], reply["fit_output"]), (8191, True))
                self.assertEqual(response.getheader("X-Crit-Max-Tokens-Applied"), "8182")
        response, data = self.call("POST", "/api/generate", {"model": "default", "prompt": "overflow:8150", "stream": False,
                                                             "options": {"num_predict": -1}})
        value = json.loads(data)
        self.assertEqual((value["done_reason"], value["eval_count"]), ("length", 42))
        # Without num_predict the default cap of 1024 stays, unfitted.
        response, data = self.call("POST", "/api/chat", {"model": "default", "stream": False, "messages": [
            {"role": "user", "content": "hello"}]})
        reply = json.loads(json.loads(data)["message"]["content"])
        self.assertEqual((reply["limit"], reply["fit_output"]), (1024, None))

    def test_the_cards_largest_output_cap_is_the_context_less_one_and_a_larger_cap_is_fitted_to_it(self):
        response, data = self.call("GET", "/v1/models")
        limits = json.loads(data)["data"][0]["limits"]
        self.assertEqual((limits["context_length"], limits["max_output_tokens"]), (8192, 8191))
        for extra in (0, 1):
            body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": limits["max_output_tokens"] + extra}
            response, data = self.call("POST", "/v1/chat/completions", body)
            self.assertEqual(response.status, 200, data)
            reply = json.loads(json.loads(data)["choices"][0]["message"]["content"])
            self.assertEqual((reply["limit"], reply["fit_output"]), (8191, True if extra else None))


class NamedOllamaTests(DialectFixture, unittest.TestCase):
    """--model-name: the default candidate is listed and loaded under the given name."""
    model_name = "local-model"
    test_ps_lists_the_loaded_model_and_tags_carry_package_digests = \
        OllamaTests.test_ps_lists_the_loaded_model_and_tags_carry_package_digests


class StreamErrorTests(DialectFixture, unittest.TestCase):
    """An engine error after a stream's headers are sent is the error the request gets as a response."""

    def failed(self, path, body):
        """(status, error) of a request that fails, once its stream has ended with the same error. Each carries its
        own request's id, its X-Request-Id, which the error returned leaves out."""
        response, data = self.call("POST", path, dict(body, stream=True))
        self.assertEqual(response.status, 200)
        if path == "/v1/messages":
            self.assertEqual([name for name, _ in events(data)], ["message_start", "error"])
            streamed = events(data)[-1][1]
        elif path == "/api/chat":
            streamed = json.loads(data.decode().splitlines()[-1])
        else:
            found = [line[6:] for line in data.decode().split("\n\n") if line.startswith("data: ")]
            self.assertEqual(found[-1], "[DONE]")
            streamed = json.loads(found[-2])
        self.assertEqual(streamed.pop("request_id"), response.getheader("X-Request-Id"))
        response, data = self.call("POST", path, dict(body, stream=False))
        value = json.loads(data)
        self.assertEqual(value.pop("request_id"), response.getheader("X-Request-Id"))
        self.assertEqual(value, streamed)
        return response.status, streamed

    def errors(self, content):
        messages = [{"role": "user", "content": content}]
        return (self.failed("/v1/chat/completions", {"messages": messages, "max_tokens": 100}),
                self.failed("/v1/messages", {"model": "x", "max_tokens": 100, "messages": messages}),
                self.failed("/api/chat", {"model": "default", "messages": messages, "options": {"num_predict": 100}}))

    def test_a_mid_stream_error_keeps_its_status_and_type_in_each_dialect(self):
        openai, messages, chat = self.errors("overflow:9000")
        self.assertEqual((openai[0], openai[1]["error"]["type"], openai[1]["error"]["code"]),
                         (400, "invalid_request_error", "context_length_exceeded"))
        self.assertEqual((messages[0], messages[1]["error"]["type"]), (400, "invalid_request_error"))
        self.assertEqual((chat[0], list(chat[1])), (400, ["error"]))
        with mock.patch.object(self.served.ownership, "blocked", "synthetic cleanup failure"):
            found = self.errors("reply:hi")
        message = "Runtime cleanup is unresolved: synthetic cleanup failure"
        error = {"message": message, "type": "server_error", "param": None, "code": None}
        self.assertEqual(found, ((503, {"error": error}),
                                 (503, {"type": "error", "error": {"type": "overloaded_error", "message": message}}),
                                 (503, {"error": message})))
        with mock.patch.object(self.served, "timeout", 1):
            status, value = self.failed("/v1/messages", {"model": "x", "max_tokens": 100,
                                                         "messages": [{"role": "user", "content": "slow"}]})
        self.assertEqual((status, value["error"]["type"]), (504, "timeout_error"))
        self.assertEqual(self.served.status()["engine_loads"], 1)


class MethodsAndBodiesTests(DialectFixture, unittest.TestCase):
    """HEAD probes, methods no endpoint takes, chunked request bodies and body-size errors, each in the path's dialect."""

    def send(self, method, path, body, headers=None, **options):
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=30)
        self.addCleanup(connection.close)
        connection.request(method, path, body, {"Content-Type": "application/json", **(headers or {})}, **options)
        response = connection.getresponse()
        return response, response.read()

    def test_head_answers_the_probe_paths_with_get_headers_and_no_body(self):
        for path in ("/", "/health", "/api/version", "/api/tags", "/v1/models"):
            response, data = self.call("HEAD", path)
            got, _ = self.call("GET", path)
            self.assertEqual((response.status, data), (200, b""), path)
            self.assertEqual(response.getheader("Content-Type"), got.getheader("Content-Type"), path)
            self.assertEqual(response.getheader("Content-Length"), got.getheader("Content-Length"), path)

    def test_other_methods_are_405_on_known_paths_and_404_elsewhere_in_each_dialect(self):
        response, data = self.call("DELETE", "/v1/chat/completions")
        error = json.loads(data)["error"]
        self.assertEqual((response.status, error["code"]), (405, "method_not_allowed"))
        self.assertEqual(response.getheader("Allow"), "POST, OPTIONS")
        response, data = self.call("PUT", "/v1/messages", {"x": 1})
        self.assertEqual((response.status, json.loads(data)["type"]), (405, "error"))
        response, data = self.call("DELETE", "/api/delete", {"model": "x"})
        self.assertEqual(response.status, 404)
        self.assertEqual(list(json.loads(data)), ["error", "request_id"])
        response, data = self.call("DELETE", "/api/tags")
        self.assertEqual((response.status, response.getheader("Allow")), (405, "GET, HEAD, OPTIONS"))
        self.assertEqual(list(json.loads(data)), ["error", "request_id"])
        response, data = self.call("PATCH", "/v1/nothing")
        self.assertEqual((response.status, json.loads(data)["error"]["code"]), (404, "not_found"))

    def test_a_chunked_body_is_read_like_a_sized_one(self):
        body = json.dumps({"model": "default", "messages": [{"role": "user", "content": "reply:hi"}], "stream": False}).encode()
        response, data = self.send("POST", "/api/chat", iter([body[:10], body[10:]]), encode_chunked=True)
        self.assertEqual(response.status, 200, data)
        self.assertEqual(json.loads(data)["message"]["content"], "hi")
        body = json.dumps({"messages": [{"role": "user", "content": "reply:hi"}]}).encode()
        response, data = self.send("POST", "/v1/chat/completions", iter([body]), encode_chunked=True)
        self.assertEqual(response.status, 200, data)

    def test_body_errors_answer_in_the_dialect_shape(self):
        half = 5 * 1024 * 1024
        chunks = b"%x\r\n" % half + b" " * half + b"\r\n%x\r\n" % half  # the second chunk is never sent
        response, data = self.send("POST", "/v1/messages", chunks, {"Transfer-Encoding": "chunked"})
        value = json.loads(data)
        self.assertEqual((response.status, value["type"], value["error"]["type"]), (413, "error", "request_too_large"))
        response, data = self.send("POST", "/api/chat", b"", {"Content-Length": "0"})
        self.assertEqual((response.status, list(json.loads(data))), (400, ["error", "request_id"]))
        response, data = self.send("POST", "/api/chat", b"", {"Content-Length": str(2 * half)})
        self.assertEqual((response.status, list(json.loads(data))), (413, ["error", "request_id"]))
        response, data = self.send("POST", "/v1/messages", b"zz\r\n{}\r\n0\r\n\r\n", {"Transfer-Encoding": "chunked"})
        value = json.loads(data)
        self.assertEqual((response.status, value["error"]["type"]), (400, "invalid_request_error"))
        self.assertIn("malformed", value["error"]["message"])
        response, data = self.send("POST", "/v1/chat/completions", b"", {"Content-Length": "0"})
        self.assertEqual((response.status, json.loads(data)["error"]["type"]), (400, "invalid_request_error"))

    def test_a_malformed_length_is_a_400_and_closes_even_where_the_body_may_be_empty(self):
        # The bytes after the headers are an unread body, never a second request.
        smuggled = b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
        for path in ("/v1/crit/unload", "/v1/chat/completions"):
            for length in (b"abc", b"-5"):
                request = b"POST %s HTTP/1.1\r\nHost: x\r\nContent-Length: %s\r\n\r\n" % (path.encode(), length)
                with socket.create_connection(self.server.server_address[:2], timeout=30) as connection:
                    connection.sendall(request + smuggled)
                    data = b"".join(iter(lambda: connection.recv(65536), b""))
                self.assertEqual((data.count(b"HTTP/1.1 "), data.split(b" ", 2)[1]), (1, b"400"), (path, length))
        self.assertTrue(json.loads(self.call("GET", "/health")[1])["loaded"])
        response, data = self.send("POST", "/v1/crit/unload", b"0\r\n\r\n", {"Transfer-Encoding": "chunked"})
        self.assertEqual((response.status, json.loads(data)["unloaded"]), (200, True))


class LifecycleTests(DialectFixture, unittest.TestCase):
    """Unloading and loading the engine through the API: crit's own endpoints, and Ollama's requests without a prompt
    and keep_alive 0."""

    def loaded(self):
        return json.loads(self.call("GET", "/health")[1])["loaded"]

    def test_unload_frees_the_weights_and_the_next_request_loads_them_again(self):
        pid = self.served.host_pid()
        response, data = self.call("POST", "/v1/crit/unload")
        value = json.loads(data)
        self.assertEqual((response.status, value["object"], value["unloaded"]), (200, "crit.unload", True), value)
        self.assertEqual([value["engine"][k] for k in ("state", "engine", "host_bytes", "device_bytes")],
                         ["unloaded", None, 0, 0])
        health = json.loads(self.call("GET", "/health")[1])
        self.assertEqual((health["status"], health["loaded"]), ("ok", False))
        self.assertEqual([m["loaded"] for m in json.loads(self.call("GET", "/v1/models")[1])["data"]], [False, False])
        self.assertEqual(json.loads(self.call("GET", "/api/ps")[1])["models"], [])
        response, data = self.call("POST", "/v1/crit/unload", {})
        self.assertEqual((response.status, json.loads(data)["unloaded"]), (200, False))
        response, data = self.call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:hi"}]})
        self.assertEqual(response.status, 200, data)
        self.assertEqual((self.loaded(), self.served.status()["engine_loads"]), (True, 2))
        self.assertEqual(self.served.host_pid(), pid)  # the host process stays; the engine and its weights went

    def test_load_preloads_a_model_and_unload_with_a_model_only_unloads_that_one(self):
        b = self.run_id + "/" + self.b
        response, data = self.call("POST", "/v1/crit/unload", {"model": b})
        self.assertEqual((response.status, json.loads(data)["unloaded"], self.loaded()), (200, False, True))
        response, data = self.call("POST", "/v1/crit/load", {"model": b})
        value = json.loads(data)
        self.assertEqual((response.status, value["object"], value["model"]), (200, "crit.load", b), value)
        self.assertEqual(value["engine"]["engine"]["candidate_id"], self.b)
        self.assertEqual([m["loaded"] for m in json.loads(self.call("GET", "/v1/models")[1])["data"]], [False, True])
        response, data = self.call("POST", "/v1/crit/load")
        self.assertEqual(json.loads(data)["engine"]["engine"]["candidate_id"], self.a)
        loads = self.served.status()["engine_loads"]
        response, data = self.call("POST", "/v1/crit/load", {"model": "default"})
        self.assertEqual((response.status, self.served.status()["engine_loads"]), (200, loads))  # already loaded
        for body, status, param in (({"model": "candidate-9999"}, 404, "model"), ({"model": 3}, 400, "model"),
                                    ({"keep_alive": 0}, 400, "keep_alive")):
            response, data = self.call("POST", "/v1/crit/load", body)
            self.assertEqual((response.status, json.loads(data)["error"]["param"]), (status, param), body)
        response, data = self.call("DELETE", "/v1/crit/unload")
        self.assertEqual((response.status, response.getheader("Allow")), (405, "POST, OPTIONS"))

    def test_ollama_requests_without_a_prompt_load_and_keep_alive_0_unloads(self):
        response, data = self.call("POST", "/api/generate", {"model": "default", "keep_alive": 0})
        value = json.loads(data)
        self.assertEqual((response.status, value["model"], value["response"], value["done"], value["done_reason"]),
                         (200, "default", "", True, "unload"), value)
        self.assertFalse(self.loaded())
        response, data = self.call("POST", "/api/chat", {"model": "default", "messages": []})
        value = json.loads(data)
        self.assertEqual((value["message"], value["done"], value["done_reason"]),
                         ({"role": "assistant", "content": ""}, True, "load"))
        self.assertTrue(self.loaded())
        response, data = self.call("POST", "/api/chat", {"model": "default", "keep_alive": "0"})
        self.assertEqual((json.loads(data)["done_reason"], self.loaded()), ("unload", False))
        response, data = self.call("POST", "/api/generate", {"model": "default", "prompt": ""})
        self.assertEqual((json.loads(data)["done_reason"], self.loaded()), ("load", True))
        # keep_alive 0 on a reply: the reply as usual, then the model unloads.
        response, data = self.call("POST", "/api/chat", {"model": "default", "stream": False, "keep_alive": "0s",
                                                         "messages": [{"role": "user", "content": "reply:hi"}]})
        self.assertEqual((response.status, json.loads(data)["done_reason"]), (200, "stop"))
        self.assertIsNone(response.getheader("X-Crit-Ignored-Parameters"))
        end = time.monotonic() + 10
        while self.loaded() and time.monotonic() < end:
            time.sleep(.05)
        self.assertFalse(self.loaded())
        response, data = self.call("POST", "/api/generate", {"model": "candidate-9999"})
        self.assertEqual((response.status, list(json.loads(data))), (404, ["error", "request_id"]))
        response, data = self.call("POST", "/api/chat", {"model": "default", "messages": [], "mystery": 1})
        self.assertEqual((response.status, list(json.loads(data))), (400, ["error", "request_id"]))

    def test_keep_alive_0_is_zero_in_any_spelling_and_idle_unload_sets_ps_expiry(self):
        for value in (0, 0.0, "0", "0s", "0m", "00.0h"):
            self.assertTrue(ollama.unloads(value), value)
        for value in (None, False, 1, -1, "", "5m", "-1", "0x"):
            self.assertFalse(ollama.unloads(value), value)
        card = {"id": "m", "label": "M", "context_length": 8}
        self.assertEqual(ollama.loaded_card(card, 0, "", {})["expires_at"], ollama.LOADED_UNTIL)
        self.assertEqual(ollama.loaded_card(card, 0, "", {}, 86400)["expires_at"], "1970-01-02T00:00:00Z")


class LifecycleKeyTests(DialectFixture, unittest.TestCase):
    api_keys = ("secret",)

    def test_load_and_unload_need_the_key(self):
        for path in ("/v1/crit/unload", "/v1/crit/load"):
            response, data = self.call("POST", path)
            self.assertEqual((response.status, json.loads(data)["error"]["code"]), (401, "invalid_api_key"))
        self.assertTrue(json.loads(self.call("GET", "/health")[1])["loaded"])
        response, data = self.call("POST", "/v1/crit/unload", headers={"Authorization": "Bearer secret"})
        self.assertEqual((response.status, json.loads(data)["unloaded"]), (200, True))


class LenientTests(DialectFixture, unittest.TestCase):
    """crit serve --lenient over a package that applies only the temperature: what crit does not apply is ignored,
    listed in X-Crit-Ignored-Parameters and logged once, in every dialect; values are still checked."""
    lenient = True
    request_options = False

    def setUp(self):
        super().setUp()
        self.logged = []
        self.server.api.log = self.logged.append

    def test_unknown_keys_hints_and_unapplied_sampling_are_ignored_listed_and_logged_once(self):
        body = {"messages": [{"role": "user", "content": "echo-request"}], "temperature": .5, "top_p": .9,
                "typical_p": .9, "mirostat": 0, "reasoning_effort": "low", "verbosity": "low", "user": "u"}
        for _ in range(2):
            response, data = self.call("POST", "/v1/chat/completions", body)
            self.assertEqual(response.status, 200, data)
            self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"),
                             "reasoning_effort,top_p,typical_p,user,verbosity")
            echoed = json.loads(json.loads(data)["choices"][0]["message"]["content"])
            self.assertEqual(echoed["sampling"], {"temperature": .5})
        warnings = [line for line in self.logged if line.startswith("warning:")]
        self.assertEqual(warnings, [f"warning: ignoring request parameter {key}; responses list it in "
                                    "X-Crit-Ignored-Parameters" for key in
                                    ("reasoning_effort", "top_p", "typical_p", "user", "verbosity")])

    def test_an_ollama_request_that_only_loads_ignores_unknown_keys_too(self):
        response, data = self.call("POST", "/api/chat", {"model": "default", "messages": [], "mystery": 1})
        self.assertEqual((response.status, json.loads(data)["done_reason"]), (200, "load"), data)
        response, data = self.call("POST", "/api/generate", {"model": "default", "keep_alive": 0, "mystery": 1})
        self.assertEqual((response.status, json.loads(data)["done_reason"]), (200, "unload"), data)

    def test_thinking_controls_of_a_package_without_template_variables_are_ignored_whatever_they_hold(self):
        for path, body, listed in (
                ("/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:hi"}],
                                          "reasoning_effort": "max", "chat_template_kwargs": ["x"]},
                 "chat_template_kwargs,reasoning_effort"),
                ("/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:hi"}],
                                          "chat_template_kwargs": {}}, None),
                ("/v1/responses", {"input": "reply:hi", "reasoning": {"effort": "max"}}, "reasoning_effort"),
                ("/v1/messages", {"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "reply:hi"}],
                                  "thinking": {"type": "sometimes"}}, "thinking"),
                ("/api/chat", {"model": "default", "stream": False, "think": "max",
                               "messages": [{"role": "user", "content": "reply:hi"}]}, "think")):
            with self.subTest(path=path, body=body):
                response, data = self.call("POST", path, body)
                self.assertEqual((response.status, response.getheader("X-Crit-Ignored-Parameters")), (200, listed),
                                 data)

    def test_values_are_still_checked_and_features_that_change_the_reply_are_still_refused(self):
        for options, param in (({"top_p": 1.5}, "top_p"), ({"typical_p": 2}, "typical_p"), ({"mirostat": 3}, "mirostat"),
                               ({"functions": [{"name": "f"}]}, "functions"), ({"logprobs": True}, "logprobs"),
                               ({"tools": OPENAI_TOOLS}, "tools"), ({"n": 2}, "n")):
            with self.subTest(options=options):
                body = {"messages": [{"role": "user", "content": "hi"}], **options}
                response, data = self.call("POST", "/v1/chat/completions", body)
                self.assertEqual((response.status, json.loads(data)["error"]["param"]), (400, param), data)
        response, data = self.call("POST", "/v1/crit/score", {"prompt_ids": [1], "typo": 1})
        self.assertEqual((response.status, json.loads(data)["error"]["param"]), (400, "typo"))
        response, value = self.messages("hi", mcp_servers=[{"url": "x"}])
        self.assertEqual(response.status, 400, value)
        response, data = self.call("POST", "/api/generate", {"model": "default", "prompt": "hi", "raw": True})
        self.assertEqual(response.status, 400, data)

    def test_anthropic_and_ollama_requests_ignore_unknown_keys_and_options(self):
        response, value = self.messages("reply:hi", top_p=.9, top_k=5, mystery=1)
        self.assertEqual((response.status, value["content"]), (200, [{"type": "text", "text": "hi"}]), value)
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "mystery,top_k,top_p")
        body = {"model": "default", "stream": False, "messages": [{"role": "user", "content": "echo-request"}],
                "options": {"temperature": .5, "top_p": .9, "dry_multiplier": .8, "num_ctx": 4096}, "mystery": 1}
        response, data = self.call("POST", "/api/chat", body)
        self.assertEqual(response.status, 200, data)
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"),
                         "mystery,options.dry_multiplier,options.num_ctx,top_p")
        self.assertEqual(json.loads(json.loads(data)["message"]["content"])["sampling"], {"temperature": .5})

    def test_responses_requests_ignore_unknown_keys_and_still_refuse_stored_state(self):
        body = {"input": "echo-request", "temperature": .5, "top_p": .9, "mystery": 1, "store": False}
        response, data = self.call("POST", "/v1/responses", body)
        self.assertEqual(response.status, 200, data)
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "mystery,store,top_p")
        self.assertEqual(json.loads(json.loads(data)["output"][-1]["content"][0]["text"])["sampling"], {"temperature": .5})
        response, data = self.call("POST", "/v1/responses", {"input": "hi", "previous_response_id": "resp_1"})
        self.assertEqual((response.status, json.loads(data)["error"]["code"]), (400, "unsupported_parameter"), data)

    def test_unknown_stream_options_keys_are_ignored_and_listed_under_stream_options(self):
        body = {"messages": [{"role": "user", "content": "reply:hi"}], "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        response, data = self.call("POST", "/v1/chat/completions", body)
        self.assertEqual((response.status, response.getheader("X-Crit-Ignored-Parameters")),
                         (200, "stream_options.continuous_usage_stats"), data)
        self.assertIn("warning: ignoring request parameter stream_options.continuous_usage_stats; responses list it in "
                      "X-Crit-Ignored-Parameters", self.logged)

    def test_a_name_the_header_cannot_carry_as_sent_is_still_refused(self):
        """X-Crit-Ignored-Parameters and the log carry ignored names as sent: a line break would add a header or a log
        line, a name outside Latin-1 would fail the reply after generating it, and a comma would split the list."""
        turn = [{"role": "user", "content": "reply:hi"}]
        anthropic = {"model": "x", "max_tokens": 5, "messages": turn}
        for name in ("a\r\nX-Injected: yes", "温度", "a,b", "x" * 65, ""):
            for path, body in (("/v1/chat/completions", {"messages": turn, name: 1}),
                               ("/v1/chat/completions", {"messages": turn, "stream": True, name: 1}),
                               ("/v1/chat/completions", {"messages": turn, "stream": True, "stream_options": {name: 1}}),
                               ("/v1/messages", {**anthropic, name: 1}),
                               ("/v1/messages", {**anthropic, "output_config": {name: 1}}),
                               ("/v1/responses", {"input": "reply:hi", name: 1}),
                               ("/api/chat", {"model": "default", "messages": turn, name: 1}),
                               ("/api/chat", {"model": "default", "messages": turn, "options": {name: 1}}),
                               ("/api/generate", {"model": "default", "prompt": "reply:hi", name: 1})):
                with self.subTest(name=name, path=path, body=body):
                    response, data = self.call("POST", path, body)
                    self.assertEqual(response.status, 400, data)
                    self.assertIsNone(response.getheader("X-Injected"))
        self.assertEqual([line for line in self.logged if line.startswith("warning:")], [])
        response, data = self.call("POST", "/v1/chat/completions", {"messages": turn, "x.y-z_9": 1})
        self.assertEqual((response.status, response.getheader("X-Crit-Ignored-Parameters")), (200, "x.y-z_9"), data)


class KeepAliveTests(DialectFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.server.api.ping_interval = .05

    def quiet(self, path, content, **options):
        """A stream's response to a request the engine takes but answers only half a second later, as while a model
        loads or a long prompt prefills."""
        turn = [{"role": "user", "content": content}]
        body = {"model": "default", "stream": True, **({"input": turn} if path == "/v1/responses" else {"messages": turn}),
                **options}
        call = self.served.runtime.call

        def prefilling(action, *args, **kwargs):
            if action == "generate":
                time.sleep(.5)
            return call(action, *args, **kwargs)
        with mock.patch.object(self.served.runtime, "call", prefilling):
            response, data = self.call("POST", path, body)
        self.assertEqual(response.status, 200, data)
        return data

    def test_a_quiet_openai_stream_gets_ping_comments_between_whole_events(self):
        frames = self.quiet("/v1/chat/completions", "reply:hello").decode().split("\n\n")
        self.assertEqual(frames.pop(), "")
        self.assertEqual(json.loads(frames[0][6:])["choices"][0]["delta"], {"role": "assistant", "content": ""})
        text = next(i for i, frame in enumerate(frames) if '"content": "hello"' in frame)
        self.assertGreaterEqual(frames[1:text].count(": ping"), 3)
        self.assertEqual(set(frames[1:text]), {": ping"})
        self.assertEqual(frames[-1], "data: [DONE]")  # pings end before the closing events
        data = [frame for frame in frames[:-1] if frame != ": ping"]
        self.assertEqual([json.loads(frame[6:])["choices"][0]["finish_reason"] for frame in data], [None, None, "stop"])

    def test_a_quiet_responses_stream_gets_ping_comments_after_in_progress(self):
        frames = self.quiet("/v1/responses", "reply:hello").decode().split("\n\n")
        self.assertEqual(frames.pop(), "")
        names = [frame.split("\n", 1)[0] for frame in frames]
        self.assertEqual(names[:2], ["event: response.created", "event: response.in_progress"])
        added = names.index("event: response.output_item.added")
        self.assertGreaterEqual(added - 2, 3)
        self.assertEqual(set(names[2:added]), {": ping"})
        self.assertEqual(names[-1], "event: response.completed")  # pings end before the closing events

    def test_a_quiet_anthropic_stream_gets_ping_events(self):
        data = self.quiet("/v1/messages", "reply:hello", max_tokens=50)
        found = events(data)
        self.assertEqual(len(found), data.count(b"\n\n"))  # every frame a whole event
        names = [name for name, _ in found]
        self.assertEqual(names[0], "message_start")
        start = names.index("content_block_start")
        self.assertGreaterEqual(start, 4)
        self.assertEqual(found[1:start], [("ping", {"type": "ping"})] * (start - 1))
        self.assertEqual([name for name in names if name != "ping"][-4:],
                         ["content_block_delta", "content_block_stop", "message_delta", "message_stop"])
        self.assertEqual(names[-1], "message_stop")

    def test_ollama_streams_and_a_zero_interval_send_no_pings(self):
        data = self.quiet("/api/chat", "reply:hello")
        lines = [json.loads(line) for line in data.decode().splitlines()]
        self.assertEqual([line["done"] for line in lines], [False, True])
        self.server.api.ping_interval = 0
        self.assertNotIn(b": ping", self.quiet("/v1/chat/completions", "reply:hello"))
        self.assertNotIn(b"event: ping", self.quiet("/v1/messages", "reply:hello", max_tokens=50))

    def test_a_queued_stream_has_no_status_yet_so_it_gets_no_pings_and_the_queue_can_still_refuse_it(self):
        """A stream opens, and its pings start, once the engine takes the request: one still waiting for the engine
        has sent nothing, so a refusal is its status, with Retry-After, not an error after a 200 of pings."""
        self.served.lock.queue_timeout = .3
        turn = [{"role": "user", "content": "reply:hello"}]
        for path, body, status in (("/v1/chat/completions", {"messages": turn, "stream": True}, 429),
                                   ("/v1/responses", {"input": turn, "stream": True}, 429),
                                   ("/v1/messages", {"model": "x", "max_tokens": 5, "messages": turn, "stream": True},
                                    529)):
            with self.subTest(path=path):
                with self.served.lock:  # the engine is busy for longer than the queue timeout
                    response, data = self.call("POST", path, body)
                self.assertEqual((response.status, response.getheader("Retry-After")), (status, "1"), data)
                self.assertNotIn(b"ping", data)
                self.assertIn("--queue-timeout", data.decode())

    def test_crit_runs_servers_send_no_pings_and_eval_still_stops_a_stalled_engine_as_hung(self):
        self.assertEqual(Api(self.served).ping_interval, 0)  # how crit run eval and measure build theirs (Served.open)
        client = chat.Server("http://%s:%d/v1" % self.server.server_address[:2], {"speed": []})
        # The engine takes the request and then works five seconds without output, until cancelled.
        body = {"model": "default", "stream": True, "messages": [{"role": "user", "content": "slow"}]}
        found, outcome = [], {}

        def read():  # pings every .05 s while the engine is quiet
            try:
                found.extend(client.events("/chat/completions", body, .3))
            except Hung as exc:
                outcome["hung"] = str(exc)
        thread = threading.Thread(target=read)
        thread.start()
        thread.join(30)
        self.assertRegex(outcome.get("hung", ""), "/chat/completions gave no answer")
        self.assertEqual([event["choices"][0]["delta"] for event in found], [{"role": "assistant", "content": ""}])


class ReadinessTests(DialectFixture, unittest.TestCase):
    """/health's ready: an engine is loaded and no switch, reload or retire is under way; ?ready=1 is 503 until then."""

    def health(self, query=""):
        response, data = self.call("GET", "/health" + query)
        value = json.loads(data)
        return response.status, value["status"], value["ready"]

    def test_a_loaded_engine_is_ready_also_while_it_generates(self):
        self.assertEqual(self.health(), (200, "ok", True))
        self.assertEqual(self.health("?ready=1"), (200, "ok", True))
        stop = threading.Event()
        wait = self.background(self.served, "slow", stop)
        self.generating(self.served)
        self.assertEqual(self.health("?ready=1"), (200, "ok", True))
        stop.set()
        self.assertEqual(wait(), ("cancelled", None))

    def test_a_switch_a_reload_and_a_retire_are_not_ready_until_the_engine_loads(self):
        self.assertTrue(self.served.lock.admit(self.b, threading.Event()))  # a request for B, loading its engine
        try:
            self.assertEqual(self.health(), (200, "ok", False))
            self.assertEqual(self.health("?ready=1"), (503, "ok", False))
            response, data = self.call("HEAD", "/health?ready=1")
            self.assertEqual((response.status, data), (503, b""))
        finally:
            self.served.lock.release()
        with self.served.lock:  # a reload holds the engine
            self.assertEqual(self.health("?ready=true"), (503, "ok", False))
        response, data = self.call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "crash"}]})
        self.assertEqual(response.status, 500, data)
        self.assertEqual(json.loads(self.call("GET", "/health")[1])["engine"]["state"], "unloaded")
        self.assertEqual(self.health("?ready=1"), (503, "ok", False))
        response, data = self.call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:hi"}]})
        self.assertEqual(response.status, 200, data)
        self.assertEqual(self.health("?ready=1"), (200, "ok", True))

    def test_a_cancelled_switch_leaves_the_loaded_engine_ready(self):
        stop, call = threading.Event(), self.served.runtime.call
        def load(action, *args, **kwargs):
            stop.set()  # the client leaves before B's engine loads
            return call(action, *args, **kwargs)
        with mock.patch.object(self.served.runtime, "call", load):
            self.assertEqual(self.served.generate([{"role": "user", "content": "reply:hi"}], 10, stop, candidate_id=self.b),
                             ("cancelled", None, None))
        self.assertEqual(self.health("?ready=1"), (200, "ok", True))
        engine = json.loads(self.call("GET", "/health")[1])["engine"]
        self.assertEqual((engine["state"], engine["engine"]["candidate_id"], engine["engine_loads"]), ("ready", self.a, 1))


class LimitsTests(DialectFixture, unittest.TestCase):
    def test_cards_and_show_state_the_limits_a_request_is_held_to(self):
        response, data = self.call("GET", "/v1/models/" + self.run_id + "/" + self.b)
        limits = json.loads(data)["limits"]
        self.assertEqual(limits, {"context_length": 8192, "max_output_tokens": 8191, "max_body_bytes": 8 * 1024 * 1024,
                                  "max_stops": 16, "max_stop_bytes": 256, "max_tools": 128, "max_top_logprobs": 20,
                                  "max_reply_characters": 65536})
        response, data = self.call("POST", "/api/show", {"model": self.run_id + "/" + self.b})
        self.assertEqual(json.loads(data)["limits"], limits)
        tool = lambda i: {"type": "function", "function": {"name": "f%d" % i}}
        for extra, status in ((0, 200), (1, 400)):
            body = {"messages": [{"role": "user", "content": "reply:hi"}], "max_tokens": limits["max_output_tokens"] + extra}
            response, data = self.call("POST", "/v1/chat/completions", body)
            self.assertEqual(response.status, status, data)
            body = {"messages": [{"role": "user", "content": "reply:hi"}],
                    "tools": [tool(i) for i in range(limits["max_tools"] + extra)]}
            response, data = self.call("POST", "/v1/chat/completions", body)
            self.assertEqual(response.status, status, data)


class MetricsTests(DialectFixture, unittest.TestCase):
    def test_metrics_count_requests_by_dialect_and_outcome_and_carry_the_last_rates(self):
        self.call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:hi"}]})
        self.call("POST", "/v1/messages", {"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "reply:hi"}]})
        self.call("POST", "/api/chat", {"model": "default", "messages": [{"role": "user", "content": "reply:hi"}], "stream": False})
        self.call("POST", "/v1/responses", {"input": "reply:hi"})
        # Rejected: three requests that fail validation and a prompt over the context.
        self.call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "n": 2})
        self.call("POST", "/v1/messages", {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        self.call("POST", "/v1/responses", {"input": "hi", "previous_response_id": "resp_1"})
        self.call("POST", "/api/chat", {"model": "default", "messages": [{"role": "user", "content": "overflow:9000"}],
                                        "stream": False})
        response, data = self.call("GET", "/metrics")
        self.assertEqual(response.status, 200)
        self.assertTrue(response.getheader("Content-Type").startswith("text/plain"))
        text = data.decode()
        for line in ('crit_serve_requests_total{dialect="anthropic",outcome="completed"} 1',
                     'crit_serve_requests_total{dialect="ollama",outcome="completed"} 1',
                     'crit_serve_requests_total{dialect="openai",outcome="completed"} 1',
                     'crit_serve_requests_total{dialect="responses",outcome="completed"} 1',
                     'crit_serve_requests_total{dialect="anthropic",outcome="rejected"} 1',
                     'crit_serve_requests_total{dialect="ollama",outcome="rejected"} 1',
                     'crit_serve_requests_total{dialect="openai",outcome="rejected"} 1',
                     'crit_serve_requests_total{dialect="responses",outcome="rejected"} 1',
                     "crit_serve_prompt_tokens_total 40", "crit_serve_completion_tokens_total 16",
                     "crit_serve_requests_in_flight 0", "crit_serve_requests_waiting 0",
                     "# TYPE crit_serve_last_decode_tokens_per_second gauge"):
            self.assertIn(line, text)
        self.assertNotIn('outcome="error"', text)
        self.assertIn('crit_serve_engine_loaded{run="%s",candidate="%s",state="ready"} 1' % (self.run_id, self.a), text)

    def test_histograms_split_the_three_completed_requests_by_their_telemetry(self):
        for _ in range(3):
            self.call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:hi"}]})
        response, data = self.call("GET", "/metrics")
        found = samples(data.decode())
        self.assertIn("# TYPE crit_serve_time_to_first_token_seconds histogram", data.decode())
        # The synthetic engine reports 4 tokens of 10 prompt tokens, ttft_ms 6 and 3 decode tokens in 10 ms.
        for name, value in (("crit_serve_requests_running", 0), ("crit_serve_requests_waiting", 0),
                            ('crit_serve_time_to_first_token_seconds_bucket{le="0.005"}', 0),
                            ('crit_serve_time_to_first_token_seconds_bucket{le="0.01"}', 3),
                            ("crit_serve_time_to_first_token_seconds_count", 3),
                            ('crit_serve_time_per_output_token_seconds_bucket{le="0.0025"}', 0),
                            ('crit_serve_time_per_output_token_seconds_bucket{le="0.005"}', 3),
                            ('crit_serve_request_prompt_tokens_bucket{le="5"}', 0),
                            ('crit_serve_request_prompt_tokens_bucket{le="10"}', 3),
                            ("crit_serve_request_prompt_tokens_sum", 30),
                            ('crit_serve_request_generation_tokens_bucket{le="2"}', 0),
                            ('crit_serve_request_generation_tokens_bucket{le="5"}', 3),
                            ('crit_serve_request_generation_tokens_bucket{le="+Inf"}', 3),
                            ("crit_serve_request_queue_time_seconds_count", 3),
                            ("crit_serve_e2e_request_latency_seconds_count", 3),
                            ("crit_serve_engine_loads_total", 1), ("crit_serve_model_switches_total", 0)):
            self.assertEqual(found[name], value, name)
        self.assertAlmostEqual(found["crit_serve_time_to_first_token_seconds_sum"], .018)
        self.assertAlmostEqual(found["crit_serve_e2e_request_latency_seconds_sum"], found["crit_serve_request_seconds_total"])

    def test_a_request_for_another_candidate_is_a_load_and_a_model_switch(self):
        for model in (self.run_id + "/" + self.b, self.run_id + "/" + self.b, None):
            body = {"messages": [{"role": "user", "content": "reply:hi"}], **({"model": model} if model else {})}
            response, data = self.call("POST", "/v1/chat/completions", body)
            self.assertEqual(response.status, 200, data)
        found = samples(self.call("GET", "/metrics")[1].decode())
        self.assertEqual((found["crit_serve_engine_loads_total"], found["crit_serve_model_switches_total"]), (3, 2))


class QueuedMetricsTests(DialectFixture, unittest.TestCase):
    adapter_suffix = GATED_ADAPTER

    def test_a_request_behind_a_running_one_is_waiting_until_the_engine_starts_it(self):
        gate = self.root / "generation"
        self.addCleanup(gate.with_suffix(".release").touch)
        results = []
        def post(content):
            results.append(self.call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": content}]}))
        running = threading.Thread(target=post, args=("gate:" + str(gate),))
        running.start()
        self.until(gate.with_suffix(".started").exists)
        queued = threading.Thread(target=post, args=("reply:hi",))
        queued.start()
        self.until(lambda: samples(self.call("GET", "/metrics")[1].decode())["crit_serve_requests_waiting"] == 1)
        found = samples(self.call("GET", "/metrics")[1].decode())
        self.assertEqual((found["crit_serve_requests_running"], found["crit_serve_requests_in_flight"]), (1, 2))
        self.assertEqual(found["crit_serve_request_queue_time_seconds_count"], 1)
        time.sleep(.1)
        gate.with_suffix(".release").touch()
        running.join(10)
        queued.join(10)
        self.assertEqual([response.status for response, _ in results], [200, 200])
        found = samples(self.call("GET", "/metrics")[1].decode())
        self.assertEqual((found["crit_serve_requests_running"], found["crit_serve_requests_waiting"],
                          found["crit_serve_requests_in_flight"]), (0, 0, 0))
        self.assertEqual(found["crit_serve_request_queue_time_seconds_count"], 2)
        self.assertGreater(found["crit_serve_request_queue_time_seconds_sum"], .1)

    def until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("The condition did not hold within %s s" % timeout)
            time.sleep(.01)


class HistogramTests(unittest.TestCase):
    def test_buckets_are_cumulative_and_inclusive_of_their_bound(self):
        found = Histogram((1, 2.5, 5))
        for value in (1, 3, 7):
            found.observe(value)
        self.assertEqual(found.lines("x"), ['x_bucket{le="1"} 1', 'x_bucket{le="2.5"} 1', 'x_bucket{le="5"} 2',
                                            'x_bucket{le="+Inf"} 3', "x_sum 11", "x_count 3"])

    def test_a_request_waits_until_dispatched_and_leaves_either_gauge_when_it_ends(self):
        metrics = Metrics()
        served = types.SimpleNamespace(status=lambda: {"state": "unloaded"}, loads=0, switches=0)
        gauges = lambda: [samples(metrics.render(served))["crit_serve_requests_" + name]
                          for name in ("running", "waiting", "in_flight")]
        first, second = metrics.begin(), metrics.begin()
        self.assertEqual(gauges(), [0, 2, 2])
        metrics.dispatched(first)
        metrics.dispatched(first)
        self.assertEqual(gauges(), [1, 1, 2])
        metrics.end("openai", "cancelled", None, second)
        self.assertEqual(gauges(), [1, 0, 1])
        metrics.end("openai", "completed", {"prompt_tokens": 3, "output_tokens": 1, "prefill_tokens": 3,
                                            "decode_tokens": None, "decode_ms": None, "ttft_ms": None}, first)
        self.assertEqual(gauges(), [0, 0, 0])
        found = samples(metrics.render(served))
        self.assertEqual([found[name + "_count"] for name in (
            "crit_serve_request_queue_time_seconds", "crit_serve_e2e_request_latency_seconds",
            "crit_serve_time_to_first_token_seconds", "crit_serve_time_per_output_token_seconds",
            "crit_serve_request_prompt_tokens")], [1, 2, 0, 0, 1])
