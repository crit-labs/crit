"""crit serve's other dialects over the synthetic resident engine: the Anthropic Messages API, the Ollama API and
the Prometheus metrics endpoint."""
import http.client
import json
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "eval", "kernels", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_api.server import Api, Server
from test_chat_residency import ServeFixture

TOOLS = [{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                                                                             "required": ["path"]}}]
OPENAI_TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": TOOLS[0]["input_schema"]}}]
CALL = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a.py\"}"}}


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
    api_key = None

    def setUp(self):
        super().setUp()
        self.served = self.engine()
        self.server = Server(("127.0.0.1", 0), Api(self.served, api_key=self.api_key, cors_origin="*"))
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
        response, data = self.messages("tool:" + json.dumps([CALL]), tools=TOOLS, stream=True)
        self.assertEqual(response.status, 200)
        found = events(data)
        starts = [d for name, d in found if name == "content_block_start"]
        self.assertEqual(starts, [{"type": "content_block_start", "index": 0,
                                   "content_block": {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {}}}])
        partial = "".join(d["delta"]["partial_json"] for name, d in found
                          if name == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
        self.assertEqual(json.loads(partial), {"path": "a.py"})
        self.assertEqual(found[-2][1]["delta"]["stop_reason"], "tool_use")


class AnthropicKeyTests(DialectFixture, unittest.TestCase):
    api_key = "secret"

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


class OllamaTests(DialectFixture, unittest.TestCase):
    def chat(self, content, **options):
        body = {"model": "default", "messages": [{"role": "user", "content": content}], "stream": False, **options}
        response, data = self.call("POST", "/api/chat", body)
        return response, json.loads(data) if body.get("stream") is False else data

    def test_the_server_answers_what_ollama_clients_probe(self):
        response, data = self.call("GET", "/")
        self.assertEqual((response.status, data), (200, b"Ollama is running"))
        response, data = self.call("GET", "/api/version")
        self.assertIn("version", json.loads(data))
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

    def test_streams_are_newline_delimited_objects_ending_with_done(self):
        response, data = self.call("POST", "/api/chat", {"model": "default", "messages": [{"role": "user", "content": "words:a b c"}]})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "application/x-ndjson")
        rows = [json.loads(line) for line in data.decode().splitlines() if line]
        self.assertEqual("".join(r["message"]["content"] for r in rows[:-1]), "a b c ")
        self.assertTrue(all(r["done"] is False for r in rows[:-1]))
        self.assertEqual((rows[-1]["done"], rows[-1]["done_reason"], rows[-1]["eval_count"]), (True, "stop", 3))

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


class MetricsTests(DialectFixture, unittest.TestCase):
    def test_metrics_count_requests_by_dialect_and_carry_the_last_rates(self):
        self.call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:hi"}]})
        self.call("POST", "/v1/messages", {"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "reply:hi"}]})
        self.call("POST", "/api/chat", {"model": "default", "messages": [{"role": "user", "content": "reply:hi"}], "stream": False})
        self.call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "n": 2})
        response, data = self.call("GET", "/metrics")
        self.assertEqual(response.status, 200)
        self.assertTrue(response.getheader("Content-Type").startswith("text/plain"))
        text = data.decode()
        for line in ('crit_serve_requests_total{dialect="anthropic",outcome="completed"} 1',
                     'crit_serve_requests_total{dialect="ollama",outcome="completed"} 1',
                     'crit_serve_requests_total{dialect="openai",outcome="completed"} 1',
                     "crit_serve_prompt_tokens_total 30", "crit_serve_completion_tokens_total 12",
                     "crit_serve_requests_in_flight 0", "# TYPE crit_serve_last_decode_tokens_per_second gauge"):
            self.assertIn(line, text)
        self.assertIn('crit_serve_engine_loaded{run="%s",candidate="%s",state="ready"} 1' % (self.run_id, self.a), text)
