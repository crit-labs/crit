"""crit serve over HTTP with a synthetic resident adapter; no model, GPU or native build."""
import http.client
import json
import os
import threading
import unittest

from test_chat_residency import ServeFixture
from crit_api.server import Api, RequestError, Server, conversation, stops


class ConversationTests(unittest.TestCase):
    def test_system_first_then_alternating_turns_ending_with_the_user(self):
        messages = conversation([{"role": "developer", "content": "Be brief."},
                                 {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                                 {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"},
                                 {"role": "user", "content": "e"}])
        self.assertEqual(messages, [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "ab\n\nc"},
                                    {"role": "assistant", "content": "d"}, {"role": "user", "content": "e"}])
        for invalid in ([], [{"role": "assistant", "content": "x"}], [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                        [{"role": "user", "content": "x"}, {"role": "system", "content": "late"}, {"role": "user", "content": "y"}],
                        [{"role": "tool", "content": "x"}], [{"role": "user", "content": [{"type": "image_url"}]}],
                        [{"role": "system", "content": "only"}]):
            with self.subTest(messages=invalid), self.assertRaises(RequestError):
                conversation(invalid)

    def test_stop_strings_are_checked_with_core_limits(self):
        self.assertEqual(stops("STOP"), ["STOP"])
        self.assertEqual(stops(None), [])
        for invalid in ("", [""], ["a"] * 5, ["x" * 257], [3], {"a": 1}):
            with self.subTest(stop=invalid), self.assertRaises(RequestError):
                stops(invalid)


class ApiTests(ServeFixture, unittest.TestCase):
    stream = True
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

    def chat(self, content="hello", **options):
        response, data = self.call("POST", "/v1/chat/completions",
                                   {"model": options.pop("model", None) or "ignored-by-default",
                                    "messages": [{"role": "user", "content": content}], **options})
        return response, json.loads(data) if data and not options.get("stream") else data

    def test_models_list_every_sealed_resident_candidate_with_the_default_first_named(self):
        response, data = self.call("GET", "/v1/models")
        self.assertEqual(response.status, 200)
        models = json.loads(data)["data"]
        self.assertEqual([m["id"] for m in models], [self.run_id + "/" + self.a, self.run_id + "/" + self.b])
        self.assertEqual([m["default"] for m in models], [True, False])
        self.assertEqual(models[0]["context_length"], 8192)
        response, data = self.call("GET", "/v1/models/" + self.run_id + "/" + self.b)
        self.assertEqual(json.loads(data)["label"], "Engine B")
        self.assertEqual(json.loads(self.call("GET", "/health")[1])["status"], "ok")

    def test_completion_returns_the_reply_usage_and_measured_telemetry(self):
        response, value = self.chat(max_tokens=50, temperature=0.2)
        self.assertEqual(response.status, 200, value)
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "temperature")
        self.assertEqual(response.getheader("Access-Control-Allow-Origin"), "*")
        reply = json.loads(value["choices"][0]["message"]["content"])
        self.assertEqual(reply["limit"], 50)
        self.assertEqual(value["choices"][0]["finish_reason"], "stop")
        self.assertEqual(value["usage"], {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14})
        self.assertEqual(value["crit_telemetry"]["decode_tps"], 300)
        self.assertEqual(value["model"], self.run_id + "/" + self.a)
        self.assertEqual(list((self.served.runtime.root / "requests").iterdir()), [])
        response, value = self.chat(model=self.b)
        self.assertEqual(value["model"], self.run_id + "/" + self.b)
        self.assertEqual(json.loads(value["choices"][0]["message"]["content"])["calls"], 1)

    def test_length_finish_when_the_output_cap_is_reached(self):
        response, value = self.chat("words:a b c d e", max_tokens=5)
        self.assertEqual(value["choices"][0]["finish_reason"], "length")

    def test_streaming_sends_role_text_finish_usage_and_done(self):
        response, data = self.chat("words:one two three", stream=True, stream_options={"include_usage": True})
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        events = [line[6:] for line in data.decode().split("\n\n") if line.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        self.assertEqual(text, "one two three ")
        self.assertGreater(len([c for c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")]), 1)
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 3)

    def test_stop_sequences_end_the_engine_turn_early_and_keep_the_model_loaded(self):
        response, value = self.chat("words:alpha beta STOP gamma delta epsilon", stop=["STOP"])
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value["choices"][0]["message"]["content"], "alpha beta ")
        self.assertEqual(value["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("usage", value)
        response, data = self.chat("words:alpha STOP beta", stream=True, stop="STOP")
        text = "".join(json.loads(e[6:])["choices"][0]["delta"].get("content", "")
                       for e in data.decode().split("\n\n") if e.startswith("data: {"))
        self.assertEqual(text, "alpha ")
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_unsupported_and_invalid_requests_are_rejected_in_openai_shape(self):
        # Without chat-constraints-v1 the engine cannot apply response_format.
        cases = [({"response_format": {"type": "json_object"}}, 400, "response_format")] * (not self.constraints)
        cases += [({"tools": [{"type": "function"}]}, 400, "tools"),
                 ({"logprobs": True}, 400, "logprobs"), ({"n": 2}, 400, "n"), ({"max_tokens": 9000}, 400, "max_tokens"),
                 ({"model": self.run_id + "/candidate-9999"}, 404, "model"), ({"model": "candidate-9999"}, 404, "model"), ({"stop": ["a", "b", "c", "d", "e"]}, 400, "stop")]
        for options, status, param in cases:
            with self.subTest(options=options):
                response, value = self.chat(**options)
                self.assertEqual(response.status, status, value)
                self.assertEqual(value["error"]["param"], param)
        response, value = self.chat(response_format={"type": "text"})
        self.assertEqual(response.status, 200)
        response, data = self.call("POST", "/v1/completions", {"prompt": "x"})
        self.assertEqual(response.status, 404)
        response, data = self.call("GET", "/v1/unknown")
        self.assertEqual(response.status, 404)

    def test_engine_failures_become_server_errors_and_the_next_request_reloads(self):
        response, value = self.chat("crash")
        self.assertEqual(response.status, 500)
        self.assertEqual(value["error"]["type"], "server_error")
        response, value = self.chat()
        self.assertEqual(response.status, 200)


class StructuredOutputTests(ApiTests):
    """response_format through Core's constraint compiler; the synthetic adapter declares chat-constraints-v1."""
    constraints = True
    schema = {"type": "object", "properties": {"answer": {"type": "string", "maxLength": 8},
              "score": {"type": "integer", "minimum": 0, "maximum": 10}},
              "required": ["answer"], "additionalProperties": False}

    def schema_format(self, schema=None):
        return {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema or self.schema}}

    def test_json_object_mode_reaches_the_engine_and_the_reply_is_checked(self):
        response, value = self.chat(response_format={"type": "json_object"})
        self.assertEqual(response.status, 200, value)
        self.assertEqual(json.loads(value["choices"][0]["message"]["content"])["format"], "json_object")
        response, value = self.chat("reply:not json", response_format={"type": "json_object"})
        self.assertEqual(response.status, 500, value)
        self.assertIn("does not match response_format", value["error"]["message"])
        response, value = self.chat("reply:[1]", max_tokens=4, response_format={"type": "json_object"})
        self.assertEqual((response.status, value["choices"][0]["finish_reason"]), (200, "length"))  # a cut reply

    def test_json_schema_is_compiled_and_replies_follow_its_language(self):
        response, value = self.chat('reply:{"answer":"yes","score":7}', response_format=self.schema_format())
        self.assertEqual(response.status, 200, value)
        self.assertEqual(json.loads(value["choices"][0]["message"]["content"]), {"answer": "yes", "score": 7})
        for reply in ('{"score":7,"answer":"yes"}', '{"answer":"far too long"}', '{"answer": "yes"}', '{"score":7}'):
            with self.subTest(reply=reply):
                response, value = self.chat("reply:" + reply, response_format=self.schema_format())
                self.assertEqual(response.status, 500, value)
        response, data = self.chat('reply:{"answer":"no"}', stream=True, response_format=self.schema_format())
        self.assertIn(b'"finish_reason": "stop"', data)
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_unsupported_schemas_and_formats_are_refused_before_generation(self):
        for value, code in (({"type": "string", "pattern": "a+"}, "unsupported_schema"),
                            ({"type": "object", "properties": {"x": {"type": "string"}}}, "unsupported_schema"),
                            ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "unsupported_schema")):
            with self.subTest(schema=value):
                response, error = self.chat(response_format=self.schema_format(value))
                self.assertEqual((response.status, error["error"]["param"], error["error"]["code"]),
                                 (400, "response_format", code), error)
                self.assertIn("supported JSON schema subset", error["error"]["message"])
        for invalid in ({"type": "regex"}, "json", {"type": "json_schema"}, {"type": "json_schema", "json_schema": {}},
                        {"type": "json_schema", "json_schema": {"name": "bad name", "schema": {"type": "null"}}}):
            with self.subTest(response_format=invalid):
                response, error = self.chat(response_format=invalid)
                self.assertEqual((response.status, error["error"]["param"]), (400, "response_format"), error)


class ApiKeyTests(ApiTests):
    api_key = "secret"

    def call(self, method, path, body=None, headers=None):
        return super().call(method, path, body, {"Authorization": "Bearer secret", **(headers or {})})

    def test_requests_without_the_key_are_refused(self):
        for headers in ({"Authorization": ""}, {"Authorization": "Bearer wrong"}):
            response, data = super().call("GET", "/v1/models", headers=headers)
            self.assertEqual(response.status, 401)
            self.assertEqual(json.loads(data)["error"]["code"], "invalid_api_key")
        response, data = super().call("GET", "/health", headers={"Authorization": ""})
        self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()


class CommandTests(ServeFixture, unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "Ctrl+C reaches the whole Windows console group, including the test runner")
    def test_crit_serve_loads_answers_and_releases_the_runtime_on_interrupt(self):
        import re, signal, subprocess, sys, time
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        process = subprocess.Popen([sys.executable, "-B", str(root / "tests/run_crit.py"), "serve", "--workspace", str(self.store.root),
                                    "--port", "0"], stderr=subprocess.PIPE, text=True, env=os.environ.copy())
        try:
            port = None
            deadline = time.monotonic() + 30
            while port is None and time.monotonic() < deadline:
                line = process.stderr.readline()
                found = re.search(r"http://127\.0\.0\.1:(\d+)/v1", line)
                port = int(found.group(1)) if found else None
                self.assertNotIn("error", line)
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            connection.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
                               {"Content-Type": "application/json"})
            value = json.loads(connection.getresponse().read())
            connection.close()
            self.assertEqual(json.loads(value["choices"][0]["message"]["content"])["calls"], 1)
            process.send_signal(signal.SIGINT)
            self.assertEqual(process.wait(30), 0)
            self.assertFalse((self.root / "runtime-owner" / "owner.json").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(10)
            process.stderr.close()

    def test_launchers_repeat_the_command_without_the_api_key(self):
        import contextlib, io
        from crit_api.cli import main
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--workspace", str(self.store.root), "--api-key", "secret", "--port", "8123", "--write-launchers"]), 0)
        for name in ("serve.cmd", "serve.sh"):
            script = (self.directory / name).read_text(encoding="utf-8")
            self.assertIn("serve", script)
            self.assertIn("8123", script)
            self.assertIn(self.run_id, script)
            self.assertNotIn("secret", script)
