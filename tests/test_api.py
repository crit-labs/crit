"""crit serve over HTTP with a synthetic resident adapter; no model, GPU or native build."""
import contextlib
import http.client
import io
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

from test_chat_residency import ServeFixture
from crit_api.disconnect import ClientDisconnect
from crit_api.engine import reply as engine_reply
from crit_api.server import Api, RequestError, Server, continuation, conversation, stops, structured_outputs
from crit_core.constraints import matches


def completion_request(capabilities, **body):
    """Api.completion_request over an engine whose package declares `capabilities`."""
    engine = mock.Mock(default="a")
    engine.model_id.return_value = "run/a"
    engine.spec.return_value = {"capabilities": capabilities, "limits": {"kv_tokens": 8192}}
    return Api(engine).completion_request({"messages": [{"role": "user", "content": "x"}], **body})


def certificate(folder):
    """(certificate, key) PEM files of a self-signed certificate for 127.0.0.1, made with the openssl command."""
    cert, key = folder / "cert.pem", folder / "key.pem"
    try:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
                        "-days", "1", "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
                        "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise unittest.SkipTest("a test certificate needs the openssl command: %s" % exc)
    return cert, key


class ConversationTests(unittest.TestCase):
    def test_system_first_then_alternating_turns_ending_with_the_user(self):
        messages = conversation([{"role": "developer", "content": "Be brief."},
                                 {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                                 {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"},
                                 {"role": "user", "content": "e"}])
        self.assertEqual(messages, [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "ab\n\nc"},
                                    {"role": "assistant", "content": "d"}, {"role": "user", "content": "e"}])
        for invalid in ([], [{"role": "assistant", "content": "x"}], [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                        [{"role": "tool", "content": "x"}], [{"role": "user", "content": [{"type": "image_url"}]}],
                        [{"role": "system", "content": "only"}]):
            with self.subTest(messages=invalid), self.assertRaises(RequestError):
                conversation(invalid)

    def test_a_later_system_or_developer_message_becomes_marked_user_text(self):
        user, assistant = {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}
        late = [{"role": "system", "content": "One step left."}, {"role": "developer", "content": "Be terse."}]
        self.assertEqual(conversation([{"role": "system", "content": "Be brief."}, user, assistant, late[1],
                                       {"role": "user", "content": "c"}]),
                         [{"role": "system", "content": "Be brief."}, user, assistant,
                          {"role": "user", "content": "System: Be terse.\n\nc"}])
        self.assertEqual(conversation([user, late[0], assistant, {"role": "user", "content": "c"}])[0],
                         {"role": "user", "content": "a\n\nSystem: One step left."})
        self.assertEqual(conversation([user, assistant] + late),
                         [user, assistant, {"role": "user", "content": "System: One step left.\n\nSystem: Be terse."}])
        image = {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "auto"}}
        self.assertEqual(conversation([{"role": "user", "content": [image]}, late[0]], ["chat-images-v1"]),
                         [{"role": "user", "content": [image, {"type": "text", "text": "\n\n"},
                                                       {"type": "text", "text": "System: One step left."}]}])
        # With tool history the marked text opens the next user message, else ends the one before, else is a turn.
        call = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        history = [{"role": "user", "content": "Read a.py"}, {"role": "assistant", "content": None, "tool_calls": [call]},
                   {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"}]
        tools = ["chat-tools-v1"]
        self.assertEqual(conversation(history + late + [{"role": "user", "content": "c", "name": "yan"}], tools)[3:],
                         [{"role": "user", "content": "System: One step left.\n\nSystem: Be terse.\n\nc", "name": "yan"}])
        self.assertEqual(conversation(history + late[:1], tools)[3:], [{"role": "user", "content": "System: One step left."}])
        self.assertEqual(conversation(history[:1] + late[:1] + history[1:], tools)[0],
                         {"role": "user", "content": "Read a.py\n\nSystem: One step left."})
        with self.assertRaises(RequestError):  # never between a call and its result
            conversation(history[:2] + late[:1] + history[2:], tools)

    def test_a_continued_conversation_ends_with_the_assistant_message_to_continue(self):
        turns = [{"role": "user", "content": "Write"}, {"role": "assistant", "content": "Once"},
                 {"role": "assistant", "content": "upon"}]
        self.assertEqual(conversation(turns, continuing=True), [{"role": "user", "content": "Write"},
                                                                {"role": "assistant", "content": "Once\n\nupon"}])
        call = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        history = [turns[0], {"role": "assistant", "content": None, "tool_calls": [call]},
                   {"role": "tool", "tool_call_id": "call_1", "content": "done"}, turns[1]]
        self.assertEqual(conversation(history, ["chat-tools-v1"], continuing=True)[-1], turns[1])
        for invalid in (turns[:1], turns[1:], history[:3], history[:2]):
            with self.subTest(messages=invalid), self.assertRaises(RequestError):
                conversation(invalid, ["chat-tools-v1"], continuing=True)
        with self.assertRaises(RequestError):
            conversation(history, ["chat-tools-v1"])
        flags = {"continue_final_message": True, "add_generation_prompt": False}
        self.assertTrue(continuation(flags, ["chat-continue-v1"]))
        for defaults in ({}, {"continue_final_message": False, "add_generation_prompt": True},
                         {"continue_final_message": None, "add_generation_prompt": None}):
            self.assertFalse(continuation(defaults, ()))
        for invalid, param in (({"continue_final_message": True}, "add_generation_prompt"),
                               ({"add_generation_prompt": False}, "add_generation_prompt"),
                               ({"continue_final_message": 1, "add_generation_prompt": False}, "continue_final_message")):
            with self.subTest(body=invalid), self.assertRaises(RequestError) as raised:
                continuation(invalid, ["chat-continue-v1"])
            self.assertEqual(raised.exception.param, param)
        with self.assertRaises(RequestError) as raised:
            continuation(flags, ["chat-tools-v1"])
        self.assertEqual((raised.exception.param, raised.exception.code), ("continue_final_message", "unsupported_parameter"))

    def test_stop_strings_are_checked_with_core_limits(self):
        self.assertEqual(stops("STOP"), ["STOP"])
        self.assertEqual(stops(None), [])
        self.assertEqual(stops(["a"] * 16), ["a"] * 16)
        for invalid in ("", [""], ["a"] * 17, ["x" * 257], [3], {"a": 1}):
            with self.subTest(stop=invalid), self.assertRaises(RequestError):
                stops(invalid)

    def test_a_kept_stop_ends_a_reply_the_engine_finished(self):
        engine = mock.Mock(default="c")
        engine.spec.return_value = {"capabilities": []}
        engine.generate.return_value = ("completed", "alpha STOP beta", None)
        request = {"candidate": "c", "messages": [], "max_tokens": 8, "stream": False, "stop": ["STOP"], "sampling": {},
                   "tool_options": {}}
        for include, text in ((False, "alpha "), (True, "alpha STOP")):
            pieces = []
            finish, _, _, _, _, stopped = Api(engine).run(dict(request, include_stop=include), pieces.append,
                                                          threading.Event())
            self.assertEqual(("".join(pieces), finish, stopped), (text, "stop", "STOP"))

    def test_structured_outputs_compile_one_regex_choice_grammar_or_schema_with_core(self):
        regex, param = structured_outputs({"structured_outputs": {"regex": "^[0-9]{3}$", "json": None}})
        self.assertEqual((param, regex["type"], regex["language"], regex["source"]),
                         ("structured_outputs.regex", "grammar", "regex", "^[0-9]{3}$"))
        self.assertEqual([matches(regex["grammar"], text) for text in (b"042", b"42", b"0420")], [True, False, False])
        choices = ["yes", "no", 'say "hi"\n', "na\u00efve", "a|b # c"]
        choice, param = structured_outputs({"guided_choice": choices})
        self.assertEqual((param, choice["language"], choice["source"]), ("guided_choice", "choice", choices))
        self.assertTrue(all(matches(choice["grammar"], text.encode()) for text in choices))
        self.assertFalse(any(matches(choice["grammar"], text) for text in (b"ye", b"yesno", b"a")))
        grammar, _ = structured_outputs({"structured_outputs": {"grammar": 'root ::= "id-" digit+  # an id\n'
                                                                           "digit ::= [0-9]"}})
        self.assertEqual(grammar["language"], "ebnf")
        self.assertEqual([matches(grammar["grammar"], text) for text in (b"id-42", b"id-")], [True, False])
        for source in ({"type": "null"}, '{"type": "null"}'):
            schema, param = structured_outputs({"structured_outputs": {"json": source}})
            self.assertEqual((param, schema["type"], schema["name"], schema["schema"]),
                             ("structured_outputs.json", "json_schema", "response", {"type": "null"}))
            self.assertTrue(matches(schema["grammar"], b"null"))
        self.assertEqual(structured_outputs({"structured_outputs": None, "guided_regex": None}), (None, None))

    def test_structured_outputs_take_exactly_one_constraint_and_refuse_what_core_cannot_compile(self):
        cases = [({"structured_outputs": "regex"}, "structured_outputs", None, "must be an object"),
                 ({"structured_outputs": {}}, "structured_outputs", None, "needs one of"),
                 ({"structured_outputs": {"regex": None}}, "structured_outputs", None, "needs one of"),
                 ({"structured_outputs": {"json_object": True}}, "structured_outputs.json_object",
                  "unsupported_parameter", "not supported"),
                 ({"structured_outputs": {"regex": "a", "choice": ["a"]}}, "structured_outputs.choice", None,
                  "one constraint"),
                 ({"structured_outputs": {"regex": "a"}, "guided_regex": "a"}, "structured_outputs.regex", None,
                  "one constraint"),
                 ({"structured_outputs": {"regex": ""}}, "structured_outputs.regex", None, "non-empty string"),
                 ({"structured_outputs": {"choice": []}}, "structured_outputs.choice", None, "non-empty list"),
                 ({"guided_choice": ["a", ""]}, "guided_choice", None, "non-empty list"),
                 ({"structured_outputs": {"json": "[1]"}}, "structured_outputs.json", None, "JSON schema object"),
                 # The compiler's own reasons.
                 ({"structured_outputs": {"regex": "(a"}}, "structured_outputs.regex", "unsupported_constraint",
                  "cannot be compiled (docs/SERVING.md#structured-output): unclosed group"),
                 ({"guided_regex": "a(?=b)"}, "guided_regex", "unsupported_constraint", "unsupported regex operator"),
                 ({"structured_outputs": {"grammar": 'item ::= "a"'}}, "structured_outputs.grammar",
                  "unsupported_constraint", "root rule required"),
                 ({"structured_outputs": {"grammar": "root ::= (\n  \"a\")"}}, "structured_outputs.grammar",
                  "unsupported_constraint", "one name ::= expression per line required"),
                 ({"structured_outputs": {"choice": ["\ud800"]}}, "structured_outputs.choice",
                  "unsupported_constraint", "invalid or excessive constraint input"),
                 ({"structured_outputs": {"json": {"type": "string", "format": "email"}}}, "structured_outputs.json",
                  "unsupported_schema", "outside the supported JSON schema subset")]
        for body, param, code, words in cases:
            with self.subTest(body=body), self.assertRaises(RequestError) as caught:
                structured_outputs(body)
            self.assertEqual((caught.exception.status, caught.exception.param, caught.exception.code),
                             (400, param, code))
            self.assertIn(words, str(caught.exception))

    def test_a_grammar_needs_its_capability_and_no_other_constraint_callable_tools_or_continuation(self):
        regex = {"structured_outputs": {"regex": "[0-9]+"}}
        constraints, grammars = ["chat-constraints-v1"], ["chat-constraints-v1", "chat-constraints-grammar-v1"]
        for capabilities, missing in ((constraints, "chat-constraints-grammar-v1"),
                                      (["chat-constraints-grammar-v1"], "chat-constraints-v1")):
            with self.subTest(capabilities=capabilities), self.assertRaises(RequestError) as caught:
                completion_request(capabilities, **regex)
            self.assertEqual((caught.exception.param, caught.exception.code),
                             ("structured_outputs.regex", "unsupported_parameter"))
            self.assertIn("cannot apply structured_outputs.regex: its package does not declare " + missing,
                          str(caught.exception))
        self.assertEqual(completion_request(constraints, guided_json={"type": "null"})["response_format"]["type"],
                         "json_schema")
        self.assertEqual(completion_request(grammars, **regex)["response_format"]["language"], "regex")
        # response_format text constrains nothing.
        request = completion_request(grammars, response_format={"type": "text"}, **regex)
        self.assertEqual(request["response_format"]["type"], "grammar")
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        tooled = grammars + ["chat-tools-v1", "chat-continue-v1"]
        # With tools a constraint applies only when no call can be made, as response_format's does.
        request = completion_request(tooled, tools=tools, tool_choice="none", **regex)
        self.assertEqual(request["response_format"]["language"], "regex")
        prefill = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "4"}]
        for body in ({"response_format": {"type": "json_object"}, **regex}, {"tools": tools, **regex},
                     {"tools": tools, "tool_choice": "auto", **regex},
                     {"messages": prefill, "continue_final_message": True, "add_generation_prompt": False, **regex}):
            with self.subTest(body=body), self.assertRaises(RequestError) as caught:
                completion_request(tooled, **body)
            self.assertEqual(caught.exception.param, "structured_outputs.regex")


class ApiTests(ServeFixture, unittest.TestCase):
    stream = True
    api_keys = ()

    def setUp(self):
        super().setUp()
        self.served = self.engine()
        self.server = Server(("127.0.0.1", 0), Api(self.served, api_keys=self.api_keys, cors_origin="*"))
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
        # The names gateways show and LiteLLM's context field.
        self.assertEqual([(m["display_name"], m["max_model_len"]) for m in models],
                         [("Fixture engine", 8192), ("Engine B", 8192)])
        response, data = self.call("GET", "/v1/models/" + self.run_id + "/" + self.b)
        self.assertEqual(json.loads(data)["label"], "Engine B")
        self.assertEqual(json.loads(self.call("GET", "/health")[1])["status"], "ok")

    def test_completion_returns_the_reply_usage_and_measured_telemetry(self):
        response, value = self.chat(max_tokens=50, user="fixture")
        self.assertEqual(response.status, 200, value)
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "user")
        self.assertEqual(response.getheader("Access-Control-Allow-Origin"), "*")
        reply = json.loads(value["choices"][0]["message"]["content"])
        # A cap below the context, as Eval sends, reaches the engine as asked, with chat-fit-output-v1 or without.
        self.assertEqual((reply["limit"], reply["fit_output"]), (50, None))
        self.assertIsNone(response.getheader("X-Crit-Max-Tokens-Applied"))
        self.assertEqual(value["choices"][0]["finish_reason"], "stop")
        # The engine computed every prompt token: none came from a prompt cache.
        self.assertEqual(value["usage"], {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                                          "prompt_tokens_details": {"cached_tokens": 0}})
        self.assertEqual(value["crit_telemetry"]["decode_tps"], 300)
        # The engine's own phase counts and times, as it reported them.
        self.assertEqual([value["crit_telemetry"][k] for k in ("prefill_tokens", "prefill_ms", "decode_tokens", "decode_ms")],
                         [10, 5, 3, 10])
        self.assertNotIn("tokens_per_round", value["crit_telemetry"])  # the engine reported no decode rounds
        self.assertEqual(value["model"], self.run_id + "/" + self.a)
        self.assertFalse((self.served.runtime.root / "requests").exists())
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
        self.assertEqual(chunks[-1]["crit_telemetry"]["tokens_per_round"], 3)  # 3 tokens over one decode round
        self.assertEqual([chunks[-1]["crit_telemetry"][k] for k in ("prefill_tokens", "decode_tokens")], [10, 2])
        self.assertNotIn("crit_telemetry", chunks[-2])

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

    def test_include_stop_str_in_output_keeps_the_stop_that_ended_the_reply(self):
        response, value = self.chat("words:alpha beta STOP gamma delta", stop=["END", "STOP"],
                                    include_stop_str_in_output=True)
        self.assertEqual(response.status, 200, value)
        self.assertEqual((value["choices"][0]["message"]["content"], value["choices"][0]["finish_reason"]),
                         ("alpha beta STOP", "stop"))
        response, data = self.chat("words:alpha STOP beta", stream=True, stop="STOP", include_stop_str_in_output=True)
        text = "".join(json.loads(e[6:])["choices"][0]["delta"].get("content", "")
                       for e in data.decode().split("\n\n") if e.startswith("data: {"))
        self.assertEqual(text, "alpha STOP")
        response, value = self.chat("words:alpha beta", include_stop_str_in_output=True)
        self.assertEqual(value["choices"][0]["message"]["content"], "alpha beta ")
        for invalid in ("true", 1, None):
            with self.subTest(include_stop_str_in_output=invalid):
                response, value = self.chat(include_stop_str_in_output=invalid)
                self.assertEqual((response.status, value["error"]["param"]), (400, "include_stop_str_in_output"))

    def test_a_request_timeout_ends_its_turn_sooner_within_the_servers(self):
        pid = self.served.host_pid()
        started = time.monotonic()
        response, value = self.chat("slow", crit={"timeout": .3})
        self.assertEqual(response.status, 504, value)
        self.assertLess(time.monotonic() - started, 4)  # the turn would take 5 s
        self.assertEqual(value["error"]["type"], "server_error")
        # The adapter sees the request's limit; one beyond --timeout keeps --timeout.
        for timeout, seen in ((30, 30), (2.5, 2.5), (10 ** 6, 600), (10 ** 400, 600)):
            response, value = self.chat("echo-request", crit={"timeout": timeout})
            self.assertEqual(response.status, 200, value)
            self.assertEqual(json.loads(value["choices"][0]["message"]["content"])["limits"]["timeout_seconds"], seen)
        self.assertEqual((self.served.status()["engine_loads"], self.served.host_pid()), (1, pid))
        for invalid in (0, -1, "5", True, float("nan"), [1]):
            with self.subTest(timeout=invalid):
                response, value = self.chat(crit={"timeout": invalid})
                self.assertEqual((response.status, value["error"]["param"]), (400, "crit.timeout"))

    def test_unsupported_and_invalid_requests_are_rejected_in_openai_shape(self):
        # Without chat-constraints-v1 the engine cannot apply response_format.
        cases = [({"response_format": {"type": "json_object"}}, 400, "response_format"),
                 ({"structured_outputs": {"regex": "a"}}, 400, "structured_outputs.regex")] * (not self.constraints)
        cases += [({"tools": [{"type": "function"}]}, 400, "tools"),
                 ({"logprobs": True}, 400, "logprobs"), ({"n": 2}, 400, "n"), ({"max_tokens": 0}, 400, "max_tokens"),
                 ({"model": self.run_id + "/candidate-9999"}, 404, "model"), ({"model": "candidate-9999"}, 404, "model"), ({"stop": ["a"] * 17}, 400, "stop"),
                 ({"stream_options": {"include_usage": True, "continuous_usage_stats": True}}, 400,
                  "stream_options.continuous_usage_stats"),
                 ({"priority": 1.5}, 400, "priority"), ({"priority": True}, 400, "priority")]
        # Without chat-fit-output-v1 a cap must leave the prompt room in the context.
        cases += [({"max_tokens": 8192}, 400, "max_tokens")] * (not self.fit_output)
        for options, status, param in cases:
            with self.subTest(options=options):
                response, value = self.chat(**options)
                self.assertEqual(response.status, status, value)
                self.assertEqual(value["error"]["param"], param)
        response, value = self.chat(response_format={"type": "text"})
        self.assertEqual(response.status, 200)
        response, data = self.call("POST", "/v1/completions", {"prompt": "x"})
        self.assertEqual((response.status, json.loads(data)["error"]["code"]), (404, "not_supported"))
        response, data = self.call("GET", "/v1/unknown")
        self.assertEqual(response.status, 404)

    def test_a_null_n_keeps_the_one_choice_like_other_null_fields(self):
        for n in (None, 1):
            with self.subTest(n=n):
                response, value = self.chat(n=n)
                self.assertEqual(response.status, 200, value)
                self.assertEqual(len(value["choices"]), 1)

    def test_priority_reaches_admission_and_defaults_to_zero(self):
        with mock.patch.object(self.served.lock, "admit", wraps=self.served.lock.admit) as admit:
            for options in ({"priority": -5}, {"priority": None}, {}):
                response, value = self.chat(**options)
                self.assertEqual(response.status, 200, value)
                self.assertIsNone(response.getheader("X-Crit-Ignored-Parameters"))
        self.assertEqual([call.args[2] for call in admit.call_args_list], [-5, 0, 0])

    def test_endpoints_clients_look_for_beside_chat_are_404_not_supported_naming_the_endpoint(self):
        response, data = self.call("POST", "/v1/embeddings", {"model": "x", "input": "hello"})
        self.assertEqual((response.status, json.loads(data)), (404, {"error": {
            "message": "/v1/embeddings is not available on a crit engine, which serves chat; run a separate embedding "
                       "server for embeddings and reranking", "type": "invalid_request_error", "param": None,
            "code": "not_supported"}, "request_id": response.getheader("X-Request-Id")}))
        retrieval = [("POST", "/v1/rerank"), ("POST", "/v2/rerank"), ("POST", "/rerank"), ("POST", "/v1/score")]
        others = [("POST", "/v1/audio/transcriptions"), ("POST", "/v1/images/generations"), ("POST", "/v1/moderations"),
                  ("GET", "/v1/files"), ("DELETE", "/v1/files/file-1"), ("POST", "/v1/batches"), ("GET", "/v1/batches/b1")]
        for method, path in retrieval + others:
            with self.subTest(method=method, path=path):
                response, data = self.call(method, path, {"input": "x"} if method == "POST" else None)
                error = json.loads(data)["error"]
                self.assertEqual((response.status, error["code"]), (404, "not_supported"))
                self.assertTrue(error["message"].startswith(path + " is not available on a crit engine"))
                self.assertEqual("separate embedding server" in error["message"], (method, path) in retrieval)
        response, data = self.call("HEAD", "/v1/embeddings")
        self.assertEqual((response.status, data), (404, b""))
        response, data = self.call("POST", "/v1/embeddingsx", {"input": "x"})
        self.assertEqual((response.status, json.loads(data)["error"]["code"]), (404, "not_found"))

    def test_a_context_overflow_is_a_400_in_openai_words_and_keeps_the_engine_loaded(self):
        pid = self.served.host_pid()
        message = ("This model's maximum context length is 8192 tokens. However, you requested 9100 tokens (9000 in "
                   "the messages, 100 in the completion). Please reduce the length of the messages or completion.")
        error = {"message": message, "type": "invalid_request_error", "param": "messages",
                 "code": "context_length_exceeded"}
        response, value = self.chat("overflow:9000", max_tokens=100)
        ident = response.getheader("X-Request-Id")
        self.assertEqual((response.status, value), (400, {"error": error, "request_id": ident}))
        # A stream has sent its headers before the engine sees the prompt: the error is its event.
        response, data = self.chat("overflow:9000", max_tokens=100, stream=True)
        events = [line[6:] for line in data.decode().split("\n\n") if line.startswith("data: ")]
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(events[-2]), {"error": error, "request_id": response.getheader("X-Request-Id")})
        self.assertEqual(events[-1], "[DONE]")
        response, value = self.chat()
        self.assertEqual(response.status, 200, value)
        self.assertEqual(self.served.status()["engine_loads"], 1)  # no second load
        self.assertEqual(self.served.host_pid(), pid)
        self.assertFalse((self.served.runtime.root / "requests").exists())

    def test_a_later_developer_message_reaches_the_engine_as_marked_user_text(self):
        messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"},
                    {"role": "developer", "content": "Answer in French."}, {"role": "assistant", "content": "Bonjour"},
                    {"role": "user", "content": "echo-request"}]
        response, data = self.call("POST", "/v1/chat/completions", {"model": "default", "messages": messages})
        self.assertEqual(response.status, 200, data)
        echoed = json.loads(json.loads(data)["choices"][0]["message"]["content"])
        self.assertEqual(echoed["messages"], [{"role": "system", "content": "Be brief."},
                                              {"role": "user", "content": "hi\n\nSystem: Answer in French."},
                                              {"role": "assistant", "content": "Bonjour"},
                                              {"role": "user", "content": "echo-request"}])

    def test_engine_failures_become_server_errors_and_the_next_request_reloads(self):
        response, value = self.chat("crash")
        self.assertEqual(response.status, 500)
        self.assertEqual(value["error"]["type"], "server_error")
        response, value = self.chat()
        self.assertEqual(response.status, 200)

    def test_every_response_carries_its_request_id_which_names_a_failed_requests_records(self):
        lines = []
        self.server.api.log = lines.append
        response, _ = self.call("GET", "/health")
        self.assertRegex(response.getheader("X-Request-Id"), "^[0-9a-f]{32}$")
        self.assertEqual(response.getheader("Access-Control-Expose-Headers"), "X-Request-Id")
        response, value = self.chat(n=2)
        self.assertEqual((response.status, value["request_id"]), (400, response.getheader("X-Request-Id")))
        root = self.served.runtime.root
        # The client's own id is logged next to crit's, which alone names the failed request's records.
        response, data = self.call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "crash"}]},
                                   {"X-Request-Id": "../client-1"})
        ident = json.loads(data)["request_id"]
        self.assertEqual((response.status, response.getheader("X-Request-Id")), (500, ident))
        self.assertEqual([path.name for path in (root / "requests").iterdir()], [ident])
        self.assertEqual(json.loads((root / "requests" / ident / "request.json").read_text())["request_id"], ident)
        self.assertTrue(any(f"request_id={ident} client_request_id=../client-1" in line for line in lines), lines)

    def test_abort_answers_whether_it_found_a_request_and_refuses_other_bodies(self):
        for body in ({"request_id": "nothing-here"}, {"all": True}):
            response, data = self.call("POST", "/v1/crit/abort", body)
            self.assertEqual((response.status, json.loads(data)),
                             (200, {"object": "crit.abort", "found": False, "request_ids": []}))
        for body, param in (({}, None), ({"all": False}, "all"), ({"request_id": ""}, "request_id"),
                            ({"request_id": 7}, "request_id"), ({"request_id": "x", "all": True}, None), (["x"], None)):
            with self.subTest(body=body):
                response, data = self.call("POST", "/v1/crit/abort", body)
                self.assertEqual((response.status, json.loads(data)["error"]["param"]), (400, param))


class StructuredOutputTests(ApiTests):
    """response_format and structured_outputs through Core's constraint compiler; the synthetic adapter declares
    chat-constraints-v1 and chat-constraints-grammar-v1."""
    constraints = True
    extra_capabilities = ("chat-constraints-grammar-v1",)
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
        for value, code in (({"type": "string", "format": "email"}, "unsupported_schema"),
                            ({"type": "object", "properties": {"x": {"type": "string"}}}, "unsupported_schema"),
                            ({"oneOf": [{"type": "string"}, {"type": "string", "maxLength": 2}]}, "unsupported_schema")):
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

    def test_structured_outputs_constrain_the_reply_to_a_regex_choice_grammar_or_schema(self):
        cases = (({"regex": "[0-9]{3}"}, "042", "42x"), ({"choice": ["yes", "no"]}, "no", "maybe"),
                 ({"grammar": 'root ::= "id-" digit+\ndigit ::= [0-9]'}, "id-7", "id-"),
                 ({"json": self.schema}, '{"answer":"yes"}', '{"answer": "yes"}'))
        for outputs, reply, other in cases:
            with self.subTest(structured_outputs=outputs):
                response, value = self.chat("reply:" + reply, structured_outputs=outputs)
                self.assertEqual((response.status, value["choices"][0]["message"]["content"]), (200, reply), value)
                response, value = self.chat("reply:" + other, structured_outputs=outputs)
                self.assertEqual(response.status, 500, value)
                self.assertIn("does not match response_format", value["error"]["message"])
        # The engine receives a grammar; vLLM's earlier guided_* fields name the same constraints.
        response, value = self.chat(structured_outputs={"regex": "\\{.*\\}"})
        self.assertEqual(json.loads(value["choices"][0]["message"]["content"])["format"], "grammar")
        response, value = self.chat("reply:yes", guided_choice=["yes", "no"])
        self.assertEqual((response.status, value["choices"][0]["message"]["content"]), (200, "yes"), value)
        response, data = self.chat("reply:no", stream=True, structured_outputs={"choice": ["yes", "no"]})
        self.assertIn(b'"finish_reason": "stop"', data)
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_structured_outputs_errors_are_400s_before_generation(self):
        cases = (({"structured_outputs": {"regex": "[a-"}}, "structured_outputs.regex", "unsupported_constraint"),
                 ({"guided_grammar": "root ::= digit"}, "guided_grammar", "unsupported_constraint"),
                 ({"structured_outputs": {"choice": ["a"]}, "response_format": {"type": "json_object"}},
                  "structured_outputs.choice", None),
                 ({"structured_outputs": {"whitespace_pattern": " "}}, "structured_outputs.whitespace_pattern",
                  "unsupported_parameter"))
        for options, param, code in cases:
            with self.subTest(options=options):
                response, error = self.chat(**options)
                self.assertEqual((response.status, error["error"]["param"], error["error"]["code"]), (400, param, code),
                                 error)
        response, error = self.chat(structured_outputs={"grammar": "root ::= digit"})
        self.assertIn("cannot be compiled (docs/SERVING.md#structured-output): unknown rule", error["error"]["message"])


class FitOutputTests(ApiTests):
    """A package declaring chat-fit-output-v1: a cap at or above the context is fitted to the room the prompt leaves.
    Every ApiTests case runs here too, so caps below the context behave as without it."""
    fit_output = True

    def test_a_cap_at_or_above_the_context_is_fitted_and_the_applied_cap_reported(self):
        for options in ({"max_tokens": 8192}, {"max_completion_tokens": 32000}, {"max_tokens": 2 ** 40}):
            with self.subTest(options=options):
                response, value = self.chat(**options)
                self.assertEqual(response.status, 200, value)
                # The engine is asked for the most the context holds and fits it to its prompt of 10 tokens.
                reply = json.loads(value["choices"][0]["message"]["content"])
                self.assertEqual((reply["limit"], reply["fit_output"]), (8191, True))
                self.assertEqual(response.getheader("X-Crit-Max-Tokens-Applied"), "8182")
                self.assertEqual(value["choices"][0]["finish_reason"], "stop")
        response, value = self.chat("overflow:8150", max_tokens=32000)
        self.assertEqual((value["choices"][0]["finish_reason"], value["usage"]["completion_tokens"]), ("length", 42))
        self.assertEqual(response.getheader("X-Crit-Max-Tokens-Applied"), "42")
        # A stream sends its headers before the engine counts the prompt; its finish says where the context ended it.
        response, data = self.chat("overflow:8150", max_tokens=32000, stream=True)
        chunks = [json.loads(line[6:]) for line in data.decode().split("\n\n") if line.startswith("data: {")]
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertIsNone(response.getheader("X-Crit-Max-Tokens-Applied"))
        # A prompt that fills the context alone is still a 400, and the engine stays loaded.
        response, value = self.chat("overflow:9000", max_tokens=32000)
        self.assertEqual((response.status, value["error"]["code"]), (400, "context_length_exceeded"))
        for invalid in (0, -1, 1.5, True, "8192"):
            with self.subTest(max_tokens=invalid):
                response, value = self.chat(max_tokens=invalid)
                self.assertEqual((response.status, value["error"]["param"]), (400, "max_tokens"))
        self.assertEqual(self.served.status()["engine_loads"], 1)


class FittedReplyTests(unittest.TestCase):
    """The cap an engine applied comes back exactly for a fitted request, within the request and its context."""
    request = {"request_id": "r", "candidate_id": "c", "package_sha256": "p",
               "limits": {"max_output_tokens": 63, "kv_tokens": 64, "timeout_seconds": 60, "fit_output": True}}
    value = {"schema_version": 1, "record_kind": "chat_response", "request_id": "r", "candidate_id": "c",
             "package_sha256": "p", "assistant": "hi", "max_output_tokens": 54,
             "telemetry": {"prompt_tokens": 10, "output_tokens": 4, "prefill_tokens": 10, "prefill_ms": 1,
                           "decode_tokens": 3, "decode_ms": 1, "ttft_ms": 1, "missing": {}}}

    def test_the_applied_cap_is_checked_and_returned_with_the_telemetry(self):
        self.assertEqual(engine_reply(self.value, self.request)[1]["max_output_tokens"], 54)
        for cap in (3, 55, 64, "54"):  # below the output, past the context, past the request, not a count
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                engine_reply(dict(self.value, max_output_tokens=cap), self.request)
        unfitted = dict(self.request, limits={k: v for k, v in self.request["limits"].items() if k != "fit_output"})
        with self.assertRaisesRegex(ValueError, "does not match the request"):
            engine_reply(self.value, unfitted)
        with self.assertRaisesRegex(ValueError, "does not match the request"):
            engine_reply({k: v for k, v in self.value.items() if k != "max_output_tokens"}, self.request)
        self.assertNotIn("max_output_tokens", engine_reply({k: v for k, v in self.value.items()
                                                            if k != "max_output_tokens"}, unfitted)[1])


class ApiKeyTests(ApiTests):
    api_keys = ("spare", "secret")

    def call(self, method, path, body=None, headers=None):
        return super().call(method, path, body, {"Authorization": "Bearer secret", **(headers or {})})

    def test_requests_without_the_key_are_refused(self):
        for headers in ({"Authorization": ""}, {"Authorization": "Bearer wrong"}):
            response, data = super().call("GET", "/v1/models", headers=headers)
            self.assertEqual(response.status, 401)
            self.assertEqual(json.loads(data)["error"]["code"], "invalid_api_key")
        response, data = super().call("POST", "/v1/crit/abort", {"all": True}, headers={"Authorization": ""})
        self.assertEqual(response.status, 401)
        response, data = super().call("GET", "/health", headers={"Authorization": ""})
        self.assertEqual(response.status, 200)

    def test_every_key_is_accepted_whole_in_either_header(self):
        for key, status in (("spare", 200), ("secret", 200), ("secre", 401), ("secrets", 401), ("spare,secret", 401)):
            with self.subTest(key=key):
                response, _ = super().call("GET", "/v1/models", headers={"Authorization": "Bearer " + key})
                self.assertEqual(response.status, status)
                response, _ = super().call("GET", "/api/tags", headers={"x-api-key": key})
                self.assertEqual(response.status, status)


class KeySourceTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        self.key_file = self.folder / "keys.txt"

    def keys(self, argv, environment=None):
        from crit_api.cli import api_keys, parser
        clean = {k: v for k, v in os.environ.items() if k != "CRIT_API_KEY"}
        with mock.patch.dict(os.environ, {**clean, **(environment or {})}, clear=True):
            return api_keys(parser().parse_args(["--workspace", str(self.folder), *argv]))

    def test_keys_are_each_flag_else_the_environments_list_and_the_key_files_lines(self):
        self.key_file.write_text("\ufeffalpha\n# the build machine\n\n  beta  \r\n#gamma\n", encoding="utf-8")
        self.assertEqual(self.keys(["--api-key", "one", "--api-key", "two"], {"CRIT_API_KEY": "env"}), ["one", "two"])
        self.assertEqual(self.keys([], {"CRIT_API_KEY": " env1, ,env2 "}), ["env1", "env2"])
        self.assertEqual(self.keys([]), [])
        self.assertEqual(self.keys(["--api-key-file", str(self.key_file)], {"CRIT_API_KEY": "env"}), ["env", "alpha", "beta"])
        self.key_file.write_text("# none yet\n\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "holds no API key"):
            self.keys(["--api-key-file", str(self.key_file)])

    def test_a_missing_key_file_or_an_unloadable_certificate_stops_crit_serve_before_loading(self):
        from crit_api.cli import main
        self.key_file.write_text("not a certificate\n", encoding="utf-8")
        workspace = ["--workspace", str(self.folder / "missing-workspace")]
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main([*workspace, "--api-key-file", str(self.folder / "missing.txt")]), 2)
            self.assertEqual(main([*workspace, "--ssl-certfile", str(self.key_file)]), 2)
            with self.assertRaises(SystemExit):
                main([*workspace, "--ssl-keyfile", str(self.key_file)])
        self.assertIn("missing.txt", errors.getvalue())
        self.assertIn("cannot load the TLS certificate from " + str(self.key_file), errors.getvalue())
        self.assertIn("--ssl-keyfile needs --ssl-certfile", errors.getvalue())


class SmallBufferServer(Server):
    def get_request(self):
        request, address = super().get_request()
        request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        return request, address


class LateExit(ClientDisconnect):
    """A request scope that exits late, as on a busy server, so the client's next keep-alive request reaches the
    watcher first."""
    def __exit__(self, kind, value, traceback):
        time.sleep(.2)
        return super().__exit__(kind, value, traceback)


def chat_request(content):
    """The bytes of a chat request for `content`, as a client writes them to the connection."""
    body = json.dumps({"messages": [{"role": "user", "content": content}]}).encode()
    return (b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n"
            b"Content-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))


def answer(reader):
    """(status, JSON body) of the next response read from a connection, for requests sent ahead of it."""
    status = int(reader.readline().split()[1])
    length = int(http.client.parse_headers(reader)["Content-Length"])
    return status, json.loads(reader.read(length))


class HttpsTests(ServeFixture, unittest.TestCase):
    stream = True

    def setUp(self):
        super().setUp()
        cert, key = certificate(self.root)
        tls = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        tls.load_cert_chain(cert, key)
        self.trusted = ssl.create_default_context(cafile=str(cert))
        self.logged = []
        self.api = Api(self.engine(), api_keys=["secret"], log=self.logged.append)
        self.server = SmallBufferServer(("127.0.0.1", 0), self.api, tls)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, content, stream=False, key="secret", pause=0):
        connection = http.client.HTTPSConnection(*self.server.server_address, timeout=30, context=self.trusted)
        self.addCleanup(connection.close)
        connection.request("POST", "/v1/chat/completions",
                           json.dumps({"messages": [{"role": "user", "content": content}], "stream": stream}),
                           {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        time.sleep(pause)
        response = connection.getresponse()
        return response, response.read()

    def until(self, predicate, message):
        deadline = time.monotonic() + 5
        while not predicate():
            if time.monotonic() > deadline:
                self.fail(message)
            time.sleep(.01)

    def test_replies_and_a_slowly_read_stream_arrive_whole_over_https(self):
        response, data = self.call("reply:hello")
        self.assertEqual((response.status, json.loads(data)["choices"][0]["message"]["content"]), (200, "hello"))
        self.assertEqual(self.call("reply:hello", key="wrong")[0].status, 401)
        # Sent as \u00e9 escapes, the reply outgrows the socket buffers while the client waits, so the server's TLS
        # writes would block.
        text = "\u00e9" * 60000
        response, data = self.call("reply:" + text, stream=True, pause=.3)
        self.assertEqual(response.status, 200)
        chunks = [json.loads(line[6:]) for line in data.decode().split("\n") if line.startswith("data: {")]
        self.assertEqual("".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks), text)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(data.endswith(b"data: [DONE]\n\n"))

    def test_a_stalled_handshake_or_a_plain_http_request_holds_up_no_one(self):
        stalled = socket.create_connection(self.server.server_address, timeout=5)
        self.addCleanup(stalled.close)
        plain = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        self.addCleanup(plain.close)
        with self.assertRaises((http.client.HTTPException, OSError)):
            plain.request("GET", "/health")
            plain.getresponse().read()
        response, data = self.call("reply:hello")
        self.assertEqual(response.status, 200, data)
        self.assertTrue(any(line.startswith("127.0.0.1 TLS handshake failed: ") for line in self.logged), self.logged)

    def started(self, content):
        """A TLS connection whose chat request for `content` is running."""
        client = self.trusted.wrap_socket(socket.create_connection(self.server.server_address, timeout=5),
                                          server_hostname="127.0.0.1")
        self.addCleanup(client.close)
        client.sendall(chat_request(content))
        self.until(lambda: self.api.metrics.running + self.api.metrics.waiting == 1, "The request did not start")
        return client

    def test_closing_an_https_connection_with_or_without_close_notify_cancels_its_turn(self):
        self.started("slow").close()
        self.until(lambda: self.api.metrics.requests, "The request did not end")
        client = self.started("slow")
        client.setblocking(False)
        with contextlib.suppress(ssl.SSLWantReadError):
            client.unwrap()  # sends close_notify, then would wait for the server's
        self.until(lambda: self.api.metrics.requests == {("openai", "cancelled"): 2}, "close_notify did not cancel")

    def test_a_request_pipelined_over_https_cancels_the_running_turn_and_is_answered(self):
        client = self.started("slow")
        client.sendall(chat_request("reply:after"))
        reader = client.makefile("rb")
        (status, value), (after, reply) = answer(reader), answer(reader)
        self.assertEqual((status, value["error"]["message"]), (499, "Request cancelled"))
        self.assertEqual((after, reply["choices"][0]["message"]["content"]), (200, "after"))

    def test_a_keep_alive_https_connection_carries_request_after_request_in_every_dialect(self):
        connection = http.client.HTTPSConnection(*self.server.server_address, timeout=30, context=self.trusted)
        self.addCleanup(connection.close)
        requests = [("/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:one"}]}),
                    ("/v1/messages", {"model": "default", "max_tokens": 50,
                                      "messages": [{"role": "user", "content": "reply:two"}]}),
                    ("/api/chat", {"model": "default", "messages": [{"role": "user", "content": "reply:three"}],
                                   "stream": False}),
                    ("/v1/responses", {"model": "default", "input": [{"role": "user", "content": "reply:four"}]}),
                    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:five"}]})]
        sockets = []
        with mock.patch("crit_api.server.ClientDisconnect", LateExit):
            for path, body in requests:
                connection.request("POST", path, json.dumps(body),
                                   {"Content-Type": "application/json", "Authorization": "Bearer secret"})
                response = connection.getresponse()
                data = response.read()
                self.assertEqual(response.status, 200, data)
                content = (body.get("messages") or body["input"])[0]["content"]
                self.assertIn(json.dumps(content[len("reply:"):]).encode(), data)
                sockets.append(connection.sock)
        self.assertEqual(len(set(sockets)), 1)
        self.assertEqual(self.api.metrics.requests,
                         {("openai", "completed"): 2, ("anthropic", "completed"): 1, ("ollama", "completed"): 1,
                          ("responses", "completed"): 1})


if __name__ == "__main__":
    unittest.main()


class CommandTests(ServeFixture, unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "Ctrl+C reaches the whole Windows console group, including the test runner")
    def test_crit_serve_loads_answers_and_releases_the_runtime_on_interrupt(self):
        import signal
        self.serve_and_stop(signal.SIGINT)

    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_sigterm_and_a_background_jobs_ignored_sigint_also_release_the_runtime(self):
        import signal
        self.serve_and_stop(signal.SIGTERM)
        # A shell's background job (`crit serve &`) starts with SIGINT ignored.
        self.serve_and_stop(signal.SIGINT, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))

    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_crit_serve_over_https_takes_the_keys_of_a_key_file_and_its_launchers_keep_the_paths(self):
        import signal
        cert, key = certificate(self.root)
        keys = self.root / "keys.txt"
        keys.write_text("# shared on the LAN\nfiled\n", encoding="utf-8")
        arguments = ["--api-key-file", str(keys), "--ssl-certfile", str(cert), "--ssl-keyfile", str(key)]
        self.serve_and_stop(signal.SIGTERM, arguments, ssl.create_default_context(cafile=str(cert)), "filed")
        from crit_api.cli import main
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--workspace", str(self.store.root), *arguments, "--write-launchers"]), 0)
        script = (self.directory / "serve.sh").read_text(encoding="utf-8")
        for path in (keys, cert, key):
            self.assertIn(str(path), script)

    def serve_and_stop(self, stop, arguments=(), context=None, key=None, **options):
        import re, subprocess, sys, time
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        process = subprocess.Popen([sys.executable, "-B", str(root / "tests/run_crit.py"), "serve", "--workspace", str(self.store.root),
                                    "--port", "0", *arguments], stderr=subprocess.PIPE, text=True, env=os.environ.copy(), **options)
        try:
            port = None
            deadline = time.monotonic() + 30
            while port is None and time.monotonic() < deadline:
                line = process.stderr.readline()
                found = re.search(r"%s://127\.0\.0\.1:(\d+)/v1" % ("https" if context else "http"), line)
                port = int(found.group(1)) if found else None
                self.assertNotIn("error", line)
            body = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
            connect = lambda: (http.client.HTTPSConnection("127.0.0.1", port, timeout=30, context=context) if context else
                               http.client.HTTPConnection("127.0.0.1", port, timeout=30))
            if key:
                connection = connect()
                connection.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
                self.assertEqual(connection.getresponse().status, 401)
                connection.close()
            connection = connect()
            connection.request("POST", "/v1/chat/completions", body,
                               {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})})
            value = json.loads(connection.getresponse().read())
            connection.close()
            self.assertEqual(json.loads(value["choices"][0]["message"]["content"])["calls"], 1)
            process.send_signal(stop)
            self.assertEqual(process.wait(30), 0)
            self.assertFalse((self.root / "runtime-owner" / "owner.json").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(10)
            process.stderr.close()

    def test_launchers_repeat_the_command_without_the_api_keys(self):
        import contextlib, io
        from crit_api.cli import main
        (self.root / "keys.txt").write_text("filed\n", encoding="utf-8")
        environment = {k: v for k, v in os.environ.items() if k != "WSL_DISTRO_NAME"}
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.dict(os.environ, environment, clear=True), \
                contextlib.chdir(self.root):
            self.assertEqual(main(["--workspace", str(self.store.root), "--api-key", "secret", "--api-key=secret-too",
                                   "--api-key-file", "keys.txt", "--port", "8123", "--lenient", "--write-launchers"]), 0)
            key_file = str(Path.cwd() / "keys.txt")
        for name in ("serve.cmd", "serve.sh"):
            script = (self.directory / name).read_text(encoding="utf-8")
            self.assertIn("serve", script)
            self.assertIn("8123", script)
            self.assertIn("--lenient", script)
            self.assertIn(self.run_id, script)
            self.assertIn("--api-key-file", script)
            self.assertIn(key_file, script)
            self.assertNotIn("secret", script)

    @unittest.skipIf(os.name == "nt", "WSL launchers are written from the Linux side")
    def test_wsl_windows_launcher_starts_the_posix_script_in_its_distribution(self):
        import contextlib, io
        from crit_api.cli import main
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.dict(os.environ, {"WSL_DISTRO_NAME": "Fixture-Linux"}):
            self.assertEqual(main(["--workspace", str(self.store.root), "--port", "8123", "--write-launchers"]), 0)
        script = (self.directory / "serve.cmd").read_text(encoding="utf-8")
        self.assertIn("wsl.exe -d Fixture-Linux -e " + str(self.directory / "serve.sh"), script)
        self.assertIn("8123", (self.directory / "serve.sh").read_text(encoding="utf-8"))

    def test_the_sse_ping_interval_defaults_to_15_seconds_and_must_be_0_to_a_day(self):
        import contextlib, io
        from crit_api.cli import main, parser
        self.assertEqual(parser().parse_args(["--workspace", str(self.store.root)]).sse_ping_interval, 15)
        for value in ("-1", "86401", "nan", "inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()) as error, \
                    self.assertRaises(SystemExit):
                main(["--workspace", str(self.store.root), "--sse-ping-interval", value])
            self.assertIn("--sse-ping-interval takes 0", error.getvalue())
