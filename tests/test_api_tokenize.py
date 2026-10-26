"""Tokenizing through crit serve (chat-tokenize-v1) with a synthetic resident adapter: /tokenize, /detokenize,
/apply-template and Anthropic's /v1/messages/count_tokens, beside generation."""
import http.client
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "../crit-eval", "kernels", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))
from test_chat_residency import ServeFixture
from crit_api.host import Cancellation, Host, Unloaded
from crit_api.server import Api, Server

# A token is a character's code point and the special token is 1; a rendered prompt is the request's tools as JSON,
# then role:content; per message. "huge" tokenizes past the response bound, "fail" fails, "slow" takes half a second
# and answers [1] when the engine was closed meanwhile.
TOKENIZE_ADAPTER = r'''
_close = Engine.close

def _closing(self):
    self.closed = True
    _close(self)

def _tokenize(self, text, add_special_tokens):
    if text == 'huge':
        return [1000000000] * 1600000
    if text == 'fail':
        raise RuntimeError('synthetic tokenizer failure')
    if text == 'slow':
        time.sleep(.5)
        return [int(getattr(self, 'closed', False))]
    return ([1] if add_special_tokens else []) + [ord(c) for c in text]

def _render(self, request):
    text = json.dumps(request.get('tools') or []) + ''.join(m['role'] + ':' + m['content'] + ';'
                                                            for m in request['messages'])
    return text, self.tokenize(text, False)

def _detokenize(self, ids):
    if any(token > 0x10FFFF for token in ids):
        from crit_core.serving import RequestRejected
        raise RequestRejected('token id out of range')
    return ''.join(map(chr, ids))

Engine.close, Engine.tokenize, Engine.render, Engine.detokenize = _closing, _tokenize, _render, _detokenize
'''
# Also the template variables and continuation a render receives, after the messages, so a test sees what reached it.
TEMPLATE_ADAPTER = TOKENIZE_ADAPTER + r'''
def _render_options(self, request):
    options = {key: request[key] for key in ('template_kwargs', 'continue_final_message') if key in request}
    text = _render(self, request)[0] + json.dumps(options, sort_keys=True)
    return text, self.tokenize(text, False)

Engine.render = _render_options
'''
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]


def rendered(messages, tools=()):
    return json.dumps(list(tools)) + "".join(m["role"] + ":" + m["content"] + ";" for m in messages)


class TokenizeFixture(ServeFixture):
    def setUp(self):
        super().setUp()
        self.served = self.engine()
        self.server = Server(("127.0.0.1", 0), Api(self.served))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, path, body, method="POST", headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=60)
        self.addCleanup(connection.close)
        connection.request(method, path, json.dumps(body), {"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        return response, json.loads(response.read())


class TokenizeApiTests(TokenizeFixture, unittest.TestCase):
    extra_capabilities = ("chat-tokenize-v1",)
    adapter_suffix = TOKENIZE_ADAPTER
    request_options = True  # tools

    def test_tokenize_detokenize_and_apply_template_answer_in_vllm_and_llama_server_shapes(self):
        response, value = self.call("/tokenize", {"model": "default", "prompt": "hi"})
        self.assertEqual((response.status, value), (200, {"tokens": [1, 104, 105], "count": 3,
                                                          "max_model_len": 8192}))
        response, value = self.call("/tokenize", {"prompt": "hi", "add_special_tokens": False})
        self.assertEqual(value["tokens"], [104, 105])
        # Messages render as a chat request's: the system message first, the tools offered.
        response, value = self.call("/tokenize", {"messages": MESSAGES, "tools": TOOLS})
        self.assertEqual((response.status, value["tokens"]), (200, [ord(c) for c in rendered(MESSAGES, TOOLS)]))
        self.assertEqual(value["count"], len(rendered(MESSAGES, TOOLS)))
        response, value = self.call("/detokenize", {"tokens": [104, 105]})
        self.assertEqual((response.status, value), (200, {"prompt": "hi"}))
        response, value = self.call("/apply-template", {"messages": MESSAGES, "tools": TOOLS})
        self.assertEqual((response.status, value), (200, {"prompt": rendered(MESSAGES, TOOLS)}))
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_count_tokens_counts_an_anthropic_request_with_its_system_prompt_and_tools(self):
        tools = [{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object"}}]
        body = {"model": "claude-opus-5-5", "system": "Be brief.", "messages": [{"role": "user", "content": "hi"}],
                "tools": tools, "thinking": {"type": "enabled", "budget_tokens": 1024}}
        response, value = self.call("/v1/messages/count_tokens", body)
        openai = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"},
                                                    "description": "Read a file"}}]
        self.assertEqual((response.status, value), (200, {"input_tokens": len(rendered(MESSAGES, openai))}))
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "thinking")
        response, value = self.call("/v1/messages/count_tokens", dict(body, max_tokens=5))
        self.assertEqual((response.status, value["error"]["type"]), (400, "invalid_request_error"))

    def test_a_tokenize_runs_beside_a_generation_and_never_waits_for_it(self):
        stop = threading.Event()
        wait = self.background(self.served, "slow", stop)
        self.generating(self.served)
        response, value = self.call("/tokenize", {"prompt": "hi"})
        self.assertEqual((response.status, value["count"]), (200, 3))
        # The five-second turn is still running: the tokenize did not queue behind it.
        self.assertEqual(self.served.status()["state"], "generating")
        stop.set()
        self.assertEqual(wait(), ("cancelled", None))
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_a_candidate_not_loaded_is_loaded_for_its_tokenize_as_for_a_reply(self):
        pid = self.served.host_pid()
        # Told B is loaded while A is, the host refuses the fast path and crit serve loads B through admission.
        with patch.object(self.served, "loaded", return_value=True):
            response, value = self.call("/tokenize", {"model": self.b, "prompt": "hi"})
        self.assertEqual((response.status, value["count"]), (200, 3))
        status = self.served.status()
        self.assertEqual((status["engine_loads"], status["engine"]["candidate_id"]), (2, self.b))
        response, value = self.call("/detokenize", {"model": self.a, "tokens": [104]})
        self.assertEqual((response.status, value, self.served.status()["engine_loads"]), (200, {"prompt": "h"}, 3))
        self.assertEqual(self.served.host_pid(), pid)

    def test_a_tokenize_waiting_to_load_its_candidate_leaves_when_its_client_disconnects(self):
        stop = threading.Event()
        wait = self.background(self.served, "slow", stop)
        self.generating(self.served)
        entered, finished = threading.Event(), threading.Event()
        tokenize = self.served.tokenize

        def observed(*args, **kwargs):
            entered.set()
            try:
                return tokenize(*args, **kwargs)
            finally:
                finished.set()

        with patch.object(self.server, "handle_error") as server_error:
            with patch.object(self.served, "tokenize", side_effect=observed):
                connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=5)
                self.addCleanup(connection.close)
                connection.request("POST", "/tokenize", json.dumps({"model": self.b, "prompt": "hi"}),
                                   {"Content-Type": "application/json"})
                self.assertTrue(entered.wait(5), "The tokenize did not reach the engine")
                connection.close()
                self.assertTrue(finished.wait(2), "The abandoned tokenize stayed queued behind the generation")
            self.assertEqual(self.served.status()["state"], "generating")
            stop.set()
            self.assertEqual(wait(), ("cancelled", None))
            server_error.assert_not_called()
        # B was never loaded for the request nobody waited on.
        status = self.served.status()
        self.assertEqual((status["engine_loads"], status["engine"]["candidate_id"]), (1, self.a))

    def test_a_tokenize_waiting_for_its_candidate_is_refused_by_a_full_queue_and_cancelled_by_an_abort(self):
        stop = threading.Event()
        wait = self.background(self.served, "slow", stop)
        self.generating(self.served)
        turn = [{"role": "user", "content": "hi"}]
        with patch.object(self.served.lock, "max_queue", 0):
            for path, body, status in (("/tokenize", {"model": self.b, "prompt": "hi"}, 429),
                                       ("/v1/messages/count_tokens", {"model": self.b, "messages": turn}, 529)):
                with self.subTest(path=path):
                    response, value = self.call(path, body)
                    self.assertEqual((response.status, response.getheader("Retry-After")), (status, "1"), value)
        found = {}
        waiting = threading.Thread(target=lambda: found.update(result=self.call(
            "/tokenize", {"model": self.b, "prompt": "hi"}, headers={"X-Request-Id": "count-1"})))
        waiting.start()
        end = time.monotonic() + 5
        while self.served.lock.waiting() != 1 and time.monotonic() < end:
            time.sleep(.01)
        response, value = self.call("/v1/crit/abort", {"request_id": "count-1"})
        self.assertEqual((response.status, value["found"]), (200, True), value)
        waiting.join(5)
        response, value = found["result"]
        self.assertEqual((response.status, value["error"]["message"]), (499, "Request cancelled"))
        self.assertEqual(self.served.status()["state"], "generating")
        stop.set()
        self.assertEqual(wait(), ("cancelled", None))
        status = self.served.status()
        self.assertEqual((status["engine_loads"], status["engine"]["candidate_id"]), (1, self.a))

    def test_refusals_and_adapter_failures_keep_the_engine_loaded(self):
        pid = self.served.host_pid()
        response, value = self.call("/detokenize", {"tokens": [0x110000]})
        self.assertEqual((response.status, value["error"]["message"]), (400, "token id out of range"))
        response, value = self.call("/tokenize", {"prompt": "huge"})
        self.assertEqual(response.status, 400, value)
        self.assertIn("exceeds", value["error"]["message"])
        response, value = self.call("/tokenize", {"prompt": "fail"})
        self.assertEqual(response.status, 500, value)
        self.assertIn("synthetic tokenizer failure", value["error"]["message"])
        response, value = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:ok"}]})
        self.assertEqual((response.status, value["choices"][0]["message"]["content"]), (200, "ok"))
        self.assertEqual((self.served.host_pid(), self.served.status()["engine_loads"]), (pid, 1))

    def test_invalid_requests_are_refused_before_the_engine(self):
        for path, body, param in (("/tokenize", {"prompt": 5}, "prompt"),
                                  ("/tokenize", {"prompt": "x", "messages": MESSAGES}, "prompt"),
                                  ("/tokenize", {"prompt": "x", "add_special_tokens": "yes"}, "add_special_tokens"),
                                  ("/tokenize", {"messages": MESSAGES, "add_special_tokens": True},
                                   "add_special_tokens"),
                                  ("/tokenize", {"prompt": "x", "return_token_strs": True}, "return_token_strs"),
                                  ("/detokenize", {"tokens": [-1]}, "tokens"),
                                  ("/apply-template", {"messages": MESSAGES, "stream": True}, "stream"),
                                  ("/apply-template", {"messages": [{"role": "assistant", "content": "x"}]},
                                   "messages"),
                                  ("/apply-template", {"messages": MESSAGES, "tool_choice": "auto"}, "tool_choice")):
            with self.subTest(path=path, body=body):
                response, value = self.call(path, body)
                self.assertEqual((response.status, value["error"]["param"]), (400, param), value)
        response, value = self.call("/tokenize", {"prompt": "x"}, method="DELETE")
        self.assertEqual((response.status, value["error"]["code"]), (405, "method_not_allowed"))

    def test_closing_the_engine_waits_for_a_running_tokenize(self):
        host = Host(self.root / "tokenize-host", 1024, 1024, lambda message: None)
        spec = self.served.spec(self.a)
        host.attach(spec)
        request = {"record_kind": "tokenize_request", "run_id": spec["run_id"], "candidate_id": spec["candidate_id"],
                   "package_sha256": spec["package_sha256"], "request_id": "r", "operation": "tokenize",
                   "prompt": "slow", "add_special_tokens": False}
        found = {}
        thread = threading.Thread(target=lambda: found.update(host.tokenize(spec, request)))
        thread.start()
        end = time.monotonic() + 5
        while not host.texts and time.monotonic() < end:
            time.sleep(.01)
        host.handle({"action": "unload", "payload": {}}, Cancellation())
        thread.join(5)
        self.assertEqual(found["tokens"], [0])  # the engine closed only after the tokenize finished
        with self.assertRaises(Unloaded):
            host.tokenize(spec, request)


class TokenizeTemplateTests(TokenizeFixture, unittest.TestCase):
    """A package declaring chat-continue-v1 and a thinking switch: a render takes the template variables and the
    continued final message that a reply to the same request takes."""
    extra_capabilities = ("chat-tokenize-v1", "chat-continue-v1", "chat-template-kwargs-v1",
                          "chat-template-enable-thinking-v1")
    adapter_suffix = TEMPLATE_ADAPTER
    request_options = True  # tools

    def test_template_variables_and_a_continued_message_render_as_a_reply_would(self):
        prefill = MESSAGES + [{"role": "assistant", "content": "Once"}]
        continued = {"continue_final_message": True, "add_generation_prompt": False}
        for body, messages, options in (
                ({"messages": MESSAGES}, MESSAGES, {}),
                ({"messages": MESSAGES, "chat_template_kwargs": {"enable_thinking": False}}, MESSAGES,
                 {"template_kwargs": {"enable_thinking": False}}),
                ({"messages": MESSAGES, "reasoning_effort": "high"}, MESSAGES,
                 {"template_kwargs": {"enable_thinking": True}}),
                ({"messages": prefill, **continued}, prefill, {"continue_final_message": True})):
            with self.subTest(body=body):
                expected = rendered(messages) + json.dumps(options, sort_keys=True)
                response, value = self.call("/apply-template", body)
                self.assertEqual((response.status, value), (200, {"prompt": expected}))
                response, value = self.call("/tokenize", body)
                self.assertEqual((response.status, value["tokens"]), (200, [ord(c) for c in expected]))

    def test_count_tokens_renders_thinking_and_a_prefill_as_messages_does(self):
        body = {"model": "claude-opus-5-5", "system": "Be brief.",
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Once"}]}
        response, value = self.call("/v1/messages/count_tokens", body)
        options = {"continue_final_message": True, "template_kwargs": {"enable_thinking": True}}
        expected = rendered(MESSAGES + [{"role": "assistant", "content": "Once"}]) + json.dumps(options, sort_keys=True)
        self.assertEqual((response.status, value), (200, {"input_tokens": len(expected)}))
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "thinking.budget_tokens")
        for extra in ({"output_config": {"effort": "high"}}, {"thinking": {"type": "sometimes"}}):
            with self.subTest(extra=extra):
                response, value = self.call("/v1/messages/count_tokens", dict(body, **extra))
                self.assertEqual((response.status, value["error"]["type"]), (400, "invalid_request_error"))

    def test_a_render_refuses_what_a_chat_request_refuses(self):
        prefill = MESSAGES + [{"role": "assistant", "content": "Once"}]
        for body, param in (({"messages": prefill, "continue_final_message": True}, "add_generation_prompt"),
                            ({"messages": prefill, "continue_final_message": True, "add_generation_prompt": False,
                              "tools": TOOLS}, "tools"),
                            ({"messages": MESSAGES, "chat_template_kwargs": {"reasoning_effort": "low"}},
                             "chat_template_kwargs"),
                            ({"prompt": "x", "chat_template_kwargs": {"enable_thinking": True}}, "prompt")):
            with self.subTest(body=body):
                response, value = self.call("/apply-template" if "messages" in body else "/tokenize", body)
                self.assertEqual((response.status, value["error"]["param"]), (400, param), value)


class TokenizeWithoutCapabilityTests(TokenizeFixture, unittest.TestCase):
    def test_every_path_is_a_404_not_supported_in_its_dialect(self):
        for path, body in (("/tokenize", {"prompt": "hi"}), ("/detokenize", {"tokens": [1]}),
                           ("/apply-template", {"messages": MESSAGES})):
            with self.subTest(path=path):
                response, value = self.call(path, body)
                self.assertEqual((response.status, value["error"]["code"]), (404, "not_supported"))
                self.assertIn("chat-tokenize-v1", value["error"]["message"])
        response, value = self.call("/v1/messages/count_tokens", {"model": "x", "messages": []})
        self.assertEqual((response.status, value["error"]["type"]), (404, "not_found_error"))


if __name__ == "__main__":
    unittest.main()
