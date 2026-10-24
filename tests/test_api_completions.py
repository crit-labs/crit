"""Raw-prompt completions through crit serve (completion-v1): POST /v1/completions and Ollama's raw /api/generate,
with a synthetic resident adapter."""
import http.client
import json
import threading
import unittest

from test_api_score import SCORE_ADAPTER
from test_chat_residency import ServeFixture
from crit_api.logprobs import completion_logprobs
from crit_api.server import Api, Server, completion_prompt
from crit_api.options import RequestError

# A raw prompt's continuation, one token per byte (token-id prompts hold bytes too): the text after "=>" in a text
# prompt, the ids as numbers after a token-id prompt, or for "echo-request" the record's raw-prompt fields. Every
# token's alternatives are itself at -0.25 and the next byte at -1.0; an echo's prompt rows are the same.
COMPLETION_ADAPTER = SCORE_ADAPTER + r'''
import re
from crit_core.serving import ContextExceeded

Engine.logprobs = True
_chat_generate, _chat_stream = Engine.generate, Engine.generate_stream


def _complete(self, request, emit):
    prompt, limit = request['prompt'], request['limits']['max_output_tokens']
    text = prompt.get('text')
    if text is not None and text.startswith('overflow:'):
        raise ContextExceeded(int(text[len('overflow:'):]), limit, request['limits']['kv_tokens'])
    if text == 'echo-request':
        reply = json.dumps({'prompt': prompt, 'echo': request.get('echo'), 'messages': 'messages' in request,
                            'sampling': request.get('sampling'), 'logprobs': request.get('logprobs'), 'limit': limit})
    elif text is not None:
        reply = text.split('=>', 1)[1] if '=>' in text else ' ok'
    else:
        reply = ''.join(' ' + str(t) for t in prompt['ids'])
    reply = reply.encode()[:limit].decode()
    ids = list(text.encode()) if text is not None else prompt['ids']
    for piece in re.findall(r'\S+\s*|\s+', reply) if emit else ():
        emit(piece)
        time.sleep(.02)
    output = max(1, len(reply.encode()))  # an empty reply still ends with its stop token
    response = dict(schema_version=1, record_kind='chat_response', assistant=reply,
                    **{k: request[k] for k in ('request_id', 'candidate_id', 'package_sha256')},
                    telemetry=dict(prompt_tokens=len(ids), output_tokens=output, prefill_tokens=len(ids), prefill_ms=5,
                                   decode_tokens=output - 1 if output > 1 else None,
                                   decode_ms=10 if output > 1 else None, ttft_ms=6,
                                   missing={} if output > 1 else {'decode': 'the reply has a single token'}))
    if request['limits'].get('fit_output'):   # chat-fit-output-v1: the cap applied, the room the prompt leaves
        response['max_output_tokens'] = min(limit, request['limits']['kv_tokens'] - len(ids))
    if request.get('echo'):
        response['prompt'] = text if text is not None else ''.join(map(chr, ids))
    n = request.get('logprobs')
    if n is not None:
        row = lambda t: [t, -0.25, [[t, -0.25], [t + 1, -1.0]][:n]]
        rows = [row(t) for t in reply.encode()]
        if request.get('echo'):
            rows_prompt = [[ids[0], None, []]] + [row(t) for t in ids[1:]]
        named = {t for r in rows + (rows_prompt if request.get('echo') else []) for t in [r[0]] + [p[0] for p in r[2]]}
        response['logprobs'] = {'tokens': rows, 'bytes': {str(t): bytes([t]).hex() for t in named}}
        if request.get('echo'):
            response['logprobs']['prompt'] = rows_prompt
    return response


def _generate(self, request, cancellation):
    return _complete(self, request, None) if 'prompt' in request else _chat_generate(self, request, cancellation)


def _generate_stream(self, request, cancellation, emit):
    return _complete(self, request, emit) if 'prompt' in request else _chat_stream(self, request, cancellation, emit)


Engine.generate, Engine.generate_stream = _generate, _generate_stream
'''


class PromptTests(unittest.TestCase):
    def test_one_prompt_as_text_or_token_ids(self):
        for value, expected in (("Hi", {"text": "Hi"}), (["Hi"], {"text": "Hi"}), ([5, 6], {"ids": [5, 6]}),
                                ([[5, 6]], {"ids": [5, 6]}), ([7], {"ids": [7]})):
            with self.subTest(value=value):
                self.assertEqual(completion_prompt(value), expected)
        for value in ("", [], [[]], ["a", "b"], [[1], [2]], [1, -2], [1.5], [True], None, {"text": "x"}, "\ud800"):
            with self.subTest(value=value), self.assertRaises(RequestError) as caught:
                completion_prompt(value)
            self.assertEqual(caught.exception.param, "prompt")
        with self.assertRaisesRegex(RequestError, "one prompt per request"):
            completion_prompt(["a", "b"])

    def test_logprobs_take_the_legacy_completions_shape(self):
        entry = lambda token, logprob, top=(): {"token": token.decode("utf-8", "replace"), "logprob": logprob,
                                                "bytes": list(token), "top_logprobs": [
                                                    {"token": t.decode("utf-8", "replace"), "logprob": p,
                                                     "bytes": list(t)} for t, p in top]}
        # An echoed prompt's first token has no log-probability; "é" comes in two tokens, which share its offset.
        entries = [dict(entry(b"h", None), top_logprobs=None), entry(b"\xc3", -0.5, [(b"\xc3", -0.5), (b"\xa9", -1.0)]),
                   entry(b"\xa9", -0.25, [(b"\xa9", -0.25)]), entry(b"!", -1.0, [(b"!", -1.0)])]
        self.assertEqual(completion_logprobs(entries), {
            "tokens": ["h", "�", "�", "!"], "token_logprobs": [None, -0.5, -0.25, -1.0],
            "top_logprobs": [None, {"�": -0.5}, {"�": -0.25}, {"!": -1.0}], "text_offset": [0, 1, 1, 2]})


class CompletionFixture(ServeFixture):
    stream = True
    extra_capabilities = ("completion-v1", "chat-logprobs-v1", "chat-score-v1")
    adapter_suffix = COMPLETION_ADAPTER
    lenient = False

    def setUp(self):
        super().setUp()
        self.served = self.engine()
        self.server = Server(("127.0.0.1", 0), Api(self.served, lenient=self.lenient))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, path, body):
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=30)
        self.addCleanup(connection.close)
        connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response, response.read()

    def complete(self, prompt, **options):
        response, data = self.call("/v1/completions", {"model": "default", "prompt": prompt, **options})
        return response, data if options.get("stream") and response.status == 200 else json.loads(data)


class CompletionTests(CompletionFixture, unittest.TestCase):
    def test_a_text_prompt_is_continued_without_the_chat_template(self):
        response, value = self.complete("2+2=>4", max_tokens=5, user="u")
        self.assertEqual(response.status, 200, value)
        self.assertTrue(value["id"].startswith("cmpl-"))
        self.assertEqual((value["object"], value["model"]), ("text_completion", self.run_id + "/" + self.a))
        self.assertEqual(value["choices"], [{"text": "4", "index": 0, "logprobs": None, "finish_reason": "stop"}])
        self.assertEqual(value["usage"], {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7,
                                          "prompt_tokens_details": {"cached_tokens": 0}})
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "user")
        # The engine gets the prompt in place of messages, with the request's output cap and sampling.
        response, value = self.complete("echo-request", temperature=0)
        record = json.loads(value["choices"][0]["text"])
        self.assertEqual(record, {"prompt": {"text": "echo-request"}, "echo": None, "messages": False,
                                  "sampling": {"temperature": 0}, "logprobs": None, "limit": 1024})
        for prompt, text in ((["x=>one"], "one"), ([72, 105], " 72 105"), ([[72, 105]], " 72 105")):
            with self.subTest(prompt=prompt):
                response, value = self.complete(prompt)
                self.assertEqual((response.status, value["choices"][0]["text"]), (200, text), value)
        response, value = self.complete("x=>abcdef", max_tokens=3)
        self.assertEqual((value["choices"][0]["text"], value["choices"][0]["finish_reason"]), ("abc", "length"))
        # A continuation may be nothing at all: the model ended at once.
        response, value = self.complete("x=>")
        self.assertEqual((response.status, value["choices"][0]["text"]), (200, ""), value)
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_logprobs_list_each_token_with_its_alternatives_and_offset(self):
        response, value = self.complete("a=>hi", logprobs=2)
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value["choices"][0]["logprobs"], {
            "tokens": ["h", "i"], "token_logprobs": [-0.25, -0.25],
            "top_logprobs": [{"h": -0.25, "i": -1.0}, {"i": -0.25, "j": -1.0}], "text_offset": [0, 1]})
        response, value = self.complete("a=>hi", logprobs=0)
        self.assertEqual(value["choices"][0]["logprobs"]["top_logprobs"], [{}, {}])

    def test_echo_returns_the_prompt_and_with_logprobs_its_rows(self):
        response, value = self.complete("a=>hi", echo=True)
        self.assertEqual((response.status, value["choices"][0]["text"]), (200, "a=>hi" + "hi"), value)
        response, value = self.complete([72, 105], echo=True)
        self.assertEqual(value["choices"][0]["text"], "Hi 72 105")
        # As lm-eval scores a continuation: the prompt's rows, the first without a log-probability, then the reply's.
        response, value = self.complete("a=>hi", echo=True, logprobs=1, max_tokens=1)
        self.assertEqual(response.status, 200, value)
        found = value["choices"][0]["logprobs"]
        self.assertEqual(found["tokens"], ["a", "=", ">", "h", "i", "h"])
        self.assertEqual(found["token_logprobs"], [None] + [-0.25] * 5)
        self.assertEqual(found["top_logprobs"], [None, {"=": -0.25}, {">": -0.25}, {"h": -0.25}, {"i": -0.25}, {"h": -0.25}])
        self.assertEqual(found["text_offset"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(value["choices"][0]["text"], "a=>hih")

    def test_a_stream_sends_text_completion_chunks_then_done(self):
        response, data = self.complete("x=> one two three", stream=True, logprobs=1,
                                       stream_options={"include_usage": True})
        self.assertEqual((response.status, response.getheader("Content-Type")), (200, "text/event-stream"))
        events = [line[6:] for line in data.decode().split("\n\n") if line.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertTrue(all(c["object"] == "text_completion" for c in chunks))
        self.assertEqual("".join(c["choices"][0]["text"] for c in chunks if c["choices"]), " one two three")
        self.assertGreater(len([c for c in chunks if c["choices"] and c["choices"][0]["text"]]), 1)
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual("".join(chunks[-2]["choices"][0]["logprobs"]["tokens"]), " one two three")
        self.assertEqual((chunks[-1]["choices"], chunks[-1]["usage"]["completion_tokens"]), ([], 14))

    def test_invalid_requests_are_refused_in_openai_shape(self):
        cases = [({"prompt": ["a", "b"]}, "prompt"), ({"prompt": ""}, "prompt"), ({"prompt": [1, -1]}, "prompt"),
                 ({"suffix": "}"}, "suffix"), ({"echo": True, "stream": True}, "echo"), ({"echo": True, "stop": "x"}, "echo"),
                 ({"echo": "yes"}, "echo"), ({"logprobs": True}, "logprobs"), ({"logprobs": 21}, "logprobs"),
                 ({"logprobs": 1, "stop": "x"}, "stop"), ({"n": 2}, "n"), ({"best_of": 3}, "best_of"),
                 ({"max_tokens": 9000}, "max_tokens"), ({"top_p": 0.5}, "top_p"), ({"messages": []}, "messages"),
                 ({"response_format": {"type": "json_object"}}, "response_format"), ({"stop": ["a"] * 17}, "stop")]
        for options, param in cases:
            with self.subTest(options=options):
                response, value = self.complete(**{"prompt": "x=>y", **options})
                self.assertEqual(response.status, 400, value)
                self.assertEqual(value["error"]["param"], param)
        response, value = self.complete("x=>y", suffix="}")
        self.assertEqual(value["error"]["code"], "unsupported_parameter")

    def test_a_prompt_over_the_context_is_a_400_in_completions_words(self):
        message = ("This model's maximum context length is 8192 tokens, however you requested 9100 tokens (9000 in "
                   "your prompt; 100 for the completion). Please reduce your prompt; or completion length.")
        error = {"message": message, "type": "invalid_request_error", "param": "prompt", "code": "context_length_exceeded"}
        response, value = self.complete("overflow:9000", max_tokens=100)
        self.assertEqual((response.status, value), (400, {"error": error, "request_id": response.getheader("X-Request-Id")}))
        response, data = self.complete("overflow:9000", max_tokens=100, stream=True)
        events = [line[6:] for line in data.decode().split("\n\n") if line.startswith("data: ")]
        self.assertEqual((json.loads(events[-2]), events[-1]),
                         ({"error": error, "request_id": response.getheader("X-Request-Id")}, "[DONE]"))
        response, value = self.complete("x=>y")
        self.assertEqual(response.status, 200, value)
        self.assertEqual(self.served.status()["engine_loads"], 1)

    def test_the_chat_requests_extensions_apply_to_a_raw_prompt(self):
        # vLLM's include_stop_str_in_output and priority, crit's timeout, n null, and the request's id.
        response, value = self.complete("x=> one two three", stop="two", include_stop_str_in_output=True, priority=-1,
                                        crit={"timeout": 30}, n=None, best_of=None)
        self.assertEqual(response.status, 200, value)
        self.assertEqual((value["choices"][0]["text"], value["choices"][0]["finish_reason"]), (" one two", "stop"))
        self.assertRegex(response.getheader("X-Request-Id"), "^[0-9a-f]{32}$")
        response, value = self.complete("x=> one two three", stop="two")
        self.assertEqual(value["choices"][0]["text"], " one ")
        for options, param in (({"priority": "high"}, "priority"), ({"include_stop_str_in_output": 1},
                                "include_stop_str_in_output"), ({"crit": {"timeout": 0}}, "crit.timeout"),
                               ({"stream_options": {"include_usage": True, "extra": 1}}, "stream_options.extra"),
                               ({"typical_p": 0.5}, "typical_p")):
            with self.subTest(options=options):
                response, value = self.complete("x=>y", **options)
                self.assertEqual((response.status, value["error"]["param"]), (400, param), value)
        # Client metadata is ignored and listed, and an extra sampler turned off is a no-op.
        response, value = self.complete("x=>y", prompt_cache_key="k", typical_p=1)
        self.assertEqual((response.status, response.getheader("X-Crit-Ignored-Parameters")), (200, "prompt_cache_key"))
        rendered = self.served_metrics()
        self.assertIn('crit_serve_requests_total{dialect="openai",outcome="completed"}', rendered)

    def served_metrics(self):
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=30)
        self.addCleanup(connection.close)
        connection.request("GET", "/metrics")
        return connection.getresponse().read().decode()

    def test_chat_requests_are_unchanged(self):
        response, data = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "reply:hi"}]})
        self.assertEqual((response.status, json.loads(data)["choices"][0]["message"]["content"]), (200, "hi"))
        response, data = self.call("/v1/chat/completions", {"prompt": "x", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((response.status, json.loads(data)["error"]["param"]), (400, "prompt"))


class OllamaRawTests(CompletionFixture, unittest.TestCase):
    def generate(self, prompt, **options):
        return self.call("/api/generate", {"model": "default", "prompt": prompt, "raw": True, **options})

    def test_raw_generate_continues_the_prompt_as_written(self):
        response, data = self.generate("echo-request", stream=False, options={"num_predict": 200, "temperature": 0.5},
                                       keep_alive="5m")
        self.assertEqual(response.status, 200, data)
        value = json.loads(data)
        self.assertEqual(json.loads(value["response"]), {"prompt": {"text": "echo-request"}, "echo": None,
                                                         "messages": False, "sampling": {"temperature": 0.5},
                                                         "logprobs": None, "limit": 200})
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "keep_alive")
        response, data = self.generate("x=>abcdef", stream=False, options={"num_predict": 3})
        value = json.loads(data)
        self.assertEqual((value["response"], value["done_reason"], value["prompt_eval_count"]), ("abc", "length", 9))
        response, data = self.generate("x=> a b c")
        rows = [json.loads(line) for line in data.decode().splitlines() if line]
        self.assertEqual("".join(r["response"] for r in rows[:-1]), " a b c")
        self.assertEqual((rows[-1]["done"], rows[-1]["done_reason"]), (True, "stop"))

    def test_raw_logprobs_take_ollamas_fields_and_think_is_ignored(self):
        response, data = self.generate("x=>hi", stream=False, logprobs=True, top_logprobs=1, think=True)
        self.assertEqual(response.status, 200, data)
        value = json.loads(data)
        self.assertEqual([(e["token"], e["logprob"], e["crit_token_id"]) for e in value["logprobs"]],
                         [("h", -0.25, ord("h")), ("i", -0.25, ord("i"))])
        self.assertEqual(value["logprobs"][0]["top_logprobs"][0]["token"], "h")
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "think")
        response, data = self.generate("echo-request", stream=False, logprobs=False, top_logprobs=2)
        self.assertEqual(json.loads(json.loads(data)["response"])["logprobs"], None)
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "top_logprobs")
        response, data = self.generate("x=>hi", stream=False, logprobs=1)
        self.assertEqual(response.status, 400, data)
        self.assertIn("logprobs", json.loads(data)["error"])

    def test_a_raw_prompt_takes_no_system_images_or_format(self):
        for options in ({"system": "Be brief."}, {"images": ["iVBOR"]}, {"format": "json"}, {"suffix": "}"},
                        {"raw": "yes"}):
            with self.subTest(options=options):
                response, data = self.generate("x=>y", stream=False, **options)
                self.assertEqual(response.status, 400, data)
                self.assertIn(next(iter(options)), json.loads(data)["error"])
        response, data = self.call("/api/generate", {"model": "default", "prompt": "hello", "raw": False, "stream": False})
        self.assertEqual(json.loads(json.loads(data)["response"])["roles"], ["user"])  # the chat path, as before


class FittedCompletionTests(CompletionFixture, unittest.TestCase):
    """A package declaring chat-fit-output-v1 fits a raw prompt's cap at or above the context, as a chat's."""
    fit_output = True

    def test_a_cap_at_or_above_the_context_is_fitted(self):
        response, value = self.complete("x=>abc", max_tokens=9000)
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value["choices"][0]["text"], "abc")
        self.assertEqual(response.getheader("X-Crit-Max-Tokens-Applied"), str(8192 - len("x=>abc")))
        response, value = self.complete("echo-request", max_tokens=1000)
        self.assertEqual(json.loads(value["choices"][0]["text"])["limit"], 1000)
        self.assertIsNone(response.getheader("X-Crit-Max-Tokens-Applied"))
        # Ollama's num_predict -1 asks for all the room the context leaves.
        response, data = self.call("/api/generate", {"model": "default", "prompt": "echo-request", "raw": True,
                                                     "stream": False, "options": {"num_predict": -1}})
        self.assertEqual(json.loads(json.loads(data)["response"])["limit"], 8191)
        self.assertEqual(response.getheader("X-Crit-Max-Tokens-Applied"), str(8192 - len("echo-request")))


class LenientCompletionTests(CompletionFixture, unittest.TestCase):
    lenient = True

    def test_lenient_ignores_unknown_keys_and_undeclared_samplers(self):
        response, value = self.complete("x=>y", mystery=1, top_p=0.5, stream_options={"extra": 1})
        self.assertEqual(response.status, 200, value)
        self.assertEqual(response.getheader("X-Crit-Ignored-Parameters"), "mystery,stream_options.extra,top_p")
        response, value = self.complete("x=>y", suffix="}")
        self.assertEqual((response.status, value["error"]["param"]), (400, "suffix"))


class UnscoredCompletionTests(CompletionFixture, unittest.TestCase):
    """An engine without chat-score-v1 has no prompt rows: echo cannot carry logprobs."""
    extra_capabilities = ("completion-v1", "chat-logprobs-v1")
    adapter_suffix = COMPLETION_ADAPTER.replace(SCORE_ADAPTER, "")

    def test_echo_with_logprobs_needs_the_score_path(self):
        response, value = self.complete("a=>hi", echo=True, logprobs=1)
        self.assertEqual((response.status, value["error"]["param"]), (400, "echo"), value)
        self.assertIn("chat-score-v1", value["error"]["message"])
        response, value = self.complete("a=>hi", echo=True)
        self.assertEqual((response.status, value["choices"][0]["text"]), (200, "a=>hihi"), value)


if __name__ == "__main__":
    unittest.main()
