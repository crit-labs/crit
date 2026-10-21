"""Reasoning split from the reply in the chat template's own markers: crit_core.text.reasoning, ResidentChat, the
stream protocol and crit serve's reasoning_content (offline)."""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "cli", "eval", "tests"): sys.path.insert(0, str(ROOT / _folder))

from core_text_fixtures import llama_files, write_files  # noqa: E402
from crit_core.serving import ResidentChat  # noqa: E402
from crit_core.text import TextError  # noqa: E402
from crit_core.text.reasoning import ReasoningFormat, ReasoningSplitter, template_reasoning_format  # noqa: E402
from crit_api.host import Cancellation  # noqa: E402
from crit_api.stream import StreamReader, StreamWriter  # noqa: E402
import test_api  # noqa: E402,F401  (puts the checkout's packages on sys.path)
from test_chat_residency import ServeFixture  # noqa: E402
from crit_api.engine import ServedEngine, reply  # noqa: E402
from crit_api.server import Api, Server  # noqa: E402

# A template that strips thinking from history and opens the generation prompt inside it (Qwen3 style).
THINK_TEMPLATE = ("{%- for message in messages %}{{ message.role }}:{{ message.content.split('</think>')[-1] }};"
                  "{%- endfor %}{%- if add_generation_prompt %}assistant:<think>\n{%- endif %}")


def split(text, chunk=None, **options):
    splitter = ReasoningSplitter("think-tags", **options)
    events = []
    for start in range(0, len(text), chunk or max(len(text), 1)):
        events += splitter.push(text[start:start + (chunk or len(text))])
    events += splitter.finish()
    joined = {"reasoning": "", "text": ""}
    for kind, value in events:
        joined[kind] += value
    return joined["reasoning"], joined["text"]


class SplitterTests(unittest.TestCase):
    def test_reasoning_and_reply_in_any_chunking(self):
        cases = [
            ("<think>\nplan it\n</think>\n\nThe answer.", {}, ("plan it", "The answer.")),
            ("  <think>a</think>b", {}, ("a", "b")),
            ("plan it\n</think>\n\nDone", {"opened": True}, ("plan it", "Done")),
            ("No thinking here.", {}, ("", "No thinking here.")),
            ("Text with <think>inside</think> stays text", {}, ("", "Text with <think>inside</think> stays text")),
            ("<thanks>", {}, ("", "<thanks>")),
            ("<think>cut before the end", {}, ("cut before the end", "")),
            ("still thinking", {"opened": True}, ("still thinking", "")),
            ("", {"opened": True}, ("", "")),
        ]
        for text, options, expected in cases:
            for chunk in (None, 1, 3):
                with self.subTest(text=text, chunk=chunk):
                    self.assertEqual(split(text, chunk, **options), expected)

    def test_template_detection_and_package_formats(self):
        self.assertEqual(template_reasoning_format([THINK_TEMPLATE, "{{ x }}"]), "think-tags")
        self.assertIsNone(template_reasoning_format(["{{ messages[0].content }}"]))
        with self.assertRaises(TextError):
            template_reasoning_format([None])
        own = ReasoningSplitter(ReasoningFormat("[[r]]", "[[/r]]"))
        self.assertEqual(own.push("[[r]]why[[/r]] so") + own.finish(), [("reasoning", "why"), ("text", "so")])
        with self.assertRaises(TextError):
            ReasoningSplitter("unknown")


class Decoder(ResidentChat):
    reply = "plan the reply\n</think>\n\nHello there"

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        self.prompt = self.text.tokenizer.decode(list(ids))
        reply = self.text.tokenizer.encode(self.reply, add_special_tokens=False)
        for token in reply:
            on_tokens([token])
        return reply, {"prefill_ms": 1.0, "decode_ms": 1.0, "first_token_ms": 1.0}


def request(**extra):
    return {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "limits": {"max_output_tokens": 64},
            "messages": [{"role": "user", "content": "hi"}], **extra}


class ResidentChatReasoningTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = write_files(Path(temp.name) / "model", llama_files(**{"chat_template.jinja": THINK_TEMPLATE}))
        self.plain = write_files(Path(temp.name) / "plain", llama_files())
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def test_reasoning_comes_apart_from_the_reply_and_streams_first(self):
        engine = Decoder(self.context, self.model)
        emitted = []
        response = engine.generate_stream(request(), threading.Event(), emitted.append)
        self.assertTrue(engine.prompt.rstrip().endswith("<think>"))
        self.assertEqual((response["assistant"], response["reasoning"]), ("Hello there", "plan the reply"))
        self.assertEqual("".join(e["reasoning"] for e in emitted if isinstance(e, dict)), "plan the reply")
        self.assertEqual("".join(e for e in emitted if isinstance(e, str)), "Hello there")
        first_text = next(i for i, e in enumerate(emitted) if isinstance(e, str))
        self.assertTrue(all(isinstance(e, dict) for e in emitted[:first_text]))

    def test_templates_without_markers_or_disabled_packages_keep_the_text(self):
        plain = Decoder(self.context, self.plain)
        self.assertNotIn("reasoning", plain.generate(request(), threading.Event()))

        class Off(Decoder):
            reasoning_format = False
        off = Off(self.context, self.model).generate(request(), threading.Event())
        self.assertEqual(off["assistant"], Decoder.reply)
        self.assertNotIn("reasoning", off)

    def test_unfinished_reasoning_from_resident_chat_is_a_valid_response(self):
        engine = Decoder(self.context, self.model)
        engine.reply = "still thinking"
        current = request(limits={"max_output_tokens": 64, "kv_tokens": 4096, "timeout_seconds": 10})
        emitted = []
        response = engine.generate_stream(current, threading.Event(), emitted.append)
        self.assertEqual(response["assistant"], "")
        self.assertEqual(response["reasoning"], "still thinking")
        self.assertEqual("".join(e["reasoning"] for e in emitted), "still thinking")
        self.assertEqual(reply(response, current)[0], {"content": "", "reasoning": "still thinking"})


class StreamReasoningTests(unittest.TestCase):
    def test_reasoning_pieces_publish_apart_and_final_reply_must_match(self):
        request_ids = dict(request_id="request", candidate_id="candidate", package_sha256="package")
        sent = []
        writer = StreamWriter(sent.append, "command", request_ids, Cancellation())
        texts, thoughts = [], []
        reader = StreamReader(request_ids, texts.append, publish_reasoning=thoughts.append)
        for chunk in ({"reasoning": "first "}, {"reasoning": "thought"}, "Answer"):
            writer.emit(chunk)
            reader.apply(sent[-1])
        self.assertEqual((thoughts, texts), (["first ", "thought"], ["Answer"]))
        with self.assertRaisesRegex(ValueError, "reasoning differs"):
            writer.finish({"assistant": "Answer", "reasoning": "other"})


# Replies "think:REASONING|REPLY" with the reasoning apart, streaming it before the reply.
ADAPTER = r'''
class ReasoningEngine(Engine):
    def _split(self, request):
        content = request['messages'][-1]['content']
        return content[len('think:'):].split('|', 1) if content.startswith('think:') else None

    def generate(self, request, cancellation):
        parts = self._split(request)
        if parts is None:
            return super().generate(request, cancellation)
        return dict(self.response(request, parts[1]), reasoning=parts[0])

    def generate_stream(self, request, cancellation, emit):
        parts = self._split(request)
        if parts is None:
            return super().generate_stream(request, cancellation, emit)
        emit({'reasoning': parts[0]})
        if parts[0] == 'hold':
            while True:
                cancellation.check()
                time.sleep(.01)
        if parts[1]:
            emit(parts[1])
        return dict(self.response(request, parts[1]), reasoning=parts[0])
def create_engine(context):
    return ReasoningEngine(context)
'''



class ServeReasoningTests(ServeFixture, unittest.TestCase):
    """crit serve gives clients reasoning_content, in JSON replies and as stream deltas before the content."""
    stream = True
    adapter_suffix = ADAPTER

    def setUp(self):
        super().setUp()
        served = ServedEngine(self.store.root, self.run_id, host_limit=1 << 24, device_limit=1024)
        self.addCleanup(served.close)
        served.open()
        self.server = Server(("127.0.0.1", 0), Api(served))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def chat(self, **options):
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=30)
        self.addCleanup(connection.close)
        connection.request("POST", "/v1/chat/completions",
                           json.dumps({"messages": [{"role": "user", "content": "think:why|Because."}], **options}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read().decode()

    def test_json_and_stream(self):
        status, body = self.chat()
        self.assertEqual(status, 200, body)
        message = json.loads(body)["choices"][0]["message"]
        self.assertEqual((message["content"], message["reasoning_content"]), ("Because.", "why"))
        status, body = self.chat(stream=True)
        self.assertEqual(status, 200, body)
        deltas = [json.loads(line[6:])["choices"][0]["delta"] for line in body.splitlines()
                  if line.startswith("data: {") and json.loads(line[6:]).get("choices")]
        kinds = [k for d in deltas for k in d if k in ("reasoning_content", "content") and d[k]]
        self.assertEqual("".join(d.get("reasoning_content") or "" for d in deltas), "why")
        self.assertEqual("".join(d.get("content") or "" for d in deltas), "Because.")
        self.assertEqual(kinds[0], "reasoning_content")

    def test_reasoning_only_completes_and_reuses_the_resident_host(self):
        served = self.server.api.engine
        # Establish the resident engine, then ensure every reasoning-only turn keeps it.
        self.assertEqual(self.chat()[0], 200)
        root, tree = served.runtime.root, served.runtime.tree
        loads = served.status()["engine_loads"]
        for maximum, finish in ((4, "length"), (128, "stop")):
            for streaming in (False, True):
                with self.subTest(maximum=maximum, stream=streaming):
                    status, body = self.chat(messages=[{"role": "user", "content": "think:still thinking|"}],
                                             max_tokens=maximum, stream=streaming)
                    self.assertEqual(status, 200, body)
                    if streaming:
                        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
                        self.assertFalse(any("error" in e for e in events), body)
                        choices = [e["choices"][0] for e in events if e.get("choices")]
                        self.assertEqual("".join(c["delta"].get("reasoning_content", "") for c in choices), "still thinking")
                        self.assertEqual("".join(c["delta"].get("content", "") for c in choices), "")
                        self.assertEqual(choices[-1]["finish_reason"], finish)
                        self.assertTrue(body.endswith("data: [DONE]\n\n"))
                    else:
                        result = json.loads(body)
                        self.assertEqual(result["choices"][0]["message"],
                                         {"role": "assistant", "content": "", "reasoning_content": "still thinking"})
                        self.assertEqual(result["choices"][0]["finish_reason"], finish)
                        self.assertEqual(result["usage"]["completion_tokens"], 4)
                    self.assertEqual(served.runtime.root, root)
                    self.assertIs(served.runtime.tree, tree)
                    self.assertIsNone(tree.process.poll())
                    self.assertEqual(served.status()["engine_loads"], loads)
        status, body = self.chat()
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "Because.")
        self.assertIs(served.runtime.tree, tree)

    def test_reasoning_consumer_exception_retires_producer_before_next_request(self):
        served = self.server.api.engine
        self.addCleanup(served._retire, "reasoning consumer regression cleanup")
        root, tree = served.runtime.root, served.runtime.tree
        error, seen = TypeError("reasoning consumer failed"), []

        def reasoning(text):
            seen.append(text)
            raise error

        with self.assertRaises(TypeError) as caught:
            served.generate([{"role": "user", "content": "think:hold|Because."}], 128,
                            threading.Event(), on_reasoning=reasoning)
        self.assertIs(caught.exception, error)
        self.assertEqual(seen, ["hold"])
        self.assertIsNone(served.runtime.tree)
        self.assertIsNotNone(tree.process.poll())
        status, body = self.chat()
        self.assertEqual(status, 200, body)
        message = json.loads(body)["choices"][0]["message"]
        self.assertEqual((message["content"], message["reasoning_content"]), ("Because.", "why"))
        self.assertNotEqual(served.runtime.root, root)


if __name__ == "__main__":
    unittest.main()
