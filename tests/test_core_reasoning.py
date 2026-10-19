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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "cli", "eval", "tests"): sys.path.insert(0, str(ROOT / _folder))

from core_text_fixtures import llama_files, write_files  # noqa: E402
from crit_core.serving import ResidentChat  # noqa: E402
from crit_core.text import TextError  # noqa: E402
from crit_core.text.reasoning import ReasoningFormat, ReasoningSplitter, template_reasoning_format  # noqa: E402
from crit_api.host import Cancellation  # noqa: E402
from crit_api.stream import StreamReader, StreamWriter  # noqa: E402
from crit.run.records import write_json  # noqa: E402
import test_api  # noqa: E402,F401  (puts the checkout's packages on sys.path)
from test_chat_residency import ServeFixture  # noqa: E402
from crit_api.engine import ServedEngine  # noqa: E402
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


class StreamReasoningTests(unittest.TestCase):
    def test_reasoning_snapshots_publish_deltas_and_final_reply_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request_ids = dict(request_id="request", candidate_id="candidate", package_sha256="package")
            writer = StreamWriter(root / "stream.json", dict(command_id="command", token="owner"), request_ids,
                                  Cancellation(root / "cancel"))
            texts, thoughts = [], []
            reader = StreamReader(root / "stream.json", "command", "owner", request_ids, texts.append,
                                  publish_reasoning=thoughts.append)
            with patch("crit_api.stream.time.monotonic", side_effect=range(10, 1000)), \
                    patch("crit_api.stream.write_json", wraps=write_json):
                for chunk in ({"reasoning": "first "}, {"reasoning": "thought"}, "Answer"):
                    writer.emit(chunk)
                    reader.poll()
            self.assertEqual(("".join(thoughts), texts[-1]), ("first thought", "Answer"))
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


if __name__ == "__main__":
    unittest.main()
