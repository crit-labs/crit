"""Image transport and engine-owned preparation, without a model or device dependency."""
import base64
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest

from core_text_fixtures import llama_files, write_files
from test_core_serving import Decoder, request
from test_chat_residency import ServeFixture
from crit_api.server import Api, Server, conversation
from crit_api.options import RequestError
from crit_core.serving import PreparedPrompt, capabilities, chat_prompt

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj", "detail": "high"}}
TEMPLATE = """{% for m in messages %}{% for p in m.content %}{{ p.text if p.type == 'text' else '<image>' }}{% endfor %}{% endfor %}"""


class ImageDecoder(Decoder):
    images = True

    @contextmanager
    def prepare_prompt(self, request, cancellation):
        self.released = False
        try:
            ids = self.prompt_ids(request)
            messages, _ = chat_prompt(request, images=True)
            images = [p["image"] for m in messages for p in m["content"] if p["type"] == "image"]
            yield PreparedPrompt(ids, images, len(ids) + 17 * len(images))
        finally:
            self.released = True

    def generate_prepared(self, prompt, *args):
        self.images_seen = prompt.inputs
        return self.generate_tokens(prompt.ids, *args)


class ImageContractTests(unittest.TestCase):
    def test_parts_template_context_accounting_and_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            model = write_files(Path(folder), llama_files(**{"chat_template.jinja": TEMPLATE}))
            context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})
            engine = ImageDecoder(context, model)
            messages = conversation([{"role": "user", "content": [{"type": "text", "text": "before"}, IMAGE]},
                                     {"role": "user", "content": "after"}], ["chat-images-v1"])
            req = request(messages=messages, sampling={"temperature": 0})
            self.assertEqual(engine.text.render(chat_prompt(req, images=True)[0]), "before<image>\n\nafter")
            chunks = []
            response = engine.generate_stream(req, threading.Event(), chunks.append)
            self.assertEqual(response["assistant"], "".join(chunks))
            self.assertEqual(engine.images_seen, [IMAGE["image_url"]["url"]])
            self.assertEqual(response["telemetry"]["prompt_tokens"], len(engine.calls[-1][0]) + 17)
            self.assertTrue(engine.released)
            self.assertIn("chat-images-v1", capabilities(engine.text, images=True))
            with self.assertRaisesRegex(ValueError, "exceeds the context"):
                engine.generate(dict(req, limits={"max_output_tokens": 4096}), threading.Event())
            self.assertTrue(engine.released)
            stop = types.SimpleNamespace(is_set=lambda: True, check=lambda: None)
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                engine.generate(req, stop)
            self.assertTrue(engine.released)

    def test_invalid_or_unsupported_images_are_rejected(self):
        for capabilities_, part in [
                ([], IMAGE), (["chat-images-v1"], 7),
                (["chat-images-v1"], {"type": "image_url", "image_url": {"url": "file:///private.png"}}),
                (["chat-images-v1"], {"type": "image_url", "image_url": {"url": "data:image/png;base64,!"}}),
                (["chat-images-v1"], {"type": "image_url", "image_url": {"url": "https://example.com/a", "detail": 1}})]:
            with self.subTest(part=part), self.assertRaises(RequestError):
                conversation([{"role": "user", "content": [part]}], capabilities_)

    def test_tool_history_and_multiple_images_preserve_order(self):
        messages = [{"role": "user", "content": [IMAGE, {"type": "text", "text": "compare"}, IMAGE]},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "c", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c", "content": "done"}]
        normalized = conversation(messages, ["chat-images-v1", "chat-tools-v1"])
        rendered, _ = chat_prompt(request(messages=normalized), images=True)
        self.assertEqual([p["type"] for p in rendered[0]["content"]], ["image", "text", "image"])
        self.assertEqual(rendered[-1]["tool_call_id"], "c")


IMAGE_ADAPTER = r'''
_plain_generate = Engine.generate
def image_generate(self, request, cancellation):
    parts = request["messages"][-1]["content"]
    if isinstance(parts, str):
        return _plain_generate(self, request, cancellation)
    value = {"types": [p["type"] for p in parts],
             "urls": [len(p["image_url"]["url"]) for p in parts if p["type"] == "image_url"]}
    return self.response(request, json.dumps(value))
Engine.generate = image_generate
def image_stream(self, request, cancellation, emit):
    response = self.generate(request, cancellation)
    emit(response["assistant"])
    return response
Engine.generate_stream = image_stream
'''


class ImageHttpTests(ServeFixture, unittest.TestCase):
    stream = True
    extra_capabilities = ("chat-images-v1",)
    adapter_suffix = IMAGE_ADAPTER

    def test_images_cross_http_and_resident_records_in_both_response_modes(self):
        served = self.engine()
        server = Server(("127.0.0.1", 0), Api(served))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        # Cross the old 1 MiB record limit with a bounded, valid data URL.
        url = "data:image/png;base64," + base64.b64encode(b"x" * 800000).decode()
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                client = http.client.HTTPConnection(*server.server_address, timeout=20)
                self.addCleanup(client.close)
                parts = [{"type": "text", "text": "describe"}, {"type": "image_url", "image_url": {"url": url}}]
                client.request("POST", "/v1/chat/completions", json.dumps(
                    {"messages": [{"role": "user", "content": parts}], "stream": streaming, "max_tokens": 128}),
                    {"Content-Type": "application/json"})
                response = client.getresponse()
                wire = response.read()
                self.assertEqual(response.status, 200, wire)
                if streaming:
                    frames = [json.loads(line[6:]) for line in wire.splitlines()
                              if line.startswith(b"data: ") and line != b"data: [DONE]"]
                    text = "".join(f["choices"][0]["delta"].get("content", "") for f in frames)
                else:
                    text = json.loads(wire)["choices"][0]["message"]["content"]
                self.assertEqual(json.loads(text), {"types": ["text", "image_url"], "urls": [len(url)]})
        self.assertFalse((served.runtime.root / "requests").exists())
