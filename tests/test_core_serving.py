"""crit_core.serving.ResidentChat and the package check that keeps resident adapters complete (offline)."""
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))

from core_text_fixtures import llama_files, write_files  # noqa: E402
from test_core_tools import BRACKET, XML_TEMPLATE  # noqa: E402
from crit.run.store import serving_gaps  # noqa: E402
from crit_core.serving import ResidentChat, capabilities  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
TEMPLATE = XML_TEMPLATE.replace("{%- for message in messages %}",
                                "{{ tools | tojson if tools else '' }}{%- for message in messages %}"
                                "{%- if message.role != 'assistant' %}{{ message.role }}:{{ message.content }};{%- endif %}", 1)


class Decoder(ResidentChat):
    sampling = ("temperature", "seed")

    def __init__(self, context, model):
        super().__init__(context, model, stop_ids=[1])
        self.calls = []

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        self.calls.append((list(ids), max_new_tokens, dict(sampling)))
        if cancelled():
            return None, {}
        reply = self.text.tokenizer.encode("Hi there", add_special_tokens=False)
        for token in reply + [1]:
            on_tokens([token])
        return reply + [1], {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}


def request(**extra):
    return {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "limits": {"max_output_tokens": 8},
            "messages": [{"role": "user", "content": "hello"}], **extra}


class ResidentChatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = write_files(Path(temp.name) / "model", llama_files(**{"chat_template.jinja": TEMPLATE}))
        self.plain = write_files(Path(temp.name) / "plain", llama_files())
        self.context = types.SimpleNamespace(configuration={"batch": 1}, limits={"kv_tokens": 4096})

    def test_capabilities_follow_the_template_and_the_decoder(self):
        engine = Decoder(self.context, self.model)
        self.assertEqual(capabilities(engine.text, Decoder.sampling),
                         ["chat-json-v1", "chat-resident-v1", "chat-stream-v1", "chat-config-v1",
                          "chat-tools-v1", "chat-tool-stream-v1", "chat-sampling-temperature-v1",
                          "chat-sampling-seed-v1"])
        self.assertNotIn("chat-tools-v1", capabilities(Decoder(self.context, self.plain).text))
        with self.assertRaises(ValueError):
            capabilities(engine.text, ["warmth"])

    def test_reply_stream_sampling_tools_and_telemetry(self):
        engine = Decoder(self.context, self.model)
        emitted = []
        response = engine.generate_stream(request(sampling={"temperature": 0}), threading.Event(), emitted.append)
        self.assertEqual((response["assistant"], "".join(emitted)), ("Hi there", "Hi there"))
        self.assertEqual(engine.calls[-1][1:], (8, {"temperature": 0}))
        produced = len(engine.text.tokenizer.encode("Hi there", add_special_tokens=False)) + 1
        self.assertEqual(response["telemetry"]["output_tokens"], produced)
        self.assertEqual(response["telemetry"]["decode_tokens"], produced - 1)
        engine.generate(request(tools=TOOLS, tool_choice="auto"), threading.Event())
        self.assertIn("read_file", engine.text.tokenizer.decode(engine.calls[-1][0]))
        engine.generate(request(tools=TOOLS, tool_choice="none"), threading.Event())
        self.assertNotIn("read_file", engine.text.tokenizer.decode(engine.calls[-1][0]))
        self.assertEqual(engine.effective_configuration(), {"batch": 1})
        with self.assertRaisesRegex(ValueError, r"sampling field\(s\): top_p"):
            engine.generate(request(sampling={"top_p": .5}), threading.Event())
        with self.assertRaisesRegex(ValueError, "exceeds the context"):
            engine.generate(dict(request(), limits={"max_output_tokens": 5000}), threading.Event())
        stop = threading.Event()
        stop.set()
        with self.assertRaises(RuntimeError):
            engine.generate(request(), types.SimpleNamespace(is_set=stop.is_set, check=lambda: None))


class XmlDecoder(Decoder):
    """A model whose chat template teaches a syntax crit registers: ResidentChat parses it with no package code."""
    reply = ("Let me read it.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n"
             "</function>\n</tool_call>")

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        reply = self.text.tokenizer.encode(self.reply, add_special_tokens=False)
        for token in reply + [1]:
            on_tokens([token])
        return reply + [1], {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}


class TemplateToolFormatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = write_files(Path(temp.name) / "model", llama_files(**{"chat_template.jinja": TEMPLATE}))
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def test_calls_in_the_templates_registered_format_come_back_parsed(self):
        engine = XmlDecoder(self.context, self.model)
        emitted = []
        response = engine.generate_stream(request(tools=TOOLS), threading.Event(), emitted.append)
        self.assertEqual(response["assistant"], "Let me read it.")
        self.assertEqual([c["function"] for c in response["tool_calls"]],
                         [{"name": "read_file", "arguments": '{"path": "a.py"}'}])
        self.assertEqual("".join(e for e in emitted if isinstance(e, str)), "Let me read it.")
        self.assertEqual([e["tool_calls"][0]["index"] for e in emitted if isinstance(e, dict)], [0])
        for kept in (engine.generate(request(tools=TOOLS, tool_choice="none"), threading.Event()),
                     engine.generate(request(), threading.Event())):
            self.assertEqual(kept["assistant"], XmlDecoder.reply)
            self.assertNotIn("tool_calls", kept)
        with self.assertRaisesRegex(ValueError, "tool_choice must be auto or none"):
            engine.generate(request(tools=TOOLS, tool_choice="required"), threading.Event())


class BracketDecoder(Decoder):
    """A package that parses its model's own call syntax (crit registers none for it)."""
    tool_format = BRACKET
    reply = 'Sure. [[call read_file {"path": "a.py"}]]'

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        self.calls.append((list(ids), max_new_tokens, dict(sampling)))
        reply = self.text.tokenizer.encode(self.reply, add_special_tokens=False)
        for token in reply + [1]:
            on_tokens([token])
        return reply + [1], {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}


class PackageToolFormatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.plain = write_files(Path(temp.name) / "plain", llama_files())
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def test_the_package_parses_calls_and_declares_chat_tools(self):
        engine = BracketDecoder(self.context, self.plain)
        declared = capabilities(engine.text, BracketDecoder.sampling, tool_format=BracketDecoder.tool_format)
        self.assertIn("chat-tools-v1", declared)
        self.assertIn("chat-tool-stream-v1", declared)
        emitted = []
        response = engine.generate_stream(request(tools=TOOLS, tool_choice="auto"), threading.Event(), emitted.append)
        self.assertEqual(response["assistant"], "Sure.")
        self.assertEqual([c["function"]["name"] for c in response["tool_calls"]], ["read_file"])
        self.assertEqual(emitted[-1]["tool_calls"][0]["function"]["arguments"], '{"path": "a.py"}')
        plain = engine.generate(request(tools=TOOLS, tool_choice="none"), threading.Event())
        self.assertEqual((plain["assistant"], "tool_calls" in plain), (BracketDecoder.reply, False))
        self.assertNotIn("tool_calls", engine.generate(request(), threading.Event()))
        with self.assertRaisesRegex(ValueError, "tool_choice must be auto or none"):
            engine.generate(request(tools=TOOLS, tool_choice="required"), threading.Event())


class PackageGapTests(unittest.TestCase):
    def test_resident_packages_need_temperature_and_tools_when_the_template_teaches_them(self):
        with tempfile.TemporaryDirectory() as temp:
            tools = write_files(Path(temp) / "tools", llama_files(**{"chat_template.jinja": TEMPLATE}))
            plain = write_files(Path(temp) / "plain", llama_files())
            resident = ["chat-json-v1", "chat-resident-v1"]
            self.assertEqual(serving_gaps(["prefill", "decode"], tools), [])
            self.assertEqual(serving_gaps(resident + ["chat-sampling-temperature-v1"], plain), [])
            self.assertEqual(len(serving_gaps(resident, plain)), 1)
            gaps = serving_gaps(resident + ["chat-sampling-temperature-v1"], tools)
            self.assertEqual(len(gaps), 1)
            self.assertIn("chat-tools-v1", gaps[0])
            for parsed in ("chat-tools-v1", "chat-tool-template-v1"):   # the second: packages sealed before
                self.assertEqual(serving_gaps(resident + ["chat-sampling-temperature-v1", parsed], tools), [])
            self.assertEqual(serving_gaps(resident + ["chat-sampling-temperature-v1"], Path(temp) / "model.gguf"), [])


if __name__ == "__main__":
    unittest.main()
