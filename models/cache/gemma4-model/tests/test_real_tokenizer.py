"""Gemma 4's thought channel and tool calls through crit serve's reply path with Gemma 4's own tokenizer, where
`<|channel>`, `<channel|>`, `<|tool_call>` and `<tool_call|>` are special tokens (test_tools.py splits text markers):
ResidentChat over a scripted engine that returns the tokens of a reply the official template writes, its reasoning,
content and calls read back as they were, every chunking, streamed and whole. The tokenizer is a downloaded
checkpoint's (GEMMA4_TOKENIZER, else the folders in CANDIDATES that hold one); skipped where none is on this machine."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import types
import unittest

from crit_core.serving import ResidentChat
from crit_core.text import load_model_text

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gemma4_tools_real", HERE / "tools.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)

CANDIDATES = ("/llm/models/crit/google-gemma-4-26B-A4B-it", "/llm/models/crit/nvidia-Gemma-4-31B-IT-NVFP4")
FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "generation_config.json",
         "special_tokens_map.json")
FUNCTIONS = {"get_weather": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}
TOOLS = [{"type": "function", "function": {"name": n, "description": f"{n} tool", "parameters": p}}
         for n, p in FUNCTIONS.items()]
REQUEST = {"request_id": "r", "candidate_id": "c", "package_sha256": "p",
           "limits": {"max_output_tokens": 512, "kv_tokens": 4096, "timeout_seconds": 10},
           "messages": [{"role": "user", "content": "Weather in Paris?"}]}


def tokenizer_folder():
    """A folder holding a Gemma 4 tokenizer.json, or None."""
    given = os.environ.get("GEMMA4_TOKENIZER")
    for root in ([given] if given else []) + list(CANDIDATES):
        root = Path(root)
        if not root.is_dir():
            continue
        for folder in [root] + sorted(p for p in root.iterdir() if p.is_dir()):
            if (folder / "tokenizer.json").is_file() and (folder / "chat_template.jinja").is_file():
                return folder
    return None


class ScriptedGemma(ResidentChat):
    """ResidentChat with Gemma 4's formats over a scripted engine: generate_tokens returns `output` in chunks."""
    tool_format = tools.TOOL_FORMAT
    reasoning_format = tools.REASONING_FORMAT
    output = ()
    chunk = 1

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        for start in range(0, len(self.output), self.chunk):
            on_tokens(self.output[start:start + self.chunk])
        return list(self.output), {"prefill_ms": 1., "decode_ms": 1., "first_token_ms": 1.}


@unittest.skipIf(tokenizer_folder() is None, "no Gemma 4 tokenizer on this machine (GEMMA4_TOKENIZER)")
class RealTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        source, folder = tokenizer_folder(), Path(cls.temp.name)
        for name in FILES:   # the text files only: the weights stay where they are
            if (source / name).is_file():
                shutil.copy2(source / name, folder / name)
        cls.folder = folder
        cls.text = load_model_text(folder)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def engine(self, reply):
        engine = ScriptedGemma(types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096}), self.folder)
        engine.output = tuple(self.text.tokenizer.encode(reply, add_special_tokens=False))
        stops = set(engine.stop_ids)
        self.assertFalse(stops & set(engine.output), "the scripted reply holds no stop token")
        return engine

    def test_the_channel_and_call_markers_are_single_special_tokens(self):
        backend = self.text.tokenizer.backend
        for marker in (tools.REASONING_FORMAT.opening, tools.REASONING_FORMAT.closing, tools.OPEN, tools.CLOSE):
            with self.subTest(marker=marker):
                ids = backend.encode(marker, add_special_tokens=False).ids
                self.assertEqual(len(ids), 1)
                self.assertIn(ids[0], self.text.tokenizer.special_ids)

    def check(self, engine, request, content, reasoning, calls):
        for chunk in (1, 3, 4096):
            for streaming in (False, True):
                with self.subTest(chunk=chunk, streaming=streaming):
                    engine.chunk, emitted = chunk, []
                    response = (engine.generate_stream(request, threading.Event(), emitted.append) if streaming
                                else engine.generate(request, threading.Event()))
                    self.assertEqual(response["assistant"], content)
                    self.assertEqual(response.get("reasoning", ""), reasoning)
                    self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"]))
                                      for c in response.get("tool_calls", [])], calls)
                    if streaming:
                        self.assertEqual("".join(e for e in emitted if isinstance(e, str)), content)

    def test_reasoning_then_a_call_split_and_parsed(self):
        reply = ('<|channel>thought\nThe user wants the weather.\n<channel|>'
                 '<|tool_call>call:get_weather{city:<|"|>Paris, France<|"|>,days:3}<tool_call|>')
        self.check(self.engine(reply), REQUEST | {"tools": TOOLS, "tool_choice": "auto"}, "",
                   "The user wants the weather.", [("get_weather", {"city": "Paris, France", "days": 3})])

    def test_reasoning_then_an_answer_without_tools(self):
        reply = "<|channel>thought\nIt is a greeting.\n<channel|>Hello! How can I help?"
        self.check(self.engine(reply), REQUEST | {"tool_choice": "none"}, "Hello! How can I help?",
                   "It is a greeting.", [])

    def test_an_answer_without_a_channel(self):
        self.check(self.engine("Sunny, 21 degrees."), REQUEST | {"tool_choice": "none"}, "Sunny, 21 degrees.", "", [])


if __name__ == "__main__":
    unittest.main()
