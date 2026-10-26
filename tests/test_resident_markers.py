"""The resident adapter must not reinterpret ordinary output tokens as special reasoning/tool delimiters."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
for folder in ("core/python", "api", "cli", "../crit-eval"):
    sys.path.insert(0, str(ROOT / folder))

from core_text_fixtures import llama_files, write_files  # noqa: E402
from crit_api.engine import reply  # noqa: E402
from crit_core.serving import ResidentChat  # noqa: E402
from crit_core.text import ReasoningFormat  # noqa: E402
from crit_core.text.tools import _format  # noqa: E402
from test_core_markers import CALL, TOOLS, encoded, marked_tokenizer  # noqa: E402
from test_core_tools import rebuilt  # noqa: E402

TEMPLATE = ("{% for message in messages %}{{ message.role }}:{{ message.content }};{% endfor %}"
            "{% if add_generation_prompt %}assistant:<think>{% endif %}")
REQUEST = {"request_id": "r", "candidate_id": "c", "package_sha256": "p",
           "limits": {"max_output_tokens": 512, "kv_tokens": 4096, "timeout_seconds": 10},
           "messages": [{"role": "user", "content": "hi"}]}


class TokenEngine(ResidentChat):
    reasoning_format = "think-tags"
    tool_format = "json-object"
    output = ()
    chunk = 1
    prompt_override = None

    def prompt_ids(self, request):
        return self.prompt_override if self.prompt_override is not None else super().prompt_ids(request)

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        for start in range(0, len(self.output), self.chunk):
            on_tokens(self.output[start:start + self.chunk])
        return list(self.output), {"prefill_ms": 1., "decode_ms": 1., "first_token_ms": 1.}


class ResidentMarkerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tok = marked_tokenizer()
        self.model = write_files(Path(temp.name), llama_files(**{
            "tokenizer.json": self.tok.backend.to_str(), "chat_template.jinja": TEMPLATE}))
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def engine(self, output, *, thinking=True, prompt=None, cls=TokenEngine):
        engine = cls(self.context, self.model)
        engine.output, engine.prompt_override = output, prompt
        if not thinking:
            engine.reasoning_format = False
        return engine

    def check(self, engine, expected_content, expected_reasoning="", expected_calls=(), **options):
        request = REQUEST | options
        request.setdefault("tool_choice", "auto" if request.get("tools") else "none")
        for chunk in (1, 3, 4096):
            for streaming in (False, True):
                with self.subTest(chunk=chunk, streaming=streaming):
                    engine.chunk = chunk
                    emitted = []
                    response = (engine.generate_stream(request, threading.Event(), emitted.append)
                                if streaming else engine.generate(request, threading.Event()))
                    self.assertEqual(response["assistant"], expected_content)
                    self.assertEqual(response.get("reasoning", ""), expected_reasoning)
                    calls = [{"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
                             for c in response.get("tool_calls", [])]
                    self.assertEqual(calls, list(expected_calls))
                    self.assertEqual(response["telemetry"]["output_tokens"], len(engine.output))
                    # The actual serve response validator accepts the result, including reasoning-only completions.
                    reply(response, request)
                    if streaming:
                        self.assertEqual("".join(e for e in emitted if isinstance(e, str)), expected_content)
                        self.assertEqual("".join(e.get("reasoning", "") for e in emitted if isinstance(e, dict)),
                                         expected_reasoning)
                        streamed = [call for e in emitted if isinstance(e, dict) for call in e.get("tool_calls", [])]
                        self.assertEqual(rebuilt(streamed), {i: (c["id"], c["function"]["name"], c["function"]["arguments"])
                                                             for i, c in enumerate(response.get("tool_calls", []))})

    def test_identical_reasoning_closer_spelling_has_different_meaning_by_id(self):
        text = "Quote </think> literally; still thinking."
        for literal in (False, True):
            output = encoded(self.tok, text, literal)
            self.assertEqual(self.tok.decode(output), text)
            self.assertEqual(self.tok.token_to_id("</think>") not in output, literal)
            self.check(self.engine(output), "" if literal else "literally; still thinking.",
                       text if literal else "Quote")

    def test_identical_tool_spelling_only_creates_calls_for_actual_control_ids(self):
        text = "<tool_call>" + CALL + "</tool_call>"
        for literal in (False, True):
            output = encoded(self.tok, text, literal)
            self.assertEqual(self.tok.decode(output), text)
            self.check(self.engine(output, thinking=False), text if literal else "",
                       expected_calls=[] if literal else [json.loads(CALL)], tools=TOOLS)

    def test_generated_reasoning_opening_checks_identity(self):
        prompt = encoded(self.tok, "assistant:")
        text = "<think>why</think>answer"
        for literal in (False, True):
            self.check(self.engine(encoded(self.tok, text, literal), prompt=prompt),
                       text if literal else "answer", "" if literal else "why")

    def test_prompt_opening_checks_identity_and_arbitrary_trailing_whitespace(self):
        for literal in (False, True):
            prompt = encoded(self.tok, "assistant:<think>" + " \n" * 32, literal)
            self.check(self.engine(encoded(self.tok, "still thinking"), prompt=prompt),
                       "still thinking" if literal else "", "" if literal else "still thinking")
        # An earlier real opener does not validate a literal lookalike at the end of a prompt.
        prompt = encoded(self.tok, "<think>history") + encoded(self.tok, "assistant:<think>", True)
        self.check(self.engine(encoded(self.tok, "answer"), prompt=prompt), "answer")

    def test_reasoning_then_literal_call_then_real_call_with_literal_closer_in_arguments(self):
        body = json.dumps({"name": "lookup", "arguments": {"quote": "</tool_call>", "word": "日本😀"}}, ensure_ascii=False)
        literal = "<tool_call>" + CALL + "</tool_call>"
        output = (encoded(self.tok, literal, True) + encoded(self.tok, "</think>")
                  + encoded(self.tok, literal, True) + encoded(self.tok, " <tool_call>")
                  + encoded(self.tok, body, True) + encoded(self.tok, "</tool_call>"))
        self.check(self.engine(output), literal, literal, [json.loads(body)], tools=TOOLS)

    def test_tool_choice_none_preserves_markers_and_no_tools_never_parses_literal_calls(self):
        text = "<tool_call>" + CALL + "</tool_call>"
        for literal in (False, True):
            self.check(self.engine(encoded(self.tok, text, literal), thinking=False), text,
                       tools=TOOLS, tool_choice="none")
        self.check(self.engine(encoded(self.tok, text, True), thinking=False), text)

    def test_explicit_text_formats_keep_spelling_semantics_even_with_special_tokens(self):
        class TextEngine(TokenEngine):
            reasoning_format = ReasoningFormat("<think>", "</think>", delimiters="text")
            tool_format = _format("json-object")._replace(delimiters="text")

        output = encoded(self.tok, "why</think><tool_call>" + CALL + "</tool_call>", True)
        self.check(self.engine(output, cls=TextEngine), "", "why", [json.loads(CALL)], tools=TOOLS)
        # Prompt suffixes follow the text override too.
        prompt = encoded(self.tok, "assistant:<think>", True)
        self.check(self.engine(encoded(self.tok, "why"), prompt=prompt, cls=TextEngine), "", "why")

    def test_a_call_the_output_limit_cuts_is_kept_and_one_the_model_ends_fails(self):
        output = (encoded(self.tok, "Look. <tool_call>")
                  + encoded(self.tok, '{"name":"lookup","arguments":{"quote":"</tool_call>"', True))
        engine, request = self.engine(output, thinking=False), REQUEST | {"tools": TOOLS, "tool_choice": "auto"}
        with self.assertRaisesRegex(ValueError, "ended its reply inside its call to lookup"):
            engine.generate(request, threading.Event())
        request["limits"] = REQUEST["limits"] | {"max_output_tokens": len(output)}
        response = engine.generate(request, threading.Event())
        self.assertEqual((response["assistant"], [c["function"] for c in response["tool_calls"]]),
                         ("Look.", [{"name": "lookup", "arguments": '{"quote": "</tool_call>"'}]))
        reply(response, request)
        # Ended before the call's name: the text stays.
        output = encoded(self.tok, "Look. <tool_call>") + encoded(self.tok, '{"na', True)
        self.check(self.engine(output, thinking=False), "Look. <tool_call>" + '{"na', tools=TOOLS)

    def test_utf8_tail_preserves_content(self):
        output = encoded(self.tok, "why 😀</think>answer 日本")
        self.check(self.engine(output), "answer 日本", "why 😀")


if __name__ == "__main__":
    unittest.main()
