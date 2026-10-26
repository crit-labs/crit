"""Chat templates crit ships for model families whose checkpoints carry none: DeepSeek V4's port of the official
encoder renders the encoder's own golden outputs (tests/fixtures/deepseek_v4), and the text layer loads it by
config.json model_type (offline)."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "tests"): sys.path.insert(0, str(ROOT / _folder))

import core_text_fixtures as fx  # noqa: E402
from crit_core.files import MappingFiles  # noqa: E402
from crit_core.text import (ChatTemplate, ModelTextError, TemplateError, TextError, chat_template_sources,  # noqa: E402
                            load_model_text, shipped_template)

GOLDENS = ROOT / "tests/fixtures/deepseek_v4"
THINKING = {1: True, 2: True, 3: True, 4: False}   # the modes test_encoding_dsv4.py encodes each case in
BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"


def deepseek_v4():
    name, source = shipped_template("deepseek_v4")
    return ChatTemplate(source, name=name, clock=None)


def as_mappings(messages):
    """The messages with each tool call's arguments parsed from the JSON text the encoder takes into the mapping a
    chat template takes (as crit_core.text.openai_messages and encode_arguments_to_dsml parse them)."""
    messages = copy.deepcopy(messages)
    for message in messages:
        for call in message.get("tool_calls") or []:
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return messages


def golden(case):
    data = json.loads((GOLDENS / f"test_input_{case}.json").read_text(encoding="utf-8"))
    messages, tools = (data["messages"], data["tools"]) if isinstance(data, dict) else (data, None)
    return as_mappings(messages), tools, (GOLDENS / f"test_output_{case}.txt").read_text(encoding="utf-8")


def mapping(files):
    return MappingFiles({name: fx.file_bytes(value) for name, value in files.items()})


def model_files(**extra):
    """A small model without a chat template whose config.json names DeepSeek V4's model_type."""
    files = fx.llama_files(**extra)
    del files["chat_template.jinja"]
    files["config.json"] = {"model_type": "deepseek_v4", "eos_token_id": 1}
    return files


class DeepSeekV4TemplateTests(unittest.TestCase):
    def test_the_template_renders_the_official_encoders_goldens_exactly(self):
        template = deepseek_v4()
        for case, thinking in THINKING.items():
            messages, tools, expected = golden(case)
            # Each golden ends with an assistant turn, after which the encoder writes nothing: the generation
            # prompt changes nothing. Case 1's tools go on its system message, as the official test puts them,
            # or as `tools`, which the template gives the first system message.
            ways = [(messages, None)]
            if tools:
                ways = [([{**messages[0], "tools": tools}] + messages[1:], None), (messages, tools)]
            for given, top in ways:
                for prompt in (False, True):
                    with self.subTest(case=case, tools=top is not None, add_generation_prompt=prompt):
                        self.assertEqual(template.render(given, tools=top, add_generation_prompt=prompt,
                                                         thinking=thinking), expected)
                        self.assertEqual(template.render(given, tools=top, add_generation_prompt=prompt,
                                                         enable_thinking=thinking), expected)

    def test_the_generation_prompt_opens_the_reply_after_a_user_turn(self):
        template = deepseek_v4()
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        self.assertEqual(template.render(messages, add_generation_prompt=True), BOS + "S<｜User｜>U<｜Assistant｜></think>")
        self.assertEqual(template.render(messages, add_generation_prompt=True, enable_thinking=True),
                         BOS + "S<｜User｜>U<｜Assistant｜><think>")
        self.assertEqual(template.render(messages, add_generation_prompt=False), BOS + "S<｜User｜>U")
        self.assertEqual(template.render(messages, add_generation_prompt=True, thinking=True, reasoning_effort="max"),
                         BOS + "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very "
                         "thorough in your thinking and comprehensively decompose the problem to resolve the root "
                         "cause, rigorously stress-testing your logic against all potential paths, edge cases, and "
                         "adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting "
                         "every intermediate step, considered alternative, and rejected hypothesis to ensure "
                         "absolutely no assumption is left unchecked.\n\nS<｜User｜>U<｜Assistant｜><think>")

    def test_tool_results_join_the_user_turn_in_the_order_of_their_calls(self):
        calls = [{"id": "a", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}},
                 {"id": "b", "type": "function", "function": {"name": "g", "arguments": {"y": "two"}}}]
        messages = [{"role": "user", "content": "U"}, {"role": "assistant", "content": "", "tool_calls": calls},
                    {"role": "tool", "tool_call_id": "b", "content": "B"},
                    {"role": "tool", "tool_call_id": "a", "content": "A"}, {"role": "user", "content": "more"}]
        self.assertEqual(deepseek_v4().render(messages, add_generation_prompt=True),
                         BOS + "<｜User｜>U<｜Assistant｜></think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n"
                         "<｜DSML｜parameter name=\"x\" string=\"false\">1</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
                         "<｜DSML｜invoke name=\"g\">\n"
                         "<｜DSML｜parameter name=\"y\" string=\"true\">two</｜DSML｜parameter>\n"
                         "</｜DSML｜invoke>\n</｜DSML｜tool_calls>" + EOS
                         + "<｜User｜><tool_result>A</tool_result>\n\n<tool_result>B</tool_result>\n\nmore"
                         "<｜Assistant｜></think>")

    def test_what_the_encoder_refuses_the_template_refuses(self):
        template = deepseek_v4()
        call = {"type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}
        refused = (([{"role": "user", "content": "U"}, {"role": "assistant", "tool_calls": [call]}], {},
                    "arguments from a mapping"),
                   ([{"role": "user", "content": "U"}], {"reasoning_effort": "low"},
                    'reasoning_effort "high" or "max"'),
                   ([{"role": "narrator", "content": "U"}], {}, "no narrator role"),
                   ([{"role": "developer", "content": ""}], {}, "content in a developer message"),
                   ([{"role": "user", "content": "U", "task": "summary"}], {}, "no task summary"))
        for messages, variables, message in refused:
            with self.subTest(message=message), self.assertRaisesRegex(TemplateError, message):
                template.render(messages, add_generation_prompt=True, **variables)


class ShippedTemplateTests(unittest.TestCase):
    def test_crit_ships_a_template_for_deepseek_v4_alone(self):
        name, source = shipped_template("deepseek_v4")
        self.assertEqual(name, "deepseek-v4.jinja")
        self.assertIn("encoding_dsv4.py", source)
        for model_type in ("llama", None, 4):
            self.assertIsNone(shipped_template(model_type))

    def test_a_deepseek_v4_checkpoint_without_a_template_renders_crits(self):
        text = load_model_text(mapping(model_files()), clock=None)
        self.assertEqual(text.template_origin, "crit")
        self.assertEqual(list(text.templates), ["default"])
        self.assertEqual(text.templates["default"].sha256, deepseek_v4().sha256)
        self.assertEqual(text.render([{"role": "user", "content": "Hi"}], add_generation_prompt=True),
                         BOS + "<｜User｜>Hi<｜Assistant｜></think>")
        # The template is no file of the model's, so the identity names it.
        self.assertEqual(text.identity.row()["template"], {"origin": "crit",
                                                           "sha256": {"default": deepseek_v4().sha256}})
        other = load_model_text(mapping({**model_files(), "config.json": {"model_type": "llama", "eos_token_id": 1}}),
                                clock=None)
        self.assertEqual((other.template_origin, len(other.templates), other.identity.template), (None, 0, None))
        with self.assertRaisesRegex(ModelTextError, "no chat template"):
            other.render([{"role": "user", "content": "Hi"}])

    def test_a_given_template_renders_a_model_without_one_and_never_replaces_the_models_own(self):
        given = "{{ bos_token }}{% for m in messages %}[{{ m.content }}]{% endfor %}"
        text = load_model_text(mapping(model_files()), clock=None, chat_template=given)
        self.assertEqual(text.template_origin, "given")
        self.assertEqual(text.render([{"role": "user", "content": "Hi"}]), "<|begin_of_text|>[Hi]")
        self.assertNotEqual(text.identity.sha256, load_model_text(mapping(model_files()), clock=None).identity.sha256)
        own = mapping(fx.llama_files())
        with self.assertRaisesRegex(ModelTextError, "carries its own chat template"):
            load_model_text(own, clock=None, chat_template=given)
        with self.assertRaisesRegex(TextError, "template source string"):
            chat_template_sources(own, chat_template=b"bytes")
        # A model's own template keeps the identity it had: no template entry joins it.
        text = load_model_text(mapping(fx.llama_files()), clock=None)
        self.assertEqual((text.template_origin, text.identity.template), ("model", None))
        self.assertNotIn("template", text.identity.row())


if __name__ == "__main__":
    unittest.main()
