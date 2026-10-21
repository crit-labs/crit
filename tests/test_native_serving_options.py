"""Request sampling overrides reach the native call without changing resident defaults."""
import threading
import unittest
from unittest.mock import Mock

from core_text_fixtures import byte_level_json
from crit_core.native import ResidentEngine
from crit_core.text import TextStream, Tokenizer


class NativeServingOptionsTests(unittest.TestCase):
    def engine(self):
        engine = ResidentEngine.__new__(ResidentEngine)
        engine.configuration = {"temperature": .8}
        engine.capacity, engine.stop_ids = 2048, set()
        engine.facts = {"sampling": {"temperature": True, "seed": True, "penalties": True, "logit_bias": True}}
        engine.sampling = ("temperature", "seed", "presence_penalty", "frequency_penalty", "repetition_penalty",
                           "logit_bias")
        engine.template_variables = {}
        engine.kept_special = frozenset()   # no special-token tool markers (ResidentChat.__init__ sets it lazily)
        engine.text = Mock()
        engine.text.templates = {}   # a template that teaches no registered tool-call format
        engine.text.encode_chat.return_value = [1, 2]
        engine.text.tokenizer = Tokenizer.from_bytes(byte_level_json())
        engine.text.text_stream.side_effect = lambda ids, **options: TextStream(engine.text.tokenizer, ids, **options)
        output = engine.text.tokenizer.encode("ok", add_special_tokens=False)
        def generate(ids, limit, **options):
            options["on_tokens"](output)
            return output, {"timing": {"prefill_tokens": 2, "reused_prompt_tokens": 0,
                "prefill_ms": 1, "decode_ms": 1, "first_token_ms": 1}}
        engine.native = Mock()
        engine.native.generate.side_effect = generate
        return engine

    def test_override_forwarding_and_default_isolation(self):
        engine = self.engine()
        request = {"messages": [{"role": "user", "content": "hi"}], "sampling": {
            "temperature": 0, "seed": 123, "frequency_penalty": .5, "logit_bias": {"12": -2}}}
        self.assertEqual(engine._run(request, threading.Event())[0], "ok")
        self.assertEqual(engine.native.generate.call_args.kwargs["sampling"], request["sampling"])
        engine._run({"messages": request["messages"]}, threading.Event())
        self.assertEqual(engine.native.generate.call_args.kwargs["sampling"], {})
        self.assertEqual(engine.configuration, {"temperature": .8})

    def test_decode_rounds_reach_the_reply_telemetry(self):
        engine = self.engine()
        output = engine.text.tokenizer.encode("okay then", add_special_tokens=False)
        timing = {"prefill_tokens": 2, "reused_prompt_tokens": 0, "prefill_ms": 1, "decode_ms": 1,
                  "first_token_ms": 1, "decode_steps": 7}
        message = {"messages": [{"role": "user", "content": "hi"}]}
        for speculative, facts, rounds in ((True, {"accepted": [3, 2]}, 2), (True, {"rounds": 4}, 4),
                                           (True, {}, 7), (False, {}, len(output) - 1)):
            with self.subTest(speculative=speculative, facts=facts):
                def generate(ids, limit, **options):
                    options["on_tokens"](output)
                    return output, {"timing": timing, **facts}
                engine.native.generate.side_effect = generate
                engine.facts["speculative"] = speculative
                self.assertEqual(engine._run(message, threading.Event())[1]["decode_rounds"], rounds)

    def test_unsupported_native_sampling_is_not_ignored(self):
        engine = self.engine()
        with self.assertRaisesRegex(ValueError, r"sampling field\(s\): top_k"):
            engine._run({"messages": [{"role": "user", "content": "hi"}], "sampling": {"top_k": 5}}, threading.Event())
        engine.native.generate.assert_not_called()

    def test_tools_and_tool_history_reach_the_chat_template_unless_tool_choice_is_none(self):
        engine = self.engine()
        tools = [{"type": "function", "function": {"name": "list"}}]
        messages = [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "list", "arguments": '{"a": 1}'}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "[]"}]
        engine._run({"messages": messages, "tools": tools, "tool_choice": "auto"}, threading.Event())
        (rendered,), options = engine.text.encode_chat.call_args
        self.assertEqual(options["tools"], tools)
        self.assertEqual(rendered[1]["tool_calls"][0]["function"]["arguments"], {"a": 1})
        self.assertEqual(rendered[2], {"role": "tool", "content": "[]", "tool_call_id": "c1"})
        engine._run({"messages": messages, "tools": tools, "tool_choice": "none"}, threading.Event())
        self.assertIsNone(engine.text.encode_chat.call_args.kwargs["tools"])
