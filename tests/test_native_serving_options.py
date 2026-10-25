"""Request sampling overrides reach the native call without changing resident defaults."""
import threading
import unittest
from unittest.mock import Mock

from core_text_fixtures import byte_level_json
from crit_core.native import ResidentEngine, Scores
from crit_core.serving import ScoreRows
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

    def test_eval_requests_take_neutral_penalties_instead_of_the_engines_defaults(self):
        from unittest.mock import patch
        from crit_core import native
        facts = {"sampling": {"temperature": True, "penalties": True}, "speculative": False, "context": 4096}
        library = Mock()
        library.return_value.info.return_value = facts
        library.return_value.generate.return_value = ([5, 6], {"timing": {"decode_steps": 1}})
        with patch.object(native, "NativeEngine", library):
            engine = native.EvalEngine({"model": "m", "config": {}}, "libengine.so")
        neutral = {"temperature": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
        self.assertEqual(engine.sampling, neutral)
        engine.generate([1, 2], 2, lambda tokens: None)
        self.assertEqual(library.return_value.generate.call_args.kwargs["sampling"], neutral)
        facts["sampling"] = {"temperature": True}   # an engine without penalties gets none
        with patch.object(native, "NativeEngine", library):
            self.assertEqual(native.EvalEngine({"model": "m", "config": {}}, "libengine.so").sampling,
                             {"temperature": 0.0})
        # Fixed-text scores replay greedily with neutral penalties too.
        resident = self.engine()
        resident.facts = {"score_paths": []}
        resident.native.generate.side_effect = lambda ids, n, **options: (list(options["replay"]), {
            "timing": {"decode_ms": 1.0}})
        resident.score_tokens([1, 2], [3], 2, lambda: False, False)
        self.assertEqual(resident.native.generate.call_args.kwargs["sampling"], neutral)

    def test_scores_pass_the_engines_arrays_and_count_both_prefills(self):
        import array
        engine = self.engine()
        engine.facts = {"score_paths": ["prefill"]}
        rows = ScoreRows(array.array("f", [-0.5]), array.array("i", [2, 0]), array.array("f", [-0.5, -1.0]), 2)
        engine.native.score.return_value = Scores([-0.5], None, None, rows)
        engine.native.generate.side_effect = lambda ids, n, **options: (
            options["on_tokens"]([3], [(-0.25, [(3, -0.25), (0, -1.0)])]) or list(options["replay"]),
            {"timing": {"prefill_ms": 7.0, "decode_ms": 1.0}})
        prompt, continuation, timing = engine.score_tokens([1, 2], [3], 2, lambda: False, True)
        self.assertIs(prompt, rows)   # packed by ResidentChat.score without a loop over its rows
        self.assertEqual(engine.native.score.call_args.kwargs, {"top_k": 2, "packed": True})
        self.assertEqual(continuation, [(-0.25, [(3, -0.25), (0, -1.0)])])
        # The replay prefills the prompt again: its prefill is engine time too, not transfer.
        self.assertGreaterEqual(timing["prefill_ms"], 7.0)
        self.assertEqual(timing["decode_ms"], 1.0)
        # Without alternatives the engine returns no top rows: each row has none.
        engine.native.score.return_value = Scores([-0.5], None, None, None)
        self.assertEqual(engine.score_tokens([1, 2], [], 0, lambda: False, True)[0], [(-0.5, [])])

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
