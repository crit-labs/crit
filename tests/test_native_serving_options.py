"""Core's adapters (crit_core.engine) over an engine's runtime: request sampling overrides reach the runtime without
changing resident defaults, Eval's requests take neutral penalties and the engine's own times, and fixed text is
scored as crit serve scores it."""
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core/python"))
from core_text_fixtures import byte_level_json
from crit_core import engine as core_engine
from crit_core.engine import ResidentEngine
from crit_core.native import NativeError, Scores
from crit_core.serving import ScoreRows
from crit_core.text import TextStream, Tokenizer


class NativeServingOptionsTests(unittest.TestCase):
    def engine(self):
        engine = ResidentEngine.__new__(ResidentEngine)
        engine.configuration = {"temperature": .8}
        engine.capacity, engine.stop_ids = 2048, set()
        engine.release_below = None   # no device memory to hand back after a request (release_threshold)
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

    def eval_engine(self, library):
        """Core's eval-v1 adapter over a package whose loaded engine is `library`'s return value."""
        from unittest.mock import patch
        with patch.object(core_engine, "read_configuration", lambda package: {"plugins": {"use": {}}}), \
                patch.object(core_engine, "load", lambda *args, **options: library.return_value):
            return core_engine.EvalEngine({"model": "m", "config": {"package": "p"}})

    def test_eval_requests_take_neutral_penalties_instead_of_the_engines_defaults(self):
        facts = {"sampling": {"temperature": True, "penalties": True}, "speculative": False, "context": 4096}
        library = Mock()
        library.return_value.info.return_value = facts
        library.return_value.generate.return_value = ([5, 6], {"timing": {"decode_steps": 1}})
        engine = self.eval_engine(library)
        neutral = {"temperature": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
        self.assertEqual(engine.sampling, neutral)
        engine.generate([1, 2], 2, lambda tokens: None)
        self.assertEqual(library.return_value.generate.call_args.kwargs["sampling"], neutral)
        facts["sampling"] = {"temperature": True}   # an engine without penalties gets none
        self.assertEqual(self.eval_engine(library).sampling, {"temperature": 0.0})
        # Fixed-text scores replay greedily with neutral penalties too.
        resident = self.engine()
        resident.facts = {"score_paths": []}
        resident.native.generate.side_effect = lambda ids, n, **options: (list(options["replay"]), {
            "timing": {"decode_ms": 1.0}})
        resident.score_tokens([1, 2], [3], 2, lambda: False, False)
        self.assertEqual(resident.native.generate.call_args.kwargs["sampling"], neutral)

    def test_eval_generations_carry_the_engines_own_prefill_and_first_token_times(self):
        facts = {"sampling": {"temperature": True}, "speculative": True, "plain": True, "context": 4096}
        library = Mock()
        library.return_value.info.return_value = facts
        library.return_value.generate.return_value = ([5, 6], {"timing": {"decode_steps": 1, "prefill_ms": 12.5,
                                                                          "first_token_ms": 13.0, "decode_ms": 4.0}})
        engine = self.eval_engine(library)
        found = engine.generate([1, 2], 2, lambda tokens: None)
        self.assertEqual((found["prefill_ms"], found["first_token_ms"]), (12.5, 13.0))
        self.assertEqual(engine.generate_plain([1, 2], 2, lambda tokens: None)["first_token_ms"], 13.0)
        library.return_value.generate.return_value = ([5, 6], {"timing": {"decode_steps": 1}})
        self.assertNotIn("prefill_ms", engine.generate([1, 2], 2, lambda tokens: None))   # not reported, left out

    def test_an_eval_engine_scores_fixed_text_as_crit_serve_does(self):
        facts = {"sampling": {"temperature": True, "penalties": True}, "speculative": False, "context": 4096,
                 "token_logprobs": True, "query_scoring": True, "replay": True, "score_paths": ["prefill", "decode"]}
        library = Mock()
        library.return_value.info.return_value = facts
        library.return_value.score.return_value = Scores([-0.5], None, None, [[(2, -0.5), (0, -1.0)]])

        def generate(ids, n, **options):
            options["on_tokens"]([3], [(-0.25, [(3, -0.25), (0, -1.0)])])
            return list(options["replay"]), {"timing": {"prefill_ms": 1.0}}
        library.return_value.generate.side_effect = generate
        engine = self.eval_engine(library)
        rows = engine.score([1, 2], [3], 2)
        self.assertEqual(rows, {"prompt": [{"logprob": -0.5, "top": [[2, -0.5], [0, -1.0]]}],
                                "continuation": [{"logprob": -0.25, "top": [[3, -0.25], [0, -1.0]]}]})
        self.assertEqual(library.return_value.score.call_args.args, ([1, 2], 1, "prefill"))
        call = library.return_value.generate.call_args
        self.assertEqual((call.kwargs["replay"], call.kwargs["logprobs"], call.kwargs["stop_tokens"]), ([3], 2, []))
        self.assertEqual(call.kwargs["sampling"], {"temperature": 0.0, "presence_penalty": 0.0,
                                                   "frequency_penalty": 0.0, "repetition_penalty": 1.0})
        self.assertEqual(engine.score([1, 2], [3], 2, prompt_rows=False)["prompt"], [])
        library.return_value.generate.side_effect = lambda ids, n, **options: ([9], {"timing": {}})
        with self.assertRaisesRegex(NativeError, "other tokens than the continuation"):
            engine.score([1, 2], [3], 2)

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
