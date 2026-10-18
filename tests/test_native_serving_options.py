"""Request sampling overrides reach the native call without changing resident defaults."""
import threading
import types
import unittest
from unittest.mock import Mock

from crit_core.native import ResidentEngine


class NativeServingOptionsTests(unittest.TestCase):
    def engine(self):
        engine = ResidentEngine.__new__(ResidentEngine)
        engine.configuration = {"temperature": .8}
        engine.capacity, engine.stop_ids = 2048, set()
        engine.facts = {"sampling": {"temperature": True, "seed": True, "penalties": True, "logit_bias": True}}
        engine.text = Mock()
        engine.text.encode_chat.return_value = [1, 2]
        engine.text.text_stream.return_value = types.SimpleNamespace(
            push=lambda tokens: types.SimpleNamespace(text="ok"), finish=lambda: types.SimpleNamespace(text=""))
        def generate(ids, limit, **options):
            options["on_tokens"]([3, 4])
            return [3, 4], {"timing": {"prefill_tokens": 2, "reused_prompt_tokens": 0,
                "prefill_ms": 1, "decode_ms": 1, "first_token_ms": 1}}
        engine.native = Mock()
        engine.native.generate.side_effect = generate
        return engine

    def test_override_forwarding_and_default_isolation(self):
        engine = self.engine()
        request = {"messages": [{"role": "user", "content": "hi"}], "sampling": {
            "temperature": 0, "seed": 123, "frequency_penalty": .5, "logit_bias": {"12": -2}}}
        engine._run(request, threading.Event())
        self.assertEqual(engine.native.generate.call_args.kwargs["sampling"], request["sampling"])
        engine._run({"messages": request["messages"]}, threading.Event())
        self.assertEqual(engine.native.generate.call_args.kwargs["sampling"], {})
        self.assertEqual(engine.configuration, {"temperature": .8})

    def test_unsupported_native_sampling_and_unparsed_tools_are_not_ignored(self):
        engine = self.engine()
        for extra, reason in (({"sampling": {"top_k": 5}}, "sampling option: top_k"),
                              ({"tools": [{"type": "function"}]}, "model-specific")):
            with self.assertRaisesRegex(ValueError, reason):
                engine._run({"messages": [{"role": "user", "content": "hi"}], **extra}, threading.Event())
        engine.native.generate.assert_not_called()
