"""The model's tool-call syntax through crit serve's parser (crit_core.text.tools.ToolReply): a call named before its
marker parses and leaves no name in the content, text around it stays, a block naming no offered function stays text."""
import importlib.util
from pathlib import Path
import unittest

from crit_core.text.tools import ToolReply

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_tools", HERE / "tools.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
REQUEST = {"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]}
CALL = 'get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}<|end_message|>'


def reply(pieces):
    deltas = []
    r = ToolReply(tools.TOOL_FORMAT, REQUEST, deltas.append)
    for piece in pieces:
        r.push(piece)
    return r.finish(), deltas


class ToolFormatTests(unittest.TestCase):
    def test_a_call_named_before_its_marker_parses_whole_and_streamed(self):
        for pieces in ([CALL], list(CALL), [CALL[:5], CALL[5:40], CALL[40:]]):
            (content, calls), _ = reply(pieces)
            self.assertEqual(content, "")
            self.assertEqual([(c["function"]["name"], c["function"]["arguments"]) for c in calls],
                             [("get_weather", '{"city": "Paris"}')])

    def test_text_before_a_call_stays_and_a_word_at_the_end_is_text(self):
        (content, calls), _ = reply(["Let me check. ", CALL])
        self.assertEqual((content, len(calls)), ("Let me check.", 1))
        (content, calls), _ = reply(["The weather is fine"])
        self.assertEqual((content, calls), ("The weather is fine", []))

    def test_a_block_naming_no_offered_function_stays_text_as_written(self):
        other = CALL.replace("get_weather", "send_mail")
        (content, calls), _ = reply([other])
        self.assertEqual((content, calls), (other, []))


class ReasoningFormatTests(unittest.TestCase):
    def test_the_reasoning_splits_from_the_answer(self):
        from crit_core.text.reasoning import ReasoningSplitter
        splitter = ReasoningSplitter(tools.REASONING_FORMAT)
        events = splitter.push("<|content_thinking|>Let me look.<|end_message|>ALPHA: 7") + splitter.finish()
        self.assertEqual("".join(v for k, v in events if k == "reasoning"), "Let me look.")
        self.assertEqual("".join(v for k, v in events if k == "text").strip(), "ALPHA: 7")


if __name__ == "__main__":
    unittest.main()
