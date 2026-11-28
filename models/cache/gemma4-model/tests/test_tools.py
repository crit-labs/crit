"""Gemma 4's tool calls and thought channel through crit serve's parsers (crit_core.text.tools.ToolReply,
crit_core.text.reasoning.ReasoningSplitter), round trip on the official chat template: an assistant turn with
reasoning and calls is rendered by Google's Gemma 4 chat template (chat_template.jinja beside this file, from
google/gemma-4-26B-A4B-it, Apache 2.0, rendered by crit_core.text.ChatTemplate) and the rendered reply parses back to
the same reasoning, function names and arguments, whole and streamed. A block naming no offered function, or one that
does not parse in full, stays text."""
import importlib.util
import json
from pathlib import Path
import unittest

from crit_core.text.reasoning import ReasoningSplitter
from crit_core.text.template import ChatTemplate
from crit_core.text.tools import ToolReply

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gemma4_tools", HERE / "tools.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)

FUNCTIONS = {
    "get_weather": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}},
    "edit_file": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array"}}},
}
REQUEST = {"tools": [{"type": "function", "function": {"name": name, "description": f"{name} tool", "parameters": p}}
                     for name, p in FUNCTIONS.items()]}
CALLS = [("get_weather", {"city": "Paris, France", "days": 3}),
         ("edit_file", {"path": "src/a b.py", "edits": [{"old": "x = 1", "new": "x = {2}", "count": 1.5},
                                                        {"old": "", "new": None, "regex": False}],
                        "meta": {"dry": True, "tags": ["a", "b,c"]}})]


def rendered_reply(reasoning, calls):
    """The assistant turn's text as the official template renders it after the user's turn: from its reasoning (or
    its first call) to its last call's end."""
    template = ChatTemplate((HERE / "tests" / "chat_template.jinja").read_text(encoding="utf-8"),
                            special_tokens={"bos_token": "<bos>"})
    user = [{"role": "user", "content": "Do it."}]
    turn = {"role": "assistant", "content": "", "reasoning_content": reasoning,
            "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]}
    text = template.render(user + [turn], tools=REQUEST["tools"])
    text = text[text.index("Do it."):]
    start = min(i for i in (text.find(tools.REASONING_FORMAT.opening), text.find(tools.OPEN)) if i >= 0)
    return text[start:text.rindex(tools.CLOSE) + len(tools.CLOSE)]


def parsed(text, pieces=None):
    """(reasoning, content, [(name, arguments)]) of a reply, split and parsed as crit serve does, fed in `pieces`."""
    splitter, reasoning, deltas = ReasoningSplitter(tools.REASONING_FORMAT), [], []
    reply = ToolReply(tools.TOOL_FORMAT, REQUEST, deltas.append)
    chunks = [text] if pieces is None else [text[i:i + pieces] for i in range(0, len(text), pieces)]
    events = []
    for chunk in chunks:
        events += splitter.push(chunk)
    events += splitter.finish()
    for kind, value in events:
        if kind == "reasoning":
            reasoning.append(value)
        else:
            reply.push(value)
    content, calls = reply.finish()
    return "".join(reasoning), content, [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                                         for c in calls]


class RoundTripTests(unittest.TestCase):
    def test_the_templates_calls_and_reasoning_parse_back_whole_and_streamed(self):
        text = rendered_reply("I need the weather first.", CALLS)
        self.assertIn("<|channel>thought\nI need the weather first.\n<channel|>", text)
        self.assertIn('<|tool_call>call:get_weather{city:<|"|>Paris, France<|"|>,days:3}<tool_call|>', text)
        for pieces in (None, 1, 7):
            with self.subTest(pieces=pieces):
                self.assertEqual(parsed(text, pieces), ("I need the weather first.", "", CALLS))

    def test_a_turn_without_reasoning_parses_its_call(self):
        text = rendered_reply("", CALLS[:1])
        self.assertNotIn("<|channel>", text)
        self.assertEqual(parsed(text), ("", "", CALLS[:1]))

    def test_an_empty_thought_channel_and_text_before_a_call(self):
        call = '<|tool_call>call:get_weather{city:<|"|>Oslo<|"|>}<tool_call|>'
        self.assertEqual(parsed("<|channel>thought\n<channel|>" + call), ("", "", [("get_weather", {"city": "Oslo"})]))
        self.assertEqual(parsed("Let me check. " + call)[1:], ("Let me check.", [("get_weather", {"city": "Oslo"})]))


class ParseTests(unittest.TestCase):
    def test_a_block_naming_no_offered_function_or_not_parsing_in_full_is_no_call(self):
        self.assertIsNone(tools.parse('call:send_mail{to:<|"|>x<|"|>}', FUNCTIONS))
        self.assertIsNone(tools.parse('call:get_weather{city:<|"|>x}', FUNCTIONS))      # unclosed string
        self.assertIsNone(tools.parse('call:get_weather{city:1} extra', FUNCTIONS))
        self.assertIsNone(tools.parse('call:get_weather{a:1,a:2}', FUNCTIONS))          # a key twice
        other = '<|tool_call>call:send_mail{to:<|"|>x<|"|>}<tool_call|>'
        self.assertEqual(parsed(other)[1:], (other, []))

    def test_values_read_as_the_template_writes_them(self):
        self.assertEqual(tools.parse('call:get_weather{a:true,b:false,c:null,d:-2.5e3,e:[],f:{},g:word}', FUNCTIONS),
                         ("get_weather", {"a": True, "b": False, "c": None, "d": -2500.0, "e": [], "f": {},
                                          "g": "word"}))
        self.assertEqual(tools.parse('call:get_weather{<|"|>quoted<|"|>:{<|"|>k<|"|>:1}}', FUNCTIONS),
                         ("get_weather", {"quoted": {"k": 1}}))

    def test_a_call_streams_its_name_once_written(self):
        self.assertIsNone(tools.stream("call:get_wea", FUNCTIONS))
        self.assertEqual(tools.stream("call:get_weather{ci", FUNCTIONS), ("get_weather", "{"))
        self.assertIsNone(tools.stream("call:send_mail{", FUNCTIONS))


if __name__ == "__main__":
    unittest.main()
