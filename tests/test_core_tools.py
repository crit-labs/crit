"""Tool calls parsed from reply text in the chat template's own format (crit_core.text.tools, offline)."""
from pathlib import Path
import json
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "cli"): sys.path.insert(0, str(ROOT / _folder))

from crit_core.text import (ChatTemplate, TextError, ToolCallParser, ToolFormat, ToolReply,  # noqa: E402
                            templates_tool_format, tool_call_format)
from crit_core.text.tools import format_markers  # noqa: E402

# The assistant tool-call rendering of the Qwen3.8 / Nemotron 3 chat templates, reduced to the call syntax.
XML_TEMPLATE = """{%- for message in messages %}{%- if message.role == 'assistant' %}{{- message.content }}
{%- for tool_call in message.tool_calls %}{%- set tool_call = tool_call.function %}
{%- if loop.first %}{%- if message.content|trim %}{{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}
{%- else %}{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}{%- endif %}
{%- else %}{{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}{%- endif %}
{%- for args_name, args_value in tool_call.arguments|items %}{{- '<parameter=' + args_name + '>\\n' }}
{%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
{{- args_value }}{{- '\\n</parameter>\\n' }}{%- endfor %}{{- '</function>\\n</tool_call>' }}
{%- endfor %}{%- endif %}{%- endfor %}{# <tools> If you call a function use <parameter=...> #}"""
JSON_TEMPLATE = """{%- for tool_call in message.tool_calls %}{{- '<tool_call>\\n' }}{{- tool_call.function | tojson }}
{{- '\\n</tool_call>' }}{%- endfor %}"""
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "lines": {"type": "integer"}, "options": {"type": "object"},
    "note": {"type": ["string", "null"]}}}}},
    {"type": "function", "function": {"name": "list", "parameters": {"type": "object", "properties": {}}}}]


def events(parser, text, chunks=None):
    out = []
    if chunks is None:
        out += parser.push(text)
    else:
        position = 0
        for size in chunks:
            out += parser.push(text[position:position + size])
            position += size
        out += parser.push(text[position:])
    return out + parser.finish()


def joined(result):
    text = "".join(value for kind, value in result if kind == "text")
    return text, [value for kind, value in result if kind == "call"]


class FormatTests(unittest.TestCase):
    def test_detects_the_format_a_template_teaches(self):
        self.assertEqual(tool_call_format(XML_TEMPLATE), "xml-function")
        self.assertEqual(tool_call_format(JSON_TEMPLATE), "json-object")
        self.assertIsNone(tool_call_format("{{ messages[0].content }}"))
        with self.assertRaises(TextError):
            tool_call_format(None)
        with self.assertRaises(TextError):
            ToolCallParser("unknown", TOOLS)

    def test_a_models_templates_teach_one_format_or_none(self):
        self.assertEqual(templates_tool_format([XML_TEMPLATE, "{{ messages[0].content }}"]), "xml-function")
        self.assertIsNone(templates_tool_format(["{{ messages[0].content }}"]))
        with self.assertRaisesRegex(TextError, "disagree"):
            templates_tool_format([XML_TEMPLATE, JSON_TEMPLATE])


class XmlFunctionTests(unittest.TestCase):
    def render(self, content, calls):
        message = {"role": "assistant", "content": content, "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": arguments}} for name, arguments in calls]}
        return ChatTemplate(XML_TEMPLATE).render([message])

    def test_rendered_calls_parse_back_exactly_in_any_chunking(self):
        calls = [("read_file", {"path": "src/a b.py", "lines": 20, "options": {"x": [1, 2]}, "note": "two\nlines"}),
                 ("list", {})]
        text = self.render("I'll look.", calls)
        expected = [{"name": name, "arguments": arguments} for name, arguments in calls]
        self.assertEqual(joined(events(ToolCallParser("xml-function", TOOLS), text)), ("I'll look.", expected))
        rng = random.Random(7)
        for _ in range(50):
            chunks = [rng.randint(1, 9) for _ in range(len(text) // 3)]
            self.assertEqual(joined(events(ToolCallParser("xml-function", TOOLS), text, chunks)),
                             ("I'll look.", expected))
        self.assertEqual(joined(events(ToolCallParser("xml-function", TOOLS), self.render("", calls[:1]))),
                         ("", expected[:1]))

    def test_schema_strings_stay_verbatim_and_other_values_decode_when_they_can(self):
        text = ("<tool_call>\n<function=read_file>\n<parameter=path>\n123\n</parameter>\n<parameter=lines>\nmany\n"
                "</parameter>\n<parameter=extra>\n[1]\n</parameter>\n</function>\n</tool_call>")
        _, calls = joined(events(ToolCallParser("xml-function", TOOLS), text))
        self.assertEqual(calls, [{"name": "read_file", "arguments": {"path": "123", "lines": "many", "extra": "[1]"}}])

    def test_plain_replies_and_tag_like_text_pass_through_unchanged(self):
        for text in ("Hello  world \n", "a < b and <tool or <tool_cal", "x" * 40 + "\n\n"):
            for chunks in (None, [1] * len(text)):
                self.assertEqual(joined(events(ToolCallParser("xml-function", TOOLS), text, chunks)), (text, []))

    def test_malformed_unknown_extra_and_unclosed_calls_stay_text(self):
        good = "<tool_call>\n<function=list>\n</function>\n</tool_call>"
        cases = {
            "<tool_call>\n<function=delete_all>\n</function>\n</tool_call>": 0,
            "<tool_call>\n<function=list>\nstray\n</function>\n</tool_call>": 0,
            "<tool_call>\n{\"name\": \"list\"}\n</tool_call>": 0,
            good + "\n" + good: 2,
        }
        for text, count in cases.items():
            content, calls = joined(events(ToolCallParser("xml-function", TOOLS), text))
            self.assertEqual(len(calls), count, text)
            self.assertEqual(content, "" if count else text)
        content, calls = joined(events(ToolCallParser("xml-function", TOOLS, parallel=False), good + "\n" + good))
        self.assertEqual((len(calls), content), (1, good))
        cut = "Checking.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na"
        self.assertEqual(joined(events(ToolCallParser("xml-function", TOOLS), cut)), ("Checking." + cut[9:], []))

    def test_whitespace_around_calls_is_dropped_but_text_after_them_is_kept(self):
        text = "Plan:\n\n<tool_call>\n<function=list>\n</function>\n</tool_call>\n\n  Done"
        self.assertEqual(joined(events(ToolCallParser("xml-function", TOOLS), text, [1] * len(text))),
                         ("Plan:Done", [{"name": "list", "arguments": {}}]))

    def test_finished_parser_refuses_more_text(self):
        parser = ToolCallParser("xml-function", TOOLS)
        parser.finish()
        with self.assertRaises(TextError):
            parser.push("x")


class JsonObjectTests(unittest.TestCase):
    def test_hermes_calls_with_object_or_string_arguments(self):
        call = {"name": "read_file", "arguments": {"path": "a.py"}}
        text = ("<tool_call>\n" + json.dumps(call) + "\n</tool_call>\n<tool_call>\n"
                + json.dumps(dict(call, arguments=json.dumps({"lines": 2}))) + "\n</tool_call>")
        self.assertEqual(joined(events(ToolCallParser("json-object", TOOLS), text, [3] * 40)),
                         ("", [call, {"name": "read_file", "arguments": {"lines": 2}}]))
        for body in ('{"name": "read_file", "arguments": [1]}', '{"name": "x"}', "not json",
                     '{"name": "list", "arguments": {}, "id": 1}', '{"name": "list", "arguments": NaN}'):
            text = "<tool_call>" + body + "</tool_call>"
            self.assertEqual(joined(events(ToolCallParser("json-object", TOOLS), text)), (text, []))

def streamed(text, format="xml-function", chunks=None, cut=False, **request):
    """ToolReply over text pushed in chunks (one character each by default): (content, calls, the deltas emitted
    after each push, the last after finish)."""
    emitted, steps = [], []
    reply = ToolReply(format, {"tools": TOOLS, **request}, emitted.append)
    position = 0
    for size in chunks or [1] * len(text):
        reply.push(text[position:position + size])
        position += size
        steps.append([e["tool_calls"][0] for e in emitted if isinstance(e, dict)])
    reply.push(text[position:])
    content, calls = reply.finish(cut)
    return content, calls, steps + [[e["tool_calls"][0] for e in emitted if isinstance(e, dict)]]


def rebuilt(deltas):
    """Calls as a client assembles them from deltas: {index: (id, name, arguments)}."""
    calls = {}
    for delta in deltas:
        ident, name, arguments = calls.get(delta["index"], (delta.get("id"), delta["function"].get("name"), ""))
        calls[delta["index"]] = ident, name, arguments + delta["function"].get("arguments", "")
    return calls


class StreamingTests(unittest.TestCase):
    def check(self, calls, deltas):
        """Each call's first delta names it and the rest only add arguments, which add up to the final call's."""
        for index, call in enumerate(calls):
            mine = [d for d in deltas if d["index"] == index]
            self.assertEqual((mine[0]["id"], mine[0]["type"], mine[0]["function"]["name"]),
                             (call["id"], "function", call["function"]["name"]))
            self.assertTrue(all(set(d) == {"index", "function"} and set(d["function"]) == {"arguments"}
                                and d["function"]["arguments"] for d in mine[1:]))
        self.assertEqual(rebuilt(deltas), {i: (c["id"], c["function"]["name"], c["function"]["arguments"])
                                           for i, c in enumerate(calls)})

    def test_xml_calls_stream_from_their_name_in_any_chunking(self):
        calls = [("read_file", {"path": "src/a b.py", "lines": 20, "options": {"x": [1, 2]}, "note": "two\nlines\n"}),
                 ("list", {})]
        text = XmlFunctionTests.render(None, "I'll look.", calls)
        rng = random.Random(11)
        for chunks in [None] + [[rng.randint(1, 9) for _ in range(len(text) // 3)] for _ in range(30)]:
            content, final, steps = streamed(text, chunks=chunks)
            self.assertEqual(content, "I'll look.")
            self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in final], calls)
            self.check(final, steps[-1])
        _, _, steps = streamed(text)
        header = text.index("<function=read_file>") + len("<function=read_file>")
        self.assertEqual(rebuilt(steps[header - 1])[0][1:], ("read_file", "{"))  # named before its block closes
        self.assertEqual(rebuilt(steps[text.index("a b.py") + 1])[0][2], '{"path": "src/a ')
        # A non-string value goes out once its parameter closes; text that may end a string value is held.
        self.assertEqual(rebuilt(steps[text.index("20") + 1])[0][2], '{"path": "src/a b.py", "lines": ')
        self.assertEqual(rebuilt(steps[text.index("two\n") + 3])[0][2].rsplit(" ", 1)[1], '"two')

    def test_json_object_calls_stream_member_by_member(self):
        call = {"name": "read_file", "arguments": {"path": "a.py", "lines": 20, "options": {"x": [1, 2]}}}
        text = "Sure.\n<tool_call>\n" + json.dumps(call) + "\n</tool_call>"
        for chunks in (None, [3] * 40, [17] * 10):
            content, final, steps = streamed(text, "json-object", chunks)
            self.assertEqual((content, json.loads(final[0]["function"]["arguments"])), ("Sure.", call["arguments"]))
            self.check(final, steps[-1])
        _, _, steps = streamed(text, "json-object")
        self.assertEqual(steps[text.index('read_file"') + 8], [])
        self.assertEqual(rebuilt(steps[text.index('read_file"') + 9])[0][1:], ("read_file", "{"))
        self.assertEqual(rebuilt(steps[text.index('a.py"') + 1])[0][2], '{"path": "a.')
        self.assertEqual(rebuilt(steps[text.index("20,") + 1])[0][2], '{"path": "a.py", "lines": ')  # 20 may go on
        self.assertEqual(rebuilt(steps[text.index("20,") + 2])[0][2], '{"path": "a.py", "lines": 20')
        # A string streams as it decodes, never half an escape: "\u00e9" goes out as "é" once whole.
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "caf\\u00e9 \\ud83d\\ude00\\n"}}</tool_call>'
        content, final, steps = streamed(text, "json-object")
        self.assertEqual(json.loads(final[0]["function"]["arguments"]), {"path": "café 😀\n"})
        self.check(final, steps[-1])
        self.assertEqual(rebuilt(steps[text.index("u00e9") + 3])[0][2], '{"path": "caf')
        self.assertEqual(rebuilt(steps[text.index("u00e9") + 4])[0][2], '{"path": "café')
        self.assertEqual(rebuilt(steps[text.index("ude00") + 3])[0][2], '{"path": "café ')
        self.assertEqual(rebuilt(steps[text.index("ude00") + 4])[0][2], '{"path": "café 😀')
        # Arguments written as a JSON string decode when the block closes.
        text = '<tool_call>{"name": "list", "arguments": "{\\"a\\": 1}"}</tool_call>'
        _, final, steps = streamed(text, "json-object")
        self.assertEqual(final[0]["function"]["arguments"], '{"a": 1}')
        self.check(final, steps[-1])

    def test_a_named_block_that_does_not_parse_fails_the_reply(self):
        for format, text in (("xml-function", "<tool_call>\n<function=read_file>\nstray\n</function>\n</tool_call>"),
                             ("xml-function", "<tool_call>\n<function=list>\n<parameter=a>\n1\n</parameter>\n"
                                              "<parameter=a>\n2\n</parameter>\n</function>\n</tool_call>"),
                             ("json-object", '<tool_call>{"name": "list", "arguments": {}, "id": 1}</tool_call>'),
                             ("json-object", '<tool_call>{"name": "list", "arguments": {"a": 1, "a": 2}}</tool_call>')):
            with self.subTest(text=text), self.assertRaisesRegex(TextError, "malformed call to (read_file|list)"):
                streamed(text, format)

    def test_unnamed_unknown_and_disallowed_blocks_stay_text(self):
        good = "<tool_call>\n<function=list>\n</function>\n</tool_call>"
        for text in ("<tool_call>\n<function=delete_all>\n</function>\n</tool_call>",
                     "<tool_call>\n{\"name\": \"list\"}\n</tool_call>"):
            self.assertEqual(streamed(text)[:2], (text, []))
        content, calls, steps = streamed(good + "\n" + good, parallel_tool_calls=False)
        self.assertEqual((content, len(calls), {d["index"] for d in steps[-1]}), (good, 1, {0}))
        # A repeated key is no call in either format.
        text = '<tool_call>{"name": "list", "arguments": {"a": 1, "a": 2}}</tool_call>'
        self.assertEqual(joined(events(ToolCallParser("json-object", TOOLS), text)), (text, []))

    def test_a_reply_cut_inside_a_streamed_call_keeps_the_call_so_far(self):
        text = "Checking.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/a"
        content, calls, steps = streamed(text, cut=True)
        self.assertEqual(content, "Checking.")
        self.assertEqual([c["function"] for c in calls], [{"name": "read_file", "arguments": '{"path": "src/a'}])
        self.check(calls, steps[-1])
        with self.assertRaisesRegex(TextError, "ended its reply inside its call to read_file"):
            streamed(text)  # not cut: the model ended its reply there
        # Ended before the name is written: no call, the text stays.
        for cut in (True, False):
            self.assertEqual(streamed("Checking.<tool_call>\n<funct", cut=cut)[:2], ("Checking.<tool_call>\n<funct", []))

    def test_a_package_format_streams_when_it_reads_unfinished_blocks(self):
        def stream(body, tools):
            name, space, _ = body.lstrip().partition(" ")
            return (name, "{") if space and name in tools else None
        streaming = BRACKET._replace(stream=stream)
        text = 'Looking. [[call read_file {"path": "a.py"}]]'
        _, calls, steps = streamed(text, streaming)
        self.check(calls, steps[-1])
        self.assertEqual((steps[text.index("{") - 2], len(steps[text.index("{") - 1])), ([], 1))
        _, calls, steps = streamed(text, BRACKET)
        self.assertEqual(steps[-3], [])  # without stream: whole, once its block closes
        self.assertEqual(steps[-1], [dict(calls[0], index=0)])


# The tool-call rendering of the Gemma 4 chat template (format_argument with bare keys), reduced to the call syntax.
def _bracket_call(body, tools):
    """A package's own syntax for the tests: [[call NAME JSON]]."""
    name, _, arguments = body.strip().partition(" ")
    try:
        arguments = json.loads(arguments or "{}")
    except ValueError:
        return None
    return (name, arguments) if name in tools and isinstance(arguments, dict) else None


BRACKET = ToolFormat(_bracket_call, "[[call ", "]]", ("<|q|>",))


class PackageFormatTests(unittest.TestCase):
    def test_a_package_format_parses_with_the_shared_parser_and_reply(self):
        self.assertEqual(format_markers(BRACKET), ("[[call ", "]]", "<|q|>"))
        self.assertEqual(format_markers("json-object"), ("<tool_call>", "</tool_call>"))
        text = 'Looking. [[call read_file {"path": "a.py"}]] [[call nope {}]]'
        expected = ("Looking.[[call nope {}]]", [{"name": "read_file", "arguments": {"path": "a.py"}}])
        for chunks in (None, [1] * len(text), [3] * 30):
            self.assertEqual(joined(events(ToolCallParser(BRACKET, TOOLS), text, chunks)), expected)
        emitted = []
        reply = ToolReply(BRACKET, {"tools": TOOLS, "parallel_tool_calls": False}, emitted.append)
        reply.push(text[:20])
        reply.push(text[20:] + ' [[call list {}]]')
        content, calls = reply.finish()
        self.assertEqual(content, "Looking.[[call nope {}]] [[call list {}]]")
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                         [("read_file", {"path": "a.py"})])
        self.assertEqual(emitted[1], {"tool_calls": [dict(calls[0], index=0)]})
        self.assertEqual("".join(e for e in emitted if isinstance(e, str)), content)


if __name__ == "__main__":
    unittest.main()
