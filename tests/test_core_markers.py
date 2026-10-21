"""Token identity at reasoning/tool boundaries, using real offline tokenizers and shared parsers."""
import json
from pathlib import Path
import sys
import unittest

import tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core/python"))

from core_text_fixtures import byte_level_json, sentencepiece_json, straddling_json  # noqa: E402
from crit_core.text import (MarkerDecoder, ReasoningFormat, ReasoningSplitter, TextError, TokenText,  # noqa: E402
                            Tokenizer, ToolCallParser)
from crit_core.text.markers import marker_ids  # noqa: E402
from crit_core.text.tools import _format  # noqa: E402

MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>")
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
CALL = '{"name":"lookup","arguments":{}}'


def marked_tokenizer(source=byte_level_json, markers=MARKERS):
    backend = tokenizers.Tokenizer.from_str(source())
    backend.add_special_tokens([tokenizers.AddedToken(m, special=True, normalized=False) for m in markers])
    return Tokenizer.from_bytes(backend.to_str().encode())


def spelling(value):
    return value.text if isinstance(value, TokenText) else value


def encoded(tokenizer, text, literal=False):
    return tokenizer.encode(text, add_special_tokens=False, split_special_tokens=literal)


def decoded_parts(tokenizer, ids, chunk=1, prompt=()):
    decoder = MarkerDecoder(tokenizer, prompt, markers={v: k for k, v in marker_ids(tokenizer, MARKERS).items()},
                            skip_special_tokens=False)
    parts = []
    for start in range(0, len(ids), chunk):
        parts += decoder.push(ids[start:start + chunk])
    return parts + decoder.finish()


def parsed(parser, parts):
    events = []
    for part in parts:
        events += parser.push(part)
    return events + parser.finish()


def joined(events, kind):
    return "".join(spelling(value) for event, value in events if event == kind)


class MarkerDecoderTests(unittest.TestCase):
    def test_identical_spelling_retains_only_real_delimiter_ids(self):
        for source in (byte_level_json, sentencepiece_json, straddling_json):
            tok = marked_tokenizer(source)
            prompt = encoded(tok, "prefix ")
            for literal in (False, True):
                ids = encoded(tok, "日本 😀 <think>plan</think>  next", literal)
                for chunk in (1, 3, 4096):
                    with self.subTest(source=source.__name__, literal=literal, chunk=chunk):
                        parts = decoded_parts(tok, ids, chunk, prompt)
                        self.assertEqual("".join(map(spelling, parts)),
                                         tok.decode(prompt + ids)[len(tok.decode(prompt)):])
                        self.assertEqual([p.text for p in parts if isinstance(p, TokenText)],
                                         [] if literal else ["<think>", "</think>"])

    def test_incomplete_utf8_before_marker_is_flushed_without_becoming_marker_text(self):
        tok = marked_tokenizer(straddling_json)
        ids = [0xE2, tok.token_to_id("</think>")] + encoded(tok, " tail")
        parts = decoded_parts(tok, ids)
        self.assertEqual(parts[:2], ["\ufffd", TokenText("</think>", tok.token_to_id("</think>"))])
        self.assertEqual("".join(map(spelling, parts)), tok.decode(ids))

    def test_incompatible_marker_decode_fails_and_closes(self):
        tok = marked_tokenizer()
        decoder = MarkerDecoder(tok, markers={tok.token_to_id("<think>"): "wrong"}, skip_special_tokens=False)
        with self.assertRaisesRegex(TextError, "did not decode"):
            decoder.push(encoded(tok, "<think>"))
        for ids in ([], encoded(tok, "next")):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                decoder.push(ids)

    def test_stop_ids_flush_pending_text_and_never_become_delimiters(self):
        tok = marked_tokenizer(straddling_json)
        stop = tok.token_to_id("</think>")
        decoder = MarkerDecoder(tok, markers={stop: "</think>"}, stop_token_ids=[stop], skip_special_tokens=False)
        self.assertEqual(decoder.push([0xE2, stop] + encoded(tok, "ignored")), ["\ufffd"])
        self.assertEqual(decoder.push(encoded(tok, "also ignored")), [])
        self.assertEqual(decoder.finish(), [])


class MarkerParserTests(unittest.TestCase):
    def setUp(self):
        self.tok = marked_tokenizer()

    def parts(self, text, literal=False, chunk=1):
        return decoded_parts(self.tok, encoded(self.tok, text, literal), chunk)

    def test_literal_reasoning_open_and_close_never_switch_channels(self):
        for chunk in (1, 3, 4096):
            for opened in (False, True):
                text = "<think>Quote </think> literally; still here."
                events = parsed(ReasoningSplitter("think-tags", opened=opened, tokenizer=self.tok),
                                self.parts(text, True, chunk))
                self.assertEqual(joined(events, "reasoning"), text if opened else "")
                self.assertEqual(joined(events, "text"), "" if opened else text)

    def test_real_close_after_literal_close_and_tools_after_reasoning_keep_identity(self):
        parts = (self.parts("<think>") + self.parts("Quote </think> literally.", True)
                 + self.parts("</think>") + self.parts("<tool_call>" + CALL + "</tool_call>", True)
                 + self.parts(" <tool_call>" + CALL + "</tool_call>"))
        events = parsed(ReasoningSplitter("think-tags", tokenizer=self.tok), parts)
        self.assertEqual(joined(events, "reasoning"), "Quote </think> literally.")
        tools = parsed(ToolCallParser("json-object", TOOLS, tokenizer=self.tok),
                       [v for k, v in events if k == "text"])
        self.assertEqual(joined(tools, "text"), "<tool_call>" + CALL + "</tool_call>")
        self.assertEqual([v for k, v in tools if k == "call"], [{"name": "lookup", "arguments": {}}])

    def test_literal_tool_closer_inside_arguments_does_not_close_the_call(self):
        body = json.dumps({"name": "lookup", "arguments": {"quote": "</tool_call>", "emoji": "😀"}}, ensure_ascii=False)
        parts = self.parts("<tool_call>") + self.parts(body, True) + self.parts("</tool_call>")
        events = parsed(ToolCallParser("json-object", TOOLS, tokenizer=self.tok), parts)
        self.assertEqual(events, [("call", json.loads(body))])

    def test_malformed_unclosed_unknown_and_extra_calls_stay_text(self):
        for body in ("not JSON", '{"name":"unknown"}', CALL):
            for closed in (False, True):
                suffix = "<tool_call>" + body + ("</tool_call>" if closed else "")
                parts = self.parts("<tool_call>" + CALL + "</tool_call>") + self.parts(suffix)
                events = parsed(ToolCallParser("json-object", TOOLS, parallel=False, tokenizer=self.tok), parts)
                self.assertEqual(joined(events, "text"), suffix)
                self.assertEqual(len([v for k, v in events if k == "call"]), 1)

    def test_explicit_text_policy_and_mixed_boundaries(self):
        fmt = ReasoningFormat("<think>", "</think>", delimiters="text")
        events = parsed(ReasoningSplitter(fmt, tokenizer=self.tok), self.parts("<think>why</think>answer", True))
        self.assertEqual((joined(events, "reasoning"), joined(events, "text")), ("why", "answer"))
        fmt = _format("json-object")._replace(delimiters="text")
        events = parsed(ToolCallParser(fmt, TOOLS, tokenizer=self.tok),
                        self.parts("<tool_call>" + CALL + "</tool_call>", True))
        self.assertEqual(events, [("call", {"name": "lookup", "arguments": {}})])
        # A format may have a special opening token and a multi-token textual closing marker.
        fmt = ReasoningFormat("<think>", "[[/r]]")
        for chunk in (1, 4096):
            events = parsed(ReasoningSplitter(fmt, tokenizer=self.tok), self.parts("<think>why[[/r]]answer", chunk=chunk))
            self.assertEqual((joined(events, "reasoning"), joined(events, "text")), ("why", "answer"))
        with self.assertRaisesRegex(TextError, "delimiters"):
            ReasoningSplitter(fmt._replace(delimiters="typo"), tokenizer=self.tok)


if __name__ == "__main__":
    unittest.main()
