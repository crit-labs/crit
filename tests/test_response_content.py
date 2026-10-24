"""Serving content rules, independent of metadata and model/tokenizer implementation."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for folder in ("api", "cli", "core/python", "eval"):
    sys.path.insert(0, str(ROOT / folder))

from crit_api.engine import reply
from crit_api.host import Cancellation
from crit_api.stream import StreamReader, StreamWriter


CALL = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"key":"value"}'}}


class ResponseContentTests(unittest.TestCase):
    def setUp(self):
        self.request = {
            "request_id": "r", "candidate_id": "c", "package_sha256": "p",
            "limits": {"kv_tokens": 100, "max_output_tokens": 8, "timeout_seconds": 10},
            "tools": [{"type": "function", "function": {"name": "lookup"}}], "tool_choice": "auto",
        }
        self.response = {
            "schema_version": 1, "record_kind": "chat_response",
            "request_id": "r", "candidate_id": "c", "package_sha256": "p", "assistant": "Answer",
            "telemetry": {"prompt_tokens": 2, "output_tokens": 4, "prefill_tokens": 2, "prefill_ms": 1,
                          "decode_tokens": 3, "decode_ms": 1, "ttft_ms": 1, "missing": {}},
        }

    def writer(self, tools=False):
        self.sent = []
        return StreamWriter(self.sent.append, "command", self.request, Cancellation(), tools=tools)

    def piece(self, **part):
        return {"kind": "progress", "command_id": "command", **part}

    def test_reasoning_only_and_tool_only_preserve_content_and_telemetry(self):
        for assistant in ("", " \n"):
            for extra in ({"reasoning": "still thinking"},
                          {"tool_calls": [CALL]},
                          {"reasoning": "why", "tool_calls": [CALL]}):
                with self.subTest(assistant=assistant, fields=list(extra)):
                    response = self.response | {"assistant": assistant} | extra
                    original = copy.deepcopy(response)
                    result, measured = reply(response, self.request)
                    self.assertEqual(result, {"content": assistant, **extra})
                    self.assertEqual(measured["output_tokens"], 4)
                    self.assertEqual(response, original)

    def test_empty_or_invalid_channels_are_rejected_even_with_other_content(self):
        for assistant in ("", " \n"):
            with self.subTest(assistant=assistant), self.assertRaises(ValueError):
                reply(self.response | {"assistant": assistant}, self.request)
        for field in ("assistant", "reasoning"):
            for invalid in (None, 1, [], "x\0y", "\ud800", "x" * 65537):
                with self.subTest(field=field, invalid=repr(invalid)[:30]), self.assertRaises(ValueError):
                    reply(self.response | {"reasoning": "why", field: invalid}, self.request)
        with self.assertRaises(ValueError):
            reply(self.response | {"reasoning": ""}, self.request)

    def test_exact_character_limits_and_combined_tool_budget(self):
        for extra in ({}, {"reasoning": "界" * 65536}):
            response = self.response | {"assistant": "界" * 65536} | extra
            result, _ = reply(response, self.request)
            self.assertEqual(result, {"content": response["assistant"], **extra} if extra else response["assistant"])
        call_size = len(json.dumps([CALL], ensure_ascii=False))
        response = self.response | {"assistant": "界" * (65536 - call_size), "tool_calls": [CALL]}
        self.assertEqual(reply(response, self.request)[0]["tool_calls"], [CALL])
        with self.assertRaises(ValueError):
            reply(response | {"assistant": response["assistant"] + "x"}, self.request)

    def test_decode_rounds_give_tokens_per_round_within_bounds(self):
        telemetry = self.response["telemetry"]
        _, measured = reply(self.response | {"telemetry": telemetry | {"decode_rounds": 2}}, self.request)
        self.assertEqual(measured["tokens_per_round"], 2.0)  # 4 output tokens over 2 rounds
        self.assertNotIn("tokens_per_round", reply(self.response, self.request)[1])
        for change in ({"decode_rounds": 0}, {"decode_rounds": 5}, {"decode_rounds": True}, {"decode_rounds": 1.5},
                       {"decode_rounds": 1, "decode_tokens": None, "decode_ms": None,
                        "missing": {"decode": "single token"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                reply(self.response | {"telemetry": telemetry | change}, self.request)

    def test_reasoning_does_not_bypass_identity_telemetry_or_tool_policy(self):
        response = self.response | {"assistant": "", "reasoning": "still thinking"}
        for change in ({"request_id": "stale"}, {"schema_version": True},
                       {"telemetry": response["telemetry"] | {"output_tokens": 9}},
                       {"tool_calls": [dict(CALL, function={"name": "unknown", "arguments": "{}"})]},
                       {"tool_calls": [dict(CALL, function={"name": "lookup", "arguments": "{"})]}):
            with self.subTest(fields=list(change)), self.assertRaises(ValueError):
                reply(response | change, self.request)
        with self.assertRaisesRegex(ValueError, "required tool call"):
            reply(response, self.request | {"tool_choice": "required"})
        # A reply its output limit cut before the forced call began ends as "length", as a cut call does.
        cut = response | {"telemetry": response["telemetry"] | {"output_tokens": 8, "decode_tokens": 7}}
        assistant, _ = reply(cut, self.request | {"tool_choice": "required"})
        self.assertEqual(assistant, {"content": "", "reasoning": "still thinking"})
        with self.assertRaisesRegex(ValueError, "reused a tool call id"):
            reply(response | {"tool_calls": [CALL]},
                  self.request | {"messages": [{"role": "assistant", "tool_calls": [CALL]}]})

    def test_final_stream_content_matches_reply_with_and_without_tool_deltas(self):
        cases = [
            ({"assistant": "Answer"}, True),
            ({"assistant": "", "reasoning": "still thinking"}, True),
            ({"assistant": " \n", "reasoning": "why"}, True),
            ({"assistant": "", "tool_calls": [CALL]}, True),
            ({"assistant": "", "reasoning": "why", "tool_calls": [CALL]}, True),
            ({"assistant": ""}, False),
            ({"assistant": " \n"}, False),
            ({"assistant": "Answer", "reasoning": ""}, False),
            ({"assistant": "", "tool_calls": [dict(CALL, function={"name": "unknown", "arguments": "{}"})]}, False),
            ({"assistant": "", "tool_calls": [dict(CALL, function={"name": "lookup", "arguments": "{"})]}, False),
        ]
        for fields, valid in cases:
            for tools in (False, True):
                with self.subTest(fields=fields, tools=tools):
                    writer = self.writer(tools)

                    def finish():
                        if fields.get("reasoning"):
                            writer.emit({"reasoning": fields["reasoning"]})
                        if fields["assistant"]:
                            writer.emit(fields["assistant"])
                        if tools and fields.get("tool_calls"):
                            writer.emit({"tool_calls": [dict(c, index=i) for i, c in enumerate(fields["tool_calls"])]})
                        writer.finish(fields)

                    if valid:
                        finish()
                        reply(self.response | fields, self.request)
                    else:
                        with self.assertRaises(ValueError):
                            finish()
                        with self.assertRaises(ValueError):
                            reply(self.response | fields, self.request)

    def test_only_a_reply_cut_by_its_output_limit_keeps_a_cut_last_call(self):
        cut = dict(CALL, id="call_2", function={"name": "lookup", "arguments": '{"key": "va'})
        telemetry = self.response["telemetry"] | {"output_tokens": 8, "decode_tokens": 7}
        response = self.response | {"assistant": "", "telemetry": telemetry}
        for calls in ([cut], [CALL, cut]):
            with self.subTest(calls=len(calls)):
                self.assertEqual(reply(response | {"tool_calls": calls}, self.request)[0],
                                 {"content": "", "tool_calls": calls})
                writer = self.writer(tools=True)
                writer.emit({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                writer.finish(response | {"tool_calls": calls})
        for fields in ({"tool_calls": [cut]} | {"telemetry": self.response["telemetry"]},  # not cut: a malformed call
                       {"tool_calls": [cut, CALL]},  # only the last call can be cut
                       {"tool_calls": [dict(cut, function={"name": "lookup", "arguments": '"va'})]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                reply(response | fields, self.request)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.writer().finish(response | fields)

    def test_stream_prefixes_preserve_whitespace_and_incomplete_tool_arguments(self):
        writer = self.writer(tools=True)
        text, deltas = [], []
        reader = StreamReader(self.request, text.append, deltas.extend)
        writer.emit(" \n")
        writer.emit({"tool_calls": [dict(CALL, index=0, function={"name": "lookup", "arguments": "{"})]})
        for piece in self.sent:
            reader.apply(piece)
        self.assertEqual(text, [" \n"])
        self.assertEqual(deltas[0]["function"]["arguments"], "{")
        writer.emit({"tool_calls": [{"index": 0, "function": {"arguments": CALL["function"]["arguments"][1:]}}]})
        writer.finish({"assistant": " \n", "tool_calls": [CALL]})
        reader.apply(self.sent[-1])
        self.assertEqual("".join(d["function"]["arguments"] for d in deltas), CALL["function"]["arguments"])

    def test_stream_budgets_count_tools_and_keep_reasoning_separate(self):
        writer = self.writer(tools=True)
        assistant = "界" * (65536 - len(json.dumps([CALL], ensure_ascii=False)))
        reasoning = "界" * 65536
        text, thoughts = [], []
        reader = StreamReader(self.request, text.append, publish_reasoning=thoughts.append)
        writer.emit(assistant)
        writer.emit({"tool_calls": [dict(CALL, index=0)]})
        writer.emit({"reasoning": reasoning})
        writer.finish({"assistant": assistant, "reasoning": reasoning, "tool_calls": [CALL]})
        for piece in self.sent:
            reader.apply(piece)
        self.assertEqual(text, [assistant])
        self.assertEqual(thoughts, [reasoning])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            writer.emit("x")
        for piece in (self.piece(text="x"), self.piece(reasoning="x")):
            with self.subTest(piece=piece), self.assertRaisesRegex(ValueError, "exceeds"):
                reader.apply(piece)
        self.assertEqual(text, [assistant])
        self.assertEqual(thoughts, [reasoning])

    def test_invalid_piece_channels_fail_before_publication(self):
        for piece in (self.piece(text=None), self.piece(text=[]), self.piece(text="\ud800"),
                      self.piece(text="x\0y"), self.piece(text=""),
                      self.piece(reasoning=""), self.piece(reasoning=None), self.piece(reasoning="\ud800"),
                      self.piece(tool_calls=False), self.piece(tool_calls=[])):
            with self.subTest(piece=piece):
                text, thoughts, calls = [], [], []
                reader = StreamReader(self.request, text.append, calls.extend, thoughts.append)
                with self.assertRaises(ValueError):
                    reader.apply(piece)
                self.assertEqual((text, thoughts, calls), ([], [], []))


if __name__ == "__main__":
    unittest.main()
