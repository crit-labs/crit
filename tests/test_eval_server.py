"""crit eval run --url: an engine behind a fake OpenAI-compatible chat server."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval import chat, prepare, reference, runner, status  # noqa: E402
from test_eval import CharacterText  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
CHAT = [{"role": "user", "content": "Explain tides."}]
AGENT = [{"role": "user", "content": "Fix it."},
         {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
             "name": "read_file", "arguments": {"path": "a.py"}}}]},
         {"role": "tool", "content": "print(1)"}]
HELDOUT = [{"role": "user", "content": "Erkläre Ebbe und Flut."}]
# The long-context curve's and retrieval's frames: their documents go where SENTINEL is.
CURVE = [{"role": "user", "content": "Summarize.\n\n" + prepare.SENTINEL}]
RETRIEVAL = [{"role": "system", "content": "Find the facts."}, {"role": "user", "content": prepare.SENTINEL + "\n\nList them."}]
REPLY = "tides rise and fall " * 9
QUALITY = [{"role": "user", "content": "Why are there two tides?"}]
LONG_QUALITY = [{"role": "user", "content": "Background. " * 250 + "Why are there two tides?"}]
TOO_LONG = [{"role": "user", "content": "Background. " * 400 + "Why are there two tides?"}]
OTHERS = "abcdefghijklmnopq"  # the alternatives the fake server reports after each reply token


class Text(CharacterText):
    def render(self, messages, tools=None, **variables):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<a>"


def ids(messages, tools=None):
    return prepare.token_ids(Text(), Text().render(messages, tools))


def frame(messages):
    head, tail = prepare.document_frame(Text(), messages)
    return {"head_ids": head, "tail_ids": tail, "messages": prepare.api_messages(messages)}


def write_data(directory):
    protocol = json.loads((ROOT / "eval/protocol.json").read_text(encoding="utf-8"))
    speed = [("chat", CHAT, None), ("tool-call", AGENT, TOOLS)]
    prepared = {"protocol": protocol["protocol"], "suite_sha256": prepare.suite_identity(), "model": {"sha256": "model"},
                "stop_ids": [], "valid_vocab": 1 << 16,
                "speed": [{"id": i, "kind": i, "prompt_ids": ids(m, t), "messages": prepare.api_messages(m), "tools": t,
                           "output_tokens": 12} for i, m, t in speed],
                "heldout": [{"id": "de", "language": "German", "kind": "prose", "prompt_ids": ids(HELDOUT),
                             "messages": HELDOUT, "output_tokens": 12}],
                "quality": [{"id": i, "set": s, "prompt_ids": ids(m), "messages": m} for i, s, m in (
                    ("tides", "conversations", QUALITY), ("tides@3000", "long-3000", LONG_QUALITY),
                    ("tides@5000", "long-5000", TOO_LONG))],
                "context": {"case": "chat", **frame(CURVE), "tools": None,
                            "document_ids": prepare.token_ids(Text(), "A long document.\n\n")},
                "retrieval": {**frame(RETRIEVAL), "filler_ids": prepare.token_ids(Text(), "Filler text. "),
                              "facts": [{"id": "start", "answer": "tides", "ids": prepare.token_ids(Text(), "\n\ntides\n\n")}],
                              "token_text": [chr(i) if chr(i).isascii() else None for i in range(256)]}}
    (directory / "prepared.json").write_text(json.dumps(prepared))


class FakeChat(BaseHTTPRequestHandler):
    """Greedy chat: the reply is REPLY's first max_tokens characters, one per chunk; with tools, some reasoning and
    one tool call that ends the reply early, as a chat server does."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.headers.append(dict(self.headers))
        if self.path != "/v1/models":
            return self.send_error(404)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"object": "list", "data": [{"id": m} for m in self.server.models]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        if body.get("logprobs"):
            return self.reply(body)
        if body.get("tools"):
            deltas = [{"reasoning_content": "look"}, {"tool_calls": [{"index": 0, "function": {"name": "read_file"}}]},
                      {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "a.py"}'}}]}]
            finish, used = "tool_calls", 9
        else:
            reply = REPLY[:body["max_tokens"]]
            deltas = [{"content": reply}] if self.server.one_piece else [{"content": c} for c in reply]
            finish, used = "length", len(reply)
        usage = {"prompt_tokens": 1000 + len(body["messages"]), "completion_tokens": used, **self.server.usage}
        events = ([{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}]
                  + [{"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
                  + [{"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}]
                  + ([{"choices": [], "usage": usage}] if body["stream_options"]["include_usage"] and self.server.streams_usage else []))
        if self.server.stream_error:
            events = events[:1] + [{"error": {"message": self.server.stream_error, "type": "server_error"}}]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


    def reply(self, body):
        """A quality reply: REPLY's first max_tokens characters, each with its bytes, its log-probability and
        top_logprobs alternatives (itself, then letters), as server.logprobs shapes them."""
        shape = self.server.logprobs

        def entry(token, logprob):
            return {"token": token, "logprob": logprob, "bytes": None if shape.get("no_bytes") else list(token.encode())}
        content = []
        for c in REPLY[:body["max_tokens"]]:
            others = [entry(o, -3.0 - i) for i, o in enumerate(o for o in OTHERS if o != c)]
            content.append({**entry(c, -0.1), "top_logprobs": ([entry(c, -0.1)] + others)[:shape.get("top", body["top_logprobs"])]})
        finish = "length"
        if shape.get("end_token"):
            content[-1:] = [{"token": "", "logprob": -0.01, "bytes": [], "top_logprobs": []}]
            finish = "stop"
        if shape.get("partial"):
            content[0] = {**content[0], "token": "\ufffd", "bytes": list("\ufffd".encode())}
        choice = {"index": 0, "message": {"role": "assistant", "content": REPLY[:body["max_tokens"]]}, "finish_reason": finish,
                  "logprobs": None if shape.get("none") else {"content": content}}
        usage = {"prompt_tokens": len(ids(body["messages"])) + shape.get("prompt_offset", 0),
                 "completion_tokens": len(content), **self.server.usage}
        data = json.dumps({"choices": [choice], "usage": usage}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


class ChatServerTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
        self.server.requests, self.server.headers, self.server.models = [], [], ["fake"]
        self.server.usage, self.server.streams_usage, self.server.logprobs = {}, True, {}
        self.server.one_piece, self.server.stream_error = False, None
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        # Text's ids are characters; a character's id is its UTF-8 bytes' (U+FFFD is no token, as in ids_by_bytes).
        by_bytes = {chr(i).encode(): i for i in range(1, 0xD800) if i != 0xFFFD}
        for patcher in (patch.object(chat, "load_text", lambda model: Text()),
                        patch.object(chat, "ids_by_bytes", lambda tokenizer: by_bytes)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.data = Path(tempfile.mkdtemp())
        write_data(self.data)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def run_eval(self, parts, repetitions=2):
        return runner.run(None, "unused-model", self.data, self.data / "result.json", parts=parts, warmups=0,
                          repetitions=repetitions, log=lambda message: None, context_tokens=4096, url=self.url + "/")

    def engine(self, served_model=None, context_tokens=4096):
        prepared = json.loads((self.data / "prepared.json").read_text())
        return runner.open_engine(chat.Server(self.url, prepared, served_model), "unused-model", {}, True, context_tokens)

    def test_speed_held_out_and_history_prompts_go_as_their_frozen_messages(self):
        result = self.run_eval(("speed", "heldout", "history"))
        self.assertEqual((result["engine"]["name"], result["engine"]["max_context"], result["server"]),
                         ("fake", 4096, self.url + "/"))
        chat_case, tool = result["speed"]["cases"]["chat"], result["speed"]["cases"]["tool-call"]
        self.assertTrue(chat_case["repeatable"] and tool["repeatable"])
        self.assertEqual(chat_case["tokens"], prepare.token_ids(Text(), REPLY[:12]))
        sample = chat_case["samples"][0]
        self.assertEqual((sample["output_tokens"], sample["served"]["prompt_tokens"]), (12, 1001))
        self.assertIsNotNone(sample["decode_tps"])
        # The tool call ends the reply early: measured on the 9 tokens the server counted.
        self.assertEqual(tool["samples"][0]["output_tokens"], 9)
        self.assertEqual(tool["samples"][0]["served"]["finish_reason"], "tool_calls")
        self.assertEqual(tool["tokens"], prepare.token_ids(Text(), 'lookread_file{"path": "a.py"}'))
        self.assertEqual(result["heldout"]["prompts"]["de"]["status"], "measured")
        self.assertTrue(result["history"]["passed"])
        sent = self.server.requests
        self.assertEqual({(r["model"], r["temperature"], r["max_tokens"], r["stream"]) for r in sent},
                         {("fake", 0, 12, True)})
        agent = next(r for r in sent if r.get("tools"))
        self.assertEqual(agent["tools"], TOOLS)
        call, answer = agent["messages"][1]["tool_calls"][0], agent["messages"][2]
        self.assertEqual((call["function"]["arguments"], answer["tool_call_id"]), ('{"path": "a.py"}', call["id"]))

    def test_a_reply_streamed_in_one_piece_has_no_decode_speed(self):
        self.server.one_piece = True
        logged = []
        result = runner.run(None, "unused-model", self.data, self.data / "result.json", parts=("speed", "heldout"),
                            warmups=0, repetitions=2, log=logged.append, context_tokens=4096, url=self.url)
        self.assertIsNone(result["speed"]["cases"]["chat"]["decode_tps"])
        self.assertTrue(any(line.startswith("speed chat: decode unmeasured") for line in logged), logged)
        self.assertTrue(any(line.startswith("held-out de (German): decode unmeasured") for line in logged), logged)

    def test_long_context_and_retrieval_prompts_go_as_their_frames_messages(self):
        result = self.run_eval(("speed", "context"), repetitions=1)
        point = result["context"]["points"]["4096"]
        self.assertEqual(point["status"], "measured")
        self.assertEqual(point["samples"][0]["output_tokens"], len(REPLY))  # the reply ends before 256 tokens
        self.assertEqual(result["context"]["retrieval"]["points"]["4096"]["found"], {"start": True})
        self.assertEqual(result["context"]["lower"]["reason"], "a served engine cannot be opened at a lower context")
        curve = next(r for r in self.server.requests if r["messages"][0]["content"].startswith("Summarize."))
        self.assertTrue(curve["messages"][0]["content"].startswith("Summarize.\n\nA long document.\n\nFiller text. Filler"))
        self.assertEqual(len(ids(curve["messages"])), 4096 - 256)
        retrieval = next(r for r in self.server.requests if r["messages"][0]["role"] == "system")
        self.assertIn("\n\ntides\n\n", retrieval["messages"][1]["content"])

    def test_a_server_must_stream_usage_and_prefill_every_request(self):
        engine, _ = self.engine()
        prompt = json.loads((self.data / "prepared.json").read_text())["speed"][0]["prompt_ids"]
        self.server.streams_usage = False
        with self.assertRaisesRegex(RuntimeError, "include_usage"):
            engine.generate(prompt, 4, lambda tokens: None)
        self.server.streams_usage, self.server.usage = True, {"prompt_tokens_details": {"cached_tokens": 900}}
        with self.assertRaisesRegex(RuntimeError, "prompt cache"):
            engine.generate(prompt, 4, lambda tokens: None)

    def test_an_error_the_server_streams_is_reported_as_its_own(self):
        engine, _ = self.engine()
        prompt = json.loads((self.data / "prepared.json").read_text())["speed"][0]["prompt_ids"]
        self.server.stream_error = "Conversation exceeds the limit"
        with self.assertRaisesRegex(RuntimeError, "streamed an error: Conversation exceeds the limit"):
            engine.generate(prompt, 4, lambda tokens: None)

    def test_a_prompt_the_data_does_not_send_fails_the_request(self):
        engine, _ = self.engine()
        with self.assertRaises(RuntimeError):
            engine.generate([1, 2, 3], 4, lambda tokens: None)

    def test_the_server_key_comes_from_the_environment_and_stays_out_of_the_result(self):
        with patch.dict(os.environ, {"CRIT_SERVER_KEY": "secret"}):
            _, info = self.engine()
        self.assertEqual(self.server.headers[-1].get("Authorization"), "Bearer secret")
        self.assertNotIn("secret", json.dumps(info))

    def test_the_run_names_the_model_and_its_context(self):
        with self.assertRaisesRegex(SystemExit, "--context"):
            self.engine(context_tokens=None)
        self.server.models = ["a", "b"]
        with self.assertRaisesRegex(SystemExit, "--served-model"):
            self.engine()
        with self.assertRaisesRegex(SystemExit, "does not list c"):
            self.engine("c")
        self.assertEqual(self.engine("b")[1]["name"], "b")

    def prompt(self, prompt_id):
        prepared = json.loads((self.data / "prepared.json").read_text())
        return next(p["prompt_ids"] for p in prepared["quality"] if p["id"] == prompt_id)

    def test_quality_replies_come_with_the_engines_log_probabilities(self):
        result = self.run_eval(("quality",))
        quality = result["quality"]
        self.assertEqual((quality["status"], quality["top_k"], quality["reply_tokens"]), ("replied", 16, 256))
        self.assertEqual([(r["id"], r["set"]) for r in quality["replies"]],
                         [("tides", "conversations"), ("tides@3000", "long-3000")])
        self.assertEqual(quality["skipped"], {"long-5000": "needs 5,089 tokens of context"})
        tides = quality["replies"][0]
        self.assertEqual(tides["tokens"], prepare.token_ids(Text(), REPLY))  # the reply ends before 256 tokens
        self.assertEqual(tides["logprobs"][0], -0.1)
        self.assertEqual(tides["top"][0][:2], [[ord("t"), -0.1], [ord("a"), -3.0]])
        self.assertEqual(len(tides["top"][0]), 16)
        self.assertEqual((tides["finish_reason"], tides["prompt_tokens"]), ("length", len(self.prompt("tides"))))
        sent = self.server.requests[0]
        self.assertEqual((sent["messages"], sent["logprobs"], sent["top_logprobs"], sent["temperature"]), (QUALITY, True, 16, 0))
        self.assertNotIn("stream", sent)
        gate = next(g for g in result["status"]["gates"] if g["name"] == "quality")
        self.assertEqual((gate["state"], gate["detail"]), ("unmeasured", "2 replies await the reference: crit eval reference --result"))

    def test_a_quality_reply_must_name_every_token_by_its_bytes_with_its_alternatives(self):
        engine, _ = self.engine()
        prompt = self.prompt("tides")
        for shape, message in (({"none": True}, "no logprobs"), ({"no_bytes": True}, "no bytes"),
                               ({"top": 3}, "returned 3 top_logprobs"), ({"partial": True}, "no token of the model"),
                               ({"prompt_offset": 1}, "renders"), ({}, None)):
            self.server.logprobs = shape
            if message is None:
                self.assertEqual(len(engine.reply(prompt, 4, 16)["tokens"]), 4)
                continue
            with self.assertRaisesRegex(RuntimeError, message):
                engine.reply(prompt, 4, 16)
        # An end token the server reports as no text ends the reply; it is not scored.
        self.server.logprobs = {"end_token": True}
        out = engine.reply(prompt, 4, 16)
        self.assertEqual((out["tokens"], out["finish_reason"]), (prepare.token_ids(Text(), "tid"), "stop"))
        self.server.logprobs, self.server.usage = {}, {"prompt_tokens_details": {"cached_tokens": 3}}
        with self.assertRaisesRegex(RuntimeError, "prompt cache"):
            engine.reply(prompt, 4, 16)

    def test_the_reference_scores_the_replies_on_the_prepared_checkpoint(self):
        self.run_eval(("quality",))
        path = self.data / "result.json"

        class Reference:
            """Scores every reply token as the fake server reported it: the reference agrees with the engine."""
            def __init__(self, model):
                self.model = model

            def score(self, sequences, top_k, valid_vocab, log, long):
                self.long = long
                return [[{"target": t, "logprob": -0.1, "top_ids": [t] + [ord(o) for o in OTHERS if ord(o) != t][:top_k - 1],
                          "top_logprobs": [-0.1] + [-3.0 - i for i in range(top_k - 1)]} for t in ids[first:]]
                        for ids, first in sequences]
        with patch.object(reference, "StreamingReference", Reference), \
                patch.object(reference, "model_identity", lambda model: {"sha256": "model" if model == "source" else "x"}), \
                patch.dict(sys.modules, {"transformers": type(sys)("transformers")}):
            sys.modules["transformers"].__version__ = "test"
            with self.assertRaisesRegex(SystemExit, "not the checkpoint"):
                reference.score_result("other", self.data, path, log=lambda m: None)
            result = reference.score_result("source", self.data, path, log=lambda m: None)
        quality = json.loads(path.read_text())["quality"]
        self.assertEqual(quality, result["quality"])
        self.assertEqual(quality["status"], "scored")
        self.assertEqual(set(quality["sets"]), {"conversations", "long-3000"})
        self.assertAlmostEqual(quality["sets"]["conversations"]["kld_mean"], 0.0, places=6)
        self.assertEqual(quality["sets"]["conversations"]["top1_agreement"], 1.0)
        self.assertTrue(quality["passed"])
        gate = next(g for g in result["status"]["gates"] if g["name"] == "quality")
        self.assertEqual(gate["state"], "pass")
        self.assertIn("skipped: long-5000", gate["detail"])
        self.assertIn("of the drift budget used", status.render(result["status"]))


class EngineLockTests(unittest.TestCase):
    def test_a_served_engine_is_measured_without_the_engine_lock(self):
        from crit_eval import cli
        taken = []

        @contextmanager
        def lock(holder):
            taken.append(holder)
            yield
        common = ["--model", "m", "--data", "d", "--out", "o.json"]
        with patch("crit_core.machine.engine_lock", lock), patch.object(cli, "_command", lambda args: 0):
            cli.main(["run", "--url", "http://127.0.0.1:1/v1", "--context", "4096", *common])
            self.assertEqual(taken, [])  # crit serve holds the lock while it serves
            cli.main(["run", "--adapter", "a.py", *common])
        self.assertEqual(taken, ["crit eval run"])


class TokenBytesTests(unittest.TestCase):
    """A server names reply tokens by their bytes; Eval maps them to the model's ids through its vocabulary."""

    def test_reply_tokens_map_back_to_their_ids_in_byte_level_and_sentencepiece_models(self):
        import core_text_fixtures as fixtures
        from crit_core.text.tokenizer import Tokenizer
        for data in (fixtures.byte_level_json(), fixtures.sentencepiece_json()):
            tokenizer = Tokenizer(data)
            by_bytes = chat.ids_by_bytes(tokenizer)
            for token in tokenizer.encode("Grüße 😀 tides", add_special_tokens=False):
                self.assertEqual(by_bytes[tokenizer.token_bytes()[token]], token)

    def test_an_ordinary_piece_wins_over_a_byte_piece_with_the_same_bytes(self):
        import core_text_fixtures as fixtures
        from crit_core.text.tokenizer import Tokenizer
        tokenizer = Tokenizer(fixtures.sentencepiece_json())
        byte = tokenizer.token_to_id("<0x61>")
        piece = tokenizer.token_to_id("a")
        self.assertIsNotNone(byte)
        self.assertIsNotNone(piece)
        self.assertEqual(chat.ids_by_bytes(tokenizer)[b"a"], piece)


if __name__ == "__main__":
    unittest.main()
