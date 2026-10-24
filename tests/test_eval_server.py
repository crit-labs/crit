"""crit eval run --url: an engine behind a fake OpenAI-compatible chat server."""
import array
import base64
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
from crit_eval import chat, fixed, prepare, reference, runner, status  # noqa: E402
from test_eval import CharacterText  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
CHAT = [{"role": "user", "content": "Explain tides."}]
AGENT = [{"role": "user", "content": "Fix it."},
         {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
             "name": "read_file", "arguments": {"path": "a.py"}}}]},
         {"role": "tool", "content": "print(1)"}]
HELDOUT = [{"role": "user", "content": "Erkläre Ebbe und Flut."}]
# The retrieval's and the long-context curve's frames, one archive: their documents go where SENTINEL is.
RETRIEVAL = [{"role": "system", "content": "Find the facts."}, {"role": "user", "content": prepare.SENTINEL + "\n\nList them."}]
CURVE = [RETRIEVAL[0], {"role": "user", "content": prepare.SENTINEL + "\n\nList them, then summarize."}]
# Follow-up turns: each resends the conversation so far.
FOLLOWUP = [CHAT, CHAT + [{"role": "assistant", "content": "The Moon."}, {"role": "user", "content": "Why?"}]]
REPLY = "tides rise and fall " * 9
QUALITY = [{"role": "user", "content": "Why are there two tides?"}]
LONG_QUALITY = [{"role": "user", "content": "Background. " * 250 + "Why are there two tides?"}]
TOO_LONG = [{"role": "user", "content": "Background. " * 400 + "Why are there two tides?"}]
OTHERS = "abcdefghijklmnopq"  # the alternatives the fake server reports after each reply token
ANSWER = "The Moon pulls."    # every quality text's continuation


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
                "quality": [{"id": i, "set": s, "prompt_ids": ids(m), "messages": m,
                             "continuation_ids": prepare.token_ids(Text(), ANSWER)} for i, s, m in (
                    ("tides", "conversations", QUALITY), ("tides@3000", "long-3000", LONG_QUALITY),
                    ("tides@5000", "long-5000", TOO_LONG))],
                "retrieval": {**frame(RETRIEVAL), "filler_ids": prepare.token_ids(Text(), "Filler text. "),
                              "curve_tail_ids": frame(CURVE)["tail_ids"], "curve_messages": prepare.api_messages(CURVE),
                              "facts": [{"id": "start", "answer": "tides", "ids": prepare.token_ids(Text(), "\n\ntides\n\n")}],
                              "token_text": [chr(i) if chr(i).isascii() else None for i in range(256)]},
                "followup": [{"turn": t, "prompt_ids": ids(m), "messages": m} for t, m in enumerate(FOLLOWUP, 1)]}
    (directory / "prepared.json").write_text(json.dumps(prepared))


def write_store(directory, shift=0.0):
    """Reference scores of every prepared quality text: each token at -0.1 - shift with the letters after it, as the
    fake server scores and serves them (shift 0)."""
    prepared = json.loads((directory / "prepared.json").read_text())
    texts = fixed.texts(prepared)
    scores = {}
    for text_id, (tokens, _) in texts.items():
        rows = [[t] + [ord(o) for o in OTHERS if ord(o) != t][:15] for t in tokens[1:]]
        scores[text_id] = ([-0.1 - shift] * len(rows), rows, [[-0.1 - shift] + [-3.0 - i for i in range(15)] for _ in rows])
    fixed.write(directory, {"model_sha256": prepared["model"]["sha256"], "reference": reference.REFERENCE_VERSION,
                            **fixed.identity(texts, 16, prepared["valid_vocab"])}, scores)


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
        cards = [{"id": m, **self.server.card} for m in self.server.models]
        self.wfile.write(json.dumps({"object": "list", "data": cards}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        if self.path == "/v1/crit/score":
            return self.score(body)
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
        # A plain request (crit's decode extension) decodes one token per target pass.
        telemetry = {"tokens_per_round": 1.0} if (body.get("crit") or {}).get("decode") == "plain" else self.server.telemetry
        events = ([{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}]
                  + [{"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
                  + [{"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}]
                  + ([{"choices": [], "usage": usage, **({"crit_telemetry": telemetry} if telemetry else {})}]
                     if body["stream_options"]["include_usage"] and self.server.streams_usage else []))
        if self.server.stream_error:
            events = events[:1] + [{"error": {"message": self.server.stream_error, "type": "server_error"}}]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


    def score(self, body):
        """chat-score-v1: every token at -0.1 with the letters after it, as the quality replies report them; with
        server.score_offset, the scores move off what the server serves."""
        if not self.server.scores:
            return self.send_error(404)

        def row(token):
            others = [[ord(o), -3.0 - i] for i, o in enumerate(o for o in OTHERS if ord(o) != token)]
            return [token, -0.1 + self.server.score_offset, ([[token, -0.1 + self.server.score_offset]] + others)
                    [:body["top_logprobs"]]]
        self.server.scored.append(body)
        if "format" in body and self.server.packed is False:
            # An earlier crit serve refuses the field it does not know.
            data = json.dumps({"error": {"message": "Unknown score parameter format", "param": "format"}}).encode()
            self.send_response(400)
            self.end_headers()
            return self.wfile.write(data)
        prompt = body["prompt_ids"][1:] if body.get("prompt_rows", True) else []
        value = {"object": "crit.score", "prompt": [row(t) for t in prompt],
                 "continuation": [row(t) for t in body["continuation_ids"]]}
        if body.get("format") == "packed" and self.server.packed:
            def packed(rows):   # crit serve's packed format (crit_api.server.packed_rows)
                encoded = lambda kind, values: base64.b64encode(array.array(kind, values).tobytes()).decode()  # noqa: E731
                return {"count": len(rows), "alternatives": body["top_logprobs"],
                        "tokens": encoded("i", [r[0] for r in rows]), "logprobs": encoded("f", [r[1] for r in rows]),
                        "top_tokens": encoded("i", [a[0] for r in rows for a in r[2]]),
                        "top_logprobs": encoded("f", [a[1] for r in rows for a in r[2]])}
            value = {"object": "crit.score", "format": "packed", "prompt": packed(value["prompt"]),
                     "continuation": packed(value["continuation"])}
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

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
        self.server.one_piece, self.server.stream_error, self.server.telemetry = False, None, None
        self.server.scores, self.server.score_offset, self.server.scored = True, 0.0, []
        self.server.card = {}  # what crit serve adds to a model's card (supported_sampling_parameters, engine_facts)
        self.server.packed = None  # True: scores in the packed format; False: an earlier crit serve refusing it
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
        # Greedy requests name the penalties neutral, so no served default changes them.
        self.assertEqual({(r["presence_penalty"], r["frequency_penalty"], r["repetition_penalty"]) for r in sent},
                         {(0.0, 0.0, 1.0)})
        agent = next(r for r in sent if r.get("tools"))
        self.assertEqual(agent["tools"], TOOLS)
        call, answer = agent["messages"][1]["tool_calls"][0], agent["messages"][2]
        self.assertEqual((call["function"]["arguments"], answer["tool_call_id"]), ('{"path": "a.py"}', call["id"]))

    def test_tokens_per_round_come_from_crit_serves_telemetry(self):
        self.server.telemetry = {"tokens_per_round": 3.0}
        result = runner.run(None, "unused-model", self.data, self.data / "result.json", parts=("speed",),
                            warmups=0, repetitions=1, log=lambda line: None, context_tokens=4096, url=self.url)
        sample = result["speed"]["cases"]["chat"]["samples"][0]
        self.assertEqual((sample["rounds"], sample["tokens_per_round"]), (4, 3.0))

    def test_sampled_requests_carry_the_models_sampling_and_a_seed_per_repetition(self):
        path = self.data / "prepared.json"
        prepared = json.loads(path.read_text())
        prepared["sampling"] = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
        path.write_text(json.dumps(prepared))
        result = self.run_eval(("speed",))
        self.assertEqual(result["sampled_speed"]["status"], "measured")
        self.assertIsNotNone(result["sampled_speed"]["cases"]["chat"]["decode_tps"])
        sampled = [r for r in self.server.requests if r["temperature"] and not r.get("logprobs")]
        self.assertEqual({(r["temperature"], r["top_p"], r["top_k"]) for r in sampled}, {(1.0, 0.95, 20)})
        self.assertEqual({r["seed"] for r in sampled}, {1, 2})   # --repetitions 2: the first two seeds
        # A server that lists no fields gets every one, the filters the model leaves unset and the penalties neutral.
        self.assertEqual({(r["min_p"], r["presence_penalty"], r["frequency_penalty"], r["repetition_penalty"])
                          for r in sampled}, {(0.0, 0.0, 0.0, 1.0)})
        # This fake server answers log-probability requests only unstreamed: the check is unsupported, not a crash.
        self.assertEqual(result["sampling_check"]["status"], "unsupported")

    def test_sampled_requests_send_the_fields_crit_serve_lists_and_need_the_models_own(self):
        path = self.data / "prepared.json"
        prepared = json.loads(path.read_text())
        prepared["sampling"] = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
        path.write_text(json.dumps(prepared))
        # A package applying the model's fields but no penalties: neutral penalties are left out.
        self.server.card = {"supported_sampling_parameters": ["seed", "temperature", "top_k", "top_p"]}
        result = self.run_eval(("speed",))
        self.assertEqual(result["sampled_speed"]["status"], "measured")
        sampled = [r for r in self.server.requests if r["temperature"]]
        self.assertTrue(sampled)
        self.assertEqual({tuple(sorted(set(r) & set(prepare.NEUTRAL_SAMPLING))) for r in sampled}, {("top_k", "top_p")})
        # One that cannot apply the model's top_k has no sampled decode, and Eval sends it no sampled request.
        self.server.requests.clear()
        self.server.card = {"supported_sampling_parameters": ["seed", "temperature", "top_p"]}
        result = self.run_eval(("speed",))
        self.assertEqual(result["sampled_speed"], {"status": "unsupported",
                                                   "reason": "the engine does not apply the sampling's top_k"})
        self.assertFalse([r for r in self.server.requests if r["temperature"]])

    def test_a_server_on_another_template_fails_at_its_first_request_when_quality_runs(self):
        # The fake server counts 1000 prompt tokens more than the prepared ids: quality's honesty check would fail.
        with self.assertRaisesRegex(RuntimeError, "rendered the prompt as 100[0-9] tokens where the model's chat template"):
            self.run_eval(("history", "speed", "quality"))
        self.assertEqual(len(self.server.requests), 1)   # nothing after the first request
        self.assertFalse([r for r in self.server.requests if "prompt_ids" in r])
        # Without quality a server's own template is measured as it renders.
        self.assertEqual(self.run_eval(("history",))["history"]["passed"], True)

    def test_a_served_engines_facts_give_exactness_and_its_pass_floor(self):
        self.server.card = {"capabilities": ["chat-plain-v1"],
                            "engine_facts": {"speculative": True, "plain": True, "draft_width": 3, "bytes_per_pass": 2e9}}
        self.server.telemetry = {"tokens_per_round": 3.0, "accepted": [2, 2, 2, 2]}
        result = runner.run(None, "unused-model", self.data, self.data / "result.json", parts=("speed", "exactness"),
                            warmups=0, repetitions=1, log=lambda line: None, context_tokens=4096, url=self.url,
                            exactness="required", bandwidth_gbs=1000.0)
        self.assertEqual((result["engine"]["speculative"], result["engine"]["draft_width"]), (True, 3))
        exact = result["exactness"]
        self.assertEqual((exact["engine"], exact["identical"]), ("shared", True))
        plain = [r for r in self.server.requests if r.get("crit")]
        self.assertTrue(plain and all(r["crit"] == {"decode": "plain"} for r in plain))
        sample = result["speed"]["cases"]["chat"]["samples"][0]
        self.assertEqual((sample["accepted"], sample["rounds"]), ([2, 2, 2, 2], 4))
        room = result["headroom"]
        self.assertEqual((room["bytes_per_pass_source"], room["pass_floor_ms"]), ("engine", 2.0))
        gates = {g["name"]: g["state"] for g in result["status"]["gates"]}
        self.assertEqual(gates["exactness"], "pass")

    def test_greedy_requests_send_the_neutral_penalties_the_server_applies(self):
        self.server.card = {"supported_sampling_parameters": ["presence_penalty", "seed", "temperature"]}
        self.run_eval(("speed",))
        greedy = [r for r in self.server.requests if r["temperature"] == 0]
        self.assertTrue(greedy)
        # The penalty it applies goes neutral; the ones it lists no support for stay out, so it refuses nothing.
        self.assertEqual({(r["presence_penalty"], "frequency_penalty" in r, "repetition_penalty" in r) for r in greedy},
                         {(0.0, False, False)})

    def test_a_server_that_drafts_without_engine_facts_counts_as_speculative(self):
        self.server.telemetry = {"tokens_per_round": 3.0}   # more than one token per target pass, and no facts
        logged = []
        result = runner.run(None, "unused-model", self.data, self.data / "result.json", parts=("speed", "exactness"),
                            warmups=0, repetitions=1, log=logged.append, context_tokens=4096, url=self.url,
                            exactness="required")
        self.assertEqual((result["engine"]["speculative"], result["engine"]["speculative_source"]), (True, "telemetry"))
        self.assertEqual(result["exactness"]["status"], "unmeasured")
        self.assertIn("engine: speculative (its telemetry reports more than one token per target pass)", logged)
        gates = {g["name"]: g["state"] for g in result["status"]["gates"]}
        self.assertEqual(gates["exactness"], "unmeasured")
        self.server.telemetry = {"tokens_per_round": 1.0}   # one token per pass: a plain engine
        plain = runner.run(None, "unused-model", self.data, self.data / "result.json", parts=("speed", "exactness"),
                           warmups=0, repetitions=1, log=lambda line: None, context_tokens=4096, url=self.url)
        self.assertFalse(plain["engine"]["speculative"])
        self.assertNotIn("exactness", plain)
        # What crit serve reports for a plain engine: its n output tokens over n - 1 rounds (the first token comes
        # from the prefill), n / (n - 1) tokens per round, above 1 without any draft.
        self.server.telemetry = {"tokens_per_round": 12 / 11}
        served = runner.run(None, "unused-model", self.data, self.data / "result.json", parts=("speed", "exactness"),
                            warmups=0, repetitions=1, log=lambda line: None, context_tokens=4096, url=self.url,
                            exactness="required")
        sample = served["speed"]["cases"]["chat"]["samples"][0]
        self.assertEqual((sample["output_tokens"], sample["rounds"]), (12, 11))
        self.assertFalse(served["engine"]["speculative"])
        self.assertNotIn("exactness", served)
        self.assertNotIn("exactness", {g["name"] for g in served["status"]["gates"]})

    def test_a_served_speculative_engine_without_plain_decoding_has_its_exactness_unmeasured(self):
        self.server.card = {"engine_facts": {"speculative": True, "plain": True}}   # the package declares no plain
        result = runner.run(None, "unused-model", self.data, self.data / "result.json", parts=("speed", "exactness"),
                            warmups=0, repetitions=1, log=lambda line: None, context_tokens=4096, url=self.url,
                            exactness="required")
        self.assertEqual(result["exactness"]["status"], "unmeasured")
        self.assertFalse([r for r in self.server.requests if r.get("crit")])
        gate = next(g for g in result["status"]["gates"] if g["name"] == "exactness")
        self.assertEqual(gate, {"name": "exactness", "state": "unmeasured",
                                "detail": "the served package declares no plain decoding (chat-plain-v1)"})

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
        # The curve's point asks for the facts and a summary of the archive that holds them, and the retrieval
        # verdict comes from that one request.
        (curve,) = [r for r in self.server.requests if r["messages"][-1]["content"].endswith("then summarize.")]
        self.assertIn("\n\ntides\n\n", curve["messages"][-1]["content"])
        self.assertTrue(curve["messages"][-1]["content"].startswith("Filler text. Filler"))
        self.assertEqual(len(ids(curve["messages"])), 4096 - 256)

    def test_follow_up_turns_may_be_served_from_a_prompt_cache(self):
        self.server.usage = {"prompt_tokens_details": {"cached_tokens": 900}}
        result = self.run_eval(("followup",))
        followup = result["followup"]
        self.assertEqual([t["cached_tokens"] for t in followup["turns"]], [900, 900])
        self.assertEqual(self.server.requests[0]["messages"], CHAT)
        self.assertTrue(followup["again"]["passed"])
        gate = next(g for g in result["status"]["gates"] if g["name"] == "prompt_cache")
        self.assertEqual(gate["state"], "pass")

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

    def store(self, shift=0.0):
        write_store(self.data, shift)

    def test_quality_scores_the_fixed_texts_through_the_servers_score_call(self):
        self.store()
        result = self.run_eval(("quality",))
        quality = result["quality"]
        self.assertEqual((quality["status"], quality["top_k"], quality["continuation_tokens"]), ("scored", 16, 256))
        self.assertEqual(list(quality["sets"]), ["conversations/prompt", "conversations/served", "long-3000/prompt",
                                                 "long-3000/served"])
        self.assertEqual(quality["skipped"], {"long-5000": "needs 4,848 tokens of context"})
        # Fixed texts are scored at every position; the honesty rescore asks for the reply's rows alone.
        self.assertEqual([body.get("prompt_rows", True) for body in self.server.scored], [True, True, False])
        timing = quality["timing"]
        self.assertEqual((timing["scores"]["requests"], timing["honesty"]["replies"]["requests"],
                          timing["honesty"]["scores"]["requests"]), (2, 1, 1))
        self.assertGreater(timing["scores"]["response_bytes"], 0)
        self.assertTrue(quality["passed"] and all(s["kld_mean"] == 0.0 for s in quality["sets"].values()))
        scored = [r for r in self.server.requests if "prompt_ids" in r]
        self.assertEqual((scored[0]["prompt_ids"], scored[0]["continuation_ids"], scored[0]["top_logprobs"]),
                         (self.prompt("tides"), prepare.token_ids(Text(), ANSWER), 16))
        # The honesty check: the conversation's served reply, then the same tokens scored as the continuation.
        replied = next(r for r in self.server.requests if r.get("logprobs"))
        self.assertEqual((replied["messages"], replied["top_logprobs"], replied["temperature"]), (QUALITY, 16, 0))
        self.assertEqual(scored[-1]["continuation_ids"], prepare.token_ids(Text(), REPLY))
        honesty = quality["honesty"]
        self.assertEqual((honesty["passed"], honesty["max_abs_logprob_diff"], [r["id"] for r in honesty["replies"]]),
                         (True, 0.0, ["tides"]))
        gate = next(g for g in result["status"]["gates"] if g["name"] == "quality")
        self.assertEqual(gate["state"], "pass")
        self.assertIn("served replies scored alike", gate["detail"])

    def test_quality_reads_packed_scores_and_falls_back_to_json_rows(self):
        self.store()
        self.server.packed = True
        packed = self.run_eval(("quality",))
        self.assertTrue(all(r.get("format") == "packed" for r in self.server.requests if "prompt_ids" in r))
        self.server.packed, self.server.requests[:] = False, []   # an earlier crit serve refuses the field
        plain = self.run_eval(("quality",))
        for name, found in packed["quality"]["sets"].items():   # float32 log-probabilities, the same scores
            self.assertAlmostEqual(found["ppl_ratio"], plain["quality"]["sets"][name]["ppl_ratio"], places=5)
            self.assertAlmostEqual(found["kld_mean"], plain["quality"]["sets"][name]["kld_mean"], places=5)
        formats = [r.get("format") for r in self.server.requests if "prompt_ids" in r]
        self.assertEqual(formats[:2], ["packed", None])   # asked once, then JSON rows
        self.assertTrue(all(f is None for f in formats[1:]))

    def test_quality_needs_the_reference_scores_and_a_score_call(self):
        result = self.run_eval(("quality",))
        self.assertIn("crit eval reference --model --data", result["quality"]["reason"])
        self.store(shift=0.01)   # scored by another reference: the store's numbers, not its identity, are off
        self.server.scores = False
        engine, _ = self.engine()
        with self.assertRaisesRegex(RuntimeError, "scores no fixed text"):
            engine.score(self.prompt("tides"), [1], 16)

    def test_a_score_call_off_the_served_replies_fails_the_honesty_check(self):
        self.store()
        self.server.score_offset = -0.02
        result = self.run_eval(("quality",))
        honesty = result["quality"]["honesty"]
        self.assertFalse(honesty["passed"])
        self.assertAlmostEqual(honesty["max_abs_logprob_diff"], 0.02)
        gate = next(g for g in result["status"]["gates"] if g["name"] == "quality")
        self.assertEqual(gate["state"], "fail")

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

    def test_the_reference_stores_the_fixed_texts_scores_once(self):
        # crit eval reference scores every prepared text on the prepared checkpoint and only that one, once.
        texts = fixed.texts(json.loads((self.data / "prepared.json").read_text()))

        class Reference:
            def __init__(self, model):
                self.model = model

            def identity(self, model_sha):
                return {"model_sha256": model_sha, "reference": reference.REFERENCE_VERSION}

            def score_texts(self, found, top_k, valid_vocab, scratch, log, long, check):
                # Rows as the real reference returns them, tensors store_texts reads with tolist(), without torch.
                class Rows(list):
                    def tolist(self):
                        return list(self)
                self.long = long
                return [(Rows([-0.1] * (len(ids) - 1)), Rows([[0] * top_k] * (len(ids) - 1)),
                         Rows([[-0.1] * top_k] * (len(ids) - 1))) for ids in found]
        with patch.object(reference, "StreamingReference", Reference), \
                patch.object(reference, "model_identity", lambda model: {"sha256": "model" if model == "source" else "x"}):
            with self.assertRaisesRegex(SystemExit, "not the checkpoint"):
                reference.score_data("other", self.data, log=lambda m: None)
            how = reference.score_data("source", self.data, log=lambda m: None)
            self.assertEqual((how["method"], how["positions"]), ("scored", sum(len(i) - 1 for i, _ in texts.values())))
            self.assertEqual(reference.score_data("source", self.data, log=lambda m: None)["method"], "stored")
        prepared = json.loads((self.data / "prepared.json").read_text())
        stored = fixed.load(self.data, {"model_sha256": "model", **fixed.identity(texts, 16, prepared["valid_vocab"])})
        self.assertEqual(set(stored), {"tides", "tides@3000", "tides@5000"})


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
