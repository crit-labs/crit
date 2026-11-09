"""crit smoke against a toy engine behind a fake crit serve: a working engine passes every test, and each planted
bug fails the test written for it (offline, CPU)."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "smoke"))

from crit_smoke import Client, run  # noqa: E402
from crit_smoke.texts import PROMPTS, PROSE, WORDS  # noqa: E402

VOCAB = 600
BOS = 1


class Toy:
    """A word-level model that knows PROSE's word pairs and copies what followed the last three words where they
    appeared earlier in the context.
    `bug` plants one fault: leak (prefill rows see later tokens), offset (decode reads the position before),
    stale (a request's reply depends on the request before), long (positions past 1500 forget the context), spec
    (the default decode takes the second choice every fifth token), flat (logits scaled to near uniform), leak_long
    (prefill rows past position 2000 see later tokens, as a sparse mask past a threshold might).
    Two plant a change of the size a W4A4 engine's rounding leaves, 0.2 to 0.3 nats a row with the same top-1:
    decode_rounding (decode rows against prefill, as GEMV against GEMM) and length_rounding (the rows of prompts
    under 129 tokens, as a GEMM whose kernel cuBLAS picks by row count)."""

    def __init__(self, bug=None):
        self.bug, self.words, self.last = bug, {}, 0
        for word in PROSE.split() + WORDS + " ".join(PROMPTS + ("user: assistant:",)).split():
            self.id(word)
        prose = self.encode(PROSE)
        self.known = {(a, b): c for a, b, c in zip(prose, prose[1:], prose[2:])}
        reply = self.id("assistant:")
        for prompt in PROMPTS:   # replies recite the prose, so their choices are clear
            self.known[(self.id(prompt.split()[-1]), reply)] = prose[0]
        self.known[(reply, prose[0])] = prose[1]

    def id(self, word):
        if word not in self.words:
            self.words[word] = 2 + len(self.words) if len(self.words) < VOCAB - 2 else 2 + hash(word) % (VOCAB - 2)
        return self.words[word]

    def encode(self, text, special=False):
        return ([BOS] if special else []) + [self.id(w) for w in text.split()]

    def logits(self, ids, p, seen, decode, extra=None):
        last = ids[p]
        values = [((t * 2654435761 + last * 97) % 1000) / 2000 for t in range(VOCAB)]
        q = p - 1 if decode and self.bug == "offset" and p >= 3 else p
        key = tuple(ids[max(0, q - 2):q + 1])
        if not (self.bug == "long" and p > 1500):
            if key[-2:] in self.known:
                values[self.known[key[-2:]]] += 9
            if key in seen:
                values[seen[key]] += 9
        if extra:
            values[extra % VOCAB] += 12
        if self.bug == "flat":
            values = [v * 0.05 for v in values]
        if self.bug == "decode_rounding" and decode or self.bug == "length_rounding" and len(ids) < 129:
            values = [v * (0.9 if decode else 0.8) for v in values]
        return values

    def row(self, values, token, top):
        peak = max(values)
        total = peak + math.log(sum(math.exp(v - peak) for v in values))
        ranked = sorted(range(VOCAB), key=lambda t: -values[t])[:top]
        return [token, values[token] - total, [[t, values[t] - total] for t in ranked]]

    def walk(self, ids, top, decode_from, start=0, extra=None):
        """Rows for positions start..len(ids)-2 (row p scores ids[p+1]); positions from decode_from decode."""
        seen, rows = {}, []
        for p in range(len(ids) - 1):
            if p >= start:
                values = self.logits(ids, p, seen, p >= decode_from, extra)
                if (self.bug == "leak" or self.bug == "leak_long" and p > 2000) and p < decode_from:
                    for later in ids[p + 2:p + 6]:
                        values[later] += 4
                rows.append(self.row(values, ids[p + 1], top))
            if p >= 2:
                seen[tuple(ids[p - 2:p + 1])] = ids[p + 1]
        return rows

    def score(self, prompt, continuation, top, prompt_rows):
        ids = prompt + continuation
        rows = self.walk(ids, top, len(prompt), 0 if prompt_rows else len(prompt) - 1)
        if not prompt_rows:
            return [], rows
        return rows[:len(prompt) - 1], rows[len(prompt) - 1:]

    def generate(self, prompt, count, top, plain):
        extra = self.last if self.bug == "stale" else None
        self.last = len(prompt) + 7
        ids, rows = list(prompt), []
        for step in range(count):
            values = self.walk(ids + [0], top, len(prompt), len(ids) - 1, extra)[0][2]
            token = values[0][0]
            if self.bug == "spec" and not plain and step % 5 == 4:
                token = values[1][0]
            rows.append([token, dict((t, p) for t, p in values)[token], values])
            ids.append(token)
        return rows


def serve(toy, capabilities, facts=None, concurrency=1):
    names = {v: k for k, v in toy.words.items()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.answer({"data": [{"id": "toy", "default": True, "loaded": True, "context_length": 8192,
                                   "capabilities": capabilities, "engine_facts": facts or {},
                                   "max_concurrent_requests": concurrency}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                if "messages" in body:
                    tokens = toy.encode("user: " + body["messages"][-1]["content"] + " assistant:", True)
                else:
                    tokens = toy.encode(body["prompt"], body.get("add_special_tokens", True))
                return self.answer({"tokens": tokens, "count": len(tokens)})
            if self.path == "/v1/crit/score":
                prompt, continuation = toy.score(body["prompt_ids"], body["continuation_ids"], body["top_logprobs"],
                                                 body.get("prompt_rows", True))
                return self.answer({"prompt": prompt, "continuation": continuation})
            if self.path == "/v1/chat/completions":
                prompt = toy.encode("user: " + body["messages"][-1]["content"] + " assistant:", True)
                rows = toy.generate(prompt, body["max_tokens"], body["top_logprobs"],
                                    (body.get("crit") or {}).get("decode") == "plain")
                content = [{"crit_token_id": t, "logprob": p,
                            "top_logprobs": [{"crit_token_id": a, "logprob": q} for a, q in alts]}
                           for t, p, alts in rows]
                text = " ".join(names.get(t, "?") for t, _, _ in rows)
                return self.answer({"choices": [{"message": {"content": text}, "logprobs": {"content": content}}]})
            self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


ALL = ["chat-score-v1", "chat-logprobs-v1", "chat-tokenize-v1", "chat-plain-v1"]


class SmokeTests(unittest.TestCase):
    def results(self, bug=None, facts=None, capabilities=ALL, only=None, budget=300, profile="loose"):
        server = serve(Toy(bug), capabilities, facts)
        try:
            # 2600 tokens reach past the toy's thresholds while keeping its pure-Python passes short.
            items = run(Client(f"http://127.0.0.1:{server.server_port}/v1"), only, profile=profile, long=2600,
                        log=lambda line: None, budget=budget)
        finally:
            server.shutdown()
            server.server_close()
        return {i["name"]: i for i in items}

    def test_a_working_engine_passes(self):
        results = self.results(facts={"speculative": True})
        for name, item in results.items():
            if name == "concurrent":
                self.assertEqual(item["state"], "skipped", item)
            else:
                self.assertEqual(item["state"], "passed", item)

    def test_each_planted_bug_fails_its_test(self):
        for bug, test in (("leak", "causal"), ("leak", "decode_matches_prefill"), ("offset", "decode_matches_prefill"),
                          ("offset", "recall"), ("stale", "isolated"), ("long", "recall"), ("spec", "plain_equal"),
                          ("spec", "greedy_is_top"), ("flat", "fluent"), ("leak_long", "causal_long"),
                          ("leak_long", "decode_long")):
            with self.subTest(bug=bug, test=test):
                item = self.results(bug, facts={"speculative": True}, only={test})[test]
                self.assertEqual(item["state"], "failed", item)

    def test_a_fault_past_a_threshold_passes_the_short_tests(self):
        results = self.results("leak_long", only={"causal", "decode_matches_prefill"})
        self.assertEqual({i["state"] for i in results.values()}, {"passed"}, results)

    def test_decode_rounding_passes_loose_and_prompt_rows_that_move_with_length_fail(self):
        decode = {"decode_matches_prefill", "decode_long"}
        rounded = self.results("decode_rounding", only=decode | {"causal", "causal_long"})
        self.assertEqual({i["state"] for i in rounded.values()}, {"passed"}, rounded)
        self.assertIn("after 16: 48 rows, mean delta 0.16", rounded["decode_matches_prefill"]["detail"])
        strict = self.results("decode_rounding", only={"decode_matches_prefill"}, profile="strict")
        self.assertEqual(strict["decode_matches_prefill"]["state"], "failed", strict)
        # Rounding of the same size in prefill fails causal, since there it depends on the prompt's length.
        moved = self.results("length_rounding", only={"causal"})["causal"]
        self.assertEqual(moved["state"], "failed", moved)
        self.assertIn("prefix of 17 tokens", moved["detail"])

    def test_a_suite_past_its_budget_stops_as_failed(self):
        results = self.results(only={"rows_valid", "fluent"}, budget=0)
        self.assertEqual(list(results), ["budget"])
        self.assertEqual(results["budget"]["state"], "failed")
        self.assertIn("rows_valid, fluent", results["budget"]["detail"])

    def test_missing_capabilities_skip(self):
        results = self.results(capabilities=["chat-logprobs-v1"])
        self.assertEqual(results["causal"]["state"], "skipped")
        self.assertIn("chat-score-v1", results["causal"]["detail"])
        self.assertEqual(results["repeatable"]["state"], "passed")


if __name__ == "__main__":
    unittest.main()
