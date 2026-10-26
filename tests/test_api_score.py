"""Fixed-text scoring through crit serve (chat-score-v1) with a synthetic resident adapter."""
import array
import base64
import http.client
import json
import math
import threading
import unittest
from types import SimpleNamespace

from test_chat_residency import ServeFixture
from crit_api.engine import PackedScores, score_reply
from crit_api.server import Api, Server
from crit_core.serving import ResidentChat, ScoreRows, capabilities, logprob_row, packed_scores

# Every row ranks the scored token first at -0.25, then token 1 at -2.0; a prompt starting with 999 makes the engine
# name another token than the one it scores.
SCORE_ADAPTER = r'''
Engine.scores = True

def _score(self, request, cancellation):
    def row(token):
        top = [[token, -0.25], [1 if token != 1 else 2, -2.0], [3 if token != 3 else 4, -3.0]]
        return [token, -0.25, top[:request['top_logprobs']]]
    prompt, continuation = request['prompt_ids'], request['continuation_ids']
    rows = [row(t) for t in prompt[1:]]
    if prompt[0] == 999:
        rows[0][0] += 1
    return {'schema_version': 1, 'record_kind': 'score_response', 'request_id': request['request_id'],
            'candidate_id': request['candidate_id'], 'package_sha256': request['package_sha256'],
            'prompt': rows, 'continuation': [row(t) for t in continuation],
            'telemetry': {'prefill_ms': 1.5, 'decode_ms': 2.0}}

Engine.score = _score
'''

# The same rows over the host's channel in the packed format crit serve's request offers (as ResidentChat.score's).
PACKED_SCORE_ADAPTER = SCORE_ADAPTER + r'''
from crit_core.serving import packed_scores

def _packed_score(self, request, cancellation):
    record = _score(self, request, cancellation)
    assert request['format'] == 'packed'   # crit serve offers it
    def part(rows):
        return packed_scores([row[0] for row in rows], [(row[1], row[2]) for row in rows], request['top_logprobs'])
    return dict(record, format='packed', prompt=part(record['prompt']), continuation=part(record['continuation']))

Engine.score = _packed_score
'''


class ScoreApiTests(ServeFixture, unittest.TestCase):
    extra_capabilities = ("chat-score-v1",)
    adapter_suffix = SCORE_ADAPTER

    def setUp(self):
        super().setUp()
        self.server = Server(("127.0.0.1", 0), Api(self.engine()))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def post(self, body):
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=30)
        self.addCleanup(connection.close)
        connection.request("POST", "/v1/crit/score", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response, json.loads(response.read())

    def test_prompt_and_continuation_rows_come_back_with_their_alternatives(self):
        response, value = self.post({"model": "default", "prompt_ids": [5, 6, 7], "continuation_ids": [8, 9],
                                     "top_logprobs": 2})
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value["object"], "crit.score")
        self.assertEqual(value["prompt"], [[6, -0.25, [[6, -0.25], [1, -2.0]]], [7, -0.25, [[7, -0.25], [1, -2.0]]]])
        self.assertEqual([row[0] for row in value["continuation"]], [8, 9])
        self.assertEqual(value["crit_telemetry"], {"prefill_ms": 1.5, "decode_ms": 2.0})

    def test_invalid_score_requests_are_rejected(self):
        for body, param in (({"prompt_ids": []}, "prompt_ids"), ({"prompt_ids": [1, -2]}, "prompt_ids"),
                            ({"prompt_ids": [1], "top_logprobs": 21}, "top_logprobs"),
                            ({"prompt_ids": [1], "continuation_ids": "x"}, "continuation_ids"),
                            ({"prompt_ids": [1], "messages": []}, "messages")):
            with self.subTest(body=body):
                response, value = self.post(body)
                self.assertEqual(response.status, 400, value)
                self.assertEqual(value["error"]["param"], param)

    def test_a_full_queue_refuses_a_score_with_retry_after(self):
        engine = self.server.api.engine
        engine.lock.max_queue = 0
        with engine.lock:  # lifecycle work holds the engine
            response, value = self.post({"prompt_ids": [5, 6], "top_logprobs": 1})
        self.assertEqual((response.status, response.getheader("Retry-After"), value["error"]["code"]),
                         (429, "1", "rate_limit_exceeded"))
        response, value = self.post({"prompt_ids": [5, 6], "top_logprobs": 1})
        self.assertEqual(response.status, 200, value)
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=30)
        self.addCleanup(connection.close)
        connection.request("GET", "/metrics")
        text = connection.getresponse().read().decode()
        for line in ('crit_serve_requests_total{dialect="crit",outcome="rejected"} 1',
                     'crit_serve_requests_total{dialect="crit",outcome="completed"} 1',
                     "crit_serve_requests_in_flight 0", "crit_serve_requests_waiting 0"):
            self.assertIn(line, text)

    def test_a_response_naming_other_tokens_fails_the_request(self):
        response, value = self.post({"prompt_ids": [999, 6, 7], "top_logprobs": 1})
        self.assertEqual(response.status, 500, value)
        self.assertIn("invalid response", value["error"]["message"])
        self.assertEqual(value["request_id"], response.getheader("X-Request-Id"))

    def test_metrics_count_score_requests_by_outcome(self):
        self.post({"prompt_ids": [5, 6, 7], "continuation_ids": [8], "top_logprobs": 1})
        self.post({"prompt_ids": []})
        self.post({"prompt_ids": [999, 6, 7], "top_logprobs": 1})
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=30)
        self.addCleanup(connection.close)
        connection.request("GET", "/metrics")
        text = connection.getresponse().read().decode()
        for line in ('crit_serve_requests_total{dialect="crit",outcome="completed"} 1',
                     'crit_serve_requests_total{dialect="crit",outcome="rejected"} 1',
                     'crit_serve_requests_total{dialect="crit",outcome="error"} 1',
                     "crit_serve_requests_in_flight 0", "crit_serve_prompt_tokens_total 0",
                     "crit_serve_requests_running 0", "crit_serve_requests_waiting 0",
                     # The two scores the engine ran were dispatched to it; the invalid one never began.
                     "crit_serve_request_queue_time_seconds_count 2", "crit_serve_e2e_request_latency_seconds_count 2",
                     "crit_serve_request_prompt_tokens_count 0"):
            self.assertIn(line, text)


class PackedChannelScoreApiTests(ScoreApiTests):
    """The same requests with the engine's rows crossing the host's channel packed."""
    adapter_suffix = PACKED_SCORE_ADAPTER

    def test_packed_rows_reach_both_formats_unchanged(self):
        body = {"model": "default", "prompt_ids": [5, 6, 7], "continuation_ids": [8, 9], "top_logprobs": 3}
        response, value = self.post(body)
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value["continuation"], [[8, -0.25, [[8, -0.25], [1, -2.0], [3, -3.0]]],
                                                 [9, -0.25, [[9, -0.25], [1, -2.0], [3, -3.0]]]])
        response, packed = self.post(dict(body, format="packed"))
        self.assertEqual((response.status, packed["format"], packed["prompt"]["count"]), (200, "packed", 2), packed)
        self.assertEqual(array.array("f", base64.b64decode(packed["continuation"]["top_logprobs"])).tolist(),
                         [-0.25, -2.0, -3.0] * 2)


class ScoreWithoutCapabilityTests(ServeFixture, unittest.TestCase):
    def test_a_package_without_chat_score_is_refused(self):
        engine = self.engine()
        with self.assertRaisesRegex(Exception, "does not declare chat-score-v1"):
            engine.score([1, 2], [3], 1, threading.Event())


class ScoreRecordTests(unittest.TestCase):
    request = {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "prompt_ids": [5, 6],
               "continuation_ids": [7], "top_logprobs": 2, "prompt_rows": True}

    def value(self, **changes):
        return {"schema_version": 1, "record_kind": "score_response", "request_id": "r", "candidate_id": "c",
                "package_sha256": "p", "prompt": [[6, -0.5, [[6, -0.5], [2, -1.0]]]],
                "continuation": [[7, -0.25, [[7, -0.25], [2, -2.0]]]],
                "telemetry": {"prefill_ms": 1, "decode_ms": None}, **changes}

    def test_rows_are_checked_against_the_request(self):
        self.assertEqual(score_reply(self.value(), self.request)["prompt"], [[6, -0.5, [[6, -0.5], [2, -1.0]]]])
        for changes in ({"prompt": []}, {"prompt": [[5, -0.5, [[6, -0.5], [2, -1.0]]]]},
                        {"prompt": [[6, -0.5, [[6, -0.5]]]]}, {"prompt": [[6, -0.5, [[2, -1.0], [6, -0.5]]]]},
                        {"prompt": [[6, 0.5, [[6, -0.5], [2, -1.0]]]]}, {"prompt": [[6, -0.5, [[6, -0.5], [6, -0.5]]]]},
                        {"request_id": "other"}, {"telemetry": {}}, {"extra": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                score_reply(self.value(**changes), self.request)

    def test_without_prompt_rows_only_the_continuation_is_scored(self):
        request = dict(self.request, prompt_rows=False)
        self.assertEqual(score_reply(self.value(prompt=[]), request)["prompt"], [])
        with self.assertRaises(ValueError):
            score_reply(self.value(), request)

    def test_resident_chat_builds_the_record_from_score_tokens(self):
        class Engine(ResidentChat):
            scores = True

            def score_tokens(self, ids, continuation, top_k, cancelled, prompt_rows):
                rows = [(-0.5, [(t, -0.5), (0, -1.0)][:top_k]) for t in ids[1:]] if prompt_rows else []
                # Only the first continuation token: an engine that stops early.
                return rows, [(-0.25, [(t, -0.25), (0, -float("inf"))][:top_k]) for t in continuation[:1]], \
                    {"prefill_ms": 3.0, "decode_ms": 4.0}
        engine = object.__new__(Engine)
        cancellation = SimpleNamespace(is_set=lambda: False, check=lambda: None)
        record = engine.score(self.request, cancellation)
        self.assertEqual(score_reply(record, self.request)["continuation"], [[7, -0.25, [[7, -0.25], [0, -9999.0]]]])
        short = dict(self.request, continuation_ids=[7, 8])
        with self.assertRaisesRegex(ValueError, "not 1 and 2"):
            engine.score(short, cancellation)
        alone = engine.score(dict(self.request, prompt_rows=False), cancellation)
        self.assertEqual((alone["prompt"], len(alone["continuation"])), ([], 1))

    def test_packed_records_hold_the_json_rows(self):
        class Engine(ResidentChat):
            scores = True

            def score_tokens(self, ids, continuation, top_k, cancelled, prompt_rows):
                # Values the clamp changes: rounding above zero, a negative zero, -inf, NaN and one below MIN_LOGPROB.
                prompt = [(1e-7, [(t, -0.0), (0, -1e5)][:top_k]) for t in ids[1:]] if prompt_rows else []
                return prompt, [(math.nan, [(t, -0.25), (0, -math.inf)][:top_k]) for t in continuation], \
                    {"prefill_ms": 3.0, "decode_ms": 4.0}
        engine = object.__new__(Engine)
        cancellation = SimpleNamespace(is_set=lambda: False, check=lambda: None)
        for request in (self.request, dict(self.request, prompt_rows=False), dict(self.request, top_logprobs=0)):
            with self.subTest(request=request):
                plain = score_reply(engine.score(request, cancellation), request)
                packed = engine.score(dict(request, format="packed"), cancellation)
                self.assertEqual(packed["format"], "packed")
                self.assertEqual(json.dumps(score_reply(packed, request)), json.dumps(plain))
                arrays = score_reply(packed, request, packed=True)
                self.assertIsInstance(arrays["continuation"], PackedScores)
                for part in ("prompt", "continuation"):
                    self.assertEqual(arrays[part], PackedScores.of(plain[part], request["top_logprobs"]))
                # JSON rows from an adapter that answers without the packed format come back as arrays too.
                self.assertEqual(score_reply(engine.score(request, cancellation), request, packed=True), arrays)
        self.assertEqual(json.dumps(score_reply(engine.score(self.request, cancellation), self.request)["prompt"]),
                         json.dumps([[6, 0.0, [[6, 0.0], [0, -9999.0]]]]))   # 0.0, not -0.0, as logprob_row clamps

    def test_score_rows_read_as_the_rows_they_hold(self):
        rows = ScoreRows(array.array("f", [-0.5, 2.0]), array.array("i", [6, 0, 7, 1]),
                         array.array("f", [-0.5, -1.0, 0.25, -1e5]), 2)
        listed = [(-0.5, [(6, -0.5), (0, -1.0)]), (2.0, [(7, 0.25), (1, -1e5)])]
        self.assertEqual((len(rows), list(rows), rows[-1]), (2, listed, listed[1]))
        self.assertEqual(packed_scores([6, 7], rows, 2), packed_scores([6, 7], listed, 2))
        self.assertEqual(logprob_row(7, rows[1], 2), [7, 0.0, [[7, 0.0], [1, -9999.0]]])
        with self.assertRaisesRegex(ValueError, "3 alternative"):
            ScoreRows(array.array("f", [-0.5]), array.array("i", [6, 0]), array.array("f", [-0.5, -1.0]), 3)
        with self.assertRaisesRegex(ValueError, "not 1"):
            packed_scores([6, 7], rows, 1)
        with self.assertRaisesRegex(ValueError, "below 2"):
            packed_scores([2 ** 31], [(-0.5, [])], 0)

    def test_packed_records_are_checked_against_the_request(self):
        request = dict(self.request, format="packed")

        def record(prompt=((6, -0.5, [(6, -0.5), (2, -1.0)]),), alternatives=2, **changes):
            part = packed_scores([row[0] for row in prompt], [(row[1], row[2]) for row in prompt], alternatives)
            return self.value(**{"format": "packed", "prompt": part,
                                 "continuation": packed_scores([7], [(-0.25, [(7, -0.25), (2, -2.0)])], 2), **changes})
        self.assertEqual(score_reply(record(), request)["prompt"], [[6, -0.5, [[6, -0.5], [2, -1.0]]]])
        part = record()["prompt"]
        malformed = (dict(part, tokens="!"), dict(part, count=2), dict(part, logprobs=part["tokens"]),
                     {k: v for k, v in part.items() if k != "count"}, dict(part, alternatives=-1))
        for changes in ({"format": "json"}, {"prompt": []}, *({"prompt": bad} for bad in malformed)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                score_reply(record(**changes), request)
        for prompt, alternatives in ((((5, -0.5, [(6, -0.5), (2, -1.0)]),), 2), (((6, -0.5, [(6, -0.5)]),), 1),
                                     (((6, -0.5, [(2, -1.0), (6, -0.5)]),), 2),
                                     (((6, -0.5, [(6, -0.5), (6, -0.75)]),), 2),
                                     (((6, -0.5, [(6, -0.5), (-1, -0.75)]),), 2)):
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, "Score prompt rows"):
                score_reply(record(prompt, alternatives), request)
        # Values outside [MIN_LOGPROB, 0], which packed_scores clamps, so they are written in by hand.
        for bad in (math.nan, math.inf, -math.inf, 0.5, -1e5):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "numbers in"):
                found = record()
                found["prompt"]["logprobs"] = base64.b64encode(array.array("d", [bad]).tobytes()).decode()
                score_reply(found, request)

    def test_the_capability_is_declared_with_score(self):
        text = SimpleNamespace(templates={})
        self.assertIn("chat-score-v1", capabilities(text, score=True))
        self.assertNotIn("chat-score-v1", capabilities(text))


if __name__ == "__main__":
    unittest.main()
