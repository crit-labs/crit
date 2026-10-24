"""Eval's request bound (crit_eval.deadline): no fixed time budget, each request bounded by what the engine served."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval import chat, runner  # noqa: E402
from crit_eval.deadline import Hung, RequestDeadline  # noqa: E402

RULE = {"floor_seconds": 2.0, "slack": 4.0}


class DeadlineTests(unittest.TestCase):
    def test_before_any_request_only_the_floor_bounds_one(self):
        self.assertEqual(RequestDeadline(RULE).seconds(262_144, 256), 2.0)

    def test_prefill_scales_at_most_quadratically_from_the_tightest_observation(self):
        deadline = RequestDeadline(RULE)
        deadline.observe(1000, 1.0)
        deadline.observe(8000, 4.0)    # faster per token than the first: the tighter bound for longer prompts
        self.assertEqual(deadline.expected(500), 1.0)
        self.assertEqual(deadline.expected(16_000), min(1.0 * 16 ** 2, 4.0 * 2 ** 2))
        self.assertEqual(deadline.seconds(16_000), 2.0 + 4.0 * 16.0)

    def test_decode_steps_scale_at_most_linearly_with_the_context(self):
        deadline = RequestDeadline(RULE)
        deadline.observe(900, 0.5, 101, 1.0)   # 100 steps in 1 s at a context of 1,001
        self.assertAlmostEqual(deadline.expected(900, 101), 0.5 + 100 * 0.01)
        # Ten times the context: each step may take ten times as long.
        self.assertAlmostEqual(deadline.expected(10_009, 2), 0.5 * (10_009 / 900) ** 2 + 0.01 * 10_011 / 1001)


class Silent(BaseHTTPRequestHandler):
    """A server that accepts a request and never answers it."""
    release = threading.Event()

    def do_POST(self):
        self.release.wait(30)

    def log_message(self, *_):
        pass


class Pinging(BaseHTTPRequestHandler):
    """A stream of server.events events .1 s apart with keep-alive comments between them, as crit serve sends while
    the engine is quiet, then server.stalled seconds of comments alone."""

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for _ in range(self.server.events):
                self.wfile.write(b'data: {"choices": []}\n\n')
                for _ in range(4):
                    time.sleep(.025)
                    self.wfile.write(b": ping\n\n")
            stalled = time.monotonic() + self.server.stalled
            while time.monotonic() < stalled:
                time.sleep(.025)
                self.wfile.write(b": ping\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        except OSError:
            pass  # the client gave up

    def log_message(self, *_):
        pass


class HungServerTests(unittest.TestCase):
    def test_a_request_past_its_bound_raises_hung(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Silent)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/v1"
            client = chat.Server(url, {"speed": []})
            with self.assertRaisesRegex(Hung, "/crit/score gave no answer within 0 s"):
                client.post("/crit/score", {}, 0.2)
        finally:
            Silent.release.set()
            server.shutdown()
            server.server_close()

    def test_keep_alive_comments_are_no_answer_and_each_event_has_the_whole_bound(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Pinging)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client = chat.Server(f"http://127.0.0.1:{server.server_address[1]}/v1", {"speed": []})
        server.events, server.stalled = 8, 0
        self.assertEqual(len(list(client.events("/chat/completions", {}, 0.5))), 8)  # .8 s in all
        server.events, server.stalled = 1, 5
        found = []
        with self.assertRaisesRegex(Hung, "/chat/completions gave no answer within 0 s"):
            found.extend(client.events("/chat/completions", {}, 0.5))
        self.assertEqual(found, [{"choices": []}])

    def test_a_hung_request_stops_the_context_checks_instead_of_failing_a_point(self):
        class Engine:
            def generate(self, prompt, max_new_tokens, on_tokens):
                raise Hung("/chat/completions gave no answer within 9 s")
        prepared = {"retrieval": {"head_ids": [1], "tail_ids": [3], "curve_tail_ids": [4], "filler_ids": [2] * 50,
                                  "facts": []}}
        settings = {"output_tokens": 4, "curve_from": 64, "repetitions": 1, "retrieval": {"positions": []}}
        with self.assertRaises(Hung):
            runner.run_context(Engine(), {"max_context": 64}, prepared, settings, 64, lambda m: None)


if __name__ == "__main__":
    unittest.main()
