import json
from pathlib import Path
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "crit-eval"))
from crit_eval import outcomes as bench  # noqa: E402


def wire(payload):
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


def good(count=7, reason="stop"):
    return (wire({"choices": [{"index": 0, "delta": {"content": "one chunk, many tokens: é"}, "finish_reason": None}]})
            + wire({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
            + wire({"choices": [], "usage": {"completion_tokens": count}})
            + b"data: [DONE]\n\n")


class BenchmarkOutcomesTests(unittest.TestCase):
    def test_fragmented_utf8_crlf_and_actual_usage(self):
        raw = good(19).replace(b"\n", b"\r\n")
        outcome = bench.validate_completion(200, (raw[i:i+1] for i in range(len(raw))))
        self.assertTrue(outcome.successful, outcome.errors)
        self.assertEqual(outcome.completion_tokens, 19)
        self.assertTrue(bench.validate_completion(200, [good(0, "tool_calls")]).successful)

    def test_http_and_in_band_errors(self):
        for status, chunks in ((503, [good()]), (200, [wire({"error": {"message": "failure"}})]),
                               (200, [b"event: error\ndata: {}\n\n"]),
                               (200, [wire({"type": "error", "choices": []})])):
            self.assertFalse(bench.validate_completion(status, chunks).successful)

    def test_missing_finish_done_usage_and_truncation(self):
        for raw in (good().replace(b"data: [DONE]\n\n", b""), good()[:-1],
                    wire({"choices": [], "usage": {"completion_tokens": 7}}) + b"data: [DONE]\n\n",
                    wire({"choices": [{"index": 0, "finish_reason": "stop"}]}) + b"data: [DONE]\n\n"):
            self.assertFalse(bench.validate_completion(200, [raw]).successful)

    def test_invalid_counts_choices_duplicate_fields_and_order(self):
        for count in (True, -1, 1.5, "7", None, 1 << 64):
            self.assertFalse(bench.validate_completion(200, [good(count)]).successful)
        for raw in (b'data: {"choices":[],"choices":[]}\n\n',
                    wire({"choices": [{"index": False, "finish_reason": "stop"}]}),
                    wire({"choices": [{"index": 0}, {"index": 1}]}),
                    good() + wire({"choices": []}),
                    wire({"choices": [], "usage": {"completion_tokens": 0}}) + good(),
                    good().replace(b"data: [DONE]", wire({"choices": [], "usage": {"completion_tokens": 8}}) + b"data: [DONE]")):
            self.assertFalse(bench.validate_completion(200, [raw]).successful)

    def test_bounds_and_failed_transport(self):
        self.assertFalse(bench.validate_completion(200, [b"data: " + b"x" * 33], max_frame_bytes=32, max_stream_bytes=64).successful)
        self.assertFalse(bench.validate_completion(200, [good()], max_frame_bytes=32, max_stream_bytes=64).successful)
        for error_type in (ConnectionError, ValueError, UnicodeError, RecursionError):
            def interrupted():
                yield wire({"choices": [{"index": 0, "finish_reason": "stop"}]})
                raise error_type("sensitive transport details")
            outcome = bench.validate_completion(200, interrupted())
            self.assertIn(f"Transport failed: {error_type.__name__}", outcome.errors)
            self.assertNotIn("sensitive", repr(outcome))
        class BadIterator:
            def __iter__(self):
                raise ValueError("sensitive iterator details")
        outcome = bench.validate_completion(200, BadIterator())
        self.assertIn("Transport failed: ValueError", outcome.errors)
        self.assertNotIn("sensitive", repr(outcome))

    def test_workload_finish_policy(self):
        self.assertFalse(bench.validate_completion(200, [good(reason="length")], allowed_finishes={"stop"}).successful)
        self.assertFalse(bench.validate_completion(200, [good(reason="error")]).successful)

    def test_unknown_telemetry_is_not_zero(self):
        self.assertIsNone(bench.telemetry_count(None))
        self.assertEqual(bench.telemetry_count("0.0"), 0)
        self.assertEqual(bench.telemetry_count("9007199254740993"), 9007199254740993)
        for raw in ("", "nan", "inf", "-1", "0.5", "unknown", "1e100", "1_0"):
            with self.assertRaises(ValueError):
                bench.telemetry_count(raw)

    def test_fixed_wall_window_actual_counts_and_concurrency(self):
        records = [bench.RequestRecord("a", 2, 7, bench.validate_completion(200, [good(2)])),
                   bench.RequestRecord("b", 4, 8, bench.validate_completion(200, [good(8)]))]
        outcome = bench.validate_cohort(["a", "b"], records, wall_started=0, wall_finished=10, minimum_peak=2)
        self.assertTrue(outcome.qualified, outcome.errors)
        self.assertEqual((outcome.successful_completion_tokens, outcome.completion_tokens_per_second, outcome.peak_concurrency), (10, 1, 2))
        self.assertIsNone(outcome.external_activity_observation)

    def test_failures_missing_duplicate_and_process_exit_disqualify_rate(self):
        passed = bench.RequestRecord("a", 0, 1, bench.validate_completion(200, [good()]))
        failed = bench.RequestRecord("b", 0, 1, bench.validate_completion(200, [wire({"error": {}})]))
        for records, exit_code in (([passed], 0), ([passed, passed], 0), ([passed, failed], 0), ([passed], 2)):
            outcome = bench.validate_cohort(["a", "b"], records, wall_started=0, wall_finished=2, process_exit_code=exit_code)
            self.assertFalse(outcome.qualified)
            self.assertIsNone(outcome.completion_tokens_per_second)
        self.assertEqual(bench.validate_cohort(["a", "b"], [passed, failed], wall_started=0, wall_finished=2).successful_completion_tokens, 7)

    def test_end_start_tie_is_not_overlap_and_window_is_checked(self):
        out = bench.validate_completion(200, [good()])
        records = [bench.RequestRecord("a", 0, 1, out), bench.RequestRecord("b", 1, 2, out)]
        result = bench.validate_cohort(["a", "b"], records, wall_started=0, wall_finished=2, minimum_peak=2)
        self.assertFalse(result.qualified)
        self.assertEqual(result.peak_concurrency, 1)
        with self.assertRaises(ValueError):
            bench.validate_cohort(["a"], [], wall_started=0, wall_finished=float("nan"))
        result = bench.validate_cohort(["a"], [bench.RequestRecord("a", 0, 3, out)], wall_started=0, wall_finished=2)
        self.assertFalse(result.qualified)

    def test_unbounded_cohort_iterators_are_stopped(self):
        def oversized():
            for _ in range(10001):
                yield "a"
            raise AssertionError("Validator consumed past its bound")
        for expected, records in ((oversized(), []), (["a"], oversized())):
            with self.assertRaisesRegex(ValueError, "iterable exceeds"):
                bench.validate_cohort(expected, records, wall_started=0, wall_finished=2)

    def test_real_http_200_error_and_short_success(self):
        class Server(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(good(2) if self.path == "/short" else wire({"error": {"message": "failed"}}))
            def log_message(self, *_):
                pass
        server = HTTPServer(("127.0.0.1", 0), Server)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            for route, successful in (("/error", False), ("/short", True)):
                with urlopen(f"http://127.0.0.1:{server.server_port}{route}", timeout=5) as response:
                    outcome = bench.validate_completion(response.status, iter(lambda: response.read(7), b""))
                    self.assertEqual(outcome.successful, successful)
                    if successful:
                        self.assertEqual(outcome.completion_tokens, 2)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
