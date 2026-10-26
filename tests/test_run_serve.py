"""crit run serve: a packaged candidate served through crit serve before it is sealed, under a check attempt that
settles when the server stops."""
import json
import threading
import unittest
import urllib.request

from test_run_measure import MeasureFixture
from crit.run import served
from crit.run.records import read_json


class RunServeTests(MeasureFixture, unittest.TestCase):
    def serve_in_thread(self, candidate):
        ready, stop, out = threading.Event(), threading.Event(), {}

        def bound(port):
            out["port"] = port
            ready.set()

        def run():
            try:
                out["result"] = served.serve(self.store, self.run_id, candidate, port=0, ready=bound, stop=stop)
            except BaseException as exc:  # reported by the test
                out["error"] = exc
                ready.set()
        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(ready.wait(120), "the server never listened")
        self.assertNotIn("error", out)
        return thread, stop, out

    def chat(self, port, content, stream=False):
        body = json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": 4,
                           "stream": stream}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode()

    def test_serves_the_unsealed_candidate_until_stopped_and_settles_its_check_attempt(self):
        candidate = self.served()
        thread, stop, out = self.serve_in_thread(candidate)
        try:
            reply = json.loads(self.chat(out["port"], "hello"))
            self.assertEqual(reply["choices"][0]["message"]["content"], "xxxx")
            self.assertIn("data:", self.chat(out["port"], "hello again", stream=True))
        finally:
            stop.set()
            thread.join(120)
        self.assertNotIn("error", out)
        reservation = read_json(self.folder / "attempts/attempt-0001/reservation.json")
        self.assertEqual(reservation["operation"], "check")
        result = self.attempt()
        self.assertEqual((result["outcome"], result["cleanup_settled"]), ("completed", True))
        self.assertEqual(out["result"], result)
        log = self.directory / result["logs"][0]["path"]
        self.assertIn("not sealed", log.read_text(encoding="utf-8"))

    def test_refuses_a_sealed_or_unpackaged_candidate(self):
        unpackaged = self.candidate()["candidate_id"]
        with self.assertRaisesRegex(ValueError, "Package the candidate"):
            served.serve(self.store, self.run_id, unpackaged, port=0)
        candidate = self.served()
        self.store.seal(self.run_id, candidate, "built")
        with self.assertRaisesRegex(ValueError, "sealed: serve it with crit serve"):
            served.serve(self.store, self.run_id, candidate, port=0)


if __name__ == "__main__":
    unittest.main()
