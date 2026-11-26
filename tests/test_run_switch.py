"""crit run ... switch-check on two sealed candidates, A selected and B (support/serve.py's ServeFixture, a scripted
engine behind crit serve): A answers, a request switches crit serve to B and another back to A, and A's greedy replies
before and after are compared; the record is work/switch-check.json."""
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("inference/core/python", "models/core/python", "cli"):
    sys.path.insert(0, str(ROOT / _folder))

from crit.run import switch  # noqa: E402
from crit.run.records import read_json  # noqa: E402
from support.serve import ServeFixture  # noqa: E402


class SwitchCheckTests(ServeFixture, unittest.TestCase):
    def test_the_selected_candidate_answers_alike_before_and_after_a_switch_and_back(self):
        record = switch.switch_check(self.store, self.run_id, prompt="reply:three rivers", log=lambda line: None)
        self.assertEqual((record["selected"], record["other"]), (self.a, self.b))
        self.assertEqual([step["candidate"] for step in record["steps"]], [self.a, self.b, self.a])
        self.assertEqual([step["model"] for step in record["steps"]],
                         [f"{self.run_id}/{c}" for c in (self.a, self.b, self.a)])
        self.assertTrue(record["same_reply"], record)
        self.assertEqual(read_json(self.directory / "work" / "switch-check.json")["steps"], record["steps"])

    def test_a_reply_that_changes_across_the_switch_fails_the_check(self):
        replies = iter(["one", "two", "three"])
        original = switch._chat

        def chat(address, model, prompt, max_tokens):
            _, served, seconds = original(address, model, prompt, max_tokens)
            return next(replies), served, seconds

        with mock.patch.object(switch, "_chat", chat):
            with self.assertRaisesRegex(ValueError, "reply changed"):
                switch.switch_check(self.store, self.run_id, prompt="reply:x", log=lambda line: None)
        self.assertFalse(read_json(self.directory / "work" / "switch-check.json")["same_reply"])

    def test_the_pair_is_the_selected_candidate_and_another_sealed_one(self):
        view = self.store.get(self.run_id)
        self.assertEqual(switch.sealed_pair(view), (self.a, self.b))
        with self.assertRaisesRegex(ValueError, "another sealed candidate"):
            switch.sealed_pair(view, other=self.a)


if __name__ == "__main__":
    unittest.main()
