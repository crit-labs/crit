"""Eval's follow-up turns (protocol followup): measured with whatever prompt cache the engine has, and its cached path
must answer as its fresh one did."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval import chat, runner, status  # noqa: E402

TURNS = [{"turn": 1, "prompt_ids": list(range(1, 101))},
         {"turn": 2, "prompt_ids": list(range(1, 121))},
         {"turn": 3, "prompt_ids": list(range(1, 141))}]


class Engine(chat.ChatEngine):
    """An engine behind a server: with `cache`, it serves the longest prompt prefix it saw before from its cache;
    with `drifts`, a reply after a cache hit differs from a fresh one."""

    def __init__(self, cache=False, drifts=False):
        self.cache, self.drifts, self.seen, self.asked = cache, drifts, [], []

    def generate(self, prompt, max_new_tokens, on_tokens, prompt_cache=False):
        self.asked.append(prompt_cache)
        cached = max((len(p) for p in self.seen if prompt[:len(p)] == p), default=0) if self.cache else 0
        self.seen.append(list(prompt))
        tokens = [len(prompt) % 50 + (1 if self.drifts and cached else 0)] * max_new_tokens
        on_tokens(tokens)
        return {"tokens": tokens, "served": {"prompt_tokens": len(prompt), "output_tokens": max_new_tokens,
                                             "cached_tokens": cached, "finish_reason": "length"}}


class FollowupTests(unittest.TestCase):
    protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")

    def measure(self, engine, max_context=1000):
        prepared = {"followup": TURNS}
        first = runner.run_followup(engine, {"max_context": max_context}, prepared, self.protocol, lambda m: None)
        if first["status"] == "measured":
            first["again"] = runner.compare_followup(
                first, runner.run_followup(engine, {"max_context": max_context}, prepared, self.protocol, lambda m: None))
        gate = next(g for g in status._gates({"requirements": {"exactness": "reported"}, "context": {"tokens": None,
                    "reason": "-", "missing": []}, "followup": first}) if g["name"] == "prompt_cache")
        return first, gate

    def test_without_a_cache_every_turn_prefills_and_the_gate_passes(self):
        engine = Engine()
        first, gate = self.measure(engine)
        self.assertEqual([t["cached_tokens"] for t in first["turns"]], [0, 0, 0])
        self.assertEqual(gate["state"], "pass")
        self.assertTrue(all(engine.asked))   # every turn allows a prompt cache
        # The KPIs cover the turns after the first: the prefill rate counts every prompt token.
        self.assertAlmostEqual(first["turns"][1]["prefill_tps"], 120 * 1000 / first["turns"][1]["ttft_ms"])

    def test_a_cache_serves_the_earlier_turns_and_its_answers_hold(self):
        first, gate = self.measure(Engine(cache=True))
        self.assertEqual([t["cached_tokens"] for t in first["turns"]], [0, 100, 120])
        # The prefill rate counts only the tokens the engine computed.
        self.assertAlmostEqual(first["turns"][2]["prefill_tps"], 20 * 1000 / first["turns"][2]["ttft_ms"])
        self.assertEqual(gate["state"], "pass")
        self.assertIn("cached prompt tokens 0, 100, 120", gate["detail"])

    def test_a_cache_that_changes_the_answer_fails(self):
        first, gate = self.measure(Engine(cache=True, drifts=True))
        self.assertEqual(gate["state"], "fail")
        self.assertIn("turn 1 differs at token 0", gate["detail"])   # cached the second time, fresh the first

    def test_turns_that_do_not_fit_are_skipped_and_unmeasured(self):
        first, gate = self.measure(Engine(), max_context=150)
        self.assertEqual((first["status"], gate["state"]), ("skipped", "unmeasured"))

    def test_timed_requests_still_refuse_a_prompt_cache(self):
        engine = Engine(cache=True)
        runner.timed_generate(engine, [1, 2, 3], 4)
        runner.timed_generate(engine, [1, 2, 3], 4, prompt_cache=True)
        self.assertEqual(engine.asked, [False, True])


if __name__ == "__main__":
    unittest.main()
