"""crit run cost: agent transcripts attributed to the candidates built in their spans, priced at recorded rates."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "../crit-eval", "kernels", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit.run import cost
from crit.run.cli import main
from crit.run.records import read_json
from test_run_records import RunFixture


def iso(when):
    return when.isoformat().replace("+00:00", "Z")


class CostTests(RunFixture, unittest.TestCase):
    def sealed(self, **updates):
        candidate = self.measured(**updates)[0]
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        return candidate["candidate_id"], datetime.fromisoformat(
            next(read_json(p)["recorded_at"] for p in sorted((self.directory / "events").glob("event-*.json"))
                 if read_json(p)["event_type"] == "candidate_sealed" and read_json(p)["candidate_id"] == candidate["candidate_id"]))

    def claude_code(self, path, rows):
        """Claude Code writes one line per content block; a response's lines share requestId and usage."""
        lines = []
        for at, model, usage, request_id in rows:
            for block in range(2):
                lines.append(json.dumps({"type": "assistant", "timestamp": iso(at), "requestId": request_id, "cwd": "x",
                                         "message": {"id": "msg_" + request_id, "model": model, "usage": usage}}))
        lines.append("not json at all")
        lines.append(json.dumps({"type": "user", "timestamp": iso(rows[0][0]), "message": {"role": "user", "content": "hi"}}))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_requests_are_attributed_to_the_candidate_built_in_their_span_and_priced(self):
        created = datetime.fromisoformat(self.run["created_at"])
        a, a_sealed = self.sealed()
        b, b_sealed = self.sealed(label="Second engine", configuration={"batch": 2})
        self.assertLess(created, a_sealed)
        self.assertLessEqual(a_sealed, b_sealed)
        usage = {"input_tokens": 10, "cache_read_input_tokens": 1_000_000, "output_tokens": 100,
                 "cache_creation_input_tokens": 500_000,
                 "cache_creation": {"ephemeral_1h_input_tokens": 500_000, "ephemeral_5m_input_tokens": 0}}
        transcript = self.root / "session.jsonl"
        self.claude_code(transcript, [
            (created - timedelta(minutes=5), "claude-opus-5-5", usage, "req-before"),
            (created, "claude-opus-5-5", usage, "req-a1"),
            (a_sealed - timedelta(microseconds=1), "claude-opus-5-5", usage, "req-a2"),
            (a_sealed, "claude-opus-5-5", usage, "req-b1"),
            (b_sealed + timedelta(seconds=1), "claude-opus-5-5", usage, "req-after"),
        ])
        until = b_sealed + timedelta(minutes=1)
        value = self.store.cost(self.run_id, [transcript], agent="Claude Opus 5.5 in Claude Code", until=iso(until))
        self.assertEqual(value["record_kind"], "run_cost")
        self.assertEqual(value["transcripts"][0]["format"], "claude-code")
        self.assertEqual(value["transcripts"][0]["requests"], 5, "a response counts once, however many lines it spans")
        rows = {row["candidate_id"]: row for row in value["candidates"]}
        self.assertEqual([row["requests"] for row in value["candidates"]], [2, 1])
        # 10 input at $4, 1M cache reads at $0.20, 500K 1h cache writes at $8, 100 output at $20 per million tokens.
        each = (10 * 4 + 1_000_000 * 0.20 + 500_000 * 8 + 100 * 20) / 1_000_000
        self.assertAlmostEqual(rows[a]["usd"], 2 * each, places=4)
        self.assertAlmostEqual(rows[b]["usd"], each, places=4)
        self.assertEqual(rows[a]["tokens"], {"input": 20, "cache_read": 2_000_000, "cache_write_5m": 0, "cache_write_1h": 1_000_000,
                                             "output": 200, "total": 3_000_220})
        self.assertEqual(rows[a]["from"], iso(created))
        self.assertEqual(rows[a]["until"], iso(a_sealed))
        self.assertEqual(rows[b]["from"], iso(a_sealed))
        self.assertTrue(rows[b]["sealed"])
        self.assertEqual(rows[b]["cumulative"]["tokens"], 3 * 1_500_110)
        self.assertAlmostEqual(rows[b]["cumulative"]["usd"], 3 * each, places=4)
        self.assertEqual(rows[b]["cumulative"]["seconds"], round((b_sealed - created).total_seconds()))
        self.assertEqual(value["selection"]["requests"], 1)
        self.assertEqual(value["before_run"]["requests"], 1)
        self.assertEqual(value["run"]["requests"], 4)
        self.assertEqual(value["run"]["until_source"], "given")
        self.assertAlmostEqual(value["run"]["usd"], 4 * each, places=4)
        self.assertEqual(value["run"]["seconds"], round((until - created).total_seconds()))
        self.assertEqual(value["prices"]["usd_per_million_tokens"]["claude-opus-5-5"]["cache_write_1h"], 8.0)
        self.assertEqual(read_json(self.directory / "cost.json"), value)
        self.assertEqual(cost.built(value)["agent_tokens"], value["run"]["tokens"]["total"])

    def test_unpriced_models_keep_their_tokens_and_prices_can_be_supplied(self):
        created = datetime.fromisoformat(self.run["created_at"])
        a, a_sealed = self.sealed()
        usage = {"input_tokens": 1000, "output_tokens": 1000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        transcript = self.root / "other.jsonl"
        self.claude_code(transcript, [(created, "future-model", usage, "r1"), (created, "claude-opus-5-5-20270101", usage, "r2")])
        value = self.store.cost(self.run_id, [transcript], until=iso(a_sealed + timedelta(seconds=1)))
        row = value["candidates"][0]
        self.assertEqual(row["tokens"]["total"], 4000)
        self.assertIsNone(row["usd"])
        self.assertEqual(row["unpriced_requests"], 1)
        self.assertAlmostEqual(row["usd_priced"], (1000 * 4 + 1000 * 20) / 1_000_000, places=6, msg="a dated variant takes its model's rates")
        self.assertIn("future-model", row["usd_note"])
        self.assertIsNone(row["cumulative"]["usd"])
        priced = self.store.cost(self.run_id, [transcript], prices=json.dumps({"future-model": {"input": 1.0, "output": 2.0}}),
                                 until=iso(a_sealed + timedelta(seconds=1)))
        self.assertAlmostEqual(priced["candidates"][0]["usd"], (1000 * 4 + 1000 * 20 + 1000 * 1 + 1000 * 2) / 1_000_000, places=6)
        self.assertIn("--prices", priced["prices"]["source"])
        with self.assertRaisesRegex(ValueError, "rates are any of"):
            self.store.cost(self.run_id, [transcript], prices=json.dumps({"future-model": {"tokens": 1}}))

    def test_codex_and_plain_transcripts_and_the_cli(self):
        created = datetime.fromisoformat(self.run["created_at"])
        a, a_sealed = self.sealed()
        codex = self.root / "rollout.jsonl"
        codex.write_text("\n".join([
            json.dumps({"timestamp": iso(created), "type": "session_meta", "payload": {"id": "s"}}),
            json.dumps({"timestamp": iso(created), "type": "event_msg", "payload": {"type": "thread_settings_applied",
                                                                                      "thread_settings": {"model": "gpt-x"}}}),
            json.dumps({"timestamp": iso(created), "type": "event_msg", "payload": {"type": "token_count", "info": {
                "last_token_usage": {"input_tokens": 1500, "cached_input_tokens": 1000, "cache_write_input_tokens": 0,
                                     "output_tokens": 50, "reasoning_output_tokens": 20, "total_tokens": 1550}}}}),
            json.dumps({"timestamp": iso(created), "type": "event_msg", "payload": {"type": "token_count", "info": None}}),
        ]) + "\n", encoding="utf-8")
        plain = self.root / "plain.jsonl"
        plain.write_text(json.dumps({"timestamp": iso(created), "model": "claude-sonnet-5-5", "input_tokens": 100,
                                     "output_tokens": 10}) + "\n", encoding="utf-8")
        value = self.store.cost(self.run_id, [codex, plain], until=iso(a_sealed + timedelta(seconds=1)))
        self.assertEqual([t["format"] for t in value["transcripts"]], ["codex", "plain"])
        row = value["candidates"][0]
        self.assertEqual(row["requests"], 2)
        self.assertEqual(row["tokens"]["input"], 600, "Codex input_tokens include the cached ones, which are split out")
        self.assertEqual(row["tokens"]["cache_read"], 1000)
        self.assertEqual(row["models"], {"gpt-x": 1, "claude-sonnet-5-5": 1})
        self.assertEqual(row["unpriced_requests"], 1)
        empty = self.root / "empty.jsonl"
        empty.write_text(json.dumps({"type": "user"}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no model requests"):
            self.store.cost(self.run_id, [empty])
        code = main(["--project", str(ROOT), "--root", str(self.store.root), "--id", self.run_id, "cost",
                     "--transcript", str(plain), "--agent", "fixture", "--until", iso(a_sealed + timedelta(seconds=1))])
        self.assertEqual(code, 0)
        self.assertEqual(read_json(self.directory / "cost.json")["agent"], "fixture")
