"""crit run eval: a packaged candidate's Eval through crit serve, under a benchmark reservation."""
import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import test_run_records as run_records
from test_eval_server import Text, write_data
from crit.run import evaluation
from crit.run.records import file_sha, read_json
from crit_eval import chat, prepare

# Answers Eval's prompts as a chat server over Text's tokens (characters): the prompt renders as Text renders it and
# the greedy reply is REPLY, each token with its alternatives when the request asks for log-probabilities.
ADAPTER = r'''
import time

REPLY = "tides rise and fall " * 20
OTHERS = "abcdefghijklmnopq"

class Engine:
    logprobs = True

    def __init__(self, context):
        self.weights = context.host('model.fixture')

    def generate(self, request, cancellation):
        return self.generate_stream(request, cancellation, None)

    def generate_stream(self, request, cancellation, emit):
        prompt = ''.join('<%s>%s' % (m['role'], m['content']) for m in request['messages']) + '<a>'
        reply = REPLY[:request['limits']['max_output_tokens']]
        for c in reply:
            cancellation.check()
            if emit:
                emit(c)
                time.sleep(.01)  # tokens arrive over time, so the stream has a decode time
        response = dict(schema_version=1, record_kind='chat_response',
                        **{k: request[k] for k in ('request_id', 'candidate_id', 'package_sha256')}, assistant=reply,
                        telemetry=dict(prompt_tokens=len(prompt), output_tokens=len(reply), prefill_tokens=len(prompt),
                                       prefill_ms=2, decode_tokens=len(reply) - 1, decode_ms=10, ttft_ms=3, missing={}))
        n = request.get('logprobs')
        if n is not None:
            def row(c):
                top = [[ord(c), -0.1]] + [[ord(o), -3.0 - i] for i, o in enumerate(o for o in OTHERS if o != c)]
                return [ord(c), -0.1, top[:n]]
            rows = [row(c) for c in reply]
            named = {t for r in rows for t in [r[0]] + [p[0] for p in r[2]]}
            response['logprobs'] = {'tokens': rows, 'bytes': {str(t): chr(t).encode().hex() for t in named}}
        return response

    def reset(self):
        pass

    def close(self):
        pass

def create_engine(context):
    return Engine(context)
'''
SERVED = ["prefill", "decode", "chat-json-v1", "chat-resident-v1", "chat-stream-v1", "chat-sampling-temperature-v1",
          "chat-logprobs-v1"]


class RunEvalTests(run_records.RunFixture, unittest.TestCase):
    capabilities = SERVED

    def setUp(self):
        super().setUp()
        for patcher in (patch.dict(os.environ, {"CRIT_ENGINE_LOCK": "off",
                                                "CRIT_RUNTIME_DIRECTORY": str(self.root / "runtime")}),
                        patch.object(chat, "load_text", lambda model: Text()),
                        patch.object(chat, "ids_by_bytes",
                                     lambda tokenizer: {chr(i).encode(): i for i in range(1, 0xD800) if i != 0xFFFD})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.candidate_id = self.candidate()["candidate_id"]
        package = self.root / "served-package"
        package.mkdir()
        (package / "engine.py").write_text(ADAPTER)
        self.store.package(self.run_id, self.candidate_id, package,
                           {"build": {"toolchain": {"fixture": "host-python"}, "dependencies": [], "command": ["fixture"]},
                            "launch": ["engine.py"], "capabilities": self.capabilities})
        self.folder = self.directory / "candidates" / self.candidate_id
        self.eval_data = self.root / "eval-data"
        self.eval_data.mkdir()
        write_data(self.eval_data)
        prepared = json.loads((self.eval_data / "prepared.json").read_text())
        prepared["model"] = prepare.model_identity(self.model)
        prepared["speed"] = [case for case in prepared["speed"] if not case["tools"]]  # the package declares no tools
        (self.eval_data / "prepared.json").write_text(json.dumps(prepared))
        self.scored = []

    def score(self, model, data, path, baseline, log):
        """The reference agrees with the engine: the quality gate passes."""
        self.scored.append((model, Path(data), Path(path)))
        result = read_json(Path(path))
        result["quality"]["status"] = "scored"
        result["status"] = {"gates": [{"name": "quality", "state": "pass", "detail": "fixture reference"}],
                            "kpis": {}}
        Path(path).write_text(json.dumps(result))
        return result

    def evaluate(self, parts=("speed", "heldout", "history", "quality", "context")):
        with patch("crit_eval.reference.score_result", self.score):
            return evaluation.evaluate(self.store, self.run_id, self.candidate_id, self.eval_data, parts)

    def test_eval_measures_the_packaged_candidate_through_crit_serve_and_scores_its_replies(self):
        measured = self.evaluate()
        path = self.directory / measured["result"]["path"]
        self.assertEqual(measured["result"], {"path": f"candidates/{self.candidate_id}/eval/attempt-0001.json",
                                              "sha256": file_sha(path)})
        result = read_json(path)
        self.assertEqual(result["server"], f"crit serve {self.run_id}/{self.candidate_id}")
        self.assertIsNone(result["adapter"])
        self.assertEqual(result["engine"]["name"], f"{self.run_id}/{self.candidate_id}")
        self.assertTrue(result["engine"]["configuration"]["url"].startswith("http://127.0.0.1:"))
        self.assertEqual(result["engine"]["max_context"], 8192)
        # The lower-context gate's engine is the candidate served again at a lower context, before the run's.
        lower = result["context"]["lower"]
        self.assertEqual((lower["context"], set(lower["cases"])), (2048, {"chat"}))
        self.assertIsNotNone(lower["cases"]["chat"]["decode_tps"])
        self.assertTrue(result["context"]["points"])  # the long-context curve, served at the run's context
        self.assertEqual(result["requirements"]["exactness"], "reported")
        chat_case = result["speed"]["cases"]["chat"]
        self.assertEqual(chat_case["tokens"], prepare.token_ids(Text(), ("tides rise and fall " * 20)[:12]))
        replies = result["quality"]["replies"]
        self.assertEqual([r["set"] for r in replies], ["conversations", "long-3000", "long-5000"])
        self.assertEqual(replies[0]["top"][0][:2], [[ord("t"), -0.1], [ord("a"), -3.0]])
        self.assertEqual(self.scored, [(str(self.model), self.eval_data.absolute(), path)])
        self.assertEqual(measured["status"]["gates"][0]["state"], "pass")
        attempt = read_json(self.folder / "attempts/attempt-0001/result.json")
        self.assertEqual((attempt["outcome"], attempt["cleanup_settled"]), ("completed", True))
        log = attempt["logs"][0]
        self.assertEqual(log["path"], f"candidates/{self.candidate_id}/logs/eval-attempt-0001.log")
        logged = (self.directory / log["path"]).read_text()
        self.assertEqual([line for line in logged.splitlines() if line.startswith("crit serve: loading")],
                         [f"crit serve: loading {self.candidate_id} at 2,048 tokens of context",
                          f"crit serve: loading {self.candidate_id} at 8,192 tokens of context"])
        self.assertEqual(logged.count("crit serve: stopped"), 2)
        self.assertFalse((self.root / "runtime/owner.json").exists())
        self.store.seal(self.run_id, self.candidate_id, "built")  # the result is part of the sealed inventory
        self.assertIn(measured["result"]["path"].split("/", 2)[2],
                      {row["path"] for row in read_json(self.folder / "seal.json")["files"]})

    def test_a_sealed_candidate_or_other_data_is_refused_before_any_reservation(self):
        other = json.loads((self.eval_data / "prepared.json").read_text())
        other["model"] = {"sha256": "another model"}
        (self.root / "other").mkdir()
        (self.root / "other/prepared.json").write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError, "prepared from another model"):
            evaluation.evaluate(self.store, self.run_id, self.candidate_id, self.root / "other")
        with self.assertRaisesRegex(ValueError, "holds no prepared.json"):
            evaluation.evaluate(self.store, self.run_id, self.candidate_id, self.root)
        self.store.seal(self.run_id, self.candidate_id, "built")
        with self.assertRaisesRegex(ValueError, "Sealed candidate is immutable"):
            self.evaluate()
        self.assertFalse((self.folder / "attempts").exists())

    def test_the_deadline_stops_the_served_engine_and_settles_a_timeout(self):
        with patch("crit_eval.reference.score_result", self.score):
            with self.assertRaisesRegex(ValueError, "Eval timed_out: Eval passed its reserved deadline"):
                evaluation.evaluate(self.store, self.run_id, self.candidate_id, self.eval_data, seconds=2)
        attempt = read_json(self.folder / "attempts/attempt-0001/result.json")
        self.assertEqual((attempt["outcome"], attempt["cleanup_settled"]), ("timed_out", True))
        self.assertEqual(self.scored, [])
        self.assertFalse((self.root / "runtime/owner.json").exists())

    def test_the_command_prints_the_retained_result(self):
        from crit.run.cli import main
        out = io.StringIO()
        with patch("crit_eval.reference.score_result", self.score), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            code = main(["--root", str(self.store.root), "--id", self.run_id, "eval", "--candidate", self.candidate_id,
                         "--data", str(self.eval_data), "--only", "speed", "--label", "served"])
        self.assertEqual(code, 0)
        printed = json.loads(out.getvalue())
        self.assertEqual(printed["result"]["path"], f"candidates/{self.candidate_id}/eval/attempt-0001.json")
        self.assertEqual(read_json(self.directory / printed["result"]["path"])["label"], "served")

    def test_a_failing_engine_settles_its_attempt_as_failed(self):
        engine = self.folder / "package/engine.py"
        with patch("crit_api.engine.ServedEngine.open", side_effect=ValueError("synthetic load failure")):
            with self.assertRaisesRegex(ValueError, "Eval failed: ValueError: synthetic load failure"):
                self.evaluate()
        attempt = read_json(self.folder / "attempts/attempt-0001/result.json")
        self.assertEqual((attempt["outcome"], attempt["cleanup_settled"]), ("failed", True))
        self.assertTrue(engine.is_file())
        self.assertFalse((self.root / "runtime/owner.json").exists())


class UnservableEvalTests(run_records.RunFixture, unittest.TestCase):
    def test_eval_needs_a_streaming_resident_package_with_logprobs_for_quality(self):
        candidate = self.candidate()["candidate_id"]
        package = self.root / "served-package"
        package.mkdir()
        (package / "engine.py").write_text(ADAPTER)
        self.store.package(self.run_id, candidate, package,
                           {"build": {"toolchain": {"fixture": "host-python"}, "dependencies": [], "command": ["fixture"]},
                            "launch": ["engine.py"], "capabilities": [c for c in SERVED if c != "chat-logprobs-v1"]})
        with self.assertRaisesRegex(ValueError, "must declare chat-logprobs-v1"):
            evaluation.evaluate(self.store, self.run_id, candidate, self.root)
        with self.assertRaisesRegex(ValueError, "holds no prepared.json"):
            evaluation.evaluate(self.store, self.run_id, candidate, self.root, ["speed"])


if __name__ == "__main__":
    unittest.main()
