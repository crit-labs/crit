"""crit run create --from-engine: a run that starts from a crit-hub engine records its origin, keeps its evidence,
checks the checkpoint at resolve and lets the first candidate declare its lineage."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "eval", "kernels", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit.run import hub
from crit.run.cli import main
from crit.run.records import LINEAGE, digest, file_sha, inventory, read_json, write_json
from test_run_records import RunFixture

try:
    import yaml
except ImportError:  # crit-hub's tools need PyYAML; without it the card cannot be read
    yaml = None

REVISION = "0123456789abcdef0123456789abcdef01234567"


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class OriginTests(RunFixture, unittest.TestCase):
    model_files = {"README.md": "# Fixture model\n"}

    def hub_engine(self, owner="someone", name="fast-engine", files=None):
        """A crit-hub engine folder for the fixture model: card.yaml, card.lock.json and one evidence file."""
        folder = self.root / "hub" / "engines" / owner / name
        (folder / "evidence").mkdir(parents=True)
        evidence = folder / "evidence/crit-eval.json"
        write_json(evidence, {"protocol": "crit-eval-v7", "speed": {"decode_score": 100.0, "complete": True}})
        card = {
            "title": "Fixture @ BF16 · RTX 5090 · fast-engine", "summary": "A fixture engine", "origin": "crit",
            "contributor": {"name": owner, "repository": f"https://github.com/{owner}/crit"}, "license": "Apache-2.0",
            "features": ["openai-api"],
            "source": {"repository": "https://github.com/crit-labs/crit-engines", "revision": REVISION, "path": name},
            "requirements": {"platforms": ["linux"], "gpu_arch": ["sm_120a"], "gpu_memory_gb": 32, "cuda": "13.1"},
            "checkpoints": {"fixture@BF16": {"repository": "fixture/model", "revision": REVISION}},
            "results": [{"gpu": "rtx5090", "model": "fixture@BF16", "measured_with": "crit-eval-v7", "measured_on": "2026-10-23",
                         "stats": {"decode_tok_s": 100.0}, "evidence": [{"file": "evidence/crit-eval.json", "sha256": file_sha(evidence)}]}],
            "readme": {"intro": "An engine."},
        }
        (folder / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")
        rows = files if files is not None else inventory(self.model)
        write_json(folder / "card.lock.json", {
            "checkpoints": {"fixture@BF16": {"repository": "fixture/model", "revision": REVISION, "files": rows,
                                             "run_model_manifest_sha256": digest(rows)}},
            "provenance": {"run_id": "run-20261023T011625Z-3141e655", "candidate_id": "candidate-0007",
                           "source_manifest_sha256": "a" * 64}})
        return folder

    def start(self, folder):
        """Create a run from the engine and point the fixture helpers at it."""
        self.view = self.store.create(self.data, origin=folder)
        self.run = self.view["run"]
        self.run_id = self.run["run_id"]
        self.directory = self.store.directory(self.run_id)
        self.store.recon(self.run_id)
        return self.run["spec"]["origin_engine"]

    def test_the_run_records_the_engine_its_source_and_its_evidence(self):
        folder = self.hub_engine()
        origin = self.start(folder)
        self.assertEqual(origin["engine"], "someone/fast-engine")
        self.assertEqual(origin["source"], {"repository": "https://github.com/crit-labs/crit-engines", "revision": REVISION, "path": "fast-engine"})
        self.assertEqual(origin["contributor"]["name"], "someone")
        self.assertEqual(origin["provenance"]["candidate_id"], "candidate-0007")
        self.assertEqual(len(origin["checkpoints"]["fixture@BF16"]["files"]), len(inventory(self.model)))
        self.assertEqual(origin["evidence"], [{"file": "evidence/crit-eval.json", "sha256": file_sha(folder / "evidence/crit-eval.json"),
                                               "measured_with": "crit-eval-v7", "gpu": "rtx5090", "model": "fixture@BF16",
                                               "path": "captured/origin/crit-eval.json"}])
        copied = self.directory / "captured/origin/crit-eval.json"
        self.assertEqual(file_sha(copied), origin["evidence"][0]["sha256"])
        self.assertEqual(digest(self.run["spec"]), self.run["request_sha256"], "the stored origin is part of the frozen request")
        request = (self.directory / "segments/segment-0001/agent_request.md").read_text(encoding="utf-8")
        self.assertIn("Origin engine: this run starts from the crit-hub engine someone/fast-engine", request)
        self.assertIn(REVISION, request)
        self.assertIn(f'"lineage": "{LINEAGE}"', request)
        self.assertIn("captured/origin/crit-eval.json (crit-eval-v7, rtx5090, fixture@BF16)", request)
        self.assertEqual(self.store.get(self.run_id)["run"]["spec"]["origin_engine"]["engine"], "someone/fast-engine")

    def test_resolve_says_whether_the_model_is_the_measured_checkpoint(self):
        self.start(self.hub_engine())
        self.resolve()
        found = read_json(self.directory / "inputs.json")["origin_checkpoint"]
        self.assertEqual(found["matched"], "fixture@BF16")
        self.assertTrue(found["checkpoints"]["fixture@BF16"]["matches"])
        # Another run on a model whose README changed and which gained a file is not that checkpoint.
        (self.model / "README.md").write_text("# Finetune\n", encoding="utf-8")
        (self.model / "extra.bin").write_text("more", encoding="utf-8")
        self.start(self.hub_engine(name="other-engine"))
        rows = inventory(self.model)
        # The lock still describes the original files, so the run's model differs from it.
        folder = self.root / "hub/engines/someone/other-engine"
        lock = read_json(folder / "card.lock.json")
        original = [row for row in lock["checkpoints"]["fixture@BF16"]["files"] if row["path"] != "extra.bin"]
        original = [dict(row, sha256="b" * 64) if row["path"] == "README.md" else row for row in original]
        lock["checkpoints"]["fixture@BF16"]["files"] = original
        write_json(folder / "card.lock.json", lock)
        self.start(folder)
        self.resolve()
        found = read_json(self.directory / "inputs.json")["origin_checkpoint"]
        self.assertIsNone(found["matched"])
        report = found["checkpoints"]["fixture@BF16"]
        self.assertFalse(report["matches"])
        self.assertEqual(report["differing"], ["README.md"])
        self.assertEqual(report["extra"], ["extra.bin"])
        self.assertEqual(len(rows), len(original) + 1)

    def test_the_first_candidate_declares_its_lineage(self):
        self.start(self.hub_engine())
        candidate = self.candidate(lineage=LINEAGE)
        self.assertEqual(candidate["lineage"], LINEAGE)
        stored = read_json(self.directory / "candidates" / candidate["candidate_id"] / "candidate.json")
        self.assertEqual(stored["lineage"], LINEAGE)
        self.assertEqual(self.store.get(self.run_id)["candidates"][0]["candidate_id"], candidate["candidate_id"])
        self.assertNotIn("lineage", self.candidate())
        with self.assertRaisesRegex(ValueError, "lineage is origin-engine"):
            self.candidate(lineage="fork")
        other = self.store.create(self.data)["run"]["run_id"]
        self.store.recon(other)
        self.run_id, self.directory = other, self.store.directory(other)
        self.run = read_json(self.directory / "run.json")
        with self.assertRaisesRegex(ValueError, "needs a run created with --from-engine"):
            self.candidate(lineage=LINEAGE)

    def test_a_damaged_or_incomplete_hub_folder_is_refused(self):
        folder = self.hub_engine()
        (folder / "evidence/crit-eval.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match its sha256"):
            hub.describe_engine(folder)
        (folder / "card.lock.json").unlink()
        with self.assertRaisesRegex(ValueError, "needs card.yaml and card.lock.json"):
            hub.describe_engine(folder)
        with self.assertRaisesRegex(ValueError, "Name the origin engine once"):
            self.store.create({**self.data, "origin_engine": {}}, origin=self.root / "hub/engines/someone/fast-engine")

    def test_the_cli_creates_a_run_from_an_engine(self):
        folder = self.hub_engine()
        request = self.root / "request.json"
        write_json(request, self.data)
        code = main(["--project", str(ROOT), "--root", str(self.store.root), "create", "--record", str(request),
                     "--from-engine", str(folder)])
        self.assertEqual(code, 0)
        runs = [p for p in self.store.root.iterdir() if p.is_dir() and p.name != self.run_id]
        self.assertEqual(len(runs), 1)
        self.assertEqual(read_json(runs[0] / "run.json")["spec"]["origin_engine"]["engine"], "someone/fast-engine")
        self.assertTrue((runs[0] / "captured/origin/crit-eval.json").is_file())
