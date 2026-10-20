"""Sealing must remain retryable across publication boundaries without changing evidence."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from test_run_records import ROOT, RunFixture
from crit.run import records, store as run_store
from crit.run.records import inventory


class SealRecoveryTests(RunFixture, unittest.TestCase):
    def prepare(self):
        candidate = self.candidate()
        self.package(candidate)
        candidate_id = candidate["candidate_id"]
        return candidate_id, self.directory / "candidates" / candidate_id

    def assert_retry(self, candidate_id, folder):
        original_outcome = (folder / "outcome.json").read_bytes()
        self.assertEqual(self.store.recover(self.run_id)["errors"], [])
        self.store.seal(self.run_id, candidate_id, "built")
        view = self.store.get(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertTrue(view["candidates"][0]["sealed"])
        self.assertEqual(original_outcome, (folder / "outcome.json").read_bytes())
        with self.assertRaises(ValueError):
            self.store.seal(self.run_id, candidate_id, "cancelled")

    def test_retry_after_outcome_publication_keeps_the_original_outcome(self):
        candidate_id, folder = self.prepare()
        with patch.object(self.store, "_refresh", side_effect=OSError("projection failed")):
            with self.assertRaisesRegex(OSError, "projection failed"):
                self.store.seal(self.run_id, candidate_id, "built")
        self.assert_retry(candidate_id, folder)

    def test_retry_after_failed_seal_write(self):
        candidate_id, folder = self.prepare()
        write = run_store.write_json

        def fail_seal(path, *args, **kwargs):
            if path.name == "seal.json":
                raise OSError("seal write failed")
            return write(path, *args, **kwargs)

        with patch.object(run_store, "write_json", side_effect=fail_seal):
            with self.assertRaisesRegex(OSError, "seal write failed"):
                self.store.seal(self.run_id, candidate_id, "built")
        self.assert_retry(candidate_id, folder)

    def test_retry_after_seal_publication_never_rewrites_candidate_files(self):
        candidate_id, folder = self.prepare()
        state = self.store._state

        def fail_state(root, **changes):
            if (folder / "seal.json").exists():
                raise OSError("state update failed")
            return state(root, **changes)

        with patch.object(self.store, "_state", side_effect=fail_state):
            with self.assertRaisesRegex(OSError, "state update failed"):
                self.store.seal(self.run_id, candidate_id, "built")
        retained = inventory(folder)
        self.assert_retry(candidate_id, folder)
        self.assertEqual(inventory(folder), retained)
        (folder / "package/engine.fixture").write_text("tampered package", encoding="utf-8")
        tampered = inventory(folder)
        with self.assertRaises(ValueError):
            self.store.seal(self.run_id, candidate_id, "built")
        self.assertEqual(inventory(folder), tampered)

    def test_abrupt_exit_after_seal_publication_keeps_the_sealed_inventory_valid(self):
        candidate_id, folder = self.prepare()
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--crash-seal",
             str(self.store.root), self.run_id, candidate_id],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
        self.assertTrue((folder / "seal.json").is_file())
        retained = inventory(folder)
        view = self.store.recover(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertTrue(view["candidates"][0]["sealed"])
        self.assertEqual(inventory(folder), retained)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--crash-seal":
        link = records.os.link

        def crash_after_link(source, destination, *args, **kwargs):
            link(source, destination, *args, **kwargs)
            if Path(destination).name == "seal.json":
                os._exit(73)

        records.os.link = crash_after_link
        run_store.RunStore(sys.argv[2]).seal(sys.argv[3], sys.argv[4], "built")
    else:
        unittest.main()
