"""Candidate creation retains complete source and identity across coordinator failure."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from test_run_records import ROOT, RunFixture
from crit.run import store as run_store
from crit.run.records import inventory


class CandidatePublicationTests(RunFixture, unittest.TestCase):
    def prepare(self):
        self.resolve()
        source = self.root / "source-input"
        source.mkdir()
        (source / "engine.fixture").write_text("complete retained source", encoding="utf-8")
        return source

    def test_entire_work_directory_can_supply_candidate_source(self):
        self.resolve()
        source = self.directory / "work"
        (source / "engine.fixture").write_text("source from work", encoding="utf-8")
        copytree = shutil.copytree
        calls = []

        def bounded_copy(*args, **kwargs):
            calls.append(args)
            if len(calls) > 5:
                raise OSError("recursive copy of staging into itself")
            return copytree(*args, **kwargs)

        with patch.object(shutil, "copytree", side_effect=bounded_copy):
            candidate = self.store.candidate(self.run_id, source, self.proposal())
        folder = self.directory / "candidates" / candidate["candidate_id"]
        self.assertEqual(inventory(folder / "source"), inventory(source))
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_identity_write_failure_does_not_leave_a_candidate_that_blocks_recovery(self):
        source = self.prepare()
        write = run_store.write_json

        def fail_identity(path, *args, **kwargs):
            if path.name == "candidate.json":
                raise OSError("identity write failed")
            return write(path, *args, **kwargs)

        with patch.object(run_store, "write_json", side_effect=fail_identity):
            with self.assertRaisesRegex(OSError, "identity write failed"):
                self.store.candidate(self.run_id, source, self.proposal())
        view = self.store.recover(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertEqual(view["candidates"], [])
        candidate = self.store.candidate(self.run_id, source, self.proposal())
        self.assertEqual(candidate["candidate_id"], "candidate-0001")

    def crash(self, phase, source):
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--crash", phase,
             str(self.store.root), self.run_id, str(source), json.dumps(self.proposal())],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 73, result.stdout + result.stderr)

    def test_abrupt_exit_during_source_copy_leaves_no_partial_candidate(self):
        source = self.prepare()
        self.crash("copy", source)
        view = self.store.recover(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertEqual(view["candidates"], [])
        candidate = self.store.candidate(self.run_id, source, self.proposal())
        folder = self.directory / "candidates" / candidate["candidate_id"]
        self.assertEqual(inventory(folder / "source"), inventory(source))

    def test_abrupt_exit_after_candidate_publication_recovers_its_original_identity(self):
        source = self.prepare()
        self.crash("published", source)
        folder = self.directory / "candidates/candidate-0001"
        original = (folder / "candidate.json").read_bytes()
        view = self.store.recover(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertEqual(len(view["candidates"]), 1)
        self.assertEqual(view["candidates"][0]["candidate_id"], "candidate-0001")
        self.assertFalse(view["candidates"][0]["sealed"])
        self.assertEqual((folder / "candidate.json").read_bytes(), original)
        self.assertEqual(inventory(folder / "source"), inventory(source))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--crash":
        if sys.argv[2] == "copy":
            def crash_copy(source, destination, *args, **kwargs):
                destination.mkdir()
                shutil.copy2(source / "engine.fixture", destination / "engine.fixture")
                os._exit(73)

            run_store.shutil.copytree = crash_copy
        else:
            run_store.RunStore._refresh = lambda *args: os._exit(73)
        run_store.RunStore(sys.argv[3]).candidate(sys.argv[4], Path(sys.argv[5]), json.loads(sys.argv[6]))
    else:
        unittest.main()
