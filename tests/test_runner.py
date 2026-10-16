"""Record discovery and owned process trees; no real optimization agent."""
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

from test_run_records import ROOT, RunFixture
from crit.process.tree import ProcessTree
from crit.run.store import RunStore


class RunnerTests(RunFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        guard = patch.dict('os.environ', {'CRIT_RUNTIME_DIRECTORY': str(self.root / 'runtime-owner')})
        guard.start()
        self.addCleanup(guard.stop)

    def test_config_reports_invalid_new_records_but_ignores_old_job_folders(self):
        runs = RunStore(self.root / "discovery-runs")
        old = runs.root / "0123456789abcdef"
        old.mkdir()
        (old / "job.json").write_text('{"schema_version":1}')
        malformed = runs.root / "run-20261005T000000Z-12345678"
        malformed.mkdir()
        (malformed / "run.json").write_text("{")
        result = runs.list()
        self.assertEqual(result["runs"], [])
        self.assertEqual(len(result["errors"]), 1)
        self.assertTrue((old / "job.json").is_file())

    def test_discovery_does_not_resume_or_rewrite(self):
        before = {p.relative_to(self.directory).as_posix(): p.read_bytes() for p in self.directory.rglob("*") if p.is_file()}
        runs = RunStore(self.store.root)
        self.assertEqual(len(runs.list()["runs"]), 1)
        after = {p.relative_to(self.directory).as_posix(): p.read_bytes() for p in self.directory.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_process_tree_closes_descendants_after_launcher_exit(self):
        marker = self.root / "child-survived.fixture"
        child = "import pathlib,time; time.sleep(1.5); pathlib.Path(" + repr(str(marker)) + ").write_text('bad')"
        parent = "import subprocess,sys; subprocess.Popen([sys.executable,'-B','-c'," + repr(child) + "])"
        with (self.root / "tree.log").open("wb") as log:
            tree = ProcessTree([sys.executable, "-B", "-c", parent], self.root, log)
            try:
                self.assertEqual(tree.process.wait(timeout=10), 0)
            finally:
                tree.close()
        time.sleep(1.8)
        self.assertFalse(marker.exists())
        # The terminated descendant inherited stdout; close must release it.
        (self.root / "tree.log").unlink()



if __name__ == "__main__":
    unittest.main()
