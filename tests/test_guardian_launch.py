"""Launch identity and real venv integration; only synthetic child processes run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv
import io
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "eval", "kernels", "cli"): sys.path.insert(0, str(ROOT / _folder))
from crit.process.ownership import process_identity
from crit.process.tree import ProcessTree
from crit.process import guardian


class GuardianOwnershipTests(unittest.TestCase):
    def run_guardian(self, reads, clock=None):
        launch = {'owner_path': 'owner.json', 'token': 'token', 'command': ['fixture'], 'cwd': '.'}
        engine = Mock(returncode=0)
        engine.poll.side_effect = [None, 0]
        diagnostic = io.StringIO()
        with patch.object(guardian, 'read_json', return_value=launch), \
                patch('crit.process.ownership.read_json', side_effect=reads), \
                patch.object(guardian, 'process_identity', side_effect=lambda pid: 'self' if pid == os.getpid() else 'owner'), \
                patch.object(guardian, 'write_json'), patch.object(guardian.sys, 'argv', ['guardian', 'launch.json']), \
                patch.object(guardian.subprocess, 'Popen', return_value=engine), \
                patch.object(guardian.time, 'sleep'), \
                patch.object(guardian.time, 'monotonic', side_effect=clock, return_value=0), \
                patch.object(guardian.os, 'killpg', create=True), patch.object(guardian.os, 'getpgrp', return_value=1, create=True), \
                patch.object(guardian.sys, 'stderr', diagnostic):
            result = guardian.main()
        return result, engine, diagnostic.getvalue()

    def owner(self):
        return {'token': 'token', 'owner_pid': -1, 'owner_identity': 'owner',
                'process': {'pid': os.getpid(), 'identity': 'self'}}

    def test_transient_denial_at_preflight_handshake_and_monitor_does_not_kill_engine(self):
        result, engine, _ = self.run_guardian([PermissionError('sharing'), self.owner()] * 3)
        self.assertEqual(result, 0)
        engine.kill.assert_not_called()

    def test_persistent_monitor_denial_still_kills_engine_after_bounded_retry(self):
        result, engine, diagnostic = self.run_guardian([self.owner(), self.owner(), PermissionError('persistent')],
                                                      clock=[0, 0, 0, 0, .3])
        self.assertEqual(result, 125)
        engine.kill.assert_called_once()
        engine.wait.assert_called_once_with(timeout=5)
        self.assertIn('persistent', diagnostic)

    def test_retry_never_reuses_old_owner_when_token_changes(self):
        changed = dict(self.owner(), token='other')
        result, engine, _ = self.run_guardian([self.owner(), self.owner(), PermissionError('sharing'), changed])
        self.assertEqual(result, 125)
        engine.kill.assert_called_once()


class GuardianLaunchTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows process-handle lifetime")
    def test_close_settles_inherited_log_handles_without_garbage_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(20):
                path = root / (str(index) + ".log")
                with path.open("wb") as log:
                    tree = ProcessTree([sys.executable, "-B", "-c", "import time; time.sleep(30)"], root, log)
                    tree.close()
                # Keep tree/Popen alive: explicit retirement must suffice.
                path.unlink()
                self.assertTrue(tree.closed)
                self.assertIsNotNone(tree.process.poll())
                tree.close()

    def test_unrelated_or_changed_process_cannot_be_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "child.log").open("wb") as log:
                tree = ProcessTree([sys.executable, "-B", "-c", "import time; time.sleep(30)"], root, log)
                try:
                    with self.assertRaisesRegex(ValueError, "outside the owned"):
                        tree.identify_guardian(os.getpid(), process_identity(os.getpid()))
                    with self.assertRaisesRegex(ValueError, "identity"):
                        tree.identify_guardian(tree.process.pid, "wrong birth")
                    self.assertIsNone(tree.guardian)
                finally:
                    tree.close()

    def test_virtualenv_guardian_preserves_engine_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "venv"
            venv.EnvBuilder(with_pip=False).create(environment)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            site = subprocess.check_output([str(python), "-B", "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True, timeout=10).strip()
            (Path(site) / "crit_venv_fixture.py").write_text("value = 'venv dependency'\n", encoding="utf-8")
            output = root / "result.json"
            client = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from crit.process.ownership import RuntimeOwnership
from crit.process.tree import guarded_tree
root = Path(sys.argv[2])
owner = RuntimeOwnership(root / 'owner')
owner.acquire()
tree = None
try:
    with (root / 'engine.log').open('wb') as log:
        command = [sys.executable, '-B', '-c',
            "import json,sys,crit_venv_fixture; from pathlib import Path; "
            "Path(sys.argv[1]).write_text(json.dumps({'prefix':sys.prefix,'dependency':crit_venv_fixture.value}))",
            str(root / 'result.json')]
        tree = guarded_tree(command, root, log, owner, root)
        value = owner.read_owned()
        assert value['process']['pid'] == tree.guardian['pid']
        assert tree.process.wait(timeout=8) == 0
finally:
    if tree: tree.close()
    owner.release(settled=True)
'''
            completed = subprocess.run([str(python), "-B", "-c", client, str(ROOT / "cli"), str(root)],
                capture_output=True, text=True, timeout=25)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr +
                             (root / "engine.log").read_text(encoding="utf-8"))
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(Path(result["prefix"]), environment)
            self.assertEqual(result["dependency"], "venv dependency")


if __name__ == "__main__":
    unittest.main()
