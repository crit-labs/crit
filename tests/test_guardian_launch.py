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
for _folder in ("core/python", "api", "../crit-eval", "kernels", "cli"): sys.path.insert(0, str(ROOT / _folder))
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
import json, os, sys
from pathlib import Path
sys.path[:0] = sys.argv[1].split(os.pathsep)
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
            # crit as installed: the CLI package with Core's Python package (a dependency of crit) beside it.
            completed = subprocess.run([str(python), "-B", "-c", client,
                                        os.pathsep.join((str(ROOT / "cli"), str(ROOT / "core" / "python"))), str(root)],
                capture_output=True, text=True, timeout=25)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr +
                             (root / "engine.log").read_text(encoding="utf-8"))
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(Path(result["prefix"]), environment)
            self.assertEqual(result["dependency"], "venv dependency")


def store_python():
    """A Microsoft Store Python's interpreter, the one running the tests or the one on PATH, else None. Its base
    prefix is a packaged app's install folder under WindowsApps."""
    if os.name != "nt":
        return None
    if "\\windowsapps\\" in sys.base_prefix.lower():
        return getattr(sys, "_base_executable", sys.executable)
    alias = Path(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps", "python.exe")
    if not alias.exists():
        return None
    try:  # The alias may be the stub that opens the Store instead of a Python.
        found = subprocess.run([str(alias), "-I", "-c", "import sys; print(sys.base_prefix); print(sys.executable)"],
                               capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = found.stdout.splitlines()
    return lines[1] if found.returncode == 0 and len(lines) == 2 and "\\windowsapps\\" in lines[0].lower() else None


class StorePythonJobTests(unittest.TestCase):
    """crit-labs/crit#3: on a Microsoft Store Python, Windows starts a packaged interpreter's children, and the
    interpreter a venv's launcher starts, outside a job unless the process tree was created with the desktop app
    policy."""

    @unittest.skipUnless(os.name == "nt", "Windows venv launcher")
    def test_tree_starts_its_own_venv_python_as_the_base_interpreter(self):
        with patch.object(ProcessTree, "_windows_start") as start, \
                patch.object(sys, "executable", r"C:\venv\Scripts\python.exe"), \
                patch.object(sys, "_base_executable", r"C:\base\python.exe", create=True):
            ProcessTree([sys.executable, "-B", "engine.py"], ".", None, env={"KEPT": "1"})
            ProcessTree([r"C:\other\python.exe", "-B"], ".", None, env={"KEPT": "1"})
        (venv_command, _, _, venv_env, _), (other_command, _, _, other_env, _) = \
            [call.args for call in start.call_args_list]
        self.assertEqual(venv_command, [r"C:\base\python.exe", "-B", "engine.py"])
        self.assertEqual(venv_env, {"KEPT": "1", "__PYVENV_LAUNCHER__": r"C:\venv\Scripts\python.exe"})
        self.assertEqual((other_command, other_env), ([r"C:\other\python.exe", "-B"], {"KEPT": "1"}))

    def test_engine_through_store_venv_python_stays_in_the_job(self):
        """A guarded engine command that runs a Store venv's python.exe, as crit serve's host does: the engine's
        interpreter and the worker it starts (multiprocessing, through the base interpreter) are members of the
        tree's job, and closing the tree ends them."""
        python = store_python()
        if python is None:
            self.skipTest("needs a Microsoft Store Python on Windows")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            environment = root / "venv"
            subprocess.run([python, "-I", "-m", "venv", "--without-pip", str(environment)], check=True,
                           capture_output=True, timeout=120, stdin=subprocess.DEVNULL)
            venv_python = environment / "Scripts" / "python.exe"
            engine = root / "engine.py"
            engine.write_text(r'''
import json, multiprocessing, os, sys, time
from pathlib import Path
if __name__ == "__main__":
    worker = multiprocessing.Process(target=time.sleep, args=(60,))
    worker.start()
    Path(sys.argv[1]).write_text(json.dumps({"engine": os.getpid(), "worker": worker.pid, "prefix": sys.prefix}))
    time.sleep(60)
''', encoding="utf-8")
            client = r'''
import ctypes as c, json, os, sys, time
from ctypes import wintypes as w
from pathlib import Path
sys.path[:0] = sys.argv[1].split(os.pathsep)
from crit.process.ownership import RuntimeOwnership
from crit.process.tree import guarded_tree
root, venv_python = Path(sys.argv[2]), sys.argv[3]
kernel = c.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
kernel.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
kernel.WaitForSingleObject.argtypes, kernel.WaitForSingleObject.restype = [w.HANDLE, w.DWORD], w.DWORD
kernel.CloseHandle.argtypes = [w.HANDLE]
kernel.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
owner = RuntimeOwnership(root / "owner")
owner.acquire()
tree, handles, report = None, {}, {}
try:
    with (root / "engine.log").open("wb") as log:
        pids = root / "pids.json"
        tree = guarded_tree([venv_python, "-B", str(root / "engine.py"), str(pids)], root, log, owner, root)
        end = time.monotonic() + 30
        while not pids.exists():
            assert tree.process.poll() is None and time.monotonic() < end, "engine reported no processes"
            time.sleep(.05)
        report = json.loads(pids.read_text())
        member = {}
        for name in ("engine", "worker"):
            # SYNCHRONIZE | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION
            handle = kernel.OpenProcess(0x00100000 | 0x1 | 0x1000, False, report[name])
            assert handle, name + " is not running"
            handles[name] = handle
            inside = w.BOOL()
            assert kernel.IsProcessInJob(handle, tree.job, c.byref(inside))
            member[name] = bool(inside.value)
        report["in_job"] = member
        tree.close()
        report["ended"] = {name: kernel.WaitForSingleObject(handle, 5000) == 0 for name, handle in handles.items()}
finally:
    if tree: tree.close()
    owner.release(settled=True)
    for handle in handles.values():
        if kernel.TerminateProcess(handle, 1):  # an escaped process the job did not end
            kernel.WaitForSingleObject(handle, 5000)
        kernel.CloseHandle(handle)
print(json.dumps(report))
'''
            # The supervisor runs in the venv too, as an installed crit does: its guardian is the base interpreter.
            completed = subprocess.run([str(venv_python), "-B", "-c", client,
                                        os.pathsep.join((str(ROOT / "cli"), str(ROOT / "core" / "python"))),
                                        str(root), str(venv_python)],
                                       capture_output=True, text=True, timeout=90, stdin=subprocess.DEVNULL)
            log = (root / "engine.log").read_text(encoding="utf-8", errors="replace") \
                if (root / "engine.log").exists() else ""
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr + log)
            report = json.loads(completed.stdout.splitlines()[-1])
            self.assertEqual(Path(report["prefix"]), environment)
            self.assertEqual(report["in_job"], {"engine": True, "worker": True}, log)
            self.assertEqual(report["ended"], {"engine": True, "worker": True}, log)


if __name__ == "__main__":
    unittest.main()
