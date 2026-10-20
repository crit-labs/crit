"""Measurement supervision with synthetic adapters and real child processes."""
import ctypes
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from test_run_records import ROOT
import test_run_records as run_records
from crit.process.ownership import RuntimeOwnership, process_identity
from crit.run import measure


class MeasureProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        env = patch.dict(os.environ, {"CRIT_RUNTIME_DIRECTORY": str(self.root / "runtime")})
        env.start()
        self.addCleanup(env.stop)

    def cleanup_helper(self, marker):
        if marker.exists():
            value = json.loads(marker.read_text())
            for pid in (value["parent"], value["pid"]):
                if process_identity(pid) is not None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        os.kill(pid, signal.SIGKILL)

    def job(self, mode):
        # The helper acknowledges startup itself. Its liveness after measurement
        # returns detects both a surviving child and handles inherited from it.
        marker = self.root / "helper.json"
        helper = ("import json,os,pathlib,time; pathlib.Path(" + repr(str(marker)) +
                  ").write_text(json.dumps({'pid':os.getpid(),'parent':os.getppid()})); time.sleep(60)")
        source = (ROOT / "eval/examples/mock_adapter.py").read_text(encoding="utf-8")
        source += (
            "\nimport subprocess, sys\n"
            "from pathlib import Path\n"
            f"subprocess.Popen([sys.executable, '-B', '-c', {helper!r}])\n"
            f"while not Path({str(marker)!r}).exists(): time.sleep(.01)\n"
        )
        if mode == "failed":
            source += "raise RuntimeError('synthetic adapter failure')\n"
        elif mode in {"timeout", "interrupt"}:
            source += "time.sleep(60)\n"
        adapter = self.root / "adapter.py"
        adapter.write_text(source, encoding="utf-8")
        # Timeout exercises the actual pinning grandchild; the other modes go
        # directly through the measurement worker.
        cases = [{"case_id": "one", "pin": None, "prompt": [1], "output_tokens": 1,
                  "warmup_runs": 0, "repetitions": 1}] if mode == "timeout" else []
        job = self.root / "job.json"
        job.write_text(json.dumps({"adapter": str(adapter), "model": "synthetic", "config": {},
                                   "context_tokens": 8, "cases": cases, "out": str(self.root / "result.json")}),
                       encoding="utf-8")
        return job, marker

    def run_mode(self, mode):
        job, marker = self.job(mode)
        deadline = datetime.now(timezone.utc) + timedelta(seconds=4 if mode == "timeout" else 20)
        original_wait = subprocess.Popen.wait
        interrupted = False

        def interrupt_once(process, *args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                end = time.monotonic() + 10
                while not marker.exists() and process.poll() is None and time.monotonic() < end:
                    time.sleep(.01)
                raise KeyboardInterrupt()
            return original_wait(process, *args, **kwargs)

        try:
            if mode == "interrupt":
                with patch.object(subprocess.Popen, "wait", interrupt_once):
                    with self.assertRaises(KeyboardInterrupt):
                        measure._run_child(job, self.root / "worker.log", deadline.isoformat())
            else:
                outcome, _, _ = measure._run_child(job, self.root / "worker.log", deadline.isoformat())
                self.assertEqual(outcome, {"timeout": "timed_out"}.get(mode, mode))
            self.assertTrue(marker.exists(), "helper never started; cleanup was not exercised")
            pid = json.loads(marker.read_text())["pid"]
            self.assertIsNone(process_identity(pid), "measurement returned with a live descendant")
            (self.root / "worker.log").unlink()
        finally:
            # Failed-before runs must not leave the deliberately leaked helper.
            self.cleanup_helper(marker)

    def test_success_retires_remaining_descendants(self):
        self.run_mode("completed")

    def test_failure_retires_remaining_descendants(self):
        self.run_mode("failed")

    def test_timeout_retires_pinning_grandchild_and_its_helper(self):
        self.run_mode("timeout")

    def test_interruption_retires_worker_and_descendants(self):
        self.run_mode("interrupt")

    def test_expired_reservation_never_launches(self):
        job, _ = self.job("completed")
        with patch("crit.process.tree.ProcessTree", side_effect=AssertionError("expired worker launched")):
            outcome, _, _ = measure._run_child(job, self.root / "worker.log",
                (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())
        self.assertEqual(outcome, "timed_out")

    def test_expiry_during_guardian_handshake_retires_without_dispatch(self):
        from crit.process.tree import ProcessTree
        job, marker = self.job("completed")
        identify = ProcessTree.identify_guardian

        def delayed(tree, *args):
            identify(tree, *args)
            time.sleep(.2)

        execution = measure._Execution()
        with patch.object(ProcessTree, "identify_guardian", delayed):
            with self.assertRaises(TimeoutError):
                measure._run_child(job, self.root / "worker.log",
                    (datetime.now(timezone.utc) + timedelta(seconds=.15)).isoformat(), execution)
        self.assertTrue(execution.cleanup_settled)
        self.assertFalse(marker.exists())
        self.assertFalse(RuntimeOwnership().path.exists())

    def test_owner_death_retires_workers_and_leaves_recoverable_ownership(self):
        job, marker = self.job("interrupt")
        command = (
            "from crit.run.measure import _run_child\n"
            f"_run_child({str(job)!r}, {str(self.root / 'worker.log')!r}, "
            f"{(datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()!r})\n"
        )
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in measure.SOURCE_FOLDERS))
        interpreter = sys.executable
        if os.name == "nt":
            interpreter = sys._base_executable
            env["__PYVENV_LAUNCHER__"] = sys.executable
        with (self.root / "supervisor.log").open("wb") as log:
            parent = subprocess.Popen([interpreter, "-B", "-c", command], env=env, stdout=log,
                                      stderr=subprocess.STDOUT)
            try:
                end = time.monotonic() + 10
                while not marker.exists() and parent.poll() is None and time.monotonic() < end:
                    time.sleep(.01)
                self.assertTrue(marker.exists(), "worker never started")
                pid = json.loads(marker.read_text())["pid"]
                parent.kill()  # No Python finally blocks run in this coordinator.
                parent.wait(timeout=10)
                end = time.monotonic() + 5
                while process_identity(pid) is not None and time.monotonic() < end:
                    time.sleep(.02)
                self.assertIsNone(process_identity(pid), "worker tree outlived its coordinator")
                owner = RuntimeOwnership()
                self.assertTrue(owner.path.exists(), "abrupt exit must retain recovery evidence")
                self.assertEqual(owner.recover()["status"], "recovered")
                self.assertFalse(owner.path.exists())
            finally:
                if parent.poll() is None:
                    parent.kill()
                parent.wait(timeout=10)
                self.cleanup_helper(marker)


class MeasureAttemptTests(run_records.RunFixture, unittest.TestCase):
    resolve = run_records.MeasureTests.resolve
    package = run_records.MeasureTests.package

    def setUp(self):
        super().setUp()
        env = patch.dict(os.environ, {"CRIT_ENGINE_LOCK": "off",
                                     "CRIT_RUNTIME_DIRECTORY": str(self.root / "runtime")})
        env.start()
        self.addCleanup(env.stop)
        self.candidate_id = self.candidate()["candidate_id"]
        self.package({"candidate_id": self.candidate_id})
        self.folder = self.directory / "candidates" / self.candidate_id

    def measure(self):
        return measure.measure(self.store, self.run_id, self.candidate_id, "mock_adapter.py")

    def result(self):
        return run_records.read_json(self.folder / "attempts/attempt-0001/result.json")

    def fake_result(self, job, log, deadline, *args):
        value = run_records.read_json(job)
        Path(value["out"]).write_text(self.output, encoding="utf-8")
        log.write_text("synthetic worker log", encoding="utf-8")
        return "completed", 0, None

    def test_setup_error_settles_without_a_child_or_log(self):
        with patch.object(measure, "write_json", side_effect=OSError("synthetic job write failure")):
            with self.assertRaisesRegex(OSError, "synthetic job write failure"):
                self.measure()
        result = self.result()
        self.assertEqual(result["outcome"], "launch_failed")
        self.assertTrue(result["cleanup_settled"])
        self.assertEqual(result["logs"], [])

    def test_waiting_for_engine_consumes_the_reservation_without_launching(self):
        from test_machine_lock import HOLD
        with patch.dict(os.environ, {"CRIT_ENGINE_LOCK": str(self.root / "engine.lock"),
                                     "CRIT_ENGINE_LEASE": "off", "CRIT_ENGINE_LOCK_HELD": ""}):
            holder = subprocess.Popen([sys.executable, "-B", "-c", HOLD, "3", str(ROOT / "core/python")],
                                      stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), "held")
                with patch.object(measure, "_run_child") as launch:
                    with self.assertRaises(TimeoutError):
                        measure.measure(self.store, self.run_id, self.candidate_id, "mock_adapter.py", seconds=.1)
                    launch.assert_not_called()
                result = self.result()
                self.assertEqual(result["outcome"], "timed_out")
                self.assertTrue(result["cleanup_settled"])
            finally:
                holder.terminate()
                holder.wait(timeout=10)
                holder.stdout.close()

    def test_uncertain_launch_failure_retains_unsettled_cleanup(self):
        with patch("crit.process.tree.ProcessTree", side_effect=OSError("synthetic launch failure")):
            with self.assertRaisesRegex(OSError, "synthetic launch failure"):
                self.measure()
        self.assertEqual(self.result()["outcome"], "launch_failed")
        self.assertFalse(self.result()["cleanup_settled"])

    def test_unreadable_result_settles_failed_attempt_and_retains_log(self):
        self.output = "{"
        with patch.object(measure, "_run_child", self.fake_result):
            with self.assertRaises(ValueError):
                self.measure()
        result = self.result()
        self.assertEqual(result["outcome"], "failed")
        self.assertTrue(result["cleanup_settled"])
        self.assertEqual(len(result["logs"]), 1)
        self.assertFalse(list((self.folder / "benchmarks").glob("*.json")))

    def test_interruption_is_settled_and_propagated(self):
        with patch.object(measure, "_run_child", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.measure()
        self.assertEqual(self.result()["outcome"], "interrupted")
        self.assertTrue(self.result()["cleanup_settled"])

    def test_failed_tree_retirement_is_never_recorded_as_settled(self):
        tree = Mock()
        tree.process.wait.return_value = 0
        tree.close.side_effect = RuntimeError("synthetic retirement failure")
        with patch("crit.process.tree.guarded_tree", return_value=tree):
            with self.assertRaisesRegex(RuntimeError, "synthetic retirement failure"):
                self.measure()
        result = self.result()
        self.assertEqual(result["outcome"], "failed")
        self.assertFalse(result["cleanup_settled"])
        self.assertIn("retirement failure", result["failure"])
        self.assertFalse(list((self.folder / "benchmarks").glob("*.json")))
        self.assertEqual(result["logs"], [])
        # An unretired worker can keep writing; this must not invalidate the
        # attempt's immutable evidence when it is validated or recovered later.
        (self.folder / "logs/measure-attempt-0001.log").write_text("later child output", encoding="utf-8")
        run_records.candidate_projection(self.directory, self.run, self.candidate_id)

    def test_failed_cleanup_blocks_followup_execution(self):
        tree = Mock()
        tree.process.wait.return_value = 0
        tree.close.side_effect = RuntimeError("synthetic retirement failure")
        with patch("crit.process.tree.guarded_tree", return_value=tree):
            with self.assertRaisesRegex(RuntimeError, "synthetic retirement failure"):
                self.measure()
        with patch("crit.process.tree.ProcessTree", side_effect=AssertionError("admitted after failed cleanup")):
            with self.assertRaisesRegex(ValueError, "unresolved cleanup"):
                self.measure()

    def test_late_completion_settles_as_timeout_and_publishes_no_benchmark(self):
        self.output = '{"pins": {}, "cases": {}}'

        def after_deadline():
            reservation = run_records.read_json(self.folder / "attempts/attempt-0001/reservation.json")
            return (datetime.fromisoformat(reservation["deadline_at"]) + timedelta(seconds=1)).isoformat()

        with patch.object(measure, "_run_child", self.fake_result), patch.object(measure, "now", after_deadline):
            with self.assertRaisesRegex(ValueError, "timed_out"):
                self.measure()
        self.assertEqual(self.result()["outcome"], "timed_out")
        self.assertTrue(self.result()["cleanup_settled"])
        self.assertFalse(list((self.folder / "benchmarks").glob("*.json")))


class MeasureWorkerTests(unittest.TestCase):
    def test_sampler_is_stopped_after_engine_load_or_close_failure(self):
        for phase in ("load", "close"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "job.json"
                path.write_text(json.dumps({"adapter": "synthetic", "model": "synthetic", "config": {},
                    "context_tokens": 8, "cases": [], "out": str(Path(tmp) / "result.json")}), encoding="utf-8")
                engine, sampler = Mock(), Mock()
                error = RuntimeError("synthetic " + phase + " failure")
                engine.close.side_effect = error if phase == "close" else None
                with patch.object(measure, "_no_core_files"), patch.object(measure, "Sampler", return_value=sampler), \
                        patch("crit_eval.runner.load_adapter"), patch("crit_eval.runner.open_engine",
                        side_effect=error if phase == "load" else None,
                        return_value=(engine, {"name": "synthetic"})):
                    with self.assertRaisesRegex(RuntimeError, "synthetic " + phase):
                        measure.worker(path)
                sampler.stop.assert_called_once_with()
                self.assertFalse((Path(tmp) / "result.json").exists())


class MeasureCliTests(run_records.RunFixture, unittest.TestCase):
    resolve = run_records.MeasureTests.resolve

    def test_cli_interrupt_retires_pinning_and_settles_attempt(self):
        signals = ("CTRL_C_EVENT",) if os.name == "nt" else (signal.SIGINT, signal.SIGTERM)
        for signum in signals:
            with self.subTest(signal=signum):
                candidate = self.candidate()
                marker = self.root / (candidate["candidate_id"] + ".json")
                package = self.root / candidate["candidate_id"]
                package.mkdir()
                (package / "adapter.py").write_text(
                    "import json,os,time\nfrom pathlib import Path\n"
                    "def create_eval_engine(context):\n"
                    f"    Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid()}}))\n"
                    "    time.sleep(60)\n", encoding="utf-8")
                self.store.package(self.run_id, candidate["candidate_id"], package,
                    {"build": {"toolchain": {"identity": "host fixture"}, "dependencies": [], "command": ["fixture"]},
                     "launch": ["adapter.py"], "capabilities": ["prefill", "decode"]})
                options = {}
                if os.name == "nt":
                    # Children inherit an ignored Ctrl+C even with a new console, and CI
                    # runners start tests that way, so re-enable it before the launch.
                    ctypes.WinDLL("kernel32").SetConsoleCtrlHandler(None, False)
                    startup = subprocess.STARTUPINFO()
                    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startup.wShowWindow = subprocess.SW_HIDE
                    options.update(creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
                runtime = self.root / "runtime"
                env = dict(os.environ, CRIT_ENGINE_LOCK="off", CRIT_RUNTIME_DIRECTORY=str(runtime))
                with (self.root / "cli.log").open("wb") as log:
                    process = subprocess.Popen([sys.executable, "-B", str(ROOT / "tests/run_crit.py"), "run",
                        "--root", str(self.store.root), "--id", self.run_id, "measure",
                        "--candidate", candidate["candidate_id"], "--adapter", "adapter.py"],
                        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options)
                    try:
                        end = time.monotonic() + 15
                        while not marker.exists() and process.poll() is None and time.monotonic() < end:
                            time.sleep(.01)
                        self.assertTrue(marker.exists(), (self.root / "cli.log").read_text())
                        pid = json.loads(marker.read_text())["pid"]
                        if os.name == "nt":
                            helper = """
import ctypes,sys,time
api = ctypes.WinDLL("kernel32", use_last_error=True)
api.FreeConsole()
if not api.AttachConsole(int(sys.argv[1])):
    raise ctypes.WinError(ctypes.get_last_error())
try:
    if not api.SetConsoleCtrlHandler(None, True) or not api.GenerateConsoleCtrlEvent(0, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    time.sleep(.2)
finally:
    api.FreeConsole()
"""
                            sent = subprocess.run([sys.executable, "-B", "-c", helper, str(process.pid)],
                                creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True, text=True, timeout=5)
                            self.assertEqual(sent.returncode, 0, sent.stderr)
                        else:
                            process.send_signal(signum)
                        self.assertNotEqual(process.wait(timeout=15), 0)
                        result = run_records.read_json(self.directory / "candidates" / candidate["candidate_id"] /
                                                       "attempts/attempt-0001/result.json")
                        self.assertEqual(result["outcome"], "interrupted")
                        self.assertTrue(result["cleanup_settled"])
                        self.assertIsNone(process_identity(pid))
                        self.assertFalse((runtime / "owner.json").exists())
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
