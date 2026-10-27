"""crit run measure: the run's cases measured on a packaged candidate through crit serve, under a benchmark
reservation."""
import contextlib
import ctypes
import io
import json
import os
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import test_run_records as run_records
from test_run_records import ROOT
from crit.process.ownership import process_identity
from crit.run import measure
from crit.run.records import file_sha, read_json

# A resident chat adapter whose prompt is one token per character of its messages and whose greedy reply is one "x"
# per token, three tokens per decode round. A message starting "early" ends after 40 tokens, one starting "hold"
# decodes until it is cancelled. Every request writes the host's process id and the sampling it got next to the model.
ADAPTER = r'''
import json
import os
import time
from pathlib import Path

class Engine:
    def __init__(self, context):
        self.weights = context.host('model.fixture')

    def generate(self, request, cancellation):
        return self.generate_stream(request, cancellation, lambda piece: None)

    def generate_stream(self, request, cancellation, emit):
        Path(request['model_path']).with_name('host.pid').write_text(str(os.getpid()))
        Path(request['model_path']).with_name('sampling.json').write_text(json.dumps(request.get('sampling')))
        content = request['messages'][-1]['content']
        n = request['limits']['max_output_tokens']
        n = min(n, 40) if content.startswith('early') else n
        for i in range(1 << 30 if content.startswith('hold') else n):
            cancellation.check()
            time.sleep(.002)
            emit('x')
        prompt = sum(len(m['content']) for m in request['messages'])
        return dict(schema_version=1, record_kind='chat_response',
                    **{k: request[k] for k in ('request_id', 'candidate_id', 'package_sha256')}, assistant='x' * n,
                    telemetry=dict(prompt_tokens=prompt, output_tokens=n, prefill_tokens=prompt, prefill_ms=1,
                                   decode_tokens=n - 1, decode_ms=2 * n, ttft_ms=1, missing={},
                                   decode_rounds=-(-(n - 1) // 3)))

    def reset(self):
        pass

    def close(self):
        pass

def create_engine(context):
    return Engine(context)
'''
SERVED = ["prefill", "decode", "chat-json-v1", "chat-resident-v1", "chat-stream-v1", "chat-sampling-temperature-v1"]


class MeasureFixture(run_records.RunFixture):
    content = "x" * 512  # the case's message: the workload's 512 prompt tokens

    def setUp(self):
        super().setUp()
        self.runtime = self.root / "runtime"
        environment = patch.dict(os.environ, {"CRIT_ENGINE_LOCK": "off", "CRIT_RUNTIME_DIRECTORY": str(self.runtime)})
        environment.start()
        self.addCleanup(environment.stop)
        self.case = {"messages": [{"role": "user", "content": self.content}]}

    def resolve(self):
        path = self.directory / "captured/case-0001/request.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.case), encoding="utf-8")
        self.resolved = self.store.freeze_inputs(self.run_id, {"hardware_manifest": {"device": "host fixture"},
            "environment_manifest": {"benchmark": "synthetic"}, "case_manifests": {"case-0001": [
                {"path": "captured/case-0001/request.json", "bytes": path.stat().st_size, "sha256": file_sha(path)}]}})

    def served(self, adapter=ADAPTER, capabilities=SERVED):
        candidate = self.candidate()["candidate_id"]
        package = self.root / ("package-" + candidate)
        package.mkdir()
        (package / "engine.py").write_text(adapter, encoding="utf-8")
        self.store.package(self.run_id, candidate, package,
                           {"build": {"toolchain": {"fixture": "host-python"}, "dependencies": [], "command": ["fixture"]},
                            "launch": ["engine.py"], "capabilities": capabilities})
        self.folder = self.directory / "candidates" / candidate
        return candidate

    def attempt(self):
        return read_json(self.folder / "attempts/attempt-0001/result.json")

    def benchmarks(self):
        return list((self.folder / "benchmarks").glob("*.json"))


class RunMeasureTests(MeasureFixture, unittest.TestCase):
    def test_measure_serves_the_candidate_and_records_one_benchmark_per_case(self):
        candidate = self.served()
        pids, sample = set(), measure.process_rss

        def rss(pid):
            pids.add(pid)
            return sample(pid)
        with patch.object(measure, "process_rss", rss):
            (record,) = measure.measure(self.store, self.run_id, candidate)
        workload = self.run["spec"]["cases"][0]["workload"]
        for key in ("prefill_tps", "decode_tps", "request_ms", "ttft_ms", "load_ms", "accepted_tokens_per_round"):
            self.assertEqual(len(record["metrics"][key]["samples"]), workload["repetitions"], key)
        # crit serve's telemetry: 128 output tokens over 43 decode rounds.
        self.assertEqual(record["metrics"]["accepted_tokens_per_round"]["samples"], [128 / 43] * 3)
        self.assertEqual(record["extensions"], {"measured_by": "crit run measure", "served_by": "crit serve",
                                                "attempt_id": "attempt-0001", "output_tokens": [128] * 3,
                                                "sampling": {"temperature": 0}})
        for key in ("peak_vram_bytes", "minimum_free_vram_bytes", "peak_ram_bytes", "minimum_free_ram_bytes"):
            self.assertTrue(key in record["metrics"] or key in record["missing_metrics"], key)
        if sys.platform.startswith("linux") or os.name == "nt":
            self.assertIn("peak_ram_bytes", record["metrics"])
        # Memory is the resident host's, where the engine runs, not this process's.
        host = int((self.root / "host.pid").read_text())
        self.assertNotEqual(host, os.getpid())
        self.assertEqual(pids, {host})
        attempt = self.attempt()
        self.assertEqual((attempt["outcome"], attempt["cleanup_settled"]), ("completed", True))
        log = attempt["logs"][0]
        self.assertEqual(log["path"], f"candidates/{candidate}/logs/measure-attempt-0001.log")
        logged = (self.directory / log["path"]).read_text()
        self.assertIn(f"crit serve: loading {candidate} at 8,192 tokens of context", logged)
        self.assertIn("case-0001 rep 3: ttft", logged)
        self.assertTrue(logged.endswith("crit serve: stopped\n"))
        self.assertFalse((self.runtime / "owner.json").exists())
        view = self.store.get(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertEqual([s["benchmark_id"] for s in view["candidates"][0]["series"]], [record["benchmark_id"]])

    def test_a_captured_requests_sampling_is_sent_and_recorded(self):
        self.case.update(temperature=0.7, top_p=0.9, seed=3)
        candidate = self.served(capabilities=SERVED + ["chat-sampling-top-p-v1", "chat-sampling-seed-v1"])
        (record,) = measure.measure(self.store, self.run_id, candidate)
        self.assertEqual(record["extensions"]["sampling"], {"temperature": 0.7, "top_p": 0.9, "seed": 3})
        self.assertEqual(json.loads((self.root / "sampling.json").read_text()),
                         {"temperature": 0.7, "top_p": 0.9, "seed": 3})

    def test_a_captured_requests_template_variables_are_sent_and_recorded(self):
        self.case.update(chat_template_kwargs={"enable_thinking": True})
        adapter = ADAPTER.replace("Path(request['model_path']).with_name('sampling.json').write_text(json.dumps(request.get('sampling')))",
                                  "Path(request['model_path']).with_name('template.json').write_text(json.dumps(request.get('template_kwargs')))")
        candidate = self.served(adapter, SERVED + ["chat-template-kwargs-v1", "chat-template-enable-thinking-v1"])
        (record,) = measure.measure(self.store, self.run_id, candidate)
        self.assertEqual(record["extensions"]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(json.loads((self.root / "template.json").read_text()), {"enable_thinking": True})

    def test_a_package_measure_cannot_serve_or_a_sealed_candidate_is_refused_before_any_reservation(self):
        candidate = self.served(capabilities=[c for c in SERVED if c != "chat-stream-v1"])
        with self.assertRaisesRegex(ValueError, "must declare chat-stream-v1"):
            measure.measure(self.store, self.run_id, candidate)
        self.store.seal(self.run_id, candidate, "built")
        with self.assertRaisesRegex(ValueError, "Sealed candidate is immutable"):
            measure.measure(self.store, self.run_id, candidate)
        self.assertFalse((self.folder / "attempts").exists())

    def test_the_deadline_stops_the_served_engine_and_settles_a_timeout(self):
        self.case["messages"][0]["content"] = "hold" + "x" * 508
        candidate = self.served()
        with self.assertRaisesRegex(ValueError, "Measurement timed_out: The measurement passed its reserved deadline"):
            measure.measure(self.store, self.run_id, candidate, seconds=3)
        self.assertEqual((self.attempt()["outcome"], self.attempt()["cleanup_settled"]), ("timed_out", True))
        self.assertEqual(self.benchmarks(), [])
        self.assertFalse((self.runtime / "owner.json").exists())

    def test_waiting_for_the_engine_lock_consumes_the_reservation_without_serving(self):
        from test_machine_lock import HOLD
        candidate = self.served()
        with patch.dict(os.environ, {"CRIT_ENGINE_LOCK": str(self.root / "engine.lock"),
                                     "CRIT_ENGINE_LEASE": "off", "CRIT_ENGINE_LOCK_HELD": ""}):
            holder = subprocess.Popen([sys.executable, "-B", "-c", HOLD, "3", str(ROOT / "core/python")],
                                      stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), "held")
                with patch("crit_api.engine.ServedEngine.open", side_effect=AssertionError("served while waiting")):
                    with self.assertRaisesRegex(ValueError, "Measurement timed_out: Engine lock acquisition"):
                        measure.measure(self.store, self.run_id, candidate, seconds=.1)
            finally:
                holder.terminate()
                holder.wait(timeout=10)
                holder.stdout.close()
        self.assertEqual((self.attempt()["outcome"], self.attempt()["cleanup_settled"]), ("timed_out", True))

    def test_an_interruption_is_settled_and_propagated(self):
        candidate = self.served()
        with patch.object(measure, "_request", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                measure.measure(self.store, self.run_id, candidate)
        self.assertEqual((self.attempt()["outcome"], self.attempt()["cleanup_settled"]), ("interrupted", True))
        self.assertFalse((self.runtime / "owner.json").exists())

    def test_a_failed_engine_cleanup_is_never_recorded_as_settled(self):
        from crit_api.engine import ServedEngine
        candidate = self.served()
        close = ServedEngine.close

        def failing(engine):
            close(engine)
            raise RuntimeError("synthetic cleanup failure")
        with patch.object(ServedEngine, "close", failing):
            with self.assertRaisesRegex(ValueError, "Measurement failed: .*synthetic cleanup failure"):
                measure.measure(self.store, self.run_id, candidate)
        attempt = self.attempt()
        self.assertEqual((attempt["outcome"], attempt["cleanup_settled"], attempt["logs"]), ("failed", False, []))
        self.assertEqual(self.benchmarks(), [])
        self.assertTrue((self.folder / "logs/measure-attempt-0001.log").is_file())

    def test_a_late_completion_settles_as_a_timeout_and_records_no_benchmark(self):
        candidate = self.served()

        def after_deadline():
            reservation = read_json(self.folder / "attempts/attempt-0001/reservation.json")
            return (run_records.timestamp_for_test(reservation["deadline_at"])
                    + run_records.timedelta(seconds=1)).isoformat()
        with patch.object(measure, "now", after_deadline):
            with self.assertRaisesRegex(ValueError, "timed_out"):
                measure.measure(self.store, self.run_id, candidate)
        self.assertEqual((self.attempt()["outcome"], self.attempt()["cleanup_settled"]), ("timed_out", True))
        self.assertEqual(self.benchmarks(), [])

    def test_the_command_prints_the_recorded_benchmarks(self):
        from crit.run.cli import main
        candidate = self.served()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--root", str(self.store.root), "--id", self.run_id, "measure", "--candidate", candidate])
        self.assertEqual(code, 0)
        (printed,) = json.loads(out.getvalue())
        self.assertEqual(printed["extensions"]["served_by"], "crit serve")


class EarlyReplyTests(MeasureFixture, unittest.TestCase):
    content = "early" + "x" * 507

    def test_a_reply_that_ends_early_is_measured_on_its_tokens(self):
        candidate = self.served()
        (record,) = measure.measure(self.store, self.run_id, candidate)
        self.assertEqual(record["extensions"]["output_tokens"], [40] * 3)
        self.assertEqual(record["metrics"]["accepted_tokens_per_round"]["samples"], [40 / 13] * 3)
        self.assertEqual(len(record["metrics"]["decode_tps"]["samples"]), 3)
        logged = (self.folder / "logs/measure-attempt-0001.log").read_text()
        self.assertIn("the reply ended after 40 of 128 tokens", logged)


class PromptMismatchTests(MeasureFixture, unittest.TestCase):
    content = "x" * 500

    def test_a_prompt_crit_serve_counts_differently_fails_naming_both_counts(self):
        candidate = self.served()
        with self.assertRaisesRegex(ValueError, "Measurement failed: ValueError: case-0001: crit serve rendered the "
                                                "case's messages as 500 prompt tokens, the workload has 512"):
            measure.measure(self.store, self.run_id, candidate)
        self.assertEqual((self.attempt()["outcome"], self.attempt()["cleanup_settled"]), ("failed", True))
        self.assertEqual(self.benchmarks(), [])


class UncapturedCaseTests(MeasureFixture, unittest.TestCase):
    def test_a_case_without_a_chat_request_is_refused_before_any_reservation(self):
        self.case = {"ids": list(range(512))}
        candidate = self.served()
        with self.assertRaisesRegex(ValueError, "No retained chat request for the case"):
            measure.measure(self.store, self.run_id, candidate)
        self.assertFalse((self.folder / "attempts").exists())


class HostMemoryTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux") or os.name == "nt", "no process memory reader here")
    def test_another_process_resident_memory_is_read_until_it_exits(self):
        child = subprocess.Popen([sys.executable, "-B", "-c", "import sys; print(flush=True); sys.stdin.read()"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            child.stdout.readline()  # the interpreter has started
            self.assertGreater(measure.process_rss(child.pid), 1 << 20)
        finally:
            child.stdin.close()
            child.wait(timeout=10)
            child.stdout.close()
        self.assertIsNone(measure.process_rss(child.pid))  # Popen still holds its handle to the exited child


class MeasureCliTests(MeasureFixture, unittest.TestCase):
    def test_cli_interrupt_retires_the_resident_host_and_settles_the_attempt(self):
        signals = ("CTRL_C_EVENT",) if os.name == "nt" else (signal.SIGINT, signal.SIGTERM)
        for signum in signals:
            with self.subTest(signal=signum):
                marker = self.root / f"loading-{len(list(self.root.glob('loading-*')))}.json"
                # The engine never finishes loading: the command is interrupted while crit serve loads it.
                candidate = self.served("import json, os, time\nfrom pathlib import Path\n"
                                        "def create_engine(context):\n"
                                        f"    Path({str(marker)!r}).write_text(json.dumps({{'pid': os.getpid()}}))\n"
                                        "    time.sleep(60)\n")
                options = {}
                if os.name == "nt":
                    # Children inherit an ignored Ctrl+C even with a new console, and CI
                    # runners start tests that way, so re-enable it before the launch.
                    ctypes.WinDLL("kernel32").SetConsoleCtrlHandler(None, False)
                    startup = subprocess.STARTUPINFO()
                    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startup.wShowWindow = subprocess.SW_HIDE
                    options.update(creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
                with (self.root / "cli.log").open("wb") as log:
                    process = subprocess.Popen([sys.executable, "-B", str(ROOT / "tests/run_crit.py"), "run",
                        "--root", str(self.store.root), "--id", self.run_id, "measure", "--candidate", candidate],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options)
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
                        self.assertEqual((self.attempt()["outcome"], self.attempt()["cleanup_settled"]),
                                         ("interrupted", True))
                        self.assertIsNone(process_identity(pid))
                        self.assertFalse((self.runtime / "owner.json").exists())
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
