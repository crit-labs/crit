"""crit run measure: the run's benchmark cases measured by crit on a candidate's package.

The candidate's Eval adapter (eval-v1, docs/eval/ADAPTER.md) runs in a child process under a benchmark
reservation. Every case decodes a text pinned once per run (the first measured candidate's plain greedy
output, stored under pins/), replayed through each candidate's speculative loop when the adapter declares
replay, so a candidate's decode speed does not depend on the text its own arithmetic happened to produce.
Host and device memory are sampled during every request. The parent settles the attempt and records one
benchmark per case, bound to the case's checks.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from .records import (ALL_METRICS, METRICS, digest, file_sha, ident, inputs, load_run, now, read_json, safe_path, timestamp,
                      write_json)

PINS = "pins"
ROOT = Path(__file__).resolve().parents[3]
SOURCE_FOLDERS = ("cli", "eval", "core/python", "api", "kernels")


@dataclass
class _Execution:
    outcome: str = "launch_failed"
    code: int | None = None
    failure: str | None = None
    cleanup_settled: bool = True  # No process has been attempted yet.


def case_prompt(root, rows):
    """A case's prompt token ids: its first retained JSON file holding {"ids": [token ids]}."""
    for row in rows:
        path = safe_path(root, row["path"])
        if path.suffix.lower() != ".json":
            continue
        try:
            value = read_json(path)
        except (ValueError, OSError):
            continue
        ids = value.get("ids") if isinstance(value, dict) else None
        if isinstance(ids, list) and ids and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in ids):
            return ids
    raise ValueError('No retained prompt for the case; keep a captured JSON file {"ids": [token ids]} in its manifest')


def read_pin(root, case_id):
    path = safe_path(root, f"{PINS}/{ident(case_id, 'case')}.json")
    if not path.is_file():
        return None
    pin = read_json(path)
    if pin.get("case_id") != case_id or pin.get("sha256") != digest(pin.get("tokens")):
        raise ValueError(f"{path}: pinned text does not match its sha256")
    return pin


def measure(store, run_id, candidate_id, adapter, model=None, config=None, seconds=None):
    """Reserve, measure every case in a child process, settle, then record one benchmark per case."""
    root = store.directory(run_id)
    run = load_run(root)
    resolved = inputs(root, run)
    folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
    if (folder / "seal.json").exists():
        raise ValueError("Sealed candidate is immutable; allocate a new candidate")
    manifest_path = folder / "package" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Package the candidate before measuring it")
    manifest = read_json(manifest_path)
    if not any(row["path"] == adapter for row in manifest["files"]):
        raise ValueError(f"{adapter} is not a file of the candidate's package")
    cases = []
    for case in run["spec"]["cases"]:
        workload = case["workload"]
        if workload["mode"] != "resident" or workload["concurrency"] != 1:
            raise ValueError(f"{case['case_id']}: crit run measure supports resident cases of concurrency 1")
        prompt = case_prompt(root, resolved["case_manifests"][case["case_id"]])
        if len(prompt) != workload["prompt_tokens"]:
            raise ValueError(f"{case['case_id']}: retained prompt has {len(prompt)} tokens, the workload {workload['prompt_tokens']}")
        pin = read_pin(root, case["case_id"])
        if pin and len(pin["tokens"]) != workload["output_tokens"]:
            raise ValueError(f"{case['case_id']}: pinned text has {len(pin['tokens'])} tokens, the workload {workload['output_tokens']}")
        cases.append({"case_id": case["case_id"], "prompt": prompt, "output_tokens": workload["output_tokens"],
                      "warmup_runs": workload["warmup_runs"], "repetitions": workload["repetitions"],
                      "pin": pin["tokens"] if pin else None})
    reservation = store.reserve(run_id, candidate_id, "benchmark", seconds)
    attempt = reservation["attempt_id"]
    log_path = folder / "logs" / f"measure-{attempt}.log"
    execution, result = _Execution(), None
    try:
        scratch = safe_path(root, f"work/measure/{candidate_id}-{attempt}")
        scratch.mkdir(parents=True, exist_ok=True)
        job = {"adapter": str(folder / "package" / adapter), "model": str(model or run["spec"]["model_path"]),
               "config": dict(config or {}), "context_tokens": run["spec"]["hardware_constraints"]["kv_tokens"],
               "cases": cases, "out": str(scratch / "result.json")}
        write_json(scratch / "job.json", job)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        from crit_core.machine import engine_lock
        deadline = time.monotonic() + _remaining(reservation["deadline_at"])
        with engine_lock(f"crit run measure {run_id} {candidate_id}", deadline=deadline):
            execution.outcome, execution.code, execution.failure = _run_child(
                scratch / "job.json", log_path, reservation["deadline_at"], execution)
        if execution.outcome == "completed":
            if not (scratch / "result.json").is_file():
                raise ValueError("The measurement wrote no result")
            result = read_json(scratch / "result.json")
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            execution.outcome = "interrupted"
        elif isinstance(exc, TimeoutError) and execution.outcome == "launch_failed":
            execution.outcome = "timed_out"
        elif execution.outcome == "completed":
            execution.outcome = "failed"
        execution.failure = (type(exc).__name__ + ": " + str(exc))[:8000]
        raise
    finally:
        ended = now()
        if execution.outcome == "completed" and timestamp(ended, "ended") > timestamp(reservation["deadline_at"], "deadline"):
            execution.outcome, execution.failure = "timed_out", "The measurement passed its reserved deadline"
        logs = []
        try:
            # Unretired descendants may still append to this file. Keep their
            # raw output, but never freeze a mutable log as retained evidence.
            if execution.cleanup_settled and log_path.is_file():
                logs.append({"path": log_path.relative_to(root).as_posix(), "sha256": file_sha(log_path)})
        except OSError as exc:
            if execution.outcome == "completed":
                execution.outcome = "failed"
            execution.failure = ((execution.failure or "") + f"; could not retain log: {exc}")[:8000]
        store.settle(run_id, reservation, {"outcome": execution.outcome, "ended_at": ended, "exit_code": execution.code,
            "cleanup_settled": execution.cleanup_settled, "logs": logs, "failure": execution.failure})
    if execution.outcome != "completed":
        raise ValueError(f"Measurement {execution.outcome}: {execution.failure} (log {log_path.relative_to(root).as_posix()})")
    for case_id, tokens in result["pins"].items():
        path = safe_path(root, f"{PINS}/{ident(case_id, 'case')}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_json(path, {"case_id": case_id, "tokens": tokens, "sha256": digest(tokens), "pinned_by": candidate_id,
                              "package_sha256": digest(manifest), "created_at": now()}, immutable=True)
        except FileExistsError:
            # Another candidate pinned this case meanwhile; this measurement replayed a different text.
            raise ValueError(f"{case_id} was pinned concurrently; measure this candidate again") from None
    recorded = []
    for case in run["spec"]["cases"]:
        measured = result["cases"][case["case_id"]]
        pin = read_pin(root, case["case_id"])
        metrics = {key: {"unit": ALL_METRICS[key], "samples": samples} for key, samples in measured["samples"].items()
                   if samples is not None}
        missing = {key: measured["missing"].get(key, "Not measured on this host") for key in METRICS if key not in metrics}
        text = ({"source": "pinned", "path": f"{PINS}/{case['case_id']}.json", "sha256": pin["sha256"], "pinned_by": pin["pinned_by"]}
                if measured["replayed"] else {"source": "generated", "reason": measured["text_reason"]})
        payload = {"case_id": case["case_id"], "package_sha256": digest(manifest), "case_sha256": digest(case),
                   "protocol_sha256": digest(run["spec"]["benchmark_protocol"]),
                   "identities": {"model_sha256": digest(resolved["model_manifest"]),
                                  "input_sha256": digest(resolved["case_manifests"][case["case_id"]]),
                                  "hardware_sha256": digest(resolved["hardware_manifest"]),
                                  "environment_sha256": digest(resolved["environment_manifest"])},
                   "workload": case["workload"], "metrics": metrics, "missing_metrics": missing, "checks": [],
                   "outcome": "completed", "started_at": measured["started_at"], "ended_at": measured["ended_at"],
                   "extensions": {"measured_by": "crit run measure", "adapter": adapter, "decode_text": text,
                                  "attempt_id": attempt}}
        recorded.append(store.append(run_id, candidate_id, "benchmark", payload))
    shutil.rmtree(scratch, ignore_errors=True)
    return recorded


def _remaining(deadline_at):
    return (timestamp(deadline_at, "deadline") - datetime.now(timezone.utc)).total_seconds()


def _run_child(job, log_path, deadline_at, execution=None):
    from crit.process.ownership import RuntimeOwnership
    from crit.process.tree import guarded_tree

    execution = execution if execution is not None else _Execution()
    command = [sys.executable, "-B", "-m", "crit.run.measure", str(job)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / f) for f in SOURCE_FOLDERS] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    remaining = _remaining(deadline_at)
    if remaining <= 0:
        return "timed_out", None, "The measurement passed its reserved deadline before launch"
    deadline = time.monotonic() + remaining
    ownership = RuntimeOwnership()
    ownership.acquire()
    try:
        with open(log_path, "wb") as log:
            # Admission and filesystem work can consume the remaining budget.
            if _remaining(deadline_at) <= 0:
                return "timed_out", None, "The measurement passed its reserved deadline before launch"
            execution.cleanup_settled = False
            try:
                tree = guarded_tree(command, Path.cwd(), log, ownership, Path(job).parent, env=env, deadline=deadline)
            except BaseException:
                execution.cleanup_settled = ownership.launch_settled
                raise
            execution.outcome = "failed"
            try:
                while True:
                    left = _remaining(deadline_at)
                    if left <= 0:
                        return "timed_out", None, "The measurement passed its reserved deadline"
                    try:
                        # Windows' native process wait can defer Python's Ctrl+C
                        # handler until it returns. Keep each wait bounded.
                        code = execution.code = tree.process.wait(timeout=min(.1, left))
                        break
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                # Includes the pinning process and adapter/sampler helpers, even
                # after normal exit. The guardian also handles coordinator death.
                tree.close()
                execution.cleanup_settled = True
    finally:
        if execution.cleanup_settled:
            ownership.release(settled=True)
        else:
            ownership.block("Measurement launch or tree retirement could not be proven settled")
    if code != 0:
        with log_path.open("rb") as log:
            log.seek(max(0, log.seek(0, os.SEEK_END) - 2000))
            tail = log.read().decode("utf-8", "replace").strip().splitlines()
        return "failed", code, "Measurement process exited with %d: %s" % (code, tail[-1] if tail else "no output")
    return "completed", 0, None


class Sampler:
    """Free host memory, this process's resident memory and free device memory, sampled during requests.
    Device memory is the driver's whole-GPU view (nvidia-smi, GPU 0), which counts other processes too; inside
    WSL2 the process's own cudaMemGetInfo disagrees with it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.min_free_ram = self.peak_rss = self.min_free_vram = self.total_vram = None
        self.smi = None
        if shutil.which("nvidia-smi"):
            self.smi = subprocess.Popen(["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total",
                                         "--format=csv,noheader,nounits", "-lms", "100"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._device, daemon=True).start()
        self.thread = threading.Thread(target=self._host, daemon=True)
        self.thread.start()

    def _device(self):
        for line in self.smi.stdout:
            try:
                used, total = (int(float(v)) * (1 << 20) for v in line.split(",")[:2])
            except ValueError:
                continue
            with self.lock:
                self.total_vram = total
                free = total - used
                self.min_free_vram = free if self.min_free_vram is None else min(self.min_free_vram, free)

    def _host(self):
        while self.running:
            self.sample()
            time.sleep(.02)

    def sample(self):
        available, rss = host_available(), process_rss()
        with self.lock:
            if available is not None:
                self.min_free_ram = available if self.min_free_ram is None else min(self.min_free_ram, available)
            if rss is not None:
                self.peak_rss = rss if self.peak_rss is None else max(self.peak_rss, rss)

    def reset(self):
        with self.lock:
            self.min_free_ram = self.peak_rss = self.min_free_vram = None
        self.sample()
        if self.smi:
            time.sleep(.15)  # at least one device reading inside the window

    def window(self):
        self.sample()
        with self.lock:
            return {"minimum_free_ram_bytes": self.min_free_ram, "peak_ram_bytes": self.peak_rss,
                    "minimum_free_vram_bytes": self.min_free_vram,
                    "peak_vram_bytes": None if self.min_free_vram is None else self.total_vram - self.min_free_vram}

    def stop(self):
        self.running = False
        self.thread.join()
        if self.smi:
            self.smi.kill()
            self.smi.wait()


def host_available():
    if sys.platform.startswith("linux"):
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        return None
    if os.name == "nt":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                        ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        return status.available if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
    return None


def process_rss():
    if sys.platform.startswith("linux"):
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD), ("peak_working_set", ctypes.c_size_t),
                        ("working_set", ctypes.c_size_t), ("quota_peak_paged", ctypes.c_size_t),
                        ("quota_paged", ctypes.c_size_t), ("quota_peak_nonpaged", ctypes.c_size_t),
                        ("quota_nonpaged", ctypes.c_size_t), ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return counters.working_set if ok else None
    return None


def _no_core_files():
    if os.name != "nt":
        import resource
        # No core file: a crash of a process holding tens of GB would write one that large (a soft limit of 1
        # byte also stops pipe-based core handlers, which ignore 0).
        resource.setrlimit(resource.RLIMIT_CORE, (1, resource.getrlimit(resource.RLIMIT_CORE)[1]))


def pin_worker(job_path):
    """Grandchild process: pin missing cases with plain greedy decoding on a non-speculative engine. It runs in its
    own process so everything that engine holds (host-resident weights, pinned memory, caches an adapter keeps for
    the next engine of the process) is gone before the speculative engine loads: two engines of a model whose
    weights live in host memory do not fit side by side."""
    _no_core_files()
    from crit_eval.runner import load_adapter, open_engine, timed_generate
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    module = load_adapter(job["adapter"])
    new_pins, unpinned = {}, {}
    engine, _ = open_engine(module, job["model"], job["config"], speculative=False, context_tokens=job["context_tokens"])
    try:
        for case in job["cases"]:
            if case["pin"] is not None:
                continue
            first, _ = timed_generate(engine, case["prompt"], case["output_tokens"])
            second, _ = timed_generate(engine, case["prompt"], case["output_tokens"])
            if first != second:
                # Like crit eval pin, only a repeatable text becomes the run's pin; the case is still measured
                # on the text the candidate generates.
                at = next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), min(len(first), len(second)))
                unpinned[case["case_id"]] = f"plain greedy output changed between two runs (from token {at}), so no text was pinned"
                print(f"{case['case_id']}: {unpinned[case['case_id']]}", flush=True)
                continue
            new_pins[case["case_id"]] = first
            print(f"pinned {case['case_id']}: {len(first)} tokens", flush=True)
    finally:
        engine.close()
    Path(job["out"] + ".pins.json").write_text(json.dumps({"pins": new_pins, "unpinned": unpinned}), encoding="utf-8")


def worker(job_path):
    """Child process: pin missing cases (in a grandchild process, pin_worker), then measure every case."""
    _no_core_files()
    from crit_eval.runner import load_adapter, open_engine, timed_generate, timed_replay
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    pins = {c["case_id"]: c["pin"] for c in job["cases"] if c["pin"] is not None}
    new_pins, unpinned = {}, {}
    if len(pins) < len(job["cases"]):
        sys.stdout.flush()
        done = subprocess.run([sys.executable, "-B", "-m", "crit.run.measure", str(job_path), "--pin"])
        pinned = Path(job["out"] + ".pins.json")
        if done.returncode or not pinned.is_file():
            raise SystemExit(f"pinning failed (exit code {done.returncode})")
        result = json.loads(pinned.read_text(encoding="utf-8"))
        new_pins, unpinned = result["pins"], result["unpinned"]
        pins.update(new_pins)
    module = load_adapter(job["adapter"])
    sampler = Sampler()
    engine = None
    try:
        started = time.perf_counter()
        engine, info = open_engine(module, job["model"], job["config"], speculative=True, context_tokens=job["context_tokens"])
        load_ms = (time.perf_counter() - started) * 1000
        print(f"loaded {info['name']} in {load_ms / 1000:.1f} s", flush=True)
        replay = bool(info.get("replay")) and bool(info.get("speculative")) and callable(getattr(engine, "replay", None))
        reason = ("" if replay else "the adapter does not declare replay" if info.get("speculative") else "ordinary decoding")
        cases = {}
        for case in job["cases"]:
            names = ("prefill_tps", "decode_tps", "request_ms", "ttft_ms", "load_ms", "peak_ram_bytes", "peak_vram_bytes",
                     "minimum_free_ram_bytes", "minimum_free_vram_bytes", "accepted_tokens_per_round")
            samples = {name: [] for name in names}
            begun = now()
            replayed = replay and case["case_id"] in pins
            text_reason = "" if replayed else unpinned.get(case["case_id"]) or reason
            for i in range(case["warmup_runs"] + case["repetitions"]):
                sampler.reset()
                if replayed:
                    _, sample = timed_replay(engine, case["prompt"], pins[case["case_id"]])
                else:
                    _, sample = timed_generate(engine, case["prompt"], case["output_tokens"])
                memory = sampler.window()
                if i < case["warmup_runs"]:
                    continue
                for name in ("prefill_tps", "decode_tps", "request_ms", "ttft_ms"):
                    samples[name].append(sample[name])
                samples["load_ms"].append(load_ms)
                samples["accepted_tokens_per_round"].append(sample["tokens_per_round"])
                for name, value in memory.items():
                    samples[name].append(value)
                print(f"{case['case_id']} rep {i - case['warmup_runs'] + 1}: ttft {sample['ttft_ms']:.0f} ms, decode "
                      f"{sample['decode_tps'] or 0:.1f} tok/s" + (" (pinned text)" if replayed else ""), flush=True)
            missing = {name: "not measurable on this host" for name, values in samples.items() if any(v is None for v in values)}
            missing.pop("accepted_tokens_per_round", None)
            cases[case["case_id"]] = {"samples": {name: (None if any(v is None for v in values) else values)
                                                  for name, values in samples.items()},
                                      "missing": missing, "replayed": replayed, "text_reason": text_reason,
                                      "started_at": begun, "ended_at": now()}
    finally:
        try:
            if engine is not None:
                engine.close()
        finally:
            sampler.stop()
    Path(job["out"]).write_text(json.dumps({"load_ms": load_ms, "cases": cases, "pins": new_pins}), encoding="utf-8")


if __name__ == "__main__":
    pin_worker(sys.argv[1]) if sys.argv[2:] == ["--pin"] else worker(sys.argv[1])
