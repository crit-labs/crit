"""crit run measure: the run's benchmark cases measured on a candidate's package, through crit serve.

Under a benchmark reservation crit serve serves the candidate (its resident chat adapter, packaged and not yet
sealed) at the run's context, and each case's chat request streams to it over the chat API with greedy decoding, as
a client of the server gets it. crit serve's usage must count the workload's prompt tokens. Every candidate decodes
its own reply; one that ends before the workload's output length is measured on the tokens it has. Host and device
memory are sampled on the resident host process during every request, and tokens per round comes from crit serve's
telemetry. The attempt settles once the server has stopped, then one benchmark is recorded per case.
"""
import functools
import http.client
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from .records import ALL_METRICS, digest, file_sha, ident, inputs, load_run, now, read_json, safe_path, timestamp
from .served import Expired, SERVED, Served, remaining

TIMED = ("prefill_tps", "decode_tps", "request_ms", "ttft_ms")
MEMORY = ("peak_ram_bytes", "minimum_free_ram_bytes", "peak_vram_bytes", "minimum_free_vram_bytes")


def case_request(root, rows):
    """A case's chat request: its first retained JSON file holding {"messages": [...]}, with optional "tools"."""
    from crit_api.server import MAX_BODY
    for row in rows:
        path = safe_path(root, row["path"])
        if path.suffix.lower() != ".json":
            continue
        try:
            value = read_json(path, max_bytes=MAX_BODY)
        except (ValueError, OSError):
            continue
        messages = value.get("messages") if isinstance(value, dict) else None
        if isinstance(messages, list) and messages and all(isinstance(m, dict) for m in messages):
            tools = value.get("tools")
            if tools is not None and not isinstance(tools, list):
                raise ValueError(f"{row['path']}: tools must be a list")
            return {"messages": messages, **({"tools": tools} if tools else {})}
    raise ValueError('No retained chat request for the case; keep a captured JSON file {"messages": [...]} in its '
                     'manifest')


def measure(store, run_id, candidate_id, seconds=None):
    """Reserve, serve the candidate, measure every case through crit serve, stop the server and settle, then record
    one benchmark per case."""
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
    missing = [c for c in SERVED if c not in manifest["capabilities"]]
    if missing:
        raise ValueError("crit run measure measures the candidate through crit serve: its package must declare "
                         + ", ".join(missing) + " (docs/SERVING.md)")
    cases = []
    for case in run["spec"]["cases"]:
        workload = case["workload"]
        if workload["mode"] != "resident" or workload["concurrency"] != 1:
            raise ValueError(f"{case['case_id']}: crit run measure supports resident cases of concurrency 1")
        cases.append({"case_id": case["case_id"], "request": case_request(root, resolved["case_manifests"][case["case_id"]]),
                      **{key: workload[key] for key in ("prompt_tokens", "output_tokens", "warmup_runs", "repetitions")}})
    reservation = store.reserve(run_id, candidate_id, "benchmark", seconds)
    attempt = reservation["attempt_id"]
    log_path = folder / "logs" / f"measure-{attempt}.log"
    outcome, failure, settled, interrupted = "completed", None, True, None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            def log(message):
                print(message, file=sys.stderr, flush=True)
                stream.write(message + "\n")
                stream.flush()
            measured = _measure(store, run_id, candidate_id, attempt, reservation["deadline_at"],
                                run["spec"]["hardware_constraints"]["kv_tokens"], cases, log)
    except BaseException as exc:
        settled = getattr(exc, "settled", True)
        if isinstance(exc, (Expired, TimeoutError)):
            outcome, failure = "timed_out", str(exc) or "The measurement passed its reserved deadline"
        elif isinstance(exc, (KeyboardInterrupt, SystemExit)):
            outcome, failure, interrupted = "interrupted", "Interrupted", exc
        else:
            outcome, failure = "failed", (type(exc).__name__ + ": " + str(exc))[:8000]
    ended = now()
    if outcome == "completed" and timestamp(ended, "ended") > timestamp(reservation["deadline_at"], "deadline"):
        outcome, failure = "timed_out", "The measurement passed its reserved deadline"
    # As for every attempt, the log is retained as evidence only once cleanup is proven.
    logs = ([{"path": log_path.relative_to(root).as_posix(), "sha256": file_sha(log_path)}]
            if settled and log_path.is_file() else [])
    # The server ran in this process; the resident host's exit is part of the cleanup.
    store.settle(run_id, reservation, {"outcome": outcome, "ended_at": ended,
                                       "exit_code": 0 if outcome == "completed" else None,
                                       "cleanup_settled": settled, "logs": logs, "failure": failure})
    if interrupted is not None:
        raise interrupted
    if outcome != "completed":
        raise ValueError(f"Measurement {outcome}: {failure} (log {log_path.relative_to(root).as_posix()})")
    recorded = []
    for case in run["spec"]["cases"]:
        result = measured[case["case_id"]]
        payload = {"case_id": case["case_id"], "package_sha256": digest(manifest), "case_sha256": digest(case),
                   "protocol_sha256": digest(run["spec"]["benchmark_protocol"]),
                   "identities": {"model_sha256": digest(resolved["model_manifest"]),
                                  "input_sha256": digest(resolved["case_manifests"][case["case_id"]]),
                                  "hardware_sha256": digest(resolved["hardware_manifest"]),
                                  "environment_sha256": digest(resolved["environment_manifest"])},
                   "workload": case["workload"], "metrics": result["metrics"], "missing_metrics": result["missing"],
                   "checks": [], "outcome": "completed", "started_at": result["started_at"],
                   "ended_at": result["ended_at"],
                   "extensions": {"measured_by": "crit run measure", "served_by": "crit serve", "attempt_id": attempt,
                                  "output_tokens": result["output_tokens"]}}
        recorded.append(store.append(run_id, candidate_id, "benchmark", payload))
    return recorded


def _measure(store, run_id, candidate_id, attempt, deadline_at, kv_tokens, cases, log):
    """Serve the candidate and measure every case; a failed engine cleanup raises with `settled` False."""
    from crit_core.machine import engine_lock
    # A request may take what is left of the reservation, whose deadline stops the server.
    served = Served(store, run_id, candidate_id, attempt, kv_tokens, log, "The measurement",
                    timeout=min(86400, max(1.0, remaining(deadline_at))))
    with engine_lock(f"crit run measure {run_id} {candidate_id}", deadline=time.monotonic() + remaining(deadline_at)):
        with served.until(deadline_at):
            started = time.perf_counter()
            engine, _, port = served.open()
            load_ms = (time.perf_counter() - started) * 1000
            sampler = Sampler(engine.host_pid())
            try:
                return {case["case_id"]: _case(port, case, load_ms, sampler, served.timeout, log) for case in cases}
            finally:
                sampler.stop()


def _case(port, case, load_ms, sampler, timeout, log):
    """A case's warmups and repetitions: its metrics and missing metrics as a benchmark records them."""
    from crit_eval.metrics import speed_sample
    body = {**case["request"], "max_tokens": case["output_tokens"], "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    samples = {name: [] for name in TIMED + MEMORY + ("accepted_tokens_per_round",)}
    outputs, begun = [], now()
    for i in range(case["warmup_runs"] + case["repetitions"]):
        sampler.reset()
        started, first, last, usage, telemetry = _request(port, body, timeout)
        memory = sampler.window()
        if usage["prompt_tokens"] != case["prompt_tokens"]:
            raise ValueError(f"{case['case_id']}: crit serve rendered the case's messages as {usage['prompt_tokens']} "
                             f"prompt tokens, the workload has {case['prompt_tokens']}")
        if i < case["warmup_runs"]:
            continue
        n = usage["completion_tokens"]
        sample = speed_sample(usage["prompt_tokens"], n, started, first, last, None)
        for name in TIMED:
            samples[name].append(sample[name])
        for name in MEMORY:
            samples[name].append(memory[name])
        samples["accepted_tokens_per_round"].append(telemetry.get("tokens_per_round"))
        outputs.append(n)
        log(f"{case['case_id']} rep {len(outputs)}: ttft {sample['ttft_ms']:.0f} ms, decode "
            f"{sample['decode_tps'] or 0:.1f} tok/s" + ("" if n == case["output_tokens"] else
                                                       f", the reply ended after {n} of {case['output_tokens']} tokens"))
    reasons = {"decode_tps": "A reply arrived in one piece, so its decode was not timed",
               "accepted_tokens_per_round": "crit serve reports no tokens per round: the engine reports no decode rounds"}
    metrics = {name: {"unit": ALL_METRICS[name], "samples": values} for name, values in samples.items()
               if all(value is not None for value in values)}
    metrics["load_ms"] = {"unit": ALL_METRICS["load_ms"], "samples": [load_ms] * case["repetitions"]}
    missing = {name: reasons.get(name, "Not measurable on this host") for name in samples if name not in metrics}
    return {"metrics": metrics, "missing": missing, "output_tokens": outputs, "started_at": begun, "ended_at": now()}


def _request(port, body, timeout):
    """Stream one chat request to crit serve. Returns when it was sent and when its first and last reply pieces
    arrived (time.perf_counter()), its usage and crit serve's telemetry."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        started = time.perf_counter()
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            data = response.read()
            try:
                data = json.loads(data)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            raise ValueError(f"crit serve answered {response.status}: {data}")
        first = last = usage = None
        telemetry = {}
        for line in response:
            arrived = time.perf_counter()
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise ValueError("crit serve failed the request: " + str(event["error"].get("message")))
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                    first, last = first or arrived, arrived
            usage = event.get("usage") or usage
            telemetry = event.get("crit_telemetry") or telemetry
    finally:
        connection.close()
    if first is None or not usage:
        raise ValueError("crit serve streamed " + ("no reply" if first is None else "no usage"))
    return started, first, last, usage, telemetry


class Sampler:
    """Free host memory, the resident memory of process `pid` (crit serve's resident host, where the engine runs) and
    free device memory, sampled during requests. Device memory is the driver's whole-GPU view (nvidia-smi, GPU 0),
    which counts other processes too; inside WSL2 the process's own cudaMemGetInfo disagrees with it."""

    def __init__(self, pid):
        self.pid = pid
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
        available, rss = host_available(), process_rss(self.pid) if self.pid else None
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


def process_rss(pid):
    """Process `pid`'s resident memory (its working set on Windows), or None once it has exited."""
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{pid}/status", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return None
    if os.name == "nt":
        import ctypes
        kernel32, psapi, Counters = _windows_memory()
        # SYNCHRONIZE, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ
        handle = kernel32.OpenProcess(0x100000 | 0x1000 | 0x0010, False, pid)
        if not handle:
            return None
        try:
            # An exited process stays open while anyone holds a handle to it, as subprocess.Popen does to its child.
            if kernel32.WaitForSingleObject(handle, 0) == 0:  # WAIT_OBJECT_0: it has exited
                return None
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return counters.working_set if ok else None
        finally:
            kernel32.CloseHandle(handle)
    return None


@functools.cache
def _windows_memory():
    """kernel32 and psapi with the prototypes process_rss calls, and PROCESS_MEMORY_COUNTERS."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD), ("peak_working_set", ctypes.c_size_t),
                    ("working_set", ctypes.c_size_t), ("quota_peak_paged", ctypes.c_size_t),
                    ("quota_paged", ctypes.c_size_t), ("quota_peak_nonpaged", ctypes.c_size_t),
                    ("quota_nonpaged", ctypes.c_size_t), ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
    return kernel32, psapi, Counters
