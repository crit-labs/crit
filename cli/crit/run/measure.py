"""crit run measure: the run's benchmark cases measured on a candidate's package, through crit serve.

Under a benchmark reservation crit serve serves the candidate (its resident chat adapter, packaged and not yet
sealed) at the run's context, and each case's chat request streams to it over the chat API at the sampling the
captured request names (greedy when it names none), as a client of the server gets it. crit serve's usage must count the workload's prompt tokens. Every candidate decodes
its own reply; one that ends before the workload's output length is measured on the tokens it has. Host and device
memory are sampled on the resident host process during every request, and tokens per round comes from crit serve's
telemetry. A case of concurrency N sends each repetition as N requests started together, one per client: every
request's timings are its samples, and aggregate_tps is the wave's completion tokens over its wall time. The attempt
settles once the server has stopped, then one benchmark is recorded per case.

A benchmark qualifies only on the checks published before it, and a candidate with a benchmark that does not qualify
is never eligible, so measure refuses a candidate whose checks would leave it exploratory (unqualified) before it
reserves anything; --exploratory measures it anyway.
"""
import functools
import http.client
import json
import os
import subprocess
import sys
import threading
import time

from crit.pull.requirements import nvidia_smi

from .attempt import Attempt, packaged, remaining, unsealed
from .records import ALL_METRICS, digest, inputs, load_run, now, read_json, safe_path, series_gates

TIMED = ("prefill_tps", "decode_tps", "request_ms", "ttft_ms")
PER_REQUEST = TIMED + ("accepted_tokens_per_round",)
MEMORY = ("peak_ram_bytes", "minimum_free_ram_bytes", "peak_vram_bytes", "minimum_free_vram_bytes")


# Sampling fields a captured chat request keeps, as crit serve takes them (crit_api.options).
CAPTURED_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
                     "repetition_penalty")


def case_request(root, rows):
    """A case's chat request: its first retained JSON file holding {"messages": [...]}, with optional "tools", the
    sampling fields (CAPTURED_SAMPLING) it was sent with and "chat_template_kwargs", the chat-template variables it
    renders with (for example a workload captured with thinking on, where the candidate serves it off by default)."""
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
            sampling = {key: value[key] for key in CAPTURED_SAMPLING if value.get(key) is not None}
            if any(type(v) not in (int, float) for v in sampling.values()):
                raise ValueError(f"{row['path']}: sampling fields must be numbers")
            variables = value.get("chat_template_kwargs")
            if variables is not None and not isinstance(variables, dict):
                raise ValueError(f"{row['path']}: chat_template_kwargs must be an object")
            return {"messages": messages, **({"tools": tools} if tools else {}), **sampling,
                    **({"chat_template_kwargs": variables} if variables else {})}
    raise ValueError('No retained chat request for the case; keep a captured JSON file {"messages": [...]} in its '
                     'manifest')


def check_case_prompts(root, run, case_manifests, variables):
    """Each case whose captured chat request crit run measure would send must render to its workload's prompt tokens
    as a candidate serving the template `variables` (recon's frozen template_variables) renders it, the request's own
    chat_template_kwargs over them, as ResidentChat counts them: a case that renders otherwise could never be measured
    on a candidate (crit serve's usage would differ), so inputs are refused before they are frozen."""
    model = None
    for case in run["spec"]["cases"]:
        try:
            request = case_request(root, case_manifests[case["case_id"]])
        except ValueError:
            continue   # no chat request: measure refuses the case on its own
        if model is None:
            from crit_core.model import open_model
            from crit_core.text import load_model_text
            model = load_model_text(open_model(run["spec"]["model_path"]).files, clock=None)
        applied = {**variables, **request.get("chat_template_kwargs", {})}
        count = len(model.encode_chat(request["messages"], tools=request.get("tools"), add_generation_prompt=True,
                                      **applied))
        expected = case["workload"]["prompt_tokens"]
        if count != expected:
            raise ValueError(f"{case['case_id']}: its captured messages render to {count} prompt tokens with template "
                             f"variables {json.dumps(applied, sort_keys=True)}, the workload has {expected}: capture them "
                             "again for the template variables candidates serve, or name the ones the workload needs in "
                             "the request's chat_template_kwargs")


def unqualified(root, run, candidate_id):
    """What keeps a benchmark crit run measure records now from qualifying, by the candidate's published records:
    {"lasting": [...], "missing": [...], "failed": {gates}}. A benchmark binds each case's latest check of every gate
    (RunStore._case_checks) and qualifies only when those pass every required gate at the run's evidence level and
    the candidate's own gates pass (records._series and candidate_gates). `lasting` is what no later record undoes and
    keeps every benchmark of the candidate exploratory: a required check that did not pass (failed, unverified,
    not_applicable; `failed` names the gates of the failed ones) or an earlier benchmark's free memory below the run's
    headroom. `missing` is what publishing the checks puts right: a case with no check of a required gate, or whose
    latest passed one is only reported where the run requires artifact_verified evidence. This measurement's own free
    memory is judged once it is recorded."""
    folder = safe_path(root, f"candidates/{candidate_id}")
    telemetry = read_json(folder / "telemetry.json")
    checks = [read_json(path) for path in sorted((folder / "checks").glob("check-*.json"))]
    found = {"lasting": [], "missing": [], "failed": set()}
    for gate in run["spec"]["correctness"]["required_gates"]:
        own = [check for check in checks if check["gate"] == gate]
        for check in own:
            if check["status"] != "passed":
                found["lasting"].append(f"{gate} check {check['check_id']} ({check['case_id']}) {check['status']}")
                found["failed"] |= {gate} if check["status"] == "failed" else set()
        found["missing"] += [f"{case['case_id']} has no {gate} check" for case in run["spec"]["cases"]
                             if not any(check["case_id"] == case["case_id"] for check in own)]
    # The run's memory rule as a series judges it: resources fails below either headroom (series_gates).
    found["lasting"] += [f"{series['benchmark_id']} ({series['case_id']}) failed resources: its free memory is below "
                         "the run's headroom or unmeasured" for series in telemetry["series"]
                         if series_gates(run, [], series["metrics"])["resources"] == "failed"]
    if run["spec"]["objectives"]["verification_level"] != "reported":
        for case in run["spec"]["cases"]:
            latest = {check["gate"]: check for check in checks if check["case_id"] == case["case_id"]}
            found["missing"] += [f"{gate} check {check['check_id']} ({case['case_id']}) is {check['evidence_level']}, "
                                 "the run requires artifact_verified evidence" for gate, check in latest.items()
                                 if check["status"] == "passed" and check["evidence_level"] != "artifact_verified"]
    return found


def refusal(candidate_id, found):
    """Why crit run measure refuses a candidate whose benchmarks could not qualify (unqualified), and what to do."""
    if found["lasting"]:
        what = (f" No later record undoes that, so {candidate_id} cannot become eligible: build the fix as a new "
                "candidate" + (" (its plan repairs a failed check with a requirement lever serving "
                               + ", ".join(f"check:{gate}" for gate in sorted(found["failed"])) + ")"
                               if found["failed"] else "") + ".")
    else:
        what = (f" Record its checks first: crit run ... check --candidate {candidate_id} --from-eval records the "
                "arithmetic, functional and resources checks of every case from its completed Eval (crit run ... eval "
                f"--candidate {candidate_id}), then measure.")
    return (f"crit run measure refuses {candidate_id}: its benchmarks would stay exploratory, since "
            + "; ".join(found["lasting"] + found["missing"]) + "." + what + " To measure it anyway (telemetry, memory), "
            f"pass --exploratory: a benchmark that does not qualify keeps {candidate_id} from ever being eligible")


def measure(store, run_id, candidate_id, seconds=None, exploratory=False):
    """Reserve, serve the candidate, measure every case through crit serve, stop the server and settle, then record
    one benchmark per case. A candidate whose published checks would leave its benchmarks exploratory (unqualified)
    is refused before the reservation, unless `exploratory` measures it anyway."""
    root = store.directory(run_id)
    run = load_run(root)
    resolved = inputs(root, run)
    manifest = packaged(unsealed(root, candidate_id), "measuring", "crit run measure measures")
    cases = []
    for case in run["spec"]["cases"]:
        workload = case["workload"]
        if workload["mode"] != "resident":
            raise ValueError(f"{case['case_id']}: crit run measure supports resident cases")
        cases.append({"case_id": case["case_id"], "request": case_request(root, resolved["case_manifests"][case["case_id"]]),
                      **{key: workload[key] for key in ("prompt_tokens", "output_tokens", "warmup_runs", "repetitions",
                                                        "concurrency")}})
    found = unqualified(root, run, candidate_id)
    reasons = found["lasting"] + found["missing"]
    if reasons and not exploratory:
        raise ValueError(refusal(candidate_id, found))
    attempt = Attempt(store, run_id, candidate_id, "benchmark", seconds, "measure", "Measurement", "The measurement")
    with attempt.running() as log:
        if reasons:
            log("crit run measure --exploratory: the benchmarks stay exploratory, since " + "; ".join(reasons))
        measured = _measure(store, run_id, candidate_id, attempt, attempt.deadline_at,
                            run["spec"]["hardware_constraints"]["kv_tokens"], cases, log)
    # The server ran in this process; the resident host's exit is part of the cleanup.
    attempt.settle()
    attempt.finish()
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
                   "extensions": {"measured_by": "crit run measure", "served_by": served_by(root, candidate_id),
                                  "attempt_id": attempt.id,
                                  "output_tokens": result["output_tokens"], "sampling": result["sampling"],
                                  **({"chat_template_kwargs": result["template_kwargs"]}
                                     if result.get("template_kwargs") else {}),
                                  **({"concurrency": result["concurrency"]} if result.get("concurrency") else {})}}
        recorded.append(store.append(run_id, candidate_id, "benchmark", payload))
    return recorded


def served_by(root, candidate_id):
    """What served the candidate's measurements: crit serve, or an external candidate's own launch command."""
    from .external import SERVED_BY, candidate_spec
    return "crit serve" if candidate_spec(safe_path(root, f"candidates/{candidate_id}")) is None else SERVED_BY


def _measure(store, run_id, candidate_id, attempt, deadline_at, kv_tokens, cases, log):
    """Serve the candidate (attempt.Attempt) and measure every case; a failed engine cleanup raises with `settled`
    False."""
    from crit_core.machine import engine_lock
    from .external import serving
    # A request may take what is left of the reservation, whose deadline stops the server.
    served = serving(store, run_id, candidate_id, attempt, kv_tokens, log, "The measurement",
                     timeout=min(86400, max(1.0, remaining(deadline_at))))
    with engine_lock(f"crit run measure {run_id} {candidate_id}", deadline=time.monotonic() + remaining(deadline_at)):
        with served.until(deadline_at):
            started = time.perf_counter()
            engine, _, port = served.open()
            load_ms = (time.perf_counter() - started) * 1000
            sampler = Sampler(engine.host_pid(), run_gpus(store, run_id))
            try:
                return {case["case_id"]: _case(port, case, load_ms, sampler, served.timeout, log, engine.capacity())
                        for case in cases}
            finally:
                sampler.stop()


def case_body(case):
    """The streamed chat request a case sends: greedy unless the captured request names its own sampling."""
    return {"temperature": 0, **case["request"], "max_tokens": case["output_tokens"], "stream": True,
            "stream_options": {"include_usage": True}}


def _case(port, case, load_ms, sampler, timeout, log, capacity=None):
    """A case's warmups and repetitions: its metrics and missing metrics as a benchmark records them. A repetition of
    a case of concurrency N is a wave of N requests started together (_wave); capacity: the requests the served engine
    runs at once (crit serve's max_concurrent_requests)."""
    from crit_eval.reports.metrics import speed_sample
    body = case_body(case)
    clients = case.get("concurrency", 1)
    samples = {name: [] for name in PER_REQUEST + MEMORY + (("aggregate_tps",) if clients > 1 else ())}
    outputs, peaks, begun = [], [], now()
    seeds = case.get("seeds")
    for i in range(case["warmup_runs"] + case["repetitions"]):
        sampler.reset()
        # A case with seeds sends each measured repetition with its own seed (crit run ablate's Eval cases).
        sent = {**body, "seed": seeds[i - case["warmup_runs"]]} if seeds and i >= case["warmup_runs"] else body
        wave = _wave(port, sent, timeout, clients)
        memory = sampler.window()
        for _, _, _, usage, _ in wave:
            if case["prompt_tokens"] is not None and usage["prompt_tokens"] != case["prompt_tokens"]:
                raise ValueError(f"{case['case_id']}: crit serve rendered the case's messages as "
                                 f"{usage['prompt_tokens']} prompt tokens, the workload has {case['prompt_tokens']}")
        if i < case["warmup_runs"]:
            continue
        rows = []
        for started, first, last, usage, telemetry in wave:
            n = usage["completion_tokens"]
            sample = speed_sample(usage["prompt_tokens"], n, started, first, last, None)
            for name in TIMED:
                samples[name].append(sample[name])
            samples["accepted_tokens_per_round"].append(telemetry.get("tokens_per_round"))
            rows.append(sample)
            outputs.append(n)
        for name in MEMORY:
            samples[name].append(memory[name])
        if clients > 1:
            wall = max(last for _, _, last, _, _ in wave) - min(started for started, _, _, _, _ in wave)
            samples["aggregate_tps"].append(sum(usage["completion_tokens"] for _, _, _, usage, _ in wave) / wall)
            peaks.append(_peak(wave))
        lines = ", ".join(f"ttft {s['ttft_ms']:.0f} ms, decode {s['decode_tps'] or 0:.1f} tok/s" for s in rows[:4])
        log(f"{case['case_id']} rep {len(outputs) // clients}: " + (lines if clients == 1 else
            f"{clients} concurrent requests, {samples['aggregate_tps'][-1]:.1f} tok/s aggregate, peak {peaks[-1]} in "
            f"flight; {lines}" + (", ..." if clients > 4 else ""))
            + "".join(f", {'the' if clients == 1 else 'a'} reply ended after {n} of {case['output_tokens']} tokens"
                      for n in outputs[-clients:] if n != case["output_tokens"]))
    reasons = {"decode_tps": "A reply arrived in one piece, so its decode was not timed",
               "accepted_tokens_per_round": "crit serve reports no tokens per round: the engine reports no decode rounds"}
    metrics = {name: {"unit": ALL_METRICS[name], "samples": values} for name, values in samples.items()
               if all(value is not None for value in values)}
    metrics["load_ms"] = {"unit": ALL_METRICS["load_ms"], "samples": [load_ms] * case["repetitions"]}
    missing = {name: reasons.get(name, "Not measurable on this host") for name in samples if name not in metrics}
    return {"metrics": metrics, "missing": missing, "output_tokens": outputs, "started_at": begun, "ended_at": now(),
            "sampling": {key: body[key] for key in CAPTURED_SAMPLING if key in body},
            "template_kwargs": body.get("chat_template_kwargs"),
            # Per repetition, the requests in flight at once at the peak, and what the engine runs at once.
            "concurrency": {"clients": clients, "peak_concurrency": peaks, "max_concurrent_requests": capacity}
                           if clients > 1 else None}


def _wave(port, body, timeout, clients):
    """`clients` copies of a request sent at once, one per thread (_request); their results in client order."""
    if clients == 1:
        return [_request(port, body, timeout)]
    start, results, failures = threading.Event(), [None] * clients, []

    def client(index):
        start.wait()
        try:
            results[index] = _request(port, body, timeout)
        except BaseException as error:
            failures.append(error)
    threads = [threading.Thread(target=client, args=(index,), daemon=True) for index in range(clients)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]
    return results


def _peak(wave):
    """The requests of a wave in flight at once at the peak: from sending to the last reply piece."""
    events = sorted([(started, 1) for started, _, _, _, _ in wave] + [(last, -1) for _, _, last, _, _ in wave])
    current = peak = 0
    for _, change in events:   # an end (-1) sorts before a start at the same instant
        current += change
        peak = max(peak, current)
    return peak


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
    free device memory, sampled during requests. Device memory is the driver's whole-GPU view (nvidia-smi; on Linux
    without it the first amdgpu card's VRAM counters, amdgpu_vram), which counts other processes too; inside WSL2 the
    process's own cudaMemGetInfo disagrees with it. `gpus`: the GPUs the engine runs on (an engine on several, from the
    run's device file), summed; default the first CUDA_VISIBLE_DEVICES names, else GPU 0."""

    def __init__(self, pid, gpus=None):
        self.pid = pid
        self.lock = threading.Lock()
        self.running = True
        self.min_free_ram = self.peak_rss = self.min_free_vram = self.total_vram = None
        self.smi = self.vram = None
        if gpus is None:
            first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
            gpus = [int(first) if first.isdigit() else 0]
        self.gpus, self.readings = [int(g) for g in gpus], {}
        smi = nvidia_smi()
        if smi:
            self.smi = subprocess.Popen([smi, "--id=" + ",".join(map(str, self.gpus)),
                                         "--query-gpu=index,memory.used,memory.total",
                                         "--format=csv,noheader,nounits", "-lms", "100"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._device, daemon=True).start()
        else:
            self.vram = amdgpu_vram()
            if self.vram:
                threading.Thread(target=self._amdgpu, daemon=True).start()
        self.thread = threading.Thread(target=self._host, daemon=True)
        self.thread.start()

    def _device(self):
        for line in self.smi.stdout:
            try:
                index, used, total = line.split(",")[:3]
                index, used, total = int(index), int(float(used)) * (1 << 20), int(float(total)) * (1 << 20)
            except ValueError:
                continue
            with self.lock:
                self.readings[index] = (used, total)
                if len(self.readings) < len(self.gpus):
                    continue   # every GPU read once before their sum counts
                self.total_vram = sum(t for _, t in self.readings.values())
                free = sum(t - u for u, t in self.readings.values())
                self.min_free_vram = free if self.min_free_vram is None else min(self.min_free_vram, free)

    def _amdgpu(self):
        used_path, total_path = self.vram
        while self.running:
            try:
                used, total = (int(path.read_text()) for path in (used_path, total_path))
            except (OSError, ValueError):
                return
            with self.lock:
                self.total_vram = total
                free = total - used
                self.min_free_vram = free if self.min_free_vram is None else min(self.min_free_vram, free)
            time.sleep(.1)

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
        if self.smi or self.vram:
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
            self.smi.stdout.close()


def run_gpus(store, run_id):
    """The GPUs a run's device file names when it is a set of several (crit eval probe --devices), else None."""
    context = (load_run(store.directory(run_id))["spec"]["hardware_constraints"].get("context") or {})
    return [row["index"] for row in context.get("devices") or []] or None


def amdgpu_vram(root="/sys/class/drm"):
    """(used, total) VRAM counter files of the first amdgpu card (Linux sysfs, bytes), or None."""
    from pathlib import Path
    for card in sorted(Path(root).glob("card[0-9]*")):
        used, total = card / "device" / "mem_info_vram_used", card / "device" / "mem_info_vram_total"
        if used.is_file() and total.is_file():
            return used, total
    return None


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
