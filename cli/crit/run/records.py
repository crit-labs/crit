"""Strict first-class run records and deterministic candidate projections."""
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import stat
import tempfile
import time

MAX_JSON_BYTES = 1_048_576
MAX_CANDIDATES = 512
MAX_FILES = 100_000
WEIGHTS_POLICIES = {"as_stored", "transformable"}  # run: may candidates change stored weights?
# Drift a run allows against the Eval reference, spent on speed; matches eval/protocol.json.
DEFAULT_QUALITY_BUDGET = {"ppl_ratio_max": 1.05, "kld_mean_max": 0.05, "kld_p99_max": 0.5}
EXACTNESS_MODES = {"reported", "required"}  # must speculative output equal plain greedy?
WEIGHTS = {"as_stored", "transformed"}             # candidate: did it change them?
HASH_PARALLEL_BYTES = 256 * 1024 * 1024
GIB = 1024 ** 3
_verification_cache = ContextVar("crit_verification_cache", default=None)

RUN_ID = r"run-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}"
LIFECYCLES = {"queued", "building", "built", "checking", "benchmarking", "completed", "build_failed",
              "check_failed", "benchmark_failed", "cancelled", "timed_out", "interrupted"}
STOP_REASONS = {"satisfied", "search_exhausted", "blocked", "budget_exhausted", "cancelled", "interrupted"}
METRICS = {"prefill_tps": "tokens/second", "decode_tps": "tokens/second", "request_ms": "milliseconds",
           "load_ms": "milliseconds", "peak_vram_bytes": "bytes", "peak_ram_bytes": "bytes",
           "minimum_free_vram_bytes": "bytes", "minimum_free_ram_bytes": "bytes"}
OPTIONAL_METRICS = {"ttft_ms": "milliseconds", "startup_ms": "milliseconds", "perplexity": "dimensionless",
                    "kl_divergence": "nats/token", "accepted_tokens_per_round": "tokens/round"}
ALL_METRICS = METRICS | OPTIONAL_METRICS
# Lowest valid sample: perplexity and accepted tokens per round are at least 1, rates are positive.
METRIC_MINIMUM = {"perplexity": 1, "accepted_tokens_per_round": 1}


def metric_minimum(key):
    return METRIC_MINIMUM.get(key, 0.000001 if key.endswith("tps") else 0)


@contextmanager
def verification_scope():
    """Deduplicate hashes only within one boundary check, observing file changes."""
    if _verification_cache.get() is not None:
        yield
        return
    token = _verification_cache.set({})
    try:
        yield
    finally:
        _verification_cache.reset(token)

def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_sha(path, cache=False):
    # Windows may retain stale timestamps while another handle is open.
    # Metadata cannot safely identify unchanged bytes there, even within a scope.
    if cache is False:
        cache = _verification_cache.get() if os.name != "nt" else None
    before = path.stat()
    identity = (str(path), before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if cache is not None and identity in cache:
        return cache[identity]
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Artifact changed while hashing")
    value = hasher.hexdigest()
    if cache is not None and len(cache) < MAX_FILES:
        cache[identity] = value
    return value


def text(value, name, limit=4096):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\0" in value:
        raise ValueError(f"{name} requires nonempty text (up to {limit} characters)")
    return value


def number(value, name, low=0, high=1_000_000, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} requires a finite {'integer' if integer else 'number'} between {low} and {high}")
    return value


# A candidate-budget segment has no working time budget; its deadline is only a safety window.
SAFETY_WINDOW_MINUTES = 43200
DEFAULT_CANDIDATE_BUDGET = 3
HEADROOM_REASONS = {"search_exhausted", "budget_exhausted"}
STOP_KPIS = ("decode", "prefill", "ttft")  # a stop's headroom accounts for each
CONTEXT_KPIS = ("context_decode", "context_ttft")  # and for these when Eval measured the required context
KV_TOKENS_MAX = 2_000_000


def quality_terms(correctness, amendments=()):
    """The run's drift budget over the defaults, and its exactness mode, with each resume's
    amendment applied in order."""
    budget = {**DEFAULT_QUALITY_BUDGET, **correctness.get("quality_budget", {})}
    exactness = correctness.get("exactness", "reported")
    for change in amendments:
        budget |= change.get("quality_budget", {})
        exactness = change.get("exactness", exactness)
    return budget, exactness


def amendment(value):
    """A resume's change to the run's quality terms, with who approved it."""
    keys(value, {"approved_by"}, {"quality_budget", "exactness"})
    text(value["approved_by"], "Amendment approval", 400)
    if not {"quality_budget", "exactness"} & set(value):
        raise ValueError("An amendment changes quality_budget or exactness")
    budget = value.get("quality_budget", {})
    keys(budget, set(), set(DEFAULT_QUALITY_BUDGET))
    for key, limit in budget.items():
        number(limit, key, 1 if key == "ppl_ratio_max" else 0, 1_000_000)
    if value.get("exactness", "reported") not in EXACTNESS_MODES:
        raise ValueError("exactness requires reported or required")
    return value


def headroom_record(value, reason):
    """Validate a stop record's account of each Eval goal: the KPI's score at the start, the score
    reached and its physical limit (tok/s for decode and prefill, ms for ttft), what was built for
    it and the physical cause of any distance left to the limit. search_exhausted and
    budget_exhausted need all three KPIs; the long-context goals (context_decode, context_ttft) may
    be added. crit checks that the account is complete, not what it concludes: the agent reasons,
    no threshold decides."""
    if value is None:
        if reason in HEADROOM_REASONS:
            raise ValueError(f"{reason} requires headroom: for each of decode, prefill and ttft the score at the start, "
                             "the score reached, the limit, what was built for it and the physical cause of what is left")
        return None
    keys(value, {"kpis"})
    rows = value["kpis"]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Headroom kpis require a list of {kpi, start, reached, limit, built, cause}")
    seen = set()
    for row in rows:
        keys(row, {"kpi", "start", "reached", "limit", "built", "cause"})
        if row["kpi"] not in STOP_KPIS + CONTEXT_KPIS or row["kpi"] in seen:
            raise ValueError("Headroom kpis are decode, prefill, ttft, context_decode and context_ttft, each at most once")
        seen.add(row["kpi"])
        for key in ("start", "reached", "limit"):
            if row[key] is not None:
                number(row[key], f"{row['kpi']} {key}", 0, 1e9)
        text(row["built"], f"What was built for {row['kpi']}", 8000)
        text(row["cause"], f"The physical cause of the distance left between {row['kpi']} and its limit", 8000)
    if reason in HEADROOM_REASONS and not set(STOP_KPIS) <= seen:
        raise ValueError(f"{reason} requires an account of every KPI: " + ", ".join(STOP_KPIS))
    return value


def sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} requires a lowercase SHA-256")
    return value


def timestamp(value, name):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset().total_seconds() != 0:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"{name} requires a UTC timestamp with timezone") from None
    return result


def ident(value, kind):
    pattern = RUN_ID if kind == "run" else kind + r"-[0-9]{4,8}"
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f"Invalid {kind} identity")
    return value


def keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        raise ValueError("Missing required fields: " + ", ".join(sorted(set(required) - (value.keys() if isinstance(value, dict) else set()))))
    unknown = value.keys() - set(required) - set(optional) - {"extensions"}
    if unknown:
        raise ValueError("Unknown fields; use extensions: " + ", ".join(sorted(unknown)))
    if "extensions" in value and not isinstance(value["extensions"], dict):
        raise ValueError("extensions must be an object")
    json.dumps(value, allow_nan=False)
    return value


def record(kind, run_id, **fields):
    return {"schema_version": 1, "record_kind": kind, "run_id": run_id, **fields}


def header(value, kind, run_id, required, optional=()):
    keys(value, {"schema_version", "record_kind", "run_id"} | set(required), optional)
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["record_kind"] != kind:
        raise ValueError(f"Requires schema_version 1, record_kind {kind}")
    if value["run_id"] != ident(run_id, "run"):
        raise ValueError("Record belongs to a different run")
    return value


def safe_path(root, relative):
    text(relative, "record path")
    parts = PurePosixPath(relative).parts
    if ("\\" in relative or ":" in relative or relative.startswith("/") or
            any(p in {"", ".", ".."} for p in relative.split("/"))):
        raise ValueError("References require canonical relative forward-slash paths")
    root = Path(root).absolute()
    if root.resolve() != root:
        raise ValueError("Reference escapes its retained directory")
    cursor = root
    # Resolve the root once. Every child is lexical and every existing component
    # is inspected without following links; repeated full ancestor resolution
    # made large sealed source trees prohibitively slow on Windows.
    for part in (None, *parts):
        if part is not None:
            cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/reparse references are not permitted")
    return cursor


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field: " + key)
        value[key] = item
    return value


def read_json(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON must be a regular file under 1 MiB")
    data = path.read_bytes()
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON must be under 1 MiB")
    return json.loads(data.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def write_json(path, value, immutable=False):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON projection exceeds 1 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            # Exclusive hard-link publication is atomic and never replaces retained bytes.
            os.link(temporary, path)
        else:
            # Windows readers can briefly deny replacement while holding an
            # ordinary file handle. Keep atomic publication and bound retries;
            # a persistent permission error must still reach the caller.
            deadline = time.monotonic() + .25
            while True:
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError as exc:
                    if getattr(exc, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                        raise
                    time.sleep(.005)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def inventory(root, exclude=()):
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        safe_path(root, relative)
        if path.is_file() and relative not in exclude:
            files.append((relative, path))
            if len(files) > MAX_FILES:
                raise ValueError("Artifact exceeds file inventory limit")
    # Large artifacts (model checkpoints) hash files concurrently; hashlib releases
    # the GIL while digesting, so threads overlap reads and hashing. Row order and
    # values are identical to a sequential pass.
    cache = _verification_cache.get() if os.name != "nt" else None
    total = sum(path.stat().st_size for _, path in files)
    workers = min(8, os.cpu_count() or 1, len(files)) if total >= HASH_PARALLEL_BYTES else 1
    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            digests = list(pool.map(lambda item: file_sha(item[1], cache), files))
    else:
        digests = [file_sha(path, cache) for _, path in files]
    return [{"path": relative, "bytes": path.stat().st_size, "sha256": value} for (relative, path), value in zip(files, digests)]


def verify_inventory(root, rows, exclude=()):
    if not isinstance(rows, list) or not rows or len(rows) > MAX_FILES:
        raise ValueError("Artifact requires a nonempty bounded file inventory")
    for row in rows:
        keys(row, {"path", "bytes", "sha256"})
        safe_path(root, row["path"])
        number(row["bytes"], "file bytes", 0, 2 ** 60, integer=True)
        sha(row["sha256"], "file sha256")
    if rows != inventory(root, exclude):
        raise ValueError("Retained artifact bytes differ from manifest")


def eval_max_context(model, device, vram_headroom_bytes, reserve_bytes=None):
    """Eval's max context (eval/crit_eval/context.py) for MODEL on DEVICE, a `crit eval probe` result. reserve_bytes
    replaces the protocol's reserve (device memory kept out of the KV cache for workspaces and speed techniques)."""
    from crit_eval import EVAL_DIR
    from crit_eval.context import max_context
    protocol = EVAL_DIR / "protocol.json"
    if not protocol.is_file():
        raise ValueError("Deriving kv_tokens from a device needs Eval from a crit checkout; set kv_tokens explicitly")
    settings = json.loads(protocol.read_text(encoding="utf-8"))["context"]
    if reserve_bytes is not None:
        settings = {**settings, "reserve_gib": reserve_bytes / GIB}
    return max_context(model, device, vram_headroom_bytes, settings)


def kv_capacity(data, model, vram_headroom_bytes, reserve_bytes=None):
    """The run's KV capacity in tokens of prompt plus output, and what it rests on. An explicit kv_tokens
    is taken as given. Otherwise the request names its device (a `crit eval probe` result) and the run is
    built for Eval's max context: the model's positions, capped by the KV cache that fits on the device."""
    if "kv_tokens" in data:
        return number(data["kv_tokens"], "KV cache", 1, KV_TOKENS_MAX, integer=True), None
    if "device" not in data:
        raise ValueError("Request needs device (a crit eval probe result, so the run is built for Eval's max context) "
                         "or an explicit kv_tokens")
    device = Path(text(data["device"], "Device file")).expanduser().resolve()
    value = eval_max_context(model, read_json(device), vram_headroom_bytes, reserve_bytes)
    if not value["tokens"]:
        raise ValueError("Eval cannot derive the max context: " + (value["reason"] or "; ".join(value["missing"])))
    tokens, bound = min(value["tokens"], KV_TOKENS_MAX), value["bound"] if value["tokens"] <= KV_TOKENS_MAX else "crit"
    basis = value.get("basis") or {}
    context = {"source": "eval", "bound": bound, "model_limit": value["model_limit"],
               "model_limit_source": value["model_limit_source"], "device_limit": value["device_limit"],
               "kv_precision": value["kv_precision"], "kv_bytes_per_token": value["kv_bytes_per_token"],
               **{key: basis[key] for key in ("device", "memory_free_bytes", "memory_total_bytes", "reserve_bytes",
                                              "resident_weight_bytes", "expert_bytes", "fixed_state_bytes",
                                              "expert_bytes_in_device_memory_at_max") if key in basis},
               "device_file": str(device), "device_sha256": file_sha(device)}
    return number(tokens, "Eval's max context", 1, KV_TOKENS_MAX, integer=True), context


def request_spec(data):
    keys(data, {"model_path", "correctness_criteria"}, {"kv_tokens", "device", "vram_headroom_gib", "ram_headroom_gib",
         "context_reserve_gib",
         "budget_minutes", "workload", "cases", "benchmark_protocol", "targets", "verification_level", "attempt_limit",
         "weights_policy", "candidate_budget", "quality_budget", "exactness"})
    path = Path(text(data["model_path"], "Model path")).expanduser().resolve()
    if not path.exists():
        raise ValueError("Model path does not exist on this computer")
    workload = data.get("workload", {})
    keys(workload, set(), {"prompt_tokens", "output_tokens", "concurrency", "repetitions", "warmup_runs", "mode"})
    workload = {key: number(workload.get(key, default), key, minimum, limit, integer=True)
                for key, default, minimum, limit in (("prompt_tokens", 512, 1, 1_000_000), ("output_tokens", 128, 1, 1_000_000),
                 ("concurrency", 1, 1, 256), ("repetitions", 3, 1, 100), ("warmup_runs", 1, 0, 100))} | {"mode": workload.get("mode", "resident")}
    if workload["mode"] not in {"resident", "cold"}:
        raise ValueError("Mode requires resident or cold")
    headroom = {kind + "_headroom_bytes": round(number(data.get(kind + "_headroom_gib", default), kind + " headroom", high=65536) * GIB)
                for kind, default in (("vram", 1), ("ram", 2))}
    reserve = (round(number(data["context_reserve_gib"], "context reserve", 0, 65536) * GIB)
               if "context_reserve_gib" in data else None)
    capacity, context = kv_capacity(data, path, headroom["vram_headroom_bytes"], reserve)
    if capacity < workload["prompt_tokens"] + workload["output_tokens"]:
        raise ValueError("KV capacity must cover prompt plus output")
    targets = data.get("targets", {})
    keys(targets, set(), {"decode_tps", "prefill_tps", "request_ms"})
    for key, value in targets.items():
        if key != "extensions":
            number(value, key + " target", 0.000001, 1_000_000_000)
    level = data.get("verification_level", "artifact_verified")
    if level not in {"reported", "artifact_verified"}:
        raise ValueError("Verification level requires reported or artifact_verified")
    spec = {"model_path": str(path), "hardware_constraints": {"kv_tokens": capacity, **headroom,
                                                              **({"context": context} if context else {})},
            "cases": data.get("cases", [{"case_id": "case-0001", "workload": workload}]),
            "benchmark_protocol": text(data.get("benchmark_protocol", "crit-matched-v1"), "Protocol", 200),
            "correctness": {"criteria": text(data["correctness_criteria"], "Correctness criteria", 8000),
                            "required_gates": ["arithmetic", "functional", "resources"],
                            **({"weights_policy": data["weights_policy"]} if "weights_policy" in data else {}),
                            **({"quality_budget": data["quality_budget"]} if "quality_budget" in data else {}),
                            **({"exactness": data["exactness"]} if "exactness" in data else {})},
            "objectives": {"targets": targets, "verification_level": level},
            "resource_limits": resource_limits(data)}
    validate_spec(spec)
    return spec


def resource_limits(data):
    """A run is bounded by candidates, not minutes, unless it asks only for a time budget."""
    limits = {"attempt_limit": number(data.get("attempt_limit", 64), "Attempt limit", 1, 1000, integer=True),
              # Eval at a long required context (curve, lower context, capacity) runs well past 10 minutes.
              "process_seconds": {"discovery": 180, "build": 1800, "check": 3600, "benchmark": 1800, "agent": 10800}}
    if "budget_minutes" in data and "candidate_budget" not in data:
        return {"budget_minutes": number(data["budget_minutes"], "Budget", 1, 180, integer=True), **limits}
    return {"budget_minutes": number(data.get("budget_minutes", SAFETY_WINDOW_MINUTES), "Safety window", 1, SAFETY_WINDOW_MINUTES, integer=True),
            "candidate_budget": number(data.get("candidate_budget", DEFAULT_CANDIDATE_BUDGET), "Candidate budget", 1, 64, integer=True),
            **limits}


def validate_limits(limits, name):
    keys(limits, {"budget_minutes", "attempt_limit", "process_seconds"}, {"candidate_budget"})
    if "candidate_budget" in limits:
        number(limits["candidate_budget"], name + " candidate budget", 1, 64, integer=True)
        number(limits["budget_minutes"], name + " safety window", 1, SAFETY_WINDOW_MINUTES, integer=True)
    else:
        number(limits["budget_minutes"], name + " budget", 1, 180, integer=True)
    number(limits["attempt_limit"], name + " attempt limit", 1, 1000, integer=True)


def validate_workload(value):
    keys(value, {"prompt_tokens", "output_tokens", "concurrency", "repetitions", "warmup_runs", "mode"})
    for key, low, high in (("prompt_tokens", 1, 1_000_000), ("output_tokens", 1, 1_000_000), ("concurrency", 1, 256), ("repetitions", 1, 100), ("warmup_runs", 0, 100)):
        number(value[key], key, low, high, integer=True)
    if value["mode"] not in {"resident", "cold"}:
        raise ValueError("Unknown workload mode")


def validate_spec(value):
    keys(value, {"model_path", "hardware_constraints", "cases", "benchmark_protocol", "correctness", "objectives", "resource_limits"})
    if not Path(text(value["model_path"], "model locator")).is_absolute():
        raise ValueError("Model locator requires an absolute local path")
    hardware = value["hardware_constraints"]
    keys(hardware, {"kv_tokens", "vram_headroom_bytes", "ram_headroom_bytes"}, {"context"})
    number(hardware["kv_tokens"], "KV capacity", 1, KV_TOKENS_MAX, integer=True)
    if "context" in hardware and (not isinstance(hardware["context"], dict) or hardware["context"].get("source") != "eval"):
        raise ValueError("A KV capacity basis requires Eval's max context")
    for key in ("vram_headroom_bytes", "ram_headroom_bytes"):
        number(hardware[key], key, 0, 65536 * GIB, integer=True)
    if not isinstance(value["cases"], list) or not 1 <= len(value["cases"]) <= 32:
        raise ValueError("Run requires 1 to 32 fixed cases")
    seen = set()
    for case in value["cases"]:
        keys(case, {"case_id", "workload"})
        ident(case["case_id"], "case")
        if case["case_id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case["case_id"])
        validate_workload(case["workload"])
        if hardware["kv_tokens"] < case["workload"]["prompt_tokens"] + case["workload"]["output_tokens"]:
            raise ValueError("Case exceeds requested KV capacity")
    text(value["benchmark_protocol"], "protocol", 200)
    keys(value["correctness"], {"criteria", "required_gates"}, {"weights_policy", "quality_budget", "exactness"})
    text(value["correctness"]["criteria"], "criteria", 8000)
    if value["correctness"].get("weights_policy", "as_stored") not in WEIGHTS_POLICIES:
        raise ValueError("weights_policy requires as_stored or transformable")
    budget = value["correctness"].get("quality_budget", {})
    keys(budget, set(), set(DEFAULT_QUALITY_BUDGET))
    for key, limit in budget.items():
        number(limit, key, 1 if key == "ppl_ratio_max" else 0, 1_000_000)
    if value["correctness"].get("exactness", "reported") not in EXACTNESS_MODES:
        raise ValueError("exactness requires reported or required")
    if value["correctness"]["required_gates"] != ["arithmetic", "functional", "resources"]:
        raise ValueError("Fixed arithmetic, functional and resource gates required")
    keys(value["objectives"], {"targets", "verification_level"})
    keys(value["objectives"]["targets"], set(), {"decode_tps", "prefill_tps", "request_ms"})
    for key, target in value["objectives"]["targets"].items():
        if key != "extensions":
            number(target, "target", .000001, 1_000_000_000)
    if value["objectives"]["verification_level"] not in {"reported", "artifact_verified"}:
        raise ValueError("Unknown evidence requirement")
    limits = value["resource_limits"]
    validate_limits(limits, "run")
    keys(limits["process_seconds"], {"discovery", "build", "check", "benchmark", "agent"})
    for cap in limits["process_seconds"].values():
        number(cap, "process cap", 1, 10800, integer=True)
    return value


def load_run(root):
    run = read_json(safe_path(root, "run.json"))
    header(run, "run", root.name, {"created_at", "spec", "request_sha256"})
    timestamp(run["created_at"], "created_at")
    validate_spec(run["spec"])
    if digest(run["spec"]) != run["request_sha256"]:
        raise ValueError("Run request identity changed")
    return run


def segment(root, run, segment_id):
    value = read_json(safe_path(root, "segments/" + ident(segment_id, "segment") + ".json"))
    header(value, "segment", run["run_id"], {"segment_id", "action", "started_at", "deadline_at", "request_sha256", "limits", "dependency_scope"},
           {"amendment"})
    if "amendment" in value:
        amendment(value["amendment"])
    if value["segment_id"] != segment_id or value["request_sha256"] != run["request_sha256"]:
        raise ValueError("Segment request identity mismatch")
    if timestamp(value["deadline_at"], "deadline_at") <= timestamp(value["started_at"], "started_at"):
        raise ValueError("Segment deadline must follow start")
    if value["action"] not in {"start", "resume"}:
        raise ValueError("Invalid segment action")
    validate_limits(value["limits"], "segment")
    if value["limits"]["process_seconds"] != run["spec"]["resource_limits"]["process_seconds"]:
        raise ValueError("Segment process caps differ from fixed request")
    elapsed = (timestamp(value["deadline_at"], "deadline") - timestamp(value["started_at"], "start")).total_seconds()
    if elapsed != value["limits"]["budget_minutes"] * 60:
        raise ValueError("Segment deadline does not match authorized budget")
    return value


def inputs(root, run):
    value = read_json(safe_path(root, "inputs.json"))
    header(value, "run_inputs", run["run_id"], {"created_at", "request_sha256", "model_manifest", "hardware_manifest", "environment_manifest", "case_manifests", "manifest_sha256"})
    timestamp(value["created_at"], "input creation")
    if value["request_sha256"] != run["request_sha256"] or value["manifest_sha256"] != digest({k: v for k, v in value.items() if k != "manifest_sha256"}):
        raise ValueError("Resolved input identity mismatch")
    for key in ("hardware_manifest", "environment_manifest"):
        if not isinstance(value[key], dict) or not value[key]:
            raise ValueError(key + " requires a content manifest")
    if set(value["case_manifests"]) != {case["case_id"] for case in run["spec"]["cases"]}:
        raise ValueError("Resolved cases differ from fixed request")
    for rows in value["case_manifests"].values():
        if not isinstance(rows, list) or not rows:
            raise ValueError("Captured case identity requires retained files")
        for row in rows:
            keys(row, {"path", "bytes", "sha256"})
            if not row["path"].startswith("captured/"):
                raise ValueError("Case input must be retained under captured/")
            path = evidence(root, {"path": row["path"], "sha256": row["sha256"]})
            if path.stat().st_size != row["bytes"]:
                raise ValueError("Captured input extent changed")
    return value


def evidence(root, ref):
    keys(ref, {"path", "sha256"})
    path = safe_path(root, ref["path"])
    if not path.is_file() or file_sha(path) != sha(ref["sha256"], "evidence sha256"):
        raise ValueError("Missing or changed retained evidence")
    return path


def candidate_projection(root, run, candidate_id, overlay=None):
    """Recompute from immutable retained artifacts/checks/benchmarks, never fastest samples."""
    overlay = overlay or {}
    def read(path):
        relative = path.relative_to(root).as_posix()
        return overlay[relative] if relative in overlay else read_json(path)
    def exists(path):
        return path.relative_to(root).as_posix() in overlay or path.exists()
    def byte_sha(path):
        relative = path.relative_to(root).as_posix()
        if relative in overlay:
            data = (json.dumps(overlay[relative], indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            return hashlib.sha256(data).hexdigest()
        return file_sha(path)
    def paths(directory, pattern):
        found = set(directory.glob(pattern))
        for relative in overlay:
            path = root / relative
            if directory.name == "attempts" and path.parent.parent == directory:
                found.add(path.parent)
            elif path.parent == directory and path.match(pattern):
                found.add(path)
        return sorted(found)
    folder = safe_path(root, "candidates/" + ident(candidate_id, "candidate"))
    candidate = read(safe_path(folder, "candidate.json"))
    required = {"candidate_id", "segment_id", "ordinal", "parent_candidate_id", "origin_candidate_id", "created_at", "label", "rationale", "strategy", "configuration", "store_decisions", "source_manifest_sha256", "inputs_manifest_sha256", "request_sha256"}
    header(candidate, "candidate", run["run_id"], required, {"weights"})
    if "weights" in candidate and candidate["weights"] not in WEIGHTS:
        raise ValueError("Candidate weights declaration requires as_stored or transformed")
    if candidate["candidate_id"] != candidate_id or candidate["request_sha256"] != run["request_sha256"]:
        raise ValueError("Candidate request/identity mismatch")
    number(candidate["ordinal"], "candidate ordinal", 1, 99_999_999, integer=True)
    if candidate["ordinal"] != int(candidate_id.split("-")[1]):
        raise ValueError("Candidate ordinal mismatch")
    text(candidate["label"], "label", 160)
    text(candidate["rationale"], "rationale", 8000)
    text(candidate["strategy"], "strategy", 2000)
    if not isinstance(candidate["configuration"], dict) or not isinstance(candidate["store_decisions"], dict):
        raise ValueError("Configuration and Inference decisions require objects")
    for decision in candidate["store_decisions"].values():
        keys(decision, {"status", "reason"})
        if decision["status"] not in {"implemented", "inherited", "rejected", "untested"}:
            raise ValueError("Invalid Inference decision")
        text(decision["reason"], "Inference decision reason")
    for key in ("parent_candidate_id", "origin_candidate_id"):
        if candidate[key] is not None:
            ident(candidate[key], "candidate")
            if int(candidate[key].split("-")[1]) >= candidate["ordinal"] or not safe_path(root, f"candidates/{candidate[key]}/candidate.json").is_file():
                raise ValueError("Lineage must reference an earlier retained candidate")
    bounds = segment(root, run, candidate["segment_id"])
    created = timestamp(candidate["created_at"], "candidate time")
    if not timestamp(bounds["started_at"], "segment start") <= created <= timestamp(bounds["deadline_at"], "segment deadline"):
        raise ValueError("Candidate lies outside its segment")
    resolved = inputs(root, run)
    if candidate["inputs_manifest_sha256"] != resolved["manifest_sha256"]:
        raise ValueError("Candidate resolved inputs changed")
    source = read(safe_path(folder, "source.manifest.json"))
    header(source, "source_manifest", run["run_id"], {"candidate_id", "files"})
    if source["candidate_id"] != candidate_id or digest(source) != candidate["source_manifest_sha256"]:
        raise ValueError("Candidate source manifest mismatch")
    verify_inventory(folder / "source", source["files"])
    artifacts = {"source": f"candidates/{candidate_id}/source", "source_manifest": f"candidates/{candidate_id}/source.manifest.json"}
    package_sha = None
    package_path = safe_path(folder, "package/manifest.json")
    if exists(package_path):
        package = read(package_path)
        header(package, "package_manifest", run["run_id"], {"candidate_id", "source_manifest_sha256", "configuration_sha256", "files", "build", "launch", "capabilities"})
        if (package["candidate_id"] != candidate_id or package["source_manifest_sha256"] != candidate["source_manifest_sha256"] or
                package["configuration_sha256"] != digest(candidate["configuration"])):
            raise ValueError("Package identity differs from frozen source/configuration")
        verify_inventory(folder / "package", package["files"], {"manifest.json"})
        keys(package["build"], {"toolchain", "dependencies", "command"})
        if not isinstance(package["launch"], list) or not package["launch"] or not all(isinstance(s, str) for s in package["launch"]):
            raise ValueError("Package launch recipe requires literal arguments")
        executable = safe_path(folder / "package", package["launch"][0])
        if not executable.is_file():
            raise ValueError("Package launch entry must be a retained file")
        if not isinstance(package["build"]["toolchain"], dict) or not package["build"]["toolchain"] or not isinstance(package["build"]["dependencies"], list):
            raise ValueError("Build recipe requires toolchain identity and dependency list")
        if not isinstance(package["build"]["command"], list) or not package["build"]["command"] or not all(isinstance(arg, str) for arg in package["build"]["command"]):
            raise ValueError("Build command requires literal arguments")
        if not isinstance(package["capabilities"], list) or not all(isinstance(capability, str) for capability in package["capabilities"]):
            raise ValueError("Package capabilities require a list")
        for dependency in package["build"]["dependencies"]:
            keys(dependency, {"name", "version", "path", "sha256"})
            text(dependency["name"], "dependency name", 200)
            text(dependency["version"], "dependency version", 200)
            dependency_path = safe_path(folder / "package", dependency["path"])
            if not dependency_path.is_file() or file_sha(dependency_path) != dependency["sha256"]:
                raise ValueError("Package dependency must bind retained bytes")
        package_sha = digest(package)
        artifacts["package"] = f"candidates/{candidate_id}/package"
        artifacts["package_manifest"] = f"candidates/{candidate_id}/package/manifest.json"
    unresolved = False
    lifecycle = "queued" if package_sha is None else "checking"
    failure = None
    attempt_count = 0
    attempts = []
    for attempt in paths(folder / "attempts", "attempt-*"):
        ident(attempt.name, "attempt")
        reservation = read(safe_path(attempt, "reservation.json"))
        header(reservation, "attempt_reservation", run["run_id"], {"segment_id", "candidate_id", "attempt_id", "operation", "reserved_at", "deadline_at", "inputs_manifest_sha256", "package_sha256", "cost"})
        if reservation["candidate_id"] != candidate_id or reservation["segment_id"] != candidate["segment_id"] or reservation["attempt_id"] != attempt.name:
            raise ValueError("Attempt owner mismatch")
        if reservation["inputs_manifest_sha256"] != resolved["manifest_sha256"]:
            raise ValueError("Attempt input identity mismatch")
        if reservation["operation"] not in {"build", "check", "benchmark"}:
            raise ValueError("Unknown candidate operation")
        if reservation["operation"] != "build" and reservation["package_sha256"] != package_sha:
            raise ValueError("Attempt package identity mismatch")
        if not created <= timestamp(reservation["reserved_at"], "reservation") <= timestamp(reservation["deadline_at"], "attempt deadline") <= timestamp(bounds["deadline_at"], "segment deadline"):
            raise ValueError("Attempt deadline outside segment")
        cap = bounds["limits"]["process_seconds"][reservation["operation"]]
        if (timestamp(reservation["deadline_at"], "deadline") - timestamp(reservation["reserved_at"], "reserved")).total_seconds() > cap:
            raise ValueError("Attempt exceeds its process cap")
        if digest(reservation["cost"]) != digest({"unit": "attempt", "amount": 1}):
            raise ValueError("Attempt counter reservation invalid")
        attempt_count += 1
        attempt_view = {"attempt_id": attempt.name, "operation": reservation["operation"], "deadline_at": reservation["deadline_at"],
                        "reservation": f"candidates/{candidate_id}/attempts/{attempt.name}/reservation.json", "outcome": None}
        attempts.append(attempt_view)
        result_path = safe_path(attempt, "result.json")
        if not exists(result_path):
            unresolved = True
            lifecycle = {"build": "building", "check": "checking", "benchmark": "benchmarking"}[reservation["operation"]]
            continue
        result = read(result_path)
        header(result, "attempt_result", run["run_id"], {"segment_id", "candidate_id", "attempt_id", "outcome", "ended_at", "exit_code", "cleanup_settled", "logs", "failure"})
        if any(result[k] != reservation[k] for k in ("segment_id", "candidate_id", "attempt_id")):
            raise ValueError("Attempt result owner mismatch")
        if result["outcome"] not in {"completed", "failed", "cancelled", "timed_out", "interrupted", "launch_failed"} or type(result["cleanup_settled"]) is not bool:
            raise ValueError("Invalid attempt outcome/settlement")
        if timestamp(result["ended_at"], "ended_at") < timestamp(reservation["reserved_at"], "reserved_at"):
            raise ValueError("Attempt ended before reservation")
        if result["outcome"] == "completed" and timestamp(result["ended_at"], "ended_at") > timestamp(reservation["deadline_at"], "deadline_at"):
            raise ValueError("Completed attempt exceeded deadline")
        if result["exit_code"] is not None:
            number(result["exit_code"], "exit_code", -(2 ** 31), 2 ** 32, integer=True)
        if result["outcome"] == "completed" and (result["exit_code"] != 0 or result["cleanup_settled"] is not True):
            raise ValueError("Completed attempt requires zero exit and settled cleanup")
        attempt_view.update(outcome=result["outcome"], cleanup_settled=result["cleanup_settled"], ended_at=result["ended_at"], logs=result["logs"])
        unresolved |= not result["cleanup_settled"]
        for ref in result["logs"]:
            evidence(root, ref)
        if result["outcome"] != "completed":
            lifecycle = result["outcome"] if result["outcome"] in LIFECYCLES else reservation["operation"] + "_failed"
            failure = {"operation": reservation["operation"], "message": text(result["failure"], "failure", 8000)}
        else:
            # The latest attempt decides: a completed retry supersedes an earlier failed attempt.
            lifecycle, failure = ("queued" if package_sha is None else "checking"), None
    checks = {}
    check_refs = {}
    for path in paths(folder / "checks", "check-*.json"):
        safe_path(folder, path.relative_to(folder).as_posix())
        check = read(path)
        header(check, "check", run["run_id"], {"candidate_id", "segment_id", "check_id", "case_id", "package_sha256", "gate", "criteria_sha256", "status", "rationale", "evidence", "evidence_level", "recorded_at"})
        ident(check["check_id"], "check")
        if (check["candidate_id"] != candidate_id or check["segment_id"] != candidate["segment_id"] or
                check["package_sha256"] != package_sha or package_sha is None or path.stem != check["check_id"]):
            raise ValueError("Check artifact/owner mismatch")
        if check["case_id"] not in resolved["case_manifests"] or check["gate"] not in run["spec"]["correctness"]["required_gates"]:
            raise ValueError("Check case or gate differs from request")
        if check["criteria_sha256"] != digest(run["spec"]["correctness"]):
            raise ValueError("Check criteria changed")
        if check["status"] not in {"passed", "failed", "unverified", "not_applicable"} or check["evidence_level"] not in {"reported", "artifact_verified"}:
            raise ValueError("Invalid check verdict/evidence level")
        text(check["rationale"], "check rationale", 8000)
        if not created <= timestamp(check["recorded_at"], "check time") <= timestamp(bounds["deadline_at"], "segment deadline"):
            raise ValueError("Check outside segment")
        if not isinstance(check["evidence"], list) or (check["status"] == "passed" and not check["evidence"]):
            raise ValueError("Passed check requires retained evidence")
        for ref in check["evidence"]:
            evidence(root, ref)
        checks[check["check_id"]] = check
        check_refs[check["check_id"]] = {"path": path.relative_to(root).as_posix(), "sha256": byte_sha(path)}
    series = []
    for path in paths(folder / "benchmarks", "benchmark-*.json"):
        safe_path(folder, path.relative_to(folder).as_posix())
        bench = read(path)
        header(bench, "benchmark", run["run_id"], {"candidate_id", "segment_id", "benchmark_id", "case_id", "package_sha256", "case_sha256", "protocol_sha256", "identities", "workload", "metrics", "checks", "outcome", "started_at", "ended_at", "missing_metrics"})
        ident(bench["benchmark_id"], "benchmark")
        case = next((c for c in run["spec"]["cases"] if c["case_id"] == bench["case_id"]), None)
        if (case is None or bench["candidate_id"] != candidate_id or bench["segment_id"] != candidate["segment_id"] or
                bench["package_sha256"] != package_sha or package_sha is None or path.stem != bench["benchmark_id"]):
            raise ValueError("Benchmark case/artifact/owner mismatch")
        expected = {"model_sha256": digest(resolved["model_manifest"]), "input_sha256": digest(resolved["case_manifests"][case["case_id"]]),
                    "hardware_sha256": digest(resolved["hardware_manifest"]), "environment_sha256": digest(resolved["environment_manifest"])}
        validate_workload(bench["workload"])
        if (bench["identities"] != expected or bench["workload"] != case["workload"] or bench["case_sha256"] != digest(case) or
                bench["protocol_sha256"] != digest(run["spec"]["benchmark_protocol"])):
            raise ValueError("Benchmark comparison identities changed")
        if not created <= timestamp(bench["started_at"], "benchmark start") <= timestamp(bench["ended_at"], "benchmark end") <= timestamp(bounds["deadline_at"], "segment deadline"):
            raise ValueError("Benchmark outside segment")
        if bench["outcome"] not in {"completed", "failed", "cancelled", "timed_out", "interrupted"}:
            raise ValueError("Invalid benchmark outcome")
        keys(bench["metrics"], set(), ALL_METRICS)
        missing = set(METRICS) - bench["metrics"].keys()
        keys(bench["missing_metrics"], missing, set(OPTIONAL_METRICS) - bench["metrics"].keys())
        for key in bench["missing_metrics"]:
            if key == "extensions":
                continue
            text(bench["missing_metrics"][key], key + " missing reason", 2000)
        metrics = {}
        for key, metric in bench["metrics"].items():
            if key == "extensions":
                continue
            keys(metric, {"samples", "unit"})
            if metric["unit"] != ALL_METRICS[key] or not isinstance(metric["samples"], list) or len(metric["samples"]) != case["workload"]["repetitions"]:
                raise ValueError("Metric units or repetition count differ from protocol")
            for sample in metric["samples"]:
                number(sample, key, metric_minimum(key),
                       2 ** 60, integer=key.endswith("bytes"))
            metrics[key] = metric | {"median": statistics.median(metric["samples"])}
        if not isinstance(bench["checks"], list):
            raise ValueError("Benchmark checks require references")
        bound_checks = []
        for ref in bench["checks"]:
            check_path = evidence(root, ref)
            check = checks.get(check_path.stem)
            if not check or ref != check_refs[check_path.stem] or check["case_id"] != case["case_id"]:
                raise ValueError("Benchmark check binding mismatch")
            bound_checks.append(check)
        gates = {}
        for gate in run["spec"]["correctness"]["required_gates"]:
            matching = [c for c in bound_checks if c["gate"] == gate]
            gates[gate] = "failed" if any(c["status"] == "failed" for c in matching) else "passed" if matching and all(c["status"] == "passed" for c in matching) else "unverified"
        verified = bool(bound_checks) and all(c["evidence_level"] == "artifact_verified" for c in bound_checks)
        memory_met = all(key in metrics and min(metrics[key]["samples"]) >= run["spec"]["hardware_constraints"][kind + "_headroom_bytes"]
                         for kind, key in (("vram", "minimum_free_vram_bytes"), ("ram", "minimum_free_ram_bytes")))
        gates["resources"] = "failed" if not memory_met else gates["resources"]
        qualified = all(v == "passed" for v in gates.values()) and (verified or run["spec"]["objectives"]["verification_level"] == "reported") and bench["outcome"] == "completed"
        target_met = qualified and all(key in metrics and (metrics[key]["median"] <= target if key.endswith("ms") else metrics[key]["median"] >= target)
                    for key, target in run["spec"]["objectives"]["targets"].items() if key != "extensions")
        group = digest({"identities": expected, "case": case, "protocol": run["spec"]["benchmark_protocol"], "correctness": run["spec"]["correctness"]})
        series.append({"benchmark_id": bench["benchmark_id"], "case_id": case["case_id"], "comparison_group_id": group,
                       "qualification": "qualified" if qualified else "exploratory", "evidence_level": "artifact_verified" if verified else "reported",
                       "metrics": metrics, "gates": gates, "target_met": target_met, "outcome": bench["outcome"],
                       "record": path.relative_to(root).as_posix(), "record_sha256": byte_sha(path), "missing_metrics": bench["missing_metrics"], "extensions": bench.get("extensions", {})})
    gates = {}
    for gate in run["spec"]["correctness"]["required_gates"]:
        matching = [c for c in checks.values() if c["gate"] == gate]
        gates[gate] = "failed" if any(c["status"] == "failed" for c in matching) else "passed" if matching and all(c["status"] == "passed" for c in matching) and all(any(c["case_id"] == case["case_id"] for c in matching) for case in run["spec"]["cases"]) else "unverified"
    if any(s["gates"]["resources"] == "failed" for s in series):
        gates["resources"] = "failed"
    if any(v != "passed" for v in gates.values()):
        for item in series:
            item["qualification"] = "exploratory"
            item["target_met"] = False
    eligible = bool(package_sha) and not unresolved and all(v == "passed" for v in gates.values()) and all(any(s["case_id"] == c["case_id"] for s in series) and all(s["qualification"] == "qualified" for s in series if s["case_id"] == c["case_id"]) for c in run["spec"]["cases"])
    policy_violation = run["spec"]["correctness"].get("weights_policy") == "as_stored" and candidate.get("weights") != "as_stored"
    if policy_violation:
        eligible = False
    target_satisfied = eligible and all(all(s["target_met"] for s in series if s["case_id"] == c["case_id"]) for c in run["spec"]["cases"])
    outcome_path = safe_path(folder, "outcome.json")
    if exists(outcome_path):
        outcome = read(outcome_path)
        header(outcome, "candidate_outcome", run["run_id"], {"candidate_id", "lifecycle", "failure", "closed_at"})
        if outcome["candidate_id"] != candidate_id or outcome["lifecycle"] not in LIFECYCLES - {"queued", "building", "checking", "benchmarking"}:
            raise ValueError("Invalid candidate outcome")
        timestamp(outcome["closed_at"], "outcome closed_at")
        if outcome["failure"] is not None:
            keys(outcome["failure"], {"message"}, {"operation"})
            text(outcome["failure"]["message"], "outcome failure", 8000)
        if unresolved:
            pending = ", ".join(a["attempt_id"] for a in attempts if a["outcome"] is None or not a.get("cleanup_settled", True))
            raise ValueError(f"Unsettled attempt cannot be sealed ({pending}); record its actual outcome and cleanup with "
                             "`crit run ... settle` first")
        lifecycle, failure = outcome["lifecycle"], outcome["failure"]
        if lifecycle == "completed" and (not package_sha or not series):
            raise ValueError("Completed candidate requires package and measurement")
        if lifecycle == "built" and not package_sha:
            raise ValueError("Built candidate requires a runnable package")
        if lifecycle != "completed":
            eligible = target_satisfied = False
    return record("candidate_telemetry", run["run_id"], candidate_id=candidate_id, segment_id=candidate["segment_id"],
                  ordinal=candidate["ordinal"], parent_candidate_id=candidate["parent_candidate_id"], origin_candidate_id=candidate["origin_candidate_id"],
                  label=candidate["label"], rationale=candidate["rationale"], strategy=candidate["strategy"], configuration=candidate["configuration"],
                  store_decisions=candidate["store_decisions"], lifecycle=lifecycle, evidence_level="artifact_verified" if package_sha and checks and all(c["evidence_level"] == "artifact_verified" for c in checks.values()) else "reported",
                  artifacts=artifacts, series=series, gates=gates, eligible=eligible, target_satisfied=target_satisfied,
                  identities={"request_sha256": run["request_sha256"], "source_manifest_sha256": candidate["source_manifest_sha256"],
                    "inputs_manifest_sha256": candidate["inputs_manifest_sha256"], "package_sha256": package_sha, "configuration_sha256": digest(candidate["configuration"])},
                  attempts=attempts, extensions=candidate.get("extensions", {}) | ({"ineligible_reason":
                      "the run requires weights used as stored; this candidate declares " + candidate.get("weights", "no weights treatment")}
                      if policy_violation else {}),
                  logs=[ref for path in paths(folder / "attempts", "attempt-*") if exists(path / "result.json") for ref in read(path / "result.json")["logs"]],
                  failure=failure, unresolved_cleanup=unresolved, attempt_count=attempt_count)


def read_candidates(root, run):
    candidates, errors = [], []
    folders = sorted(safe_path(root, "candidates").glob("candidate-*"))
    if len(folders) > MAX_CANDIDATES:
        errors.append({"file": "candidates", "error": "Candidate limit exceeded"})
        return [], errors
    for folder in folders:
        try:
            actual = candidate_projection(root, run, folder.name)
            stored = read_json(safe_path(folder, "telemetry.json"))
            revision = stored.pop("revision", None)
            number(revision, "telemetry revision", 1, 2 ** 31, integer=True)
            if digest(stored) != digest(actual):
                raise ValueError("Telemetry differs from retained records; coordinator refresh required")
            seal_path = safe_path(folder, "seal.json")
            sealed = seal_path.exists()
            if sealed:
                seal = read_json(seal_path)
                header(seal, "candidate_seal", run["run_id"], {"candidate_id", "sealed_at", "telemetry_revision", "files"})
                number(seal["telemetry_revision"], "seal revision", 1, 2 ** 31, integer=True)
                timestamp(seal["sealed_at"], "sealed_at")
                if seal["candidate_id"] != folder.name or seal["telemetry_revision"] != revision or not (folder / "outcome.json").is_file():
                    raise ValueError("Seal identity/revision mismatch")
                verify_inventory(folder, seal["files"], {"seal.json"})
            actual.update(revision=revision, sealed=sealed)
            if not sealed:
                actual["eligible"] = actual["target_satisfied"] = False
                for series in actual["series"]:
                    series["qualification"] = "exploratory"
                    series["target_met"] = False
            candidates.append(actual)
        except (ValueError, OSError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            errors.append({"file": "candidates/" + folder.name, "error": str(exc)})
    return candidates, errors
