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
# Drift a run allows against the Eval reference, spent on speed; matches crit_eval/protocol.json (quality.thresholds,
# and quality.mode_thresholds for as_quantized: twice preserve's, since 4-bit activation math swings KL far more).
DEFAULT_QUALITY_BUDGET = {"ppl_ratio_max": 1.05, "kld_mean_max": 0.05, "kld_p99_max": 0.5}
DEFAULT_QUALITY_BUDGETS = {"preserve": DEFAULT_QUALITY_BUDGET,
                           "as_quantized": {"ppl_ratio_max": 1.1, "kld_mean_max": 0.1, "kld_p99_max": 1.0}}
EXACTNESS_MODES = {"reported", "required"}  # must speculative output equal plain greedy?
# How Eval judges quality (crit-eval quality modes): against the checkpoint with exact activations, or the same with
# each set's limits widened by the drift of the checkpoint's own activation quantization.
QUALITY_MODES = ("preserve", "as_quantized")
# Runs created from this version on require a recorded plan (crit run ... plan) for every candidate; older runs
# have no plans field and keep plan_id optional.
NEW_RUN_PLANS = "required"
# Runs created from this version on rank plans with the decode limits crit computes from the Eval attempt's facts
# when Eval states none (crit.run.plans.computed_limits); older runs keep the agent's estimates.
NEW_RUN_PLAN_LIMITS = "computed"
# Runs created from this version on give every lever of a candidate's plan its fate in the proposal; older runs keep
# fates optional.
NEW_RUN_LEVER_FATES = "required"
# Runs created from this version on check plans with the fourth version of the plan rules (crit.run.plans.rules);
# older runs keep the rules they were created with.
NEW_RUN_PLAN_RULES = 4
WEIGHTS = {"as_stored", "transformed"}             # candidate: did it change them?
HASH_PARALLEL_BYTES = 256 * 1024 * 1024
GIB = 1024 ** 3
_verification_cache = ContextVar("crit_verification_cache", default=None)

RUN_ID = r"run-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}"
LIFECYCLES = {"queued", "building", "built", "checking", "benchmarking", "completed", "build_failed",
              "check_failed", "benchmark_failed", "eval_failed", "cancelled", "timed_out", "interrupted"}
# A candidate attempt's lifecycle while it runs. Eval (crit run eval) has no fixed process cap: its work scales with
# the model and the device, so it runs to the segment's deadline and bounds each request by the rates it measures.
RUNNING = {"build": "building", "check": "checking", "benchmark": "benchmarking", "eval": "benchmarking"}
UNCAPPED = {"eval"}
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
# A run started from a crit-hub engine (crit run create --from-engine) keeps the engine's published evidence here,
# and its first candidate declares this lineage to say it is the imported source re-measured.
ORIGIN_DIR = "captured/origin"
LINEAGE = "origin-engine"
COMMIT = r"[0-9a-f]{40}"
HTTPS_REPOSITORY = r"https://[A-Za-z0-9.-]+(/[A-Za-z0-9_.~-]+)+/?"


def validate_origin(value, stored=None):
    """A run's origin_engine (hub.describe_engine): the crit-hub engine it starts from. Evidence entries name their
    source file before the run exists (`source`) and their copy under captured/origin/ once it does (`path`);
    `stored` True or False requires one form, None accepts either."""
    keys(value, {"engine", "title", "origin", "summary", "contributor", "source", "card", "lock", "checkpoints",
                 "provenance", "evidence"})
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(value["engine"])):
        raise ValueError("origin_engine.engine is <owner>/<name>, the engine's folder in crit-hub")
    text(value["title"], "origin_engine.title", 400)
    if value["origin"] not in ("crit", "external"):
        raise ValueError("origin_engine.origin is crit or external")
    if value["summary"] is not None:
        text(value["summary"], "origin_engine.summary", 400)
    keys(value["contributor"], {"name", "repository"})
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(value["contributor"]["name"])) \
            or not re.fullmatch(HTTPS_REPOSITORY, str(value["contributor"]["repository"])):
        raise ValueError("origin_engine.contributor needs a name and an https repository")
    keys(value["source"], {"repository", "revision", "path"})
    if not re.fullmatch(HTTPS_REPOSITORY, str(value["source"]["repository"])) or not re.fullmatch(COMMIT, str(value["source"]["revision"])):
        raise ValueError("origin_engine.source needs an https repository and a full commit revision")
    if value["source"]["path"] is not None:
        text(value["source"]["path"], "origin_engine.source.path", 512)
    for name in ("card", "lock"):
        keys(value[name], {"path", "sha256"})
        text(value[name]["path"], f"origin_engine.{name}.path")
        sha(value[name]["sha256"], f"origin_engine.{name}.sha256")
    if not isinstance(value["checkpoints"], dict) or not value["checkpoints"]:
        raise ValueError("origin_engine.checkpoints lists the checkpoints the engine was measured on")
    for model, entry in value["checkpoints"].items():
        text(model, "origin_engine checkpoint id", 200)
        keys(entry, {"repository", "revision", "files", "files_sha256"})
        if not isinstance(entry["files"], list) or not entry["files"]:
            raise ValueError(f"origin_engine.checkpoints.{model}.files lists the checkpoint's files")
        for row in entry["files"]:
            keys(row, {"path", "bytes", "sha256"})
            text(row["path"], "checkpoint file path", 1024)
            number(row["bytes"], "checkpoint file bytes", 0, 2 ** 53, integer=True)
            sha(row["sha256"], "checkpoint file sha256")
        if digest(entry["files"]) != entry["files_sha256"]:
            raise ValueError(f"origin_engine.checkpoints.{model}.files_sha256 does not match its files")
    if not isinstance(value["provenance"], dict):
        raise ValueError("origin_engine.provenance is an object")
    if not isinstance(value["evidence"], list):
        raise ValueError("origin_engine.evidence is a list")
    for ref in value["evidence"]:
        keys(ref, {"file", "sha256", "measured_with", "gpu", "model"}, {"source", "path"})
        text(ref["file"], "origin evidence file", 512)
        sha(ref["sha256"], "origin evidence sha256")
        has_path, has_source = "path" in ref, "source" in ref
        if has_path == has_source:
            raise ValueError("origin_engine.evidence entries name either their source file or their stored path")
        if stored is True and not has_path or stored is False and not has_source:
            raise ValueError("origin_engine.evidence is " + ("not yet " if stored else "already ") + "stored in the run")
        if has_path and not str(ref["path"]).startswith(ORIGIN_DIR + "/"):
            raise ValueError(f"origin_engine.evidence is kept under {ORIGIN_DIR}/")
    return value


def quality_mode(correctness):
    """The run's quality mode (QUALITY_MODES): preserve unless the request chose as_quantized."""
    return correctness.get("quality_mode", "preserve")


def quality_terms(correctness, amendments=()):
    """The run's drift budget over its quality mode's defaults, and its exactness mode, with each resume's
    amendment applied in order."""
    budget = {**DEFAULT_QUALITY_BUDGETS[quality_mode(correctness)], **correctness.get("quality_budget", {})}
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


def read_json(path, *, max_bytes=MAX_JSON_BYTES):
    limit = "1 MiB" if max_bytes == MAX_JSON_BYTES else str(max_bytes) + " bytes"
    if not path.exists() and not path.is_symlink():
        # A relative path resolves against the current directory, not the run.
        raise ValueError(f"JSON file not found: {path} (from {Path.cwd()})" if not path.is_absolute()
                         else f"JSON file not found: {path}")
    if not path.is_file() or path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError("JSON must be a regular file under " + limit)
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError("JSON must be under " + limit)
    return json.loads(data.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def write_json(path, value, immutable=False, *, max_bytes=MAX_JSON_BYTES, temporary_directory=None):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > max_bytes:
        raise ValueError("JSON projection exceeds " + ("1 MiB" if max_bytes == MAX_JSON_BYTES else str(max_bytes) + " bytes"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=temporary_directory or path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
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


def eval_max_context(model, device, vram_headroom_bytes, reserve_bytes=None, requested=None, drafters=()):
    """Eval's max context (crit_eval/limits/context.py) for MODEL on DEVICE, a `crit eval probe` result, capped at the
    requested context. reserve_bytes replaces the protocol's reserve (device memory kept out of the KV cache for
    workspaces and speed techniques); drafters: companion drafter folders whose stored weights are kept out too."""
    from crit_eval import EVAL_DIR
    from crit_eval.limits.context import max_context
    protocol = EVAL_DIR / "protocol.json"
    if not protocol.is_file():
        raise ValueError("Deriving kv_tokens from a device needs Eval from a crit checkout; set kv_tokens explicitly")
    settings = json.loads(protocol.read_text(encoding="utf-8"))["context"]
    if reserve_bytes is not None:
        settings = {**settings, "reserve_gib": reserve_bytes / GIB}
    return max_context(model, device, vram_headroom_bytes, settings, requested, drafters=drafters)


def kv_capacity(data, model, vram_headroom_bytes, reserve_bytes=None):
    """The run's KV capacity in tokens of prompt plus output, and what it rests on. An explicit kv_tokens
    is taken as given. Otherwise the request names its device (a `crit eval probe` result) and the run is
    built for Eval's max context: the context the user wants (context_tokens, from settings.toml; default 262,144),
    capped by the model's positions and the KV cache that fits on the device next to the companion drafters the
    request names (drafters: folders a candidate may load beside the model)."""
    if "kv_tokens" in data:
        return number(data["kv_tokens"], "KV cache", 1, KV_TOKENS_MAX, integer=True), None
    if "device" not in data:
        raise ValueError("Request needs device (a crit eval probe result, so the run is built for Eval's max context) "
                         "or an explicit kv_tokens")
    device = Path(text(data["device"], "Device file")).expanduser().resolve()
    from crit_eval import DEFAULT_CONTEXT_TOKENS
    requested = number(data.get("context_tokens", DEFAULT_CONTEXT_TOKENS), "context_tokens", 1, KV_TOKENS_MAX, integer=True)
    drafters = data.get("drafters", [])
    if not isinstance(drafters, list) or len(drafters) > 8:
        raise ValueError("drafters is a list of up to 8 companion drafter folders")
    drafters = [Path(text(path, "Drafter folder")).expanduser().resolve() for path in drafters]
    for path in drafters:
        if not path.is_dir():
            raise ValueError(f"Drafter folder {path} does not exist on this computer")
    value = eval_max_context(model, read_json(device), vram_headroom_bytes, reserve_bytes, requested, drafters)
    if not value["tokens"]:
        raise ValueError("Eval cannot derive the max context: " + (value["reason"] or "; ".join(value["missing"])))
    tokens, bound = min(value["tokens"], KV_TOKENS_MAX), value["bound"] if value["tokens"] <= KV_TOKENS_MAX else "crit"
    basis = value.get("basis") or {}
    context = {"source": "eval", "bound": bound, "requested": requested,
               **({"note": value["note"]} if value.get("note") else {}), "model_limit": value["model_limit"],
               "model_limit_source": value["model_limit_source"], "device_limit": value["device_limit"],
               "kv_precision": value["kv_precision"], "kv_bytes_per_token": value["kv_bytes_per_token"],
               **{key: basis[key] for key in ("device", "memory_free_bytes", "memory_total_bytes", "reserve_bytes",
                                              "resident_weight_bytes", "expert_bytes", "fixed_state_bytes",
                                              "expert_bytes_in_device_memory_at_max", "companion_drafter_bytes",
                                              "companion_drafters") if key in basis},
               "device_file": str(device), "device_sha256": file_sha(device)}
    return number(tokens, "Eval's max context", 1, KV_TOKENS_MAX, integer=True), context


def request_spec(data):
    keys(data, {"model_path", "correctness_criteria"}, {"kv_tokens", "device", "vram_headroom_gib", "ram_headroom_gib",
         "context_reserve_gib", "context_tokens", "drafters",
         "budget_minutes", "workload", "cases", "benchmark_protocol", "targets", "verification_level", "attempt_limit",
         "weights_policy", "candidate_budget", "quality_budget", "exactness", "quality_mode", "eval_sampling",
         "origin_engine"})
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
                            **({"exactness": data["exactness"]} if "exactness" in data else {}),
                            **({"quality_mode": data["quality_mode"]} if "quality_mode" in data else {})},
            "objectives": {"targets": targets, "verification_level": level},
            "resource_limits": resource_limits(data),
            **({"eval_sampling": eval_sampling(data["eval_sampling"])} if "eval_sampling" in data else {}),
            **({"origin_engine": validate_origin(data["origin_engine"])} if "origin_engine" in data else {}),
            **({"plans": NEW_RUN_PLANS} if NEW_RUN_PLANS else {}),
            **({"plan_limits": NEW_RUN_PLAN_LIMITS} if NEW_RUN_PLANS and NEW_RUN_PLAN_LIMITS else {}),
            **({"lever_fates": NEW_RUN_LEVER_FATES} if NEW_RUN_PLANS and NEW_RUN_LEVER_FATES else {}),
            **({"plan_rules": NEW_RUN_PLAN_RULES} if NEW_RUN_PLANS and NEW_RUN_PLAN_LIMITS and NEW_RUN_PLAN_RULES else {})}
    validate_spec(spec)
    return spec


def eval_sampling(value):
    """The user's override of the sampling Eval's sampled suite measures, for the whole run: {"settings": {temperature,
    and any of top_p, top_k, min_p} or "temperature=0.6,top_p=0.95", "approved_by": who asked for it}. Without it
    Eval measures at the model's own sampling; the engine builder never sets it."""
    keys(value, {"settings", "approved_by"})
    text(value["approved_by"], "eval_sampling approved_by", 400)
    from crit_eval.quality.prepare import sampling_override
    return {"settings": sampling_override(value["settings"]), "approved_by": value["approved_by"]}


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
    keys(value, {"model_path", "hardware_constraints", "cases", "benchmark_protocol", "correctness", "objectives", "resource_limits"},
         {"eval_sampling", "origin_engine", "plans", "plan_limits", "lever_fates", "plan_rules"})
    if value.get("plans", "required") != "required":
        raise ValueError("plans is required when present")
    if value.get("plan_limits", "computed") != "computed" or ("plan_limits" in value and "plans" not in value):
        raise ValueError("plan_limits is computed when present, in a run that requires plans")
    if value.get("lever_fates", "required") != "required" or ("lever_fates" in value and "plans" not in value):
        raise ValueError("lever_fates is required when present, in a run that requires plans")
    if value.get("plan_rules", 2) not in (2, 3, 4) or ("plan_rules" in value and value.get("plan_limits") != "computed"):
        raise ValueError("plan_rules is 2, 3 or 4 when present, in a run that computes plan limits")
    if "eval_sampling" in value and eval_sampling(value["eval_sampling"]) != value["eval_sampling"]:
        raise ValueError("eval_sampling is not in its validated form")
    if "origin_engine" in value:
        validate_origin(value["origin_engine"])
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
    keys(value["correctness"], {"criteria", "required_gates"}, {"weights_policy", "quality_budget", "exactness",
                                                                  "quality_mode"})
    text(value["correctness"]["criteria"], "criteria", 8000)
    if value["correctness"].get("weights_policy", "as_stored") not in WEIGHTS_POLICIES:
        raise ValueError("weights_policy requires as_stored or transformable")
    budget = value["correctness"].get("quality_budget", {})
    keys(budget, set(), set(DEFAULT_QUALITY_BUDGET))
    for key, limit in budget.items():
        number(limit, key, 1 if key == "ppl_ratio_max" else 0, 1_000_000)
    if value["correctness"].get("exactness", "reported") not in EXACTNESS_MODES:
        raise ValueError("exactness requires reported or required")
    if quality_mode(value["correctness"]) not in QUALITY_MODES:
        raise ValueError("quality_mode requires preserve or as_quantized")
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
    header(value, "run_inputs", run["run_id"], {"created_at", "request_sha256", "model_manifest", "hardware_manifest", "environment_manifest", "case_manifests", "manifest_sha256"},
           {"origin_checkpoint"})
    timestamp(value["created_at"], "input creation")
    if "origin_checkpoint" in value:
        if "origin_engine" not in run["spec"]:
            raise ValueError("origin_checkpoint belongs to a run started from a crit-hub engine")
        keys(value["origin_checkpoint"], {"matched", "checkpoints"})
        if value["origin_checkpoint"]["matched"] is not None and value["origin_checkpoint"]["matched"] not in run["spec"]["origin_engine"]["checkpoints"]:
            raise ValueError("origin_checkpoint names a checkpoint the origin engine does not have")
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


def attempt_process_cap(bounds, candidate_id, operation):
    """Check the operation's owner and return its fixed process cap, None for an operation the segment's deadline
    alone bounds (UNCAPPED)."""
    if candidate_id is not None:
        ident(candidate_id, "candidate")
    if (not isinstance(operation, str) or operation not in set(bounds["limits"]["process_seconds"]) | UNCAPPED
            or (candidate_id is None) != (operation in {"agent", "discovery"})):
        raise ValueError("Invalid reservation operation/owner")
    return None if operation in UNCAPPED else bounds["limits"]["process_seconds"][operation]


def validate_attempt_reservation(root, run, value, *, bounds=None):
    """Common authority; callers may reuse a validated segment and additionally bind paths/artifacts."""
    header(value, "attempt_reservation", run["run_id"], {"segment_id", "candidate_id", "attempt_id",
           "operation", "reserved_at", "deadline_at", "inputs_manifest_sha256", "package_sha256", "cost"})
    ident(value["attempt_id"], "attempt")
    if bounds is None:
        bounds = segment(root, run, value["segment_id"])
    if value["segment_id"] != bounds["segment_id"]:
        raise ValueError("Reservation segment mismatch")
    cap = attempt_process_cap(bounds, value["candidate_id"], value["operation"])
    start = timestamp(value["reserved_at"], "reservation")
    deadline = timestamp(value["deadline_at"], "attempt deadline")
    if not (timestamp(bounds["started_at"], "segment start") <= start <= deadline
            <= timestamp(bounds["deadline_at"], "segment deadline")):
        raise ValueError("Reservation exceeds fixed segment bounds")
    if cap is not None and (deadline - start).total_seconds() > cap:
        raise ValueError("Attempt exceeds its process cap")
    if digest(value["cost"]) != digest({"unit": "attempt", "amount": 1}):
        raise ValueError("Invalid attempt counter reservation")
    return value


def validate_attempt_result(root, run_id, reservation, value):
    """Validate a result against its reservation before publication or when rebuilding state."""
    header(value, "attempt_result", run_id, {"segment_id", "candidate_id", "attempt_id", "outcome",
           "ended_at", "exit_code", "cleanup_settled", "logs", "failure"})
    if any(value[key] != reservation[key] for key in ("segment_id", "candidate_id", "attempt_id")):
        raise ValueError("Attempt result owner mismatch")
    if (not isinstance(value["outcome"], str) or
            value["outcome"] not in {"completed", "failed", "cancelled", "timed_out", "interrupted", "launch_failed"}
            or type(value["cleanup_settled"]) is not bool):
        raise ValueError("Invalid attempt outcome/settlement")
    ended = timestamp(value["ended_at"], "ended_at")
    if ended < timestamp(reservation["reserved_at"], "reserved_at"):
        raise ValueError("Attempt ended before reservation")
    if value["outcome"] == "completed" and ended > timestamp(reservation["deadline_at"], "deadline_at"):
        raise ValueError("Completed attempt exceeded deadline")
    if value["exit_code"] is not None:
        number(value["exit_code"], "exit_code", -(2 ** 31), 2 ** 32, integer=True)
    if value["outcome"] == "completed" and (value["exit_code"] != 0 or not value["cleanup_settled"]):
        raise ValueError("Completed attempt requires zero exit and settled cleanup")
    if not isinstance(value["logs"], list):
        raise ValueError("Logs require retained evidence references")
    for ref in value["logs"]:
        evidence(root, ref)
    # Candidate failures feed the lifecycle's failure message. Agent/discovery results may omit it.
    if value["failure"] is not None or (reservation["candidate_id"] is not None and value["outcome"] != "completed"):
        text(value["failure"], "failure", 8000)
    return value


def candidate_projection(root, run, candidate_id, overlay=None, *, candidate_root=None, package_root=None):
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
    if candidate_root is not None:
        folder = candidate_root
    candidate = read(safe_path(folder, "candidate.json"))
    required = {"candidate_id", "segment_id", "ordinal", "parent_candidate_id", "origin_candidate_id", "created_at", "label", "rationale", "strategy", "configuration", "store_decisions", "source_manifest_sha256", "inputs_manifest_sha256", "request_sha256"}
    header(candidate, "candidate", run["run_id"], required, {"weights", "lineage", "plan_id", "lever_fates"})
    if "lever_fates" in candidate and ("plan_id" not in candidate or not isinstance(candidate["lever_fates"], dict)):
        raise ValueError("Candidate lever_fates belong to the candidate's plan")
    if "plan_id" in candidate:
        ident(candidate["plan_id"], "plan")
    if "weights" in candidate and candidate["weights"] not in WEIGHTS:
        raise ValueError("Candidate weights declaration requires as_stored or transformed")
    if "lineage" in candidate and (candidate["lineage"] != LINEAGE or "origin_engine" not in run["spec"]):
        raise ValueError(f"Candidate lineage {LINEAGE} belongs to a run started from a crit-hub engine")
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
    package_root = package_root if package_root is not None else folder / "package"
    package_path = safe_path(folder, "package/manifest.json")
    if exists(package_path):
        package = read(package_path)
        header(package, "package_manifest", run["run_id"], {"candidate_id", "source_manifest_sha256", "configuration_sha256", "files", "build", "launch", "capabilities"})
        if (package["candidate_id"] != candidate_id or package["source_manifest_sha256"] != candidate["source_manifest_sha256"] or
                package["configuration_sha256"] != digest(candidate["configuration"])):
            raise ValueError("Package identity differs from frozen source/configuration")
        verify_inventory(package_root, package["files"], {"manifest.json"})
        keys(package["build"], {"toolchain", "dependencies", "command"})
        if not isinstance(package["launch"], list) or not package["launch"] or not all(isinstance(s, str) for s in package["launch"]):
            raise ValueError("Package launch recipe requires literal arguments")
        executable = safe_path(package_root, package["launch"][0])
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
            dependency_path = safe_path(package_root, dependency["path"])
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
        reservation = validate_attempt_reservation(root, run, read(safe_path(attempt, "reservation.json")), bounds=bounds)
        if reservation["candidate_id"] != candidate_id or reservation["segment_id"] != candidate["segment_id"] or reservation["attempt_id"] != attempt.name:
            raise ValueError("Attempt owner mismatch")
        if reservation["inputs_manifest_sha256"] != resolved["manifest_sha256"]:
            raise ValueError("Attempt input identity mismatch")
        if reservation["operation"] != "build" and reservation["package_sha256"] != package_sha:
            raise ValueError("Attempt package identity mismatch")
        if timestamp(reservation["reserved_at"], "reservation") < created:
            raise ValueError("Attempt precedes candidate creation")
        attempt_count += 1
        attempt_view = {"attempt_id": attempt.name, "operation": reservation["operation"], "deadline_at": reservation["deadline_at"],
                        "reservation": f"candidates/{candidate_id}/attempts/{attempt.name}/reservation.json", "outcome": None}
        attempts.append(attempt_view)
        result_path = safe_path(attempt, "result.json")
        if not exists(result_path):
            unresolved = True
            lifecycle = RUNNING[reservation["operation"]]
            continue
        result = validate_attempt_result(root, run["run_id"], reservation, read(result_path))
        attempt_view.update(outcome=result["outcome"], cleanup_settled=result["cleanup_settled"], ended_at=result["ended_at"], logs=result["logs"])
        unresolved |= not result["cleanup_settled"]
        if result["outcome"] != "completed":
            lifecycle = result["outcome"] if result["outcome"] in LIFECYCLES else reservation["operation"] + "_failed"
            failure = {"operation": reservation["operation"], "message": result["failure"]}
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


def exploratory_reason(item):
    """Why a benchmark series (telemetry `series` entry) does not qualify."""
    unbound = [g for g, v in item["gates"].items() if v == "unverified"]
    failed = [g for g, v in item["gates"].items() if v == "failed"]
    if item["outcome"] != "completed":
        return f"its outcome is {item['outcome']}"
    if failed:
        return "gate " + ", ".join(failed) + " failed" + (" (free memory below the run's headroom)" if "resources" in failed else "")
    if unbound:
        return ("it cites no passed check for " + ", ".join(unbound) +
                " (publish the case's checks first and list their records in the benchmark's checks)")
    return "its cited checks are reported, not artifact_verified"


def ineligibility(run, candidate):
    """Why a candidate (its view from RunStore.get) is not eligible, one entry per blocker, so seal and select can
    say what to fix. Derived when asked and never stored, so telemetry keeps the projection earlier runs retain."""
    if candidate["eligible"]:
        return []
    if not candidate.get("sealed"):
        return ["it is not sealed"]
    reasons = []
    if not candidate["identities"]["package_sha256"]:
        reasons.append("no package is retained")
    if candidate["unresolved_cleanup"]:
        reasons.append("an attempt's cleanup is unresolved")
    for gate, status in candidate["gates"].items():
        if status != "passed":
            reasons.append(f"required gate {gate} is {status}" + (" (some case has no check for it)" if status == "unverified" else ""))
    for case in run["spec"]["cases"]:
        mine = [s for s in candidate["series"] if s["case_id"] == case["case_id"]]
        if not mine:
            reasons.append(f"{case['case_id']} has no benchmark")
        for item in mine:
            if item["qualification"] != "qualified":
                reasons.append(f"{item['benchmark_id']} ({case['case_id']}) is exploratory: {exploratory_reason(item)}")
    # The weights policy is the one reason telemetry records itself.
    policy = (candidate.get("extensions") or {}).get("ineligible_reason")
    if policy:
        reasons.append(policy)
    if candidate["lifecycle"] != "completed":
        reasons.append(f"the candidate was sealed {candidate['lifecycle']}, not completed")
    return reasons


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
