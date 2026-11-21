"""crit run ... profile: where a served engine's time goes per token, by kernel family.

The candidate (a plugin package through crit serve, or an external engine from its launch command) is served with
crit-eval's kernel tracer in its processes' environment (crit_eval.limits.kernels: CUPTI on a CUDA device,
rocprofiler-sdk on a HIP one, the profiler the run's device has), and one short request of the run's case is sent
twice: once to warm (graphs and workspaces for its length), once traced. The traced request's prompt runs until its
first reply piece and its decode from there to its last. Each phase's kernels are charged their exclusive time
(crit_eval.limits.kernels.exclusive: a kernel overlapping an earlier one is charged from that one's end) and sorted
into kernel families by name (FAMILIES, then the rules --families gives ahead of them); a name no rule classes is
counted as other and listed. The device's memory copies join the copies family with the time the device only
copied (copy_terms' alone_ms), so the families sum to the time the device was busy. A decode step (one round of the
engine) is set against recon's plain-pass floor (work/recon.json's placement), the time one pass's weight reads take
at the device's bandwidth. Where no profiler can attach, the report says why. Keeps
candidates/ID/logs/profile-ATTEMPT.json under a check attempt; records no check and no benchmark.
"""
import json
from pathlib import Path
import sys
import tempfile
import time

TOKENS = 32   # the traced request's output tokens: enough decode steps to average, short enough to run in seconds
# Kernel families by name, checked in order on the lower-cased demangled name: the first family with a part the
# name holds classes it. Norms before attention (attn_norm is a norm) and routing before matrix products (a router's
# top-k is not a GEMV); copies first, since a copy kernel's name can hold any of the others'. Besides the common
# libraries' names, the parts name crit's own kernels as a trace of a library engine shows them: grouped-experts'
# `crit::techniques::grouped_experts::...`, quantized-projections' and the kernel core's `rows_kernel` / `tiles_kernel`
# and the activation quantizers ahead of a W4A4 or W8A8 product (matmul), paged-kv-attention's `_sparse_kernel`,
# `_window_kernel`, `_prefill_kernel` and their merges (attention), recurrent-scans' pool-delta and the gated
# recurrences (recurrence), fused-norms' stream sums (norms) and its fused activations beside torch's elementwise
# kernels (elementwise). Triton kernels are named by their function alone, so a plugin's own generic name still
# lands in other, listed for a --families rule.
FAMILIES = (("copies", ("memcpy", "memset", "copy", "transpose", "_cast", "cast_", "convert")),
            ("routing", ("topk", "top_k", "router", "routing", "moe_align", "gating", "expert_sort", "sort_tokens",
                         "permute", "combine")),
            ("norms", ("rmsnorm", "rms_norm", "layernorm", "layer_norm", "_norm", "norm_", "fused_add",
                       "stream_add", "stream_sum")),
            ("recurrence", ("recurren", "pool_delta", "delta_rule", "gated_delta", "softplus_decay", "conv1d",
                            "causal_conv", "mamba", "ssm_")),
            ("attention", ("attn", "attention", "flash", "fmha", "sdpa", "_attend", "rope", "rotary", "paged",
                           "_kv", "kv_", "softmax", "_sparse_kernel", "_window_kernel", "_prefill_kernel",
                           "merge_kernel", "sparse_indexer", "indexer")),
            ("matmul", ("gemv", "gemm", "matmul", "_mm_", "_mm", "mma", "cutlass", "cublas", "xmma", "linear",
                        "marlin", "stored_tile", "w4a", "w8a", "a16_", "dot", "dense", "grouped_experts",
                        "expert_kernel", "rows_kernel", "tiles_kernel", "quantize")),
            ("elementwise", ("act_mul", "swiglu", "silu", "gelu", "sigmoid", "elementwise")))
# The table's rows in order (matmul: GEMV and GEMM); a family only a --families rule names follows them.
ORDER = ("matmul", "attention", "recurrence", "norms", "elementwise", "routing", "copies", "other")
OTHER = 8   # the largest names classed other, listed per phase
BACKENDS = {"cuda": "CUPTI", "hip": "rocprofiler-sdk"}


def classify(name, rules=()):
    """(family, "rules" or "names") of a kernel name: the first rule ({match, family}) whose `match` the name holds,
    else the first of FAMILIES with a part the lower-cased name holds, else other."""
    for rule in rules:
        if rule["match"].lower() in name.lower():
            return rule["family"], "rules"
    lowered = name.lower()
    for family, parts in FAMILIES:
        if any(part in lowered for part in parts):
            return family, "names"
    return "other", "names"


def family_rules(value):
    """--families: [{"match": NAME_PART, "family": FAMILY}] as JSON text or a file, checked ahead of FAMILIES."""
    if value is None:
        return []
    path = Path(value)
    rules = json.loads(path.read_text(encoding="utf-8") if not value.lstrip().startswith("[") and path.is_file()
                       else value)
    if not isinstance(rules, list) or not all(
            isinstance(rule, dict) and set(rule) == {"match", "family"} and isinstance(rule["match"], str)
            and rule["match"] and isinstance(rule["family"], str) and rule["family"] for rule in rules):
        raise ValueError('--families is a list of {"match": NAME_PART, "family": FAMILY}')
    return rules


def ceiling(root):
    """recon's plain-pass floor for the run's placement (work/recon.json): {path, floor_ms, ceiling_tps}, the path the
    placement's experts take (the fastest way past the device for experts off it, else the device), or {reason}."""
    from .recon import RECON_JSON
    from .records import read_json, safe_path
    path = safe_path(root, RECON_JSON)
    placement = ((read_json(path).get("gathered") or {}).get("placement") or {}) if path.is_file() else {}
    floors = {key: ms for key, ms in (placement.get("plain_pass_floor_ms") or {}).items() if ms}
    if not floors:
        return {"reason": "recon recorded no plain-pass floor (run crit run ... recon with the run's device)"
                if path.is_file() else "the run has no recon.json (crit run ... recon)"}
    past = {key: ms for key, ms in floors.items() if key != "device"}
    key = min(past, key=past.get) if past else "device"
    return {"path": key, "floor_ms": floors[key], "ceiling_tps": 1000 / floors[key]}


def phase(kernels, copies, names, lo, hi, tokens, rules):
    """One phase's GPU time and launches by family over [lo, hi), per token of `tokens`, and the names classed
    other."""
    from crit_eval.limits.kernels import copy_terms, exclusive, short_name
    by = {family: {"ms": 0.0, "launches": 0} for family in ORDER}
    other, ruled = {}, 0.0
    for kernel, ns in exclusive([k for k in kernels if lo <= k[0] < hi]):
        name = names.get(kernel[4], kernel[4])
        family, source = classify(name, rules)
        row = by.setdefault(family, {"ms": 0.0, "launches": 0})
        row["ms"] += ns / 1e6
        row["launches"] += 1
        ruled += ns / 1e6 if source == "rules" else 0.0
        if family == "other":
            entry = other.setdefault(short_name(name), {"ms": 0.0, "launches": 0})
            entry["ms"] += ns / 1e6
            entry["launches"] += 1
    moved = copy_terms(copies, kernels, lo, hi)
    if moved:
        # A copy the device ran beside a kernel cost the step nothing the kernel did not: only its time alone counts.
        by["copies"]["ms"] += moved["alone_ms"]
        by["copies"]["launches"] += moved["count"]
    busy = sum(row["ms"] for row in by.values())
    launches = sum(row["launches"] for row in by.values())
    per = max(1, tokens)
    return {"wall_ms": (hi - lo) / 1e6, "tokens": tokens, "gpu_ms": busy, "launches": launches,
            "idle_ms": max(0.0, (hi - lo) / 1e6 - busy), "gpu_us_per_token": 1000 * busy / per,
            "launches_per_token": launches / per, "classed_by_rules_ms": ruled,
            "families": {family: {**row, "us_per_token": 1000 * row["ms"] / per,
                                  "launches_per_token": row["launches"] / per,
                                  "share": row["ms"] / busy if busy else 0.0} for family, row in by.items()},
            **({"copies": moved} if moved else {}),
            "other": [{"kernel": name, **row} for name, row in
                      sorted(other.items(), key=lambda item: -item[1]["ms"])[:OTHER]]}


def analyze(kernels, device, copies, mark, rules=(), floor=None):
    """The table of one traced request: its prompt phase (start to first reply piece, per prompt token) and decode
    phase (first to last, per decoded token and per decode step), the decode step against recon's plain-pass
    `floor` (ceiling())."""
    from crit_eval.limits.kernels import demangle
    names = demangle(k[4] for k in kernels)
    decoded = max(0, mark["output_tokens"] - 1)   # the first reply token ends the prompt's pass
    prompt = phase(kernels, copies, names, mark["start"], mark["first"], mark["prefill_tokens"], rules)
    decode = phase(kernels, copies, names, mark["first"], mark["end"], decoded, rules)
    steps = mark.get("rounds") or decoded
    decode["steps"] = steps
    if steps:
        decode["step_ms"] = decode["wall_ms"] / steps
        decode["gpu_step_ms"] = decode["gpu_ms"] / steps
    floor = floor or {}
    if steps and floor.get("floor_ms"):
        decode["against_floor"] = {**floor, "step_over_floor": decode["step_ms"] / floor["floor_ms"],
                                   "gpu_step_over_floor": decode["gpu_step_ms"] / floor["floor_ms"],
                                   "tps": 1000 * decoded / decode["wall_ms"] if decode["wall_ms"] else None}
    return {"device": device, "case_id": mark["case_id"], "prompt_tokens": mark["prompt_tokens"],
            "prefill_tokens": mark["prefill_tokens"], "output_tokens": mark["output_tokens"],
            "phases": {"prompt": prompt, "decode": decode}, "ceiling": floor}


def render(report):
    """The profile as a table, or why there is none."""
    if not report.get("available"):
        return f"crit run profile: no kernel profile: {report.get('reason')}"
    prompt, decode = report["phases"]["prompt"], report["phases"]["decode"]
    device = report.get("device") or {}
    lines = [f"profile of {report['candidate_id']} ({report['profiler']}"
             + (f" on {device['name']}" if device.get("name") else "") + f"), {report['case_id']}: "
             f"{report['prefill_tokens']:,} prompt tokens computed, {decode['tokens']:,} decoded in "
             f"{decode['steps']:,} steps",
             f"{'family':10s} {'prompt us/token':>16s} {'launches':>9s} {'share':>6s}   "
             f"{'decode us/token':>16s} {'launches':>9s} {'share':>6s}"]
    families = list(ORDER) + sorted((set(prompt["families"]) | set(decode["families"])) - set(ORDER))
    for family in families:
        p, d = prompt["families"].get(family), decode["families"].get(family)
        cells = [f"{row['us_per_token']:16,.1f} {row['launches_per_token']:9,.2f} {row['share']:6.1%}" if row else
                 f"{'-':>16s} {'-':>9s} {'-':>6s}" for row in (p, d)]
        lines.append(f"{family:10s} {cells[0]}   {cells[1]}")
    lines.append(f"{'GPU busy':10s} {prompt['gpu_us_per_token']:16,.1f} {prompt['launches_per_token']:9,.2f} "
                 f"{'':6s}   {decode['gpu_us_per_token']:16,.1f} {decode['launches_per_token']:9,.2f}")
    lines.append(f"{'wall':10s} {1000 * prompt['wall_ms'] / max(1, prompt['tokens']):16,.1f} {'':9s} {'':6s}   "
                 f"{1000 * decode['wall_ms'] / max(1, decode['tokens']):16,.1f}")
    if decode.get("step_ms") is not None:
        line = (f"decode step: {decode['step_ms']:.2f} ms wall, {decode['gpu_step_ms']:.2f} ms GPU busy, "
                f"{decode['launches'] / decode['steps']:,.1f} launches")
        floor = decode.get("against_floor")
        if floor:
            line += (f"; recon's plain pass ({floor['path']}) {floor['floor_ms']:.2f} ms: the step takes "
                     f"{floor['step_over_floor']:.2f}x it ({floor['gpu_step_over_floor']:.2f}x GPU busy)"
                     + (f", {floor['tps']:,.1f} tok/s against the plain-pass ceiling of {floor['ceiling_tps']:,.1f}"
                        if floor.get("tps") else ""))
        else:
            line += f"; no plain-pass ceiling: {(report.get('ceiling') or {}).get('reason')}"
        lines.append(line)
    for name, entry in (("prompt", prompt), ("decode", decode)):
        if entry["other"]:
            lines.append(f"{name} other by name: " + ", ".join(
                f"{row['kernel'][:60]} {row['ms']:,.2f} ms x{row['launches']}" for row in entry["other"])
                + " (class them with --families)")
    return "\n".join(lines)


class Tracer:
    """crit-eval's kernel tracer on `backend` (cuda: CUPTI, hip: rocprofiler-sdk) built from `toolkit`: the library
    the engine's processes load, the variables that load it, its clock and its records."""

    def __init__(self, backend, toolkit):
        self.backend, self.toolkit, self.name = backend, toolkit, BACKENDS[backend]

    def library(self, directory):
        from crit_eval.limits.kernels import build
        return build(self.toolkit, directory)

    def environment(self, library, directory):
        from crit_eval.limits.kernels import LOADERS
        return {LOADERS[self.backend]: str(library), "CRIT_KTRACE_DIR": str(directory),
                "CRIT_KTRACE_BACKEND": self.backend}

    def clock(self, library):
        from crit_eval.limits.kernels import CLOCKS
        from .external import _tracer_clock
        return _tracer_clock(library, CLOCKS[self.backend])

    def read(self, directory, clock):
        """(kernels, device, copies) the engine's processes recorded, once they have exited."""
        from crit_eval.limits.kernels import read_traces
        return read_traces(directory, clock, copies=True)


def profiler(run, nvcc=None, hipcc=None):
    """(None, Tracer) for the profiler the run's device has (its crit eval probe file's backend, cuda without one),
    else (why there is none, None)."""
    from .records import read_json
    from .store import run_device_file
    from crit_eval.limits.ceiling import device_backend
    path = run_device_file(run)
    try:
        backend = device_backend(read_json(Path(path))) if path and Path(path).is_file() else "cuda"
    except (ValueError, OSError):
        backend = "cuda"
    if backend not in BACKENDS:
        return (f"the run's device runs {backend}, which no kernel profiler crit-eval's tracer attaches to covers "
                "(CUPTI on cuda, rocprofiler-sdk on hip)"), None
    if not sys.platform.startswith("linux"):
        return (f"kernel tracing injects {BACKENDS[backend]} into the engine's processes, which needs Linux "
                f"(this is {sys.platform})"), None
    try:
        from crit_eval.limits.kernels import find_rocm, find_toolkit
        return None, Tracer(backend, find_toolkit(nvcc) if backend == "cuda" else find_rocm(hipcc))
    except SystemExit as missing:
        return str(missing), None
    except ImportError as missing:
        return f"crit-eval is not installed: {missing}", None


def profile(store, run_id, candidate_id, seconds=None, nvcc=None, hipcc=None, case_id=None, tokens=TOKENS,
            families=None, tracer=None):
    """crit run ... profile: the table of one short request of the run's case (`case_id`, default its first resident
    case) at `tokens` output tokens, traced on the profiler the run's device has. `families`: rules ahead of
    FAMILIES (family_rules). `tracer`: the Tracer to use instead (tests)."""
    from .attempt import Attempt, packaged, unsealed
    from .external import serving
    from .measure import _request, case_body, case_request
    from .records import inputs, load_run, write_json
    root = store.directory(run_id)
    run = load_run(root)
    folder = unsealed(root, candidate_id, "The candidate is sealed: profile it before sealing it")
    packaged(folder, "profiling", "crit run profile profiles")
    if not isinstance(tokens, int) or not 2 <= tokens <= 4096:
        raise ValueError("--tokens is a whole number from 2 to 4096")
    rules = list(families or [])
    resident = [case for case in run["spec"]["cases"] if case["workload"]["mode"] == "resident"]
    case = next((c for c in resident if case_id in (None, c["case_id"])), None)
    if case is None:
        raise ValueError(f"The run has no resident case {case_id}" if case_id else "The run has no resident case")
    resolved = inputs(root, run)
    request = {"case_id": case["case_id"], "output_tokens": tokens,
               "request": case_request(root, resolved["case_manifests"][case["case_id"]])}
    attempt = Attempt(store, run_id, candidate_id, "check", seconds, "profile", "crit run profile", "The profile")
    report = folder / "logs" / f"profile-{attempt.id}.json"
    found = {"attempt_id": attempt.id, "candidate_id": candidate_id, "case_id": case["case_id"], "available": False}
    with attempt.running() as log:
        reason = None
        if tracer is None:
            reason, tracer = profiler(run, nvcc, hipcc)
        if reason is not None:
            found["reason"] = reason
            log(f"crit run profile: no kernel profile: {reason}")
        else:
            with tempfile.TemporaryDirectory() as directory:
                library = tracer.library(directory)
                served = serving(store, run_id, candidate_id, attempt, run["spec"]["hardware_constraints"]["kv_tokens"],
                                 log, "The profile", environment=tracer.environment(library, directory))
                # Marks on the tracer's own clock, which its kernel records carry.
                now, clock = tracer.clock(library)
                with served.until(attempt.deadline_at):
                    _, _, port = served.open()
                    body = case_body(request, port)
                    _request(port, body, served.timeout)   # warmup: graphs and workspaces for this length
                    offset = now() - time.perf_counter_ns()
                    started, first, last, usage, telemetry = _request(port, body, served.timeout)
                    stamp = lambda t: offset + round(t * 1e9)   # noqa: E731
                    per_round = telemetry.get("tokens_per_round")   # crit serve's output tokens per decode round
                    mark = {"case_id": case["case_id"], "prompt_tokens": usage["prompt_tokens"],
                            "prefill_tokens": telemetry.get("prefill_tokens") or usage["prompt_tokens"],
                            "output_tokens": usage["completion_tokens"],
                            "rounds": round(usage["completion_tokens"] / per_round) if per_round else None,
                            "start": stamp(started), "first": stamp(first), "end": stamp(last)}
                    log(f"crit run profile: traced {case['case_id']}")
                # The tracer writes its records as each process exits: read them once the engine is gone.
                kernels, device, copies = tracer.read(directory, clock)
            if kernels or copies:
                found.update(available=True, profiler=tracer.name,
                             **analyze(kernels, device, copies, mark, rules, ceiling(root)))
            else:
                found["reason"] = ("no kernels were recorded: the engine may not run on the device, or a process of it "
                                   "dropped the tracer (a static device runtime, or an environment it does not pass to "
                                   "its workers)")
                log("crit run profile: " + found["reason"])
    write_json(report, found, immutable=True)
    result = attempt.settle(evidence=(report,))
    attempt.finish()
    return {"attempt": result, "report": report.relative_to(root).as_posix(), **found}
