"""Single-writer run coordinator; records are data, never executable recipes."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import os
import json
from pathlib import Path
import platform
import shutil
import socket
import tempfile
import time
import uuid

from .records import (
    HEADROOM_REASONS, LIFECYCLES, LINEAGE, MAX_CANDIDATES, SAFETY_WINDOW_MINUTES, STOP_REASONS, WEIGHTS, amendment,
    attempt_process_cap,
    candidate_projection, quality_terms, digest, evidence, file_sha, header, headroom_record, ident,
    ineligibility, inputs, inventory, keys, load_run, now, number, read_candidates, read_json, record,
    request_spec, safe_path, segment, text, timestamp, validate_attempt_reservation, validate_attempt_result,
    validate_spec, verify_inventory, write_json,
)
from . import cost, hub, plans, recon

ROOT = Path(__file__).resolve().parents[3]


def lease_owner():
    """Identity written into the lease: a PID is only meaningful on its own host and OS."""
    return {"pid": os.getpid(), "host": socket.gethostname(), "system": platform.system(), "started_at": now()}


def read_lease_owner(path):
    """Lease owner from the lock file; a legacy lock holds only the PID."""
    raw = path.read_text(encoding="utf-8").strip()
    if raw.isdigit():
        return {"pid": int(raw), "host": None, "system": None, "started_at": None}
    value = json.loads(raw)
    keys(value, {"pid", "host", "system", "started_at"})
    number(value["pid"], "lease owner PID", 1, 2 ** 32, integer=True)
    return value


@contextmanager
def lease(root, *, recover=False):
    """Pin ownership while checking/removing a dead lease and throughout every write."""
    from crit_core.machine import _try_lock, _unlock
    # Never unlink this guard: replacing its inode would split the kernel lock.
    # The separate owner lease still excludes writers across Windows/WSL.
    with safe_path(root, ".coordinator.guard").open("a+b") as guard:
        if not _try_lock(guard):
            raise ValueError("Run coordinator is busy; another writer or recoverer holds its guard")
        try:
            if recover:
                release_dead_lease(root)
            with exclusive_lease(root):
                yield
        finally:
            _unlock(guard)


@contextmanager
def exclusive_lease(root):
    """An abandoned exclusive lease blocks writers until explicit dead-owner recovery."""
    path = safe_path(root, ".coordinator.lock")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            owner = read_lease_owner(path)
            held = f" (held by PID {owner['pid']}" + (f" on {owner['host']}, {owner['system']}, since {owner['started_at']})" if owner["host"] else ")")
        except (OSError, ValueError):
            held = ""
        raise ValueError(f"Run coordinator is busy{held}; if that process has exited, run recover on the same host and OS") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(lease_owner()))
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        path.unlink()


def release_dead_lease(root):
    """Called only while holding the persistent coordinator guard."""
    lock_path = safe_path(root, ".coordinator.lock")
    if not lock_path.exists():
        return
    owner = read_lease_owner(lock_path)
    if owner["host"] is not None and (owner["host"], owner["system"]) != (socket.gethostname(), platform.system()):
        # Another OS (for example WSL versus Windows) has its own PID space:
        # a missing PID here proves nothing about the owner.
        raise ValueError(f"Lease owner ran on {owner['host']} ({owner['system']}); recover from that host and OS")
    pid = owner["pid"]
    if os.name == "nt":
        import ctypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.restype = ctypes.c_void_p
        api.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        handle = api.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
        if handle:
            api.CloseHandle.argtypes = [ctypes.c_void_p]
            api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            api.WaitForSingleObject.restype = ctypes.c_ulong
            try:
                # An exited process can still be opened while another process
                # retains its handle. Signalled state proves it has terminated.
                if api.WaitForSingleObject(handle, 0) != 0:
                    raise ValueError("Coordinator owner may still be alive; recovery blocked")
            finally:
                api.CloseHandle(handle)
        elif ctypes.get_last_error() != 87:
            raise ValueError("Cannot prove coordinator owner exited")
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("Coordinator owner is still alive")
    lock_path.unlink()


def next_id(directory, kind, suffix=""):
    values = []
    for path in directory.glob(kind + "-*" + suffix):
        value = path.name[:-len(suffix)] if suffix else path.name
        ident(value, kind)
        values.append(int(value.split("-")[1]))
    return f"{kind}-{max(values, default=0) + 1:04d}"


def copy_package(source, destination, rows, retained_source, source_rows):
    """Copy a package, hard-linking files whose bytes equal a file of the candidate's retained source (same
    size and sha256), so large shared files such as draft-only weights are stored once. Both trees stay
    inventoried byte for byte; a filesystem without hard links falls back to a copy. Windows keeps directory
    metadata of hard-linked files stale until a file is opened, which verification would read as a change, so
    packages are always copied there."""
    by_digest = {(r["bytes"], r["sha256"]): retained_source / r["path"] for r in source_rows}
    identity = {r["path"]: (r["bytes"], r["sha256"]) for r in rows}

    def copy(src, dst):
        relative = Path(src).relative_to(source).as_posix()
        twin = by_digest.get(identity.get(relative))
        if twin is not None and twin.is_file() and os.name != "nt":
            try:
                os.link(twin, dst)
                return dst
            except OSError:
                pass
        return shutil.copy2(src, dst)
    shutil.copytree(source, destination, copy_function=copy)


@contextmanager
def staging(root, kind):
    """Unpublished bytes live outside candidate inventories, even after abrupt exit."""
    # A source may be the entire work/ directory. Stage beside it, never inside
    # it; callers already reject a source containing the run root itself.
    scratch = Path(tempfile.mkdtemp(dir=root, prefix=f".{kind}-"))
    try:
        yield scratch / kind
    finally:
        # Revalidate the exact scratch path before recursive cleanup.
        path = safe_path(root, scratch.name)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def publish_directory(source, destination):
    """The coordinator lease serializes publication; Windows readers may hold a handle."""
    deadline = time.monotonic() + .25
    while True:
        try:
            source.rename(destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                raise
            time.sleep(.005)


def amendments(root, segment_id):
    """The quality-term amendments of the run's segments up to segment_id, in order."""
    return [s["amendment"] for s in (read_json(p) for p in sorted((root / "segments").glob("segment-*.json")))
            if "amendment" in s and s["segment_id"] <= segment_id]


def described(model_path):
    """Whether the model has description files (tokenizer, templates, config): a folder, or a weight file whose
    metadata derives them (a GGUF, crit_core.model.open_model)."""
    try:
        from crit_core.model import open_model
        source = open_model(model_path)
    except Exception:
        return Path(model_path).is_dir()
    return source.folder is not None or bool(source.derived)


def serving_gaps(capabilities, model_path):
    """What a chat-resident-v1 package leaves out that crit serve clients rely on: per-request temperature, and
    tool calls when the model's chat template teaches a call format crit parses."""
    if not isinstance(capabilities, list) or "chat-resident-v1" not in capabilities:
        return []
    gaps = []
    if "chat-sampling-temperature-v1" not in capabilities:
        gaps.append("it must apply a request's temperature and declare chat-sampling-temperature-v1")
    if not {"chat-tools-v1", "chat-tool-template-v1"} & set(capabilities) and described(model_path):
        from crit_core.model import open_model
        from crit_core.text import template_sources, templates_tool_format
        try:
            taught = templates_tool_format(template_sources(open_model(model_path).files).values())
        except Exception:
            taught = None
        if taught:
            gaps.append("the model's chat template teaches tool calls, so it must render tools, parse the calls and "
                        "declare chat-tools-v1")
    return gaps


class RunStore:
    def __init__(self, root, project=ROOT):
        self.project = Path(project).resolve()
        if not (self.project / "core/include/crit/core.h").is_file():
            raise ValueError("Select a crit repository with --project")
        requested = Path(root).expanduser().absolute()
        if requested.resolve() != requested or requested.is_symlink():
            raise ValueError("Run root must not traverse symlinks/reparse points")
        self.root = requested.resolve()
        if self.root == self.project or any(self.root.is_relative_to(self.project / name)
               for name in ("docs", "tools", "tests", "core", "inference", "kernels", "eval", "api", "cli", ".github", "archive")):
            raise ValueError("Runs belong outside publication and protected retention")
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, run_id):
        path = safe_path(self.root, ident(run_id, "run"))
        if not path.is_dir():
            raise ValueError("Unknown run")
        return path

    def get(self, run_id):
        root = self.directory(run_id)
        run = load_run(root)
        state = read_json(safe_path(root, "state.json"))
        header(state, "run_state", run_id, {"revision", "updated_at", "current_segment_id", "phase", "candidates", "baseline_candidate_id",
            "proposed_candidate_id", "selected_candidate_id", "target_satisfied", "budget", "stopping_reason", "critique", "opportunities", "cleanup", "message"},
            {"plans"})
        number(state["revision"], "state revision", 1, 2 ** 31, integer=True)
        timestamp(state["updated_at"], "state update")
        if state["phase"] not in LIFECYCLES | STOP_REASONS | {"awaiting_agent", "running", "discovering", "finished", "failed", "launch_failed", "expired"}:
            raise ValueError("Unknown run phase")
        keys(state["budget"], {"segments", "attempts", "segment_attempts", "remaining_seconds"})
        for value in state["budget"].values():
            number(value, "budget projection", 0, 2 ** 31, integer=True)
        if not isinstance(state["opportunities"], list):
            raise ValueError("Ranked opportunities require a list")
        if state["critique"] is not None:
            text(state["critique"], "critique", 16000)
        bounds = segment(root, run, state["current_segment_id"])
        candidates, errors = read_candidates(root, run)
        expected_summaries = [{"candidate_id": c["candidate_id"], "telemetry_revision": c["revision"]} for c in candidates]
        if state["candidates"] != expected_summaries:
            errors.append({"file": "state.json", "error": "Candidate summaries/revisions are stale; coordinator refresh required"})
        if state["cleanup"] not in {"owned", "external", "settled", "unresolved"} or state["stopping_reason"] not in STOP_REASONS | {None}:
            raise ValueError("Invalid run cleanup/stopping projection")
        for key in ("baseline_candidate_id", "proposed_candidate_id", "selected_candidate_id"):
            if state[key] is not None:
                ident(state[key], "candidate")
        by_id = {c["candidate_id"]: c for c in candidates}
        selected = by_id.get(state["selected_candidate_id"])
        # Always derive selection satisfaction from valid retained bytes, including missing/mutated artifacts.
        state["target_satisfied"] = bool(selected and selected["target_satisfied"])
        if state["stopping_reason"] == "satisfied" and not state["target_satisfied"]:
            errors.append({"file": "state.json", "error": "Selected artifact no longer satisfies this run"})
        remaining = max(0, (timestamp(bounds["deadline_at"], "deadline") - datetime.now(timezone.utc)).total_seconds())
        state["budget"]["remaining_seconds"] = round(remaining)
        if not remaining and state["phase"] in {"awaiting_agent", "running", "discovering", "building", "checking", "benchmarking"}:
            state["phase"] = "expired"
        prompt_path = safe_path(root, f"segments/{bounds['segment_id']}/agent_request.md")
        return {"run": run, "state": state, "segment": bounds, "candidates": candidates, "errors": errors,
                "folder": str(root), "prompt": prompt_path.read_text(encoding="utf-8")}

    def list(self):
        items, errors = [], []
        for path in sorted(self.root.glob("run-*/run.json"), reverse=True):
            try:
                data = self.get(path.parent.name)
                items.append({"run_id": data["run"]["run_id"], "created_at": data["run"]["created_at"],
                              "model_path": data["run"]["spec"]["model_path"], "phase": data["state"]["phase"]})
            except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
                errors.append({"file": path.parent.name, "error": str(exc)})
        return {"runs": items[:100], "errors": errors}

    def _state(self, root, **changes):
        run = load_run(root)
        path = safe_path(root, "state.json")
        state = read_json(path) if path.exists() else record("run_state", run["run_id"], revision=0,
            current_segment_id="segment-0001", phase="awaiting_agent", candidates=[], baseline_candidate_id=None,
            proposed_candidate_id=None, selected_candidate_id=None, target_satisfied=False, budget={}, stopping_reason=None,
            critique=None, opportunities=[], cleanup="external", message="")
        state.update(changes)
        candidates, errors = read_candidates(root, run)
        if errors:
            # Recoverable partial publication is retained and reported, not deleted.
            state["message"] = "; ".join(error["error"] for error in errors)[:8000]
        state["candidates"] = [{"candidate_id": c["candidate_id"], "telemetry_revision": c["revision"]} for c in candidates]
        if "phase" not in changes and state["stopping_reason"] is None and candidates and state["phase"] not in {"finished", "failed", "launch_failed", "timed_out", "interrupted", "cancelled"}:
            state["phase"] = candidates[-1]["lifecycle"]
        sealed = [c for c in candidates if c["sealed"]]
        state["baseline_candidate_id"] = sealed[0]["candidate_id"] if sealed else None
        selected = next((c for c in candidates if c["candidate_id"] == state["selected_candidate_id"]), None)
        state["target_satisfied"] = bool(selected and selected["target_satisfied"])
        state["plans"] = plans.summaries(root, run)
        segments = sorted((root / "segments").glob("segment-*.json"))
        attempts = self._attempts(root)
        bounds = segment(root, run, state["current_segment_id"])
        state["budget"] = {"segments": len(segments), "attempts": len(attempts),
            "segment_attempts": sum(a["segment_id"] == bounds["segment_id"] for a in attempts),
            "remaining_seconds": max(0, round((timestamp(bounds["deadline_at"], "deadline") - datetime.now(timezone.utc)).total_seconds()))}
        state["revision"] += 1
        state["updated_at"] = now()
        write_json(path, state)

    def _event(self, root, event_type, candidate_id=None, detail=None, reference=None):
        run = load_run(root)
        state = read_json(root / "state.json")
        events = safe_path(root, "events")
        event_id = next_id(events, "event", ".json")
        value = record("event", run["run_id"], event_id=event_id, sequence=int(event_id.split("-")[1]),
            recorded_at=now(), segment_id=state["current_segment_id"], candidate_id=candidate_id,
            event_type=event_type, reference=reference, detail=detail or {})
        write_json(events / (event_id + ".json"), value, immutable=True)

    def _segment(self, root, run, action, minutes, attempt_limit, candidates=None, amend=None):
        limits = {key: value for key, value in run["spec"]["resource_limits"].items() if key != "candidate_budget"}
        if candidates is None and action == "start":
            candidates = run["spec"]["resource_limits"].get("candidate_budget")
        if candidates is not None:
            minutes = SAFETY_WINDOW_MINUTES if minutes is None else minutes
            limits |= {"candidate_budget": number(candidates, "Candidate budget", 1, 64, integer=True),
                       "budget_minutes": number(minutes, "Safety window", 1, SAFETY_WINDOW_MINUTES, integer=True)}
        else:
            if minutes is None:
                raise ValueError("Resume requires --candidates or --minutes")
            limits |= {"budget_minutes": number(minutes, "Budget", 1, 180, integer=True)}
        limits["attempt_limit"] = number(attempt_limit, "Attempt limit", 1, 1000, integer=True)
        segment_id = next_id(root / "segments", "segment", ".json")
        started = datetime.now(timezone.utc)
        bounds = record("segment", run["run_id"], segment_id=segment_id, action=action, started_at=started.isoformat(),
            deadline_at=(started + timedelta(minutes=minutes)).isoformat(), request_sha256=run["request_sha256"],
            limits=limits, dependency_scope="Local construction dependencies; preserve source, notices and exact identities outside publication",
            **({"amendment": amendment(amend)} if amend else {}))
        write_json(root / "segments" / (segment_id + ".json"), bounds, immutable=True)
        folder = root / "segments" / segment_id
        folder.mkdir()
        prompt = self.prompt(root, run, bounds)
        (folder / "agent_request.md").write_text(prompt, encoding="utf-8", newline="\n")
        if action == "start":
            (root / "agent_request.md").write_text(prompt, encoding="utf-8", newline="\n")
        return bounds

    def create(self, data, origin=None):
        """A new run; with `origin`, a crit-hub engine folder the run starts from (hub.describe_engine), whose
        published evidence is kept under captured/origin/."""
        if origin is not None:
            if not isinstance(data, dict):
                raise ValueError("A request object is required")
            if "origin_engine" in data:
                raise ValueError("Name the origin engine once: --from-engine or the request's origin_engine")
            data = {**data, "origin_engine": hub.describe_engine(origin)}
        spec = request_spec(data)
        while True:
            run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
            root = self.root / run_id
            try:
                root.mkdir()
                break
            except FileExistsError:
                continue
        with lease(root):
            for name in ("segments", "events", "candidates", "work", "captured"):
                (root / name).mkdir()
            if "origin_engine" in spec:
                spec["origin_engine"] = hub.store_evidence(spec["origin_engine"], root)
                validate_spec(spec)
            run = record("run", run_id, created_at=now(), spec=spec, request_sha256=digest(spec))
            write_json(root / "run.json", run, immutable=True)
            bounds = self._segment(root, run, "start", spec["resource_limits"]["budget_minutes"], spec["resource_limits"]["attempt_limit"])
            self._state(root, current_segment_id=bounds["segment_id"])
            self._event(root, "run_created", detail={"request_sha256": run["request_sha256"]})
        return self.get(run_id)

    def _active(self, root):
        run = load_run(root)
        state = read_json(root / "state.json")
        bounds = segment(root, run, state["current_segment_id"])
        if state["stopping_reason"] or datetime.now(timezone.utc) >= timestamp(bounds["deadline_at"], "deadline"):
            raise ValueError("Segment is closed/expired; explicit Resume is required")
        return run, state, bounds

    def _attempts(self, root):
        paths = list((root / "candidates").glob("candidate-*/attempts/attempt-*/reservation.json"))
        paths += list((root / "segments").glob("segment-*/attempts/attempt-*/reservation.json"))
        result = []
        for path in paths:
            safe_path(root, path.relative_to(root).as_posix())
            run = load_run(root)
            value = validate_attempt_reservation(root, run, read_json(path))
            owner = f"candidates/{value['candidate_id']}" if value["candidate_id"] else f"segments/{value['segment_id']}"
            if path.relative_to(root).as_posix() != f"{owner}/attempts/{value['attempt_id']}/reservation.json":
                raise ValueError("Reservation owner/path mismatch")
            outcome_path = safe_path(root, f"{owner}/attempts/{value['attempt_id']}/result.json")
            if outcome_path.exists():
                validate_attempt_result(root, run["run_id"], value, read_json(outcome_path))
            result.append(value)
        for segment_id in {value["segment_id"] for value in result}:
            if sum(value["segment_id"] == segment_id for value in result) > segment(root, load_run(root), segment_id)["limits"]["attempt_limit"]:
                raise ValueError("Retained attempts exceed their authorized segment cap")
        return result

    def resume(self, run_id, minutes=None, attempt_limit=64, external_cleanup_confirmed=False, candidates=None, amend=None):
        """amend: {quality_budget?, exactness?, approved_by} changes the run's quality terms from this segment on."""
        if type(external_cleanup_confirmed) is not bool:
            raise ValueError("External cleanup confirmation requires a boolean")
        root = self.directory(run_id)
        with lease(root):
            run = load_run(root)
            previous = read_json(root / "state.json")
            for reservation in self._attempts(root):
                owner = (f"candidates/{reservation['candidate_id']}" if reservation["candidate_id"] else f"segments/{reservation['segment_id']}")
                path = safe_path(root, f"{owner}/attempts/{reservation['attempt_id']}/result.json")
                if not path.exists() or read_json(path).get("cleanup_settled") is not True:
                    raise ValueError(f"Unsettled process ownership blocks Resume: {owner}/attempts/{reservation['attempt_id']} has "
                                     "no settled result; record its actual outcome and cleanup with `crit run ... settle` first")
            if previous["cleanup"] != "settled" and external_cleanup_confirmed is not True:
                raise ValueError("Confirm that the external agent has stopped before Resume")
            if previous["cleanup"] == "owned":
                raise ValueError("Stop the owned runner before Resume")
            existing, errors = read_candidates(root, run)
            if errors or any(not c["sealed"] for c in existing):
                raise ValueError("Retain and seal unfinished candidate outcomes before Resume")
            self._event(root, "segment_closed", detail={"cleanup": "settled", "external_user_confirmation": external_cleanup_confirmed,
                "previous_segment_id": previous["current_segment_id"]})
            if amend:
                amendment(amend)
            bounds = self._segment(root, run, "resume", minutes, attempt_limit, candidates, amend)
            self._state(root, current_segment_id=bounds["segment_id"], phase="awaiting_agent", stopping_reason=None, cleanup="external", message="")
            self._event(root, "segment_resumed", detail={"segment_id": bounds["segment_id"]})
        return self.get(run_id)

    def freeze_inputs(self, run_id, data):
        root = self.directory(run_id)
        with lease(root):
            run, _, _ = self._active(root)
            keys(data, {"hardware_manifest", "environment_manifest", "case_manifests"})
            if set(data["case_manifests"]) != {c["case_id"] for c in run["spec"]["cases"]}:
                raise ValueError("Case identities differ from request")
            for case_rows in data["case_manifests"].values():
                if not isinstance(case_rows, list) or not case_rows:
                    raise ValueError("Case requires captured input file identities")
                for row in case_rows:
                    keys(row, {"path", "bytes", "sha256"})
                    path = evidence(root, {"path": row["path"], "sha256": row["sha256"]})
                    if not row["path"].startswith("captured/") or path.stat().st_size != row["bytes"]:
                        raise ValueError("Inputs must reference exact retained captured bytes")
            for key in ("hardware_manifest", "environment_manifest"):
                if not isinstance(data[key], dict) or not data[key]:
                    raise ValueError(key + " must be a nonempty object")
            model = Path(run["spec"]["model_path"])
            model_rows = inventory(model) if model.is_dir() else [{"path": model.name, "bytes": model.stat().st_size, "sha256": file_sha(model)}]
            if not model_rows:
                raise ValueError("Model locator contains no files")
            if "origin_engine" in run["spec"]:
                # Is this the exact checkpoint the origin engine was measured on? Recorded, never required.
                data = {**data, "origin_checkpoint": hub.checkpoint_match(run["spec"]["origin_engine"], model_rows)}
            value = record("run_inputs", run_id, created_at=now(), request_sha256=run["request_sha256"], model_manifest=model_rows, **data)
            value["manifest_sha256"] = digest(value)
            write_json(root / "inputs.json", value, immutable=True)
            self._state(root, phase="building")
            self._event(root, "inputs_frozen", reference={"path": "inputs.json", "sha256": file_sha(root / "inputs.json")})
        return value

    def _refresh(self, root, candidate_id):
        projection = candidate_projection(root, load_run(root), candidate_id)
        path = safe_path(root, f"candidates/{candidate_id}/telemetry.json")
        try:
            previous = read_json(path)["revision"] if path.exists() else 0
            number(previous, "projection revision", 0, 2 ** 31, integer=True)
        except (ValueError, OSError, KeyError, TypeError):
            state = read_json(root / "state.json")
            previous = next((c["telemetry_revision"] for c in state["candidates"] if c["candidate_id"] == candidate_id), 0)
        revision = previous + 1
        write_json(path, projection | {"revision": revision})

    def _publish(self, root, candidate_id, relative, value):
        folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
        if safe_path(folder, "seal.json").exists():
            raise ValueError("Sealed candidate is immutable; allocate a new candidate")
        # Validate projected semantics with the proposed record before exclusive publication.
        path = safe_path(root, f"candidates/{candidate_id}/{relative}")
        candidate_projection(root, load_run(root), candidate_id, {path.relative_to(root).as_posix(): value})
        write_json(path, value, immutable=True)
        self._published(root, candidate_id, path, value)
        return value

    def _published(self, root, candidate_id, path, value):
        self._refresh(root, candidate_id)
        self._state(root)
        self._event(root, value["record_kind"], candidate_id, reference={"path": path.relative_to(root).as_posix(), "sha256": file_sha(path)})

    @staticmethod
    def _segment_candidates(root, run, bounds):
        candidates, _ = read_candidates(root, run)
        return sum(c.get("segment_id") == bounds["segment_id"] for c in candidates)

    def candidate(self, run_id, source, proposal):
        root = self.directory(run_id)
        with lease(root):
            run, _, bounds = self._active(root)
            resolved = inputs(root, run)
            keys(proposal, {"label", "rationale", "strategy", "configuration", "store_decisions"},
                 {"parent_candidate_id", "origin_candidate_id", "weights", "lineage", "plan_id", "lever_fates"})
            if "weights" in proposal and proposal["weights"] not in WEIGHTS:
                raise ValueError("weights requires as_stored or transformed")
            if "lineage" in proposal:
                if proposal["lineage"] != LINEAGE:
                    raise ValueError(f"lineage is {LINEAGE}: the candidate is the origin engine's source as imported")
                if "origin_engine" not in run["spec"]:
                    raise ValueError(f"lineage {LINEAGE} needs a run created with --from-engine")
            if run["spec"]["correctness"].get("weights_policy") == "as_stored" and "weights" not in proposal:
                raise ValueError("This run requires weights used as stored: declare weights as_stored or transformed")
            if "plan_id" in proposal:
                plan = plans.check_link(root, run, proposal["plan_id"], proposal.get("parent_candidate_id"))
                required = run["spec"].get("lever_fates") == "required"
                if required or "lever_fates" in proposal:
                    plans.check_fates(plan, proposal.get("lever_fates", {}), required)
            elif "lever_fates" in proposal:
                raise ValueError("lever_fates give the fate of each lever of the candidate's plan: name its plan_id")
            elif run["spec"].get("plans") == "required":
                raise ValueError("This run requires a plan for every candidate: record it with crit run ... plan --record "
                                 "plan.json after the parent's Eval and before any code, and name its plan_id in the proposal")
            if len(list((root / "candidates").glob("candidate-*"))) >= MAX_CANDIDATES:
                raise ValueError("Candidate limit exceeded")
            budget = bounds["limits"].get("candidate_budget")
            if budget is not None and self._segment_candidates(root, run, bounds) >= budget:
                raise ValueError(f"This segment's candidate budget ({budget}) is used; stop, or Resume with more candidates")
            source = Path(source).absolute()
            if root.is_relative_to(source):
                raise ValueError("Source folder must not contain its run")
            rows = inventory(source)
            if not rows:
                raise ValueError("Candidate needs complete retained source")
            # The model recon's sampling, chat template and stop tokens are frozen into every candidate.
            frozen, recon_ref = recon.load(root, run)
            if frozen is None and described(run["spec"]["model_path"]):
                raise ValueError(f"Run recon before the first candidate: crit run --id {run_id} recon writes "
                                 f"{recon.RECON_JSON}, then finish {recon.RECON_MD}")
            # Plan rules 3: decode's drafting plan and prefill's arithmetic plan are written before any candidate.
            missing = recon.unfinished(root) if run["spec"].get("plan_rules", 1) >= 3 and frozen is not None else []
            if missing:
                raise ValueError(f"Finish {recon.RECON_MD} before the candidate: " + ", ".join(missing) + " (still TODO "
                                 "or missing). The prefill arithmetic plan names each matrix group's prompt activation "
                                 "format and the quality measurement per group (crit eval run --attribute-drift)")
            configuration, extensions = proposal["configuration"], proposal.get("extensions", {})
            if frozen is not None:
                if not isinstance(configuration, dict) or configuration.get("recon", frozen) != frozen:
                    raise ValueError(f"configuration.recon differs from {recon.RECON_JSON}; change its frozen block, with "
                                     f"the reason in {recon.RECON_MD}, before the candidate")
                configuration, extensions = {**configuration, "recon": frozen}, {**extensions, "recon": recon_ref}
            candidate_id = next_id(root / "candidates", "candidate")
            folder = root / "candidates" / candidate_id
            # Validate complete staged records and source before publishing the candidate ID.
            source_manifest = record("source_manifest", run_id, candidate_id=candidate_id, files=rows)
            value = record("candidate", run_id, candidate_id=candidate_id, segment_id=bounds["segment_id"], ordinal=int(candidate_id.split("-")[1]),
                parent_candidate_id=proposal.get("parent_candidate_id"), origin_candidate_id=proposal.get("origin_candidate_id"), created_at=now(),
                **{k: proposal[k] for k in ("label", "rationale", "strategy", "store_decisions")}, configuration=configuration,
                **({"weights": proposal["weights"]} if "weights" in proposal else {}),
                **({"lineage": proposal["lineage"]} if "lineage" in proposal else {}),
                **({"plan_id": proposal["plan_id"]} if "plan_id" in proposal else {}),
                **({"lever_fates": proposal["lever_fates"]} if "lever_fates" in proposal else {}),
                extensions=extensions, source_manifest_sha256=digest(source_manifest), inputs_manifest_sha256=resolved["manifest_sha256"], request_sha256=run["request_sha256"])
            with staging(root, "candidate") as staged:
                staged.mkdir()
                shutil.copytree(source, staged / "source")
                write_json(staged / "source.manifest.json", source_manifest, immutable=True)
                write_json(staged / "candidate.json", value, immutable=True)
                candidate_projection(root, run, candidate_id, candidate_root=staged)
                publish_directory(staged, folder)
            # The agent may have prepared the candidate's work folder before creating it.
            (root / "work" / candidate_id).mkdir(exist_ok=True)
            self._refresh(root, candidate_id)
            self._state(root)
            self._event(root, "candidate_created", candidate_id, detail={"parent_candidate_id": value["parent_candidate_id"], "rationale": value["rationale"],
                                                                         **({"plan_id": value["plan_id"]} if "plan_id" in value else {})})
        return value

    def plan(self, run_id, data):
        """Record a candidate plan (plans.py): checked against the parent's newest Eval, which it freezes, with the
        priorities crit computes. Never changed afterwards; a revision is a new plan that supersedes it."""
        root = self.directory(run_id)
        with lease(root):
            run, _, bounds = self._active(root)
            value = plans.build(root, run, bounds, data)
            plan_id = next_id(safe_path(root, "plans"), "plan")
            value = record("plan", run_id, plan_id=plan_id, **value)
            path = safe_path(root, f"plans/{plan_id}/plan.json")
            write_json(path, value, immutable=True)
            self._state(root)
            self._event(root, "plan_recorded", reference={"path": f"plans/{plan_id}/plan.json", "sha256": file_sha(path)},
                        detail={"plan_id": plan_id, "decision": value["decision"], "parent_candidate_id": value["parent_candidate_id"],
                                "supersedes": value["supersedes"], "priorities": value["priorities"]})
        return value

    def plan_outcome(self, run_id, candidate_id, attempt_id):
        """After a retained Eval attempt of a candidate created from a plan: the plan's forecast next to what Eval
        measured, at plans/PLAN_ID/outcomes/ATTEMPT.json. None when the candidate has no plan."""
        root = self.directory(run_id)
        with lease(root):
            run = load_run(root)
            candidate = read_json(safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}/candidate.json"))
            if candidate.get("plan_id") is None:
                return None
            plan = plans.load(root, run, candidate["plan_id"])
            result = safe_path(root, f"candidates/{candidate_id}/eval/{ident(attempt_id, 'attempt')}.json")
            reference = {"path": result.relative_to(root).as_posix(), "sha256": file_sha(result)}
            value = plans.outcome(run, plan, candidate_id, attempt_id, reference, plans.read_status(result),
                                  candidate.get("lever_fates"))
            path = safe_path(root, f"plans/{plan['plan_id']}/outcomes/{attempt_id}.json")
            write_json(path, value, immutable=True)
            self._event(root, "plan_outcome_recorded", candidate_id,
                        reference={"path": path.relative_to(root).as_posix(), "sha256": file_sha(path)},
                        detail={"plan_id": plan["plan_id"], "flags": [flag["kpi"] for flag in value["flags"]]})
        return value

    def recon(self, run_id, force=False, drafters=()):
        root = self.directory(run_id)
        with lease(root):
            value = recon.write(root, load_run(root), force, drafters)
            self._event(root, "model_recon", reference={"path": recon.RECON_JSON, "sha256": file_sha(root / recon.RECON_JSON)})
        return value

    def recon_compare(self, run_id, other):
        """What another checkpoint changes for an engine built for this run's model (recon.compare_with_run)."""
        root = self.directory(run_id)
        return recon.compare_with_run(root, load_run(root), Path(other).expanduser().resolve())

    def package(self, run_id, candidate_id, source, recipe):
        root = self.directory(run_id)
        with lease(root):
            run, _, _ = self._active(root)
            folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
            if (folder / "seal.json").exists():
                raise ValueError("Sealed candidate is immutable")
            candidate = read_json(folder / "candidate.json")
            keys(recipe, {"build", "launch", "capabilities"})
            source = Path(source).absolute()
            if root.is_relative_to(source):
                raise ValueError("Package folder must not contain its run")
            rows = inventory(source)
            if any(r["path"] == "manifest.json" for r in rows):
                raise ValueError("Package input must not contain coordinator manifest.json")
            gaps = serving_gaps(recipe.get("capabilities"), run["spec"]["model_path"])
            if gaps:
                raise ValueError("The resident chat adapter is incomplete: " + "; ".join(gaps)
                                 + ". crit_core.serving.ResidentChat provides all of it (docs/SERVING.md)")
            # The package carries the candidate's configuration so exports and launches
            # can read it; a supplied copy must be exactly that configuration.
            supplied = source / "configuration.json"
            if supplied.is_file() and read_json(supplied) != candidate["configuration"]:
                raise ValueError("Package configuration.json differs from the candidate configuration")
            destination = safe_path(folder, "package")
            manifest_path = safe_path(folder, "package/manifest.json")
            with staging(root, "package") as staged:
                copy_package(source, staged, rows, folder / "source", read_json(folder / "source.manifest.json")["files"])
                verify_inventory(staged, rows)
                if not supplied.is_file():
                    (staged / "configuration.json").write_text(
                        json.dumps(candidate["configuration"], indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
                    rows = inventory(staged)
                value = record("package_manifest", run_id, candidate_id=candidate_id, source_manifest_sha256=candidate["source_manifest_sha256"],
                    configuration_sha256=digest(candidate["configuration"]), files=rows, **recipe)
                candidate_projection(root, run, candidate_id,
                    {manifest_path.relative_to(root).as_posix(): value}, package_root=staged)
                if destination.exists():
                    if not manifest_path.exists() or read_json(manifest_path) != value:
                        raise ValueError("Package already exists with different or incomplete contents; retain it and use a new candidate")
                    # An identical retry finishes projections without replacing published bytes.
                    candidate_projection(root, run, candidate_id)
                else:
                    write_json(staged / "manifest.json", value, immutable=True)
                    publish_directory(staged, destination)
            self._published(root, candidate_id, manifest_path, value)
        return value

    def append(self, run_id, candidate_id, kind, data):
        root = self.directory(run_id)
        with lease(root):
            run = load_run(root)  # Result publication grants no new process execution.
            candidate = read_json(safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}/candidate.json"))
            if kind not in {"check", "benchmark"}:
                raise ValueError("Append requires check or benchmark")
            directory = root / "candidates" / candidate_id / ("checks" if kind == "check" else "benchmarks")
            if kind == "benchmark" and isinstance(data, dict) and not data.get("checks"):
                data = {**data, "checks": self._case_checks(root, candidate_id, data.get("case_id"), data.get("package_sha256"))}
            record_id = next_id(directory, kind, ".json")
            value = record(kind, run_id, candidate_id=candidate_id, segment_id=candidate["segment_id"], **{kind + "_id": record_id}, **data)
            return self._publish(root, candidate_id, f"{directory.name}/{record_id}.json", value)

    @staticmethod
    def _case_checks(root, candidate_id, case_id, package_sha256):
        """A benchmark that cites no checks binds the case's latest check per gate for its package: the checks
        recorded before it, so publication order no longer decides whether it can qualify."""
        latest = {}
        for path in sorted((root / "candidates" / candidate_id / "checks").glob("check-*.json")):
            check = read_json(path)
            if check.get("case_id") == case_id and check.get("package_sha256") == package_sha256:
                latest[check.get("gate")] = path
        return [{"path": path.relative_to(root).as_posix(), "sha256": file_sha(path)} for path in sorted(latest.values())]

    def reserve(self, run_id, candidate_id, operation, seconds=None):
        root = self.directory(run_id)
        with lease(root):
            run, _, bounds = self._active(root)
            cap = attempt_process_cap(bounds, candidate_id, operation)
            if sum(a["segment_id"] == bounds["segment_id"] for a in self._attempts(root)) >= bounds["limits"]["attempt_limit"]:
                raise ValueError("Segment attempt limit exhausted")
            window = (timestamp(bounds["deadline_at"], "deadline") - datetime.now(timezone.utc)).total_seconds()
            limit = window if cap is None else cap
            seconds = limit if seconds is None else number(seconds, "Process deadline", .001, max(.001, limit))
            owner = f"candidates/{ident(candidate_id, 'candidate')}" if candidate_id else f"segments/{bounds['segment_id']}"
            directory = safe_path(root, owner + "/attempts")
            attempt_id = next_id(directory, "attempt")
            reserved = datetime.now(timezone.utc)
            package_path = safe_path(root, owner + "/package/manifest.json")
            value = record("attempt_reservation", run_id, segment_id=bounds["segment_id"], candidate_id=candidate_id, attempt_id=attempt_id,
                operation=operation, reserved_at=reserved.isoformat(), deadline_at=min(reserved + timedelta(seconds=seconds), timestamp(bounds["deadline_at"], "deadline")).isoformat(),
                inputs_manifest_sha256=inputs(root, run)["manifest_sha256"] if candidate_id else None,
                package_sha256=digest(read_json(package_path)) if package_path.exists() else None, cost={"unit": "attempt", "amount": 1})
            written = directory / attempt_id / "reservation.json"
            try:
                if candidate_id:
                    self._publish(root, candidate_id, f"attempts/{attempt_id}/reservation.json", value)
                else:
                    validate_attempt_reservation(root, run, value, bounds=bounds)
                    write_json(written, value, immutable=True)
                    self._state(root, phase="running" if operation == "agent" else "discovering", **({"cleanup": "owned"} if operation == "agent" else {}))
                    self._event(root, "agent_reserved", reference={"path": f"{owner}/attempts/{attempt_id}/reservation.json", "sha256": file_sha(written)})
            except BaseException:
                self._retract_reservation(root, candidate_id, owner, attempt_id, written)
                raise
        return value

    def _retract_reservation(self, root, candidate_id, owner, attempt_id, path):
        """Remove a reservation that reserve() could not finish publishing. No process was started for it, and
        an event that already references it keeps it (the reservation was observable)."""
        if not path.exists():
            return
        reference = f"{owner}/attempts/{attempt_id}/reservation.json"
        for event in sorted((root / "events").glob("event-*.json")):
            try:
                if (read_json(event).get("reference") or {}).get("path") == reference:
                    return
            except (OSError, ValueError):
                return  # unreadable history: keep the reservation and let the operator settle it
        try:
            path.chmod(0o600)
            path.unlink()
            path.parent.rmdir()
        except OSError:
            return
        try:
            if candidate_id:
                self._refresh(root, candidate_id)
            self._state(root)
        except (OSError, ValueError):
            pass

    def settle(self, run_id, reservation, data):
        root = self.directory(run_id)
        with lease(root):
            candidate_id = reservation["candidate_id"]
            owner = f"candidates/{candidate_id}" if candidate_id else f"segments/{reservation['segment_id']}"
            path = safe_path(root, f"{owner}/attempts/{reservation['attempt_id']}/reservation.json")
            if read_json(path) != reservation:
                raise ValueError("Reservation changed")
            keys(data, {"outcome", "ended_at", "exit_code", "cleanup_settled", "logs", "failure"})
            value = record("attempt_result", run_id, **{k: reservation[k] for k in ("segment_id", "candidate_id", "attempt_id")}, **data)
            if candidate_id:
                self._publish(root, candidate_id, f"attempts/{reservation['attempt_id']}/result.json", value)
            else:
                validate_attempt_reservation(root, load_run(root), reservation)
                validate_attempt_result(root, run_id, reservation, value)
                write_json(path.with_name("result.json"), value, immutable=True)
                phase = ("finished" if reservation["operation"] == "agent" else "building") if value["outcome"] == "completed" else value["outcome"]
                cleanup = ("settled" if reservation["operation"] == "agent" else "external") if value["cleanup_settled"] else "unresolved"
                self._state(root, phase=phase, cleanup=cleanup, message=value["failure"] or ("Runner ended; inspect candidate gates" if reservation["operation"] == "agent" else ""))
                self._event(root, "agent_settled", detail={"outcome": value["outcome"], "cleanup_settled": value["cleanup_settled"]})
        return value

    def seal(self, run_id, candidate_id, lifecycle, failure=None):
        root = self.directory(run_id)
        with lease(root):
            folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
            outcome_path = safe_path(folder, "outcome.json")
            seal_path = safe_path(folder, "seal.json")
            if outcome_path.exists():
                outcome = read_json(outcome_path)
                if digest({key: outcome[key] for key in ("lifecycle", "failure")}) != digest({"lifecycle": lifecycle, "failure": failure}):
                    raise ValueError("Candidate outcome already exists with different lifecycle/failure")
            else:
                outcome = record("candidate_outcome", run_id, candidate_id=candidate_id, lifecycle=lifecycle, failure=failure, closed_at=now())
            if seal_path.exists():
                # A retry repairs only run projections/history, never a sealed inventory.
                candidates, errors = read_candidates(root, load_run(root))
                if not any(c["candidate_id"] == candidate_id for c in candidates):
                    error = next((e["error"] for e in errors if e["file"] == f"candidates/{candidate_id}"), "invalid candidate")
                    raise ValueError("Cannot retry an invalid seal: " + error)
                value = read_json(seal_path)
            else:
                if outcome_path.exists():
                    self._published(root, candidate_id, outcome_path, outcome)
                else:
                    self._publish(root, candidate_id, "outcome.json", outcome)
                revision = read_json(folder / "telemetry.json")["revision"]
                value = record("candidate_seal", run_id, candidate_id=candidate_id, sealed_at=now(), telemetry_revision=revision, files=inventory(folder))
                # A killed writer can leave its temporary hard link behind. Keep it
                # outside the inventory that this very record seals.
                write_json(seal_path, value, immutable=True, temporary_directory=safe_path(root, "work"))
            self._state(root)
            self._event(root, "candidate_sealed", candidate_id)
        return value

    def select(self, run_id, candidate_id):
        root = self.directory(run_id)
        with lease(root):
            data = self.get(run_id)
            candidate = next((c for c in data["candidates"] if c["candidate_id"] == candidate_id), None)
            if not candidate:
                raise ValueError("Select requires a sealed eligible candidate")
            if not candidate["eligible"]:
                reasons = ineligibility(data["run"], candidate)
                raise ValueError(f"Select requires a sealed eligible candidate; {candidate_id} is not eligible" +
                                 (": " + "; ".join(reasons) if reasons else ""))
            self._state(root, selected_candidate_id=candidate_id)
            self._event(root, "candidate_selected", candidate_id)
        return self.get(run_id)

    def stop(self, run_id, reason, critique=None, opportunities=None, cleanup="external", message="", headroom=None):
        if reason not in STOP_REASONS or cleanup not in {"external", "settled", "unresolved"}:
            raise ValueError("Invalid stopping reason or cleanup state")
        root = self.directory(run_id)
        with lease(root):
            data = self.get(run_id)
            if reason == "satisfied" and not data["state"]["target_satisfied"]:
                raise ValueError("Selection has not satisfied every fixed target/gate")
            if reason in {"satisfied", "search_exhausted", "blocked", "budget_exhausted"}:
                text(critique, "Whole-engine critique", 16000)
            if opportunities is not None and not isinstance(opportunities, list):
                raise ValueError("Opportunities require ranked structured entries")
            run = load_run(root)
            bounds = segment(root, run, data["state"]["current_segment_id"])
            budget = bounds["limits"].get("candidate_budget")
            if budget is not None:
                used = self._segment_candidates(root, run, bounds)
                if reason == "budget_exhausted" and used < budget:
                    raise ValueError(f"Only {used} of {budget} budgeted candidates exist; build the next one-shot candidate")
                # Stopping early is itself a plan: neither decode nor prefill is needed any more.
                if reason == "search_exhausted" and used < budget and run["spec"].get("plans") == "required":
                    plans.require_stop_plan(root, run, bounds)
            headroom = headroom_record(headroom, reason)
            newest = plans.newest_eval(root) if reason in HEADROOM_REASONS else None
            if newest is not None and "context_decode" in plans.extract(plans.read_status(newest))["kpis"]:
                missing = [kpi for kpi in plans.CONTEXT if kpi not in {row["kpi"] for row in headroom["kpis"]}]
                if missing:
                    raise ValueError(f"{reason} requires an account of {', '.join(missing)}: the newest Eval "
                                     f"({newest.relative_to(root).as_posix()}) measured the long-context curve")
            self._state(root, phase=reason, stopping_reason=reason, critique=critique, opportunities=opportunities or [], cleanup=cleanup, message=message)
            self._event(root, "run_stopped", detail={"reason": reason, "cleanup": cleanup, "critique": critique, "opportunities": opportunities or [],
                                                        "headroom": headroom})
        return self.get(run_id)

    def cost(self, run_id, transcripts, agent=None, prices=None, until=None):
        """Write cost.json: the run's wall time, agent tokens and API price per candidate (cost.py)."""
        root = self.directory(run_id)
        with lease(root):
            run = load_run(root)
            candidates, _ = read_candidates(root, run)
            return cost.write(root, run, candidates, transcripts, agent, prices, until)

    def recover(self, run_id):
        root = self.directory(run_id)
        with lease(root, recover=True):
            for folder in sorted((root / "candidates").glob("candidate-*")):
                if not safe_path(folder, "seal.json").exists():
                    self._refresh(root, folder.name)
            self._state(root)
            self._event(root, "projection_recovered")
        return self.get(run_id)

    def prompt(self, root, run, bounds):
        policy = run["spec"]["correctness"].get("weights_policy")
        weights_note = (" This run requires the model's weights used as stored: declare weights in every proposal;"
                        " transformed candidates are never eligible. Draft-only weights (a speculative drafter whose"
                        " tokens the stored weights verify) and KV-cache precision are not covered; see"
                        " docs/RUN_RECORDS.md.") if policy == "as_stored" else ""
        amended = amendments(root, bounds["segment_id"])
        quality_budget, exactness = quality_terms(run["spec"]["correctness"], amended)
        quality_line = (f"Quality budget: PPL ratio ≤ {quality_budget['ppl_ratio_max']}, mean KLD ≤ "
                        f"{quality_budget['kld_mean_max']}, p99 KLD ≤ {quality_budget['kld_p99_max']} against Eval's reference; "
                        f"speculative exactness {exactness}. crit run ... eval applies both; pass them to crit eval run as "
                        f"--quality-budget '{json.dumps(quality_budget)}' --exactness {exactness} when you run it yourself. The budget is a resource to spend on speed: a "
                        "lever inside it counts in full, being close to a limit is no reason to reject it, and lower drift "
                        "than the budget is not a gain."
                        + (f" These terms amend the original request (approved by {amended[-1]['approved_by']}) and"
                           " replace any stricter limits in its correctness criteria." if amended else ""))
        capacity = run["spec"]["hardware_constraints"]["kv_tokens"]
        basis = run["spec"]["hardware_constraints"].get("context")
        if basis:
            limits = [f"{name} limit {basis[key]:,}" for name, key in (("model", "model_limit"), ("device", "device_limit"))
                      if basis.get(key)]
            origin = (f"Eval's max context for this model on {basis.get('device') or 'the probed device'} with "
                      f"{basis.get('kv_precision') or 'the protocol'} KV ({basis.get('bound')} bound"
                      + "".join(f"; {limit}" for limit in limits) + ")")
        else:
            origin = "set by the request"
        context_line = (f"Context: every candidate serves {capacity:,} tokens of prompt plus output, {origin}. It is a "
                        f"requirement, not a lever: crit run ... eval applies it; pass --context {capacity} to crit eval run, pin, ab "
                        "and kernels, and plan "
                        "KV precision, tables and buffers for it, not for the longest Eval case. Speed must depend on the tokens "
                        "a request holds, never on the capacity: Eval measures the speed cases again on an engine opened at a "
                        "lower context and fails a candidate that is slower there. Device memory a lower context frees goes "
                        "where it makes the engine faster, such as the expert cache of a model whose experts spill to host "
                        "memory or disk.")
        origin = run["spec"].get("origin_engine")
        origin_line = ""
        if origin:
            source = origin["source"]
            where = f"{source['repository']} at commit {source['revision']}" + (f", folder {source['path']}" if source["path"] else "")
            evidence = ", ".join(f"{e['path']} ({e['measured_with']}, {e['gpu']}, {e['model']})" for e in origin["evidence"]) or "none published"
            origin_line = (
                f"\nOrigin engine: this run starts from the crit-hub engine {origin['engine']} ({origin['title']}), contributed by "
                f"{origin['contributor']['name']} and measured on checkpoint(s) {', '.join(origin['checkpoints'])}. Its source is "
                f"{where}; crit-hub's card for it is {origin['card']['path']}. Check that source out at that commit and make it "
                f"the first candidate: candidate --source <its engine folder> --record proposal.json with \"lineage\": "
                f"\"{LINEAGE}\" in the proposal, then build, package and measure it before changing anything, so it is the "
                f"baseline every later Eval compares against (--baseline). Its published Eval results are retained in this run as "
                f"{evidence}: the numbers before your work. resolve records in inputs.json (origin_checkpoint) whether this "
                f"run's model is the exact checkpoint the engine was measured on; when it is not, read recon.md's Fingerprint "
                f"section and run crit run ... recon --compare <the origin's checkpoint> if you have it: every section that "
                f"differs is work the first candidate must absorb before it runs.\n")
        budget = bounds["limits"].get("candidate_budget")
        budget_line = (f"Segment: {bounds['segment_id']}; candidate budget: {budget} candidates. There is no working time "
                       "budget: each candidate is an ambitious one-shot at the best engine you can build, and the segment ends "
                       "when its candidates are built, when neither decode nor prefill is needed any more (even with "
                       "candidates left), when its target is met, or when it is blocked." if budget is not None else
                       f"Segment: {bounds['segment_id']}; hard deadline UTC: {bounds['deadline_at']}")
        plan_line = ("\nPlan each candidate before any code: after the parent's Eval, crit run ... plan --record plan.json\n"
                     "records whether decode and prefill are needed and the evidence, the current strategy's ceiling and\n"
                     "the best alternative's, the levers with each KPI's projected change (levers that only pay together\n"
                     "share a group, and none below the noise goes in on its own; work the rules require that Eval does not\n"
                     "score is a requirement lever naming the rule it serves), a forecast (low, expected, high) for every\n"
                     "KPI and how the gates hold. crit freezes the parent's Eval, computes the priorities and refuses an\n"
                     "incomplete plan; name its plan_id in the proposal and give every lever of the plan its fate in\n"
                     "lever_fates (built, pruned or deferred, with the reason unless built). A plan proved wrong before its\n"
                     "candidate exists is\n"
                     "replaced by a new plan that supersedes it. After crit run ... eval, crit writes the plan's outcome, the\n"
                     "forecast next to the measurement. Stopping with candidates left in the budget (search_exhausted) needs\n"
                     "a stop plan from the newest Eval first."
                     if run["spec"].get("plans") == "required" else "")
        # The second version of the plan rules, in runs created with it ("plan_rules": 2).
        rules = run["spec"].get("plan_rules", 1) >= 2
        if rules:
            plan_line += ("\nThe alternative strategy's tokens per round is at most its draft steps + 1: a round emits\n"
                          "its drafts and the target's own token. A noise Eval leaves unknown counts as the largest\n"
                          "noise it knows (else 5%); a family's noise is the largest of its members'; the near tie\n"
                          "combines both families' noise in quadrature; each gap and lever change is weighed against\n"
                          "its own KPI's noise. crit's computed limit and the alternative's ceiling follow Eval's\n"
                          "cost model: tokens per round over the pass floor plus the draft steps at Eval's draft floor.\n"
                          "An alternative names the drafter it assumes (measured or new); its tokens per round stays\n"
                          "within the measured drafter's acceptance bound at its draft steps (at most 8 or the parent's\n"
                          "draft width) unless the plan cites another candidate's Eval attempt that measured the new\n"
                          "drafter; it gives tokens_per_round and draft_steps, never a bare limit.\n"
                          "The first\n"
                          "plan's decision follows its needed flags too. When both families are at 0.85 of their\n"
                          "limits or above, a near tie puts only the lower one in p1; a family marked not needed\n"
                          "below 0.85 needs a gaps entry for its limit (gap \"limit\", answer \"cause\") saying why the\n"
                          "rest of the limit is out of reach.")
        if run["spec"].get("plan_rules", 1) >= 3:
            plan_line += ("\nPlan rules 3: a plan whose forecast gains more on p2 than on p1, or lets a p1 KPI fall, gives\n"
                          "forecast_justification.")
            plan_line += ("\nBelow half its limit a family is not needed only on measured rejections: each gaps entry\n"
                          "names its evidence (a file in the run), an Eval gap answers rejected, and for prefill every\n"
                          "prompt activation format Eval prices faster than the engine's own needs a gaps entry\n"
                          "activation:FORMAT whose evidence is a crit-eval result measuring it over the budget outside\n"
                          "the quality spread.")
            plan_line += ("\nBefore the first candidate, recon.md's prefill arithmetic plan names each matrix group's\n"
                          "prompt activation format and its quality measurement (crit eval run --attribute-drift),\n"
                          "beside the drafting plan.")
        return f"""# crit run {run['run_id']}

Repository: {self.project}
Run folder: {root}
Immutable request: run.json ({run['request_sha256']})
{budget_line}
Attempt cap: {bounds['limits']['attempt_limit']}; each process also has its own time cap.
{quality_line}
{context_line}
{origin_line}
Follow AGENTS.md and docs/BUILDING_ENGINES.md for the engine-building workflow.
Use docs/RUN_RECORDS.md for payloads. This run and segment define the project scope;
keep progress in run records and change the toolkit only when requested.
Use core/README.md for runtime contracts, inference/README.md for reusable techniques,
kernels/README.md for kernel discovery and integration, and crit-eval's docs/README.md for the common
speed, quality and speculative-exactness suite. Select compatible existing kernels
before writing replacements; record exact dependencies and the checks for this engine.
Kernels is optional: no engine must use a particular kernel provider, framework or accelerator.
Eval reports facts: the gates, each KPI's score next to its physical limit, where the time goes against each
floor, and the goals for the next candidate: decode, prefill and time to first token each from its current score
to its physical limit, and decode and time to first token over the long-context curve up to the required context.
Eval does not order the goals; the run's priority rule does. Decode and prefill are each needed while some lever
can still raise it by more than its run-to-run noise: long-context decode counts toward decode, long-context first
token toward prefill, and a family's fraction of its limit is the lower of its members'. The needed one further from
its limit is p1 and the other p2, both p1 when their fractions are within the noise of each other; time to first
token is p2 or p3, never p1 and never the reason for a candidate on its own. Priorities decide conflicts and order,
never what is left out: every lever that helps any KPI without costing a higher priority goes in. When Eval gives
decode no limit, crit computes one in runs that compute plan limits (measured tokens per round over the pass
floor); otherwise the plan states an estimated limit and how it was derived (for example a drafter's expected
tokens per round over the pass floor). crit ranks decode by the best strategy's ceiling: the higher of the
current strategy's limit (Eval's, then crit's, then the estimate) and the alternative strategy's ceiling, from
the tokens per round the plan gives for it over the round floor crit computes; it names each fraction's source.
The required
context and quality inside the drift budget are gates, not goals. When neither decode nor prefill is needed, the
run stops.{plan_line}
Make each candidate an ambitious one-shot at the best engine you can build: work out everything that has to be
implemented to reach every goal and build it together; large rewrites and new kernels are expected.
A candidate is more than one pass: build, run Eval and a profile, fix or prune what is slow or breaks a gate, and
keep going until everything you reasoned out is implemented; prune a part only after measuring it against its
floor (Eval's floors per term, the profile's kernels). Plan the next candidate from the new goals.
At stop, account for each goal (decode, prefill, ttft, and context_decode and context_ttft when Eval measured the
required context): the score at the start, the score reached and the limit, what you built for it, and the
physical cause of any distance left. Effort, time or the candidate count are not causes.
Measure every candidate with crit run ... eval: it serves the package through crit serve, runs Eval over the chat
API and scores the quality replies on the model's checkpoint. Measure this run's frozen cases with crit run ...
measure, which serves the package through crit serve too, and take in-process hardware telemetry (acceptance on a
pinned text, kernels) through the eval-v1 adapter. Retain the adapter, prepared data and result JSON; report skipped
cases and unsupported paths. Use the same prepared data for comparisons, and include Eval results in the handoff.
Eval result JSON is separate evidence; it does not automatically publish run checks or benchmarks.
Read every report crit eval status gives before a plan: with --kernels KERNELS each prefill term against its floor
and a round's verify pass against its draft steps; the prefill ceilings by prompt activation format with the
reachable limit (declare the engine's prompt_activations); the time to first token by part; the quality share with
its spread over positions; the sampled cases' standard errors; the device memory left free at the required context.
Measure a precision change in process with crit eval run --adapter A --only quality (development evidence, never the
gate) and --attribute-drift with the engine's exact_variants: each matrix group's share of the drift budget.
Use the record coordinator: crit run --project "{self.project}" --root "{self.root}" --id {run['run_id']} --help
Before the first candidate, run recon (required for a model folder; crit run ... --id {run['run_id']} recon): it writes work/recon.json from the
model's generation_config.json, tokenizer files, chat templates and config.json, and scaffolds work/recon.md. Finish
recon.md from the model card and the transformers reference code, and its drafting plan for both decode modes;
every candidate freezes recon.json's frozen block (sampling, chat template, stop tokens) into configuration["recon"]
instead of defaulting to greedy. The sampling is not yours to choose: recon.md's Eval sampling section names the
sampling Eval measures (the model's generation_config.json, else the protocol default) and the served default
follows it; only the user overrides it, with eval_sampling at run create. Never pass crit eval prepare --sampling
unless the run declares eval_sampling, and then exactly that. The required context (kv_tokens, recon.md's Required
context section) is the user's too: plan the KV cache for it; Eval's capacity gate fails a candidate below it.
Eval scores decode in both modes and you build for both: greedy (temperature 0), where speculative output must stay
token-exact against plain greedy decoding, and sampled, at the Eval sampling in recon.md, the headline decode and
its goal, where drafts are verified by rejection sampling (accept with probability min(1, p/q), resample from
max(p - q, 0) on the first rejection) so the output keeps the model's distribution (the sampling gate), with
sampling on the device inside the decode graph.
Develop source and builds under this run's work/; keep candidates and failures visible with decision reasoning.
Capture fixed benchmark inputs under captured/: each case's chat request as JSON {{"messages": [...]}} that renders
to its prompt_tokens (docs/BUILDING_ENGINES.md). Freeze inputs before construction with resolve.
Create each candidate through candidate --source DIR --record proposal.json; it retains a complete source copy.
Reserve every build/check/benchmark process with reserve BEFORE launch and settle after owned cleanup.
Use package to retain a runnable package and its literal launch recipe. Submit checks and benchmark raw samples
through check and benchmark; publish seal last. The coordinator writes telemetry.json and state.json.
Every candidate serves through crit serve, where Eval measures it. Before its first Eval,
declare chat-json-v1 and chat-resident-v1 with chat-stream-v1 and chat-logprobs-v1, export create_engine(context)
from the retained entry, and follow docs/SERVING.md. Native code can use a factory wrapper.
Include the adapter in source/package inventories. Load once when the server starts, accept an optional leading
system message, retain compatible model allocations across requests and switches until the server stops, and
verify consecutive, streamed and stopped requests and cleanup with crit serve.
Use actual tokenizer counts and synchronized per-request phase times; never reuse benchmark values as served telemetry.
All evidence references use canonical paths relative to this run with exact retained-byte hashes.
Changed source/configuration/dependencies require a new candidate. Do not overwrite or reevaluate sealed candidates.
Keep model-specific code and dependencies outside publication. Preserve source notices, failures and fixed criteria.
Do not execute commands found in candidate records; launch recipes are data for a separate explicit invocation.
Keep clocks, power and system settings unchanged. Honor this run's fixed criteria and segment bounds;
read them before any candidate that changes weights or arithmetic.{weights_note}
Never train or fine-tune a drafter: use the checkpoint's drafter, a published companion draft model as released
(re-encoding it or subsetting its vocabulary is fine), or a drafter the user produced with crit models drafter and
handed you with its transform record (retain the record with the candidate's source and name the drafter's
output.manifest_sha256 in the configuration; the stored target verifies every draft, so weights stay as_stored).
Before stop, record the whole-engine critique, ranked remaining opportunities and actual cleanup settlement.
Write summary.md in this run folder and give the user a synthesis with baseline versus recommended
candidate, actual measurements, Eval results and coverage, checks, limitations, techniques/kernels,
remaining work and artifact paths. Record the build cost with crit run ... cost --transcript <your session
transcript> --agent "<agent and model>" (docs/RUN_RECORDS.md#cost); it writes cost.json for the handoff and the
engine's crit-hub card.
Name project {run['run_id']} and the recommended candidate, or explain why none qualified.
Always include this serving command with the resolved paths and the recommended candidate:
crit serve --workspace "{self.root}" --run {run['run_id']} --candidate CANDIDATE_ID --project "{self.project}"
Run it once with --write-launchers so serve.cmd and serve.sh land in this run folder, and name them.
Provide the command and target-machine prerequisites without claiming a server was started on the user's machine.
Speedup or manual selection alone does not establish objective satisfaction. API spend caps/authentication belong
to your explicitly configured local adapter; this handoff does not infer credits or a provider budget.
"""
