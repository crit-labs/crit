"""Single-writer run coordinator; records are data, never executable recipes."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import os
import json
from pathlib import Path
import platform
import shutil
import socket
import uuid

from .records import (LIFECYCLES, MAX_CANDIDATES, SAFETY_WINDOW_MINUTES, STOP_REASONS, WEIGHTS, amendment, candidate_projection, quality_terms, digest, evidence,
    file_sha, header, headroom_record, ident, ineligibility, inputs, inventory, keys, load_run, now, number, read_candidates,
    read_json, record, request_spec, safe_path, segment, text, timestamp, write_json)
from . import recon

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
def lease(root):
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


def serving_gaps(capabilities, model_path):
    """What a chat-resident-v1 package leaves out that crit serve clients rely on: per-request temperature, and
    tool calls when the model's chat template teaches a call format crit parses."""
    if not isinstance(capabilities, list) or "chat-resident-v1" not in capabilities:
        return []
    gaps = []
    if "chat-sampling-temperature-v1" not in capabilities:
        gaps.append("it must apply a request's temperature and declare chat-sampling-temperature-v1")
    if not {"chat-tools-v1", "chat-tool-template-v1"} & set(capabilities) and Path(model_path).is_dir():
        from crit_core.text import template_sources, templates_tool_format
        try:
            taught = templates_tool_format(template_sources(Path(model_path)).values())
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
            "proposed_candidate_id", "selected_candidate_id", "target_satisfied", "budget", "stopping_reason", "critique", "opportunities", "cleanup", "message"})
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

    def create(self, data):
        spec = request_spec(data)
        while True:
            run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
            root = self.root / run_id
            try:
                root.mkdir()
                break
            except FileExistsError:
                continue
        run = record("run", run_id, created_at=now(), spec=spec, request_sha256=digest(spec))
        with lease(root):
            for name in ("segments", "events", "candidates", "work", "captured"):
                (root / name).mkdir()
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
            value = read_json(path)
            run = load_run(root)
            header(value, "attempt_reservation", run["run_id"], {"segment_id", "candidate_id", "attempt_id", "operation", "reserved_at", "deadline_at", "inputs_manifest_sha256", "package_sha256", "cost"})
            ident(value["attempt_id"], "attempt")
            bounds = segment(root, run, value["segment_id"])
            if value["candidate_id"] is not None:
                ident(value["candidate_id"], "candidate")
            owner = f"candidates/{value['candidate_id']}" if value["candidate_id"] else f"segments/{value['segment_id']}"
            if path.relative_to(root).as_posix() != f"{owner}/attempts/{value['attempt_id']}/reservation.json":
                raise ValueError("Reservation owner/path mismatch")
            operation = value["operation"]
            if operation not in bounds["limits"]["process_seconds"] or (value["candidate_id"] is None) != (operation in {"agent", "discovery"}):
                raise ValueError("Invalid reservation operation/owner")
            start, deadline = timestamp(value["reserved_at"], "reservation"), timestamp(value["deadline_at"], "deadline")
            if not timestamp(bounds["started_at"], "start") <= start <= deadline <= timestamp(bounds["deadline_at"], "segment deadline") or (deadline-start).total_seconds() > bounds["limits"]["process_seconds"][operation]:
                raise ValueError("Reservation exceeds fixed segment/process bounds")
            if digest(value["cost"]) != digest({"unit": "attempt", "amount": 1}):
                raise ValueError("Invalid attempt counter reservation")
            outcome_path = safe_path(root, f"{owner}/attempts/{value['attempt_id']}/result.json")
            if outcome_path.exists():
                outcome = read_json(outcome_path)
                header(outcome, "attempt_result", run["run_id"], {"segment_id", "candidate_id", "attempt_id", "outcome", "ended_at", "exit_code", "cleanup_settled", "logs", "failure"})
                if any(outcome[key] != value[key] for key in ("segment_id", "candidate_id", "attempt_id")) or type(outcome["cleanup_settled"]) is not bool:
                    raise ValueError("Invalid attempt result binding/cleanup")
                if outcome["outcome"] not in {"completed", "failed", "cancelled", "timed_out", "interrupted", "launch_failed"}:
                    raise ValueError("Invalid process outcome")
                ended = timestamp(outcome["ended_at"], "process end")
                if ended < start or (outcome["outcome"] == "completed" and (ended > deadline or outcome["exit_code"] != 0 or not outcome["cleanup_settled"])):
                    raise ValueError("Process result violates its reserved bounds")
                if outcome["exit_code"] is not None:
                    number(outcome["exit_code"], "exit code", -(2 ** 31), 2 ** 32, integer=True)
                if not isinstance(outcome["logs"], list):
                    raise ValueError("Logs require retained evidence references")
                for reference in outcome["logs"]:
                    evidence(root, reference)
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
        self._refresh(root, candidate_id)
        self._state(root)
        self._event(root, value["record_kind"], candidate_id, reference={"path": path.relative_to(root).as_posix(), "sha256": file_sha(path)})
        return value

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
                 {"parent_candidate_id", "origin_candidate_id", "weights"})
            if "weights" in proposal and proposal["weights"] not in WEIGHTS:
                raise ValueError("weights requires as_stored or transformed")
            if run["spec"]["correctness"].get("weights_policy") == "as_stored" and "weights" not in proposal:
                raise ValueError("This run requires weights used as stored: declare weights as_stored or transformed")
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
            if frozen is None and Path(run["spec"]["model_path"]).is_dir():
                raise ValueError(f"Run recon before the first candidate: crit run --id {run_id} recon writes "
                                 f"{recon.RECON_JSON}, then finish {recon.RECON_MD}")
            configuration, extensions = proposal["configuration"], proposal.get("extensions", {})
            if frozen is not None:
                if not isinstance(configuration, dict) or configuration.get("recon", frozen) != frozen:
                    raise ValueError(f"configuration.recon differs from {recon.RECON_JSON}; change its frozen block, with "
                                     f"the reason in {recon.RECON_MD}, before the candidate")
                configuration, extensions = {**configuration, "recon": frozen}, {**extensions, "recon": recon_ref}
            candidate_id = next_id(root / "candidates", "candidate")
            folder = root / "candidates" / candidate_id
            # Validate every record before the folder exists: a rejected proposal leaves nothing behind and
            # does not consume a candidate ID.
            source_manifest = record("source_manifest", run_id, candidate_id=candidate_id, files=rows)
            value = record("candidate", run_id, candidate_id=candidate_id, segment_id=bounds["segment_id"], ordinal=int(candidate_id.split("-")[1]),
                parent_candidate_id=proposal.get("parent_candidate_id"), origin_candidate_id=proposal.get("origin_candidate_id"), created_at=now(),
                **{k: proposal[k] for k in ("label", "rationale", "strategy", "store_decisions")}, configuration=configuration,
                **({"weights": proposal["weights"]} if "weights" in proposal else {}),
                extensions=extensions, source_manifest_sha256=digest(source_manifest), inputs_manifest_sha256=resolved["manifest_sha256"], request_sha256=run["request_sha256"])
            overlays = {f"candidates/{candidate_id}/candidate.json": value, f"candidates/{candidate_id}/source.manifest.json": source_manifest}
            folder.mkdir()
            try:
                shutil.copytree(source, folder / "source")
                candidate_projection(root, run, candidate_id, overlays)
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                raise
            write_json(folder / "source.manifest.json", source_manifest, immutable=True)
            write_json(folder / "candidate.json", value, immutable=True)
            # The agent may have prepared the candidate's work folder before creating it.
            (root / "work" / candidate_id).mkdir(exist_ok=True)
            self._refresh(root, candidate_id)
            self._state(root)
            self._event(root, "candidate_created", candidate_id, detail={"parent_candidate_id": value["parent_candidate_id"], "rationale": value["rationale"]})
        return value

    def recon(self, run_id, force=False):
        root = self.directory(run_id)
        with lease(root):
            value = recon.write(root, load_run(root), force)
            self._event(root, "model_recon", reference={"path": recon.RECON_JSON, "sha256": file_sha(root / recon.RECON_JSON)})
        return value

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
            copy_package(source, folder / "package", rows, folder / "source", read_json(folder / "source.manifest.json")["files"])
            if not supplied.is_file():
                (folder / "package" / "configuration.json").write_text(
                    json.dumps(candidate["configuration"], indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
                rows = inventory(folder / "package")
            value = record("package_manifest", run_id, candidate_id=candidate_id, source_manifest_sha256=candidate["source_manifest_sha256"],
                configuration_sha256=digest(candidate["configuration"]), files=rows, **recipe)
            self._publish(root, candidate_id, "package/manifest.json", value)
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
            if operation not in bounds["limits"]["process_seconds"] or (candidate_id is None) != (operation in {"agent", "discovery"}):
                raise ValueError("Invalid reservation owner/operation")
            if sum(a["segment_id"] == bounds["segment_id"] for a in self._attempts(root)) >= bounds["limits"]["attempt_limit"]:
                raise ValueError("Segment attempt limit exhausted")
            cap = bounds["limits"]["process_seconds"][operation]
            seconds = cap if seconds is None else number(seconds, "Process deadline", .001, cap)
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
                if value["outcome"] not in {"completed", "failed", "cancelled", "timed_out", "interrupted", "launch_failed"} or type(value["cleanup_settled"]) is not bool:
                    raise ValueError("Invalid agent outcome")
                ended = timestamp(value["ended_at"], "process end")
                if ended < timestamp(reservation["reserved_at"], "reserved") or (value["outcome"] == "completed" and (ended > timestamp(reservation["deadline_at"], "deadline") or value["exit_code"] != 0 or not value["cleanup_settled"])):
                    raise ValueError("Agent outcome violates its reserved bounds")
                if value["exit_code"] is not None:
                    number(value["exit_code"], "exit code", -(2 ** 31), 2 ** 32, integer=True)
                if not isinstance(value["logs"], list):
                    raise ValueError("Logs require retained references")
                for ref in value["logs"]:
                    evidence(root, ref)
                write_json(path.with_name("result.json"), value, immutable=True)
                phase = ("finished" if reservation["operation"] == "agent" else "building") if value["outcome"] == "completed" else value["outcome"]
                cleanup = ("settled" if reservation["operation"] == "agent" else "external") if value["cleanup_settled"] else "unresolved"
                self._state(root, phase=phase, cleanup=cleanup, message=value["failure"] or ("Runner ended; inspect candidate gates" if reservation["operation"] == "agent" else ""))
                self._event(root, "agent_settled", detail={"outcome": value["outcome"], "cleanup_settled": value["cleanup_settled"]})
        return value

    def seal(self, run_id, candidate_id, lifecycle, failure=None):
        root = self.directory(run_id)
        with lease(root):
            outcome = record("candidate_outcome", run_id, candidate_id=candidate_id, lifecycle=lifecycle, failure=failure, closed_at=now())
            self._publish(root, candidate_id, "outcome.json", outcome)
            folder = safe_path(root, f"candidates/{candidate_id}")
            revision = read_json(folder / "telemetry.json")["revision"]
            value = record("candidate_seal", run_id, candidate_id=candidate_id, sealed_at=now(), telemetry_revision=revision, files=inventory(folder))
            write_json(folder / "seal.json", value, immutable=True)
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
            headroom = headroom_record(headroom, reason)
            self._state(root, phase=reason, stopping_reason=reason, critique=critique, opportunities=opportunities or [], cleanup=cleanup, message=message)
            self._event(root, "run_stopped", detail={"reason": reason, "cleanup": cleanup, "critique": critique, "opportunities": opportunities or [],
                                                        "headroom": headroom})
        return self.get(run_id)

    def recover(self, run_id):
        root = self.directory(run_id)
        lock_path = safe_path(root, ".coordinator.lock")
        if lock_path.exists():
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
                handle = api.OpenProcess(0x1000, False, pid)
                if handle:
                    api.CloseHandle.argtypes = [ctypes.c_void_p]
                    api.CloseHandle(handle)
                    raise ValueError("Coordinator owner may still be alive; recovery blocked")
                if ctypes.get_last_error() != 87:
                    raise ValueError("Cannot prove coordinator owner exited")
            else:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError("Coordinator owner is still alive")
            lock_path.unlink()
        with lease(root):
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
        amendments = [s["amendment"] for s in (read_json(p) for p in sorted((root / "segments").glob("segment-*.json")))
                      if "amendment" in s and s["segment_id"] <= bounds["segment_id"]]
        quality_budget, exactness = quality_terms(run["spec"]["correctness"], amendments)
        quality_line = (f"Quality budget: PPL ratio ≤ {quality_budget['ppl_ratio_max']}, mean KLD ≤ "
                        f"{quality_budget['kld_mean_max']}, p99 KLD ≤ {quality_budget['kld_p99_max']} against Eval's reference; "
                        f"speculative exactness {exactness}. Pass both to crit eval run as --quality-budget "
                        f"'{json.dumps(quality_budget)}' --exactness {exactness}. The budget is a resource to spend on speed: a "
                        "lever inside it counts in full, being close to a limit is no reason to reject it, and lower drift "
                        "than the budget is not a gain."
                        + (f" These terms amend the original request (approved by {amendments[-1]['approved_by']}) and"
                           " replace any stricter limits in its correctness criteria." if amendments else ""))
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
                        f"requirement, not a lever: pass --context {capacity} to crit eval run, pin, ab and kernels, and plan "
                        "KV precision, tables and buffers for it, not for the longest Eval case. Speed must depend on the tokens "
                        "a request holds, never on the capacity: Eval measures the speed cases again on an engine opened at a "
                        "lower context and fails a candidate that is slower there. Device memory a lower context frees goes "
                        "where it makes the engine faster, such as the expert cache of a model whose experts spill to host "
                        "memory or disk.")
        budget = bounds["limits"].get("candidate_budget")
        budget_line = (f"Segment: {bounds['segment_id']}; candidate budget: {budget} candidates. There is no working time "
                       "budget: each candidate is an ambitious one-shot at the best engine you can build, and the segment ends "
                       "when its candidates are built, its target is met, or it is blocked." if budget is not None else
                       f"Segment: {bounds['segment_id']}; hard deadline UTC: {bounds['deadline_at']}")
        return f"""# crit run {run['run_id']}

Repository: {self.project}
Run folder: {root}
Immutable request: run.json ({run['request_sha256']})
{budget_line}
Attempt cap: {bounds['limits']['attempt_limit']}; each process also has its own time cap.
{quality_line}
{context_line}

Follow AGENTS.md and docs/BUILDING_ENGINES.md for the engine-building workflow.
Use docs/RUN_RECORDS.md for payloads. This run and segment define the project scope;
keep progress in run records and change the toolkit only when requested.
Use core/README.md for runtime contracts, inference/README.md for reusable techniques,
kernels/README.md for kernel discovery and integration, and docs/eval/README.md for the common
speed, quality and speculative-exactness suite. Select compatible existing kernels
before writing replacements; record exact dependencies and the checks for this engine.
Kernels is optional: no engine must use a particular kernel provider, framework or accelerator.
Eval reports facts: the gates, each KPI's score next to its physical limit, where the time goes against each
floor, and the goals for the next candidate: decode, prefill and time to first token each from its current score
to its physical limit, and decode and time to first token over the long-context curve up to the required context.
Every goal is the candidate's goal at once; none is enough on its own and none trades against another. Eval does
not rank them; you reason out how to reach them.
Make each candidate an ambitious one-shot at the best engine you can build: work out everything that has to be
implemented to reach every goal and build it together; large rewrites and new kernels are expected.
A candidate is more than one pass: build, run Eval and a profile, fix or prune what is slow or breaks a gate, and
keep going until everything you reasoned out is implemented. Plan the next candidate from the new goals.
At stop, account for each goal (decode, prefill, ttft, and context_decode and context_ttft when Eval measured the
required context): the score at the start, the score reached and the limit, what you built for it, and the
physical cause of any distance left. Effort, time or the candidate count are not causes.
Measure candidates with Eval alongside this run's frozen cases. Retain the eval-v1 adapter,
applied settings, prepared data, reference and result JSON; report skipped cases and unsupported paths.
Use the same prepared data and reference for comparisons, and include Eval results in the handoff.
Eval result JSON is separate evidence; it does not automatically publish run checks or benchmarks.
Use the record coordinator: crit run --project "{self.project}" --root "{self.root}" --id {run['run_id']} --help
Before the first candidate, run recon (required for a model folder; crit run ... --id {run['run_id']} recon): it writes work/recon.json from the
model's generation_config.json, tokenizer files, chat templates and config.json, and scaffolds work/recon.md. Finish
recon.md from the model card and the transformers reference code; every candidate freezes recon.json's frozen
block (sampling, chat template, stop tokens) into configuration["recon"] instead of defaulting to greedy.
Develop source and builds under this run's work/; keep candidates and failures visible with decision reasoning.
Capture fixed benchmark inputs under captured/. Freeze inputs before construction with resolve.
Create each candidate through candidate --source DIR --record proposal.json; it retains a complete source copy.
Reserve every build/check/benchmark process with reserve BEFORE launch and settle after owned cleanup.
Use package to retain a runnable package and its literal launch recipe. Submit checks and benchmark raw samples
through check and benchmark; publish seal last. The coordinator writes telemetry.json and state.json.
Include serving through crit serve unless the user explicitly requested a narrower deliverable. Before sealing,
declare chat-json-v1 and chat-resident-v1, export create_engine(context) from the retained entry,
and follow docs/SERVING.md. Native code can use a factory wrapper.
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
Never train or fine-tune a drafter: use the checkpoint's drafter or a published companion draft model as released
(re-encoding it or subsetting its vocabulary is fine).
Before stop, record the whole-engine critique, ranked remaining opportunities and actual cleanup settlement.
Write summary.md in this run folder and give the user a synthesis with baseline versus recommended
candidate, actual measurements, Eval results and coverage, checks, limitations, techniques/kernels,
remaining work and artifact paths.
Name project {run['run_id']} and the recommended candidate, or explain why none qualified.
Always include this serving command with the resolved paths and the recommended candidate:
crit serve --workspace "{self.root}" --run {run['run_id']} --candidate CANDIDATE_ID --project "{self.project}"
Run it once with --write-launchers so serve.cmd and serve.sh land in this run folder, and name them.
Provide the command and target-machine prerequisites without claiming a server was started on the user's machine.
Speedup or manual selection alone does not establish objective satisfaction. API spend caps/authentication belong
to your explicitly configured local adapter; this handoff does not infer credits or a provider budget.
"""
