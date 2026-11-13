"""Single-writer run coordinator; records are data, never executable recipes."""
from contextlib import contextmanager
from datetime import timedelta
import functools
import os
import json
from pathlib import Path
import shutil
import tempfile
import uuid

from ..cli import checkout, installed
# The coordinator lease each write takes (crit.lease); named from here as before.
from ..lease import exclusive_lease, lease, lease_owner, read_lease_owner, release_dead_lease  # noqa: F401
from ..records import retry_denied, utc_now, verification_scope
from .records import (
    HEADROOM_REASONS, LIFECYCLES, LINEAGE, MAX_CANDIDATES, SAFETY_WINDOW_MINUTES, STOP_REASONS, WEIGHTS, amendment,
    attempt_process_cap,
    candidate_projection, quality_mode, quality_terms, digest, evidence, file_sha,
    header, headroom_record, ident,
    ineligibility, inputs, inventory, keys, load_run, now, number, read_candidates, read_json, record,
    request_spec, safe_path, segment, text, timestamp, validate_attempt_reservation, validate_attempt_result,
    validate_spec, verify_inventory, write_json,
)
from . import cost, hub, ladder, plans, recon, replay

ROOT = Path(__file__).resolve().parents[3]
# The folders the repository publishes (its .gitignore allowlist, which a test holds this to) and the ones it keeps
# for retention: no run root lies inside them.
# The run's phase in its segment (state.json `phase`): the segment's own state only. A candidate's state is its
# lifecycle (and crit.run.state), the stop's reason is stopping_reason. Older runs stored a candidate lifecycle or a
# stop reason as the phase; get() reads those as building and stopped.
PHASES = {"awaiting_agent", "running", "discovering", "building", "finished", "failed", "launch_failed", "expired",
          "stopped"}
PUBLISHED = ("core", "inference", "kernels", "models", "tests", "docs", "eval", "cli", "api", "mcp", "smoke", ".github")
PROTECTED = PUBLISHED + ("tools", "archive")


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
    retry_denied(lambda: source.rename(destination))


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


def engine_package(folder, model_path, configuration, what, device_file=None, launchable=True):
    """Refuse `folder` unless it is an engine crit run can serve: an external engine its configuration declares
    (configuration.engine, crit.run.external.check_package; `launchable`: its launch folder must exist, as in a
    package), else a plugin package Core's loader runs (plugin_package)."""
    from .external import check_package, spec_of
    if spec_of(configuration) is not None:
        check_package(folder, configuration, what, launchable)
        return
    plugin_package(folder, model_path, configuration, what, device_file)


def plugin_package(folder, model_path, configuration, what, device_file=None):
    """Refuse `folder` unless it is an engine Core's loader runs: plugins/ and a configuration whose plugins selection
    fills every slot an engine fills, checked without running any of its code (crit_core.engine.check_package),
    against the run's device (its crit eval probe device file, else the machine's)."""
    from crit_core.engine import check_package
    from crit_plugins import PluginError
    probe = read_json(Path(device_file)) if device_file and Path(device_file).is_file() else None
    try:
        check_package(folder, model_path, configuration, probe=probe)
    except PluginError as exc:
        raise ValueError(f"{what} is no engine Core can load: {exc}. An engine is Core plus a plugin selection: "
                         "the package holds plugins/ and configuration.plugins names the plugin of each slot "
                         "(docs/PLUGINS.md)") from None


def configuration_text(configuration):
    """A candidate's configuration.json as crit writes it: into a build folder before the recipe's command runs, and
    into a package that supplies none."""
    return json.dumps(configuration, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def run_device_file(run):
    """The crit eval probe device file the run's context was derived from, if any."""
    return ((run["spec"].get("hardware_constraints") or {}).get("context") or {}).get("device_file")


def store_operation(method):
    """A RunStore operation as one verification scope (crit.records.verification_scope): however many of its
    projections read a candidate, it walks and hashes the candidate's retained inventories once, and the next
    operation verifies them again. An operation run inside another (select reads the run with get) is part of it."""
    @functools.wraps(method)
    def scoped(*args, **kwargs):
        with verification_scope():
            return method(*args, **kwargs)
    return scoped


class RunStore:
    def __init__(self, root, project=ROOT):
        self.project = Path(project).resolve()
        # The crit the run's agent follows, which its agent_request.md names (repository): the project when it is a
        # crit checkout, else the installed crit, when crit runs without a checkout. Its project is then any folder,
        # the one of settings.toml and the default runs/, and its guides are crit's repository at its release.
        crit = None
        if not checkout(self.project) and (crit := installed()) is None:
            raise ValueError("Select a crit repository with --project")
        self.repository = (str(self.project) if crit is None else
                           f"{crit.guides}, the release of the installed crit {crit.version} ({crit.package})")
        requested = Path(root).expanduser().absolute()
        if requested.resolve() != requested or requested.is_symlink():
            raise ValueError("Run root must not traverse symlinks/reparse points")
        self.root = requested.resolve()
        if crit is None and (self.root == self.project or
                             any(self.root.is_relative_to(self.project / name) for name in PROTECTED)):
            raise ValueError("Runs belong outside publication and protected retention")
        if crit is not None and self.root.is_relative_to(crit.package.parent):
            raise ValueError(f"Runs belong outside the folder crit is installed in ({crit.package.parent})")
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, run_id):
        path = safe_path(self.root, ident(run_id, "run"))
        if not path.is_dir():
            raise ValueError("Unknown run")
        return path

    def _recording(self, run_id):
        """The run's folder for a command that records or changes something: never a replay's (replay.py)."""
        root = self.directory(run_id)
        replay.refuse(root)
        return root

    @store_operation
    def get(self, run_id):
        root = self.directory(run_id)
        run = load_run(root)
        state = read_json(safe_path(root, "state.json"))
        header(state, "run_state", run_id, {"revision", "updated_at", "current_segment_id", "phase", "candidates", "baseline_candidate_id",
            "proposed_candidate_id", "selected_candidate_id", "target_satisfied", "budget", "stopping_reason", "critique", "opportunities", "cleanup", "message"},
            {"plans"})
        number(state["revision"], "state revision", 1, 2 ** 31, integer=True)
        timestamp(state["updated_at"], "state update")
        if state["phase"] in STOP_REASONS:
            state["phase"] = "stopped"
        elif state["phase"] in LIFECYCLES:
            state["phase"] = "building"
        if state["phase"] not in PHASES:
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
        remaining = max(0, (timestamp(bounds["deadline_at"], "deadline") - utc_now()).total_seconds())
        state["budget"]["remaining_seconds"] = round(remaining)
        if not remaining and state["phase"] in {"awaiting_agent", "running", "discovering", "building"}:
            state["phase"] = "expired"
        prompt_path = safe_path(root, f"segments/{bounds['segment_id']}/agent_request.md")
        return {"run": run, "state": state, "segment": bounds, "candidates": candidates, "errors": errors,
                "folder": str(root), "prompt": prompt_path.read_text(encoding="utf-8")}

    @store_operation
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
        sealed = [c for c in candidates if c["sealed"]]
        state["baseline_candidate_id"] = sealed[0]["candidate_id"] if sealed else None
        selected = next((c for c in candidates if c["candidate_id"] == state["selected_candidate_id"]), None)
        state["target_satisfied"] = bool(selected and selected["target_satisfied"])
        state["plans"] = plans.summaries(root, run)
        segments = sorted((root / "segments").glob("segment-*.json"))
        attempts = self.attempts(root)
        bounds = segment(root, run, state["current_segment_id"])
        state["budget"] = {"segments": len(segments), "attempts": len(attempts),
            "segment_attempts": sum(a["segment_id"] == bounds["segment_id"] for a in attempts),
            "remaining_seconds": max(0, round((timestamp(bounds["deadline_at"], "deadline") - utc_now()).total_seconds()))}
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
        if candidates is None:
            raise ValueError("Resume requires --candidates: a segment's budget is candidates on the candidate ladder")
        minutes = SAFETY_WINDOW_MINUTES if minutes is None else minutes
        limits |= {"candidate_budget": number(candidates, "Candidate budget", 1, 64, integer=True),
                   "budget_minutes": number(minutes, "Safety window", 1, SAFETY_WINDOW_MINUTES, integer=True)}
        limits["attempt_limit"] = number(attempt_limit, "Attempt limit", 1, 1000, integer=True)
        segment_id = next_id(root / "segments", "segment", ".json")
        started = utc_now()
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

    @store_operation
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
            run_id = "run-" + utc_now().strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
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
        if state["stopping_reason"] or utc_now() >= timestamp(bounds["deadline_at"], "deadline"):
            raise ValueError("Segment is closed/expired; explicit Resume is required")
        return run, state, bounds

    def attempts(self, root):
        """Every attempt reservation the run retains, validated against its segment, and its result where it settled
        (crit serve checks them before it serves a candidate)."""
        paths = list((root / "candidates").glob("candidate-*/attempts/attempt-*/reservation.json"))
        paths += list((root / "segments").glob("segment-*/attempts/attempt-*/reservation.json"))
        result = []
        run = load_run(root)
        for path in paths:
            safe_path(root, path.relative_to(root).as_posix())
            value = validate_attempt_reservation(root, run, read_json(path))
            owner = f"candidates/{value['candidate_id']}" if value["candidate_id"] else f"segments/{value['segment_id']}"
            if path.relative_to(root).as_posix() != f"{owner}/attempts/{value['attempt_id']}/reservation.json":
                raise ValueError("Reservation owner/path mismatch")
            outcome_path = safe_path(root, f"{owner}/attempts/{value['attempt_id']}/result.json")
            if outcome_path.exists():
                validate_attempt_result(root, run["run_id"], value, read_json(outcome_path))
            result.append(value)
        for segment_id in {value["segment_id"] for value in result}:
            if sum(value["segment_id"] == segment_id for value in result) > segment(root, run, segment_id)["limits"]["attempt_limit"]:
                raise ValueError("Retained attempts exceed their authorized segment cap")
        return result

    @store_operation
    def resume(self, run_id, minutes=None, attempt_limit=64, external_cleanup_confirmed=False, candidates=None, amend=None):
        """amend: {quality_budget?, exactness?, approved_by} changes the run's quality terms from this segment on."""
        if type(external_cleanup_confirmed) is not bool:
            raise ValueError("External cleanup confirmation requires a boolean")
        root = self._recording(run_id)
        with lease(root):
            run = load_run(root)
            previous = read_json(root / "state.json")
            for reservation in self.attempts(root):
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

    @store_operation
    def freeze_inputs(self, run_id, data):
        root = self._recording(run_id)
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
            # A captured chat request must render to its workload's prompt tokens as candidates will serve it.
            frozen, _ = recon.load(root, run)
            if frozen is not None:
                from .measure import check_case_prompts
                check_case_prompts(root, run, data["case_manifests"], frozen["template_variables"])
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
    def _segment_candidates(root, bounds):
        """The candidates the segment's budget has spent: those on the candidate ladder in it."""
        return sum(found == bounds["segment_id"] for found in ladder.counted(root).values())

    @staticmethod
    def _spent(budget):
        """The refusal once the segment's candidate budget is used."""
        return ValueError(f"This segment's candidate budget ({budget} candidates on the candidate ladder) is used; "
                          "stop, or Resume with more candidates")

    @store_operation
    def candidate(self, run_id, source, proposal):
        root = self._recording(run_id)
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
            if "plan_id" not in proposal:
                raise ValueError("Every candidate has a plan: record it with crit run ... plan --record plan.json after "
                                 "the parent's Eval (or seal) and before any code, and name its plan_id in the proposal")
            plan = plans.check_link(root, run, proposal["plan_id"], proposal.get("parent_candidate_id"))
            plans.check_fates(plan, proposal.get("lever_fates", {}))
            if len(list((root / "candidates").glob("candidate-*"))) >= MAX_CANDIDATES:
                raise ValueError("Candidate limit exceeded")
            budget = bounds["limits"].get("candidate_budget")
            if budget is not None and self._segment_candidates(root, bounds) >= budget:
                raise self._spent(budget)
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
            # recon.md's drafting, prefill arithmetic and placement plans are written before any candidate.
            missing = recon.unfinished(root) if frozen is not None else []
            if missing:
                raise ValueError(f"Finish {recon.RECON_MD} before the candidate: " + ", ".join(missing) + " (still TODO "
                                 "or missing). The prefill arithmetic plan names each matrix group's prompt activation "
                                 "format and, in preserve, its quality measurement (crit eval run --quality-mode "
                                 "preserve --attribute-drift)")
            configuration, extensions = proposal["configuration"], proposal.get("extensions", {})
            if frozen is not None:
                if not isinstance(configuration, dict) or configuration.get("recon", frozen) != frozen:
                    raise ValueError(f"configuration.recon differs from {recon.RECON_JSON}; change its frozen block, with "
                                     f"the reason in {recon.RECON_MD}, before the candidate")
                configuration, extensions = {**configuration, "recon": frozen}, {**extensions, "recon": recon_ref}
            engine_package(source, run["spec"]["model_path"], configuration, "The candidate's source",
                           run_device_file(run), launchable=False)
            plans.check_engine(plan, configuration)
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

    @store_operation
    def plan(self, run_id, data, check=False):
        """Record a candidate plan (plans.py): checked against the parent's newest Eval, which it freezes, with the
        priorities crit computes. Never changed afterwards; a revision is a new plan that supersedes it. check: run
        the same checks and return the record crit would store, writing nothing (the only plan command a replay
        takes; there the parent's Eval status is the one the replay rendered again)."""
        root = self.directory(run_id)
        if check:
            with lease(root):
                run = load_run(root)
                bounds = segment(root, run, read_json(safe_path(root, "state.json"))["current_segment_id"])
                value = plans.build(root, run, bounds, data, replay.status_for(root))
            return {"checked": True, "recorded": False, **value}
        replay.refuse(root)
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

    @store_operation
    def plan_draft(self, run_id, path):
        """crit run ... plan --draft: write the next plan's skeleton to `path` (refusing to overwrite a file) and
        return it with the facts crit computes for it (plans.draft)."""
        root = self.directory(run_id)
        with lease(root):
            run = load_run(root)
            bounds = segment(root, run, read_json(safe_path(root, "state.json"))["current_segment_id"])
            value = plans.draft(root, run, bounds)
        path = Path(path)
        if path.exists():
            raise ValueError(f"{path} exists: plan --draft writes a new skeleton, so name another file")
        path.write_text(json.dumps(value["plan"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"written": str(path), **value}

    @store_operation
    def next(self, run_id):
        """crit run ... next: the run's one next step (crit.run.state)."""
        from .state import next_step
        return next_step(self, run_id)

    @store_operation
    def plan_outcome(self, run_id, candidate_id, attempt_id):
        """After a retained Eval attempt of a candidate created from a plan: the plan's forecast next to what Eval
        measured, at plans/PLAN_ID/outcomes/ATTEMPT.json. None when the candidate has no plan."""
        root = self._recording(run_id)
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

    @store_operation
    def recon(self, run_id, force=False, drafters=()):
        root = self._recording(run_id)
        with lease(root):
            value = recon.write(root, load_run(root), force, drafters)
            self._event(root, "model_recon", reference={"path": recon.RECON_JSON, "sha256": file_sha(root / recon.RECON_JSON)})
        return value

    @store_operation
    def recon_compare(self, run_id, other):
        """What another checkpoint changes for an engine built for this run's model (recon.compare_with_run)."""
        root = self.directory(run_id)
        return recon.compare_with_run(root, load_run(root), Path(other).expanduser().resolve())

    @store_operation
    def package(self, run_id, candidate_id, source, recipe):
        root = self._recording(run_id)
        with lease(root):
            run, _, _ = self._active(root)
            folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
            if (folder / "seal.json").exists():
                raise ValueError("Sealed candidate is immutable")
            candidate = read_json(folder / "candidate.json")
            keys(recipe, {"build"})
            source = Path(source).absolute()
            if root.is_relative_to(source):
                raise ValueError("Package folder must not contain its run")
            rows = inventory(source)
            if any(r["path"] == "manifest.json" for r in rows):
                raise ValueError("Package input must not contain coordinator manifest.json")
            engine_package(source, run["spec"]["model_path"], candidate["configuration"], "The package",
                           run_device_file(run))
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
                    (staged / "configuration.json").write_bytes(
                        configuration_text(candidate["configuration"]).encode("utf-8"))
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

    @store_operation
    def append(self, run_id, candidate_id, kind, data):
        root = self._recording(run_id)
        with lease(root):
            load_run(root)  # Result publication grants no new process execution.
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

    @store_operation
    def reserve(self, run_id, candidate_id, operation, seconds=None):
        root = self._recording(run_id)
        with lease(root):
            run, _, bounds = self._active(root)
            cap = attempt_process_cap(bounds, candidate_id, operation)
            if sum(a["segment_id"] == bounds["segment_id"] for a in self.attempts(root)) >= bounds["limits"]["attempt_limit"]:
                raise ValueError("Segment attempt limit exhausted")
            # The candidate ladder: a candidate's rung says whether Eval measures it at all.
            if operation == "eval" and candidate_id and ladder.scope_of(root, candidate_id) == "none":
                raise ValueError(f"{candidate_id} is on the candidate ladder's first rung: {ladder.SCOPES['none']}")
            # A candidate's first Eval spends one of the segment's budgeted candidates unless it already counts; later
            # Evals of it (a resume, a partial rerun) cost nothing more.
            budget = bounds["limits"].get("candidate_budget")
            if (operation == "eval" and budget is not None and candidate_id not in ladder.counted(root)
                    and self._segment_candidates(root, bounds) >= budget):
                raise self._spent(budget)
            window = (timestamp(bounds["deadline_at"], "deadline") - utc_now()).total_seconds()
            limit = window if cap is None else cap
            seconds = limit if seconds is None else number(seconds, "Process deadline", .001, max(.001, limit))
            owner = f"candidates/{ident(candidate_id, 'candidate')}" if candidate_id else f"segments/{bounds['segment_id']}"
            directory = safe_path(root, owner + "/attempts")
            attempt_id = next_id(directory, "attempt")
            reserved = utc_now()
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

    @store_operation
    def eval_started(self, run_id, reservation, journal, resumes=None):
        """The run's event for an Eval attempt that starts measuring: its reservation, the journal it writes and the
        earlier attempt's journal it resumes (crit run ... eval --resume)."""
        root = self._recording(run_id)
        with lease(root):
            owner = f"candidates/{reservation['candidate_id']}/attempts/{reservation['attempt_id']}"
            reserved = safe_path(root, owner + "/reservation.json")
            resumed = None
            if resumes is not None:
                resumes = Path(resumes)
                resumed = {"path": resumes.relative_to(root).as_posix(), "sha256": file_sha(resumes)}
            self._event(root, "eval_resumed" if resumed else "eval_started", reservation["candidate_id"],
                        detail={"attempt_id": reservation["attempt_id"],
                                "journal": Path(journal).relative_to(root).as_posix(), "resumes": resumed},
                        reference={"path": f"{owner}/reservation.json", "sha256": file_sha(reserved)})

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

    @store_operation
    def settle(self, run_id, reservation, data):
        root = self._recording(run_id)
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

    def _unmeasured_cases(self, root, folder, candidate_id):
        """While the segment runs, a completed seal waits for a benchmark of every run case: `select` needs one, and a
        sealed candidate takes no new measurement. A stopped segment refuses `measure`, so there an Eval alone seals."""
        if read_json(root / "state.json").get("stopping_reason") is not None:
            return
        measured = {s.get("case_id") for s in read_json(folder / "telemetry.json").get("series") or []}
        missing = [c["case_id"] for c in load_run(root)["spec"]["cases"] if c["case_id"] not in measured]
        if missing:
            raise ValueError(f"Measure {', '.join(missing)} before sealing {candidate_id} as completed (crit run ... "
                             f"check --candidate {candidate_id} --from-eval, then crit run ... measure --candidate "
                             f"{candidate_id}): select needs every case's benchmark, and a sealed candidate takes no "
                             "new measurement")

    @store_operation
    def seal(self, run_id, candidate_id, lifecycle, failure=None):
        root = self._recording(run_id)
        with lease(root):
            folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
            outcome_path = safe_path(folder, "outcome.json")
            seal_path = safe_path(folder, "seal.json")
            if outcome_path.exists():
                outcome = read_json(outcome_path)
                if digest({key: outcome[key] for key in ("lifecycle", "failure")}) != digest({"lifecycle": lifecycle, "failure": failure}):
                    raise ValueError("Candidate outcome already exists with different lifecycle/failure")
            else:
                if lifecycle == "completed":
                    self._unmeasured_cases(root, folder, candidate_id)
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

    @store_operation
    def select(self, run_id, candidate_id):
        root = self._recording(run_id)
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

    @store_operation
    def stop(self, run_id, reason, critique=None, opportunities=None, cleanup="external", message="", headroom=None):
        if reason not in STOP_REASONS or cleanup not in {"external", "settled", "unresolved"}:
            raise ValueError("Invalid stopping reason or cleanup state")
        root = self._recording(run_id)
        with lease(root):
            data = self.get(run_id)
            headroom = headroom_record(headroom, reason)
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
                used = self._segment_candidates(root, bounds)
                if reason == "budget_exhausted" and used < budget:
                    raise ValueError(f"Only {used} of {budget} budgeted candidates are on the candidate ladder; build "
                                     "the next rung's candidate")
                # Stopping early is itself a plan: neither decode nor prefill is needed any more.
                if reason == "search_exhausted" and used < budget:
                    plans.require_stop_plan(root, run, bounds)
            newest = plans.newest_eval(root) if reason in HEADROOM_REASONS else None
            if newest is not None and "context_decode" in plans.extract(plans.read_status(newest))["kpis"]:
                missing = [kpi for kpi in plans.CONTEXT if kpi not in {row["kpi"] for row in headroom["kpis"]}]
                if missing:
                    raise ValueError(f"{reason} requires an account of {', '.join(missing)}: the newest Eval "
                                     f"({newest.relative_to(root).as_posix()}) measured the long-context curve")
            self._state(root, phase="stopped", stopping_reason=reason, critique=critique, opportunities=opportunities or [], cleanup=cleanup, message=message)
            self._event(root, "run_stopped", detail={"reason": reason, "cleanup": cleanup, "critique": critique, "opportunities": opportunities or [],
                                                        "headroom": headroom})
        return self.get(run_id)

    @store_operation
    def cost(self, run_id, transcripts, agent=None, prices=None, until=None, idle=None):
        """Write cost.json: the run's wall and active time, agent tokens and API price per candidate (cost.py)."""
        root = self._recording(run_id)
        with lease(root):
            run = load_run(root)
            candidates, _ = read_candidates(root, run)
            return cost.write(root, run, candidates, transcripts, agent, prices, until, idle)

    @store_operation
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
        """The run's brief (agent_request.md): this run's facts and the user's terms only. How to build an engine is
        AGENTS.md's, and the order of the steps is crit run ... next's, so the brief repeats neither."""
        correctness = run["spec"]["correctness"]
        amended = amendments(root, bounds["segment_id"])
        quality_budget, exactness = quality_terms(correctness, amended)
        mode = quality_mode(correctness)
        policy = correctness.get("weights_policy")
        capacity = run["spec"]["hardware_constraints"]["kv_tokens"]
        basis = run["spec"]["hardware_constraints"].get("context")
        context = (f"Eval's max context for this model on {basis.get('device') or 'the probed device'} "
                   f"({basis.get('bound')} bound)" if basis else "set by the request")
        sampling = run["spec"].get("eval_sampling")
        targets = run["spec"]["objectives"]["targets"]
        base = f'crit run --project "{self.project}" --root "{self.root}" --id {run["run_id"]}'
        terms = [
            f"- Model: {run['spec']['model_path']}",
            f"- Quality mode {mode} (the run's; never change it): " + (
                "a budget that keeps the output close to the checkpoint's, so measure every precision change (activation "
                "quantization, a lower-precision KV cache or head) against it" if mode == "preserve" else
                "a budget that only keeps out a broken engine: the stored format's activations, FP8 or 4-bit KV, an FP8 "
                "head and, where the weights policy allows, re-encoded weights are the starting arithmetic, and output "
                "forks at near-ties pass") + " (docs/RUN_RECORDS.md#quality-modes).",
            f"- Quality budget: PPL ratio ≤ {quality_budget['ppl_ratio_max']}, mean KLD ≤ "
            f"{quality_budget['kld_mean_max']}, p99 KLD ≤ {quality_budget['kld_p99_max']} against Eval's reference; "
            f"speculative exactness {exactness}" + (
                f"; these terms amend the request (approved by {amended[-1]['approved_by']})" if amended else "") +
            f". Spend it on speed. crit run ... eval applies it; pass --quality-mode {mode} --quality-budget "
            f"'{json.dumps(quality_budget)}' --exactness {exactness} to the crit eval commands you run yourself.",
            f"- Context: every candidate serves {capacity:,} tokens of prompt plus output, {context}. Pass --context "
            f"{capacity} to the crit eval commands you run yourself.",
            "- Weights policy: " + ({"as_stored": "the target weights are used as stored; declare weights in every "
                                                  "proposal",
                                     "transformable": "the engine may re-encode the stored 16-bit target weights as "
                                                      "it loads them (declare weights transformed and "
                                                      "configuration.re_encoding)"}.get(policy, "none stated")) + ".",
            "- Eval sampling: " + (f"the user's override {sampling['settings']} (approved by {sampling['approved_by']}); "
                                   "prepare Eval data with crit eval prepare --sampling and exactly these settings"
                                   if sampling else "the model's own (recon.md's Eval sampling section)") + ".",
        ]
        if targets:
            terms.append("- Targets: " + ", ".join(f"{key} {value}" for key, value in targets.items()
                                                   if key != "extensions") + ".")
        origin = run["spec"].get("origin_engine")
        origin_line = ""
        if origin:
            source = origin["source"]
            where = f"{source['repository']} at commit {source['revision']}" + (
                f", folder {source['path']}" if source["path"] else "")
            other = sorted({e["quality_mode"] for e in origin["evidence"]} - {mode})
            evidence = ", ".join(f"{e['path']} ({e['measured_with']}, {e['gpu']}, {e['model']}, quality mode "
                                 f"{e['quality_mode']})" for e in origin["evidence"]) or "none published"
            origin_line = (
                f"\nOrigin engine: this run starts from the crit-hub engine {origin['engine']} ({origin['title']}), source "
                f"{where}. Make it the first candidate as imported (\"lineage\": \"{LINEAGE}\" in the proposal): its "
                f"small Eval is the run's baseline. Its published results: {evidence}"
                + (f". Results judged in quality mode {' and '.join(other)} are not judged in this run's, {mode}: their "
                   "numbers do not compare with this run's Evals" if other else "")
                + ". resolve records whether this run's model is the exact checkpoint it was measured on; when not, "
                "recon --compare says what candidate 1 must absorb.\n")
        budget = bounds["limits"].get("candidate_budget")
        budget_line = (f"candidate budget: {budget} candidates on the candidate ladder (docs/RUN_RECORDS.md#the-candidate-ladder); "
                       f"safety window until {bounds['deadline_at']} UTC" if budget is not None else
                       f"hard deadline UTC: {bounds['deadline_at']} (a time budget, from an earlier crit release)")
        return f"""# crit run {run['run_id']}

Build the best engine you can for this run's model and machine. Follow AGENTS.md in the repository below.
After every step run

    {base} next

which reads the run's records and names the one next step and its command. crit refuses a step out of order with
what to do instead.

Repository: {self.repository}
Run folder: {root}
Immutable request: run.json ({run['request_sha256']})
Segment: {bounds['segment_id']}; {budget_line}; attempt cap {bounds['limits']['attempt_limit']}.

The user's terms (never change them; only the user amends them):
{chr(10).join(terms)}
{origin_line}
Hand-off serve command, with the recommended candidate:

    crit serve --workspace "{self.root}" --run {run['run_id']} --candidate CANDIDATE_ID --project "{self.project}"
"""
