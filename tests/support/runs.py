"""A run with a candidate, its package, checks and benchmarks, all synthetic records (RunFixture): nothing it
records executes. Every candidate's source and package is a plugin package, as crit run requires: the toy engine's
plugins (support/engines.py) beside the fixture's files. Importing it puts crit's packages, and crit Eval beside this
checkout, on the path."""
from datetime import datetime
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for _folder in ("core/python", "api", "../crit-eval", "kernels", "cli"): sys.path.insert(0, str(ROOT / _folder))
from crit.run.records import GIB, digest, file_sha, now, read_json  # noqa: E402
from crit.run.store import RunStore  # noqa: E402
from .engines import TOY, selection  # noqa: E402


def with_plugins(folder):
    """The toy engine's plugins in folder/plugins (a candidate's source, a package), beside any it holds."""
    plugins = Path(folder) / "plugins"
    for plugin in sorted(TOY.iterdir()):
        if (plugin / "plugin.json").is_file() and not (plugins / plugin.name).exists():
            shutil.copytree(plugin, plugins / plugin.name, ignore=shutil.ignore_patterns("__pycache__"))
    return folder


class RunFixture:
    model_files = None  # {name: text} makes the model a recon'd fixture folder (core_text_fixtures.llama_files)
    # Whether a candidate's rung on the candidate ladder limits its Eval. These tests measure Eval, plans and records
    # on any candidate, so by default no rung limits it (crit.run.ladder.scope_of gives none); CandidateLadderTests
    # climb the real ladder.
    ladder_scopes = False
    # With model_files, candidate() writes recon.md's plans first, as crit asks before any candidate.
    finish_recon = True

    def setUp(self):
        if not self.ladder_scopes:
            patched = patch("crit.run.ladder.scope_of", lambda *args, **kwargs: None)
            patched.start()
            self.addCleanup(patched.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        if self.model_files:
            from core_text_fixtures import llama_files, write_files
            self.model = write_files(self.root / "model", llama_files(
                **{"model.fixture": "synthetic input, not a model", **self.model_files}))
        else:
            self.model = self.root / "model.fixture"
            self.model.write_text("synthetic input, not a model", encoding="utf-8")
        # A segment long enough for every test's records: a check or benchmark recorded after the segment's deadline is
        # refused, and a loaded machine takes more than a minute between a test's run and its later records.
        self.data = {"model_path": str(self.model), "correctness_criteria": "Fixed fixture gates", "budget_minutes": 30,
                     "kv_tokens": 8192, "targets": {"decode_tps": 15}, "workload": {"repetitions": 3}}
        self.store = RunStore(self.root / "runs")
        self.view = self.store.create(self.data)
        self.run = self.view["run"]
        self.run_id = self.run["run_id"]
        self.directory = self.store.directory(self.run_id)
        if self.model_files:
            self.store.recon(self.run_id)

    def resolve(self):
        path = self.directory / "captured/input.fixture"
        path.write_text("fixed captured workload", encoding="utf-8")
        self.resolved = self.store.freeze_inputs(self.run_id, {"hardware_manifest": {"device": "host fixture"},
            "environment_manifest": {"benchmark": "synthetic"}, "case_manifests": {"case-0001": [{"path": "captured/input.fixture", "bytes": path.stat().st_size, "sha256": file_sha(path)}]}})

    def proposal(self, **updates):
        """A proposal; without a plan_id it names a first plan recorded for it (every candidate has a plan)."""
        if "plan_id" not in updates:
            updates = {"plan_id": self.first_plan(), "lever_fates": {}, **updates}
        value = {"label": "Fixture engine", "rationale": "Test retained artifact contracts", "strategy": "Generic fixture",
                 "configuration": {"batch": 1}, "store_decisions": {"attention-state": {"status": "untested", "reason": "Host artifact fixture only"}}, **updates}
        if isinstance(value["configuration"], dict) and "plugins" not in value["configuration"]:
            value["configuration"] = {**value["configuration"], "plugins": selection()}
        return value

    def first_plan(self, **updates):
        """Record a first plan (no parent, no levers: it imports a baseline as is), as every candidate needs one while
        the run has no candidate on the candidate ladder. Returns its plan_id."""
        value = {"parent_candidate_id": None, "decision": "build",
                 "needed": {name: {"needed": True, "evidence": "fixture"} for name in ("decode", "prefill")},
                 "gaps": [], "levers": [], "rejected": [], "forecast": None, "gates": "fixture gates",
                 "strategy": {"current": {"name": "fixture", "ceiling": "unknown", "evidence": "fixture"},
                              "alternative": {"name": "none", "ceiling": "unknown", "evidence": "fixture"},
                              "choice": "current", "reason": "fixture"}, **updates}
        if self.run["spec"]["correctness"]["quality_mode"] == "speed":
            value.setdefault("prompt_activations", "stored")
        return self.store.plan(self.run_id, value)["plan_id"]

    def child_plan(self, parent, **updates):
        """Record a build plan from `parent`, a candidate on the candidate ladder without an Eval (on_ladder), with one
        lever. Returns its plan_id."""
        value = {"parent_candidate_id": parent, "levers": [
            {"id": "fixture-lever", "technique": "fixture", "description": "fixture", "group": None, "confidence": "low",
             "provider": "inference:attention-state", "changes": {"decode": {"change": 0.2, "source": "estimate"}}}],
            "forecast": {kpi: {"low": 1, "expected": 2, "high": 3} for kpi in ("decode", "prefill", "ttft")}, **updates}
        return self.first_plan(**value)

    def on_ladder(self, candidate):
        """Put a candidate below the full Eval on the candidate ladder, as serve --check and seal built do: a passed
        functional check that serve --check recorded, and the seal. Returns its id."""
        candidate_id = candidate["candidate_id"]
        folder = self.directory / "candidates" / candidate_id
        if not (folder / "package").is_dir():
            self.package(candidate)
        package = read_json(folder / "package" / "manifest.json")
        path = folder / "logs" / "serve-check-fixture.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"checks": []}', encoding="utf-8")
        self.store.append(self.run_id, candidate_id, "check", {
            "case_id": "case-0001", "package_sha256": digest(package), "gate": "functional",
            "criteria_sha256": digest(self.run["spec"]["correctness"]), "status": "passed", "rationale": "fixture",
            "evidence": [{"path": path.relative_to(self.directory).as_posix(), "sha256": file_sha(path)}],
            "evidence_level": "artifact_verified", "recorded_at": now(),
            "extensions": {"recorded_by": "crit run serve --check"}})
        self.store.seal(self.run_id, candidate_id, "built")
        return candidate_id

    def candidate(self, **updates):
        """A candidate from a recorded plan: the given plan_id, else a plan crit records for it, a child plan of its
        parent_candidate_id (which must stand on the ladder: on_ladder) or a first plan."""
        if not (self.directory / "inputs.json").exists():
            self.resolve()
        notes = self.directory / "work" / "recon.md"
        if self.finish_recon and notes.is_file():
            from crit.run import recon
            text = notes.read_text(encoding="utf-8")
            for title in recon.PLANS + (recon.PLACEMENT_PLAN,):
                text = text.replace(f"## {title}\n\nTODO", f"## {title}\n\nWritten for the fixture. TODO")
            notes.write_text(text, encoding="utf-8")
        parent = updates.get("parent_candidate_id")
        if "plan_id" not in updates and parent:
            updates = {"plan_id": self.child_plan(parent), "lever_fates": {"fixture-lever": {"fate": "built"}},
                       **updates}
        source = self.root / "source-input"
        source.mkdir(exist_ok=True)
        (source / "engine.fixture").write_text("complete synthetic source", encoding="utf-8")
        with_plugins(source)
        return self.store.candidate(self.run_id, source, self.proposal(**updates))

    def package(self, candidate):
        package = self.root / "package-input"
        package.mkdir(exist_ok=True)
        (package / "engine.fixture").write_text("synthetic package bytes; never execute", encoding="utf-8")
        with_plugins(package)
        return self.store.package(self.run_id, candidate["candidate_id"], package,
            {"build": {"toolchain": {"identity": "host fixture"}, "dependencies": [], "command": ["fixture"]}})

    def checks(self, candidate, package, arithmetic="passed", evidence_level="artifact_verified"):
        result = []
        for gate in ("arithmetic", "functional", "resources"):
            path = self.directory / "candidates" / candidate["candidate_id"] / "checks/raw" / (gate + ".fixture")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic " + gate + " result", encoding="utf-8")
            value = self.store.append(self.run_id, candidate["candidate_id"], "check", {"case_id": "case-0001", "package_sha256": digest(package),
                "gate": gate, "criteria_sha256": digest(self.run["spec"]["correctness"]),
                "status": arithmetic if gate == "arithmetic" else "passed", "rationale": "Synthetic fixture outcome",
                "evidence": [{"path": path.relative_to(self.directory).as_posix(), "sha256": file_sha(path)}], "evidence_level": evidence_level, "recorded_at": now()})
            record_path = self.directory / "candidates" / candidate["candidate_id"] / "checks" / (value["check_id"] + ".json")
            result.append({"path": record_path.relative_to(self.directory).as_posix(), "sha256": file_sha(record_path)})
        return result

    def benchmark_data(self, package, checks, **updates):
        case = self.run["spec"]["cases"][0]
        return {"case_id": case["case_id"], "package_sha256": digest(package), "case_sha256": digest(case),
            "protocol_sha256": digest(self.run["spec"]["benchmark_protocol"]), "identities": {
                "model_sha256": digest(self.resolved["model_manifest"]), "input_sha256": digest(self.resolved["case_manifests"][case["case_id"]]),
                "hardware_sha256": digest(self.resolved["hardware_manifest"]), "environment_sha256": digest(self.resolved["environment_manifest"])},
            "workload": case["workload"], "metrics": {"decode_tps": {"samples": [10, 12, 11], "unit": "tokens/second"},
                "prefill_tps": {"samples": [100, 120, 110], "unit": "tokens/second"}, "request_ms": {"samples": [10, 12, 11], "unit": "milliseconds"},
                "peak_vram_bytes": {"samples": [3 * GIB] * 3, "unit": "bytes"}, "peak_ram_bytes": {"samples": [4 * GIB] * 3, "unit": "bytes"},
                "minimum_free_vram_bytes": {"samples": [GIB] * 3, "unit": "bytes"}, "minimum_free_ram_bytes": {"samples": [2 * GIB] * 3, "unit": "bytes"}},
            "missing_metrics": {"load_ms": "Not measured by synthetic fixture"}, "checks": checks, "outcome": "completed", "started_at": now(), "ended_at": now(), **updates}

    def measured(self, arithmetic="passed", evidence_level="artifact_verified", **updates):
        candidate = self.candidate(**updates)
        package = self.package(candidate)
        checks = self.checks(candidate, package, arithmetic, evidence_level)
        data = self.benchmark_data(package, checks)
        self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data)
        return candidate, package, checks, data


def timestamp_for_test(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
