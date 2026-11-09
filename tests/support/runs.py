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
from crit.run.records import GIB, digest, file_sha, now  # noqa: E402
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
    # Runs as created before plans (no plans field, plan_id optional); plan tests create runs that require them.
    new_run_plans = None
    # Runs as created before budget counts (every candidate spends the budget); budget tests create runs that count
    # the candidates through Eval.
    budget_counts = None

    def setUp(self):
        for name, value in (("NEW_RUN_PLANS", self.new_run_plans), ("NEW_RUN_BUDGET_COUNTS", self.budget_counts)):
            patched = patch(f"crit.run.records.{name}", value)
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
        value = {"label": "Fixture engine", "rationale": "Test retained artifact contracts", "strategy": "Generic fixture",
                 "configuration": {"batch": 1}, "store_decisions": {"attention-state": {"status": "untested", "reason": "Host artifact fixture only"}}, **updates}
        if isinstance(value["configuration"], dict) and "plugins" not in value["configuration"]:
            value["configuration"] = {**value["configuration"], "plugins": selection()}
        return value

    def candidate(self, **updates):
        if not (self.directory / "inputs.json").exists():
            self.resolve()
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
