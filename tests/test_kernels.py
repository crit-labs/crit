"""Executable applicability, exact dependency retention and read-only navigation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "../crit-eval", "kernels", "cli"): sys.path.insert(0, str(ROOT / _folder))
import crit_kernels as kernels
from crit.run.records import digest, inventory, read_json, write_json


class KernelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "provider"
        self.source.mkdir()
        (self.source / "kernel.py").write_text("raise RuntimeError('inspection must never import me')\n")
        (self.source / "LICENSE").write_text("Synthetic test notice, not a project license.\n")
        self.req = {"schema_version": 1, "record_kind": "kernels_request", "operation": {
            "name": "gemm", "dtype": "float32", "layout": "row-major", "m": 7, "n": 11, "k": 16},
            "target": {"os": "windows", "arch": "x86_64", "backend": "cpu", "abi": "cblas-ilp64", "features": ["sse2", "avx2"]}}
        self.entry = {"schema_version": 1, "record_kind": "kernels_implementation", "id": "fixture.sgemm",
            "summary": "Synthetic SGEMM applicability fixture; never executed",
            "provenance": {"repository": "https://example.invalid/fixture", "revision": "sha256:" + "a" * 64, "variant": "host-ilp64"},
            "license": "LicenseRef-SyntheticFixture",
            "requirements": {"operation." + name: {"one_of": [value]} for name, value in self.req["operation"].items() if name not in {"m", "n", "k"}},
            "contract": {name: "Synthetic fixture contract: " + name for name in ("semantics", "invocation", "workspace", "completion", "ownership")},
            "artifacts": [row | {"role": "notice" if row["path"] == "LICENSE" else "source"} for row in inventory(self.source)],
            "runtime_dependencies": [], "unverified": []}
        self.entry["requirements"].update({"target." + name: {"one_of": [value]} for name, value in self.req["target"].items() if name != "features"})
        self.entry["requirements"].update({"operation.m": {"minimum": 1, "maximum": 128},
            "operation.n": {"minimum": 1}, "operation.k": {"minimum": 8, "multiple_of": 8},
            "target.features": {"contains": ["avx2"]}})
        self.catalog = self.root / "catalog"
        self.catalog.mkdir()
        write_json(self.catalog / "kernel.json", self.entry)
        self.request_path = self.root / "request.json"
        write_json(self.request_path, self.req)

    def save(self, **changes):
        return kernels.retain(self.req, {self.entry["id"]: self.entry}, self.entry["id"], self.source,
                             changes.pop("destination", self.root / "retained"), "Chosen for this exact fixture ABI", **changes)

    def command(self, *arguments, expected=0):
        result = subprocess.run([sys.executable, "-S", "-B", str(ROOT / "tests/run_crit.py"), "kernels", *map(str, arguments)],
            cwd=self.root, capture_output=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, expected, result.stderr)
        return result

    def test_inspection_explains_all_matches_and_never_imports_or_writes(self):
        before = inventory(self.root)
        value = json.loads(self.command("inspect", "--request", self.request_path, "--catalog", self.catalog).stdout)
        self.assertEqual(value["request_sha256"], digest(self.req))
        decision = value["decisions"][0]
        self.assertEqual(decision["status"], "declared_compatible")
        self.assertTrue(all(check["status"] == "matched" for check in decision["checks"]))
        self.assertIn("no operator", decision["qualification"])
        self.assertEqual(before, inventory(self.root))
        result = json.loads(self.command("search", "sgemm", "--catalog", self.catalog).stdout)
        self.assertEqual([entry["id"] for entry in result], ["fixture.sgemm"])
        target = json.loads(self.command("target").stdout)
        self.assertIsNone(target["target"]["abi"])

    def test_shape_abi_dtype_layout_device_and_feature_rejections_are_specific(self):
        for key, wrong in (("operation.dtype", "bfloat16"), ("operation.layout", "column-major"),
                           ("operation.k", 15), ("operation.m", 129), ("operation.n", 0),
                           ("target.abi", "cblas-lp64"), ("target.os", "linux"),
                           ("target.arch", "aarch64"), ("target.backend", "cuda"), ("target.features", ["sse2"])):
            with self.subTest(key=key):
                value = copy.deepcopy(self.req)
                namespace, name = key.split(".")
                value[namespace][name] = wrong
                decision = kernels.inspect(value, self.entry)
                self.assertEqual(decision["status"], "rejected")
                check = next(c for c in decision["checks"] if c["fact"] == key)
                self.assertEqual(check["actual"], wrong)
                self.assertEqual(check["status"], "rejected")

    def test_missing_facts_uncovered_semantics_and_incomplete_provenance_remain_unknown(self):
        for mode in ("missing_target", "missing_rule", "new_mask", "floating_revision", "no_notice", "unreviewed"):
            with self.subTest(mode=mode):
                req, entry = copy.deepcopy(self.req), copy.deepcopy(self.entry)
                if mode == "missing_target":
                    req["target"].pop("abi")
                elif mode == "missing_rule":
                    entry["requirements"].pop("operation.dtype")
                elif mode == "new_mask":
                    req["operation"]["mask"] = "causal"
                elif mode == "floating_revision":
                    entry["provenance"]["revision"] = "main"
                elif mode == "no_notice":
                    entry["artifacts"] = [a for a in entry["artifacts"] if a["role"] != "notice"]
                else:
                    entry["unverified"] = ["Aliasing behavior needs source review"]
                self.assertEqual(kernels.inspect(req, entry)["status"], "unknown")
                with self.assertRaisesRegex(ValueError, "declared-compatible"):
                    kernels.retain(req, {entry["id"]: entry}, entry["id"], self.source, self.root / "blocked", "reason")
                self.assertFalse((self.root / "blocked").exists())

    def test_rules_are_typed_and_cannot_execute_arbitrary_expressions(self):
        self.assertFalse(kernels.matches(True, {"one_of": [1]}))
        self.assertFalse(kernels.matches(16.0, {"multiple_of": 8}))
        self.assertTrue(kernels.matches("any layout", {"any_value": True}))
        for rule in ({"eval": "__import__('os')"}, {"multiple_of": 0}, {"minimum": 5, "maximum": 1},
                     {"one_of": []}, {"contains": "avx2"}, {"any_value": 1}, {"minimum": float("nan")}):
            with self.subTest(rule=rule):
                entry = copy.deepcopy(self.entry)
                entry["requirements"]["operation.k"] = rule
                with self.assertRaises(ValueError):
                    kernels.implementation(entry)

    def test_retention_copies_exact_bytes_notices_and_recomputable_comparisons(self):
        rejected = copy.deepcopy(self.entry)
        rejected["id"] = "fixture.cuda"
        rejected["requirements"]["target.backend"] = {"one_of": ["cuda"]}
        write_json(self.catalog / "other.json", rejected)
        dest = self.root / "retained"
        saved = json.loads(self.command("retain", "--request", self.request_path, "--catalog", self.catalog,
            "--select", self.entry["id"], "--root", self.source, "--destination", dest, "--reason", "Native ABI matches").stdout)
        verified = json.loads(self.command("verify", dest).stdout)
        self.assertEqual(verified["dependency_sha256"], digest(saved))
        self.assertEqual(saved["rationale"], "Native ABI matches")
        self.assertEqual((dest / "artifacts/LICENSE").read_bytes(), (self.source / "LICENSE").read_bytes())
        self.assertEqual(len(read_json(dest / "assessment.json")["decisions"]), 2)
        self.assertEqual(saved["implementation_sha256"], digest(self.entry))
        with self.assertRaisesRegex(ValueError, "new destination"):
            self.save(destination=dest)
        (dest / "artifacts/kernel.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "differ from inventory"):
            kernels.verify(dest)

    def test_changed_source_copy_race_and_resource_limit_never_publish_a_dependency(self):
        with self.assertRaisesRegex(ValueError, "byte limit"):
            self.save(max_bytes=1)
        self.assertFalse((self.root / "retained").exists())
        original = kernels.verify_files
        def mutate_after_verification(root, rows):
            original(root, rows)
            (self.source / "kernel.py").write_text("changed after initial verification")
        with patch.object(kernels, "verify_files", mutate_after_verification), self.assertRaisesRegex(ValueError, "changed during retention"):
            self.save()
        self.assertFalse((self.root / "retained/dependency.json").exists())
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.save(destination=self.root / "another")

    def test_unsafe_duplicate_paths_and_catalog_ids_are_rejected(self):
        for path in ("../kernel.py", "/kernel.py", "C:/kernel.py", "a//b", "a\\b", "a/./b", "a.", "a/CON.txt", "x?y"):
            with self.subTest(path=path):
                entry = copy.deepcopy(self.entry)
                entry["artifacts"][0]["path"] = path
                with self.assertRaises(ValueError):
                    kernels.implementation(entry)
        entry = copy.deepcopy(self.entry)
        entry["artifacts"].append(entry["artifacts"][0] | {"path": entry["artifacts"][0]["path"].lower()})
        with self.assertRaisesRegex(ValueError, "[Cc]ase-aliased"):
            kernels.implementation(entry)
        for paths in (("Parent/a", "parent/b"), ("parent", "parent/b")):
            entry = copy.deepcopy(self.entry)
            for artifact, path in zip(entry["artifacts"], paths):
                artifact["path"] = path
            with self.assertRaises(ValueError):
                kernels.implementation(entry)
        write_json(self.catalog / "duplicate.json", self.entry)
        with self.assertRaisesRegex(ValueError, "Duplicate implementation"):
            kernels.catalog(self.catalog)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.save(destination=self.source / "retained")

    def test_retained_assessment_cannot_be_changed_by_merely_updating_inventory(self):
        self.save()
        dest = self.root / "retained"
        assessment = read_json(dest / "assessment.json")
        assessment["decisions"][0]["checks"][0]["reason"] = "fabricated"
        write_json(dest / "assessment.json", assessment)
        record = read_json(dest / "dependency.json")
        record["files"] = inventory(dest, exclude=("dependency.json",))
        write_json(dest / "dependency.json", record)
        with self.assertRaisesRegex(ValueError, "assessment differs"):
            kernels.verify(dest)

    def test_retained_metadata_cannot_claim_operator_qualification(self):
        self.save()
        dest = self.root / "retained"
        original = read_json(dest / "dependency.json")
        for changes in ({"qualification": "arithmetic passed"}, {"rationale": ""}, {"selected": []}):
            with self.subTest(changes=changes):
                write_json(dest / "dependency.json", original | changes)
                with self.assertRaises(ValueError):
                    kernels.verify(dest)
        write_json(dest / "dependency.json", original | {"rationale": "r" * 8000})
        kernels.verify(dest)
        with self.assertRaises(ValueError):
            kernels.scalar(10 ** 1000)
