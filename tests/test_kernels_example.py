"""OpenBLAS example preflight and failed-launch retention without native builds."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("openblas_example", ROOT / "kernels/examples/openblas/example.py")
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


class OpenBlasExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.library = self.root / "provider.bin"
        self.library.write_bytes(b"Synthetic bytes; never load as a library")
        self.notice = self.root / "NOTICE.txt"
        self.notice.write_text("Synthetic test notice, not a project license.")
        self.args = argparse.Namespace(work=self.root / "work", library=self.library,
            notice=[self.notice], runtime_file=[], metadata=[], runtime_requirement=[], integer_bits=64,
            sgemm_symbol="fixture_sgemm", config_symbol="fixture_config", threads_symbol="fixture_threads",
            variant="synthetic-fixture", repository="fixture", revision=None, license="LicenseRef-Test")
        self.runtime = patch.dict(os.environ, {"CRIT_RUNTIME_DIRECTORY": str(self.root / "runtime")})
        self.runtime.start()
        self.addCleanup(self.runtime.stop)

    def prepare(self):
        value = example.prepare(self.args)
        self.assertIn("unqualified", value["qualification"])
        return self.args.work / "dependency"

    def test_prepare_is_data_only_and_retains_inputs(self):
        with patch.object(example, "guarded_tree", side_effect=AssertionError("must not launch")):
            dependency = self.prepare()
        self.assertEqual((dependency / "artifacts/lib/provider.bin").read_bytes(), self.library.read_bytes())
        self.assertEqual((dependency / "artifacts/notices/0-NOTICE.txt").read_bytes(), self.notice.read_bytes())
        with self.assertRaisesRegex(ValueError, "new work"):
            example.prepare(self.args)
        self.assertFalse((self.root / "runtime").exists())

    def test_tampering_is_rejected_before_execution(self):
        dependency = self.prepare()
        (dependency / "artifacts/lib/provider.bin").write_bytes(b"changed")
        with patch.object(example, "guarded_tree") as launch, self.assertRaisesRegex(ValueError, "inventory"):
            example.check(dependency, self.library, self.root / "results")
        launch.assert_not_called()
        self.assertFalse((self.root / "results").exists())

    def test_invalid_symbols_and_duplicate_runtime_names_fail_before_copy(self):
        self.args.sgemm_symbol = "invalid;symbol"
        with self.assertRaisesRegex(ValueError, "ASCII"):
            example.prepare(self.args)
        self.args.sgemm_symbol = "fixture_sgemm"
        self.args.runtime_file = [self.library]
        with self.assertRaisesRegex(ValueError, "[Cc]ase-aliased"):
            example.prepare(self.args)
        self.assertFalse(self.args.work.exists())

    def test_launch_failure_retains_diagnostics_and_releases_ownership(self):
        dependency = self.prepare()
        output = self.root / "results"
        with patch.object(example, "guarded_tree", side_effect=ValueError("launch failed")):
            with self.assertRaisesRegex(ValueError, "launch failed"):
                example.check(dependency, self.library, output)
        record = example.read_json(output / "result.json")
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["cleanup_settled"])
        self.assertFalse((self.root / "runtime/owner.json").exists())

    def test_output_creation_race_never_writes_into_another_directory(self):
        dependency = self.prepare()
        output = self.root / "results"
        mkdir = Path.mkdir
        def create_racer(path, *args, **kwargs):
            if path == output:
                mkdir(path)
                (path / "other.txt").write_text("Another writer owns this directory")
                raise FileExistsError("output race")
            return mkdir(path, *args, **kwargs)
        with patch.object(Path, "mkdir", create_racer), patch.object(example, "guarded_tree") as launch:
            with self.assertRaisesRegex(ValueError, "output race"):
                example.check(dependency, self.library, output)
        launch.assert_not_called()
        self.assertEqual([p.name for p in output.iterdir()], ["other.txt"])
        self.assertFalse((self.root / "runtime/owner.json").exists())

    def test_attention_profile_requires_coverage_finite_errors_and_settled_cleanup(self):
        dependency = self.prepare()
        native = {"schema_version": 1, "record_kind": "attention_check", "cases": 2304,
            "integration_cases": 8, "integer_bits": 64, "configuration": "OpenBLAS synthetic-fixture USE64BITINT",
            "max_output_error": 1e-7, "max_logsumexp_error": 2e-7}
        def launch(command, output, log, owner, _):
            self.assertEqual(command[1], "openblas")
            log.write(json.dumps(native).encode("utf-8"))
            log.flush()
            return SimpleNamespace(process=SimpleNamespace(poll=lambda: 0, returncode=0), close=lambda: None)
        with patch.object(example, "guarded_tree", launch):
            result = example.check(dependency, self.library, self.root / "good", profile="attention")
            self.assertEqual(result["record_kind"], "openblas_attention_validation")
            self.assertTrue(result["cleanup_settled"])
            for i, (field, bad) in enumerate((("cases", 648), ("integration_cases", 7),
                    ("integer_bits", 32), ("max_output_error", True), ("max_output_error", .01),
                    ("max_logsumexp_error", float("inf")))):
                original = native[field]
                native[field] = bad
                output = self.root / str(i)
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    example.check(dependency, self.library, output, profile="attention")
                record = example.read_json(output / "result.json")
                self.assertEqual(record["status"], "failed")
                self.assertTrue(record["cleanup_settled"])
                native[field] = original
        self.assertFalse((self.root / "runtime/owner.json").exists())
