"""Synthetic retained-artifact fixtures; no inference/build/model process starts."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from support.runs import ROOT, RunFixture, timestamp_for_test, with_plugins
from crit import __version__
from crit.cli import REPOSITORY, Installed
from crit.run.records import (GIB, candidate_projection, digest, file_sha, ineligibility, inventory, now, read_json, safe_path,
                              write_json)
from crit.run.store import PUBLISHED, RunStore, lease


class RunRecordTests(RunFixture, unittest.TestCase):
    def test_rejected_proposal_leaves_no_candidate_folder(self):
        with self.assertRaises(ValueError):
            self.candidate(label="x" * 500)
        self.assertEqual(list((self.directory / "candidates").glob("candidate-*")), [])
        self.assertEqual(self.candidate()["candidate_id"], "candidate-0001")

    def test_a_prepared_work_folder_does_not_break_candidate_creation(self):
        (self.directory / "work" / "candidate-0001").mkdir(parents=True)
        self.assertEqual(self.candidate()["candidate_id"], "candidate-0001")
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_start_identity_and_input_freezing_precede_construction(self):
        self.assertEqual(self.run["request_sha256"], digest(self.run["spec"]))
        self.assertFalse((self.directory / "job.json").exists())
        with self.assertRaises(ValueError):
            self.store.candidate(self.run_id, self.root, self.proposal())
        self.resolve()
        with self.assertRaises(FileExistsError):
            self.resolve()
        handoff = (self.directory / "agent_request.md").read_text(encoding="utf-8")
        # The brief holds the run's facts and points to AGENTS.md and crit run ... next for the rest.
        for line in ("Follow AGENTS.md", f"--id {self.run_id} next\n", "Repository: ", "Run folder: ",
                     "The user's terms", "crit serve --workspace"):
            self.assertIn(line, handoff)
        self.assertLess(len(handoff.split()), 600)
        self.assertNotIn("create_engine(context)", handoff)
        self.assertNotIn("--crit-chat-request", handoff)
        self.assertIn(f'crit serve --workspace "{self.store.root}" --run {self.run_id} --candidate CANDIDATE_ID --project "{self.store.project}"', handoff)
        segment_handoff = self.directory / "segments/segment-0001/agent_request.md"
        self.assertEqual(segment_handoff.read_text(encoding="utf-8"), handoff)

    def test_run_root_rejects_eval_sources_before_creating_directories(self):
        project = self.root / "project"
        header = project / "core/include/crit/core.h"
        header.parent.mkdir(parents=True)
        header.write_text("synthetic project marker", encoding="utf-8")
        evaluation = project / "eval"
        evaluation.mkdir()
        suite = evaluation / "protocol.json"
        suite.write_text('{"fixture": true}', encoding="utf-8")
        before = sorted(p.relative_to(project) for p in project.rglob("*"))
        for target in (evaluation, evaluation / "runs/new"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "outside publication"):
                RunStore(target, project=project)
        self.assertEqual(sorted(p.relative_to(project) for p in project.rglob("*")), before)
        self.assertEqual(suite.read_text(encoding="utf-8"), '{"fixture": true}')

    def test_request_rejects_legacy_and_invalid_fields_before_creation(self):
        for update in ({"job_id": "old"}, {"budget_minutes": True}, {"kv_tokens": 10}, {"targets": {"decode_tps": float("nan")}}, {"workload": None}):
            with self.subTest(update=update), self.assertRaises((ValueError, TypeError)):
                self.store.create(self.data | update)
        self.assertEqual(len(list(self.store.root.iterdir())), 1)

    def test_kv_capacity_is_eval_max_context_on_the_requested_device(self):
        model = self.root / "model"
        model.mkdir()
        header = json.dumps({"lm_head.weight": {"dtype": "BF16", "shape": [64, 64], "data_offsets": [0, 8192]}}).encode()
        (model / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(8192))
        # Two full-attention layers of one KV head of 8: 32 bytes per token at FP8.
        (model / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                                                       "head_dim": 8, "max_position_embeddings": 262144}), encoding="utf-8")
        device = self.root / "device.json"
        # 1 GiB headroom and the 3 GiB reserve leave the weights and 100,000 tokens of KV cache.
        device.write_text(json.dumps({"device": "fixture gpu", "memory_free_bytes": 4 * GIB + 8192 + 32 * 100000}), encoding="utf-8")
        data = {key: value for key, value in self.data.items() if key != "kv_tokens"} | {"model_path": str(model)}
        with self.assertRaisesRegex(ValueError, "Request needs device"):
            self.store.create(data)
        view = self.store.create(data | {"device": str(device)})
        hardware = view["run"]["spec"]["hardware_constraints"]
        self.assertEqual(hardware["kv_tokens"], 100000 // 1024 * 1024)
        basis = hardware["context"]
        self.assertEqual((basis["source"], basis["bound"], basis["model_limit"]), ("eval", "device", 262144))
        self.assertEqual((basis["device"], basis["device_file"], basis["device_sha256"]),
                         ("fixture gpu", str(device), file_sha(device)))
        prompt = (self.store.directory(view["run"]["run_id"]) / "agent_request.md").read_text(encoding="utf-8")
        self.assertIn("every candidate serves 99,328 tokens of prompt plus output, Eval's max context", prompt)
        self.assertIn("--context 99328", prompt)
        # Sixteen concurrent requests of 512 + 128 tokens hold 10,240 tokens of KV at once: a wanted context of 8,192
        # is raised to it, within what the device holds.
        sized = self.store.create(data | {"device": str(device), "context_tokens": 8192,
                                          "workload": {"concurrency": 16}})["run"]["spec"]["hardware_constraints"]
        self.assertEqual(sized["kv_tokens"], 10240)
        self.assertEqual((sized["context"]["concurrency_tokens"], sized["context"]["context_tokens"],
                          sized["context"]["bound"]), (10240, 8192, "requested"))
        # An explicit capacity is taken as given.
        explicit = self.store.create(data | {"device": str(device), "kv_tokens": 65536})["run"]["spec"]["hardware_constraints"]
        self.assertEqual(explicit["kv_tokens"], 65536)
        self.assertNotIn("context", explicit)
        # A device file of two GPUs (crit eval probe --devices) names them, and their memory holds more context.
        one = {"device": "fixture gpu", "memory_free_bytes": 4 * GIB + 8192 + 32 * 100000}
        device.write_text(json.dumps({**one, "devices": [{**one, "index": 0}, {**one, "index": 1}]}), encoding="utf-8")
        pair = self.store.create(data | {"device": str(device)})["run"]["spec"]["hardware_constraints"]
        self.assertEqual(pair["context"]["devices"], [{"index": 0, "device": "fixture gpu"},
                                                      {"index": 1, "device": "fixture gpu"}])
        self.assertGreater(pair["kv_tokens"], hardware["kv_tokens"])
        device.write_text(json.dumps({"memory_free_bytes": 4 * GIB}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Eval cannot derive the max context: the weights read on every pass"):
            self.store.create(data | {"device": str(device)})

    def test_complete_source_package_and_configuration_lineage(self):
        candidate = self.candidate()
        package = self.package(candidate)
        folder = self.directory / "candidates" / candidate["candidate_id"]
        self.assertTrue((folder / "source/engine.fixture").is_file())
        self.assertTrue((folder / "package/engine.fixture").is_file())
        self.on_ladder(candidate)
        second = self.candidate(parent_candidate_id=candidate["candidate_id"], configuration={"batch": 2})
        self.assertNotEqual(candidate["candidate_id"], second["candidate_id"])
        self.assertEqual(second["parent_candidate_id"], candidate["candidate_id"])
        self.assertEqual(package["configuration_sha256"], digest(candidate["configuration"]))

    def test_package_retains_exactly_the_candidate_configuration(self):
        candidate = self.candidate(configuration={"batch": 3})
        self.package(candidate)
        folder = self.directory / "candidates" / candidate["candidate_id"] / "package"
        self.assertEqual(json.loads((folder / "configuration.json").read_text(encoding="utf-8")),
                         candidate["configuration"])
        self.assertEqual(candidate["configuration"]["batch"], 3)
        manifest = read_json(folder / "manifest.json")
        self.assertIn("configuration.json", [row["path"] for row in manifest["files"]])
        other = self.candidate(configuration={"batch": 4})
        source = with_plugins(self.root / "package-mismatch")
        (source / "configuration.json").write_text('{"batch": 5}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from the candidate configuration"):
            self.store.package(self.run_id, other["candidate_id"], source,
                {"build": {"toolchain": {"identity": "host fixture"}, "dependencies": [], "command": ["fixture"]}})
        self.assertFalse((self.directory / "candidates" / other["candidate_id"] / "package").exists())

    def test_a_candidate_and_its_package_are_plugin_packages(self):
        """crit run takes an engine only as Core's loader runs it: plugins/ and a configuration whose plugins
        selection fills every slot an engine fills. A launch entry or capabilities are no recipe's to give."""
        recipe = {"build": {"toolchain": {"identity": "host fixture"}, "dependencies": [], "command": ["fixture"]}}
        source = self.root / "monolith"
        source.mkdir()
        (source / "engine.py").write_text("def create_engine(context): ...", encoding="utf-8")
        if not (self.directory / "inputs.json").exists():
            self.resolve()
        with self.assertRaisesRegex(ValueError, "The candidate's source is no engine Core can load: .*has no plugins/ "
                                                "folder"):
            self.store.candidate(self.run_id, source, self.proposal())
        with_plugins(source)
        with self.assertRaisesRegex(ValueError, "the configuration names no plugins"):
            self.store.candidate(self.run_id, source, {**self.proposal(), "configuration": {"batch": 1}})
        selection = self.proposal()["configuration"]["plugins"]
        del selection["use"]["sampler"]
        with self.assertRaisesRegex(ValueError, "leaves sampler empty"):
            self.store.candidate(self.run_id, source, {**self.proposal(), "configuration": {"plugins": selection}})
        candidate = self.store.candidate(self.run_id, source, self.proposal())
        bare = self.root / "bare-package"
        bare.mkdir()
        (bare / "engine.py").write_text("def create_engine(context): ...", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "The package is no engine Core can load"):
            self.store.package(self.run_id, candidate["candidate_id"], bare, recipe)
        with self.assertRaisesRegex(ValueError, "Unknown fields; use extensions: capabilities, launch"):
            self.store.package(self.run_id, candidate["candidate_id"], with_plugins(bare),
                               {**recipe, "launch": ["engine.py"], "capabilities": ["chat-json-v1"]})
        manifest = self.store.package(self.run_id, candidate["candidate_id"], bare, recipe)
        self.assertFalse({"launch", "capabilities"} & set(manifest))

    def test_weights_policy_keeps_transformed_candidates_ineligible(self):
        view = self.store.create(dict(self.data, weights_policy="as_stored"))
        self.run, self.run_id = view["run"], view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        self.assertEqual(self.run["spec"]["correctness"]["weights_policy"], "as_stored")
        with self.assertRaises(ValueError):
            self.candidate()  # the run requires a declaration
        with self.assertRaises(ValueError):
            self.candidate(weights="requantized")
        kept = self.measured(weights="as_stored")[0]
        changed = self.measured(weights="transformed")[0]
        for c in (kept, changed):
            self.store.seal(self.run_id, c["candidate_id"], "completed")
        result = {c["candidate_id"]: c for c in self.store.get(self.run_id)["candidates"]}
        self.assertTrue(result[kept["candidate_id"]]["eligible"])
        self.assertFalse(result[changed["candidate_id"]]["eligible"])
        self.assertIn("as stored", result[changed["candidate_id"]]["extensions"]["ineligible_reason"])
        with self.assertRaises(ValueError):
            self.store.select(self.run_id, changed["candidate_id"])
        self.store.select(self.run_id, kept["candidate_id"])
        with self.assertRaises(ValueError):
            self.store.create(dict(self.data, weights_policy="sometimes"))

    def test_speed_runs_default_to_transformable_weights(self):
        # A request that names neither is in speed, which may re-encode stored weights.
        self.assertEqual(self.run["spec"]["correctness"]["weights_policy"], "transformable")
        view = self.store.create(dict(self.data, quality_mode="speed"))
        self.assertEqual(view["run"]["spec"]["correctness"]["weights_policy"], "transformable")
        kept = self.store.create(dict(self.data, quality_mode="speed", weights_policy="as_stored"))
        self.assertEqual(kept["run"]["spec"]["correctness"]["weights_policy"], "as_stored")
        preserve = self.store.create(dict(self.data, quality_mode="preserve"))["run"]["spec"]["correctness"]
        self.assertNotIn("weights_policy", preserve)
        self.assertEqual(self.store.create(dict(self.data, quality_mode="preserve", weights_policy="transformable"))
                         ["run"]["spec"]["correctness"]["weights_policy"], "transformable")

    def test_every_run_record_states_its_quality_mode(self):
        # speed is the default, recorded explicitly so the run never depends on what the default was.
        self.assertEqual(self.run["spec"]["correctness"]["quality_mode"], "speed")
        self.assertEqual(self.store.create(dict(self.data, quality_mode="preserve"))["run"]["spec"]["correctness"]
                         ["quality_mode"], "preserve")
        # A record that states no mode, or one no longer named, is refused: no mode is read for another.
        from crit.run import records
        for mode in (None, "as_quantized"):
            with self.subTest(mode=mode):
                path = self.directory / "run.json"
                old = read_json(path)
                if mode is None:
                    del old["spec"]["correctness"]["quality_mode"]
                else:
                    old["spec"]["correctness"]["quality_mode"] = mode
                old["request_sha256"] = digest(old["spec"])
                write_json(path, old)
                with self.assertRaisesRegex(ValueError, "quality_mode requires preserve or speed"):
                    records.load_run(self.directory)

    def test_crit_names_the_quality_modes_and_their_default_as_crit_eval_does(self):
        # crit imports crit Eval only where a command needs it (default_quality_budgets says how to install it), so
        # it keeps its own copy of the modes; the two copies stay alike.
        from crit.run import records
        from crit_eval.quality import fixed
        self.assertEqual((records.QUALITY_MODES, records.DEFAULT_QUALITY_MODE),
                         (fixed.QUALITY_MODES, fixed.DEFAULT_QUALITY_MODE))
        self.assertEqual(set(records.default_quality_budgets()), set(records.QUALITY_MODES))

    def test_parallel_inventory_matches_sequential_hashing(self):
        folder = self.root / "artifact"
        folder.mkdir()
        for i in range(12):
            (folder / f"part-{i:02d}.bin").write_bytes(bytes([i]) * (4096 + i))
        sequential = inventory(folder)
        with patch("crit.run.records.HASH_PARALLEL_BYTES", 0):
            self.assertEqual(inventory(folder), sequential)

    def test_busy_lease_names_its_owner_and_refuses_recovery_from_another_system(self):
        lock = self.directory / ".coordinator.lock"
        lock.write_text(json.dumps({"pid": 4242, "host": "another-host", "system": "OtherOS", "started_at": now()}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "PID 4242 on another-host"):
            self.store.resume(self.run_id, 1, external_cleanup_confirmed=True)
        with self.assertRaisesRegex(ValueError, "recover from that host"):
            self.store.recover(self.run_id)
        self.assertTrue(lock.exists())
        lock.unlink()

    def test_raw_samples_medians_pairing_and_selection_without_target_satisfaction(self):
        candidate, package, checks, data = self.measured()
        active = self.store.get(self.run_id)["candidates"][0]
        self.assertFalse(active["eligible"])
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        view = self.store.select(self.run_id, candidate["candidate_id"])
        result = view["candidates"][0]
        self.assertEqual(result["series"][0]["metrics"]["decode_tps"]["median"], 11)
        self.assertEqual(result["series"][0]["metrics"]["prefill_tps"]["median"], 110)
        self.assertTrue(result["eligible"])
        self.assertFalse(view["state"]["target_satisfied"])
        with self.assertRaises(ValueError):
            self.store.stop(self.run_id, "satisfied", critique="fixture critique")

    def test_two_reads_of_crits_clock_never_share_a_time_while_the_wall_clock_is_behind(self):
        # Records written while a stepped-back wall clock catches up (an install, then its reinstall) keep distinct,
        # increasing times.
        from datetime import datetime as wall
        from crit import records

        class Clock(wall):
            back = timedelta(0)

            @classmethod
            def now(cls, tz=None):
                return wall.now(tz) - cls.back
        with patch.object(records, "datetime", Clock):
            first = records.utc_now()
            Clock.back = timedelta(seconds=1)
            reads = [records.utc_now() for _ in range(3)]
        self.assertEqual(sorted(set([first] + reads)), [first] + reads)

    def test_records_stay_in_their_segment_when_the_wall_clock_steps_back(self):
        # A virtual machine's clock kept in step with its host's moves back by up to a second at times (WSL's does,
        # every half minute or so): a check recorded after its candidate in the same process must not read as recorded
        # before it, by however much the clock stepped (here past the slack two processes get).
        from datetime import datetime as wall
        from crit import records
        from crit.run.records import CLOCK_SLACK

        class Clock(wall):
            back = timedelta(0)

            @classmethod
            def now(cls, tz=None):
                return wall.now(tz) - cls.back
        with patch.object(records, "datetime", Clock):
            candidate = self.candidate()
            Clock.back = CLOCK_SLACK + timedelta(seconds=1)
            package = self.package(candidate)
            data = self.benchmark_data(package, self.checks(candidate, package))
            self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data)
            self.assertGreaterEqual(records.utc_now(), timestamp_for_test(candidate["created_at"]))
        view = self.store.get(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertEqual(len(view["candidates"][0]["series"]), 1)

    def test_records_another_process_stamped_a_moment_early_stay_in_their_segment(self):
        # Two processes' clocks can disagree by a step back of the wall clock (records.CLOCK_SLACK): a benchmark
        # stamped half a second before its candidate's creation still belongs to it, one stamped past the slack not.
        from crit.run.records import CLOCK_SLACK
        candidate = self.candidate()
        package = self.package(candidate)
        created = timestamp_for_test(candidate["created_at"])
        checks = self.checks(candidate, package)
        early = (created - timedelta(seconds=0.5)).isoformat()
        self.store.append(self.run_id, candidate["candidate_id"], "benchmark",
                          self.benchmark_data(package, checks, started_at=early))
        view = self.store.get(self.run_id)
        self.assertEqual(view["errors"], [])
        self.assertEqual(len(view["candidates"][0]["series"]), 1)
        too_early = (created - CLOCK_SLACK - timedelta(seconds=0.5)).isoformat()
        with self.assertRaisesRegex(ValueError, "Benchmark outside segment"):
            self.store.append(self.run_id, candidate["candidate_id"], "benchmark",
                              self.benchmark_data(package, checks, started_at=too_early))

    def test_failed_gate_keeps_exploratory_speeds_and_cannot_be_selected(self):
        candidate, *_ = self.measured(arithmetic="failed")
        self.store.seal(self.run_id, candidate["candidate_id"], "check_failed", {"message": "fixed numerical failure"})
        result = self.store.get(self.run_id)["candidates"][0]
        self.assertEqual(result["series"][0]["metrics"]["decode_tps"]["median"], 11)
        self.assertEqual(result["series"][0]["qualification"], "exploratory")
        self.assertEqual(result["evidence_level"], "artifact_verified", "Retained-byte verification is separate from a failed numerical gate")
        self.assertFalse(result["eligible"])
        with self.assertRaises(ValueError):
            self.store.select(self.run_id, candidate["candidate_id"])

    def test_measure_warning_reads_the_unsealed_candidates_own_qualification(self):
        import contextlib, io
        from crit.run.cli import warn_exploratory
        good, *_ = self.measured()
        bad, *_ = self.measured(arithmetic="failed")
        for candidate, expected in ((good, ""), (bad, "gate arithmetic failed")):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                warn_exploratory(self.store, self.run_id, candidate["candidate_id"], "benchmark-0001")
            self.assertIn(expected, stderr.getvalue())
            self.assertEqual(stderr.getvalue() == "", expected == "")

    def test_benchmark_without_bound_checks_names_why_the_candidate_is_ineligible(self):
        candidate = self.candidate()
        package = self.package(candidate)
        # Published before the case's checks, so it cites none.
        self.store.append(self.run_id, candidate["candidate_id"], "benchmark", self.benchmark_data(package, []))
        self.checks(candidate, package)
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        view = self.store.get(self.run_id)
        result = view["candidates"][0]
        self.assertFalse(result["eligible"])
        # The reason is derived when asked: telemetry keeps the projection runs recorded by crit 0.1.0 retain.
        self.assertEqual(view["errors"], [])
        telemetry = read_json(self.directory / "candidates" / candidate["candidate_id"] / "telemetry.json")
        self.assertNotIn("ineligible_reason", telemetry["extensions"])
        reason = "; ".join(ineligibility(view["run"], result))
        self.assertIn("benchmark-0001 (case-0001) is exploratory", reason)
        self.assertIn("publish the case's checks first", reason)
        with self.assertRaisesRegex(ValueError, "cites no passed check for arithmetic"):
            self.store.select(self.run_id, candidate["candidate_id"])

    def test_benchmark_without_cited_checks_binds_the_cases_checks(self):
        candidate = self.candidate()
        package = self.package(candidate)
        latest = self.checks(candidate, package)
        data = self.benchmark_data(package, [])
        del data["checks"]
        bench = self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data)
        self.assertEqual(bench["checks"], latest)
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        self.assertTrue(self.store.get(self.run_id)["candidates"][0]["eligible"])

    def test_reported_evidence_does_not_satisfy_verified_request(self):
        candidate, *_ = self.measured(evidence_level="reported")
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        self.assertFalse(self.store.get(self.run_id)["candidates"][0]["eligible"])

    def test_headroom_uses_worst_sample_and_missing_metrics_have_no_zero_values(self):
        candidate = self.candidate()
        package = self.package(candidate)
        checks = self.checks(candidate, package)
        data = self.benchmark_data(package, checks)
        data["metrics"]["minimum_free_vram_bytes"]["samples"][0] = GIB - 1
        del data["metrics"]["prefill_tps"]
        data["missing_metrics"]["prefill_tps"] = "Not measured by fixture"
        self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data)
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        result = self.store.get(self.run_id)["candidates"][0]
        self.assertNotIn("prefill_tps", result["series"][0]["metrics"])
        self.assertFalse(result["eligible"])

    def test_quality_and_acceptance_metrics_are_optional_and_bounded(self):
        candidate = self.candidate()
        package = self.package(candidate)
        checks = self.checks(candidate, package)
        bad = self.benchmark_data(package, checks)
        bad["metrics"]["accepted_tokens_per_round"] = {"samples": [3.5, 0.5, 3.6], "unit": "tokens/round"}
        with self.assertRaises(ValueError):
            self.store.append(self.run_id, candidate["candidate_id"], "benchmark", bad)
        wrong_unit = self.benchmark_data(package, checks)
        wrong_unit["metrics"]["kl_divergence"] = {"samples": [0.01] * 3, "unit": "dimensionless"}
        with self.assertRaises(ValueError):
            self.store.append(self.run_id, candidate["candidate_id"], "benchmark", wrong_unit)
        data = self.benchmark_data(package, checks)
        data["metrics"]["kl_divergence"] = {"samples": [0.0, 0.013, 0.013], "unit": "nats/token"}
        data["metrics"]["accepted_tokens_per_round"] = {"samples": [3.7, 3.8, 3.6], "unit": "tokens/round"}
        self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data)
        metrics = self.store.get(self.run_id)["candidates"][0]["series"][0]["metrics"]
        self.assertEqual((metrics["kl_divergence"]["median"], metrics["accepted_tokens_per_round"]["median"]), (0.013, 3.7))

    def test_multiple_benchmarks_remain_paired_with_no_cherry_picked_best(self):
        candidate, package, checks, first = self.measured()
        second = copy.deepcopy(first)
        second["metrics"]["decode_tps"]["samples"] = [30, 32, 31]
        second["metrics"]["prefill_tps"]["samples"] = [50, 52, 51]
        self.store.append(self.run_id, candidate["candidate_id"], "benchmark", second)
        result = self.store.get(self.run_id)["candidates"][0]
        self.assertEqual([(s["metrics"]["decode_tps"]["median"],s["metrics"]["prefill_tps"]["median"]) for s in result["series"]], [(11,110),(31,51)])
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        result = self.store.select(self.run_id, candidate["candidate_id"])
        self.assertFalse(result["state"]["target_satisfied"], "A fast repeated benchmark cannot hide the slow one")

    def test_every_target_and_gate_met_can_satisfy_selected_candidate(self):
        candidate = self.candidate()
        package = self.package(candidate)
        checks = self.checks(candidate, package)
        data = self.benchmark_data(package, checks)
        data["metrics"]["decode_tps"]["samples"] = [20, 22, 21]
        self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data)
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        self.assertTrue(self.store.select(self.run_id, candidate["candidate_id"])["state"]["target_satisfied"])
        result = self.store.stop(self.run_id, "satisfied", critique="Bounded synthetic whole-engine critique", opportunities=[], cleanup="settled")
        self.assertEqual(result["state"]["stopping_reason"], "satisfied")

    def test_voluntary_stop_requires_an_account_of_every_kpi(self):
        with self.assertRaisesRegex(ValueError, "requires headroom"):
            self.store.stop(self.run_id, "search_exhausted", critique="critique")
        decode = {"kpi": "decode", "start": 201.5, "reached": 212.8, "limit": 306.0, "built": "smaller draft vocabulary",
                  "cause": "later draft depths are accepted less often than the first"}
        with self.assertRaisesRegex(ValueError, "account of every KPI"):
            self.store.stop(self.run_id, "budget_exhausted", critique="critique", headroom={"kpis": [decode]})
        with self.assertRaisesRegex(ValueError, "nonempty text"):
            self.store.stop(self.run_id, "blocked", critique="critique", headroom={"kpis": [{**decode, "cause": " "}]})
        with self.assertRaisesRegex(ValueError, "each at most once"):
            self.store.stop(self.run_id, "blocked", critique="critique", headroom={"kpis": [decode, decode]})
        with self.assertRaises(ValueError):
            self.store.stop(self.run_id, "blocked", critique="critique", headroom={"kpis": [{**decode, "kpi": "vibes"}]})
        # Stopping with budgeted candidates left is a stop plan first.
        self.first_plan(decision="stop", needed={name: {"needed": False, "evidence": "fixture"}
                                                 for name in ("decode", "prefill")})
        self.store.stop(self.run_id, "blocked", critique="critique", headroom={"kpis": [decode]})
        with self.assertRaises(ValueError):  # the old shape without a start or a cause
            self.store.stop(self.run_id, "blocked", critique="critique", headroom={"kpis": [
                {"kpi": "decode", "score": 212.8, "limit": 306.0, "built": "x", "stopped": "y"}]})
        account = {"kpis": [decode, {**decode, "kpi": "prefill", "start": None, "reached": 5203.0, "limit": 13906.0},
                            {**decode, "kpi": "ttft", "reached": 241.0, "limit": None},
                            # The long-context goals join when Eval measured the required context.
                            {**decode, "kpi": "context_decode", "reached": 151.0, "limit": 240.0},
                            {**decode, "kpi": "context_ttft", "reached": 41000.0, "limit": 30500.0}]}
        result = self.store.stop(self.run_id, "search_exhausted", critique="critique", headroom=account)
        self.assertEqual(result["state"]["stopping_reason"], "search_exhausted")

    def test_candidate_budget_replaces_the_time_budget(self):
        data = {key: value for key, value in self.data.items() if key != "budget_minutes"} | {"candidate_budget": 2}
        self.run_id = self.store.create(data)["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        bounds = self.store.get(self.run_id)["segment"]
        self.assertEqual(bounds["limits"]["candidate_budget"], 2)
        prompt = (self.directory / "agent_request.md").read_text(encoding="utf-8")
        self.assertIn("candidate budget: 2 candidates", prompt)
        self.assertNotIn("hard deadline", prompt)
        headroom = {"kpis": [{"kpi": kpi, "start": 1.0, "reached": 1.0, "limit": 2.0, "built": "fixture", "cause": "fixture"}
                             for kpi in ("decode", "prefill", "ttft")]}
        # A draft costs nothing; a candidate on the ladder spends the budget.
        draft = self.candidate()
        self.store.seal(self.run_id, draft["candidate_id"], "build_failed", {"message": "fixture"})
        first = self.on_ladder(self.candidate())
        with self.assertRaisesRegex(ValueError, "Only 1 of 2 budgeted candidates"):
            self.store.stop(self.run_id, "budget_exhausted", critique="critique", headroom=headroom)
        second = self.on_ladder(self.candidate(parent_candidate_id=first))
        with self.assertRaisesRegex(ValueError, "candidate budget \\(2 candidates on the candidate ladder\\) is used"):
            self.candidate(parent_candidate_id=second)
        self.store.stop(self.run_id, "budget_exhausted", critique="critique", headroom=headroom)
        result = self.store.resume(self.run_id, attempt_limit=8, external_cleanup_confirmed=True, candidates=4)
        self.assertEqual(result["segment"]["limits"]["candidate_budget"], 4)
        self.assertIsNotNone(self.candidate(parent_candidate_id=second))

    def test_quality_budget_and_exactness_reach_the_agent_request(self):
        # A request that names no quality mode is in speed, with Eval's budget for it, eight times
        # preserve's, and the brief says what the mode means: drift from the selected checkpoint traded for speed.
        prompt = (self.directory / "agent_request.md").read_text(encoding="utf-8")
        self.assertIn("PPL ratio ≤ 1.4, mean KLD ≤ 0.4, p99 KLD ≤ 4.0", prompt)
        self.assertIn("exactness reported", prompt)
        self.assertIn("Spend it on speed", prompt)
        self.assertIn("Quality mode speed", prompt)
        self.assertIn("--quality-mode speed", prompt)
        self.assertIn("output forks at near-ties pass", prompt)
        data = self.data | {"quality_budget": {"ppl_ratio_max": 1.0, "kld_mean_max": 0.0}, "exactness": "required"}
        run_id = self.store.create(data)["run"]["run_id"]
        strict = (self.store.directory(run_id) / "agent_request.md").read_text(encoding="utf-8")
        self.assertIn("PPL ratio ≤ 1.0, mean KLD ≤ 0.0, p99 KLD ≤ 4.0", strict)
        self.assertIn("--exactness required", strict)
        for bad in ({"quality_budget": {"ppl_ratio_max": 0.9}}, {"quality_budget": {"kld_max": 1}}, {"exactness": "maybe"},
                    {"quality_mode": "fast"}, {"quality_mode": None}, {"quality_mode": "as_quantized"}):
            with self.assertRaises(ValueError):
                self.store.create(self.data | bad)
        # The user's other quality mode: the run records it, with Eval's budget for it, and the brief says what it means.
        run = self.store.create(self.data | {"quality_mode": "preserve"})["run"]
        self.assertEqual(run["spec"]["correctness"]["quality_mode"], "preserve")
        kept = (self.store.directory(run["run_id"]) / "agent_request.md").read_text(encoding="utf-8")
        self.assertIn("Quality mode preserve (the run's; never change it)", kept)
        self.assertIn("--quality-mode preserve", kept)
        self.assertIn("PPL ratio ≤ 1.05, mean KLD ≤ 0.05, p99 KLD ≤ 0.5", kept)
        self.assertIn("activation quantization, a lower-precision KV cache or head", kept)
        self.assertNotIn("near-ties", kept.split("Quality budget:")[1].split("\n")[0])

    def test_resume_amends_quality_terms_for_later_segments(self):
        self.store.stop(self.run_id, "cancelled", critique="critique")
        with self.assertRaisesRegex(ValueError, "approved_by"):
            self.store.resume(self.run_id, external_cleanup_confirmed=True, candidates=2, amend={"exactness": "reported"})
        amend = {"quality_budget": {"ppl_ratio_max": 1.1}, "exactness": "required", "approved_by": "Yan"}
        result = self.store.resume(self.run_id, external_cleanup_confirmed=True, candidates=2, amend=amend)
        segment_id = result["segment"]["segment_id"]
        prompt = (self.directory / "segments" / segment_id / "agent_request.md").read_text(encoding="utf-8")
        self.assertIn("PPL ratio ≤ 1.1, mean KLD ≤ 0.4", prompt)   # over the run's mode's budget (speed)
        self.assertIn("--exactness required", prompt)
        self.assertIn("approved by Yan", prompt)
        self.store.stop(self.run_id, "cancelled", critique="critique")
        later = self.store.resume(self.run_id, external_cleanup_confirmed=True, candidates=1)
        prompt = (self.directory / "segments" / later["segment"]["segment_id"] / "agent_request.md").read_text(encoding="utf-8")
        self.assertIn("PPL ratio ≤ 1.1", prompt)  # amendments carry over

    def test_time_budget_run_can_resume_with_candidates(self):
        self.store.stop(self.run_id, "cancelled", critique="critique")
        result = self.store.resume(self.run_id, external_cleanup_confirmed=True, candidates=3)
        self.assertEqual(result["segment"]["limits"]["candidate_budget"], 3)
        with self.assertRaisesRegex(ValueError, "Resume requires --candidates"):
            self.store.resume(self.run_id, external_cleanup_confirmed=True)

    def test_later_failed_check_keeps_all_candidate_timings_exploratory(self):
        candidate, package, *_ = self.measured()
        self.checks(candidate, package, arithmetic="failed")
        result = self.store.get(self.run_id)["candidates"][0]
        self.assertEqual(result["gates"]["arithmetic"], "failed")
        self.assertTrue(all(series["qualification"] == "exploratory" for series in result["series"]))

    def test_changed_captured_inputs_exclude_candidates(self):
        candidate, *_ = self.measured()
        (self.directory / "captured/input.fixture").write_text("changed bytes")
        result = self.store.get(self.run_id)
        self.assertEqual(result["candidates"], [])
        self.assertTrue(any("evidence" in issue["error"] for issue in result["errors"]))

    def test_unknown_telemetry_version_and_duplicate_json_keys_are_rejected(self):
        candidate = self.candidate()
        path = self.directory / "candidates" / candidate["candidate_id"] / "telemetry.json"
        value = read_json(path)
        value["schema_version"] = 2
        write_json(path, value)
        self.assertEqual(self.store.get(self.run_id)["candidates"], [])
        path.write_text('{"duplicate":1,"duplicate":2}')
        with self.assertRaises(ValueError):
            read_json(path)

    def test_a_missing_json_file_is_named_as_missing(self):
        missing = self.directory / "no-such-record.json"
        with self.assertRaisesRegex(ValueError, "JSON file not found: .*no-such-record.json"):
            read_json(missing)
        with self.assertRaisesRegex(ValueError, r"JSON file not found: no-such-record\.json \(from "):
            read_json(Path("no-such-record.json"))
        big = self.directory / "big.json"
        big.write_text(" " * (2 << 20) + "{}")
        with self.assertRaisesRegex(ValueError, "JSON must be a regular file under 1 MiB"):
            read_json(big)

    def test_bool_cannot_masquerade_as_numeric_projection_or_ordinal(self):
        candidate = self.candidate()
        package = self.package(candidate)
        checks = self.checks(candidate, package)
        data = self.benchmark_data(package, checks)
        data["metrics"]["decode_tps"]["samples"] = [1, 1, 1]
        self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data)
        path = self.directory / "candidates" / candidate["candidate_id"] / "telemetry.json"
        value = read_json(path)
        value["series"][0]["metrics"]["decode_tps"]["median"] = True
        write_json(path, value)
        self.assertEqual(self.store.get(self.run_id)["candidates"], [])
        raw = path.parent / "candidate.json"
        value = read_json(raw)
        value["ordinal"] = True
        write_json(raw, value)
        with self.assertRaises(ValueError):
            candidate_projection(self.directory, self.run, candidate["candidate_id"])

    def test_distinct_case_workloads_create_distinct_comparison_groups(self):
        workload = self.run["spec"]["cases"][0]["workload"]
        self.view = self.store.create(self.data | {"cases": [{"case_id": "case-0001", "workload": workload},
            {"case_id": "case-0002", "workload": workload | {"prompt_tokens": 1024, "concurrency": 2}}]})
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        path = self.directory / "captured/input.fixture"
        path.write_text("fixed two-case fixture")
        rows = [{"path": "captured/input.fixture", "bytes": path.stat().st_size, "sha256": file_sha(path)}]
        self.resolved = self.store.freeze_inputs(self.run_id, {"hardware_manifest": {"device": "fixture"}, "environment_manifest": {"driver": "fixture"},
            "case_manifests": {"case-0001": rows, "case-0002": rows}})
        candidate = self.candidate()
        package = self.package(candidate)
        first = self.benchmark_data(package, [])
        self.store.append(self.run_id, candidate["candidate_id"], "benchmark", first)
        case = self.run["spec"]["cases"][1]
        second = first | {"case_id": case["case_id"], "case_sha256": digest(case), "workload": case["workload"]}
        self.store.append(self.run_id, candidate["candidate_id"], "benchmark", second)
        series = self.store.get(self.run_id)["candidates"][0]["series"]
        self.assertNotEqual(series[0]["comparison_group_id"], series[1]["comparison_group_id"])

    def test_unknown_version_identity_units_counts_and_nonfinite_rejected_before_publish(self):
        candidate = self.candidate()
        package = self.package(candidate)
        checks = self.checks(candidate, package)
        original = self.benchmark_data(package, checks)
        bad = [original | {"schema_version": 2}, original | {"case_sha256": "0" * 64}, original | {"protocol_sha256": "0" * 64}, original | {"unknown": 1}]
        for metric_update in ({"unit": "seconds"}, {"samples": [1]}, {"samples": [float("inf")] * 3}):
            changed = copy.deepcopy(original)
            changed["metrics"]["decode_tps"].update(metric_update)
            bad.append(changed)
        for data in bad:
            with self.subTest(data=data), self.assertRaises((ValueError, TypeError)):
                self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data)
        self.assertFalse((self.directory / "candidates" / candidate["candidate_id"] / "benchmarks").exists())

    def test_frozen_source_package_evidence_and_seal_mutation_exclude_candidate(self):
        candidate, *_ = self.measured()
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        folder = self.directory / "candidates" / candidate["candidate_id"]
        for relative in ("source/engine.fixture", "package/engine.fixture", "checks/raw/arithmetic.fixture", "telemetry.json"):
            path = folder / relative
            retained = path.read_bytes()
            path.write_bytes(retained + b" ")
            view = self.store.get(self.run_id)
            self.assertEqual(view["candidates"], [], relative)
            self.assertTrue(view["errors"], relative)
            path.write_bytes(retained)
        (folder / "unexpected.fixture").write_text("changed inventory")
        self.assertEqual(self.store.get(self.run_id)["candidates"], [])

    def test_an_operation_walks_each_retained_inventory_once_and_the_next_walks_it_again(self):
        from crit import records
        sealed, *_ = self.measured()
        self.store.seal(self.run_id, sealed["candidate_id"], "completed")
        candidate, _, _, data = self.measured(label="Second engine")
        walk, walked = records.inventory, []

        def counted(root, *args, **kwargs):
            walked.append(Path(root).relative_to(self.directory / "candidates").as_posix())
            return walk(root, *args, **kwargs)

        # The append projects the new record, refreshes the candidate's telemetry, and the run's state reads every
        # candidate: three reads of the appended candidate, one of the sealed one.
        with patch.object(records, "inventory", counted):
            self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data)
        first, second = sealed["candidate_id"], candidate["candidate_id"]
        self.assertEqual(sorted(walked), sorted([f"{second}/source", f"{second}/package", first, f"{first}/source",
                                                 f"{first}/package"]))
        path = self.directory / "candidates" / first / "package/engine.fixture"
        path.write_bytes(path.read_bytes() + b" ")
        view = self.store.get(self.run_id)
        self.assertEqual([c["candidate_id"] for c in view["candidates"]], [second])
        self.assertIn("candidates/" + first, [e["file"] for e in view["errors"]])

    def test_sealed_candidate_cannot_publish_new_measurements(self):
        candidate, package, checks, data = self.measured()
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        for action in (lambda: self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data),
                       lambda: self.store.reserve(self.run_id, candidate["candidate_id"], "check")):
            with self.assertRaises(ValueError):
                action()

    def test_failed_build_retains_source_logs_and_has_no_invented_series(self):
        candidate = self.candidate()
        reservation = self.store.reserve(self.run_id, candidate["candidate_id"], "build")
        self.assertEqual(self.store.get(self.run_id)["candidates"][0]["lifecycle"], "building")
        path = self.directory / "candidates" / candidate["candidate_id"] / "logs/failure.fixture"
        path.parent.mkdir()
        path.write_text("synthetic build failure")
        self.store.settle(self.run_id, reservation, {"outcome": "failed", "ended_at": now(), "exit_code": 1, "cleanup_settled": True,
            "logs": [{"path": path.relative_to(self.directory).as_posix(), "sha256": file_sha(path)}], "failure": "synthetic compiler failure"})
        self.store.seal(self.run_id, candidate["candidate_id"], "build_failed", {"message": "synthetic compiler failure"})
        result = self.store.get(self.run_id)["candidates"][0]
        self.assertEqual(result["series"], [])
        self.assertNotIn("package", result["artifacts"])
        self.assertTrue((path.parent.parent / "source/engine.fixture").is_file())

    def test_package_files_equal_to_source_files_are_stored_once(self):
        candidate = self.candidate()
        package = self.root / "shared-package"
        package.mkdir()
        (package / "engine.fixture").write_text("complete synthetic source", encoding="utf-8")  # same bytes as the source
        (package / "launcher.fixture").write_text("launcher only in the package", encoding="utf-8")
        self.store.package(self.run_id, candidate["candidate_id"], with_plugins(package),
            {"build": {"toolchain": {"identity": "host fixture"}, "dependencies": [], "command": ["fixture"]}})
        folder = self.directory / "candidates" / candidate["candidate_id"]
        shared, own = folder / "package/engine.fixture", folder / "package/launcher.fixture"
        self.assertEqual(shared.read_text(encoding="utf-8"), "complete synthetic source")
        if shared.stat().st_nlink > 1:  # hard links available on this filesystem
            self.assertTrue(os.path.samefile(shared, folder / "source/engine.fixture"))
        self.assertEqual(own.stat().st_nlink, 1)
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_a_completed_retry_supersedes_a_failed_launch(self):
        candidate = self.candidate()
        failed = self.store.reserve(self.run_id, candidate["candidate_id"], "build")
        self.store.settle(self.run_id, failed, {"outcome": "launch_failed", "ended_at": now(), "exit_code": None,
            "cleanup_settled": True, "logs": [], "failure": "runner did not start"})
        view = self.store.get(self.run_id)["candidates"][0]
        self.assertEqual((view["lifecycle"], view["failure"]["operation"]), ("build_failed", "build"))
        retry = self.store.reserve(self.run_id, candidate["candidate_id"], "build")
        self.store.settle(self.run_id, retry, {"outcome": "completed", "ended_at": now(), "exit_code": 0,
            "cleanup_settled": True, "logs": [], "failure": None})
        view = self.store.get(self.run_id)["candidates"][0]
        self.assertEqual((view["lifecycle"], view["failure"]), ("queued", None))  # no package yet, nothing failed
        self.assertEqual(view["attempt_count"], 2)

    def test_a_reservation_that_cannot_be_published_is_retracted(self):
        candidate = self.candidate()
        original = self.store._refresh
        def failing(root, candidate_id):
            raise OSError("synthetic projection write failure")
        self.store._refresh = failing
        try:
            with self.assertRaises(OSError):
                self.store.reserve(self.run_id, candidate["candidate_id"], "build")
        finally:
            self.store._refresh = original
        attempts = self.directory / "candidates" / candidate["candidate_id"] / "attempts"
        self.assertEqual(list(attempts.glob("attempt-*/reservation.json")), [])
        reservation = self.store.reserve(self.run_id, candidate["candidate_id"], "build")  # nothing blocks a new one
        self.store.settle(self.run_id, reservation, {"outcome": "failed", "ended_at": now(), "exit_code": 1,
            "cleanup_settled": True, "logs": [], "failure": "fixture failure"})
        self.store.seal(self.run_id, candidate["candidate_id"], "build_failed", {"message": "fixture failure"})

    def test_resume_appends_segment_preserves_request_seals_and_consumed_attempts(self):
        candidate = self.candidate()
        reservation = self.store.reserve(self.run_id, candidate["candidate_id"], "build")
        self.store.settle(self.run_id, reservation, {"outcome": "failed", "ended_at": now(), "exit_code": 1,
            "cleanup_settled": True, "logs": [], "failure": "fixture failure"})
        self.store.seal(self.run_id, candidate["candidate_id"], "build_failed", {"message": "fixture failure"})
        originals = {p: p.read_bytes() for p in (self.directory / "run.json", self.directory / "segments/segment-0001.json", self.directory / "candidates/candidate-0001/seal.json")}
        with self.assertRaises(ValueError):
            self.store.resume(self.run_id, 2, candidates=1)
        result = self.store.resume(self.run_id, 2, 3, True, candidates=1)
        self.assertEqual(result["segment"]["segment_id"], "segment-0002")
        self.assertEqual(result["state"]["budget"]["attempts"], 1)
        self.assertEqual(result["state"]["budget"]["segment_attempts"], 0)
        for path, value in originals.items():
            self.assertEqual(path.read_bytes(), value)

    def test_unsettled_attempts_and_unsealed_candidates_block_resume(self):
        candidate = self.candidate()
        with self.assertRaises(ValueError):
            self.store.resume(self.run_id, 1, external_cleanup_confirmed=True)
        self.store.reserve(self.run_id, candidate["candidate_id"], "build")
        with self.assertRaises(ValueError):
            self.store.seal(self.run_id, candidate["candidate_id"], "interrupted")
        with self.assertRaises(ValueError):
            self.store.resume(self.run_id, 1, external_cleanup_confirmed=True)
        self.assertFalse((self.directory / "segments/segment-0002.json").exists())

    def test_attempt_caps_and_deadlines_cannot_reset(self):
        candidate = self.candidate()
        bounds_path = self.directory / "segments/segment-0001.json"
        bounds = read_json(bounds_path)
        bounds["limits"]["attempt_limit"] = 1
        write_json(bounds_path, bounds)  # Only synthetic fixtures may alter authority for this test.
        reservation = self.store.reserve(self.run_id, candidate["candidate_id"], "build", 2)
        self.assertLessEqual(timestamp_for_test(reservation["deadline_at"]), timestamp_for_test(bounds["deadline_at"]))
        with self.assertRaises(ValueError):
            self.store.reserve(self.run_id, candidate["candidate_id"], "build", 2)
        self.assertEqual(self.store.get(self.run_id)["state"]["budget"]["attempts"], 1)

    def test_discovery_reservation_precedes_resolved_inputs_and_consumes_its_cap(self):
        reservation = self.store.reserve(self.run_id, None, "discovery", 10)
        self.assertFalse((self.directory / "inputs.json").exists())
        self.assertEqual(self.store.get(self.run_id)["state"]["phase"], "discovering")
        self.store.settle(self.run_id, reservation, {"outcome": "completed", "ended_at": now(), "exit_code": 0,
            "cleanup_settled": True, "logs": [], "failure": None})
        self.assertEqual(self.store.get(self.run_id)["state"]["budget"]["attempts"], 1)
        self.assertEqual(self.store.get(self.run_id)["state"]["cleanup"], "external")

    def test_late_publication_of_in_window_result_grants_no_execution_extension(self):
        candidate = self.candidate()
        package = self.package(candidate)
        checks = self.checks(candidate, package)
        data = self.benchmark_data(package, checks)
        deadline = timestamp_for_test(self.view["segment"]["deadline_at"])
        with patch("crit.run.store.utc_now", return_value=deadline + timedelta(seconds=1)):
            self.store.append(self.run_id, candidate["candidate_id"], "benchmark", data)
            with self.assertRaises(ValueError):
                self.store.reserve(self.run_id, candidate["candidate_id"], "benchmark")
        self.assertEqual(len(self.store.get(self.run_id)["candidates"][0]["series"]), 1)

    def test_exclusive_lease_serializes_ids_and_blocks_live_owner_recovery(self):
        with lease(self.directory):
            with self.assertRaises(ValueError):
                self.store.resume(self.run_id, 1, external_cleanup_confirmed=True)
            with self.assertRaises(ValueError):
                self.store.recover(self.run_id)
        self.assertFalse((self.directory / ".coordinator.lock").exists())

    def test_partial_projection_recovery_keeps_raw_records_and_missing_seal_incomplete(self):
        candidate = self.candidate()
        path = self.directory / "candidates" / candidate["candidate_id"] / "telemetry.json"
        raw = (path.parent / "candidate.json").read_bytes()
        path.write_text("{")
        self.assertTrue(self.store.get(self.run_id)["errors"])
        result = self.store.recover(self.run_id)
        self.assertFalse(result["candidates"][0]["sealed"])
        self.assertEqual((path.parent / "candidate.json").read_bytes(), raw)

    def test_one_invalid_candidate_does_not_hide_valid_run_or_candidates(self):
        first = self.candidate()
        self.on_ladder(first)
        second = self.candidate(parent_candidate_id=first["candidate_id"])
        (self.directory / "candidates" / first["candidate_id"] / "telemetry.json").write_text("{")
        result = self.store.get(self.run_id)
        self.assertEqual([c["candidate_id"] for c in result["candidates"]], [second["candidate_id"]])
        self.assertEqual(len(self.store.list()["runs"]), 1)

    def test_paths_reject_traversal_absolute_backslash_and_symlink_escape(self):
        for relative in ("../outside", "/absolute", "C:/outside", "x\\outside", "a/./b"):
            with self.assertRaises(ValueError):
                safe_path(self.directory, relative)
        try:
            (self.directory / "escape").symlink_to(self.root, target_is_directory=True)
        except OSError:
            return
        with self.assertRaises(ValueError):
            safe_path(self.directory, "escape/model.fixture")

    def test_atomic_exclusive_records_and_temporary_files_are_not_candidates(self):
        path = self.directory / "events/fixture.json"
        write_json(path, {"fixture": 1}, immutable=True)
        with self.assertRaises(FileExistsError):
            write_json(path, {"fixture": 2}, immutable=True)
        self.assertEqual(read_json(path), {"fixture": 1})
        (self.directory / "candidates/random.tmp").write_text("{")
        self.assertEqual(self.store.get(self.run_id)["candidates"], [])

    def test_windows_reader_contention_retries_atomic_projection_publication(self):
        from crit import records as run_records   # write_json's module
        path = self.directory / "events/live.json"
        write_json(path, {"version": 1})
        denied = PermissionError("Synthetic Windows reader holds the destination")
        denied.winerror = 5
        replace = run_records.os.replace
        calls = []
        def transient(source, destination):
            calls.append(destination)
            if len(calls) == 1:
                self.assertEqual(read_json(path), {"version": 1})
                raise denied
            return replace(source, destination)
        with patch.object(run_records.os, "replace", side_effect=transient):
            write_json(path, {"version": 2})
        self.assertEqual(len(calls), 2)
        self.assertEqual(read_json(path), {"version": 2})
        with patch.object(run_records.os, "replace", side_effect=denied), \
                patch.object(run_records.time, "monotonic", side_effect=[0, 1]):
            with self.assertRaises(PermissionError):
                write_json(path, {"version": 3})
        self.assertEqual(read_json(path), {"version": 2})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])


class RunRootTests(unittest.TestCase):
    def test_no_run_root_lies_in_a_folder_the_repository_publishes(self):
        allowlist = [line[2:-1] for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
                     if line.startswith("!/") and line.endswith("/")]
        self.assertEqual(sorted(allowlist), sorted(PUBLISHED))
        for name in allowlist:
            with self.subTest(folder=name), self.assertRaisesRegex(ValueError, "Runs belong outside publication"):
                RunStore(ROOT / name / "runs", ROOT)

    def test_a_crit_installed_without_a_checkout_keeps_runs_in_any_folder_outside_its_own(self):
        """An installed crit (crit.cli.installed) takes any folder as the project, names its release in the run's
        brief and keeps runs out of the folder it is installed in; a crit run from a checkout still wants one."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base, request = Path(temp.name).resolve(), {"correctness_criteria": "Fixture gates", "kv_tokens": 8192}
        (base / "model.fixture").write_text("synthetic input, not a model", encoding="utf-8")
        workspace = base / "workspace"
        crit = Installed(base / "site-packages" / "crit", __version__, f"{REPOSITORY}/tree/v{__version__}")
        with patch("crit.run.store.installed", return_value=crit):
            store = RunStore(workspace / "runs", workspace)
            run_id = store.create({**request, "model_path": str(base / "model.fixture")})["run"]["run_id"]
            brief = (store.directory(run_id) / "agent_request.md").read_text(encoding="utf-8")
            self.assertIn(f"\nRepository: {crit.guides}, the release of the installed crit {__version__} "
                          f"({crit.package})\n", brief)
            self.assertIn(f'--candidate CANDIDATE_ID --project "{workspace}"', brief)
            with self.assertRaisesRegex(ValueError, "Runs belong outside the folder crit is installed in"):
                RunStore(base / "site-packages" / "runs", workspace)
            # A checkout given as the project is the repository, as before.
            self.assertEqual(RunStore(base / "runs", ROOT).repository, str(ROOT))
        with self.assertRaisesRegex(ValueError, "Select a crit repository with --project"):
            RunStore(workspace / "runs", workspace)


if __name__ == "__main__":
    unittest.main()
