"""crit run ... profile: GPU time and launches by kernel family per prompt and decode token of one short served
request, and the decode step against recon's plain-pass floor. The profiler is faked (no GPU): a tracer whose records
are written in crit-eval's trace format over the time the toy engine served, read by crit-eval's own reader."""
import json
import os
from pathlib import Path
import time
import unittest

from test_run_measure import ADAPTER, MeasureFixture
from crit.run import profile
from crit.run.records import read_json

# The toy engine also writes the tracer's variable its host process got, next to the model.
TRACED = ADAPTER.replace(
    "        Path(request['model_path']).with_name('host.pid').write_text(str(os.getpid()))\n",
    "        Path(request['model_path']).with_name('host.pid').write_text(str(os.getpid()))\n"
    "        Path(request['model_path']).with_name('ktrace.txt').write_text(os.environ.get('CRIT_KTRACE_DIR', ''))\n")
# Kernels the fake tracer records in turn: one per family, a name a rule classes and one nothing classes.
NAMES = ("gemv_bf16_kernel", "_flash_kernel", "fused_add_rmsnorm_kernel", "moe_topk_router", "custom_mixer_kernel",
         "mystery_kernel")


class FakeTracer(profile.Tracer):
    """A tracer that records a kernel every 50 us, 30 us long, and a 20 us host-to-device copy every 500 us, from when
    its library is built until its records are read, on the host's realtime clock."""

    def __init__(self):
        self.backend, self.toolkit, self.name = "cuda", None, "fake tracer"

    def library(self, directory):
        self.began = time.time_ns()
        return Path(directory) / "libfake.so"

    def clock(self, library):
        return time.time_ns, "realtime"

    def read(self, directory, clock):
        ended = time.time_ns()
        lines = ["#clock 0 0", "#device 2 Fake GPU"]
        for i, start in enumerate(range(self.began, ended, 50_000)):
            lines.append(f"{start} {start + 30_000} 4 128 {NAMES[i % len(NAMES)]}")
            if i % 10 == 5:
                lines.append(f"#copy {start + 31_000} {start + 51_000} 4096 1")   # alone for 19 us of its 20
        (Path(directory) / "trace-1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return super().read(directory, clock)


class ProfileTableTests(unittest.TestCase):
    def test_names_go_to_their_family_rules_first_and_the_rest_to_other(self):
        for name, family in (("gemv_kernel", "matmul"), ("void cutlass::Kernel<gemm>", "matmul"),
                             ("_flash_kernel", "attention"), ("paged_attention_v2", "attention"),
                             ("attn_rmsnorm", "norms"), ("fused_add_rms_norm", "norms"),
                             ("moe_topk_softmax", "routing"), ("router_gemv", "routing"),
                             ("memcpy_dtoh", "copies"), ("transpose_kernel", "copies"),
                             ("_window_kernel", "attention"),
                             ("crit::techniques::grouped_experts::(anonymous namespace)::expert_kernel<3, 1>", "matmul"),
                             ("rows_kernel", "matmul"), ("crit::recurrent_kernels::pool_delta_pipe", "recurrence"),
                             ("act_mul_kernel", "elementwise"), ("stream_add_kernel", "norms"),
                             ("mystery_kernel", "other")):
            self.assertEqual(profile.classify(name)[0], family, name)
        rules = profile.family_rules('[{"match": "Window", "family": "attention"}, {"match": "gemv", "family": "experts"}]')
        self.assertEqual(profile.classify("_window_kernel", rules), ("attention", "rules"))
        self.assertEqual(profile.classify("gemv_kernel", rules), ("experts", "rules"))
        for bad in ('{"match": "x"}', '[{"match": "", "family": "x"}]', '[{"match": "x", "family": "y", "z": 1}]'):
            with self.assertRaisesRegex(ValueError, "--families"):
                profile.family_rules(bad)

    def test_a_traced_request_per_token_by_family_against_the_plain_pass_floor(self):
        ms = 1_000_000
        kernels = [(0, ms, 8, 128, "gemm_kernel"), (ms, 2 * ms, 8, 128, "_flash_kernel"),      # prompt: 2 ms
                   (10 * ms, 12 * ms, 8, 128, "gemv_kernel"), (11 * ms, 13 * ms, 8, 128, "gemv_kernel"),
                   (14 * ms, 15 * ms, 8, 128, "rms_norm"), (16 * ms, 17 * ms, 8, 128, "my_scan")]
        copies = [(18 * ms, 19 * ms, 1 << 20, 1), (14 * ms, 15 * ms, 1 << 10, 2)]   # alone 1 ms; beside a kernel
        mark = {"case_id": "case-0001", "prompt_tokens": 100, "prefill_tokens": 80, "output_tokens": 5, "rounds": 2,
                "start": 0, "first": 5 * ms, "end": 25 * ms}
        floor = {"path": "device", "floor_ms": 5.0, "ceiling_tps": 200.0}
        table = profile.analyze(kernels, {"sms": 2, "name": "Fake GPU"}, copies, mark, floor=floor)
        prompt, decode = table["phases"]["prompt"], table["phases"]["decode"]
        self.assertAlmostEqual(prompt["gpu_ms"], 2.0)
        self.assertEqual(prompt["tokens"], 80)   # the prompt tokens computed, not those a cache served
        self.assertAlmostEqual(prompt["families"]["matmul"]["us_per_token"], 1000 / 80)
        self.assertAlmostEqual(prompt["families"]["attention"]["share"], 0.5)
        # Decode: the second GEMV overlaps the first and is charged from its end; the copy beside the norm costs
        # nothing, the one alone 1 ms.
        self.assertEqual(decode["tokens"], 4)
        self.assertAlmostEqual(decode["families"]["matmul"]["ms"], 3.0)
        self.assertEqual(decode["families"]["matmul"]["launches"], 2)
        self.assertAlmostEqual(decode["families"]["copies"]["ms"], 1.0)
        self.assertEqual(decode["families"]["copies"]["launches"], 2)
        self.assertAlmostEqual(decode["gpu_ms"], 6.0)
        self.assertAlmostEqual(decode["gpu_us_per_token"], 1500.0)
        self.assertEqual(decode["launches"], 6)
        self.assertEqual(decode["other"], [{"kernel": "my_scan", "ms": 1.0, "launches": 1}])
        self.assertAlmostEqual(decode["step_ms"], 10.0)
        self.assertAlmostEqual(decode["against_floor"]["step_over_floor"], 2.0)
        self.assertAlmostEqual(decode["against_floor"]["gpu_step_over_floor"], 0.6)
        self.assertAlmostEqual(decode["against_floor"]["tps"], 200.0)
        text = profile.render({"available": True, "candidate_id": "candidate-0001", "profiler": "CUPTI", **table})
        self.assertIn("2.00x it", text)
        self.assertIn("decode other by name: my_scan", text)

    def test_the_floor_is_the_placements_path_past_the_device(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIn("no recon.json", profile.ceiling(root)["reason"])
            (root / "work").mkdir()
            facts = {"gathered": {"placement": {"plain_pass_floor_ms": {"device": 5.0, "link": 40.0, "cpu": 25.0}}}}
            (root / "work/recon.json").write_text(json.dumps(facts), encoding="utf-8")
            self.assertEqual(profile.ceiling(root), {"path": "cpu", "floor_ms": 25.0, "ceiling_tps": 40.0})

    def test_a_device_without_a_profiler_says_why(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            device = Path(directory) / "device.json"
            device.write_text(json.dumps({"backend": "cpu"}), encoding="utf-8")
            run = {"spec": {"hardware_constraints": {"context": {"device_file": str(device)}}}}
            reason, tracer = profile.profiler(run)
        self.assertIsNone(tracer)
        self.assertIn("runs cpu", reason)


class ProfileServedTests(MeasureFixture, unittest.TestCase):
    def test_the_toy_engine_served_with_the_tracer_prints_the_table(self):
        self.resolve()
        candidate = self.served(adapter=TRACED)
        (self.directory / "work").mkdir(exist_ok=True)
        (self.directory / "work/recon.json").write_text(json.dumps({"gathered": {"placement": {
            "scenario": "fits_device", "plain_pass_floor_ms": {"device": 1.0}}}}), encoding="utf-8")
        rules = profile.family_rules('[{"match": "custom_mixer", "family": "attention"}]')
        found = profile.profile(self.store, self.run_id, candidate, tokens=16, families=rules, tracer=FakeTracer())
        self.assertTrue(found["available"], found.get("reason"))
        # The tracer's variables reached the engine's host process, and left this one's environment.
        self.assertTrue(self.model.with_name("ktrace.txt").read_text(encoding="utf-8"))
        self.assertNotIn("CRIT_KTRACE_DIR", os.environ)
        decode = found["phases"]["decode"]
        self.assertEqual(decode["tokens"], 15)
        self.assertEqual(decode["steps"], 5)   # the toy engine's rounds of three tokens
        for family in ("matmul", "attention", "norms", "routing", "copies", "other"):
            self.assertGreater(decode["families"][family]["launches"], 0, family)
        self.assertEqual([row["kernel"] for row in decode["other"]], ["mystery_kernel"])
        self.assertGreater(decode["classed_by_rules_ms"], 0)
        self.assertEqual(decode["against_floor"]["path"], "device")
        self.assertEqual(found["prefill_tokens"], 512)
        report = read_json(self.directory / found["report"])
        self.assertEqual(report["phases"]["decode"]["steps"], 5)
        self.assertEqual(self.attempt()["outcome"], "completed")
        text = profile.render(found)
        rows = [line.split()[0] for line in text.splitlines()[2:10]]
        self.assertEqual(rows, ["matmul", "attention", "recurrence", "norms", "elementwise", "routing", "copies",
                                "other"])
        self.assertEqual(rows.count("routing"), 1)
        self.assertEqual(sum(line.startswith("routing ") for line in text.splitlines()), 1)
        for part in ("profile of " + candidate, "Fake GPU", "decode us/token", "routing", "recon's plain pass (device)",
                     "decode other by name: mystery_kernel"):
            self.assertIn(part, text)

    def test_without_a_profiler_nothing_is_served_and_the_report_says_why(self):
        # A host without one: Windows by its platform, Linux without a CUDA toolkit with CUPTI, which this host may
        # have, so the test takes it away (as test_run_external does its own).
        from unittest.mock import patch
        from crit_eval.limits import kernels

        def no_toolkit(nvcc=None):
            raise SystemExit("no CUDA toolkit with CUPTI found (looked for include/cupti.h and libcupti.so); pass --nvcc")
        self.resolve()
        candidate = self.served()
        with patch.object(kernels, "find_toolkit", no_toolkit):
            found = profile.profile(self.store, self.run_id, candidate)
        self.assertFalse(found["available"])
        self.assertIn("Linux" if os.name == "nt" else "no CUDA toolkit with CUPTI", found["reason"])
        self.assertFalse(self.model.with_name("host.pid").exists(), "nothing was served")
        self.assertEqual(read_json(self.directory / found["report"])["available"], False)
        self.assertIn("no kernel profile", profile.render(found))


if __name__ == "__main__":
    unittest.main()
