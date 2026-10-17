"""crit eval kernels: trace parsing, phase split, ranking and share of peak (no GPU)."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval import kernels  # noqa: E402


def write_trace(directory, pid, offset, rows, sms=170):
    lines = [f"#clock 1000 {1000 + offset}"]
    lines += [f"{s} {e} {blocks} 256 {name}" for s, e, blocks, name in rows]
    lines += [f"#clock 9000000000 {9000000000 + offset}", f"#device {sms} Test GPU"]
    (Path(directory) / f"trace-{pid}.txt").write_text("\n".join(lines) + "\n")


class KernelTraceTest(unittest.TestCase):
    def test_phase_split_ranking_and_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            # CUPTI clock runs 5,000 ns behind realtime; marks are realtime.
            write_trace(directory, 1, 5_000, [
                (10_000, 1_010_000, 340, "gemm_kernel"),        # prefill, 1 ms
                (1_100_000, 1_600_000, 96, "scan_kernel"),      # prefill, 0.5 ms, underfills 170 SMs
                (2_100_000, 2_200_000, 400, "gemv_kernel"),     # decode
                (2_300_000, 2_400_000, 400, "gemv_kernel"),     # decode
            ])
            marks = {"info": {"name": "toy", "kernel_work": [
                        {"match": "gemm", "flops_per_token": 1e9, "format": "fp8"},
                        {"match": "gemv", "bytes_per_call": 1e8}]},
                     "cases": [{"id": "c", "prompt_tokens": 1000, "output_tokens": 2, "rounds": 2,
                                "start": 0, "first": 2_000_000, "end": 3_000_000}]}
            found, device = kernels.read_traces(directory)
            self.assertEqual(device, {"sms": 170, "name": "Test GPU"})
            self.assertEqual(found[0][0], 15_000)  # converted to the realtime clock
            result = kernels.analyze(found, device, marks, {"tflops": {"fp8": 2000.0}, "bandwidth_gbs": 4000.0})
        prefill, decode = result["phases"]["prefill"], result["phases"]["decode"]
        self.assertEqual([r["kernel"] for r in prefill["kernels"]], ["gemm_kernel", "scan_kernel"])
        self.assertAlmostEqual(prefill["kernel_ms"], 1.5)
        self.assertAlmostEqual(prefill["wall_ms"], 2.0)
        gemm, scan = prefill["kernels"]
        self.assertAlmostEqual(gemm["tflops"], 1000.0)  # 1e12 FLOPs in 1 ms
        self.assertAlmostEqual(gemm["flops_of_peak"], 0.5)
        self.assertFalse(gemm["underfills"])
        self.assertTrue(scan["underfills"])
        (gemv,) = decode["kernels"]
        self.assertEqual(gemv["calls"], 2)
        self.assertAlmostEqual(gemv["gbs"], 1000.0)  # 2e8 bytes in 0.2 ms
        self.assertAlmostEqual(gemv["bandwidth_of_peak"], 0.25)
        self.assertEqual(result["cases"]["c"]["prefill"]["top"][0]["kernel"], "gemm_kernel")
        text = kernels.render({"engine": {"name": "toy"}, **result})
        self.assertIn("only 96 blocks for 170 SMs", text)
        self.assertIn("50% of peak", text)

    def test_overlapping_launch_is_charged_only_after_its_predecessor(self):
        with tempfile.TemporaryDirectory() as directory:
            # norm_kernel launches early (programmatic dependent launch) and waits 0.9 ms on gemm_kernel.
            write_trace(directory, 1, 0, [
                (0, 1_000_000, 340, "gemm_kernel"),
                (100_000, 1_100_000, 340, "norm_kernel"),
                (1_200_000, 1_300_000, 340, "gemm_kernel"),
            ])
            marks = {"info": {"name": "toy"}, "cases": [{"id": "c", "prompt_tokens": 10, "output_tokens": 1,
                                                          "start": 0, "first": 2_000_000, "end": 2_000_001}]}
            found, device = kernels.read_traces(directory)
            result = kernels.analyze(found, device, marks)
        prefill = result["phases"]["prefill"]
        rows = {r["kernel"]: r for r in prefill["kernels"]}
        self.assertAlmostEqual(rows["norm_kernel"]["ms"], 0.1)
        self.assertAlmostEqual(rows["norm_kernel"]["span_ms"], 1.0)
        self.assertAlmostEqual(rows["gemm_kernel"]["ms"], 1.1)
        self.assertAlmostEqual(prefill["kernel_ms"], 1.2)  # never more than the GPU was busy
        self.assertAlmostEqual(prefill["idle_ms"], 0.8)
        self.assertIn("not charged", kernels.render({"engine": {"name": "toy"}, **result}))

    def test_short_name_drops_parameters_not_templates(self):
        self.assertEqual(kernels.short_name("void ns::k<a(b), 2>(float const*, int)"), "ns::k<a(b), 2>")
        self.assertTrue(kernels.short_name("x" * 200).endswith("..."))

    def test_internal_linkage_prefix(self):
        self.assertEqual(kernels._mangled("__nv_static_54__110d7f6d_27_f_cu_1f652c57_6886__ZN2ns1kEv"), "_ZN2ns1kEv")
        self.assertEqual(kernels._mangled("_ZN2ns1kEv"), "_ZN2ns1kEv")


if __name__ == "__main__":
    unittest.main()
