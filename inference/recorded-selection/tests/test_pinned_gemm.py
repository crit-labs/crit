import importlib.util
import json
import os
import sys
import tempfile
import unittest

CODE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
sys.path.insert(0, CODE)
_spec = importlib.util.spec_from_file_location("pinned_gemm", os.path.join(CODE, "pinned_gemm.py"))
pinned = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pinned)

try:
    import torch
    TORCH, CUDA = True, torch.cuda.is_available()
except ImportError:
    TORCH = CUDA = False

ENTRY = {"n": 5120, "k": 17408, "dtype": "fp8", "rep_rows": 2048, "index": 1, "algo": [67, 18, 16, 1, 0, 1, 18]}


class TableTests(unittest.TestCase):
    def test_key_format_uses_the_selection_key_without_a_width_bucket(self):
        self.assertEqual(pinned.gemm_key(5120, 17408, "fp8"), "pinned-gemm-fp8:5120x17408:0")
        self.assertEqual(pinned.gemm_key(17408, 5120, "bf16"), "pinned-gemm-bf16:17408x5120:0")
        self.assertEqual(pinned.gemm_key(5120, 5120, "fp8e4m3-rowscale:bf16"),
                         "pinned-gemm-fp8e4m3-rowscale-to-bf16:5120x5120:0")
        for bad in ((0, 5120, "bf16"), (5120, 5120, "fp4"), (5120.0, 5120, "fp8"), (5120, 5120, "bf16:fp8"),
                    (5120, 5120, "bf16-tensorscale:bf16"), (5120, 5120, "bf16xfp16:bf16"),
                    (5120, 5120, "nvfp4-rowscale:bf16"), (5120, 5120, "mxfp8xnvfp4:bf16"), (5120, 5120, None)):
            with self.assertRaises(ValueError):
                pinned.gemm_key(*bad)

    def test_every_format_parses_into_distinct_keys(self):
        keys = {pinned.gemm_key(64, 64, fmt) for fmt in pinned.FORMATS + tuple(pinned.DTYPES)}
        self.assertEqual(len(keys), len(pinned.FORMATS) + len(pinned.DTYPES))
        for fmt in pinned.FORMATS:
            with self.subTest(fmt=fmt):
                f = pinned.parse_format(fmt)
                self.assertEqual(f["name"], fmt)
                self.assertIn(f["out"], pinned.OUTPUTS)
        self.assertEqual(pinned.parse_format("fp8"), pinned.parse_format("fp8e4m3-tensorscale:bf16"))
        self.assertEqual(pinned.parse_format("bf16"), pinned.parse_format("bf16:bf16"))
        mixed = pinned.parse_format("fp8e4m3xfp8e5m2-rowscale:fp32")
        self.assertEqual((mixed["a"], mixed["b"], mixed["scale"], mixed["out"]), ("e4m3", "e5m2", "row", "fp32"))
        self.assertEqual([pinned.parse_format(f)[p] for f, p in (("mxfp8:bf16", "scale"), ("mxfp4:fp16", "block"),
                                                                  ("nvfp4:fp32", "block"), ("nvfp4:bf16", "a"))],
                         ["vec32", 32, 16, "e2m1"])

    def test_block_scale_layout_pads_rows_to_128_and_columns_to_4(self):
        self.assertEqual(pinned.scale_layout(1, 5120, 16), (128, 320))
        self.assertEqual(pinned.scale_layout(129, 5120, 32), (256, 160))
        self.assertEqual(pinned.scale_layout(128, 96, 32), (128, 4))
        for bad in ((0, 5120, 16), (1, 5121, 16), (1, 5120, 0)):
            with self.assertRaises(ValueError):
                pinned.scale_layout(*bad)
        for rows, cols in ((1, 1), (3, 5), (128, 4), (129, 8), (300, 10)):
            with self.subTest(rows=rows, cols=cols):
                order = pinned.blocked_order(rows, cols)
                self.assertEqual(len(order), (-(-rows // 128) * 128) * (-(-cols // 4) * 4))
                self.assertEqual(sorted(i for i in order if i >= 0), list(range(rows * cols)))
        # Within a 128 x 4 tile, row r and column c sit at (r % 32) * 16 + (r // 32) * 4 + c.
        order = pinned.blocked_order(128, 4)
        self.assertEqual([order[i] for i in (0, 1, 4, 16, 511)], [0, 1, 32 * 4, 1 * 4, 127 * 4 + 3])
        # A prefix's own layout keeps each row's scales at the positions they take in the longer prompt's.
        long, short = pinned.blocked_order(200, 8), pinned.blocked_order(70, 8)
        self.assertEqual([i for i in short if i >= 0], [i for i in long if 0 <= i < 70 * 8])

    @unittest.skipUnless(TORCH, "needs torch")
    def test_torch_block_scales_follow_the_blocked_order(self):
        for rows, cols in ((1, 3), (17, 8), (130, 9)):
            with self.subTest(rows=rows, cols=cols):
                scales = torch.randint(1, 255, (rows, cols), dtype=torch.uint8)
                expected = [0 if i < 0 else int(v) for i, v in zip(
                    pinned.blocked_order(rows, cols), (scales.flatten()[max(i, 0)] for i in
                                                       pinned.blocked_order(rows, cols)))]
                self.assertEqual(pinned.blocked_scales(scales).tolist(), expected)

    def test_a_table_round_trips_through_its_file(self):
        table = pinned.GemmTable(130101, "Some GPU", entries={"x": ENTRY})
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "gemms.json")
            table.save(path)
            again = pinned.GemmTable.load(path)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["entries"], {"pinned-gemm-fp8:5120x17408:0": ENTRY})
        self.assertEqual(again.to_json(), table.to_json())
        entry = again.entry(5120, 17408, "fp8", cublaslt=130101, device="Some GPU")
        self.assertEqual(entry, dict(ENTRY, cublaslt=130101, device="Some GPU", workspace_bytes=pinned.WORKSPACE))
        with self.assertRaises(KeyError):
            again.entry(5120, 5120, "fp8", cublaslt=130101, device="Some GPU")

    def test_short_bucket_choices_round_trip_and_entries_without_them_still_load(self):
        buckets = {"8": {"rep_rows": 8, "index": 0, "algo": [67, 18, 16, 1, 0, 1, 18], "ms": 0.03},
                   "64": {"rep_rows": 64, "index": 2, "algo": [67, 15, 18, 1, 0, 1, 18], "ms": 0.04}}
        table = pinned.GemmTable(130101, "Some GPU", entries={"x": dict(ENTRY, buckets=buckets), "y": dict(
            ENTRY, n=17408)})
        again = pinned.GemmTable.from_json(json.loads(json.dumps(table.to_json())))
        self.assertEqual(again.to_json(), table.to_json())
        self.assertEqual(again.entry(5120, 17408, "fp8", 130101, "Some GPU")["buckets"], buckets)
        self.assertNotIn("buckets", again.entry(17408, 17408, "fp8", 130101, "Some GPU"))
        for bad in ({"6": dict(buckets["8"], rep_rows=6)}, {"8": dict(buckets["8"], rep_rows=16)},
                    {"8": dict(buckets["8"], index=-1)}, {"8": None}):
            with self.assertRaises(ValueError):
                pinned.GemmTable(130101, "Some GPU", entries={"x": dict(ENTRY, buckets=bad)})

    def test_replay_on_another_device_or_cublaslt_version_is_refused(self):
        table = pinned.GemmTable(130101, "Some GPU", entries={"x": ENTRY})
        for cublaslt, device in ((130200, "Some GPU"), (130101, "Other GPU")):
            with self.assertRaisesRegex(ValueError, "record it again"):
                table.entry(5120, 17408, "fp8", cublaslt=cublaslt, device=device)

    def test_malformed_tables_are_refused(self):
        good = pinned.GemmTable(130101, "Some GPU", entries={"x": ENTRY}).to_json()
        for bad in (dict(good, kind="other"), dict(good, cublaslt=0), dict(good, device=""),
                    dict(good, entries={"wrong-key": ENTRY}), dict(good, entries={"x": dict(ENTRY, index=-1)}),
                    dict(good, entries={"x": dict(ENTRY, algo=None)})):
            with self.assertRaises(ValueError):
                pinned.GemmTable.from_json(bad)


@unittest.skipUnless(CUDA, "needs CUDA")
class DeviceTests(unittest.TestCase):
    SHAPES = ((5120, 5120), (17408, 5120), (5120, 17408))
    LENGTHS = (1, 2, 3, 5, 7, 9, 17, 33, 65, 128, 513)
    TOLERANCE = {"bf16": 1e-2, "fp16": 2e-3, "fp32": 1e-4}   # norm-relative error against the FP32 product

    def check(self, fmt, n, k):
        cublaslt, device = pinned.environment()
        table = pinned.GemmTable(cublaslt, device)
        try:
            table.add(pinned.record(n, k, fmt, rows=(128, 512), short=16))
        except pinned.UnsupportedFormat as exc:
            self.skipTest(str(exc))
        entry = table.entry(n, k, fmt)
        a, b, scales = pinned.operands(n, k, fmt, 1024, seed=1)
        whole = pinned.pinned_mm(a, b, entry, **pinned.prompt_scales(fmt, scales, 1024))
        for m in self.LENGTHS:
            part = pinned.pinned_mm(a[:m], b, entry, **pinned.prompt_scales(fmt, scales, m))
            self.assertTrue(torch.equal(part, whole[:m]), m)
        expected = pinned.torch_mm(fmt, a, b, scales)
        if expected is None or expected.dtype != whole.dtype:
            expected = pinned.reference(fmt, a, b, scales)
        error = ((whole.float() - expected.float()).norm() / expected.float().norm()).item()
        self.assertLess(error, self.TOLERANCE[pinned.parse_format(fmt)["out"]])

    def test_a_recorded_entry_replays_directly_and_through_a_table(self):
        entry = pinned.record(5120, 5120, "bf16:bf16", rows=(128,), short=0)
        a, b, scales = pinned.operands(5120, 5120, "bf16:bf16", 64, seed=2)
        direct = pinned.pinned_mm(a, b, entry)
        table = pinned.GemmTable(*pinned.environment())
        table.add(entry)
        self.assertTrue(torch.equal(direct, pinned.pinned_mm(a, b, table.entry(5120, 5120, "bf16:bf16"))))
        self.assertEqual(pinned.GemmTable.from_json(table.to_json()).entry(5120, 5120, "bf16:bf16"),
                         table.entry(5120, 5120, "bf16:bf16"))

    def test_every_format_this_device_takes_is_length_exact_and_matches_a_reference(self):
        for fmt in pinned.FORMATS:
            with self.subTest(fmt=fmt):
                self.check(fmt, 5120, 5120)

    def test_the_old_formats_are_length_exact_on_every_shape(self):
        for dtype in pinned.DTYPES:
            for n, k in self.SHAPES:
                with self.subTest(dtype=dtype, n=n, k=k):
                    self.check(dtype, n, k)


if __name__ == "__main__":
    unittest.main()
