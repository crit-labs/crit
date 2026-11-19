"""model-template on the host: its layouts are well formed and cover the model types its manifest needs (each with the
slots it requires there), its manifest declares the options compose() reads with the same defaults, prompt spans follow
the timed cost model within the budget and the room, and row groups multiply each prompt chunk alone and one-token rows
in calls of at most invariant_rows."""
import importlib
import json
import sys
import unittest
from pathlib import Path

from crit_plugins.testing import PluginTestCase

HERE = Path(__file__).resolve().parents[1]


def template(name):
    """A module of the template's code, imported as the plugin imports it (its code/ folder a package)."""
    from crit_plugins.assemble import load_module
    from crit_plugins.manifest import load_manifest
    module = load_module(load_manifest(HERE))
    return importlib.import_module(f"{module.__package__}.template.{name}")


class LayoutTests(PluginTestCase):
    def test_every_library_layout_checks_and_each_model_type_has_one(self):
        compose = template("compose")
        found = compose.builtin()
        manifest = json.loads((HERE / "plugin.json").read_text(encoding="utf-8"))
        needed = {fact.split(":", 1)[1] for fact in manifest["needs"][0]}
        self.assertEqual(set(found), needed)
        for path in sorted((HERE / "code" / "layouts").glob("*.json")):
            self.assertEqual(compose.check(json.loads(path.read_text(encoding="utf-8")), path.name), [])

    def test_the_manifest_requires_each_layouts_slots_where_its_model_types_hold(self):
        compose = template("compose")
        manifest = json.loads((HERE / "plugin.json").read_text(encoding="utf-8"))
        when = manifest["requires_when"]
        for model_type, layout in compose.builtin().items():
            if "linear_attention" in layout["mixers"]:
                self.assertIn("recurrence", when.get(f"model_type:{model_type}", []), model_type)
            if "moe" in model_type or model_type in ("qwen3_next", "qwen4_exp", "qwen4_exp_text", "deepseek_v3"):
                self.assertTrue({"router", "experts"} <= set(when.get(f"model_type:{model_type}", [])), model_type)

    def test_the_manifest_declares_composes_options_with_its_defaults(self):
        compose = template("compose")
        manifest = json.loads((HERE / "plugin.json").read_text(encoding="utf-8"))
        self.assertEqual({k: v["default"] for k, v in manifest["options"].items()}, compose.DEFAULTS)

    def test_a_layout_naming_an_unknown_block_or_key_is_refused(self):
        compose = template("compose")
        layout = {"schema": compose.SCHEMA, "model_types": ["x"], "mixers": {"full_attention": {"block": "lstm"}},
                  "colour": 1}
        problems = compose.check(layout)
        self.assertTrue(any("unknown keys colour" in p for p in problems), problems)
        self.assertTrue(any("mixers.full_attention names a block" in p for p in problems), problems)
        known = {"schema": compose.SCHEMA, "model_types": ["x"], "mixers": {"full_attention": {"block": "lstm"}}}
        self.assertEqual(compose.check(known, blocks={"lstm": object}), [])

    def test_a_checkpoint_without_a_layout_names_the_option(self):
        compose = template("compose")
        with self.assertRaisesRegex(ValueError, "no layout for model_type 'mamba'.*option layout"):
            compose.layout_for({"model_type": "mamba"})
        self.assertEqual(compose.layout_for({"model_type": "x", "text_config": {"model_type": "qwen3_5_text"}})["name"],
                         "gated-hybrid")


class SpanTests(PluginTestCase):
    def test_spans_are_the_longest_within_the_budget_at_each_position(self):
        rows = template("rows")
        costs = rows.fit({(128, 0): 0.02, (512, 0): 0.06, (512, 200_000): 0.9}, {128: 1e8, 512: 4e8})
        spans = rows.Spans(512, costs, 0.5)
        for position in (0, 50_000, 120_000, 200_000):
            found = spans.rows(position)
            self.assertEqual(found % rows.QUANTUM, 0)
            self.assertLessEqual(spans.predicted(found, position), 0.5)
            if found < 512:
                self.assertGreater(spans.predicted(found + rows.QUANTUM, position), 0.5)
        self.assertEqual(spans.rows(0), 512)
        self.assertLess(spans.rows(200_000), 512)
        self.assertEqual(spans.stop(1000), 1000 + spans.rows(1000))

    def test_a_span_leaves_the_rooms_bytes_and_one_without_a_calibration_is_the_chunk(self):
        rows = template("rows")
        costs = rows.fit({(16, 0): 0.001, (64, 0): 0.002, (64, 1000): 0.003}, {16: 16 << 20, 64: 64 << 20})
        self.assertEqual(costs["m"], float(f"{1 << 20:.4g}"))
        self.assertEqual(rows.Spans(64, costs, 10.0, room=int(costs["m"] * 40)).rows(0), 32)
        self.assertEqual(rows.Spans(64).rows(10 ** 6), 64)

    def test_where_no_span_meets_the_budget_spans_are_the_ones_that_prefill_fastest(self):
        # A step whose fixed part alone passes the budget (an experts plugin copying a chunk's experts every step):
        # shorter spans pay it again for fewer tokens, so the span is the whole chunk, or the longest that still ends
        # inside the cancel grace, and the facts name the fixed part.
        rows = template("rows")
        fixed = {"c": 0.6, "a": 2e-4, "b": 0.0, "m": 0.0}
        spans = rows.Spans(1024, fixed, 0.5, deepest=200_000)
        self.assertEqual([spans.rows(p) for p in (0, 100_000, 200_000)], [1024, 1024, 1024])
        self.assertIn("0.60 s of it whatever its span", spans.facts()["over_budget"])
        self.assertEqual(spans.facts()["steps"][-1], {"position": 200_000, "rows": 1024, "seconds": 0.8048,
                                                      "tokens_per_second": 1272.0})
        # The default budget leaves the grace a quarter: a step that copies its experts (0.45 s) and takes 0.48 ms a
        # token runs 624 tokens, 830 a second, where half a second ran 96 and 190.
        staged = {"c": 0.45, "a": 4.8e-4, "b": 0.0, "m": 0.0}
        self.assertEqual(rows.SECONDS, 0.75 * rows.GRACE)
        self.assertEqual((rows.Spans(1024, staged, rows.SECONDS).rows(0), rows.Spans(1024, staged, 0.5).rows(0)),
                         (624, 96))
        self.assertEqual(template("compose").DEFAULTS["span_seconds"], rows.SECONDS)
        slow = rows.Spans(1024, {**fixed, "a": 5e-4}, 0.5)     # the chunk would take 1.11 s: past the grace
        self.assertEqual(slow.rows(0), 784)                     # the longest that ends inside it (0.992 s)
        self.assertLess(slow.predicted(784, 0), rows.GRACE)
        hopeless = rows.Spans(1024, {**fixed, "c": 1.5}, 0.5)   # nothing ends inside the grace: the fastest
        self.assertEqual(hopeless.rows(0), 1024)
        # A step whose time grows with its rows keeps the rule: the longest span within the budget.
        grows = rows.Spans(1024, {"c": 0.05, "a": 1e-3, "b": 0.0, "m": 0.0}, 0.5)
        self.assertEqual(grows.rows(0), 448)
        self.assertNotIn("over_budget", grows.facts())
        self.assertEqual(rows.Spans(64, {**fixed, "m": 1.0}, 0.0, room=0).rows(0), rows.QUANTUM)   # no room at all

    def test_the_fit_reads_back_its_own_rounding(self):
        rows = template("rows")
        costs = rows.fit({(16, 0): 0.0011111, (64, 0): 0.0023456, (64, 999): 0.0034567}, {64: 12345678})
        self.assertEqual(rows.fit({(16, 0): 0.0011111, (64, 0): 0.0023456, (64, 999): 0.0034567}, {64: 12345678}),
                         costs)
        self.assertEqual(json.loads(json.dumps(costs)), costs)


class CheckpointTests(PluginTestCase):
    def test_the_index_names_the_shards_wherever_the_checkpoint_keeps_them(self):
        # A checkpoint may keep its shards in a subfolder, which only its index names; a file beside the index that
        # the index does not list (a draft layer stored apart) joins it, and a tensor two files hold is refused.
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch and safetensors")
        import tempfile
        checkpoint = template("checkpoint")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tensors").mkdir()
            shards = {"tensors/model-00001-of-00002.safetensors": {"a.weight": torch.ones(2, 3)},
                      "tensors/model-00002-of-00002.safetensors": {"b.weight": torch.full((4,), 2.0)}}
            for name, tensors in shards.items():
                save_file(tensors, str(root / name))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 40},
                "weight_map": {tensor: name for name, tensors in shards.items() for tensor in tensors}}),
                encoding="utf-8")
            save_file({"mtp.weight": torch.zeros(5)}, str(root / "mtp.safetensors"))
            ck = checkpoint.Checkpoint(root)
            self.assertEqual(sorted(ck.names()), ["a.weight", "b.weight", "mtp.weight"])
            self.assertEqual((ck.shape("a.weight"), float(ck.get("b.weight", "cpu").sum()), ck.nbytes("mtp.weight")),
                             ((2, 3), 8.0, 20))
            ck.close()
            save_file({"a.weight": torch.zeros(1)}, str(root / "again.safetensors"))
            with self.assertRaisesRegex(ValueError, "a.weight is in model-00001-of-00002.safetensors and in again"):
                checkpoint.Checkpoint(root)

    def test_held_tensors_are_the_mappings_tensors_read_past_the_page_cache(self):
        # A layer's experts are held before they are read one by one (hold): direct reads at a queue depth into
        # memory of the checkpoint's own. Every held tensor is the one the shard's mapping gives, in dtype, shape
        # and bits, across shards, a first tensor of a file and a last one that ends inside a block.
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch and safetensors")
        import tempfile
        checkpoint = template("checkpoint")
        g = torch.Generator().manual_seed(5)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = {"layers.0.experts.0.weight": torch.randint(0, 255, (33, 77), generator=g, dtype=torch.uint8),
                     "layers.0.experts.0.scale": torch.randn(33, 5, generator=g).to(torch.bfloat16),
                     "layers.0.experts.1.weight": torch.randn(4097, generator=g).to(torch.float8_e4m3fn),
                     "layers.0.experts.1.global": torch.tensor(0.25),
                     "layers.0.other.weight": torch.randn(9000, generator=g)}
            second = {"layers.0.experts.2.words": torch.randint(-9, 9, (3, 1), generator=g, dtype=torch.int32),
                      "layers.1.experts.0.weight": torch.randn(7, 3, generator=g, dtype=torch.float64)}
            save_file(first, str(root / "model-00001-of-00002.safetensors"))
            save_file(second, str(root / "model-00002-of-00002.safetensors"))
            ck = checkpoint.Checkpoint(root)
            mapped = {name: ck.get(name, "cpu") for name in ck.names()}
            held = [name for name in ck.names() if name.startswith("layers.0.experts.")]
            for _ in range(2):   # a second hold reuses the memory of the first
                self.assertEqual(ck.hold("layers.0.experts."), sum(ck.nbytes(name) for name in held))
                self.assertEqual(sorted(ck._held), sorted(held))
                for name, want in mapped.items():
                    found = ck.get(name, "cpu")   # held or not, the same tensor
                    self.assertEqual((found.dtype, tuple(found.shape)), (want.dtype, tuple(want.shape)), name)
                    self.assertEqual(found.contiguous().view(-1).view(torch.uint8).tolist()
                                     if found.dim() else found.item(),
                                     want.contiguous().view(-1).view(torch.uint8).tolist() if want.dim() else want.item(),
                                     name)
                ck.release()
                self.assertEqual(ck._held, {})
            self.assertEqual(ck.hold("layers.9."), 0)   # nothing of that name: the mappings serve
            ck.close()
            # Tensors read in place keep the files mapped, and Windows deletes no mapped file.
            del mapped, found, want
            import gc
            gc.collect()

    def test_matrices_are_read_as_stored_with_compressed_scales_decoded(self):
        # MXFP8 (FP8 codes with one exponent byte per 32 inputs of a row) goes to the projections slot as mxfp8, FP8
        # in other blocks is read and re-encoded for a drafter, and NVFP4 whose scales the checkpoint keeps CSF-compressed (a base byte
        # a row, 4-bit codes above it, exception words) comes with its E4M3 scale bytes again.
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch and safetensors")
        import tempfile
        checkpoint = template("checkpoint")
        base = torch.arange(16, dtype=torch.uint8) + 0x30
        packed = torch.full((16, 1), 1 | (2 << 4), dtype=torch.uint8)       # two groups a row: codes 1 and 2
        fixed = torch.cat([base.view(1, 16), packed.view(1, 16)], dim=1)
        words = torch.tensor([(3 * 2 + 1) | (0x77 << 24)], dtype=torch.int32)   # row 3, group 1 is 0x77
        with tempfile.TemporaryDirectory() as folder:
            save_file({"mx.weight": torch.zeros(4, 64).to(torch.float8_e4m3fn),
                       "mx.weight_scale": torch.full((4, 2), 127, dtype=torch.uint8),
                       "wide.weight": torch.zeros(256, 256).to(torch.float8_e4m3fn),
                       "wide.weight_scale": torch.ones(2, 2),
                       "csf.weight": torch.zeros(16, 16, dtype=torch.uint8),
                       "csf.weight_scale.nvfp4_csf_fixed": fixed,
                       "csf.weight_scale.nvfp4_csf_exceptions": words,
                       "csf.weight_scale_2": torch.tensor(0.5)}, str(Path(folder) / "model.safetensors"))
            ck = checkpoint.Checkpoint(Path(folder))
            matrices = checkpoint.Matrices(ck, None, {}, "cpu")
            self.assertEqual([matrices.format(name) for name in ("mx", "wide", "csf")],
                             ["mxfp8", "fp8:block", "nvfp4"])
            kind, stored = matrices.stored("mx")
            self.assertEqual((kind, tuple(stored["weight"].shape), tuple(stored["scale"].shape)),
                             ("mxfp8", (4, 64), (4, 2)))
            kind, stacked = matrices.stacked(["mx", "mx"])
            self.assertEqual((kind, tuple(stacked["scale"].shape)), ("mxfp8", (8, 2)))
            # FP8 in 128 x 128 blocks: read with its multipliers, refused by name where the projections slot would
            # load it as stored, re-encoded where only a drafter reads it
            kind, stored = matrices.stored("wide")
            self.assertEqual((kind, stored["block"], tuple(stored["scale"].shape)), ("fp8:block", (128, 128), (2, 2)))
            with self.assertRaisesRegex(ValueError, "FP8 in blocks of"):
                matrices.load(["wide"], "draft")
            kind, encoded = matrices.reformat("wide", "nvfp4")
            self.assertEqual((kind, tuple(encoded["weight"].shape), tuple(encoded["scale"].shape)),
                             ("nvfp4", (256, 128), (256, 16)))
            self.assertEqual(matrices.reformat("wide", "bf16")[1]["weight"].dtype, torch.bfloat16)
            kind, stored = matrices.stored("csf")
            scales = stored["scale"].view(torch.uint8)
            want = torch.stack([base + 1, base + 2], dim=1)
            want[3, 1] = 0x77
            self.assertEqual((kind, stored["scale"].dtype), ("nvfp4", torch.float8_e4m3fn))
            self.assertTrue(torch.equal(scales, want))
            self.assertEqual(float(stored["global_scale"]), 2.0)   # ModelOpt's multiplier, as crit's divisor
            ck.close()
            # Tensors read in place keep the file mapped, and Windows deletes no mapped file.
            del matrices, stored, stacked, encoded, scales
            import gc
            gc.collect()


class _Batch:
    def __init__(self, lengths):
        self.round, self.rows = False, len(lengths)
        self.starts = [0]
        for n in lengths:
            self.starts.append(self.starts[-1] + n)
        self.tokens = self.starts[-1]


class _Projections:
    def facts(self):
        return {"row_invariance": {"rows": 8}}


class RowGroupTests(PluginTestCase):
    def test_prompt_chunks_go_alone_and_decode_rows_in_calls_of_invariant_rows(self):
        rows = template("rows")
        groups = rows.RowGroups(_Projections())
        self.assertEqual(groups.decode_rows, 8)
        # ten decode rows, a prompt chunk of 5, two decode rows (StepLayout puts decode rows first; this mixes them)
        found = groups.of(_Batch([1] * 10 + [5] + [1] * 2))
        self.assertEqual(found, [(0, 8, False), (8, 10, False), (10, 15, True), (15, 17, False)])
        self.assertEqual(groups.rows(17), [(0, 8, False), (8, 16, False), (16, 17, False)])


if __name__ == "__main__":
    sys.exit(unittest.main())
