"""The models core on the host: every finished model plugin's layout is well formed, its manifest declares the options
compose() reads with the same defaults, a layout naming what no technique gives is refused, prompt spans follow the
timed cost model within the budget and the room, and row groups multiply each prompt chunk alone and one-token rows
in calls of at most invariant_rows."""
import importlib
import json
import sys
import unittest
from pathlib import Path

from crit_plugins.testing import PluginTestCase

HERE = Path(__file__).resolve().parents[1]


def template(name):
    """A module of the core's code, imported as a model plugin imports it (its code/ folder the source `template`)."""
    from crit_plugins.assemble import plugin_package
    from crit_plugins.manifest import load_manifest
    return importlib.import_module(f"{plugin_package(load_manifest(HERE)).__name__}.template.{name}")


class LayoutTests(PluginTestCase):
    def test_the_layout_of_every_model_plugin_that_extends_this_core_checks(self):
        compose = template("compose")
        checked = 0
        for manifest in self.registry():
            if (manifest.get("extends") or {}).get("id") != "model-template":
                continue
            path = Path(manifest["folder"]) / compose.LAYOUT
            self.assertEqual(compose.check(json.loads(path.read_text(encoding="utf-8")), manifest["id"]), [])
            checked += 1
        self.assertTrue(checked or not (HERE.parents[1] / "cache").is_dir())   # crit's library holds some

    def test_the_manifest_declares_composes_options_with_its_defaults(self):
        compose = template("compose")
        manifest = json.loads((HERE / "plugin.json").read_text(encoding="utf-8"))
        self.assertEqual({k: v["default"] for k, v in manifest["options"].items()}, compose.DEFAULTS)
        self.assertEqual(manifest["fills"], [])   # a core: model plugins extend it, no engine selects it

    def test_the_draft_layers_precision_on_auto_follows_the_quality_mode(self):
        from crit_plugins import kernels
        shape, compose = template("shape"), template("compose")
        self.addCleanup(kernels.use_quality_mode, None)
        for mode, want in (("speed", ("nvfp4", "fp8:row")), ("preserve", ("bf16", "bf16")), (None, ("bf16", "bf16"))):
            with self.subTest(mode=mode):
                kernels.use_quality_mode(mode)
                found = shape.precision(compose.DEFAULTS)
                self.assertEqual((found["mtp_encode"], found["mtp_kv"]), want)
                self.assertEqual(found["kv"], "auto")   # resolved with the checkpoint's KV scales (Shape)
                set_ = shape.precision({**compose.DEFAULTS, "mtp_encode": "fp8", "mtp_kv": "bf16"})
                self.assertEqual((set_["mtp_encode"], set_["mtp_kv"]), ("fp8", "bf16"))   # an option set is that way

    def test_a_layout_naming_an_unknown_block_or_key_is_refused(self):
        compose = template("compose")
        layout = {"schema": compose.SCHEMA, "mixers": {"full_attention": {"block": "lstm"}}, "colour": 1,
                  "residual": {"block": "braid"}}
        problems = compose.check(layout)
        self.assertTrue(any("unknown keys colour" in p for p in problems), problems)
        self.assertTrue(any("mixers.full_attention names a block" in p for p in problems), problems)
        self.assertTrue(any("residual names how blocks join" in p for p in problems), problems)
        del layout["colour"]   # the plugin's own techniques, which compose() takes beside the library's
        self.assertEqual(compose.check(layout, blocks={"lstm": object}, residuals={"braid": object}), [])

    def test_the_core_alone_composes_no_model(self):
        compose = template("compose")
        with self.assertRaisesRegex(ValueError, "the core a model plugin extends"):
            compose.layout_of(None)
        with self.assertRaisesRegex(ValueError, "the core a model plugin extends"):
            compose.layout_of(compose.LAYOUT)   # a layout lies in a model plugin's folder


class SpanTests(PluginTestCase):
    def test_the_calibration_fits_a_steps_seconds_and_a_rows_bytes(self):
        # The fit is Core's prompt-step budget's first guess (crit_plugins.steps, tests/test_steps.py).
        rows = template("rows")
        costs = rows.fit({(16, 0): 0.001, (64, 0): 0.002, (64, 1000): 0.003}, {16: 16 << 20, 64: 64 << 20})
        self.assertEqual(costs["m"], float(f"{1 << 20:.4g}"))
        self.assertEqual(set(costs), {"c", "a", "b", "m"})

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
            # FP8 in 128 x 128 blocks: read with its multipliers and handed to the projections slot as stored (it
            # loads fp8:block on fp8-block-rows), refused by name by a slot whose formats leave it out, re-encoded
            # where only a drafter reads it
            kind, stored = matrices.stored("wide")
            self.assertEqual((kind, stored["block"], tuple(stored["scale"].shape)), ("fp8:block", (128, 128), (2, 2)))

            class Slot:
                formats = ("bf16", "fp8:channel", "fp8:block")

                def load(self, tensors, fmt, role=None, encode=None):
                    return fmt, tensors
            matrices.proj = Slot()
            fmt, tensors = matrices.load(["wide"], "draft")
            self.assertEqual((fmt, tensors["block"]), ("fp8:block", (128, 128)))
            class Narrow(Slot):
                formats = ("bf16", "fp8:channel")
            matrices.proj = Narrow()
            with self.assertRaisesRegex(ValueError, "FP8 in blocks of"):
                matrices.load(["wide"], "draft")
            matrices.proj = None
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
            del matrices, stored, stacked, encoded, scales, tensors
            import gc
            gc.collect()

    def test_expert_stacks_under_a_layouts_names_read_as_stored(self):
        # A checkpoint that names its tensors its own way (the layout's `names`) and stores its experts as two stacks:
        # NVFP4 whose factor is an amax over a denominator, one amax an expert, with each gate row beside its up row;
        # and MXFP8. Each expert reaches the slot as stored, gate rows then up rows, and decodes to the values the
        # checkpoint's reference decodes it to (E2M1 codes x E4M3 scale x amax / denominator, rows de-interleaved).
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch and safetensors")
        import tempfile
        checkpoint = template("checkpoint")
        g = torch.Generator().manual_seed(7)
        experts, inter, hidden = 3, 16, 64
        codes = torch.randint(0, 256, (experts, 2 * inter, hidden // 2), dtype=torch.uint8, generator=g)
        scale = (torch.rand(experts, 2 * inter, hidden // 16, generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
        amax = torch.rand(experts, generator=g) + 0.5
        denominator = torch.tensor([3.0])
        down_codes = torch.randint(0, 256, (experts, hidden, inter // 2), dtype=torch.uint8, generator=g)
        down_scale = torch.ones(experts, hidden, 1).to(torch.float8_e4m3fn)
        mx = (torch.randn(experts, 2 * inter, hidden, generator=g)).to(torch.float8_e4m3fn)
        mx_scale = torch.randint(120, 130, (experts, 2 * inter, hidden // 32), dtype=torch.uint8, generator=g)
        names = [[r"experts\.w13_", "experts.gate_up_proj."], [r"experts\.w2_", "experts.down_proj."]]
        e2m1 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0))

        def reference(weight, block_scale, factor):   # the reference's decode, then its Interleave(dim=1)
            pairs = torch.stack((e2m1[weight.long() & 0xF], e2m1[weight.long() >> 4]), dim=-1)
            values = pairs.view(*block_scale.shape, 16) * block_scale.float()[..., None] * factor
            values = values.reshape(weight.shape[0], -1)
            return values.reshape(values.shape[0] // 2, 2, -1).transpose(0, 1).reshape(values.shape)

        with tempfile.TemporaryDirectory() as folder:
            save_file({"l0.experts.w13_weight": codes, "l0.experts.w13_weight_scale": scale,
                       "l0.experts.w13_weight_amax": amax, "l0.experts.w13_weight_outer_scale_denominator": denominator,
                       "l0.experts.w2_weight": down_codes, "l0.experts.w2_weight_scale": down_scale,
                       "l0.experts.w2_weight_amax": amax.clone(),
                       "l0.experts.w2_weight_outer_scale_denominator": denominator.clone(),
                       "l1.experts.w13_weight": mx, "l1.experts.w13_weight_scale": mx_scale,
                       "l1.experts.w2_weight": torch.zeros(experts, hidden, inter * 8).to(torch.float8_e4m3fn),
                       "l1.experts.w2_weight_scale": torch.full((experts, hidden, inter * 8 // 32), 127,
                                                                dtype=torch.uint8)},
                      str(Path(folder) / "model.safetensors"))
            ck = checkpoint.Checkpoint(Path(folder), names)
            self.assertIn("l0.experts.gate_up_proj.weight_amax", ck)
            self.assertEqual(ck.stored_names["l0.experts.gate_up_proj.weight"], "l0.experts.w13_weight")
            matrices = checkpoint.Matrices(ck, None, {}, "cpu")
            self.assertEqual([matrices.format(f"l{i}.experts.gate_up_proj") for i in (0, 1)], ["nvfp4", "mxfp8"])
            for e in range(experts):
                fmt, gate_up, down = matrices.stacked_expert("l0.experts.gate_up_proj", "l0.experts.down_proj", e,
                                                             interleaved=True)
                self.assertEqual((fmt, tuple(gate_up["weight"].shape)), ("nvfp4", (2 * inter, hidden // 2)))
                self.assertAlmostEqual(float(gate_up["global_scale"]), float(denominator / amax[e]), places=6)
                # crit's convention: code x scale / global_scale (a divisor), the rows as they reach the slot
                w, sc = gate_up["weight"].long(), gate_up["scale"].float()
                pairs = torch.stack((e2m1[w & 0xF], e2m1[w >> 4]), dim=-1).view(*sc.shape, 16)
                decoded = (pairs * sc[..., None] / gate_up["global_scale"]).reshape(2 * inter, hidden)
                want = reference(codes[e], scale[e], float(amax[e] / denominator))
                self.assertTrue(torch.allclose(decoded, want, rtol=1e-6, atol=1e-6), e)
                fmt, gate_up, down = matrices.stacked_expert("l1.experts.gate_up_proj", "l1.experts.down_proj", e,
                                                          interleaved=True)
                self.assertEqual(fmt, "mxfp8")
                self.assertTrue(torch.equal(gate_up["weight"].view(torch.uint8),
                                            torch.cat([mx[e, 0::2], mx[e, 1::2]]).view(torch.uint8)))
                self.assertTrue(torch.equal(gate_up["scale"], torch.cat([mx_scale[e, 0::2], mx_scale[e, 1::2]])))
            ck.close()
            del matrices, gate_up, down
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
