"""model-template's load planned over every real checkpoint crit targets, from their manifests (tests/manifests: the
safetensors headers and configs, no weights; tests/support/manifests.py writes them), on the CPU.

Each checkpoint whose model_type a library layout covers loads through model-template's own code (tests/support/
dry_plan.py: its ComposedModel over the manifest's headers, the slots filled by stand-ins that take the formats the
library's plugins take), with its lookup tables on the device and off it, and must read every tensor of the
checkpoint outside its vision and audio towers: a tensor no part of the model reads is one its load ignores (a
table's scale left behind, a draft layer dropped). A checkpoint no layout covers is listed, not failed."""
from pathlib import Path
import tempfile
import unittest

from support import manifests

try:
    import torch
    from support import dry_plan
except ImportError:   # pragma: no cover - torch is in every environment the plugin tests run in
    torch = None


class ManifestFileTests(unittest.TestCase):
    def test_a_manifest_keeps_every_tensors_name_dtype_and_shape_and_small_values(self):
        import json
        import struct
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
            entries, data = {}, b""
            names = [f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight" for i in range(3) for e in range(4)]
            names += ["model.layers.1.ple.multipliers", "model.layers.7.self_attn.q_proj.weight"]
            for name in names:
                shape, dtype = ([2], "I64") if "multipliers" in name else ([4, 8], "BF16")
                size = 2 * 8 if dtype == "I64" else 4 * 8 * 2
                payload = bytes(range(size)) if dtype == "I64" else bytes(size)
                entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + size]}
                data += payload
            header = json.dumps(entries).encode()
            (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)
            manifest = manifests.read(folder, name="probe")
            path = manifests.write(manifest, folder / "probe.json.gz")
            found = manifests.tensors(manifests.load(path))
            self.assertEqual(sorted(found), sorted(names))
            self.assertEqual(found["model.layers.2.mlp.experts.3.gate_proj.weight"].shape, (4, 8))
            self.assertEqual(found["model.layers.1.ple.multipliers"].value, bytes(range(16)))
            experts = next(g for g in manifest["groups"] if "experts" in g["name"])
            self.assertEqual(experts["grid"], [[[0, 2]], [[0, 3]]])   # 12 tensors as two ranges


class LayerCutTests(unittest.TestCase):
    def test_a_cut_keeps_its_layers_byte_for_byte_and_rewrites_every_layer_reference(self):
        import json
        import os
        import struct
        from support import layer_cut

        def shard(path, tensors):
            entries, data = {}, b""
            for name, payload in tensors.items():
                entries[name] = {"dtype": "U8", "shape": [len(payload)], "data_offsets": [len(data), len(data) + len(payload)]}
                data += payload
            header = json.dumps(entries).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + data)

        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder, "source"), Path(folder, "cut")
            source.mkdir()
            layers = {f"model.layers.{i}.{part}.weight": bytes([i, n]) * 8
                      for i in range(4) for n, part in enumerate(("mlp.experts.0.up_proj", "self_attn.q_proj"))}
            shard(source / "model-1.safetensors", {**layers, "model.embed_tokens.weight": b"e" * 16,
                                                   "model.visual.blocks.0.weight": b"v" * 16,
                                                   "mtp.layers.0.mlp.gate.weight": b"m" * 16})
            shard(source / "table.safetensors", {"model.layers.2.ple.ngram_embedding.shard_0.weight": b"t" * 64})
            weight_map = {n: "model-1.safetensors" for n in [*layers, "model.embed_tokens.weight",
                                                             "model.visual.blocks.0.weight",
                                                             "mtp.layers.0.mlp.gate.weight"]}
            weight_map["model.layers.2.ple.ngram_embedding.shard_0.weight"] = "table.safetensors"
            (source / layer_cut.INDEX).write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
            quant = {"targets": [f"model.layers.{i}.mlp.experts" for i in range(4)],
                     "ignore": ["model.layers.1.self_attn*", "model.layers.2.self_attn*", "mtp.layers.0.self_attn*"],
                     "quantized_layers": {f"model.layers.{i}.self_attn.q_proj": {"quant_algo": "MXFP8"} for i in range(4)}}
            config = {"model_type": "probe", "num_hidden_layers": 4, "ple_layer_ids": [3], "mlp_only_layers": [0],
                      "layer_types": ["linear", "linear", "linear", "full"], "quantization_config": quant}
            (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (source / "tokenizer.json").write_text("{}", encoding="utf-8")
            summary = layer_cut.cut(source, target, [0, 2, 3])
            found = json.loads((target / "config.json").read_text(encoding="utf-8"))
            self.assertEqual((found["num_hidden_layers"], found["layer_types"], found["ple_layer_ids"],
                              found["mlp_only_layers"]), (3, ["linear", "linear", "full"], [2], [0]))
            q = found["quantization_config"]
            self.assertEqual(q["targets"], [f"model.layers.{i}.mlp.experts" for i in range(3)])
            self.assertEqual(q["ignore"], ["model.layers.1.self_attn*", "mtp.layers.0.self_attn*"])
            self.assertEqual(sorted(q["quantized_layers"]), [f"model.layers.{i}.self_attn.q_proj" for i in range(3)])
            # the table's shard keeps its own name for its layer 2... renumbered to 1: written again, not linked
            self.assertEqual(summary["linked"], [])
            got = {}
            for name in ("model-1.safetensors", "table.safetensors"):
                begin, header = layer_cut._header(target / name)
                data = (target / name).read_bytes()
                got.update({k: data[begin + v["data_offsets"][0]:begin + v["data_offsets"][1]]
                            for k, v in header.items() if k != "__metadata__"})
            self.assertEqual(got["model.layers.1.mlp.experts.0.up_proj.weight"], layers["model.layers.2.mlp.experts.0.up_proj.weight"])
            self.assertEqual(got["model.layers.2.self_attn.q_proj.weight"], layers["model.layers.3.self_attn.q_proj.weight"])
            self.assertEqual(got["model.layers.1.ple.ngram_embedding.shard_0.weight"], b"t" * 64)
            self.assertEqual(got["mtp.layers.0.mlp.gate.weight"], b"m" * 16)
            self.assertNotIn("model.visual.blocks.0.weight", got)
            self.assertEqual(sorted(json.loads((target / layer_cut.INDEX).read_text())["weight_map"]), sorted(got))
            self.assertTrue((target / "tokenizer.json").is_file())
            # a prefix of the layers renames nothing: a shard of kept tensors alone is the same file
            prefix = layer_cut.cut(source, Path(folder, "prefix"), [0, 1, 2])
            self.assertEqual(prefix["linked"], ["table.safetensors"])
            self.assertEqual(os.stat(source / "table.safetensors").st_ino,
                             os.stat(Path(folder, "prefix", "table.safetensors")).st_ino)


@unittest.skipIf(torch is None, "torch")
class ManifestPlanTests(unittest.TestCase):
    def test_every_covered_checkpoint_loads_and_reads_every_tensor(self):
        found = manifests.available()
        self.assertTrue(found, "tests/manifests holds the target checkpoints' manifests")
        uncovered = []
        for name, path in found.items():
            manifest = manifests.load(path)
            for placement in ("tables-on", "tables-off"):
                with self.subTest(checkpoint=name, placement=placement):
                    report = dry_plan.plan(manifest, placement)
                    if not report["covered"]:
                        uncovered.append(f"{name} ({report['model_type']})")
                        break
                    self.assertIsNone(report["error"], report["error"])
                    self.assertEqual(report["unread"][:10], [], f"{len(report['unread'])} tensors no part of the "
                                                                 "model read")
        if uncovered:
            print("checkpoints no model-template layout covers: " + ", ".join(uncovered))

    def test_a_draft_layers_link_reserve_counts_every_plan_of_a_captured_round(self):
        # FP4-Local Flash-Next stores its draft layer's experts W4A16 beside the decoder's W4A4: beside the CPU tier
        # that layer's misses go over the link, from a reserve of the slots one decode round routes in it. A captured
        # round lists its victims once, and the draft layer runs each request's catch-up (width + 1 columns) and then
        # its chain's width - 1 steps in it: 4 requests at width 3 and top 10 route 4 x 6 x 10 = 240, not the catch-up's
        # 160 alone, so the reserve is 240 and a prompt takes 512 + 240 + 1 = 753 of the cache's slots.
        found = manifests.available()
        name = "nvidia-Qwen3.8-Flash-Next-FP4-Local"
        if name not in found:
            self.skipTest(f"no {name} manifest")
        report = dry_plan.plan(manifests.load(found[name]), "tables-on")
        self.assertIsNone(report["error"], report["error"])
        begun = report["experts"]
        draft = begun["layers"] - 1
        self.assertEqual((begun["a4"][draft], {begun["a4"][layer] for layer in range(draft)}), (False, {True}))
        concurrency, width = 4, 3   # model-template's defaults
        self.assertEqual((begun["decode_rows"], begun["top"]), (concurrency * (width + 1 + width - 1), 10))
        reserve = min(begun["experts"], begun["decode_rows"] * begun["top"])
        self.assertEqual((reserve, begun["experts"] + reserve + 1), (240, 753))


if __name__ == "__main__":
    unittest.main()
