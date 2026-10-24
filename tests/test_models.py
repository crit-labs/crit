"""crit models: recipes, plans, calibration windows, codecs, the PTQ workflow and its records, on tiny host fixtures."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "eval", "kernels", "models", "cli"): sys.path.insert(0, str(ROOT / _folder))
from crit.run.records import write_json
from crit_core.model import inspect_model
from crit_models import calibration, recipe as recipes
LAUNCHER = str(ROOT / "tests/run_crit.py")

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None
HAS_TRANSFORMERS = HAS_TORCH and importlib.util.find_spec("transformers") is not None


def recipe(**overrides):
    value = {"schema_version": 1, "record_kind": "models_recipe", "method": "ptq", "format": "nvfp4", "algorithm": "rtn"}
    value.update(overrides)
    return value


def write_header_only(path, tensors):
    """A safetensors file of zero bytes per tensor spec {name: (dtype, shape)}; the standard library is enough."""
    width = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = width[dtype] * shape[0] * (shape[1] if len(shape) > 1 else 1)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(data)) + data + bytes(offset))


def layer_specs(layers=2, hidden=32, inter=64):
    specs = {"model.embed_tokens.weight": ("BF16", (64, hidden)), "model.norm.weight": ("BF16", (hidden,)),
             "lm_head.weight": ("BF16", (64, hidden))}
    for i in range(layers):
        p = f"model.layers.{i}."
        specs.update({p + "input_layernorm.weight": ("BF16", (hidden,)),
                      p + "post_attention_layernorm.weight": ("BF16", (hidden,)),
                      p + "self_attn.q_proj.weight": ("BF16", (hidden, hidden)),
                      p + "self_attn.o_proj.weight": ("BF16", (hidden, hidden)),
                      p + "mlp.gate_proj.weight": ("BF16", (inter, hidden)),
                      p + "mlp.down_proj.weight": ("BF16", (hidden, inter))})
    return specs


class RecipeTests(unittest.TestCase):
    def test_defaults_are_frozen_into_the_recipe(self):
        value = recipes.load_recipe(recipe(algorithm="gptq", format="fp8-channel"))
        self.assertEqual(value["include"], list(recipes.DEFAULT_INCLUDE))
        self.assertEqual(value["calibration"], recipes.DEFAULT_CALIBRATION)
        self.assertEqual(value["gptq"], recipes.DEFAULT_GPTQ)
        self.assertEqual(value["seed"], 0)
        self.assertNotIn("calibration", recipes.load_recipe(recipe()))

    def test_invalid_recipes_are_rejected(self):
        for bad in (recipe(method="prune"), recipe(format="int4"), recipe(algorithm="awq"),
                    recipe(calibration={"samples": 4}), recipe(algorithm="gptq", gptq={"block": 24}),
                    recipe(include=["("]), recipe(include=[]), recipe(seed=-1), recipe(extra=1),
                    recipe(algorithm="gptq", calibration={"samples": 4, "unknown": 1}), recipe(schema_version=2)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                recipes.load_recipe(bad)


class PlanTests(unittest.TestCase):
    def test_plan_matches_projections_and_counts_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            write_header_only(Path(directory) / "model.safetensors", layer_specs())
            plan = recipes.plan(recipes.load_recipe(recipe(exclude=[r"\.o_proj$"])), inspect_model(directory))
            self.assertEqual(plan["prefix"], "model.")
            self.assertEqual([t["module"] for t in plan["targets"]],
                             ["layers.0.mlp.down_proj", "layers.0.mlp.gate_proj", "layers.0.self_attn.q_proj",
                              "layers.1.mlp.down_proj", "layers.1.mlp.gate_proj", "layers.1.self_attn.q_proj"])
            self.assertEqual(plan["stored_bytes"], 2 * 2 * (64 * 32 * 2 + 32 * 32))
            self.assertEqual(plan["targets"][0]["transformed_bytes"], 32 * 64 // 2 + 32 * 64 // 16 + 4)
            fp8 = recipes.plan(recipes.load_recipe(recipe(format="fp8-channel")), inspect_model(directory))
            self.assertEqual(fp8["targets"][0]["transformed_bytes"], 32 * 64 + 32 * 4)

    def test_plan_rejects_quantized_odd_and_empty_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = layer_specs(layers=1)
            specs["model.layers.0.self_attn.q_proj.weight"] = ("F8_E4M3", (32, 32))
            specs["model.layers.0.self_attn.q_proj.weight_scale"] = ("F32", (32, 1))
            write_header_only(Path(directory) / "model.safetensors", specs)
            with self.assertRaisesRegex(ValueError, "unquantized"):
                recipes.plan(recipes.load_recipe(recipe()), inspect_model(directory))
            with self.assertRaisesRegex(ValueError, "matches no module"):
                recipes.plan(recipes.load_recipe(recipe(include=["nothing"])), inspect_model(directory))
        with tempfile.TemporaryDirectory() as directory:
            specs = layer_specs(layers=1)
            specs["model.layers.0.mlp.down_proj.weight"] = ("BF16", (32, 40))
            write_header_only(Path(directory) / "model.safetensors", specs)
            with self.assertRaisesRegex(ValueError, "divisible by 16"):
                recipes.plan(recipes.load_recipe(recipe()), inspect_model(directory))
            self.assertTrue(recipes.plan(recipes.load_recipe(recipe(format="fp8-channel")), inspect_model(directory)))

    def test_plan_command_prints_targets_without_torch_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_header_only(root / "model.safetensors", layer_specs(layers=1))
            write_json(root / "recipe.json", recipe())
            out = subprocess.run([sys.executable, "-B", "-c", "import sys, runpy; sys.modules['torch'] = None; "
                                  f"sys.argv = ['crit', 'models', 'plan', '--recipe', {str(root / 'recipe.json')!r}, "
                                  f"'--source', {directory!r}]; runpy.run_path({LAUNCHER!r}, run_name='__main__')"],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("layers.0.mlp.gate_proj  BF16 64x32", out.stdout)
            self.assertIn("4 modules", out.stdout)


class CalibrationTests(unittest.TestCase):
    def test_windows_are_seeded_and_bounded(self):
        documents = [f"doc {i}" for i in range(10)]

        def encode(text):
            return [ord(c) for c in text]
        first = calibration.windows(documents, encode, [0], 3, 5, seed=7)
        self.assertEqual(first, calibration.windows(documents, encode, [0], 3, 5, seed=7))
        self.assertNotEqual(first, calibration.windows(documents, encode, [0], 3, 5, seed=8))
        self.assertTrue(all(len(w) == 5 for w in first))
        with self.assertRaisesRegex(ValueError, "needs 30 x 5"):
            calibration.windows(documents, encode, [0], 30, 5, seed=7)

    def test_documents_quoting_the_eval_suite_are_found(self):
        texts = calibration.suite_texts()
        self.assertTrue(texts)
        quality = json.loads((ROOT / "eval/suite/quality.json").read_text(encoding="utf-8"))
        quoted = max((m["content"] for c in quality["conversations"] for m in c["messages"]), key=len)
        found = calibration.suite_overlap(["An unrelated document about rivers and maps.",
                                           "Notes:\n" + quoted + "\nend"], texts)
        self.assertEqual(found, [1])

    def test_jsonl_documents_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('{"text": "one"}\n\n{"text": "two"}\n', encoding="utf-8")
            self.assertEqual(calibration.read_documents(path), ["one", "two"])
            path.write_text('{"text": ""}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                calibration.read_documents(path)


@unittest.skipUnless(HAS_TORCH, "needs torch and safetensors")
class CodecTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        from crit_models import codecs
        self.codecs = codecs
        generator = torch.Generator().manual_seed(3)
        self.weight = torch.randn(24, 64, generator=generator) * torch.linspace(0.01, 2.0, 64)

    def test_nvfp4_matches_the_inference_codec(self):
        spec = importlib.util.spec_from_file_location("nvfp4", ROOT / "inference/quantized-projections/code/nvfp4.py")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        tensors, values, errors = self.codecs.quantize("nvfp4", self.weight)
        g = float(tensors["weight_global_scale"][0])
        self.assertEqual(g, self.torch.tensor(reference.global_scale(self.weight.flatten().tolist())).float().item())
        for r, row in enumerate(self.weight.tolist()):
            codes, scales = reference.quantize_row(row, g)
            self.assertEqual(bytes(tensors["weight_packed"][r].tolist()), reference.pack(codes))
            self.assertEqual(bytes(tensors["weight_scale"][r].view(self.torch.uint8).tolist()), bytes(scales))
            decoded = reference.dequantize_row(codes, scales, g)
            self.assertTrue(all(abs(a - b) <= 1e-6 * max(1.0, abs(b)) for a, b in zip(values[r].tolist(), decoded)))
        self.assertLess(errors["weight_error"], 0.2)

    def test_fp8_channel_scales_rows_and_round_trips(self):
        tensors, values, errors = self.codecs.quantize("fp8-channel", self.weight)
        self.assertEqual(tensors["weight"].dtype, self.torch.float8_e4m3fn)
        self.assertEqual(list(tensors["weight_scale"].shape), [24, 1])
        self.assertTrue(self.torch.equal(tensors["weight"].float() * tensors["weight_scale"], values))
        self.assertTrue(self.torch.equal(tensors["weight"].float().abs().amax(dim=1), self.torch.full((24,), 448.0)))
        self.assertLess(errors["weight_error"], 0.05)

    def test_gptq_with_an_identity_hessian_is_round_to_nearest(self):
        for format in ("fp8-channel", "nvfp4"):
            with self.subTest(format=format):
                rtn, _, _ = self.codecs.quantize(format, self.weight)
                gptq, _, _ = self.codecs.quantize(format, self.weight, self.torch.eye(64), {"dampening": 0.0, "block": 32})
                self.assertEqual(rtn.keys(), gptq.keys())
                for key in rtn:
                    self.assertTrue(self.torch.equal(rtn[key].view(self.torch.uint8) if rtn[key].dtype == self.torch.float8_e4m3fn else rtn[key],
                                                     gptq[key].view(self.torch.uint8) if gptq[key].dtype == self.torch.float8_e4m3fn else gptq[key]), key)

    def test_gptq_lowers_the_output_error_on_correlated_inputs(self):
        torch = self.torch
        generator = torch.Generator().manual_seed(5)
        mix = torch.randn(64, 64, generator=generator)
        inputs = torch.randn(512, 64, generator=generator) @ mix
        hessian = 2 * inputs.T @ inputs / inputs.shape[0]
        for format in ("fp8-channel", "nvfp4"):
            with self.subTest(format=format):
                _, rtn, _ = self.codecs.quantize(format, self.weight)
                _, gptq, errors = self.codecs.quantize(format, self.weight, hessian, {"dampening": 0.01, "block": 32})
                rtn_error = ((inputs @ (self.weight - rtn).T) ** 2).sum()
                gptq_error = ((inputs @ (self.weight - gptq).T) ** 2).sum()
                self.assertLess(float(gptq_error), float(rtn_error))
                self.assertIn("output_error", errors)


def tiny_model(directory, layers=2, sharded=True):
    """A random Qwen3-style checkpoint: config, a byte-level tokenizer and BF16 weights in one or two shards."""
    import torch
    from safetensors.torch import save_file
    from core_text_fixtures import byte_level_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = byte_level_json()
    vocab = len(json.loads(tokenizer)["model"]["vocab"]) + 16
    (directory / "tokenizer.json").write_text(tokenizer, encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|end_of_text|>"}), encoding="utf-8")
    hidden, inter, heads, kv, head_dim = 64, 128, 4, 2, 16
    config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": hidden,
              "intermediate_size": inter, "num_hidden_layers": layers, "num_attention_heads": heads,
              "num_key_value_heads": kv, "head_dim": head_dim, "vocab_size": vocab, "max_position_embeddings": 512,
              "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
              "hidden_act": "silu", "attention_bias": False, "eos_token_id": json.loads(tokenizer)["added_tokens"][1]["id"]}
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    generator = torch.Generator().manual_seed(11)

    def tensor(*shape, scale=0.08):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)
    shards = [{"model.embed_tokens.weight": tensor(vocab, hidden, scale=0.5)}, {}]
    for i in range(layers):
        p = f"model.layers.{i}."
        target = shards[i % 2] if sharded else shards[0]
        target.update({p + "input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
                       p + "post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
                       p + "self_attn.q_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
                       p + "self_attn.k_norm.weight": torch.ones(head_dim, dtype=torch.bfloat16),
                       p + "self_attn.q_proj.weight": tensor(heads * head_dim, hidden),
                       p + "self_attn.k_proj.weight": tensor(kv * head_dim, hidden),
                       p + "self_attn.v_proj.weight": tensor(kv * head_dim, hidden),
                       p + "self_attn.o_proj.weight": tensor(hidden, heads * head_dim),
                       p + "mlp.gate_proj.weight": tensor(inter, hidden),
                       p + "mlp.up_proj.weight": tensor(inter, hidden),
                       p + "mlp.down_proj.weight": tensor(hidden, inter)})
    shards[-1 if sharded else 0].update({"model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
                                         "lm_head.weight": tensor(vocab, hidden)})
    if not sharded:
        save_file(shards[0], str(directory / "model.safetensors"), metadata={"format": "pt"})
        return directory
    weight_map = {}
    for n, shard in enumerate(shards, 1):
        name = f"model-0000{n}-of-00002.safetensors"
        save_file(shard, str(directory / name), metadata={"format": "pt"})
        weight_map.update({key: name for key in shard})
    total = sum(t.numel() * t.element_size() for s in shards for t in s.values())
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
                                                                         "weight_map": weight_map}), encoding="utf-8")
    return directory


@unittest.skipUnless(HAS_TORCH, "needs torch and safetensors")
class PtqTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sys.path.insert(0, str(ROOT / "tests"))

    def test_rtn_exports_targets_keeps_everything_else_and_records_hashes(self):
        import torch
        from safetensors import safe_open
        from crit_eval.reference import Checkpoint
        from crit_models.evaluate import load_record
        from crit_models.ptq import run_ptq, verify
        source = tiny_model(self.root / "source")
        value = recipes.load_recipe(recipe(exclude=[r"\.down_proj$"], seed=4))
        record = run_ptq(value, source, self.root / "out", self.root / "transform.json", device="cpu", log=lambda m: None)
        self.assertEqual(load_record(self.root / "transform.json"), record)
        out = self.root / "out"
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                          "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"])
        self.assertEqual(record["output"]["copied_files"], ["tokenizer.json", "tokenizer_config.json"])
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        group = config["quantization_config"]["config_groups"]["group_0"]
        self.assertEqual(len(group["targets"]), record["plan"]["modules"])
        self.assertEqual(record["plan"]["modules"], 2 * 6)
        self.assertNotIn("model.layers.0.mlp.down_proj", group["targets"])
        index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["weight_map"]["model.layers.1.self_attn.q_proj.weight_packed"],
                         "model-00002-of-00002.safetensors")
        self.assertEqual(inspect_model(out).tensor_bytes, index["metadata"]["total_size"])
        before, after = Checkpoint(source), Checkpoint(out)
        for key in before.where:
            base, _, leaf = key.rpartition(".")
            if base in group["targets"]:
                self.assertFalse(after.has(key))
                restored = after.dense(base, "cpu")
                original = before.raw(key).float()
                self.assertLess(float((restored - original).norm() / original.norm()), 0.2)
            else:
                self.assertTrue(torch.equal(before.raw(key), after.raw(key)), key)
        self.assertEqual(record["seed"], 4)
        self.assertIsNone(record["calibration"])
        self.assertEqual(record["source"]["manifest_sha256"],
                         __import__("crit.run.records", fromlist=["digest"]).digest(record["source"]["files"]))
        self.assertIn("quantize_and_export", record["cost"]["phases"])
        self.assertGreater(record["errors"]["weight_error"]["mean"], 0)
        self.assertNotEqual(record["source"]["eval_model_sha256"], record["output"]["eval_model_sha256"])
        self.assertEqual(verify(record), [])
        with safe_open(str(out / "model-00001-of-00002.safetensors"), "pt") as handle:
            self.assertEqual(handle.metadata(), {"format": "pt"})
        (out / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        self.assertEqual(verify(record), ["output differs from the record: tokenizer_config.json"])

    def test_rtn_twice_gives_identical_bytes(self):
        from crit_models.ptq import run_ptq
        source = tiny_model(self.root / "source", sharded=False)
        value = recipes.load_recipe(recipe(format="fp8-channel"))
        first = run_ptq(value, source, self.root / "a", self.root / "a.json", device="cpu", log=lambda m: None)
        second = run_ptq(value, source, self.root / "b", self.root / "b.json", device="cpu", log=lambda m: None)
        self.assertEqual(first["output"]["manifest_sha256"], second["output"]["manifest_sha256"])
        self.assertFalse((self.root / "a/model.safetensors.index.json").exists())

    def test_workflow_refuses_unsafe_paths_and_quantized_sources(self):
        from crit_models.ptq import run_ptq
        source = tiny_model(self.root / "source", sharded=False)
        value = recipes.load_recipe(recipe())
        quiet = lambda m: None
        (self.root / "full").mkdir()
        (self.root / "full/file").write_text("x")
        for out, record in ((source / "out", self.root / "r.json"), (self.root / "full", self.root / "r.json"),
                            (self.root / "out", self.root / "out/r.json"), (self.root / "out", source / "r.json")):
            with self.subTest(out=out, record=record), self.assertRaises(ValueError):
                run_ptq(value, source, out, record, device="cpu", log=quiet)
        with self.assertRaisesRegex(ValueError, "rtn takes none"):
            run_ptq(value, source, self.root / "out", self.root / "r.json", calibration=self.root / "c.jsonl",
                    device="cpu", log=quiet)
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        config["quantization_config"] = {"quant_method": "fp8"}
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unquantized checkpoint"):
            run_ptq(value, source, self.root / "out", self.root / "r.json", device="cpu", log=quiet)

    @unittest.skipUnless(HAS_TRANSFORMERS, "needs transformers")
    def test_gptq_calibrates_layer_by_layer_and_tracks_the_original(self):
        from crit_eval.reference import StreamingReference
        from crit_models.ptq import run_ptq
        source = tiny_model(self.root / "source")
        words = "river map stone cloud lantern harbor meadow copper violin orchard".split()
        lines = [json.dumps({"text": " ".join(words[(i + j) % len(words)] for j in range(40))}) for i in range(20)]
        (self.root / "calibration.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        value = recipes.load_recipe(recipe(algorithm="gptq", format="nvfp4", seed=2,
                                           calibration={"samples": 4, "sequence_tokens": 32}, gptq={"block": 32}))
        record = run_ptq(value, source, self.root / "out", self.root / "transform.json",
                         calibration=self.root / "calibration.jsonl", device="cpu", log=lambda m: None)
        facts = record["calibration"]
        self.assertEqual((facts["samples"], facts["sequence_tokens"], facts["documents"]), (4, 32, 20))
        self.assertEqual(facts["suite_overlap"], "checked")
        self.assertEqual(len(facts["windows_sha256"]), 64)
        self.assertIn("calibrate_and_quantize", record["cost"]["phases"])
        self.assertIn("output_error", record["errors"])
        self.assertFalse((self.root / "out/.crit-staging").exists())
        ids = list(range(1, 40))
        original = StreamingReference(source, "cpu").score_texts([ids], 1, None, self.root / "scratch-a",
                                                                 log=lambda m: None)[0][0]
        transformed = StreamingReference(self.root / "out", "cpu").score_texts([ids], 1, None, self.root / "scratch-b",
                                                                              log=lambda m: None)[0][0]
        self.assertLess(float((original - transformed).abs().mean()), 0.5)
        repeat = run_ptq(value, source, self.root / "again", self.root / "again.json",
                         calibration=self.root / "calibration.jsonl", device="cpu", log=lambda m: None)
        self.assertEqual(repeat["calibration"]["windows_sha256"], facts["windows_sha256"])


@unittest.skipUnless(HAS_TORCH, "needs torch and safetensors")
class EvaluationTests(unittest.TestCase):
    status = {"gates": [{"name": "quality", "state": "pass"}],
              "kpis": {"quality": {"passed": True, "worst_ppl_ratio": 1.01}, "decode": {"score": 90.0},
                       "prefill": {"score": 900.0}, "ttft": {"score_ms": 12.0}}}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sys.path.insert(0, str(ROOT / "tests"))
        from crit_models.ptq import run_ptq
        self.source = tiny_model(self.root / "source", sharded=False)
        run_ptq(recipes.load_recipe(recipe()), self.source, self.root / "out", self.root / "transform.json",
                device="cpu", log=lambda m: None)
        self.data = self.root / "data"
        self.data.mkdir()

    def prepared(self, model):
        from crit_eval.prepare import model_identity
        self.model_sha256 = model_identity(model)["sha256"]
        write_json(self.data / "prepared.json", {"model": model_identity(model), "suite_sha256": "suite"})

    def result(self, **fields):
        value = {"model_sha256": self.model_sha256, "suite_sha256": "suite", "adapter": None,
                 "server": "http://127.0.0.1:8000/v1", "engine": {"name": "out"}, "status": self.status,
                 "ended_at": "2026-10-21T00:00:00+00:00", **fields}
        write_json(self.root / "result.json", {k: v for k, v in value.items() if v is not ...})

    def evaluate(self):
        from crit_models.evaluate import evaluate
        return evaluate(self.root / "transform.json", self.data, self.root / "result.json",
                        self.root / "evaluation.json", log=lambda m: None)

    def test_eval_requires_data_prepared_from_the_source_model(self):
        with self.assertRaisesRegex(ValueError, "prepare it from the source"):
            self.evaluate()
        self.prepared(self.root / "out")
        with self.assertRaisesRegex(ValueError, "another model than the source"):
            self.evaluate()

    def test_eval_binds_the_served_result_to_the_transform(self):
        self.prepared(self.source)
        self.result()
        evaluation = self.evaluate()
        self.assertEqual(evaluation["server"], {"url": "http://127.0.0.1:8000/v1", "model": "out"})
        self.assertEqual(evaluation["summary"]["gates"], {"quality": "pass"})
        self.assertEqual(evaluation["summary"]["decode_score"], 90.0)
        self.assertEqual(json.loads((self.root / "evaluation.json").read_text(encoding="utf-8")), evaluation)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.evaluate()
        (self.root / "evaluation.json").unlink()
        (self.root / "out/config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "output differs"):
            self.evaluate()

    def test_eval_refuses_other_data_adapter_and_unfinished_results(self):
        self.prepared(self.source)
        for fields, message in (({"suite_sha256": "other"}, "other prepared data"),
                                ({"adapter": "adapter.py", "server": None}, "adapter's engine"),
                                ({"aborted": {"reason": "disturbed"}}, "no finished Eval round"),
                                ({"status": ...}, "no finished Eval round")):
            with self.subTest(fields=fields):
                self.result(**fields)
                with self.assertRaisesRegex(ValueError, message):
                    self.evaluate()
        self.assertFalse((self.root / "evaluation.json").exists())

    def test_eval_records_the_quality_its_round_judged_without_running_the_reference(self):
        # The round judged quality against the source's reference scores stored under the data (crit eval reference
        # --model SOURCE --data DATA): the record takes its gates as they are.
        self.prepared(self.source)
        self.result(quality={"status": "scored", "passed": True})
        with patch("crit_eval.reference.score_data", side_effect=AssertionError("ran the reference")):
            evaluation = self.evaluate()
        self.assertEqual(evaluation["summary"]["gates"], {"quality": "pass"})


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_TRANSFORMERS, "needs torch, safetensors and transformers")
class DrafterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sys.path.insert(0, str(ROOT / "tests"))

    def drafter_recipe(self, **overrides):
        value = {"schema_version": 1, "record_kind": "models_recipe", "method": "drafter", "seed": 3,
                 "calibration": {"samples": 6, "sequence_tokens": 32}, "heldout_fraction": 0.34,
                 "train": {"steps": 12, "windows_per_step": 2, "learning_rate": 1e-3, "warmup_steps": 2}}
        value.update(overrides)
        return value

    def test_drafter_recipes_fill_their_defaults_and_refuse_the_unknown(self):
        value = recipes.load_recipe(self.drafter_recipe())
        self.assertEqual((value["kind"], value["dtype"], value["init_layer"]), ("mtp-head", "bf16", -1))
        self.assertEqual(value["train"]["weight_decay"], 0.0)
        self.assertEqual(recipes.load_recipe({"schema_version": 1, "record_kind": "models_recipe", "method": "drafter"})["calibration"],
                         {"samples": 64, "sequence_tokens": 1024})
        for bad in (self.drafter_recipe(kind="eagle"), self.drafter_recipe(heldout_fraction=0.9),
                    self.drafter_recipe(format="nvfp4"), self.drafter_recipe(train={"epochs": 1})):
            with self.assertRaises(ValueError):
                recipes.load_recipe(bad)
        with self.assertRaises(ValueError):
            recipes.load_recipe(recipe(method="prune"))

    def test_a_drafter_is_distilled_recorded_and_reloadable(self):
        import torch
        from safetensors import safe_open
        from crit_models import main
        from crit_models.drafter import run_drafter
        from crit_models.evaluate import load_record
        from crit_models.ptq import verify
        source = tiny_model(self.root / "source")
        words = "river map stone cloud lantern harbor meadow copper violin orchard".split()
        lines = [json.dumps({"text": " ".join(words[(i + j) % len(words)] for j in range(40))}) for i in range(20)]
        (self.root / "calibration.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(self.root / "recipe.json", self.drafter_recipe())
        value = recipes.load_recipe(self.drafter_recipe())
        record = run_drafter(value, source, self.root / "drafter", self.root / "drafter.json", self.root / "calibration.jsonl",
                             device="cpu", log=lambda m: None)
        self.assertEqual(load_record(self.root / "drafter.json"), record)
        out = self.root / "drafter"
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["config.json", "model.safetensors"])
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["num_hidden_layers"], 1)
        self.assertEqual(config["crit_drafter"]["kind"], "mtp-head")
        self.assertEqual((config["crit_drafter"]["init_layer"], config["crit_drafter"]["target_layers"]), (1, 2))
        self.assertEqual(config["crit_drafter"]["target"]["eval_model_sha256"], record["source"]["eval_model_sha256"])
        with safe_open(str(out / "model.safetensors"), "pt") as handle:
            keys = set(handle.keys())
            self.assertEqual(handle.get_tensor("drafter.fc.weight").shape, (64, 128))
            self.assertEqual(handle.get_tensor("drafter.fc.weight").dtype, torch.bfloat16)
        for name in ("drafter.layers.0.self_attn.q_proj.weight", "drafter.norm.weight", "drafter.pre_fc_norm_hidden.weight",
                     "drafter.pre_fc_norm_embedding.weight"):
            self.assertIn(name, keys)
        self.assertEqual(record["plan"]["init_layer"], 1)
        self.assertEqual(sorted(record["plan"]["tensors"]), sorted(keys))
        self.assertEqual((record["calibration"]["train_windows"], record["calibration"]["heldout_windows"]), (4, 2))
        errors = record["errors"]
        self.assertEqual(errors["train_loss"]["steps"], 12)
        self.assertTrue(0 <= errors["heldout"]["top1_agreement"] <= 1)
        self.assertGreaterEqual(errors["heldout"]["kl_mean"], 0)
        self.assertIn("heldout_before", errors)
        self.assertEqual(sorted(record["cost"]["phases"]), ["export", "hash_source", "target_pass", "train"])
        self.assertEqual(verify(record), [])
        shown = subprocess.run([sys.executable, "-B", LAUNCHER, "models", "plan", "--recipe", str(self.root / "recipe.json"),
                                "--source", str(source)], capture_output=True, text=True)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("drafter from layer 1 of 2", shown.stdout)
        with self.assertRaises(SystemExit):
            main(["ptq", "--recipe", str(self.root / "recipe.json"), "--source", str(source), "--out", str(self.root / "x"),
                  "--record", str(self.root / "x.json")])
