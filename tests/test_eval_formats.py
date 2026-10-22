"""Eval over weight formats and quantization schemes beyond plain safetensors: PyTorch state dicts, and the schemes of
crit_eval.schemes. For each, the reference decodes the stored weights to exactly the FP32 values the format's own
library produces, the model scores exactly as transformers scores its dequantized twin, and the ceiling counts the
same matrices as the unquantized model at the scheme's rate.

The fixtures are tiny models written on the CPU by each format's own tool: GPTQModel (GPTQ, AWQ), mlx-lm,
bitsandbytes through transformers, llm-compressor (compressed-tensors), and Optimum with onnxruntime's MatMulNBits
quantizer (ONNX). A test is skipped when its tool is not
installed; they need torch, transformers, safetensors and tokenizers."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval import ceiling  # noqa: E402


def _has(*names):
    return all(importlib.util.find_spec(name) for name in names)


HAS_REFERENCE = _has("torch", "transformers", "safetensors", "tokenizers")
CALIBRATION = ["the quick brown fox jumps over the lazy dog " * 8] * 16


def tokenizer(out):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=298, special_tokens=["<|endoftext|>", "<|im_start|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(["the quick brown fox jumps over the lazy dog " * 4, "gguf models"], trainer)
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast",
                                                           "eos_token": "<|endoftext|>",
                                                           "pad_token": "<|endoftext|>"}))


def tiny(out, hidden=128, intermediate=256, head_dim=32):
    """A two-layer Qwen3 in FP32 with a tokenizer, its weights small and random."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    out.mkdir(parents=True)
    tokenizer(out)
    torch.manual_seed(0)
    config = AutoConfig.for_model("qwen3", vocab_size=320, eos_token_id=0, hidden_size=hidden,
                                  intermediate_size=intermediate, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=head_dim, tie_word_embeddings=False)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05 + (1.0 if "norm" in key else 0))
    model.save_pretrained(out)
    return out


def sequences():
    import torch
    torch.manual_seed(1)
    return [(torch.randint(0, 300, (40,)).tolist(), 5), (torch.randint(0, 300, (23,)).tolist(), 3)]


def scored(folder, sequences, top_k=5):
    """Per (ids, first target): the reference's (log-probabilities, top_k ids, their log-probabilities) of the targets
    from the first on, as lists (score_texts scores every position)."""
    import tempfile
    from crit_eval.reference import StreamingReference
    with tempfile.TemporaryDirectory() as scratch:
        rows = StreamingReference(folder, device="cpu").score_texts([ids for ids, _ in sequences], top_k, None, scratch,
                                                                    log=lambda *_: None)
    return [tuple(part[first - 1:].tolist() for part in row) for (_, first), row in zip(sequences, rows)]


@unittest.skipUnless(HAS_REFERENCE, "needs torch, transformers, safetensors and tokenizers")
class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.small = tiny(cls.root / "small")
        # Wide enough for several quantization groups and FP8 blocks per row (llm-compressor groups by 128).
        cls.wide = tiny(cls.root / "wide", hidden=256, intermediate=640, head_dim=64)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def checkpoint(self, folder):
        from crit_eval.reference import _text_config, open_checkpoint
        return open_checkpoint(folder, _text_config(folder))

    def assert_scores_like_dequantized(self, folder):
        """The reference over `folder` scores exactly as transformers scores the model rebuilt from the weights the
        reference decodes, so the stored scheme reaches the modeling code intact."""
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from crit_eval.schemes import scheme
        checkpoint = self.checkpoint(folder)
        scores = scored(folder, sequences())
        config = json.loads((folder / "config.json").read_text())
        for key in ("quantization_config", "quantization"):
            config.pop(key, None)
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model(**config), dtype=torch.float32,
                                                 attn_implementation="eager")
        state = {}
        for key in model.state_dict():
            base = key[:-len(".weight")]
            stored = key.endswith(".weight") and (checkpoint.has(key) or scheme(checkpoint, base))
            state[key] = checkpoint.dense(base, "cpu") if stored else checkpoint.raw(key).float()
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            for (ids, first), (got, _, _) in zip(sequences(), scores):
                logprobs = torch.log_softmax(model(torch.tensor([ids])).logits[0].double(), -1)
                self.assertEqual(got, [logprobs[i - 1, ids[i]].item() for i in range(first, len(ids))], folder.name)

    def assert_counted_like(self, folder, original, rate):
        """The ceiling counts the original's matrices and FLOPs, at `rate`."""
        plain, counted = ceiling.model_ceiling(original), ceiling.model_ceiling(folder)
        self.assertEqual(counted["matmul_flops_per_token"], {rate: plain["matmul_flops_per_token"]["bf16"]})
        self.assertEqual(counted["matrices"], {rate: plain["matrices"]["bf16"]})
        self.assertLess(counted["weight_bytes"], plain["weight_bytes"])

    def assert_decodes_like(self, folder, library):
        """Every quantized matrix decodes to the library's own FP32 dequantization: library(checkpoint, base)."""
        import torch
        from crit_eval.schemes import scheme
        checkpoint = self.checkpoint(folder)
        bases = sorted({key.rsplit(".", 1)[0] for key in checkpoint.where if scheme(checkpoint, key.rsplit(".", 1)[0])})
        self.assertGreaterEqual(len(bases), 14, folder.name)
        for base in bases:
            expected = library(checkpoint, base).float()
            got = checkpoint.dense(base, "cpu")
            self.assertEqual(got.shape, expected.shape, base)
            self.assertTrue(torch.equal(got, expected), f"{folder.name} {base}")

    def test_a_state_dict_scores_like_its_safetensors(self):
        import torch
        from safetensors.torch import load_file
        from crit_core.model import open_model
        tensors = load_file(self.small / "model.safetensors")
        whole, split = self.root / "torch", self.root / "torch-split"
        for folder in (whole, split):
            shutil.copytree(self.small, folder, ignore=shutil.ignore_patterns("*.safetensors"))
        torch.save(tensors, whole / "pytorch_model.bin")
        names = sorted(tensors)
        halves = {"pytorch_model-00001-of-00002.bin": names[::2], "pytorch_model-00002-of-00002.bin": names[1::2]}
        for file, keys in halves.items():
            torch.save({key: tensors[key] for key in keys}, split / file)
        (split / "pytorch_model.bin.index.json").write_text(json.dumps(
            {"weight_map": {key: file for file, keys in halves.items() for key in keys}}))
        expected = scored(self.small, sequences())
        for folder in (whole, split):
            self.assertEqual(open_model(folder).format, "torch")
            self.assertEqual(scored(folder, sequences()), expected)
            self.assertEqual(ceiling.model_ceiling(folder), ceiling.model_ceiling(self.small))

    def test_block_fp8_scales_decode_as_transformers_does(self):
        # DeepSeek and Qwen FP8 checkpoints: one weight_scale_inv per 128 x 128 block, FP32 or E8M0.
        import torch
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize
        from crit_eval.reference import StateDictCheckpoint
        torch.manual_seed(0)
        weight = (torch.randn(256, 384) * 100).to(torch.float8_e4m3fn)
        for scale in (torch.rand(2, 3) * 0.01,
                      torch.pow(2.0, torch.randint(-12, -4, (2, 3)).float()).to(torch.float8_e8m0fnu)):
            with self.subTest(scale.dtype):
                checkpoint = object.__new__(StateDictCheckpoint)
                checkpoint.where = {"m.weight": weight, "m.weight_scale_inv": scale}
                checkpoint.quantization = {"quant_method": "fp8", "weight_block_size": [128, 128]}
                expected = Fp8Dequantize(None)._dequantize_one(weight, scale, torch.float32)
                self.assertTrue(torch.equal(checkpoint.dense("m", "cpu"), expected))

    @unittest.skipUnless(_has("gptqmodel"), "needs gptqmodel")
    def test_gptq(self):
        from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
        for bits, symmetric in ((2, True), (3, True), (3, False), (4, True), (4, False), (8, True)):
            with self.subTest(bits=bits, symmetric=symmetric):
                folder = self.root / f"gptq-{bits}-{symmetric}"
                model = GPTQModel.load(str(self.small), QuantizeConfig(bits=bits, group_size=32, desc_act=True,
                                                                       sym=symmetric), device="cpu")
                model.quantize(CALIBRATION, batch_size=4)
                model.save(str(folder))
                quantized = GPTQModel.load(str(folder), device="cpu", backend=BACKEND.TORCH)
                modules = dict(quantized.model.named_modules())

                def library(checkpoint, base):
                    # GPTQModel's own dequantization, over the stored scales (it casts them to its load dtype)
                    module = modules[base]
                    module.scales.data = checkpoint.raw(base + ".scales").float()
                    return module.dequantize_weight().T
                self.assert_decodes_like(folder, library)
                if (bits, symmetric) in ((4, True), (3, False)):
                    self.assert_scores_like_dequantized(folder)
                    self.assert_counted_like(folder, self.small, "bf16")

    @unittest.skipUnless(_has("gptqmodel", "awq"), "needs gptqmodel and autoawq")
    def test_awq(self):
        from awq.utils.packing_utils import dequantize_gemm
        from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
        from gptqmodel.quantization import FORMAT, METHOD
        folder = self.root / "awq"
        config = QuantizeConfig(bits=4, group_size=32, quant_method=METHOD.AWQ, format=FORMAT.GEMM)
        model = GPTQModel.load(str(self.small), config, device="cpu", backend=BACKEND.AWQ_TORCH)
        model.quantize(CALIBRATION, batch_size=4, backend=BACKEND.AWQ_TORCH)
        model.save(str(folder))

        def library(checkpoint, base):   # AutoAWQ's dequantize_gemm
            return dequantize_gemm(checkpoint.raw(base + ".qweight"), checkpoint.raw(base + ".qzeros"),
                                   checkpoint.raw(base + ".scales").float(), 4, 32).T
        self.assert_decodes_like(folder, library)
        self.assert_scores_like_dequantized(folder)
        self.assert_counted_like(folder, self.small, "bf16")

    @unittest.skipUnless(_has("mlx", "mlx_lm", "numpy"), "needs mlx and mlx-lm")
    def test_mlx(self):
        import subprocess
        import mlx.core as mx
        import numpy as np
        import torch
        from crit_eval.schemes import _mlx
        cases = {"affine-4": (["--q-bits", "4", "--q-group-size", "32"], "bf16"),
                 "affine-3": (["--q-bits", "3", "--q-group-size", "64"], "bf16"),
                 "affine-6-bf16": (["--q-bits", "6", "--q-group-size", "32", "--dtype", "bfloat16"], "bf16"),
                 "mixed": (["--quant-predicate", "mixed_3_6", "--q-group-size", "32"], "bf16"),
                 "mxfp4": (["--q-mode", "mxfp4"], "nvfp4"), "nvfp4": (["--q-mode", "nvfp4"], "nvfp4"),
                 "mxfp8": (["--q-mode", "mxfp8"], "fp8")}
        for name, (options, rate) in cases.items():
            with self.subTest(name):
                folder = self.root / f"mlx-{name}"
                subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(self.small),
                                "--mlx-path", str(folder), "-q", *options], check=True, capture_output=True)
                arrays = mx.load(str(folder / "model.safetensors"))
                quantization = json.loads((folder / "config.json").read_text())["quantization"]

                def library(checkpoint, base):   # mx.dequantize, with FP32 affine scales and biases
                    bits, group, mode = _mlx(quantization, base)
                    scales, biases = arrays[base + ".scales"], arrays.get(base + ".biases")
                    if mode == "affine":
                        scales, biases = scales.astype(mx.float32), biases.astype(mx.float32)
                    weight = mx.dequantize(arrays[base + ".weight"], scales, biases, group_size=group, bits=bits,
                                           mode=mode)
                    return torch.from_numpy(np.array(weight.astype(mx.float32)))
                self.assert_decodes_like(folder, library)
                if name in ("affine-4", "mixed", "mxfp4"):
                    self.assert_scores_like_dequantized(folder)
                self.assertEqual(ceiling.model_ceiling(folder)["matmul_flops_per_token"],
                                 {rate: ceiling.model_ceiling(self.small)["matmul_flops_per_token"]["bf16"]})

    @unittest.skipUnless(_has("bitsandbytes"), "needs bitsandbytes")
    def test_bitsandbytes(self):
        import torch
        import bitsandbytes.functional as functional
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        cases = {"nf4-double": (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_use_double_quant=True), "bf16"),
                 "nf4": (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"), "bf16"),
                 "fp4-double": (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4",
                                                   bnb_4bit_use_double_quant=True), "bf16"),
                 "int8": (BitsAndBytesConfig(load_in_8bit=True), "fp8")}
        for name, (config, rate) in cases.items():
            with self.subTest(name):
                folder = self.root / f"bnb-{name}"
                AutoModelForCausalLM.from_pretrained(self.small, quantization_config=config, device_map="cpu",
                                                     dtype=torch.float32).save_pretrained(folder)

                def library(checkpoint, base):   # bitsandbytes' own dequantization, in FP32
                    weight = checkpoint.raw(base + ".weight")
                    if checkpoint.has(base + ".SCB"):
                        return functional.int8_vectorwise_dequant(weight, checkpoint.raw(base + ".SCB"))
                    prefix = base + ".weight."
                    state = functional.QuantState.from_dict(
                        {k[len(prefix):]: checkpoint.raw(k) for k in checkpoint.where if k.startswith(prefix)}, "cpu")
                    state.dtype = torch.float32
                    return functional.dequantize_4bit(weight, state)
                self.assert_decodes_like(folder, library)
                if name in ("nf4-double", "int8"):
                    self.assert_scores_like_dequantized(folder)
                    self.assert_counted_like(folder, self.small, rate)

    @unittest.skipUnless(_has("llmcompressor", "compressed_tensors"), "needs llm-compressor")
    def test_compressed_tensors(self):
        import torch
        from compressed_tensors.compressors.naive_quantized.base import NaiveQuantizationCompressor
        from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
        from compressed_tensors.quantization import QuantizationScheme
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from transformers import AutoModelForCausalLM
        cases = {"W4A16": "bf16", "W4A16_ASYM": "bf16", "W8A16": "bf16", "W8A8": "fp8", "FP8_BLOCK": "fp8",
                 "FP8_DYNAMIC": "fp8"}
        for name, rate in cases.items():
            with self.subTest(name):
                folder = self.root / f"ct-{name}"
                model = AutoModelForCausalLM.from_pretrained(self.wide, dtype=torch.bfloat16)
                oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme=name, ignore=["lm_head"]))
                model.save_pretrained(folder, save_compressed=True)
                group = json.loads((folder / "config.json").read_text())["quantization_config"]["config_groups"]
                scheme = QuantizationScheme(targets=["Linear"], weights=group["group_0"]["weights"])

                def library(checkpoint, base):   # compressed-tensors' decompressors, with FP32 scales
                    local = {k[len(base) + 1:]: checkpoint.raw(k) for k in checkpoint.where if k.startswith(base + ".")}
                    local["weight_scale"] = local["weight_scale"].float()
                    packed = "weight_packed" in local
                    compressor = PackedQuantizationCompressor if packed else NaiveQuantizationCompressor
                    return compressor.decompress(local, scheme)["weight"]
                self.assert_decodes_like(folder, library)
                if name in ("W4A16_ASYM", "W8A8", "FP8_BLOCK"):
                    self.assert_scores_like_dequantized(folder)
                self.assert_counted_like(folder, self.wide, rate)


    @unittest.skipUnless(_has("optimum", "onnx", "onnxruntime", "numpy"), "needs optimum-onnx, onnx and onnxruntime")
    def test_onnx(self):
        import numpy as np
        import onnx
        import onnxruntime
        import torch
        from onnx import TensorProto, helper, numpy_helper
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        from optimum.exporters.onnx import main_export
        from crit_core.model import open_model
        exported, quantized = self.root / "onnx", self.root / "onnx-q4"
        main_export(str(self.small), output=str(exported), task="text-generation", no_post_process=True)
        self.assertEqual(open_model(exported).format, "onnx")
        self.assert_scores_like_dequantized(exported)
        self.assertEqual(ceiling.model_ceiling(exported)["matmul_flops_per_token"],
                         ceiling.model_ceiling(self.small)["matmul_flops_per_token"])
        quantizer = MatMulNBitsQuantizer(onnx.load(exported / "model.onnx"), block_size=32, is_symmetric=False)
        quantizer.process()
        shutil.copytree(exported, quantized, ignore=shutil.ignore_patterns("*.onnx*"))
        quantizer.model.save_model_to_file(str(quantized / "model.onnx"), use_external_data_format=True)
        graph = onnx.load(quantized / "model.onnx")
        stored = {t.name: numpy_helper.to_array(t, base_dir=str(quantized)) for t in graph.graph.initializer}

        def library(checkpoint, base):
            # onnxruntime's own MatMulNBits kernel times the identity: the dequantized weight, transposed.
            node = next(n for n in graph.graph.node if n.op_type == "MatMulNBits"
                        and n.name.strip("/").replace("/", ".").startswith(base + "."))
            attributes = {a.name: a.i for a in node.attribute}
            k, n = attributes["K"], attributes["N"]
            single = helper.make_graph(
                [helper.make_node("MatMulNBits", ["A", *node.input[1:]], ["Y"], domain="com.microsoft",
                                  **attributes)], "one",
                [helper.make_tensor_value_info("A", TensorProto.FLOAT, [k, k])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [k, n])],
                [numpy_helper.from_array(stored[name], name) for name in node.input[1:] if name])
            model = helper.make_model(single, ir_version=10, opset_imports=[helper.make_opsetid("", 18),
                                                                           helper.make_opsetid("com.microsoft", 1)])
            session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            return torch.from_numpy(session.run(None, {"A": np.eye(k, dtype=np.float32)})[0]).T
        self.assert_decodes_like(quantized, library)
        self.assert_scores_like_dequantized(quantized)
        self.assert_counted_like(quantized, self.small, "bf16")


if __name__ == "__main__":
    unittest.main()
