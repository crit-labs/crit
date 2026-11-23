"""model-template's GGUF reader (code/gguf.py) on small checkpoints written as GGUF's converter writes them
(crit_models.testing.gguf_checkpoint, by the same layout entries): every Hugging Face tensor of each architecture a
finished plugin's layout reads comes back under its name, in its shape and layout, with its values (the stored values
of a quantized matrix, the converter's transforms undone); quantized matrices load as stored, routed experts as each
expert's rows of the file's stacks; joined tensors read as their parts; a step applies to the tensors its entry
names; what maps to no name, an architecture no layout reads and an entry the reader does not take are refused.
Needs torch and safetensors."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reader():
    """code/gguf.py as the template's package imports it."""
    return _load("model_template_gguf", HERE.parent / "code" / "gguf.py")


from crit_models.testing import gguf_checkpoint as fixture  # noqa: E402

# The GGUF architectures the finished model plugins read, each with the layout that reads it (its gguf entry, its draft
# prefix).
LAYOUTS = {arch: layout for path in sorted((HERE.parents[2] / "cache").glob("*/layout.json"))
           for layout in [json.loads(path.read_text(encoding="utf-8"))] for arch in layout.get("gguf", {})}

# The GGUF architectures these tests write, each from its finished model plugin's layout.
GGUF_ARCHS = {"deepseek2", "llama", "qwen3", "qwen35moe", "qwen4exp"}

HIDDEN, LAYERS = 64, 2
LINEAR = {"linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 32,
          "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4}


def _gemma4(t):
    """Gemma 4's layers: four norms around the blocks, a scalar, an MLP beside experts at the layer's prefix."""
    out = {}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm", "post_feedforward_layernorm_1", "pre_feedforward_layernorm_2",
                     "post_feedforward_layernorm_2"):
            out[p + name + ".weight"] = t(HIDDEN)
        out[p + "layer_scalar"] = t(1)
        out[p + "self_attn.q_proj.weight"] = t(4 * 32, HIDDEN)
        out[p + "self_attn.k_proj.weight"] = t(2 * 32, HIDDEN)
        if i == 0:   # the full layer's values are its keys' rows (no value projection)
            out[p + "self_attn.v_proj.weight"] = t(2 * 32, HIDDEN)
        out[p + "self_attn.o_proj.weight"] = t(HIDDEN, 4 * 32)
        out[p + "self_attn.q_norm.weight"] = t(32)
        out[p + "self_attn.k_norm.weight"] = t(32)
        out[p + "mlp.gate_proj.weight"] = t(64, HIDDEN)
        out[p + "mlp.up_proj.weight"] = t(64, HIDDEN)
        out[p + "mlp.down_proj.weight"] = t(HIDDEN, 64)
        out[p + "router.proj.weight"] = t(4, HIDDEN)
        out[p + "router.scale"] = t(HIDDEN)
        out[p + "router.per_expert_scale"] = t(4)
        out[p + "experts.gate_up_proj"] = t(4, 2 * 32, HIDDEN)
        out[p + "experts.down_proj"] = t(4, HIDDEN, 32)
    return out


def _deepseek(t):
    """DeepSeek V3's layers: latent attention with low-rank queries, a dense first layer, then routed experts with a
    score correction and shared experts."""
    out = {}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        out[p + "input_layernorm.weight"] = t(HIDDEN)
        out[p + "post_attention_layernorm.weight"] = t(HIDDEN)
        a = p + "self_attn."
        out[a + "q_a_proj.weight"] = t(64, HIDDEN)
        out[a + "q_a_layernorm.weight"] = t(64)
        out[a + "q_b_proj.weight"] = t(4 * (32 + 16), 64)
        out[a + "kv_a_proj_with_mqa.weight"] = t(32 + 16, HIDDEN)
        out[a + "kv_a_layernorm.weight"] = t(32)
        out[a + "kv_b_proj.weight"] = t(4 * (32 + 32), 32)
        out[a + "o_proj.weight"] = t(HIDDEN, 4 * 32)
        if i == 0:
            out[p + "mlp.gate_proj.weight"] = t(128, HIDDEN)
            out[p + "mlp.up_proj.weight"] = t(128, HIDDEN)
            out[p + "mlp.down_proj.weight"] = t(HIDDEN, 128)
            continue
        out[p + "mlp.gate.weight"] = t(4, HIDDEN)
        out[p + "mlp.gate.e_score_correction_bias"] = t(4)
        for e in range(4):
            for name, shape in (("gate", (32, HIDDEN)), ("up", (32, HIDDEN)), ("down", (HIDDEN, 32))):
                out[f"{p}mlp.experts.{e}.{name}_proj.weight"] = t(*shape)
        for name, shape in (("gate", (32, HIDDEN)), ("up", (32, HIDDEN)), ("down", (HIDDEN, 32))):
            out[f"{p}mlp.shared_experts.{name}_proj.weight"] = t(*shape)
    return out


def _draft(out, arch, t):
    """A draft (MTP) layer: a copy of the last layer's tensors and the layer's own fc and norms."""
    last = f"model.layers.{LAYERS - 1}."
    for name in [n for n in out if n.startswith(last) and ".ple." not in n]:
        out["mtp.layers.0." + name[len(last):]] = out[name].clone()
    if arch == "qwen4exp":
        out["mtp.fc_embedding.weight"] = t(HIDDEN, HIDDEN)
        out["mtp.fc_hidden.weight"] = t(HIDDEN, HIDDEN)
        out["mtp.pre_fc_norm_hidden.weight"] = t(4 * HIDDEN)
        for name in [n for n in out if n.startswith("model.hyper_connection_mixer.")]:
            out["mtp." + name[len("model."):]] = out[name].clone()
    else:
        out["mtp.fc.weight"] = t(HIDDEN, 2 * HIDDEN)
        out["mtp.pre_fc_norm_hidden.weight"] = t(HIDDEN)
    out["mtp.pre_fc_norm_embedding.weight"] = t(HIDDEN)
    out["mtp.norm.weight"] = t(HIDDEN)
    return out


def _tensors(arch, g, draft=False):
    """{hf name: tensor} of a small checkpoint of `arch`, every part its rules cover (with a draft layer: its MTP
    layer)."""
    def t(*shape):
        return torch.randn(*shape, generator=g)
    out = {"model.embed_tokens.weight": t(96, HIDDEN), "lm_head.weight": t(96, HIDDEN)}
    if arch in ("gemma4", "deepseek2"):
        out.update(_gemma4(t) if arch == "gemma4" else _deepseek(t))
        out["model.norm.weight"] = t(HIDDEN)
        return out
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        attention = arch != "qwen4exp" and (arch not in ("qwen35moe",) or i == 1)
        if arch != "qwen4exp":
            out[p + "input_layernorm.weight"] = t(HIDDEN)
            out[p + "post_attention_layernorm.weight"] = t(HIDDEN)
        if attention or (arch == "qwen4exp" and i == 1):
            q_rows = 4 * 32 * (2 if arch in ("qwen35moe", "qwen4exp") else 1)
            out[p + "self_attn.q_proj.weight"] = t(q_rows, HIDDEN)
            out[p + "self_attn.k_proj.weight"] = t(2 * 32, HIDDEN)
            out[p + "self_attn.v_proj.weight"] = t(2 * 32, HIDDEN)
            out[p + "self_attn.o_proj.weight"] = t(HIDDEN, 4 * 32)
            if arch != "llama":
                out[p + "self_attn.q_norm.weight"] = t(32)
                out[p + "self_attn.k_norm.weight"] = t(32)
            if arch == "qwen4exp":
                out[p + "self_attn.indexer.index_qk_proj.weight"] = t(2 * 32 + 32, HIDDEN)
                out[p + "self_attn.indexer.q_layernorm.weight"] = t(32)
                out[p + "self_attn.indexer.k_layernorm.weight"] = t(32)
        elif arch in ("qwen35moe", "qwen4exp"):
            conv = 2 * 2 * 32 + 4 * 32
            out[p + "linear_attn.in_proj_qkv.weight"] = t(conv, HIDDEN)
            out[p + "linear_attn.in_proj_z.weight"] = t(4 * 32, HIDDEN)
            out[p + "linear_attn.in_proj_a.weight"] = t(4, HIDDEN)
            out[p + "linear_attn.in_proj_b.weight"] = t(4, HIDDEN)
            out[p + "linear_attn.A_log"] = t(4)
            out[p + "linear_attn.dt_bias"] = t(4)
            out[p + "linear_attn.conv1d.weight"] = t(conv, 1, 4)
            out[p + "linear_attn.norm.weight"] = t(32)
            out[p + "linear_attn.out_proj.weight"] = t(HIDDEN, 4 * 32)
        if arch == "llama":
            out[p + "mlp.gate_proj.weight"] = t(128, HIDDEN)
            out[p + "mlp.up_proj.weight"] = t(128, HIDDEN)
            out[p + "mlp.down_proj.weight"] = t(HIDDEN, 128)
        else:
            out[p + "mlp.gate.weight"] = t(4, HIDDEN)
            for e in range(4):
                for name, shape in (("gate", (32, HIDDEN)), ("up", (32, HIDDEN)), ("down", (HIDDEN, 32))):
                    out[f"{p}mlp.experts.{e}.{name}_proj.weight"] = t(*shape)
        if arch in ("qwen35moe", "qwen4exp"):
            out[p + "mlp.shared_expert.gate_proj.weight"] = t(32, HIDDEN)
            out[p + "mlp.shared_expert.up_proj.weight"] = t(32, HIDDEN)
            out[p + "mlp.shared_expert.down_proj.weight"] = t(HIDDEN, 32)
            out[p + "mlp.shared_expert_gate.weight"] = t(1, HIDDEN)
        if arch == "qwen4exp":
            for which in ("attn", "mlp"):
                h = f"{p}{which}_hyper_connection."
                out[h + "input_mix_weight_down.weight"] = t(32, 4 * HIDDEN)
                out[h + "input_mix_weight_up.weight"] = t(4 * HIDDEN, 32)
                out[h + "block_inject_weight.weight"] = t(4, 4 * HIDDEN)
                out[h + "hc_norm.weight"] = t(4 * HIDDEN)
    if arch == "qwen4exp":
        out["model.hyper_connection_mixer.input_mix_weight_down.weight"] = t(32, 4 * HIDDEN)
        out["model.hyper_connection_mixer.input_mix_weight_up.weight"] = t(4 * HIDDEN, 32)
        out["model.hyper_connection_mixer.hc_norm.weight"] = t(4 * HIDDEN)
        p = "model.layers.1.ple."
        out[p + "conv1d.weight"] = t(4 * HIDDEN, 1, 4)
        out[p + "key_proj.weight"] = t(4 * HIDDEN, HIDDEN)
        out[p + "value_proj.weight"] = t(HIDDEN, HIDDEN)
        for name in ("norm_key", "norm_query", "norm_conv"):
            out[p + name + ".weight"] = t(4 * HIDDEN)
        out[p + "ple_embedding.ngram_embedding.weight"] = t(40, 16)
        out[p + "ple_embedding.layer_multipliers"] = torch.tensor([3, 5 << 40, 7], dtype=torch.int64)
        out[p + "ple_embedding.ngram_heads_offsets"] = torch.tensor([0, 20], dtype=torch.int64)
        out[p + "ple_embedding.ngram_heads_vocab_sizes"] = torch.tensor([20, 20], dtype=torch.int64)
    else:
        out["model.norm.weight"] = t(HIDDEN)
    if arch == "qwen4exp":
        out["model.norm.weight"] = t(HIDDEN)
    return _draft(out, arch, t) if draft else out


def _config(arch):
    config = {"hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "num_attention_heads": 4, "num_key_value_heads": 2,
              "head_dim": 32, "vocab_size": 96}
    if arch in ("qwen35moe", "qwen4exp"):
        config.update(LINEAR)
    if arch == "qwen4exp":
        config.update({"indexer_n_heads": 2, "indexer_head_dim": 32, "ple_layer_ids": [2], "hc_lowrank": 32})
    if arch == "deepseek2":
        config.update({"kv_lora_rank": 32, "q_lora_rank": 64, "qk_nope_head_dim": 32, "qk_rope_head_dim": 16,
                       "v_head_dim": 32})
    return config


def _expected(tensors, values):
    """{name: (shape, FP32 values or None)} the reader gives for the HF tensors: a fused experts stack (Gemma 4's) as
    each expert's gate, up and down matrices; the stored values of every quantized matrix."""
    out = {}
    for name, want in tensors.items():
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            at = name.rsplit(".", 1)[0] + "."
            down = name.endswith("down_proj")
            for e in range(want.shape[0]):
                if down:
                    out[f"{at}{e}.down_proj.weight"] = torch.from_numpy(values[f"{name}:{e}"])
                    continue
                for which in ("gate", "up"):
                    out[f"{at}{e}.{which}_proj.weight"] = torch.from_numpy(values[f"{name}:{e}:{which}"])
            continue
        out[name] = (torch.from_numpy(values[name]).reshape(want.shape) if name in values
                     else want if want.dtype == torch.int64 else want.float())
    return out


@unittest.skipIf(torch is None, "torch, numpy and safetensors")
class ReaderTests(unittest.TestCase):
    def setUp(self):
        # The finished model plugins' layouts lie beside the core in the repo (models/cache); an engine package
        # carries its own model plugin only.
        missing = sorted(GGUF_ARCHS - set(LAYOUTS))
        if missing:
            self.skipTest(f"the layouts of {', '.join(missing)} are not beside this core (models/cache)")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gguf = _reader()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, arch, tensors=None, config=None, name="hf", draft=False, layout=None):
        folder = self.root / name
        folder.mkdir()
        tensors = tensors if tensors is not None else _tensors(arch, torch.Generator().manual_seed(len(arch)), draft)
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(folder / "model.safetensors"))
        config = config or _config(arch)
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
        path, values = fixture.write_gguf(folder, self.root / (name + "-gguf"), arch, layout=layout or LAYOUTS[arch])
        return path, tensors, values, config

    def assertReadsBack(self, arch, draft=False):
        path, tensors, values, config = self.write(arch, draft=draft)
        ck = self.gguf.GgufCheckpoint(path.parent, config, LAYOUTS[arch])
        steps = self.gguf.rules_of(LAYOUTS[arch]["gguf"][arch]).transforms
        zero = steps.get("zero-centred", {"ends": []})
        constants = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
        expected = _expected(tensors, values)
        self.assertEqual(sorted(n for n in ck.names() if not n.endswith(constants)),
                         sorted(n for n in expected if not n.endswith(constants)), arch)
        for name, want in expected.items():
            self.assertEqual(ck.shape(name), tuple(want.shape), name)
            got = ck.get(name, "cpu")
            if want.dtype == torch.int64:
                self.assertEqual(got.tolist(), want.tolist(), name)
                continue
            # the converter's elementwise steps undone in FP32 (1 + w, -exp): one rounding each way
            transformed = name.endswith(tuple(steps.get("negated-exp", ()))) or (
                name.endswith(tuple(zero["ends"])) and not name.endswith(tuple(zero.get("except", ()))))
            if transformed:
                torch.testing.assert_close(got, want, rtol=0, atol=2e-6 if "A_log" not in name else 2e-5,
                                           msg=name)
            else:
                self.assertTrue(torch.equal(got, want), name)
        return ck, tensors, values

    def test_every_tensor_reads_back_for_each_architectures_rules(self):
        for arch, draft in (("llama", False), ("qwen3moe", False), ("qwen35moe", True), ("qwen4exp", True),
                            ("gemma4", False), ("deepseek2", False)):
            with self.subTest(arch):
                self.assertReadsBack(arch, draft)
                for path in self.root.iterdir():   # one architecture's folders at a time
                    shutil.rmtree(path)

    def test_a_draft_block_reads_as_the_draft_layer_and_its_fc_halves_as_stored_blocks(self):
        # Qwen4Exp's converter joins fc_embedding and fc_hidden by inputs into one eh_proj; each half loads as the
        # stored blocks of its columns, and decodes to that half's stored values.
        ck, tensors, values = self.assertReadsBack("qwen4exp", draft=True)
        m = self.gguf.GgufMatrices(ck, None, None, "cpu")
        from crit_core.model.gguf_blocks import TYPES, stored_values
        for name in ("mtp.fc_embedding", "mtp.fc_hidden"):
            fmt, found = m.stored(name)
            self.assertEqual((fmt, tuple(found["weight"].shape)), ("gguf:Q8_0", (HIDDEN, HIDDEN // 32 * 34)))
            self.assertTrue(np.array_equal(stored_values(found["weight"].numpy(), TYPES["Q8_0"], HIDDEN),
                                           values[name + ".weight"]))
        self.assertEqual(m.expert_formats("mtp.layers.0.mlp.experts."), ("Q8_0", "Q8_0"))

    def test_deepseeks_split_latent_matrix_reads_joined_and_decoded(self):
        ck, tensors, values = self.assertReadsBack("deepseek2")
        m = self.gguf.GgufMatrices(ck, None, None, "cpu")
        self.assertEqual(m.format("model.layers.0.self_attn.kv_b_proj"), "bf16")   # a plain matrix, decoded
        self.assertEqual(m.format("model.layers.0.self_attn.kv_a_proj_with_mqa"), "gguf:Q8_0")

    def test_quantized_matrices_load_as_stored_and_experts_as_their_stack_rows(self):
        ck, tensors, values = self.assertReadsBack("qwen35moe")
        loaded = []

        class Projections:
            def load(self, found, fmt, role=None):
                loaded.append((fmt, found["weight"]))
                return fmt
        m = self.gguf.GgufMatrices(ck, Projections(), None, "cpu")
        base = "model.layers.0.linear_attn.out_proj"
        self.assertEqual(m.format(base), "gguf:Q8_0")
        self.assertEqual(m.format("model.layers.0.linear_attn.norm"), "bf16")
        fmt, found = m.stored(base)
        from crit_core.model.gguf_blocks import TYPES, stored_values
        decoded = stored_values(found["weight"].numpy(), TYPES["Q8_0"], 128)
        self.assertTrue(np.array_equal(decoded, values[base + ".weight"]))   # the value heads moved back as blocks
        self.assertEqual(m.load([base], "out"), "gguf:Q8_0")
        self.assertEqual(m.rows(base), HIDDEN)
        fmt, gate_up, down = m.expert(*(f"model.layers.1.mlp.experts.2.{n}_proj" for n in ("gate", "up", "down")))
        self.assertEqual((fmt, gate_up["type"], down["type"]), ("gguf", "Q8_0", "Q8_0"))
        self.assertEqual(tuple(gate_up["weight"].shape), (64, 2 * 34))
        both = stored_values(gate_up["weight"].numpy(), 8, HIDDEN)
        want = np.concatenate([values[f"model.layers.1.mlp.experts.2.{n}_proj.weight"] for n in ("gate", "up")])
        self.assertTrue(np.array_equal(both, want))
        self.assertEqual(m.expert_formats("model.layers.1.mlp.experts."), ("Q8_0", "Q8_0"))
        m.stored_experts = False   # an experts plugin that takes no GGUF experts: their values in BF16
        fmt, gate_up, down = m.expert(*(f"model.layers.1.mlp.experts.2.{n}_proj" for n in ("gate", "up", "down")))
        self.assertEqual((fmt, gate_up.dtype, tuple(down.shape)), ("bf16", torch.bfloat16, (HIDDEN, 32)))
        self.assertTrue(torch.equal(gate_up, torch.from_numpy(want).to(torch.bfloat16)))

    def test_merged_tensors_read_as_their_halves(self):
        # Converters merge tensors that read one input ([a; b]); the template reads each half as its own tensor.
        g = torch.Generator().manual_seed(5)
        conv = 2 * 4 * 32 + 4 * 32   # four key heads, as many as value heads: no value-head order
        qkv, z, b, a = (torch.randn(r, HIDDEN, generator=g) for r in (conv, 4 * 32, 4, 4))
        down, inject = torch.randn(32, 4 * HIDDEN, generator=g), torch.randn(4, 4 * HIDDEN, generator=g)
        f32 = lambda x: x.float().numpy().astype(np.float32)  # noqa: E731
        tensors = [fixture._Tensor("blk.0.attn_qkv_gate.weight", 0, (conv + 4 * 32, HIDDEN), f32(torch.cat([qkv, z])).tobytes()),
                   fixture._Tensor("blk.0.ssm_beta_alpha.weight", 0, (8, HIDDEN), f32(torch.cat([b, a])).tobytes()),
                   fixture._Tensor("blk.0.hc_attn_down_inject.weight", 0, (36, 4 * HIDDEN),
                                   f32(torch.cat([down, inject])).tobytes())]
        path = self.root / "merged.gguf"
        fixture._write(path, {"general.architecture": "qwen4exp", "qwen4exp.block_count": 1}, tensors)
        config = {**_config("qwen4exp"), "num_hidden_layers": 1, "linear_num_key_heads": 4}   # no value-head order
        ck = self.gguf.GgufCheckpoint(path, config, LAYOUTS["qwen4exp"])
        p = "model.layers.0."
        for name, want in ((p + "linear_attn.in_proj_qkv.weight", qkv), (p + "linear_attn.in_proj_z.weight", z),
                           (p + "linear_attn.in_proj_b.weight", b), (p + "linear_attn.in_proj_a.weight", a),
                           (p + "attn_hyper_connection.input_mix_weight_down.weight", down),
                           (p + "attn_hyper_connection.block_inject_weight.weight", inject)):
            self.assertTrue(torch.equal(ck.get(name, "cpu"), want), name)

    def test_a_tensor_no_name_maps_to_and_an_unknown_architecture_are_refused(self):
        x = np.zeros((4, 32), np.float32).tobytes()
        path = self.root / "odd.gguf"
        fixture._write(path, {"general.architecture": "qwen3", "qwen3.block_count": 1},
                       [fixture._Tensor("blk.0.attn_mystery.weight", 0, (4, 32), x)])
        with self.assertRaisesRegex(ValueError, "map to no name the template reads: blk.0.attn_mystery.weight"):
            self.gguf.GgufCheckpoint(path, {"num_hidden_layers": 1}, LAYOUTS["qwen3"])
        path = self.root / "other.gguf"
        fixture._write(path, {"general.architecture": "mamba", "mamba.block_count": 1}, [])
        with self.assertRaisesRegex(ValueError, "general.architecture is 'mamba'"):
            self.gguf.GgufCheckpoint(path, {}, LAYOUTS["qwen3"])

    def test_value_heads_move_only_where_the_format_keeps_heads_whole(self):
        # Q8_0's 32-value blocks hold whole 32-wide value heads; a format whose blocks span heads cannot be moved.
        path, tensors, values, config = self.write("qwen35moe")
        config = {**config, "linear_value_head_dim": 16, "linear_key_head_dim": 16}
        with self.assertRaisesRegex(ValueError, "do not align with the 16-wide heads"):
            ck = self.gguf.GgufCheckpoint(path.parent, config, LAYOUTS["qwen35moe"])
            ck.rows("model.layers.0.linear_attn.out_proj.weight")

    def test_a_gguf_folder_is_told_from_a_safetensors_one(self):
        path, *_ = self.write("llama")
        self.assertTrue(self.gguf.is_gguf(path.parent))
        self.assertTrue(self.gguf.is_gguf(path))
        self.assertFalse(self.gguf.is_gguf(self.root / "hf"))

    def test_a_step_applies_to_the_tensors_its_parameters_name(self):
        # The reader's steps take their tensors from the layout's entry: the attention norms of a file whose entry says
        # its converter stored them 1 + w read back as w through that entry, and as stored through one that does not.
        llama = LAYOUTS["llama"]["gguf"]["llama"]
        layout = {"gguf": {"llama": {**llama, "transforms": {
            **llama["transforms"], "zero-centred": {"ends": ["input_layernorm.weight"]}}}}}
        path, tensors, values, config = self.write("llama", layout=layout)
        name = "model.layers.0.input_layernorm.weight"
        through = self.gguf.GgufCheckpoint(path.parent, config, layout)
        plain = self.gguf.GgufCheckpoint(path.parent, config, LAYOUTS["llama"])
        want = tensors[name].float()
        torch.testing.assert_close(through.get(name, "cpu"), want, rtol=0, atol=2e-6)
        torch.testing.assert_close(plain.get(name, "cpu"), want + 1, rtol=0, atol=2e-6)
        self.assertTrue(torch.equal(through.get("model.norm.weight", "cpu"), plain.get("model.norm.weight", "cpu")))
        through.close()
        plain.close()

    def test_an_entry_the_reader_does_not_take_is_refused_by_name(self):
        for entry, problem in (({"parts": ["gated-delta"]}, "parts is"), ({"streams": True}, "streams is no key"),
                               ({"transforms": ["zero-centred"]}, "transforms is"),
                               ({"transforms": {"latent": True}}, "latent is no step"),
                               ({"transforms": {"joined-rows": {"x": {"names": ["a"], "rows": 1}}}}, "joined-rows is"),
                               ({"transforms": {"squeezed": {"conv1d.weight": {"axis": 3, "rank": 3}}}},
                                "squeezed is")):
            with self.subTest(problem), self.assertRaisesRegex(ValueError, problem):
                self.gguf.rules_of(entry)
        # a draft block's parts where the layout has no draft prefix to read them under
        path, *_ = self.write("llama")
        with self.assertRaisesRegex(ValueError, "no draft prefix"):
            self.gguf.GgufCheckpoint(path.parent, _config("llama"),
                                     {"gguf": {"llama": {"draft": {"nextn.enorm": "x"}}}})


if __name__ == "__main__":
    unittest.main()
