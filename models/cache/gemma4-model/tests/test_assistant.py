"""The companion MTP drafter (techniques/assistant-drafter) against the official implementation, transformers'
Gemma4AssistantForCausalLM, on tiny random checkpoints: a Gemma 4 target with sliding and full layers of two head
shapes (its full layers' values its keys' rows) and a gemma4_assistant drafter of both layer types beside it.

On the host: the drafter's own arithmetic (step(), the code its device head runs) over plain FP32 products, its
weights read from the companion folder by the core's companion reader, in both checkpoint layouts (the drafter as
published, and under another spelling its layout's draft `names` rules map back), gives the official drafter's
logits and next row for the same target keys, values and final-normed row, and the same chain of greedy drafts; and
the target's row the official generator feeds the drafter is its final-normed row, which the models core keeps as
the draft feature. The engine finds the folder beside the checkpoint and plans the drafter as its draft head. On a GPU
(drafting runs in Core's captured rounds), an engine drafting from it replies the plain greedy tokens."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from crit_models import testing as t

HERE = Path(__file__).resolve().parents[1]
torch, transformers, GREEDY = t.torch, t.transformers, t.GREEDY

TARGET = dict(vocab_size=256, vocab_size_per_layer_input=256, hidden_size=128, intermediate_size=256,
              num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32, global_head_dim=64,
              num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=24,
              layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
              enable_moe_block=False, hidden_size_per_layer_input=0, max_position_embeddings=2048, rms_norm_eps=1e-6,
              tie_word_embeddings=True, final_logit_softcapping=30.0, pad_token_id=0, eos_token_id=1, bos_token_id=2,
              rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                                  "rope_type": "proportional"},
                               "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}})
# The drafter: as wide as its own layers want, reading the target's last sliding and last full layer.
DRAFTER = dict(TARGET, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
               layer_types=["sliding_attention", "full_attention"], num_kv_shared_layers=2,
               final_logit_softcapping=None, vocab_size_per_layer_input=0)
# Another spelling of the drafter's tensors, and the draft `names` rules a layout serving it carries.
SPELLING = [["^pre_projection\\.", "draft.into."], ["^post_projection\\.", "draft.out_of."],
            ["^model\\.layers\\.", "draft.block."], ["^model\\.", "draft."]]
NAMES = [["^draft\\.into\\.", "pre_projection."], ["^draft\\.out_of\\.", "post_projection."],
         ["^draft\\.block\\.", "model.layers."], ["^draft\\.", "model."]]


def available():
    return torch is not None and transformers is not None and hasattr(transformers, "Gemma4AssistantForCausalLM")


def randomize(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("q_norm.weight", "k_norm.weight")):
                parameter.copy_(0.35 + torch.randn(parameter.shape, generator=g) * 0.03)
            elif "post_" in name and name.endswith(".weight") and "norm" in name:
                parameter.copy_(0.1 + torch.randn(parameter.shape, generator=g) * 0.02)
            elif "norm" in name and name.endswith(".weight"):
                parameter.copy_(1.0 + torch.randn(parameter.shape, generator=g) * 0.1)
            else:
                parameter.copy_(torch.randn(parameter.shape, generator=g) * parameter.shape[-1] ** -0.5)
        for name, buffer in model.named_buffers():
            if name.endswith("layer_scalar"):
                buffer.fill_(0.5 + 0.25 * int(name.split(".")[-2]))


def checkpoints(root):
    """(target, drafter folder, the drafter's official model) saved under root in BF16, the models in FP32."""
    target = transformers.Gemma4ForCausalLM(transformers.Gemma4TextConfig(**TARGET))
    randomize(target, 5)
    config = transformers.Gemma4AssistantConfig(text_config=transformers.Gemma4TextConfig(**DRAFTER),
                                                backbone_hidden_size=TARGET["hidden_size"])
    drafter = transformers.Gemma4AssistantForCausalLM(config)
    randomize(drafter, 7)
    drafter.lm_head.weight = drafter.model.embed_tokens.weight
    for model, folder in ((target, root / "model"), (drafter, root / "assistant")):
        model.to(torch.bfloat16).save_pretrained(folder, safe_serialization=True)
    return target.float().eval(), root / "assistant", drafter.float().eval()


def respell(folder, out):
    """The drafter folder with its tensors under SPELLING."""
    import re
    from safetensors.torch import load_file, save_file
    shutil.copytree(folder, out)
    for file in out.glob("*.safetensors"):
        tensors = {}
        for name, tensor in load_file(str(file)).items():
            for pattern, replacement in SPELLING:
                name = re.sub(pattern, replacement, name)
            tensors[name] = tensor
        save_file(tensors, str(file), metadata={"format": "pt"})
    return out


def rotate(q, inv_freq, position):
    """q [heads][dim] turned at `position` by a table's inverse frequencies (the halves' pairs, as transformers
    turns them)."""
    angles = position * inv_freq.double()
    cos, sin = torch.cat([angles.cos(), angles.cos()]).float(), torch.cat([angles.sin(), angles.sin()]).float()
    half = q.shape[-1] // 2
    turned = torch.cat([-q[..., half:], q[..., :half]], dim=-1)
    return q * cos + turned * sin


def package_module(name):
    """A module of the package as an engine imports it (its core's code under .core)."""
    import importlib
    from crit_plugins.assemble import plugin_package
    from crit_plugins.manifest import load_manifest
    return importlib.import_module(f"{plugin_package(load_manifest(HERE)).__name__}.{name}")


@unittest.skipIf(not available(), "transformers with Gemma4AssistantForCausalLM")
class HostReferenceTests(unittest.TestCase):
    """The drafter's step() over FP32 host products against Gemma4AssistantForCausalLM."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.target, cls.folder, cls.official = checkpoints(cls.root)
        cls.drafter = package_module("assistant_drafter.assistant_drafter")
        g = torch.Generator().manual_seed(3)
        cls.ids = torch.randint(3, 256, (1, 40), generator=g)   # longer than the sliding window
        with torch.no_grad():
            out = cls.target(cls.ids, output_hidden_states=True, return_shared_kv_states=True, use_cache=False)
            cls.normed = cls.target.model(cls.ids, use_cache=False).last_hidden_state
        cls.hidden = out.hidden_states[-1][:, -1:]
        cls.kv = {k: (v[0].float(), v[1].float()) for k, v in out.shared_kv_states.items()}
        cls.pending = int(out.logits[0, -1].argmax())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def companion(self, folder, names=None):
        return package_module("core.template.companion").Companion(folder, names)

    def chain(self, companion, tokens, h, position):
        """The drafter's step() on the host over FP32 products and the official target's keys and values, a draft
        at the pending position each: [(logits, next h)]."""
        from crit_core.model import parse_model_config
        from crit_models.config import architecture
        d, tables = self.drafter, package_module("core.rope.rope_table")
        text = d.text_config(companion)
        per, top = d.arrange(*d.weights(companion.ck, len(text["layer_types"]), lambda name, tensor: tensor.float()),
                             lambda g, u: torch.cat([g, u]))
        config = architecture(parse_model_config(json.loads((self.root / "model" / "config.json").read_text(encoding="utf-8"))))
        types, window = text["layer_types"], TARGET["sliding_window"]

        def inv_freq(kind, dim):
            rope = next((e.rope for e in config.rope_by_layer_type if e.layer_type == kind), None)
            return torch.tensor(tables.rope_table(config.rope if rope is None else rope, dim,
                                                  config.max_positions).inv_freq)

        def mm(x, w):
            return x.float() @ w.float().T

        def rms(x, weight):
            x = x.float()
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + text["rms_norm_eps"]) * weight

        def rope(i, q):
            return rotate(rms(q, per[i]["self_attn.q_norm.weight"]), inv_freq(types[i], q.shape[-1]), position)

        def attend(i, q):
            k, v = self.kv[types[i]]
            k, v = k[0, :, :position], v[0, :, :position]             # [kv_heads][positions][dim]
            if types[i] == "sliding_attention":
                k, v = k[:, -window:], v[:, -window:]
            group = q.shape[1] // k.shape[0]
            k, v = k.repeat_interleave(group, 0), v.repeat_interleave(group, 0)
            return torch.einsum("rhp,hpd->rhd", torch.einsum("rhd,hpd->rhp", q.float(), k).softmax(-1), v)

        def gelu(x):
            return torch.nn.functional.gelu(x, approximate="tanh")
        ops, found = d.Ops(mm, rms, rope, attend, gelu), []
        h = h.reshape(1, -1)
        for token in tokens:
            embedded = self.target.get_input_embeddings()(torch.tensor([token]))
            logits, h = d.step(per, top, text["num_attention_heads"], embedded, h, ops)
            found.append((logits[0], h))
        return found

    def official_chain(self, tokens, h, position):
        found = []
        for token in tokens:
            embedded = self.target.get_input_embeddings()(torch.tensor([[token]]))
            with torch.no_grad():
                out = self.official(inputs_embeds=torch.cat([embedded, h.reshape(1, 1, -1)], dim=-1),
                                    position_ids=torch.tensor([[position]]), shared_kv_states=self.kv)
            h = out.last_hidden_state
            found.append((out.logits[0, -1], h.reshape(1, -1)))
        return found

    def assertDrafts(self, found, want):
        for i, ((logits, h), (theirs, their_h)) in enumerate(zip(found, want)):
            # BF16 steps between the drafter's layers and at each product's inputs (as the device head runs them)
            # against transformers in FP32: within a few hundredths of the logits' spread, the same choice
            spread = float(theirs.std())
            self.assertLess(float((logits - theirs).abs().max()), 0.05 * spread, i)
            self.assertEqual(int(logits.argmax()), int(theirs.argmax()), i)
            self.assertLess(float((h.float() - their_h).abs().max()), 0.05 * float(their_h.std()), i)

    def test_the_row_the_drafter_reads_is_the_targets_final_normed_row(self):
        self.assertTrue(torch.allclose(self.hidden, self.normed[:, -1:], atol=1e-5))

    def test_its_drafts_are_the_official_drafters_in_both_layouts(self):
        position = self.ids.shape[1]   # the pending token's, not yet in the target's KV
        tokens = [self.pending]
        for _ in range(3):   # a greedy chain at the same position, each draft fed the last one's row
            want = self.official_chain(tokens, self.hidden, position)
            tokens.append(int(want[-1][0].argmax()))
        want = self.official_chain(tokens, self.hidden, position)
        respelled = respell(self.folder, self.root / "respelled")
        for folder, names in ((self.folder, None), (respelled, NAMES)):
            with self.subTest(folder=folder.name):
                companion = self.companion(folder, names)
                self.assertDrafts(self.chain(companion, tokens, self.hidden, position), want)
                companion.close()

    def test_the_drafter_fits_the_target_it_was_made_for(self):
        class Shape:
            hidden, vocabulary, types = 128, 256, tuple(TARGET["layer_types"])
        config = json.loads((self.folder / "config.json").read_text(encoding="utf-8"))
        fits = self.drafter.AssistantDrafter.fits
        self.assertIsNone(fits(config, Shape))
        Shape.hidden = 256
        self.assertIn("backbone", fits(config, Shape))
        Shape.hidden, Shape.types = 128, ("sliding_attention",) * 4
        self.assertIn("full_attention", fits(config, Shape))


@unittest.skipIf(not available(), "transformers with Gemma4AssistantForCausalLM")
class CompanionEngineTests(t.Engines):
    """The target beside its drafter folder, served by gemma4-model with a drafter in the selection: the models core
    finds the companion beside the checkpoint and plans it as the draft head; on a GPU (captured rounds) drafting
    from it replies the plain greedy tokens."""
    plugin = HERE
    library = t.BASE + ["greedy-head-drafter"]

    @classmethod
    def checkpoint(cls, root):
        target, _, _ = checkpoints(root)
        cls.model = root / "model"
        cls.reference = target.to(cls.device).eval()
        return cls.model

    def test_the_companion_beside_the_checkpoint_is_the_draft_head(self):
        engine = self.load(use={"drafter": "greedy-head-drafter"}, mtp=True, width=3, rounds=self.device == "cuda")
        info = engine.model.info()
        self.assertEqual(Path(info["draft_folder"]).resolve(), (Path(self.temp.name) / "assistant").resolve())
        self.assertEqual(info["drafter_kind"], "companion MTP drafter (gemma4_assistant)")
        self.assertGreater(info["draft_bytes_per_step"], 0)
        self.assertEqual(info["draft_width"], 3)
        if self.device != "cuda":
            return
        prompts = self.prompts(3, 30)
        plain = [self.engine.generate(p, 24, stop_tokens=[], sampling=GREEDY)[0] for p in prompts]
        found = [engine.generate(p, 24, stop_tokens=[], sampling=GREEDY) for p in prompts]
        self.assertEqual([tokens for tokens, _ in found], plain)
        for tokens, result in found:
            self.assertEqual(sum(result["accepted"]) + result["rounds"], len(tokens) - 1)

    def test_a_draft_folder_that_holds_no_drafter_is_refused(self):
        with self.assertRaisesRegex(Exception, "holds no gemma4_assistant checkpoint"):
            self.load(use={"drafter": "greedy-head-drafter"}, mtp=True, draft_folder=str(self.model))


if __name__ == "__main__":
    unittest.main()
