"""A hybrid mixture-of-experts engine of the library's device plugins on a GPU. `crit plugins add` copies the KV pool,
attention, projections, sampling and placement plugins with the gated-delta recurrence, the top-k router, the experts
block, the expert store and fused-norms (zero-centred norms, query/key norms with partial rotary positions, the gated
norm and the output gate) into a package beside a Qwen3-Next-layout model plugin (support/device_plugins/next-model);
Core's loader runs it on Core's concurrent runtime. Its fixed-text scores follow transformers' log-probabilities of the
same checkpoint (its BF16 weights in FP32, so only the engine rounds), a text longer than a prefill chunk among
them (scored chunk by chunk, the state carried), each token of its greedy continuations is
transformers' choice up to BF16 rounding, and requests in flight together decode what each decodes alone. Needs
CUDA, transformers with Qwen3-Next, safetensors and Core's crit_core_steps (CRIT_CORE_STEP_LIBRARY).

The checkpoint's decays are milder than transformers' initial ones (A up to 1, not 16): with A up to 16 this random
checkpoint's recurrent outputs fall to a thousandth of its values, and every chunked form of the recurrence
(fla's kernel and transformers' own chunked path) parts from the exact one by a few percent there, which the
sharpened head turns into whole nats on a few positions."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "cli"):
    sys.path.insert(0, str(ROOT / _folder))

from crit_core.native import step_library  # noqa: E402

try:
    import torch
    import transformers
    import safetensors  # noqa: F401
    READY = (torch.cuda.is_available() and step_library() is not None
             and hasattr(transformers, "Qwen3NextForCausalLM"))
except ImportError:
    READY = False

SELECTION = {"use": {"model": "next-model", "placement": "device-placement", "projections": "projections-dense",
                     "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "token-sampling",
                     "recurrence": "sequential-recurrence", "router": "topk-router", "experts": "experts-reference",
                     "expert_source": "expert-tensors", "norms": "fused-norms"},
             "base": {"norms": "torch-norms"},
             # Qwen3-Next's router: a softmax over every expert, the top ones renormalized.
             "options": {"topk-router": {"score_before_selection": True, "renormalize": True}}}
LIBRARY = ["attention-state", "softmax-attention", "projections-dense", "token-sampling", "device-placement",
           "sequential-recurrence", "topk-router", "experts-reference", "expert-tensors", "fused-norms"]


@unittest.skipUnless(READY, "a GPU engine: CUDA, transformers with Qwen3-Next, safetensors and CRIT_CORE_STEP_LIBRARY")
class HybridEngineTests(unittest.TestCase):
    selection, library = SELECTION, LIBRARY

    @classmethod
    def setUpClass(cls):
        from crit.plugins import add
        from crit_core.engine import load
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        torch.manual_seed(0)
        config = transformers.Qwen3NextConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
            decoder_sparse_step=1, norm_topk_prob=True, max_position_embeddings=1024, rms_norm_eps=1e-6,
            tie_word_embeddings=False)
        model = transformers.Qwen3NextForCausalLM(config)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("norm.weight") and "linear_attn" not in name:
                    parameter.normal_(0.0, 0.1)   # Qwen3-Next's norms scale by 1 + weight: not all ones
                if name.endswith("A_log"):
                    parameter.copy_(torch.empty_like(parameter).uniform_(0.05, 1.0).log())   # milder decays
            model.lm_head.weight.mul_(30)   # sharp logits: a greedy choice no rounding flips
        model.to(torch.bfloat16)
        cls.model = root / "model"
        model.save_pretrained(cls.model, safe_serialization=True)
        cls.reference = model.float().cuda().eval()   # the stored BF16 weights, computed in FP32
        package = root / "package"
        add(package / "plugins", cls.library)
        shutil.copytree(ROOT / "tests" / "support" / "device_plugins" / "next-model", package / "plugins" / "next-model")
        (package / "configuration.json").write_text(json.dumps({"plugins": cls.selection}), encoding="utf-8")
        cls.engine = load(package, cls.model)
        # The first requests after load, before any test serves one (a later test would find what earlier ones
        # compiled): prompts of lengths no load request used, alone and together.
        from crit_plugins.testing import compiles
        prompts = [cls.prompts(cls, 1, length)[0] for length in (2, 45, 140, 260)]
        greedy = {"temperature": 0.0}
        with compiles() as found:
            for prompt in prompts:
                cls.engine.generate(prompt, 4, stop_tokens=[], sampling=greedy)
            with ThreadPoolExecutor(len(prompts)) as pool:
                list(pool.map(lambda p: cls.engine.generate(p, 4, stop_tokens=[], sampling=greedy), prompts))
        cls.compiled_after_load = found

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.temp.cleanup()

    def prompts(self, count, length=17):
        g = torch.Generator().manual_seed(count * 103 + length)
        return [torch.randint(3, 256, (length + 5 * i,), generator=g).tolist() for i in range(count)]

    def logits(self, ids):
        with torch.no_grad():
            return self.reference(torch.tensor([ids]).cuda()).logits[0].float()

    def test_requests_after_load_compile_nothing_at_new_prompt_lengths(self):
        # A plugin compiles at load: the first requests after it (setUpClass, before any test), prompts of lengths no
        # load request used, alone and together, compiled no kernel and built no extension.
        self.assertEqual(self.compiled_after_load, [])

    def test_the_engine_runs_the_librarys_device_plugins_in_every_slot(self):
        info = self.engine.info()
        self.assertEqual(info["plugins"], self.selection["use"])

    def test_scores_follow_transformers_log_probabilities(self):
        # 40 tokens: the recurrence's prompt rows run through fla's chunked kernel, where it is installed. The engine
        # computes in BF16 and the reference in FP32, and the sharpened head turns a BF16 rounding of the hidden state
        # into up to a nat or so on a token 20 nats down: the targets (random tokens) agree on average, and every
        # position's most likely token and its log-probability agree closely.
        (ids,) = self.prompts(1, 40)
        scored = self.engine.score(ids, 1, "prefill", top_k=1)
        logp = self.logits(ids)[:len(ids) - 1].log_softmax(-1)
        expected = logp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].tolist()
        differences = [abs(a - b) for a, b in zip(scored.target, expected)]
        self.assertLess(sum(differences) / len(differences), 0.15, differences)
        best, token = logp.max(-1)
        for i, ((found, value),) in enumerate(scored.top):
            second = float(logp[i].topk(2).values[1])
            if found != int(token[i]):   # only a near-tie may rank them the other way
                self.assertLess(float(best[i]) - second, 0.5, (i, found, int(token[i])))
                continue
            self.assertLess(abs(value - float(best[i])), 0.5, (i, value, float(best[i])))

    def test_a_text_longer_than_a_prefill_chunk_scores_chunk_by_chunk(self):
        # 150 tokens, three of the model's prefill chunks: the first scores as the text cut after it does, and the later
        # ones, which read the recurrent state and KV the chunks before them kept, follow transformers on average.
        (ids,) = self.prompts(1, 150)
        chunk = self.engine.model.options["prefill_chunk"]
        scored = self.engine.score(ids, 1, "prefill")
        self.assertEqual(scored.target[:chunk], self.engine.score(ids[:chunk + 1], 1, "prefill").target)
        logp = self.logits(ids)[:len(ids) - 1].log_softmax(-1)
        expected = logp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].tolist()
        later = [abs(a - b) for a, b in zip(scored.target[chunk:], expected[chunk:])]
        self.assertLess(sum(later) / len(later), 0.15, later)

    def test_greedy_continuations_are_transformers_choices(self):
        for prompt in self.prompts(3):
            tokens, _ = self.engine.generate(prompt, 12, stop_tokens=[], sampling={"temperature": 0.0})
            logits = self.logits(prompt + tokens)
            for i, token in enumerate(tokens):
                row = logits[len(prompt) - 1 + i]
                self.assertGreaterEqual(float(row[token]), float(row.max()) - 0.5, (i, token, int(row.argmax())))

    def test_requests_in_flight_together_decode_what_each_decodes_alone(self):
        prompts = self.prompts(6, 9)
        alone = [self.engine.generate(p, 10, stop_tokens=[], sampling={"temperature": 0.0})[0] for p in prompts]
        with ThreadPoolExecutor(6) as pool:
            together = list(pool.map(lambda p: self.engine.generate(p, 10, stop_tokens=[],
                                                                    sampling={"temperature": 0.0})[0], prompts))
        self.assertEqual(together, alone)


class HybridGroupedExpertsTests(HybridEngineTests):
    """The same engine with grouped-experts in the experts slot: the experts as BF16 bundles, the routes grouped by
    expert on the device (crit's kernel grouped-experts where it builds here, else its torch fallback)."""
    selection = {**SELECTION, "use": {**SELECTION["use"], "experts": "grouped-experts"}}
    library = [p for p in LIBRARY if p != "experts-reference"] + ["grouped-experts"]

    def test_the_experts_run_on_the_path_the_kernel_build_gives(self):
        source = self.engine.facts["kernels"]["grouped-experts"]["source"]
        self.assertIn(source, ("tuned", "built", "fallback"))
        self.assertEqual(self.engine.facts["experts"]["path"], "torch" if source == "fallback" else "kernel")


if __name__ == "__main__":
    unittest.main()
